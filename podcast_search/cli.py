"""podcast-search: find episodes in podcast back catalogues.

A plan of 1-4 questions (written by you, your agent, or Claude) is put to TypeSafe's Jev for every
episode, one episode at a time; results are ranked in code. Every answer is cached, so repeat searches
are free. Agents: run `podcast-search guide` first.

Commands:
  podcast-search guide                                  how to use this tool well (start here)
  podcast-search add <feed url | podcast name>          add a podcast, or refresh it
  podcast-search list                                   podcasts, stored answers, saved plans
  podcast-search search --plan-file plan.json           search with a plan you wrote ("-" reads stdin)
  podcast-search search --plan NAME                     search with a saved plan
  podcast-search search "query"                         Claude writes the plan (needs ANTHROPIC_API_KEY)
  podcast-search episode 3                              details for a result of the last search
  podcast-search spotify-login | spotify-link | save 1-5

Add --json to list, search, episode and save for machine-readable output.
"""
import argparse
import asyncio
import json
import sys
import time

from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

from . import library
from .cache import AnswerCache, question_hash
from .config import ROOT, load_env

PLANS = ROOT / "plans"
LAST = ROOT / "last_search.json"
CLAUDE_MODEL = "claude-opus-5"
JEV_MODEL = "jev-1.13.0"   # pinned: cached answers are only valid for the model that gave them
JEV_RATE = 18              # requests/second, under TypeSafe's 1,200/min
JEV_CONCURRENCY = 12
JEV_PRICE = 0.042          # $ per million input tokens

JSON_OUT = False


def say(*a, **k):
    """Human-readable output: stdout normally, stderr under --json so stdout stays pure JSON."""
    print(*a, file=sys.stderr if JSON_OUT else sys.stdout, **k)


def emit(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- plans: format, rules, validation

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["score", "noul"]},
                    "role": {"type": "string", "enum": ["rank", "exclude", "require"]},
                    "weight": {"type": "number"},
                    "instructions": {"type": "string"},
                    "levels": {"type": "array", "items": {"type": "string"}},
                    "does_not_count": {"type": "string"},
                },
                "required": ["id", "kind", "role", "weight", "instructions", "levels", "does_not_count"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["interpretation", "questions"],
    "additionalProperties": False,
}

# The one copy of the question-writing rules: the built-in Claude planner and `guide` both use it.
PLAN_RULES = """Jev judges ONE podcast episode at a time, from its title and summary only, and returns a typed
answer, never text. It sees each question's instructions, levels and does_not_count; it never sees the
question id, the role or the weight.

Question kinds:
- score: places the episode on 3-5 ordered levels you describe. Use for "how much does this episode match".
- noul: probability (0-1) that a statement about the episode is true. Use only for hard yes/no conditions.

Write 1-4 questions, each one narrow, independent judgment. Split a compound request into separate
questions instead of one question that asks several things.

Only ask what a title and summary can actually show. Summaries describe an episode's topic, guest and
angle; they rarely list the specific techniques, steps or examples inside it. A question that needs
detail the summary won't have pushes genuinely on-topic episodes down for missing it.

Make exactly one rank question the core topic of the request, with weight 1.0. Any other preference
(practicality, tone, format, audience) is secondary: include it only if a summary could plausibly show
it, and give it a weight of 0.3 or less.

For each question:
- role "rank" (kind score): orders results. Weights of rank questions are relative.
- role "exclude" (kind noul): its truth pushes an episode down, e.g. "the episode is mainly about X".
- role "require" (kind noul): must be true for an episode to be useful.
- instructions: the full judgment, readable on its own.
- levels (score only): concrete, self-contained descriptions from lowest to highest match. The lowest
  level must mean "not about this at all". Use [] for a noul.
- does_not_count: what should NOT count as a match, to stop superficial keyword hits. Be specific.

Do not ask about audio, length, popularity or anything else a title and summary can't show."""

PLANNER_SYSTEM = ("You turn a listener's search request into questions for Jev. Follow these rules exactly.\n\n"
                  + PLAN_RULES)


def validate_plan(plan: dict) -> dict:
    """Check a plan from any source (Claude, an agent, a file) and normalise it. Exits with a clear message."""
    def bad(msg):
        raise SystemExit(f"plan error: {msg}")
    if not isinstance(plan, dict) or not isinstance(plan.get("questions"), list) or not plan["questions"]:
        bad('expected {"questions": [ ... ]} with at least one question')
    if len(plan["questions"]) > 10:
        bad("Jev takes at most 10 questions per call")
    plan.setdefault("interpretation", "")
    seen = set()
    for n, q in enumerate(plan["questions"], 1):
        for k in ("id", "kind", "role", "instructions", "does_not_count"):
            if not isinstance(q.get(k), str) or not q[k].strip():
                bad(f"question {n} needs a non-empty '{k}'")
        if q["id"] in seen:
            bad(f"duplicate question id '{q['id']}'")
        seen.add(q["id"])
        if q["kind"] not in ("score", "noul"):
            bad(f"'{q['id']}': kind must be score or noul")
        if q["role"] not in ("rank", "exclude", "require"):
            bad(f"'{q['id']}': role must be rank, exclude or require")
        if q["role"] == "rank" and q["kind"] != "score":
            bad(f"'{q['id']}': rank questions must be kind score")
        if q["role"] != "rank" and q["kind"] != "noul":
            bad(f"'{q['id']}': {q['role']} questions must be kind noul")
        q["weight"] = float(q.get("weight", 1.0 if q["role"] == "rank" else 0))
        if q["kind"] == "score":
            if not isinstance(q.get("levels"), list) or not 2 <= len(q["levels"]) <= 10:
                bad(f"'{q['id']}': a score needs 2-10 levels")
        else:
            q["levels"] = []
    if not any(q["role"] == "rank" for q in plan["questions"]):
        bad("at least one rank question is needed")
    return plan


def make_plan(query: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": f"Search request: {query}"}],
        output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
    )
    if resp.stop_reason == "refusal":
        raise SystemExit("Claude declined to plan this query.")
    return validate_plan(json.loads(next(b.text for b in resp.content if b.type == "text")))


def jev_question(q: dict):
    instr = {"judgment": q["instructions"], "does_not_count": q["does_not_count"]}
    return Score(instructions=instr, criteria=q["levels"]) if q["kind"] == "score" else Noul(instructions=instr)


def save_plan(name: str, query: str, plan: dict):
    PLANS.mkdir(exist_ok=True)
    path = PLANS / f"{name}.json"
    path.write_text(json.dumps({"name": name, "query": query, "saved": time.strftime("%Y-%m-%d %H:%M"),
                                "plan": plan}, indent=1))
    return path


def load_plan(name: str):
    path = PLANS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no saved plan '{name}' (see: podcast-search list)")
    d = json.loads(path.read_text())
    return d["query"], validate_plan(d["plan"])


def read_plan_file(path: str):
    """A plan written by you or an agent: either a bare plan, or a saved-plan file with "plan" and "query"."""
    text = sys.stdin.read() if path == "-" else open(path).read()
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"plan file is not valid JSON: {e}")
    if "plan" in d and "questions" not in d:
        return d.get("query", ""), validate_plan(d["plan"])
    return d.pop("query", ""), validate_plan(d)

# ---------------------------------------------------------------- Jev judges, reusing stored answers

async def judge_all(slug: str, episodes: list, plan: dict, cache: AnswerCache):
    """Answers for every episode x question. Only (episode, question) pairs not already stored go to Jev,
    and each episode's missing questions are asked together in one call."""
    qh = {q["id"]: question_hash(q) for q in plan["questions"]}
    known = cache.lookup(slug, JEV_MODEL, set(qh.values()))
    answers = [{} for _ in episodes]
    todo = []
    for i, e in enumerate(episodes):
        missing = []
        for q in plan["questions"]:
            hit = known.get((e["guid"], e["content_hash"], qh[q["id"]]))
            if hit is None:
                missing.append(q)
            else:
                answers[i][q["id"]] = hit
        if missing:
            todo.append((i, e, missing))

    stats = {"from_cache": sum(len(a) for a in answers), "asked": sum(len(m) for _, _, m in todo),
             "calls": len(todo), "tokens": 0}
    if not todo:
        return answers, stats

    sem, gate, last = asyncio.Semaphore(JEV_CONCURRENCY), asyncio.Lock(), [0.0]
    pending = []

    async def throttle():
        async with gate:
            wait = last[0] + 1 / JEV_RATE - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            last[0] = time.perf_counter()

    async with AsyncTypeSafeClient() as jev:
        async def one(i, e, missing):
            async with sem:
                await throttle()
                r = await jev.system_one(state={"episode": {"title": e["title"], "summary": e["summary"]}},
                                         questions={qh[q["id"]]: jev_question(q) for q in missing}, model=JEV_MODEL)
            stats["tokens"] += r.usage.input_tokens or 0
            for q in missing:
                h = qh[q["id"]]
                if q["kind"] == "score":
                    v = r.scores[h]
                    ans = {"score": v.score, "max": max(v.probabilities), "confidence": v.confidence}
                else:
                    ans = {"p": r.nouls[h].noul}
                answers[i][q["id"]] = ans
                pending.append((slug, e["guid"], e["content_hash"], h, JEV_MODEL, ans))
            if len(pending) >= 200:          # save as we go, so an interrupted run keeps its progress
                cache.store(pending[:]); pending.clear()

        await asyncio.gather(*(one(*t) for t in todo))
    if pending:
        cache.store(pending)
    return answers, stats

# ---------------------------------------------------------------- rank in code

def rank(episodes, answers, plan):
    rank_qs = [q for q in plan["questions"] if q["role"] == "rank"]
    total_w = sum(max(q["weight"], 0) for q in rank_qs) or 1
    rows = []
    for e, a in zip(episodes, answers):
        s = sum(max(q["weight"], 0) * a[q["id"]]["score"] / a[q["id"]]["max"] for q in rank_qs) / total_w
        for q in plan["questions"]:
            if q["role"] == "exclude":
                s *= 1 - a[q["id"]]["p"]
            elif q["role"] == "require":
                s *= a[q["id"]]["p"]
        rows.append((s, e, a))
    return sorted(rows, key=lambda r: -r[0])


def explain(query, top):
    import anthropic
    client = anthropic.Anthropic()
    listing = "\n\n".join(f"[{i}] {e['title']}\n{e['summary']}" for i, (_, e, _) in enumerate(top, 1))
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL, max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content":
                   f"A listener searched a podcast for: {query}\n\nThese episodes ranked highest:\n\n{listing}\n\n"
                   "For each, write one plain sentence on why it fits the search, or say plainly if it doesn't. "
                   "Number them to match."}],
    )
    if resp.stop_reason == "refusal":
        return "(Claude declined to explain.)"
    return next(b.text for b in resp.content if b.type == "text")

# ---------------------------------------------------------------- commands

def spotify_urls(slug: str) -> dict:
    from . import spotify
    return {g: spotify.episode_url(m["id"]) for g, m in spotify.existing_mapping(slug).items()}


def answers_text(a: dict, ids: list) -> str:
    return "  ".join(f"{k}={a[k]['score']:.2f}" if "score" in a[k] else f"{k}={a[k]['p']:.2f}" for k in ids)


def print_plan(plan):
    for q in plan["questions"]:
        say(f"  [{q['role']}{' w=' + str(q['weight']) if q['role'] == 'rank' else ''}] {q['kind']} {q['id']}: {q['instructions']}")
        for i, lv in enumerate(q["levels"]):
            say(f"        {i}: {lv}")
        say(f"        not: {q['does_not_count']}")


def cmd_list(cache):
    pods = [{"slug": p["slug"], "title": p["title"], "feed_url": p["feed_url"], "episodes": len(p["episodes"]),
             "stored_answers": cache.count(p["slug"]), "fetched": p["fetched"]} for p in library.podcasts()]
    plans = []
    for f in sorted(PLANS.glob("*.json")) if PLANS.exists() else []:
        d = json.loads(f.read_text())
        plans.append({"name": d["name"], "query": d["query"], "saved": d["saved"],
                      "questions": [q["id"] for q in d["plan"]["questions"]]})
    if JSON_OUT:
        return emit({"podcasts": pods, "plans": plans})
    for p in pods:
        say(f"{p['slug']:44s} {p['episodes']:4d} episodes  {p['stored_answers']:6,} stored answers   {p['title']}")
    for p in plans:
        say(f"plan  {p['name']:30s} {p['saved']}  {p['query']}")


def cmd_search(args, cache):
    pod = library.load(args.podcast)
    episodes = pod["episodes"]
    if args.plan_file:
        query, plan = read_plan_file(args.plan_file)
        query = args.query or query
        source = f"plan file {args.plan_file}"
    elif args.plan:
        query, plan = load_plan(args.plan)
        source = f"saved plan '{args.plan}'"
    else:
        query = args.query
        t0 = time.perf_counter()
        plan = make_plan(query)
        source = f"Claude ({time.perf_counter() - t0:.1f}s): {plan['interpretation']}"
    say(f"\nPODCAST {pod['title']}\nQUERY   {query or '(none given)'}\nPLAN    {source}\n")
    print_plan(plan)

    t1 = time.perf_counter()
    answers, st = asyncio.run(judge_all(pod["slug"], episodes, plan, cache))
    st.update(seconds=round(time.perf_counter() - t1, 1), cost_usd=round(st["tokens"] * JEV_PRICE / 1e6, 5),
              needed=len(episodes) * len(plan["questions"]))
    say(f"\nJEV     {st['needed']:,} answers needed: {st['from_cache']:,} from cache, {st['asked']:,} asked in "
        f"{st['calls']:,} calls  ({st['seconds']}s, {st['tokens']:,} tokens, ${st['cost_usd']:.4f})\n")

    ranked = rank(episodes, answers, plan)
    LAST.write_text(json.dumps({"podcast": pod["slug"], "query": query, "plan_name": args.plan, "plan": plan,
                                "ranked": [{"rank_score": sc, "guid": e["guid"], "title": e["title"], "date": e["date"],
                                            "answers": a} for sc, e, a in ranked]}, indent=1))
    saved_path = save_plan(args.save_plan, query, plan) if args.save_plan else None
    urls = spotify_urls(pod["slug"])
    ids = [q["id"] for q in plan["questions"]]
    top = ranked[: args.top]

    if JSON_OUT:
        emit({"podcast": {"slug": pod["slug"], "title": pod["title"]}, "query": query, "plan": plan,
              "plan_source": "file" if args.plan_file else "saved" if args.plan else "claude",
              "saved_plan": saved_path.stem if saved_path else None, "jev": st,
              "results": [{"rank": i, "rank_score": round(sc, 4), "guid": e["guid"], "title": e["title"],
                           "date": e["date"], "link": e.get("link"), "spotify_url": urls.get(e["guid"]),
                           "answers": a} for i, (sc, e, a) in enumerate(top, 1)]})
    else:
        for i, (sc, e, a) in enumerate(top, 1):
            say(f"{i:2d}. {sc:.2f}  {e['title'][:78]}\n      {e['date']}   {answers_text(a, ids)}")
        if saved_path:
            say(f"\nsaved plan -> plans/{saved_path.name}")
    if args.explain:
        say("\nWHY (Claude, top 5)\n" + explain(query, ranked[:5]))


def last_search():
    if not LAST.exists():
        raise SystemExit("no search yet - run podcast-search search first")
    return json.loads(LAST.read_text())


def parse_ranks(spec: str, n: int) -> list[int]:
    picks = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        picks.update(range(int(a), int(b or a) + 1))
    return [r for r in sorted(picks) if 0 < r <= n]


def cmd_episode(args):
    last = last_search()
    pod = library.load(last["podcast"])
    by_guid = {e["guid"]: e for e in pod["episodes"]}
    urls = spotify_urls(pod["slug"])
    out = []
    for r in parse_ranks(args.ranks, len(last["ranked"])):
        row = last["ranked"][r - 1]
        e = by_guid[row["guid"]]
        out.append({"rank": r, "rank_score": round(row["rank_score"], 4), "guid": e["guid"], "title": e["title"],
                    "date": e["date"], "link": e.get("link"), "spotify_url": urls.get(e["guid"]),
                    "summary": e["summary"], "answers": row["answers"]})
    if JSON_OUT:
        return emit({"query": last["query"], "episodes": out})
    for x in out:
        say(f"\n#{x['rank']}  {x['title']}\n{x['date']}   score {x['rank_score']:.2f}"
            + (f"   {x['spotify_url']}" if x["spotify_url"] else "") + f"\n\n{x['summary']}")


def cmd_save(args):
    from . import spotify
    last = last_search()
    chosen = [last["ranked"][r - 1] for r in parse_ranks(args.ranks, len(last["ranked"]))]
    say(f"saving {len(chosen)} episode(s) from the last search ('{(last['query'] or '')[:60]}')")
    results = spotify.save(last["podcast"], chosen)
    n = sum(r["status"] == "confirmed" for r in results)
    if JSON_OUT:
        return emit({"requested": len(chosen), "confirmed": n, "results": results})
    labels = {"confirmed": "saved, confirmed in library", "not_confirmed": "NOT in library after saving",
              "no_spotify_match": "no Spotify match, skipped"}
    for r in results:
        say(f"  {labels[r['status']]:30s} {r['title'][:62]}" + (f"\n  {'':30s} {r['spotify_url']}" if r["spotify_url"] else ""))
    say(f"{n} of {len(chosen)} confirmed in Your Episodes")


EXAMPLE_PLAN = {
 "interpretation": "The listener wants episodes about calming nerves and performance anxiety before public speaking or a big presentation, ideally with actionable techniques, and explicitly not episodes centered on job interview preparation.",
 "questions": [
  {
   "id": "core_presentation_nerves",
   "kind": "score",
   "role": "rank",
   "weight": 1.0,
   "instructions": "Judge how centrally this episode is about managing nerves, anxiety or stage fright connected to speaking in front of an audience — a presentation, pitch, talk, speech or similar high-stakes public speaking moment. Base this on the stated topic, guest and angle in the title and summary. Top levels require that calming or handling the fear/nerves of speaking is the episode's main subject, not a passing mention.",
   "levels": [
    "Not about public speaking or performance nerves at all.",
    "Touches public speaking or anxiety only in passing, or covers general stress/anxiety with no link to speaking in front of people.",
    "Substantially about public speaking, presenting or pitching, with nerves, confidence or fear discussed as one part of it.",
    "Largely about the fear, nerves or anxiety of speaking in front of an audience, with the presentation context clear.",
    "Entirely devoted to managing nerves, stage fright or anxiety before and during a presentation or public talk."
   ],
   "does_not_count": "General mental-health or anxiety-disorder episodes with no speaking context; communication or storytelling craft episodes focused only on slide design, structure or persuasion; episodes about performance anxiety in sports or music unless speaking to an audience is the focus."
  },
  {
   "id": "practical_techniques",
   "kind": "score",
   "role": "rank",
   "weight": 0.3,
   "instructions": "Judge how much the title and summary signal that the episode offers usable techniques, exercises or steps a listener could apply, rather than only discussion, theory or personal narrative. Signals include promises of tips, drills, routines, frameworks, breathing or preparation methods, or a coach walking through what to do.",
   "levels": [
    "Purely conversational, theoretical or a personal story with no hint of applicable advice.",
    "Mostly reflection or research with maybe an implicit takeaway.",
    "Mixes discussion with some advice or suggested approaches.",
    "Clearly framed around concrete techniques, steps or routines the listener can try."
   ],
   "does_not_count": "Generic promotional phrasing like 'actionable insights' with no indication of what is taught; sales of a course or book without described methods."
  },
  {
   "id": "job_interview_focus",
   "kind": "noul",
   "role": "exclude",
   "weight": 1.0,
   "instructions": "Judge the probability that this episode is mainly about job interviews — interview preparation, answering interview questions, hiring, recruiting, resumes or landing a job.",
   "levels": [],
   "does_not_count": "A brief mention of interviews as one of several high-pressure situations does not make the episode mainly about job interviews; neither does an episode that simply features an interview-format conversation."
  }
 ]
}


def cmd_guide():
    print(GUIDE.format(rules=PLAN_RULES, example=json.dumps(EXAMPLE_PLAN, indent=1, ensure_ascii=False), root=ROOT))


GUIDE = """# podcast-search: guide for agents

Finds episodes in podcast back catalogues. You write a plan of 1-4 questions; TypeSafe's Jev answers
them for every episode, one episode at a time (so list order never matters); the tool ranks the results.
Answers are cached, so re-running an unchanged question is free and instant.

## Workflow

1. `podcast-search list --json` - podcasts in the library and saved plans.
   Missing podcast: `podcast-search add "<name or feed URL>"`. It prints the match it picked and any
   alternatives; check it is the show the user means.
2. A saved plan that fits the request? `podcast-search search --plan NAME --podcast SLUG --json`.
3. Otherwise write a plan (rules and format below), save it to a file, and run
   `podcast-search search --plan-file plan.json --podcast SLUG --json` (or pipe it in with `--plan-file -`).
   Briefly tell the user what the questions are, so they can correct the interpretation.
4. Read `results`: `rank_score` orders them; `answers` shows each question's score or probability.
   To explain or check picks, read the summaries: `podcast-search episode 1-5 --json`. Explain them
   yourself; flag any that only partly fit.
5. Refine on feedback. Changing weights or roles re-ranks from cache for free. Changing a question's
   wording, levels or does_not_count asks Jev again for every episode (about 20s and 1-2 cents per 300).
6. If the user is happy with the plan, re-run with `--save-plan NAME` so it can be reused.
7. Spotify: ask the user before saving anything to their library. Then `podcast-search save 1-5 --json`;
   each result is "confirmed", "not_confirmed" or "no_spotify_match". If it says the user is not signed
   in, they must run `podcast-search spotify-login` themselves (it opens a browser).

`--podcast` takes a slug or unique prefix and can be omitted when the library holds one podcast.

## Reading results

- Use the order, not absolute thresholds: higher scores reliably mean better matches, but the numbers
  themselves are not calibrated for a newly written question.
- Identical calls can differ by a few hundredths; don't treat small gaps between neighbours as meaningful.
- An exclude/require probability near 0.5 means Jev genuinely can't tell, not "somewhat".

## Writing a plan

{rules}

## Plan format

A JSON object with "questions" (and optionally "interpretation" and "query"):

{example}

Fields per question: id (your label), kind (score|noul), role (rank|exclude|require), weight (rank
questions; relative), instructions, levels (score: 2-10 strings, lowest = no match; noul: []),
does_not_count. Plans are validated before anything is sent; errors say what to fix.

## Where things live

{root}: library/ (episodes), plans/ (saved plans), cache.sqlite (Jev answers), .env (keys),
last_search.json (the latest results, used by `episode` and `save`).
"""


def main():
    global JSON_OUT
    load_env()
    ap = argparse.ArgumentParser(prog="podcast-search", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("guide", help="how to use this tool well - agents start here")
    a = sub.add_parser("add", help="add a podcast by feed URL or name, or refresh one")
    a.add_argument("feed")
    ls = sub.add_parser("list", help="podcasts, stored answers and saved plans")
    ls.add_argument("--json", action="store_true")
    s = sub.add_parser("search", help="search a podcast")
    s.add_argument("query", nargs="?", help="free-form request; Claude writes the plan unless --plan/--plan-file")
    s.add_argument("--podcast", help="podcast slug (or unique prefix); optional with one podcast")
    s.add_argument("--plan", help="use a saved plan")
    s.add_argument("--plan-file", metavar="PATH", help="use a plan from a JSON file, or - for stdin")
    s.add_argument("--save-plan", metavar="NAME", help="save this run's plan under NAME")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--explain", action="store_true", help="Claude explains the top 5 (needs ANTHROPIC_API_KEY)")
    s.add_argument("--json", action="store_true")
    ep = sub.add_parser("episode", help="details for results of the last search, e.g. 3 or 1-5")
    ep.add_argument("ranks")
    ep.add_argument("--json", action="store_true")
    sub.add_parser("spotify-login", help="sign in to Spotify (opens your browser)")
    lk = sub.add_parser("spotify-link", help="match a podcast's episodes to Spotify")
    lk.add_argument("--podcast")
    sv = sub.add_parser("save", help="save results of the last search to Spotify, e.g. 1-5")
    sv.add_argument("ranks")
    sv.add_argument("--json", action="store_true")
    args = ap.parse_args()
    JSON_OUT = getattr(args, "json", False)

    if args.cmd == "guide":
        cmd_guide()
    elif args.cmd == "add":
        library.add(args.feed)
    elif args.cmd == "list":
        cmd_list(AnswerCache())
    elif args.cmd == "search":
        if args.plan and args.plan_file:
            s.error("use --plan or --plan-file, not both")
        if not (args.query or args.plan or args.plan_file):
            s.error("give a query, --plan NAME or --plan-file PATH")
        cmd_search(args, AnswerCache())
    elif args.cmd == "episode":
        cmd_episode(args)
    elif args.cmd == "save":
        cmd_save(args)
    else:
        from . import spotify
        spotify.login() if args.cmd == "spotify-login" else spotify.link(args.podcast)


if __name__ == "__main__":
    main()
