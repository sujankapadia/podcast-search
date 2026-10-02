"""Search podcast back catalogues: Claude writes the questions, Jev judges every episode.

    LLM (once)   turn the query into 1-4 independent judgments (Scores + Nouls)
    Jev (N)      judge each episode on its own, in parallel - no long context
    cache        every answer is stored; a question is never asked twice about an episode
    code         rank: weighted Score average, softened by exclude/require Nouls

Commands:
  search.py add <feed url | podcast name>          add a podcast, or refresh it
  search.py list                                   podcasts, cached answers, saved plans
  search.py search "query" [--podcast SLUG] [--save-plan NAME] [--top 10] [--explain]
  search.py search --plan NAME [--podcast SLUG]    reuse a saved plan (no Claude call)
  search.py spotify-login                          sign in to Spotify once (browser)
  search.py spotify-link [--podcast SLUG]          match a podcast's episodes to Spotify
  search.py save 1-5                               save ranked results from the last search to Your Episodes

Run with: uv run --env-file .env search.py ...
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import anthropic
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

import library
from cache import AnswerCache, question_hash

HERE = Path(__file__).parent
PLANS = HERE / "plans"
CLAUDE_MODEL = "claude-opus-5"
JEV_MODEL = "jev-1.13.0"   # pinned: cached answers are only valid for the model that gave them
JEV_RATE = 18              # requests/second, under TypeSafe's 1,200/min
JEV_CONCURRENCY = 12

# ---------------------------------------------------------------- step 1: Claude writes the plan

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

PLANNER_SYSTEM = """You turn a listener's search request into questions for Jev, a model that judges ONE podcast \
episode at a time from its title and summary and returns a typed answer, never text.

Jev question types:
- score: places the episode on 3-5 ordered levels you describe. Use for "how much does this episode match".
- noul: probability (0-1) that a statement about the episode is true. Use only for hard yes/no conditions.

Write 1-4 questions, each one narrow, independent judgment. Split a compound request into separate \
questions instead of one question that asks several things.

Only ask what a title and summary can actually show. Summaries describe an episode's topic, guest and \
angle; they rarely list the specific techniques, steps or examples inside it. A question that needs \
detail the summary won't have pushes genuinely on-topic episodes down for missing it.

Make exactly one rank question the core topic of the request, with weight 1.0. Any other preference \
(practicality, tone, format, audience) is secondary: include it only if a summary could plausibly show \
it, and give it a weight of 0.3 or less.

For each question:
- role "rank": a score that orders results. Give it a weight; weights of rank questions are relative.
- role "exclude": a noul whose truth should push an episode down (e.g. "the episode is mainly about X").
- role "require": a noul that must be true for an episode to be useful.
- instructions: the full judgment, readable on its own. Jev never sees the question id.
- levels (score only): concrete, self-contained descriptions from lowest to highest match. The lowest \
  level must mean "not about this at all". Use [] for a noul.
- does_not_count: what should NOT count as a match, to stop superficial keyword hits. Be specific.

Judge only what a title and short summary can show. Do not ask about audio, length, or popularity."""


def make_plan(client: anthropic.Anthropic, query: str) -> dict:
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
    text = next(b.text for b in resp.content if b.type == "text")
    plan = json.loads(text)
    # Guard rails the schema can't express.
    for q in plan["questions"]:
        if q["kind"] == "score" and not 2 <= len(q["levels"]) <= 10:
            raise SystemExit(f"plan error: score {q['id']} has {len(q['levels'])} levels (Jev needs 2-10)")
        if q["kind"] == "noul":
            q["levels"] = []
    if not any(q["role"] == "rank" for q in plan["questions"]):
        raise SystemExit("plan error: no rank question")
    return plan


def jev_question(q: dict):
    instr = {"judgment": q["instructions"], "does_not_count": q["does_not_count"]}
    return Score(instructions=instr, criteria=q["levels"]) if q["kind"] == "score" else Noul(instructions=instr)

# ---------------------------------------------------------------- step 2: Jev judges, reusing stored answers

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

# ---------------------------------------------------------------- step 3: rank in code

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

# ---------------------------------------------------------------- optional: Claude explains the top picks

def explain(client, query, top):
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

def save_plan(name: str, query: str, plan: dict):
    PLANS.mkdir(exist_ok=True)
    path = PLANS / f"{name}.json"
    path.write_text(json.dumps({"name": name, "query": query, "saved": time.strftime("%Y-%m-%d %H:%M"),
                                "plan": plan}, indent=1))
    return path


def load_plan(name: str):
    path = PLANS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no saved plan '{name}' (see --list-plans)")
    d = json.loads(path.read_text())
    return d["query"], d["plan"]


def print_plan(plan):
    for q in plan["questions"]:
        print(f"  [{q['role']}{' w=' + str(q['weight']) if q['role'] == 'rank' else ''}] {q['kind']} {q['id']}: {q['instructions']}")
        for i, lv in enumerate(q["levels"]):
            print(f"        {i}: {lv}")
        print(f"        not: {q['does_not_count']}")


def cmd_list(cache):
    for p in library.podcasts():
        print(f"{p['slug']:44s} {len(p['episodes']):4d} episodes  {cache.count(p['slug']):6,} stored answers   {p['title']}")
    for f in sorted(PLANS.glob("*.json")) if PLANS.exists() else []:
        d = json.loads(f.read_text())
        print(f"plan  {d['name']:24s} {d['saved']}  {d['query']}")


def cmd_search(args, cache):
    pod = library.load(args.podcast)
    episodes = pod["episodes"]
    claude = None
    if args.plan:
        query, plan = load_plan(args.plan)
        print(f"\nPODCAST {pod['title']}\nQUERY   {query}\nPLAN    saved plan '{args.plan}' (no Claude call)\n")
    else:
        query = args.query
        claude = anthropic.Anthropic()
        t0 = time.perf_counter()
        plan = make_plan(claude, query)
        print(f"\nPODCAST {pod['title']}\nQUERY   {query}\nCLAUDE  {plan['interpretation']}  ({time.perf_counter() - t0:.1f}s)\n")
    print_plan(plan)

    t1 = time.perf_counter()
    answers, st = asyncio.run(judge_all(pod["slug"], episodes, plan, cache))
    el = time.perf_counter() - t1
    total = len(episodes) * len(plan["questions"])
    print(f"\nJEV     {total:,} answers needed: {st['from_cache']:,} from cache, {st['asked']:,} asked in "
          f"{st['calls']:,} calls  ({el:.1f}s, {st['tokens']:,} tokens, ${st['tokens'] * 0.042 / 1e6:.4f})\n")

    ranked = rank(episodes, answers, plan)
    ids = [q["id"] for q in plan["questions"]]
    for i, (sc, e, a) in enumerate(ranked[: args.top], 1):
        detail = "  ".join(f"{k}={a[k]['score']:.2f}" if "score" in a[k] else f"{k}={a[k]['p']:.2f}" for k in ids)
        print(f"{i:2d}. {sc:.2f}  {e['title'][:78]}\n      {e['date']}   {detail}")

    out = HERE / "last_search.json"
    out.write_text(json.dumps({"podcast": pod["slug"], "query": query, "plan_name": args.plan, "plan": plan,
                               "ranked": [{"rank_score": sc, "guid": e["guid"], "title": e["title"], "date": e["date"],
                                           "answers": a} for sc, e, a in ranked]}, indent=1))
    if args.save_plan:
        print(f"\nsaved plan -> {save_plan(args.save_plan, query, plan).relative_to(HERE)}")
    if args.explain:
        claude = claude or anthropic.Anthropic()
        print("\nWHY (Claude, top 5)\n" + explain(claude, query, ranked[:5]))
    print(f"\nfull ranking -> {out.relative_to(HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add a podcast by feed URL or name, or refresh one")
    a.add_argument("feed")
    sub.add_parser("list", help="podcasts, stored answers and saved plans")
    s = sub.add_parser("search", help="search a podcast")
    s.add_argument("query", nargs="?")
    s.add_argument("--podcast", help="podcast slug (or unique prefix); optional with one podcast")
    s.add_argument("--plan", help="reuse a saved plan instead of asking Claude")
    s.add_argument("--save-plan", metavar="NAME", help="save this run's plan under NAME")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--explain", action="store_true")
    sub.add_parser("spotify-login", help="sign in to Spotify (opens your browser)")
    lk = sub.add_parser("spotify-link", help="match a podcast's episodes to Spotify")
    lk.add_argument("--podcast")
    sv = sub.add_parser("save", help="save ranked results from the last search to Spotify")
    sv.add_argument("ranks", help="e.g. 1-5 or 1,3,7")
    args = ap.parse_args()

    if args.cmd in ("spotify-login", "spotify-link", "save"):
        import spotify
        if args.cmd == "spotify-login":
            spotify.login()
        elif args.cmd == "spotify-link":
            spotify.link(args.podcast)
        else:
            last = json.loads((HERE / "last_search.json").read_text())
            picks = set()
            for part in args.ranks.split(","):
                a, _, b = part.partition("-")
                picks.update(range(int(a), int(b or a) + 1))
            chosen = [last["ranked"][n - 1] for n in sorted(picks) if 0 < n <= len(last["ranked"])]
            print(f"saving {len(chosen)} episode(s) from the last search ('{last['query'][:60]}')")
            n = spotify.save(last["podcast"], chosen)
            print(f"{n} of {len(chosen)} confirmed in Your Episodes")
        return

    cache = AnswerCache()
    if args.cmd == "add":
        library.add(args.feed)
    elif args.cmd == "list":
        cmd_list(cache)
    else:
        if not (args.query or args.plan):
            s.error("give a query, or --plan NAME")
        cmd_search(args, cache)


if __name__ == "__main__":
    main()
