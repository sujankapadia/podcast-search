"""Baseline: one Claude call that scores every episode on a saved plan's questions.

The top-10 baseline (one_call.py) returns a finished list. This one asks Claude for a
per-episode answer to each question in the plan, so its output can be stored and
re-ranked the same way Jev's answers are. It measures time, cost and completeness, and
compares the resulting ranking with Jev's stored answers for the same plan.

Usage: uv run python -m podcast_search.one_call_scores --plan NAME [--podcast SLUG] [--reversed]
"""
import argparse
import json
import time

import anthropic

from . import library
from .cache import AnswerCache, question_hash
from .cli import CLAUDE_MODEL, ENGINES, load_plan, rank
from .config import ROOT, load_env

PRICE_IN, PRICE_OUT = 5.00, 25.00  # $/MTok, Claude Opus 5


def describe(q: dict) -> str:
    if q["kind"] == "score":
        levels = "\n".join(f"  {i}: {lvl}" for i, lvl in enumerate(q["levels"]))
        answer = (f"a number from 0 to {len(q['levels']) - 1}; use a decimal between two levels "
                  f"if the episode falls between them")
        return f"{q['id']} ({answer})\n{q['instructions']}\nLevels:\n{levels}\nDoes not count: {q['does_not_count']}"
    return (f"{q['id']} (the probability from 0 to 1 that this is true)\n{q['instructions']}\n"
            f"Does not count: {q['does_not_count']}")


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--podcast")
    ap.add_argument("--reversed", action="store_true", help="list the episodes in reverse order")
    args = ap.parse_args()

    query, plan = load_plan(args.plan)
    pod = library.load(args.podcast)
    episodes = pod["episodes"]
    qs = plan["questions"]

    seq = list(enumerate(episodes))
    if args.reversed:
        seq.reverse()
    listing = "\n\n".join(f"[{i}] {e['title']}\n{e['summary']}" for i, e in seq)
    questions = "\n\n".join(describe(q) for q in qs)
    schema = {
        "type": "object",
        "properties": {"episodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, **{q["id"]: {"type": "number"} for q in qs}},
            "required": ["id"] + [q["id"] for q in qs], "additionalProperties": False}}},
        "required": ["episodes"], "additionalProperties": False,
    }

    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system="You judge podcast episodes from their title and summary only.",
        messages=[{"role": "user", "content":
                   f"Here are all {len(episodes)} episodes of the podcast, each with an id in brackets.\n\n{listing}\n\n"
                   f"Answer these questions for EVERY episode, judging each episode on its own title and summary:\n\n"
                   f"{questions}\n\nReturn one entry per episode, for all {len(episodes)} episodes, using the bracketed ids."}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    ) as stream:
        msg = stream.get_final_message()
    seconds = time.perf_counter() - t0
    if msg.stop_reason == "refusal":
        raise SystemExit("Claude declined.")
    u = msg.usage
    cost = u.input_tokens * PRICE_IN / 1e6 + u.output_tokens * PRICE_OUT / 1e6
    rows = json.loads(next(b.text for b in msg.content if b.type == "text"))["episodes"]

    # completeness and validity
    by_id, dupes, bad = {}, 0, 0
    for r in rows:
        if r["id"] in by_id:
            dupes += 1
        by_id[r["id"]] = r
    for r in by_id.values():
        for q in qs:
            hi = len(q["levels"]) - 1 if q["kind"] == "score" else 1
            if not 0 <= r[q["id"]] <= hi:
                bad += 1
    missing = [i for i in range(len(episodes)) if i not in by_id]
    print(f"Claude ({CLAUDE_MODEL}, {'reversed' if args.reversed else 'feed'} order): {seconds:.0f}s, "
          f"{u.input_tokens:,} in / {u.output_tokens:,} out, ${cost:.3f}, stop={msg.stop_reason}")
    print(f"returned {len(rows)} rows for {len(episodes)} episodes: {len(missing)} missing, {dupes} duplicates, "
          f"{bad} out-of-range values")

    # rank Claude's answers with the plan's own formula, in the same shape as Jev's
    def as_answer(q, v):
        return {"score": v, "max": len(q["levels"]) - 1} if q["kind"] == "score" else {"p": v}
    claude = [{q["id"]: as_answer(q, by_id[i][q["id"]]) for q in qs} if i in by_id else None
              for i in range(len(episodes))]
    keep = [i for i in range(len(episodes)) if claude[i]]
    c_rank = rank([episodes[i] for i in keep], [claude[i] for i in keep], plan)

    # Jev's stored answers for the same plan and the same episode text
    qh = {q["id"]: question_hash(q) for q in qs}
    known = AnswerCache(ROOT / "cache.sqlite").lookup(pod["slug"], ENGINES["jev"]["model"], set(qh.values()))
    jev = [{q["id"]: known.get((e["guid"], e["content_hash"], qh[q["id"]])) for q in qs} for e in episodes]
    if any(None in a.values() for a in jev):
        print("Jev answers missing for some episodes: run `podcast-search search --plan "
              f"{args.plan} --podcast {pod['slug']}` first"); return
    j_rank = rank(episodes, jev, plan)

    num = {e["guid"]: e["title"].split(".")[0] if e["title"][:1].isdigit() else e["title"][:20] for e in episodes}
    cj = [e["guid"] for _, e, _ in j_rank]
    cc = [e["guid"] for _, e, _ in c_rank]
    print("\nJev top 10:   ", [num[g] for g in cj[:10]])
    print("Claude top 10:", [num[g] for g in cc[:10]])
    for k in (5, 10, 20):
        print(f"top-{k} overlap: {len(set(cj[:k]) & set(cc[:k]))}/{k}")

    def spearman(x, y):
        def rk(v):
            order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[order[k]] = (i + j) / 2
                i = j + 1
            return r
        a, b = rk(x), rk(y); n = len(a); ma, mb = sum(a) / n, sum(b) / n
        cov = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        return cov / (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    val = lambda a: a.get("score", a.get("p"))
    for q in qs:
        xs = [val(claude[i][q["id"]]) for i in keep]; ys = [val(jev[i][q["id"]]) for i in keep]
        print(f"rank correlation with Jev, {q['id']}: {spearman(xs, ys):.2f}")

    prev = ROOT / "last_one_call.json"
    if prev.exists():
        d = json.loads(prev.read_text())
        if d["query"] == query:
            g = {i: e["guid"] for i, e in enumerate(episodes)}
            for r in d["runs"]:
                t = {g[x["id"]] for x in r["top"]}
                print(f"overlap with the top-10 call ({r['order']} order): {len(t & set(cc[:10]))}/10")

    (ROOT / f"last_one_call_scores{'_reversed' if args.reversed else ''}.json").write_text(json.dumps(
        {"plan": args.plan, "order": "reversed" if args.reversed else "feed", "seconds": seconds,
         "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "cost": cost,
         "missing": missing, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
