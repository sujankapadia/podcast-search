"""Baseline: one Claude call with every episode in context, asked for the top matches.

Runs the same query with the episode list in two orders (feed order and reversed).
If position in a long context matters, the two top-10 lists will differ.

Usage: uv run python -m podcast_search.one_call "your query" [--podcast SLUG] [--top 10]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from . import library
from .cli import CLAUDE_MODEL
from .config import ROOT, load_env

HERE = ROOT
PRICE_IN, PRICE_OUT = 5.00, 25.00  # $/MTok, Claude Opus 5

SCHEMA = {
    "type": "object",
    "properties": {"top": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
        "required": ["id", "reason"], "additionalProperties": False}}},
    "required": ["top"], "additionalProperties": False,
}


def ask(client, query, episodes, order, k):
    seq = list(enumerate(episodes))
    if order == "reversed":
        seq.reverse()
    listing = "\n\n".join(f"[{i}] {e['title']}\n{e['summary']}" for i, e in seq)
    pos = {i: n for n, (i, _) in enumerate(seq)}           # where each episode sat in the prompt
    t0 = time.perf_counter()
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system="You help a listener find podcast episodes that match what they're looking for.",
        messages=[{"role": "user", "content":
                   f"Here are all {len(episodes)} episodes of the podcast, each with an id in brackets.\n\n{listing}\n\n"
                   f"Search request: {query}\n\nReturn the {k} best-matching episodes, best first, "
                   f"using their bracketed ids, with one short reason each."}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    ) as stream:
        msg = stream.get_final_message()
    el = time.perf_counter() - t0
    if msg.stop_reason == "refusal":
        raise SystemExit("Claude declined.")
    top = json.loads(next(b.text for b in msg.content if b.type == "text"))["top"][:k]
    u = msg.usage
    cost = u.input_tokens * PRICE_IN / 1e6 + u.output_tokens * PRICE_OUT / 1e6
    return {"order": order, "seconds": el, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cost": cost, "top": [{**t, "prompt_position": pos[t["id"]] / len(episodes)} for t in top]}


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--podcast")
    args = ap.parse_args()
    episodes = library.load(args.podcast)["episodes"]
    client = anthropic.Anthropic()

    with ThreadPoolExecutor(2) as ex:
        runs = list(ex.map(lambda o: ask(client, args.query, episodes, o, args.top), ("feed", "reversed")))

    for r in runs:
        print(f"\n=== {r['order']} order: {r['seconds']:.0f}s, {r['input_tokens']:,} in / {r['output_tokens']:,} out, ${r['cost']:.3f}")
        for n, t in enumerate(r["top"], 1):
            print(f"{n:2d}. [{t['id']:3d}] pos {t['prompt_position']:4.0%}  {episodes[t['id']]['title'][:72]}")

    a, b = ([t["id"] for t in r["top"]] for r in runs)
    print(f"\nfeed vs reversed: {len(set(a) & set(b))}/{args.top} episodes in common, "
          f"{sum(x == y for x, y in zip(a, b))} in the same rank")

    jev_file = HERE / "last_search.json"
    if jev_file.exists():
        jev = json.loads(jev_file.read_text())
        if jev["query"] == args.query:
            title_to_id = {e["title"]: i for i, e in enumerate(episodes)}
            j = [title_to_id[x["title"]] for x in jev["ranked"][: args.top]]
            print(f"Jev top {args.top} vs Claude (feed): {len(set(j) & set(a))} in common;  "
                  f"vs Claude (reversed): {len(set(j) & set(b))} in common;  in all three: {len(set(j) & set(a) & set(b))}")
            print("Where Claude's picks sit in Jev's ranking:",
                  sorted({next(n for n, x in enumerate(jev['ranked'], 1) if title_to_id[x['title']] == i) for i in set(a) | set(b)}))

    (HERE / "last_one_call.json").write_text(json.dumps({"query": args.query, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
