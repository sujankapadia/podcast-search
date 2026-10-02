"""Stored Jev answers, so a question is never asked twice about the same episode.

An answer is reused only when all of these match:
  podcast + episode guid     which episode
  content_hash               the episode text Jev saw (edits trigger a re-judge)
  question hash              exactly what Jev was asked: kind, wording, levels, exclusions
                             (not the weight or role - they don't change Jev's answer)
  model                      the pinned Jev version
Answers are stored per question, so plans that share a question share its answers.
"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .config import ROOT

DB = ROOT / "cache.sqlite"


def question_hash(q: dict) -> str:
    sent = {k: q[k] for k in ("kind", "instructions", "levels", "does_not_count")}
    return hashlib.sha256(json.dumps(sent, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


class AnswerCache:
    def __init__(self, path: Path = DB):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS answers (
            podcast TEXT, episode TEXT, content_hash TEXT, question TEXT, model TEXT,
            answer TEXT, created TEXT,
            PRIMARY KEY (podcast, episode, content_hash, question, model))""")

    def lookup(self, podcast: str, model: str, questions: set[str]) -> dict:
        if not questions:
            return {}
        marks = ",".join("?" * len(questions))
        rows = self.db.execute(
            f"SELECT episode, content_hash, question, answer FROM answers "
            f"WHERE podcast = ? AND model = ? AND question IN ({marks})", (podcast, model, *questions))
        return {(ep, ch, q): json.loads(a) for ep, ch, q, a in rows}

    def store(self, rows: list[tuple]):
        """rows: (podcast, episode, content_hash, question, model, answer_dict)"""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.db.executemany("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?,?)",
                            [(*r[:5], json.dumps(r[5]), now) for r in rows])
        self.db.commit()

    def count(self, podcast: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM answers WHERE podcast = ?", (podcast,)).fetchone()[0]
