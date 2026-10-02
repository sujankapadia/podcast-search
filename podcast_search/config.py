"""Where podcast-search keeps its data, and loading its .env.

Data (library/, plans/, cache.sqlite, .env, the Spotify token, last_search.json)
lives in the project root, so the installed command finds it from any folder.
Set PODCAST_SEARCH_HOME to keep it somewhere else.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PODCAST_SEARCH_HOME") or Path(__file__).resolve().parents[1])


def load_env():
    """Read ROOT/.env without overriding variables already set in the environment."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
