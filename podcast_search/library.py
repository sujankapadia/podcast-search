"""Podcast library: add a feed and keep its episodes on disk.

Each podcast is keyed by its feed URL. On disk it gets a readable name: the
title slug plus a short hash of the feed URL, so two shows with the same title
never collide. Each episode is keyed by its RSS guid (the feed's own permanent
id), and carries a hash of its cleaned text so edited descriptions get re-judged.
"""
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from .config import ROOT

LIBRARY = ROOT / "library"
UA = {"User-Agent": "Mozilla/5.0 (podcast-search prototype)"}
NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd", "content": "http://purl.org/rss/1.0/modules/content/"}

# Show-notes clean-up: keep the summary and chapter titles; drop links, ads, sponsors and timestamps.
# Everything from the first marker on is removed (case-insensitive), in the summary and in the chapter list.
CUT_MARKERS = (
    # Think Fast Talk Smart
    "Printable Expanded", "Episode Reference Links", "Connect:", "********", "Thank you to our sponsors",
    # Money For Couples
    "This episode is brought to you by", "This episode is also brought to you by", "This episode brought to you by",
    "Links mentioned in this episode", "PODCAST NEWSLETTER", "Apply to be coached",
)
# Sponsor sentences in the middle of a summary, with real content after them, are removed one by one.
PROMO_SENTENCE = re.compile(r"https?://|promo code|\d+% off|sponsoring this episode|sponsored by|brought to you by", re.I)
GENERIC_CHAPTERS = re.compile(r"introduction|conclusion|final (three )?questions|wrap", re.I)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _cut(text: str) -> str:
    low = text.lower()
    hits = [low.find(m.lower()) for m in CUT_MARKERS if m.lower() in low]
    return text[:min(hits)] if hits else text


def _drop_promo(text: str) -> str:
    return " ".join(x for x in re.split(r"(?<=[.!?])\s+", text) if not PROMO_SENTENCE.search(x))


def clean(desc: str) -> str:
    chapters = ""
    if "Chapters:" in desc:
        desc, raw = desc.split("Chapters:", 1)
        titles = [t.strip(" -") for t in re.split(r"\(\d{2}:\d{2}(?::\d{2})?\)", _cut(raw)) if t.strip(" -")]
        titles = [t for t in titles if not GENERIC_CHAPTERS.search(t)]
        if titles:
            chapters = " Chapters: " + "; ".join(titles) + "."
    return (_drop_promo(_cut(desc)).strip() + chapters).strip()


def slugify(title: str, feed_url: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32].rstrip("-") or "podcast"
    return f"{base}-{hashlib.sha1(feed_url.encode()).hexdigest()[:6]}"


def find_feed(name_or_url: str) -> str:
    """A URL is used as-is; anything else is looked up in the Apple Podcasts directory."""
    if name_or_url.startswith(("http://", "https://")):
        return name_or_url
    q = urllib.parse.urlencode({"term": name_or_url, "entity": "podcast", "limit": 5})
    res = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://itunes.apple.com/search?{q}", headers=UA)).read())
    hits = [r for r in res["results"] if r.get("feedUrl")]
    if not hits:
        raise SystemExit(f"no podcast found for '{name_or_url}'")
    for r in hits[1:]:
        print(f"  (also found: {r['collectionName']} - {r['feedUrl']})")
    print(f"using: {hits[0]['collectionName']} by {hits[0]['artistName']}")
    return hits[0]["feedUrl"]


def add(name_or_url: str) -> str:
    """Fetch (or refresh) a podcast feed. Returns the podcast's slug."""
    feed_url = find_feed(name_or_url)
    root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(feed_url, headers=UA)).read())
    channel = root.find("channel")
    title = _text(channel.findtext("title")) or feed_url
    slug = slugify(title, feed_url)
    episodes = []
    for it in channel.iter("item"):
        ep_title = _text(it.findtext("title"))
        date = (it.findtext("pubDate") or "")[:16]
        raw = _text(it.findtext("description") or it.findtext("content:encoded", namespaces=NS))
        summary = clean(raw)
        guid = (it.findtext("guid") or "").strip() or hashlib.sha1(f"{ep_title}|{date}".encode()).hexdigest()
        episodes.append({
            "guid": guid, "title": ep_title, "date": date, "link": it.findtext("link"),
            "summary": summary,
            "content_hash": hashlib.sha1(f"{ep_title}\n{summary}".encode()).hexdigest()[:16],
        })
    path = LIBRARY / slug / "podcast.json"
    before = {e["guid"] for e in json.loads(path.read_text())["episodes"]} if path.exists() else set()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"slug": slug, "title": title, "feed_url": feed_url,
                                "fetched": time.strftime("%Y-%m-%d %H:%M"), "episodes": episodes}, indent=1))
    new = sum(e["guid"] not in before for e in episodes)
    print(f"{'refreshed' if before else 'added'} {title}: {len(episodes)} episodes"
          + (f", {new} new" if before else "") + f"  [{slug}]")
    return slug


def podcasts() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(LIBRARY.glob("*/podcast.json"))]


def load(slug: str | None) -> dict:
    """Load a podcast by slug or unique slug prefix; with one podcast in the library, slug may be omitted."""
    all_ = podcasts()
    if not all_:
        raise SystemExit("library is empty - run: search.py add <feed url or podcast name>")
    if slug is None:
        if len(all_) == 1:
            return all_[0]
        raise SystemExit("several podcasts in the library - pick one with --podcast:\n  " +
                         "\n  ".join(p["slug"] for p in all_))
    matches = [p for p in all_ if p["slug"] == slug] or [p for p in all_ if p["slug"].startswith(slug)]
    if len(matches) != 1:
        raise SystemExit(f"'{slug}' matches {len(matches)} podcasts: " + ", ".join(p["slug"] for p in matches))
    return matches[0]
