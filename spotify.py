"""Spotify: sign in once, match RSS episodes to Spotify episodes, save picks to Your Episodes.

Sign-in uses Authorization Code with PKCE (no client secret). The token is kept in
.spotify_token.json (git-ignored) and refreshed automatically.

Matching: Spotify needs its own episode ids, so each show's Spotify episodes are listed
once and matched to RSS episodes by title (exact, then without a leading episode number),
falling back to release date plus a close title. The mapping is stored next to the
podcast in library/<slug>/spotify.json; unmatched episodes are listed, never guessed.
"""
import base64
import difflib
import hashlib
import http.server
import json
import os
import re
import secrets
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import library

HERE = Path(__file__).parent
TOKEN = HERE / ".spotify_token.json"
REDIRECT = "http://127.0.0.1:8888/callback"
# market=from_token now needs more scope than this (403 "Insufficient client scope"), so no market param is sent.
SCOPES = "user-library-modify user-library-read user-read-playback-position"
API = "https://api.spotify.com/v1"


def client_id() -> str:
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    if not cid:
        raise SystemExit("SPOTIFY_CLIENT_ID is not set (run with --env-file .env)")
    return cid

# ---------------------------------------------------------------- sign-in (PKCE)

def _token_request(form: dict) -> dict:
    req = urllib.request.Request("https://accounts.spotify.com/api/token",
                                 data=urllib.parse.urlencode(form).encode(),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        tok = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Spotify token request failed ({e.code}): {e.read().decode()[:300]}")
    tok["expires_at"] = time.time() + tok.get("expires_in", 3600) - 60
    return tok


def _save_token(tok: dict, previous: dict | None = None):
    if previous and "refresh_token" not in tok:          # refreshes may omit it; keep the old one
        tok["refresh_token"] = previous["refresh_token"]
    TOKEN.write_text(json.dumps(tok))
    TOKEN.chmod(0o600)


def login(timeout: int = 300):
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
        "client_id": client_id(), "response_type": "code", "redirect_uri": REDIRECT, "scope": SCOPES,
        "state": state, "code_challenge_method": "S256", "code_challenge": challenge})
    got = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got and got.get("state") == state
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(("<h2>Signed in to Spotify. You can close this tab.</h2>" if ok else
                              f"<h2>Sign-in did not complete: {got.get('error', 'unknown error')}</h2>").encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 8888), Handler)
    server.timeout = 1
    print("Opening Spotify in your browser. If it doesn't open, visit:\n" + url, flush=True)
    webbrowser.open(url)
    deadline = time.time() + timeout
    while not got and time.time() < deadline:
        server.handle_request()
    server.server_close()
    if got.get("state") != state or "code" not in got:
        raise SystemExit(f"sign-in failed: {got.get('error', 'timed out waiting for the browser')}")
    tok = _token_request({"grant_type": "authorization_code", "code": got["code"], "redirect_uri": REDIRECT,
                          "client_id": client_id(), "code_verifier": verifier})
    _save_token(tok)
    print("signed in; token saved to .spotify_token.json")


def access_token() -> str:
    if not TOKEN.exists():
        raise SystemExit("not signed in to Spotify - run: search.py spotify-login")
    tok = json.loads(TOKEN.read_text())
    if time.time() >= tok["expires_at"]:
        new = _token_request({"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                              "client_id": client_id()})
        _save_token(new, tok)
        tok = new
    return tok["access_token"]

# ---------------------------------------------------------------- API helper

def api(method: str, path: str, params: dict | None = None, retry: bool = True):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {access_token()}"})
    try:
        body = urllib.request.urlopen(req).read()
        return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        if e.code == 429 and retry:
            time.sleep(int(e.headers.get("Retry-After", "2")) + 1)
            return api(method, path, params, retry=False)
        raise SystemExit(f"Spotify {method} {path} failed ({e.code}): {e.read().decode()[:400]}")

# ---------------------------------------------------------------- matching RSS episodes to Spotify

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _no_number(t: str) -> str:
    return re.sub(r"^(ep(isode)?\s*)?\d+\s+", "", t)


def _rss_date(d: str):
    try:
        return datetime.strptime(d.strip(), "%a, %d %b %Y").date()
    except ValueError:
        return None


def find_show(title: str) -> dict:
    res = api("GET", "/search", {"q": title, "type": "show", "limit": 10})
    shows = [s for s in res["shows"]["items"] if s]
    if not shows:
        raise SystemExit(f"no Spotify show found for '{title}'")
    exact = [s for s in shows if _norm(s["name"]) == _norm(title)]
    best = exact[0] if exact else max(shows, key=lambda s: difflib.SequenceMatcher(None, _norm(s["name"]), _norm(title)).ratio())
    print(f"Spotify show: {best['name']} ({best.get('total_episodes', '?')} episodes)")
    return best


def show_episodes(show_id: str) -> list[dict]:
    out, offset = [], 0
    while True:
        page = api("GET", f"/shows/{show_id}/episodes", {"limit": 50, "offset": offset})
        out += [e for e in page["items"] if e]
        if not page.get("next"):
            return out
        offset += 50


def link(slug: str | None) -> dict:
    """Match a podcast's RSS episodes to Spotify and store the mapping."""
    pod = library.load(slug)
    show = find_show(pod["title"])
    sp = show_episodes(show["id"])
    by_title, by_short = {}, {}
    for e in sp:
        by_title.setdefault(_norm(e["name"]), e)
        by_short.setdefault(_no_number(_norm(e["name"])), e)
    mapping, unmatched, how = {}, [], {"title": 0, "title without number": 0, "date + similar title": 0}
    for ep in pod["episodes"]:
        n = _norm(ep["title"])
        hit, kind = by_title.get(n), "title"
        if not hit:
            hit, kind = by_short.get(_no_number(n)), "title without number"
        if not hit:
            d = _rss_date(ep["date"])
            near = [e for e in sp if d and e.get("release_date") and
                    abs(datetime.strptime(e["release_date"][:10], "%Y-%m-%d").date() - d) <= timedelta(days=1)]
            scored = [(difflib.SequenceMatcher(None, n, _norm(e["name"])).ratio(), e) for e in near]
            scored = [s for s in scored if s[0] >= 0.6]
            if scored:
                hit, kind = max(scored, key=lambda s: s[0])[1], "date + similar title"
        if hit:
            mapping[ep["guid"]] = {"id": hit["id"], "uri": hit["uri"], "name": hit["name"], "matched_by": kind}
            how[kind] += 1
        else:
            unmatched.append(ep["title"])
    data = {"show_id": show["id"], "show_name": show["name"], "linked": time.strftime("%Y-%m-%d %H:%M"),
            "episodes": mapping, "unmatched": unmatched}
    (library.LIBRARY / pod["slug"] / "spotify.json").write_text(json.dumps(data, indent=1))
    print(f"matched {len(mapping)}/{len(pod['episodes'])} episodes  ({', '.join(f'{v} by {k}' for k, v in how.items())})"
          f"; Spotify lists {len(sp)}")
    for t in unmatched[:10]:
        print(f"  unmatched: {t[:80]}")
    if len(unmatched) > 10:
        print(f"  ... and {len(unmatched) - 10} more")
    return data


def mapping_for(slug: str) -> dict:
    f = library.LIBRARY / slug / "spotify.json"
    return json.loads(f.read_text()) if f.exists() else link(slug)

# ---------------------------------------------------------------- saving

def save(slug: str, episodes: list[dict]):
    """Save RSS episodes (dicts with guid/title) to the user's Your Episodes, then confirm with Spotify
    that each one is actually in the library - a 200 from the save call alone isn't treated as proof."""
    m = mapping_for(slug)["episodes"]
    found = [(e, m[e["guid"]]) for e in episodes if e["guid"] in m]
    missing = [e for e in episodes if e["guid"] not in m]
    confirmed = []
    for i in range(0, len(found), 40):                      # both endpoints take at most 40 URIs
        uris = ",".join(sp["uri"] for _, sp in found[i:i + 40])
        api("PUT", "/me/library", {"uris": uris})
        confirmed += api("GET", "/me/library/contains", {"uris": uris})
    for (e, sp), ok in zip(found, confirmed):
        status = "saved, confirmed in library" if ok else "NOT in library after saving"
        print(f"  {status:30s} {e['title'][:62]}\n  {'':30s} https://open.spotify.com/episode/{sp['id']}")
    for e in missing:
        print(f"  {'no Spotify match, skipped':30s} {e['title'][:62]}")
    n_ok = sum(confirmed)
    if n_ok < len(found):
        print(f"  warning: {len(found) - n_ok} episode(s) did not show up in your library")
    return n_ok
