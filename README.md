# podcast-search

Search a podcast's back catalogue for episodes that match what you're looking for, then save the best ones to Spotify.

```
your query ──► Claude: writes the questions (once per query, or reuse a saved plan)
                    │
episodes (RSS) ──► Jev: judges each episode on its own, in parallel
                    │
              code: ranks ──► Spotify: save picks to Your Episodes
```

Claude turns a free-form request into 1–4 narrow questions for [TypeSafe's Jev](https://docs.typesafe.ai) model, which judges every episode independently and returns scores rather than text. Results are ranked in code, every answer is cached, and saved plans can be rerun without calling Claude.

## Setup

1. `uv sync`
2. Create `.env`:
   ```
   TYPESAFE_API_KEY=...
   SPOTIFY_CLIENT_ID=...      # only needed for saving to Spotify
   ```
   `ANTHROPIC_API_KEY` is read from the environment (or add it to `.env`). It's only needed when Claude writes a new plan.
3. For Spotify: create an app at developer.spotify.com/dashboard with the redirect URI `http://127.0.0.1:8888/callback`, put its Client ID in `.env`, then run `spotify-login` once. Development Mode requires Spotify Premium.

## Commands

```
uv run search.py add "podcast name or feed URL"          # add a podcast, or refresh it
uv run search.py list                                    # podcasts, cached answers, saved plans

uv run --env-file .env search.py search "your query" [--podcast SLUG] [--save-plan NAME] [--top 10] [--explain]
uv run --env-file .env search.py search --plan NAME [--podcast SLUG]     # reuse a saved plan, no Claude call

uv run --env-file .env search.py spotify-login           # once
uv run --env-file .env search.py spotify-link            # match episodes to Spotify (runs automatically on first save)
uv run --env-file .env search.py save 1-5                # save results from the last search, confirmed against your library

uv run --env-file .env one_call.py "your query"          # baseline: one Claude call over the whole catalogue
```

## How answers are cached

An answer is reused when the podcast, episode (RSS guid), episode text, exact question and Jev model version all match. Weights aren't part of the key, so changing them re-ranks instantly without calling Jev. A repeat search costs nothing, and a new episode costs one call per saved plan.

## Files

| Path | What | In git |
|---|---|---|
| `search.py`, `library.py`, `cache.py`, `spotify.py`, `one_call.py` | the tool | yes |
| `plans/` | saved searches | yes |
| `library/` | fetched episodes and Spotify mappings | no (rebuilt by `add`) |
| `cache.sqlite` | stored Jev answers | no |
| `.env`, `.spotify_token.json` | secrets | no |
