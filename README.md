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

1. Install the `podcast-search` command globally (editable, so it runs straight from this repo):
   ```
   uv tool install -e .
   ```
2. Create `.env` in this folder. The command reads it from any directory:
   ```
   TYPESAFE_API_KEY=...
   SPOTIFY_CLIENT_ID=...      # only needed for saving to Spotify
   OPENROUTER_API_KEY=...     # only needed for the OpenRouter engines below
   ```
   `ANTHROPIC_API_KEY` (environment or `.env`) is only needed when Claude writes a plan or explains results.
   An agent can write plans itself, with no Claude call.
3. For Spotify: create an app at developer.spotify.com/dashboard with the redirect URI `http://127.0.0.1:8888/callback`,
   put its Client ID in `.env`, then run `podcast-search spotify-login` once. Development Mode requires Spotify Premium.

## Using it from an agent

Tell your agent to use `podcast-search`. It runs `podcast-search guide`, which explains the workflow,
the rules for writing good questions and the plan format; the agent then writes plans itself and reads
`--json` output. The guide and the built-in Claude planner share one copy of the question-writing rules.

## Commands

```
podcast-search guide                                     # how to use the tool well (agents start here)
podcast-search add "podcast name or feed URL"            # add a podcast, or refresh it
podcast-search list [--json]                             # podcasts, cached answers, saved plans

podcast-search search --plan-file plan.json [--json]     # a plan you or your agent wrote ("-" reads stdin)
podcast-search search --plan NAME [--json]               # a saved plan
podcast-search search "your query" [--explain]           # Claude writes the plan
   common options: --podcast SLUG  --save-plan NAME  --top 10
   --engine: jev (default) | mercury, d1, solar, or-jev (OpenRouter) | jebadiah-9b, decider-4b, decider-2b (local)

podcast-search episode 1-5 [--json]                      # summaries and links for results of the last search
podcast-search spotify-login                             # once
podcast-search spotify-link                              # match episodes to Spotify (runs automatically on first save)
podcast-search save 1-5 [--json]                         # save results to Your Episodes, confirmed against your library

uv run python -m podcast_search.one_call "your query"   # baseline: one Claude call over the whole catalogue
```

## How answers are cached

An answer is reused when the podcast, episode (RSS guid), episode text, exact question and model (engine and version) all match. Weights aren't part of the key, so changing them re-ranks instantly without calling Jev. A repeat search costs nothing, and a new episode costs one call per saved plan.

## Files

| Path | What | In git |
|---|---|---|
| `podcast_search/` | the tool (`cli.py` is the command) | yes |
| `plans/` | saved searches | yes |
| `library/` | fetched episodes and Spotify mappings | no (rebuilt by `add`) |
| `cache.sqlite` | stored Jev answers | no |
| `.env`, `.spotify_token.json` | secrets | no |

## License

MIT. See [LICENSE](LICENSE).
