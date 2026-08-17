# CLAUDE.md

Huronalytics — a public baseball leaderboard and a set of derived metrics, built from Google Sheets and the MLB/Savant feeds, published as a static site.

- `pipeline/` — the build. Sheets in, leaderboard JSON and gzipped embeds out. Entry point `python3 -m pipeline.process_data`.
- `stuff_plus/` — the one offline-trained model (xgboost). Runs in CI after the pipeline, injects scores back into the leaderboard JSONs.
- `scrapers/` — feed and Savant pulls, plus the Sheets append path.
- `cards/` — matplotlib pitcher and hitter cards.
- `js/`, `css/`, `index.html`, `trade.html` — the site. Vanilla JS, no build step, no framework.
- `R-scripts/` — the printed pitcher reports.
- `scripts/` — everything that is not the build. `scripts/README.md` is the authority on its layout; read it before adding a file there.

Nested rules live in `pipeline/CLAUDE.md` and `js/CLAUDE.md`. Longer workflows live as skills in `.claude/skills/`. Do not paste them here.

The global `~/.claude/CLAUDE.md` carries the tuning-constant rule (never estimate a constant, sweep it until an interior optimum is bracketed or the curve is proven flat). It applies to every number in this repo. It is not repeated below.

## Ask before you assume

Never guess at intent. If a task leaves anything open — which metric, which season, which player pool, whether a change ships or stays research — stop and ask. One question up front is cheaper than half a day in the wrong direction.

- Ask when the request could reasonably mean two different things.
- Ask before changing a shipped metric's definition, a column set, or anything that lands in `data/*_rs.json`.
- A request is a hypothesis, not an order. If the thing asked for has no value, or the value is already captured by something shipped, say so before building it.
- Do not widen scope past what was asked. Note the adjacent thing you spotted, do not fix it unprompted.
- Do not "clean up" from a snapshot whose age you have not checked.
- If you had to assume something you could not resolve, list it explicitly at the top of your summary.

## The loop

There are two loops here and they are not interchangeable. Pick by what you changed.

### Code changes (refactors, plumbing, moves, new columns)

The proof is byte-identical output, not a test suite.

```bash
# copy scripts/tools/golden_harness somewhere disposable first
python3 capture_inputs.py                          # once: freeze Sheets rows (2m30s)
PYTHONHASHSEED=0 python3 golden_run.py prestate    # ~3s
PYTHONHASHSEED=0 python3 golden_run.py run BASE    # 5m45s
# ... make the change ...
PYTHONHASHSEED=0 python3 golden_run.py run CHANGE  # 5m45s
PYTHONHASHSEED=0 python3 golden_run.py compare BASE CHANGE
```

Budget about 15 minutes for a full verification. Measured 2026-08-17 on 583,619 frozen rows.

Then, on real output:

```bash
python3 -m pipeline.process_data
python3 scripts/ci/validate_output.py
```

Rules:

- Capture, baseline, and verify in the **same session**. Day-boundary cache drift is real, and comparing runs from different days proves nothing.
- `PYTHONHASHSEED=0` on every golden run. Without it the diff is noise.
- A behavior-neutral change that does not reproduce the baseline is not behavior-neutral. Find out why before shipping.
- If the change is intentionally behavior-changing, say what should differ **before** you run the compare, then confirm only that differed.

### Metric changes (a new stat, a re-weighting, a shrinkage constant, a gate)

A within-season holdout is not green. It has passed while the config lost every independent replicate.

- Rebuild self-contained on each of 2021 through 2025, plus the live season. Require the config to win in most of the seasons it was never fitted to.
- Report the objective that decided it, the grid, and the curve. "Best on the grid," "optimal," and "flat, so anything here works" are three different claims.
- Check the objective cannot be gamed by the direction you are searching. Reliability inflates under smoothing toward a constant. That makes it a diagnostic, not an objective.
- Tune at the sample size production actually runs at.
- Validate any replication against a known-shipped value before you trust it.

Rules for both loops:

- Never report success on a red loop. Never loosen a gate or narrow a pool to get a number to pass.
- If a check is wrong, say so and explain why before changing it.
- Do not start long-running processes to "verify." Use the commands above.

## Numeric correctness

There is no type checker here. The boundary discipline has to be explicit instead, because every one of these has silently corrupted output at least once.

- **No silent `fillna`.** A missing value is either meaningful (impute it deliberately, with the frozen constant written down) or a bug. Never `.fillna(0)` to make a computation run.
- **NaN is not zero.** A NaN arm angle, a NaN velocity, and a missing pitch type are different failures and none of them are `0`.
- **Full precision until display.** `runValue`, `xRunValue`, `rv100`, `xRv100` are summed and aggregated at full precision. Rounding happens in the display layer only.
- **Sheet writes use `USER_ENTERED` with a text-forced Count and NaN mapped to blank.** `RAW` plus `str` stores numerics as text, which breaks sorting and renders `<NA>`.
- **Parse the feed, do not trust the shape.** Outs come from per-pitch `count.outs`, not `about.outs`. Pitcher and batter attribution handles mid-PA subs. Savant `pitch_number` counts auto balls and the feed does not.
- **Currency is not universal.** MiLB RunExp is league-denominated in the raw cache. Any pickle reader converts with `compute_runexp_scale` from `pipeline/utils.py`.
- **One home per constant.** `PITCHING_W_STUFF` lives in `pipeline/utils.py:32` and nowhere else. It once ran 70/30 in one file and 80/20 in another for about three weeks.
- **The qualification constants are the exception, and they are paired by hand.** Seven of them exist twice, in `pipeline/utils.py` and in the `QUAL` block of `js/aggregator.js`, because JS cannot import Python. Change one and you change the other in the same commit, or the site colors rows the pipeline did not qualify.
- **Same name does not mean same quantity.** `CELL_SHRINK_K` is 3 in `xwoba3d.py` and 50 in `sdplus.py` and `contact.py`. `HITTER_PRIOR_N` is 190 and 66. `MIN_POOL` counts pitchers in one file and pitches in another. Each was measured independently and each is correct. Do not "unify" them.

Respect these from the start. Do not write loose code and fix it after the output looks wrong.

## Naming

Pick the existing word. Do not coin a new one.

**Domain vocabulary, one word per concept.** This table starts empty on purpose. A row earns its place when a real synonym conflict is found in this repo, not when one seems plausible. An invented "never" is worse than no row, because it reads as authority and can send an agent to "fix" correct code.

| Concept | Use | Never |
| --- | --- | --- |
| _(none yet)_ | | |

Before adding a row, grep for both spellings and confirm the loser is genuinely a mistake rather than a different concept wearing a similar name.

**Code:**

- Metric modules are named for the metric: `locplus.py`, `sdplus.py`, `commandplus.py`.
- Research scripts are `scripts/research/<topic>/<what_it_tests>.py`, run from the repo root.
- Scratch artifacts get a leading underscore: `data/_era_battery.json`. Shipped artifacts do not.
- Shipped leaderboard artifacts end `_rs.json`.
- Scripts compute the repo root from `__file__`. Same-folder imports are bare, cross-folder imports carry their own `sys.path.insert` next to the import.

**Display conventions.** These are wrong in output more often than anything else:

- No leading `+` on positive numbers. Negatives still take `-`. The one exception is a parenthesised vs-baseline delta, which is signed both ways.
- Three-decimal rate stats drop the leading zero: `.318`, not `0.318`. ERA, FIP, and velocity keep theirs.
- Date columns in any CSV deliverable are ISO `yyyy-mm-dd`.
- No em-dashes in prose, copy, or article drafts. Commas, colons, periods.

## Project structure

```
pipeline/          # the build: fetch -> compute -> process_data -> data/
  utils.py         #   shared constants and helpers, single home for weights
  process_data.py  #   entry point, python3 -m pipeline.process_data
stuff_plus/        # xgboost trainer + cached model bundle
scrapers/          # feed/Savant pulls, sheets_append
cards/             # matplotlib pitcher + hitter cards
js/ css/           # the site, no build step
R-scripts/         # printed pitcher reports
scripts/           # everything else, see scripts/README.md
data/              # artifacts. _-prefixed = research scratch
```

- New metric math goes in `pipeline/`. New research goes in `scripts/research/<topic>/`.
- `scripts/ci/` paths are pinned in `.github/workflows/update-leaderboard.yml`. Move them only in lockstep with it.
- `scripts/abs_daily.py` and `scripts/auto-pull.sh` are pinned by launchd. Moving them breaks the daily run.
- Nothing new at the repo root without asking.

## Dependencies

Code is cheap, maintenance is not. Prefer the standard library, then numpy/pandas, then a well-established package.

- `requirements.txt` is the full local environment. CI installs a smaller subset inline in the workflow. Adding a pipeline dependency means editing **both**.
- `xgboost==3.3.0` is pinned exactly on purpose. Model output is version-sensitive.
- Ask before adding a dependency. Never add one as a side effect of another task.
- `.venv/` is load-bearing for the IDE. Never delete it.

## Performance

There is no server, so the budgets are build time and payload.

- **Payload.** Per-visitor bytes and main-thread parse are the real cost. The embed is sharded and split across `data_core` / `data_tables` / `data_heavy` for that reason. Anything that moves data into `data_core` needs a stated reason.
- **CI.** A score-only run is roughly a minute. A retrain is 30 to 40 minutes and runs only on `-f retrain=true`. Do not make a normal run pay retrain cost.
- **Savant pulls** cap at 25k rows per request. Page or narrow the query, do not assume a full result.
- **Vectorize.** Per-row Python over a pitch-level frame is a bug, not a tuning opportunity. These frames run to hundreds of thousands of rows.
- Cache expensive pulls in `data/`, and say in the summary what you cached and what you re-pulled.

## Error handling

The failure mode here is not a crash, it is a silently wrong number reaching the site.

**Fail closed.** On 2026-08-12 a `curl` without `-f` wrote an HTTP error page into the model bundle, and the Stuff+ and Pitching+ columns were blanked in every workbook. Nothing errored.

- Never write a partial artifact. Build to a temp path and move, or do not write.
- A missing input never silently blanks a column. Either abort or preserve the prior value, and log which one happened.
- Every fallback announces itself. `0 Stuff+ grades` in a CI log is the shape of tell that should exist for every degrade path.
- No bare `except:`. No `except Exception: pass`. Catch the specific thing and handle it, or let it propagate.
- Downloads use `curl -sfL --retry 3`. Without `-f` an error body becomes a corrupt file, which surfaces as a confusing decompression error rather than a download failure.
- Error messages name the repair tool. If the fix is `scripts/ops/fix_unformatted_blocks.py --apply`, the message says so.
- Never edit an artifact in `data/` by hand to make a number look right.

## End-to-end verification

After any change that spans the pipeline and the site, exercise it the way a visitor would.

```bash
python3 -m pipeline.process_data
python3 scripts/ci/validate_output.py
# serve the site: preview config "site" in .claude/launch.json, port 8899
```

- Pick a real player you know the numbers for and check the value on the page against the shipped value. Not a synthetic row.
- Check a qualified hitter, an unqualified hitter, and a ROC player. They render differently by design.
- Check the leaderboard, a player page, and the filtered view. Filters reshape team numbers, so a change can be right in one and wrong in another.
- Report what you loaded and what you saw. If a step failed, keep the failure. Do not work around it and call it passing.

## Visual checking

A passing validation says nothing about whether the page or the card looks right.

- Screenshot every card or view you touch and look at it before saying it is done.
- Pitch colors are Okabe-Ito, sinker is amber everywhere. The site and cards share one palette.
- There is no dark palette in `js/`. Do not add one.
- Check: clipped labels, overlapping annotations, a legend that lost a pitch type, an empty state, a player with one pitch type, a player with seven.
- Use a real qualified player and a real unqualified one. Unqualified renders hatched by design, that is not a bug.
- ROC team rows show but stay uncolored. Confirm that still holds after any coloring change.

## Architecture

Read this before exploring. If you find yourself searching for something that belongs here, add it.

Build path: Google Sheets (six per-division workbooks) → `pipeline.fetch` → `pipeline.process_data` → leaderboard JSONs plus `data_core` / `data_tables` / `data_heavy` gzipped embeds plus shards → static site.

Stuff+ is trained offline and runs after the pipeline in CI, injecting scores into the leaderboard JSONs, after which `rebuild_embed.py` swaps them into `data_core.json.gz`. Normal runs score with a cached fold bundle, which is leakage-free for new pitches. A retrain is deliberate and manual.

The workflow is manual dispatch only. There is no schedule. Every run is an intentional update.

**Where things live:**

| You need | Look in |
| --- | --- |
| The build entry point | `pipeline/process_data.py` |
| Shared constants, weights, currency helpers | `pipeline/utils.py` |
| Loc+ / SD+ / CT+ / Command+ / ERA+ math | `pipeline/locplus.py`, `sdplus.py`, `contact.py`, `commandplus.py`, `eraplus.py` |
| Stuff+ training and features | `stuff_plus/train_stuff.py` |
| Feed and Savant pulls | `scrapers/pitcher2026.py` |
| Sheets append path | `scrapers/sheets_append.py` |
| The output gate | `scripts/ci/validate_output.py` |
| Regression proof | `scripts/tools/golden_harness/` |
| Everything under scripts/ | `scripts/README.md` |
| Site rendering and coloring | `js/leaderboard.js`, `js/player-page.js` |
| Data loading and cache-bust stamp | `js/data.js` |
| CI pipeline | `.github/workflows/update-leaderboard.yml` |

The pre-commit hook bumps the `?v=` cache-bust stamp whenever `js/` or `css/` changes. Install it once with `git config core.hooksPath .githooks`. Without it, a JS edit will not reach browsers.

## Keeping this file current

This file is a failure log, not a wishlist. Every line below exists because it went wrong at least once.

When you make a mistake, get corrected, or discover something about this repo that was not written down:

1. Add one line to the failure log below, in the imperative, describing the correct behaviour.
2. Keep it **mechanical and repo-specific**: run this before that, this file is generated, this path is pinned. Decision history, metric rationale, and what-we-tried-and-rejected belong in the memory directory, not here. Two homes for the same fact means one of them goes stale unnoticed.
3. If the fix is a workflow rather than a rule, put it in `.claude/skills/` and link it from here.
4. Include the change in the same commit and mention it in your summary.

Keep this file under 500 lines. It loads into every session, and long context makes you less reliable, not more. If a section outgrows its usefulness, move it to `pipeline/CLAUDE.md`, `js/CLAUDE.md`, or a skill.

## Failure log

- Run the golden harness capture, baseline, and verification in one session. Never compare runs made on different days.
- Set `PYTHONHASHSEED=0` on every golden run, or the comparison is noise.
- Copy `scripts/tools/golden_harness/` somewhere disposable before running it. Its state lives next to the scripts and grows to several GB.
- The harness cannot restore a file over 100 MB (`COPY_LIMIT`); it hashes those instead. `data/all_pitches_rs_cache.pkl` is one, so a run leaves its own output there. Untracked and rebuildable, but the harness will not undo damage to it.
- The harness `WATCH` list is `data`, `js`, `index.html`. The pipeline also stamps `trade.html` and `catch.html`, which are therefore **not** restored: a run rolls their `?v=` back to the run's own timestamp. Check `git status` after any golden run and `git checkout -- trade.html catch.html` if they moved.
- The reorg left stale `pipeline_utils` / `pipeline_fetch` / `pipeline_compute` names in about 17 comments. They are navigation traps, not bugs. Fix one when you touch the line, and check any such reference resolves before trusting it.
- Do not hand-edit `data/` artifacts. Regenerate them.
- Republish the release assets after **any** pickle augmentation, or CI scores against a stale bundle.
- `curl` in CI needs `-f`. Without it an HTTP error body lands in the file and surfaces as a decompression failure.
- `0 Stuff+ grades` in the CI log means the grade dump is missing. The run is bad even though it succeeded.
- Do not move anything in `scripts/ci/` without editing `.github/workflows/update-leaderboard.yml` in the same commit.
- Do not move `scripts/abs_daily.py` or `scripts/auto-pull.sh`. launchd pins their paths.
- The settings blocks at the top of `scrapers/pitcher2026.py` and the ford_comps scripts are per-run scratch. Stage files explicitly when committing logic changes so team, dates, and player_id edits do not ride along.
- Read the README in `archive/` and `scripts/archive/` before re-running anything there. Some of it is destructive.
- Concurrent sessions can swallow each other's staged files. Check `git status` before committing if another session is open.
- `.venv/` is load-bearing for the IDE. Never delete it.
- Player-ID pulls always produce a CSV and never push to Sheets. Only team and game modes push.
- Deliverables go to `~/Downloads/` as `.docx`, `.pdf`, or `.csv`, never `.md`. Analysis and utility scripts go to `scripts/`, never Downloads.
- Bunts are not swings. Foul Bunt and Missed Bunt stay out of Swing%, Whiff%, and Chase% for both sides.
- Barrel is the official `launch_speed_angle` column. Never substitute the EV/LA recompute, which undercounts by about 5%.
- Percentile pools are all MLB players. Qualification is a render-time coloring gate, not a pool filter.
- Hitter qualification is 3.1 PA times team games played. Never substitute swing count or BIP count.
