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

## Write all replies in Simplified Technical English

Write every reply in ASD-STE100 Simplified Technical English. This rule applies to all replies in a conversation. It applies at turn 200 the same as at turn 1. A long conversation is not a reason to stop.

The rule governs replies to the user. It does not govern code, code comments, commit messages, or file content, which follow the conventions of the file they go in.

The core rules:

- Write one instruction in one sentence. Do not join two instructions with "and".
- Keep procedural sentences to 20 words or fewer. Keep descriptive sentences to 25 words or fewer.
- Keep a paragraph to 6 sentences or fewer.
- Use the active voice. Write "The pipeline writes the file." Do not write "The file is written."
- Use the simple present tense when you can. Use the simple past tense for a completed action.
- Use one approved word for one meaning. Do not use a synonym for variety.
- Write the article. Write "the file", not "file".
- Do not use a verb in the -ing form, unless it is part of a technical name.
- Do not use contractions. Write "do not", not "don't".
- Do not use slang, idiom, or a figure of speech.
- Start a warning with the command. Start a caution with the condition.

Number the steps of a procedure. State the result of a step if the result is not obvious.

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
- The harness `WATCH` list is `data`, `js`, `index.html`, and a golden run RESTORES those trees to its prestate when it finishes. Never edit a watched tree while a golden run is in flight: on 2026-08-27 an uncommitted `js/aggregator.js` edit made mid-BASE was silently wiped by the restore. Commit or hold watched-tree edits until the harness session is over. The pipeline also stamps `trade.html` and `catch.html`, which are therefore **not** restored: a run rolls their `?v=` back to the run's own timestamp. Check `git status` after any golden run and `git checkout -- trade.html catch.html` if they moved.
- The reorg left stale `pipeline_utils` / `pipeline_fetch` / `pipeline_compute` names in about 17 comments. They are navigation traps, not bugs. Fix one when you touch the line, and check any such reference resolves before trusting it.
- Do not hand-edit `data/` artifacts. Regenerate them.
- Republish the release assets after **any** pickle augmentation, or CI scores against a stale bundle.
- `curl` in CI needs `-f`. Without it an HTTP error body lands in the file and surfaces as a decompression failure.
- `0 Stuff+ grades` in the CI log means the grade dump is missing. The run is bad even though it succeeded.
- Do not move anything in `scripts/ci/` without editing `.github/workflows/update-leaderboard.yml` in the same commit.
- Do not move `scripts/abs_daily.py` or `scripts/auto-pull.sh`. launchd pins their paths.
- The settings blocks at the top of `scrapers/pitcher2026.py`, `cards/pitcher.py`, `cards/hitter.py` and the ford_comps scripts are per-run scratch. Every option-type card control (social, bats, rv_mode, layout, la_view, ...) ALWAYS gets a variable in that block with the CLI flag as override — never a CLI-only option (per Wally 2026-08-27). Stage files explicitly when committing logic changes so team, dates, and player_id edits do not ride along. Reset the dates to `None` in the committed version: a shipped `start_date` makes a bare run render a window instead of the season.
- Read the README in `archive/` and `scripts/archive/` before re-running anything there. Some of it is destructive.
- Concurrent sessions can swallow each other's staged files. Check `git status` before committing if another session is open.
- `.venv/` is load-bearing for the IDE. Never delete it.
- Player-ID pulls always produce a CSV and never push to Sheets. Only team and game modes push.
- Deliverables go to `~/Downloads/` as `.docx`, `.pdf`, or `.csv`, never `.md`. Analysis and utility scripts go to `scripts/`, never Downloads.
- Bunts are not swings. Foul Bunt and Missed Bunt stay out of Swing%, Whiff%, and Chase% for both sides.
- Barrel is the official `launch_speed_angle` column. Never substitute the EV/LA recompute, which undercounts by about 5%.
- Percentile pools are all MLB players. Qualification is a render-time coloring gate, not a pool filter.
- Hitter qualification is 3.1 PA times team games played. Never substitute swing count or BIP count.
- Never run bare `git commit` here. Use `git commit -- <paths>`. The index routinely holds work in progress (the `pitcher2026.py` settings block, R-scripts, cards), and a bare commit takes everything staged. On 2026-08-17 `c3b9ba28` swept five unrelated files into a commit whose message described only a scraper fix.
- `scripts/auto-pull.sh` runs `git pull --rebase --autostash` on a launchd timer. It can fire mid-session and leave the artifacts in `data/` conflicted, which makes the next commit fail. Reset them to HEAD; they are all regenerable.
- Hitter `+` metrics are percents; pitcher `+` metrics are standard deviations, and that split is deliberate. Stuff+/Loc+/Pitching+ are borrowed public vocabulary built FanGraphs-style, so their 100 +/- 10 spread is the shared language and rescaling would cost cross-site comparability. Command+ is the one pitcher exception (a miss-distance ratio) because it is not shared vocabulary. Do not "unify" these.
- The `+` suffix is a contract: 100 is league average and one point is one percent. Hitter+/BB+/SD+/CT+ all honour it as of 2026-08-18. Never rescale a `+` stat to match another stat's spread — matching Hitter+ to wRC+'s full SD drops its slope to r and makes 115 worth 12%, not 15%. The deflation to `r x SD(wRC+)` IS the wRC+ scale.
- Hitter+'s run-truth `r` is measured LIVE each run so the slope stays exactly 1. A pool under 30 or an r outside 0.40-0.98 falls back to the frozen 0.82 and logs it. The value used ships in `metadata.hitterPlusStandardization.wrcScaleMatch.r`.
- Percentile precision differs by family: the pipeline ranks FULL-precision values (`process_data.py` says so where the ranks are computed), while the Stuff+ family ranks display-rounded ones (Stuff+ at 1 decimal, xRVOE/100 at 2, in `train_stuff.py`). Only in the second family can an affine rescale shift a few ranks by exactly 1 through tie-breaking; that is not a rank change. Check which family a stat is in before assuming either behaviour.
- An IBB is a PA. For HITTERS it also counts as a walk, so PA and BB% both include it; for PITCHERS BB% uses unintentional walks only, but the IBB still counts toward batters faced. The season path already does this via the boxscore merge. Pitch-derived paths CANNOT: a no-pitch IBB leaves no pitch, so they must merge an official line or they run short (Wood 449 PA against 454, BB% 16.5% against 17.4%).
- FanGraphs serves custom date ranges (`month=1000` plus `startdate`/`enddate`). `fg_overrides.fetch_mlb_hitters_range` uses it for the window official line. Do not conclude FG is season-only; that has been wrong twice.
- Date-window cards: VALUES come from the window, PERCENTILES come from the SEASON pool, and there is no sample-size gate (`pipeline/window_pool.py`). A three-week hot stretch is supposed to read as an elite percentile; that is what makes two windows comparable to each other. Do not rebuild a window-specific percentile pool - that was tried, it made every window its own ruler, and it needed a qualification gate that returned nothing on short ranges.
- The + family on a window is scored against SEASON league anchors and cell tables, never window-rebuilt ones, for the same reason.
- Percentile inversions (`HITTER_INVERT_PCTL`) are a SEPARATE pass in `process_data`, not part of `compute_percentile_ranks_with_aaa`. Any code that computes percentiles outside the pipeline must apply them, or K%/Chase%/Whiff%/GB% colour backwards and nothing errors.
- `scripts/builders/build_window_leaderboard.py START END [--cached]` is the EXACT window pool (real season code path, official-boxscore PA), at ~8 min. The card only uses it if it already exists; otherwise it builds its own in ~9s. Use the builder when PA must reconcile to official totals. `--cached` reuses `data/_window_pitches_cache.pkl` and skips the 2m30s Sheets read; drop it after Sheets is appended to.
- Never pass `window_mode=True` to `process_game_type` on a run that writes a shipped `_rs` artifact. It suppresses the FanGraphs wRC+/FIP overrides and the Savant sprint-speed merge on purpose, so its output is a window artifact only.
- Set `PYTHONHASHSEED=0` on window builds. A full-season control reproduced the shipped leaderboard on 150 of 164 keys, and the two rows that moved at the third decimal were both multi-team (2TM) rows, where aggregation order is not pinned.
- Column decimal depth is single-homed in `scrapers/sheet_precision.py`. `pitcher2026.py`, `backfill_full.py` and `backfill_supplement.py` all read it and none of them may carry a rounding list of its own, or a fresh scrape writes truncated values that the next sweep re-upgrades forever.
- Qualification is single-homed in `Utils.buildQualContext` / `Utils.isQualified` (`js/utils.js`). `js/aggregator.js`, `js/data.js` and `js/player-page.js` all call it and none of them may answer the question themselves. Three copies existed until 2026-08-19 and they disagreed on 16 MLB players.
- A multi-team label (`2TM` ... `10TM`) is a player, not a franchise, so it never supplies a team-games denominator. A multi-team player is measured against the club he most recently played for, resolved from `lastGameDate` on his MLB stint rows. Feeding the label to `teamGames` gave `2TM` a 144-game season, `3TM` a 109-game one, and `4TM` a threshold of zero.
- A ROC/AAA row never resolves a combined row. `Utils.playerKey` groups on `mlbId`, so a ROC stint of a traded player shares a key with his MLB rows, and the combined row is built from MLB stints only. `pipeline/compute.py:769` routes ROC rows out before the same check.
- Pitcher and hitter leaderboard rows both carry `lastGameDate`. Qualification reads it on both sides. Do not drop it from either row builder in `pipeline/process_data.py`.
- The pipeline resolves the same denominator through `current_team_by_player` / `is_combined_team` / `player_key` in `pipeline/utils.py`. They mirror `js/utils.js` and must move together, or the shipped percentile pool and the site's render gate disagree.
- `metadata.teamGamesPlayed` (30 MLB teams, from `compute_team_games_played`) and `metadata.teamGames` (33 entries, includes ROC 112 and the 2TM/3TM labels) are different dicts. The pipeline qualifiers use the first, the client uses the second. Never feed a combined label to either.
- Never commit the output of a local `python3 -m pipeline.process_data`. CI runs `stuff_plus/train_stuff.py` AFTER the pipeline and injects `stuffScore_lowSupport` plus the `rvoe`/`xrvoe` family into the leaderboards, so a local rebuild silently strips them (985 pitcher and 4573 pitch rows, measured 2026-08-19). Run it locally to verify, then `git checkout -- data/` and dispatch the workflow.
- `metadata.generatedAt` is the generating machine's LOCAL time with no timezone marker. CI writes UTC, a local run writes ET, so the site's "Generated" stamp can appear to move backwards after a local run. Compare commit times, not the stamp.
- Key every per-club lookup by MLB club id through `TEAM_ABBREV_TO_ID`, never by abbreviation. Savant spells six clubs `AZ`/`KC`/`SD`/`SF`/`TB`/`OAK` and the leaderboard spells them `ARI`/`KCR`/`SDP`/`SFG`/`TBR`/`ATH`; a hand-copied abbreviation map made 167 pitcher rows score hpERA against a neutral park for four days, with no error.
- Savant's rolling-3 park-factor endpoint OMITS any club whose venue lacks three seasons, and it does not narrow the window itself. `scripts/builders/park_factors_pull.py` cascades 3 to 2 to 1 and records the window under `_window`. Do not read that endpoint directly.
- ROC/AAA rows score hpERA but NOT hdERA. `apply_era_plus` needs `roc_pitches` passed in (train_stuff hands it the currency-corrected list) or every ROC row loses the xRV channel and hpERA stays None. ROC never enters the pool, the league rates or the z statistics, and its home park scores neutral because no minor-league park factors exist.
- A combined `2TM`..`10TM` row has no park of its own, so `apply_era_plus` resolves it as the MOST RECENT club's park via `current_team_by_player` (`combined_park_map`). Measured 2026-08-24: hpERA forecasts future runs, and IP-weighted stint history LOSES to the current club's park in every LOSO replicate. `park.get('2TM')` falls through to neutral, which scored 93 rows against a neutral park until 2026-08-19. ROC stints never resolve the club.
- The bulk stats endpoint (`stats?stats=season`) returns ONE season-combined row per pitcher with only the FINAL club attached. Per-team stints need the person hydrate (`/people?personIds=...&hydrate=stats(...)`), which also splits byDateRange pulls by team.
- Savant serves bat speed for ROC on EVENT pitches (balls in play and strikeouts) as of 2026-08-21, applied by hand to the AAA tab's `BatSpeed` column. `cards/hitter.py` used to drop the Bat Speed bubble on every ROC card on the assumption the value could never exist. A note that encodes "this can never happen" goes stale silently; gate on the data, not the assumption. The ROC value is event-only, the MLB value is every swing, and the card says so.
- R report tables drop an all-empty column by its SOURCE data (`STAT_COL_SOURCE` / `keep_populated_cols` in `pitcher_report_utils.R`), never by the rendered string. Zone% and GB% print a false `0.0%` when `InZone` or `BBType` is empty, because both divide a real denominator by zero hits, so a blank-text test keeps them. `format_table` also located the platoon table's second Pitch Type column at a hardcoded index 9 behind an `ncol >= 16` gate; it now finds it by name, because dropping columns shrinks both halves.
- Report Total rows RECOMPUTE every rate over the whole outing; never average the per-type rows, which would weight an 11-pitch type the same as a 27-pitch one. Only Count, % Thrown, release point (Ext/RelZ/RelX/Arm Angle), Stuff+ and the outcome rates aggregate; velocity, max velo, spin, tilt, IVB/HB and VAA/HAA stay blank because a mean across pitch types is not a pitch anyone threw. Do NOT point `Daily.R` at `summarize_total_row`: Daily carries a Stuff+ column the shared helper does not model, and it formats arm angle `%.1f` where the shared reports use `%.1f°`.
- A postponed game is `abstractGameState: Final` with `codedGameState: D` on its ORIGINAL date, and the makeup game reuses the gamePk. Gate schedule pulls on `codedGameState == 'F'` (`_game_is_completed` in `pipeline/fetch.py`) and never let an empty boxscore claim a gamePk, or the dedup drops the real game. Six makeup games and 150 players were short one game until 2026-08-23.
- The v1 Stuff+ LOSO gate (`scripts/research/stuff/stuff_pertype_loso_gate.py`, `stuff_features_loso.py`) applies the nVAA transform on the pitch dicts and then `build_df` applies the frozen `NVAA_SLOPES` again. Every gate run after v12 shipped scored a double-adjusted VAA. Use `stuff_gate_v2.py`, which adjusts VAA once from `vaa_raw`.
- A Stuff+ feature or hyperparameter decision needs the NEXT-SEASON objective (`nxt_r` in `stuff_gate_v2.py`, year Y grade against Y+1 outcomes, model fit on neither) with a paired bootstrap SE. The within-season `fut_r` passed depth 7 and failed to detect a +0.015 gain from pitcher height; `nxt_r` found both. Wins in 5/5 pairs on a delta inside one SE is not a result.
- `scripts/builders/build_historical_training_set.py` imports `scripts.build_2025_training_set`, a pre-reorg path. It does not import. Copy `GUTS` or fix the import before relying on it.
- Never push to main while an update-leaderboard run is in flight. The run rebases before its artifact push; a push that moved `catch.html`/`index.html` stamps makes that rebase conflict (UU), the run FAILS at the commit step, and its entire output — leaderboard data AND, on retrains, the published bundle asset — is dropped. On 2026-08-23 the v14 retrain lost ~50 minutes this way. Dispatch, then freeze main until green.
- Pitcher K%/BB%/K-BB% divide by the OFFICIAL boxscore TBF, never the pitch-derived `pa`. A no-pitch intentional walk (automatic since 2017) leaves no pitch row, so `pa` runs short by that many batters faced and K% reads high. `js/aggregator.js` must re-apply the shipped season rate whenever the aggregation covered the whole season (gate on `rows[i].pa === pre.pa`), because `DataStore.getFilteredDataV2` calls `needsReaggregation()`, which returns `true` unconditionally: a server-only fix does not change a single number on the leaderboard. The player page reads `PITCHER_DATA` directly and needs no client fix.
- Pitcher BB% is UNINTENTIONAL walks over TBF; hitter BB% is TOTAL walks (incl. IBB) over PA. FanGraphs counts IBB for both, so its pitcher BB% reads higher than ours by design. Do not "fix" that gap.
- No-pitch intentional walks are ALREADY in the sheets as marker rows (`PitchID` ends `_00`, Event `Intent Walk`, every measurement blank). `read_all_pitches_from_sheets` drops them at the boundary via `real_pitches`/`is_no_pitch`; PA-level callers pass `include_no_pitch=True`. Do NOT teach the per-pitch denominators to skip them instead: that was tried on 2026-08-18 and did not converge (`c[0] += 1` in four micro loops, `usagePct`, and a `not in BALL_DESCRIPTIONS` numerator that scored a no-pitch row as a STRIKE). Because the pipeline excludes them, pitch-derived `pa` is short by design, which is why the PA-denominated rates read official boxscore counts. Keep the marker set current with `scripts/audits/enumerate_missing_ibb.py` then `scripts/ops/write_missing_ibb.py --apply`.
- Exactly ONE row per at-bat may carry a plate-appearance Event. A live scrape stamps the outcome on the then-final pitch; the re-pull appends the real final pitch and the PitchID dedupe cannot tell the earlier row is now wrong, so both keep it. The stale row also carries `BBType` and the batted-ball columns, so one ball in play is counted twice (BABIP, GB%/FB%/PU%, hard-hit, avg EV). `process_data` warns right after the Sheets read via `duplicate_pa_events`; repair with `scripts/ops/fix_duplicate_pa_events.py --apply`. It clears `PA_OUTCOME_COLUMNS` only. Never clear `Description`, `Count`, `Runners`, `Outs` or `RunExp` on those rows: the stale row is a real pitch and those are per-pitch facts.
- `_sheet_row` in the pitch cache goes stale, because rows shift. Any tool that WRITES to a sheet row must re-read the cell and confirm the `PitchID` first, then look the row up by `PitchID` if it does not match. Measured 2026-08-26: one of three target rows had moved by 129 rows.
