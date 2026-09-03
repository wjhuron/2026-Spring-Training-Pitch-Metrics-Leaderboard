# pipeline/CLAUDE.md

The build. Sheets in, leaderboard JSON and gzipped embeds out. Read the root `CLAUDE.md` first; this file covers only what is specific to the metric math.

`process_data.py` is 5,000 lines and runs as one ordered pass. Stage boundaries are marked with `# ====` banners. Order matters: percentiles are computed in a single pass near the end (around line 4315) and everything upstream must have written its values by then.

## Module map

| File | What it owns |
| --- | --- |
| `utils.py` | Shared constants, zone geometry, event sets, qualification, RunExp currency |
| `fetch.py` | Sheets reads, boxscore refresh, FanGraphs guts and park factors |
| `compute.py` | Rate stats, percentile inversion lists |
| `process_data.py` | The ordered pass and all output writing |
| `locplus.py` | Loc+, the decomposition location-value model |
| `sdplus.py` | SD+, zone-band swing decisions |
| `contact.py` | CT+ |
| `commandplus.py` | Command+, K=1 cell means with a thin-cell cascade |
| `eraplus.py` | The ERA estimator family |
| `xwoba3d.py` | xwOBA on the EV × LA × spray grid |
| `pitcherplus.py` | Pitcher+ composite |
| `xmove.py` | Expected movement (xIVB/xHB), per-type ridge on spin + release axis, scored per pitch |

## Constants are measured, not chosen

The global rule in `~/.claude/CLAUDE.md` applies to every number in this directory: sweep it, bracket an interior optimum or prove the curve flat, say which, then validate on independent replicates.

Two directory-specific consequences:

- **Every constant here carries its provenance in a comment.** `MIN_HITTER_SWINGS = 65` says "split-half r=.50 point." `N0_XRV = 800.0` says "flat 500-1500." `DH_B = 0.917` says "LOSO slope." A constant without that note is either undocumented or unmeasured, and you should find out which before trusting it.
- **Do not re-derive a constant from the live season alone.** The season is partial, and tuning on a partial season fits the sample size rather than the sample.

## Shrinkage is a cascade, and each level has its own measured constant

The pattern across `sdplus`, `contact`, `xwoba3d`, and `commandplus` is the same: a thin cell shrinks toward its zone, the zone shrinks toward the league, and the player shrinks toward the population. Each arrow has its own pseudo-observation count, and they were measured separately.

- `CELL_SHRINK_K` is **3** in `xwoba3d.py`, **50** in `sdplus.py`, **50** in `contact.py`. Same name, three different quantities, all correct. Do not unify them.
- `HITTER_PRIOR_N` is **190** in `sdplus.py` and **66** in `contact.py`. Same.
- Reliability and split-half sweeps are **gameable** for cell-level `k`: smoothing toward a constant inflates them while destroying information. Use out-of-sample cell-model fit to choose a cell `k`, and treat reliability as a diagnostic only.

## Zone geometry

Two different zone definitions live in `utils.py` on purpose.

- **`compute_in_zone`** (line 369) is the exact Savant geometry: rounded-rect with the ball-radius rule, built from `HALF_PLATE_FT` (8.5") and `BALL_RADIUS_FT` (1.45"). This is what InZone means everywhere it is displayed. It is validated to 100% agreement with Savant.
- **`ZONE_HALF_WIDTH = 0.83`** is the older rectangle. It survives **only** as the band boundary for SD+ zones, where the rectangle is what the bands were fit against. It over-counts corners by about 0.22% and must never be substituted for `compute_in_zone`.

SD+ bands run off `HEART_X` (6.7"), `SHADOW_X` (13.3"), and `CHASE_X` (20.0") with vertical fractions of the measured zone height.

## Currency

Values from different levels are not on the same scale, and mixing them silently is the easiest way to ship a wrong number.

- **MiLB RunExp is league-denominated in the raw cache**, roughly 1.25x MLB. `process_data` corrects it. Any other pickle reader must import `compute_runexp_scale` / `runexp_factor` from `utils.py` rather than reading the raw values.
- **ROC pitchers are scored against the MLB baseline** for Loc+ and the other plus metrics. That is deliberate: the tooltip text says so, and it makes ROC and MLB rows comparable.
- `AAA_TEAMS = {'ROC', 'AAA'}` covers both, and they are **not** synonyms. `BTeam='AAA'` rows are the opponent byproduct of the ROC tab, not ROC hitters.

## Pools, gates, and gates that are not pools

Three different things get confused constantly:

- **The percentile pool is all MLB players.** Every row gets a stored rank. Qualification does not filter the pool.
- **Qualification is a render-time coloring gate**, applied in the site layer, not here. Hitters: 3.1 PA times team games (2.7 for ROC). Constants live at `utils.py:101`.
- **A computation floor is neither.** `MIN_HITTER_SWINGS = 65`, `MIN_CELL = 20`, `MIN_POOL = 300` in `commandplus` and `50` in `pitcherplus` are floors below which a value is not computed at all. Note that the two `MIN_POOL` values count different units: pitches in one file, qualified pitchers in the other.

## Event and swing semantics

- **Bunts are not swings.** `SWING_DESCRIPTIONS` is exactly `{'Swinging Strike', 'Foul', 'In Play'}`. Foul Bunt and Missed Bunt stay out of Swing%, Whiff%, and Chase% on both sides, though a foul bunt still counts as a strike.
- **Barrel is the official `launch_speed_angle`**, never the EV/LA recompute, which undercounts by about 5%.
- **Outs come from per-pitch `count.outs`.** `about.outs` is absent in Triple-A and archived MLB feeds.
- Position-player pitching is excluded from skill metrics but their PAs still count for hitters.

## Inversion lists

`compute.py` holds four sets that decide which metrics rank low-is-good: `PITCH_BB_INVERT`, `PITCHER_INVERT_PCTL`, `HITTER_INVERT_PCTL`, `PITCHER_BB_INVERT`. The site mirrors these in `js/aggregator.js`. Adding a metric to one without the other means the leaderboard and the player page color it in opposite directions. Change both in the same commit.

Hand-signed shape metrics (`haa`, `nHAA`, `horzBrk`, `hbOE`) rank on absolute value, not signed value.

## Failure log

- Do not round inside an aggregation. `runValue`, `xRunValue`, `rv100`, `xRv100` stay at full precision until the display layer.
- Do not add a weight or threshold to a second file. `PITCHING_W_STUFF` ran 70/30 in one place and 80/20 in another for about three weeks.
- Raw HAA is a location channel. Always residualize before using it as a shape feature.
- Loc+ count mix carries location skill, not just stuff. Post-stratification, a BIP anchor, and a median-zone baseline were all tested and rejected.
- The percentile pass is single-pass and late. A value written after it will have no rank.
- `NVAA_SLOPES` in the Stuff+ build are frozen. Re-deriving them per run makes retrains unstable.
- A debut arm with no measured arm angle falls back to its ROC/AAA angle. A NaN angle is retrain-unstable garbage, not a zero.
