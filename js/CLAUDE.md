# js/CLAUDE.md

The site. Read the root `CLAUDE.md` first; this file covers only the front end.

**No build step, no bundler, no framework, no npm.** Scripts are loaded by plain `<script>` tags in `index.html`, `trade.html`, and `catch.html`, and they run as-authored in the browser. Modern syntax is fine and already used (`const`, `let`, arrow functions, `Promise`), but nothing may require transpiling, importing, or installing. Do not introduce a build. Do not add a dependency.

## File map

| File | Lines | What it owns |
| --- | --- | --- |
| `player-page.js` | ~5,000 | The player page: bubbles, bars, zone grids, command map |
| `aggregator.js` | ~3,400 | Team aggregation, filter application, inversion lists |
| `app.js` | ~1,700 | Shell, routing, tab wiring |
| `leaderboard.js` | ~1,050 | Column definitions, table render, cell coloring |
| `abs.js` | ~1,000 | ABS challenge matrix views |
| `utils.js` | ~460 | Formatters, palette, percentile color scales, qualification helpers |
| `data.js` | ~450 | Chunked load, shard fetch, prefetch, cache-bust |
| `catch.js`, `trade.js`, `scatter.js` | small | Their own pages and the scatter widget |

## Payload

Per-visitor bytes and main-thread parse time are the real cost. Hosting is a static CDN, so concurrency is not an issue.

The load is staged deliberately. Do not collapse it.

| Chunk | Carries | When |
| --- | --- | --- |
| `data_core.json.gz` | Pitcher leaderboard | First paint blocks on this |
| `data_tables.json.gz` | pitchData, hitterData, hitterPitchData | Background, after idle |
| `data_heavy.json.gz` | microData, hitter details, swing locations | Background, after idle |
| `pitchdetails/<id>.json.gz` | One shard per pitcher | On demand, plus idle prefetch |

Pitch details left `data_heavy` in the 2026-08-03 split because they were 18.6 MB gz and 120.6 MB of JSON that every visitor parsed to read at most a handful of pitchers. Anything you are tempted to move **into** `data_core` needs a stated reason, because it moves onto the first-paint critical path.

Background chunks wait for idle so a prefetch never delays first paint. Keep that.

## Cache-bust

`DATA_VERSION` in `data.js` reads the same `?v=` build tag the pipeline stamps into the HTML, so data fetches and asset fetches share one stamp.

Two things bump it, and they are different:

- **The pipeline** stamps `index.html`, `trade.html`, and `catch.html` on every run.
- **The pre-commit hook** bumps it whenever a `js/` or `css/` file is part of a commit, so a code edit reaches browsers instead of serving stale cache. Install once with `git config core.hooksPath .githooks`.

A comment-only edit does not need a bust. `git commit --no-verify` is the documented skip.

## Coloring

Two percentile scales exist and they are not interchangeable. Both live in `utils.js`.

- **`percentileColor`** for table cells. Base is mid-paper `rgb(236,227,209)`, brick `rgb(176,64,47)` for good, slate `rgb(86,120,155)` for bad, with a 1.3 exponent scaled by 0.72. The near-paper midpoint is deliberate: dense tables need mid values to recede.
- **`percentileBubbleColor`** for player-page circles and bars. Keeps a visible warm-greige floor `rgb(203,184,156)` at the 50th and uses a 0.72 exponent, so a mid bubble still reads as a filled disc on cream rather than vanishing.

Do not use the cell scale for bubbles or the bubble scale for cells. If you need a third context, add a third function rather than bending one of these.

Related render states:

- **Unqualified** renders as a gray bar with white diagonal hatching, not as absent data.
- **Low-sample zones** get their own diagonal hatch on the zone grids.
- **ROC team rows show but stay uncolored.** ROC players are hidden from leaderboards unless the user explicitly selects their team.

**There is no dark palette here.** The print redesign removed it. Do not reintroduce one.

## Palette

`Utils.PITCH_COLORS` is Okabe-Ito, colorblind-safe: FF `#0072B2`, SI `#E0A81E` (amber, everywhere), FC `#8B5A2B`, SL `#D55E00`, ST `#56B4E9`, CU `#332288`, CH `#009E73`, FS `#CC79A7`. Always go through `getPitchColor` / `getPitchBorderColor` rather than hardcoding a hex.

R reports use a punched-up variant of the same palette. Cards and the site share this one.

## Mirrors of the Python side

Two blocks here are hand-maintained copies of pipeline constants, because JS cannot import Python. **Both halves change in the same commit or the site disagrees with the pipeline.**

| Here | Mirrors |
| --- | --- |
| `aggregator.js` `QUAL` block | `pipeline/utils.py` `QUAL_*` constants |
| `utils.js` `hitterPaPerGame` / `pitcherIpPerGame` | `pipeline/utils.py` same-named helpers |
| `aggregator.js` `INVERT` maps | `pipeline/compute.py` `*_INVERT*` sets |

A metric added to the Python inversion set but not here gets colored backwards on one surface and correctly on the other, which is very hard to spot.

## Qualification is a render concern

The percentile pool is all MLB players and every row already carries a stored rank from the pipeline. Qualification only decides whether a cell gets colored. Do not filter the pool client-side, and do not recompute ranks here.

Multi-team players: the combined 2TM/3TM row shows in the All Teams view and per-team rows are hidden; selecting a specific team inverts that. The qualifier denominator for a multi-team player is the max team games across their teams.

## Failure log

- Verify in the browser, not by reading the diff. Start the preview with the `site` config in `.claude/launch.json` (python http.server on 8899).
- A local `http.server` sets no content-encoding header, so `data.js` sniffs the gzip magic bytes and re-heads the stream itself. Do not "simplify" that path; it is what makes local dev work.
- Check a qualified player, an unqualified player, and a ROC player after any coloring change. They are three different render paths.
- Filters reshape team numbers, so check the filtered view as well as the unfiltered one.
- Team stats are weighted true totals, not averages of player values.
