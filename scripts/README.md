# scripts/ layout (2026-08 reorg)

Two files stay at this level because launchd pins their paths:
`abs_daily.py` (7:30 daily ABS pipeline) and `auto-pull.sh` (keep-alive
git pull). Do not move them without updating `~/Library/LaunchAgents/`.

| Folder | Contents |
|---|---|
| `ci/` | The five scripts the GitHub Actions workflow runs (rebuild_embed, refresh_micro_grades, sheets_write_grades, validate_output, build_kinematics_2026) plus kinematics_lib. Paths are referenced in `.github/workflows/update-leaderboard.yml` — move only in lockstep with it. |
| `abs/` | ABS challenge-matrix suite. `../abs_daily.py` runs these as subprocesses. |
| `ops/` | Operational repair tools that production error messages point at (fix_unformatted_blocks, fix_text_typed_supplements). |
| `audits/` | Read-only data-integrity audits: drift checks, attribution audits, tag audits, PA completeness, PlateZ resync. `scrapers/backfill_supplement.py` dynamically loads resync_recent_platez from here. |
| `builders/` | Asset and cache builders: training pickles, augmentations, park factors, priors, Statcast pulls. Outputs land in `data/` or GitHub release assets. |
| `tradevalue/` | The trade-value program: corpus pulls, engines, market fit, stuffedge. Ships `data/tradevalue_data.json.gz` for trade.html. |
| `research/<topic>/` | Metric research (era, locplus, commandplus, hitter, stuff, xmove, comps, cards, misc). Scripts run from the repo root: `python3 scripts/research/era/era_weights_final.py`. Cross-topic imports declare an explicit `sys.path.insert` next to the import. |
| `tools/` | Standalone utilities and render one-offs (catch_prob, platoon cards, zone renders, prospect pulls, MiLB FA sheets). |
| `archive/` | Superseded scripts, kept greppable. See its README before re-running anything. |

Convention: scripts compute the repo root from `__file__` and read/write
`data/` at the root. Same-folder imports are bare; cross-folder imports
always carry their own `sys.path.insert` line.
