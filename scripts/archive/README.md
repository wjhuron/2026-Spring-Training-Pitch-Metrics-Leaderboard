# scripts/archive/ — completed one-off scripts

Archived 2026-07-29 during the full-codebase audit. Everything here is a
finished experiment, a superseded model round, or an already-applied
correction. Nothing in the live tree imports or invokes these files; the
shipped configs they produced live in the pipeline modules, whose comments
cite these scripts by name as provenance (grep the name, find it here).

## DANGEROUS TO RE-RUN — Sheets/data writers

These were surgical one-shot writes against a specific historical state of
the six division workbooks. Re-running them today would overwrite current
data with stale corrections:

- `apply_ivbhb_review.py`, `apply_mismatch_fixes.py`, `apply_spin_rtilt_curation.py`
- `fix_attribution.py`, `fix_counts.py`, `fix_lv_ballpark_weather.py`,
  `fix_milb_runners_ghost.py`, `fix_missing_pitches.py`, `fix_outcomes.py`,
  `fix_platexz_feed.py`, `fix_platez_earlyseason.py`, `fix_restored_row_types.py`
- `readd_spins.py`, `repair_spin.py`, `repull_battracking.py`
- `restore_feedrev_pas.py`, `restore_missing_pitch_pas.py`, `restore_seq_revision_pas.py`
- `clear_shifted_supplement.py`, `delete_sheet_columns.py` (destructive),
  `unhide_column_ac.py`, `migrate_sheets_grade_columns.py` (schema-altering),
  `verify_sheets_parity.py` (--sync mode writes), `backfill_reconcile.py`,
  `backfill_milb_weather.py`

(The two live repair tools stayed in scripts/: `fix_unformatted_blocks.py`,
`fix_text_typed_supplements.py`. `backfill_milb_feed.py` and
`resync_recent_platez.py` also stayed — backfill_supplement.py imports them.)

## Families (what each was, where the result shipped)

| Family | What it was | Where the result lives |
|---|---|---|
| `locplus_*` (24) | Loc+ v2/v3 research: bandwidth, shrinkage, gates, stabilization, three generations of supersession | `pipeline_locplus.py` (locked config in header) |
| `stuff_lab*` (6) + `stuff_*` A/Bs | Stuff+ rounds 1-6 and later hyperparameter / weight audits | `stuff_plus_v11/train_stuff_v11.py` |
| `prototype_*` (22) | Visual mockups: LA-spray panels, Okabe-Ito palette, card/website styles | shipped in Cards.py / css (print redesign) |
| `commandplus_*` (5) | Command+ research chain: firstlook → battery → port validation | `pipeline_commandplus.py`; engine of record `../commandplus_v1.py` stayed |
| `pitcherplus_*` (5) | Pitcher+ phase 1-2 research (v2 found no headroom) | `pipeline_pitcherplus.py`; `../pitcherplus_search.py` + `../pitcherplus_build_prior.py` stayed (production) |
| `pitching_plus*` / `pitchingplus_weight_*` (3) | Stuff+/Loc+ blend-weight search | `../pitchingplus_loso_full.py` (stayed) supersedes all |
| `derive_*`, `sdct_k_predictive`, `phase2*`, `bb/pd/sd/hitter_plus_analysis`, `*_multiseason_test` | Hitter-side (BB+/SD+/CT+/Hitter+) weight derivation + multi-season defensibility tests | `process_data.py`, `pipeline_sdplus.py`, `pipeline_contact.py` |
| `xrvoe_*` (4) | xRVOE feasibility → ROC validity chain (concluded) | notes in `train_stuff_v11.py` comments |
| Jul 5-6 corrective session (`fix_*`, `apply_*`, `restore_*`, `tracking_diff_*`, `spin_repair_review`, `show_8pa_details`, `missing_pa_breakdown`, `data_diff_savant`) | The data-quality repair sprint | corrections live in the Sheets |
| `abs_metric_bakeoff`, `abs_research_round2/3` | ABS challenge metric research | `../abs_*` production chain |
| misc (`accel_test`, `axis_retest`, `depth7_confirm`, `fb_anchor_experiment`, `squared_up_explore`, `velo_diff_by_type`, `midpa_*`, `count_sweep`, `changed_teams_holdout`, `multi_season_retrain`, `train_2526_experiment`, `v12_round2_battery`, `agnostic_stuff_experiment`, `rv_vs_xrv_reliability`, `verify_runexp_alignment`, `test_approach_angles`, `wsh_hitter_analysis`, `_characterize_revisions`, `_platexz_relationship`) | one-off experiments and session scratch | conclusions in memory/commit messages |
| `stuff_data.py`, `stuff_v10_compare.py` | pre-v11 dataset builder; v10 comparison | superseded |

`results/` holds committed output artifacts that used to sit loose in scripts/.
