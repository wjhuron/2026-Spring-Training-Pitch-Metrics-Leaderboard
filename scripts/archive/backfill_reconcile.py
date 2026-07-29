"""backfill_reconcile.py — one-time corrective backfill.

The permanent fix lives in download_statcast (auto-ball renumbering). This script
cleans up the historical damage that the old numbering left behind: the ~800 real
pitches sitting after an auto ball whose RunExp / bat-tracking were filled from the
neighbouring Savant pitch, plus the 108 batted balls left missing xwOBA/xBA/xSLG.

It runs a complete backfill but treats every supplement field (except Event) as an
OVERWRITE column, so wrong non-blank values are reconciled to current Savant — a
normal fill-if-empty run would leave them. Event stays scoring-correction-only.
"skip if identical" means correct cells are never rewritten. After this run, normal
fill-if-empty backfills are correct again.

  python3 scripts/backfill_reconcile.py --teams DET --dry   # validate counts, no writes
  python3 scripts/backfill_reconcile.py --teams DET         # one team for real
  python3 scripts/backfill_reconcile.py                     # complete (all teams)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backfill_supplement as B

# Reconcile every supplement field to current Savant. Event excluded: it is only
# meaningful on the terminal pitch and OVERWRITE_ONLY already handles scoring flips.
B.ALWAYS_OVERWRITE_COLS = set(B.SUPPLEMENT_MAP.keys()) - {'Event'}

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--dry' in argv:
        argv.remove('--dry')
        _orig = B.update_cells_with_retry
        def _noop(ws, cells, value_input_option='RAW'):
            return None
        B.update_cells_with_retry = _noop
        print(">>> DRY RUN: reading + computing changes, NO writes <<<\n")
    if '--teams' in argv:
        t = argv[argv.index('--teams') + 1]
        B.filter_teams = [x.strip().upper() for x in t.split(',') if x.strip()]
    B.main()
