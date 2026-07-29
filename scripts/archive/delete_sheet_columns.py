"""Delete 7 now-unneeded columns from the AL 2026 / NL 2026 pitcher tabs to
reclaim cells (NL 2026 was at ~99.75% of Google's 10M-cell limit).

Removed: EffectiveVelo, PlateTime, wOBAval, wOBAdom, Int_X, Int_Y, Barrel.

Downstream handling (already committed in the pipeline):
  - Barrel            -> recomputed at ingestion from ExitVelo + LaunchAngle
                         (pipeline_fetch.read_pitches_from_sheet), so barrel%
                         on cards / leaderboard / filters is unchanged.
  - wOBAval / wOBAdom -> recomputed from Event in process_data._bip_woba_value,
                         so xwOBAsp and SACQ% are unchanged.
  - EffectiveVelo     -> dropped entirely (leaderboard column + plumbing removed).
  - PlateTime/Int_X/Int_Y -> had no readers.
The producers (Pitcher2026 final_columns) and backfill_supplement no longer
write these, so positional appends stay aligned with the reduced schema.

Matches by HEADER NAME per tab and deletes via one batchUpdate per workbook,
descending column index so earlier deletions don't shift later ones.

Dry run by default. Pass --write to apply. IRREVERSIBLE.
"""

import argparse
import sys

sys.path.insert(0, '/Users/wallyhuron/Huronalytics')
import sheets_append as sa

DELETE_COLS = ['EffectiveVelo', 'PlateTime', 'wOBAval', 'wOBAdom',
               'Int_X', 'Int_Y', 'Barrel']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='Apply the deletions (default: dry run)')
    args = ap.parse_args()

    gc = sa._get_client()
    print(f"Columns to delete: {DELETE_COLS}")
    print(f"Mode: {'WRITE' if args.write else 'DRY RUN'}")

    for label, wbid in [('AL 2026', sa.SHEETS_AL), ('NL 2026', sa.SHEETS_NL)]:
        wb = gc.open_by_key(wbid)
        print(f"\n=== {label} ===")
        requests = []
        tabs_touched = 0
        for ws in wb.worksheets():
            header = ws.row_values(1)
            present = [(name, header.index(name)) for name in DELETE_COLS if name in header]
            if not present:
                print(f"  {ws.title:<8} no target columns; skip ({ws.col_count} cols)")
                continue
            tabs_touched += 1
            names = [n for n, _ in present]
            missing = [n for n in DELETE_COLS if n not in header]
            note = f"  (missing: {missing})" if missing else ""
            print(f"  {ws.title:<8} delete {len(present)}: {names}{note}")
            # Descending index so each deletion leaves the remaining indices valid.
            for name, i in sorted(present, key=lambda t: -t[1]):
                requests.append({'deleteDimension': {'range': {
                    'sheetId': ws.id, 'dimension': 'COLUMNS',
                    'startIndex': i, 'endIndex': i + 1}}})

        print(f"  -> {tabs_touched} tabs, {len(requests)} column-deletions queued")
        if args.write and requests:
            wb.batch_update({'requests': requests})
            print(f"  WROTE {len(requests)} deletions to {label}")
        elif not args.write:
            print(f"  dry run — no changes made to {label}")

    print("\nDone." if args.write else "\nDry run complete. Re-run with --write to apply.")


if __name__ == '__main__':
    main()
