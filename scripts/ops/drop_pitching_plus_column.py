#!/usr/bin/env python3
"""drop_pitching_plus_column.py — delete the per-pitch Pitching+ column (Z,
1-based 26) from every migrated tab of the six division workbooks.

Part of the season-Pitching+ retirement (2026-08-28, per Wally). The name
now means the per-outing grade on the daily cards; the per-pitch blend
column is dead weight. This is a SCHEMA CHANGE: every column right of Z
shifts left by one, so it must land together with the code that stops
naming the column (sheets_write_grades X:Y, pitcher2026 / backfill_full
column lists, sheet_precision) — same-day commit b… (see git log).

Safety:
  - a tab is touched ONLY when its header cell 26 reads exactly 'Pitching+'
    AND cells 24-25 read 'Stuff+' / 'Loc+' (the migrated schema); anything
    else is skipped with a note (NEW/FCL/scratch tabs are not migrated)
  - after each delete the header is re-read and verified
  - dry run by default; nothing is written without --apply

Usage:
  python3 scripts/ops/drop_pitching_plus_column.py          # dry run
  python3 scripts/ops/drop_pitching_plus_column.py --apply
"""
import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pipeline.fetch import _gspread_client, DIVISION_WORKBOOK_IDS

COL_IDX = 25            # 0-based index of column Z (1-based 26)
EXPECT = ['Stuff+', 'Loc+', 'Pitching+']   # 1-based cols 24-26


def _retry(fn, what, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            wait = 15 * (i + 1)
            print(f'  {what}: {e} — retry in {wait}s', flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    gc = _gspread_client()
    n_drop = n_skip = 0
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        wb = _retry(lambda: gc.open_by_key(wid), f'{name} open')
        for ws in _retry(lambda: wb.worksheets(), f'{name} tabs'):
            tab = ws.title
            header = _retry(lambda: ws.row_values(1), f'{name}/{tab} header')
            if len(header) < 26 or header[23:26] != EXPECT:
                n_skip += 1
                print(f'  {name}/{tab}: not on the migrated schema — skip')
                continue
            if not args.apply:
                n_drop += 1
                print(f'  {name}/{tab}: WOULD drop column Z (Pitching+)')
                continue
            _retry(lambda: wb.batch_update({'requests': [{
                'deleteDimension': {'range': {
                    'sheetId': ws.id, 'dimension': 'COLUMNS',
                    'startIndex': COL_IDX, 'endIndex': COL_IDX + 1}}}]}),
                f'{name}/{tab} delete')
            check = _retry(lambda: ws.row_values(1), f'{name}/{tab} verify')
            if 'Pitching+' in check:
                sys.exit(f'ABORT: {name}/{tab} still shows Pitching+ after '
                         f'the delete — inspect by hand before re-running')
            n_drop += 1
            print(f'  {name}/{tab}: dropped (now {len(check)} cols)')
            time.sleep(1.2)          # write-quota pacing
    verb = 'dropped' if args.apply else 'would drop'
    print(f'{verb} {n_drop} tabs, skipped {n_skip}')


if __name__ == '__main__':
    main()
