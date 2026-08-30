#!/usr/bin/env python3
"""fix_pagan_xmovement.py — one-off repair, 2026-08-30.

The Savant overwrite pass (backfill_supplement) updated one retracked
Pagán pitch (CIN 2026-08-26): IndVertBrk 15.7 -> 18.1, HorzBrk 7.6 -> 8.6.
xIndVrtBrk/xHorzBrk are IndVertBrk/HorzBrk times the weather factor and
were computed from the OLD raw values, so that row's x-movement is stale.

No weather re-pull is needed: the factor is the stored x/raw ratio of the
pre-update pair. This writes exactly two cells, located by PitchID (never
by cached row number), and aborts unless the row holds the post-update
raw values it expects.

  python3 scripts/ops/fix_pagan_xmovement.py            # dry run
  python3 scripts/ops/fix_pagan_xmovement.py --apply
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import argparse

import gspread

from scrapers.sheets_append import _workbook_id_for_team

PITCH_ID = None      # resolved below: the one CIN 2026-08-26 Pagán CU retrack
OLD_IVB, OLD_HB = 15.7, 7.6     # pre-apply raw values the factors divide out
NEW_IVB, NEW_HB = 18.1, 8.6     # post-apply raw values the row must hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    gc = gspread.service_account()
    ws = gc.open_by_key(_workbook_id_for_team('CIN')).worksheet('CIN')
    rows = ws.get_all_values()
    ci = {n: i for i, n in enumerate(rows[0])}

    hits = []
    for r_i, r in enumerate(rows[1:], start=2):
        if (r[ci['Game Date']] == '2026-08-26'
                and r[ci['Pitcher']] == 'Pagán, Emilio'
                and r[ci['IndVertBrk']] == f'{NEW_IVB}'
                and r[ci['HorzBrk']] == f'{NEW_HB}'):
            hits.append((r_i, r))
    if len(hits) != 1:
        raise SystemExit(f'expected exactly 1 matching row, found {len(hits)} — abort')
    r_i, r = hits[0]

    xivb_old = float(r[ci['xIndVrtBrk']])
    xhb_old = float(r[ci['xHorzBrk']])
    xivb_new = round(NEW_IVB * (xivb_old / OLD_IVB), 1)
    xhb_new = round(NEW_HB * (xhb_old / OLD_HB), 1)
    print(f'row {r_i} PitchID {r[ci["PitchID"]]}: '
          f'xIVB {xivb_old} -> {xivb_new}, xHB {xhb_old} -> {xhb_new}')

    if not args.apply:
        print('DRY RUN — pass --apply to write the 2 cells')
        return
    ws.update_cells(
        [gspread.Cell(r_i, ci['xIndVrtBrk'] + 1, f'{xivb_new:.1f}'),
         gspread.Cell(r_i, ci['xHorzBrk'] + 1, f'{xhb_new:.1f}')],
        value_input_option='USER_ENTERED')
    print('wrote 2 cells')


if __name__ == '__main__':
    main()
