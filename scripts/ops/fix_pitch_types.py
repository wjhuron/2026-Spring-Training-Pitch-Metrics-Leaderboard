#!/usr/bin/env python3
"""Set Pitch Type from the MLB feed for a named list of pitches.

Pitch Type is on the backfill's exclusion list, so the sweep never writes it. But
every band comparison the sweep makes is KEYED on it, so a wrong tag silently
routes a pitch to the wrong reference group and every judgement about that pitch is
made against the wrong population.

Measured 2026-08-17 on the consolidated reject file: 401 of 1,920 rejected pitches
carried a sheet tag the feed disagrees with, accounting for 553 of the 2,404
rejections. One mistagged pitch produced up to 10 rejections across 10 tabs, which
is why the review felt like the same pitch over and over.

Scope is deliberately the pitches that surfaced in review, not the whole sheet —
Wally's call: "my sheets are good, it is these specific pitches."

A pitch whose feed type is UN is SKIPPED. The feed lost the classification there, so
the sheet's tag is the better one and overwriting it would destroy information.

Wally, 2026-08-17, on where these tags came from: some are his own guesses, made when
a pitch arrived with no tracking data at all and a type had to be entered by hand. Those
were never wrong, they were placeholders. The feed now carries a real measurement for
them, which is precisely the case where it should win.

    python3 scripts/ops/fix_pitch_types.py <reject_workbook.xlsx>
    python3 scripts/ops/fix_pitch_types.py --apply <reject_workbook.xlsx>
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse, collections, os
import gspread, requests, openpyxl

from pipeline.fetch import DIVISION_WORKBOOK_IDS, read_sheet_with_retry
from scrapers.backfill_supplement import _retry_sheets_call
from scrapers.backfill_full import feed_rows


def main(path, apply=False):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    seen = {}
    for n in wb.sheetnames:
        ws = wb[n]; hdr = None
        for row in ws.iter_rows(values_only=True):
            if hdr is None:
                hdr = list(row)
                if 'PitchID' not in hdr or 'Pitch Type' not in hdr:
                    break
                continue
            d = dict(zip(hdr, row))
            if d.get('PitchID'):
                seen.setdefault(d['PitchID'], (d.get('Team'), d.get('Pitch Type')))
    print(f'{len(seen)} distinct pitches in {os.path.basename(path)}')

    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
    todo = collections.defaultdict(list)   # tab -> [(pid, sheet_pt, feed_pt)]
    skipped_un = 0
    for pid, (tab, sheet_pt) in seen.items():
        fr = feed_rows(pid.split('_')[0], s).get(pid)
        if not fr:
            continue
        feed_pt = str(fr.get('_PitchType') or '').strip()
        sheet_pt = str(sheet_pt or '').strip()
        if not sheet_pt or not feed_pt or sheet_pt == feed_pt:
            continue
        if feed_pt == 'UN':
            skipped_un += 1
            continue
        todo[str(tab).upper()].append((pid, sheet_pt, feed_pt))

    n = sum(len(v) for v in todo.values())
    print(f'to retag: {n} pitches across {len(todo)} tabs')
    print(f'skipped because the feed says UN: {skipped_un}')
    print('\ndirections:')
    for k, c in collections.Counter(
            f'{a}->{b}' for v in todo.values() for _, a, b in v).most_common():
        print(f'   {k:12} {c}')

    written = 0
    for label, sid in DIVISION_WORKBOOK_IDS.items():
        sh = _retry_sheets_call(lambda: gc.open_by_key(sid), 'workbook open')
        for ws in _retry_sheets_call(sh.worksheets, 'tab list'):
            tab = ws.title.upper()
            if tab not in todo:
                continue
            rows = read_sheet_with_retry(ws)
            hdr = rows[0]
            jp, jt = hdr.index('PitchID'), hdr.index('Pitch Type')
            where = {r[jp]: i + 1 for i, r in enumerate(rows) if jp < len(r) and r[jp]}
            cells, bad = [], 0
            for pid, sheet_pt, feed_pt in todo[tab]:
                r = where.get(pid)
                if r is None:
                    print(f'    {tab} {pid}: not in this tab, skipping'); bad += 1
                    continue
                # Assert the cell still holds what the workbook said before touching it.
                have = (rows[r - 1][jt] or '').strip()
                if have != sheet_pt:
                    print(f'    {tab} {pid}: expected {sheet_pt!r}, found {have!r}, '
                          f'skipping'); bad += 1
                    continue
                cells.append(gspread.Cell(row=r, col=jt + 1, value=feed_pt))
            print(f'  {tab}: {len(cells)} to retag'
                  + (f', {bad} skipped' if bad else ''))
            if apply and cells:
                _retry_sheets_call(
                    lambda: ws.update_cells(cells, value_input_option='USER_ENTERED'),
                    'pitch type write')
                written += len(cells)
                print(f'     wrote {len(cells)}')
    print(f'\n{"WROTE " + str(written) if apply else "DRY RUN"} pitch types')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('workbook')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    gc = gspread.service_account()
    main(a.workbook, apply=a.apply)
