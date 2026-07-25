"""Retype supplement columns that were stored as TEXT so the sheet sorts them.

Background
----------
backfill_supplement wrote every supplement column with
value_input_option='RAW', which stores "53.6" as the STRING "53.6". Nothing
downstream broke — the pipeline reads with get_all_values() and parses through
safe_float, so the website was always right — but Sheets sorts text
lexicographically, so an ArmAngle column sorts 9.9 above 74.8 and is useless to
sort or filter in the spreadsheet. This affects every tab, MLB included, for
every column in SUPPLEMENT_NUMBER_FORMATS.

backfill_supplement now writes USER_ENTERED; this repairs what RAW already
wrote. Values are re-sent unchanged (parsed as numbers this time) and the
column's number format is pinned, so the FORMATTED value each cell reads back
as is byte-identical to the string it held before. That matters: the backfill
decides "already filled" and "identical to existing" by string-comparing
against get_all_values(), so a format that rendered 41 as "41" instead of
"41.0" would make every later run phantom-overwrite the cell.

Blank cells stay blank. Text that is not a number (a stray note) is left alone
and reported.

Usage
-----
    python scripts/fix_text_typed_supplements.py                   # scan
    python scripts/fix_text_typed_supplements.py --apply           # repair all
    python scripts/fix_text_typed_supplements.py --apply --teams ROC,AAA
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backfill_supplement as B  # noqa: E402
from sheets_append import _col_letter, _sheets_retry  # noqa: E402


def _numeric_text_cells(sh, tab, header, name):
    """Return (n_text, n_num, n_blank, n_bad) for one column."""
    idx0 = header.index(name)
    a1 = f"'{tab}'!{_col_letter(idx0 + 1)}2:{_col_letter(idx0 + 1)}"
    meta = _sheets_retry(
        lambda: sh.fetch_sheet_metadata({
            'includeGridData': True, 'ranges': [a1],
            'fields': 'sheets(data(rowData(values(effectiveValue))))'}),
        label=f'{tab}.{name} scan')
    rows = (meta['sheets'][0].get('data') or [{}])[0].get('rowData', [])
    n_text = n_num = n_blank = n_bad = 0
    for r in rows:
        ev = (r.get('values') or [{}])[0].get('effectiveValue')
        if not ev:
            n_blank += 1
        elif 'numberValue' in ev:
            n_num += 1
        elif 'stringValue' in ev:
            try:
                float(ev['stringValue'])
                n_text += 1
            except ValueError:
                n_bad += 1
        else:
            n_bad += 1
    return n_text, n_num, n_blank, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--teams', help='comma-separated tab names (default: all)')
    args = ap.parse_args()
    only = ({t.strip().upper() for t in args.teams.split(',')}
            if args.teams else None)

    gc = B.get_client() if hasattr(B, 'get_client') else __import__('gspread').service_account()
    grand_text = grand_fixed = 0

    for wb_name, wb_id in B.SPREADSHEET_IDS.items():
        sh = _sheets_retry(lambda: gc.open_by_key(wb_id), label=f'open {wb_name}')
        for ws in _sheets_retry(sh.worksheets, label=f'{wb_name} tabs'):
            tab = ws.title.upper()
            if tab not in B.ALL_TRACKED_TEAMS:
                continue
            if only and tab not in only:
                continue
            header = _sheets_retry(lambda: ws.row_values(1), label=f'{tab} header')
            if not header:
                continue

            targets = [c for c in B.SUPPLEMENT_NUMBER_FORMATS if c in header]
            dirty = []
            for name in targets:
                n_text, n_num, n_blank, n_bad = _numeric_text_cells(sh, ws.title, header, name)
                if n_text:
                    dirty.append((name, n_text, n_num, n_bad))
            if not dirty:
                continue

            print(f"\n[{wb_name}/{ws.title}]")
            for name, n_text, n_num, n_bad in dirty:
                grand_text += n_text
                print(f"   {name:16} {n_text:6} text  {n_num:6} numeric"
                      + (f"  {n_bad} non-numeric (left alone)" if n_bad else ""))

            if not args.apply:
                continue

            for name, n_text, _n, _b in dirty:
                idx0 = header.index(name)
                col = _col_letter(idx0 + 1)
                rng = f"{col}2:{col}"
                vals = _sheets_retry(lambda: ws.get(rng), label=f'{ws.title}.{name} read')
                # Re-send unchanged; USER_ENTERED parses the numerics this time.
                body = [[(v[0] if v else '')] for v in vals]
                _sheets_retry(
                    lambda: ws.update(values=body,
                                      range_name=f"{col}2:{col}{len(body) + 1}",
                                      value_input_option='USER_ENTERED'),
                    label=f'{ws.title}.{name} rewrite')
                grand_fixed += n_text
                print(f"   retyped {name} ({len(body)} cells)")
            B.pin_supplement_formats(ws, header, [d[0] for d in dirty])
            print(f"   pinned number formats")

    print()
    if not grand_text:
        print("All supplement columns are already stored as numbers.")
    elif args.apply:
        print(f"Retyped {grand_fixed} text-stored cells to real numbers.")
    else:
        print(f"{grand_text} cells stored as text. Re-run with --apply to fix.")


if __name__ == '__main__':
    main()
