"""Find and repair appended blocks that never got their number formats.

Background
----------
sheets_append writes rows with USER_ENTERED, so Sheets *parses* them: the
Game Date string "2026-07-24" is stored as the date serial 46227, and RTilt
"1:05" as the day fraction 0.04513888. What makes them render (and CSV-export)
correctly is the number format applied to the block right after the write.

That formatting used to be ~10 separate un-retried API calls. A single 429 or
transient 5xx on any of them left the block raw — the sheet then shows 46227
in Game Date and 0.045138 in RTilt/OTilt, and the row looks like a botched
download. sheets_append now applies formatting as one atomic, retried,
verified batch, so new pushes can't land in this state; this script repairs
blocks written before that fix (or by any other writer that skips formatting).

Detection is exact rather than heuristic: a data row whose column A carries
anything other than a DATE number format is unformatted, full stop.

Usage
-----
    python scripts/fix_unformatted_blocks.py            # scan only (default)
    python scripts/fix_unformatted_blocks.py --apply    # scan and repair
    python scripts/fix_unformatted_blocks.py --apply --team ATL
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheets_append import (  # noqa: E402
    WORKBOOKS, _get_client, _sheets_retry, apply_block_format,
)


def _unformatted_blocks(sh, title):
    """Return [(start_row, end_row), ...] of contiguous data rows in `title`
    whose column A lacks a DATE number format."""
    meta = _sheets_retry(
        lambda: sh.fetch_sheet_metadata({
            'includeGridData': True,
            'ranges': [f"'{title}'!A2:A"],
            'fields': ('sheets(data(rowData(values('
                       'effectiveFormat(numberFormat/type),'
                       'effectiveValue/numberValue))))'),
        }),
        label=f'{title} format scan')

    row_data = (meta['sheets'][0].get('data') or [{}])[0].get('rowData', [])
    bad = []
    for i, row in enumerate(row_data):
        cell = (row.get('values') or [{}])[0]
        if not cell.get('effectiveValue'):
            continue  # blank row — nothing to format
        fmt = (cell.get('effectiveFormat') or {}).get('numberFormat', {})
        if fmt.get('type') != 'DATE':
            bad.append(i + 2)  # rowData starts at sheet row 2

    blocks = []
    for row in bad:
        if blocks and row == blocks[-1][1] + 1:
            blocks[-1][1] = row
        else:
            blocks.append([row, row])
    return [tuple(b) for b in blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='repair the blocks found (default: report only)')
    ap.add_argument('--team', help='limit to a single tab')
    args = ap.parse_args()

    gc = _get_client()
    found = repaired = 0

    for wb_name, wb_id in WORKBOOKS.items():
        sh = _sheets_retry(lambda: gc.open_by_key(wb_id), label=f'open {wb_name}')
        for ws in _sheets_retry(sh.worksheets, label=f'{wb_name} tab list'):
            if args.team and ws.title != args.team:
                continue
            blocks = _unformatted_blocks(sh, ws.title)
            if not blocks:
                continue

            header = _sheets_retry(lambda: ws.row_values(1),
                                   label=f'{ws.title} header')
            if not header:
                print(f"  {wb_name}/{ws.title}: no header row; skipping")
                continue

            for start, end in blocks:
                found += end - start + 1
                print(f"  {wb_name}/{ws.title}: rows {start}-{end} "
                      f"({end - start + 1}) unformatted")
                if not args.apply:
                    continue
                try:
                    apply_block_format(ws, header, start, end)
                    repaired += end - start + 1
                    print(f"    repaired rows {start}-{end}")
                except Exception as e:
                    print(f"    REPAIR FAILED ({type(e).__name__}: {e})")

    if not found:
        print("All tabs clean — every data row has a DATE-formatted Game Date.")
    elif args.apply:
        print(f"\nRepaired {repaired}/{found} unformatted rows.")
    else:
        print(f"\n{found} unformatted rows found. Re-run with --apply to fix.")


if __name__ == '__main__':
    main()
