"""fix_counts.py — apply the AUTO_BALL_STRIKE count corrections to the Sheets.

Surgical: touches ONLY the 'Count' cell of the affected PitchIDs, preserving every
other column (pitch-type retags etc.). Safety guard: a cell is overwritten ONLY if
it still holds the exact wrong value from the review; anything else (already fixed,
or an unexpected value) is skipped and reported, never clobbered.

  python3 scripts/fix_counts.py            # DRY RUN (read-only): show what would change
  python3 scripts/fix_counts.py --apply    # write the corrections to the Sheets
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
CSV = os.path.expanduser('~/Downloads/count_review_2026.csv')


def main():
    fixes = {}
    for r in csv.DictReader(open(CSV)):
        if r['category'] == 'AUTO_BALL_STRIKE':
            fixes[r['PitchID']] = {'stored': r['stored_count'], 'correct': r['correct_count']}
    print(f"{'APPLY' if APPLY else 'DRY RUN'}: {len(fixes)} AUTO count fixes to place\n", flush=True)

    gc = gspread.service_account()
    found = set()
    will_fix = already = mismatch = 0
    mismatches = []
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for i, ws in enumerate(sh.worksheets()):
            if i:
                time.sleep(1.2)
            rows = B.read_sheet_with_retry(ws)
            if not rows or len(rows) < 2:
                continue
            ci = {n: j for j, n in enumerate(rows[0]) if n}
            if 'PitchID' not in ci or 'Count' not in ci:
                continue
            pc, cc = ci['PitchID'], ci['Count']
            cells = []
            for li in range(1, len(rows)):
                row = rows[li]
                pid = row[pc] if pc < len(row) else ''
                if pid not in fixes:
                    continue
                found.add(pid)
                cur = row[cc] if cc < len(row) else ''
                st, co = fixes[pid]['stored'], fixes[pid]['correct']
                if cur == co:
                    already += 1
                elif cur == st:
                    will_fix += 1
                    cells.append(gspread.Cell(row=li + 1, col=cc + 1, value=co))
                else:
                    mismatch += 1
                    mismatches.append(f"{pid} [{label}/{ws.title}] sheet='{cur}' expected wrong='{st}' -> skip")
            if cells:
                print(f"  [{label}/{ws.title}] {len(cells)} count cell(s) "
                      f"{'WRITING' if APPLY else 'would fix'}", flush=True)
                if APPLY:
                    B.update_cells_with_retry(ws, cells, value_input_option='RAW')

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} SUMMARY ===")
    print(f"  will-fix (wrong -> correct): {will_fix}")
    print(f"  already correct (skipped):   {already}")
    print(f"  unexpected value (skipped):  {mismatch}")
    print(f"  not found in any sheet:      {len(fixes) - len(found)}")
    for m in mismatches[:20]:
        print(f"    {m}")
    missing = [p for p in fixes if p not in found]
    for p in missing[:20]:
        print(f"    NOT FOUND: {p} (stored {fixes[p]['stored']} -> {fixes[p]['correct']})")
    if not APPLY:
        print("\n(dry run only — re-run with --apply to write)")


if __name__ == '__main__':
    main()
