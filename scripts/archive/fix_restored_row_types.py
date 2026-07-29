"""fix_restored_row_types.py — repair the cell TYPES on the 10 restored PAs.

restore_missing_pitch_pas.py appended rows with RAW + str(), so numeric fields
(Velocity, Spin, RTilt/OTilt, movement, location, hit data) were stored as TEXT
and pandas NA was stored as the literal string "<NA>"/"nan". That makes those
rows sort separately from the numeric originals and shows junk text in blank
cells. This rewrites each affected row the way sheets_append does:

  - determine each column's canonical type from the existing (non-restored) rows
  - blank any NA-like string
  - text-force values in text-typed columns (so Count "1-2" etc. don't misparse)
  - write via USER_ENTERED so numeric columns parse to real numbers / time serials
  - repaint the per-column number formats from row 2

  python3 scripts/fix_restored_row_types.py           # DRY RUN
  python3 scripts/fix_restored_row_types.py --apply
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
import sheets_append as SA

APPLY = '--apply' in sys.argv
CASES = [('824448', 43), ('824527', 53), ('824525', 45), ('824280', 51), ('824600', 48),
         ('823056', 69), ('823545', 25), ('823700', 63), ('822725', 65), ('822968', 70)]
PREFIXES = tuple(f"{pk}_{ab:03d}_" for pk, ab in CASES)
NA_STRINGS = {'<NA>', 'nan', 'NaN', 'NAN', 'None', 'NaT', '#N/A', '#n/a'}


def col_letter(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def main():
    gc = gspread.service_account()
    total_rows = 0
    total_na = 0
    total_num = 0
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            if wi:
                time.sleep(0.6)
            fmt = ws.get_all_values()                      # display strings
            if not fmt or len(fmt) < 2:
                continue
            header = fmt[0]
            ci = {n: j for j, n in enumerate(header) if n}
            if 'PitchID' not in ci:
                continue
            pc = ci['PitchID']
            our_rows = [ri for ri in range(1, len(fmt))
                        if pc < len(fmt[ri]) and str(fmt[ri][pc]).startswith(PREFIXES)]
            if not our_rows:
                continue
            ncol = len(header)
            unf = ws.get_all_values(value_render_option='UNFORMATTED_VALUE')  # native types
            # canonical per-column type from NON-restored rows
            base_num = [False] * ncol
            seen = [False] * ncol
            for ri in range(1, len(unf)):
                if ri in our_rows:
                    continue
                row = unf[ri]
                for c in range(min(ncol, len(row))):
                    if seen[c]:
                        continue
                    v = row[c]
                    if v == '' or v is None:
                        continue
                    base_num[c] = isinstance(v, (int, float)) and not isinstance(v, bool)
                    seen[c] = True

            data = []   # {range, values}
            for ri in our_rows:
                src = fmt[ri]
                out = []
                for c in range(ncol):
                    v = src[c] if c < len(src) else ''
                    if str(v).strip() in NA_STRINGS:
                        v = ''
                        total_na += 1
                    elif v != '' and not base_num[c]:
                        # text-typed column: protect from USER_ENTERED coercion
                        v = "'" + str(v)
                    elif v != '' and base_num[c]:
                        total_num += 1
                    out.append(v)
                data.append({'range': f"A{ri + 1}:{col_letter(ncol)}{ri + 1}", 'values': [out]})
                total_rows += 1
            print(f"[{label}/{ws.title}] {len(our_rows)} restored rows to retype")
            if APPLY:
                ws.batch_update(data, value_input_option='USER_ENTERED')
                for ri in our_rows:
                    SA._paste_number_formats_from_row(ws, 2, ri + 1, ri + 1, ncol)
                    time.sleep(0.2)
                time.sleep(1.0)

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'}: {total_rows} rows, "
          f"{total_num} numeric cells retyped, {total_na} NA strings blanked ===")


if __name__ == '__main__':
    main()
