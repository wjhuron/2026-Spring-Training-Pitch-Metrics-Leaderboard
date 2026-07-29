"""fix_missing_pitches.py — insert the 10 feed-revision missing pitches.

For each affected PA: re-scrape the game with the patched Pitcher2026 (the current,
correct sequence), match its pitches to the existing Sheet rows by physics
(velocity + plate coords), then:
  - UPDATE existing rows' PitchID / Count / Description / Event where the feed
    revision shifted them, preserving Pitch Type retags + all other columns.
  - INSERT the one new pitch (raw feed type; Statcast supplement fields fill in
    on the next backfill_supplement run, keyed on PitchID).
Guarded + dry-run-first. Only the listed 10 PAs are touched.

  python3 scripts/fix_missing_pitches.py            # DRY RUN
  python3 scripts/fix_missing_pitches.py --apply
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
from Pitcher2026 import BaseballSavantFocusedDownloader

APPLY = '--apply' in sys.argv
CASES = [('824448', 43), ('824527', 53), ('824525', 45), ('824280', 51), ('824600', 48),
         ('823056', 69), ('823545', 25), ('823700', 63), ('822725', 65), ('822968', 70)]
UPD = ['PitchID', 'Count', 'Description', 'Event']   # feed-revision-affected fields


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pkey(velo, px, pz):
    return (round(sf(velo) if sf(velo) is not None else -9, 1),
            round(sf(px) if sf(px) is not None else -9, 2),
            round(sf(pz) if sf(pz) is not None else -9, 2))


def main():
    dl = BaseballSavantFocusedDownloader()
    # re-scrape each game once; index fresh rows by PA
    fresh = {}
    for pk, ab in CASES:
        if pk not in fresh:
            df = dl.download_game_data(int(pk))
            fresh[pk] = df
    # fresh rows per PA (list of dicts), plus a physics->fresh-row map
    pa_fresh = {}
    for pk, ab in CASES:
        df = fresh[pk]
        rows = [r._asdict() if hasattr(r, '_asdict') else dict(r)
                for _, r in df[df['PitchID'].str.startswith(f"{pk}_{ab:03d}_")].iterrows()]
        pa_fresh[(pk, ab)] = rows
    want_pids_prefix = {f"{pk}_{ab:03d}_" for pk, ab in CASES}

    gc = gspread.service_account()
    total_upd = total_ins = 0
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for i, ws in enumerate(sh.worksheets()):
            if i:
                time.sleep(1.0)
            data = B.read_sheet_with_retry(ws)
            if not data or len(data) < 2:
                continue
            header = data[0]
            ci = {n: j for j, n in enumerate(header) if n}
            if 'PitchID' not in ci:
                continue
            pcol = ci['PitchID']
            # which of our PAs live in this tab?
            tab_pas = set()
            for li in range(1, len(data)):
                pid = data[li][pcol] if pcol < len(data[li]) else ''
                for pk, ab in CASES:
                    if pid.startswith(f"{pk}_{ab:03d}_"):
                        tab_pas.add((pk, ab))
            if not tab_pas:
                continue
            cells = []
            inserts = []   # (sheet_row_index_to_insert_at, row_values)
            for pk, ab in sorted(tab_pas):
                # existing sheet rows for this PA: li -> row
                sheet_rows = [(li, data[li]) for li in range(1, len(data))
                              if (data[li][pcol] if pcol < len(data[li]) else '').startswith(f"{pk}_{ab:03d}_")]
                fresh_rows = pa_fresh[(pk, ab)]
                # map physics -> fresh row
                fmap = {pkey(r.get('Velocity'), r.get('PlateX'), r.get('PlateZ')): r for r in fresh_rows}
                matched_fresh = set()
                first_li = sheet_rows[0][0]
                for li, row in sheet_rows:
                    k = pkey(row[ci['Velocity']] if ci.get('Velocity', 99) < len(row) else None,
                             row[ci['PlateX']] if ci.get('PlateX', 99) < len(row) else None,
                             row[ci['PlateZ']] if ci.get('PlateZ', 99) < len(row) else None)
                    fr = fmap.get(k)
                    if fr is None:
                        print(f"  !! {label}/{ws.title} {pk}_{ab:03d}: sheet row {row[pcol]} had no physics match in fresh scrape — SKIPPING PA")
                        break
                    matched_fresh.add(id(fr))
                    for fld in UPD:
                        if fld in ci:
                            newv = fr.get(fld)
                            newv = '' if newv is None else str(newv)
                            cur = row[ci[fld]] if ci[fld] < len(row) else ''
                            if cur != newv:
                                cells.append(gspread.Cell(li + 1, ci[fld] + 1, newv))
                else:
                    # the unmatched fresh pitch(es) = new; build full row(s)
                    for fr in fresh_rows:
                        if id(fr) in matched_fresh:
                            continue
                        newrow = [('' if fr.get(h) is None else str(fr.get(h))) for h in header]
                        inserts.append((first_li + 1, newrow, fr.get('PitchID'), fr.get('Pitch Type'), fr.get('Description'), fr.get('Count')))
            # report
            if cells or inserts:
                print(f"\n[{label}/{ws.title}]")
                for c in cells:
                    print(f"   update r{c.row} {header[c.col-1]} -> '{c.value}'")
                for pos, _, pid, pt, desc, cnt in inserts:
                    print(f"   INSERT at row {pos}: {pid}  {desc} ({cnt})  type={pt}")
                total_upd += len(cells); total_ins += len(inserts)
                if APPLY:
                    if cells:
                        B.update_cells_with_retry(ws, cells, value_input_option='RAW')
                    # insert from highest row down so indices stay valid
                    for pos, newrow, *_ in sorted(inserts, key=lambda x: -x[0]):
                        ws.insert_row(newrow, index=pos, value_input_option='RAW')

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'}: {total_upd} cell updates, {total_ins} pitch inserts ===")
    if not APPLY:
        print("(dry run — re-run with --apply; then run backfill_supplement to fill the new pitches' Statcast fields)")


if __name__ == '__main__':
    main()
