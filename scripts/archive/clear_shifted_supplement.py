"""clear_shifted_supplement.py — blank the supplement fields on the 659 pitches
that sat after an automatic ball, so the (now auto-ball-aware) backfill refills
them from the correct Savant pitch.

Only the fill-if-empty fields that the offset corrupted are cleared: RunExp,
bat-tracking, xwOBA/xBA/xSLG, Runners. ArmAngle and Barrel self-correct on the
backfill (ALWAYS_OVERWRITE); Event is left alone (set by the base scrape). Guarded:
a cell is only blanked if it currently holds a value. Dry-run first.

  python3 scripts/clear_shifted_supplement.py            # DRY RUN
  python3 scripts/clear_shifted_supplement.py --apply
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
CLEAR = ['RunExp', 'BatSpeed', 'SwingLength', 'AttackAngle', 'AttackDirection',
         'SwingPathTilt', 'xwOBA', 'xBA', 'xSLG', 'Runners']


def main():
    shifted = pickle.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', '_shifted_pitchids.pkl'), 'rb'))
    print(f"shifted pitches to clear: {len(shifted)}  fields: {CLEAR}\n")
    gc = gspread.service_account()
    total_cells = 0
    total_rows = 0
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
            cells = []
            rows_hit = 0
            for r_idx in range(1, len(data)):
                row = data[r_idx]
                pid = row[pcol] if pcol < len(row) else ''
                if pid not in shifted:
                    continue
                hit = False
                for f in CLEAR:
                    if f not in ci:
                        continue
                    c = ci[f]
                    cur = row[c] if c < len(row) else ''
                    if cur not in ('', None):
                        cells.append(gspread.Cell(row=r_idx + 1, col=c + 1, value=''))
                        hit = True
                if hit:
                    rows_hit += 1
            if cells:
                print(f"[{label}/{ws.title}] {rows_hit} pitches, {len(cells)} cells to blank")
                total_cells += len(cells)
                total_rows += rows_hit
                if APPLY:
                    B.update_cells_with_retry(ws, cells, value_input_option='RAW')
                    time.sleep(1.5)
    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'}: {total_rows} pitches, {total_cells} cells "
          f"{'blanked' if APPLY else 'would be blanked'} ===")
    if not APPLY:
        print("re-run with --apply, then run the complete backfill to refill them correctly.")


if __name__ == '__main__':
    main()
