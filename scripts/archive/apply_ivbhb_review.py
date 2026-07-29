"""apply_ivbhb_review.py — apply the IVB/HB >= 2 discrepancies (the ones earlier
left for review). Overwrites IndVertBrk/HorzBrk with Savant and recomputes the
weather-adjusted xIndVrtBrk/xHorzBrk = round(new * game_factor, 1). Guarded +
dry-run first. (The <2 group was already applied by apply_mismatch_fixes.py.)

  python3 scripts/apply_ivbhb_review.py            # DRY RUN
  python3 scripts/apply_ivbhb_review.py --apply
"""
import os, sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
from numbers_parser import Document

APPLY = '--apply' in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weather = json.load(open(os.path.join(ROOT, 'data', 'game_weather_rs.json')))


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def factor_of(pid):
    return weather.get(pid.split('_')[0], {}).get('factor', 1.0) or 1.0


def main():
    doc = Document(os.path.expanduser('~/Downloads/MismatchFixes.numbers'))
    tabs = {s.name: s.tables[0].rows(values_only=True) for s in doc.sheets}
    writes = defaultdict(list)   # pid -> (col, new, guard_col, guard_old)
    n = 0
    for tab, raw, xcol in [('IndVertBrk all', 'IndVertBrk', 'xIndVrtBrk'),
                           ('HorzBrk all', 'HorzBrk', 'xHorzBrk')]:
        rows = tabs[tab]; h = rows[0]
        pi = h.index('PitchID'); sa = h.index('Savant'); shc = h.index('Sheet'); di = h.index('Diff')
        for r in rows[1:]:
            d = sf(r[di]); sv = sf(r[sa])
            if d is None or sv is None or abs(d) < 2:
                continue
            pid = str(r[pi]); old = sf(r[shc]); newv = round(sv, 1)
            writes[pid].append((raw, str(newv), raw, old))
            writes[pid].append((xcol, str(round(newv * factor_of(pid), 1)), raw, old))
            n += 1
    print(f"IVB/HB >=2 pitches to apply: {n}  (across {len(writes)} pitches, +xIVB/xHB)")

    gc = gspread.service_account()
    total = 0; skips = []
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            if ws.title.upper() not in B.ALL_TRACKED_TEAMS:
                continue
            if wi:
                time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n2: j for j, n2 in enumerate(vals[0]) if n2}
            pc = ci['PitchID']
            cells = []
            for ri in range(1, len(vals)):
                pid = str(vals[ri][pc]) if pc < len(vals[ri]) else ''
                if pid not in writes:
                    continue
                for col, newstr, gcol, gold in writes[pid]:
                    if col not in ci or gcol not in ci:
                        continue
                    curg = sf(vals[ri][ci[gcol]]) if ci[gcol] < len(vals[ri]) else None
                    if gold is not None and (curg is None or abs(curg - gold) > 0.06):
                        skips.append(f"{pid}.{col} (guard {gcol} cur={curg} reviewed={gold})")
                        continue
                    cur = vals[ri][ci[col]] if ci[col] < len(vals[ri]) else ''
                    if str(cur) == newstr:
                        continue
                    cells.append(gspread.Cell(ri + 1, ci[col] + 1, newstr))
            if cells:
                print(f"  [{label}/{ws.title}] {len(cells)} cells")
                total += len(cells)
                if APPLY:
                    B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
                    time.sleep(1.0)
    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'}: {total} cells ===")
    if skips:
        print(f"guard skips: {len(skips)}")
        for s in skips[:15]:
            print("  ", s)


if __name__ == '__main__':
    main()
