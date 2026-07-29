"""apply_mismatch_fixes.py — apply Wally's reviewed tracking corrections from
MismatchFixes.numbers. Guarded (only overwrites a cell that still holds the
reviewed 'Sheet' value) + dry-run first.

Rules (per Wally, 2026-07):
  Spin Rate : Verdict KEEP -> leave; DELETE -> blank the cell; blank -> Savant.
  Velocity  : Savant for all.
  IndVertBrk/HorzBrk : |Diff| < 2 -> Savant, AND recompute the weather-adjusted
              xIndVrtBrk/xHorzBrk = round(newValue * game_factor, 1); |Diff| >= 2 left.
  Extension, SzTop, SzBot, ExitVelo, Distance, HC_X, HC_Y : Savant for all.
  PlateX / RelPosX / RelPosZ / PlateZ : untouched.

  python3 scripts/apply_mismatch_fixes.py            # DRY RUN
  python3 scripts/apply_mismatch_fixes.py --apply
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


def fmt(v, prec):
    return str(int(round(v))) if prec == 0 else str(round(v, prec))


def main():
    doc = Document(os.path.expanduser('~/Downloads/MismatchFixes.numbers'))
    tabs = {s.name: s.tables[0].rows(values_only=True) for s in doc.sheets}
    # writes[pid] = list of (col, new_str, guard_col, guard_old_float_or_None)
    writes = defaultdict(list)
    cat_counts = defaultdict(int)

    def col_idx(rows, name):
        return rows[0].index(name)

    def simple(tab, sheetcol, prec):
        rows = tabs[tab]; pi = col_idx(rows, 'PitchID'); sa = col_idx(rows, 'Savant'); shc = col_idx(rows, 'Sheet')
        for r in rows[1:]:
            pid = str(r[pi]); sv = sf(r[sa])
            if sv is None:
                continue
            writes[pid].append((sheetcol, fmt(sv, prec), sheetcol, sf(r[shc])))
            cat_counts[tab] += 1

    # Spin Rate
    rows = tabs['Spin Rate all']; pi = col_idx(rows, 'PitchID'); sa = col_idx(rows, 'Savant'); shc = col_idx(rows, 'Sheet'); vi = col_idx(rows, 'Verdict')
    for r in rows[1:]:
        pid = str(r[pi]); verd = (str(r[vi]) if r[vi] is not None else '').strip().upper()
        old = sf(r[shc])
        if verd == 'KEEP':
            continue
        if verd == 'DELETE':
            writes[pid].append(('Spin Rate', '', 'Spin Rate', old)); cat_counts['Spin DELETE'] += 1
        else:
            sv = sf(r[sa])
            if sv is not None:
                writes[pid].append(('Spin Rate', fmt(sv, 0), 'Spin Rate', old)); cat_counts['Spin ->Savant'] += 1

    simple('Velocity all', 'Velocity', 1)
    simple('Extension all', 'Extension', 2)
    simple('SzTop all', 'SzTop', 2)
    simple('SzBot all', 'SzBot', 2)
    simple('ExitVelo all', 'ExitVelo', 1)
    simple('Distance all', 'Distance', 0)
    simple('HC_X all', 'HC_X', 2)
    simple('HC_Y all', 'HC_Y', 2)

    # IVB / HB: <2 only, plus weather-adjusted dependent
    for tab, raw, xcol in [('IndVertBrk all', 'IndVertBrk', 'xIndVrtBrk'),
                           ('HorzBrk all', 'HorzBrk', 'xHorzBrk')]:
        rows = tabs[tab]; pi = col_idx(rows, 'PitchID'); sa = col_idx(rows, 'Savant'); shc = col_idx(rows, 'Sheet'); di = col_idx(rows, 'Diff')
        for r in rows[1:]:
            pid = str(r[pi]); d = sf(r[di]); sv = sf(r[sa])
            if d is None or sv is None or abs(d) >= 2:
                continue
            old = sf(r[shc]); newv = round(sv, 1)
            writes[pid].append((raw, fmt(newv, 1), raw, old))
            writes[pid].append((xcol, fmt(round(newv * factor_of(pid), 1), 1), raw, old))  # guard on raw cell
            cat_counts[tab + ' (<2)'] += 1

    print("planned changes by category:")
    for k in sorted(cat_counts):
        print(f"  {k}: {cat_counts[k]}")
    print(f"  distinct pitches touched: {len(writes)}\n")

    # apply per tab
    gc = gspread.service_account()
    total = 0; guard_skips = []
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
            ci = {n: j for j, n in enumerate(vals[0]) if n}
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
                    # guard: the field we key on must still hold the reviewed value
                    if gold is None:
                        ok = True
                    elif curg is None:
                        ok = False
                    else:
                        tol = 1.0 if gcol in ('Distance', 'HC_X', 'HC_Y') else (0.5 if gcol == 'Spin Rate' else 0.06)
                        ok = abs(curg - gold) <= tol
                    if not ok:
                        guard_skips.append(f"{pid}.{col} (guard {gcol}: cur={curg} reviewed={gold})")
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
    if guard_skips:
        print(f"guard skips (cell no longer holds reviewed value): {len(guard_skips)}")
        for s in guard_skips[:15]:
            print(f"   {s}")


if __name__ == '__main__':
    main()
