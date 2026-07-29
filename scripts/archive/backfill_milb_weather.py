"""backfill_milb_weather.py — apply weather-adjusted xIVB/xHB to ROC/AAA.

The Triple-A feed omits venue elevation, so MiLB xIndVrtBrk/xHorzBrk were stored
un-adjusted (factor 1.0 = copies of IVB/HB). Now that the 9 Triple-A parks have
elevations in Pitcher2026.VENUE_ELEVATION_FT_OVERRIDE, recompute the per-game
factor (elevation + the feed's recorded temp) and rewrite
  xIndVrtBrk = round(IndVertBrk * factor, 1)
  xHorzBrk   = round(HorzBrk   * factor, 1)
and refresh the per-game sidecar (data/game_weather_rs.json) so the stored
factor matches. Guarded, dry-run first.

  python3 scripts/backfill_milb_weather.py            # DRY RUN
  python3 scripts/backfill_milb_weather.py --apply
"""
import os, sys, json, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import gspread, backfill_supplement as B
import Pitcher2026 as P

APPLY = '--apply' in sys.argv
SIDECAR = os.path.join(ROOT, 'data', 'game_weather_rs.json')
MILB_VENUES = {2773, 2756, 3230, 2797, 2823, 4670, 5410, 2852, 4271}  # the 9 Triple-A parks
MILB_BOOK = 'NLE2026'
MILB_TABS = ('ROC', 'AAA')


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    weather = json.load(open(SIDECAR))
    # per-game factor for MiLB games at known-elevation parks
    gf = {}          # game_pk(str) -> (factor, elev, rho, temp)
    for g, e in weather.items():
        vid = e.get('venueId')
        if vid not in MILB_VENUES:
            continue
        elev = P.VENUE_ELEVATION_FT_OVERRIDE.get(vid)
        temp = e.get('tempF')
        cond = (e.get('condition') or '').strip().lower()
        if cond in P.ROOF_CLOSED_CONDITIONS:
            temp = P.ROOF_CLOSED_TEMP_F
        t = temp if temp is not None else P.DEFAULT_TEMP_F
        rho = P.compute_air_density(elev, t)
        factor = P.compute_weather_adj_factor(rho)
        gf[str(g)] = (factor, elev, rho, temp)
    print(f"MiLB games with a computed factor: {len(gf)}", flush=True)
    fs = sorted(f[0] for f in gf.values())
    if fs:
        print(f"  factor range: {fs[0]:.4f} .. {fs[-1]:.4f}  (was 1.0000 for all)")

    gc = gspread.service_account()
    sh = gc.open_by_key(B.SPREADSHEET_IDS[MILB_BOOK])
    staged = []   # (tab, ws, ri, col1, newstr)
    per = defaultdict(int)
    for name in MILB_TABS:
        ws = sh.worksheet(name)
        vals = ws.get_all_values()
        ci = {n: j for j, n in enumerate(vals[0]) if n}
        need = ['PitchID', 'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk']
        if not all(c in ci for c in need):
            print(f"[{name}] missing columns — skipping"); continue
        pc = ci['PitchID']
        for ri in range(1, len(vals)):
            r = vals[ri]
            pid = r[pc] if pc < len(r) else ''
            g = pid.split('_')[0]
            if not g.isdigit() or g not in gf:
                continue
            factor = gf[g][0]
            for raw, xcol in (('IndVertBrk', 'xIndVrtBrk'), ('HorzBrk', 'xHorzBrk')):
                mv = sf(r[ci[raw]]) if ci[raw] < len(r) else None
                if mv is None:
                    continue
                newv = round(mv * factor, 1)
                cur = sf(r[ci[xcol]]) if ci[xcol] < len(r) else None
                if cur is None or abs(cur - newv) > 0.049:
                    staged.append((name, ws, ri + 1, ci[xcol] + 1, f"{newv:.1f}"))
                    per[xcol] += 1
    print(f"\ncells to rewrite: {len(staged)}   by column: {dict(per)}", flush=True)

    if not APPLY:
        print("=== DRY RUN (no writes) ===")
        return
    # write cells per tab
    byws = defaultdict(list)
    for name, ws, ri, col, val in staged:
        byws[(name, ws)].append((ri, col, val))
    total = 0
    for (name, ws), items in byws.items():
        cells = [gspread.Cell(ri, col, val) for (ri, col, val) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        print(f"  [{name}] wrote {len(cells)}", flush=True)
        time.sleep(1.0)
    # refresh the sidecar so stored factors match
    for g, (factor, elev, rho, temp) in gf.items():
        if g in weather:
            weather[g]['elevationFt'] = elev
            weather[g]['rho'] = round(rho, 5)
            weather[g]['factor'] = round(factor, 5)
    json.dump(weather, open(SIDECAR, 'w'), indent=0, sort_keys=True)  # match _record_game_weather
    print(f"=== APPLIED: {total} cells; sidecar refreshed for {len(gf)} games ===")


if __name__ == '__main__':
    main()
