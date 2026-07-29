"""fix_platez_earlyseason.py — sync PlateZ to the current feed for the games
Statcast reprocessed after scrape (the March/early-April vertical recalibration).

Characterization (scripts/_characterize_revisions.py) showed the sheet-vs-feed
PlateZ gap is NOT random: 2026-03 games run +0.081 ft high (99.6% of pitches),
2026-04 +0.035 (42.6%), and 2026-05 onward is 0.0% revised — a one-time,
settled early-season calibration. This rewrites PlateZ = current feed pZ for
every pitch that still differs (|sheet-feed| > THRESH), leaving PlateX alone.
The already-fixed lag corruption is skipped (those cells now equal the feed).

Guarded (re-reads each tab, only overwrites a cell still holding the detected
sheet value) + dry-run.
  python3 scripts/fix_platez_earlyseason.py            # DRY RUN
  python3 scripts/fix_platez_earlyseason.py --apply
"""
import os, sys, time, pickle, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
THRESH = 0.02            # sync any genuine revision above this (below = rounding/no-op)
CAP = 0.20               # safety: never touch >0.2 here (that was the corruption pass)
COORD_CACHE = os.path.join(ROOT, 'scripts', '_feed_platexz_cache.pkl')


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    feed = pickle.load(open(COORD_CACHE, 'rb'))
    print(f"feed coord cache: {len(feed)} games", flush=True)
    gc = gspread.service_account()

    staged = []   # (label,title,ws,ri,cz,sheet_pz,feed_pz)
    diffs = []
    print("reading sheets ...", flush=True)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.5)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            if 'PlateZ' not in ci:
                continue
            pc = ci['PitchID']; zc = ci['PlateZ']
            for ri in range(1, len(vals)):
                r = vals[ri]
                pid = r[pc] if pc < len(r) else ''
                p = pid.split('_')
                if len(p) != 3 or not p[0].isdigit(): continue
                pk, pa, pn = int(p[0]), int(p[1]), int(p[2])
                pm = feed.get(pk, {}).get(pa)
                if not pm or pn not in pm: continue
                fz = pm[pn][1]
                if fz is None: continue
                sz = sf(r[zc]) if zc < len(r) else None
                if sz is None: continue
                d = abs(sz - fz)
                if THRESH < d <= CAP:
                    staged.append((label, ws.title, ws, ri + 1, zc + 1, sz, fz))
                    diffs.append(d)
    # report
    print(f"\n=== PlateZ cells to sync: {len(staged)} ===")
    if diffs:
        ds = sorted(diffs)
        print(f"  |diff| min={ds[0]:.3f} p50={ds[len(ds)//2]:.3f} max={ds[-1]:.3f}")
        for b in (0.02, 0.05, 0.1):
            print(f"    > {b}: {sum(x > b for x in ds)}")
    bytab = defaultdict(int)
    for s in staged:
        bytab[s[1]] += 1
    print("  by tab:", dict(sorted(bytab.items())))

    if not APPLY:
        print("\n=== DRY RUN (no writes) ===")
        return
    byws = defaultdict(list)
    for label, title, ws, ri, cz, sz, fz in staged:
        byws[(label, title, ws)].append((ri, cz, sz, fz))
    total = 0; skipped = 0
    for (label, title, ws), items in byws.items():
        cur = ws.get_all_values()
        cells = []
        for ri, cz, sz, fz in items:
            row = cur[ri - 1] if ri - 1 < len(cur) else []
            curz = sf(row[cz - 1]) if cz - 1 < len(row) else None
            if curz is not None and abs(curz - sz) <= 0.005:
                cells.append(gspread.Cell(ri, cz, f"{fz:.3f}"))
            else:
                skipped += 1
        if cells:
            B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
            total += len(cells)
            print(f"  [{label}/{title}] wrote {len(cells)}", flush=True)
            time.sleep(1.0)
    print(f"\n=== APPLIED: {total} PlateZ cells   (guard-skipped: {skipped}) ===")


if __name__ == '__main__':
    main()
