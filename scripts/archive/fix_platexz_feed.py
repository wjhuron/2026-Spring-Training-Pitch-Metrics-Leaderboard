"""fix_platexz_feed.py — fix ONLY lag-corrupted PlateX/PlateZ against the live feed.

The auto-ball coordinate-lag bug (since fixed in Pitcher2026) copied a NEIGHBOR
pitch's plate coordinates onto a pitch. We detect that signature precisely:
a pitch is corrupt iff its STORED (x,z) matches a *sibling* pitch's FEED (x,z)
in the same PA (both within MATCH_TOL) while differing from its OWN feed coords.
That isolates the bug from benign Statcast reprocessing (sub-inch revisions that
touch ~18% of pitches but match no sibling). A magnitude backstop (|diff|>BIG on
either coord) catches any lag whose sibling coords aren't a clean match.

For every flagged pitch we rewrite BOTH PlateX and PlateZ to that pitch's own
feed value, so the (x,z) pair stays from one consistent source. We do NOT touch
Savant (feed PlateZ sits +0.071 ft above Savant by design) and do NOT touch the
~108k revision-noise cells.

Guarded (only overwrites a cell that still holds the flagged value) + dry-run.
  python3 scripts/fix_platexz_feed.py            # DRY RUN + writes review CSV
  python3 scripts/fix_platexz_feed.py --apply
"""
import os, sys, time, csv, pickle, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import requests, gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
MATCH_TOL = 0.03   # stored coords "match" a sibling's feed coords within this
OWN_TOL   = 0.05   # ...while differing from its own feed by more than this
BIG       = 0.20   # magnitude backstop (either coord)
FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
CACHE = os.path.join(ROOT, 'scripts', '_feed_platexz_cache.pkl')


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def feed_coords(pk):
    """{PA(int) -> {feed_pitchNum(int) -> (px3, pz3)}} for one game."""
    try:
        r = requests.get(FEED.format(pk), timeout=60)
        plays = r.json().get('liveData', {}).get('plays', {}).get('allPlays', [])
    except Exception:
        return None
    out = defaultdict(dict)
    for play in plays:
        ab = play.get('atBatIndex', 0) + 1
        for ev in play.get('playEvents', []):
            if not ev.get('isPitch', False):
                continue
            pn = ev.get('pitchNumber') or 0
            c = ev.get('pitchData', {}).get('coordinates', {})
            px, pz = c.get('pX'), c.get('pZ')
            out[ab][pn] = (round(px, 3) if px is not None else None,
                           round(pz, 3) if pz is not None else None)
    return dict(out)


def main():
    gc = gspread.service_account()
    tabs = []; pks = set()
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
            if 'PlateX' not in ci or 'PlateZ' not in ci:
                continue
            tabs.append((label, ws.title, ws, vals, ci))
            pc = ci['PitchID']
            for r in vals[1:]:
                p = (r[pc] if pc < len(r) else '').split('_')
                if len(p) == 3 and p[0].isdigit():
                    pks.add(int(p[0]))
    print(f"MLB tabs: {len(tabs)}   distinct games: {len(pks)}", flush=True)

    if os.path.exists(CACHE):
        feed = pickle.load(open(CACHE, 'rb'))
        print(f"loaded feed cache: {len(feed)} games", flush=True)
    else:
        print("pulling feeds ...", flush=True)
        feed = {}; failed = []
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(feed_coords, pk): pk for pk in pks}
            done = 0
            for fu in cf.as_completed(futs):
                pk = futs[fu]; res = fu.result()
                if res is None: failed.append(pk)
                else: feed[pk] = res
                done += 1
                if done % 300 == 0: print(f"  {done}/{len(pks)}", flush=True)
        print(f"feed games: {len(feed)}   failed: {len(failed)} {failed[:8]}", flush=True)
        pickle.dump(feed, open(CACHE, 'wb'))

    # detect
    flagged = {}   # pid -> dict(tab,label,ri,ci_x,ci_z,sx,sz,fx,fz,reason,dx,dz)
    for label, title, ws, vals, ci in tabs:
        pc = ci['PitchID']
        for ri in range(1, len(vals)):
            r = vals[ri]
            pid = r[pc] if pc < len(r) else ''
            p = pid.split('_')
            if len(p) != 3 or not p[0].isdigit(): continue
            pk, pa, pn = int(p[0]), int(p[1]), int(p[2])
            pa_map = feed.get(pk, {}).get(pa)
            if not pa_map or pn not in pa_map: continue
            fx, fz = pa_map[pn]
            if fx is None or fz is None: continue
            sx = sf(r[ci['PlateX']]) if ci['PlateX'] < len(r) else None
            sz = sf(r[ci['PlateZ']]) if ci['PlateZ'] < len(r) else None
            if sx is None or sz is None: continue
            dx, dz = sx - fx, sz - fz
            own_off = (abs(dx) > OWN_TOL or abs(dz) > OWN_TOL)
            if not own_off:
                continue
            # structural: does stored (sx,sz) match a SIBLING's feed coords?
            sib = None
            for opn, (ofx, ofz) in pa_map.items():
                if opn == pn or ofx is None or ofz is None: continue
                if abs(sx - ofx) < MATCH_TOL and abs(sz - ofz) < MATCH_TOL:
                    sib = opn; break
            reason = None
            if sib is not None:
                reason = f"lag<-pn{sib:02d}"
            elif abs(dx) > BIG or abs(dz) > BIG:
                reason = "big"
            if reason:
                flagged[pid] = dict(label=label, title=title, ws=ws, ri=ri + 1,
                                    cx=ci['PlateX'] + 1, cz=ci['PlateZ'] + 1,
                                    sx=sx, sz=sz, fx=fx, fz=fz, reason=reason,
                                    dx=dx, dz=dz, pk=pk)

    # report
    byreason = defaultdict(int); bycol = [0, 0]
    for pid, f in flagged.items():
        byreason[f['reason'].split('<-')[0]] += 1
        if abs(f['dx']) > OWN_TOL: bycol[0] += 1
        if abs(f['dz']) > OWN_TOL: bycol[1] += 1
    print(f"\n=== flagged pitches: {len(flagged)} ===")
    print(f"  by reason: {dict(byreason)}")
    print(f"  PlateX actually off: {bycol[0]}   PlateZ actually off: {bycol[1]}")
    mx = max((max(abs(f['dx']), abs(f['dz'])) for f in flagged.values()), default=0)
    mn = min((max(abs(f['dx']), abs(f['dz'])) for f in flagged.values()), default=0)
    print(f"  max coord error: {mx:.3f}   min (of flagged): {mn:.3f}")
    print("  examples:")
    for pid, f in sorted(flagged.items(), key=lambda kv: -max(abs(kv[1]['dx']), abs(kv[1]['dz'])))[:20]:
        print(f"    {pid} [{f['title']}] {f['reason']:12s} "
              f"X {f['sx']:+.3f}->{f['fx']:+.3f}  Z {f['sz']:+.3f}->{f['fz']:+.3f}")

    # review CSV
    csvp = os.path.expanduser('~/Downloads/platexz_lag_fixes.csv')
    with open(csvp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['PitchID', 'Team', 'reason', 'sheet_PlateX', 'feed_PlateX',
                    'sheet_PlateZ', 'feed_PlateZ'])
        for pid, f in sorted(flagged.items()):
            w.writerow([pid, f['title'], f['reason'], f['sx'], f['fx'], f['sz'], f['fz']])
    print(f"\n  review CSV -> {csvp}")

    if not APPLY:
        print("\n=== DRY RUN (no writes) ===")
        return
    # guarded apply: re-read each tab, only overwrite cells still holding the flagged value
    byws = defaultdict(list)
    for pid, f in flagged.items():
        byws[(f['label'], f['title'], f['ws'])].append(f)
    total = 0; skipped = 0
    for (label, title, ws), items in byws.items():
        cur_vals = ws.get_all_values()
        cells = []
        for f in items:
            row = cur_vals[f['ri'] - 1] if f['ri'] - 1 < len(cur_vals) else []
            cx0, cz0 = f['cx'] - 1, f['cz'] - 1
            curx = sf(row[cx0]) if cx0 < len(row) else None
            curz = sf(row[cz0]) if cz0 < len(row) else None
            if curx is not None and abs(curx - f['sx']) <= 0.005:
                cells.append(gspread.Cell(f['ri'], f['cx'], f"{f['fx']:.3f}"))
            else:
                skipped += 1
            if curz is not None and abs(curz - f['sz']) <= 0.005:
                cells.append(gspread.Cell(f['ri'], f['cz'], f"{f['fz']:.3f}"))
            else:
                skipped += 1
        if cells:
            B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
            total += len(cells)
            print(f"  [{label}/{title}] wrote {len(cells)} cells ({len(items)} pitches)", flush=True)
            time.sleep(1.0)
    print(f"\n=== APPLIED: {total} cells across {len(flagged)} pitches   (guard-skipped: {skipped}) ===")


if __name__ == '__main__':
    main()
