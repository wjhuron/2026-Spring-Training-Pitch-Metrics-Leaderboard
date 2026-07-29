"""Characterize sheet-vs-current-feed PlateX/PlateZ revisions: random jitter vs
systematic (whole-game reprocessing, or park/date recalibration). READ-ONLY.

Uses the feed coord cache from fix_platexz_feed.py + a light date/venue pull."""
import os, sys, time, pickle, warnings, statistics as st, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import requests, gspread
import backfill_supplement as B

FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
COORD_CACHE = os.path.join(ROOT, 'scripts', '_feed_platexz_cache.pkl')
META_CACHE = os.path.join(ROOT, 'scripts', '_feed_meta_cache.pkl')
REV = 0.02   # |signed diff| above this = "revised"


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def game_meta(pk):
    try:
        gd = requests.get(FEED.format(pk), timeout=60).json().get('gameData', {})
        return (gd.get('datetime', {}).get('officialDate', ''),
                gd.get('venue', {}).get('id'), gd.get('venue', {}).get('name', ''))
    except Exception:
        return ('', None, '')


def main():
    feed = pickle.load(open(COORD_CACHE, 'rb'))
    print(f"feed coord cache: {len(feed)} games", flush=True)

    if os.path.exists(META_CACHE):
        meta = pickle.load(open(META_CACHE, 'rb'))
    else:
        print("pulling game date/venue ...", flush=True)
        meta = {}
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(game_meta, pk): pk for pk in feed}
            for i, fu in enumerate(cf.as_completed(futs)):
                meta[futs[fu]] = fu.result()
                if (i + 1) % 400 == 0: print(f"  {i+1}/{len(feed)}", flush=True)
        pickle.dump(meta, open(META_CACHE, 'wb'))

    gc = gspread.service_account()
    # per pitch signed diffs (sheet - feed), skipping anything still >0.2 (residual corruption)
    dz_all = []; dx_all = []
    by_game = defaultdict(lambda: {'dz': [], 'dx': [], 'n': 0})
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
            pc = ci['PitchID']
            for r in vals[1:]:
                pid = r[pc] if pc < len(r) else ''
                p = pid.split('_')
                if len(p) != 3 or not p[0].isdigit(): continue
                pk, pa, pn = int(p[0]), int(p[1]), int(p[2])
                pm = feed.get(pk, {}).get(pa)
                if not pm or pn not in pm: continue
                fx, fz = pm[pn]
                sx = sf(r[ci['PlateX']]) if ci['PlateX'] < len(r) else None
                sz = sf(r[ci['PlateZ']]) if ci['PlateZ'] < len(r) else None
                g = by_game[pk]
                if sx is not None and fx is not None and abs(sx - fx) <= 0.2:
                    dx_all.append(sx - fx); g['dx'].append(sx - fx)
                if sz is not None and fz is not None and abs(sz - fz) <= 0.2:
                    dz_all.append(sz - fz); g['dz'].append(sz - fz); g['n'] += 1

    def desc(a, name):
        n = len(a)
        rev = sum(abs(v) > REV for v in a)
        print(f"\n{name}: n={n}")
        print(f"  signed  mean={st.mean(a):+.4f}  median={st.median(a):+.4f}  sd={st.pstdev(a):.4f}")
        print(f"  |diff|>{REV}: {rev} ({100*rev/n:.1f}%)   >0.05: {sum(abs(v)>0.05 for v in a)}  >0.1: {sum(abs(v)>0.1 for v in a)}")
    desc(dz_all, "PlateZ  sheet-feed (|d|<=0.2)")
    desc(dx_all, "PlateX  sheet-feed (|d|<=0.2)")

    # --- per-game clustering test (PlateZ): is fraction-revised bimodal? ---
    fracs = []
    gmeans = []
    for pk, g in by_game.items():
        if g['n'] < 30: continue
        fr = sum(abs(v) > REV for v in g['dz']) / len(g['dz'])
        fracs.append((pk, fr, st.mean(g['dz']), len(g['dz'])))
        gmeans.append(st.mean(g['dz']))
    fracs.sort(key=lambda t: t[1])
    print(f"\n=== per-game PlateZ revision clustering (games with >=30 pitches: {len(fracs)}) ===")
    buckets = [0, 0, 0, 0, 0]  # <5% 5-20 20-50 50-80 >80
    for pk, fr, mn, n in fracs:
        buckets[min(int(fr * 5), 4) if fr < 0.8 else 4] += 1
    print(f"  games by fraction-of-pitches-revised:  <20%: {sum(1 for _,fr,_,_ in fracs if fr<0.2)}"
          f"   20-50%: {sum(1 for _,fr,_,_ in fracs if 0.2<=fr<0.5)}"
          f"   50-80%: {sum(1 for _,fr,_,_ in fracs if 0.5<=fr<0.8)}"
          f"   >=80%: {sum(1 for _,fr,_,_ in fracs if fr>=0.8)}")
    print(f"  per-game mean PlateZ diff: sd across games = {st.pstdev(gmeans):.4f} (0=all games same; large=games differ)")
    print("  most-revised games:")
    for pk, fr, mn, n in fracs[-12:][::-1]:
        d, vid, vn = meta.get(pk, ('', None, ''))
        print(f"    {pk} {d} {vn[:22]:22s} revised={100*fr:4.0f}%  meanZdiff={mn:+.3f}  n={n}")
    print("  least-revised games:")
    for pk, fr, mn, n in fracs[:5]:
        d, vid, vn = meta.get(pk, ('', None, ''))
        print(f"    {pk} {d} {vn[:22]:22s} revised={100*fr:4.0f}%  meanZdiff={mn:+.3f}  n={n}")

    # --- by venue (park) ---
    byven = defaultdict(lambda: {'dz': [], 'n': 0})
    for pk, g in by_game.items():
        _, vid, vn = meta.get(pk, ('', None, ''))
        if vid is None: continue
        byven[(vid, vn)]['dz'].extend(g['dz'])
    print(f"\n=== by park (mean PlateZ diff, revised%) — top offsets ===")
    rows = []
    for (vid, vn), d in byven.items():
        if len(d['dz']) < 200: continue
        rows.append((vn, st.mean(d['dz']), 100 * sum(abs(v) > REV for v in d['dz']) / len(d['dz']), len(d['dz'])))
    for vn, mn, rv, n in sorted(rows, key=lambda t: -abs(t[1]))[:12]:
        print(f"    {vn[:26]:26s} meanZdiff={mn:+.4f}  revised={rv:4.1f}%  n={n}")

    # --- by date ---
    bydate = defaultdict(list)
    for pk, g in by_game.items():
        d, _, _ = meta.get(pk, ('', None, ''))
        if d: bydate[d[:7]].extend(g['dz'])   # by month
    print(f"\n=== by month (mean PlateZ diff, revised%) ===")
    for mo in sorted(bydate):
        a = bydate[mo]
        print(f"    {mo}: meanZdiff={st.mean(a):+.4f}  revised={100*sum(abs(v)>REV for v in a)/len(a):4.1f}%  n={len(a)}")


if __name__ == '__main__':
    main()
