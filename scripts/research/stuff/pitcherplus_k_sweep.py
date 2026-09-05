"""pitcherplus_k_sweep.py — re-measure the season Pitcher+ stabilization
constants (2026-09-05).

The shipped k (42/215/398/421/1046/333, pipeline/pitcherplus.py) were
screen-measured 2026-07-24 on v11 Stuff+ and the pre-count-surface Loc+.
Two questions, on the dual panel of pitcherplus_v14_audit (S: odd/even
halves within season both directions; Y: year pairs 2021->22..2024->25;
target = future xRV/100; frozen shipped weights):
  1. What are the r = 0.5 reliability crossings now, with the v14/v15
     LOSO stuff series and a Loc+ series rebuilt on the CURRENT surfaces
     (data/_pplus_locplus_hist.csv, regenerated 2026-09-05)?
  2. Does the COMPOSITE objective care? Sweep each component's k with the
     other five at shipped; report r_S and r_Y per k. The 2026-09-02 outing
     refit showed the composite is flat in stuff k over 20-150; the
     reliability crossing is the wrong objective for a composite's
     shrinkage, so the curve decides and the crossing is a diagnostic.
Usage: PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_k_sweep.py
Output: console + data/_pplus_k_sweep.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
import pitcherplus_search as ps
import pitcherplus_v14_audit as V

GRID = {'stuffRaw': [5, 10, 14, 20, 30, 42, 60, 100, 150, 300],
        'locRaw': [30, 60, 100, 150, 215, 300, 450, 700, 1000],
        'kPct': [100, 200, 300, 398, 500, 700, 1000],
        'izWhiffPct': [100, 200, 300, 421, 600, 800, 1200],
        'xrv100': [300, 500, 700, 1046, 1500, 2000, 3000],
        'gbPct': [100, 200, 333, 500, 700, 1000]}

def crossing(t, feat):
    """r = 0.5 crossing from the odd/even halves: n_half * (1 - r) / r."""
    h = t[t['half'].isin(['A', 'B']) & (t['n'] >= V.MIN_HALF)]
    A = h[h['half'] == 'A']; B = h[h['half'] == 'B']
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b')).dropna(subset=[feat + '_a', feat + '_b'])
    r = float(np.corrcoef(ab[feat + '_a'], ab[feat + '_b'])[0, 1])
    n_half = float(np.mean(np.minimum(ab['n_a'], ab['n_b'])))
    return r, n_half, n_half * (1 - r) / r, len(ab)

def eval_k(t, kmap):
    feats = V.SHIPPED_SUBSET
    halves = V.shrunk_z(t[t['half'].isin(['A', 'B'])], feats, kmap, V.Q_HALF)
    A = halves[(halves['half'] == 'A') & (halves['n'] >= V.MIN_HALF)]
    B = halves[(halves['half'] == 'B') & (halves['n'] >= V.MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    comp = lambda df, suf: sum(w * df[f + '_sz' + suf] for f, w, _ in V.SHIPPED)
    ps_ = np.concatenate([comp(ab, '_a'), comp(ab, '_b')])
    ys_ = np.concatenate([ab['xrv100_b'].to_numpy(float), ab['xrv100_a'].to_numpy(float)])
    ok = np.isfinite(ys_); r_s = float(np.corrcoef(ps_[ok], ys_[ok])[0, 1])
    full = V.shrunk_z(t[(t['half'] == 'full') & (t['n'] >= V.MIN_FULL)], feats, kmap, V.Q_FULL)
    pairs = full.merge(full.assign(season=full['season'] - 1), on=['pid', 'season'], suffixes=('', '_n1'))
    py = comp(pairs, ''); yy = pairs['xrv100_n1'].to_numpy(float); ok = np.isfinite(yy)
    r_y = float(np.corrcoef(py[ok], yy[ok])[0, 1])
    return r_s, r_y

def main():
    t = V.load_tables(ps.STUFF_CSV)
    out = {'crossings': {}, 'sweep': {}}
    print("r = 0.5 crossings on the current series (odd/even halves, n >= %d):" % V.MIN_HALF)
    for f, _w, k in V.SHIPPED:
        r, nh, cross, n = crossing(t, f)
        out['crossings'][f] = dict(r=r, n_half=nh, crossing=cross, n=n, shipped=k)
        print(f"  {f:11} half r {r:.3f}  mean half n {nh:6.0f}  crossing {cross:7.0f}  shipped k {k:6.0f}  (n {n})")
    ship = {f: k for f, _w, k in V.SHIPPED}
    r_s0, r_y0 = eval_k(t, ship)
    print(f"\nshipped k set: r_S {r_s0:.4f}  r_Y {r_y0:.4f}")
    for f in V.SHIPPED_SUBSET:
        print(f"\n-- {f} (others at shipped)")
        rows = []
        for k in GRID[f]:
            km = dict(ship); km[f] = float(k)
            r_s, r_y = eval_k(t, km)
            rows.append(dict(k=k, r_s=r_s, r_y=r_y))
            print(f"   k {k:5d}  r_S {r_s:.4f} ({r_s - r_s0:+.4f})  r_Y {r_y:.4f} ({r_y - r_y0:+.4f})" + ("  <- shipped" if k == ship[f] else ""))
        out['sweep'][f] = rows
    json.dump(out, open(os.path.join(ROOT, 'data', '_pplus_k_sweep.json'), 'w'), indent=1)
    print('\nwrote data/_pplus_k_sweep.json')

if __name__ == '__main__':
    main()
