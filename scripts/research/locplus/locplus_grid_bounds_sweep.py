"""locplus_grid_bounds_sweep.py — sweep the Loc+ grid BOUNDS, 2021-2025.

Produces the measurement recorded in the provenance comment above
X_MIN/X_MAX/Z_MIN/Z_MAX in pipeline/locplus.py (2026-08-30).

Holds the bin widths fixed (BIN_X 2 inches, BIN_Z 0.10) so that only COVERAGE
changes and the smoothing kernels — which are defined in physical units —
stay identical across every configuration. The bin WIDTHS are therefore not
measured by this script.

Objectives are the shipped gate's (see locplus_final_gate.py):
  raw     first-half Loc+ vs second-half actual RV        <- THE DECIDER
  partial the same, controlling FF velocity
  rel     odd/even-date split-half reliability            <- diagnostic only

Read `rel` as a diagnostic, never an objective: it climbs monotonically as the
grid shrinks, peaking where over half the pitches are clamped into a handful
of bins and `raw` is worse than shipped. `partial` is likewise unusable in the
X direction, because r(Loc+, velo) rises as the grid narrows and mechanically
lifts it while `raw` stays flat.

  python3 scripts/research/locplus/locplus_grid_bounds_sweep.py x
  python3 scripts/research/locplus/locplus_grid_bounds_sweep.py z
"""
import gc
import json
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.sdplus import make_rv_xrv

# X: symmetric half-width in feet. Shipped 1.5. Runs 0.5-2.5 so the plateau is
# bracketed on both sides rather than sitting at a grid edge.
X_GRID = [6/12, 8/12, 10/12, 1.0, 14/12, 16/12, 1.5, 20/12, 1.75, 2.0, 2.25, 2.5]
# Z: symmetric margin in zone-heights beyond each edge. Shipped 0.6.
Z_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]

SHIPPED = {'x': 1.5, 'z': 0.6}
ORIG = (lp.X_MIN, lp.X_MAX, lp.NX, lp.Z_MIN, lp.Z_MAX, lp.NZ)

SEASONS = [(2021, 'data/_statcast2021_cache.pkl'),
           (2022, 'data/_statcast2022_cache.pkl'),
           (2023, 'data/_statcast2023_cache.pkl'),
           (2024, 'data/_statcast2024_cache.pkl'),
           (2025, 'data/_statcast2025_full_cache.pkl')]


def apply_setting(axis, v):
    if axis == 'x':
        lp.X_MIN, lp.X_MAX = -v, v
        lp.NX = int(round((lp.X_MAX - lp.X_MIN) / lp.BIN_X))
        return lp.NX
    lp.Z_MIN, lp.Z_MAX = -v, 1.0 + v
    lp.NZ = int(round((lp.Z_MAX - lp.Z_MIN) / lp.BIN_Z))
    return lp.NZ


def clamped_fraction(axis, pitches):
    """Share of baseline pitches that fall outside the current bounds."""
    if axis == 'x':
        vals = [abs(v) for v in (lp.safe_float(p.get('PlateX')) for p in pitches)
                if v is not None]
        return sum(1 for v in vals if v >= lp.X_MAX) / len(vals) if vals else 0.0
    vals = [z for z in (lp._znorm(p) for p in pitches) if z is not None]
    return (sum(1 for z in vals if z <= lp.Z_MIN or z >= lp.Z_MAX) / len(vals)
            if vals else 0.0)


def eval_season(axis, pitches, rv_fn):
    """One season, every grid value. Segmentation is grid-independent, so it
    is built once and reused — is_eligible_baseline does not read the bounds."""
    b = [p for p in pitches if lp.is_eligible_baseline(p)]
    dates = sorted({p['Game Date'] for p in b if p['Game Date']})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in b:
        d = p['Game Date']
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)
    par = {d: i % 2 for i, d in enumerate(dates)}
    g0 = [p for p in b if par.get(p['Game Date']) == 0]
    g1 = [p for p in b if par.get(p['Game Date']) == 1]

    prep = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        actual = {}
        for k, ps in base.by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= base.MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        velo = {}
        for k, ps in base.by_pitcher(seg[f]).items():
            v = [x for x in (lp.safe_float(p['Velocity']) for p in ps
                             if p['Pitch Type'] == 'FF') if x is not None]
            if len(v) >= base.MIN_VELO:
                velo[k] = sum(v) / len(v)
        prep[half] = {'first': seg[f], 'byp_f': base.by_pitcher(seg[f]),
                      'actual': actual, 'velo': velo}

    res = {}
    for v in (X_GRID if axis == 'x' else Z_GRID):
        n_bins = apply_setting(axis, v)
        raws, parts, rlvs = [], [], []
        for half in ('A', 'B'):
            c = prep[half]
            S = lp.build_surfaces(c['first'], base.LG, base.SCALE)
            loc = base.score_map(c['byp_f'], S, base.MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
            rlvs.append(r_ls)
        a0 = base.score_map(base.by_pitcher(g0),
                            lp.build_surfaces(g0, base.LG, base.SCALE), base.MIN_REL)
        a1 = base.score_map(base.by_pitcher(g1),
                            lp.build_surfaces(g1, base.LG, base.SCALE), base.MIN_REL)
        ks = [k for k in a0 if k in a1]
        res[v] = {'raw': sum(raws) / len(raws), 'partial': sum(parts) / len(parts),
                  'rlv': sum(rlvs) / len(rlvs),
                  'rel': base.pearson([a0[k] for k in ks], [a1[k] for k in ks]),
                  'bins': n_bins, 'clamp': clamped_fraction(axis, b)}
        r = res[v]
        print(f"    {axis}={v:<6.3f} bins={n_bins:<3} raw={r['raw']:+.4f} "
              f"partial={r['partial']:+.4f} rel={r['rel']:.4f} "
              f"clamp={r['clamp'] * 100:.2f}%", flush=True)
    return res


def main():
    axis = (sys.argv[1] if len(sys.argv) > 1 else 'x').lower()
    if axis not in ('x', 'z'):
        raise SystemExit('usage: locplus_grid_bounds_sweep.py [x|z]')
    grid = X_GRID if axis == 'x' else Z_GRID
    rv_fn = make_rv_xrv(base.LG, base.SCALE)

    table = {}
    for yr, path in SEASONS:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr}: cache missing, skipped", flush=True)
            continue
        print(f"=== {yr} ===", flush=True)
        pitches = base.adapt(p)
        print(f"  {len(pitches)} usable pitches", flush=True)
        table[yr] = eval_season(axis, pitches, rv_fn)
        del pitches
        gc.collect()
    lp.X_MIN, lp.X_MAX, lp.NX, lp.Z_MIN, lp.Z_MAX, lp.NZ = ORIG

    if not table:
        raise SystemExit('no season caches found')
    print(f"\n===== mean across {len(table)} seasons =====")
    print(f"{axis:>7} {'bins':>5} {'clamp':>8} {'RAW':>9} {'partial':>9} "
          f"{'r(Loc,velo)':>12} {'rel':>8}")
    for v in grid:
        vals = [table[y][v] for y in table if v in table[y]]
        if not vals:
            continue
        m = lambda k: st.mean(x[k] for x in vals)
        star = '  <-- shipped' if abs(v - SHIPPED[axis]) < 1e-9 else ''
        print(f"{v:>7.3f} {vals[0]['bins']:>5} {m('clamp') * 100:>7.2f}% "
              f"{m('raw'):>+9.4f} {m('partial'):>+9.4f} {m('rlv'):>+12.4f} "
              f"{m('rel'):>8.4f}{star}")

    ship = SHIPPED[axis]
    print(f"\n===== RAW vs shipped {ship} (paired; RAW is the decider) =====")
    for v in grid:
        if abs(v - ship) < 1e-9:
            continue
        ys = [y for y in table if v in table[y] and ship in table[y]]
        if len(ys) < 3:
            continue
        df = [table[y][v]['raw'] - table[y][ship]['raw'] for y in ys]
        mu = st.mean(df)
        se = st.stdev(df) / len(df) ** 0.5
        t = mu / se if se else float('nan')
        verdict = 'WORSE' if t < -2 else ('better' if t > 2 else 'flat')
        print(f"  {v:>6.3f}: delta {mu:+.5f} SE {se:.5f} t {t:>6.2f} "
              f"wins {sum(1 for x in df if x > 0)}/{len(df)}  {verdict}")

    out = os.path.join(ROOT, 'data', f'_locplus_grid_{axis}_sweep.json')
    json.dump({str(y): {str(k): v for k, v in t.items()} for y, t in table.items()},
              open(out, 'w'), indent=1)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
