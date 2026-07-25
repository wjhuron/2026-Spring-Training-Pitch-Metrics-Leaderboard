"""locplus_bandwidth_2d.py — joint sweep of the Loc+ smoothing bandwidths.

The 1-D sweeps in locplus_bandwidth_sweep.py showed the shipped 4.5in / 0.22z
pair is not on the ridge, and that the two axes want to move in OPPOSITE
directions: narrower horizontally, wider vertically. x and z interact through
the same surfaces, so the follow-up has to be a joint 2-D sweep, not two more
1-D ones.

READ RELIABILITY AS A DIAGNOSTIC, NOT AN OBJECTIVE. rel_r rises monotonically
with bandwidth in both axes, all the way to the edge of the grid. That is an
over-smoothing artifact and not a result: in the limit, infinite smoothing
gives every pitcher the same flat surface, which is perfectly reliable and
carries zero information. Maximizing split-half reliability here selects the
most degenerate model available. The 1-D per-group pass in the earlier script
DID rank groups by reliability and duly picked the largest bandwidth on the
grid for four of six groups — that pass was measuring the artifact, and its
"best" column should be ignored.

The objectives that actually discriminate:
  predictive validity  — first-half score vs second-half ACTUAL xRV allowed.
                         Cannot be gamed by smoothing: a flat surface predicts
                         nothing. This is the primary objective.
  stuff-independence   — |r| vs whiff% and FF velo. Also cannot be gamed by
                         smoothing toward a constant, since a flat surface
                         correlates with nothing.

Usage: python3 scripts/locplus_bandwidth_2d.py
"""
import os, sys, math, pickle, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_FULL, MIN_HALF = 250, 125

X_GRID = [2.0, 2.5, 3.0, 3.75, 4.5]
Z_GRID = [0.22, 0.28, 0.34, 0.42, 0.52]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def raw_by_pitcher(byp, S, min_n):
    out = {}
    for k, ps in byp.items():
        vals = [v for v in (lp.score_pitch(p, S) for p in ps) if v is not None]
        if len(vals) >= min_n:
            out[k] = sum(vals) / len(vals)
    return out


def set_bw(x_in, z_frac):
    lp.PHYS_X_IN = x_in
    lp.PHYS_Z_FRAC = z_frac
    lp._KX = lp._k1d(x_in / 2.0)
    lp._KZ = lp._k1d(z_frac / lp.BIN_Z)


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    print(f"baseline pitches: {len(base)} (EP excluded)", file=sys.stderr)

    by_p = defaultdict(list)
    for p in base:
        by_p[(p.get('Pitcher'), p.get('Throws'))].append(p)
    whiff, ffv = {}, {}
    for k, ps in by_p.items():
        sw = [p for p in ps if p.get('Description') in lp.SWING_DESC]
        wh = [p for p in sw if p.get('Description') == 'Swinging Strike']
        if len(sw) >= 100:
            whiff[k] = len(wh) / len(sw)
        v = [f for f in (lp.safe_float(p.get('Velocity')) for p in ps
                         if p.get('Pitch Type') == 'FF') if f is not None]
        if len(v) >= 50:
            ffv[k] = sum(v) / len(v)

    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    parity = {d: i % 2 for i, d in enumerate(dates)}
    mid = dates[len(dates) // 2]
    halves = [[p for p in base if parity.get(p.get('Game Date')) == h] for h in (0, 1)]
    byp_halves = []
    for h in (0, 1):
        d = defaultdict(list)
        for p in halves[h]:
            d[(p.get('Pitcher'), p.get('Throws'))].append(p)
        byp_halves.append(d)
    early = [p for p in base if p.get('Game Date') and p.get('Game Date') < mid]
    late = [p for p in base if p.get('Game Date') and p.get('Game Date') >= mid]
    byp_early = defaultdict(list)
    for p in early:
        byp_early[(p.get('Pitcher'), p.get('Throws'))].append(p)
    rv_fn = make_rv_xrv(LG, SCALE)
    byp_l = defaultdict(list)
    for p in late:
        byp_l[(p.get('Pitcher'), p.get('Throws'))].append(p)
    actual_l = {}
    for k, ps in byp_l.items():
        vals = [v for v in (rv_fn(p) for p in ps) if v is not None]
        if len(vals) >= MIN_FULL:
            actual_l[k] = sum(vals) / len(vals)

    results = {}
    print()
    print(f"{'x_in':>5s} {'z':>5s} {'rel_r*':>7s} {'|r|whf':>7s} {'|r|velo':>8s} "
          f"{'pred_r':>7s}   (* diagnostic only)")
    print('-' * 56)
    for x in X_GRID:
        for z in Z_GRID:
            t0 = time.time()
            set_bw(x, z)
            hv = []
            for h in (0, 1):
                S_h = lp.build_surfaces(halves[h], LG, SCALE)
                hv.append(raw_by_pitcher(byp_halves[h], S_h, MIN_HALF))
            keys = [k for k in hv[0] if k in hv[1]]
            rel = pearson([hv[0][k] for k in keys], [hv[1][k] for k in keys])

            S_full = lp.build_surfaces(base, LG, SCALE)
            full = raw_by_pitcher(by_p, S_full, MIN_FULL)
            kw = [k for k in full if k in whiff]
            rw = abs(pearson([full[k] for k in kw], [whiff[k] for k in kw]))
            kv = [k for k in full if k in ffv]
            rv_ = abs(pearson([full[k] for k in kv], [ffv[k] for k in kv]))

            S_e = lp.build_surfaces(early, LG, SCALE)
            sc_e = raw_by_pitcher(byp_early, S_e, MIN_FULL)
            kp = [k for k in sc_e if k in actual_l]
            pred = pearson([sc_e[k] for k in kp], [actual_l[k] for k in kp])

            results[(x, z)] = (rel, rw, rv_, pred)
            mark = '  <- shipped' if (abs(x - 4.5) < 1e-9 and abs(z - 0.22) < 1e-9) else ''
            print(f"{x:>5.2f} {z:>5.2f} {rel:>7.3f} {rw:>7.3f} {rv_:>8.3f} "
                  f"{pred:>7.3f}{mark}", flush=True)
            print(f"   ({time.time()-t0:.0f}s)", file=sys.stderr)

    print()
    ship = results[(4.5, 0.22)]
    best_pred = max(results.items(), key=lambda kv: kv[1][3])
    best_leak = min(results.items(), key=lambda kv: kv[1][1] + kv[1][2])
    print(f"shipped   x=4.50 z=0.22 -> pred {ship[3]:.3f}, whf {ship[1]:.3f}, velo {ship[2]:.3f}")
    print(f"best pred x={best_pred[0][0]:.2f} z={best_pred[0][1]:.2f} -> pred {best_pred[1][3]:.3f}, "
          f"whf {best_pred[1][1]:.3f}, velo {best_pred[1][2]:.3f}")
    print(f"min leak  x={best_leak[0][0]:.2f} z={best_leak[0][1]:.2f} -> pred {best_leak[1][3]:.3f}, "
          f"whf {best_leak[1][1]:.3f}, velo {best_leak[1][2]:.3f}")
    print()
    print("Prefer a FLAT region over a point argmax — the v2 lock's lesson was that")
    print("chasing a per-sample argmax on one half-season does not generalize.")


if __name__ == '__main__':
    main()
