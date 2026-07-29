"""locplus_bandwidth_sweep.py — is one smoothing bandwidth right for every
pitch-type group?

Loc+ smooths every physical surface with the same anisotropic Gaussian
(PHYS_X_IN=4.5in, PHYS_Z_FRAC=0.22). A changeup's chase surface plausibly has
a different gradient scale than a four-seamer's, so a single bandwidth could
be over-smoothing one group and under-smoothing another.

Two passes, cheapest-first:

  PASS 1 (global): sweep the shared bandwidth over the 3-objective harness.
  If the objectives are FLAT in bandwidth globally, per-group tuning is
  noise-chasing by construction and PASS 2 is decoration. This ordering is
  deliberate — the v2 lock's stated lesson was "round where the objective is
  flat", and a per-group sweep has 6x the chances to fit noise.

  PASS 2 (per group): hold every other group at the default and sweep one
  group's bandwidth, scoring reliability on THAT GROUP'S cells only (a
  group-restricted objective — a global objective would drown a single
  group's contribution). Only run when pass 1 shows real curvature.

Objectives (same contract as phase2_locplus_eval / locplus_phase3_eval):
  reliability   — odd/even split-half r of per-pitcher raw_loc, surfaces
                  rebuilt per half, >=125 pitches/half
  stuff-indep   — |r| vs whiff% and FF velo (lower is better)
  predictive    — first-half score vs second-half actual xRV allowed

Usage: python3 scripts/locplus_bandwidth_sweep.py
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
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']

X_GRID = [3.0, 3.75, 4.5, 5.25, 6.0, 7.5]       # inches
Z_GRID = [0.14, 0.18, 0.22, 0.26, 0.32]         # zone fraction


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


def raw_by_pitcher(byp, S, min_n, group=None):
    out = {}
    for k, ps in byp.items():
        vals = []
        for p in ps:
            if group is not None and lp.GROUP.get(p.get('Pitch Type')) != group:
                continue
            v = lp.score_pitch(p, S)
            if v is not None:
                vals.append(v)
        if len(vals) >= min_n:
            out[k] = sum(vals) / len(vals)
    return out


def set_global_bw(x_in, z_frac):
    lp.PHYS_X_IN = x_in
    lp.PHYS_Z_FRAC = z_frac
    lp._KX = lp._k1d(x_in / 2.0)
    lp._KZ = lp._k1d(z_frac / lp.BIN_Z)


def evaluate(ctx, group=None, min_half=MIN_HALF, min_full=MIN_FULL):
    """Returns (rel, |r|whiff, |r|velo, pred, n_rel)."""
    halves = []
    for h in (0, 1):
        S_h = lp.build_surfaces(ctx['halves'][h], LG, SCALE)
        halves.append(raw_by_pitcher(ctx['byp_halves'][h], S_h, min_half, group))
    keys = [k for k in halves[0] if k in halves[1]]
    rel = pearson([halves[0][k] for k in keys], [halves[1][k] for k in keys])

    S_full = lp.build_surfaces(ctx['base'], LG, SCALE)
    full = raw_by_pitcher(ctx['by_p'], S_full, min_full, group)
    kw = [k for k in full if k in ctx['whiff']]
    rw = pearson([full[k] for k in kw], [ctx['whiff'][k] for k in kw])
    kv = [k for k in full if k in ctx['ffv']]
    rv_ = pearson([full[k] for k in kv], [ctx['ffv'][k] for k in kv])

    S_e = lp.build_surfaces(ctx['early'], LG, SCALE)
    score_e = raw_by_pitcher(ctx['byp_early'], S_e, min_full, group)
    kp = [k for k in score_e if k in ctx['actual_l']]
    pred = pearson([score_e[k] for k in kp], [ctx['actual_l'][k] for k in kp])
    return (rel, abs(rw) if rw is not None else None,
            abs(rv_) if rv_ is not None else None, pred, len(keys))


def fmt(x):
    return f"{x:.3f}" if x is not None else "  n/a"


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]

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

    ctx = {'base': base, 'by_p': by_p, 'halves': halves, 'byp_halves': byp_halves,
           'early': early, 'byp_early': byp_early, 'actual_l': actual_l,
           'whiff': whiff, 'ffv': ffv}

    x0, z0 = lp.PHYS_X_IN, lp.PHYS_Z_FRAC

    # ── PASS 1a: horizontal bandwidth ───────────────────────────────────
    print()
    print(f"PASS 1a — global HORIZONTAL bandwidth (z held at {z0})")
    print(f"{'x_in':>6s} {'rel_r':>7s} {'|r|whf':>7s} {'|r|velo':>8s} {'pred_r':>7s}")
    print('-' * 40)
    for x in X_GRID:
        set_global_bw(x, z0)
        rel, rw, rv_, pred, _n = evaluate(ctx)
        star = '  <- shipped' if abs(x - x0) < 1e-9 else ''
        print(f"{x:>6.2f} {fmt(rel):>7s} {fmt(rw):>7s} {fmt(rv_):>8s} {fmt(pred):>7s}{star}")

    # ── PASS 1b: vertical bandwidth ─────────────────────────────────────
    print()
    print(f"PASS 1b — global VERTICAL bandwidth (x held at {x0}in)")
    print(f"{'z_frac':>6s} {'rel_r':>7s} {'|r|whf':>7s} {'|r|velo':>8s} {'pred_r':>7s}")
    print('-' * 40)
    for z in Z_GRID:
        set_global_bw(x0, z)
        rel, rw, rv_, pred, _n = evaluate(ctx)
        star = '  <- shipped' if abs(z - z0) < 1e-9 else ''
        print(f"{z:>6.2f} {fmt(rel):>7s} {fmt(rw):>7s} {fmt(rv_):>8s} {fmt(pred):>7s}{star}")

    # ── PASS 2: per-group horizontal sweep, group-restricted reliability ─
    set_global_bw(x0, z0)
    print()
    print("PASS 2 — per-group HORIZONTAL bandwidth (others held at default),")
    print("reliability scored on that group's cells only (min 60 pitches/half)")
    print(f"{'group':>6s} " + ''.join(f"{x:>9.2f}" for x in X_GRID) + f"{'best':>8s}")
    print('-' * (7 + 9 * len(X_GRID) + 8))
    for G in GROUPS:
        row, best, bestx = f"{G:>6s} ", None, None
        for x in X_GRID:
            lp.PHYS_BW_PT = {G: (x, z0)}
            rel, _rw, _rv, _pred, n = evaluate(ctx, group=G, min_half=60, min_full=100)
            row += f"{rel:>9.3f}" if rel is not None else f"{'n/a':>9s}"
            if rel is not None and (best is None or rel > best):
                best, bestx = rel, x
        lp.PHYS_BW_PT = {}
        row += f"{bestx:>8.2f}" if bestx else f"{'-':>8s}"
        print(row)
    print()
    print(f"shipped bandwidth is {x0}in / {z0}z for every group.")


if __name__ == '__main__':
    main()
