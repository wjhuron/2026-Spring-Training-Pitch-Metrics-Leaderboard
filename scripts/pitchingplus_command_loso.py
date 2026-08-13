"""pitchingplus_command_loso.py — should Command+ be in Pitching+, and is
0.80 still the right Stuff+/Loc+ weight under Stuff+ v12?

Extends scripts/pitchingplus_loso_full.py (which derived the shipped 0.80)
with a third component and re-runs the identical protocol. Two reasons to
re-run rather than trust the stored answer:

  1. The 0.80 was derived 2026-07-25 on Stuff+ **v11**. v12 shipped
     2026-08-09 (nVAA + release-axis cross features + FF/SI velo_diff mask).
     This harness fits the model from the CURRENT feature code, so re-running
     it re-derives the weight under v12 automatically.
  2. Command+ did not exist when the weight was set, and changed again on
     2026-08-12 (K=1 cell means + thin-cell cascade).

PROTOCOL (unchanged from the original, deliberately):
  - per season, split at the median game date
  - Stuff+  = first half, model LOSO-fit on the other DERIVATION seasons, put
              on the SHIPPED atom scale (mean of per-pitch-type atoms)
  - Loc+    = first half, surfaces built on the first half, 100 - 10z
  - Command+= first half, shipped scorer (K=1 + cascade), 100 + 10z sign-
              flipped so higher = better, matching the other two
  - target  = SECOND-half xRV/100, min 200 pitches each side
  - objective: Pearson r of the blend vs second-half xRV, MINIMIZED (lower
    xRV allowed = better pitching), averaged across seasons via Fisher z

TWO KEY SETS, which the first version of this script got wrong. The 2-way
sweep must run on loc n splus n actuals — exactly the original's join. Adding
`k in cmdp` (Command+ needs its own 200-pitch minimum) drops ~5% of pitchers
and makes the 2-way answer non-comparable to the stored derivation. So:
  K2 = loc n splus n actuals            -> the 2-way question
  K3 = K2 n cmdp                        -> the 3-way question
and the 2-way is ALSO reported on K3, so the Command+ verdict is read against
a like-for-like baseline rather than across two different populations.

2026 IS CONFIRMATION, NOT DERIVATION. It is ~70% complete, and the standing
lesson is to beware tuning on a partial season. The argmin is derived on
2021-2025 only; 2026 is scored by a model fit on all five and reported
separately as a sixth, never-fitted replicate.

PRIOR EXPECTATION, stated up front so a surprise gets scrutinized rather than
believed: Command+'s validated contract is that it does NOT predict future
xRV beyond Loc+ and velocity. If this sweep puts real weight on it, that
CONTRADICTS the contract and the result should be distrusted until the
partial correlation explains why.

Usage: python3 scripts/pitchingplus_command_loso.py [--seasons 2024,2025]
"""
import gc
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v11'))

import pipeline_locplus as lp
import pipeline_commandplus as cp
from pipeline_sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt
import train_stuff_v11 as T

from pitchingplus_loso_full import (CACHE, GUTS, MIN_ACTUAL, MIN_PITCH,
                                    TRAIN_PKL, build_season_df, pearson, sd,
                                    stuff_plus_from_raw)

LG, SCALE = 0.3169, 1.2393
W_GRID = [round(0.01 * i, 2) for i in range(101)]
TRI_STEP = 0.02
DERIV = [2021, 2022, 2023, 2024, 2025]     # the argmin is derived on these
CONFIRM = [2026]                            # never fitted, reported separately
SERIES_CACHE = os.path.join(ROOT, 'data', '_pplus_cmd_series.pkl')


def fisher(r):
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def unfisher(z):
    e = math.exp(2 * z)
    return (e - 1) / (e + 1)


def load_2026_pitches():
    """His own 2026 cache is already in the pipeline schema — no adapt()."""
    import pandas as pd
    D = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    return [p for p in D if p.get('_source', 'MLB') == 'MLB'
            and (p.get('Pitcher'), p.get('PTeam')) not in ep]


def build_2026_df():
    return T.build_df(load_2026_pitches())


def command_plus(early):
    by_p = defaultdict(list)
    for p in early:
        by_p[p['Pitcher']].append(p)
    res = cp.score_misses(by_p)
    raw = {k: v['raw_miss'] for k, v in res.items() if v['n_pitches'] >= MIN_PITCH}
    if len(raw) < 10:
        return {}
    mu = sum(raw.values()) / len(raw)
    sg = sd(list(raw.values()))
    return {k: 100.0 + 10.0 * (mu - v) / sg for k, v in raw.items()}


def partial(r_xy, r_xz, r_zy):
    d = (1 - r_xz ** 2) * (1 - r_zy ** 2)
    return (r_xy - r_xz * r_zy) / math.sqrt(d) if d > 0 else None


def sweep2(S, L, y):
    return {w: pearson([w * s + (1 - w) * l for s, l in zip(S, L)], y)
            for w in W_GRID}


def tri_grid():
    g, n = [], int(round(1.0 / TRI_STEP))
    for i in range(n + 1):
        for k in range(n + 1 - i):
            a = round(i * TRI_STEP, 2); c = round(k * TRI_STEP, 2)
            g.append((a, round(1.0 - a - c, 2), c))
    return g


def agg_over(curves, keys, grid_keys):
    return {g: sum(fisher(curves[y][g]) for y in keys) / len(keys)
            for g in grid_keys}


def report_flat(agg, best, label):
    tol = abs(agg[best]) * 0.01
    flat = [w for w in agg if agg[w] <= agg[best] + tol]
    print(f'  {label}: argmin {best:.2f}  mean z {agg[best]:.4f} '
          f'(r {unfisher(agg[best]):.4f})   flat [{min(flat):.2f}, {max(flat):.2f}]')
    return flat


def main():
    import pandas as pd
    import xgboost as xgb
    seasons = DERIV + CONFIRM
    if '--seasons' in sys.argv:
        want = {int(x) for x in sys.argv[sys.argv.index('--seasons') + 1].split(',')}
        seasons = [y for y in seasons if y in want]
    deriv = [y for y in seasons if y in DERIV]
    confirm = [y for y in seasons if y in CONFIRM]

    bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11',
                                           'stuff_models_v11.pkl'), 'rb'))
    params = dict(bundle['params'])
    print(f'Stuff+ features: {len(bundle["features"])}, v12 cross='
          f'{"cross" in bundle["features"]}', file=sys.stderr)

    dfs = {}
    for y in seasons:
        dfs[y] = build_2026_df() if y == 2026 else build_season_df(y)
        dfs[y] = dfs[y][dfs[y]['target_xrv'].notna()].reset_index(drop=True) \
            if y == 2026 else dfs[y]
        print(f'  df {y}: {len(dfs[y])} rows', file=sys.stderr)
        gc.collect()

    rv_fn = make_rv_xrv(LG, SCALE)
    locp, cmdp, actuals, mids = {}, {}, {}, {}
    for y in seasons:
        pitches = (load_2026_pitches() if y == 2026
                   else adapt(os.path.join(ROOT, CACHE[y])))
        base = [p for p in pitches if lp.is_eligible_baseline(p)]
        dates = sorted({p['Game Date'] for p in base if p['Game Date']})
        mid = dates[len(dates) // 2]
        mids[y] = mid
        early = [p for p in base if p['Game Date'] < mid]
        late = [p for p in base if p['Game Date'] >= mid]
        S = lp.build_surfaces(early, LG, SCALE)
        acc = defaultdict(lambda: [0.0, 0])
        for p in early:
            v = lp.score_pitch(p, S)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        raw = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PITCH}
        mu = sum(raw.values()) / len(raw); sg = sd(list(raw.values()))
        locp[y] = {k: 100.0 - 10.0 * (v - mu) / sg for k, v in raw.items()}
        cmdp[y] = command_plus([p for p in pitches if p['Game Date'] < mid])
        acc = defaultdict(lambda: [0.0, 0])
        for p in late:
            v = rv_fn(p)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        actuals[y] = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_ACTUAL}
        print(f'  {y}: loc {len(locp[y])} cmd {len(cmdp[y])} '
              f'act {len(actuals[y])} (mid {mid})', file=sys.stderr)
        del pitches, base, early, late, S
        gc.collect()

    series = {}
    for y in seasons:
        train_years = [o for o in deriv if o != y]
        tr = pd.concat([dfs[o] for o in train_years], ignore_index=True)
        Xtr = T.design(tr); ytr = tr['target_xrv'].values
        model = xgb.XGBRegressor(**params); model.fit(Xtr, ytr)
        del tr, Xtr, ytr; gc.collect()
        te = dfs[y]
        te = te[te['date'].astype(str) < mids[y]].reset_index(drop=True)
        Xte = T.design(te).reindex(columns=model.get_booster().feature_names,
                                   fill_value=0)
        te = te.assign(stuff_raw=-model.predict(Xte))
        splus = stuff_plus_from_raw(te)
        del model, Xte, te; gc.collect()
        k2 = [k for k in locp[y] if k in splus and k in actuals[y]]
        k3 = [k for k in k2 if k in cmdp[y]]
        if len(k2) < 40:
            continue
        series[y] = {
            'K2': {'S': [splus[k] for k in k2], 'L': [locp[y][k] for k in k2],
                   'y': [actuals[y][k] for k in k2], 'n': len(k2)},
            'K3': {'S': [splus[k] for k in k3], 'L': [locp[y][k] for k in k3],
                   'C': [cmdp[y][k] for k in k3],
                   'y': [actuals[y][k] for k in k3], 'n': len(k3)},
            'trained_on': train_years}
        print(f'  {y}: K2 {len(k2)}, K3 {len(k3)} pitchers '
              f'(model trained on {train_years})', file=sys.stderr)
    pickle.dump(series, open(SERIES_CACHE, 'wb'))
    print(f'series cached -> {SERIES_CACHE}', file=sys.stderr)

    D = [y for y in deriv if y in series]
    C = [y for y in confirm if y in series]

    print('\n' + '=' * 94)
    print('REALIZED SPREADS (shipped scales — nominal weight != effective weight)')
    print('=' * 94)
    print(f'{"season":>7} {"n(K2)":>6} {"n(K3)":>6} {"SD S+":>7} {"SD L+":>7} '
          f'{"SD C+":>7}  {"r(S,L)":>8}{"r(S,C)":>8}{"r(L,C)":>8}   role')
    for y in sorted(series):
        a, b = series[y]['K2'], series[y]['K3']
        role = 'derive' if y in D else 'CONFIRM'
        print(f'{y:>7} {a["n"]:>6} {b["n"]:>6} {sd(a["S"]):>7.2f} '
              f'{sd(a["L"]):>7.2f} {sd(b["C"]):>7.2f}  '
              f'{pearson(a["S"], a["L"]):>8.3f}{pearson(b["S"], b["C"]):>8.3f}'
              f'{pearson(b["L"], b["C"]):>8.3f}   {role}')

    # ── 1. two-way, on the ORIGINAL key set ──
    print('\n' + '=' * 94)
    print('1. TWO-WAY  w*Stuff+ + (1-w)*Loc+   on K2 (original join, no Command+ filter)')
    print('=' * 94)
    cols = (0.60, 0.66, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90)
    c2 = {y: sweep2(series[y]['K2']['S'], series[y]['K2']['L'],
                    series[y]['K2']['y']) for y in series}
    print(f'{"season":>7} {"n":>5} ' + ''.join(f'{w:>7.2f}' for w in cols)
          + f' {"argmin":>8}  role')
    for y in sorted(series):
        cur = c2[y]
        print(f'{y:>7} {series[y]["K2"]["n"]:>5} '
              + ''.join(f'{cur[w]:>7.3f}' for w in cols)
              + f' {min(cur, key=lambda w: cur[w]):>8.2f}'
              + f'  {"derive" if y in D else "CONFIRM"}')
    agg2 = agg_over(c2, D, W_GRID)
    w2 = min(agg2, key=lambda w: agg2[w])
    print(f'\n  DERIVATION (2021-2025):')
    flat2 = report_flat(agg2, w2, '  cross-season')
    print(f'    shipped 0.80  mean z {agg2[0.80]:.4f} (r {unfisher(agg2[0.80]):.4f})'
          f'   inside flat region: {0.80 in flat2}')
    print(f'    argmin beats 0.80 in '
          f'{sum(1 for y in D if c2[y][w2] < c2[y][0.80])}/{len(D)} seasons')
    for y in C:
        print(f'\n  CONFIRMATION {y} (never fitted):')
        print(f'    r at 0.80 = {c2[y][0.80]:.4f}   at {w2:.2f} = {c2[y][w2]:.4f}'
              f'   -> {"argmin" if c2[y][w2] < c2[y][0.80] else "shipped"} wins')
        print(f'    {y} own argmin = {min(c2[y], key=lambda w: c2[y][w]):.2f}')

    # ── 2. three-way, on K3, with a K3 two-way baseline ──
    print('\n' + '=' * 94)
    print('2. THREE-WAY  a*Stuff+ + b*Loc+ + c*Command+   on K3 (like-for-like)')
    print('=' * 94)
    G = tri_grid()
    c3 = {y: {g: pearson([g[0] * s + g[1] * l + g[2] * m for s, l, m in
                          zip(series[y]['K3']['S'], series[y]['K3']['L'],
                              series[y]['K3']['C'])], series[y]['K3']['y'])
              for g in G} for y in series}
    agg3 = agg_over(c3, D, G)
    g3 = min(agg3, key=lambda g: agg3[g])
    base3 = min(agg3[g] for g in G if g[2] == 0.0)
    print(f'  best 3-way   Stuff+ {g3[0]:.2f}  Loc+ {g3[1]:.2f}  '
          f'Command+ {g3[2]:.2f}    mean z {agg3[g3]:.4f} (r {unfisher(agg3[g3]):.4f})')
    print(f'  best c=0     (same grid, Command+ excluded)      '
          f'mean z {base3:.4f} (r {unfisher(base3):.4f})')
    print(f'  gain from Command+: {base3 - agg3[g3]:+.4f} mean z')
    tol3 = abs(agg3[g3]) * 0.01
    cs = sorted({g[2] for g in G if agg3[g] <= agg3[g3] + tol3})
    print(f'  Command+ weight inside 1% flat region: [{min(cs):.2f}, {max(cs):.2f}]'
          f'   c=0 inside: {0.0 in cs}')
    print(f'\n  per-season best Command+ weight:')
    for y in sorted(series):
        gy = min(G, key=lambda g: c3[y][g])
        print(f'    {y}: S {gy[0]:.2f}  L {gy[1]:.2f}  C {gy[2]:.2f}   '
              f'r {c3[y][gy]:.4f}   {"derive" if y in D else "CONFIRM"}')

    # ── 3. partial ──
    print('\n' + '=' * 94)
    print('3. PARTIAL — Command+ vs 2nd-half xRV, controlling for the blend')
    print('=' * 94)
    print(f'{"season":>7} {"r(P+,xRV)":>11} {"r(C+,xRV)":>11} {"r(P+,C+)":>10}'
          f' {"r(C+,xRV)|P+":>14}   role')
    parts = []
    for y in sorted(series):
        d = series[y]['K3']
        P = [w2 * s + (1 - w2) * l for s, l in zip(d['S'], d['L'])]
        r_py, r_cy = pearson(P, d['y']), pearson(d['C'], d['y'])
        r_pc = pearson(P, d['C'])
        pr = partial(r_cy, r_pc, r_py)
        if y in D:
            parts.append(pr)
        print(f'{y:>7} {r_py:>11.3f} {r_cy:>11.3f} {r_pc:>10.3f} {pr:>14.3f}'
              f'   {"derive" if y in D else "CONFIRM"}')
    print(f'\n  mean partial over derivation seasons = {sum(parts)/len(parts):+.3f}')
    print('  Command+\'s contract says ~0. A large NEGATIVE value would mean it')
    print('  adds run-prevention signal beyond Stuff+/Loc+.')


if __name__ == '__main__':
    main()
