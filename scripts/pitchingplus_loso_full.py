"""pitchingplus_loso_full.py — the Pitching+ blend weight on the ACTUAL shipped
scales, leave-one-season-out.

scripts/stuffplus_loso_pitchingplus.py got the leakage right but the SCALES
wrong: it z-scored raw regressor output. That is not what Pitching+ blends.

WHY THE SCALE MATTERS, and it is not cosmetic. Shipped Stuff+ is the
pitch-weighted mean of per-pitch-type ATOMS, each atom being
100 + 10*(stuff_raw - mu_type)/sd_type. Atoms have SD 10 at the
pitcher-team-pitch-type level, but averaging across a pitcher's arsenal
shrinks the spread, so OVERALL Stuff+ has a cross-pitcher SD well below 10.
Loc+ meanwhile is z-scored at the pitcher level to exactly SD 10 by
construction (100 - 10*z). Blending two series with different spreads at
0.7/0.3 does NOT give Stuff+ 70% of the influence — the nominal weight and
the effective weight come apart. Any blend-weight estimate made on z-scores
is therefore answering a different question than the shipped constant.

This reproduces the shipped path per held-out season:
  stuff_raw   = -prediction from a model fit on the OTHER four seasons
                (negated exactly as _oof_predict / _cached_oof do)
  mu/sd       per pitch type from the QUALIFIED pool of (pitcher, team,
                pitch_type) units with n >= QUAL_N, with ANCHOR_BORROW for
                KN/SV, matching _standardize
  atom        100 + K_SCALE*(stuff_raw - mu_type)/sd_type per pitch
  Stuff+      mean atom per pitcher (the coherent-canon definition)
  Loc+        100 - 10*(raw - mu)/sigma over the season's pitcher pool,
                matching pipeline_locplus._normalize with n_prior=0
  Pitching+   w*Stuff+ + (1-w)*Loc+, exactly the shipped form

Reported alongside is each series' realized SD, so the gap between nominal
and effective weight is visible rather than assumed.

Usage: python3 scripts/pitchingplus_loso_full.py
"""
import os, sys, math, pickle, gc, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v11'))

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt
import train_stuff_v11 as T

LG, SCALE = 0.3169, 1.2393
MIN_PITCH, MIN_ACTUAL = 200, 200
W_GRID = [round(0.01 * i, 2) for i in range(101)]
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = dict(T.HIST_GUTS); GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)
TRAIN_PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL) for y in SEASONS}
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl',
         2023: 'data/_statcast2023_cache.pkl', 2024: 'data/_statcast2024_cache.pkl',
         2025: 'data/_statcast2025_full_cache.pkl'}


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


def sd(vs):
    m = sum(vs) / len(vs)
    return math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))


def build_season_df(year):
    lg, sc = GUTS[year]
    old = (T.LG_WOBA, T.WOBA_SCALE)
    T.LG_WOBA, T.WOBA_SCALE = lg, sc
    try:
        d = T.build_df(pickle.load(open(TRAIN_PKL[year], 'rb')))
    finally:
        T.LG_WOBA, T.WOBA_SCALE = old
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    return d


def stuff_plus_from_raw(df):
    """Replicates _standardize + _atom_grade + the coherent-canon overall."""
    import numpy as np, pandas as pd
    # 2021-2024 training pickles carry no PTeam, so build_df leaves team NaN and
    # pandas groupby would silently DROP every row (yielding no atoms at all).
    # Fill it: team only sub-divides the qualified-pool units, and historical
    # seasons have no mid-season team splits worth preserving here.
    df = df.assign(team=df['team'].fillna('') if 'team' in df.columns else '')
    a = df.groupby(['pitcher', 'team', 'pitch_type'])['stuff_raw'].agg(
        rawmean='mean', n='size').reset_index()
    scale = {}
    for key, sub in a.groupby('pitch_type'):
        q = sub[sub['n'] >= T.QUAL_N]
        base = q if len(q) >= 5 else sub
        if key in T.ANCHOR_BORROW:
            donor = a[a['pitch_type'].isin(T.ANCHOR_BORROW[key])]
            dq = donor[donor['n'] >= T.QUAL_N]
            base = dq if len(dq) >= 5 else donor
        mu, s = float(base['rawmean'].mean()), float(base['rawmean'].std())
        scale[key] = (mu, s)
    mus = df['pitch_type'].map(lambda k: scale.get(k, (np.nan, np.nan))[0])
    sds = df['pitch_type'].map(lambda k: scale.get(k, (np.nan, np.nan))[1])
    atom = 100 + T.K_SCALE * (df['stuff_raw'] - mus) / sds
    out = pd.DataFrame({'pitcher': df['pitcher'].values, 'atom': atom.values})
    out = out.dropna(subset=['atom'])
    g = out.groupby('pitcher')['atom'].agg(['mean', 'size'])
    return {p: float(r['mean']) for p, r in g.iterrows() if r['size'] >= MIN_PITCH}


def main():
    import pandas as pd, numpy as np, xgboost as xgb
    bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11',
                                           'stuff_models_v11.pkl'), 'rb'))
    params = dict(bundle['params'])

    print("building per-season feature frames...", file=sys.stderr)
    dfs = {}
    for y in SEASONS:
        dfs[y] = build_season_df(y)
        print(f"  {y}: {len(dfs[y])} rows", file=sys.stderr)
        gc.collect()

    rv_fn = make_rv_xrv(LG, SCALE)
    locp, actuals, mids = {}, {}, {}
    for y in SEASONS:
        pitches = adapt(os.path.join(ROOT, CACHE[y]))
        base = [p for p in pitches if lp.is_eligible_baseline(p)]
        dates = sorted({p['Game Date'] for p in base if p['Game Date']})
        mid = dates[len(dates) // 2]; mids[y] = mid
        early = [p for p in base if p['Game Date'] < mid]
        late = [p for p in base if p['Game Date'] >= mid]
        S = lp.build_surfaces(early, LG, SCALE)
        acc = defaultdict(lambda: [0.0, 0])
        for p in early:
            v = lp.score_pitch(p, S)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        raw = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PITCH}
        # pipeline_locplus._normalize with n_prior=0: locPlus = 100 - 10*z
        mu = sum(raw.values()) / len(raw); sg = sd(list(raw.values()))
        locp[y] = {k: 100.0 - 10.0 * (v - mu) / sg for k, v in raw.items()}
        acc = defaultdict(lambda: [0.0, 0])
        for p in late:
            v = rv_fn(p)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        actuals[y] = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_ACTUAL}
        del pitches, base, early, late, S
        gc.collect()

    print()
    print(f"{'season':>7s} {'n':>5s} {'SD(S+)':>7s} {'SD(L+)':>7s} | "
          + "".join(f"{w:>7.2f}" for w in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0))
          + f" | {'best':>5s}")
    print('-' * 84)
    bests, curves = {}, {}
    for y in SEASONS:
        tr = pd.concat([dfs[o] for o in SEASONS if o != y], ignore_index=True)
        Xtr = T.design(tr); ytr = tr['target_xrv'].values
        model = xgb.XGBRegressor(**params); model.fit(Xtr, ytr)
        del tr, Xtr, ytr; gc.collect()

        te = dfs[y]
        te = te[te['date'].astype(str) < mids[y]].reset_index(drop=True)
        Xte = T.design(te).reindex(columns=model.get_booster().feature_names,
                                   fill_value=0)
        # NEGATED, exactly as _oof_predict/_cached_oof do: higher = better
        te = te.assign(stuff_raw=-model.predict(Xte))
        splus = stuff_plus_from_raw(te)
        del model, Xte, te; gc.collect()

        keys = [k for k in locp[y] if k in splus and k in actuals[y]]
        if len(keys) < 40:
            print(f"{y:>7d}  only {len(keys)} joined, skipped", flush=True)
            continue
        S_ = [splus[k] for k in keys]; L_ = [locp[y][k] for k in keys]
        ys = [actuals[y][k] for k in keys]
        curve = {w: pearson([w * s + (1 - w) * l for s, l in zip(S_, L_)], ys)
                 for w in W_GRID}
        best = min(curve, key=lambda w: curve[w])
        bests[y] = best; curves[y] = curve
        cells = "".join(f"{curve[w]:>7.3f}" for w in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0))
        print(f"{y:>7d} {len(keys):>5d} {sd(S_):>7.2f} {sd(L_):>7.2f} | {cells} "
              f"| {best:>5.2f}", flush=True)

    if not bests:
        print("\nno seasons evaluated")
        return
    # ── CROSS-SEASON OPTIMUM ──────────────────────────────────────────────
    # Picking the median of five per-season argmaxes is eyeballing. The actual
    # objective is "one w for all seasons", so optimize THAT: average each
    # season's correlation via Fisher z (the correct way to average r), then
    # find the argmin and, just as important, how flat it is around there.
    import pickle as _pk
    _pk.dump({'curves': curves, 'bests': bests},
             open(os.path.join(ROOT, 'data', '_pplus_loso_curves.pkl'), 'wb'))

    def fisher(r):
        r = max(-0.999999, min(0.999999, r))
        return 0.5 * math.log((1 + r) / (1 - r))

    agg = {}
    for w in W_GRID:
        zs = [fisher(curves[y][w]) for y in curves]
        agg[w] = sum(zs) / len(zs)
    wbest = min(agg, key=lambda w: agg[w])          # most negative = best
    zbest = agg[wbest]
    # Flat region: every w whose mean-z is within 1% of the best in magnitude.
    tol = abs(zbest) * 0.01
    flat = [w for w in W_GRID if agg[w] <= zbest + tol]
    print()
    print(f"CROSS-SEASON OPTIMUM (mean Fisher-z over {len(curves)} seasons)")
    print(f"  argmin w = {wbest:.2f}   (mean z {zbest:.4f})")
    print(f"  flat region within 1% of best: w in [{min(flat):.2f}, {max(flat):.2f}]")
    print(f"  interior optimum: {'YES' if 0 < wbest < 1 and min(flat) > 0 and max(flat) < 1 else 'NO — at a grid edge'}")
    print()
    print(f"{'w':>6s} {'mean z':>8s} " + "".join(f"{y:>8d}" for y in sorted(curves)))
    for w in [round(x, 2) for x in (0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82,
                                    0.85, 0.90, 0.95, 1.00)]:
        cells = "".join(f"{curves[y][w]:>8.3f}" for y in sorted(curves))
        mark = '  <- argmin' if abs(w - wbest) < 1e-9 else (
               '  <- shipped' if abs(w - 0.70) < 1e-9 else '')
        print(f"{w:>6.2f} {agg[w]:>8.4f} {cells}{mark}")
    print()
    print("(cells are corr(Pitching+, next-half xRV allowed); MORE NEGATIVE = better)")
    print("SD(S+) vs SD(L+) is the scale gap — where nominal and effective weight diverge.")
    ws = sorted(bests.values())
    print()
    print("best w: " + ", ".join(f"{y}={bests[y]:.2f}" for y in sorted(bests))
          + f"   median {ws[len(ws)//2]:.2f}   (shipped 0.70)")
    print()
    print("Loss at the shipped 0.70 vs each season's own optimum:")
    for y in sorted(curves):
        c = curves[y]
        print(f"  {y}: {c[0.7]:.3f} vs {c[bests[y]]:.3f}  (gives up {abs(c[0.7]-c[bests[y]]):.3f})")


if __name__ == '__main__':
    main()
