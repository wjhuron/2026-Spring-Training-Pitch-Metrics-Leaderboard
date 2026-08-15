"""stuffplus_loso_pitchingplus.py — leave-one-season-out Stuff+, then the
Pitching+ blend weight measured without leakage.

WHY THIS EXISTS. scripts/pitchingplus_weight_multiseason.py could not answer
the blend-weight question. 2021-2025 sit inside the v11 training set, and the
bundle's OOF fold models do NOT help: in train_stuff._oof_predict the
prior-season frame Xp is concatenated into EVERY fold's training set, so
GroupKFold only holds out 2026 pitchers' 2026 pitches. Every fold model has
seen all of 2021-2025, which is why OOF and full-model scoring agreed to three
decimals. The measured optimum there (median w 0.85 vs the shipped 0.70) is
biased toward Stuff+ by an unknown amount and must not be used.

WHAT THIS DOES. For each season Y in 2021-2025, fit a fresh XGBRegressor on
every OTHER season with the shipped hyperparameters, then score Y with it.
That is also the honest production analog: a new season is always scored by a
model trained on prior data. Loc+ and the actual-xRV target come from the same
per-season harness as before (surfaces built in-season, first half predicts
second half). Seasons are never pooled for Loc+.

FIDELITY NOTES — this reuses the trainer's own pieces rather than reimplementing:
  build_df / design / BASE_FEATS   feature engineering, imported directly
  HIST_GUTS + PRIOR_LG_WOBA        per-season wOBA guts, so target_xrv is built
                                   with each season's own constants (using 2026
                                   guts for a 2021 target would be wrong)
  bundle['params']                 the shipped XGBoost hyperparameters
Tag harmonization is deliberately NOT applied: it maps prior tags onto a
CURRENT season's pitcher tags, which has no meaning when the current season is
not in the training set. The agnostic model ignores pitch-type labels anyway.

COST: five fits on ~2.8M rows each. Expect this to run long.

Usage: python3 scripts/stuffplus_loso_pitchingplus.py
"""
import os, sys, math, pickle, gc, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt

import stuff_plus.train_stuff as T

LG, SCALE = 0.3169, 1.2393
MIN_PITCH, MIN_ACTUAL = 200, 200
W_GRID = [round(0.05 * i, 2) for i in range(21)]
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = dict(T.HIST_GUTS)
GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)
TRAIN_PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL)
             for y in SEASONS}
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


def zscore(d):
    vs = list(d.values())
    mu = sum(vs) / len(vs)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / len(vs))
    return {k: (v - mu) / sd for k, v in d.items()} if sd > 1e-12 else {k: 0.0 for k in d}


def build_season_df(year):
    """build_df with THAT season's wOBA guts, as the trainer does."""
    lg, sc = GUTS[year]
    old = (T.LG_WOBA, T.WOBA_SCALE)
    T.LG_WOBA, T.WOBA_SCALE = lg, sc
    try:
        d = T.build_df(pickle.load(open(TRAIN_PKL[year], 'rb')))
    finally:
        T.LG_WOBA, T.WOBA_SCALE = old
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    d['_season'] = year
    return d


def main():
    import pandas as pd, numpy as np, xgboost as xgb
    bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus',
                                           'stuff_models.pkl'), 'rb'))
    params = dict(bundle['params'])
    print(f"params: {params}", file=sys.stderr)

    print("building per-season feature frames...", file=sys.stderr)
    dfs = {}
    for y in SEASONS:
        t0 = time.time()
        dfs[y] = build_season_df(y)
        print(f"  {y}: {len(dfs[y])} rows ({time.time()-t0:.0f}s)", file=sys.stderr)
        gc.collect()

    # Loc+ and the target, per season, from the raw caches
    rv_fn = make_rv_xrv(LG, SCALE)
    locs, actuals, mids = {}, {}, {}
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
        locs[y] = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PITCH}
        acc = defaultdict(lambda: [0.0, 0])
        for p in late:
            v = rv_fn(p)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        actuals[y] = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_ACTUAL}
        print(f"  {y}: Loc+ for {len(locs[y])} pitchers, target for "
              f"{len(actuals[y])}", file=sys.stderr)
        del pitches, base, early, late, S
        gc.collect()

    print()
    print(f"{'season':>7s} {'n':>5s} | " + "".join(f"{w:>7.2f}" for w in
          (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)) + f" | {'best w':>7s}")
    print('-' * 70)
    bests, curves = {}, {}
    for y in SEASONS:
        t0 = time.time()
        tr = pd.concat([dfs[o] for o in SEASONS if o != y], ignore_index=True)
        Xtr = T.design(tr)
        ytr = tr['target_xrv'].values
        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, ytr)
        print(f"  {y}: fit on {len(tr)} rows from {len(SEASONS)-1} other seasons "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
        del tr, Xtr, ytr
        gc.collect()

        te = dfs[y]
        # BUGFIX: build_df emits 'date', not 'game_date'. The old guarded form
        # silently skipped filtering entirely, so Stuff+ was computed from the
        # WHOLE season including the second half that is the prediction target
        # — target leakage into the predictor, biasing w upward. Superseded by
        # scripts/research/stuff/pitchingplus_loso_full.py, which also fixes the scales.
        te = te[te['date'].astype(str) < mids[y]].reset_index(drop=True)
        Xte = T.design(te).reindex(columns=model.get_booster().feature_names,
                                   fill_value=0)
        pred = model.predict(Xte)
        agg = defaultdict(lambda: [0.0, 0])
        for pit, pr in zip(te['pitcher'].values, pred):
            a = agg[pit]; a[0] += float(pr); a[1] += 1
        stuff = {k: v[0] / v[1] for k, v in agg.items() if v[1] >= MIN_PITCH}
        del model, Xte, te
        gc.collect()

        keys = [k for k in locs[y] if k in stuff and k in actuals[y]]
        if len(keys) < 40:
            print(f"{y:>7d}  only {len(keys)} joined pitchers, skipped", flush=True)
            continue
        lz = zscore({k: -locs[y][k] for k in keys})
        sz = zscore({k: -stuff[k] for k in keys})
        ys = [actuals[y][k] for k in keys]
        curve = {w: pearson([w * sz[k] + (1 - w) * lz[k] for k in keys], ys)
                 for w in W_GRID}
        best = min(curve, key=lambda w: curve[w])
        bests[y] = best; curves[y] = curve
        cells = "".join(f"{curve[w]:>7.3f}" for w in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0))
        print(f"{y:>7d} {len(keys):>5d} | {cells} | {best:>7.2f}", flush=True)

    if not bests:
        print("\nno seasons evaluated")
        return
    print()
    print("(cells are corr(blend, next-half xRV allowed); MORE NEGATIVE = better)")
    ws = sorted(bests.values())
    print(f"best w by season: " + ", ".join(f"{y}={bests[y]:.2f}" for y in sorted(bests)))
    print(f"median best w: {ws[len(ws) // 2]:.2f}   (shipped: 0.70)")
    print()
    print("How much does the choice matter? Loss vs each season's own optimum:")
    for y in sorted(curves):
        c = curves[y]
        b = c[bests[y]]
        print(f"  {y}: at w=0.70 {c[0.7]:.3f} vs best {b:.3f} "
              f"(gives up {abs(c[0.7]-b):.3f})")
    print()
    print("This run is leakage-free: each season is scored by a model that never")
    print("saw it. A flat loss column means 0.70 is fine even if the argmax moved.")


if __name__ == '__main__':
    main()
