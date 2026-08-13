"""pitchingplus_weight_multiseason.py — is Pitching+ 0.70 Stuff+ / 0.30 Loc+ right
across seasons?

Pitching+ ships as exactly 0.7*Stuff+ + 0.3*Loc+. That weight is a tuned
constant and, unlike the seven constants in the 2026-07-13 battery, I find no
multi-season record for it. Blend weights are exactly the kind of constant that
moves with sample size, and the 2026-07-25 Loc+ work showed what happens when a
constant is validated only inside the season it was fitted to (a config that won
every within-2026 test lost 5 of 5 across 2021-2025).

DESIGN — per season, never pooled:
  Stuff+  from data/_pitches{yr}_training.pkl, which is in the pipeline's own
          format with Wally's retagged pitch types, scored through the shipped
          v11 bundle (stuff_plus/stuff_models.pkl).
  Loc+    from data/_statcast{yr}_cache.pkl via the adapter in
          locplus_constants_multiseason.py — the training pickles carry no
          SzTop/SzBot, so zone-normalized location has to come from the raw
          cache. The two are joined at the PITCHER-SEASON level (both sources
          use "Last, First"), not per pitch.
  target  actual xRV allowed in the SECOND half, predictors from the FIRST.

Both components are z-scored within season and oriented so higher = better
pitcher, then blended. The reported optimum is the w minimizing correlation
with runs allowed.

LEAKAGE. 2021-2025 are inside the v11 training set, so scoring them with the
full model would flatter Stuff+ and bias the optimal w upward. This uses the
bundle's OOF fold models instead — each pitcher is scored by the fold that
held him out — reproducing the shipped leakage-free path. A first pass WITH
the full model returned a median best w of 0.85, exactly the direction
leakage predicts, and 2023 returned w=1.00 (Loc+ contributing nothing),
which is the tell. Those numbers should not be used.

Usage: python3 scripts/pitchingplus_weight_multiseason.py
"""
import os, sys, math, pickle, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt

LG, SCALE = 0.3169, 1.2393
MIN_PITCH, MIN_ACTUAL = 200, 200
W_GRID = [round(0.05 * i, 2) for i in range(21)]
BUNDLE = os.path.join(ROOT, 'stuff_plus', 'stuff_models.pkl')


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


def stuff_by_pitcher(train_path, bundle, cutoff):
    """Mean predicted xRV per pitcher over pitches BEFORE cutoff (hitter
    perspective, so it is negated later to make higher = better)."""
    import pandas as pd
    from train_stuff import build_df
    pitches = pickle.load(open(train_path, 'rb'))
    early = [p for p in pitches if str(p.get('Game Date', ''))[:10] < cutoff]
    del pitches
    gc.collect()
    df = build_df(early)
    feats = bundle['features']
    have = [f for f in feats if f in df.columns]
    if len(have) != len(feats):
        # fall back to the no-arm-angle model if arm_angle is unavailable
        feats = bundle['noarm_feats']
        model = bundle['model_na']
        have = [f for f in feats if f in df.columns]
        if len(have) != len(feats):
            return None
    else:
        model = bundle['model']
    X = df[feats].astype(float)
    # LEAKAGE-FREE: score each pitcher with the fold model that held him out.
    # fold_pitchers[k] is the TEST fold for fold_models[k] (train_stuff
    # _oof_predict / _cached_oof), so this reproduces the shipped OOF path
    # rather than scoring training data with the full model. Pitchers newer
    # than the retrain fall to fold 0, which never saw them either.
    import numpy as np
    fmodels = bundle['fold_models'] if model is bundle['model'] else bundle['fold_models_na']
    fold_of = {p: k for k, ps in enumerate(bundle['fold_pitchers']) for p in ps}
    groups = (df['pitcher'] if 'pitcher' in df.columns else df['Pitcher']).values
    pf = np.array([fold_of.get(p, 0) for p in groups])
    out = np.full(len(X), np.nan)
    for k, mm in enumerate(fmodels):
        mask = pf == k
        if mask.any():
            out[mask] = mm.predict(X[mask])
    n_seen = int(np.isin(groups, list(fold_of)).sum())
    print(f"    OOF: {n_seen}/{len(groups)} pitches from pitchers with a known fold",
          file=sys.stderr)
    df['_pred'] = out
    agg = defaultdict(lambda: [0.0, 0])
    for pit, pred in zip(df['pitcher'] if 'pitcher' in df.columns else df['Pitcher'],
                         df['_pred']):
        a = agg[pit]; a[0] += float(pred); a[1] += 1
    del df, early
    gc.collect()
    return {k: v[0] / v[1] for k, v in agg.items() if v[1] >= MIN_PITCH}


def main():
    bundle = pickle.load(open(BUNDLE, 'rb'))
    print(f"bundle v{bundle.get('version')}, {len(bundle['features'])} features",
          file=sys.stderr)
    rv_fn = make_rv_xrv(LG, SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl', 'data/_pitches2021_training.pkl'),
               (2022, 'data/_statcast2022_cache.pkl', 'data/_pitches2022_training.pkl'),
               (2023, 'data/_statcast2023_cache.pkl', 'data/_pitches2023_training.pkl'),
               (2024, 'data/_statcast2024_cache.pkl', 'data/_pitches2024_training.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl', 'data/_pitches2025_training.pkl')]

    print()
    print(f"{'season':>7s} {'n':>5s} | " + "".join(f"{w:>7.2f}" for w in
          (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)) + f" | {'best w':>7s}")
    print('-' * 70)
    bests = {}
    for yr, cache, train in seasons:
        cp, tp = os.path.join(ROOT, cache), os.path.join(ROOT, train)
        if not (os.path.exists(cp) and os.path.exists(tp)):
            print(f"{yr:>7d}  missing input, skipped", flush=True)
            continue
        pitches = adapt(cp)
        base = [p for p in pitches if lp.is_eligible_baseline(p)]
        dates = sorted({p['Game Date'] for p in base if p['Game Date']})
        mid = dates[len(dates) // 2]
        early = [p for p in base if p['Game Date'] < mid]
        late = [p for p in base if p['Game Date'] >= mid]

        S = lp.build_surfaces(early, LG, SCALE)
        loc = defaultdict(lambda: [0.0, 0])
        for p in early:
            v = lp.score_pitch(p, S)
            if v is not None:
                a = loc[p['Pitcher']]; a[0] += v; a[1] += 1
        loc = {k: v[0] / v[1] for k, v in loc.items() if v[1] >= MIN_PITCH}

        actual = defaultdict(lambda: [0.0, 0])
        for p in late:
            v = rv_fn(p)
            if v is not None:
                a = actual[p['Pitcher']]; a[0] += v; a[1] += 1
        actual = {k: v[0] / v[1] for k, v in actual.items() if v[1] >= MIN_ACTUAL}
        del pitches, base, early, late
        gc.collect()

        stuff = stuff_by_pitcher(tp, bundle, mid)
        if stuff is None:
            print(f"{yr:>7d}  feature mismatch, skipped", flush=True)
            continue

        keys = [k for k in loc if k in stuff and k in actual]
        if len(keys) < 40:
            print(f"{yr:>7d}  only {len(keys)} joined pitchers, skipped", flush=True)
            continue
        # orient both so higher = better pitcher (both are hitter-perspective xRV)
        lz = zscore({k: -loc[k] for k in keys})
        sz = zscore({k: -stuff[k] for k in keys})
        ys = [actual[k] for k in keys]
        curve = {}
        for w in W_GRID:
            blend = [w * sz[k] + (1 - w) * lz[k] for k in keys]
            curve[w] = pearson(blend, ys)
        best = min(curve, key=lambda w: curve[w])       # most negative = best
        bests[yr] = (best, curve)
        cells = "".join(f"{curve[w]:>7.3f}" for w in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0))
        print(f"{yr:>7d} {len(keys):>5d} | {cells} | {best:>7.2f}", flush=True)
        del stuff, loc, actual
        gc.collect()

    if not bests:
        print("\nno seasons evaluated")
        return
    print()
    print("(cells are corr(blend, next-half xRV allowed); MORE NEGATIVE = better)")
    print()
    ws = [b for b, _ in bests.values()]
    print(f"best w by season: " + ", ".join(f"{yr}={bests[yr][0]:.2f}" for yr in sorted(bests)))
    print(f"median best w: {sorted(ws)[len(ws) // 2]:.2f}   (shipped: 0.70)")
    print()
    print("Leakage note: 2021-2025 are inside the v11 training set and are scored")
    print("here with the full model, which FLATTERS Stuff+ and biases w upward.")
    print("A result below 0.70 is therefore conservative; above 0.70 is suspect.")


if __name__ == '__main__':
    main()
