"""stuff_hp_loso_check.py — do the Stuff+ hyperparameters survive seasons they
were never tuned on?

TUNED (max_depth 7, n_estimators 800, lr 0.025, mcw 10, lambda 1.5) came out of
scripts/stuff_hp_retune.py, whose evaluation was 2026 OOF ONLY — the same
single-(partial-)season protocol that made the Loc+ bandwidth search look like
a win before it lost 0/5 across 2021-2025. The training DATA is multi-season;
the hyperparameter CHOICE was never validated outside 2026. This closes that.

PROTOCOL. For each season Y in 2021-2025 and each config: fit on the OTHER
four seasons (per-season wOBA guts, production monotone velocity constraint),
score Y held out. Two objectives per (config, season):

  pitch_r   r(prediction, target_xrv) over every held-out pitch. The direct
            model-fit measure — most sensitive to hyperparameters.
  fut_r     pitcher-level: mean prediction over first-half pitches (>=200)
            vs actual xRV/100 allowed in the second half (>=200). The
            decision-relevant measure, matching the repo's standard.

CONFIGS: one-axis neighbors around shipped. This is a VALIDATION screen, not a
re-tune — the question is "does shipped hold up out of season", answered by
whether any neighbor beats it consistently across seasons. A full grid search
would need its own multi-season protocol and is only warranted if shipped
loses here.

Usage: python3 scripts/stuff_hp_loso_check.py
"""
import os, sys, math, pickle, gc, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import train_stuff as T
from locplus_constants_multiseason import adapt
from pipeline_sdplus import make_rv_xrv
import pipeline_locplus as lp

LG, SCALE = 0.3169, 1.2393
MIN_PITCH, MIN_ACTUAL = 200, 200
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = dict(T.HIST_GUTS); GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)
TRAIN_PKL = {y: (T.HIST_PKL.format(year=y) if y != 2025 else T.PRIOR_PKL) for y in SEASONS}
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl',
         2023: 'data/_statcast2023_cache.pkl', 2024: 'data/_statcast2024_cache.pkl',
         2025: 'data/_statcast2025_full_cache.pkl'}

CONFIGS = [
    ('shipped',   {}),
    ('depth5',    {'max_depth': 5}),
    ('depth9',    {'max_depth': 9}),
    ('lr05_n400', {'learning_rate': 0.05, 'n_estimators': 400}),
    ('lr0125_n1600', {'learning_rate': 0.0125, 'n_estimators': 1600}),
    ('mcw40',     {'min_child_weight': 40}),
    ('lambda5',   {'reg_lambda': 5.0}),
]


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


def build_season_df(year):
    lg, sc = GUTS[year]
    old = (T.LG_WOBA, T.WOBA_SCALE)
    T.LG_WOBA, T.WOBA_SCALE = lg, sc
    try:
        d = T.build_df(pickle.load(open(TRAIN_PKL[year], 'rb')))
    finally:
        T.LG_WOBA, T.WOBA_SCALE = old
    return d[d['target_xrv'].notna()].reset_index(drop=True)


def main():
    import pandas as pd, numpy as np, xgboost as xgb
    print("building per-season frames...", file=sys.stderr)
    dfs = {y: build_season_df(y) for y in SEASONS}
    for y in SEASONS:
        print(f"  {y}: {len(dfs[y])} rows", file=sys.stderr)
    gc.collect()

    # second-half actual xRV/100 per pitcher, and each season's midpoint,
    # from the raw caches (same harness as the Pitching+ LOSO work)
    rv_fn = make_rv_xrv(LG, SCALE)
    actuals, mids = {}, {}
    for y in SEASONS:
        pitches = adapt(os.path.join(ROOT, CACHE[y]))
        base = [p for p in pitches if lp.is_eligible_baseline(p)]
        dates = sorted({p['Game Date'] for p in base if p['Game Date']})
        mids[y] = dates[len(dates) // 2]
        acc = defaultdict(lambda: [0.0, 0])
        for p in base:
            if p['Game Date'] < mids[y]:
                continue
            v = rv_fn(p)
            if v is not None:
                a = acc[p['Pitcher']]; a[0] += v; a[1] += 1
        actuals[y] = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_ACTUAL}
        del pitches, base
        gc.collect()

    print()
    print(f"{'config':>13s} {'season':>7s} | {'pitch_r':>8s} | {'fut_r':>7s} {'n':>5s}")
    print('-' * 50)
    results = defaultdict(dict)
    for name, over in CONFIGS:
        for y in SEASONS:
            t0 = time.time()
            tr = pd.concat([dfs[o] for o in SEASONS if o != y], ignore_index=True)
            Xtr = T.design(tr)
            params = T._params_for(Xtr)
            params.update(over)
            model = xgb.XGBRegressor(**params)
            model.fit(Xtr, tr['target_xrv'].values)
            del tr, Xtr
            gc.collect()

            te = dfs[y]
            Xte = T.design(te).reindex(columns=model.get_booster().feature_names,
                                       fill_value=0)
            pred = model.predict(Xte)
            pitch_r = pearson(list(map(float, pred)), list(te['target_xrv'].values))

            first = te['date'].astype(str) < mids[y]
            agg = defaultdict(lambda: [0.0, 0])
            for pit, pr, ok in zip(te['pitcher'].values, pred, first.values):
                if ok:
                    a = agg[pit]; a[0] += float(pr); a[1] += 1
            stuff = {k: v[0] / v[1] for k, v in agg.items() if v[1] >= MIN_PITCH}
            kk = [k for k in stuff if k in actuals[y]]
            fut_r = pearson([stuff[k] for k in kk], [actuals[y][k] for k in kk])
            results[name][y] = (pitch_r, fut_r, len(kk))
            print(f"{name:>13s} {y:>7d} | {pitch_r:>8.4f} | {fut_r:>7.3f} {len(kk):>5d}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
            del model, Xte, pred
            gc.collect()

    print()
    print("VERDICT — per config: seasons beating shipped on each objective")
    ship = results['shipped']
    print(f"{'config':>13s} {'pitch_r wins':>13s} {'fut_r wins':>11s} "
          f"{'mean d(pitch_r)':>16s} {'mean d(fut_r)':>14s}")
    print('-' * 72)
    for name, _o in CONFIGS:
        if name == 'shipped':
            continue
        r = results[name]
        pw = sum(1 for y in SEASONS if r[y][0] > ship[y][0])
        fw = sum(1 for y in SEASONS if r[y][1] > ship[y][1])
        dp = sum(r[y][0] - ship[y][0] for y in SEASONS) / len(SEASONS)
        df_ = sum(r[y][1] - ship[y][1] for y in SEASONS) / len(SEASONS)
        print(f"{name:>13s} {pw:>10d}/5 {fw:>8d}/5 {dp:>+16.4f} {df_:>+14.4f}")
    print()
    print("Shipped SURVIVES if no neighbor beats it in most seasons on both")
    print("objectives. A consistent winner would mean the 2026-only retune")
    print("picked the wrong region and a proper multi-season re-tune is owed.")


if __name__ == '__main__':
    main()
