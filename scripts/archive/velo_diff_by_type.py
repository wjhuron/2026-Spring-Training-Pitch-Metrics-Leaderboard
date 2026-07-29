"""velo_diff_by_type.py — does velo_diff (velocity off the primary fastball)
earn its place per pitch type? (2026-07-08)

Hypothesis (Wally): velo separation should matter more for CH/FS (deception
pitches whose value IS arriving slower than a look-alike fastball) than for
SL (which sells on shape/gyro deception).

Two lenses, both at the (pitcher, throws, pitch_type) unit level:

  1. MODEL-FREE signal: within each pitch type, correlation between velo
     SEPARATION (mph slower than the pitcher's primary fastball) and RUN
     PREVENTION (-target_xrv, higher = better for pitcher). Positive = more
     separation associates with better results. Also reports the SD of
     velo_diff per type — velo_diff can only matter where it varies.

  2. PERMUTATION (model reliance that's earned): production 8-fold OOF model,
     scramble velo_diff, measure the drop in each pitch type's descriptive r
     (mean OOF stuff vs mean target, >=100-pitch units). Bigger drop = the
     model's grade for that type genuinely leans on velo_diff. Isolates
     velo_diff's MARGINAL contribution (velocity itself is held fixed), which
     the raw within-type correlation cannot.

Uses the retune feature cache (df25/df26). Season-blocked protocol, same as
stuff_weight_audit.

Usage: STUFF_RETUNE_CACHE=<dir> python3 scripts/velo_diff_by_type.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import stuff_plus_v11.train_stuff_v11 as T
import stuff_hp_retune as R

ALL_FEATS = list(T.BASE_FEATS) + ['platoon_same']
MIN_UNIT = 100


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    df25, df26 = R.load_dfs(use_cache=True)
    y = df26['target_xrv'].values
    groups = df26['pitcher'].values
    X26 = df26[ALL_FEATS].reset_index(drop=True)
    X25 = df25[ALL_FEATS].reset_index(drop=True)
    y25 = df25['target_xrv'].values
    params = dict(T.TUNED)
    params['monotone_constraints'] = tuple(-1 if c == T.MONO_FEAT else 0
                                           for c in X26.columns)

    # 8-fold OOF, keep fold models + test indices for permutation
    base = np.full(len(df26), np.nan)
    models, tests = [], []
    for tr, te in GroupKFold(n_splits=8).split(X26, y, groups):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X26.iloc[tr], X25], ignore_index=True),
              np.concatenate([y[tr], y25]))
        base[te] = m.predict(X26.iloc[te])
        models.append(m); tests.append(te)
    df26['stuff_base'] = -base

    # permute velo_diff, re-score with the SAME fold models
    rng = np.random.RandomState(20260708)
    Xp = X26.copy()
    Xp['velo_diff'] = Xp['velo_diff'].values[rng.permutation(len(Xp))]
    perm = np.full(len(df26), np.nan)
    for m, te in zip(models, tests):
        perm[te] = m.predict(Xp.iloc[te])
    df26['stuff_perm'] = -perm

    print(f"\n{'pt':4s} {'n_units':>7s} {'veloDiff_mean':>13s} {'veloDiff_sd':>11s} "
          f"{'sep~prev_r':>10s} {'desc_base':>9s} {'desc_perm':>9s} {'drop':>6s}")
    print('-' * 76)
    rows = []
    for pt, sub in df26.groupby('pitch_type'):
        g = sub.groupby(['pitcher', 'throws']).agg(
            vd=('velo_diff', 'mean'), tgt=('target_xrv', 'mean'),
            sb=('stuff_base', 'mean'), sp=('stuff_perm', 'mean'),
            n=('velo_diff', 'size')).reset_index()
        gq = g[g['n'] >= MIN_UNIT]
        if len(gq) < 8:
            continue
        # model-free: separation (= -velo_diff) vs run prevention (= -target)
        sep_prev = pearson(-gq['vd'], -gq['tgt'])
        desc_base = pearson(gq['sb'], gq['tgt'])
        desc_perm = pearson(gq['sp'], gq['tgt'])
        db = None if desc_base is None else -desc_base
        dp = None if desc_perm is None else -desc_perm
        drop = (db - dp) if (db is not None and dp is not None) else float('nan')
        vd_sd = float(sub['velo_diff'].std())
        vd_mean = float(sub['velo_diff'].mean())
        rows.append((pt, len(gq), vd_mean, vd_sd, sep_prev, db, dp, drop))
        print(f"{pt:4s} {len(gq):7d} {vd_mean:13.1f} {vd_sd:11.2f} "
              f"{(sep_prev if sep_prev is not None else float('nan')):10.3f} "
              f"{(db if db is not None else float('nan')):9.3f} "
              f"{(dp if dp is not None else float('nan')):9.3f} {drop:6.3f}")
    print('-' * 76)
    print("sep~prev_r  : model-free, corr(velo separation, run prevention); + = separation helps")
    print("desc_base   : model's descriptive r for that type (higher = grade tracks results)")
    print("drop        : desc_base - desc_perm; how much that type's grade LEANS on velo_diff")
    pd.DataFrame(rows, columns=['pt', 'n', 'vd_mean', 'vd_sd', 'sep_prev_r',
                                'desc_base', 'desc_perm', 'drop']).to_csv(
        os.path.join(ROOT, 'scripts', 'velo_diff_by_type.csv'), index=False)


if __name__ == '__main__':
    main()
