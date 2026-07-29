"""axis_retest.py — AXIS at 100% prior coverage (the definitive test).

Round 2 rejected AXIS at 80% prior coverage (2025 had no SpinAxis — a
whole-season feature gap the trees could exploit as an era marker). With
augment_2025_spinaxis.py run, every prior season now carries SpinAxis.
Same protocol as v12_round2_battery.py: depth-7 TUNED, 2021-25 prior in
every fold, 2026 pitcher-grouped 8-fold OOF. BASE vs AXIS only.

Usage: python3 scripts/axis_retest.py
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
import scripts.build_historical_training_set as H

GUTS = dict(H.GUTS)
GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)

p26 = [p for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
       if p.get('_source') == 'MLB']
_ep = {(p.get('Pitcher'), p.get('PTeam')) for p in p26 if p.get('Pitch Type') == 'EP'}
p26 = [p for p in p26 if (p.get('Pitcher'), p.get('PTeam')) not in _ep]
df26 = T.build_df(p26)
df26 = df26[df26['target_xrv'].notna()].reset_index(drop=True)
order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')

prior_dfs = []
for yr in (2021, 2022, 2023, 2024, 2025):
    pk = pickle.load(open(os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
    if yr == 2025:
        T._harmonize_tags(pk, p26)
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = GUTS[yr]
    d = T.build_df(pk)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    prior_dfs.append(d[d['target_xrv'].notna()].reset_index(drop=True))
prior = pd.concat(prior_dfs, ignore_index=True)
print(f'prior axis_dev coverage: {prior.axis_dev.notna().mean()*100:.1f}% '
      f'(2025 slice: {prior_dfs[-1].axis_dev.notna().mean()*100:.1f}%)', flush=True)

y = df26['target_xrv'].values
groups = df26['pitcher'].values
yp = prior['target_xrv'].values


def pear(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


for name, feats in (('BASE', list(T.BASE_FEATS)),
                    ('AXIS', list(T.BASE_FEATS) + ['axis_dev', 'axis_dev_abs'])):
    t0 = time.time()
    X = T.design(df26, feats)
    Xp = T.design(prior, feats).reindex(columns=X.columns, fill_value=0)
    params = T._params_for(X)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X, y, groups):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X.iloc[tr], Xp], ignore_index=True),
              np.concatenate([y[tr], yp]))
        oof[te] = m.predict(X.iloc[te])
    d = df26.copy()
    d['stuff'] = -oof
    a0, a1, est, late = [], [], {}, {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.mean(); est[key] = e.stuff.mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    ks = list(est)
    rel = pear(a0, a1)
    pred = -pear([est[k] for k in ks], [late[k] for k in ks])
    dx = [g.stuff.mean() for _, g in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(g) >= 100]
    dy = [g.target_xrv.mean() for _, g in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(g) >= 100]
    desc = -pear(dx, dy)
    print(f'{name:5s} reliab {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}  [{time.time()-t0:.0f}s]', flush=True)
