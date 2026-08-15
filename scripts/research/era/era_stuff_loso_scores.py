"""era_stuff_loso_scores.py — per-pitcher-season Stuff+ (raw, xRV scale),
2021-2025, leakage-free, for the ERA-estimator screen.

Protocol matches scripts/research/stuff/stuff_features_loso.py SHIPPED config: for each
held-out season Y, fit the production feature set (T.BASE_FEATS, production
params, per-season Guts) on the OTHER four seasons, score Y. A pitcher's
score = mean per-pitch stuff (pitcher-positive, -prediction of xRV):
'full' = all season pitches, 'h1' = pitches on/before the All-Star date.

2026 is NOT scored here — the battery already carries the shipped v13
per-pitch means from the sheet. Scale differs (raw xRV here vs 100-scale
there); the screen correlates within season, so scale is irrelevant.

Output: data/_era_internal_stuff.json
  {season: {pid: {stuff_full, stuff_h1, n_full, n_h1}}}
"""
import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import stuff_plus.train_stuff as T
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stuff'))  # stuff_features_loso moved in 2026-08 reorg
import stuff_features_loso as L
from pipeline.utils import _fullname_to_lastfirst

OUT = os.path.join(ROOT, 'data', '_era_internal_stuff.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
SEASONS = (2021, 2022, 2023, 2024, 2025)
BASE = list(T.BASE_FEATS)
MIN_N = 50


def name_map(season):
    m, ambig = {}, set()
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        full = (rec['name'] or '').strip()
        variants = {_fullname_to_lastfirst(full).lower()}
        parts = full.split()
        if len(parts) >= 3:
            variants.add((' '.join(parts[-2:]) + ', '
                          + ' '.join(parts[:-2])).lower())
        for lf in variants:
            if lf in m and m[lf] != int(pid):
                ambig.add(lf)
            m[lf] = int(pid)
    for lf in ambig:
        del m[lf]
    return m


def main():
    result = {}
    for Y in SEASONS:
        train_years = [y for y in SEASONS if y != Y]
        print(f'=== held-out {Y} (train {train_years}) ===', flush=True)
        tr_dfs = [L.build_year(y) for y in train_years]
        dY = L.build_year(Y)
        Xtr = pd.concat([T.design(d, BASE) for d in tr_dfs],
                        ignore_index=True)
        ytr = np.concatenate([d['target_xrv'].values for d in tr_dfs])
        del tr_dfs
        gc.collect()
        XY = T.design(dY, BASE).reindex(columns=Xtr.columns, fill_value=0)
        m = xgb.XGBRegressor(**T._params_for(Xtr))
        m.fit(Xtr, ytr)
        stuff = -m.predict(XY)
        del Xtr, XY, m
        gc.collect()

        dY = dY.assign(stuff=stuff)
        asg = TARGETS[str(Y)]['asg']
        nm = name_map(Y)
        rec = {}
        unmatched = 0
        for name, g in dY.groupby('pitcher'):
            pid = nm.get((str(name) or '').lower())
            if pid is None:
                unmatched += 1
                continue
            gh = g[g['date'].astype(str).str[:10] <= asg]
            r = {}
            if len(g) >= MIN_N:
                r['stuff_full'] = float(g['stuff'].mean())
                r['n_full'] = int(len(g))
            if len(gh) >= MIN_N:
                r['stuff_h1'] = float(gh['stuff'].mean())
                r['n_h1'] = int(len(gh))
            if r:
                rec[str(pid)] = r
        result[str(Y)] = rec
        print(f'  {Y}: {len(rec)} pitchers, {unmatched} unmatched names',
              flush=True)
        del dY
        gc.collect()
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
