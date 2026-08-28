#!/usr/bin/env python3
"""pitcherplus_aging_proj.py — does a curved aging term improve Pitcher+
Proj? (Approved by Wally 2026-08-28.)

The July research shipped Proj = 0.70*current + 0.30*prior Pitcher+ with
aging explicitly untested (no birthdates then; cached now at
data/_pplus_birthdates.json). The linear cross-sectional age term failed
the stage B screen, but Marcel-style aging is a CURVE applied to a
projection, so this tests curve forms directly on the Proj task:

  unit      pitcher-season triples (Y-1 prior, Y current, Y+1 target),
            plus current-only pairs (no prior -> proj = current, the
            shipped Marcel pattern)
  score     shipped frozen composite (v14 stuff series), standardized per
            season on the qualified pool — the research analog of the
            production Pitcher+ z
  target    xRV/100 in season Y+1 (>= 800 pitches both ends, the panel Y
            convention)
  folds     leave-one-target-season-out over Y in 2022..2024 (3 folds:
            triples need three consecutive seasons)
  forms     PROJ alone | +linear age | +quadratic (age-peak)^2 with peak
            swept 26..33 | +piecewise slopes below/above swept peak
  verdict   OOF pooled r + per-fold wins vs PROJ alone. Adopt only on
            majority-fold wins with a gain that clears the fold spread.

Usage: PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_aging_proj.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pitcherplus_search as ps  # noqa: E402

SHIPPED = (
    ('stuffRaw',   0.20, 42.0),
    ('locRaw',     0.06, 215.0),
    ('kPct',       0.21, 398.0),
    ('izWhiffPct', 0.19, 421.0),
    ('xrv100',     0.23, 1046.0),
    ('gbPct',      0.12, 333.0),
)
MIN_FULL = 800
Q_FULL = 800
CUR_W, PRIOR_W = 0.70, 0.30


def load_scores():
    """(pid, season) -> standardized shipped-composite z + xrv100 + age."""
    t = pd.read_pickle(ps.TABLES_PKL) if ps.TABLES_PKL.endswith('.pkl') else None
    import pickle
    t = pickle.load(open(ps.TABLES_PKL, 'rb'))
    for path, cols in ((ps.LOC_CSV, ['locRaw']), (ps.STUFF_CSV, ['stuffRaw'])):
        ext = pd.read_csv(path)
        t = t.merge(ext[['pid', 'season', 'half'] + cols],
                    on=['pid', 'season', 'half'], how='left')
    feats = [f for f, _w, _k in SHIPPED]
    for c in feats + ['xrv100']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    full = t[(t['half'] == 'full') & (t['n'] >= MIN_FULL)].copy()

    # shrunk-z composite per season (production convention: n-weighted mu,
    # unweighted sd, qualified pool)
    full['comp'] = 0.0
    for season, g in full.groupby('season'):
        q = g[g['n'] >= Q_FULL]
        comp = np.zeros(len(g))
        for f, w, k in SHIPPED:
            vals = q[f].dropna()
            mu = float(np.average(vals, weights=q.loc[vals.index, 'n']))
            sd = float(vals.std())
            z = ((g[f] - mu) / sd).fillna(0.0)
            comp += w * (z * (g['n'] / (g['n'] + k))).to_numpy()
        # standardize the composite itself (the 100+/-10 z)
        cmu, csd = comp.mean(), comp.std(ddof=1)
        full.loc[g.index, 'comp'] = (comp - cmu) / csd

    bd = json.load(open(os.path.join(ps.DATA, '_pplus_birthdates.json')))
    bdt = {int(k): pd.Timestamp(v) for k, v in bd.items() if v}
    mid = full['season'].map(lambda s: pd.Timestamp(int(s), 7, 1))
    full['age'] = (mid - full['pid'].map(bdt)).dt.days / 365.25
    return full[['pid', 'season', 'n', 'comp', 'xrv100', 'age']]


def build_panel(s):
    """One row per (pid, Y): proj at Y (blend when a prior exists),
    age at Y, target xrv100 at Y+1."""
    cur = s.rename(columns={'comp': 'cur'})
    pri = s[['pid', 'season', 'comp']].rename(columns={'comp': 'prior'})
    pri = pri.assign(season=pri['season'] + 1)
    nxt = s[['pid', 'season', 'xrv100']].rename(columns={'xrv100': 'y_next'})
    nxt = nxt.assign(season=nxt['season'] - 1)
    p = cur.merge(pri, on=['pid', 'season'], how='left') \
           .merge(nxt, on=['pid', 'season'], how='inner')
    p['proj'] = np.where(p['prior'].notna(),
                         CUR_W * p['cur'] + PRIOR_W * p['prior'], p['cur'])
    p = p[p['y_next'].notna() & p['age'].notna()]
    return p.reset_index(drop=True)


def oof_eval(p, cols):
    pred = np.full(len(p), np.nan)
    folds = {}
    y = p['y_next'].to_numpy(float)
    for Y in sorted(p['season'].unique()):
        tr = p['season'] != Y
        te = ~tr
        A = np.column_stack([np.ones(tr.sum())] + [p.loc[tr, c] for c in cols])
        beta, *_ = np.linalg.lstsq(A, y[tr.to_numpy()], rcond=None)
        At = np.column_stack([np.ones(te.sum())] + [p.loc[te, c] for c in cols])
        pv = At @ beta
        pred[te.to_numpy()] = pv
        folds[int(Y)] = float(np.corrcoef(pv, y[te.to_numpy()])[0, 1])
    return float(np.corrcoef(pred, y)[0, 1]), folds


def main():
    s = load_scores()
    p = build_panel(s)
    print(f'panel: {len(p)} pitcher-seasons, target seasons '
          f'{sorted(p.season.unique())}, prior available '
          f'{p["prior"].notna().mean():.2f}, age {p.age.min():.1f}-'
          f'{p.age.max():.1f} (median {p.age.median():.1f})')

    results = {}
    r0, f0 = oof_eval(p, ['proj'])
    results['PROJ alone'] = (r0, f0)
    p['age_lin'] = p['age']
    results['+linear age'] = oof_eval(p, ['proj', 'age_lin'])

    best_quad, best_pw = None, None
    for peak in range(26, 34):
        p['age_q'] = -np.square(p['age'] - peak)
        rq = oof_eval(p, ['proj', 'age_q'])
        if best_quad is None or rq[0] > best_quad[1][0]:
            best_quad = (peak, rq)
        p['age_lo'] = np.minimum(p['age'] - peak, 0)      # rise to peak
        p['age_hi'] = np.maximum(p['age'] - peak, 0)      # decline after
        rp = oof_eval(p, ['proj', 'age_lo', 'age_hi'])
        if best_pw is None or rp[0] > best_pw[1][0]:
            best_pw = (peak, rp)
    results[f'+quadratic (peak {best_quad[0]})'] = best_quad[1]
    results[f'+piecewise (peak {best_pw[0]})'] = best_pw[1]

    print('\n══ OOF results (target: next-season xRV/100) ══')
    base_folds = f0
    for name, (r, folds) in results.items():
        wins = sum(1 for Y, fr in folds.items() if fr > base_folds[Y])
        spread = float(np.std(list(folds.values())))
        print(f'{name:26s} pooled r {r:.4f}  '
              f'folds ' + ' '.join(f'{Y}:{fr:.4f}' for Y, fr in
                                   sorted(folds.items()))
              + (f'  wins {wins}/3' if name != 'PROJ alone' else '')
              + f'  (fold sd {spread:.4f})')

    # peak sweep curve for the quadratic (interior optimum or flat?)
    print('\nquadratic peak sweep (pooled OOF r):')
    for peak in range(26, 34):
        p['age_q'] = -np.square(p['age'] - peak)
        r, _ = oof_eval(p, ['proj', 'age_q'])
        print(f'  peak {peak}: {r:.4f}')


if __name__ == '__main__':
    main()
