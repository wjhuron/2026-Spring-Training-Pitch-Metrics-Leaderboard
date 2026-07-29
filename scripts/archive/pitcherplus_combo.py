#!/usr/bin/env python3
"""pitcherplus_combo.py — Pitcher+ phase 2: combination search.

Finds the best small linear composite of the phase 1 survivors for
predicting FUTURE xRV/100, with every component shrunk by its own
stabilization constant before combination.

Design:
  features   z-scored per season (n-weighted league mean, qualified pool),
             then shrunk z * n/(n + k_f) with k_f = the feature's split-half
             stabilize_n from data/_pplus_screen.csv; missing -> 0 (a missing
             component IS league-average under shrinkage logic)
  panel S    half A features -> half B xRV/100, both directions stacked,
             OOF by leave-one-season-out (5 folds)
  panel Y    full-season features year N -> xRV/100 year N+1,
             OOF by leave-one-year-pair-out (4 folds)
  search     exhaustive best-subset sizes 1-6 over the survivor pool +
             LASSO stability selection (per-fold LassoCV, selection freq)
  pick       smallest subset within 1 SE of the best combined OOF score
             (combined = mean of panel S and panel Y OOF r)
  benchmarks kbbPct (kwERA-core), fipCore (FIP-core), xrv100 (regressed
             current performance), 0.7*z(stuffRaw)+0.3*z(locRaw) (shipped
             Pitching+ proxy) — all run through the same OOF machinery

Usage: python3 scripts/pitcherplus_combo.py            (~5-15 min)
Outputs: data/_pplus_combo_results.csv + console report
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pitcherplus_search as ps  # noqa: E402

OUT_CSV = os.path.join(ROOT, 'data', '_pplus_combo_results.csv')

SURVIVORS = ['stuffRaw', 'locRaw', 'xrvoe', 'fbVelo', 'kPct', 'kbbPct',
             'cswPct', 'putawayPct', 'whiffPct', 'izWhiffPct', 'chasePct',
             'xrv100', 'fipCore', 'xwobacon', 'barrelPct', 'gbPct',
             'weakSideXrv100', 'bestPitchXrv100', 'usageEntropy',
             'pitchesPerG']
MAX_K = 6
MIN_HALF, MIN_FULL = 300, 800
Q_HALF, Q_FULL = 200, 800          # qualified pools for z params


def load_tables():
    t = pickle.load(open(ps.TABLES_PKL, 'rb'))
    t = ps.merge_external(t)
    feats = [c for c in t.columns if c not in ps.META]
    for c in feats:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t = ps.add_xrvoe(t)
    return t


def stab_constants():
    scr = pd.read_csv(ps.SCREEN_CSV).set_index('feature')
    k = {}
    for f in SURVIVORS:
        v = scr['stabilize_n'].get(f, np.nan)
        k[f] = float(v) if v == v and v > 0 else 1000.0
    return k


def shrunk_z(rows, feats, kmap, qmin):
    """Per-season z (n-weighted mean, qualified-pool sd), shrunk by
    n/(n+k_f); missing -> 0. Returns a copy with <feat>_sz columns."""
    rows = rows.copy()
    for f in feats:
        rows[f + '_sz'] = 0.0
    for season, g in rows.groupby('season'):
        q = g[g['n'] >= qmin]
        for f in feats:
            vals = q[f].dropna()
            if len(vals) < 30:
                continue
            w = q.loc[vals.index, 'n']
            mu = float(np.average(vals, weights=w))
            sd = float(vals.std())
            if not sd:
                continue
            idx = g.index[g[f].notna()]
            z = (g.loc[idx, f] - mu) / sd
            shrink = g.loc[idx, 'n'] / (g.loc[idx, 'n'] + kmap[f])
            rows.loc[idx, f + '_sz'] = z * shrink
    return rows


def build_panels(t, kmap):
    halves = t[t['half'].isin(['A', 'B'])]
    halves = shrunk_z(halves, SURVIVORS, kmap, Q_HALF)
    A = halves[(halves['half'] == 'A') & (halves['n'] >= MIN_HALF)]
    B = halves[(halves['half'] == 'B') & (halves['n'] >= MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    sz = [f + '_sz' for f in SURVIVORS]
    S_X = np.vstack([ab[[c + '_a' for c in sz]].to_numpy(float),
                     ab[[c + '_b' for c in sz]].to_numpy(float)])
    S_y = np.concatenate([ab['xrv100_b'].to_numpy(float),
                          ab['xrv100_a'].to_numpy(float)])
    S_grp = np.concatenate([ab['season'].to_numpy()] * 2)

    full = t[(t['half'] == 'full') & (t['n'] >= MIN_FULL)]
    full = shrunk_z(full, SURVIVORS, kmap, Q_FULL)
    pairs = full.merge(full.assign(season=full['season'] - 1),
                       on=['pid', 'season'], suffixes=('', '_n1'))
    Y_X = pairs[sz].to_numpy(float)
    Y_y = pairs['xrv100_n1'].to_numpy(float)
    Y_grp = pairs['season'].to_numpy()

    ok_s = np.isfinite(S_y)
    ok_y = np.isfinite(Y_y)
    return ((S_X[ok_s], S_y[ok_s], S_grp[ok_s]),
            (Y_X[ok_y], Y_y[ok_y], Y_grp[ok_y]))


def oof_r(X, y, grp, cols):
    """OLS per left-out group; returns pooled OOF Pearson r and per-fold rs."""
    Xs = X[:, cols]
    pred = np.full(len(y), np.nan)
    fold_rs = []
    for g in np.unique(grp):
        tr, te = grp != g, grp == g
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
        pred[te] = p
        if te.sum() >= 30 and np.std(p) > 0:
            fold_rs.append(float(np.corrcoef(p, y[te])[0, 1]))
    r = float(np.corrcoef(pred, y)[0, 1])
    return r, fold_rs


def fit_weights(X, y, cols):
    A = np.column_stack([np.ones(len(y)), X[:, cols]])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def lasso_stability(X, y, grp):
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    freq = np.zeros(X.shape[1])
    for g in np.unique(grp):
        tr = grp != g
        Xs = StandardScaler().fit_transform(X[tr])
        m = LassoCV(cv=5, n_alphas=40, max_iter=20000).fit(Xs, y[tr])
        freq += (np.abs(m.coef_) > 1e-8)
    return freq / len(np.unique(grp))


def main():
    t = load_tables()
    kmap = stab_constants()
    print('shrinkage constants:',
          {f: round(k) for f, k in sorted(kmap.items(), key=lambda x: x[1])})
    (S_X, S_y, S_grp), (Y_X, Y_y, Y_grp) = build_panels(t, kmap)
    print(f'panel S: {len(S_y)} obs, panel Y: {len(Y_y)} obs')

    try:
        print('\nLASSO stability selection (panel S folds / panel Y folds):')
        fs = lasso_stability(S_X, S_y, S_grp)
        fy = lasso_stability(Y_X, Y_y, Y_grp)
        stab = pd.DataFrame({'feature': SURVIVORS, 'freq_S': fs,
                             'freq_Y': fy})
        print(stab.sort_values('freq_S', ascending=False)
              .round(2).to_string(index=False))
    except ImportError:
        print('  (sklearn unavailable — skipping LASSO stability)')

    # exhaustive best-subset
    results = []
    idx_all = range(len(SURVIVORS))
    n_sub = 0
    for k in range(1, MAX_K + 1):
        for cols in itertools.combinations(idx_all, k):
            cols = list(cols)
            r_s, frs = oof_r(S_X, S_y, S_grp, cols)
            r_y, fry = oof_r(Y_X, Y_y, Y_grp, cols)
            comb = (r_s + r_y) / 2
            folds = frs + fry
            se = float(np.std(folds) / np.sqrt(len(folds))) if folds else np.nan
            results.append({'k': k,
                            'subset': '+'.join(SURVIVORS[i] for i in cols),
                            'r_S': r_s, 'r_Y': r_y, 'combined': comb,
                            'se': se})
            n_sub += 1
        print(f'  size {k} done ({n_sub} subsets total)', flush=True)
    res = pd.DataFrame(results).sort_values('combined', ascending=False)

    # benchmarks through the same machinery
    bench = []
    for name, colset in (('BENCH kwERA-core (kbbPct)', ['kbbPct']),
                         ('BENCH FIP-core', ['fipCore']),
                         ('BENCH regressed xRV/100', ['xrv100']),
                         ('BENCH Pitching+ proxy', None)):
        if colset is not None:
            cols = [SURVIVORS.index(c) for c in colset]
            r_s, _ = oof_r(S_X, S_y, S_grp, cols)
            r_y, _ = oof_r(Y_X, Y_y, Y_grp, cols)
        else:
            iS, iL = SURVIVORS.index('stuffRaw'), SURVIVORS.index('locRaw')
            pp_s = 0.7 * S_X[:, iS] + 0.3 * S_X[:, iL]
            pp_y = 0.7 * Y_X[:, iS] + 0.3 * Y_X[:, iL]
            r_s = float(np.corrcoef(pp_s, S_y)[0, 1])
            r_y = float(np.corrcoef(pp_y, Y_y)[0, 1])
        bench.append({'k': 0, 'subset': name, 'r_S': r_s, 'r_Y': r_y,
                      'combined': (r_s + r_y) / 2, 'se': np.nan})

    out = pd.concat([pd.DataFrame(bench), res], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    best = res.iloc[0]
    thresh = best['combined'] - best['se']
    onese = res[res['combined'] >= thresh].sort_values('k').iloc[0]
    print('\n════ benchmarks ════')
    print(pd.DataFrame(bench).round(3).to_string(index=False))
    print('\n════ top 12 subsets (combined OOF r) ════')
    print(res.head(12).round(3).to_string(index=False))
    print(f'\nbest: {best["subset"]}  combined {best["combined"]:.3f} '
          f'(se {best["se"]:.3f})')
    print(f'1-SE pick: {onese["subset"]}  combined {onese["combined"]:.3f} '
          f'(k={onese["k"]})')
    print('\nbest-per-size:')
    print(res.groupby('k').head(1).round(3).to_string(index=False))
    # full-data weights for the 1-SE pick and the best subset, panel S fit
    for label, row in (('1-SE', onese), ('best', best)):
        cols = [SURVIVORS.index(c) for c in row['subset'].split('+')]
        beta = fit_weights(S_X, S_y, cols)
        terms = ', '.join(f'{b:+.3f}*{SURVIVORS[i]}'
                          for b, i in zip(beta[1:], cols))
        print(f'{label} weights (panel S, full fit): {beta[0]:+.3f} {terms}')


if __name__ == '__main__':
    main()
