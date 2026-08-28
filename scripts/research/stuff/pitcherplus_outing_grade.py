#!/usr/bin/env python3
"""pitcherplus_outing_grade.py — design search for the daily-card outing
grade (Pitcher+ overhaul part 2, 2026-08-28).

Question: which mix of outing-level components best estimates how well the
pitcher pitched TONIGHT? Operationalized without a latent variable: fit
components measured on the odd-PA half of an outing to the xRV/100 the
pitcher produced on the even-PA half of the SAME outing (both directions
stacked). The mix that transfers across halves of one game is the mix that
captures tonight's quality rather than tonight's noise.

Panels (data/_pplus_outing_tables.pkl + _pplus_stuff_loso_games.csv):
  screen   per-feature: odd/even reliability, outing-grain stabilize_n,
           split-half predictive r
  combo    OOF (leave-one-season-out) OLS on shrunk z-features,
           benchmark subsets + exhaustive search over the survivor set
  context  chosen composite vs NEXT outing xRV/100 (how much of an outing
           grade is signal about the pitcher at all)

Usage:
  PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_outing_grade.py
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
TABLES = os.path.join(DATA, '_pplus_outing_tables.pkl')
STUFF_GAMES = os.path.join(DATA, '_pplus_stuff_loso_games.csv')

FEATS = ['stuffRaw', 'locRaw', 'kPct', 'bbPct', 'kbbPct', 'cswPct',
         'whiffPct', 'izWhiffPct', 'chasePct', 'gbPct', 'xrv100', 'rv100']
MIN_N = 20            # outing floor for the analysis pool
MIN_HALF_N = 8        # per-half floor for the split panel


def load():
    t = pickle.load(open(TABLES, 'rb'))
    sg = pd.read_csv(STUFF_GAMES)
    t['date'] = pd.to_datetime(t['date']).dt.strftime('%Y-%m-%d')
    sg['date'] = pd.to_datetime(sg['date']).dt.strftime('%Y-%m-%d')
    t = t.merge(sg[['pid', 'season', 'date', 'stuffRaw']],
                on=['pid', 'season', 'date'], how='left')
    print(f'{len(t)} outings, stuffRaw matched '
          f'{t["stuffRaw"].notna().mean():.3f}')
    # stuffRaw is a game-level merge (no per-half split in the games CSV):
    # use the game value for both halves. Its rel_r is 1.0 by construction
    # and its fitted weight is biased UP relative to a true half
    # measurement; flagged in the report. locRaw_o/_e are REAL per-half
    # values from the outing table.
    t['stuffRaw_o'] = t['stuffRaw']
    t['stuffRaw_e'] = t['stuffRaw']
    return t


def pear(a, b):
    a = pd.to_numeric(pd.Series(a), errors='coerce')
    b = pd.to_numeric(pd.Series(b), errors='coerce')
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def screen(t):
    pool = t[t['n'] >= MIN_N]
    hp = pool[(pool['n_o'] >= MIN_HALF_N) & (pool['n_e'] >= MIN_HALF_N)]
    rows = []
    for f in FEATS:
        rel, n_rel = pear(hp[f + '_o'], hp[f + '_e'])
        half_n = float(np.nanmean(np.minimum(hp['n_o'], hp['n_e'])))
        stab = half_n * (1 - rel) / rel if rel and rel > 0 else np.nan
        pa_ = np.concatenate([hp[f + '_o'], hp[f + '_e']])
        tb = np.concatenate([hp['xrv100_e'], hp['xrv100_o']])
        pred, n_pred = pear(pa_, tb)
        rows.append({'feature': f, 'rel_r': rel, 'stabilize_n': stab,
                     'pred_half_r': pred, 'n': n_pred})
    res = pd.DataFrame(rows)
    print('\n══ outing-grain screen (odd/even PA halves) ══')
    print(res.round(3).to_string(index=False))
    print('  NOTE stuffRaw/locRaw halves reuse the game value: their rel_r '
          'is 1.0 by construction, ignore it; their pred_half_r is real.')
    return res


def pool_params(t):
    """Per (season, feature) mu/sd from the FULL-outing pool (n >= MIN_N).
    One set of z params serves halves and full outings alike, so weights
    fit on halves apply to full outings without a per-feature rescale."""
    params = {}
    pool = t[t['n'] >= MIN_N]
    for season, g in pool.groupby('season'):
        for f in FEATS:
            vals = g[f].dropna()
            if len(vals) < 100:
                continue
            sd = float(vals.std())
            if sd:
                params[(season, f)] = (float(vals.mean()), sd)
    return params


def apply_shrunk_z(df, feats, kmap, params, n_col, suffix_in=''):
    """<feat>_sz = z(value | season pool params) * n/(n+k); missing -> 0."""
    df = df.reset_index(drop=True)
    for f in feats:
        arr = np.zeros(len(df))
        col = pd.to_numeric(df[f + suffix_in], errors='coerce')
        for season, g in df.groupby('season'):
            p = params.get((season, f))
            if p is None:
                continue
            mu, sd = p
            idx = g.index[col[g.index].notna()]
            z = (col[idx] - mu) / sd
            shrink = g.loc[idx, n_col] / (g.loc[idx, n_col] + kmap[f])
            arr[idx] = (z * shrink).to_numpy()
        df[f + '_sz'] = arr
    return df


def build_split_panel(t, kmap, params):
    hp = t[(t['n'] >= MIN_N) & (t['n_o'] >= MIN_HALF_N)
           & (t['n_e'] >= MIN_HALF_N)].copy()
    frames = []
    for fit_suf, tgt_suf in (('_o', '_e'), ('_e', '_o')):
        d = hp.copy()
        for f in FEATS:
            d[f + '_fit'] = d[f + fit_suf]
        d['_n_half'] = d['n' + fit_suf]
        d['_target'] = d['xrv100' + tgt_suf]
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel = apply_shrunk_z(panel, FEATS, kmap, params, '_n_half',
                           suffix_in='_fit')
    X = panel[[f + '_sz' for f in FEATS]].to_numpy(float)
    y = panel['_target'].to_numpy(float)
    grp = panel['season'].to_numpy()
    ok = np.isfinite(y)
    return X[ok], y[ok], grp[ok]


def oof_r(X, y, grp, cols):
    Xs = X[:, cols]
    pred = np.full(len(y), np.nan)
    fold_rs = []
    for g in np.unique(grp):
        tr, te = grp != g, grp == g
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
        pred[te] = p
        if te.sum() >= 100 and np.std(p) > 0:
            fold_rs.append(float(np.corrcoef(p, y[te])[0, 1]))
    return float(np.corrcoef(pred, y)[0, 1]), fold_rs


def main():
    t = load()
    scr = screen(t)
    kmap = {}
    for _, r in scr.iterrows():
        v = r['stabilize_n']
        kmap[r['feature']] = float(v) if v == v and v > 0 else 200.0
    # game-level process features get the season-search constants scaled
    # to the outing grain? No: they carry no within-outing split, so use
    # the season k (pitch-denominated, grain-free by construction).
    kmap['stuffRaw'] = 42.0
    kmap['locRaw'] = 215.0
    print('\nshrinkage k:', {f: round(k) for f, k in kmap.items()})

    params = pool_params(t)
    X, y, grp = build_split_panel(t, kmap, params)
    print(f'\nsplit panel: {len(y)} obs')

    named = {
        'xrv100 alone': ['xrv100'],
        'rv100 alone': ['rv100'],
        'stuff+loc (process only)': ['stuffRaw', 'locRaw'],
        'season six': ['stuffRaw', 'locRaw', 'kPct', 'izWhiffPct',
                       'xrv100', 'gbPct'],
        'csw swap (csw for k/izw)': ['stuffRaw', 'locRaw', 'cswPct',
                                     'xrv100', 'gbPct'],
    }
    print('\n══ named subsets (OOF by season) ══')
    for name, cs in named.items():
        cols = [FEATS.index(c) for c in cs]
        r, folds = oof_r(X, y, grp, cols)
        se = float(np.std(folds) / np.sqrt(len(folds))) if folds else np.nan
        print(f'{name:28s} r {r:.4f}  se {se:.4f}')

    print('\n══ exhaustive search (sizes 1-5, no rv100) ══', flush=True)
    cand = [f for f in FEATS if f != 'rv100']
    results = []
    for k in range(1, 6):
        for cc in itertools.combinations(range(len(cand)), k):
            cols = [FEATS.index(cand[i]) for i in cc]
            r, folds = oof_r(X, y, grp, cols)
            se = (float(np.std(folds) / np.sqrt(len(folds)))
                  if folds else np.nan)
            results.append({'k': k,
                            'subset': '+'.join(cand[i] for i in cc),
                            'r': r, 'se': se})
        print(f'  size {k} done', flush=True)
    res = pd.DataFrame(results).sort_values('r', ascending=False)
    out_csv = os.path.join(DATA, '_pplus_outing_search.csv')
    res.to_csv(out_csv, index=False)
    print(res.head(12).round(4).to_string(index=False))
    best = res.iloc[0]
    thresh = best['r'] - best['se']
    onese = res[res['r'] >= thresh].sort_values('k').iloc[0]
    print(f'\n1-SE pick: {onese["subset"]} (r {onese["r"]:.4f}, '
          f'k={onese["k"]})')

    # weights for the 1-SE pick, full fit, normalized
    cols = [FEATS.index(c) for c in onese['subset'].split('+')]
    A = np.column_stack([np.ones(len(y)), X[:, cols]])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = beta[1:] / np.abs(beta[1:]).sum()
    print('weights (full fit, normalized): '
          + '  '.join(f'{onese["subset"].split("+")[i]}:{v:+.3f}'
                      for i, v in enumerate(w)))

    # ── context: does the outing composite say anything about the NEXT
    # outing (fixed weights from above, full-outing values) ──
    full = t[t['n'] >= MIN_N].copy()
    full = apply_shrunk_z(full, FEATS, kmap, params, 'n')
    comp = np.zeros(len(full))
    for i, c in enumerate(onese['subset'].split('+')):
        comp += w[i] * full[c + '_sz'].to_numpy()
    full['_comp'] = comp
    full = full.sort_values(['pid', 'season', 'date'])
    full['_next_xrv'] = full.groupby(['pid', 'season'])['xrv100'].shift(-1)
    r_next, n_next = pear(full['_comp'], full['_next_xrv'])
    r_next_x, _ = pear(full['xrv100'], full['_next_xrv'])
    print(f'\ncontext: composite -> next outing xRV/100 r {r_next:.3f} '
          f'(n {n_next}); outing xrv100 alone -> next {r_next_x:.3f}')


if __name__ == '__main__':
    main()
