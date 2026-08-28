#!/usr/bin/env python3
"""pitcherplus_outing_pick.py — settle the outing-grade config.

Follow-up to pitcherplus_outing_grade.py. The exhaustive search left a flat
top region (r .0885-.0888); this script picks WITHIN that region by
replicate agreement and interpretability:

  1. per-season OOF folds for the candidate configs (does each win in most
     seasons it was never fitted on vs the stuff-only floor?)
  2. fitted weights per config + effective contribution at a 90-pitch
     start and a 20-pitch relief outing
  3. face validity: the top and bottom 2025 outings under the chosen
     config, with their lines

Usage:
  PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_outing_pick.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pitcherplus_outing_grade import (load, pool_params, apply_shrunk_z,
                                      build_split_panel, oof_r, FEATS,
                                      MIN_N, DATA)

CONFIGS = {
    'stuff+csw':          ['stuffRaw', 'cswPct'],
    'stuff+loc+csw':      ['stuffRaw', 'locRaw', 'cswPct'],
    'stuff+loc+csw+xrv':  ['stuffRaw', 'locRaw', 'cswPct', 'xrv100'],
    'season six':         ['stuffRaw', 'locRaw', 'kPct', 'izWhiffPct',
                           'xrv100', 'gbPct'],
    'stuff only':         ['stuffRaw'],
}

KMAP = {'stuffRaw': 42.0, 'locRaw': 185.0, 'kPct': 316.0, 'bbPct': 954.0,
        'kbbPct': 1518.0, 'cswPct': 398.0, 'whiffPct': 205.0,
        'izWhiffPct': 307.0, 'chasePct': 397.0, 'gbPct': 416.0,
        'xrv100': 1581.0, 'rv100': 1018.0}


def main():
    t = load()
    params = pool_params(t)
    X, y, grp = build_split_panel(t, KMAP, params)
    print(f'split panel: {len(y)} obs')

    print('\n══ per-season folds ══')
    fold_table = {}
    for name, cs in CONFIGS.items():
        cols = [FEATS.index(c) for c in cs]
        r, folds = oof_r(X, y, grp, cols)
        fold_table[name] = folds
        print(f'{name:20s} pooled {r:.4f}  folds '
              + ' '.join(f'{f:.4f}' for f in folds))
    base = np.array(fold_table['stuff only'])
    for name in CONFIGS:
        if name == 'stuff only':
            continue
        wins = int((np.array(fold_table[name]) > base).sum())
        print(f'{name:20s} beats stuff-only in {wins}/{len(base)} seasons')

    print('\n══ weights + effective contributions ══')
    chosen = 'stuff+loc+csw+xrv'
    for name, cs in CONFIGS.items():
        if name == 'stuff only':
            continue
        cols = [FEATS.index(c) for c in cs]
        A = np.column_stack([np.ones(len(y)), X[:, cols]])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        w = beta[1:] / np.abs(beta[1:]).sum()
        print(f'{name}: '
              + '  '.join(f'{c}:{v:+.3f}' for c, v in zip(cs, w)))
        if name == chosen:
            for n in (90, 20):
                eff = np.array([abs(wi) * n / (n + KMAP[c])
                                for wi, c in zip(w, cs)])
                eff = eff / eff.sum()
                print(f'   effective share at n={n}: '
                      + '  '.join(f'{c}:{v:.2f}' for c, v in zip(cs, eff)))
            chosen_w = {c: float(v) for c, v in zip(cs, w)}

    # ── face validity on 2025 ──
    full = t[t['n'] >= MIN_N].copy()
    full = apply_shrunk_z(full, list(chosen_w), KMAP, params, 'n')
    comp = np.zeros(len(full))
    for c, wv in chosen_w.items():
        comp += wv * full[c + '_sz'].to_numpy()
    # rescale to 100 +/- 10 against the pool
    mu, sd = float(np.mean(comp)), float(np.std(comp))
    full['grade'] = 100 + 10 * (comp - mu) / sd
    g25 = full[full['season'] == 2025]
    cols = ['pid', 'date', 'n', 'pa', 'kPct', 'bbPct', 'cswPct', 'gbPct',
            'xrv100', 'locRaw', 'stuffRaw', 'grade']
    print('\n══ 2025 top 8 outings ══')
    print(g25.nlargest(8, 'grade')[cols].round(3).to_string(index=False))
    print('\n══ 2025 bottom 8 outings ══')
    print(g25.nsmallest(8, 'grade')[cols].round(3).to_string(index=False))
    print('\n══ grade distribution by outing length (2025) ══')
    g25 = g25.copy()
    g25['bin'] = pd.cut(g25['n'], [19, 30, 50, 75, 120])
    print(g25.groupby('bin', observed=True)['grade']
          .agg(['mean', 'std', 'count']).round(2).to_string())
    out = os.path.join(DATA, '_pplus_outing_grades_2025.csv')
    g25[cols].to_csv(out, index=False)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
