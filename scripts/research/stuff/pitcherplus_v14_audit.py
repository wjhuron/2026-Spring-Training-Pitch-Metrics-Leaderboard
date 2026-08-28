#!/usr/bin/env python3
"""pitcherplus_v14_audit.py — does the shipped Pitcher+ recipe hold under
Stuff+ v14?

The frozen weights (pipeline/pitcherplus.py) were fit 2026-07-24 against a
LOSO v11 stuff series. v14 shipped 2026-08-23. This audit re-runs the
phase 2 evaluation with a stuff series regenerated under the CURRENT
feature code (scripts/research/stuff/pitcherplus_stuff_loso_v14.py) and
answers three questions on the original two panels (S: split-half within
season, both directions; Y: year-pairs 2021->22 ... 2024->25; target =
future xRV/100):

  1. FROZEN: how does the shipped formula (frozen weights, production
     shrinkage constants) score, v11 stuff vs v14 stuff?
  2. REFIT: does refitting the same six components under v14 move the
     weights or the score?
  3. SEARCH: does any other subset (sizes 1-6 over the 20 phase 1
     survivors) now beat the shipped six by more than 1 SE?

Panel machinery is copied from scripts/archive/pitcherplus_combo.py
verbatim (shrunk z per stabilize_n from _pplus_screen.csv, OOF by
leave-one-group-out OLS) so the numbers are comparable to the stored
2026-07-24 results.

Usage:
  PYTHONHASHSEED=0 python3 scripts/research/stuff/pitcherplus_v14_audit.py \
      --stuff-csv data/_pplus_stuff_loso.csv [--search]
Compare against the v11 baseline by pointing --stuff-csv at
data/_pplus_stuff_loso_v11.csv.
"""
import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pitcherplus_search as ps  # noqa: E402

SURVIVORS = ['stuffRaw', 'locRaw', 'xrvoe', 'fbVelo', 'kPct', 'kbbPct',
             'cswPct', 'putawayPct', 'whiffPct', 'izWhiffPct', 'chasePct',
             'xrv100', 'fipCore', 'xwobacon', 'barrelPct', 'gbPct',
             'weakSideXrv100', 'bestPitchXrv100', 'usageEntropy',
             'pitchesPerG']
MAX_K = 6
MIN_HALF, MIN_FULL = 300, 800
Q_HALF, Q_FULL = 200, 800

# the shipped recipe: research-feature name -> (weight, k) from
# pipeline/pitcherplus.py COMPONENTS (production keys mapped to the
# research table's names)
SHIPPED = (
    ('stuffRaw',   0.20, 42.0),
    ('locRaw',     0.06, 215.0),
    ('kPct',       0.21, 398.0),
    ('izWhiffPct', 0.19, 421.0),
    ('xrv100',     0.23, 1046.0),
    ('gbPct',      0.12, 333.0),
)
SHIPPED_SUBSET = [f for f, _w, _k in SHIPPED]


def load_tables(stuff_csv):
    t = pickle.load(open(ps.TABLES_PKL, 'rb'))
    # merge_external with the chosen stuff CSV
    for path, cols in ((ps.LOC_CSV, ['locRaw']), (stuff_csv, ['stuffRaw'])):
        if os.path.exists(path):
            ext = pd.read_csv(path)
            t = t.merge(ext[['pid', 'season', 'half'] + cols],
                        on=['pid', 'season', 'half'], how='left')
            print(f'merged {os.path.basename(path)}')
        else:
            sys.exit(f'missing {path}')
    feats = [c for c in t.columns if c not in ps.META]
    for c in feats:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t = ps.add_xrvoe(t)
    return t


def add_extras(t):
    """Stage B candidates: age (July 1 of the season, from the MLB API
    birthdate cache) and sieraCore (public SIERA point estimate from the
    table's own rates; the +/- convention on the netGB^2 term follows the
    FanGraphs glossary: minus when netGB/PA is positive)."""
    import json
    bd = json.load(open(os.path.join(ps.DATA, '_pplus_birthdates.json')))
    bdt = {int(k): pd.Timestamp(v) for k, v in bd.items() if v}
    mid = t['season'].map(lambda s: pd.Timestamp(int(s), 7, 1))
    born = t['pid'].map(bdt)
    t['age'] = (mid - born).dt.days / 365.25

    so_pa = t['kPct']
    bb_pa = t['bbPct']
    net_gb = ((t['gbPct'].fillna(0) - t['fbPct'].fillna(0)
               - t['puPct'].fillna(0))
              * t['nbip'] / t['pa']).where(t['pa'] > 0)
    siera = (6.145 - 16.986 * so_pa + 11.434 * bb_pa - 1.858 * net_gb
             + 7.653 * so_pa ** 2
             - np.sign(net_gb) * 6.664 * net_gb ** 2
             + 10.130 * so_pa * net_gb - 5.195 * bb_pa * net_gb)
    t['sieraCore'] = -siera        # orient higher = better like the rest
    return t


def stab_constants():
    scr = pd.read_csv(ps.SCREEN_CSV).set_index('feature')
    k = {}
    for f in SURVIVORS:
        v = scr['stabilize_n'].get(f, np.nan)
        k[f] = float(v) if v == v and v > 0 else 1000.0
    return k


def shrunk_z(rows, feats, kmap, qmin):
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


def frozen_eval(t):
    """Score the SHIPPED formula exactly: production weights AND production
    shrinkage constants (not the screen-derived kmap), then r vs each
    panel's target. No fitting anywhere, so this is the purest 'does the
    shipped number still rank pitchers' check."""
    kmap = {f: k for f, _w, k in SHIPPED}
    feats = SHIPPED_SUBSET
    halves = t[t['half'].isin(['A', 'B'])]
    halves = shrunk_z(halves, feats, kmap, Q_HALF)
    A = halves[(halves['half'] == 'A') & (halves['n'] >= MIN_HALF)]
    B = halves[(halves['half'] == 'B') & (halves['n'] >= MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))

    def comp(df, suf):
        return sum(w * df[f + '_sz' + suf] for f, w, _ in SHIPPED)

    ps_ = np.concatenate([comp(ab, '_a'), comp(ab, '_b')])
    ys_ = np.concatenate([ab['xrv100_b'].to_numpy(float),
                          ab['xrv100_a'].to_numpy(float)])
    ok = np.isfinite(ys_)
    r_s = float(np.corrcoef(ps_[ok], ys_[ok])[0, 1])

    full = t[(t['half'] == 'full') & (t['n'] >= MIN_FULL)]
    full = shrunk_z(full, feats, kmap, Q_FULL)
    pairs = full.merge(full.assign(season=full['season'] - 1),
                       on=['pid', 'season'], suffixes=('', '_n1'))
    py = comp(pairs, '')
    yy = pairs['xrv100_n1'].to_numpy(float)
    ok = np.isfinite(yy)
    r_y = float(np.corrcoef(py[ok], yy[ok])[0, 1])
    return r_s, r_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stuff-csv', default=ps.STUFF_CSV)
    ap.add_argument('--search', action='store_true',
                    help='also run the full best-subset search (slow)')
    ap.add_argument('--extras', action='store_true',
                    help='add stage B candidates (age, sieraCore)')
    args = ap.parse_args()

    t = load_tables(args.stuff_csv)
    if args.extras:
        t = add_extras(t)
        SURVIVORS.extend(['age', 'sieraCore'])
    print(f'\n══ frozen shipped formula ({os.path.basename(args.stuff_csv)}) ══')
    r_s, r_y = frozen_eval(t)
    print(f'FROZEN  r_S {r_s:.4f}  r_Y {r_y:.4f}  '
          f'combined {(r_s + r_y) / 2:.4f}')

    kmap = stab_constants()
    if args.extras:
        kmap['age'] = 0.0                  # a birthdate has no sampling noise
        kmap['sieraCore'] = kmap['kbbPct']  # K-BB body, same stabilization
    (S_X, S_y, S_grp), (Y_X, Y_y, Y_grp) = build_panels(t, kmap)
    print(f'panel S: {len(S_y)} obs, panel Y: {len(Y_y)} obs')

    print('\n══ benchmarks (OOF refit) ══')
    for name, colset in (('kwERA-core (kbbPct)', ['kbbPct']),
                         ('FIP-core', ['fipCore']),
                         ('regressed xRV/100', ['xrv100']),
                         ('SHIPPED subset (refit)', SHIPPED_SUBSET)):
        cols = [SURVIVORS.index(c) for c in colset]
        rs, frs = oof_r(S_X, S_y, S_grp, cols)
        ry, fry = oof_r(Y_X, Y_y, Y_grp, cols)
        folds = frs + fry
        se = float(np.std(folds) / np.sqrt(len(folds))) if folds else np.nan
        print(f'{name:26s} r_S {rs:.4f}  r_Y {ry:.4f}  '
              f'combined {(rs + ry) / 2:.4f}  se {se:.4f}')

    # refit weights for the shipped six (both panels, full fit, normalized)
    cols = [SURVIVORS.index(c) for c in SHIPPED_SUBSET]
    for label, X, y in (('S', S_X, S_y), ('Y', Y_X, Y_y)):
        A = np.column_stack([np.ones(len(y)), X[:, cols]])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        w = beta[1:] / np.abs(beta[1:]).sum()   # target is pitcher-
        # perspective xRV/100 (higher = better), so good components carry
        # positive weight
        terms = '  '.join(f'{f}:{v:+.3f}' for f, v in zip(SHIPPED_SUBSET, w))
        print(f'refit weights (panel {label}, normalized): {terms}')

    if args.search:
        print('\n══ best-subset search 1-6 ══', flush=True)
        results = []
        for k in range(1, MAX_K + 1):
            for cc in itertools.combinations(range(len(SURVIVORS)), k):
                cc = list(cc)
                rs, frs = oof_r(S_X, S_y, S_grp, cc)
                ry, fry = oof_r(Y_X, Y_y, Y_grp, cc)
                folds = frs + fry
                se = (float(np.std(folds) / np.sqrt(len(folds)))
                      if folds else np.nan)
                results.append({'k': k,
                                'subset': '+'.join(SURVIVORS[i] for i in cc),
                                'r_S': rs, 'r_Y': ry,
                                'combined': (rs + ry) / 2, 'se': se})
            print(f'  size {k} done', flush=True)
        res = pd.DataFrame(results).sort_values('combined', ascending=False)
        out_csv = os.path.join(ps.DATA, '_pplus_v14_search.csv')
        res.to_csv(out_csv, index=False)
        print(res.head(12).round(4).to_string(index=False))
        print(f'saved {out_csv}')


if __name__ == '__main__':
    main()
