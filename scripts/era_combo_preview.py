"""era_combo_preview.py — does a SECOND variable reliably add to the best
univariate metric? Phase-2 preview for the ERA-estimator screen; the full
composite build is a separate project.

Protocol (replicates decide, never the fitting sample):
  * Unit = pitcher-season (or pitcher-half for ROS), features z-scored
    within their own season/scope, targets left raw.
  * For each test (DESC 6 replicates / NEXT 5 pairs / ROS 6 replicates),
    hold out one replicate; fit OLS on the others pooled; score the
    held-out replicate with the frozen coefficients; record r.
  * A pair (b1, f2) earns a row only if its held-out mean r beats the
    same-protocol univariate r of b1, and wins in the majority of
    replicates.

Gate: 60 IP equivalent (matching the screen's main tables).
Usage: python3 scripts/era_combo_preview.py
"""
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from era_estimator_screen import (feature_rows, targets_for, pearson,
                                  SEASONS, MAIN_GATE)

BASES = {
    'desc': ['xwoba', 'fip_core', 'rv100'],
    'next': ['siera_core', 'xfip_core', 'k_pct'],
    'ros': ['xwoba', 'siera_core', 'k_pct'],
}
CANDS = ['k_pct', 'bb_pct', 'kbb_off', 'csw_pct', 'whiff_pct', 'chase_pct',
         'gb_pct', 'ld_pct', 'brl_pct', 'ev', 'hh_pct', 'xwobacon',
         'xwoba', 'fip_core', 'xfip_core', 'siera_core', 'hr_fb',
         'ff_velo', 'velo_max', 'ff_nvaa', 'stuff_raw', 'loc_raw',
         'cmd_miss', 'pitchingplus_z', 'zcon_pct', 'babip_off', 'rv100',
         'fps_pct', 'ext']


def zscore_within(rows, feats):
    vals = defaultdict(list)
    for r in rows.values():
        for f in feats:
            if f in r:
                vals[f].append(r[f])
    mu_sd = {}
    for f, v in vals.items():
        if len(v) < 30:
            continue
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        if s > 0:
            mu_sd[f] = (m, s)
    out = {}
    for pid, r in rows.items():
        out[pid] = {f: (r[f] - mu_sd[f][0]) / mu_sd[f][1]
                    for f in feats if f in r and f in mu_sd}
    return out


def replicate_units(test):
    """-> list of (label, [(x_dict, y), ...]) one entry per replicate."""
    feats = sorted(set(sum(BASES.values(), []) + CANDS))
    reps = []
    if test == 'desc':
        for season in SEASONS:
            fr = zscore_within(feature_rows(season, 'full'), feats)
            targ = targets_for(season, 'full')
            units = [(fr[pid], targ[pid]['era']) for pid in fr
                     if pid in targ and targ[pid]['outs'] >= MAIN_GATE * 3]
            reps.append((str(season), units))
    elif test == 'next':
        for season in SEASONS[:-1]:
            fr = zscore_within(feature_rows(season, 'full'), feats)
            targ_c = targets_for(season, 'full')
            targ_n = targets_for(season + 1, 'full')
            units = [(fr[pid], targ_n[pid]['era']) for pid in fr
                     if pid in targ_n and pid in targ_c
                     and targ_c[pid]['outs'] >= MAIN_GATE * 3
                     and targ_n[pid]['outs'] >= MAIN_GATE * 3]
            reps.append((f'{season}->{season + 1}', units))
    else:
        for season in SEASONS:
            fr = zscore_within(feature_rows(season, 'h1'), feats)
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = MAIN_GATE * 3 // 2
            units = [(fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and pid in t1
                     and t1[pid]['outs'] >= hg and t2[pid]['outs'] >= hg]
            reps.append((f'{season} h1->h2', units))
    return reps


def ols(units, feats):
    """OLS coefficients (with intercept) on units restricted to complete
    cases. Pure-python normal equations (tiny k)."""
    rows = [(x, y) for x, y in units if all(f in x for f in feats)]
    n = len(rows)
    if n < 50:
        return None
    k = len(feats)
    X = [[1.0] + [x[f] for f in feats] for x, _ in rows]
    Y = [y for _, y in rows]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n))
            for b in range(k + 1)] for a in range(k + 1)]
    XtY = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(k + 1)]
    # gaussian elimination
    for col in range(k + 1):
        piv = max(range(col, k + 1), key=lambda r2: abs(XtX[r2][col]))
        if abs(XtX[piv][col]) < 1e-9:
            return None
        XtX[col], XtX[piv] = XtX[piv], XtX[col]
        XtY[col], XtY[piv] = XtY[piv], XtY[col]
        d = XtX[col][col]
        XtX[col] = [v / d for v in XtX[col]]
        XtY[col] /= d
        for r2 in range(k + 1):
            if r2 != col and XtX[r2][col]:
                f2 = XtX[r2][col]
                XtX[r2] = [a - f2 * b for a, b in zip(XtX[r2], XtX[col])]
                XtY[r2] -= f2 * XtY[col]
    return XtY


def loso_r(reps, feats):
    """Frozen-coefficient held-out r per replicate."""
    rs = []
    for i, (label, test_units) in enumerate(reps):
        train = [u for j, (_, us) in enumerate(reps) if j != i for u in us]
        beta = ols(train, feats)
        if beta is None:
            return None
        preds, ys = [], []
        for x, y in test_units:
            if all(f in x for f in feats):
                preds.append(beta[0] + sum(b * x[f]
                                           for b, f in zip(beta[1:], feats)))
                ys.append(y)
        r = pearson(preds, ys)
        if r is None:
            return None
        rs.append((label, r))
    return rs


def main():
    for test in ('desc', 'next', 'ros'):
        reps = replicate_units(test)
        print(f'\n=== {test.upper()} — held-out replicate r, ERA target, '
              f'{MAIN_GATE} IP gate ===')
        for base in BASES[test]:
            solo = loso_r(reps, [base])
            if solo is None:
                print(f'  {base}: insufficient coverage')
                continue
            solo_mean = sum(r for _, r in solo) / len(solo)
            print(f'  base {base:<14} mean held-out r {solo_mean:+.3f}')
            gains = []
            for f2 in CANDS:
                if f2 == base:
                    continue
                pair = loso_r(reps, [base, f2])
                if pair is None:
                    continue
                pair_mean = sum(r for _, r in pair) / len(pair)
                wins = sum(1 for (_, rp), (_, rs2) in zip(pair, solo)
                           if abs(rp) > abs(rs2))
                gains.append((abs(pair_mean) - abs(solo_mean), f2,
                              pair_mean, wins, len(pair)))
            gains.sort(reverse=True)
            for g, f2, pm, wins, nrep in gains[:5]:
                print(f'    +{f2:<14} mean r {pm:+.3f}  gain {g:+.3f}  '
                      f'wins {wins}/{nrep}')


if __name__ == '__main__':
    main()
