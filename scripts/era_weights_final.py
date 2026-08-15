"""era_weights_final.py — the accuracy-first pass on dhERA/phERA:
shrunk inputs, freed component weights, the Pitcher+ component set tested
against future ERA, and the untested channels (age, role, park).

Inputs are SHRUNK at the constants measured by era_shrinkage_sweep.py
(interior optima, per-season replicates):
    xwOBA n0=250 PA · K% n0=90 PA · BB% n0=180 PA · xwOBAcon n0=700 BIP
    Stuff+ n0=15 · Pitching+ n0=20 · Loc+ n0=170 (pitches)
Shrink target = league (role target won 6/6 but by ~0.3% RMSE; excluded
on the documented Pitcher+ no-role philosophy).

All tests: LOSO OLS or fixed-weight sweeps on per-season replicates,
production z-convention (30+ IP pool), evaluated at gates 30 and 60.

  D-series (dhERA, DESC objective):
    D1 z(xwOBA_sh) alone           (current dhERA)
    D2 free [K%_sh, BB%_sh, xwOBAcon_sh]  (unpackage the wOBA weights)
    D3 D2 + xwOBA_sh               (packaged + free residuals)
  P-series (phERA, NEXT + ROS objectives):
    P1 2-channel [xwOBA_sh, Pitching+_sh] re-swept w (does shrinkage move w?)
    P2 free [xwOBA_sh, Stuff+_sh, Loc+_sh]   (free the 0.8/0.2)
    P3 Pitcher+ set free [Stuff+_sh, Loc+_sh, K%_sh, izWhiff, xRV/100, GB%]
    P4 P3 + BB%_sh + xwOBAcon_sh   (kitchen sink ceiling)
  C-series (completeness, added to the P-winner and to D1):
    age (as of Jun 30), starter share (gs/g), park exposure (home PF)

Usage: python3 scripts/era_weights_final.py
Output: console + data/_era_weights_final.json
"""
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from era_estimator_screen import pearson, SEASONS, BATTERY, CMDLOC, STUFF
from era_estimator_screen import targets_for, park_exposure
from era_combo_preview import ols

TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
XRV = json.load(open(os.path.join(ROOT, 'data', '_era_xrv100.json')))
AGES = json.load(open(os.path.join(ROOT, 'data', '_era_ages.json')))

N0 = {'xw': 250.0, 'k': 90.0, 'bb': 180.0, 'xwc': 700.0,
      'stuff': 15.0, 'pplus': 20.0, 'loc': 170.0}
POOL_OUTS = {'full': 90, 'h1': 45}


def season_age(pid, season):
    bd = AGES.get(str(pid))
    if not bd:
        return None
    y, m, d = map(int, bd.split('-'))
    return (season - y) + ((6 - m) * 30 + (30 - d)) / 365.0


def raw_features(season, scope):
    """Unshrunk numerators/denominators + rates per pitcher."""
    sk = str(season)
    out = {}
    for pid, brec in BATTERY.get(sk, {}).items():
        m = brec[scope]
        kc = m['k_counts']
        line = TARGETS[sk]['pitchers'].get(pid)
        if line is None:
            continue
        lrec = line if scope == 'full' else line[scope]
        if not lrec or lrec['outs'] <= 0 or lrec['bf'] <= 0:
            continue
        r = {}
        r['outs'] = lrec['outs']
        # xwOBA numerator reconstruction: den ~ pa - ibb - sh - ci
        den = m['pa'] - kc['ibb'] - kc['sh'] - kc['ci']
        if m.get('xwoba') is not None and den > 0:
            r['xw_num'] = m['xwoba'] * den
            r['xw_den'] = den
        r['k_n'], r['bb_n'], r['bf'] = lrec['so'], lrec['bb'], lrec['bf']
        if m.get('xwobacon') is not None and m['bip'] > 0:
            r['xwc_num'] = m['xwobacon'] * m['bip']
            r['xwc_den'] = m['bip']
        if m.get('gb_pct') is not None:
            r['gb_pct'] = m['gb_pct']
        if m.get('zcon_pct') is not None:
            r['izwhiff'] = 1.0 - m['zcon_pct']
        r['pitches'] = m['pitches']
        # internals
        suf = scope
        srec = STUFF.get(sk, {}).get(pid, {})
        if f'stuff_{suf}' in srec:
            r['stuff'], r['stuff_n'] = srec[f'stuff_{suf}'], srec[f'n_{suf}']
        elif m.get('stuff_plus') is not None:
            r['stuff'], r['stuff_n'] = m['stuff_plus'], m['pitches']
        crec = CMDLOC.get(sk, {}).get(pid, {})
        if f'loc_{suf}' in crec:
            r['loc'] = -crec[f'loc_{suf}']
            r['loc_n'] = crec[f'loc_n_{suf}']
        elif m.get('loc_plus') is not None:
            r['loc'], r['loc_n'] = m['loc_plus'], m['pitches']
        xrec = XRV.get(sk, {}).get(pid, {})
        if scope in xrec:
            r['xrv100'] = -xrec[scope]     # pitcher-positive -> ERA dir
            r['xrv_n'] = xrec[f'n_{scope}']
        r['age'] = season_age(pid, season)
        r['gs_share'] = (line['gs'] / line['g']) if line.get('g') else 0.0
        r['park'] = park_exposure(season, line['teams'])
        out[pid] = r
    return out


def shrunk_features(season, scope):
    """Apply measured shrinkage; z-score within the 30+ IP pool."""
    raw = raw_features(season, scope)
    pool = {pid: r for pid, r in raw.items()
            if r['outs'] >= POOL_OUTS[scope]}

    def lg(num_k, den_k):
        n = sum(r[num_k] for r in pool.values() if num_k in r)
        d = sum(r[den_k] for r in pool.values() if num_k in r)
        return n / d if d else 0.0

    def lg_mean(val_k, n_k):
        s = sum(r[val_k] * r[n_k] for r in pool.values() if val_k in r)
        d = sum(r[n_k] for r in pool.values() if val_k in r)
        return s / d if d else 0.0

    lg_xw = lg('xw_num', 'xw_den')
    lg_k = sum(r['k_n'] for r in pool.values()) / \
        sum(r['bf'] for r in pool.values())
    lg_bb = sum(r['bb_n'] for r in pool.values()) / \
        sum(r['bf'] for r in pool.values())
    lg_xwc = lg('xwc_num', 'xwc_den')
    lg_st = lg_mean('stuff', 'stuff_n')
    lg_lo = lg_mean('loc', 'loc_n')
    lg_xrv = lg_mean('xrv100', 'xrv_n')

    feats = {}
    for pid, r in pool.items():
        f = {}
        if 'xw_num' in r:
            f['xw'] = (r['xw_num'] + N0['xw'] * lg_xw) / \
                (r['xw_den'] + N0['xw'])
        f['k'] = -(r['k_n'] + N0['k'] * lg_k) / (r['bf'] + N0['k'])
        f['bb'] = (r['bb_n'] + N0['bb'] * lg_bb) / (r['bf'] + N0['bb'])
        if 'xwc_num' in r:
            f['xwc'] = (r['xwc_num'] + N0['xwc'] * lg_xwc) / \
                (r['xwc_den'] + N0['xwc'])
        if 'stuff' in r:
            f['stuff'] = -(r['stuff'] * r['stuff_n']
                           + N0['stuff'] * lg_st) / \
                (r['stuff_n'] + N0['stuff'])
        if 'loc' in r:
            f['loc'] = -(r['loc'] * r['loc_n'] + N0['loc'] * lg_lo) / \
                (r['loc_n'] + N0['loc'])
        if 'xrv100' in r:
            f['xrv'] = (r['xrv100'] * r['xrv_n']
                        + N0['pplus'] * lg_xrv) / \
                (r['xrv_n'] + N0['pplus'])
        if 'gb_pct' in r:
            f['gb'] = -r['gb_pct']
        if 'izwhiff' in r:
            f['izwh'] = -r['izwhiff']
        if r['age'] is not None:
            f['age'] = r['age']
        f['gs_share'] = r['gs_share']
        f['park'] = r['park']
        feats[pid] = f
    # z within pool
    keys = sorted({k for f in feats.values() for k in f})
    mu_sd = {}
    for k in keys:
        v = [f[k] for f in feats.values() if k in f]
        if len(v) < 30:
            continue
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        if s > 0:
            mu_sd[k] = (m, s)
    z = {pid: {k: (f[k] - mu_sd[k][0]) / mu_sd[k][1]
               for k in f if k in mu_sd} for pid, f in feats.items()}
    # Pitching+ is a Z-SPACE composite (0.8/0.2 is a ratio of z-scores);
    # both inputs are already shrunk and ERA-direction here
    for f in z.values():
        if 'stuff' in f and 'loc' in f:
            f['pplus'] = 0.8 * f['stuff'] + 0.2 * f['loc']
    return z


def build_reps(test, gate):
    reps = []
    if test == 'desc':
        for season in SEASONS:
            fr = shrunk_features(season, 'full')
            targ = targets_for(season, 'full')
            units = [(fr[pid], targ[pid]['era']) for pid in fr
                     if pid in targ and targ[pid]['outs'] >= gate * 3]
            reps.append((str(season), units))
    elif test == 'next':
        for season in SEASONS[:-1]:
            fr = shrunk_features(season, 'full')
            tc = targets_for(season, 'full')
            tn = targets_for(season + 1, 'full')
            units = [(fr[pid], tn[pid]['era']) for pid in fr
                     if pid in tn and tc[pid]['outs'] >= gate * 3
                     and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
    else:
        for season in SEASONS:
            fr = shrunk_features(season, 'h1')
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = max(gate * 3 // 2, 45)
            units = [(fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and t1[pid]['outs'] >= hg
                     and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
    return reps


def loso(reps, feats):
    per = []
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
        per.append((label, r))
    return per, sum(r for _, r in per) / len(per)


def report(tag, reps, feats, base_mean=None):
    ev = loso(reps, feats)
    if ev is None:
        print(f'  {tag:<34} insufficient coverage')
        return None
    per, mean = ev
    d = '' if base_mean is None else f'  d {mean - base_mean:+.4f}'
    print(f'  {tag:<34} {mean:+.4f}{d}   '
          + ' '.join(f'{r:+.3f}' for _, r in per))
    return mean


def main():
    out = {}
    for gate in (60, 30):
        print(f'\n########## GATE {gate} IP ##########')
        print('=== D-series (DESC) ===')
        reps = build_reps('desc', gate)
        d1 = report('D1 xwOBA_sh', reps, ['xw'])
        report('D2 free K/BB/xwOBAcon', reps, ['k', 'bb', 'xwc'], d1)
        report('D3 D2 + xwOBA_sh', reps, ['k', 'bb', 'xwc', 'xw'], d1)
        report('C  D1 + gs_share', reps, ['xw', 'gs_share'], d1)
        report('C  D1 + park', reps, ['xw', 'park'], d1)

        for test in ('next', 'ros'):
            print(f'=== P-series ({test.upper()}) ===')
            reps = build_reps(test, gate)
            p1 = report('P1 xwOBA_sh + Pitching+_sh', reps,
                        ['xw', 'pplus'])
            report('P2 free xwOBA/Stuff/Loc', reps,
                   ['xw', 'stuff', 'loc'], p1)
            p3 = report('P3 Pitcher+ set', reps,
                        ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb'], p1)
            report('P4 P3 + BB + xwOBAcon', reps,
                   ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'bb',
                    'xwc'], p1)
            report('C  P1 + age', reps, ['xw', 'pplus', 'age'], p1)
            report('C  P1 + gs_share', reps, ['xw', 'pplus', 'gs_share'],
                   p1)
            report('C  P1 + park', reps, ['xw', 'pplus', 'park'], p1)
            out[f'{test}_{gate}'] = {'p1': p1, 'p3': p3}

    with open(os.path.join(ROOT, 'data', '_era_weights_final.json'),
              'w') as f:
        json.dump(out, f)
    print('\nwrote data/_era_weights_final.json')


if __name__ == '__main__':
    main()
