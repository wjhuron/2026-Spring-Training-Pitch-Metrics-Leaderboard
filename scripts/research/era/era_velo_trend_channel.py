"""era_velo_trend_channel.py — does a year-over-year velocity-change
channel improve hpERA?

Public prior (THT / Driveline): an early-season fastball velocity drop of
1.5+ mph preceded about +0.26 ERA of decline, and velocity loss predicts
attrition. hpERA carries velocity only through Stuff+, which grades the
pitch as thrown; a pitcher can lose 1.5 mph and still grade well if the
shape holds, so the TREND is a candidate orthogonal channel.

Channel: dvelo = current FF velocity minus PRIOR full-season FF velocity
(era battery ff_velo; 'current' = h1 for the ROS test, full for NEXT).
Pitchers without a prior season impute dvelo = 0 (no-change; keeps
rookies in the pool and is the honest production behavior). z within the
30+ IP pool over REAL values only; imputed rows take z = 0.

Comparison: the production 8-channel fold-mean OLS fit (era_weights_final
harness) with and without dvelo, LOSO across replicate seasons. 2021 has
no prior season in the battery, so BOTH arms run on 2022+ replicates to
keep the comparison apples-to-apples. Adoption bar: the 9-channel arm
beats the 8-channel arm on held-out r in most folds.

Usage: PYTHONHASHSEED=0 python3 scripts/research/era/era_velo_trend_channel.py
Output: console + data/_era_velo_trend.json
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import era_weights_final as wf
from era_estimator_screen import pearson, BATTERY, targets_for
from era_combo_preview import ols

BASE_FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']
SEASONS = [2022, 2023, 2024, 2025, 2026]      # 2021 has no prior season


def dvelo_map(season, scope):
    """{pid: (dvelo, real)} — ff_velo now minus prior full ff_velo."""
    cur = BATTERY.get(str(season), {})
    prior = BATTERY.get(str(season - 1), {})
    out = {}
    for pid, rec in cur.items():
        v_now = (rec.get(scope) or {}).get('ff_velo')
        v_pri = (prior.get(pid, {}).get('full') or {}).get('ff_velo')
        if v_now is not None and v_pri is not None:
            out[pid] = (v_now - v_pri, True)
        else:
            out[pid] = (0.0, False)
    return out


def features(season, scope):
    z = wf.shrunk_features(season, scope)
    dv = dvelo_map(season, scope)
    real = [dv[p][0] for p in z if p in dv and dv[p][1]]
    n_real = len(real)
    if n_real >= 30:
        m = sum(real) / n_real
        s = math.sqrt(sum((x - m) ** 2 for x in real) / n_real)
    else:
        m, s = 0.0, 0.0
    for pid, f in z.items():
        v, is_real = dv.get(pid, (0.0, False))
        f['dvelo'] = ((v - m) / s) if (is_real and s > 0) else 0.0
    return z, (n_real / len(z) if z else 0.0)


def build_reps(test, gate):
    reps, cov = [], []
    if test == 'next':
        for season in SEASONS:
            if season + 1 not in SEASONS and season + 1 != 2027:
                pass
            if str(season + 1) not in BATTERY:
                continue
            fr, c = features(season, 'full')
            tc = targets_for(season, 'full')
            tn = targets_for(season + 1, 'full')
            units = [(fr[pid], tn[pid]['era']) for pid in fr
                     if pid in tn and pid in tc
                     and tc[pid]['outs'] >= gate * 3
                     and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
            cov.append(c)
    else:                                       # ros
        for season in SEASONS:
            fr, c = features(season, 'h1')
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = max(gate * 3 // 2, 45)
            units = [(fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and pid in t1
                     and t1[pid]['outs'] >= hg
                     and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
            cov.append(c)
    return reps, cov


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


def main():
    out = {}
    for gate in (60, 30):
        for test in ('ros', 'next'):
            reps, cov = build_reps(test, gate)
            print(f'\n===== {test.upper()} gate {gate} '
                  f'(real-dvelo coverage {sum(cov) / len(cov):.1%}) =====')
            res = {}
            for tag, feats in (('control8', BASE_FEATS),
                               ('with_dvelo', BASE_FEATS + ['dvelo'])):
                ev = loso(reps, feats)
                if ev is None:
                    print(f'  {tag}: insufficient coverage')
                    continue
                per, mean = ev
                res[tag] = {'per': dict(per), 'mean': mean}
                print(f'  {tag:<11} mean r {mean:+.4f}   '
                      + ' '.join(f'{r:+.3f}' for _, r in per))
            if len(res) == 2:
                c, t = res['control8'], res['with_dvelo']
                wins = sum(1 for k in t['per']
                           if t['per'][k] > c['per'].get(k, -9))
                print(f'  dvelo wins {wins}/{len(t["per"])} folds, '
                      f'delta mean {t["mean"] - c["mean"]:+.4f}')
            out[f'{test}_{gate}'] = res
    tmp = os.path.join(ROOT, 'data', '_era_velo_trend.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_era_velo_trend.json'))
    print('\nwrote data/_era_velo_trend.json')


if __name__ == '__main__':
    main()
