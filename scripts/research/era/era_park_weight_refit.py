"""era_park_weight_refit.py — refit the hpERA park channel on innings-
weighted stint exposure.

Motivation (pipeline/eraplus.py combined_park_map docstring): production
resolves a traded pitcher's park as the IP-weighted mean of his own MLB
stints, but W_PH['park'] = 0.168 was fit against the research harness's
park_exposure. On inspection the harness input was worse than the
docstring believed: the bulk stats endpoint returns ONE season-combined
row per pitcher with only the FINAL club attached, so every traded
pitcher's exposure was his last park, full stop (Scherzer 2023 scored as
pure TEX despite 107.2 IP with NYM). ~11-14% of each season's pool is
multi-team.

This battery re-runs the production 8-channel fold-mean OLS fit
(era_weights_final harness, shrunk inputs, LOSO replicates) under three
park exposures:
  A control   final-club exposure (reproduces the shipped fit)
  B weighted  IP-weighted stint exposure (data/_era_team_outs.json,
              scope-matched: h1 stints for the ROS fit)
  C unweighted  plain mean over stint clubs (the docstring's belief),
              to separate "any stint data" from "the weighting"

Adoption bar: B beats A on held-out LOSO r in most folds it was never
fitted on, on the deciding objective (ROS gate 60, the shipped fit's
objective), with NEXT as the transfer check.

Usage: PYTHONHASHSEED=0 python3 scripts/research/era/era_park_weight_refit.py
Output: console + data/_era_park_refit.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import era_weights_final as wf
from era_estimator_screen import PF, pearson
from era_combo_preview import ols

TEAM_OUTS = json.load(open(os.path.join(ROOT, 'data', '_era_team_outs.json')))
FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']
# shipped production weights (pipeline/eraplus.py W_PH), for the control
# replication check. Harness betas are in ERA-per-z units on the same
# z-convention, so the control fold-mean should land near these.
SHIPPED = {'stuff': 0.297, 'loc': 0.136, 'k': 0.088, 'izwh': 0.117,
           'xrv': 0.139, 'gb': 0.162, 'gs_share': 0.277, 'park': 0.168}

_orig_raw = wf.raw_features


def _pf_half(season, tid):
    return PF.get(str(season), {}).get(str(tid), 100.0) / 100.0 / 2.0 + 0.5


def _exposure(season, pid, scope, weighted):
    rec = TEAM_OUTS.get(str(season), {}).get(str(pid))
    if not rec:
        return None
    stints = rec['full' if scope == 'full' else 'h1']
    stints = {t: o for t, o in stints.items() if o > 0}
    if not stints:
        return None
    if weighted:
        tot = sum(stints.values())
        return sum(_pf_half(season, t) * o for t, o in stints.items()) / tot
    vals = [_pf_half(season, t) for t in stints]
    return sum(vals) / len(vals)


_MODE = {'v': 'control'}


def _patched_raw(season, scope):
    out = _orig_raw(season, scope)
    if _MODE['v'] == 'control':
        return out
    weighted = _MODE['v'] == 'weighted'
    for pid, r in out.items():
        e = _exposure(season, pid, scope, weighted)
        if e is not None:
            r['park'] = e
        # missing stint record: keep the harness final-club value
    return out


wf.raw_features = _patched_raw
# shrunk_features reads module-global raw_features, so the patch takes.


def loso_with_betas(reps, feats):
    per, betas = [], []
    for i, (label, test_units) in enumerate(reps):
        train = [u for j, (_, us) in enumerate(reps) if j != i for u in us]
        beta = ols(train, feats)
        if beta is None:
            return None
        betas.append(beta)
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
    mean_beta = [sum(b[j] for b in betas) / len(betas)
                 for j in range(len(betas[0]))]
    return per, sum(r for _, r in per) / len(per), mean_beta


def run(mode, test, gate):
    _MODE['v'] = mode
    reps = wf.build_reps(test, gate)
    ev = loso_with_betas(reps, FEATS)
    if ev is None:
        print(f'  {mode:<10} {test} g{gate}: insufficient coverage')
        return None
    per, mean, beta = ev
    w = {f: round(b, 3) for f, b in zip(FEATS, beta[1:])}
    print(f'  {mode:<10} {test} g{gate}  mean r {mean:+.4f}   '
          + ' '.join(f'{r:+.3f}' for _, r in per))
    print(f'             fold-mean weights: {w}')
    return {'per': dict(per), 'mean': mean, 'weights': w}


def main():
    out = {}
    for gate in (60, 30):
        for test in ('ros', 'next'):
            print(f'\n===== {test.upper()} gate {gate} =====')
            for mode in ('control', 'weighted', 'unweighted'):
                res = run(mode, test, gate)
                if res:
                    out[f'{mode}_{test}_{gate}'] = res
            c = out.get(f'control_{test}_{gate}')
            b = out.get(f'weighted_{test}_{gate}')
            if c and b:
                wins = sum(1 for k in b['per']
                           if b['per'][k] > c['per'].get(k, -9))
                print(f'  weighted beats control in {wins}/{len(b["per"])} '
                      f'folds, delta mean {b["mean"] - c["mean"]:+.4f}')
    ck = out.get('control_ros_60')
    if ck:
        print('\ncontrol vs SHIPPED W_PH (replication check):')
        for f in FEATS:
            print(f'  {f:<9} control {ck["weights"][f]:+.3f}   '
                  f'shipped {SHIPPED[f]:+.3f}')
    tmp = os.path.join(ROOT, 'data', '_era_park_refit.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_era_park_refit.json'))
    print('\nwrote data/_era_park_refit.json')


if __name__ == '__main__':
    main()
