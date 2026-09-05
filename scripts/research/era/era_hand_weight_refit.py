"""era_hand_weight_refit.py — refit the hpERA weights with a LHP indicator
as a ninth channel (2026-09-05), on the rebuilt replicates.

Recipe = the shipped one (era_park_weight_refit.loso_with_betas): LOSO OLS
per replicate, weights = mean of the per-fold betas, deciding objective
ROS gate 60, NEXT gate 60 as the transfer check, 30 IP for reference.
The 8-channel control must land near the shipped W_PH (replication check).
The indicator enters RAW (0/1), so its weight reads as the ERA shift for a
LHP; in production it would be centered at the pool LHP share so the
pool-mean forecast is unchanged.
Usage: python3 scripts/research/era/era_hand_weight_refit.py
Output: console + data/_era_hand_refit.json
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, HERE)
import era_weights_final as wf
from era_estimator_screen import pearson
from era_combo_preview import ols
# Inlined from era_park_weight_refit (which loads data/_era_team_outs.json at
# import, another lost scratch file). Same recipe: LOSO fits, fold-mean betas.
FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']
SHIPPED = {'stuff': 0.297, 'loc': 0.136, 'k': 0.088, 'izwh': 0.117,
           'xrv': 0.139, 'gb': 0.162, 'gs_share': 0.277, 'park': 0.168}

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
                preds.append(beta[0] + sum(b * x[f] for b, f in zip(beta[1:], feats)))
                ys.append(y)
        r = pearson(preds, ys)
        if r is None:
            return None
        per.append((label, r))
    mean_beta = [sum(b[j] for b in betas) / len(betas) for j in range(len(betas[0]))]
    return per, sum(r for _, r in per) / len(per), mean_beta
HAND = {sk: {pid: rec.get('hand') for pid, rec in v['pitchers'].items()} for sk, v in wf.TARGETS.items()}

def reps_hand(test, gate):
    """era_hand_residuals.reps_with_pid units (season, pid, feats, y) with a
    raw LHP indicator added, in the (feats, y) shape loso_with_betas takes."""
    import era_hand_residuals as H
    out = []
    for label, units in H.reps_with_pid(test, gate):
        out.append((label, [(dict(x, lhp=1.0 if HAND[sk].get(pid) == 'L' else 0.0), y)
                            for sk, pid, x, y in units]))
    return out

def main():
    out = {}
    for test in ('ros', 'next'):
        for gate in (60, 30):
            reps = reps_hand(test, gate)
            print(f"\n===== {test.upper()} gate {gate} =====")
            res = {}
            for name, feats in (('control8', FEATS), ('hand9', FEATS + ['lhp'])):
                ev = loso_with_betas(reps, feats)
                if ev is None:
                    print(f"  {name}: insufficient coverage"); continue
                per, mean, beta = ev
                w = dict(zip(feats, beta[1:])); res[name] = {'per': dict(per), 'mean': mean, 'int': beta[0], 'weights': w}
                print(f"  {name:9} mean held-out r {mean:+.4f}   " + ' '.join(f"{r:+.3f}" for _, r in per))
                print(f"            fold-mean weights: int {beta[0]:+.3f} | " + ' '.join(f"{f} {v:+.3f}" for f, v in w.items()))
            if 'control8' in res and 'hand9' in res:
                c, h = res['control8'], res['hand9']
                wins = sum(1 for k in h['per'] if h['per'][k] > c['per'][k])
                print(f"  hand9 beats control8 in {wins}/{len(h['per'])} folds, delta mean r {h['mean'] - c['mean']:+.4f}")
                print("  weight shift control8 -> hand9: " + ' '.join(f"{f} {h['weights'][f] - c['weights'][f]:+.3f}" for f in FEATS))
            out[f'{test}_{gate}'] = res
    c = out['ros_60'].get('control8')
    if c:
        print("\nreplication check, ROS gate 60 control8 vs shipped W_PH:")
        for f in FEATS:
            print(f"  {f:9} control {c['weights'][f]:+.3f}   shipped {SHIPPED[f]:+.3f}   d {c['weights'][f] - SHIPPED[f]:+.3f}")
    # LHP share in the ROS-60 pools, for the centering note
    shares = []
    for label, units in reps_hand('ros', 60):
        v = [x['lhp'] for x, _ in units]; shares.append(sum(v) / len(v))
    print(f"\nLHP share of the ROS-60 units by replicate: {' '.join(f'{s:.3f}' for s in shares)} (mean {sum(shares)/len(shares):.3f})")
    json.dump(out, open(os.path.join(ROOT, 'data', '_era_hand_refit.json'), 'w'), indent=1)
    print('wrote data/_era_hand_refit.json')

if __name__ == '__main__':
    main()
