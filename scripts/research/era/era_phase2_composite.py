"""era_phase2_composite.py — phase 2 of the ERA-estimator project: build
and validate the PREDICTIVE composite (skeleton + stuff channel).

Everything is decided by held-out replicates, per the multiseason
standard. Units are pitcher-seasons (NEXT, 5 pairs) or pitcher-halves
(ROS, 6 seasons), features z-scored within their own season/scope,
targets raw ERA. Gate 60 IP equivalent unless stated.

Sections:
  A. Skeleton x stuff-channel grid: which 2-var pair wins held-out.
  B. Fixed-weight sweep w in [0,1] for the winning pair, evaluated
     directly on every replicate (no fitting): the weight curve, its
     interior optimum or flatness, per-replicate agreement.
  C. LOSO OLS coefficient stability (the fitted-weights reference).
  D. Third variable: forward selection over the full candidate set,
     LOSO OLS, adopt only on majority wins + real gain.
  E. Gate sensitivity of the chosen config (40 / 60 / 100 IP).
  F. Cross-horizon: NEXT-chosen composite evaluated on ROS replicates
     vs the ROS-native winner — one metric or two?
  G. ERA-scale calibration: ERA_hat = lgERA_target + b * comp_z, slope
     per fold, and the compression check (predicted vs actual SD).

Usage: python3 scripts/research/era/era_phase2_composite.py
Output: console + data/_era_phase2_results.json
"""
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from era_estimator_screen import (feature_rows, targets_for, pearson,
                                  SEASONS)
from era_combo_preview import zscore_within, ols, CANDS

GATE = 60


def replicate_units(test, gate=GATE, extra_feats=()):
    feats = sorted(set(CANDS) | set(extra_feats))
    reps = []
    if test == 'next':
        for season in SEASONS[:-1]:
            fr = zscore_within(feature_rows(season, 'full'), feats)
            tc = targets_for(season, 'full')
            tn = targets_for(season + 1, 'full')
            units = [(fr[pid], tn[pid]['era']) for pid in fr
                     if pid in tn and pid in tc
                     and tc[pid]['outs'] >= gate * 3
                     and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
    else:
        for season in SEASONS:
            fr = zscore_within(feature_rows(season, 'h1'), feats)
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = gate * 3 // 2
            units = [(fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and pid in t1
                     and t1[pid]['outs'] >= hg and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
    return reps


def loso_eval(reps, feats):
    """LOSO OLS: fit on other replicates, score held-out. Returns
    (per-rep [(label, r)], mean r, per-fold betas)."""
    per, betas = [], []
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
        betas.append(beta)
    mean = sum(r for _, r in per) / len(per)
    return per, mean, betas


def fixed_w_eval(reps, f_skel, f_stuff, w):
    """Composite = (1-w)*z(skel) + w*(-z(stuff)), ERA direction. No
    fitting, so every replicate is a pure evaluation."""
    per = []
    for label, units in reps:
        xs, ys = [], []
        for x, y in units:
            if f_skel in x and f_stuff in x:
                xs.append((1 - w) * x[f_skel] - w * x[f_stuff])
                ys.append(y)
        r = pearson(xs, ys)
        if r is not None:
            per.append((label, r))
    mean = sum(r for _, r in per) / len(per)
    return per, mean


def main():
    out = {}
    reps_next = replicate_units('next')
    reps_ros = replicate_units('ros')

    # ── A. skeleton x stuff grid ──────────────────────────────────────
    print('=== A. Skeleton x stuff channel (NEXT, held-out mean r) ===')
    skels = ['siera_core', 'xfip_core', 'kbb_off', 'kwera_core', 'k_pct',
             'xwoba', 'fip_core', 'csw_pct']
    stuffs = ['stuff_raw', 'pitchingplus_z']
    grid = {}
    for sk in skels:
        for st in stuffs:
            ev = loso_eval(reps_next, [sk, st])
            if ev:
                per, mean, _ = ev
                wins = sum(1 for _, r in per if abs(r) > 0)
                grid[f'{sk}+{st}'] = {'mean': mean,
                                      'per': {l: r for l, r in per}}
                print(f'  {sk:<12} + {st:<15} mean {mean:+.4f}   '
                      + ' '.join(f'{r:+.3f}' for _, r in per))
    out['A_grid'] = grid
    best_pair = max(grid, key=lambda k: abs(grid[k]['mean']))
    print(f'  best: {best_pair}')
    sk_best, st_best = best_pair.split('+')

    # ── B. fixed-weight sweep ────────────────────────────────────────
    print(f'\n=== B. Weight sweep: (1-w)*z({sk_best}) + w*(-z({st_best})) ===')
    sweep = {}
    for wi in range(0, 21):
        w = wi / 20.0
        per, mean = fixed_w_eval(reps_next, sk_best, st_best, w)
        sweep[round(w, 2)] = {'mean': mean, 'per': {l: r for l, r in per}}
        marks = ' '.join(f'{r:+.3f}' for _, r in per)
        print(f'  w={w:.2f}  mean {mean:+.4f}   {marks}')
    out['B_sweep'] = sweep
    best_w = max(sweep, key=lambda w2: sweep[w2]['mean'])
    print(f'  argmax w = {best_w}')

    # per-replicate argmax (does the optimum agree across replicates?)
    print('  per-replicate argmax:')
    for label, _ in reps_next:
        wbest = max(sweep, key=lambda w2: sweep[w2]['per'].get(label, -9))
        print(f'    {label}: w = {wbest}')

    # ── C. LOSO OLS coefficient stability ────────────────────────────
    print(f'\n=== C. LOSO OLS betas for [{sk_best}, {st_best}] ===')
    per, mean, betas = loso_eval(reps_next, [sk_best, st_best])
    out['C_ols'] = {'mean': mean, 'per': {l: r for l, r in per},
                    'betas': betas}
    print(f'  held-out mean r {mean:+.4f}')
    for (label, r), b in zip(per, betas):
        share = abs(b[2]) / (abs(b[1]) + abs(b[2]))
        print(f'  fold {label}: r {r:+.3f}  b_skel {b[1]:+.3f} '
              f'b_stuff {b[2]:+.3f}  stuff share {share:.2f}')

    # ── D. third variable ────────────────────────────────────────────
    print(f'\n=== D. Third variable on [{sk_best}, {st_best}] (NEXT) ===')
    base_per = {l: r for l, r in per}
    thirds = []
    for f3 in CANDS:
        if f3 in (sk_best, st_best):
            continue
        ev = loso_eval(reps_next, [sk_best, st_best, f3])
        if ev is None:
            continue
        per3, mean3, _ = ev
        wins = sum(1 for l, r in per3 if abs(r) > abs(base_per[l]))
        thirds.append((mean3 - mean, f3, mean3, wins, len(per3)))
    thirds.sort(reverse=True)
    out['D_thirds'] = [(f, m, w, n) for _, f, m, w, n in thirds[:10]]
    for g, f3, m3, wins, n in thirds[:8]:
        print(f'  +{f3:<15} mean {m3:+.4f}  gain {g:+.4f}  wins {wins}/{n}')

    # ── E. gate sensitivity of the chosen 2-var config ───────────────
    print('\n=== E. Gate sensitivity (fixed w from B) ===')
    wq = float(best_w)
    out['E_gates'] = {}
    for gate in (40, 60, 100):
        reps_g = replicate_units('next', gate=gate)
        per_g, mean_g = fixed_w_eval(reps_g, sk_best, st_best, wq)
        out['E_gates'][gate] = mean_g
        print(f'  gate {gate:>3} IP: mean r {mean_g:+.4f}  '
              + ' '.join(f'{r:+.3f}' for _, r in per_g))

    # ── F. cross-horizon ─────────────────────────────────────────────
    print('\n=== F. One metric or two? (ROS replicates) ===')
    per_x, mean_x = fixed_w_eval(reps_ros, sk_best, st_best, wq)
    print(f'  NEXT-composite on ROS: mean {mean_x:+.4f}  '
          + ' '.join(f'{r:+.3f}' for _, r in per_x))
    ros_grid = {}
    for sk in ('xwoba', 'siera_core', 'k_pct'):
        for st in stuffs:
            ev = loso_eval(reps_ros, [sk, st])
            if ev:
                p2, m2, _ = ev
                ros_grid[f'{sk}+{st}'] = m2
                print(f'  ROS-native {sk:<10}+{st:<15} held-out mean {m2:+.4f}')
    out['F_cross'] = {'next_on_ros': mean_x, 'ros_native': ros_grid}
    # ROS weight sweep for the best ROS pair
    best_ros = max(ros_grid, key=lambda k: abs(ros_grid[k]))
    rsk, rst = best_ros.split('+')
    print(f'  ROS weight sweep for {best_ros}:')
    ros_sweep = {}
    for wi in range(0, 21, 2):
        w = wi / 20.0
        _, mean_r = fixed_w_eval(reps_ros, rsk, rst, w)
        ros_sweep[round(w, 2)] = mean_r
        print(f'    w={w:.2f}  mean {mean_r:+.4f}')
    out['F_ros_sweep'] = ros_sweep

    # ── G. ERA-scale calibration ─────────────────────────────────────
    print('\n=== G. Calibration: ERA_hat = a + b * comp ===')
    calib = []
    for i, (label, test_units) in enumerate(reps_next):
        train = [u for j, (_, us) in enumerate(reps_next) if j != i
                 for u in us]
        xs = [(1 - wq) * x[sk_best] - wq * x[st_best]
              for x, _ in train if sk_best in x and st_best in x]
        ys = [y for x, y in train if sk_best in x and st_best in x]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((v - mx) ** 2 for v in xs)
        b = sum((v - mx) * (y - my) for v, y in zip(xs, ys)) / sxx
        a = my - b * mx
        # held-out RMSE + compression
        px = [(1 - wq) * x[sk_best] - wq * x[st_best]
              for x, _ in test_units if sk_best in x and st_best in x]
        py = [y for x, y in test_units if sk_best in x and st_best in x]
        preds = [a + b * v for v in px]
        rmse = math.sqrt(sum((p - y) ** 2
                             for p, y in zip(preds, py)) / len(py))
        sd_p = math.sqrt(sum((p - sum(preds) / len(preds)) ** 2
                             for p in preds) / len(preds))
        sd_y = math.sqrt(sum((y - sum(py) / len(py)) ** 2
                             for y in py) / len(py))
        calib.append({'fold': label, 'a': a, 'b': b, 'rmse': rmse,
                      'sd_pred': sd_p, 'sd_act': sd_y})
        print(f'  fold {label}: a {a:+.3f}  b {b:+.3f}  '
              f'held-out RMSE {rmse:.3f}  sd(pred) {sd_p:.2f} '
              f'vs sd(actual) {sd_y:.2f}')
    out['G_calib'] = calib

    path = os.path.join(ROOT, 'data', '_era_phase2_results.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
