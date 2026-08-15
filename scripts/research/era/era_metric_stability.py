"""era_metric_stability.py — year-to-year self-correlation of every
candidate metric (gate 60 IP both seasons), 2021->2022 .. 2025->2026.

Context for the ERA-estimator screen: a metric predicts next-season ERA
through two channels — how much pitcher skill it captures and how stable
that skill is. r(metric_N, metric_N+1) is the stability half; reading it
next to the NEXT-test correlations shows which metrics are stable but
ERA-irrelevant (e.g. arm angle) vs unstable but ERA-loaded (e.g. wOBA).

Usage: python3 scripts/research/era/era_metric_stability.py
Output: data/_era_metric_stability.json + console table.
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from era_estimator_screen import (feature_rows, targets_for, pearson,
                                  SEASONS, MAIN_GATE)


def main():
    acc = defaultdict(list)
    for season in SEASONS[:-1]:
        f1 = feature_rows(season, 'full')
        f2 = feature_rows(season + 1, 'full')
        t1 = targets_for(season, 'full')
        t2 = targets_for(season + 1, 'full')
        pids = [pid for pid in f1 if pid in f2
                and t1.get(pid, {}).get('outs', 0) >= MAIN_GATE * 3
                and t2.get(pid, {}).get('outs', 0) >= MAIN_GATE * 3]
        feats = set()
        for pid in pids:
            feats.update(f1[pid].keys())
        for f in feats:
            xs = [f1[pid][f] for pid in pids
                  if f in f1[pid] and f in f2[pid]]
            ys = [f2[pid][f] for pid in pids
                  if f in f1[pid] and f in f2[pid]]
            r = pearson(xs, ys)
            if r is not None:
                acc[f].append(r)
    out = {f: {'mean': sum(rs) / len(rs), 'n_pairs': len(rs),
               'per_pair': rs} for f, rs in acc.items()}
    path = os.path.join(ROOT, 'data', '_era_metric_stability.json')
    with open(path, 'w') as fh:
        json.dump(out, fh)
    print(f'wrote {path}\n')
    for f, rec in sorted(out.items(), key=lambda kv: -kv[1]['mean']):
        print(f'  {f:<16} self-r {rec["mean"]:+.3f} ({rec["n_pairs"]} pairs)')


if __name__ == '__main__':
    main()
