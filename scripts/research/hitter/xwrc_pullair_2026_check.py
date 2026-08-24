"""xwrc_pullair_2026_check.py — live-season confirmation of the pulled-air
xwOBA term, on the production data the ship would run on.

The 2021-2025 battery (xwrc_pullair_adjust.py) passed DESC LOSO 4/5 with
interior optima c* ~ .12-.25 and a gain that GROWS toward recent seasons.
The speed-term lesson (same day): a term that passes replicates can still
be dead on the live board, so this check is the adoption gate.

Air ball = In Play, LaunchAngle >= 20, not a bunt. Pull = spray beyond
15 degrees to the pull side (pipeline spray_direction 'pull'+'pull_side').
Pool = one row per player (max-PA row, so combined rows represent traded
players), MLB only, 300+ PA. Adjusted per-hitter:

    xwOBA_adj = xwOBA + c * (nPullAir - lgShare * nAir) / pa

Objective: r vs the official wOBA and wRC+ on the live board, swept over
c. A pass = positive gain at the replicate-chosen c with a sane curve.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/xwrc_pullair_2026_check.py
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pipeline.utils import (spray_angle, spray_direction, safe_float,
                            BUNT_BB_TYPES)

PULL_BINS = {'pull', 'pull_side'}
C_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MIN_PA = 300


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sx / sy


def main():
    pitches = pickle.load(open(os.path.join(
        ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    counts = defaultdict(lambda: [0, 0])          # name -> [nAir, nPull]
    for p in pitches:
        if p.get('BTeam') in ('ROC', 'AAA'):
            continue
        if p.get('Description') != 'In Play':
            continue
        if (p.get('BBType') or '') in BUNT_BB_TYPES:
            continue
        la = safe_float(p.get('LaunchAngle'))
        if la is None or la < 20.0:
            continue
        c = counts[p.get('Batter')]
        c[0] += 1
        ang = spray_angle(safe_float(p.get('HC_X')), safe_float(p.get('HC_Y')))
        if spray_direction(ang, p.get('Bats')) in PULL_BINS:
            c[1] += 1
    tot_air = sum(v[0] for v in counts.values())
    tot_pull = sum(v[1] for v in counts.values())
    lgshare = tot_pull / tot_air
    print(f'MLB air balls {tot_air}, pulled share {lgshare:.3f}')

    rows = json.load(open(os.path.join(
        ROOT, 'data', 'hitter_leaderboard_rs.json')))
    best = {}
    for r in rows:
        t = r.get('team') or ''
        if t in ('ROC', 'AAA'):
            continue
        name = r.get('hitter') or r.get('name')
        if name is None:
            continue
        if None in (r.get('xwOBA'), r.get('wOBA'), r.get('wRCplus')):
            continue
        pa = r.get('pa') or 0
        if pa < MIN_PA:
            continue
        prev = best.get(name)
        if prev is None or pa > (prev.get('pa') or 0):
            best[name] = r
    pool = [(r, counts.get(n, [0, 0])) for n, r in best.items()]
    print(f'pool {len(pool)} hitters (300+ PA, one row per player)')
    for c in C_GRID:
        xs, yw, yr = [], [], []
        for r, (nair, npull) in pool:
            adj = r['xwOBA'] + c * (npull - lgshare * nair) / r['pa']
            xs.append(adj)
            yw.append(r['wOBA'])
            yr.append(r['wRCplus'])
        print(f'  c={c:<5} r vs wOBA {pearson(xs, yw):+.4f}   '
              f'r vs wRC+ {pearson(xs, yr):+.4f}')


if __name__ == '__main__':
    main()
