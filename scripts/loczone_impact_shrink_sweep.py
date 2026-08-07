#!/usr/bin/env python3
"""Sweep the cell-grade shrinkage constant k for zone x count Loc+ impacts.

Impact per cell i of a (pitcher, pitch-group) unit:
    impact_i(k) = s_i * g'_i(k) - ls_i * lg_i          (Loc+ points / 100 scale)
    g'_i(k)     = (n_i * g_i + k * lg_i) / (n_i + k)
with s_i the pitcher's joint share of the unit's pitches in cell i, g_i his
mean Loc+ atom there, and ls_i / lg_i the league share and grade for that
pitch group. Cells he never visits carry impact -ls_i * lg_i (a pure mix
effect), which keeps the 15-cell vector complete.

Objective: split-half PREDICTIVE validity, not plain stability (which is
gamed by k -> inf collapsing to the mix term). Each qualifying unit's pitches
are split even/odd by pitch index; we correlate the SHRUNK impact vector from
half A with the RAW (k=0) impact vector from half B, both directions, and
average across units. The k maximizing that out-of-half prediction is the one
that best separates signal from single-pitch noise.

Usage: python3 scripts/loczone_impact_shrink_sweep.py
"""
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_sdplus import classify_zone
from pipeline_locplus import group_of_code

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
STATES = ['ahead', 'even', 'behind']
CELLS = [(z, s) for z in ZONES for s in STATES]
MIN_UNIT_PITCHES = 250
K_GRID = [0, 2, 5, 8, 12, 18, 25, 35, 50, 75, 110]


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def count_state(p):
    try:
        b, s = (int(x) for x in str(p.get('Count', '')).split('-'))
    except ValueError:
        return None
    return 'ahead' if s > b else ('behind' if b > s else 'even')


def cell_agg(pitches):
    acc = defaultdict(lambda: [0.0, 0])
    for v, z, st in pitches:
        acc[(z, st)][0] += v
        acc[(z, st)][1] += 1
    return acc


def impact_vector(acc, n_total, lg, k):
    out = []
    for cell in CELLS:
        ls, lgg = lg[cell]
        s_sum, n = acc.get(cell, [0.0, 0])
        share = n / n_total
        if n > 0:
            g = s_sum / n
            g_shr = (n * g + k * lgg) / (n + k)
        else:
            g_shr = lgg
        out.append((share * g_shr - ls * lgg))
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else None


def main():
    print('Loading cache ...')
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        allp = pickle.load(f)

    units = defaultdict(list)      # (pitcher, team, group) -> [(atom, zone, state)]
    lg_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    lg_n = defaultdict(int)
    for p in allp:
        if p.get('_source') != 'MLB':
            continue
        v = sf(p.get('Loc+'))
        if v is None:
            continue
        z = classify_zone(p)
        st = count_state(p)
        if z is None or st is None:
            continue
        grp = group_of_code(p.get('Pitch Type'))
        units[(p.get('Pitcher'), p.get('PTeam'), grp)].append((v, z, st))
        lg_acc[grp][(z, st)][0] += v
        lg_acc[grp][(z, st)][1] += 1
        lg_n[grp] += 1

    lg = {}
    for grp, cells in lg_acc.items():
        tot = lg_n[grp]
        lg[grp] = {c: (n / tot, s / n) for c, (s, n) in cells.items()}
        for c in CELLS:
            lg[grp].setdefault(c, (0.0, 0.0))

    qual = {u: ps for u, ps in units.items() if len(ps) >= MIN_UNIT_PITCHES}
    print(f'{len(qual)} qualifying (pitcher, group) units of {len(units)}')

    results = []
    for k in K_GRID:
        corrs = []
        for (pit, team, grp), ps in qual.items():
            half_a = ps[0::2]
            half_b = ps[1::2]
            acc_a, acc_b = cell_agg(half_a), cell_agg(half_b)
            ia_k = impact_vector(acc_a, len(half_a), lg[grp], k)
            ib_raw = impact_vector(acc_b, len(half_b), lg[grp], 0)
            ib_k = impact_vector(acc_b, len(half_b), lg[grp], k)
            ia_raw = impact_vector(acc_a, len(half_a), lg[grp], 0)
            c1 = pearson(ia_k, ib_raw)
            c2 = pearson(ib_k, ia_raw)
            for c in (c1, c2):
                if c is not None:
                    corrs.append(c)
        mean_r = sum(corrs) / len(corrs)
        results.append((k, mean_r))
        print(f'  k={k:>3}  cross-half predictive r={mean_r:.4f}  (n={len(corrs)})')

    best = max(results, key=lambda kr: kr[1])
    print(f'\nBest on grid: k={best[0]} (r={best[1]:.4f})')
    interior = results[0][1] < best[1] and results[-1][1] < best[1]
    print('Interior optimum bracketed.' if interior else
          'WARNING: optimum at grid edge; extend the grid.')


if __name__ == '__main__':
    main()
