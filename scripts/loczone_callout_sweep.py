#!/usr/bin/env python3
"""Sweep callout-ranker configs for the zone x count impact pages.

Task being tuned: from a pitcher's (pitch type, zone, count-state) impact
cells, pick the top-2 positive and top-2 negative "callout" cells. A good
ranker picks cells whose impact is REAL, i.e. replicates out of sample.

Objective: split every qualifying MLB pitcher's graded pitches into even/odd
halves. Each candidate ranker selects callout cells from half A by its score;
quality is the realized RAW impact spread of those same cells in half B
(mean B-impact of positive picks minus mean B-impact of negative picks),
averaged over both directions (A->B and B->A) and all pitchers. Coverage
(picks actually made / 4) is reported because a floor that filters everything
"wins" nothing.

Families swept:
  FLOOR:  score = impact(k=8), pitch types eligible only with n >= F pitches
          in the half (full-season equivalent ~ 2F).
  KRANK:  score = impact(k_rank), stronger shrinkage at rank time only.
  SEDISC: score = sign-aware SE discount on impact(k=8):
          d = sign(imp) * max(0, |imp| - lam * SE(imp)),
          SE^2 = (s * sd_cell / sqrt(n))^2 + (lg_grade * se_share)^2.

Usage: python3 scripts/loczone_callout_sweep.py
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
K_DISPLAY = 8
MIN_PITCHER_PITCHES = 400


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


def build_half(pitches):
    """-> {ptype: {'n': n, 'cells': {(z,st): [sum, n]}}}"""
    out = defaultdict(lambda: {'n': 0, 'cells': defaultdict(lambda: [0.0, 0])})
    for v, z, st, pt in pitches:
        d = out[pt]
        d['n'] += 1
        d['cells'][(z, st)][0] += v
        d['cells'][(z, st)][1] += 1
    return out


def cell_rows(half, lg):
    """All cells of one half -> list of dicts with impact ingredients."""
    rows = []
    for pt, d in half.items():
        grp = group_of_code(pt)
        L = lg.get(grp)
        if L is None or d['n'] == 0:
            continue
        for cell in CELLS:
            ls, lgg, sd = L[cell]
            s_sum, n = d['cells'].get(cell, [0.0, 0])
            share = n / d['n']
            g_raw = (s_sum / n) if n else None
            rows.append({'pt': pt, 'cell': cell, 'n': n, 'n_type': d['n'],
                         'share': share, 'g_raw': g_raw,
                         'ls': ls, 'lg': lgg, 'sd': sd})
    return rows


def impact(row, k):
    if row['n']:
        g = (row['n'] * row['g_raw'] + k * row['lg']) / (row['n'] + k)
    else:
        g = row['lg']
    return row['share'] * g - row['ls'] * row['lg']


def se_impact(row):
    se_g = (row['sd'] / math.sqrt(row['n'])) if row['n'] else 0.0
    se_s = math.sqrt(max(row['share'] * (1 - row['share']), 1e-9) / row['n_type'])
    return math.sqrt((row['share'] * se_g) ** 2 + (row['lg'] * se_s) ** 2)


def score(row, family, param):
    if family == 'FLOOR':
        if row['n_type'] < param:
            return None
        return impact(row, K_DISPLAY)
    if family == 'KRANK':
        return impact(row, param)
    if family == 'SEDISC':
        imp = impact(row, K_DISPLAY)
        d = abs(imp) - param * se_impact(row)
        return math.copysign(max(0.0, d), imp) if d > 0 else 0.0
    raise ValueError(family)


def evaluate(units, lg, family, param):
    spreads, picks_made = [], 0
    for half_a, half_b in units:
        rows_a = cell_rows(half_a, lg)
        # raw impacts in B for the same (pt, cell)
        b_map = {(r['pt'], r['cell']): impact(r, 0) for r in cell_rows(half_b, lg)}
        scored = [(score(r, family, param), r) for r in rows_a]
        scored = [(sc, r) for sc, r in scored if sc is not None]
        pos = sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])[:2]
        neg = sorted((x for x in scored if x[0] < 0), key=lambda x: x[0])[:2]
        vals_p = [b_map.get((r['pt'], r['cell'])) for _, r in pos]
        vals_n = [b_map.get((r['pt'], r['cell'])) for _, r in neg]
        vals_p = [v for v in vals_p if v is not None]
        vals_n = [v for v in vals_n if v is not None]
        picks_made += len(vals_p) + len(vals_n)
        if vals_p and vals_n:
            spreads.append(sum(vals_p) / len(vals_p) - sum(vals_n) / len(vals_n))
    cov = picks_made / (4.0 * len(units))
    return (sum(spreads) / len(spreads) if spreads else 0.0), cov, len(spreads)


def main():
    print('Loading cache ...')
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        allp = pickle.load(f)

    by_pitcher = defaultdict(list)
    lg_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
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
        pt = p.get('Pitch Type') or '?'
        by_pitcher[(p.get('Pitcher'), p.get('PTeam'))].append((v, z, st, pt))
        grp = group_of_code(pt)
        c = lg_acc[grp][(z, st)]
        c[0] += v
        c[1] += v * v
        c[2] += 1
        lg_n[grp] += 1

    lg = {}
    for grp, cells in lg_acc.items():
        tot = lg_n[grp]
        out = {}
        for cell in CELLS:
            s, ss, n = cells.get(cell, [0.0, 0.0, 0])
            if n:
                mean = s / n
                var = max(ss / n - mean * mean, 0.0)
                out[cell] = (n / tot, mean, math.sqrt(var))
            else:
                out[cell] = (0.0, 0.0, 0.0)
        lg[grp] = out

    units = []
    for key, ps in by_pitcher.items():
        if len(ps) >= MIN_PITCHER_PITCHES:
            a, b = build_half(ps[0::2]), build_half(ps[1::2])
            units.append((a, b))
            units.append((b, a))
    print(f'{len(units)} evaluation directions from {len(units) // 2} pitchers\n')

    grids = [
        ('FLOOR', [0, 12, 25, 37, 50, 75, 100]),   # half-sample floors (~2x season)
        ('KRANK', [8, 16, 30, 60, 120, 240]),
        ('SEDISC', [0.5, 1.0, 1.5, 2.0, 3.0]),
    ]
    best = {}
    for family, grid in grids:
        print(f'--- {family} ---')
        curve = []
        for param in grid:
            spread, cov, n_used = evaluate(units, lg, family, param)
            curve.append((param, spread, cov))
            print(f'  {family} {param:>6}: realized spread {spread:.3f} pts '
                  f'· coverage {cov:.2f} · units {n_used}')
        best[family] = max(curve, key=lambda c: c[1])
        p, s, c = best[family]
        print(f'  best: {p} (spread {s:.3f}, coverage {c:.2f})\n')

    print('=== winners ===')
    for fam, (p, s, c) in best.items():
        print(f'{fam}: param {p} spread {s:.3f} coverage {c:.2f}')


if __name__ == '__main__':
    main()
