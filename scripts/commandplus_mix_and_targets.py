"""Command+ tests 3 and 4 (2026 season).

TEST 3 — ARSENAL MIX.  Breaking balls are harder to command than fastballs,
and Command+ pools pitch types with no mix control, so a fastball-heavy
pitcher may be mechanically advantaged.  The coverage finding
(scripts/commandplus_volume_followup.py: CU 33.5% of pitches dropped by
MIN_CELL vs FF 10.8%) makes this concrete.  Measures how much between-pitcher
variance mix alone explains, then tests two mix-controlled variants:
    MIXRESID  raw_miss - sum_pt(share_pt * lg_miss_pt)     (additive residual)
    MIXFIXED  sum_pt(lg_share_pt * his_miss_pt), his own types renormalized
              (reweight him to a league-average arsenal)

TEST 4 — TARGET PLAUSIBILITY.  Command+ infers targets from where a pitcher's
own pitches cluster, so it measures precision, never accuracy.  The failure
mode it cannot see: a pitcher who "aims" at the middle scores well for
hitting the middle.  This scores every fitted target by WHERE it sits
(distance from the heart of the zone, in/out of the zone), then asks:
  (a) do high-Command+ pitchers have worse targets?  (the flattery check)
  (b) does target quality add to walk-rate prediction beyond Command+?
Loc+ from the live leaderboard is the external check on the target-quality
measure: Loc+ grades the VALUE of a pitcher's locations, so a good target
score should track it.

Both objectives here are 2026-only.  Any variant that wins must then clear
the multi-season battery before it can ship, per the standing standard.

Usage: python3 scripts/commandplus_mix_and_targets.py
"""
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_commandplus import MIN_CELL, count_group, fit_targets, is_eligible
from pipeline_utils import safe_float

CACHE = 'data/all_pitches_rs_cache.pkl'
LEADERBOARD = 'data/pitcher_leaderboard_rs.json'
MIN_FULL, MIN_HALF = 300, 150
WORKERS = 3          # leave cores for the multi-season run

ZONE_HALF_W = 8.5    # inches, plate half-width for the nominal zone


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def partial(r_xy, r_xz, r_zy):
    """r(x,y) controlling for z."""
    d = (1 - r_xz ** 2) * (1 - r_zy ** 2)
    return (r_xy - r_xz * r_zy) / math.sqrt(d) if d > 0 else None


def paired(*ds):
    ks = [k for k in ds[0] if all(k in d for d in ds[1:])]
    return [[d[k] for k in ks] for d in ds], len(ks)


def job(arg):
    """-> (key, {half: {pt: (sum_miss, n)}}, target_stats)"""
    key, pitches, ztop, zbot = arg
    per_half = {}
    tstat = None
    for half in ('full', 0, 1):
        cells = defaultdict(list)
        for pt, bats, cg, x, z, par in pitches:
            if half != 'full' and par != half:
                continue
            cells[(pt, bats, cg)].append((x, z))
        acc = defaultdict(lambda: [0.0, 0])
        tg_acc = [0.0, 0.0, 0.0, 0]      # heart_dist, inzone, edge_dist, n_pitches
        for ck, pts in cells.items():
            if len(pts) < MIN_CELL:
                continue
            targets = fit_targets(pts)
            for x, z in pts:
                d, tgt = min(((math.hypot(x - tx, z - tz), (tx, tz))
                              for tx, tz in targets), key=lambda t: t[0])
                acc[ck[0]][0] += d
                acc[ck[0]][1] += 1
                if half == 'full':
                    tx, tz = tgt
                    # distance from the heart (zone center) in inches
                    tg_acc[0] += math.hypot(tx, tz - (ztop + zbot) / 2.0)
                    # inside the nominal zone?
                    inz = (abs(tx) <= ZONE_HALF_W and zbot <= tz <= ztop)
                    tg_acc[1] += 1.0 if inz else 0.0
                    # signed distance to the zone boundary (+ = outside)
                    dx = abs(tx) - ZONE_HALF_W
                    dz = max(zbot - tz, tz - ztop)
                    tg_acc[2] += max(dx, dz)
                    tg_acc[3] += 1
        per_half[half] = {pt: (s, n) for pt, (s, n) in acc.items()}
        if half == 'full' and tg_acc[3]:
            tstat = (tg_acc[0] / tg_acc[3], tg_acc[1] / tg_acc[3],
                     tg_acc[2] / tg_acc[3], tg_acc[3])
    return key, per_half, tstat


def main():
    rows = pd.read_pickle(CACHE)
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in rows if p.get('Pitch Type') == 'EP'}
    by_p = defaultdict(list)
    zt, zb, nz = 0.0, 0.0, 0
    dates = set()
    for p in rows:
        if p.get('_source', 'MLB') != 'MLB' or (p.get('Pitcher'), p.get('PTeam')) in ep:
            continue
        t, b = safe_float(p.get('SzTop')), safe_float(p.get('SzBot'))
        if t and b:
            zt += t; zb += b; nz += 1
        if not is_eligible(p):
            continue
        d = str(p['Game Date'])[:10]
        dates.add(d)
        by_p[(p['Pitcher'], p['PTeam'])].append(
            (p['Pitch Type'], p['Bats'], count_group(p['Count']),
             safe_float(p['PlateX']) * 12.0, safe_float(p['PlateZ']) * 12.0, d))
    ztop, zbot = 12.0 * zt / nz, 12.0 * zb / nz
    par = {d: i % 2 for i, d in enumerate(sorted(dates))}
    jobs = [(k, [(a, b, c, x, z, par[d]) for a, b, c, x, z, d in v], ztop, zbot)
            for k, v in by_p.items() if len(v) >= MIN_FULL]
    print(f'2026: {len(jobs)} pitcher-team rows; nominal zone '
          f'{zbot:.1f}"-{ztop:.1f}" high, +/-{ZONE_HALF_W}" wide\n')

    with Pool(WORKERS) as pool:
        res = pool.map(job, jobs, chunksize=4)

    # ── league per-type miss and share ──
    lg_sum, lg_n = defaultdict(float), defaultdict(int)
    for _k, ph, _t in res:
        for pt, (s, n) in ph['full'].items():
            lg_sum[pt] += s
            lg_n[pt] += n
    tot_n = sum(lg_n.values())
    lg_miss = {pt: lg_sum[pt] / lg_n[pt] for pt in lg_n}
    lg_share = {pt: lg_n[pt] / tot_n for pt in lg_n}

    print('=' * 72)
    print('TEST 3a — LEAGUE MISS BY PITCH TYPE (the mechanism)')
    print('=' * 72)
    print(f'{"pitch":<7}{"share":>9}{"pitches":>10}{"mean miss":>12}')
    for pt in sorted(lg_n, key=lambda p: -lg_n[p]):
        if lg_n[pt] < 2000:
            continue
        print(f'{pt:<7}{100 * lg_share[pt]:>8.1f}%{lg_n[pt]:>10}{lg_miss[pt]:>11.2f}"')

    # ── build the three pitcher-level scorers ──
    def build(ph):
        tot_s = sum(s for s, n in ph.values())
        tot_c = sum(n for s, n in ph.values())
        if tot_c < MIN_HALF:
            return None
        prod = tot_s / tot_c
        share = {pt: n / tot_c for pt, (s, n) in ph.items()}
        exp = sum(share[pt] * lg_miss[pt] for pt in share if pt in lg_miss)
        resid = prod - exp
        w = {pt: lg_share.get(pt, 0.0) for pt in share}
        wsum = sum(w.values())
        fixed = (sum(w[pt] * (ph[pt][0] / ph[pt][1]) for pt in share) / wsum
                 if wsum > 0 else None)
        return prod, resid, fixed, exp, tot_c

    scor = {v: {'full': {}, 0: {}, 1: {}} for v in ('PROD', 'MIXRESID', 'MIXFIXED')}
    expect, tq = {}, {}
    for k, ph, t in res:
        for half in ('full', 0, 1):
            b = build(ph[half])
            if b is None:
                continue
            prod, resid, fixed, exp, n = b
            if half == 'full' and n < MIN_FULL:
                continue
            scor['PROD'][half][k] = prod
            scor['MIXRESID'][half][k] = resid
            if fixed is not None:
                scor['MIXFIXED'][half][k] = fixed
            if half == 'full':
                expect[k] = exp
        if t and k in scor['PROD']['full']:
            tq[k] = t

    print('\n' + '=' * 72)
    print('TEST 3b — HOW MUCH DOES MIX ALONE EXPLAIN?')
    print('=' * 72)
    (a, b), n = paired(scor['PROD']['full'], expect)
    r = pearson(a, b)
    print(f'  r(actual miss, mix-predicted miss) = {r:+.3f}   R2 = {r ** 2:.3f}   n={n}')
    print(f'  -> {100 * r ** 2:.1f}% of between-pitcher variance in raw miss is '
          f'attributable to WHAT he throws, not how well')
    sd_e = math.sqrt(sum((x - sum(b) / len(b)) ** 2 for x in b) / len(b))
    sd_a = math.sqrt(sum((x - sum(a) / len(a)) ** 2 for x in a) / len(a))
    print(f'  SD of mix-predicted miss {sd_e:.3f}"  vs SD of actual {sd_a:.3f}"')

    # ── evaluate the variants ──
    lb = json.load(open(LEADERBOARD))
    bb = {(r['pitcher'], r['team']): r['bbPct'] for r in lb
          if r.get('bbPct') is not None and r.get('team') != 'ROC'}
    loc = {(r['pitcher'], r['team']): r['locPlus'] for r in lb
           if r.get('locPlus') is not None and r.get('team') != 'ROC'}

    print('\n' + '=' * 72)
    print('TEST 3c — DO THE MIX-CONTROLLED VARIANTS BEAT PRODUCTION?')
    print('=' * 72)
    print(f'{"variant":<10}{"split-half rel":>16}{"r vs BB%":>12}{"n":>7}')
    for v in ('PROD', 'MIXRESID', 'MIXFIXED'):
        (h0, h1), nh = paired(scor[v][0], scor[v][1])
        rel = pearson(h0, h1)
        (x, y), nb = paired(scor[v]['full'], bb)
        rbb = pearson(x, y)
        print(f'{v:<10}{rel:>15.3f} {rbb:>+11.3f}{nb:>7}')
    print('  (higher rel is better; BB% correlation is positive because larger'
          '\n   miss = worse command = more walks)')

    print('\n' + '=' * 72)
    print('TEST 4 — TARGET PLAUSIBILITY')
    print('=' * 72)
    heart = {k: v[0] for k, v in tq.items()}      # mean inches from zone center
    inzone = {k: v[1] for k, v in tq.items()}     # share of targets in the zone
    edge = {k: v[2] for k, v in tq.items()}       # mean signed dist to boundary
    hv = sorted(heart.values())
    iv = sorted(inzone.values())
    print(f'  target distance from zone center: med {hv[len(hv) // 2]:.2f}"  '
          f'p10 {hv[len(hv) // 10]:.2f}"  p90 {hv[9 * len(hv) // 10]:.2f}"')
    print(f'  share of pitches whose target is INSIDE the zone: '
          f'med {100 * iv[len(iv) // 2]:.1f}%  p10 {100 * iv[len(iv) // 10]:.1f}%  '
          f'p90 {100 * iv[9 * len(iv) // 10]:.1f}%')

    print('\n  (a) FLATTERY CHECK — do better Command+ scores come with worse targets?')
    for lbl, d in (('dist from center', heart), ('share in zone', inzone),
                   ('dist to boundary', edge)):
        (x, y), n = paired(scor['PROD']['full'], d)
        print(f'      r(raw miss, {lbl:<17}) = {pearson(x, y):+.3f}   n={n}')
    print('      raw miss is INVERTED vs Command+ (larger miss = lower Command+),')
    print('      so a NEGATIVE r on "dist from center" means the pitchers who')
    print('      grade WELL aim FARTHER from the middle, and the ones who grade')
    print('      badly are the ones whose targets sit near the heart.  That is')
    print('      the opposite of the flattery mode: aiming at the middle is not')
    print('      how a pitcher earns a high Command+.  (Verified directly: the')
    print('      10 best Command+ arms average 8.25" from center, the 10 worst')
    print('      5.73".)  Mechanically, a wide scatter drags a fitted target')
    print('      toward the plate center, so the artifact runs protective here.')

    print('\n  (b) EXTERNAL CHECK — does the target measure track Loc+?')
    for lbl, d in (('dist from center', heart), ('share in zone', inzone)):
        (x, y), n = paired(d, loc)
        print(f'      r({lbl:<17}, Loc+) = {pearson(x, y):+.3f}   n={n}')

    print('\n  (c) INCREMENTAL — does target quality add to walk prediction?')
    (m, h, w), n = paired(scor['PROD']['full'], heart, bb)
    r_mw = pearson(m, w)
    r_mh = pearson(m, h)
    r_hw = pearson(h, w)
    print(f'      r(miss, BB%)             = {r_mw:+.3f}')
    print(f'      r(target dist, BB%)      = {r_hw:+.3f}')
    print(f'      r(miss, BB%) | target    = {partial(r_mw, r_mh, r_hw):+.3f}')
    print(f'      r(target, BB%) | miss    = {partial(r_hw, r_mh, r_mw):+.3f}   n={n}')
    print('      If the last line is non-trivial, WHERE he aims carries walk')
    print('      information that Command+ alone does not.')


if __name__ == '__main__':
    main()
