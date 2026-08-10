#!/usr/bin/env python3
"""Tune the zone x count impact-page constants for SD+ / CT+ hitter pages.

Mirrors the two Loc+ sweeps (loczone_impact_shrink_sweep.py and
loczone_callout_sweep.py) on the hitter decision/contact atoms — the
pitcher-page constants (k=8, lam=0.6) were tuned on Loc+ atoms, which are
near-deterministic in location; SD+/CT+ atoms are signed near-binary
variables with cell SDs ~25x larger, so the constants must be re-swept,
not adopted.

Atoms (both make the league overall grade exactly 100 by construction):
  SD: atom = 100 * dv / lg_mean_dv, dv = RV(chosen) - RV(opposite) from the
      shipped sdPlusWeights table; unit = (hitter, team, pitch category).
  CT: atom = 100 * lev * I[contact] / lg_mean(lev * (1 - p_whiff)), lev and
      p_whiff from the shipped ctPlusWeights table; league cell grade
      100 * lev * (1 - pw) / D, so cell-level league contact ties out.

Sweep 1 (shrinkage k): even/odd pitch split per unit; correlate shrunk
impact vector from half A with the raw (k=0) vector from half B, both
directions, averaged. Sweep 2 (callout ranker): SEDISC family — score =
sign(imp) * max(0, |imp| - lam * SE); objective = realized raw impact
spread of the top-2/bottom-2 picks in the opposite half.

Sample-size note: units are capped at CAP pitches (drawn as every other
pitch first, then the rest) so MLB tuning halves match a ROC season's
per-category volume — tuning at a larger n than production would favor
less shrinkage than the ROC pages need.

Usage: python3 scripts/sdzone_impact_sweeps.py
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_sdplus import (  # noqa: E402
    is_eligible, classify_zone, classify_decision, get_count, cat_of,
)
from pipeline_contact import is_ct_eligible, classify_contact_outcome  # noqa: E402

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
STATES = ['ahead', 'even', 'behind']
CELLS = [(z, s) for z in ZONES for s in STATES]

# Unit gates/caps (decisions for SD, swings for CT): floor ~ smallest cat a
# ROC season page will show, cap ~ its largest (FB) cat.
SD_MIN, SD_CAP = 250, 800
CT_MIN, CT_CAP = 120, 400
K_GRID = [0, 5, 10, 20, 40, 80, 160, 320, 640]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def count_state_hitter(p):
    """Hitter perspective: ahead = more balls than strikes."""
    c = get_count(p)
    if c is None:
        return None
    b, s = c
    return 'ahead' if b > s else ('behind' if s > b else 'even')


def cell_agg(pitches):
    acc = defaultdict(lambda: [0.0, 0])
    for v, z, st in pitches:
        acc[(z, st)][0] += v
        acc[(z, st)][1] += 1
    return acc


def impact_vector(acc, n_total, lg, k):
    out = []
    for cell in CELLS:
        ls, lgg, _sd = lg[cell]
        s_sum, n = acc.get(cell, [0.0, 0])
        share = n / n_total
        if n > 0:
            g = s_sum / n
            g_shr = (n * g + k * lgg) / (n + k)
        else:
            g_shr = lgg
        out.append(share * g_shr - ls * lgg)
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else None


def league_table(atoms):
    """[(atom, zone, state)] -> {cell: (share, grade, sd)}."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for v, z, st in atoms:
        c = acc[(z, st)]
        c[0] += v
        c[1] += v * v
        c[2] += 1
    tot = len(atoms)
    out = {}
    for cell in CELLS:
        s, ss, n = acc.get(cell, [0.0, 0.0, 0])
        if n:
            mean = s / n
            out[cell] = (n / tot, mean, math.sqrt(max(ss / n - mean ** 2, 0.0)))
        else:
            out[cell] = (0.0, 0.0, 0.0)
    return out


def sweep_k(tag, units, lg_by_grp):
    print(f'--- {tag}: shrinkage k sweep ({len(units)} units) ---')
    results = []
    for k in K_GRID:
        corrs = []
        for grp, ps in units:
            lg = lg_by_grp[grp]
            half_a, half_b = ps[0::2], ps[1::2]
            acc_a, acc_b = cell_agg(half_a), cell_agg(half_b)
            for x, y in ((impact_vector(acc_a, len(half_a), lg, k),
                          impact_vector(acc_b, len(half_b), lg, 0)),
                         (impact_vector(acc_b, len(half_b), lg, k),
                          impact_vector(acc_a, len(half_a), lg, 0))):
                c = pearson(x, y)
                if c is not None:
                    corrs.append(c)
        mean_r = sum(corrs) / len(corrs)
        results.append((k, mean_r))
        print(f'  k={k:>4}  cross-half predictive r={mean_r:.4f}')
    best = max(results, key=lambda kr: kr[1])
    interior = results[0][1] < best[1] and results[-1][1] < best[1]
    print(f'  best on grid: k={best[0]} (r={best[1]:.4f}) — '
          + ('interior optimum bracketed'
         if interior else 'EDGE — extend the grid') + '\n')
    return best[0]


def se_impact(share, n, n_type, lg_grade, sd_cell):
    se_g = (sd_cell / math.sqrt(n)) if n else 0.0
    se_s = math.sqrt(max(share * (1 - share), 1e-9) / n_type)
    return math.sqrt((share * se_g) ** 2 + (lg_grade * se_s) ** 2)


def sweep_lam(tag, hitter_units, lg_by_grp, k_display):
    """hitter_units: list of {grp: [(atom, z, st), ...]} per hitter."""
    print(f'--- {tag}: callout SE-discount sweep (k_display={k_display}) ---')

    def half_rows(halves):
        rows = []
        for grp, ps in halves.items():
            n_type = len(ps)
            if not n_type:
                continue
            acc = cell_agg(ps)
            lg = lg_by_grp[grp]
            for cell in CELLS:
                ls, lgg, sd = lg[cell]
                s_sum, n = acc.get(cell, [0.0, 0])
                share = n / n_type
                g = (s_sum / n) if n else None
                g_shr = ((n * g + k_display * lgg) / (n + k_display)
                         if n else lgg)
                rows.append({'grp': grp, 'cell': cell, 'n': n,
                             'n_type': n_type, 'share': share,
                             'imp': share * g_shr - ls * lgg,
                             'imp_raw': share * (g if n else lgg) - ls * lgg,
                             'se': se_impact(share, n, n_type, lgg, sd)})
        return rows

    dirs = []
    for unit in hitter_units:
        a = {grp: ps[0::2] for grp, ps in unit.items()}
        b = {grp: ps[1::2] for grp, ps in unit.items()}
        dirs.append((a, b))
        dirs.append((b, a))

    # Objective: EXPECTED realized spread — a direction whose ranker makes
    # no picks contributes 0, so the metric cannot be gamed by filtering
    # down to a cherry-picked handful of extreme pages (the conditional
    # spread rises monotonically with lam for exactly that reason).
    results = []
    for lam in LAM_GRID:
        spreads = []
        for ha, hb in dirs:
            rows_a = half_rows(ha)
            b_map = {(r['grp'], r['cell']): r['imp_raw'] for r in half_rows(hb)}
            scored = []
            for r in rows_a:
                d = abs(r['imp']) - lam * r['se']
                sc = math.copysign(max(0.0, d), r['imp']) if d > 0 else 0.0
                scored.append((sc, r))
            pos = sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])[:2]
            neg = sorted((x for x in scored if x[0] < 0), key=lambda x: x[0])[:2]
            vp = [b_map.get((r['grp'], r['cell'])) for _, r in pos]
            vn = [b_map.get((r['grp'], r['cell'])) for _, r in neg]
            vp = [v for v in vp if v is not None]
            vn = [v for v in vn if v is not None]
            spreads.append((sum(vp) / len(vp) - sum(vn) / len(vn))
                           if (vp and vn) else 0.0)
        exp_sp = sum(spreads) / len(spreads)
        n_made = sum(1 for s in spreads if s != 0.0)
        results.append((lam, exp_sp, n_made))
        print(f'  lam={lam:>4}: expected realized spread {exp_sp:.3f} pts '
              f'(picks on {n_made}/{len(dirs)} directions)')
    best = max(results, key=lambda r: r[1])
    interior = results[0][1] < best[1] and results[-1][1] < best[1]
    print(f'  best on grid: lam={best[0]} (expected spread {best[1]:.3f}) — '
          + ('interior optimum bracketed'
         if interior else 'EDGE — extend the grid') + '\n')
    return best[0]


def cap_unit(ps, cap):
    if len(ps) <= cap:
        return ps
    # every-other draw keeps both halves of the even/odd split balanced
    half = ps[0::2] + ps[1::2]
    return half[:cap]


def main():
    print('Loading cache + tables ...')
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        allp = pickle.load(f)
    meta = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
    sdw, ctw = meta['sdPlusWeights'], meta['ctPlusWeights']

    mlb = [p for p in allp if p.get('_source', 'MLB') == 'MLB' and is_eligible(p)]

    def dv(p):
        z, c, cat = classify_zone(p), get_count(p), cat_of(p)
        s = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|swing']['rv']
        t = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|take']['rv']
        return (s - t) if classify_decision(p) == 'swing' else (t - s)

    lg_mean_dv = sum(dv(p) for p in mlb) / len(mlb)
    print(f'  league mean dv = {lg_mean_dv:.5f} runs/decision')

    def sd_atom(p):
        return 100.0 * dv(p) / lg_mean_dv

    mlb_sw = [p for p in mlb if is_ct_eligible(p)]

    def lev_pw(p):
        cell = ctw[f'{classify_zone(p)}|{get_count(p)[0]}-{get_count(p)[1]}']
        return cell['rv_contact'] - cell['rv_whiff'], cell['p_whiff']

    D = sum(lv * (1 - pw) for lv, pw in (lev_pw(p) for p in mlb_sw)) / len(mlb_sw)
    print(f'  league mean lev*(1-pw) = {D:.5f}')

    def ct_atom(p):
        lv, _pw = lev_pw(p)
        made = 1.0 if classify_contact_outcome(p) == 'contact' else 0.0
        return 100.0 * lv * made / D

    for tag, pool, atom_fn, floor, cap in (
            ('SD', mlb, sd_atom, SD_MIN, SD_CAP),
            ('CT', mlb_sw, ct_atom, CT_MIN, CT_CAP)):
        lg_atoms = defaultdict(list)
        by_unit = defaultdict(list)
        by_hitter = defaultdict(lambda: defaultdict(list))
        for p in pool:
            z, st = classify_zone(p), count_state_hitter(p)
            if st is None:
                continue
            grp = cat_of(p)
            a = atom_fn(p)
            lg_atoms[grp].append((a, z, st))
            by_unit[(p.get('Batter'), p.get('BTeam'), grp)].append((a, z, st))
            by_hitter[(p.get('Batter'), p.get('BTeam'))][grp].append((a, z, st))
        lg_by_grp = {g: league_table(v) for g, v in lg_atoms.items()}
        units = [(grp, cap_unit(ps, cap))
                 for (_h, _t, grp), ps in by_unit.items() if len(ps) >= floor]
        k_best = sweep_k(tag, units, lg_by_grp)
        h_units = []
        for _key, grps in by_hitter.items():
            capped = {g: cap_unit(ps, cap) for g, ps in grps.items()
                      if len(ps) >= floor}
            if capped:
                h_units.append(capped)
        sweep_lam(tag, h_units, lg_by_grp, k_best)


if __name__ == '__main__':
    main()
