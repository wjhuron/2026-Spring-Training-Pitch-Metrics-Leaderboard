"""phase2_sdct_harness.py — A/B validation for the Phase 2 SD+/CT+ changes.

Variants tested (cumulative and isolated):
  SD+:  baseline | +count-anchored BIP offsets | +heart=1/6 | both
  CT+:  baseline (leverage-weighted raw contact rate)
        | +offsets | lift-ratio (actual/expected) | both

Metrics per variant:
  - split-half reliability (odd/even calendar dates, unshrunk raw values),
    reported at half-n >= 40 and >= 80, plus implied stabilization n0
    (Spearman-Brown: n0 ~= mean_half_n * (1-r)/r) and the r=.50 floor (=n0,
    in FULL-sample decisions ~= 2*half-floor)
  - descriptive validity: corr of full-sample raw vs season wRC+ (hitters
    with wRC+ and >= 200 full-sample events)
  - spread of the shipped Plus scale

Usage: python3 scripts/research/hitter/phase2_sdct_harness.py
"""
import os, sys, pickle, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import pipeline.sdplus as sd
import pipeline.contact as ct
from pipeline.sdplus import (
    is_eligible, classify_zone, classify_decision, get_count,
    build_weight_table, zone_level_means, shrink_table, compute_dv,
    make_rv_xrv, build_bip_count_offsets,
)
import json

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
HL = os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return cov / (sx * sy) if sx > 0 and sy > 0 else None


def build_sd_table(pitches, use_offsets, heart_frac):
    old_frac = sd.HEART_VERT_FRAC
    sd.HEART_VERT_FRAC = heart_frac
    try:
        offsets = build_bip_count_offsets(pitches, LG_WOBA, WOBA_SCALE) if use_offsets else None
        rv_fn = make_rv_xrv(LG_WOBA, WOBA_SCALE, offsets)
        raw = build_weight_table(pitches, rv_fn)
        zm = zone_level_means(pitches, rv_fn)
        return shrink_table(raw, zm)
    finally:
        sd.HEART_VERT_FRAC = old_frac


def sd_raw_for(pitches, table, heart_frac):
    old_frac = sd.HEART_VERT_FRAC
    sd.HEART_VERT_FRAC = heart_frac
    try:
        elig = [p for p in pitches if is_eligible(p)]
        if not elig:
            return None, 0
        dvs = [compute_dv(p, table) for p in elig]
        return sum(dvs) / len(dvs), len(dvs)
    finally:
        sd.HEART_VERT_FRAC = old_frac


def build_ct_table(swings, use_offsets, heart_frac):
    old_frac = sd.HEART_VERT_FRAC
    sd.HEART_VERT_FRAC = heart_frac
    try:
        offsets = build_bip_count_offsets(swings, LG_WOBA, WOBA_SCALE) if use_offsets else None
        rv_fn = make_rv_xrv(LG_WOBA, WOBA_SCALE, offsets)
        raw = ct.build_contact_cell_weights(swings, rv_fn)
        zm = ct.zone_level_contact_means(swings, rv_fn)
        return ct.shrink_contact_cells(raw, zm)
    finally:
        sd.HEART_VERT_FRAC = old_frac


def ct_raw_for(pitches, table, heart_frac, lift):
    """Returns (raw, n_swings). raw = leverage-weighted contact rate
    (baseline) or actual/expected ratio (lift)."""
    old_frac = sd.HEART_VERT_FRAC
    sd.HEART_VERT_FRAC = heart_frac
    try:
        swings = [p for p in pitches if ct.is_ct_eligible(p)]
        if not swings:
            return None, 0
        A = E = W = 0.0
        for p in swings:
            lev, con = ct.compute_ct_swing(p, table)
            if lev <= 0:
                continue
            cell = table[(classify_zone(p), get_count(p))]
            A += lev * con
            E += lev * (1.0 - cell['p_whiff'])
            W += lev
        if W <= 0:
            return None, 0
        if lift:
            return (A / W) / (E / W) if E > 0 else None, len(swings)
        return A / W, len(swings)
    finally:
        sd.HEART_VERT_FRAC = old_frac


def evaluate(name, raw_fn, table_builder, pitches_by_hitter, mlb_elig,
             wrc_by_key, heart_frac, min_half=40):
    """table_builder(pitch_subset) -> table; raw_fn(pitches, table) -> (raw, n)."""
    # full-sample table + values
    full_table = table_builder(mlb_elig)
    full = {}
    for key, pitches in pitches_by_hitter.items():
        raw, n = raw_fn(pitches, full_table)
        if raw is not None:
            full[key] = (raw, n)

    # split halves by global date parity (tables rebuilt per half)
    dates = sorted({p.get('Game Date') for p in mlb_elig if p.get('Game Date')})
    half_of = {d: i % 2 for i, d in enumerate(dates)}
    elig_h = ([p for p in mlb_elig if half_of.get(p.get('Game Date')) == 0],
              [p for p in mlb_elig if half_of.get(p.get('Game Date')) == 1])
    tables_h = (table_builder(elig_h[0]), table_builder(elig_h[1]))
    halves = ({}, {})
    for key, pitches in pitches_by_hitter.items():
        for h in (0, 1):
            sub = [p for p in pitches if half_of.get(p.get('Game Date')) == h]
            raw, n = raw_fn(sub, tables_h[h])
            if raw is not None:
                halves[h][key] = (raw, n)

    out = {'name': name}
    for mh in (min_half, 2 * min_half):
        xs, ys, ns = [], [], []
        for key in halves[0]:
            if key not in halves[1]:
                continue
            r0, n0_ = halves[0][key]
            r1, n1_ = halves[1][key]
            if n0_ >= mh and n1_ >= mh:
                xs.append(r0); ys.append(r1); ns.append((n0_ + n1_) / 2)
        r = pearson(xs, ys)
        out[f'r@half{mh}'] = (round(r, 3) if r is not None else None, len(xs))
        if r and 0 < r < 1 and ns:
            nbar = sum(ns) / len(ns)
            out[f'n0@half{mh}'] = round(nbar * (1 - r) / r)

    # descriptive vs wRC+
    xs, ys = [], []
    for key, (raw, n) in full.items():
        if n >= 200 and key in wrc_by_key:
            xs.append(raw); ys.append(wrc_by_key[key])
    r = pearson(xs, ys)
    out['r_wrc'] = (round(r, 3) if r is not None else None, len(xs))
    return out


def main():
    D = pickle.load(open(PKL, 'rb'))
    hitter_groups = defaultdict(list)
    for p in D:
        h, t = p.get('Batter'), p.get('BTeam')
        if h and t:
            hitter_groups[(h, t)].append(p)

    hl = json.load(open(HL))
    wrc_by_key = {(r['hitter'], r['team']): r['wRCplus'] for r in hl
                  if r.get('wRCplus') is not None and not r.get('_isROC')}

    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']

    print('=== SD+ variants ===')
    for name, use_off, frac in [('baseline', False, 1/3), ('offsets', True, 1/3),
                                ('heart16', False, 1/6), ('offsets+heart16', True, 1/6)]:
        old = sd.HEART_VERT_FRAC
        sd.HEART_VERT_FRAC = frac
        elig = [p for p in mlb if is_eligible(p)]
        sd.HEART_VERT_FRAC = old
        res = evaluate(
            name,
            lambda ps, tb, f=frac: sd_raw_for(ps, tb, f),
            lambda ps, u=use_off, f=frac: build_sd_table(ps, u, f),
            hitter_groups, elig, wrc_by_key, frac, min_half=110)
        print(res)

    print('\n=== CT+ variants ===')
    for name, use_off, frac, lift in [
            ('baseline', False, 1/3, False), ('offsets', True, 1/3, False),
            ('lift-ratio', False, 1/3, True), ('offsets+lift', True, 1/3, True),
            ('offs+lift+heart16', True, 1/6, True)]:
        old = sd.HEART_VERT_FRAC
        sd.HEART_VERT_FRAC = frac
        swings = [p for p in mlb if ct.is_ct_eligible(p)]
        sd.HEART_VERT_FRAC = old
        res = evaluate(
            name,
            lambda ps, tb, f=frac, l=lift: ct_raw_for(ps, tb, f, l),
            lambda ps, u=use_off, f=frac: build_ct_table(ps, u, f),
            hitter_groups, swings, wrc_by_key, frac, min_half=42)
        print(res)


if __name__ == '__main__':
    main()
