"""sdplus_v2_prototype.py — would SD+ improve on continuous surfaces, the way
Loc+ v2 did?

MOTIVATION. Shipped SD+ grades each swing/take against a 360-cell table
(5 zones x 12 counts x 3 pitch categories x 2 decisions). Loc+ was rebuilt
from exactly this kind of zone table onto continuous smoothed surfaces in
2026-06 and roughly DOUBLED its signal (reliability 0.48->0.59, predictive
0.09->0.17). The same headroom argument applies to SD+, arguably more
directly: a cell mixes a barely-off-heart shadow pitch with one at the far
edge, so two hitters making genuinely different decisions grade identically.

THE BUILD IS NEARLY FREE. Loc+'s decomposition already computes both branch
values at the exact (x, z) for each pitch's group/hands/count —
    swing_val = Pwhiff*rvWhiff + Pfoul*rvFoul + Pbip*xwOBAcon_value
    take_val  = Pcs*rvCS + (1-Pcs)*rvBall
(hitter-perspective, count-specific weights, CS count-transform included).
SD+ v2 per-pitch decision value = value(chosen branch) - value(other branch),
the same opportunity-cost definition as shipped.

PROTOTYPE PROTOCOL (2026, MLB only; multi-season comes ONLY if this wins):
  - identical per-hitter pitch sample for both models: decision pitches
    (swing or take) passing BOTH models' eligibility
  - both models rebuilt per data slice (tables and surfaces alike)
  - aggregation mix-neutral for BOTH, using the same 5-zone league mix, so
    the comparison isolates the SCORING function, not the aggregation
  - objectives:
      rel   odd/even game-date split-half r of per-hitter raw score
            (>=125 decisions per half)
      pred  first-half raw score (>=200 decisions) vs second-half actual
            per-pitch hitter xRV (>=250 pitches) — does decision quality
            translate to future production?

Usage: python3 scripts/sdplus_v2_prototype.py
"""
import os, sys, math, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.locplus as lp
import pipeline.sdplus as sd
from pipeline.sdplus import make_rv_xrv, classify_zone

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_REL, MIN_PRED, MIN_ACT = 125, 200, 250

SWING = sd.SWING_DESCRIPTIONS if hasattr(sd, 'SWING_DESCRIPTIONS') else \
    {'Swinging Strike', 'Foul', 'In Play'}
TAKE = {'Called Strike', 'Ball'}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def branch_values(p, S):
    """(swing_val, take_val) from Loc+ surfaces — score_pitch's two branches,
    hitter-perspective."""
    key = (lp.group_of(p), p.get('Bats'), p.get('Throws'))
    if key not in S['WH']:
        return None
    c = lp.get_count(p)
    px = lp.safe_float(p.get('PlateX')); zn = lp._znorm(p)
    if c is None or px is None or zn is None:
        return None
    i = lp._xbin(px); j = lp._zbin(zn)
    pwh = S['WH'][key][i][j]; pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = S['XW'][key][i][j] + S['BIPOFF'].get(c, 0.0)
    pcs = S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return swing_val, take_val


def eligible_decision(p):
    """A pitch usable by BOTH models: SD+ eligibility, Loc+ scorability, and
    an actual swing/take decision."""
    if not sd.is_eligible(p):
        return False
    if not lp._is_scorable(p):
        return False
    return p.get('Description') in SWING or p.get('Description') in TAKE


def mix_neutral(dv_by_zone, lg_zone_w):
    """Per-zone mean dv reweighted to the league zone mix (shipped SD+'s
    aggregation, applied identically to both models)."""
    zm = {z: sum(v) / len(v) for z, v in dv_by_zone.items() if v}
    wsum = sum(lg_zone_w.get(z, 0.0) for z in zm)
    if wsum <= 0:
        return None
    return sum(m * lg_zone_w.get(z, 0.0) for z, m in zm.items()) / wsum


def score_slice(pitches, by_hitter, min_n):
    """Both models on one data slice. Returns (shipped_raw, v2_raw) dicts."""
    mlb = [p for p in pitches if p.get('_source', 'MLB') == 'MLB'
           and sd.is_eligible(p)]
    # shipped machinery: count-anchored rv_fn -> weight table -> smoothing
    offsets = sd.build_bip_count_offsets(mlb, LG, SCALE)
    rv_fn = make_rv_xrv(LG, SCALE, offsets)
    table = sd.shrink_table(sd.build_weight_table(mlb, rv_fn),
                            sd.zone_level_means(mlb, rv_fn))
    # v2 machinery: Loc+ surfaces on the slice
    S = lp.build_surfaces([p for p in pitches if lp.is_eligible_baseline(p)],
                          LG, SCALE)
    # league zone mix over eligible decisions (shared by both aggregations)
    lg_zone_n = defaultdict(int)
    for p in mlb:
        if p.get('Description') in SWING or p.get('Description') in TAKE:
            z = classify_zone(p)
            if z:
                lg_zone_n[z] += 1
    tot = sum(lg_zone_n.values())
    lg_zone_w = {z: n / tot for z, n in lg_zone_n.items()}

    ship_out, v2_out = {}, {}
    for hitter, ps in by_hitter.items():
        ship_z, v2_z = defaultdict(list), defaultdict(list)
        n = 0
        for p in ps:
            z = classify_zone(p)
            if z is None:
                continue
            desc = p.get('Description')
            swung = desc in SWING
            # shipped: the production dv function against the slice's table
            try:
                dv_ship = sd.compute_dv(p, table)
            except KeyError:
                continue
            bv = branch_values(p, S)
            if dv_ship is None or bv is None:
                continue
            dv_v2 = (bv[0] - bv[1]) if swung else (bv[1] - bv[0])
            ship_z[z].append(dv_ship)
            v2_z[z].append(dv_v2)
            n += 1
        if n < min_n:
            continue
        a = mix_neutral(ship_z, lg_zone_w)
        b = mix_neutral(v2_z, lg_zone_w)
        if a is not None and b is not None:
            ship_out[hitter] = a
            v2_out[hitter] = b
    return ship_out, v2_out


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    dec = [p for p in base if eligible_decision(p)]
    by_hitter_all = defaultdict(list)
    for p in dec:
        by_hitter_all[p.get('Batter')].append(p)
    print(f"{len(dec)} decision pitches, {len(by_hitter_all)} hitters",
          file=sys.stderr)

    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    parity = {d: i % 2 for i, d in enumerate(dates)}
    mid = dates[len(dates) // 2]

    # objective 1 — split-half reliability
    rel = {}
    halves = []
    for h in (0, 1):
        slice_p = [p for p in base if parity.get(p.get('Game Date')) == h]
        byh = defaultdict(list)
        for p in slice_p:
            if eligible_decision(p):
                byh[p.get('Batter')].append(p)
        halves.append(score_slice(slice_p, byh, MIN_REL))
        print(f"half {h} scored", file=sys.stderr)
    for name, idx in (('shipped', 0), ('v2', 1)):
        a, b = halves[0][idx], halves[1][idx]
        keys = [k for k in a if k in b]
        rel[name] = (pearson([a[k] for k in keys], [b[k] for k in keys]), len(keys))

    # objective 2 — first-half score vs second-half actual hitter xRV
    early = [p for p in base if p.get('Game Date') and p.get('Game Date') < mid]
    late = [p for p in base if p.get('Game Date') and p.get('Game Date') >= mid]
    byh_e = defaultdict(list)
    for p in early:
        if eligible_decision(p):
            byh_e[p.get('Batter')].append(p)
    ship_e, v2_e = score_slice(early, byh_e, MIN_PRED)
    rv_plain = make_rv_xrv(LG, SCALE)
    acc = defaultdict(lambda: [0.0, 0])
    for p in late:
        v = rv_plain(p)
        if v is not None:
            a = acc[p.get('Batter')]; a[0] += v; a[1] += 1
    actual = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_ACT}
    pred = {}
    for name, sc in (('shipped', ship_e), ('v2', v2_e)):
        keys = [k for k in sc if k in actual]
        pred[name] = (pearson([sc[k] for k in keys], [actual[k] for k in keys]),
                      len(keys))
    # agreement between the two scores
    kk = [k for k in ship_e if k in v2_e]
    agree = pearson([ship_e[k] for k in kk], [v2_e[k] for k in kk])

    print()
    print(f"{'model':>9s} {'rel_r':>7s} {'n_rel':>6s} {'pred_r':>7s} {'n_pred':>7s}")
    print('-' * 42)
    for name in ('shipped', 'v2'):
        print(f"{name:>9s} {rel[name][0]:>7.3f} {rel[name][1]:>6d} "
              f"{pred[name][0]:>7.3f} {pred[name][1]:>7d}")
    print()
    print(f"shipped-vs-v2 score agreement r = {agree:.3f}")
    print()
    print("If v2 wins both objectives here, the next step is NOT shipping — it")
    print("is the 2021-2025 replicate test (per the multi-season standard).")
    print("If it loses or splits, the 360-cell table is adequate and this")
    print("direction closes.")


if __name__ == '__main__':
    main()
