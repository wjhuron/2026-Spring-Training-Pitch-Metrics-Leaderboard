"""sdplus_personal_damage.py — does hitter-personalized swing-branch damage
improve SD+?

Context. Shipped SD+ is already SEAGER-shaped: a count-conditional,
luck-neutral swing-vs-take run-value table, mix-neutral aggregation. The
continuous-surface refinement was tested and REJECTED 2026-07-28 with a
mechanism (spatial error cancels in the swing-take difference). The one
SEAGER component NOT in the shipped metric is PERSONALIZATION: a swing at
a location is a better decision for a hitter who does damage THERE.

Variant. Per hitter h and zone z, over h's own eligible swings:
    d_h(z)   = mean[ rv(p) - league swing cell value ]   (shrunk, n/(n+K))
    shape(z) = d_h(z) - swing-count-weighted mean of d_h  (CENTERED, so
               overall contact quality — BB+ territory — cannot enter;
               only the spatial SHAPE of damage personalizes decisions)
    dv_p(p)  = dv_base(p) + shape(z) on swings, - shape(z) on takes
Aggregation, floors, and mix-neutrality are the shipped path unchanged.

Protocol (mirrors hitter_phase2_multiseason):
  reliability  split-half by game date, 3 seeds, per-half floor 125
               decisions, EVERYTHING (league table + shape) rebuilt
               self-contained per half — no cross-half leakage
  prediction   full-season raw score (floor 200 decisions) vs year-N+1
               wOBA (>= 200 PA events), pairs 2021->22 .. 2024->25
  contamination control: partial r of the personalized score with next
               wOBA CONTROLLING for the BASE score — if the pred gain
               vanishes there, the shape is smuggling outcome quality,
               not decision skill, and the variant is rejected
K grid: 25, 50, 100, 200, 400 swings (BASE = no personalization).
Adoption bar: a K must win reliability in most seasons AND not lose
prediction, with an interior optimum or a proven-flat curve.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/sdplus_personal_damage.py
Output: console + data/_sdplus_personal_results.json
"""
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
SEEDS = (0, 1, 2)
HALF_MIN_DEC = 125
FULL_MIN_DEC = 200
MIN_PA_NEXT = 200
K_GRID = [25, 50, 100, 200, 400]
GUTS_2026 = (0.3172, 1.2343)
OUT = os.path.join(ROOT, 'data', '_sdplus_personal_results.json')

WOBA_W = {'Walk': .69, 'Hit By Pitch': .72, 'Single': .89, 'Double': 1.27,
          'Triple': 1.61, 'Home Run': 2.10,
          'walk': .69, 'hit_by_pitch': .72, 'single': .89, 'double': 1.27,
          'triple': 1.61, 'home_run': 2.10}
NON_PA_TOKENS = ('stealing', 'pickoff', 'stolen', 'wild_pitch',
                 'passed_ball', 'truncated', 'game_advisory',
                 'Wild Pitch', 'Passed Ball', 'Stolen Base',
                 'Caught Stealing', 'Pickoff')
IBB = {'Intent Walk', 'intent_walk'}


def pearson(xs, ys):
    n = len(xs)
    if n < 20:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def partial_r(y, x, z):
    """r(y, x | z) via residual correlation."""
    ryx, ryz, rxz = pearson(y, x), pearson(y, z), pearson(x, z)
    if None in (ryx, ryz, rxz):
        return None
    den = math.sqrt((1 - ryz ** 2) * (1 - rxz ** 2))
    return (ryx - ryz * rxz) / den if den > 0 else None


def load_season(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(
            ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB']
    return A.season_dicts(year)


def guts(year):
    return GUTS_2026 if year == 2026 else A.GUTS[year]


def atoms(pitches, rv_fn):
    """One compact tuple per eligible pitch:
    (hitter, date, zone, count, cat, decision, rv). Also wOBA aggregates
    per hitter for the prediction target."""
    out = []
    woba = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        ev = p.get('Event')
        tok = ev if ev in IBB else (p.get('event_raw') or ev)
        if tok and not any(t in str(tok) for t in NON_PA_TOKENS):
            if tok not in IBB:
                w = woba[p.get('Batter')]
                w[0] += WOBA_W.get(tok, 0.0)
                w[1] += 1
        if not sd.is_eligible(p):
            continue
        rv = rv_fn(p)
        if rv is None:
            continue
        out.append((p.get('Batter'), p.get('Game Date'),
                    sd.classify_zone(p), sd.get_count(p), sd._sd_cat(p),
                    sd.classify_decision(p), rv))
    wt = {h: s / n for h, (s, n) in woba.items() if n >= MIN_PA_NEXT}
    return out, wt


def league_tables(rows):
    """Shipped cascade on atom tuples: smoothed cell table + zone weights."""
    raw = defaultdict(lambda: [0.0, 0])
    zc = defaultdict(lambda: [0.0, 0])
    zz = defaultdict(lambda: [0.0, 0])
    zone_n = defaultdict(int)
    for _, _, zone, count, cat, dec, rv in rows:
        raw[(zone, count, cat, dec)][0] += rv
        raw[(zone, count, cat, dec)][1] += 1
        zc[(zone, cat, dec)][0] += rv
        zc[(zone, cat, dec)][1] += 1
        zz[(zone, dec)][0] += rv
        zz[(zone, dec)][1] += 1
        zone_n[zone] += 1
    raw_t = {k: (s / n, n) for k, (s, n) in raw.items()}
    zc_t = {k: (s / n, n) for k, (s, n) in zc.items()}
    zz_t = {k: (s / n, n) for k, (s, n) in zz.items()}
    smoothed = sd.shrink_table(raw_t, (zc_t, zz_t))
    tot = sum(zone_n.values())
    lg_zone_w = {z: n / tot for z, n in zone_n.items()}
    return smoothed, lg_zone_w


def hitter_aggs(rows, table):
    """Per hitter: per-zone dv sums, n, swing-take imbalance, and per-zone
    swing residual sums (vs the league swing cell) for the shape."""
    H = {}
    for h, _, zone, count, cat, dec, rv in rows:
        swing_rv = table[(zone, count, cat, 'swing')][0]
        take_rv = table[(zone, count, cat, 'take')][0]
        dv = (swing_rv - take_rv) if dec == 'swing' else (take_rv - swing_rv)
        r = H.setdefault(h, {})
        z = r.setdefault(zone, [0.0, 0, 0, 0.0, 0])
        # [dv_sum, n, n_swing, swing_resid_sum, n_swing_resid]
        z[0] += dv
        z[1] += 1
        if dec == 'swing':
            z[2] += 1
            z[3] += rv - swing_rv
            z[4] += 1
    return H


def score(aggs, lg_zone_w, K):
    """Mix-neutral raw score per hitter; K = shape pseudo-obs, None = BASE."""
    out = {}
    for h, zones in aggs.items():
        n_dec = sum(z[1] for z in zones.values())
        shape = {}
        if K is not None:
            d = {}
            wsum = 0
            for zn, z in zones.items():
                n_sw = z[4]
                d[zn] = (z[3] / (n_sw + K)) if (n_sw + K) > 0 else 0.0
                # z[3]/(n+K) == shrunk mean: (n * mean) / (n + K)
                wsum += n_sw
            center = (sum(d[zn] * zones[zn][4] for zn in d) / wsum
                      if wsum > 0 else 0.0)
            shape = {zn: d[zn] - center for zn in d}
        zmeans = {}
        for zn, z in zones.items():
            m = z[0] / z[1]
            if K is not None:
                sh = shape.get(zn, 0.0)
                m += sh * (2 * z[2] - z[1]) / z[1]   # +sh swings, -sh takes
            zmeans[zn] = m
        w = sum(lg_zone_w.get(zn, 0.0) for zn in zmeans)
        raw = (sum(m * lg_zone_w.get(zn, 0.0) for zn, m in zmeans.items())
               / w if w > 0 else sum(z[0] for z in zones.values()) / n_dec)
        out[h] = (raw, n_dec)
    return out


def main():
    res = {'rel': {}, 'pred': {}}
    full_scores, season_woba = {}, {}
    for y in SEASONS:
        lg, sc = guts(y)
        rv_fn = sd.make_rv_xrv(lg, sc)
        pitches = load_season(y)
        rows, wt = atoms(pitches, rv_fn)
        del pitches
        season_woba[y] = wt
        print(f'{y}: {len(rows)} eligible decisions, '
              f'{len(wt)} wOBA hitters', flush=True)

        # ── reliability ──
        dates = sorted({r[1] for r in rows})
        for seed in SEEDS:
            rnd = random.Random(seed * 1000 + y)
            half_of = {d: (rnd.random() < 0.5) for d in dates}
            halves = ([r for r in rows if half_of[r[1]]],
                      [r for r in rows if not half_of[r[1]]])
            scored = []
            for hrows in halves:
                tbl, zw = league_tables(hrows)
                ag = hitter_aggs(hrows, tbl)
                scored.append({K: score(ag, zw, K)
                               for K in [None] + K_GRID})
            for K in [None] + K_GRID:
                a, b = scored[0][K], scored[1][K]
                xs, ys = [], []
                for h in a:
                    if h in b and a[h][1] >= HALF_MIN_DEC \
                            and b[h][1] >= HALF_MIN_DEC:
                        xs.append(a[h][0])
                        ys.append(b[h][0])
                r = pearson(xs, ys)
                res['rel'].setdefault(str(y), {}).setdefault(
                    'BASE' if K is None else str(K), []).append(r)

        # full-season scores for the prediction pass; rows freed after
        tbl, zw = league_tables(rows)
        ag = hitter_aggs(rows, tbl)
        full_scores[y] = {K: score(ag, zw, K) for K in [None] + K_GRID}
        del rows

    # ── prediction ──
    for y0, y1 in PAIRS:
        base = full_scores[y0][None]
        nxt = season_woba[y1]
        hs = [h for h in base
              if base[h][1] >= FULL_MIN_DEC and h in nxt]
        yv = [nxt[h] for h in hs]
        bv = [base[h][0] for h in hs]
        r_base = pearson(bv, yv)
        rec = {'n': len(hs), 'BASE': r_base}
        for K in K_GRID:
            pv = [full_scores[y0][K][h][0] for h in hs]
            rec[str(K)] = pearson(pv, yv)
            rec[f'partial_{K}'] = partial_r(yv, pv, bv)
        res['pred'][f'{y0}->{y1}'] = rec

    # ── report ──
    print('\n=== RELIABILITY (split-half r, mean over 3 seeds) ===')
    for y in SEASONS:
        rr = res['rel'][str(y)]
        base = sum(rr['BASE']) / len(rr['BASE'])
        line = f'  {y}: BASE {base:+.4f}  '
        line += ' '.join(
            f'K{K} {sum(rr[str(K)]) / len(rr[str(K)]) - base:+.4f}'
            for K in K_GRID)
        print(line)
    print('\n=== PREDICTION (r vs next-season wOBA) ===')
    for pair, rec in res['pred'].items():
        line = f'  {pair} (n {rec["n"]}): BASE {rec["BASE"]:+.4f}  '
        line += ' '.join(f'K{K} {rec[str(K)] - rec["BASE"]:+.4f}'
                         f'(p{rec[f"partial_{K}"]:+.3f})'
                         for K in K_GRID)
        print(line)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f)
    os.replace(tmp, OUT)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
