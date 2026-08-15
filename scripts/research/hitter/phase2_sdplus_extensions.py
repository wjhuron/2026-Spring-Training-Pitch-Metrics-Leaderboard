"""phase2_sdplus_extensions.py — test the two optional SD+ extensions on top
of the adopted Phase 2 config (count-anchored offsets + heart=1/6):

  1. cat3: split cells by pitch category (FB/BRK/OFF) -> 360 cells with a
     shrinkage cascade cell -> (zone x cat) -> zone.
  2. mixneutral: reweight the hitter's per-zone mean dv to the LEAGUE zone
     distribution, so facing a more separable pitch diet (more heart+waste,
     fewer coin-flip shadow pitches) stops leaking into the score.

Metrics: split-half reliability (odd/even dates), implied n0, corr vs wRC+.

Usage: python3 scripts/research/hitter/phase2_sdplus_extensions.py
"""
import os, sys, pickle, math, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import pipeline.sdplus as sd
from pipeline.sdplus import (
    is_eligible, classify_zone, classify_decision, get_count,
    make_rv_xrv, build_bip_count_offsets, ZONES, COUNTS,
)

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
HL = os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393
K = 50

FB_TYPES = {'FF', 'SI', 'FC', 'FA'}
OFF_TYPES = {'CH', 'FS', 'SC', 'KN'}


def cat_of(p):
    pt = p.get('Pitch Type')
    if pt in FB_TYPES:
        return 'FB'
    if pt in OFF_TYPES:
        return 'OFF'
    return 'BRK'


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)
            if sx > 0 and sy > 0 else None)


def build_table(pitches, cat3):
    offsets = build_bip_count_offsets(pitches, LG_WOBA, WOBA_SCALE)
    rv_fn = make_rv_xrv(LG_WOBA, WOBA_SCALE, offsets)
    cells = defaultdict(lambda: [0.0, 0])
    zc = defaultdict(lambda: [0.0, 0])   # (zone, cat, decision)
    z = defaultdict(lambda: [0.0, 0])    # (zone, decision)
    for p in pitches:
        rv = rv_fn(p)
        if rv is None:
            continue
        zone, dec, c = classify_zone(p), classify_decision(p), get_count(p)
        cat = cat_of(p) if cat3 else None
        cells[(zone, c, cat, dec)][0] += rv; cells[(zone, c, cat, dec)][1] += 1
        zc[(zone, cat, dec)][0] += rv; zc[(zone, cat, dec)][1] += 1
        z[(zone, dec)][0] += rv; z[(zone, dec)][1] += 1
    table = {}
    cats = ('FB', 'BRK', 'OFF') if cat3 else (None,)
    for zone in ZONES:
        for c in COUNTS:
            for cat in cats:
                for dec in ('swing', 'take'):
                    zs, zn = z.get((zone, dec), (0.0, 0))
                    zmean = zs / zn if zn else 0.0
                    zcs, zcn = zc.get((zone, cat, dec), (0.0, 0))
                    zcmean = (zcn * (zcs / zcn) + K * zmean) / (zcn + K) if zcn else zmean
                    cs, cn = cells.get((zone, c, cat, dec), (0.0, 0))
                    cmean = (cn * (cs / cn) + K * zcmean) / (cn + K) if cn else zcmean
                    table[(zone, c, cat, dec)] = cmean
    return table


def hitter_raw(pitches, table, cat3, mixneutral, lg_zone_w):
    elig = [p for p in pitches if is_eligible(p)]
    if not elig:
        return None, 0
    if not mixneutral:
        tot = 0.0
        for p in elig:
            zone, dec, c = classify_zone(p), classify_decision(p), get_count(p)
            cat = cat_of(p) if cat3 else None
            sw = table[(zone, c, cat, 'swing')]
            tk = table[(zone, c, cat, 'take')]
            tot += (sw - tk) if dec == 'swing' else (tk - sw)
        return tot / len(elig), len(elig)
    zone_acc = defaultdict(lambda: [0.0, 0])
    for p in elig:
        zone, dec, c = classify_zone(p), classify_decision(p), get_count(p)
        cat = cat_of(p) if cat3 else None
        sw = table[(zone, c, cat, 'swing')]
        tk = table[(zone, c, cat, 'take')]
        dv = (sw - tk) if dec == 'swing' else (tk - sw)
        zone_acc[zone][0] += dv; zone_acc[zone][1] += 1
    wsum = sum(lg_zone_w[zn] for zn in zone_acc)
    if wsum <= 0:
        return None, 0
    val = sum((s / n) * lg_zone_w[zn] for zn, (s, n) in zone_acc.items() if n) / wsum
    return val, len(elig)


def main():
    D = pickle.load(open(PKL, 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    elig_all = [p for p in mlb if is_eligible(p)]
    lg_zone_w = defaultdict(int)
    for p in elig_all:
        lg_zone_w[classify_zone(p)] += 1
    tot = sum(lg_zone_w.values())
    lg_zone_w = {zn: n / tot for zn, n in lg_zone_w.items()}

    by_h = defaultdict(list)
    for p in D:
        h, t = p.get('Batter'), p.get('BTeam')
        if h and t:
            by_h[(h, t)].append(p)
    hl = json.load(open(HL))
    wrc = {(r['hitter'], r['team']): r['wRCplus'] for r in hl
           if r.get('wRCplus') is not None and not r.get('_isROC')}

    dates = sorted({p.get('Game Date') for p in elig_all if p.get('Game Date')})
    half_of = {d: i % 2 for i, d in enumerate(dates)}

    for name, cat3, mixn in [('adopted', False, False), ('cat3', True, False),
                             ('mixneutral', False, True), ('cat3+mixn', True, True)]:
        tables_h = []
        for h in (0, 1):
            sub = [p for p in elig_all if half_of.get(p.get('Game Date')) == h]
            tables_h.append(build_table(sub, cat3))
        full_table = build_table(elig_all, cat3)

        halves = ({}, {})
        full = {}
        for key, ps in by_h.items():
            v, n = hitter_raw(ps, full_table, cat3, mixn, lg_zone_w)
            if v is not None:
                full[key] = (v, n)
            for h in (0, 1):
                sub = [p for p in ps if half_of.get(p.get('Game Date')) == h]
                v, n = hitter_raw(sub, tables_h[h], cat3, mixn, lg_zone_w)
                if v is not None:
                    halves[h][key] = (v, n)

        res = {'name': name}
        for mh in (110, 220):
            xs, ys, ns = [], [], []
            for key in halves[0]:
                if key in halves[1]:
                    v0, n0_ = halves[0][key]; v1, n1_ = halves[1][key]
                    if n0_ >= mh and n1_ >= mh:
                        xs.append(v0); ys.append(v1); ns.append((n0_ + n1_) / 2)
            r = pearson(xs, ys)
            res[f'r@{mh}'] = (round(r, 3) if r else None, len(xs))
            if r and 0 < r < 1 and ns:
                res[f'n0@{mh}'] = round((sum(ns) / len(ns)) * (1 - r) / r)
        xs = [v for k, (v, n) in full.items() if n >= 200 and k in wrc]
        ys = [wrc[k] for k, (v, n) in full.items() if n >= 200 and k in wrc]
        r = pearson(xs, ys)
        res['r_wrc'] = (round(r, 3) if r else None, len(xs))
        print(res, flush=True)


if __name__ == '__main__':
    main()
