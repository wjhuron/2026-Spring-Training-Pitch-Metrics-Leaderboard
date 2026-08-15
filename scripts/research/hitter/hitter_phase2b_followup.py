"""hitter_phase2b_followup.py — follow-ups to hitter_phase2_multiseason.py.

Three questions, same replicate protocol (rel: 2021-2026 x 3 seeds; pred:
4 pairs 21->22..24->25 vs next-season wOBA):

1. SD+ count anchor — mechanism. NOANCHOR won rel 6/6 and pred 3/4. Is that
   a real gain, or count-mix leak (no-anchor lets the hitter's count
   distribution into the score)? Discriminator: repeat the comparison under
   ZONE x COUNT mix-neutral aggregation (reweight per-(zone,count) mean dv
   to the league (zone,count) distribution). If NOANCHOR's edge vanishes
   under count-neutral aggregation, the edge was count mix; if it survives,
   the anchor itself is hurting (estimated offsets add noise).
     sd_BASE, sd_NOANCHOR, sd_BASE_ZC, sd_NOANCHOR_ZC

2. CT+ pitch-category cells (never tested; SD+'s cat3 won, CT+ was left at
   zone x count with a "<1% residual variance" claim from 2026):
     ct_BASE, ct_CAT3 (cells (zone,count,cat), cascade cell->(zone,cat)->zone,
     k=200 each level)

3. BB+ platoon-exposure neutrality (never tested): hitter xwOBAcon computed
   per pitcher hand and reweighted to the league L/R BIP exposure, so an
   unusually favorable platoon diet doesn't inflate the raw.
     bb_BASE, bb_HANDMIX

Usage: python3 scripts/research/hitter/hitter_phase2b_followup.py
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
import hitter_phase2_multiseason as H

SEEDS = (0, 1, 2)
HALF_MIN_DEC, HALF_MIN_SW, HALF_MIN_BIP = 125, 45, 40
FULL_MIN_DEC, FULL_MIN_SW, FULL_MIN_BIP = 200, 65, 80
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
OUT_JSON = os.path.join(ROOT, 'data', '_hitter_phase2b_results.json')

KEYS = ('sd_BASE', 'sd_NOANCHOR', 'sd_BASE_ZC', 'sd_NOANCHOR_ZC',
        'ct_BASE', 'ct_CAT3', 'bb_BASE', 'bb_HANDMIX')


def sd_score_zc(by_hitter, table, lg_zc_w, min_n):
    """Zone x count mix-neutral SD raw: per-(zone,count) mean dv reweighted
    to the league (zone,count) distribution."""
    out = {}
    for h, pitches in by_hitter.items():
        if len(pitches) < min_n:
            continue
        cell_dvs = defaultdict(list)
        for p in pitches:
            cell_dvs[(sd.classify_zone(p), sd.get_count(p))].append(
                sd.compute_dv(p, table))
        cmeans = {c: sum(v) / len(v) for c, v in cell_dvs.items()}
        wsum = sum(lg_zc_w.get(c, 0.0) for c in cmeans)
        if wsum <= 0:
            continue
        out[h] = sum(m * lg_zc_w.get(c, 0.0) for c, m in cmeans.items()) / wsum
    return out


def ct_cat3_tables(swings, rv_fn, k=200):
    """CT+ cell table keyed (zone, count, cat), cascade shrinkage
    cell -> (zone, cat) -> zone for all three quantities."""
    def acc():
        return {'n_swings': 0, 'n_whiff': 0, 'sum_rv_contact': 0.0,
                'sum_rv_whiff': 0.0}
    cells = defaultdict(acc)
    zc_l = defaultdict(acc)
    z_l = defaultdict(acc)
    for p in swings:
        zone = sd.classify_zone(p)
        count = sd.get_count(p)
        cat = sd.cat_of(p)
        outcome = ct.classify_contact_outcome(p)
        rv = rv_fn(p)
        if rv is None:
            continue
        for c in (cells[(zone, count, cat)], zc_l[(zone, cat)], z_l[zone]):
            c['n_swings'] += 1
            if outcome == 'whiff':
                c['n_whiff'] += 1
                c['sum_rv_whiff'] += rv
            else:
                c['sum_rv_contact'] += rv

    def stats(c):
        n_sw, n_wh = c['n_swings'], c['n_whiff']
        n_ct = n_sw - n_wh
        return {'n': n_sw,
                'p_whiff': (n_wh / n_sw) if n_sw else 0.0,
                'rv_contact': (c['sum_rv_contact'] / n_ct) if n_ct else 0.0,
                'rv_whiff': (c['sum_rv_whiff'] / n_wh) if n_wh else 0.0}

    DEFAULT = {'n': 0, 'p_whiff': 0.25, 'rv_contact': 0.0, 'rv_whiff': -0.05}
    out = {}
    for zone in sd.ZONES:
        zst = stats(z_l[zone]) if zone in z_l else DEFAULT
        for cat in sd.CATS:
            zcst = stats(zc_l[(zone, cat)]) if (zone, cat) in zc_l else DEFAULT
            n_zc = zcst['n']
            zc_shrunk = {q: ((n_zc * zcst[q] + k * zst[q]) / (n_zc + k))
                         for q in ('p_whiff', 'rv_contact', 'rv_whiff')}
            for count in sd.COUNTS:
                key = (zone, count, cat)
                cst = stats(cells[key]) if key in cells else DEFAULT
                n = cst['n']
                out[key] = {q: ((n * cst[q] + k * zc_shrunk[q]) / (n + k))
                            for q in ('p_whiff', 'rv_contact', 'rv_whiff')}
                out[key]['n_swings'] = n
    return out


def ct_score_cat3(by_hitter_sw, table, min_n):
    out = {}
    for h, swings in by_hitter_sw.items():
        if len(swings) < min_n:
            continue
        A_ = E = 0.0
        for p in swings:
            cell = table[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p))]
            lev = cell['rv_contact'] - cell['rv_whiff']
            if lev <= 0:
                continue
            con = 1 if ct.classify_contact_outcome(p) == 'contact' else 0
            A_ += lev * con
            E += lev * (1.0 - cell['p_whiff'])
        if E > 0:
            out[h] = A_ / E
    return out


def bb_raws(elig, min_bip):
    """bb_BASE: mean xwOBA on non-bunt BIP. bb_HANDMIX: per pitcher hand,
    reweighted to league L/R BIP exposure."""
    lg_hand = defaultdict(int)
    per = defaultdict(lambda: defaultdict(list))
    for p in elig:
        if p.get('Description') != 'In Play':
            continue
        if p.get('BBType') in sd.BUNT_BB_TYPES:
            continue
        xw = sd.safe_float(p.get('xwOBA'))
        th = p.get('Throws')
        h = p.get('Batter')
        if xw is None or th not in ('L', 'R') or not h:
            continue
        lg_hand[th] += 1
        per[h][th].append(xw)
    tot = sum(lg_hand.values())
    if not tot:
        return {}, {}
    wL = lg_hand['L'] / tot
    base, handmix = {}, {}
    for h, d in per.items():
        allx = d['L'] + d['R']
        if len(allx) < min_bip:
            continue
        base[h] = sum(allx) / len(allx)
        if d['L'] and d['R']:
            mL = sum(d['L']) / len(d['L'])
            mR = sum(d['R']) / len(d['R'])
            handmix[h] = wL * mL + (1 - wL) * mR
        else:
            handmix[h] = base[h]
    return base, handmix


def season_components(elig, lg, sc, min_dec, min_sw, min_bip):
    res = {}
    with H.patched('_z16', True):
        by_hitter = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                by_hitter[h].append(p)
        swings = [p for p in elig if ct.is_ct_eligible(p)]
        by_hitter_sw = defaultdict(list)
        for p in swings:
            h = p.get('Batter')
            if h:
                by_hitter_sw[h].append(p)

        # league mixes
        zc_zone = defaultdict(int)
        zc_cell = defaultdict(int)
        for p in elig:
            z = sd.classify_zone(p)
            zc_zone[z] += 1
            zc_cell[(z, sd.get_count(p))] += 1
        tz = sum(zc_zone.values())
        lgw_zone = {z: n / tz for z, n in zc_zone.items()}
        tc = sum(zc_cell.values())
        lgw_zc = {c: n / tc for c, n in zc_cell.items()}

        for anchor, tag in ((True, ''), (False, 'NOANCHOR')):
            offsets = sd.build_bip_count_offsets(elig, lg, sc) if anchor else None
            rv_fn = sd.make_rv_xrv(lg, sc, offsets)
            raw = sd.build_weight_table(elig, rv_fn)
            zm = sd.zone_level_means(elig, rv_fn)
            table = sd.shrink_table(raw, zm)
            name = 'sd_BASE' if anchor else 'sd_NOANCHOR'
            res[name] = H.sd_score(by_hitter, table, lgw_zone, min_dec)
            res[name + '_ZC'] = sd_score_zc(by_hitter, table, lgw_zc, min_dec)

        # CT variants (anchored rv, per shipped)
        offsets = ct.build_bip_count_offsets(swings, lg, sc)
        rv_fn = ct.make_rv_xrv(lg, sc, offsets)
        craw = ct.build_contact_cell_weights(swings, rv_fn)
        czm = ct.zone_level_contact_means(swings, rv_fn)
        ctab = ct.shrink_contact_cells(craw, czm)
        res['ct_BASE'] = H.ct_score(by_hitter_sw, ctab, min_sw, lift=True)
        ctab3 = ct_cat3_tables(swings, rv_fn)
        res['ct_CAT3'] = ct_score_cat3(by_hitter_sw, ctab3, min_sw)

        res['bb_BASE'], res['bb_HANDMIX'] = bb_raws(elig, min_bip)
    return res


def main():
    results = {'split_half': {}, 'predictive': {}}
    agg = defaultdict(list)
    print(f"SPLIT-HALF RELIABILITY (floors {HALF_MIN_DEC}/{HALF_MIN_SW}/"
          f"{HALF_MIN_BIP})", flush=True)
    for year in SEASONS:
        P = H.load_season(year)
        elig = H.precompute(P)
        lg, sc = H.guts(year)
        dates = sorted({p.get('Game Date') for p in elig if p.get('Game Date')})
        for seed in SEEDS:
            rnd = random.Random(seed)
            sh = dates[:]
            rnd.shuffle(sh)
            ha = set(sh[:len(sh) // 2])
            Ea = [p for p in elig if p.get('Game Date') in ha]
            Eb = [p for p in elig if p.get('Game Date') and p.get('Game Date') not in ha]
            ra = season_components(Ea, lg, sc, HALF_MIN_DEC, HALF_MIN_SW, HALF_MIN_BIP)
            rb = season_components(Eb, lg, sc, HALF_MIN_DEC, HALF_MIN_SW, HALF_MIN_BIP)
            row = {}
            for k in KEYS:
                common = [h for h in ra[k] if h in rb[k]]
                r = H.pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                row[k] = [r, len(common)]
                if r is not None:
                    agg[k].append((year, seed, r))
            results['split_half'][f'{year}_s{seed}'] = row
            print(f"  {year} seed{seed}: " + '  '.join(
                f"{k}={row[k][0]:.3f}" if row[k][0] is not None else f"{k}=NA"
                for k in KEYS), flush=True)
        del P, elig
        import gc
        gc.collect()

    print("\n  MEAN split-half r:")
    for k in KEYS:
        rs = [r for _, _, r in agg[k]]
        by_season = defaultdict(list)
        for y, _, r in agg[k]:
            by_season[y].append(r)
        seas = '  '.join(f"{y}:{sum(v)/len(v):.3f}" for y, v in sorted(by_season.items()))
        print(f"    {k}: mean {sum(rs)/len(rs):.4f}   {seas}")

    print(f"\nPREDICTIVE (floors {FULL_MIN_DEC}/{FULL_MIN_SW}/{FULL_MIN_BIP})",
          flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = H.load_season(yn)
        elig = H.precompute(P)
        lg, sc = H.guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_DEC, FULL_MIN_SW, FULL_MIN_BIP)
        y_map = A.target_y(yn1)
        row = {}
        for k in KEYS:
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = H.pearson(xs, ys)
            row[k] = [r, len(xs)]
            if r is not None:
                pagg[k].append((yn, r))
        results['predictive'][f'{yn}_{yn1}'] = row
        print(f"  {yn}->{yn1}: " + '  '.join(
            f"{k}={row[k][0]:+.3f}" if row[k][0] is not None else f"{k}=NA"
            for k in KEYS), flush=True)
        del P, elig

    print("\n  MEAN predictive r:")
    for k in KEYS:
        rs = [r for _, r in pagg[k]]
        if rs:
            print(f"    {k}: {sum(rs)/len(rs):+.4f}")

    print("\nDELTAS OF INTEREST (per-season rel means)")
    def delta(k1, k0):
        d = {}
        for y in SEASONS:
            a = [r for yy, _, r in agg[k1] if yy == y]
            b = [r for yy, _, r in agg[k0] if yy == y]
            if a and b:
                d[y] = sum(a) / len(a) - sum(b) / len(b)
        pd_ = None
        p1 = [r for _, r in pagg[k1]]
        p0 = [r for _, r in pagg[k0]]
        if p1 and p0:
            pd_ = sum(p1) / len(p1) - sum(p0) / len(p0)
        wins = sum(1 for v in d.values() if v > 0)
        cells = '  '.join(f"{y}:{v:+.3f}" for y, v in sorted(d.items()))
        print(f"  {k1} - {k0}: rel wins {wins}/{len(d)}  {cells}"
              + (f"   pred {pd_:+.4f}" if pd_ is not None else ""))

    delta('sd_NOANCHOR', 'sd_BASE')          # plain aggregation (replicates 2a)
    delta('sd_NOANCHOR_ZC', 'sd_BASE_ZC')    # count-neutral aggregation
    delta('sd_BASE_ZC', 'sd_BASE')           # what count-neutrality itself does
    delta('ct_CAT3', 'ct_BASE')
    delta('bb_HANDMIX', 'bb_BASE')

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == '__main__':
    main()
