"""ct_threeoutcome_multiseason.py — should CT+ separate foul from fair
contact? Shipped CT+ scores binary contact (foul + In Play pooled) with
leverage weighting. But a 2-strike foul (spoil, PA survives) and a ball in
play are different outcomes, and spoil-ability is plausibly a distinct
execution skill the binary form averages away.

Variant CT3 (outcome-over-expected, additive):
  cells (zone, count, cat) carry p_whiff / p_foul / p_fair given swing and
  class values rv_whiff / rv_foul / rv_fair (anchored rv class means),
  cascade-shrunk cell -> (zone,cat) -> zone, k=200 (same as shipped).
  Per swing: actual = rv of the ACTUAL outcome class (class value, never
  the pitch's own xwOBA — no BB+ leakage); expected = sum p_i * rv_i.
  raw_ct3 = mean(actual - expected) per swing (runs above expectation).

Metrics: rel 2021-2026 x 3 seeds; pred 4 pairs. Ship consideration only on
consistent predictive wins (the placebo standard: rel gains from changed
structure do not count on their own).

Usage: python3 scripts/research/hitter/ct_threeoutcome_multiseason.py
"""
import random
import sys
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
import hitter_phase2_multiseason as H

SEEDS = (0, 1, 2)
HALF_MIN_SW, FULL_MIN_SW = 45, 65
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
KEYS = ('ct_lift', 'ct3_ooe')


def outcome_class(p):
    d = p.get('Description')
    if d == 'Swinging Strike':
        return 'wh'
    if d == 'Foul':
        return 'fl'
    if d == 'In Play':
        return 'fair'
    return None


def build_ct3(swings, rv_fn, k=200):
    def acc():
        return {'n': 0, 'n_wh': 0, 'n_fl': 0,
                's_wh': 0.0, 's_fl': 0.0, 's_fair': 0.0}
    cells = defaultdict(acc)
    zc = defaultdict(acc)
    z_ = defaultdict(acc)
    for p in swings:
        oc = outcome_class(p)
        rv = rv_fn(p)
        if oc is None or rv is None:
            continue
        key3 = (sd.classify_zone(p), sd.get_count(p), sd.cat_of(p))
        for c in (cells[key3], zc[(key3[0], key3[2])], z_[key3[0]]):
            c['n'] += 1
            if oc == 'wh':
                c['n_wh'] += 1
                c['s_wh'] += rv
            elif oc == 'fl':
                c['n_fl'] += 1
                c['s_fl'] += rv
            else:
                c['s_fair'] += rv

    def stats(c):
        n, nw, nf = c['n'], c['n_wh'], c['n_fl']
        nfair = n - nw - nf
        return {'n': n,
                'p_wh': nw / n if n else 0.0,
                'p_fl': nf / n if n else 0.0,
                'rv_wh': c['s_wh'] / nw if nw else -0.05,
                'rv_fl': c['s_fl'] / nf if nf else 0.0,
                'rv_fair': c['s_fair'] / nfair if nfair else 0.0}

    DEFAULT = {'n': 0, 'p_wh': 0.25, 'p_fl': 0.4, 'rv_wh': -0.05,
               'rv_fl': 0.0, 'rv_fair': 0.0}
    QS = ('p_wh', 'p_fl', 'rv_wh', 'rv_fl', 'rv_fair')
    out = {}
    for zone in sd.ZONES:
        zst = stats(z_[zone]) if zone in z_ else DEFAULT
        for cat in sd.CATS:
            zcst = stats(zc[(zone, cat)]) if (zone, cat) in zc else DEFAULT
            nzc = zcst['n']
            mid = {q: (nzc * zcst[q] + k * zst[q]) / (nzc + k) for q in QS}
            for count in sd.COUNTS:
                key3 = (zone, count, cat)
                cst = stats(cells[key3]) if key3 in cells else DEFAULT
                n = cst['n']
                out[key3] = {q: (n * cst[q] + k * mid[q]) / (n + k)
                             for q in QS}
    return out


def ct3_score(by_sw, table, min_n):
    out = {}
    for h, swings in by_sw.items():
        if len(swings) < min_n:
            continue
        tot = m = 0.0
        for p in swings:
            oc = outcome_class(p)
            if oc is None:
                continue
            cell = table[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p))]
            exp = (cell['p_wh'] * cell['rv_wh'] + cell['p_fl'] * cell['rv_fl']
                   + (1 - cell['p_wh'] - cell['p_fl']) * cell['rv_fair'])
            act = cell['rv_wh'] if oc == 'wh' else (
                cell['rv_fl'] if oc == 'fl' else cell['rv_fair'])
            tot += act - exp
            m += 1
        if m >= min_n:
            out[h] = tot / m
    return out


def season_components(elig, lg, sc, min_sw):
    res = {}
    with H.patched('_z16', True):
        swings = [p for p in elig if ct.is_ct_eligible(p)]
        by_sw = defaultdict(list)
        for p in swings:
            h = p.get('Batter')
            if h:
                by_sw[h].append(p)
        offsets = ct.build_bip_count_offsets(swings, lg, sc)
        rv_fn = ct.make_rv_xrv(lg, sc, offsets)
        craw = ct.build_contact_cell_weights(swings, rv_fn)
        czm = ct.zone_level_contact_means(swings, rv_fn)
        tab = ct.shrink_contact_cells(craw, czm)
        res['ct_lift'] = H.ct_score(by_sw, tab, min_sw, lift=True)
        tab3 = build_ct3(swings, rv_fn)
        res['ct3_ooe'] = ct3_score(by_sw, tab3, min_sw)
    return res


def main():
    agg = defaultdict(list)
    print("SPLIT-HALF RELIABILITY", flush=True)
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
            ra = season_components(Ea, lg, sc, HALF_MIN_SW)
            rb = season_components(Eb, lg, sc, HALF_MIN_SW)
            row = []
            for k in KEYS:
                common = [h for h in ra[k] if h in rb[k]]
                r = H.pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                if r is not None:
                    agg[k].append((year, r))
                row.append(f"{k}={r:.3f}" if r is not None else f"{k}=NA")
            print(f"  {year} s{seed}: " + '  '.join(row), flush=True)
        del P, elig
        import gc
        gc.collect()

    print("\nMEAN split-half r:")
    for k in KEYS:
        rs = [r for _, r in agg[k]]
        by = defaultdict(list)
        for y, r in agg[k]:
            by[y].append(r)
        print(f"  {k}: {sum(rs)/len(rs):.4f}  "
              + '  '.join(f"{y}:{sum(v)/len(v):.3f}" for y, v in sorted(by.items())))

    print("\nPREDICTIVE (4 pairs, vs next-season wOBA >=200 events)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = H.load_season(yn)
        elig = H.precompute(P)
        lg, sc = H.guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_SW)
        y_map = A.target_y(yn1)
        row = []
        for k in KEYS:
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = H.pearson(xs, ys)
            if r is not None:
                pagg[k].append(r)
            row.append(f"{k}={r:+.3f}" if r is not None else f"{k}=NA")
        print(f"  {yn}->{yn1}: " + '  '.join(row), flush=True)
        del P, elig

    print("\nMEAN predictive r:")
    for k in KEYS:
        if pagg[k]:
            print(f"  {k}: {sum(pagg[k])/len(pagg[k]):+.4f}")
    print("\nAdoption consideration only on consistent predictive wins;")
    print("agreement corr and a composite check would follow a win.")


if __name__ == '__main__':
    main()
