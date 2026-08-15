"""hitter_meta_validation.py — the Loc+ lessons applied back to the hitter
atoms: a placebo control for the shipped CT+ cat3 (it added parameters and
was never placebo-tested), and a count-mix adjustment candidate for BB+.

1. CT+ cat3 placebo: rebuild the cat3 tables with pitch-category labels
   permuted among swings (seed-fixed), score with real categories. The real
   cat3 must beat the placebo; a winning placebo means the shipped gain was
   parameter flexibility, not category signal.
     ct_zc (zone x count) | ct_cat3 (real, SHIPPED) | ct_cat3_placebo

2. BB+ count-mix adjustment: xwOBAcon is count-confounded (2-strike
   defensive contact is weaker) — an early-count swinger's raw xwOBAcon is
   inflated by his count mix, and deep counts come partly from decisions
   (SD+ channel). Candidate: per-BIP standardized xw-value minus the league
   count offset (the xw_clevel table), then re-based to xwOBA units.
     bb_BASE | bb_CADJ
   Also reports corr(bb, sd_raw) both ways — the adjustment should REDUCE
   BB+/SD+ overlap if the confound is real.

Metrics: split-half rel 2021-2026 x 3 seeds; predictive 4 pairs.

Usage: python3 scripts/research/hitter/hitter_meta_validation.py
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
import hitter_phase2b_followup as F

SEEDS = (0, 1, 2)
HALF_MIN_SW, HALF_MIN_BIP, HALF_MIN_DEC = 45, 40, 125
FULL_MIN_SW, FULL_MIN_BIP, FULL_MIN_DEC = 65, 80, 200
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
PLACEBO_SEED = 17
KEYS = ('ct_zc', 'ct_cat3', 'ct_cat3_placebo', 'bb_BASE', 'bb_CADJ')


def xw_count_offsets(elig, lg, sc):
    acc = defaultdict(lambda: [0.0, 0])
    for p in elig:
        if p.get('Description') != 'In Play':
            continue
        if p.get('BBType') in sd.BUNT_BB_TYPES:
            continue
        xw = sd.safe_float(p.get('xwOBA'))
        c = sd.get_count(p)
        if xw is None or c is None:
            continue
        acc[c][0] += (xw - lg) / sc
        acc[c][1] += 1
    tot_s = sum(s for s, _ in acc.values())
    tot_n = sum(n for _, n in acc.values())
    overall = tot_s / tot_n if tot_n else 0.0
    return {c: ((s / n - overall) if n >= 200 else 0.0)
            for c, (s, n) in acc.items()}


def bb_raws(elig, lg, sc, min_bip):
    offs = xw_count_offsets(elig, lg, sc)
    per = defaultdict(lambda: ([], []))
    for p in elig:
        if p.get('Description') != 'In Play':
            continue
        if p.get('BBType') in sd.BUNT_BB_TYPES:
            continue
        xw = sd.safe_float(p.get('xwOBA'))
        c = sd.get_count(p)
        h = p.get('Batter')
        if xw is None or not h:
            continue
        base_l, adj_l = per[h]
        base_l.append(xw)
        adj_l.append(xw - (offs.get(c, 0.0) * sc if c else 0.0))
    base, adj = {}, {}
    for h, (bl, al) in per.items():
        if len(bl) >= min_bip:
            base[h] = sum(bl) / len(bl)
            adj[h] = sum(al) / len(al)
    return base, adj


def ct_score3(by_hitter_sw, table, min_n, catfn=None):
    """CT+ lift score against a cat3 table ((zone, count, cat) keys).
    catfn defaults to the real category."""
    if catfn is None:
        catfn = sd.cat_of
    out = {}
    for h, swings in by_hitter_sw.items():
        if len(swings) < min_n:
            continue
        A_ = E = 0.0
        for p in swings:
            cell = table[(sd.classify_zone(p), sd.get_count(p), catfn(p))]
            lev = cell['rv_contact'] - cell['rv_whiff']
            if lev <= 0:
                continue
            con = 1 if ct.classify_contact_outcome(p) == 'contact' else 0
            A_ += lev * con
            E += lev * (1.0 - cell['p_whiff'])
        if E > 0:
            out[h] = A_ / E
    return out


def season_components(elig, lg, sc, min_dec, min_sw, min_bip):
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

        # real cat3 (shipped pipeline path)
        craw = ct.build_contact_cell_weights(swings, rv_fn)
        czm = ct.zone_level_contact_means(swings, rv_fn)
        tab3 = ct.shrink_contact_cells(craw, czm)
        res['ct_cat3'] = ct_score3(by_sw, tab3, min_sw)

        # zone x count (pre-cat3): collapse cat via single-cat patch.
        # NOTE: pipeline_contact binds cat_of by name at import, so BOTH
        # module references must be patched or the build silently uses
        # real categories (the bug that voided the first run).
        orig_cat = sd.cat_of
        try:
            sd.cat_of = ct.cat_of = (lambda p: 'FB')
            craw0 = ct.build_contact_cell_weights(swings, rv_fn)
            czm0 = ct.zone_level_contact_means(swings, rv_fn)
            tab0 = ct.shrink_contact_cells(craw0, czm0)
            res['ct_zc'] = ct_score3(by_sw, tab0, min_sw,
                                     catfn=lambda p: 'FB')
        finally:
            sd.cat_of = ct.cat_of = orig_cat

        # placebo cat3: permuted category labels for BUILD, real for SCORE
        cats = [sd.cat_of(p) for p in swings]
        random.Random(PLACEBO_SEED).shuffle(cats)
        for p, cshuf in zip(swings, cats):
            p['_catshuf'] = cshuf
        try:
            sd.cat_of = ct.cat_of = (lambda p: p['_catshuf'])
            crawp = ct.build_contact_cell_weights(swings, rv_fn)
            czmp = ct.zone_level_contact_means(swings, rv_fn)
            tabp = ct.shrink_contact_cells(crawp, czmp)
        finally:
            sd.cat_of = ct.cat_of = orig_cat
        worst = max(abs(tabp[k]['p_whiff'] - tab3[k]['p_whiff'])
                    for k in tab3)
        assert worst > 1e-6, 'placebo tables identical to real — patch inert'
        res['ct_cat3_placebo'] = ct_score3(by_sw, tabp, min_sw)

        res['bb_BASE'], res['bb_CADJ'] = bb_raws(elig, lg, sc, min_bip)

        # SD raw for the orthogonality check (shipped config, no anchor)
        rvsd = sd.make_rv_xrv(lg, sc)
        raw = sd.build_weight_table(elig, rvsd)
        zm = sd.zone_level_means(elig, rvsd)
        table = sd.shrink_table(raw, zm)
        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}
        by_h = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                by_h[h].append(p)
        res['_sd'] = H.sd_score(by_h, table, lgw, min_dec)
    return res


def main():
    agg = defaultdict(list)
    orth = defaultdict(list)
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
            ra = season_components(Ea, lg, sc, HALF_MIN_DEC, HALF_MIN_SW, HALF_MIN_BIP)
            rb = season_components(Eb, lg, sc, HALF_MIN_DEC, HALF_MIN_SW, HALF_MIN_BIP)
            row = []
            for k in KEYS:
                common = [h for h in ra[k] if h in rb[k]]
                r = H.pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                if r is not None:
                    agg[k].append((year, r))
                row.append(f"{k}={r:.3f}" if r is not None else f"{k}=NA")
            print(f"  {year} s{seed}: " + '  '.join(row), flush=True)
        # orthogonality on the full season
        comp = season_components(elig, lg, sc, FULL_MIN_DEC, FULL_MIN_SW, FULL_MIN_BIP)
        for bk in ('bb_BASE', 'bb_CADJ'):
            common = [h for h in comp[bk] if h in comp['_sd']]
            r = H.pearson([comp[bk][h] for h in common],
                          [comp['_sd'][h] for h in common])
            if r is not None:
                orth[bk].append((year, r))
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

    print("\ncorr(BB raw, SD raw) per season (lower |r| = cleaner separation):")
    for bk in ('bb_BASE', 'bb_CADJ'):
        print(f"  {bk}: " + '  '.join(f"{y}:{r:+.3f}" for y, r in orth[bk]))

    print("\nPREDICTIVE (4 pairs, vs next-season wOBA >=200 events)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = H.load_season(yn)
        elig = H.precompute(P)
        lg, sc = H.guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_DEC, FULL_MIN_SW, FULL_MIN_BIP)
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

    print("\nVERDICT GUIDE: ct_cat3 must beat ct_cat3_placebo (else the")
    print("shipped cat3 was flexibility bias). bb_CADJ adopts only if rel")
    print("holds, pred does not drop, and corr(bb, sd) shrinks.")


if __name__ == '__main__':
    main()
