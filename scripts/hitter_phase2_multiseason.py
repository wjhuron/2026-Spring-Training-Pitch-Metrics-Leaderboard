"""hitter_phase2_multiseason.py — multi-season replicate revalidation of the
Phase 2 SD+/CT+ structural config (adopted 2026-07-02 on a single 2026
split-half; see scripts/phase2_sdct_harness.py + phase2_sdplus_extensions.py).

Variants (each toggles ONE structural choice off the shipped config):
  SD+:  BASE (shipped: count-anchor ON, heart=1/6, cat3 ON, mix-neutral ON)
        NOANCHOR   count-anchor offsets removed from the BIP branch
        HEART13    HEART_VERT_FRAC back to 1/3 (pre-Phase-2 heart)
        NOCAT3     single pitch category (zone x count table)
        NOMIX      plain per-decision mean (no league zone reweighting)
  CT+:  BASE (shipped: count-anchor ON, lift-ratio actual/expected)
        NOANCHOR   count-anchor off
        RAWRATE    leverage-weighted raw contact rate (pre-Phase-2 form)

Metrics, seasons 2021-2025 (public Statcast via adapter) + 2026 (cache):
  1. Split-half reliability: 3 random game-date partitions per season,
     per-half floors 125 decisions / 45 swings (handsplit-harness protocol).
  2. Predictive: full-season year-N raw component (production floors
     200 dec / 65 sw) vs year-N+1 wOBA (>=200 PA events), pairs 21->22..24->25.

Ship bar (per feedback_multiseason_defensibility): the shipped config must
win or hold in most replicates it was never fitted on. A variant that beats
BASE on reliability in most seasons AND does not lose prediction flags a
config change; otherwise the shipped choice is confirmed.

Zones are precomputed per pitch under both heart fractions (_z16 / _z13)
and classify_zone/get_count are redirected to the stash — the table build,
shrinkage cascade, dv and CT+ scoring all run through the production
pipeline functions unchanged. A parity check against the unpatched pipeline
runs first on 2026 (BASE must match to 1e-9).

Results: printed + data/_hitter_phase2_multiseason.json

Usage: python3 scripts/hitter_phase2_multiseason.py
"""
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statcast_hitter_adapter as A
import pipeline_sdplus as sd
import pipeline_contact as ct

SEEDS = (0, 1, 2)
HALF_MIN_DEC = 125
HALF_MIN_SW = 45
FULL_MIN_DEC = sd.MIN_HITTER_DECISIONS   # 200
FULL_MIN_SW = ct.MIN_HITTER_SWINGS       # 65
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
GUTS_2026 = (0.3172, 1.2343)   # train_stuff fallback constants
OUT_JSON = os.path.join(ROOT, 'data', '_hitter_phase2_multiseason.json')

SD_VARIANTS = ('BASE', 'NOANCHOR', 'HEART13', 'NOCAT3', 'NOMIX')
CT_VARIANTS = ('BASE', 'NOANCHOR', 'RAWRATE')

_orig_classify_zone = sd.classify_zone
_orig_get_count = sd.get_count
_orig_cat_of = sd.cat_of


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


def load_season(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB']
    return A.season_dicts(year)


def guts(year):
    return GUTS_2026 if year == 2026 else A.GUTS[year]


def precompute(P):
    """Stash _z16 / _z13 / _cnt on every pitch, using the real classify_zone
    under each heart fraction. Returns the SD+-eligible subset (eligibility
    is heart-invariant: the fraction moves the heart/shadow boundary but
    never produces/removes a None zone)."""
    old = sd.HEART_VERT_FRAC
    try:
        sd.HEART_VERT_FRAC = 1.0 / 6.0
        for p in P:
            p['_z16'] = _orig_classify_zone(p)
        sd.HEART_VERT_FRAC = 1.0 / 3.0
        for p in P:
            p['_z13'] = _orig_classify_zone(p)
    finally:
        sd.HEART_VERT_FRAC = old
    for p in P:
        p['_cnt'] = _orig_get_count(p)
    return [p for p in P if sd.is_eligible(p)]


class patched:
    """Redirect classify_zone/get_count (and optionally cat_of) in BOTH
    pipeline namespaces to the precomputed stash for one variant."""

    def __init__(self, heart_key='_z16', cat3=True):
        self.zfn = (lambda p, k=heart_key: p[k])
        self.cfn = (lambda p: p['_cnt'])
        self.catfn = _orig_cat_of if cat3 else (lambda p: 'FB')

    def __enter__(self):
        sd.classify_zone = ct.classify_zone = self.zfn
        sd.get_count = ct.get_count = self.cfn
        sd.cat_of = self.catfn
        return self

    def __exit__(self, *a):
        sd.classify_zone = ct.classify_zone = _orig_classify_zone
        sd.get_count = ct.get_count = _orig_get_count
        sd.cat_of = _orig_cat_of


def sd_score(by_hitter, table, lgw, min_n):
    """Raw SD per hitter off the current patched zone fn. lgw None = plain mean."""
    out = {}
    for h, pitches in by_hitter.items():
        if len(pitches) < min_n:
            continue
        if lgw is None:
            dvs = [sd.compute_dv(p, table) for p in pitches]
            out[h] = sum(dvs) / len(dvs)
            continue
        zone_dvs = defaultdict(list)
        for p in pitches:
            zone_dvs[sd.classify_zone(p)].append(sd.compute_dv(p, table))
        zmeans = {z: sum(v) / len(v) for z, v in zone_dvs.items()}
        wsum = sum(lgw.get(z, 0.0) for z in zmeans)
        if wsum <= 0:
            continue
        out[h] = sum(m * lgw.get(z, 0.0) for z, m in zmeans.items()) / wsum
    return out


def ct_score(by_hitter_sw, table, min_n, lift=True):
    out = {}
    for h, swings in by_hitter_sw.items():
        if len(swings) < min_n:
            continue
        A_ = E = W = 0.0
        for p in swings:
            lev, con = ct.compute_ct_swing(p, table)
            if lev <= 0:
                continue
            # 3-part key since CT+ cat3 shipped 2026-08-15 (pipeline tables
            # are (zone, count, cat); sd.cat_of respects patched contexts)
            cell = table[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p))]
            A_ += lev * con
            E += lev * (1.0 - cell['p_whiff'])
            W += lev
        if lift:
            if E > 0:
                out[h] = A_ / E
        else:
            if W > 0:
                out[h] = A_ / W
    return out


def season_components(elig, lg, sc, min_dec, min_sw):
    """All 8 variant raw-component dicts for one data slice (elig = the
    precomputed SD+-eligible pitches of that slice)."""
    res = {}

    for name in SD_VARIANTS:
        heart_key = '_z13' if name == 'HEART13' else '_z16'
        cat3 = (name != 'NOCAT3')
        anchor = (name != 'NOANCHOR')
        mix = (name != 'NOMIX')
        with patched(heart_key, cat3):
            offsets = sd.build_bip_count_offsets(elig, lg, sc) if anchor else None
            rv_fn = sd.make_rv_xrv(lg, sc, offsets)
            raw = sd.build_weight_table(elig, rv_fn)
            zm = sd.zone_level_means(elig, rv_fn)
            table = sd.shrink_table(raw, zm)
            lgw = None
            if mix:
                zc = defaultdict(int)
                for p in elig:
                    zc[sd.classify_zone(p)] += 1
                tot = sum(zc.values())
                lgw = {z: n / tot for z, n in zc.items()}
            by_hitter = defaultdict(list)
            for p in elig:
                h = p.get('Batter')
                if h:
                    by_hitter[h].append(p)
            res[f'sd_{name}'] = sd_score(by_hitter, table, lgw, min_dec)

    with patched('_z16', True):
        swings = [p for p in elig if ct.is_ct_eligible(p)]
        by_hitter_sw = defaultdict(list)
        for p in swings:
            h = p.get('Batter')
            if h:
                by_hitter_sw[h].append(p)
        for name in CT_VARIANTS:
            anchor = (name != 'NOANCHOR')
            lift = (name != 'RAWRATE')
            offsets = ct.build_bip_count_offsets(swings, lg, sc) if anchor else None
            rv_fn = ct.make_rv_xrv(lg, sc, offsets)
            raw = ct.build_contact_cell_weights(swings, rv_fn)
            zm = ct.zone_level_contact_means(swings, rv_fn)
            table = ct.shrink_contact_cells(raw, zm)
            res[f'ct_{name}'] = ct_score(by_hitter_sw, table, min_sw, lift)
    return res


def parity_check():
    """BASE via the stash-patched path must reproduce the unpatched pipeline
    on 2026 (guards against a stash/patch bug corrupting every variant)."""
    P = load_season(2026)
    lg, sc = guts(2026)
    elig = precompute(P)

    with patched('_z16', True):
        offsets = sd.build_bip_count_offsets(elig, lg, sc)
        rv_fn = sd.make_rv_xrv(lg, sc, offsets)
        raw = sd.build_weight_table(elig, rv_fn)
        zm = sd.zone_level_means(elig, rv_fn)
        t_patch = sd.shrink_table(raw, zm)

    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)
    raw = sd.build_weight_table(elig, rv_fn)
    zm = sd.zone_level_means(elig, rv_fn)
    t_ref = sd.shrink_table(raw, zm)

    worst = max(abs(t_patch[k][0] - t_ref[k][0]) for k in t_ref)
    assert worst < 1e-9, f'parity check failed: max cell delta {worst}'
    print(f'parity check OK (max cell delta {worst:.2e})', flush=True)
    return P, elig


def main():
    results = {'split_half': {}, 'predictive': {}}
    P2026, elig2026 = parity_check()

    all_keys = [f'sd_{v}' for v in SD_VARIANTS] + [f'ct_{v}' for v in CT_VARIANTS]

    # ── 1. split-half reliability ──
    print(f"\nSPLIT-HALF RELIABILITY (per-half floors {HALF_MIN_DEC} dec / "
          f"{HALF_MIN_SW} sw)", flush=True)
    agg = defaultdict(list)
    for year in SEASONS:
        if year == 2026:
            P, elig = P2026, elig2026
        else:
            P = load_season(year)
            elig = precompute(P)
        lg, sc = guts(year)
        dates = sorted({p.get('Game Date') for p in elig if p.get('Game Date')})
        for seed in SEEDS:
            rnd = random.Random(seed)
            sh = dates[:]
            rnd.shuffle(sh)
            ha = set(sh[:len(sh) // 2])
            Ea = [p for p in elig if p.get('Game Date') in ha]
            Eb = [p for p in elig if p.get('Game Date') and p.get('Game Date') not in ha]
            ra = season_components(Ea, lg, sc, HALF_MIN_DEC, HALF_MIN_SW)
            rb = season_components(Eb, lg, sc, HALF_MIN_DEC, HALF_MIN_SW)
            row = {}
            for k in all_keys:
                common = [h for h in ra[k] if h in rb[k]]
                r = pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                row[k] = [r, len(common)]
                if r is not None:
                    agg[k].append((year, seed, r))
            results['split_half'][f'{year}_s{seed}'] = row
            print(f"  {year} seed{seed}: " + '  '.join(
                f"{k}={row[k][0]:.3f}(n={row[k][1]})" if row[k][0] is not None
                else f"{k}=NA" for k in all_keys), flush=True)
        del P, elig
        if year != 2026:
            import gc
            gc.collect()

    print("\n  MEAN split-half r (and per-season means):")
    for k in all_keys:
        rows = agg[k]
        rs = [r for _, _, r in rows]
        by_season = defaultdict(list)
        for y, _, r in rows:
            by_season[y].append(r)
        seas = '  '.join(f"{y}:{sum(v)/len(v):.3f}" for y, v in sorted(by_season.items()))
        print(f"    {k}: mean {sum(rs)/len(rs):.4f}   {seas}")

    # ── 2. predictive: year N -> year N+1 wOBA ──
    print(f"\nPREDICTIVE (full-season raw, floors {FULL_MIN_DEC} dec / "
          f"{FULL_MIN_SW} sw, vs next-season wOBA >=200 events)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = load_season(yn)
        elig = precompute(P)
        lg, sc = guts(yn)
        comp = season_components(elig, lg, sc, FULL_MIN_DEC, FULL_MIN_SW)
        y_map = A.target_y(yn1)
        row = {}
        for k in all_keys:
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = pearson(xs, ys)
            row[k] = [r, len(xs)]
            if r is not None:
                pagg[k].append((yn, r))
        results['predictive'][f'{yn}_{yn1}'] = row
        print(f"  {yn}->{yn1}: " + '  '.join(
            f"{k}={row[k][0]:+.3f}(n={row[k][1]})" if row[k][0] is not None
            else f"{k}=NA" for k in all_keys), flush=True)
        del P, elig

    print("\n  MEAN predictive r:")
    for k in all_keys:
        rs = [r for _, r in pagg[k]]
        if rs:
            print(f"    {k}: {sum(rs)/len(rs):+.4f}")

    # ── 3. verdict table: variant vs BASE per season replicate ──
    print("\nVERDICTS (variant minus BASE, per season; positive = variant better)")
    for fam, variants, base in (('sd', SD_VARIANTS, 'sd_BASE'),
                                ('ct', CT_VARIANTS, 'ct_BASE')):
        base_rel = defaultdict(list)
        for y, s, r in agg[base]:
            base_rel[y].append(r)
        for v in variants:
            if v == 'BASE':
                continue
            k = f'{fam}_{v}'
            wins = 0
            terms = []
            var_rel = defaultdict(list)
            for y, s, r in agg[k]:
                var_rel[y].append(r)
            for y in sorted(base_rel):
                d = sum(var_rel[y]) / len(var_rel[y]) - sum(base_rel[y]) / len(base_rel[y])
                terms.append(f"{y}:{d:+.3f}")
                if d > 0:
                    wins += 1
            pb = [r for _, r in pagg[base]]
            pv = [r for _, r in pagg[k]]
            pd = (sum(pv) / len(pv) - sum(pb) / len(pb)) if pb and pv else None
            print(f"  {k}: rel wins {wins}/{len(base_rel)}  " + '  '.join(terms)
                  + (f"   pred delta {pd:+.4f}" if pd is not None else ""))

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == '__main__':
    main()
