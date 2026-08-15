"""bbplus_n0_cliff_test.py — measure BB+'s n0 and test shrinkage vs the 80-BIP cliff.

Part A — n0: split-half r of raw BB+ (0.85 conRatio + 0.15 sprayRatio, league
tables as full-season constants) at fixed BIPs-per-half N, hitters subsampled
to exactly N per half, 5 seeds, seasons 2021-2025 + 2026. The N where r=0.5
is the MMSE pseudo-count (same convention as SD+ n0=250, CT+ n0=85).

Part B — cliff: does sub-80-BIP BB+ carry usable signal? Year-N BB+ in BIP
bands [30,50) / [50,80) / [80,inf) vs year-N+1 wOBA (pairs 21->22 .. 24->25),
raw and shrunk-to-league(n0). If the low bands predict, replace the hard
`nBip >= 80 else None` gate with (n·raw + n0·lg)/(n+n0).

Usage: python3 scripts/research/hitter/bbplus_n0_cliff_test.py
"""
import os, sys, math, pickle, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
from pipeline.utils import safe_float, BUNT_BB_TYPES, spray_angle, spray_direction
from derive_weights_multiseason import (LA_BINS, MIN_BIP_ZONE, la_bin_of,
                                        classify, pear)

W_CON, W_SP = 0.85, 0.15
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
NS = [20, 30, 40, 60, 80, 120, 160]
SEEDS = range(5)
BANDS = [(30, 50), (50, 80), (80, 10 ** 9)]


def load_season(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB']
    return A.season_dicts(year)


def bip_records(P):
    """Per hitter: list of (xw, spdiff) records where BOTH ingredients exist,
    plus the league tables. Returns (records_by_hitter, lg_xc)."""
    acc = defaultdict(lambda: [0.0, 0])
    lg_acc = [0.0, 0]
    for p in P:
        if p.get('Description') != 'In Play':
            continue
        xw = safe_float(p.get('xwOBA'))
        if xw is not None:
            lg_acc[0] += xw
            lg_acc[1] += 1
        k = classify(p)
        if k is None:
            continue
        ev = p.get('event_raw') if 'event_raw' in p else None
        if ev is None:  # 2026 cache path: Wally-schema Event names
            from derive_weights_multiseason import A as _A
            evmap = {'Single': 0.9, 'Double': 1.25, 'Triple': 1.6,
                     'Home Run': 2.0, 'Field Error': 0.9,
                     'Fielders Choice': 0.9, 'Fielders Choice Out': 0.9}
            w = evmap.get(p.get('Event'), 0.0 if p.get('Event') else None)
        else:
            w = A.BIP_WOBA_PUB.get(ev, 0.0 if ev else None)
        if w is None:
            continue
        d, bi, bats = k
        for key in ((d, bi, bats), (d, bi), (bi, bats), (bi,)):
            acc[key][0] += w
            acc[key][1] += 1
    lg_xc = lg_acc[0] / lg_acc[1]

    def lookup(*keys):
        for key in keys:
            s, n = acc.get(key, (0.0, 0))
            if n >= MIN_BIP_ZONE:
                return s / n
        return None

    recs = defaultdict(list)
    for p in P:
        h = p.get('Batter')
        if not h or p.get('Description') != 'In Play':
            continue
        xw = safe_float(p.get('xwOBA'))
        k = classify(p)
        if xw is None or k is None:
            continue
        d, bi, bats = k
        zv = lookup((d, bi, bats), (d, bi))
        lv = lookup((bi, bats), (bi,))
        if zv is None or lv is None:
            continue
        recs[h].append((xw, zv - lv))
    return recs, lg_xc


def bb_raw(sample, lg_xc):
    con = sum(r[0] for r in sample) / len(sample)
    sp = sum(r[1] for r in sample) / len(sample)
    return (W_CON * 100.0 * con / lg_xc
            + W_SP * 100.0 * (lg_xc + sp) / lg_xc)


def main():
    # ── Part A: n0 ──
    print("PART A — BB+ split-half r by BIPs-per-half", flush=True)
    r_by_n = defaultdict(list)
    season_recs = {}
    for year in SEASONS:
        P = load_season(year)
        recs, lg_xc = bip_records(P)
        season_recs[year] = (recs, lg_xc)
        del P
        for N in NS:
            elig = {h: r for h, r in recs.items() if len(r) >= 2 * N}
            if len(elig) < 30:
                continue
            for seed in SEEDS:
                rnd = random.Random(1000 * seed + N)
                xs, ys = [], []
                for h, r in elig.items():
                    s = r[:]
                    rnd.shuffle(s)
                    xs.append(bb_raw(s[:N], lg_xc))
                    ys.append(bb_raw(s[N:2 * N], lg_xc))
                rr = pear(xs, ys)
                if rr is not None:
                    r_by_n[N].append(rr)
        print(f"  {year}: {len(recs)} hitters with BIP records", flush=True)
    print(f"\n  {'N/half':>7s} {'mean r':>7s} {'n_meas':>7s} {'implied n0=N(1-r)/r':>20s}")
    for N in NS:
        rs = r_by_n.get(N, [])
        if not rs:
            continue
        m = sum(rs) / len(rs)
        n0 = N * (1 - m) / m if m > 0 else float('inf')
        print(f"  {N:7d} {m:7.3f} {len(rs):7d} {n0:20.0f}")

    # ── Part B: cliff ──
    print("\nPART B — BIP-band prediction of next-season wOBA", flush=True)
    # n0 for shrinkage: harmonic-consensus from part A rows at N=40..120
    n0_est = []
    for N in (40, 60, 80, 120):
        rs = r_by_n.get(N, [])
        if rs:
            m = sum(rs) / len(rs)
            if 0 < m < 1:
                n0_est.append(N * (1 - m) / m)
    n0 = sum(n0_est) / len(n0_est) if n0_est else 80.0
    print(f"  using n0 = {n0:.0f}")
    band_rows = defaultdict(lambda: ([], []))   # band -> (raw list, shrunk list) with y
    band_y = defaultdict(list)
    for yn, yn1 in PAIRS:
        recs, lg_xc = season_recs[yn]
        y_map = A.target_y(yn1)
        for h, r in recs.items():
            n = len(r)
            if n < 30:
                continue
            yv = y_map.get(h)
            if yv is None or yv[1] < 200:
                continue
            raw = bb_raw(r, lg_xc)
            shrunk = (n * raw + n0 * 100.0) / (n + n0)
            for lo, hi in BANDS:
                if lo <= n < hi:
                    band_rows[(lo, hi)][0].append(raw)
                    band_rows[(lo, hi)][1].append(shrunk)
                    band_y[(lo, hi)].append(yv[0] / yv[1])
    for band in BANDS:
        raws, shr = band_rows[band]
        ys = band_y[band]
        if len(ys) < 20:
            print(f"  band {band}: n={len(ys)} (too few)")
            continue
        print(f"  band {band[0]:>3d}-{band[1] if band[1] < 10**9 else 'inf':>4}: "
              f"n={len(ys):4d}  r_raw={pear(raws, ys):+.3f}  "
              f"r_shrunk={pear(shr, ys):+.3f}")
    # pooled >=30 with shrinkage vs current cliff coverage
    all_raw, all_shr, all_y, cliff_raw, cliff_y = [], [], [], [], []
    for band in BANDS:
        raws, shr = band_rows[band]
        ys = band_y[band]
        all_raw += raws
        all_shr += shr
        all_y += ys
        if band[0] >= 80:
            cliff_raw += raws
            cliff_y += ys
    print(f"\n  pooled >=30 BIP shrunk: n={len(all_y)}  r={pear(all_shr, all_y):+.3f}")
    print(f"  current >=80 cliff:     n={len(cliff_y)}  r={pear(cliff_raw, cliff_y):+.3f}")
    print(f"  coverage gain: +{len(all_y) - len(cliff_y)} hitters "
          f"({(len(all_y) - len(cliff_y)) / max(1, len(cliff_y)) * 100:.0f}%)")


if __name__ == '__main__':
    main()
