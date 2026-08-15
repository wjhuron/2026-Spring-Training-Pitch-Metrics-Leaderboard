"""locplus_count_stability.py — face validity + cross-season stationarity of
the learned count structure, and the decomposition of the rejected anchor.

Prints per season 2021-2026:
  1. Whiff-rate count multipliers (league whiff|swing in count / overall) —
     should be monotone-ish in strikes and near-constant across seasons.
  2. Contact-quality offsets (xw_clevel) per count — 2-strike negative
     (defensive contact), hitter counts positive, stable across seasons.
  3. Anchor decomposition: the rejected BIP_COUNT_ANCHOR offset equals
     (RE-state currency term) + (contact-quality term). The batteries said
     the bundled anchor hurts while the contact-quality half helps; the
     table shows the two halves separately:
        anchor(c)  = mean(-RunExp | BIP,c) - mean(xwval | BIP,c)
        quality(c) = mean(xwval | BIP,c) - overall mean(xwval)
        currency(c)= anchor(c) + quality(c)
                   = mean(-RunExp | BIP,c) - overall mean(xwval)

If stable, the pooled means become the early-season fallback table for
xw_clevel (mirroring FALLBACK_COUNT_OFFSETS).

Usage: python3 scripts/research/locplus/locplus_count_stability.py
"""
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.sdplus import SWING_DESCRIPTIONS

GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259), 2023: (0.318, 1.204),
        2024: (0.310, 1.242), 2025: (0.3131, 1.2317), 2026: (0.3172, 1.2343)}


def season_pitches(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB'
                and lp.is_eligible_baseline(p)]
    fname = ('_statcast2025_full_cache.pkl' if year == 2025
             else f'_statcast{year}_cache.pkl')
    path = os.path.join(ROOT, 'data', fname)
    return [p for p in base.adapt(path) if lp.is_eligible_baseline(p)]


def main():
    counts = sorted(lp.COUNTS)
    wh_mult = {c: {} for c in counts}
    quality = {c: {} for c in counts}
    anchor = {c: {} for c in counts}
    currency = {c: {} for c in counts}

    for year in (2021, 2022, 2023, 2024, 2025, 2026):
        lg, sc = GUTS[year]
        P = season_pitches(year)
        sw = defaultdict(lambda: [0, 0])         # count -> [whiffs, swings]
        xw = defaultdict(lambda: [0.0, 0])       # count -> [xwval sum, n]
        re = defaultdict(lambda: [0.0, 0])       # count -> [-RunExp sum, n]
        for p in P:
            c = lp.get_count(p)
            if c is None:
                continue
            d = p.get('Description')
            if d in SWING_DESCRIPTIONS:
                sw[c][1] += 1
                if d == 'Swinging Strike':
                    sw[c][0] += 1
            if d == 'In Play':
                v = lp.safe_float(p.get('xwOBA'))
                if v is not None:
                    xw[c][0] += (v - lg) / sc
                    xw[c][1] += 1
                r = lp.safe_float(p.get('RunExp'))
                if r is not None:
                    re[c][0] += -r
                    re[c][1] += 1
        tot_wh = sum(v[0] for v in sw.values())
        tot_sw = sum(v[1] for v in sw.values())
        ov_wh = tot_wh / tot_sw
        tot_x = sum(v[0] for v in xw.values())
        tot_xn = sum(v[1] for v in xw.values())
        ov_x = tot_x / tot_xn
        for c in counts:
            if sw[c][1]:
                wh_mult[c][year] = (sw[c][0] / sw[c][1]) / ov_wh
            if xw[c][1] >= 200:
                qx = xw[c][0] / xw[c][1] - ov_x
                quality[c][year] = qx
                if re[c][1] >= 200:
                    a = re[c][0] / re[c][1] - xw[c][0] / xw[c][1]
                    anchor[c][year] = a
                    currency[c][year] = a + qx
        del P
        import gc
        gc.collect()
        print(f"{year} done", file=sys.stderr)

    def table(name, d, fmt='+.3f'):
        print(f"\n{name}  (rows = count, cols = 2021..2026, then mean, spread)")
        for c in counts:
            vals = [d[c].get(y) for y in (2021, 2022, 2023, 2024, 2025, 2026)]
            cells = '  '.join(format(v, fmt) if v is not None else '   --  '
                              for v in vals)
            got = [v for v in vals if v is not None]
            if got:
                m = sum(got) / len(got)
                sp = max(got) - min(got)
                print(f"  {c[0]}-{c[1]}: {cells}   mean {format(m, fmt)}  "
                      f"spread {sp:.3f}")

    table('1. WHIFF MULTIPLIER (whiff rate in count / overall)', wh_mult, '.3f')
    table('2. CONTACT-QUALITY OFFSET (xw_clevel, standardized units)', quality)
    table('3a. ANCHOR (rejected: currency + quality bundled)', anchor)
    table('3b. CURRENCY HALF (anchor + quality = pure RE-state term)', currency)
    print("\nFace validity: whiff multiplier should rise with strikes;")
    print("quality offset should be negative at 2 strikes. Small spreads =")
    print("near-constants of baseball = safe pooled early-season fallbacks.")


if __name__ == '__main__':
    main()
