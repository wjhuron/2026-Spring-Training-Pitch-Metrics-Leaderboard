#!/usr/bin/env python3
"""RV vs xRV: which to show on the pitcher card?

Two empirical tests on the season pitch-level pickle:

  A. Random split-half RELIABILITY — randomly halve each unit's pitches,
     compute the metric on each half, correlate halves across units. Higher
     correlation = more signal / less noise = more reliable.

  B. Chronological PREDICTIVENESS — split each pitcher's pitches at their
     season midpoint (by date), and ask which first-half metric better
     predicts SECOND-half ACTUAL RV/100. The actual results are what we
     ultimately care about; the better leading indicator wins.

Run at both the pitcher level and the pitch-type level (the table's unit).
"""
import sys, os, pickle, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Cards import GUTS_LG_WOBA, GUTS_WOBA_SCALE, sf, MILB_TEAMS

random.seed(17)
PICKLE = '/Users/wallyhuron/Huronalytics/data/all_pitches_rs_cache.pkl'


def per_pitch_rv(p):
    """Actual RV (pitcher perspective)."""
    return sf(p.get('RunExp'))


def per_pitch_xrv(p):
    """Expected RV: xwOBA-based on BIP, actual RunExp otherwise."""
    if p.get('Description') == 'In Play':
        xw = sf(p.get('xwOBA'))
        if xw is not None:
            return -(xw - GUTS_LG_WOBA) / GUTS_WOBA_SCALE
    return sf(p.get('RunExp'))


def rate(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals) * 100) if vals else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None, n
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None, n
    return sxy / (sxx * syy) ** 0.5, n


def main():
    print('Loading pickle …')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    # MLB only (drop MiLB + ROC), need a usable RunExp.
    mlb = [p for p in allp
           if p.get('PTeam') not in MILB_TEAMS and p.get('PTeam') != 'ROC'
           and per_pitch_rv(p) is not None]
    print(f'  {len(mlb):,} MLB pitches with RunExp\n')

    # ---- Group by pitcher and by (pitcher, pitch-type) ----
    by_pitcher = defaultdict(list)
    by_type = defaultdict(list)
    for p in mlb:
        key = (p.get('Pitcher'), p.get('PTeam'))
        by_pitcher[key].append(p)
        pt = p.get('Pitch Type')
        if pt:
            by_type[(key, pt)].append(p)

    # ================= TEST A: random split-half reliability =================
    def split_half_reliability(groups, min_n, label):
        a_rv, b_rv, a_xrv, b_xrv = [], [], [], []
        for g in groups.values():
            if len(g) < min_n:
                continue
            gg = list(g)
            random.shuffle(gg)
            half = len(gg) // 2
            h1, h2 = gg[:half], gg[half:2 * half]
            r1, r2 = rate([per_pitch_rv(p) for p in h1]), rate([per_pitch_rv(p) for p in h2])
            x1, x2 = rate([per_pitch_xrv(p) for p in h1]), rate([per_pitch_xrv(p) for p in h2])
            if None in (r1, r2, x1, x2):
                continue
            a_rv.append(r1); b_rv.append(r2); a_xrv.append(x1); b_xrv.append(x2)
        r_rv, n = pearson(a_rv, b_rv)
        r_xrv, _ = pearson(a_xrv, b_xrv)
        print(f'  {label} (n={n}, min {min_n} pitches/unit):')
        print(f'     actual RV/100  split-half r = {r_rv:.3f}')
        print(f'     xRV/100        split-half r = {r_xrv:.3f}')
        print(f'     → {"xRV" if r_xrv > r_rv else "RV"} more reliable '
              f'(Δr = {abs(r_xrv - r_rv):.3f})\n')

    print('=' * 64)
    print('TEST A — RANDOM SPLIT-HALF RELIABILITY (higher r = more signal)')
    print('=' * 64)
    split_half_reliability(by_pitcher, 200, 'Pitcher level')
    split_half_reliability(by_type, 150, 'Pitch-type level')
    split_half_reliability(by_type, 60,  'Pitch-type level (small samples)')

    # ============ TEST B: chronological predictiveness ============
    # First-half metric → second-half ACTUAL RV/100.
    def chrono_predict(groups, min_n, label):
        f_rv, f_xrv, s_actual = [], [], []
        for g in groups.values():
            if len(g) < min_n:
                continue
            gg = sorted(g, key=lambda p: str(p.get('Game Date') or ''))
            half = len(gg) // 2
            h1, h2 = gg[:half], gg[half:]
            fr = rate([per_pitch_rv(p) for p in h1])
            fx = rate([per_pitch_xrv(p) for p in h1])
            sa = rate([per_pitch_rv(p) for p in h2])
            if None in (fr, fx, sa):
                continue
            f_rv.append(fr); f_xrv.append(fx); s_actual.append(sa)
        r_rv, n = pearson(f_rv, s_actual)
        r_xrv, _ = pearson(f_xrv, s_actual)
        print(f'  {label} (n={n}, min {min_n} pitches/unit):')
        print(f'     1st-half actual RV → 2nd-half actual RV : r = {r_rv:.3f}')
        print(f'     1st-half xRV       → 2nd-half actual RV : r = {r_xrv:.3f}')
        print(f'     → {"xRV" if r_xrv > r_rv else "RV"} predicts future results better '
              f'(Δr = {abs(r_xrv - r_rv):.3f})\n')

    print('=' * 64)
    print('TEST B — CHRONOLOGICAL PREDICTIVENESS (which leads future results)')
    print('=' * 64)
    chrono_predict(by_pitcher, 300, 'Pitcher level')
    chrono_predict(by_type, 150, 'Pitch-type level')

    # ============ Descriptive: how much do RV and xRV diverge? ============
    print('=' * 64)
    print('DESCRIPTIVE — RV vs xRV agreement (per pitch-type unit, ≥150 p)')
    print('=' * 64)
    rvs, xrvs = [], []
    for g in by_type.values():
        if len(g) < 150:
            continue
        r = rate([per_pitch_rv(p) for p in g])
        x = rate([per_pitch_xrv(p) for p in g])
        if r is not None and x is not None:
            rvs.append(r); xrvs.append(x)
    r_corr, n = pearson(rvs, xrvs)
    sd_rv = (sum((v - sum(rvs)/len(rvs))**2 for v in rvs)/len(rvs))**0.5
    sd_xrv = (sum((v - sum(xrvs)/len(xrvs))**2 for v in xrvs)/len(xrvs))**0.5
    print(f'  n={n} pitch-type units')
    print(f'  corr(RV/100, xRV/100) = {r_corr:.3f}')
    print(f'  SD of actual RV/100 = {sd_rv:.2f}   SD of xRV/100 = {sd_xrv:.2f}')
    print(f'  → actual RV is {sd_rv/sd_xrv:.2f}x as spread out (extra spread = BIP noise)\n')


if __name__ == '__main__':
    main()
