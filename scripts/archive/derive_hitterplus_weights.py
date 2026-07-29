"""derive_hitterplus_weights.py — Hitter+ component weights, out-of-sample.

Replaces the retired same-season wRC+ OLS (which produced 65/7/28 and
structurally underweighted the stabler skills because wRC+ noise correlates
with BB+ noise, and BB+/CT+ collinearity destabilized the coefficients).

Protocol: FIRST-HALF component values (chronological median date split;
league tables built from first-half data only for SD+/CT+, full-season
zone tables for the BB+ ingredients since those are league constants) →
standardized → OLS and RIDGE regression on SECOND-HALF wOBA. Ridge is
reported at several lambdas; recommend the stable region.

Components (2026-07-02 definitions):
  BB+  = 0.85·xwOBAcon+ + 0.15·sprayPlus (LA-residualized spray value)
  SD+  = mix-neutral count-anchored decision value (pipeline_sdplus)
  CT+  = actual/expected leverage-weighted contact ratio (pipeline_contact)

Usage: python3 scripts/derive_hitterplus_weights.py
"""
import os, sys, pickle, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_sdplus as sd
import pipeline_contact as ct
from pipeline_utils import safe_float, BUNT_BB_TYPES, spray_angle, spray_direction

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393
LA_BINS = [(-999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
           (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 999)]
MIN_BIP_ZONE = 20
BIP_WOBA = {'Single': 0.9, 'Double': 1.25, 'Triple': 1.6, 'Home Run': 2.0,
            'Field Error': 0.9, 'Fielders Choice': 0.9,
            'Fielders Choice Out': 0.9}
PA_WOBA = {'Walk': 0.7, 'Hit By Pitch': 0.72}
W_CON, W_SP = 0.85, 0.15


def la_bin_of(la):
    for bi, (lo, hi) in enumerate(LA_BINS):
        if lo <= la < hi:
            return bi
    return None


def classify(p):
    bb = p.get('BBType')
    if not bb or bb in BUNT_BB_TYPES:
        return None
    hcx, hcy = safe_float(p.get('HC_X')), safe_float(p.get('HC_Y'))
    la, bats = safe_float(p.get('LaunchAngle')), p.get('Bats')
    if None in (hcx, hcy, la) or not bats:
        return None
    d = spray_direction(spray_angle(hcx, hcy), bats)
    bi = la_bin_of(la)
    if not d or bi is None:
        return None
    return d, bi, bats


def zscore(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    return [(v - m) / s if s > 0 else 0.0 for v in vals]


def ridge3(z, y, lam):
    """3-predictor ridge on standardized predictors. z: list of 3 lists."""
    n = len(y)
    my = sum(y) / n
    yc = [v - my for v in y]
    S = [[sum(z[i][k] * z[j][k] for k in range(n)) + (lam * n if i == j else 0.0)
          for j in range(3)] for i in range(3)]
    b = [sum(z[i][k] * yc[k] for k in range(n)) for i in range(3)]
    # solve 3x3 by Cramer
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(S)
    if abs(D) < 1e-12:
        return None
    out = []
    for i in range(3):
        M = [row[:] for row in S]
        for r in range(3):
            M[r][i] = b[r]
        out.append(det3(M) / D)
    return out


def main():
    D = pickle.load(open(PKL, 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    dates = sorted({p.get('Game Date') for p in mlb if p.get('Game Date')})
    mid = dates[len(dates) // 2]
    first = [p for p in mlb if (p.get('Game Date') or '') < mid]
    second = [p for p in mlb if (p.get('Game Date') or '') >= mid]

    # ── full-season league zone tables for BB+ ingredients ──
    acc = defaultdict(lambda: [0.0, 0])
    for p in mlb:
        if p.get('Description') != 'In Play':
            continue
        k = classify(p)
        if k is None:
            continue
        w = BIP_WOBA.get(p.get('Event'), 0.0 if p.get('Event') else None)
        if w is None:
            continue
        d, bi, bats = k
        for key in ((d, bi, bats), (d, bi), (bi, bats), (bi,)):
            acc[key][0] += w; acc[key][1] += 1

    def lookup(*keys):
        for key in keys:
            s, n = acc.get(key, (0.0, 0))
            if n >= MIN_BIP_ZONE:
                return s / n
        return None

    lg_xwobacon_acc = [0.0, 0]
    for p in mlb:
        if p.get('Description') == 'In Play':
            xw = safe_float(p.get('xwOBA'))
            if xw is not None:
                lg_xwobacon_acc[0] += xw; lg_xwobacon_acc[1] += 1
    lg_xc = lg_xwobacon_acc[0] / lg_xwobacon_acc[1]

    # ── first-half SD+/CT+ (tables from first-half MLB data) ──
    by_h_first = defaultdict(list)
    for p in first:
        h, t = p.get('Batter'), p.get('BTeam')
        if h and t:
            by_h_first[(h, t)].append(p)
    sd_res, _w1 = sd.compute_sd_plus(first, dict(by_h_first), LG_WOBA, WOBA_SCALE)
    ct_res, _w2 = ct.compute_ct_plus(first, dict(by_h_first), LG_WOBA, WOBA_SCALE)

    # ── first-half BB+ components + second-half wOBA target ──
    per = defaultdict(lambda: {'con': [0.0, 0], 'sp': [0.0, 0], 'y': [0.0, 0]})
    for p in first:
        h, t = p.get('Batter'), p.get('BTeam')
        if not h or not t or p.get('Description') != 'In Play':
            continue
        a = per[(h, t)]
        k = classify(p)
        xw = safe_float(p.get('xwOBA'))
        if xw is not None and k is not None:
            a['con'][0] += xw; a['con'][1] += 1
        if k is not None:
            d, bi, bats = k
            zv = lookup((d, bi, bats), (d, bi))
            lv = lookup((bi, bats), (bi,))
            if zv is not None and lv is not None:
                a['sp'][0] += zv - lv; a['sp'][1] += 1
    for p in second:
        h, t = p.get('Batter'), p.get('BTeam')
        if not h or not t:
            continue
        ev = p.get('Event')
        if ev == 'Intent Walk':
            continue
        a = per[(h, t)]
        val = None
        if p.get('Description') == 'In Play':
            val = BIP_WOBA.get(ev, 0.0 if ev else None)
        elif ev in PA_WOBA:
            val = PA_WOBA[ev]
        elif ev and ('Strikeout' in ev or ev == 'Batter Out'):
            val = 0.0
        if val is not None:
            a['y'][0] += val; a['y'][1] += 1

    rows = []
    for key, a in per.items():
        sdv = sd_res.get(key)
        ctv = ct_res.get(key)
        if (a['con'][1] >= 40 and a['sp'][1] >= 40 and a['y'][1] >= 100
                and sdv is not None and ctv is not None):
            con_plus = 100.0 * (a['con'][0] / a['con'][1]) / lg_xc
            sp_plus = 100.0 * (lg_xc + a['sp'][0] / a['sp'][1]) / lg_xc
            bb = W_CON * con_plus + W_SP * sp_plus
            rows.append((bb, sdv['sdPlus'], ctv['ctPlus'], a['y'][0] / a['y'][1]))

    print(f'derivation sample: {len(rows)} hitters '
          f'(>=40 BIP + SD/CT floors 1st half, >=100 PA-events 2nd half)')
    zb = zscore([r[0] for r in rows])
    zs = zscore([r[1] for r in rows])
    zc = zscore([r[2] for r in rows])
    y = [r[3] for r in rows]

    def pear(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((x - mb) ** 2 for x in b))
        return sum((x - ma) * (v - mb) for x, v in zip(a, b)) / (sa * sb) if sa > 0 and sb > 0 else None

    print(f'component correlations: bb-sd {pear(zb, zs):+.2f}, '
          f'bb-ct {pear(zb, zc):+.2f}, sd-ct {pear(zs, zc):+.2f}')
    print(f'univariate vs 2nd-half wOBA: bb {pear(zb, y):+.3f}, '
          f'sd {pear(zs, y):+.3f}, ct {pear(zc, y):+.3f}')

    print(f"\n{'lambda':>7s} {'w_bb':>6s} {'w_sd':>6s} {'w_ct':>6s}  (normalized |beta|)")
    for lam in (0.0, 0.05, 0.1, 0.25, 0.5):
        b = ridge3([zb, zs, zc], y, lam)
        if b is None:
            continue
        tot = sum(abs(x) for x in b)
        print(f'{lam:7.2f} {abs(b[0])/tot:6.3f} {abs(b[1])/tot:6.3f} {abs(b[2])/tot:6.3f}'
              f'   raw: {b[0]:+.5f} {b[1]:+.5f} {b[2]:+.5f}')


if __name__ == '__main__':
    main()
