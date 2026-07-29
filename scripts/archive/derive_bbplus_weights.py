"""derive_bbplus_weights.py — BB+ component weights, predictively derived.

BB+ = W_CON x xwOBAcon+ + W_SP x sprayPlus, where sprayPlus is built from the
LA-residualized spray value (sprayVal): the zone (spray x LA) wOBAcon minus
the LA-only league wOBAcon per BIP. This script derives the weights by
regressing SECOND-HALF wOBA on standardized FIRST-HALF components — a
predictive target, unlike the retired same-season wRC+ OLS, which
structurally flattered the spray component (its zone values are realized
outcomes, so it shares noise with same-season results).

Also reports: component orthogonality (the point of residualizing), the
same-season descriptive fit for reference, and split-half stickiness of
sprayVal itself.

League zone tables are built from the FULL season (they are league
constants); only the per-hitter component samples are split.

Usage: python3 scripts/derive_bbplus_weights.py
"""
import os, sys, pickle, math, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline_utils import safe_float, BUNT_BB_TYPES, spray_angle, spray_direction

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LA_BINS = [(-999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
           (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 999)]
MIN_BIP_ZONE = 20
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393
# Statcast woba_value weights, mirroring process_data._bip_woba_value
BIP_WOBA = {'Single': 0.9, 'Double': 1.25, 'Triple': 1.6, 'Home Run': 2.0,
            'Field Error': 0.9, 'Fielders Choice': 0.9,
            'Fielders Choice Out': 0.9}
PA_WOBA = {'Walk': 0.7, 'Hit By Pitch': 0.72}


def bip_woba_value(event):
    if event in BIP_WOBA:
        return BIP_WOBA[event]
    return 0.0 if event else None


def la_bin_of(la):
    for bi, (lo, hi) in enumerate(LA_BINS):
        if lo <= la < hi:
            return bi
    return None


def classify(p):
    """(direction, la_bin, bats) for an eligible BIP, else None."""
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


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return (sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)
            if sx > 0 and sy > 0 else None)


def zscore(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    return [(v - m) / s if s > 0 else 0.0 for v in vals], m, s


def ols2(z1, z2, y):
    """Two-var OLS on standardized predictors; returns (b1, b2)."""
    n = len(y)
    my = sum(y) / n
    yc = [v - my for v in y]
    s11 = sum(a * a for a in z1); s22 = sum(a * a for a in z2)
    s12 = sum(a * b for a, b in zip(z1, z2))
    s1y = sum(a * b for a, b in zip(z1, yc)); s2y = sum(a * b for a, b in zip(z2, yc))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-9:
        return None, None
    return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


def main():
    D = pickle.load(open(PKL, 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']

    # full-season league tables (hand-specific + pooled, spray x LA and LA-only)
    acc = defaultdict(lambda: [0.0, 0])
    for p in mlb:
        if p.get('Description') != 'In Play':
            continue
        k = classify(p)
        if k is None:
            continue
        w = bip_woba_value(p.get('Event'))
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

    dates = sorted({p.get('Game Date') for p in mlb if p.get('Game Date')})
    mid = dates[len(dates) // 2]

    per = defaultdict(lambda: {'con1': [0.0, 0], 'sp1': [0.0, 0],
                               'woba2': [0.0, 0.0], 'bip1': 0})
    for p in mlb:
        h, t = p.get('Batter'), p.get('BTeam')
        if not h or not t:
            continue
        date = p.get('Game Date') or ''
        a = per[(h, t)]
        first = date < mid
        if p.get('Description') == 'In Play':
            k = classify(p)
            xw = safe_float(p.get('xwOBA'))
            if first:
                if xw is not None and k is not None:
                    a['con1'][0] += xw; a['con1'][1] += 1
                if k is not None:
                    d, bi, bats = k
                    zv = lookup((d, bi, bats), (d, bi))
                    lv = lookup((bi, bats), (bi,))
                    if zv is not None and lv is not None:
                        a['sp1'][0] += zv - lv; a['sp1'][1] += 1
                        a['bip1'] += 1
        # second-half wOBA target from PA-ending events
        if not first:
            ev = p.get('Event')
            if ev in ('Intent Walk',):
                continue
            val = None
            dom = 0
            if p.get('Description') == 'In Play':
                val = bip_woba_value(ev); dom = 1
            elif ev in PA_WOBA:
                val = PA_WOBA[ev]; dom = 1
            elif ev and ('Strikeout' in ev or ev in ('Batter Out',)):
                val = 0.0; dom = 1
            if dom and val is not None:
                a['woba2'][0] += val; a['woba2'][1] += 1

    rows = []
    for k, a in per.items():
        if a['con1'][1] >= 40 and a['sp1'][1] >= 40 and a['woba2'][1] >= 100:
            rows.append((a['con1'][0] / a['con1'][1],
                         a['sp1'][0] / a['sp1'][1],
                         a['woba2'][0] / a['woba2'][1]))
    print(f'hitters in derivation sample: {len(rows)} '
          f'(>=40 BIP 1st half, >=100 PA-events 2nd half)')
    cons = [r[0] for r in rows]; sps = [r[1] for r in rows]; y = [r[2] for r in rows]
    print(f'orthogonality corr(xwOBAcon, sprayVal) first half: '
          f'{pearson(cons, sps):+.3f}')
    print(f'stickiness check corr(sprayVal_1st, wOBA_2nd): {pearson(sps, y):+.3f}; '
          f'corr(con_1st, wOBA_2nd): {pearson(cons, y):+.3f}')

    zc, _, _ = zscore(cons)
    zs, _, _ = zscore(sps)
    b1, b2 = ols2(zc, zs, y)
    tot = abs(b1) + abs(b2)
    print(f'\nPREDICTIVE weights (2nd-half wOBA on standardized 1st-half components):')
    print(f'  raw betas: con {b1:+.5f}, spray {b2:+.5f}')
    print(f'  normalized: W_CON = {abs(b1)/tot:.3f}, W_SP = {abs(b2)/tot:.3f}')

    # split-half stickiness of sprayVal itself (odd/even dates)
    parity = {d: i % 2 for i, d in enumerate(dates)}
    halves = (defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0]))
    for p in mlb:
        if p.get('Description') != 'In Play':
            continue
        k = classify(p)
        if k is None:
            continue
        h = parity.get(p.get('Game Date'))
        if h is None:
            continue
        d, bi, bats = k
        zv = lookup((d, bi, bats), (d, bi)); lv = lookup((bi, bats), (bi,))
        if zv is None or lv is None:
            continue
        key = (p.get('Batter'), p.get('BTeam'))
        halves[h][key][0] += zv - lv; halves[h][key][1] += 1
    xs, ys2 = [], []
    for key in halves[0]:
        if key in halves[1] and halves[0][key][1] >= 40 and halves[1][key][1] >= 40:
            xs.append(halves[0][key][0] / halves[0][key][1])
            ys2.append(halves[1][key][0] / halves[1][key][1])
    r = pearson(xs, ys2)
    if r and 0 < r < 1:
        nbar = 40  # conservative: threshold as sample proxy
        print(f'\nsprayVal split-half r (>=40 BIP/half): {r:.3f} (n={len(xs)})')


if __name__ == '__main__':
    main()
