"""derive_weights_multiseason.py — BB+ / Hitter+ weights from 2021-2025 season pairs.

The half-season derivation (derive_hitterplus_weights.py) could not price SD+
and CT+ (betas noise-level at n=209). This runs the same protocol at season
scale: FULL-SEASON year-N components → FULL-SEASON year-N+1 wOBA, over pairs
21→22, 22→23, 23→24, 24→25, components z-scored within pair, rows pooled.

Reports:
  1. Component correlations + univariate predictive r (pooled).
  2. BB+ internal 2-predictor ridge (con, sp) — re-derives 0.85/0.15.
  3. Hitter+ 3-predictor ridge (BB+ @ current 85/15, SD+, CT+) — re-derives 70/15/15.
  4. 4-predictor ridge (con, sp, sd, ct) — joint check.
  5. Per-pair OLS betas — year-to-year stability of the weights.
  6. Leave-one-pair-out evaluation of candidate weight vectors — the decision
     table: does re-weighting actually predict better out of sample?

Usage: python3 scripts/research/hitter/derive_weights_multiseason.py
"""
import os, sys, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
from pipeline.utils import safe_float, BUNT_BB_TYPES, spray_angle, spray_direction

LA_BINS = [(-999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
           (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 999)]
MIN_BIP_ZONE = 20
MIN_BIP_HITTER = 80    # production BB_PLUS_MIN_BIP
MIN_Y_EVENTS = 200
W_CON, W_SP = 0.85, 0.15
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]


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


def pear(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa <= 0 or sb <= 0:
        return None
    return sum((x - ma) * (v - mb) for x, v in zip(a, b)) / (sa * sb)


def ridge_k(Z, y, lam):
    """k-predictor ridge on standardized predictors via Gaussian elimination.
    Z: list of k lists. Returns list of k betas."""
    k, n = len(Z), len(y)
    my = sum(y) / n
    yc = [v - my for v in y]
    S = [[sum(Z[i][t] * Z[j][t] for t in range(n)) + (lam * n if i == j else 0.0)
          for j in range(k)] for i in range(k)]
    b = [sum(Z[i][t] * yc[t] for t in range(n)) for i in range(k)]
    M = [row[:] + [b[i]] for i, row in enumerate(S)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(k):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, k + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][k] / M[i][i] for i in range(k)]


def build_pair(year_n, year_n1):
    """Return rows [(con_plus, sp_plus, sdPlus, ctPlus, y), ...] for one pair."""
    P = A.season_dicts(year_n)
    lg, sc = A.GUTS[year_n]

    byh = defaultdict(list)
    for p in P:
        if p['Batter']:
            byh[(p['Batter'], 'X')].append(p)
    sd_res, _ = sd.compute_sd_plus(P, dict(byh), lg, sc)
    ct_res, _ = ct.compute_ct_plus(P, dict(byh), lg, sc)

    # league zone tables (full year N) for BB+ ingredients
    acc = defaultdict(lambda: [0.0, 0])
    lg_xc_acc = [0.0, 0]
    for p in P:
        if p['Description'] != 'In Play':
            continue
        xw = p.get('xwOBA')
        if xw is not None:
            lg_xc_acc[0] += xw
            lg_xc_acc[1] += 1
        k = classify(p)
        if k is None:
            continue
        w = A.BIP_WOBA_PUB.get(p.get('event_raw'), 0.0 if p.get('event_raw') else None)
        if w is None:
            continue
        d, bi, bats = k
        for key in ((d, bi, bats), (d, bi), (bi, bats), (bi,)):
            acc[key][0] += w
            acc[key][1] += 1
    lg_xc = lg_xc_acc[0] / lg_xc_acc[1]

    def lookup(*keys):
        for key in keys:
            s, n = acc.get(key, (0.0, 0))
            if n >= MIN_BIP_ZONE:
                return s / n
        return None

    per = defaultdict(lambda: {'con': [0.0, 0], 'sp': [0.0, 0]})
    for p in P:
        h = p['Batter']
        if not h or p['Description'] != 'In Play':
            continue
        a = per[h]
        k = classify(p)
        xw = p.get('xwOBA')
        if xw is not None and k is not None:
            a['con'][0] += xw
            a['con'][1] += 1
        if k is not None:
            d, bi, bats = k
            zv = lookup((d, bi, bats), (d, bi))
            lv = lookup((bi, bats), (bi,))
            if zv is not None and lv is not None:
                a['sp'][0] += zv - lv
                a['sp'][1] += 1

    y_map = A.target_y(year_n1)

    rows = []
    for h, a in per.items():
        if a['con'][1] < MIN_BIP_HITTER or a['sp'][1] < MIN_BIP_HITTER:
            continue
        sdv = sd_res.get((h, 'X'))
        ctv = ct_res.get((h, 'X'))
        if sdv is None or ctv is None:
            continue
        yv = y_map.get(h)
        if yv is None or yv[1] < MIN_Y_EVENTS:
            continue
        con_plus = 100.0 * (a['con'][0] / a['con'][1]) / lg_xc
        sp_plus = 100.0 * (lg_xc + a['sp'][0] / a['sp'][1]) / lg_xc
        rows.append((con_plus, sp_plus, sdv['sdPlus'], ctv['ctPlus'],
                     yv[0] / yv[1]))
    del P
    return rows


def main():
    pair_rows = {}
    for yn, yn1 in PAIRS:
        rows = build_pair(yn, yn1)
        pair_rows[(yn, yn1)] = rows
        print(f"pair {yn}->{yn1}: {len(rows)} hitters", flush=True)

    # z-score components within pair, pool
    pooled = {k: [] for k in ('con', 'sp', 'sd', 'ct', 'bb', 'y', 'pair')}
    per_pair_z = {}
    for pk, rows in pair_rows.items():
        if len(rows) < 30:
            continue
        con = zscore([r[0] for r in rows])
        sp = zscore([r[1] for r in rows])
        sdz = zscore([r[2] for r in rows])
        ctz = zscore([r[3] for r in rows])
        bb = zscore([W_CON * r[0] + W_SP * r[1] for r in rows])
        y = [r[4] for r in rows]
        per_pair_z[pk] = (con, sp, sdz, ctz, bb, y)
        pooled['con'] += con
        pooled['sp'] += sp
        pooled['sd'] += sdz
        pooled['ct'] += ctz
        pooled['bb'] += bb
        pooled['y'] += y
        pooled['pair'] += [pk] * len(y)

    n = len(pooled['y'])
    print(f"\npooled: {n} hitter-pairs")

    print("\ncomponent correlations (pooled z):")
    for a, b in (('con', 'sp'), ('con', 'sd'), ('con', 'ct'), ('sp', 'sd'),
                 ('sp', 'ct'), ('sd', 'ct'), ('bb', 'sd'), ('bb', 'ct')):
        print(f"  {a}-{b}: {pear(pooled[a], pooled[b]):+.3f}")
    print("\nunivariate r vs next-season wOBA:")
    for k in ('con', 'sp', 'bb', 'sd', 'ct'):
        print(f"  {k}: {pear(pooled[k], pooled['y']):+.3f}")

    print(f"\nBB+ internal (con, sp) ridge  [current {W_CON:.2f}/{W_SP:.2f}]:")
    print(f"{'lambda':>7s} {'w_con':>7s} {'w_sp':>7s}")
    for lam in (0.0, 0.05, 0.1, 0.25, 0.5):
        b = ridge_k([pooled['con'], pooled['sp']], pooled['y'], lam)
        tot = sum(abs(x) for x in b)
        print(f"{lam:7.2f} {abs(b[0])/tot:7.3f} {abs(b[1])/tot:7.3f}"
              f"   raw: {b[0]:+.5f} {b[1]:+.5f}")

    print("\nHitter+ (bb, sd, ct) ridge  [current 0.70/0.15/0.15]:")
    print(f"{'lambda':>7s} {'w_bb':>7s} {'w_sd':>7s} {'w_ct':>7s}")
    for lam in (0.0, 0.05, 0.1, 0.25, 0.5):
        b = ridge_k([pooled['bb'], pooled['sd'], pooled['ct']], pooled['y'], lam)
        tot = sum(abs(x) for x in b)
        print(f"{lam:7.2f} {abs(b[0])/tot:7.3f} {abs(b[1])/tot:7.3f} {abs(b[2])/tot:7.3f}"
              f"   raw: {b[0]:+.5f} {b[1]:+.5f} {b[2]:+.5f}")

    print("\n4-predictor (con, sp, sd, ct) ridge:")
    for lam in (0.05, 0.1, 0.25):
        b = ridge_k([pooled['con'], pooled['sp'], pooled['sd'], pooled['ct']],
                    pooled['y'], lam)
        tot = sum(abs(x) for x in b)
        print(f"{lam:7.2f} " + ' '.join(f"{abs(x)/tot:6.3f}" for x in b)
              + "   raw: " + ' '.join(f"{x:+.5f}" for x in b))

    print("\nper-pair OLS (bb, sd, ct) normalized |beta| (stability):")
    for pk, (con, sp, sdz, ctz, bb, y) in sorted(per_pair_z.items()):
        b = ridge_k([bb, sdz, ctz], y, 0.05)
        tot = sum(abs(x) for x in b)
        print(f"  {pk[0]}->{pk[1]}: "
              + ' '.join(f"{abs(x)/tot:.3f}" for x in b)
              + "   raw: " + ' '.join(f"{x:+.5f}" for x in b)
              + f"   (n={len(y)})")

    # leave-one-pair-out candidate evaluation
    CANDS = {
        'current 70/15/15': (0.70, 0.15, 0.15),
        'old 65/7/28':      (0.65, 0.07, 0.28),
        '80/10/10':         (0.80, 0.10, 0.10),
        '60/20/20':         (0.60, 0.20, 0.20),
        'bb-only 100/0/0':  (1.00, 0.00, 0.00),
    }
    print("\nleave-one-pair-out r of composite vs next wOBA:")
    results = {k: [] for k in CANDS}
    results['LOPO-derived'] = []
    for hold in per_pair_z:
        train_bb, train_sd, train_ct, train_y = [], [], [], []
        for pk, (con, sp, sdz, ctz, bb, y) in per_pair_z.items():
            if pk == hold:
                continue
            train_bb += bb; train_sd += sdz; train_ct += ctz; train_y += y
        bder = ridge_k([train_bb, train_sd, train_ct], train_y, 0.05)
        tot = sum(abs(x) for x in bder)
        wder = tuple(abs(x) / tot for x in bder)
        con, sp, sdz, ctz, bb, y = per_pair_z[hold]
        for name, w in list(CANDS.items()) + [('LOPO-derived', wder)]:
            comp = [w[0] * bb[i] + w[1] * sdz[i] + w[2] * ctz[i]
                    for i in range(len(y))]
            results[name].append(pear(comp, y))
    for name, rs in results.items():
        print(f"  {name:18s} mean r {sum(rs)/len(rs):+.4f}   "
              + ' '.join(f"{r:+.3f}" for r in rs))


if __name__ == '__main__':
    main()
