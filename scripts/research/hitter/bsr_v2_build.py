"""bsr_v2_build.py — the constructive rebuild of Bat Speed Responsiveness.

bsr_improve.py showed the published metric conflates two nearly
uncorrelated quantities (below-mean slope ~3.2 mph/ft, above-mean ~0.5)
and that its half-foot window has no interior optimum. This build makes
the improved object:

  BSR-up  the lengthening return: slope of bat speed on swing length on
          swings LONGER than the hitter's own mean. What the original
          metric claimed to measure.
  DSC     defensive-swing cost: slope on swings SHORTER than the mean.
          How much bat speed the hitter sacrifices when he shortens up.
          The component that dominated the published metric.

INGREDIENTS (all from the 2026-08-25 batteries):
  1. Player-relative validity gate: keep swings with
     bat_speed >= max(50, ref - OFFSET), ref = median of the hitter's
     top-half bat speeds. OFFSET is the one free constant and is SWEPT
     below on a common-hitter pool; the 50 floor survives because the
     2026 sheet already nulled sub-50 rows at scrape time.
  2. No window: all gated swings enter the fit.
  3. Hinge fit at the hitter's mean gated swing length m:
         bs = a + b_dn * min(sl - m, 0) + b_up * max(sl - m, 0)
     one OLS per hitter-season; per-branch standard errors from the
     coefficient covariance. Qualification: >= 100 gated swings and
     >= 40 on each side of the hinge (stated as a convention).
  4. Empirical-Bayes shrinkage per branch per season: tau^2 by method
     of moments from the raw slopes and their SEs (knob-free).
  5. DSC reported as its own metric, not folded into a single slope.

OBJECTIVES (fixed before the sweep was run):
  S  split-half r (odd/even swings) and year-to-year r per branch.
  V  branch-specific prospective test on each season pair:
         d_bs ~ 1 + dsl_p + dsl_n + up_c + dn_c
                  + b_vu * (dsl_p * up_c) + b_vd * (dsl_n * dn_c)
     dsl_p/dsl_n = positive/negative part of the year-over-year change
     in mean gated swing length, up_c/dn_c = centered prior-year branch
     values. Both interactions are expected POSITIVE (a high BSR-up
     amplifies the gain of a lengthener; a high DSC amplifies the loss
     of a shortener).
  OFFSET decision: interior optimum on the year-to-year curve over the
  common-hitter pool, V as the tiebreaker; if the curve is flat the
  choice is declared a convention.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bsr_v2_build.py
Output: data/_bsr_v2.json, ~/Downloads/bsr_v2_2026.csv, printed tables.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bsr_screen import swing_frame, pearson, name_bridge  # noqa: E402

SEASONS = (2024, 2025, 2026)
PAIRS = ((2024, 2025), (2025, 2026))
OFFSETS = (8, 10, 12, 14, 16, 20, None)   # None = 50-floor only
MIN_SW = 100
MIN_SIDE = 40
FLOOR = 50.0


# ── gate + hinge ────────────────────────────────────────────────────────

def gate(g, offset):
    """Player-relative validity gate on one hitter's swings."""
    bs = g['bs'].to_numpy(float)
    if offset is None:
        return g[bs >= FLOOR]
    top = bs[bs >= np.median(bs)]
    ref = float(np.median(top))
    return g[bs >= max(FLOOR, ref - offset)]


def hinge_fit(sl, bs, min_sw=MIN_SW, min_side=MIN_SIDE):
    """bs = a + b_dn*min(sl-m,0) + b_up*max(sl-m,0), m = mean(sl).
    Returns dict with slopes, SEs, m, counts — or None."""
    ok = np.isfinite(sl) & np.isfinite(bs)
    sl, bs = sl[ok], bs[ok]
    if len(sl) < min_sw:
        return None
    m = float(sl.mean())
    d = sl - m
    x_dn, x_up = np.minimum(d, 0.0), np.maximum(d, 0.0)
    n_dn, n_up = int((d < 0).sum()), int((d > 0).sum())
    if n_dn < min_side or n_up < min_side:
        return None
    X = np.column_stack([np.ones(len(d)), x_dn, x_up])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ (X.T @ bs)
    resid = bs - X @ beta
    s2 = float((resid ** 2).sum()) / (len(d) - 3)
    se = np.sqrt(s2 * np.diag(XtX_inv))
    return {'dn': float(beta[1]), 'up': float(beta[2]),
            'se_dn': float(se[1]), 'se_up': float(se[2]),
            'm': m, 'n': len(d), 'n_dn': n_dn, 'n_up': n_up,
            'mean_bs': float(bs.mean())}


def season_build(frame, offset):
    out = {}
    for hid, g in frame.groupby('hid'):
        gg = gate(g, offset)
        fit = hinge_fit(gg['sl'].to_numpy(float), gg['bs'].to_numpy(float))
        if fit:
            out[hid] = fit
    return out


def eb_shrink(build):
    """Per-branch EB toward the season mean, tau^2 by method of moments.
    Adds up_eb / dn_eb in place; returns the (mu, tau2) pairs."""
    hyper = {}
    for br in ('up', 'dn'):
        b = np.array([v[br] for v in build.values()])
        se = np.array([v[f'se_{br}'] for v in build.values()])
        tau2 = max(float(np.var(b) - np.mean(se ** 2)), 1e-6)
        mu = float(np.mean(b))
        hyper[br] = (mu, tau2)
        for v in build.values():
            k = tau2 / (tau2 + v[f'se_{br}'] ** 2)
            v[f'{br}_eb'] = mu + k * (v[br] - mu)
    return hyper


# ── objectives ──────────────────────────────────────────────────────────

def yy_r(b0, b1, key, pool=None):
    common = [h for h in b0 if h in b1
              and (pool is None or h in pool)]
    return pearson([b0[h][key] for h in common],
                   [b1[h][key] for h in common])


def split_half(frame, offset):
    """Odd/even split-half r per branch (raw hinge slopes)."""
    res = {}
    for br in ('up', 'dn'):
        ha, hb = [], []
        for hid, g in frame.groupby('hid'):
            gg = gate(g, offset)
            vals = []
            for k in (0, 1):
                h = gg.iloc[k::2]
                f = hinge_fit(h['sl'].to_numpy(float),
                              h['bs'].to_numpy(float),
                              min_sw=MIN_SW // 2, min_side=MIN_SIDE // 2)
                vals.append(f[br] if f else np.nan)
            ha.append(vals[0])
            hb.append(vals[1])
        r, n = pearson(ha, hb)
        res[br] = {'r': r, 'sb': (2 * r / (1 + r)) if r is not None else None,
                   'n': n}
    return res


def validity(b0, b1, up_key, dn_key):
    """Branch-specific prospective interaction test on one pair."""
    rows = []
    for h, r0 in b0.items():
        r1 = b1.get(h)
        if not r1:
            continue
        rows.append((r1['m'] - r0['m'], r1['mean_bs'] - r0['mean_bs'],
                     r0[up_key], r0[dn_key]))
    if len(rows) < 60:
        return None
    d_sl, d_bs, up, dn = (np.array(c) for c in zip(*rows))
    up_c, dn_c = up - up.mean(), dn - dn.mean()
    dsl_p, dsl_n = np.maximum(d_sl, 0), np.minimum(d_sl, 0)
    X = np.column_stack([np.ones(len(d_sl)), dsl_p, dsl_n, up_c, dn_c,
                         dsl_p * up_c, dsl_n * dn_c])
    beta, *_ = np.linalg.lstsq(X, d_bs, rcond=None)
    resid = d_bs - X @ beta
    s2 = float((resid ** 2).sum()) / (len(d_sl) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {'b_up': float(beta[5]), 't_up': float(beta[5] / se[5]),
            'b_dn': float(beta[6]), 't_dn': float(beta[6] / se[6]),
            'n': len(d_sl)}


def main():
    print('── loading swing frames ──')
    frames = {y: swing_frame(y) for y in SEASONS}
    for y in SEASONS:
        print(f'  {y}: {len(frames[y])} eligible swings')

    # ── offset sweep ────────────────────────────────────────────────────
    print('\n── OFFSET SWEEP (raw branch slopes, common-hitter pool) ──')
    builds = {off: {y: season_build(frames[y], off) for y in SEASONS}
              for off in OFFSETS}
    # common pool: hitters qualified under EVERY offset in the pair years
    sweep = {}
    print('  offset   yy up 24-25/25-26     yy dn 24-25/25-26     '
          'V t_up / t_dn (pooled pairs)')
    for off in OFFSETS:
        cells_u, cells_d = [], []
        for y0, y1 in PAIRS:
            pool = set.intersection(*[
                set(builds[o][y0]) & set(builds[o][y1]) for o in OFFSETS])
            cells_u.append(yy_r(builds[off][y0], builds[off][y1], 'up', pool))
            cells_d.append(yy_r(builds[off][y0], builds[off][y1], 'dn', pool))
        vs = [validity(builds[off][y0], builds[off][y1], 'up', 'dn')
              for y0, y1 in PAIRS]
        sweep[str(off)] = {'yy_up': cells_u, 'yy_dn': cells_d, 'v': vs}
        label = 'floor' if off is None else f'{off:5d}'
        vtxt = '  '.join(f'{v["t_up"]:+.2f}/{v["t_dn"]:+.2f}' if v else '--'
                         for v in vs)
        print(f'  {label}   '
              + ' '.join(f'{r:+.3f}({n})' for r, n in cells_u) + '   '
              + ' '.join(f'{r:+.3f}({n})' for r, n in cells_d) + '   '
              + vtxt)

    # OFFSET DECISION (2026-08-26, read off the sweep above): the
    # year-to-year curve is monotone toward the loose end and FLAT from
    # 14 through floor-only (within noise); strict gates (8-10) cost
    # stability. No interior optimum exists, so 14 is a CONVENTION: the
    # start of the plateau, keeping the partial-swing guard at zero
    # measurable stability cost. Override with BSR_OFFSET to re-examine.
    OFFSET_FINAL = int(os.environ.get('BSR_OFFSET', '14'))
    print(f'\n── FINAL BUILD (offset {OFFSET_FINAL}) ──')
    B = builds[OFFSET_FINAL]
    hyper = {y: eb_shrink(B[y]) for y in SEASONS}
    for y in SEASONS:
        (mu_u, t_u), (mu_d, t_d) = hyper[y]['up'], hyper[y]['dn']
        print(f'  {y}: {len(B[y])} hitters | up mu {mu_u:.2f} tau '
              f'{t_u ** .5:.2f} | dn mu {mu_d:.2f} tau {t_d ** .5:.2f}')

    print('\n── S: split-half (raw branches) ──')
    sh = {}
    for y in SEASONS:
        sh[y] = split_half(frames[y], OFFSET_FINAL)
        print(f'  {y}: up r {sh[y]["up"]["r"]:.3f} '
              f'(full {sh[y]["up"]["sb"]:.3f}) | dn r '
              f'{sh[y]["dn"]["r"]:.3f} (full {sh[y]["dn"]["sb"]:.3f})')

    print('\n── S: year-to-year (EB values, full pool) ──')
    yy_final = {}
    for br in ('up_eb', 'dn_eb'):
        cells = [yy_r(B[y0], B[y1], br) for y0, y1 in PAIRS]
        yy_final[br] = cells
        print(f'  {br}: ' + '  '.join(f'{r:+.3f} (n {n})' for r, n in cells))

    print('\n── V: prospective test (EB values) ──')
    v_final = {}
    for y0, y1 in PAIRS:
        v = validity(B[y0], B[y1], 'up_eb', 'dn_eb')
        v_final[f'{y0}-{y1}'] = v
        print(f'  {y0}->{y1}: b_up {v["b_up"]:+.3f} (t {v["t_up"]:+.2f}) | '
              f'b_dn {v["b_dn"]:+.3f} (t {v["t_dn"]:+.2f}) | n {v["n"]}')

    # ── characterization: what do the branches correlate with? ─────────
    print('\n── characterization vs 2026 levels (descriptive) ──')
    bt26 = json.load(open(os.path.join(
        ROOT, 'data', '_bt_seasons.json')))['2026']
    chars = {}
    for br in ('up_eb', 'dn_eb'):
        for stat in ('avg_sweetspot_speed_mph', 'swing_length_qualified',
                     'strike_swinging_per_swing', 'squared_up_per_swing'):
            xs, ys = [], []
            for h, v in B[2026].items():
                row = bt26.get(h)
                if not row or row.get(stat) is None:
                    continue
                try:
                    xs.append(float(row[stat]))
                except (TypeError, ValueError):
                    continue
                ys.append(v[br])
            r, n = pearson(ys, xs)
            chars[f'{br}~{stat}'] = r
            print(f'  {br:6s} ~ {stat:28s} r {r:+.3f} (n {n})')

    # ── 2026 leaderboard + CSV ──────────────────────────────────────────
    names = {}
    for pid, row in bt26.items():
        if row.get('name'):
            names[pid] = row['name']
    board = sorted(B[2026].items(), key=lambda kv: -kv[1]['up_eb'])
    print('\n── 2026 BSR-up leaders (EB, mph/ft) ──')
    for h, v in board[:8]:
        print(f'  {names.get(h, h):24s} up {v["up_eb"]:5.2f}  '
              f'dsc {v["dn_eb"]:5.2f}  (n {v["n"]})')
    print('  ...')
    for h, v in board[-8:]:
        print(f'  {names.get(h, h):24s} up {v["up_eb"]:5.2f}  '
              f'dsc {v["dn_eb"]:5.2f}  (n {v["n"]})')

    csv_path = os.path.expanduser('~/Downloads/bsr_v2_2026.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['player', 'mlbam_id', 'bsr_up', 'dsc', 'bsr_up_raw',
                    'dsc_raw', 'mean_swing_length_ft', 'swings',
                    'swings_short_side', 'swings_long_side'])
        for h, v in board:
            w.writerow([names.get(h, h), h, round(v['up_eb'], 2),
                        round(v['dn_eb'], 2), round(v['up'], 2),
                        round(v['dn'], 2), round(v['m'], 2),
                        v['n'], v['n_dn'], v['n_up']])
    print(f'\nwrote {csv_path} ({len(board)} hitters)')

    out = os.path.join(ROOT, 'data', '_bsr_v2.json')
    payload = {
        'offset_sweep': sweep, 'offset_final': OFFSET_FINAL,
        'hyper': {str(y): hyper[y] for y in SEASONS},
        'split_half': {str(y): sh[y] for y in SEASONS},
        'yy_final': yy_final, 'validity_final': v_final,
        'characterization': chars,
        'values': {str(y): {h: {k: v[k] for k in
                                ('up', 'dn', 'up_eb', 'dn_eb', 'se_up',
                                 'se_dn', 'm', 'n', 'n_up', 'n_dn')}
                            for h, v in B[y].items()} for y in SEASONS},
    }
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=1, default=float)
    os.replace(tmp, out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
