"""bsr_v2_build.py — the constructive rebuild of Bat Speed Responsiveness.

bsr_improve.py showed the published metric conflates two nearly
uncorrelated quantities (below-mean slope ~3.2 mph/ft, above-mean ~0.5)
and that its half-foot window has no interior optimum. bsr_v2_reach.py
then showed the above-mean side is itself two regimes: a small true
lengthening return near the hitter's norm and a steep penalty on reach
swings beyond it. The final build therefore produces THREE metrics:

  BSR-near  the lengthening return: slope of bat speed on swing length
            on swings 0 to CAP ft LONGER than the hitter's own mean.
            The validity-bearing headroom metric — cap 0.6 beat the
            uncapped branch on the prospective test in both season
            pairs, with an interior optimum on the cap sweep.
  BSR-reach the reach bleed: slope on swings more than CAP ft beyond
            the mean. A stable descriptive trait (how much bat speed
            the hitter loses on reach/lunge swings). Estimated only
            where the hitter has >= 20 such swings.
  DSC       defensive-swing cost: slope on swings SHORTER than the
            mean. The component that dominated the published metric.

INGREDIENTS (all from the 2026-08-25/26 batteries):
  1. Player-relative validity gate: keep swings with
     bat_speed >= max(50, ref - OFFSET), ref = median of the hitter's
     top-half bat speeds. OFFSET swept below; the year-to-year curve is
     flat from 14 through floor-only and strict gates cost stability,
     so 14 is a stated CONVENTION at the plateau edge.
  2. No window: all gated swings enter the fit.
  3. Three-regime hinge at the hitter's mean gated swing length m:
         bs = a + b_dn * min(d, 0) + b_near * clip(d, 0, CAP)
                + b_reach * max(d - CAP, 0),      d = sl - m
     one OLS per hitter-season; per-branch SEs from the coefficient
     covariance. CAP = 0.6 ft: interior optimum on the min-across-pairs
     prospective-validity objective (bsr_v2_reach.py sweep over
     {0.4, 0.6, 0.8, 1.0}: t 1.58 / 2.09 / 1.67 / 0.94). Qualification:
     >= 100 gated swings, >= 40 below the hinge, >= 40 in (0, CAP].
  4. Empirical-Bayes shrinkage per branch per season: tau^2 by method
     of moments from the raw slopes and their SEs (knob-free).
  5. The branches ship separately, never blended into one slope.

OBJECTIVES reported: split-half r (odd/even swings), year-to-year r,
and the branch-specific prospective test
    d_bs ~ 1 + dsl_p + dsl_n + near_c + dn_c
             + b_vn * (dsl_p * near_c) + b_vd * (dsl_n * dn_c)
per season pair (both interactions expected positive).

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
CAP = 0.6
MIN_SW = 100
MIN_SIDE = 40
FLOOR = 50.0
BRANCHES = ('near', 'dn', 'reach')


# ── gate + two-regime hinge (kept for the offset sweep) ────────────────

def gate(g, offset):
    """Player-relative validity gate on one hitter's swings."""
    bs = g['bs'].to_numpy(float)
    if offset is None:
        return g[bs >= FLOOR]
    top = bs[bs >= np.median(bs)]
    ref = float(np.median(top))
    return g[bs >= max(FLOOR, ref - offset)]


def hinge_fit(sl, bs, min_sw=MIN_SW, min_side=MIN_SIDE):
    """Two-regime v2 fit, kept because the offset sweep predates the
    reach split and bsr_v2_reach.py imports it as its baseline."""
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


def eb_shrink(build, branches=BRANCHES):
    """Per-branch EB toward the season mean, tau^2 by method of moments.
    Adds <br>_eb in place (only where the raw value exists); returns the
    (mu, tau2) pairs."""
    hyper = {}
    for br in branches:
        vals = [(v[br], v[f'se_{br}']) for v in build.values()
                if v.get(br) is not None and v.get(f'se_{br}') is not None]
        if len(vals) < 30:
            continue
        b = np.array([x for x, _ in vals])
        se = np.array([x for _, x in vals])
        tau2 = max(float(np.var(b) - np.mean(se ** 2)), 1e-6)
        mu = float(np.mean(b))
        hyper[br] = (mu, tau2)
        for v in build.values():
            if v.get(br) is None or v.get(f'se_{br}') is None:
                v[f'{br}_eb'] = None
                continue
            k = tau2 / (tau2 + v[f'se_{br}'] ** 2)
            v[f'{br}_eb'] = mu + k * (v[br] - mu)
    return hyper


# ── objectives ──────────────────────────────────────────────────────────

def yy_r(b0, b1, key, pool=None):
    common = [h for h in b0 if h in b1
              and (pool is None or h in pool)
              and b0[h].get(key) is not None and b1[h].get(key) is not None]
    return pearson([b0[h][key] for h in common],
                   [b1[h][key] for h in common])


def validity(b0, b1, near_key, dn_key):
    """Branch-specific prospective interaction test on one pair."""
    rows = []
    for h, r0 in b0.items():
        r1 = b1.get(h)
        if not r1 or r0.get(near_key) is None or r0.get(dn_key) is None:
            continue
        rows.append((r1['m'] - r0['m'], r1['mean_bs'] - r0['mean_bs'],
                     r0[near_key], r0[dn_key]))
    if len(rows) < 60:
        return None
    d_sl, d_bs, near, dn = (np.array(c) for c in zip(*rows))
    near_c, dn_c = near - near.mean(), dn - dn.mean()
    dsl_p, dsl_n = np.maximum(d_sl, 0), np.minimum(d_sl, 0)
    X = np.column_stack([np.ones(len(d_sl)), dsl_p, dsl_n, near_c, dn_c,
                         dsl_p * near_c, dsl_n * dn_c])
    beta, *_ = np.linalg.lstsq(X, d_bs, rcond=None)
    resid = d_bs - X @ beta
    s2 = float((resid ** 2).sum()) / (len(d_sl) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {'b_near': float(beta[5]), 't_near': float(beta[5] / se[5]),
            'b_dn': float(beta[6]), 't_dn': float(beta[6] / se[6]),
            'n': len(d_sl)}


def main():
    from bsr_v2_reach import three_regime  # late import: avoids a cycle

    def season_build_final(frame):
        out = {}
        for hid, g in frame.groupby('hid'):
            gg = gate(g, OFFSET_FINAL)
            fit = three_regime(gg['sl'].to_numpy(float),
                               gg['bs'].to_numpy(float), CAP)
            if fit:
                fit['n'] = fit['n_dn'] + fit['n_near'] + fit['n_reach']
                out[hid] = fit
        return out

    def split_half_final(frame):
        res = {}
        for br in ('near', 'dn'):
            ha, hb = [], []
            for hid, g in frame.groupby('hid'):
                gg = gate(g, OFFSET_FINAL)
                vals = []
                for k in (0, 1):
                    h = gg.iloc[k::2]
                    f = three_regime(h['sl'].to_numpy(float),
                                     h['bs'].to_numpy(float), CAP,
                                     min_sw=MIN_SW // 2,
                                     min_side=MIN_SIDE // 2, min_reach=10)
                    vals.append(f[br] if f else np.nan)
                ha.append(vals[0])
                hb.append(vals[1])
            r, n = pearson(ha, hb)
            res[br] = {'r': r,
                       'sb': (2 * r / (1 + r)) if r is not None else None,
                       'n': n}
        return res

    print('── loading swing frames ──')
    frames = {y: swing_frame(y) for y in SEASONS}
    for y in SEASONS:
        print(f'  {y}: {len(frames[y])} eligible swings')

    # ── offset sweep (two-regime, predates the reach split) ────────────
    print('\n── OFFSET SWEEP (raw branch slopes, common-hitter pool) ──')
    builds = {off: {y: season_build(frames[y], off) for y in SEASONS}
              for off in OFFSETS}
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
        vtxt = '  '.join(f'{v["t_near"]:+.2f}/{v["t_dn"]:+.2f}' if v else '--'
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
    global OFFSET_FINAL
    OFFSET_FINAL = int(os.environ.get('BSR_OFFSET', '14'))

    # ── FINAL BUILD: three regimes at CAP (see docstring for the cap
    # decision; sweep lives in bsr_v2_reach.py) ─────────────────────────
    print(f'\n── FINAL BUILD (offset {OFFSET_FINAL}, cap {CAP} ft, '
          f'three regimes) ──')
    B = {y: season_build_final(frames[y]) for y in SEASONS}
    hyper = {y: eb_shrink(B[y]) for y in SEASONS}
    for y in SEASONS:
        n_reach = sum(1 for v in B[y].values() if v.get('reach') is not None)
        parts = []
        for br in BRANCHES:
            if br in hyper[y]:
                mu, t2 = hyper[y][br]
                parts.append(f'{br} mu {mu:+.2f} tau {t2 ** .5:.2f}')
        print(f'  {y}: {len(B[y])} hitters ({n_reach} with a reach '
              f'estimate) | ' + ' | '.join(parts))

    print('\n── S: split-half (raw near/dn branches) ──')
    sh = {}
    for y in SEASONS:
        sh[y] = split_half_final(frames[y])
        print(f'  {y}: near r {sh[y]["near"]["r"]:.3f} '
              f'(full {sh[y]["near"]["sb"]:.3f}) | dn r '
              f'{sh[y]["dn"]["r"]:.3f} (full {sh[y]["dn"]["sb"]:.3f})')

    print('\n── S: year-to-year (EB values, full pool) ──')
    yy_final = {}
    for br in ('near_eb', 'dn_eb', 'reach_eb'):
        cells = [yy_r(B[y0], B[y1], br) for y0, y1 in PAIRS]
        yy_final[br] = cells
        print(f'  {br:8s}: ' + '  '.join(
            f'{r:+.3f} (n {n})' if r is not None else '   --    '
            for r, n in cells))

    print('\n── V: prospective test (EB values) ──')
    v_final = {}
    for y0, y1 in PAIRS:
        v = validity(B[y0], B[y1], 'near_eb', 'dn_eb')
        v_final[f'{y0}-{y1}'] = v
        print(f'  {y0}->{y1}: b_near {v["b_near"]:+.3f} '
              f'(t {v["t_near"]:+.2f}) | b_dn {v["b_dn"]:+.3f} '
              f'(t {v["t_dn"]:+.2f}) | n {v["n"]}')

    # ── characterization: what do the branches correlate with? ─────────
    print('\n── characterization vs 2026 levels (descriptive) ──')
    bt26 = json.load(open(os.path.join(
        ROOT, 'data', '_bt_seasons.json')))['2026']
    chars = {}
    for br in ('near_eb', 'reach_eb', 'dn_eb'):
        for stat in ('avg_sweetspot_speed_mph', 'swing_length_qualified',
                     'strike_swinging_per_swing', 'squared_up_per_swing'):
            xs, ys = [], []
            for h, v in B[2026].items():
                row = bt26.get(h)
                if not row or row.get(stat) is None or v.get(br) is None:
                    continue
                try:
                    xs.append(float(row[stat]))
                except (TypeError, ValueError):
                    continue
                ys.append(v[br])
            r, n = pearson(ys, xs)
            chars[f'{br}~{stat}'] = r
            print(f'  {br:8s} ~ {stat:28s} r {r:+.3f} (n {n})')

    # ── 2026 leaderboard + CSV ──────────────────────────────────────────
    names = {}
    for pid, row in bt26.items():
        if row.get('name'):
            names[pid] = row['name']
    board = sorted(B[2026].items(), key=lambda kv: -kv[1]['near_eb'])
    print('\n── 2026 BSR-near leaders (EB, mph/ft) ──')
    for h, v in board[:8]:
        rch = f'{v["reach_eb"]:5.2f}' if v.get('reach_eb') is not None \
            else '   --'
        print(f'  {names.get(h, h):24s} near {v["near_eb"]:5.2f}  '
              f'reach {rch}  dsc {v["dn_eb"]:5.2f}  (n {v["n"]})')
    print('  ...')
    for h, v in board[-8:]:
        rch = f'{v["reach_eb"]:5.2f}' if v.get('reach_eb') is not None \
            else '   --'
        print(f'  {names.get(h, h):24s} near {v["near_eb"]:5.2f}  '
              f'reach {rch}  dsc {v["dn_eb"]:5.2f}  (n {v["n"]})')

    csv_path = os.path.expanduser('~/Downloads/bsr_v2_2026.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['player', 'mlbam_id', 'bsr_near', 'bsr_reach', 'dsc',
                    'bsr_near_raw', 'bsr_reach_raw', 'dsc_raw',
                    'mean_swing_length_ft', 'swings', 'swings_short_side',
                    'swings_near_side', 'swings_reach_side'])
        rnd = lambda x: round(x, 2) if x is not None else ''
        for h, v in board:
            w.writerow([names.get(h, h), h, rnd(v['near_eb']),
                        rnd(v.get('reach_eb')), rnd(v['dn_eb']),
                        rnd(v['near']), rnd(v.get('reach')), rnd(v['dn']),
                        round(v['m'], 2), v['n'], v['n_dn'], v['n_near'],
                        v['n_reach']])
    print(f'\nwrote {csv_path} ({len(board)} hitters)')

    out = os.path.join(ROOT, 'data', '_bsr_v2.json')
    keys = ('near', 'dn', 'reach', 'near_eb', 'dn_eb', 'reach_eb',
            'se_near', 'se_dn', 'se_reach', 'm', 'n', 'n_near', 'n_dn',
            'n_reach')
    payload = {
        'cap': CAP, 'offset_final': OFFSET_FINAL,
        'offset_sweep': sweep,
        'hyper': {str(y): hyper[y] for y in SEASONS},
        'split_half': {str(y): sh[y] for y in SEASONS},
        'yy_final': yy_final, 'validity_final': v_final,
        'characterization': chars,
        'values': {str(y): {h: {k: v.get(k) for k in keys}
                            for h, v in B[y].items()} for y in SEASONS},
    }
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=1, default=float)
    os.replace(tmp, out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
