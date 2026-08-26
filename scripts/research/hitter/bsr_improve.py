"""bsr_improve.py — can Bat Speed Responsiveness itself be built better?

Follow-up to bsr_screen.py (which rejected BSR as a Hitter+ input). This
battery evaluates candidate IMPROVEMENTS to the metric on its own terms,
judged by the two things a metric of this kind must do:

  S  stability: odd/even split-half r (Spearman-Brown to full length)
     and year-to-year r (2024->2025, 2025->2026).
  V  prospective validity, the paper's own purpose: prior-year BSR
     should moderate the bat-speed return a hitter actually realizes
     when his mean swing length changes the next season. Test:
         d_bs ~ b0 + b1 * d_sl + b2 * BSR_c + b3 * (d_sl * BSR_c)
     per season pair, where d_sl / d_bs are the year-over-year changes
     in mean competitive-swing length / bat speed and BSR_c is the
     prior-year estimate, centered. b3 > 0 with a real t-stat is the
     validity signal (the paper's p=.005 test, continuous form).

VARIANTS (each produces dict[hid] -> slope for a season):
  w025 / w050 / w075 / w100 / wINF   window sweep around the hitter's
        mean swing length (paper = 0.5 ft, adopted unswept there)
  adj   w050 on residuals of pooled-league OLS (plate_z, side-signed
        plate_x, release_speed) — measurement-context controls
  eb    w050 with empirical-Bayes shrinkage toward the season mean;
        prior variance tau^2 by method of moments from the season's
        slopes and their OLS standard errors (no free knob)
  up / dn  asymmetric: slope on swings above / below the hitter's mean
        only (0 to +0.5 ft / -0.5 to 0 ft) — tests the linearity
        assumption; report also r(up, dn) within season
  nofoul  w050 excluding foul contact (timing-mistake proxy)

All variants share the >= 50 mph validity gate and >= 100 windowed
swings (except up/dn: >= 60 per side). d_sl / d_bs use ALL eligible
swings, identical across variants.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bsr_improve.py
Output: data/_bsr_improve_results.json + printed tables.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bsr_screen import swing_frame, pooled_residuals, pearson  # noqa: E402

SEASONS = (2024, 2025, 2026)
PAIRS = ((2024, 2025), (2025, 2026))
MIN_SW = 100
MIN_SIDE = 60
FOUL_DESCS = {'foul', 'foul_tip', 'Foul', 'Foul Tip'}


def slope_se(x, y):
    """OLS slope and its standard error on centered pairs."""
    x = x - x.mean()
    y = y - y.mean()
    sxx = float((x * x).sum())
    if sxx <= 0 or len(x) < 3:
        return None
    b = float((x * y).sum()) / sxx
    resid = y - b * x
    s2 = float((resid * resid).sum()) / (len(x) - 2)
    return b, (s2 / sxx) ** 0.5


def windowed(g, bs_col, sl_col, w, min_n=MIN_SW, side=None):
    sl = g[sl_col].to_numpy(float)
    bs = g[bs_col].to_numpy(float)
    ok = np.isfinite(sl) & np.isfinite(bs)
    sl, bs = sl[ok], bs[ok]
    if not len(sl):
        return None
    d = sl - sl.mean()
    if side == 'up':
        m = (d > 0) & (d <= w)
    elif side == 'dn':
        m = (d < 0) & (d >= -w)
    else:
        m = np.abs(d) <= w
    if m.sum() < min_n or sl[m].std() < 0.03:
        return None
    return slope_se(sl[m], bs[m])


def season_variants(year):
    sw = pooled_residuals(swing_frame(year)).reset_index(drop=True)
    per_h = {}
    for hid, g in sw.groupby('hid'):
        rec = {}
        for w, name in ((0.25, 'w025'), (0.5, 'w050'), (0.75, 'w075'),
                        (1.0, 'w100'), (99.0, 'wINF')):
            r = windowed(g, 'bs', 'sl', w)
            if r:
                rec[name] = r
        ga = g.dropna(subset=['bs_r', 'sl_r'])
        r = windowed(ga, 'bs_r', 'sl_r', 0.5) if len(ga) else None
        if r:
            rec['adj'] = r
        for side in ('up', 'dn'):
            r = windowed(g, 'bs', 'sl', 0.5, min_n=MIN_SIDE, side=side)
            if r:
                rec[side] = r
        if rec:
            per_h[hid] = rec
            rec['n'] = len(g)
            rec['mean_sl'] = float(g['sl'].mean())
            rec['mean_bs'] = float(g['bs'].mean())
    # empirical-Bayes on w050: tau^2 by method of moments
    slopes = np.array([v['w050'][0] for v in per_h.values() if 'w050' in v])
    ses = np.array([v['w050'][1] for v in per_h.values() if 'w050' in v])
    if len(slopes) > 30:
        tau2 = max(float(np.var(slopes) - np.mean(ses ** 2)), 1e-6)
        mu = float(np.mean(slopes))
        for v in per_h.values():
            if 'w050' in v:
                b, se = v['w050']
                k = tau2 / (tau2 + se ** 2)
                v['eb'] = (mu + k * (b - mu), se)
    return per_h


def season_variants_nofoul(year):
    """w050 excluding fouls — needs description, so a separate pass."""
    if year == 2026:
        raw = pd.read_pickle(os.path.join(ROOT, 'data',
                                          'all_pitches_rs_cache.pkl'))
        df = pd.DataFrame(raw)
        df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
        from bsr_screen import name_bridge
        f = pd.DataFrame({
            'hid': df['Batter'].astype(str).map(name_bridge()),
            'bs': pd.to_numeric(df['BatSpeed'], errors='coerce'),
            'sl': pd.to_numeric(df['SwingLength'], errors='coerce'),
            'desc': df['Description'].astype(str),
        })
    else:
        bt = pd.read_pickle(os.path.join(ROOT, 'data',
                                         f'_bsr_swing_{year}.pkl'))
        f = pd.DataFrame({
            'hid': bt['batter'].astype('Int64').astype(str),
            'bs': pd.to_numeric(bt['bat_speed'], errors='coerce'),
            'sl': pd.to_numeric(bt['swing_length'], errors='coerce'),
            'desc': bt['description'].astype(str),
        })
    m = (f['bs'].notna() & (f['bs'] >= 50) & f['sl'].notna()
         & f['hid'].notna() & ~f['desc'].isin(FOUL_DESCS))
    f = f[m]
    out = {}
    for hid, g in f.groupby('hid'):
        r = windowed(g, 'bs', 'sl', 0.5)
        if r:
            out[hid] = r
    return out


def split_half(year):
    """Split-half r of the w050 slope (odd/even swings)."""
    sw = swing_frame(year)
    ha, hb = [], []
    for hid, g in sw.groupby('hid'):
        vals = []
        for k in (0, 1):
            r = windowed(g.iloc[k::2], 'bs', 'sl', 0.5, min_n=MIN_SW // 2)
            vals.append(r[0] if r else np.nan)
        ha.append(vals[0])
        hb.append(vals[1])
    r, n = pearson(ha, hb)
    return r, (2 * r / (1 + r)) if r is not None else None, n


def interaction_test(v0, v1, key):
    """d_bs ~ d_sl + BSR_c + d_sl*BSR_c. Returns (b3, t, n)."""
    rows = []
    for hid, rec in v0.items():
        if key not in rec or hid not in v1:
            continue
        r1 = v1[hid]
        if 'mean_sl' not in rec or 'mean_sl' not in r1:
            continue
        rows.append((r1['mean_sl'] - rec['mean_sl'],
                     r1['mean_bs'] - rec['mean_bs'],
                     rec[key][0]))
    if len(rows) < 60:
        return None
    d_sl, d_bs, bsr = (np.array(c) for c in zip(*rows))
    bsr_c = bsr - bsr.mean()
    X = np.column_stack([np.ones(len(d_sl)), d_sl, bsr_c, d_sl * bsr_c])
    beta, *_ = np.linalg.lstsq(X, d_bs, rcond=None)
    resid = d_bs - X @ beta
    s2 = float((resid ** 2).sum()) / (len(d_sl) - 4)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[3]), float(beta[3] / cov[3, 3] ** 0.5), len(d_sl)


def main():
    print('── building variants ──')
    V = {y: season_variants(y) for y in SEASONS}
    NF = {y: season_variants_nofoul(y) for y in SEASONS}
    for y in SEASONS:
        for hid, r in NF[y].items():
            if hid in V[y]:
                V[y][hid]['nofoul'] = r
        print(f'  {y}: {len(V[y])} hitters')

    variants = ['w025', 'w050', 'w075', 'w100', 'wINF', 'adj', 'eb',
                'nofoul', 'up', 'dn']
    results = {'yy': {}, 'validity': {}, 'split_half': {}, 'updn': {}}

    print('\n── split-half (w050 baseline) ──')
    for y in SEASONS:
        r, sb, n = split_half(y)
        results['split_half'][y] = {'r': r, 'sb': sb, 'n': n}
        print(f'  {y}: r {r:.3f} -> full-length {sb:.3f} (n {n})')

    print('\n── S: year-to-year r by variant ──')
    hdr = '  variant   24->25        25->26'
    print(hdr)
    for k in variants:
        cells = []
        for y0, y1 in PAIRS:
            common = [h for h in V[y0] if h in V[y1]
                      and k in V[y0][h] and k in V[y1][h]]
            r, n = pearson([V[y0][h][k][0] for h in common],
                           [V[y1][h][k][0] for h in common])
            cells.append((r, n))
        results['yy'][k] = cells
        print(f'  {k:8s}' + ''.join(
            f'  {r:+.3f} ({n:3d})' if r is not None else '     --      '
            for r, n in cells))

    print('\n── V: interaction b3 (d_bs ~ d_sl x BSR), the paper\'s own '
          'validity test ──')
    print('  variant   24->25 b3 (t)      25->26 b3 (t)')
    for k in variants:
        cells = []
        for y0, y1 in PAIRS:
            out = interaction_test(V[y0], V[y1], k)
            cells.append(out)
        results['validity'][k] = cells
        print(f'  {k:8s}' + ''.join(
            f'  {b:+.3f} ({t:+.2f}, n{n})' if c else '      --        '
            for c in cells for b, t, n in ([c] if c else [(0, 0, 0)])))

    print('\n── up vs dn agreement (linearity check) ──')
    for y in SEASONS:
        both = [h for h, r in V[y].items() if 'up' in r and 'dn' in r]
        r, n = pearson([V[y][h]['up'][0] for h in both],
                       [V[y][h]['dn'][0] for h in both])
        mu_up = np.mean([V[y][h]['up'][0] for h in both])
        mu_dn = np.mean([V[y][h]['dn'][0] for h in both])
        results['updn'][y] = {'r': r, 'n': n, 'mean_up': float(mu_up),
                              'mean_dn': float(mu_dn)}
        print(f'  {y}: r(up,dn) {r:+.3f} (n {n}), mean slope up '
              f'{mu_up:.2f} / dn {mu_dn:.2f}')

    out = os.path.join(ROOT, 'data', '_bsr_improve_results.json')
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    os.replace(tmp, out)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
