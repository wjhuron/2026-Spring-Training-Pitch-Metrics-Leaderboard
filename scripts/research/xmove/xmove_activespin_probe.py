"""Is Savant's published Active Spin an independent MEASUREMENT, or is it
back-computed from the observed movement?

This decides whether it can be used in an expected-movement model at all. If
active spin is derived from movement, then conditioning on it is conditioning
on the answer: R^2 would leap, the residual would collapse, and the metric
would be meaningless while looking excellent. That is precisely the failure
mode CLAUDE.md warns about, so it gets tested, not assumed.

The test: try to reconstruct active spin from movement. If Savant computes it
from movement by some fixed formula, then a flexible model given (spin, velo,
IVB, HB, extension) must recover it almost exactly -- R^2 ~ 0.99+ -- because it
IS a deterministic function of those. If instead it carries independent
optical-tracking content, reconstruction plateaus well below that, and the
shortfall is the independent information.

Control: the same reconstruction from RELEASE-ONLY features (spin, velo, slot,
extension, release axis -- no movement). A measurement of the release spin
vector should be better predicted by release variables than a movement-derived
quantity would be.
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
COLMAP = {'fourseam': 'FF', 'sinker': 'SI', 'cutter': 'FC', 'changeup': 'CH',
          'splitter': 'FS', 'curve': 'CU', 'slider': 'SL', 'sweeper': 'ST',
          'slurve': 'SV'}
MIN_N = 100


def load_active(years=(2024, 2025)):
    out = []
    for y in years:
        path = f'{DIR}/as_{y}.csv'
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, encoding='utf-8-sig')
        for col, pt in COLMAP.items():
            c = f'active_spin_{col}'
            if c not in d.columns:
                continue
            sub = d[['entity_name', 'pitch_hand', c]].copy()
            sub.columns = ['pitcher', 'thr', 'active']
            sub['pt'] = pt
            sub['season'] = y
            out.append(sub.dropna(subset=['active']))
    a = pd.concat(out, ignore_index=True)
    a['active'] = pd.to_numeric(a.active, errors='coerce')
    return a.dropna(subset=['active'])


def load_pitches(years=(2024, 2025)):
    d = pd.concat([pd.read_parquet(f'{DIR}/xmove_{y}.parquet') for y in years],
                  ignore_index=True)
    d = d.rename(columns={'Pitch Type': 'pt', 'Spin Rate': 'spin', 'Velocity': 'velo',
                          'xIndVrtBrk': 'ivb', 'xHorzBrk': 'hb', 'Extension': 'ext',
                          'ArmAngle': 'aa', 'SpinAxis': 'axis', 'Throws': 'thr',
                          'Pitcher': 'pitcher', 'RelPosZ': 'rz', 'RelPosX': 'rx'})
    d = d.dropna(subset=['ivb', 'hb', 'spin', 'velo', 'ext', 'aa', 'axis'])
    d = d[d.thr.isin(['L', 'R'])]
    s = np.where(d.thr == 'R', 1.0, -1.0)
    d['hb_s'] = d.hb * s
    d['rx_s'] = d.rx * s
    rt = np.radians(((d.axis - 180.0) % 360.0) * s)
    d['ax_sin'], d['ax_cos'] = np.sin(rt), np.cos(rt)
    d['mag'] = np.hypot(d.ivb, d.hb)
    g = d.groupby(['season', 'pitcher', 'thr', 'pt']).agg(
        n=('ivb', 'size'), ivb=('ivb', 'mean'), hb_s=('hb_s', 'mean'),
        mag=('mag', 'mean'), spin=('spin', 'mean'), velo=('velo', 'mean'),
        ext=('ext', 'mean'), aa=('aa', 'mean'), rz=('rz', 'mean'),
        rx_s=('rx_s', 'mean'), ax_sin=('ax_sin', 'mean'),
        ax_cos=('ax_cos', 'mean'),
        # within-season dispersion: if Savant computes active spin PER PITCH
        # from movement and averages, the aggregate depends on these too, so
        # withholding them would fake independence
        sd_ivb=('ivb', 'std'), sd_hb=('hb_s', 'std'), sd_spin=('spin', 'std'),
        sd_velo=('velo', 'std'), sd_mag=('mag', 'std')).reset_index()
    g = g[g.n >= MIN_N].copy()
    # explicit movement-based efficiency estimate: Magnus break ~ w_T / v, so
    # transverse spin ~ |break| * v, and efficiency ~ |break| * v / w_total
    g['eff_est'] = g.mag * g.velo / g.spin
    return g


def cv_r2(X, y, seed=0, folds=5):
    """Out-of-fold R^2 from gradient boosting -- flexible enough to recover any
    smooth deterministic formula, so a plateau below ~0.99 is evidence the
    target is not a function of these inputs."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(y))
    pred = np.empty(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          min_samples_leaf=15, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


if __name__ == '__main__':
    act = load_active()
    pit = load_pitches()
    df = pit.merge(act, on=['season', 'pitcher', 'thr', 'pt'], how='inner')
    print(f'{len(df)} pitcher-season-pitchtype units matched to Savant active spin')
    print(f'active spin: mean {df.active.mean():.1f}%  sd {df.active.std():.1f}  '
          f'range {df.active.min():.0f}-{df.active.max():.0f}\n')

    MOVE = ['spin', 'velo', 'ivb', 'hb_s', 'mag', 'ext', 'eff_est']
    MOVE_D = MOVE + ['sd_ivb', 'sd_hb', 'sd_spin', 'sd_velo', 'sd_mag', 'n']
    REL = ['spin', 'velo', 'ext', 'aa', 'rz', 'rx_s', 'ax_sin', 'ax_cos']
    y = df.active.values

    print(f"{'pitch type':>11} {'units':>6} {'R2 move':>8} {'R2 move+disp':>13} "
          f"{'R2 release':>11}")
    print('-' * 54)
    for pt in ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']:
        g = df[df.pt == pt]
        if len(g) < 120:
            continue
        print(f'{pt:>11} {len(g):>6} {cv_r2(g[MOVE].values, g.active.values):>8.3f} '
              f'{cv_r2(g[MOVE_D].values, g.active.values):>13.3f} '
              f'{cv_r2(g[REL].values, g.active.values):>11.3f}')
    print(f"{'ALL pooled':>11} {len(df):>6} {cv_r2(df[MOVE].values, y):>8.3f} "
          f"{cv_r2(df[MOVE_D].values, y):>13.3f} {cv_r2(df[REL].values, y):>11.3f}")

    # persistence: a leak-free way to use active spin is the PRIOR season's
    # value, which requires it to be a stable pitcher trait
    print('\nYear-over-year persistence of active spin (2024 -> 2025):')
    a24 = df[df.season == 2024][['pitcher', 'thr', 'pt', 'active']]
    a25 = df[df.season == 2025][['pitcher', 'thr', 'pt', 'active']]
    mg = a24.merge(a25, on=['pitcher', 'thr', 'pt'], suffixes=('_24', '_25'))
    for pt in ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH']:
        g = mg[mg.pt == pt]
        if len(g) < 40:
            continue
        r = np.corrcoef(g.active_24, g.active_25)[0, 1]
        print(f'  {pt:>3}  n={len(g):>4}  r={r:.3f}  '
              f'mean |change| {np.abs(g.active_25 - g.active_24).mean():.1f} pts')
    print(f'  ALL  n={len(mg):>4}  r={np.corrcoef(mg.active_24, mg.active_25)[0,1]:.3f}')
    print('\nR^2 ~0.99 from movement => active spin IS a function of movement '
          '(circular; unusable as a predictor of movement).'
          '\nA clear plateau below that => independent tracking content.')
