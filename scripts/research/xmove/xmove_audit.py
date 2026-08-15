"""Audit the shipped expected-movement model (xIVB/xHB -> IVBOE/HBOE).

Three questions:
  A. Is SpinAxis a RELEASE measurement (usable for SSW) or just the
     movement-derived axis (useless — it would reproduce OTilt exactly)?
  B. How much of IVB/HB does the shipped regressor set (ArmAngle, Extension,
     Velocity) actually explain, vs sets that add Spin Rate and the release
     axis?
  C. Is the shipped residual distinct from the raw movement it is supposed to
     contextualize, and is it a persistent pitcher skill?

The shipped model is an MVN conditional mean per (pitch type, hand), which is
algebraically identical to per-group OLS of [IVB, HB] on the regressors, so
OLS is used here (same numbers, ~1000x faster).
"""
import os, sys, math
import numpy as np
import pandas as pd

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_N = 150            # matches MVN_MIN_N in process_data.py
PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'CH', 'FS']


def load(seasons=SEASONS):
    df = pd.concat([pd.read_parquet(f'{DIR}/xmove_{y}.parquet') for y in seasons],
                   ignore_index=True)
    df = df.rename(columns={'Pitch Type': 'pt', 'Spin Rate': 'spin', 'Velocity': 'velo',
                            'IndVertBrk': 'ivb_raw', 'HorzBrk': 'hb_raw',
                            'xIndVrtBrk': 'ivb', 'xHorzBrk': 'hb',
                            'Extension': 'ext', 'ArmAngle': 'aa', 'SpinAxis': 'axis',
                            'RelPosZ': 'rz', 'RelPosX': 'rx', 'Throws': 'thr',
                            'Pitcher': 'pitcher', 'Game Date': 'date'})
    df = df[df.pt.isin(PITCH_TYPES) & df.thr.isin(['L', 'R'])]
    need = ['ivb', 'hb', 'velo', 'ext', 'aa', 'axis', 'spin']
    df = df.dropna(subset=need)
    # hand sign: mirror LHP into the RHP frame (arm-side positive)
    df['s'] = np.where(df.thr == 'R', 1.0, -1.0)
    # tilt in "clock degrees" (12:00 = 0, clockwise positive)
    df['rtilt'] = (df.axis - 180.0) % 360.0
    df['otilt'] = np.degrees(np.arctan2(df.hb, df.ivb)) % 360.0
    dev = (df.otilt - df.rtilt + 180.0) % 360.0 - 180.0
    df['dev'] = dev * df.s              # hand-signed: positive = toward arm side
    df['mag'] = np.hypot(df.ivb, df.hb)
    df['hb_s'] = df.hb * df.s           # hand-signed HB
    df['aa_s'] = df.aa                  # arm angle is already hand-agnostic (0=sidearm)
    df['rx_s'] = df.rx * df.s
    return df


def circ_stats(deg):
    r = np.radians(deg)
    m = math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))
    R = math.hypot(np.sin(r).mean(), np.cos(r).mean())
    sd = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12)))))
    return m, sd


def section_a(df):
    print('\n' + '=' * 78)
    print('A. Is SpinAxis a RELEASE measurement?  (OTilt - RTilt, hand-signed deg)')
    print('   If SpinAxis were movement-derived, every row would read 0.0 +/- 0.0')
    print('=' * 78)
    print(f"{'pt':>4} {'n':>9} {'mean dev':>9} {'sd dev':>8} {'p10':>7} {'p90':>7} "
          f"{'|dev| mean':>10}")
    for pt in PITCH_TYPES:
        g = df[df.pt == pt]
        if len(g) < 1000:
            continue
        m, sd = circ_stats(g.dev.values)
        print(f'{pt:>4} {len(g):>9,} {m:>9.2f} {sd:>8.2f} '
              f'{np.percentile(g.dev, 10):>7.1f} {np.percentile(g.dev, 90):>7.1f} '
              f'{g.dev.abs().mean():>10.2f}')


FEATURE_SETS = {
    'S0 group mean':        [],
    'S1 shipped (aa,ext,v)': ['aa', 'ext', 'velo'],
    'S2 +spin':              ['aa', 'ext', 'velo', 'spin'],
    'S3 +spin,axis':         ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos'],
    'S4 spin,axis,v only':   ['velo', 'spin', 'ax_sin', 'ax_cos'],
    'S5 all':                ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'rz', 'rx_s'],
}


def add_axis_trig(df):
    r = np.radians(df.rtilt * df.s)   # hand-mirrored release axis
    df['ax_sin'] = np.sin(r)
    df['ax_cos'] = np.cos(r)
    return df


def fit_group(g, feats, targets=('ivb', 'hb_s')):
    """Per-group OLS. Returns dict target -> (r2, resid array)."""
    n = len(g)
    X = np.column_stack([np.ones(n)] + [g[f].values for f in feats])
    out = {}
    for t in targets:
        y = g[t].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        resid = y - pred
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
        out[t] = (r2, resid, pred)
    return out


def section_b(df):
    print('\n' + '=' * 78)
    print('B. Pitch-level R^2 by regressor set (per pitch type x hand, pooled 2021-25)')
    print('=' * 78)
    rows = []
    for name, feats in FEATURE_SETS.items():
        num_i = den_i = num_h = den_h = 0.0
        for (pt, thr), g in df.groupby(['pt', 'thr']):
            if len(g) < MIN_N:
                continue
            res = fit_group(g, feats)
            for t, (num, den) in (('ivb', (0, 0)), ):
                pass
            ri, resid_i, _ = res['ivb']
            rh, resid_h, _ = res['hb_s']
            vi = ((g.ivb.values - g.ivb.values.mean()) ** 2).sum()
            vh = ((g.hb_s.values - g.hb_s.values.mean()) ** 2).sum()
            num_i += (resid_i ** 2).sum(); den_i += vi
            num_h += (resid_h ** 2).sum(); den_h += vh
        rows.append((name, 1 - num_i / den_i, 1 - num_h / den_h,
                     math.sqrt(num_i / len(df)), math.sqrt(num_h / len(df))))
    print(f"{'set':>24} {'R2 IVB':>8} {'R2 HB':>8} {'RMSE IVB':>9} {'RMSE HB':>8}")
    for name, ri, rh, ei, eh in rows:
        print(f'{name:>24} {ri:>8.3f} {rh:>8.3f} {ei:>9.2f} {eh:>8.2f}')

    print('\n  per pitch type, R^2 for IVB / HB  (shipped S1 -> S3 with spin+axis)')
    print(f"{'pt':>4} {'n':>9} {'S1 ivb':>7} {'S3 ivb':>7} {'S1 hb':>7} {'S3 hb':>7}")
    for pt in PITCH_TYPES:
        g0 = df[df.pt == pt]
        if len(g0) < MIN_N * 2:
            continue
        acc = {}
        for name in ('S1 shipped (aa,ext,v)', 'S3 +spin,axis'):
            ni = di = nh = dh = 0.0
            for thr, g in g0.groupby('thr'):
                if len(g) < MIN_N:
                    continue
                res = fit_group(g, FEATURE_SETS[name])
                ni += (res['ivb'][1] ** 2).sum()
                nh += (res['hb_s'][1] ** 2).sum()
                di += ((g.ivb.values - g.ivb.values.mean()) ** 2).sum()
                dh += ((g.hb_s.values - g.hb_s.values.mean()) ** 2).sum()
            acc[name] = (1 - ni / di, 1 - nh / dh)
        s1, s3 = acc['S1 shipped (aa,ext,v)'], acc['S3 +spin,axis']
        print(f'{pt:>4} {len(g0):>9,} {s1[0]:>7.3f} {s3[0]:>7.3f} {s1[1]:>7.3f} {s3[1]:>7.3f}')


if __name__ == '__main__':
    df = add_axis_trig(load())
    print(f'loaded {len(df):,} pitches, {df.season.nunique()} seasons')
    section_a(df)
    section_b(df)
