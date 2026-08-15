"""Release-axis-frame ("polar") expected movement, vs the shipped model.

The idea Wally described: RTilt is the measured RELEASE spin axis, so it is a
legitimate upstream predictor; OTilt is atan2(HB, IVB) so it is the answer, not
a predictor. The physics: a purely-Magnus ball breaks ALONG the release axis
direction. Everything perpendicular to that is seam-shifted wake (plus any
gyro-mismeasurement).

So rotate the break vector into the release-axis frame:
    u      = (cos rtilt, sin rtilt)     unit vector in (IVB, HB_armside) space
    along  = break . u          Magnus-direction break, inches
    cross  = break . u_perp     non-Magnus / SSW deflection, inches

Both are linear in (IVB, HB), so nothing blows up on gyro sliders the way a
tilt-in-degrees residual does (a 2-inch slider's observed tilt is noise).

Model E[along] and E[cross] per (pitch type, hand) from spin, velo, extension,
arm angle. Rotate back to get xIVB/xHB. This makes the FF/SI distinction Wally
wants: same RTilt, but E[cross] differs by pitch class, so the two ellipses
land in different places -- and the residual is the pitcher's OWN seam skill on
top of his pitch class's normal seam effect.
"""
import os, math, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_audit import load, add_axis_trig, FEATURE_SETS, PITCH_TYPES, MIN_N
from xmove_residual_tests import unit_agg, sb, corr, UNIT_MIN

BASE_FEATS = ['spin', 'velo', 'ext', 'aa', 'spin_v']


def prep(df):
    df = df.copy()
    df['spin_v'] = df.spin / df.velo
    th = np.radians(df.rtilt * df.s)            # hand-mirrored release axis
    df['th'] = th
    ct, st = np.cos(th), np.sin(th)
    df['along'] = df.ivb * ct + df.hb_s * st
    df['cross'] = -df.ivb * st + df.hb_s * ct   # positive = arm-side of the axis
    df['ct'], df['st'] = ct, st
    return df


def fit_polar(train, feats=BASE_FEATS):
    models = {}
    for (pt, thr), g in train.groupby(['pt', 'thr']):
        if len(g) < MIN_N:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        m = {}
        for t in ('along', 'cross'):
            beta, *_ = np.linalg.lstsq(X, g[t].values, rcond=None)
            m[t] = beta
        models[(pt, thr)] = m
    return models


def score_polar(test, models, feats=BASE_FEATS):
    out = test.copy()
    out['a_hat'] = np.nan
    out['c_hat'] = np.nan
    for (pt, thr), g in out.groupby(['pt', 'thr']):
        m = models.get((pt, thr))
        if m is None:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        out.loc[g.index, 'a_hat'] = X @ m['along']
        out.loc[g.index, 'c_hat'] = X @ m['cross']
    out = out.dropna(subset=['a_hat', 'c_hat'])
    # rotate the predicted (along, cross) back into (IVB, HB_armside)
    out['x_ivb'] = out.a_hat * out.ct - out.c_hat * out.st
    out['x_hb'] = out.a_hat * out.st + out.c_hat * out.ct
    out['ivb_oe'] = out.ivb - out.x_ivb
    out['hb_oe'] = out.hb_s - out.x_hb
    out['along_oe'] = out.along - out.a_hat
    out['cross_oe'] = out.cross - out.c_hat
    return out


def pitch_r2(scored):
    ni = nh = di = dh = 0.0
    for (pt, thr), g in scored.groupby(['pt', 'thr']):
        ni += ((g.ivb - g.x_ivb) ** 2).sum(); di += ((g.ivb - g.ivb.mean()) ** 2).sum()
        nh += ((g.hb_s - g.x_hb) ** 2).sum(); dh += ((g.hb_s - g.hb_s.mean()) ** 2).sum()
    return 1 - ni / di, 1 - nh / dh


def report(name, u, u_prev):
    print(f'\n--- {name} ---')
    print(f"{'pt':>4} {'units':>6} {'r(ivbOE,IVB)':>13} {'r(hbOE,HB)':>11} "
          f"{'rel ivbOE':>10} {'rel hbOE':>9} {'YoY ivbOE':>10} {'YoY hbOE':>9}")
    for pt in PITCH_TYPES:
        g = u[(u.pt == pt) & (u.n >= UNIT_MIN)]
        if len(g) < 30:
            continue
        gh = g[(g.n_0 >= UNIT_MIN / 2) & (g.n_1 >= UNIT_MIN / 2)]
        p = u_prev[(u_prev.pt == pt) & (u_prev.n >= UNIT_MIN)]
        mg = g.merge(p, on=['pitcher', 'thr', 'pt'], suffixes=('', '_p'))
        print(f'{pt:>4} {len(g):>6} {corr(g.ivb_oe.values, g.ivb.values):>13.3f} '
              f'{corr(g.hb_oe.values, g.hb.values):>11.3f} '
              f'{sb(corr(gh.ivb_oe_0.values, gh.ivb_oe_1.values)):>10.3f} '
              f'{sb(corr(gh.hb_oe_0.values, gh.hb_oe_1.values)):>9.3f} '
              f'{corr(mg.ivb_oe.values, mg.ivb_oe_p.values):>10.3f} '
              f'{corr(mg.hb_oe.values, mg.hb_oe_p.values):>9.3f}')


if __name__ == '__main__':
    df = prep(add_axis_trig(load()))
    train = df[df.season <= 2023]
    t25, t24 = df[df.season == 2025], df[df.season == 2024]
    models = fit_polar(train)
    s25, s24 = score_polar(t25, models), score_polar(t24, models)
    r2i, r2h = pitch_r2(s25)
    print(f'\nS6 polar (release-axis frame), OUT OF SAMPLE 2025:')
    print(f'  pitch-level R^2  IVB {r2i:.3f}   HB {r2h:.3f}')
    print(f'\n  per pitch type, out-of-sample R^2 and the two polar residuals')
    print(f"{'pt':>4} {'R2 ivb':>7} {'R2 hb':>7} {'sd along_oe':>12} {'sd cross_oe':>12} "
          f"{'mean cross':>11}")
    for pt in PITCH_TYPES:
        g = s25[s25.pt == pt]
        if len(g) < 5000:
            continue
        ri = 1 - ((g.ivb - g.x_ivb) ** 2).sum() / ((g.ivb - g.ivb.mean()) ** 2).sum()
        rh = 1 - ((g.hb_s - g.x_hb) ** 2).sum() / ((g.hb_s - g.hb_s.mean()) ** 2).sum()
        print(f'{pt:>4} {ri:>7.3f} {rh:>7.3f} {g.along_oe.std():>12.2f} '
              f'{g.cross_oe.std():>12.2f} {g.cross.mean():>11.2f}')
    report('S6 polar', unit_agg(s25), unit_agg(s24))
