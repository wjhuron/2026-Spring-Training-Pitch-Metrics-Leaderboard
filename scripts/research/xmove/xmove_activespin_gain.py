"""Does PRIOR-season active spin actually improve expected movement?

Why prior season and not current: even if Savant's active spin turns out to be
partly movement-derived (the probe rules out reconstruction from season-
aggregate movement, but cannot rule out a per-pitch formula over the full
trajectory), last season's value cannot leak this season's residual. It carries
only the pitcher's stable efficiency trait -- which is exactly what we want to
condition on. Persistence is r = 0.79-0.92 by pitch type, so little is lost.

Fit on 2025 pitches for pitchers who have a 2024 active-spin value, in the
release-axis frame, cross-fit by game parity. Two targets:

  ALONG  Magnus-magnitude break along the measured release axis. This is the
         component efficiency should explain. If active spin helps anywhere,
         here is where.
  CROSS  break perpendicular to the axis -- seam-shifted wake. Gyro spin
         scales magnitude without rotating it, so active spin should add
         LITTLE here. If it adds a lot, something is wrong with the framing.

That asymmetry is the real test, not the headline R^2.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_activespin_probe import load_active, COLMAP

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
MIN_GROUP = 400
PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']


def load_2025_with_prior():
    d = pd.read_parquet(f'{DIR}/xmove_2025.parquet')
    d = d.rename(columns={'Pitch Type': 'pt', 'Spin Rate': 'spin', 'Velocity': 'velo',
                          'xIndVrtBrk': 'ivb', 'xHorzBrk': 'hb', 'Extension': 'ext',
                          'ArmAngle': 'aa', 'SpinAxis': 'axis', 'Throws': 'thr',
                          'Pitcher': 'pitcher', 'RelPosZ': 'rz', 'RelPosX': 'rx',
                          '_game_pk': 'game'})
    d = d.dropna(subset=['ivb', 'hb', 'spin', 'velo', 'ext', 'aa', 'axis', 'game'])
    d = d[d.thr.isin(['L', 'R']) & d.pt.isin(PITCH_TYPES)]
    prior = load_active(years=(2024,))[['pitcher', 'thr', 'pt', 'active']]
    prior = prior.rename(columns={'active': 'active_prior'})
    d = d.merge(prior, on=['pitcher', 'thr', 'pt'], how='inner')
    s = np.where(d.thr == 'R', 1.0, -1.0)
    d['hb_s'] = d.hb * s
    d['rx_s'] = d.rx * s
    th = np.radians(((d.axis - 180.0) % 360.0) * s)
    d['ct'], d['st'] = np.cos(th), np.sin(th)
    d['along'] = d.ivb * d.ct + d.hb_s * d.st
    d['cross'] = -d.ivb * d.st + d.hb_s * d.ct
    d['sv'] = d.spin / d.velo
    d['ax_sin'], d['ax_cos'] = np.sin(th), np.cos(th)
    d['ax_sin2'], d['ax_cos2'] = np.sin(2 * th), np.cos(2 * th)
    d['sv_sin'], d['sv_cos'] = d.sv * d.ax_sin, d.sv * d.ax_cos
    d['aa_sin'], d['aa_cos'] = d.aa * d.ax_sin, d.aa * d.ax_cos
    d['aa2'] = d.aa ** 2
    # transverse spin implied by last season's efficiency -- the physically
    # meaningful form, since Magnus force scales with TRANSVERSE spin, not total
    d['spin_t'] = d.spin * d.active_prior / 100.0
    d['svt'] = d.spin_t / d.velo
    d['svt_sin'], d['svt_cos'] = d.svt * d.ax_sin, d.svt * d.ax_cos
    return d


BASE = ['ext', 'aa', 'aa2', 'spin', 'velo', 'ax_sin', 'ax_cos', 'ax_sin2',
        'ax_cos2', 'sv_sin', 'sv_cos', 'aa_sin', 'aa_cos']
WITH = BASE + ['active_prior', 'spin_t', 'svt_sin', 'svt_cos']


def crossfit_r2(d, feats, target):
    num = den = 0.0
    per_pt = {}
    for (pt, thr), g in d.groupby(['pt', 'thr']):
        if len(g) < MIN_GROUP:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        y = g[target].values
        par = (g.game.values % 2)
        pred = np.full(len(g), np.nan)
        for p in (0, 1):
            tr, te = par == p, par == 1 - p
            if tr.sum() < max(MIN_GROUP // 2, 20 * X.shape[1]) or te.sum() == 0:
                continue
            b = np.linalg.lstsq(X[tr], y[tr], rcond=1e-8)[0]
            pred[te] = X[te] @ b
        ok = np.isfinite(pred)
        if ok.sum() == 0:
            continue
        a = ((y[ok] - pred[ok]) ** 2).sum()
        b_ = ((y[ok] - y[ok].mean()) ** 2).sum()
        num += a; den += b_
        acc = per_pt.setdefault(pt, [0.0, 0.0])
        acc[0] += a; acc[1] += b_
    return 1 - num / den, {k: 1 - v[0] / v[1] for k, v in per_pt.items()}


if __name__ == '__main__':
    d = load_2025_with_prior()
    print(f'{len(d):,} 2025 pitches from pitchers with a 2024 active-spin value '
          f'({d.groupby(["pitcher","thr","pt"]).ngroups} pitcher x pitch-type units)\n')
    res = {}
    for target in ('along', 'cross'):
        for tag, feats in (('base', BASE), ('with active spin', WITH)):
            res[(target, tag)] = crossfit_r2(d, feats, target)
    print(f"{'target':>7} {'base R2':>9} {'+active spin':>13} {'gain':>7}")
    print('-' * 40)
    for target in ('along', 'cross'):
        b = res[(target, 'base')][0]
        w = res[(target, 'with active spin')][0]
        print(f'{target:>7} {b:>9.3f} {w:>13.3f} {w-b:>7.3f}')
    print(f"\n{'pt':>4} {'along base':>11} {'along +AS':>10} {'gain':>7} | "
          f"{'cross base':>11} {'cross +AS':>10} {'gain':>7}")
    for pt in PITCH_TYPES:
        ab = res[('along', 'base')][1].get(pt)
        aw = res[('along', 'with active spin')][1].get(pt)
        cb = res[('cross', 'base')][1].get(pt)
        cw = res[('cross', 'with active spin')][1].get(pt)
        if ab is None or cb is None:
            continue
        print(f'{pt:>4} {ab:>11.3f} {aw:>10.3f} {aw-ab:>7.3f} | '
              f'{cb:>11.3f} {cw:>10.3f} {cw-cb:>7.3f}')
    print('\nExpected if the framing is right: a large gain on ALONG (efficiency '
          'is\nexactly the missing magnitude term) and a small one on CROSS (gyro '
          'cannot\nrotate the break).')
