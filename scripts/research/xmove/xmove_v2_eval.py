"""End-to-end evaluation of the rebuilt expected-movement model (items 1+2+3).

Ladder, all scored out of sample on 2026 by game-date parity so the scoring
model never sees the pitch's own day:

  A  shipped        arm angle, extension, velocity            (linear, per group)
  B  + release axis and spin                                  (linear)
  C  as B but gradient boosted                                (the nonlinear gap)
  D  + prior-season active spin                               (item 1)
  E  + venue offset removed from the target                   (item 3)

Reported per target in the release-axis frame, because that is where the two
quantities mean different things: ALONG is the Magnus-magnitude miss (which
spin efficiency should explain) and CROSS is the seam deflection (which it
should not, since gyro spin scales break without rotating it). A model that
lifts both equally would be a warning, not a win.

DIST is carried as the redundancy guard: |corr(OE, raw movement)| at the
rendered unit. It has a floor -- seam effect genuinely correlates with movement
-- so lower is not automatically better, but a jump back toward the shipped
model's level would mean the residual has collapsed into a copy of the column
beside it.
"""
import os, sys, json, math, pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_gray_plot import tilt_to_axis, sf, PITCH_COLORS

MIN_GROUP = 400
UNIT_MIN = 50
PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']


def load():
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        raw = pickle.load(f)
    with open(os.path.join(ROOT, 'data', 'active_spin_prior.json')) as f:
        asp = json.load(f)
    with open(os.path.join(ROOT, 'data', 'venue_offsets.json')) as f:
        vof = json.load(f)['venues']
    with open(os.path.join(ROOT, 'data', 'game_weather_rs.json')) as f:
        wx = json.load(f)
    ent, lg = asp['entries'], asp['leagueMeanByPitchType']

    rows = []
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        if pt not in PITCH_TYPES or thr not in ('L', 'R'):
            continue
        ivb, hb = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        velo, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        ext, aa = sf(p.get('Extension')), sf(p.get('ArmAngle'))
        rz, rx = sf(p.get('RelPosZ')), sf(p.get('RelPosX'))
        axis = tilt_to_axis(p.get('RTilt'))
        pid = p.get('PitchID')
        if None in (ivb, hb, velo, spin, ext, aa, axis, rz, rx) or not pid:
            continue
        pitcher = p.get('Pitcher')
        rec = ent.get(f'{pitcher}|{thr}|{pt}')
        gp = str(pid).split('_')[0]
        rows.append((pitcher, thr, pt, ivb, hb, velo, spin, ext, aa, rz, rx, axis,
                     p.get('Game Date'), gp,
                     rec['active'] if rec else lg.get(pt, 70.0),
                     1.0 if rec else 0.0,
                     (wx.get(gp) or {}).get('venueId')))
    d = pd.DataFrame(rows, columns=['pitcher', 'thr', 'pt', 'ivb', 'hb', 'velo',
                                    'spin', 'ext', 'aa', 'rz', 'rx', 'axis',
                                    'date', 'gp', 'active', 'has_active', 'venue'])
    s = np.where(d.thr == 'R', 1.0, -1.0)
    d['hb_s'] = d.hb * s
    d['rx_s'] = d.rx * s
    th = np.radians(((d.axis - 180.0) % 360.0) * s)
    d['ct'], d['st'] = np.cos(th), np.sin(th)
    d['along'] = d.ivb * d.ct + d.hb_s * d.st
    d['cross'] = -d.ivb * d.st + d.hb_s * d.ct
    d['ax_sin'], d['ax_cos'] = np.sin(th), np.cos(th)
    d['ax_sin2'], d['ax_cos2'] = np.sin(2 * th), np.cos(2 * th)
    sv = d.spin / d.velo
    d['sv_sin'], d['sv_cos'] = sv * d.ax_sin, sv * d.ax_cos
    d['aa_sin'], d['aa_cos'] = d.aa * d.ax_sin, d.aa * d.ax_cos
    d['aa2'] = d.aa ** 2
    d['spin_t'] = d.spin * d.active / 100.0
    svt = d.spin_t / d.velo
    d['svt_sin'], d['svt_cos'] = svt * d.ax_sin, svt * d.ax_cos
    d['par'] = pd.to_datetime(d.date).dt.dayofyear % 2
    # item 3: venue offset in the release-axis frame, hand-mirrored already
    d['v_along'] = [vof.get(str(v), {}).get('along', 0.0) if v else 0.0 for v in d.venue]
    d['v_cross'] = [vof.get(str(v), {}).get('cross', 0.0) if v else 0.0 for v in d.venue]
    return d


A_F = ['aa', 'ext', 'velo']
B_F = A_F + ['spin', 'ax_sin', 'ax_cos', 'ax_sin2', 'ax_cos2', 'sv_sin', 'sv_cos',
             'aa_sin', 'aa_cos', 'aa2']
GBM_F = ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'rz', 'rx_s']
GBM_D = GBM_F + ['active']


def crossfit(d, feats, targets, gbm=False, devenue=False):
    out = {t: np.full(len(d), np.nan) for t in targets}
    for (pt, thr), g in d.groupby(['pt', 'thr']):
        if len(g) < MIN_GROUP:
            continue
        pos = d.index.get_indexer(g.index)
        X = g[feats].values
        if not gbm:
            X = np.column_stack([np.ones(len(g)), X])
        for p in (0, 1):
            tr = (g.par.values == p)
            te = ~tr
            if tr.sum() < max(MIN_GROUP // 2, 25 * X.shape[1]) or te.sum() == 0:
                continue
            for t in targets:
                y = g[t].values.copy()
                if devenue:
                    y = y - g['v_' + t].values
                if gbm:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    m = HistGradientBoostingRegressor(
                        max_iter=250, learning_rate=0.07, min_samples_leaf=60,
                        l2_regularization=1.0, random_state=0)
                    m.fit(X[tr], y[tr])
                    pred = m.predict(X[te])
                else:
                    b = np.linalg.lstsq(X[tr], y[tr], rcond=1e-8)[0]
                    pred = X[te] @ b
                if devenue:
                    pred = pred + g['v_' + t].values[te]
                out[t][pos[te]] = pred
    return out


def score(d, preds, label, rows):
    res = {}
    for t in ('along', 'cross'):
        p = preds[t]
        ok = np.isfinite(p)
        num = den = 0.0
        for (pt, thr), g in d.groupby(['pt', 'thr']):
            pos = d.index.get_indexer(g.index)
            k = ok[pos]
            if k.sum() < 50:
                continue
            y = g[t].values[k]
            num += ((y - p[pos][k]) ** 2).sum()
            den += ((y - y.mean()) ** 2).sum()
        res[t] = 1 - num / den
    # redundancy guard at the rendered unit
    oe_a = d.along.values - preds['along']
    oe_c = d.cross.values - preds['cross']
    u = pd.DataFrame({'k': d.pitcher + '|' + d.thr + '|' + d.pt, 'pt': d.pt,
                      'a': d.along.values, 'c': d.cross.values,
                      'oa': oe_a, 'oc': oe_c}).dropna()
    ug = u.groupby(['k', 'pt']).agg(n=('a', 'size'), a=('a', 'mean'), c=('c', 'mean'),
                                    oa=('oa', 'mean'), oc=('oc', 'mean')).reset_index()
    ug = ug[ug.n >= UNIT_MIN]
    da, dc = [], []
    for pt, g in ug.groupby('pt'):
        if len(g) >= 60:
            da.append(abs(np.corrcoef(g.oa, g.a)[0, 1]))
            dc.append(abs(np.corrcoef(g.oc, g.c)[0, 1]))
    rows.append((label, res['along'], res['cross'], np.mean(da), np.mean(dc)))


if __name__ == '__main__':
    d = load().reset_index(drop=True)
    print(f'{len(d):,} 2026 MLB pitches; '
          f'{d.has_active.mean()*100:.1f}% carry a measured prior-season '
          f'active spin (rest use the pitch-type league mean)\n')
    rows = []
    score(d, crossfit(d, A_F, ('along', 'cross')), 'A shipped (aa,ext,velo)', rows)
    score(d, crossfit(d, B_F, ('along', 'cross')), 'B + axis + spin (linear)', rows)
    score(d, crossfit(d, GBM_F, ('along', 'cross'), gbm=True), 'C  as B, gradient boosted', rows)
    score(d, crossfit(d, GBM_D, ('along', 'cross'), gbm=True), 'D + prior active spin', rows)
    score(d, crossfit(d, GBM_D, ('along', 'cross'), gbm=True, devenue=True),
          'E + venue offset', rows)
    print(f"{'model':<28} {'R2 along':>9} {'R2 cross':>9} {'DIST a':>8} {'DIST c':>8}")
    print('-' * 66)
    for r in rows:
        print(f'{r[0]:<28} {r[1]:>9.3f} {r[2]:>9.3f} {r[3]:>8.3f} {r[4]:>8.3f}')
    print('\nALONG is the Magnus-magnitude miss (efficiency should explain it); '
          'CROSS is the\nseam deflection (efficiency should NOT, since gyro spin '
          'cannot rotate break).')
