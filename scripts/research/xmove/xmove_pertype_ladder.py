"""Per-type expected movement: what else closes the gap to the nonlinear ceiling?

Review Finding 4/5 (docs/expected_movement_review.md) settled: spin rate and the
release axis are the big inputs; two axis harmonics plus a spin x axis tensor
bracket the interior optimum; release point adds 0.006; the GBM ceiling sits
0.13 R^2 above the best linear form. This ladder asks what remains, per
(pitch type, hand), fit and scored the way production would:

  B0  current      arm angle, extension, velocity
  B1  proposed     + spin rate, 2 axis harmonics, spin x axis        (Finding 5)
  B2  + velo x axis
  B3  + arm angle x axis, extension x axis
  B4  + velo^2, spin^2, spin_v (spin factor proxy) and spin_v x axis
  B5  B2 + B3 + B4 together
  B6  B5 + release point (rz, rx)
  E1  B1 + flight-inferred spin efficiency (KinEff, + eff x axis)  FLAGGED:
      inferred from the lift magnitude, so it is partly the answer. Reported
      in the ALONG/CROSS split, where CROSS is the honest column.
  G   HistGradientBoosting on B1's raw inputs (ceiling; ships only per pitch)

Every form is cross-fit by game parity WITHIN each (type, hand, season), so a
pitch is never scored by a model that saw its game, and every form blanks the
same groups (training floor from the widest design). Objective is out-of-sample
pitch-level R^2 within type, IVB and HB, and in the release-axis frame ALONG and
CROSS. DIST (|corr(OE, raw)| at the rendered unit) is the guard. Per-season
columns are the replicate test: a term must win in most seasons it was never
fit on, which here means every season, since each is fit self-contained.

Usage: python3 scripts/research/xmove/xmove_pertype_ladder.py [--no-gbm] [--seasons 2023 2024 2025]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xmove_agnostic_flight import build_cache, SEASONS as ALL_SEASONS, PTS, UNIT_MIN  # noqa: E402

MIN_N = 400          # per (type, hand, season, fold) training floor, widest design
UNIT_MIN_N = UNIT_MIN


def to_arrays(d):
    d = d[d['Pitch Type'].isin(PTS) & d['Throws'].isin(['L', 'R'])]
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension', 'ArmAngle',
                         'SpinAxis', 'Spin Rate', 'KinEff', 'RelPosZ', 'RelPosX'])
    s = np.where(d['Throws'].values == 'R', 1.0, -1.0)
    ivb = d['xIndVrtBrk'].values.astype('f8')
    hb_s = d['xHorzBrk'].values.astype('f8') * s
    th = np.radians(((d['SpinAxis'].values.astype('f8') - 180.0) % 360.0) * s)
    velo = d['Velocity'].values.astype('f8')
    spin = d['Spin Rate'].values.astype('f8')
    A = dict(pt=d['Pitch Type'].values, thr=d['Throws'].values, pitcher=d['Pitcher'].values,
             season=d['season'].values.astype('i4'), game=d['_game_pk'].values.astype('i8'),
             ivb=ivb, hb_s=hb_s, velo=velo, spin=spin, spin_v=spin / velo,
             velo2=(velo - 88.0) ** 2 / 100.0, spin2=(spin - 2300.0) ** 2 / 1e5,
             ext=d['Extension'].values.astype('f8'), aa=d['ArmAngle'].values.astype('f8'),
             eff=d['KinEff'].values.astype('f8'),
             rz=d['RelPosZ'].values.astype('f8'), rx_s=d['RelPosX'].values.astype('f8') * s,
             ct=np.cos(th), st=np.sin(th), th=th)
    A['along'] = ivb * A['ct'] + hb_s * A['st']
    A['cross'] = -ivb * A['st'] + hb_s * A['ct']
    for k in (1, 2):
        hs, hc = np.sin(k * th), np.cos(k * th)
        A[f'h{k}s'], A[f'h{k}c'] = hs, hc
        for nm, v in (('sp', spin / 1000.0), ('vl', velo / 10.0), ('aa', A['aa'] / 10.0),
                      ('ex', A['ext']), ('sv', A['spin_v']), ('ef', A['eff'])):
            A[f'{nm}{k}s'], A[f'{nm}{k}c'] = v * hs, v * hc
    A['gid'] = pd.factorize(pd.Series(A['pt']) + '_' + pd.Series(A['thr']) + '_' +
                            pd.Series(A['season']).astype(str))[0]
    return A


def harm(prefixes):
    return [f'{p}{k}{c}' for p in prefixes for k in (1, 2) for c in ('s', 'c')]


B0 = ['aa', 'ext', 'velo']
B1 = B0 + ['spin'] + harm(['h', 'sp'])
B2 = B1 + harm(['vl'])
B3 = B1 + harm(['aa', 'ex'])
B4 = B1 + ['velo2', 'spin2', 'spin_v'] + harm(['sv'])
B5 = B1 + harm(['vl', 'aa', 'ex']) + ['velo2', 'spin2', 'spin_v'] + harm(['sv'])
B6 = B5 + ['rz', 'rx_s']
E1 = B1 + ['eff'] + harm(['ef'])
FORMS = [('B0 current', B0), ('B1 proposed', B1), ('B2 +velo x ax', B2),
         ('B3 +aa,ext x ax', B3), ('B4 +sq, spin_v', B4), ('B5 all', B5),
         ('B6 B5+relpt', B6), ('E1 B1+eff FLAG', E1)]
GBM_RAW = ['aa', 'ext', 'velo', 'spin', 'ct', 'st']


def design(A, feats, idx):
    return np.column_stack([np.ones(len(idx))] + [A[f][idx] for f in feats])


def fit_cv(A, feats, gbm=False):
    n = len(A['ivb'])
    xi, xh = np.full(n, np.nan), np.full(n, np.nan)
    gid = A['gid']
    order = np.argsort(gid, kind='stable')
    bounds = np.searchsorted(gid[order], np.arange(gid.max() + 2))
    if gbm:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    for gi in range(gid.max() + 1):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) == 0:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N or len(te) == 0:
                continue
            if gbm:
                Xt = np.column_stack([A[f][tr] for f in GBM_RAW])
                Xs = np.column_stack([A[f][te] for f in GBM_RAW])
                for tgt, out in (('ivb', xi), ('hb_s', xh)):
                    m = HGB(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                            min_samples_leaf=40, random_state=0)
                    m.fit(Xt, A[tgt][tr])
                    out[te] = m.predict(Xs)
            else:
                Xt, Xs = design(A, feats, tr), design(A, feats, te)
                xi[te] = Xs @ np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
                xh[te] = Xs @ np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]
    return xi, xh


def r2(y, yhat, m):
    yy, hh = y[m], yhat[m]
    return 1 - ((yy - hh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def within_type(A, xi, xh, m0):
    """Mean over types of R^2 (IVB, HB, ALONG, CROSS) on mask m0."""
    xa = xi * A['ct'] + xh * A['st']
    xc = -xi * A['st'] + xh * A['ct']
    out = []
    for pt in PTS:
        m = m0 & (A['pt'] == pt)
        if m.sum() < 2000:
            continue
        out.append([r2(A['ivb'], xi, m), r2(A['hb_s'], xh, m),
                    r2(A['along'], xa, m), r2(A['cross'], xc, m)])
    return np.mean(out, axis=0)


def dist(A, xi, xh, m0):
    d = pd.DataFrame({'pitcher': A['pitcher'][m0], 'pt': A['pt'][m0], 'thr': A['thr'][m0],
                      'season': A['season'][m0], 'ivb': A['ivb'][m0], 'hb': A['hb_s'][m0],
                      'oi': A['ivb'][m0] - xi[m0], 'oh': A['hb_s'][m0] - xh[m0]})
    u = d.groupby(['pitcher', 'pt', 'thr', 'season']).agg(
        n=('ivb', 'size'), ivb=('ivb', 'mean'), hb=('hb', 'mean'),
        oi=('oi', 'mean'), oh=('oh', 'mean')).reset_index()
    u = u[u.n >= UNIT_MIN_N]
    di, dh = [], []
    for pt in PTS:
        b = u[u.pt == pt]
        if len(b) < 100:
            continue
        di.append(abs(np.corrcoef(b.oi, b.ivb)[0, 1]))
        dh.append(abs(np.corrcoef(b.oh, b.hb)[0, 1]))
    return np.mean(di), np.mean(dh)


def main():
    seasons = ALL_SEASONS
    if '--seasons' in sys.argv:
        i = sys.argv.index('--seasons') + 1
        seasons = [int(a) for a in sys.argv[i:] if a.isdigit()]
    frames = [build_cache(y) for y in seasons]
    A = to_arrays(pd.concat(frames, ignore_index=True))
    del frames
    print(f'{len(A["ivb"]):,} pitches, seasons {seasons}, per (type, hand, season), '
          f'cross-fit by game parity\n')

    # constant coverage: blank the same groups for every form (widest design floor)
    forms = list(FORMS)
    if '--no-gbm' not in sys.argv:
        forms.append(('G  gbm ceiling', None))
    results = {}
    for name, feats in forms:
        t0 = time.time()
        xi, xh = fit_cv(A, feats, gbm=feats is None)
        results[name] = (xi, xh)
        print(f'  {name:<18} fitted in {time.time() - t0:5.0f}s', file=sys.stderr)
    cov = np.all([np.isfinite(results[n][0]) for n, _ in forms], axis=0)
    print(f'coverage held constant at {cov.mean():.3f} across every form\n')

    # pooled table
    print(f'{"form":<18}{"k":>4}{"R2 IVB":>8}{"R2 HB":>7}{"ALONG":>7}{"CROSS":>7}{"DIST i":>8}{"DIST h":>8}')
    print('-' * 67)
    for name, feats in forms:
        xi, xh = results[name]
        r = within_type(A, xi, xh, cov)
        di, dh = dist(A, xi, xh, cov)
        k = len(feats) + 1 if feats else 0
        print(f'{name:<18}{k:>4}{r[0]:>8.3f}{r[1]:>7.3f}{r[2]:>7.3f}{r[3]:>7.3f}{di:>8.3f}{dh:>8.3f}')

    # per-season replicate: R2 IVB + HB summed, vs B1
    print(f'\nPer-season (R2 IVB + R2 HB, within type), delta vs B1 proposed:')
    base = results['B1 proposed']
    hdr = f'{"form":<18}' + ''.join(f'{y:>9}' for y in seasons) + f'{"wins":>7}'
    print(hdr)
    for name, feats in forms:
        if name == 'B1 proposed':
            continue
        xi, xh = results[name]
        line, wins = f'{name:<18}', 0
        for y in seasons:
            m = cov & (A['season'] == y)
            r = within_type(A, xi, xh, m)
            rb = within_type(A, base[0], base[1], m)
            d = (r[0] + r[1]) - (rb[0] + rb[1])
            wins += d > 0
            line += f'{d:>+9.4f}'
        print(line + f'{wins:>4}/{len(seasons)}')


if __name__ == '__main__':
    main()
