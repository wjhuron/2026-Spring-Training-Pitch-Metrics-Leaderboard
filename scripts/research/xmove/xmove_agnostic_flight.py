"""Label-free expected movement with FLIGHT-derived inputs: does it separate FF / SI / FC?

Wally's objection to the July agnostic form (xmove_agnostic.py, review Finding 6):
a pitcher's four-seam and sinker share release tilt, velocity, spin and slot, so
their expected bubbles land on top of each other. The label was the only thing
that separated them, and the label is what he wants gone.

Two per-pitch quantities from scripts/ci/kinematics_lib.py are NOT the label and
NOT the movement direction:
  KinEff  transverse-spin fraction inferred from lift magnitude
  KinCd   drag coefficient from the flight deceleration
Seam-shifted wake makes the wake asymmetric, which changes drag as well as lift,
so KinCd may carry seam information that tells a sinker from a four-seam at the
same tilt. KinEff carries gyro. This measures whether either input, added to the
release-only pooled surface, pushes the expected bubbles apart without a label.

Ladder, pooled by hand x season, cross-fit by game parity (a pitch is never
scored by a model that saw its game):
  REL      release only: arm angle, extension, velocity, spin, 2 axis harmonics,
           spin x axis tensor (the July form)
  +EFF     REL + KinEff (+ KinEff x axis harmonics)
  +CD      REL + KinCd  (+ KinCd x axis harmonics)
  +BOTH    both
  PERTYPE  the per-class reference, REL basis fit per (type, hand, season)

Reported, at the rendered unit (pitcher x hand x type x season, >= 50 pitches):
  GAP     median |expected FF - expected SI| in inches (and FF/FC, SI/FC) over
          pitcher-seasons throwing both. This is the bubble-separation question.
  APART   share of those pitcher-seasons with the two expected bubbles >= 3"
          apart. 3" is the per-type OE ring the review proposed; it is a
          convention here, not a fit.
  RESID   median residual gap on the same pairs (what the seam story is left with)
  R2      within-type R^2, pooled over types, ALONG and CROSS the release axis
          separately. KinEff is inferred from lift magnitude, so ALONG is
          partly self-referential under +EFF; CROSS is gyro-independent and
          is the honest column.
  DIST    mean within-type |corr(OE, raw)| at the rendered unit.

Then Cade Cavalli 2026, scored per pitch type under every form by the 2025 fit.

Usage: python3 scripts/research/xmove/xmove_agnostic_flight.py [--cache-only]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SCRATCH = os.environ.get(
    'XMOVE_DIR',
    '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/c68c2281-f8bc-4edf-b57d-429f9aa4c530/scratchpad')
SEASONS = [2021, 2022, 2023, 2024, 2025]
PTS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']
MIN_N = 2000
UNIT_MIN = 50
APART_IN = 3.0

COLS = ['Pitcher', 'Throws', 'Pitch Type', 'Velocity', 'Spin Rate', 'SpinAxis',
        'xIndVrtBrk', 'xHorzBrk', 'Extension', 'ArmAngle', 'KinEff', 'KinCd',
        'RelPosZ', 'RelPosX', '_game_pk']
NUM = ['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk', 'xHorzBrk', 'Extension',
       'ArmAngle', 'KinEff', 'KinCd', 'RelPosZ', 'RelPosX']


def build_cache(year):
    path = f'{SCRATCH}/xmove_flight_{year}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    with open(f'{ROOT}/data/_pitches{year}_training.pkl', 'rb') as f:
        rows = pickle.load(f)
    df = pd.DataFrame([{c: r.get(c) for c in COLS} for r in rows])
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['season'] = year
    df.to_parquet(path, index=False)
    print(f'{year}: {len(df):,} rows cached', file=sys.stderr)
    return df


def to_arrays(d):
    d = d[d['Pitch Type'].isin(PTS) & d['Throws'].isin(['L', 'R'])]
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension',
                         'ArmAngle', 'SpinAxis', 'Spin Rate', 'KinEff', 'KinCd'])
    s = np.where(d['Throws'].values == 'R', 1.0, -1.0)
    ivb = d['xIndVrtBrk'].values.astype('f8')
    hb_s = d['xHorzBrk'].values.astype('f8') * s
    rtilt = (d['SpinAxis'].values.astype('f8') - 180.0) % 360.0
    th = np.radians(rtilt * s)
    velo = d['Velocity'].values.astype('f8')
    spin = d['Spin Rate'].values.astype('f8')
    A = dict(
        pt=d['Pitch Type'].values, thr=d['Throws'].values,
        pitcher=d['Pitcher'].values, season=d['season'].values.astype('i4'),
        game=d['_game_pk'].values.astype('i8'),
        ivb=ivb, hb_s=hb_s, velo=velo, spin=spin, spin_v=spin / velo,
        ext=d['Extension'].values.astype('f8'), aa=d['ArmAngle'].values.astype('f8'),
        eff=d['KinEff'].values.astype('f8'), cd=d['KinCd'].values.astype('f8'),
        ct=np.cos(th), st=np.sin(th),
    )
    A['along'] = ivb * A['ct'] + hb_s * A['st']
    A['cross'] = -ivb * A['st'] + hb_s * A['ct']
    for k in (1, 2):
        A[f'h{k}s'], A[f'h{k}c'] = np.sin(k * th), np.cos(k * th)
        A[f'sv{k}s'], A[f'sv{k}c'] = A['spin_v'] * A[f'h{k}s'], A['spin_v'] * A[f'h{k}c']
        A[f'ef{k}s'], A[f'ef{k}c'] = A['eff'] * A[f'h{k}s'], A['eff'] * A[f'h{k}c']
        A[f'cd{k}s'], A[f'cd{k}c'] = A['cd'] * A[f'h{k}s'], A['cd'] * A[f'h{k}c']
    return A


REL = ['aa', 'ext', 'velo', 'spin', 'h1s', 'h1c', 'h2s', 'h2c', 'sv1s', 'sv1c', 'sv2s', 'sv2c']
EFF = ['eff', 'ef1s', 'ef1c', 'ef2s', 'ef2c']
CD = ['cd', 'cd1s', 'cd1c', 'cd2s', 'cd2c']
FORMS = [
    ('REL', REL, 'pool'),
    ('+EFF', REL + EFF, 'pool'),
    ('+CD', REL + CD, 'pool'),
    ('+BOTH', REL + EFF + CD, 'pool'),
    ('PERTYPE', REL, 'type'),
]


def design(A, feats, idx):
    return np.column_stack([np.ones(len(idx))] + [A[f][idx] for f in feats])


def fit_cv(A, feats, gid):
    n = len(A['ivb'])
    xi, xh = np.full(n, np.nan), np.full(n, np.nan)
    order = np.argsort(gid, kind='stable')
    bounds = np.searchsorted(gid[order], np.arange(gid.max() + 2))
    for gi in range(gid.max() + 1):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) == 0:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N or len(te) == 0:
                continue
            Xt, Xs = design(A, feats, tr), design(A, feats, te)
            bi = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
            bh = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]
            xi[te], xh[te] = Xs @ bi, Xs @ bh
    return xi, xh


def fit_full(A, feats, mask):
    """Coefficients on every pitch in mask (for scoring an outside season)."""
    idx = np.where(mask)[0]
    X = design(A, feats, idx)
    bi = np.linalg.lstsq(X, A['ivb'][idx], rcond=None)[0]
    bh = np.linalg.lstsq(X, A['hb_s'][idx], rcond=None)[0]
    return bi, bh


def r2(y, yhat, m):
    yy, hh = y[m], yhat[m]
    return 1 - ((yy - hh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def unit_frame(A, xi, xh):
    ok = np.isfinite(xi) & np.isfinite(xh)
    d = pd.DataFrame({
        'pitcher': A['pitcher'][ok], 'thr': A['thr'][ok], 'pt': A['pt'][ok],
        'season': A['season'][ok], 'ivb': A['ivb'][ok], 'hb': A['hb_s'][ok],
        'xivb': xi[ok], 'xhb': xh[ok],
        'ivb_oe': A['ivb'][ok] - xi[ok], 'hb_oe': A['hb_s'][ok] - xh[ok],
    })
    u = d.groupby(['pitcher', 'thr', 'pt', 'season']).agg(
        n=('ivb', 'size'), **{c: (c, 'mean') for c in
                              ['ivb', 'hb', 'xivb', 'xhb', 'ivb_oe', 'hb_oe']}
    ).reset_index()
    return u[u.n >= UNIT_MIN]


def pair_stats(u, a, b):
    ua = u[u.pt == a].set_index(['pitcher', 'thr', 'season'])
    ub = u[u.pt == b].set_index(['pitcher', 'thr', 'season'])
    j = ua.join(ub, lsuffix='_a', rsuffix='_b', how='inner')
    if len(j) == 0:
        return None
    gap = np.hypot(j.xivb_a - j.xivb_b, j.xhb_a - j.xhb_b)
    obs = np.hypot(j.ivb_a - j.ivb_b, j.hb_a - j.hb_b)
    res = np.hypot(j.ivb_oe_a - j.ivb_oe_b, j.hb_oe_a - j.hb_oe_b)
    return dict(n=len(j), obs=np.median(obs), gap=np.median(gap),
                apart=(gap >= APART_IN).mean(), resid=np.median(res))


def evaluate(name, A, xi, xh):
    m0 = np.isfinite(xi)
    # residual in the release-axis frame
    oi, oh = A['ivb'] - xi, A['hb_s'] - xh
    x_along = xi * A['ct'] + xh * A['st']
    x_cross = -xi * A['st'] + xh * A['ct']
    r_al, r_cr, r_i, r_h, d_i, d_h = [], [], [], [], [], []
    u = unit_frame(A, xi, xh)
    for pt in PTS:
        m = m0 & (A['pt'] == pt)
        if m.sum() < 5000:
            continue
        r_al.append(r2(A['along'], x_along, m))
        r_cr.append(r2(A['cross'], x_cross, m))
        r_i.append(r2(A['ivb'], xi, m))
        r_h.append(r2(A['hb_s'], xh, m))
        b = u[u.pt == pt]
        d_i.append(abs(corr(b.ivb_oe.values, b.ivb.values)))
        d_h.append(abs(corr(b.hb_oe.values, b.hb.values)))
    print(f'\n== {name}  coverage {m0.mean():.3f}')
    print(f'   within-type R2  IVB {np.mean(r_i):.3f}  HB {np.mean(r_h):.3f}  '
          f'ALONG {np.mean(r_al):.3f}  CROSS {np.mean(r_cr):.3f}   '
          f'DIST i {np.mean(d_i):.3f} h {np.mean(d_h):.3f}')
    print(f'   {"pair":<8}{"n":>6}{"observed":>10}{"GAP":>8}{"APART":>8}{"RESID":>8}')
    for a, b in (('FF', 'SI'), ('FF', 'FC'), ('SI', 'FC'), ('SL', 'ST'), ('CU', 'ST')):
        p = pair_stats(u, a, b)
        if p:
            print(f'   {a}/{b:<5}{p["n"]:>6}{p["obs"]:>9.1f}"{p["gap"]:>7.1f}"'
                  f'{p["apart"]:>8.2f}{p["resid"]:>7.1f}"')
    return u


def clock_to_deg(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v)
    if ':' not in s:
        return pd.to_numeric(s, errors='coerce')
    h, m = s.split(':')
    return (float(h) % 12) * 30.0 + float(m) * 0.5


def cavalli_2026(A, coefs):
    with open(f'{ROOT}/data/all_pitches_rs_cache.pkl', 'rb') as f:
        rows = [r for r in pickle.load(f) if 'Cavalli' in str(r.get('Pitcher', ''))]
    with open(f'{ROOT}/data/kinematics_2026_sidecar.pkl', 'rb') as f:
        kin = pickle.load(f)
    df = pd.DataFrame(rows)
    k = df['PitchID'].map(kin)
    df['KinEff'] = k.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    df['KinCd'] = k.map(lambda t: t[2] if isinstance(t, tuple) else np.nan)
    # the 2026 cache carries RTilt as a clock string (12:00 = pure backspin);
    # clock degrees equal the (SpinAxis - 180) % 360 convention used above
    df['rtilt'] = df['RTilt'].map(clock_to_deg)
    for c in ['Velocity', 'Spin Rate', 'xIndVrtBrk', 'xHorzBrk', 'IndVertBrk', 'HorzBrk',
              'Extension', 'ArmAngle']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ivb'] = df['xIndVrtBrk'].fillna(df['IndVertBrk'])
    df['hb'] = df['xHorzBrk'].fillna(df['HorzBrk'])
    n_all = len(df)
    df = df.dropna(subset=['ivb', 'hb', 'Velocity', 'Spin Rate', 'rtilt', 'Extension',
                           'ArmAngle', 'KinEff', 'KinCd'])
    print(f'\n== Cade Cavalli 2026: {len(df)} of {n_all} pitches have every input '
          f'(RHP, hand-signed frame: HB positive = arm side)')
    th = np.radians(df['rtilt'].values)
    B = dict(aa=df['ArmAngle'].values, ext=df['Extension'].values,
             velo=df['Velocity'].values, spin=df['Spin Rate'].values,
             eff=df['KinEff'].values, cd=df['KinCd'].values)
    B['spin_v'] = B['spin'] / B['velo']
    for kk in (1, 2):
        B[f'h{kk}s'], B[f'h{kk}c'] = np.sin(kk * th), np.cos(kk * th)
        B[f'sv{kk}s'], B[f'sv{kk}c'] = B['spin_v'] * B[f'h{kk}s'], B['spin_v'] * B[f'h{kk}c']
        B[f'ef{kk}s'], B[f'ef{kk}c'] = B['eff'] * B[f'h{kk}s'], B['eff'] * B[f'h{kk}c']
        B[f'cd{kk}s'], B[f'cd{kk}c'] = B['cd'] * B[f'h{kk}s'], B['cd'] * B[f'h{kk}c']
    idx = np.arange(len(df))
    out = {}
    for name, feats, how in FORMS:
        if how == 'pool':
            bi, bh = coefs[name]
            X = design(B, feats, idx)
            out[name] = (X @ bi, X @ bh)
        else:
            xi, xh = np.full(len(df), np.nan), np.full(len(df), np.nan)
            for pt in df['Pitch Type'].unique():
                if pt not in coefs[name]:
                    continue
                bi, bh = coefs[name][pt]
                m = (df['Pitch Type'] == pt).values
                X = design(B, feats, idx[m])
                xi[m], xh[m] = X @ bi, X @ bh
            out[name] = (xi, xh)
    hdr = f'{"type":<5}{"n":>4}{"velo":>6}{"spin":>6}{"tilt":>6}{"eff":>5}{"Cd":>6}' \
          f'{"actual IVB/HB":>15}' + ''.join(f'{n:>15}' for n, _, _ in FORMS)
    print(hdr)
    for pt in ['FF', 'SI', 'FC', 'CU', 'ST', 'CH']:
        m = (df['Pitch Type'] == pt).values
        if m.sum() < 10:
            continue
        line = (f'{pt:<5}{m.sum():>4}{df.Velocity[m].mean():>6.1f}{df["Spin Rate"][m].mean():>6.0f}'
                f'{df.rtilt[m].mean():>6.0f}{df.KinEff[m].mean():>5.2f}{df.KinCd[m].mean():>6.3f}'
                f'{df.ivb[m].mean():>8.1f}/{df.hb[m].mean():>5.1f}')
        for name, _, _ in FORMS:
            xi, xh = out[name]
            if np.isfinite(xi[m]).any():
                line += f'{np.nanmean(xi[m]):>8.1f}/{np.nanmean(xh[m]):>5.1f}'
            else:
                line += f'{"-":>15}'
        print(line)
    print('   (expected columns are xIVB/xHB; residual = actual minus expected)')


def main():
    frames = [build_cache(y) for y in SEASONS]
    if '--cache-only' in sys.argv:
        return
    A = to_arrays(pd.concat(frames, ignore_index=True))
    del frames
    print(f'{len(A["ivb"]):,} pitches with movement, axis, spin, arm angle, extension, '
          f'KinEff and KinCd, 2021-2025', file=sys.stderr)
    gid_pool = pd.factorize(pd.Series(A['thr']) + '_' + pd.Series(A['season']).astype(str))[0]
    gid_type = pd.factorize(pd.Series(A['pt']) + '_' + pd.Series(A['thr']) + '_' +
                            pd.Series(A['season']).astype(str))[0]
    coefs = {}
    m25 = A['season'] == 2025
    for name, feats, how in FORMS:
        gid = gid_pool if how == 'pool' else gid_type
        xi, xh = fit_cv(A, feats, gid)
        evaluate(name, A, xi, xh)
        # 2025 RHP coefficients for scoring Cavalli
        if how == 'pool':
            coefs[name] = fit_full(A, feats, m25 & (A['thr'] == 'R'))
        else:
            coefs[name] = {pt: fit_full(A, feats, m25 & (A['thr'] == 'R') & (A['pt'] == pt))
                           for pt in PTS if (m25 & (A['thr'] == 'R') & (A['pt'] == pt)).sum() >= MIN_N}
        sys.stdout.flush()
    cavalli_2026(A, coefs)


if __name__ == '__main__':
    main()
