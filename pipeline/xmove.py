"""Expected movement: xIVB / xHB and the IVBOE / HBOE residuals.

What a pitch "should" do, judged the way a HITTER judges it. One ridge fit per
(pitch type, throwing hand), fit on the season's MLB pitches, on the three
things a hitter can read before the ball moves:

    arm angle, extension, velocity.

Spin rate and the release spin axis are deliberately NOT inputs. Both are in
the data and a basis on them (BASIS = 'physics' below: two axis harmonics plus
a spin x axis tensor) explains far more of the movement itself, 0.49 / 0.42
within-type R^2 against 0.27 / 0.18. It shipped for about four hours on
2026-09-03 and was pulled the same day, because it answers the wrong question
for the chart it sits on. Measured across 2021-2025, each season predicted from
the other four (scripts/research/xmove/xmove_residual_value.py):

    residual measured against      whiff r   run-value r   whiff r with raw movement in
    slot only                        0.11        0.02              0.15
    slot + extension + velocity      0.13        0.07              0.22
    slot + ext + velo + spin + axis  0.05        0.05              0.15

Extra spin and a clean axis ARE the weapon. A physics expectation credits them
away, so its residual explains results worse than raw movement does. The
hitter expectation leaves them in the residual, where they read as surprise.
So the chart's ghost is "what a pitcher who looks like this usually throws",
and the arrow to the actual pitch is what the hitter did not see coming. Per
Wally, 2026-09-03: "put the hitter model back on the chart." The physics
basis stays in this module for research and scouting use; it is not scored.

The 'roc' variant swaps arm angle for the release point (RelPosZ, RelPosX) and
serves only pitches with no arm angle; both variants are fit on MLB pitches so
a ROC arm is measured against the MLB baseline.

Scoring is per pitch and the site sums the per-pitch expectations (micro rows
sumXIVB/nXIVB/sumXHB/nXHB). For the hitter basis that is exact either way; it
is kept per pitch so the physics basis, which is nonlinear in tilt, can be
switched in without touching a consumer.
"""
import math

import numpy as np

from pipeline.utils import break_tilt_to_minutes, safe_float

BASIS = 'hitter'          # 'hitter' (shipped) or 'physics' (research only)

# Ridge penalty on the standardised columns (intercept unpenalised) and the
# per-(type, hand) pool floor, swept JOINTLY per basis on 2025
# (scripts/research/xmove/xmove_ridge_sweep.py: four groups incl. FS_L, held-out
# RMSE excess over the full-pool OLS fit, median of 20 draws, after the MAD
# screen).
#   hitter (shipped): the excess is FLAT in lambda from 0 to 3e-2 at every n
#     (worst group 0.046" at n=150, 0.025" at 300), so 0 is a convention on a
#     flat region, i.e. plain least squares; the floor stays at the original
#     150 because the worst group is already under the 0.10" bar there (0.10"
#     = display rounding, itself a convention).
#   physics (research): 3e-3 is an interior optimum (0 and 1.0 both worse at
#     every n) because the harmonic columns go collinear in narrow-tilt groups
#     (CH_L: 1.8" excess at n=1000 unpenalised); floor 300 = first n under
#     0.10" on the worst group.
XMOVE_RIDGE = {'hitter': 0.0, 'physics': 3e-3}[BASIS]
XMOVE_MIN_N = {'hitter': 150, 'physics': 300}[BASIS]
MAD_THRESH = 6.0
SPIN_SCALE = 1000.0

HARMONIC_TERMS = ['spin', 'h1s', 'h1c', 'h2s', 'h2c', 'sp1s', 'sp1c', 'sp2s', 'sp2c']
TERMS_MLB = ['aa', 'ext', 'velo'] + (HARMONIC_TERMS if BASIS == 'physics' else [])
TERMS_ROC = ['rz', 'rx', 'ext', 'velo'] + (HARMONIC_TERMS if BASIS == 'physics' else [])


def hand_sign(throws):
    return 1.0 if throws == 'R' else -1.0


def release_theta(rtilt, throws):
    """Hand-signed release axis in radians from a clock string. None if absent."""
    minutes = break_tilt_to_minutes(rtilt)
    if minutes is None:
        return None
    return minutes / 720.0 * 2.0 * math.pi * hand_sign(throws)


def _spin_theta(p, throws):
    """Spin rate and hand-signed release axis for the physics basis. The
    hitter basis reads neither, so it returns placeholders that never gate a
    pitch out (coverage then matches the pre-2026-09-03 model exactly)."""
    if BASIS != 'physics':
        return 0.0, 0.0
    return safe_float(p.get('Spin Rate')), release_theta(p.get('RTilt'), throws)


def _design(cols, theta):
    """cols: dict of 1-D arrays (aa or rz/rx, ext, velo, and spin for the
    physics basis); theta: hand-signed release axis, read only by the physics
    basis. Returns the design matrix with an intercept column first."""
    n = len(cols['ext'])
    parts = [np.ones(n)]
    for k in ('rz', 'rx', 'aa'):
        if k in cols:
            parts.append(cols[k])
    parts += [cols['ext'], cols['velo']]
    if BASIS == 'physics':
        sp = cols['spin'] / SPIN_SCALE
        parts.append(cols['spin'])
        for k in (1, 2):
            s, c = np.sin(k * theta), np.cos(k * theta)
            parts += [s, c]
        for k in (1, 2):
            s, c = np.sin(k * theta), np.cos(k * theta)
            parts += [sp * s, sp * c]
    return np.column_stack(parts)


def _mad_keep(X):
    """Rows within MAD_THRESH MADs of the column median on every column."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    ok = np.ones(len(X), dtype=bool)
    for j in range(X.shape[1]):
        if mad[j] > 1e-9:
            ok &= np.abs(X[:, j] - med[j]) <= MAD_THRESH * mad[j]
    return ok


def _fit(rows, use_relpt):
    """rows: list of (ivb, hb_s, aa, rz, rx, ext, velo, spin, theta)."""
    A = np.asarray(rows, dtype='f8')
    screen_cols = [0, 1, 5, 6] + ([7] if BASIS == 'physics' else []) + ([3, 4] if use_relpt else [2])
    keep = _mad_keep(A[:, screen_cols])
    if keep.sum() < XMOVE_MIN_N:
        keep[:] = True
    A = A[keep]
    cols = {'ext': A[:, 5], 'velo': A[:, 6], 'spin': A[:, 7]}
    if use_relpt:
        cols['rz'], cols['rx'] = A[:, 3], A[:, 4]
    else:
        cols['aa'] = A[:, 2]
    X = _design(cols, A[:, 8])
    return {'ivb': _ridge(X, A[:, 0]).tolist(), 'hb': _ridge(X, A[:, 1]).tolist(),
            'n': int(len(A))}


def _ridge(X, y, lam=None):
    """Ridge on standardised columns (intercept, column 0, unpenalised).
    Returns coefficients in RAW units with the intercept first, so scoring is
    one dot product with the design row."""
    lam = XMOVE_RIDGE if lam is None else lam
    Z = X[:, 1:]
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Zs = (Z - mu) / sd
    ym = y.mean()
    n, k = Zs.shape
    b = np.linalg.solve(Zs.T @ Zs + lam * n * np.eye(k), Zs.T @ (y - ym))
    braw = b / sd
    return np.concatenate([[ym - braw @ mu], braw])


def fit_models(all_pitches):
    """One model per 'PT_H' key, MLB pitches only. Each key holds an 'mlb'
    variant (arm angle) and, where the pool allows, a 'roc' variant (release
    point) for pitches with no arm angle."""
    pools_mlb, pools_roc = {}, {}
    for p in all_pitches:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        pt = p.get('Pitch Type')
        throws = p.get('Throws')
        if not pt or throws not in ('L', 'R'):
            continue
        ivb = safe_float(p.get('xIndVrtBrk'))
        hb = safe_float(p.get('xHorzBrk'))
        ext = safe_float(p.get('Extension'))
        velo = safe_float(p.get('Velocity'))
        spin, theta = _spin_theta(p, throws)
        if None in (ivb, hb, ext, velo, spin, theta):
            continue
        s = hand_sign(throws)
        aa = safe_float(p.get('ArmAngle'))
        rz = safe_float(p.get('RelPosZ'))
        rx = safe_float(p.get('RelPosX'))
        key = pt + '_' + throws
        if aa is not None:
            pools_mlb.setdefault(key, []).append((ivb, hb * s, aa, 0.0, 0.0, ext, velo, spin, theta))
        if rz is not None and rx is not None:
            pools_roc.setdefault(key, []).append((ivb, hb * s, 0.0, rz, rx * s, ext, velo, spin, theta))
    models = {}
    for key in sorted(set(pools_mlb) | set(pools_roc)):
        m = {}
        if len(pools_mlb.get(key, ())) >= XMOVE_MIN_N:
            m['mlb'] = _fit(pools_mlb[key], use_relpt=False)
        if len(pools_roc.get(key, ())) >= XMOVE_MIN_N:
            m['roc'] = _fit(pools_roc[key], use_relpt=True)
        if m:
            models[key] = m
    return models


def score_pitch(models, p):
    """(xIVB, xHB) for one pitch dict, HB in the display frame. (None, None)
    when the group has no model or the pitch lacks an input. Arm-angle
    variant first, release-point variant as the fallback."""
    pt = p.get('Pitch Type')
    throws = p.get('Throws')
    m = models.get((pt or '') + '_' + (throws or ''))
    if not m:
        return None, None
    ext = safe_float(p.get('Extension'))
    velo = safe_float(p.get('Velocity'))
    spin, theta = _spin_theta(p, throws)
    if None in (ext, velo, spin, theta):
        return None, None
    s = hand_sign(throws)
    aa = safe_float(p.get('ArmAngle'))
    rz = safe_float(p.get('RelPosZ'))
    rx = safe_float(p.get('RelPosX'))
    if m.get('mlb') and aa is not None:
        cols = {'aa': np.array([aa])}
        variant = m['mlb']
    elif m.get('roc') and rz is not None and rx is not None:
        cols = {'rz': np.array([rz]), 'rx': np.array([rx * s])}
        variant = m['roc']
    else:
        return None, None
    cols.update({'ext': np.array([ext]), 'velo': np.array([velo]), 'spin': np.array([spin])})
    X = _design(cols, np.array([theta]))
    xivb = float(X @ np.asarray(variant['ivb']))
    xhb = float(X @ np.asarray(variant['hb'])) * s
    return xivb, xhb


def score_all(models, all_pitches):
    """Vectorised score_pitch over a list of pitch dicts. Writes '_xivb' and
    '_xhb' onto each dict (absent when unscored) and returns the count scored."""
    by_key = {}
    for i, p in enumerate(all_pitches):
        pt = p.get('Pitch Type')
        throws = p.get('Throws')
        key = (pt or '') + '_' + (throws or '')
        if key not in models:
            continue
        ext = safe_float(p.get('Extension'))
        velo = safe_float(p.get('Velocity'))
        spin, theta = _spin_theta(p, throws)
        if None in (ext, velo, spin, theta):
            continue
        aa = safe_float(p.get('ArmAngle'))
        rz = safe_float(p.get('RelPosZ'))
        rx = safe_float(p.get('RelPosX'))
        s = hand_sign(throws)
        if models[key].get('mlb') and aa is not None:
            by_key.setdefault((key, 'mlb'), []).append((i, aa, 0.0, 0.0, ext, velo, spin, theta, s))
        elif models[key].get('roc') and rz is not None and rx is not None:
            by_key.setdefault((key, 'roc'), []).append((i, 0.0, rz, rx * s, ext, velo, spin, theta, s))
    scored = 0
    for (key, variant), rows in by_key.items():
        A = np.asarray(rows, dtype='f8')
        cols = {'ext': A[:, 4], 'velo': A[:, 5], 'spin': A[:, 6]}
        if variant == 'mlb':
            cols['aa'] = A[:, 1]
        else:
            cols['rz'], cols['rx'] = A[:, 2], A[:, 3]
        X = _design(cols, A[:, 7])
        xi = X @ np.asarray(models[key][variant]['ivb'])
        xh = (X @ np.asarray(models[key][variant]['hb'])) * A[:, 8]
        for r, i in enumerate(A[:, 0].astype(int)):
            all_pitches[i]['_xivb'] = float(xi[r])
            all_pitches[i]['_xhb'] = float(xh[r])
        scored += len(rows)
    return scored


def export(models):
    """JSON-ready copy for metadata_rs.json (coefficients at 6 decimals)."""
    out = {}
    for key, m in models.items():
        out[key] = {}
        for variant, v in m.items():
            out[key][variant] = {
                'terms': TERMS_ROC if variant == 'roc' else TERMS_MLB,
                'ivb': [round(c, 6) for c in v['ivb']],
                'hb': [round(c, 6) for c in v['hb']],
                'n': v['n'],
            }
    return out
