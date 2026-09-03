"""Expected movement: xIVB / xHB and the IVBOE / HBOE residuals.

What a pitch "should" do, given how it was thrown. One linear model per
(pitch type, throwing hand), fit on the season's MLB pitches, on:

    arm angle, extension, velocity, spin rate,
    sin/cos of the release spin axis (two harmonics),
    spin rate x each of those four harmonics.

The release axis is the measured RELEASE tilt (Sheets column RTilt, from the
feed's spin axis), never the movement-derived OTilt: OTilt is atan2(HB, IVB),
the answer restated. The axis enters as a hand-signed angle (12:00 = 0, arm
side positive) so one fit serves either hand's mirror image, and HB is fit
hand-signed for the same reason and mirrored back for display.

Why this basis and not the earlier one. The shipped model until 2026-09-03
conditioned on arm angle, extension and velocity only. Spin rate and the
release axis are the two largest physical causes of movement and both are in
the data at 99%+, so that model explained 0.27 / 0.18 of within-type movement
and its residual restated the raw column (|corr| 0.80 / 0.84). This basis
explains 0.49 / 0.42 out of sample, wins in every season 2021-2025 when each is
fit self-contained, and its residual correlates 0.62 / 0.66 with the raw
column. The two harmonics and the spin x axis tensor bracket an interior
optimum (harmonics 1/2/3 give 0.519 / 0.528 / 0.517; a spin spline buys
nothing); velocity x axis, slot x axis, squares and release point each add at
most 0.01 and lose in at least one season. docs/expected_movement_review.md
Findings 1-5 and scripts/research/xmove/xmove_pertype_ladder.py hold the
grids.

Two things this model is NOT. It is not label-free: a re-tagged pitch moves
its own expectation, because the class is an interaction with the release
parameters and nothing measured at release separates a four-seam from a
sinker (xmove_agnostic_flight.py, 2026-09-03: release-only, spin-efficiency
and drag inputs all leave the FF/SI expected gap under 3 inches; the label
gives 9.3). And it is not a physics calculation: the expectation is what
pitches THROWN like this usually do, so the residual is seam-shifted wake plus
gyro fraction plus whatever the sensors miss, measured against that pool.

The fit is a ridge (XMOVE_RIDGE) on standardised columns; see the constant
note for why and for the sweep.

Scoring is per pitch. The basis is nonlinear in tilt, so scoring at group
means is wrong by a median 0.1" and up to 3" on gyro sliders
(xmove_basis_at_means.py); every consumer sums per-pitch expectations instead.

The ROC variant swaps arm angle for the release point (RelPosZ, RelPosX) and
serves only pitches with no arm angle; both variants are fit on MLB pitches so
a ROC arm is measured against the MLB baseline.
"""
import math

import numpy as np

from pipeline.utils import break_tilt_to_minutes, safe_float

# Ridge penalty on the standardised columns, intercept unpenalised. When a
# group's release tilt barely varies its harmonic columns are nearly straight
# lines in tilt and the spin tensor copies them, so an unpenalised fit is
# near-singular at small pools (CH_L: 1.8" excess RMSE at n = 1000 with no
# penalty). Swept 2026-09-03 (scripts/research/xmove/xmove_ridge_sweep.py,
# 2025, four groups, held-out RMSE excess over the full-pool OLS fit): both
# grid edges (0 and 1.0) are worse at every n, 3e-3 to 1e-2 is the interior
# optimum, and at n >= 5000 the cost is 0.004" (indistinguishable from OLS, so
# the five-season ladder verdict carries).
XMOVE_RIDGE = 3e-3
# Per-(type, hand) pool floor. Same sweep: under XMOVE_RIDGE the worst group's
# excess drops under 0.10" at n = 300 (0.064"), 0.032" at 500. The 0.10" bar
# is a convention: below it the floor's cost is invisible at the 0.1" the site
# rounds to. A group under the floor blanks, as before (the old floor was 150
# for 3 regressors).
XMOVE_MIN_N = 300
MAD_THRESH = 6.0
SPIN_SCALE = 1000.0

TERMS_MLB = ['aa', 'ext', 'velo', 'spin',
             'h1s', 'h1c', 'h2s', 'h2c', 'sp1s', 'sp1c', 'sp2s', 'sp2c']
TERMS_ROC = ['rz', 'rx', 'ext', 'velo', 'spin',
             'h1s', 'h1c', 'h2s', 'h2c', 'sp1s', 'sp1c', 'sp2s', 'sp2c']


def hand_sign(throws):
    return 1.0 if throws == 'R' else -1.0


def release_theta(rtilt, throws):
    """Hand-signed release axis in radians from a clock string. None if absent."""
    minutes = break_tilt_to_minutes(rtilt)
    if minutes is None:
        return None
    return minutes / 720.0 * 2.0 * math.pi * hand_sign(throws)


def _design(cols, theta):
    """cols: dict of 1-D arrays (aa or rz/rx, ext, velo, spin); theta: array.
    Returns the design matrix with an intercept column first."""
    n = len(theta)
    sp = cols['spin'] / SPIN_SCALE
    parts = [np.ones(n)]
    for k in ('rz', 'rx', 'aa'):
        if k in cols:
            parts.append(cols[k])
    parts += [cols['ext'], cols['velo'], cols['spin']]
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
    screen_cols = [0, 1, 5, 6, 7] + ([3, 4] if use_relpt else [2])
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
        spin = safe_float(p.get('Spin Rate'))
        theta = release_theta(p.get('RTilt'), throws)
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
    spin = safe_float(p.get('Spin Rate'))
    theta = release_theta(p.get('RTilt'), throws)
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
        spin = safe_float(p.get('Spin Rate'))
        theta = release_theta(p.get('RTilt'), throws)
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
