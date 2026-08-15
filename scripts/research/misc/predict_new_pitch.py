#!/usr/bin/env python3
"""
Predict hypothetical new pitch metrics for a pitcher given his current
arsenal and biomechanics.

Fits a multivariate normal over:
  [arm_angle, extension, FF_velo, FF_IVB, FF_HB, FF_spin, (other_arsenal)?, T_velo, T_IVB, T_HB, T_spin]
per (target pitch type, pitcher hand) across all pitchers who throw both the
anchor fastball and the target pitch. Then conditions on the target pitcher's
known values to produce the posterior distribution of the target pitch.

Two tiers:
  Tier 1: condition only on biomechanics and the anchor fastball (FF, or SI
          if no FF). Always available.
  Tier 2: also condition on the pitcher's other existing pitches. Sharpens the
          prediction but training sample shrinks. Skipped for a given target if
          training sample drops below 30.

Usage:
  python3 scripts/research/misc/predict_new_pitch.py "Beeter, Clayton" WSH CU
  python3 scripts/research/misc/predict_new_pitch.py "Beeter, Clayton" WSH CU SI CH
  python3 scripts/research/misc/predict_new_pitch.py --tier 1 "Beeter, Clayton" WSH CU
  python3 scripts/research/misc/predict_new_pitch.py --no-plot "Beeter, Clayton" WSH CU

Outputs:
  - printed arsenal summary, predictions, and Mahalanobis comps
  - two-panel plot at scripts/outputs/new_pitch/<pitcher>_<pitches>.png
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = Path.home() / 'Downloads'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT = 25       # Conditioning / training / calibration threshold
MIN_DISPLAY = 1      # Plot + arsenal table + skip-existing threshold. No real
                     # floor — if the pitcher threw the pitch even once, it's
                     # part of his arsenal and should display. The MVN math
                     # still uses MIN_COUNT for shape reliability separately.
TIER2_MIN_TRAIN = 30
N_COMPS = 10

# Metrics we fit over in the MVN. Spin stays in the joint distribution so it
# sharpens the velo/IVB/HB conditional, but we do not display predicted spin
# because the cross-pitcher variance on spin-per-grip is high enough that
# "don't trust it" is the right call.
TARGET_METRICS = ['velocity', 'indVertBrk', 'horzBrk', 'spinRate']
PITCH_METRICS = ['velocity', 'indVertBrk', 'horzBrk', 'spinRate']
# Subset actually shown to the user
DISPLAY_METRICS = ['velocity', 'indVertBrk', 'horzBrk']
DISPLAY_INDICES = [TARGET_METRICS.index(m) for m in DISPLAY_METRICS]
# Biomechanics conditioning
BIOMECH = ['armAngle', 'extension']

# Outcome metrics projected via shape-comp regression. Each entry is
# (key, display_label, fmt, getter) where:
#   key        — string identifier (used for column ordering / dict keys)
#   label      — display label
#   fmt        — 'pct' (rate, x100 → %) or 'woba' (wOBA-scale, .XXX)
#   getter     — None for `row.get(key)`, OR a function(row) → float|None
#                for derived metrics like CSR = cswPct − swStrRate.
def _csr(row):
    """Called-strike rate per pitch = cswPct - swStrRate (called strikes / total pitches)."""
    csw = row.get('cswPct'); sw_rate = row.get('swStrRate')
    if csw is None or sw_rate is None:
        return None
    return csw - sw_rate

OUTCOME_METRICS = [
    ('swStrPct',   'Whiff%',     'pct',  None),
    ('callStrPct', 'CalledStr%', 'pct',  _csr),
    ('chasePct',   'Chase%',     'pct',  None),
    ('gbPct',      'GB%',        'pct',  None),
    ('hardHitPct', 'Hard-Hit%',  'pct',  None),
    ('xwOBAcon',   'xwOBAcon',   'woba', None),
]


def _outcome_getter(okey):
    """Resolve OUTCOME_METRICS getter — fall back to row.get(key) when not provided."""
    for k, _, _, getter in OUTCOME_METRICS:
        if k == okey:
            return getter if getter is not None else (lambda r, kk=k: r.get(kk))
    return lambda r: None
# Min comp-pitch sample to include in outcome projection. Below this the
# comp's rates are too noisy to contribute meaningfully. Set low enough that
# early-season comps still pass; the sqrt(n) weighting handles the rest.
OUTCOME_MIN_PITCHES = 30
# Empirical-Bayes shrinkage strength — pseudo-pitches added to (pitch_type, hand)
# league mean to stabilize comp rates with small samples. With k=30, a comp
# with 100 pitches is ~75% comp / 25% league. Light enough to preserve signal,
# heavy enough to discount 30-pitch outliers.
OUTCOME_SHRINK_K = 30
# Distance-weight bandwidth (in pop-shape SDs). Smaller = sharper falloff,
# more weight on the closest-shape comps. With sigma=1.0, a comp 1 SD away
# in shape gets ~60% weight; at 2 SD, ~14%; at 3 SD, ~1%. This prevents
# the average from being dominated by the long tail of dissimilar comps.
OUTCOME_DIST_SIGMA = 1.0

# Pitch colors matching js/utils.js site palette
PITCH_COLORS = {
    'FF': '#0072B2', 'SI': '#FFD700', 'FC': '#8B5A2B', 'SL': '#D55E00',
    'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288', 'CH': '#009E73',
    'FS': '#CC79A7', 'KN': '#AAAAAA', 'SC': '#999999', 'CS': '#666666',
}
PITCH_FULL_NAMES = {
    'FF': 'Four-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
    'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball', 'CH': 'Changeup',
    'FS': 'Splitter', 'KN': 'Knuckleball', 'SC': 'Screwball', 'CS': 'Slow Curve',
}

# Dark theme palette
BG_COLOR = '#0f172a'
GRID_COLOR = '#334155'
TEXT_COLOR = '#e2e8f0'
AXIS_COLOR = '#475569'


# ────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────

def load_arsenals():
    """Return a dict: (pitcher, team) -> {pitch_type: row, 'throws': hand}."""
    with (DATA_DIR / 'pitch_leaderboard_rs.json').open() as f:
        rows = json.load(f)
    arsenals = defaultdict(dict)
    for r in rows:
        key = (r['pitcher'], r['team'])
        pt = r['pitchType']
        arsenals[key][pt] = r
        # Store throws once per pitcher
        arsenals[key]['_throws'] = r.get('throws')
    return dict(arsenals)


def compute_pitch_population_stats(arsenals):
    """For each (pitch_type, hand) pair, compute population shape mean+covariance
    AND outcome league averages. Used by shape-comp regression for distance
    weighting and by the report formatter for vs-league deltas.

    Returns:
        shape_stats: {(pt, hand): (mean_3vec, cov_3x3)} — over (velo, IVB, HB)
        outcome_lg:  {(pt, hand): {okey: weighted_mean}} — pitch-count-weighted
    """
    groups = defaultdict(list)
    for arsenal in arsenals.values():
        hand = arsenal.get('_throws')
        if not hand:
            continue
        for pt, row in arsenal.items():
            if pt.startswith('_') or not isinstance(row, dict):
                continue
            if (row.get('count') or 0) < MIN_COUNT:
                continue
            groups[(pt, hand)].append(row)

    shape_stats = {}
    outcome_lg = {}
    for (pt, hand), rows in groups.items():
        # Shape mean + covariance over (velo, IVB, HB)
        shape_vecs = []
        for r in rows:
            v = r.get('velocity'); iv = r.get('indVertBrk'); hb = r.get('horzBrk')
            if v is None or iv is None or hb is None:
                continue
            shape_vecs.append([float(v), float(iv), float(hb)])
        if len(shape_vecs) >= 5:
            X = np.array(shape_vecs)
            shape_stats[(pt, hand)] = (
                X.mean(axis=0),
                np.cov(X, rowvar=False, ddof=1) + 1e-3 * np.eye(3),
            )
        # League outcome avg, weighted by pitch count
        lg = {}
        for okey, _, _, _ in OUTCOME_METRICS:
            getter = _outcome_getter(okey)
            pairs = [(getter(r), r.get('count') or 0) for r in rows
                     if getter(r) is not None and (r.get('count') or 0) > 0]
            if pairs:
                tw = sum(w for _, w in pairs)
                lg[okey] = sum(v * w for v, w in pairs) / tw
            else:
                lg[okey] = None
        outcome_lg[(pt, hand)] = lg
    return shape_stats, outcome_lg


def extract_vec(row, fields):
    """Pull a float vector from a row in field order. Returns None if any are missing."""
    vec = []
    for f in fields:
        v = row.get(f)
        if v is None:
            return None
        vec.append(float(v))
    return np.array(vec)


# ────────────────────────────────────────────────────────────────────────
# MVN conditional distribution
# ────────────────────────────────────────────────────────────────────────

def mvn_conditional(mu, cov, x_a, d_a):
    """Standard MVN conditional: given observed values x_a for the first d_a
    dimensions, return (mu_b_given_a, cov_b_given_a) for the remaining dims.
    """
    mu_a = mu[:d_a]
    mu_b = mu[d_a:]
    cov_aa = cov[:d_a, :d_a]
    cov_ab = cov[:d_a, d_a:]
    cov_ba = cov[d_a:, :d_a]
    cov_bb = cov[d_a:, d_a:]

    # Use solve instead of inv for numerical stability
    # cov_ba @ inv(cov_aa) @ (x_a - mu_a)
    delta = x_a - mu_a
    z = np.linalg.solve(cov_aa, delta)
    mu_b_given_a = mu_b + cov_ba @ z

    # cov_bb - cov_ba @ inv(cov_aa) @ cov_ab
    K = np.linalg.solve(cov_aa, cov_ab)
    cov_b_given_a = cov_bb - cov_ba @ K

    return mu_b_given_a, cov_b_given_a


def mahalanobis_dist_batch(X, x_target, cov_aa):
    """Mahalanobis distance from each row of X to x_target under cov_aa."""
    diffs = X - x_target
    # d^2 = diff^T @ inv(cov_aa) @ diff for each row
    solved = np.linalg.solve(cov_aa, diffs.T).T  # (n, d)
    d2 = np.sum(diffs * solved, axis=1)
    d2 = np.clip(d2, 0, None)  # guard against tiny negatives from float error
    return np.sqrt(d2)


# ────────────────────────────────────────────────────────────────────────
# Training matrix construction
# ────────────────────────────────────────────────────────────────────────

def pick_anchor(arsenal):
    """FF if available with count >= MIN_COUNT, else SI, else None."""
    for pt in ('FF', 'SI'):
        r = arsenal.get(pt)
        if r and (r.get('count') or 0) >= MIN_COUNT:
            return pt
    return None


def build_feature_vector(arsenal, anchor, other_pitches, target_pt):
    """Concatenate [biomech (from anchor row), anchor metrics, other_pitches metrics in order, target metrics]."""
    anchor_row = arsenal.get(anchor)
    if not anchor_row:
        return None
    biomech = extract_vec(anchor_row, BIOMECH)
    if biomech is None:
        return None
    anchor_m = extract_vec(anchor_row, PITCH_METRICS)
    if anchor_m is None:
        return None
    others_m = []
    for op in other_pitches:
        r = arsenal.get(op)
        if not r or (r.get('count') or 0) < MIN_COUNT:
            return None
        m = extract_vec(r, PITCH_METRICS)
        if m is None:
            return None
        others_m.append(m)
    target_row = arsenal.get(target_pt)
    if not target_row or (target_row.get('count') or 0) < MIN_COUNT:
        return None
    target_m = extract_vec(target_row, TARGET_METRICS)
    if target_m is None:
        return None

    parts = [biomech, anchor_m]
    for m in others_m:
        parts.append(m)
    parts.append(target_m)
    return np.concatenate(parts)


def assemble_training_matrix(arsenals, anchor, other_pitches, target_pt, hand, exclude_key=None):
    """Build the (n, d) training matrix for the specified conditioning scheme.

    Returns:
        X: (n, d) matrix of feature vectors
        train_keys: list of (pitcher, team) tuples in row order
    """
    X = []
    keys = []
    for key, arsenal in arsenals.items():
        if key == exclude_key:
            continue
        if arsenal.get('_throws') != hand:
            continue
        vec = build_feature_vector(arsenal, anchor, other_pitches, target_pt)
        if vec is None:
            continue
        X.append(vec)
        keys.append(key)
    if not X:
        return np.empty((0, 0)), []
    return np.vstack(X), keys


def fit_mvn(X):
    """Return (mean, covariance) of X. Uses ddof=1."""
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    # If X has a single row, cov is 0-d; make it 2-d
    cov = np.atleast_2d(cov)
    # Ridge regularizer for numerical stability
    d = cov.shape[0]
    cov = cov + 1e-6 * np.eye(d)
    return mu, cov


# ────────────────────────────────────────────────────────────────────────
# Prediction (one tier, one target pitch)
# ────────────────────────────────────────────────────────────────────────

def predict(arsenals, target_key, anchor, other_pitches, target_pt):
    """Fit MVN and condition. Returns dict with keys mu_b, cov_b, n_train, comps, x_a.
    Returns None if training sample is insufficient.
    """
    target_arsenal = arsenals[target_key]
    hand = target_arsenal.get('_throws')
    X, train_keys = assemble_training_matrix(
        arsenals, anchor, other_pitches, target_pt, hand,
        exclude_key=target_key,
    )
    n_train = X.shape[0]
    d_a = len(BIOMECH) + len(PITCH_METRICS) * (1 + len(other_pitches))
    d_b = len(TARGET_METRICS)
    # Need at least d_a + d_b rows for invertible cov, with a safety margin
    min_needed = max(d_a + d_b + 10, TIER2_MIN_TRAIN if other_pitches else 30)
    if n_train < min_needed:
        return {'insufficient': True, 'n_train': n_train, 'min_needed': min_needed,
                'other_pitches': other_pitches, 'anchor': anchor, 'target': target_pt}

    # Build the target pitcher's x_a (conditioning block from arsenal)
    anchor_row = target_arsenal.get(anchor)
    if not anchor_row:
        return None
    biomech = extract_vec(anchor_row, BIOMECH)
    anchor_m = extract_vec(anchor_row, PITCH_METRICS)
    if biomech is None or anchor_m is None:
        return None
    x_a_parts = [biomech, anchor_m]
    for op in other_pitches:
        op_row = target_arsenal.get(op)
        if not op_row or (op_row.get('count') or 0) < MIN_COUNT:
            return None
        m = extract_vec(op_row, PITCH_METRICS)
        if m is None:
            return None
        x_a_parts.append(m)
    x_a = np.concatenate(x_a_parts)

    mu, cov = fit_mvn(X)
    mu_b, cov_b = mvn_conditional(mu, cov, x_a, d_a)

    # Mahalanobis comps in the conditioning space
    X_a_only = X[:, :d_a]
    _, cov_aa_only = mu[:d_a], cov[:d_a, :d_a]
    dists = mahalanobis_dist_batch(X_a_only, x_a, cov_aa_only)
    order = np.argsort(dists)
    comps = []
    for idx in order[:N_COMPS]:
        ck = train_keys[idx]
        carsenal = arsenals[ck]
        target_row = carsenal[target_pt]
        tm = extract_vec(target_row, TARGET_METRICS)
        comps.append({
            'pitcher': ck[0],
            'team': ck[1],
            'distance': float(dists[idx]),
            'metrics': tm.tolist() if tm is not None else None,
        })

    return {
        'mu_b': mu_b,
        'cov_b': cov_b,
        'n_train': n_train,
        'comps': comps,
        'x_a': x_a,
        'd_a': d_a,
    }


def get_candidate_other_pitches(arsenal, anchor, target_pt):
    """List of pitch types in candidate's arsenal (excluding anchor and target)
    with sufficient sample, ordered by usage descending. Used by fallback Tier 2."""
    others = []
    for pt, row in arsenal.items():
        if pt.startswith('_') or pt == anchor or pt == target_pt:
            continue
        if not isinstance(row, dict):
            continue
        if (row.get('count') or 0) >= MIN_COUNT:
            others.append((pt, row.get('count') or 0))
    others.sort(key=lambda x: -x[1])
    return [pt for pt, _ in others]


def predict_tier2_with_fallback(arsenals, target_key, anchor, target_pt):
    """Generalized Tier 2: auto-detect candidate's arsenal and try the largest
    conditioning set that yields sufficient training sample. Drops the
    least-used pitches first if the joint training pool is too thin.

    Returns (result, conditioning_used). result may be None or have
    {'insufficient': True} if even a single-other-pitch attempt fails.
    """
    candidate = arsenals[target_key]
    others = get_candidate_other_pitches(candidate, anchor, target_pt)
    last_result = None
    for n_keep in range(len(others), -1, -1):
        attempt = others[:n_keep]
        result = predict(arsenals, target_key, anchor, attempt, target_pt)
        last_result = result
        if result and not result.get('insufficient'):
            return result, attempt
    return last_result, []


def compute_calibration(arsenals, target_key, anchor, shape_pop_stats, outcome_lg):
    """Self-calibration: project each existing pitch in the candidate's arsenal
    as if it were new (excluding self from training), compare projected shape
    and outcomes to the pitcher's actual values. Gives the article a defensible
    "model is within X% on his existing pitches" line.
    """
    candidate = arsenals[target_key]
    hand = candidate.get('_throws')
    rows = []
    for pt, row in candidate.items():
        if pt.startswith('_') or pt == anchor or not isinstance(row, dict):
            continue
        if (row.get('count') or 0) < MIN_COUNT:
            continue
        result, others_used = predict_tier2_with_fallback(arsenals, target_key, anchor, pt)
        # If even Tier 2 fallback all the way to anchor-only fails, try Tier 1
        if result is None or result.get('insufficient'):
            result = predict(arsenals, target_key, anchor, [], pt)
            others_used = []
        if result is None or result.get('insufficient'):
            continue
        mu_b = result['mu_b']
        proj_shape = {
            'velocity': mu_b[TARGET_METRICS.index('velocity')],
            'indVertBrk': mu_b[TARGET_METRICS.index('indVertBrk')],
            'horzBrk': mu_b[TARGET_METRICS.index('horzBrk')],
        }
        actual_shape = {m: row.get(m) for m in DISPLAY_METRICS}
        proj_outcomes = project_outcomes_from_shape(
            arsenals, pt, hand, mu_b, target_key, shape_pop_stats, outcome_lg,
        )
        actual_outcomes = {okey: _outcome_getter(okey)(row) for okey, _, _, _ in OUTCOME_METRICS}
        rows.append({
            'pitch': pt,
            'sample': row.get('count'),
            'actual_shape': actual_shape,
            'projected_shape': proj_shape,
            'actual_outcomes': actual_outcomes,
            'projected_outcomes': proj_outcomes,
            'others_used': others_used,
        })
    return rows


# ────────────────────────────────────────────────────────────────────────
# Pretty printing
# ────────────────────────────────────────────────────────────────────────

def fmt_pitch(row):
    v = row.get('velocity')
    iv = row.get('indVertBrk')
    hb = row.get('horzBrk')
    n = row.get('count')
    return (f"{v:5.1f} mph  {iv:5.1f}\" IVB  {hb:5.1f}\" HB  (n={n})")


def project_outcomes_from_shape(arsenals, target_pt, hand, predicted_shape,
                                  exclude_key, shape_pop_stats, outcome_lg):
    """Shape-comp regression for outcome projection.

    For each pitcher (excluding self) throwing target_pt with sufficient
    sample, weight their outcome rates by Mahalanobis distance from their
    actual pitch shape to predicted_shape, using POPULATION-level shape
    covariance (so the weighting is in natural units of pitcher-to-pitcher
    pitch variation). Then apply empirical-Bayes shrinkage toward the
    (pitch_type, hand) league mean before averaging.

    Returns dict per metric: {mean, sd, min, max, n_comps, total_pitches, lg}
    or None if no qualifying comps.
    """
    pop_key = (target_pt, hand)
    if pop_key not in shape_pop_stats:
        return {okey: None for okey, _, _, _ in OUTCOME_METRICS}
    _, pop_cov = shape_pop_stats[pop_key]
    lg = outcome_lg.get(pop_key, {})

    # Predicted shape in display dims (velo, IVB, HB)
    pred = np.array([predicted_shape[TARGET_METRICS.index(m)] for m in DISPLAY_METRICS])

    contributors = {okey: [] for okey, _, _, _ in OUTCOME_METRICS}
    for ck, arsenal in arsenals.items():
        if ck == exclude_key:
            continue
        if arsenal.get('_throws') != hand:
            continue
        row = arsenal.get(target_pt)
        if not row:
            continue
        n = row.get('count') or 0
        if n < OUTCOME_MIN_PITCHES:
            continue
        v = row.get('velocity'); iv = row.get('indVertBrk'); hb = row.get('horzBrk')
        if v is None or iv is None or hb is None:
            continue
        x = np.array([float(v), float(iv), float(hb)])
        delta = x - pred
        try:
            d2 = float(delta @ np.linalg.solve(pop_cov, delta))
        except np.linalg.LinAlgError:
            continue
        d = max(d2, 0.0) ** 0.5
        # Gaussian distance weight (sharp falloff) × sqrt(sample size).
        # Far-away-shape comps barely contribute, so the projection reflects
        # pitchers whose actual pitch closely matches the predicted shape.
        weight = np.exp(-0.5 * (d / OUTCOME_DIST_SIGMA) ** 2) * (n ** 0.5)
        if weight < 1e-9:
            continue
        for okey, _, _, _ in OUTCOME_METRICS:
            getter = _outcome_getter(okey)
            ov = getter(row)
            if ov is None:
                continue
            # Shrink the comp's rate toward the league mean for that pitch type
            ov_shrunk = float(ov)
            lg_val = lg.get(okey)
            if lg_val is not None and OUTCOME_SHRINK_K > 0:
                ov_shrunk = (n / (n + OUTCOME_SHRINK_K)) * float(ov) + \
                            (OUTCOME_SHRINK_K / (n + OUTCOME_SHRINK_K)) * float(lg_val)
            contributors[okey].append({
                'val': ov_shrunk, 'raw': float(ov), 'weight': weight, 'n': n,
            })

    out = {}
    for okey, _, _, _ in OUTCOME_METRICS:
        items = contributors[okey]
        if not items:
            out[okey] = None
            continue
        weights = np.array([it['weight'] for it in items])
        vals = np.array([it['val'] for it in items])
        wmean = float(np.sum(vals * weights) / np.sum(weights))
        wvar = float(np.sum(weights * (vals - wmean) ** 2) / np.sum(weights))
        wsd = wvar ** 0.5
        out[okey] = {
            'mean': wmean,
            'sd': wsd,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n_comps': len(items),
            'total_pitches': int(sum(it['n'] for it in items)),
            'lg': lg.get(okey),
        }
    return out


def fmt_prediction(mu_b, cov_b):
    sigmas = np.sqrt(np.diag(cov_b))
    v = mu_b[TARGET_METRICS.index('velocity')]
    iv = mu_b[TARGET_METRICS.index('indVertBrk')]
    hb = mu_b[TARGET_METRICS.index('horzBrk')]
    sv = sigmas[TARGET_METRICS.index('velocity')]
    siv = sigmas[TARGET_METRICS.index('indVertBrk')]
    shb = sigmas[TARGET_METRICS.index('horzBrk')]
    return (f"{v:5.1f} +/- {sv:3.1f} mph   "
            f"{iv:5.1f} +/- {siv:3.1f}\" IVB   "
            f"{hb:5.1f} +/- {shb:3.1f}\" HB")


def fmt_outcome_value(op, fmt):
    """Format an outcome dict {mean, sd, lg} according to its display kind."""
    if op is None:
        return '—'
    if fmt == 'pct':
        mean = op['mean'] * 100
        sd = op['sd'] * 100
        s = f"{mean:5.1f}% \u00b1{sd:3.1f}"
        if op.get('lg') is not None:
            delta = mean - op['lg'] * 100
            sign = '+' if delta >= 0 else ''
            s += f" ({sign}{delta:.1f} vs lg)"
        return s
    if fmt == 'woba':
        mean = op['mean']
        sd = op['sd']
        s = f".{int(round(mean*1000)):03d} \u00b1{int(round(sd*1000)):03d}"
        if op.get('lg') is not None:
            delta = mean - op['lg']
            sign = '+' if delta >= 0 else ''
            s += f" ({sign}{int(round(delta*1000))} vs lg)"
        return s
    return str(op.get('mean'))


def print_report(target_key, arsenals, anchor, targets, results,
                  shape_pop_stats, outcome_lg, calibration_rows=None):
    name, team = target_key
    arsenal = arsenals[target_key]
    hand = arsenal.get('_throws')
    anchor_row = arsenal.get(anchor)
    arm_angle = anchor_row.get('armAngle')
    ext = anchor_row.get('extension')

    print()
    print(f"{name}, {team} ({hand}HP)")
    print(f"  arm angle {arm_angle:.1f} deg, extension {ext:.1f} ft   "
          f"[anchor: {anchor}]")
    print()
    print(f"  Current arsenal (2026):")
    for pt in sorted(arsenal.keys(), key=lambda k: -(arsenal[k].get('count') or 0)
                     if isinstance(arsenal[k], dict) else 1):
        if pt.startswith('_'):
            continue
        r = arsenal[pt]
        if not isinstance(r, dict) or (r.get('count') or 0) < MIN_DISPLAY:
            continue
        print(f"    {pt:3s}  {fmt_pitch(r)}")

    print()
    print(f"  Predicted pitches (full-arsenal conditioning, fall back to anchor-only when needed):")
    for t in targets:
        res = results.get(t)
        if res is None:
            print(f"    {t:3s}  [skipped: target is the anchor pitch]")
            continue
        if res.get('insufficient'):
            combo = ' + '.join([res['anchor']] + res.get('other_pitches', []) + [res['target']])
            print(f"    {t:3s}  [skipped: only {res['n_train']} pitchers throw "
                  f"{combo} (need {res['min_needed']})]")
            continue
        cond = res.get('_t2_conditioning', [])
        cond_suffix = f"   [cond: {anchor}+{','.join(cond) if cond else 'anchor only'}]"
        line = (f"    {t:3s}  {fmt_prediction(res['mu_b'], res['cov_b'])}"
                f"   [n_train={res['n_train']}]{cond_suffix}")
        print(line)
        # Projected outcomes via shape-comp regression (Mahalanobis distance
        # in (velo, IVB, HB) space using population covariance, then EB shrunk)
        outcomes_proj = project_outcomes_from_shape(
            arsenals, t, hand, res['mu_b'], target_key,
            shape_pop_stats, outcome_lg,
        )
        sample_summary = ''
        for okey, _, _, _ in OUTCOME_METRICS:
            op = outcomes_proj.get(okey)
            if op is not None:
                sample_summary = (f"  [{op['n_comps']} comps, {op['total_pitches']} pitches]")
                break
        print(f"         projected outcomes:{sample_summary}")
        for okey, olabel, fmt, _ in OUTCOME_METRICS:
            op = outcomes_proj.get(okey)
            print(f"           {olabel:11s} {fmt_outcome_value(op, fmt)}")

    has_any_comps = any(r and not r.get('insufficient') for r in results.values())
    if has_any_comps:
        print(f"\n    Top {N_COMPS} Mahalanobis comps per target:")
        for t in targets:
            res = results.get(t)
            if res is None or res.get('insufficient'):
                continue
            print(f"      {t}:")
            for c in res['comps']:
                if c['metrics'] is None:
                    continue
                cv = c['metrics'][TARGET_METRICS.index('velocity')]
                civ = c['metrics'][TARGET_METRICS.index('indVertBrk')]
                chb = c['metrics'][TARGET_METRICS.index('horzBrk')]
                print(f"        d={c['distance']:4.2f}  {c['pitcher']:<28s} ({c['team']})  "
                      f"{cv:5.1f} mph  {civ:5.1f}\" IVB  {chb:5.1f}\" HB")

    # ── Self-calibration on existing pitches ────────────────────────────
    if calibration_rows:
        print()
        print(f"  Self-calibration (each existing pitch projected as if new):")
        print(f"    {'pitch':5s} {'n':>4s}  "
              f"{'velo (proj/act/Δ)':>22s}  {'IVB (proj/act/Δ)':>22s}  {'HB (proj/act/Δ)':>22s}")
        for cr in calibration_rows:
            pt = cr['pitch']
            ps = cr['projected_shape']; ash = cr['actual_shape']; n = cr['sample']
            def _trio(p_val, a_val, unit):
                if p_val is None or a_val is None:
                    return f"{'—':>22s}"
                d = p_val - a_val
                return f"{p_val:5.1f}/{a_val:5.1f}/{d:+5.1f} {unit}"
            v_str = _trio(ps['velocity'], ash['velocity'], 'mph')
            iv_str = _trio(ps['indVertBrk'], ash['indVertBrk'], '"  ')
            hb_str = _trio(ps['horzBrk'], ash['horzBrk'], '"  ')
            print(f"    {pt:5s} {n:4d}  {v_str:>22s}  {iv_str:>22s}  {hb_str:>22s}")
            # Outcome calibration row
            out_parts = []
            for okey, olabel, fmt, _ in OUTCOME_METRICS:
                op = cr['projected_outcomes'].get(okey)
                act = cr['actual_outcomes'].get(okey)
                if op is None or act is None:
                    out_parts.append(f"{olabel} —")
                    continue
                if fmt == 'pct':
                    proj_v = op['mean'] * 100
                    act_v = act * 100
                    out_parts.append(f"{olabel} {proj_v:.1f}/{act_v:.1f} ({proj_v-act_v:+.1f})")
                else:  # woba
                    out_parts.append(
                        f"{olabel} .{int(round(op['mean']*1000)):03d}/"
                        f".{int(round(act*1000)):03d} ({(op['mean']-act)*1000:+.0f})"
                    )
            print(f"          outcomes (proj/act/Δ):  {'   '.join(out_parts)}")


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def plot_cov_ellipse(ax, center, cov2, n_sigma=1, **kwargs):
    """Plot n-sigma ellipse for a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(cov2)  # ascending eigenvalues
    major_vec = vecs[:, 1]
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    width = 2 * n_sigma * np.sqrt(max(vals[1], 0))
    height = 2 * n_sigma * np.sqrt(max(vals[0], 0))
    e = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(e)


def _style_plot_axes(ax):
    """Apply the dark Movement-Profile look to an axes."""
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.axhline(0, color=TEXT_COLOR, ls='--', lw=0.7, alpha=0.45, zorder=1)
    ax.axvline(0, color=TEXT_COLOR, ls='--', lw=0.7, alpha=0.45, zorder=1)
    ax.grid(True, color=GRID_COLOR, lw=0.5, alpha=0.7, zorder=0)
    ax.set_xticks(range(-25, 26, 5))
    ax.set_yticks(range(-25, 26, 5))
    for sp in ax.spines.values():
        sp.set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel('Horizontal Break (in)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Induced Vertical Break (in)', color=TEXT_COLOR, fontsize=10)


def _plot_arsenal_panel(ax, target_arsenal, targets, results, panel_label):
    """Plot current pitches as solid dots, predicted pitches as hatched ellipses."""
    _style_plot_axes(ax)

    # Current arsenal: solid colored dots with pitch-type label
    for pt, row in target_arsenal.items():
        if pt.startswith('_') or not isinstance(row, dict):
            continue
        if (row.get('count') or 0) < MIN_DISPLAY:
            continue
        ivb = row.get('indVertBrk')
        hb = row.get('horzBrk')
        if ivb is None or hb is None:
            continue
        color = PITCH_COLORS.get(pt, '#999')
        ax.scatter(hb, ivb, s=240, color=color, edgecolor='#000',
                   linewidths=0.6, zorder=5)
        ax.text(hb, ivb, pt, fontsize=8, fontweight='bold', ha='center',
                va='center', color='#000' if pt in ('SL', 'SI', 'FC') else '#ffffff',
                zorder=6)

    # Predictions: hatched ellipses plus small "+T" marker at center
    for t in targets:
        res = results.get(t)
        if res is None or res.get('insufficient'):
            continue
        mu = res['mu_b']
        cov = res['cov_b']
        ivb_hat = mu[TARGET_METRICS.index('indVertBrk')]
        hb_hat = mu[TARGET_METRICS.index('horzBrk')]
        idx_ivb = TARGET_METRICS.index('indVertBrk')
        idx_hb = TARGET_METRICS.index('horzBrk')
        cov2 = np.array([
            [cov[idx_hb, idx_hb], cov[idx_hb, idx_ivb]],
            [cov[idx_ivb, idx_hb], cov[idx_ivb, idx_ivb]],
        ])
        color = PITCH_COLORS.get(t, '#999')
        plot_cov_ellipse(
            ax, (hb_hat, ivb_hat), cov2, n_sigma=1,
            facecolor=color, alpha=0.25, hatch='///', edgecolor=color, lw=1.2,
            zorder=4,
        )
        ax.text(hb_hat, ivb_hat, t, fontsize=8, fontweight='bold', ha='center',
                va='center', color='#000' if t in ('SL', 'SI', 'FC') else '#ffffff',
                zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=0.9))

    ax.set_title(panel_label, color=TEXT_COLOR, fontsize=11, fontweight='bold',
                 loc='left', pad=8)


def plot_predictions(target_key, arsenal, targets, results, out_path,
                      arsenals=None, shape_pop_stats=None, outcome_lg=None):
    """Single-panel movement plot + shape table + projected-outcomes table.
    Outcomes table renders only when arsenals/shape_pop_stats/outcome_lg are
    provided (so the function still works for plot-only callers).
    """
    fig = plt.figure(figsize=(13, 13), facecolor=BG_COLOR)
    gs = fig.add_gridspec(
        nrows=3, ncols=1, height_ratios=[3, 1.1, 1.4],
        hspace=0.32,
        left=0.07, right=0.95, top=0.93, bottom=0.04,
    )
    ax_panel = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_outcomes = fig.add_subplot(gs[2])

    _plot_arsenal_panel(ax_panel, arsenal, targets, results,
                        'Predicted pitches (full-arsenal conditioning, anchor-only fallback)')

    if not any(r and not r.get('insufficient') for r in results.values()):
        ax_panel.text(0, 24, '(no predictions: insufficient training sample)',
                       fontsize=9, color='#94a3b8', ha='center', va='top')

    # Shape table: current arsenal first (sorted by count desc), then predictions
    table_rows = []
    current_sorted = sorted(
        ((pt, r) for pt, r in arsenal.items()
         if not pt.startswith('_') and isinstance(r, dict)
         and (r.get('count') or 0) >= MIN_DISPLAY),
        key=lambda kv: -(kv[1].get('count') or 0),
    )
    for pt, r in current_sorted:
        table_rows.append({
            'pitch': pt, 'kind': 'current',
            'mph': r.get('velocity'),
            'ivb': r.get('indVertBrk'),
            'hb': r.get('horzBrk'),
        })
    for t in targets:
        res = results.get(t)
        if res is None or res.get('insufficient'):
            continue
        mu = res['mu_b']
        table_rows.append({
            'pitch': t, 'kind': 'predicted',
            'tier_suffix': '',
            'mph': mu[TARGET_METRICS.index('velocity')],
            'ivb': mu[TARGET_METRICS.index('indVertBrk')],
            'hb': mu[TARGET_METRICS.index('horzBrk')],
        })

    _draw_arsenal_table(ax_table, table_rows)

    # Outcomes table (only if we have the data to compute it)
    if arsenals is not None and shape_pop_stats is not None and outcome_lg is not None:
        hand = arsenal.get('_throws')
        outcome_rows = []
        for t in targets:
            res = results.get(t)
            if res is None or res.get('insufficient'):
                continue
            outcomes_proj = project_outcomes_from_shape(
                arsenals, t, hand, res['mu_b'], target_key,
                shape_pop_stats, outcome_lg,
            )
            outcome_rows.append({'pitch': t, 'outcomes': outcomes_proj})
        _draw_outcomes_table(ax_outcomes, outcome_rows)
    else:
        ax_outcomes.set_facecolor(BG_COLOR)
        ax_outcomes.axis('off')

    name, team = target_key
    hand = arsenal.get('_throws')
    anchor_row = next((r for pt, r in arsenal.items()
                       if not pt.startswith('_') and isinstance(r, dict)
                       and (r.get('count') or 0) >= MIN_COUNT
                       and pt in ('FF', 'SI')), None)
    arm_angle = anchor_row.get('armAngle') if anchor_row else None

    title = f"{name}, {team}   {hand}HP"
    subtitle = "* = predicted new pitch   |   hatched region = 1-sigma uncertainty"
    if arm_angle is not None:
        subtitle = f"Arm Angle = {arm_angle:.1f}°   |   " + subtitle
    fig.text(0.06, 0.965, 'MOVEMENT PROFILE', color='#67e8f9', fontsize=13,
             fontweight='bold', ha='left', va='center')
    fig.text(0.06, 0.94, title, color=TEXT_COLOR, fontsize=14,
             fontweight='bold', ha='left', va='center')
    fig.text(0.97, 0.965, subtitle, color='#94a3b8', fontsize=9,
             ha='right', va='center')

    fig.savefig(out_path, dpi=140, facecolor=BG_COLOR)
    plt.close(fig)


def _draw_outcomes_table(ax, rows):
    """Render the projected-outcomes table (one row per predicted pitch).
    Columns: PITCH | Whiff% | Chase% | GB% | Hard-Hit% | xwOBAcon
    Each cell shows mean and (Δ vs lg) on a second sub-line.
    """
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    if not rows:
        ax.text(0.5, 0.5, '(no projected outcomes)',
                color='#94a3b8', fontsize=10, ha='center', va='center',
                transform=ax.transAxes)
        return

    # Section title
    ax.text(0.0, 1.05, 'PROJECTED OUTCOMES', color='#67e8f9',
            fontsize=12, fontweight='bold', ha='left', va='center',
            transform=ax.transAxes)

    headers = ['PITCH'] + [olabel for _, olabel, _ in OUTCOME_METRICS]
    n_cols = len(headers)
    # Even spacing across columns; pitch column slightly narrower
    col_xs = [0.08] + [0.20 + i * (0.78 / max(len(OUTCOME_METRICS) - 1, 1))
                       for i in range(len(OUTCOME_METRICS))]

    y_header = 0.92
    for x, h in zip(col_xs, headers):
        ax.text(x, y_header, h, color='#67e8f9', fontsize=10, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
    ax.plot([0.04, 0.96], [0.83, 0.83], color=GRID_COLOR, lw=0.7,
            transform=ax.transAxes, clip_on=False)

    n_rows = len(rows)
    row_height = 0.78 / max(n_rows, 1)
    for i, r in enumerate(rows):
        y = 0.79 - (i + 0.5) * row_height
        pt = r['pitch']
        color = PITCH_COLORS.get(pt, '#999')
        text_color = '#000000' if pt in ('SL', 'SI', 'FC') else '#ffffff'
        ax.text(col_xs[0], y, pt + '*', color=text_color,
                fontsize=10, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.35', fc=color, ec=color, lw=0))

        for x, (okey, _, fmt) in zip(col_xs[1:], OUTCOME_METRICS):
            op = (r['outcomes'] or {}).get(okey)
            if op is None:
                ax.text(x, y, '—', color=TEXT_COLOR, fontsize=10,
                        ha='center', va='center', transform=ax.transAxes)
                continue
            if fmt == 'pct':
                main_str = f"{op['mean']*100:.1f}%"
                if op.get('lg') is not None:
                    delta = op['mean']*100 - op['lg']*100
                    sub_str = f"{'+' if delta>=0 else ''}{delta:.1f} vs lg"
                else:
                    sub_str = ''
            else:  # woba
                main_str = f".{int(round(op['mean']*1000)):03d}"
                if op.get('lg') is not None:
                    delta = (op['mean'] - op['lg']) * 1000
                    sub_str = f"{'+' if delta>=0 else ''}{int(round(delta))} vs lg"
                else:
                    sub_str = ''
            # Color the delta sub-line by sign convention. For Hard-Hit% and
            # xwOBAcon, lower is better (pitcher); flip the green/red mapping.
            higher_better = okey not in ('hardHitPct', 'xwOBAcon')
            ax.text(x, y + 0.02, main_str, color=TEXT_COLOR, fontsize=10,
                    fontweight='bold', ha='center', va='center',
                    transform=ax.transAxes)
            if sub_str:
                if delta == 0:
                    sub_color = '#94a3b8'
                else:
                    is_good = (delta > 0) if higher_better else (delta < 0)
                    sub_color = '#22c55e' if is_good else '#ef4444'
                ax.text(x, y - 0.04, sub_str, color=sub_color, fontsize=8,
                        ha='center', va='center', transform=ax.transAxes)


def _draw_arsenal_table(ax, rows):
    """Render the metrics table (PITCH | MPH | IVB | HB) with colored pill
    badges for pitch labels and tier suffixes on predicted rows."""
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    if not rows:
        return

    headers = ['PITCH', 'MPH', 'IVB', 'HB']
    col_x = [0.12, 0.38, 0.60, 0.82]

    y_header = 0.95
    for x, h in zip(col_x, headers):
        ax.text(x, y_header, h, color='#67e8f9', fontsize=11, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.86, 0.86], color=GRID_COLOR, lw=0.7,
            transform=ax.transAxes, clip_on=False)

    n_rows = len(rows)
    row_height = 0.82 / max(n_rows, 1)
    for i, r in enumerate(rows):
        y = 0.82 - (i + 0.5) * row_height
        pt = r['pitch']
        label = pt + (r.get('tier_suffix') or '') + ('*' if r['kind'] == 'predicted' else '')
        color = PITCH_COLORS.get(pt, '#999')
        text_color = '#000000' if pt in ('SL', 'SI', 'FC') else '#ffffff'
        ax.text(col_x[0], y, label, color=text_color, fontsize=10, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.35', fc=color, ec=color, lw=0))
        # Italicize predicted-row values to visually distinguish them from
        # the candidate's actual measured pitches in the same table.
        is_predicted = (r['kind'] == 'predicted')
        val_style = 'italic' if is_predicted else 'normal'
        for x, key in zip(col_x[1:], ['mph', 'ivb', 'hb']):
            val = r[key]
            if val is None:
                txt = '—'
            elif key == 'mph':
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.1f}\""
            ax.text(x, y, txt, color=TEXT_COLOR, fontsize=10,
                    fontstyle=val_style,
                    ha='center', va='center', transform=ax.transAxes)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    # ── Settings (edit these directly or override via command line) ──
    pitcher  = "Granillo, Andre"   # Pitcher name, "Last, First" format
    team     = "WSH"                 # Team abbreviation
    pitches  = []          # Target pitch types; empty = auto (all common pitches he doesn't already throw)

    no_plot   = False                # True to skip plot generation
    plot_only = False                # True to write plot but skip printed text
    include_existing = False         # True to project pitches the pitcher already throws
    no_calibrate = False             # True to skip the self-calibration table

    # ── CLI overrides (optional — values above are used if no args passed) ──
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('pitcher', nargs='?', default=None,
                        help='Pitcher name, "Last, First" format')
    parser.add_argument('team', nargs='?', default=None,
                        help='Team abbreviation, e.g. WSH')
    parser.add_argument('pitches', nargs='*',
                        help='Target pitch types to predict. Omit for auto (all common pitches he does not already throw).')
    parser.add_argument('--no-plot', action='store_true', default=None,
                        help='Skip plot generation')
    parser.add_argument('--plot-only', action='store_true', default=None,
                        help='Write plot but skip printed text')
    parser.add_argument('--include-existing', action='store_true', default=None,
                        help='Also project pitches the pitcher already throws (default: skip)')
    parser.add_argument('--no-calibrate', action='store_true', default=None,
                        help='Skip the self-calibration table on existing pitches')
    args = parser.parse_args()

    if args.pitcher is not None: pitcher = args.pitcher
    if args.team is not None: team = args.team
    if args.pitches: pitches = args.pitches
    if args.no_plot: no_plot = True
    if args.plot_only: plot_only = True
    if args.include_existing: include_existing = True
    if args.no_calibrate: no_calibrate = True

    arsenals = load_arsenals()
    target_key = (pitcher, team)
    if target_key not in arsenals:
        print(f"Error: pitcher {pitcher} on team {team} not found.", file=sys.stderr)
        # Suggest close matches
        close = [k for k in arsenals if pitcher.lower() in k[0].lower() and k[1] == team]
        if close:
            print("  Closest name matches on this team:", file=sys.stderr)
            for k in close[:5]:
                print(f"    '{k[0]}'", file=sys.stderr)
        sys.exit(1)

    target_arsenal = arsenals[target_key]
    anchor = pick_anchor(target_arsenal)
    if not anchor:
        print(f"Error: {pitcher} has no FF or SI with >={MIN_COUNT} pitches; "
              "cannot anchor prediction.", file=sys.stderr)
        sys.exit(1)

    # Determine other existing pitches (for Tier 2)
    existing = [
        pt for pt, r in target_arsenal.items()
        if (not pt.startswith('_'))
        and isinstance(r, dict)
        and (r.get('count') or 0) >= MIN_DISPLAY
    ]
    other_existing_full = [pt for pt in existing if pt != anchor]

    # Auto-build target list when not specified: common pitch types the pitcher
    # does not already throw. Skip-existing is the default; --include-existing overrides.
    COMMON_TARGETS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS']
    if not pitches:
        pitches = [pt for pt in COMMON_TARGETS if pt not in existing and pt != anchor]
    elif not include_existing:
        # Filter user-specified pitches to drop ones already in arsenal
        skipped = [t for t in pitches if t in existing]
        pitches = [t for t in pitches if t not in existing]
        if skipped:
            print(f"Note: skipping {', '.join(skipped)} (already in arsenal); "
                  f"use --include-existing to override.", file=sys.stderr)

    if not pitches:
        print(f"No target pitches to project (arsenal already covers common types).",
              file=sys.stderr)
        sys.exit(0)

    # Pre-compute population shape stats and league outcome averages by (pt, hand).
    # Used by the new shape-comp regression (replaces the biomech-comp average path).
    shape_pop_stats, outcome_lg = compute_pitch_population_stats(arsenals)

    # Single-tier predictions: full-arsenal conditioning with anchor-only fallback.
    results = {}
    for t in pitches:
        if t == anchor:
            print(f"Note: {t} is the anchor pitch; skipping prediction for it.",
                  file=sys.stderr)
            results[t] = None
            continue
        res, conditioning_used = predict_tier2_with_fallback(arsenals, target_key, anchor, t)
        if res is not None:
            res['_t2_conditioning'] = conditioning_used
        results[t] = res

    # Self-calibration on existing pitches (skip if --no-calibrate or no existing pitches)
    calibration_rows = None
    if not no_calibrate and other_existing_full:
        calibration_rows = compute_calibration(
            arsenals, target_key, anchor, shape_pop_stats, outcome_lg,
        )

    if not plot_only:
        print_report(target_key, arsenals, anchor, pitches, results,
                     shape_pop_stats, outcome_lg,
                     calibration_rows=calibration_rows)

    if not no_plot:
        name_slug = pitcher.replace(', ', '_').replace(' ', '_').replace("'", '')
        pitches_slug = '_'.join(pitches)
        out_path = OUTPUT_DIR / f"{name_slug}_{team}_{pitches_slug}.png"
        plot_predictions(target_key, target_arsenal, pitches, results, out_path,
                         arsenals=arsenals,
                         shape_pop_stats=shape_pop_stats,
                         outcome_lg=outcome_lg)
        if not plot_only:
            print(f"\n  plot: {out_path}")


if __name__ == '__main__':
    main()
