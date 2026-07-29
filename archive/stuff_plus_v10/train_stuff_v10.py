#!/usr/bin/env python3
"""
Stuff+ v10 — Training, Scoring & Comparison Pipeline
=====================================================
v10 improvements over v9:
  1. Movement differentials from FB (ivb_diff, hb_diff) — pitcher-specific deception
  2. Pitch-type-specific target weights — each pitch weighted by its primary job
  3. Count-aware training with neutral-count scoring — removes count confound
  4. Called strike probability component — captures "freeze" ability
  5. Removed spin efficiency (redundant with raw features)

Usage:
  python train_stuff_v10.py                    # Train + score (v10)
  python train_stuff_v10.py --compare          # Train both v9 and v10, compare metrics
  python train_stuff_v10.py --score-only       # Score using saved v10 artifacts
  python train_stuff_v10.py --csv input.csv    # Score a CSV file
"""

import os
import sys
import json
import math
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'ST Leaderboard')
DATA_DIR = os.path.join(LEADERBOARD_DIR, 'data')

SUPPORTED_PT = ['FF', 'SI', 'SL', 'CH', 'ST', 'FC', 'CU', 'FS', 'SV']
FASTBALL_TYPES = {'FF', 'SI', 'FC'}

# ==========================================================================
# v10: PER-PITCH-TYPE TARGET WEIGHTS
# ==========================================================================
# (whiff, ground_ball, contact_quality, called_strike)
# Each pitch type weighted by its primary job.
PT_WEIGHTS = {
    'FF': (0.45, 0.20, 0.20, 0.15),  # Balanced + freeze (high ride freezes batters)
    'SI': (0.15, 0.50, 0.25, 0.10),  # Primary: ground balls
    'SL': (0.55, 0.15, 0.20, 0.10),  # Primary: whiffs
    'ST': (0.60, 0.10, 0.20, 0.10),  # Primary: whiffs (sweeper)
    'CH': (0.40, 0.25, 0.25, 0.10),  # Balanced with GB component
    'FC': (0.35, 0.25, 0.25, 0.15),  # Versatile: whiffs + GBs + freeze (paints corners)
    'CU': (0.55, 0.15, 0.20, 0.10),  # Primary: whiffs
    'FS': (0.30, 0.35, 0.25, 0.10),  # Primary: ground balls (splitter)
    'SV': (0.60, 0.10, 0.20, 0.10),  # Primary: whiffs
}

# v10b: PT weights without CS component, tuned after v10a results
# CU/FS/FC get more conservative shifts; no CS model, no count features
PT_WEIGHTS_B = {
    #           (whiff,  gb,    contact, cs=0)
    'FF': (0.50, 0.20, 0.30, 0.00),  # Balanced (slight whiff reduction, more contact emphasis)
    'SI': (0.20, 0.45, 0.35, 0.00),  # Primary: ground balls
    'SL': (0.55, 0.15, 0.30, 0.00),  # Primary: whiffs
    'ST': (0.60, 0.10, 0.30, 0.00),  # Primary: whiffs (sweeper)
    'CH': (0.40, 0.25, 0.35, 0.00),  # Balanced with GB component
    'FC': (0.45, 0.25, 0.30, 0.00),  # More whiff weight (was too low at 0.35)
    'CU': (0.50, 0.20, 0.30, 0.00),  # More GB weight (was too low at 0.15)
    'FS': (0.35, 0.30, 0.35, 0.00),  # Balanced whiff+GB (was too GB-heavy at 0.35)
    'SV': (0.60, 0.10, 0.30, 0.00),  # Primary: whiffs
}

# v9 uniform weights (for comparison)
V9_UNIFORM_WEIGHTS = {pt: (0.55, 0.20, 0.25, 0.00) for pt in SUPPORTED_PT}

# Minimum pitches per pitch type for a pitcher to be included in training
MIN_PITCHES_TRAIN = 30
MIN_SWINGS = 10
MIN_BIP = 5

# ==========================================================================
# FEATURE DEFINITIONS
# ==========================================================================
CORE_FEATURES = [
    'velocity',        # Raw velocity (mph)
    'perceived_velo',  # Velocity + extension adjustment
    'spin_rate',       # Spin rate (rpm)
    'ivb',             # Induced vertical break (inches)
    'hb',              # Horizontal break (inches, handedness-normalized)
    'ivb_oe',          # IVB over expected (MVN conditional residual)
    'hb_oe',           # HB over expected (MVN conditional residual)
    'vaa',             # Vertical approach angle (degrees)
    'haa',             # Horizontal approach angle (degrees)
    'extension',       # Release extension (feet)
    'arm_angle',       # Arm angle (degrees)
    'rel_pos_z',       # Release height (feet)
    'rel_pos_x_norm',  # Release side (feet, normalized for handedness)
]

# Arsenal-aware features (require pitcher-level aggregation)
ARSENAL_FEATURES_V9 = [
    'velo_diff',       # Pitcher's FB velo minus this pitch's velo
    'vaa_diff',        # Pitcher's FB VAA minus this pitch's VAA
    'velo_boost',      # Effective velo minus raw velo
]

ARSENAL_FEATURES_V10 = ARSENAL_FEATURES_V9 + [
    'ivb_diff',        # v10: Pitcher's FB IVB minus this pitch's IVB
    'hb_diff',         # v10: Pitcher's FB HB minus this pitch's HB (normalized)
]

# Advanced derived features
DERIVED_FEATURES = [
    'mov_angle',       # Movement direction: arctan2(IVB, HB) in degrees
    'total_mov',       # Total movement: sqrt(IVB^2 + HB^2)
    'spin_per_mph',    # Spin rate / velocity ratio
]

# Count features (v10: used in training, neutralized at scoring)
COUNT_FEATURES = [
    'balls',           # Current ball count (0-3)
    'strikes',         # Current strike count (0-2)
]

# Full feature sets for each version
ALL_FEATURES_V9 = CORE_FEATURES + ARSENAL_FEATURES_V9 + DERIVED_FEATURES
ALL_FEATURES_V10 = CORE_FEATURES + ARSENAL_FEATURES_V10 + DERIVED_FEATURES + COUNT_FEATURES

# Per-pitch-type feature exclusions (same for both versions)
EXCLUDE_BY_PT = {
    'FF': set(),
    'SI': {'mov_angle'},
    'SL': set(),
    'ST': set(),
    'CH': set(),
    'FC': set(),
    'CU': set(),
    'FS': set(),
    'SV': {'mov_angle', 'spin_per_mph'},
}

# XGBoost hyperparameters (shared across all models)
XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    verbosity=0,
)


# ==========================================================================
# MODEL CONFIGS (for comparison mode)
# ==========================================================================
def make_v9_config():
    return {
        'name': 'v9',
        'all_features': ALL_FEATURES_V9,
        'weights': V9_UNIFORM_WEIGHTS,
        'use_cs_model': False,
        'neutralize_count': False,
    }

def make_v10_config():
    return {
        'name': 'v10',
        'all_features': ALL_FEATURES_V10,
        'weights': PT_WEIGHTS,
        'use_cs_model': True,
        'neutralize_count': True,
    }

def make_v10b_config():
    """v10b: movement diffs + tuned PT weights, NO CS model, NO count features."""
    # Use v10 arsenal features (with ivb_diff, hb_diff) but no count
    all_feats = CORE_FEATURES + ARSENAL_FEATURES_V10 + DERIVED_FEATURES
    return {
        'name': 'v10b',
        'all_features': all_feats,
        'weights': PT_WEIGHTS_B,
        'use_cs_model': False,
        'neutralize_count': False,
    }


# ==========================================================================
# MVN CONDITIONAL MODEL (from leaderboard pipeline)
# ==========================================================================
def load_mvn_models():
    meta_path = os.path.join(DATA_DIR, 'metadata.json')
    if not os.path.exists(meta_path):
        print("WARNING: metadata.json not found, IVBOE/HBOE will be zero")
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get('mvnModels', {})


def mat_inv_2x2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-12:
        return None
    return [[m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det]]


def mat_inv_general(m):
    n = len(m)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        pivot = aug[col][col]
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
    return [row[n:] for row in aug]


def mvn_conditional_xivb_xhb(model_params, rel_values):
    mu = model_params['mu']
    cov = model_params['cov']
    n_acc = 2
    n_rel = len(mu) - n_acc
    if len(rel_values) != n_rel:
        return None, None
    sigma_rel = [[cov[n_acc + i][n_acc + j] for j in range(n_rel)] for i in range(n_rel)]
    sigma_rel_inv = mat_inv_general(sigma_rel)
    if sigma_rel_inv is None:
        return None, None
    r_diff = [rel_values[k] - mu[n_acc + k] for k in range(n_rel)]
    sri_rdiff = [sum(sigma_rel_inv[i][j] * r_diff[j] for j in range(n_rel)) for i in range(n_rel)]
    mu_bar = []
    for a in range(n_acc):
        adj = sum(cov[a][n_acc + b] * sri_rdiff[b] for b in range(n_rel))
        mu_bar.append(mu[a] + adj)
    return mu_bar[0], mu_bar[1]


def compute_expected_movement(mvn_models, pitch_type, throws, arm_angle, extension, velocity, rel_z, rel_x):
    pt_key = pitch_type
    if pt_key not in mvn_models:
        return None, None
    model = mvn_models[pt_key]
    if 'mlb' in model and arm_angle is not None and extension is not None and velocity is not None:
        xivb, xhb = mvn_conditional_xivb_xhb(model['mlb'], [arm_angle, extension, velocity])
        if xivb is not None:
            return xivb, xhb
    if 'roc' in model and rel_z is not None and rel_x is not None and extension is not None and velocity is not None:
        xivb, xhb = mvn_conditional_xivb_xhb(model['roc'], [rel_z, rel_x, extension, velocity])
        if xivb is not None:
            return xivb, xhb
    return None, None


# ==========================================================================
# DATA LOADING
# ==========================================================================
def load_pitch_data_from_sheets():
    sys.path.insert(0, LEADERBOARD_DIR)
    import gspread

    # Six 2026 division workbooks (huronalytics). Pair each with its league so
    # the _league tag stays 'AL'/'NL'; NLE2026 also carries ROC/AAA/FCL ('NL').
    DIVISION_BOOKS = [
        ('AL', '1YbgAliQzXePiFan-ruwJ50G80l4AjeyTGN8cO3KJ1XI'),  # ALE2026
        ('AL', '14gglESfgJoT90crQb5hHoEZNUFDZ5chPLbUIV9mlm4E'),  # ALC2026
        ('AL', '1eSFfKRo5kSImjP0SZ1SMssGrOhrKSZM9GOHiwntIlhs'),  # ALW2026
        ('NL', '1BypxxlWgQAltETOLqccOYigeo8nXX-FIuVv6rhT4anA'),  # NLE2026
        ('NL', '1-I8BVEw9bR9rzGVYJao_Ar0bjYZF54pi5pm3YEluB9w'),  # NLC2026
        ('NL', '1vm257A676FORcSRzXcNj6txgehGhYI7k5mnmsgQCYH0'),  # NLW2026
    ]

    gc = gspread.service_account()

    all_pitches = []
    for league, sid in DIVISION_BOOKS:
        print(f"Reading {league} spreadsheet...")
        ss = gc.open_by_key(sid)
        worksheets = ss.worksheets()
        for ws in worksheets:
            title = ws.title.strip()
            if title.lower() in ('template', 'readme', 'notes', 'instructions'):
                continue
            print(f"  {title}...")
            rows = ws.get_all_records()
            for r in rows:
                r['PTeam'] = title
                r['_league'] = league
            all_pitches.extend(rows)

    print(f"Loaded {len(all_pitches)} total pitches from sheets")
    return all_pitches


def load_pitch_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pitches from {csv_path}")
    return df.to_dict('records')


def safe_float(val):
    if val is None or val == '' or val == 'None':
        return None
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


# ==========================================================================
# FEATURE ENGINEERING
# ==========================================================================
def engineer_features(all_pitches, mvn_models):
    """
    Compute all model features from raw pitch data.
    Returns DataFrame with features for BOTH v9 and v10 (superset).
    """
    records = []
    lg_avg_ext = 6.434

    # --- First pass: per-pitcher FB baselines ---
    pitcher_fb_velo = defaultdict(list)
    pitcher_fb_vaa = defaultdict(list)
    pitcher_fb_ivb = defaultdict(list)    # v10: FB IVB baseline
    pitcher_fb_hb = defaultdict(list)     # v10: FB HB baseline (handedness-normalized)
    pitcher_all_velo = defaultdict(list)
    pitcher_pitch_types = defaultdict(set)

    for p in all_pitches:
        pitcher = p.get('Pitcher', '').strip().strip('"')
        team = p.get('PTeam', '')
        pt = p.get('Pitch Type', '')
        throws = p.get('Throws', '')
        velo = safe_float(p.get('Velocity'))
        vaa = safe_float(p.get('VAA'))
        ivb = safe_float(p.get('IndVertBrk'))
        hb = safe_float(p.get('HorzBrk'))
        is_lhp = (throws == 'L')

        pk = pitcher + '|' + team
        if pt:
            pitcher_pitch_types[pk].add(pt)
        if velo is not None:
            pitcher_all_velo[pk].append(velo)
            if pt in ('FF', 'SI'):
                pitcher_fb_velo[pk].append(velo)
                if vaa is not None:
                    pitcher_fb_vaa[pk].append(vaa)
                if ivb is not None:
                    pitcher_fb_ivb[pk].append(ivb)
                if hb is not None:
                    pitcher_fb_hb[pk].append(-hb if is_lhp else hb)

    # Compute baselines
    fb_baselines = {}
    for pk in pitcher_pitch_types:
        fb_velos = pitcher_fb_velo.get(pk, [])
        fb_vaas = pitcher_fb_vaa.get(pk, [])
        fb_ivbs = pitcher_fb_ivb.get(pk, [])
        fb_hbs = pitcher_fb_hb.get(pk, [])
        all_velos = pitcher_all_velo.get(pk, [])

        fb_baselines[pk] = {
            'fb_velo': np.mean(fb_velos) if fb_velos else (np.mean(all_velos) if all_velos else None),
            'fb_vaa': np.mean(fb_vaas) if fb_vaas else None,
            'fb_ivb': np.mean(fb_ivbs) if fb_ivbs else None,
            'fb_hb': np.mean(fb_hbs) if fb_hbs else None,
            'n_pitch_types': len(pitcher_pitch_types[pk]),
        }

    # League-wide fallbacks
    all_fb_velo = [v for vl in pitcher_fb_velo.values() for v in vl]
    all_fb_vaa = [v for vl in pitcher_fb_vaa.values() for v in vl]
    all_fb_ivb = [v for vl in pitcher_fb_ivb.values() for v in vl]
    all_fb_hb = [v for vl in pitcher_fb_hb.values() for v in vl]
    lg_fb_velo = np.mean(all_fb_velo) if all_fb_velo else 94.0
    lg_fb_vaa = np.mean(all_fb_vaa) if all_fb_vaa else -4.7
    lg_fb_ivb = np.mean(all_fb_ivb) if all_fb_ivb else 15.0
    lg_fb_hb = np.mean(all_fb_hb) if all_fb_hb else 8.0

    # --- Second pass: compute features for each pitch ---
    for p in all_pitches:
        pitcher = p.get('Pitcher', '').strip().strip('"')
        team = p.get('PTeam', '')
        throws = p.get('Throws', '')
        stands = p.get('Bats', '') or p.get('Stands', '')
        pt = p.get('Pitch Type', '')

        if not pitcher or not pt or pt not in SUPPORTED_PT:
            continue

        pk = pitcher + '|' + team
        is_lhp = (throws == 'L')

        # Raw values
        velocity = safe_float(p.get('Velocity'))
        spin_rate = safe_float(p.get('Spin Rate'))
        ivb = safe_float(p.get('IndVertBrk'))
        hb = safe_float(p.get('HorzBrk'))
        extension = safe_float(p.get('Extension'))
        arm_angle = safe_float(p.get('ArmAngle') or p.get('Arm Angle'))
        vaa = safe_float(p.get('VAA'))
        haa = safe_float(p.get('HAA'))
        rel_z = safe_float(p.get('RelPosZ'))
        rel_x = safe_float(p.get('RelPosX'))
        eff_velo = safe_float(p.get('EffectiveVelo'))

        if velocity is None or ivb is None or hb is None:
            continue

        # Perceived velocity
        ext_val = extension if extension is not None else lg_avg_ext
        perceived_velo = velocity + (ext_val - lg_avg_ext) * 1.0

        # Handedness normalization
        hb_norm = -hb if is_lhp else hb
        rel_x_norm = -rel_x if (is_lhp and rel_x is not None) else rel_x

        # IVBOE / HBOE from MVN model
        xivb, xhb = compute_expected_movement(
            mvn_models, pt, throws, arm_angle, extension, velocity, rel_z, rel_x
        )
        ivb_oe = (ivb - xivb) if xivb is not None else 0.0
        hb_oe = (hb_norm - xhb) if xhb is not None else 0.0

        # Movement angle and total movement
        mov_angle = math.degrees(math.atan2(ivb, hb_norm))
        total_mov = math.sqrt(ivb**2 + hb**2)

        # Spin per mph
        spin_per_mph = (spin_rate / velocity) if (spin_rate is not None and velocity > 0) else None

        # Arsenal-aware features
        bl = fb_baselines.get(pk, {})
        fb_velo_val = bl.get('fb_velo', lg_fb_velo) or lg_fb_velo
        fb_vaa_val = bl.get('fb_vaa', lg_fb_vaa) or lg_fb_vaa
        fb_ivb_val = bl.get('fb_ivb', lg_fb_ivb) or lg_fb_ivb
        fb_hb_val = bl.get('fb_hb', lg_fb_hb) or lg_fb_hb

        velo_diff = fb_velo_val - perceived_velo
        vaa_diff = fb_vaa_val - vaa if vaa is not None else 0.0
        velo_boost = (eff_velo - velocity) if eff_velo is not None else (ext_val - lg_avg_ext) * 1.0

        # v10: Movement differentials from FB
        ivb_diff = fb_ivb_val - ivb       # How much less rise than FB
        hb_diff = fb_hb_val - hb_norm     # How different horizontal movement from FB

        # v10: Count features
        count_str = str(p.get('Count', ''))
        balls_val, strikes_val = 1, 1  # neutral fallback
        if '-' in count_str:
            try:
                parts = count_str.split('-')
                balls_val = int(parts[0])
                strikes_val = int(parts[1])
            except (ValueError, IndexError):
                pass

        # --- Outcome labels ---
        desc = p.get('Description', '')
        is_swing = desc in ('Swinging Strike', 'Foul', 'In Play')
        is_whiff = desc == 'Swinging Strike'
        is_bip = desc == 'In Play'
        run_value = safe_float(p.get('RunExp') or p.get('Delta Run Exp'))

        bb_type = p.get('BBType') or p.get('BB Type') or ''
        is_gb = 1 if bb_type.lower() in ('ground_ball', 'groundball', 'gb') else 0

        exit_velo = safe_float(p.get('ExitVelo') or p.get('Exit Velocity'))
        launch_angle = safe_float(p.get('LaunchAngle') or p.get('Launch Angle'))
        xwoba = safe_float(p.get('xwOBA'))

        # v10: Called strike outcome
        is_take_eligible = desc in ('Called Strike', 'Ball')
        is_called_strike = 1 if desc == 'Called Strike' else 0

        record = {
            'pitcher': pitcher, 'team': team, 'throws': throws,
            'stands': stands, 'pitch_type': pt,
            # Core features
            'velocity': velocity, 'perceived_velo': perceived_velo,
            'spin_rate': spin_rate, 'ivb': ivb, 'hb': hb_norm,
            'ivb_oe': ivb_oe, 'hb_oe': hb_oe,
            'vaa': vaa, 'haa': haa,
            'extension': extension, 'arm_angle': arm_angle,
            'rel_pos_z': rel_z, 'rel_pos_x_norm': rel_x_norm,
            # Arsenal features (v9)
            'velo_diff': velo_diff, 'vaa_diff': vaa_diff, 'velo_boost': velo_boost,
            # Arsenal features (v10)
            'ivb_diff': ivb_diff, 'hb_diff': hb_diff,
            # Derived features
            'mov_angle': mov_angle, 'total_mov': total_mov, 'spin_per_mph': spin_per_mph,
            # Count features (v10)
            'balls': balls_val, 'strikes': strikes_val,
            # Outcomes
            'is_swing': is_swing, 'is_whiff': is_whiff,
            'is_bip': is_bip, 'is_gb': is_gb,
            'run_value': run_value,
            'exit_velo': exit_velo, 'launch_angle': launch_angle, 'xwoba': xwoba,
            'is_take_eligible': is_take_eligible,
            'is_called_strike': is_called_strike,
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Engineered features for {len(df)} pitches")
    print(f"  Pitch types: {dict(df['pitch_type'].value_counts())}")

    # Print v10 feature stats
    if 'ivb_diff' in df.columns:
        non_fb = df[~df['pitch_type'].isin(['FF', 'SI'])]
        print(f"  ivb_diff (off-speed): mean={non_fb['ivb_diff'].mean():.1f}, std={non_fb['ivb_diff'].std():.1f}")
        print(f"  hb_diff (off-speed):  mean={non_fb['hb_diff'].mean():.1f}, std={non_fb['hb_diff'].std():.1f}")

    return df


# ==========================================================================
# TRAINING & EVALUATION (unified for both configs)
# ==========================================================================
def get_feature_list(config, pitch_type):
    """Get features for a given pitch type, excluding irrelevant ones."""
    exclude = EXCLUDE_BY_PT.get(pitch_type, set())
    return [f for f in config['all_features'] if f not in exclude]


def train_and_evaluate(df, config, n_folds=5, verbose=True):
    """
    Train models with a given config and return metrics + models.

    Uses GroupKFold (pitcher-level) for out-of-fold evaluation:
    all 4 components trained in each fold, composite OOF stuff scores computed.

    Returns: (models, league_stats, oof_stuff_scores, cv_metrics)
    """
    from sklearn.metrics import roc_auc_score

    name = config['name']
    models = {}
    league_stats = {}
    oof_stuff = pd.Series(np.nan, index=df.index, dtype=float)
    cv_metrics = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TRAINING STUFF+ {name.upper()}")
        print(f"{'='*60}")

    for pt in SUPPORTED_PT:
        pt_mask = df['pitch_type'] == pt
        pt_df = df[pt_mask].copy()
        if len(pt_df) < 100:
            if verbose:
                print(f"\n  {pt}: Only {len(pt_df)} pitches, SKIPPING")
            continue

        features = get_feature_list(config, pt)
        w_whiff, w_gb, w_contact, w_cs = config['weights'][pt]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  {pt} ({len(pt_df)} pitches, {pt_df['pitcher'].nunique()} pitchers, "
                  f"{len(features)} features)")
            print(f"  Weights: whiff={w_whiff}, gb={w_gb}, contact={w_contact}, cs={w_cs}")

        # Filter pitchers with enough data
        pitcher_groups = pt_df.groupby('pitcher')
        valid_pitchers = [p for p, grp in pitcher_groups if len(grp) >= MIN_PITCHES_TRAIN]
        if len(valid_pitchers) < 20:
            valid_pitchers = [p for p, grp in pitcher_groups if len(grp) >= 15]

        train_df = pt_df[pt_df['pitcher'].isin(valid_pitchers)].copy()
        X_full = train_df[features].copy().fillna(0)
        pitchers = train_df['pitcher']
        unique_pitchers = pitchers.unique()
        n_cv = min(n_folds, len(unique_pitchers))

        # ============================================================
        # CROSS-VALIDATED OOF SCORING
        # Train all components per fold, compute composite OOF stuff scores
        # ============================================================
        oof_raw = np.full(len(train_df), np.nan)
        fold_whiff_aucs = []
        fold_gb_aucs = []
        fold_contact_corrs = []
        fold_cs_aucs = []

        if n_cv >= 2:
            gkf = GroupKFold(n_splits=n_cv)

            for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_full, groups=pitchers)):
                X_tr = X_full.iloc[train_idx]
                X_va = X_full.iloc[val_idx].copy()
                tr = train_df.iloc[train_idx]
                va = train_df.iloc[val_idx]

                # Neutralize count for validation scoring
                if config.get('neutralize_count'):
                    if 'balls' in X_va.columns:
                        X_va['balls'] = 1
                    if 'strikes' in X_va.columns:
                        X_va['strikes'] = 1

                # --- Whiff ---
                swing_tr = tr['is_swing'].astype(bool)
                z_w_va = np.zeros(len(X_va))
                if swing_tr.sum() >= MIN_SWINGS and tr.loc[swing_tr.index[swing_tr], 'is_whiff'].sum() >= 10:
                    wm = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
                    wm.fit(X_tr.loc[swing_tr], tr.loc[swing_tr.index[swing_tr], 'is_whiff'].astype(int))
                    # Score ALL val pitches (stuff is physical, not swing-dependent)
                    whiff_probs_va = wm.predict_proba(X_va)[:, 1]
                    # Mean/std from training set (all pitches scored)
                    whiff_probs_tr = wm.predict_proba(X_tr)[:, 1]
                    w_mean = whiff_probs_tr.mean()
                    w_std = max(whiff_probs_tr.std(), 0.001)
                    z_w_va = (whiff_probs_va - w_mean) / w_std
                    # CV AUC (on swing subset of val)
                    swing_va = va['is_swing'].astype(bool)
                    if swing_va.sum() >= 10:
                        y_va_whiff = va.loc[swing_va.index[swing_va], 'is_whiff'].astype(int).values
                        if len(np.unique(y_va_whiff)) > 1:
                            auc_w = roc_auc_score(y_va_whiff, whiff_probs_va[swing_va.values])
                            fold_whiff_aucs.append(auc_w)

                # --- GB ---
                bip_tr = tr['is_bip'].astype(bool)
                z_gb_va = np.zeros(len(X_va))
                if bip_tr.sum() >= MIN_BIP and tr.loc[bip_tr.index[bip_tr], 'is_gb'].sum() >= 5:
                    gm = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
                    gm.fit(X_tr.loc[bip_tr], tr.loc[bip_tr.index[bip_tr], 'is_gb'].astype(int))
                    gb_probs_va = gm.predict_proba(X_va)[:, 1]
                    gb_probs_tr = gm.predict_proba(X_tr)[:, 1]
                    g_mean = gb_probs_tr.mean()
                    g_std = max(gb_probs_tr.std(), 0.001)
                    z_gb_va = (gb_probs_va - g_mean) / g_std
                    bip_va = va['is_bip'].astype(bool)
                    if bip_va.sum() >= 5:
                        y_va_gb = va.loc[bip_va.index[bip_va], 'is_gb'].astype(int).values
                        if len(np.unique(y_va_gb)) > 1:
                            auc_g = roc_auc_score(y_va_gb, gb_probs_va[bip_va.values])
                            fold_gb_aucs.append(auc_g)

                # --- Contact quality ---
                contact_tr = bip_tr & tr['xwoba'].notna()
                z_c_va = np.zeros(len(X_va))
                if contact_tr.sum() >= MIN_BIP:
                    cm = XGBRegressor(**XGB_PARAMS)
                    cm.fit(X_tr.loc[contact_tr], tr.loc[contact_tr.index[contact_tr], 'xwoba'])
                    contact_pred_va = cm.predict(X_va)
                    contact_pred_tr = cm.predict(X_tr)
                    c_mean = contact_pred_tr.mean()
                    c_std = max(contact_pred_tr.std(), 0.001)
                    z_c_va = (contact_pred_va - c_mean) / c_std
                    contact_va = va['is_bip'].astype(bool) & va['xwoba'].notna()
                    if contact_va.sum() >= 5:
                        corr_c = np.corrcoef(
                            va.loc[contact_va.index[contact_va], 'xwoba'].values,
                            contact_pred_va[contact_va.values]
                        )[0, 1]
                        if not np.isnan(corr_c):
                            fold_contact_corrs.append(corr_c)

                # --- Called strike (v10 only) ---
                z_cs_va = np.zeros(len(X_va))
                if config.get('use_cs_model') and w_cs > 0:
                    take_tr = tr['is_take_eligible'].astype(bool)
                    if take_tr.sum() >= 100 and tr.loc[take_tr.index[take_tr], 'is_called_strike'].sum() >= 30:
                        csm = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
                        csm.fit(X_tr.loc[take_tr], tr.loc[take_tr.index[take_tr], 'is_called_strike'].astype(int))
                        cs_probs_va = csm.predict_proba(X_va)[:, 1]
                        cs_probs_tr = csm.predict_proba(X_tr)[:, 1]
                        cs_mean = cs_probs_tr.mean()
                        cs_std = max(cs_probs_tr.std(), 0.001)
                        z_cs_va = (cs_probs_va - cs_mean) / cs_std
                        take_va = va['is_take_eligible'].astype(bool)
                        if take_va.sum() >= 30:
                            y_va_cs = va.loc[take_va.index[take_va], 'is_called_strike'].astype(int).values
                            if len(np.unique(y_va_cs)) > 1:
                                auc_cs = roc_auc_score(y_va_cs, cs_probs_va[take_va.values])
                                fold_cs_aucs.append(auc_cs)

                # Composite raw score (negative = better stuff)
                fold_raw = -w_whiff * z_w_va - w_gb * z_gb_va + w_contact * z_c_va - w_cs * z_cs_va
                oof_raw[val_idx] = fold_raw

        # Store CV metrics
        cv_metrics[pt] = {
            'whiff_auc': np.mean(fold_whiff_aucs) if fold_whiff_aucs else None,
            'gb_auc': np.mean(fold_gb_aucs) if fold_gb_aucs else None,
            'contact_corr': np.mean(fold_contact_corrs) if fold_contact_corrs else None,
            'cs_auc': np.mean(fold_cs_aucs) if fold_cs_aucs else None,
            'n_pitches': len(pt_df),
            'n_pitchers': pt_df['pitcher'].nunique(),
        }

        if verbose:
            m = cv_metrics[pt]
            if m['whiff_auc']:
                print(f"  Whiff CV AUC:   {m['whiff_auc']:.4f}")
            if m['gb_auc']:
                print(f"  GB CV AUC:      {m['gb_auc']:.4f}")
            if m['contact_corr']:
                print(f"  Contact CV r:   {m['contact_corr']:.4f}")
            if m['cs_auc']:
                print(f"  CS CV AUC:      {m['cs_auc']:.4f}")

        # ============================================================
        # TRAIN FINAL MODELS ON ALL DATA (for deployment)
        # ============================================================
        swing_mask = train_df['is_swing'].astype(bool)
        bip_mask = train_df['is_bip'].astype(bool)

        # Whiff
        whiff_model = None
        if swing_mask.sum() >= MIN_SWINGS and train_df.loc[swing_mask, 'is_whiff'].sum() >= 10:
            whiff_model = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
            whiff_model.fit(X_full.loc[swing_mask], train_df.loc[swing_mask, 'is_whiff'].astype(int))
            if verbose:
                imp = dict(zip(features, whiff_model.feature_importances_))
                top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
                print(f"  Whiff top: {[(k, round(v, 3)) for k, v in top5]}")

        # GB
        gb_model = None
        if bip_mask.sum() >= MIN_BIP and train_df.loc[bip_mask, 'is_gb'].sum() >= 5:
            gb_model = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
            gb_model.fit(X_full.loc[bip_mask], train_df.loc[bip_mask, 'is_gb'].astype(int))
            if verbose:
                imp = dict(zip(features, gb_model.feature_importances_))
                top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
                print(f"  GB top:    {[(k, round(v, 3)) for k, v in top5]}")

        # Contact
        contact_model = None
        contact_mask = bip_mask & train_df['xwoba'].notna()
        if contact_mask.sum() >= MIN_BIP:
            contact_model = XGBRegressor(**XGB_PARAMS)
            contact_model.fit(X_full.loc[contact_mask], train_df.loc[contact_mask, 'xwoba'])
            if verbose:
                imp = dict(zip(features, contact_model.feature_importances_))
                top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
                print(f"  Contact top: {[(k, round(v, 3)) for k, v in top5]}")

        # Called strike (v10 only)
        cs_model = None
        if config.get('use_cs_model') and w_cs > 0:
            take_mask = train_df['is_take_eligible'].astype(bool)
            if take_mask.sum() >= 100 and train_df.loc[take_mask, 'is_called_strike'].sum() >= 30:
                cs_model = XGBClassifier(**XGB_PARAMS, eval_metric='logloss')
                cs_model.fit(X_full.loc[take_mask], train_df.loc[take_mask, 'is_called_strike'].astype(int))
                if verbose:
                    imp = dict(zip(features, cs_model.feature_importances_))
                    top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
                    print(f"  CS top:    {[(k, round(v, 3)) for k, v in top5]}")

        # ============================================================
        # NORMALIZATION (pitcher-level composite distribution)
        # ============================================================
        X_all = pt_df[features].fillna(0)
        X_all_score = X_all.copy()
        if config.get('neutralize_count'):
            if 'balls' in X_all_score.columns:
                X_all_score['balls'] = 1
            if 'strikes' in X_all_score.columns:
                X_all_score['strikes'] = 1

        swing_all = pt_df['is_swing'].astype(bool)
        bip_all = pt_df['is_bip'].astype(bool)

        # Component means/stds
        if whiff_model is not None and swing_all.any():
            all_whiff = whiff_model.predict_proba(X_all_score)[:, 1]
            whiff_mean = float(np.mean(all_whiff))
            whiff_std = float(np.std(all_whiff))
        else:
            whiff_mean, whiff_std = 0.25, 0.1

        if gb_model is not None and bip_all.any():
            all_gb = gb_model.predict_proba(X_all_score)[:, 1]
            gb_mean = float(np.mean(all_gb))
            gb_std = float(np.std(all_gb))
        else:
            gb_mean, gb_std = 0.45, 0.1

        if contact_model is not None and bip_all.any():
            contact_mask_all = bip_all & pt_df['xwoba'].notna()
            if contact_mask_all.any():
                all_contact = contact_model.predict(X_all_score)
                contact_mean = float(np.mean(all_contact))
                contact_std = float(np.std(all_contact))
            else:
                contact_mean, contact_std = 0.35, 0.05
        else:
            contact_mean, contact_std = 0.35, 0.05

        if cs_model is not None:
            all_cs = cs_model.predict_proba(X_all_score)[:, 1]
            cs_mean = float(np.mean(all_cs))
            cs_std = float(np.std(all_cs))
        else:
            cs_mean, cs_std = 0.20, 0.05

        # Pitcher-level raw composites for normalization
        pitcher_raws = []
        for pitcher, grp in pt_df.groupby('pitcher'):
            if len(grp) < 15:
                continue
            X_p = grp[features].fillna(0)
            if config.get('neutralize_count'):
                X_p = X_p.copy()
                if 'balls' in X_p.columns:
                    X_p['balls'] = 1
                if 'strikes' in X_p.columns:
                    X_p['strikes'] = 1

            swing_p = grp['is_swing'].astype(bool)
            bip_p = grp['is_bip'].astype(bool)

            z_w = 0.0
            if whiff_model is not None and swing_p.any():
                z_w = (np.mean(whiff_model.predict_proba(X_p)[:, 1]) - whiff_mean) / max(whiff_std, 0.001)

            z_gb = 0.0
            if gb_model is not None and bip_p.any():
                z_gb = (np.mean(gb_model.predict_proba(X_p)[:, 1]) - gb_mean) / max(gb_std, 0.001)

            z_c = 0.0
            if contact_model is not None and bip_p.any():
                z_c = (np.mean(contact_model.predict(X_p)) - contact_mean) / max(contact_std, 0.001)

            z_cs = 0.0
            if cs_model is not None:
                z_cs = (np.mean(cs_model.predict_proba(X_p)[:, 1]) - cs_mean) / max(cs_std, 0.001)

            raw = -w_whiff * z_w - w_gb * z_gb + w_contact * z_c - w_cs * z_cs
            pitcher_raws.append(raw)

        if len(pitcher_raws) >= 5:
            raw_mean = float(np.mean(pitcher_raws))
            raw_std = float(np.std(pitcher_raws))
        else:
            raw_mean, raw_std = 0.0, 1.0

        if verbose:
            print(f"  Normalization: raw_mean={raw_mean:.4f}, raw_std={raw_std:.4f}")

        models[pt] = {
            'whiff_model': whiff_model,
            'gb_model': gb_model,
            'contact_model': contact_model,
            'cs_model': cs_model,
            'features': features,
        }
        league_stats[pt] = {
            'whiff_mean': whiff_mean, 'whiff_std': max(whiff_std, 0.001),
            'gb_mean': gb_mean, 'gb_std': max(gb_std, 0.001),
            'contact_mean': contact_mean, 'contact_std': max(contact_std, 0.001),
            'cs_mean': cs_mean, 'cs_std': max(cs_std, 0.001),
            'raw_mean': raw_mean, 'raw_std': max(raw_std, 0.001),
            'n_pitches': len(pt_df), 'n_pitchers': pt_df['pitcher'].nunique(),
        }

        # Convert OOF raw to stuff+ scale for this PT's training pitchers
        valid_oof = ~np.isnan(oof_raw)
        if valid_oof.any():
            oof_stuff_pt = 100 - 10 * (oof_raw[valid_oof] - raw_mean) / max(raw_std, 0.001)
            oof_stuff_pt = np.clip(oof_stuff_pt, 40, 160)
            oof_stuff.iloc[train_df.index[valid_oof]] = oof_stuff_pt

    return models, league_stats, oof_stuff, cv_metrics


# ==========================================================================
# SCORING (for deployment)
# ==========================================================================
def score_pitches(df, models, league_stats, config):
    """Score each pitch with Stuff+ using trained models and config."""
    results = pd.Series(np.nan, index=df.index, dtype=float)

    for pt in df['pitch_type'].unique():
        if pt not in models or pt not in league_stats:
            continue

        mask = df['pitch_type'] == pt
        pt_df = df.loc[mask]
        m = models[pt]
        ls = league_stats[pt]
        features = m['features']
        w_whiff, w_gb, w_contact, w_cs = config['weights'][pt]

        X = pt_df[features].fillna(0)

        # v10: Neutralize count at scoring time
        if config.get('neutralize_count'):
            X = X.copy()
            if 'balls' in X.columns:
                X['balls'] = 1
            if 'strikes' in X.columns:
                X['strikes'] = 1

        # Whiff
        if m['whiff_model'] is not None:
            z_w = (m['whiff_model'].predict_proba(X)[:, 1] - ls['whiff_mean']) / ls['whiff_std']
        else:
            z_w = np.zeros(len(pt_df))

        # GB
        if m['gb_model'] is not None:
            z_gb = (m['gb_model'].predict_proba(X)[:, 1] - ls['gb_mean']) / ls['gb_std']
        else:
            z_gb = np.zeros(len(pt_df))

        # Contact
        if m['contact_model'] is not None:
            z_c = (m['contact_model'].predict(X) - ls['contact_mean']) / ls['contact_std']
        else:
            z_c = np.zeros(len(pt_df))

        # Called strike
        if m.get('cs_model') is not None:
            z_cs = (m['cs_model'].predict_proba(X)[:, 1] - ls['cs_mean']) / ls['cs_std']
        else:
            z_cs = np.zeros(len(pt_df))

        raw = -w_whiff * z_w - w_gb * z_gb + w_contact * z_c - w_cs * z_cs
        stuff_plus = 100 - 10 * (raw - ls['raw_mean']) / ls['raw_std']
        stuff_plus = np.clip(stuff_plus, 40, 160)
        results.loc[mask] = stuff_plus

    return results


# ==========================================================================
# VALIDATION
# ==========================================================================
def validate_and_correlate(df, stuff_scores, label=""):
    """
    Compute pitcher-level correlations between Stuff+ and outcomes.
    Returns a dict of {pitch_type: {outcome: correlation}}.
    """
    df_scored = df.copy()
    df_scored['stuff_plus'] = stuff_scores
    correlations = {}

    for pt in SUPPORTED_PT:
        pt_df = df_scored[df_scored['pitch_type'] == pt]
        pitcher_groups = pt_df.groupby('pitcher')

        records = []
        for pitcher, grp in pitcher_groups:
            if len(grp) < MIN_PITCHES_TRAIN:
                continue
            n_swings = grp['is_swing'].sum()
            n_whiffs = grp['is_whiff'].sum()
            whiff_rate = n_whiffs / n_swings if n_swings >= MIN_SWINGS else None

            rv_vals = grp['run_value'].dropna()
            rv_per_100 = rv_vals.mean() * 100 if len(rv_vals) > 0 else None

            xwoba_vals = grp.loc[grp['xwoba'].notna(), 'xwoba']
            avg_xwoba = xwoba_vals.mean() if len(xwoba_vals) >= 5 else None

            n_bip = grp['is_bip'].sum()
            gb_rate = grp.loc[grp['is_bip'].astype(bool), 'is_gb'].mean() if n_bip >= MIN_BIP else None

            sp = grp['stuff_plus'].dropna()
            if len(sp) == 0:
                continue

            records.append({
                'pitcher': pitcher,
                'stuff': sp.mean(),
                'whiff_rate': whiff_rate,
                'gb_rate': gb_rate,
                'rv100': rv_per_100,
                'xwoba': avg_xwoba,
                'n': len(grp),
            })

        if len(records) < 10:
            continue

        val_df = pd.DataFrame(records)
        pt_corrs = {}

        for outcome in ['whiff_rate', 'rv100', 'xwoba']:
            valid = val_df.dropna(subset=['stuff', outcome])
            if len(valid) >= 10:
                r = valid['stuff'].corr(valid[outcome])
                pt_corrs[outcome] = round(r, 4)

        correlations[pt] = pt_corrs

    return correlations


def print_validation(correlations, label=""):
    """Print pitcher-level correlation table."""
    print(f"\n{'='*60}")
    print(f"  PITCHER-LEVEL CORRELATIONS{' (' + label + ')' if label else ''}")
    print(f"{'='*60}")
    print(f"  {'PT':<4} {'vs whiff_rate':>14} {'vs rv100':>14} {'vs xwoba':>14}")
    print(f"  {'-'*4} {'-'*14} {'-'*14} {'-'*14}")
    for pt in SUPPORTED_PT:
        if pt not in correlations:
            continue
        c = correlations[pt]
        wr = f"{c.get('whiff_rate', 'N/A'):>+.4f}" if 'whiff_rate' in c else '       N/A'
        rv = f"{c.get('rv100', 'N/A'):>+.4f}" if 'rv100' in c else '       N/A'
        xw = f"{c.get('xwoba', 'N/A'):>+.4f}" if 'xwoba' in c else '       N/A'
        print(f"  {pt:<4} {wr:>14} {rv:>14} {xw:>14}")


# ==========================================================================
# COMPARISON MODE
# ==========================================================================
def print_comparison(v9_cv, v10_cv, v9_corrs, v10_corrs):
    """Print side-by-side comparison of v9 vs v10."""

    print(f"\n{'='*70}")
    print(f"  COMPARISON: v9 vs v10")
    print(f"{'='*70}")

    # CV metrics
    print(f"\n  Cross-Validated Component Metrics:")
    print(f"  {'PT':<4} {'v9 Whiff':>10} {'v10 Whiff':>10} {'Δ':>7}"
          f"  {'v9 GB':>8} {'v10 GB':>8} {'Δ':>7}"
          f"  {'v9 Cntct':>9} {'v10 Cntct':>9} {'Δ':>7}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*7}"
          f"  {'-'*8} {'-'*8} {'-'*7}"
          f"  {'-'*9} {'-'*9} {'-'*7}")

    total_w9, total_w10 = [], []
    total_g9, total_g10 = [], []
    total_c9, total_c10 = [], []
    cs_aucs = []

    for pt in SUPPORTED_PT:
        m9 = v9_cv.get(pt, {})
        m10 = v10_cv.get(pt, {})

        w9 = m9.get('whiff_auc')
        w10 = m10.get('whiff_auc')
        g9 = m9.get('gb_auc')
        g10 = m10.get('gb_auc')
        c9 = m9.get('contact_corr')
        c10 = m10.get('contact_corr')
        cs10 = m10.get('cs_auc')

        w9s = f"{w9:.4f}" if w9 else "  N/A"
        w10s = f"{w10:.4f}" if w10 else "  N/A"
        dw = f"{w10-w9:+.4f}" if (w9 and w10) else "   N/A"
        g9s = f"{g9:.4f}" if g9 else " N/A"
        g10s = f"{g10:.4f}" if g10 else " N/A"
        dg = f"{g10-g9:+.4f}" if (g9 and g10) else "   N/A"
        c9s = f"{c9:.4f}" if c9 else "  N/A"
        c10s = f"{c10:.4f}" if c10 else "  N/A"
        dc = f"{c10-c9:+.4f}" if (c9 and c10) else "   N/A"

        print(f"  {pt:<4} {w9s:>10} {w10s:>10} {dw:>7}"
              f"  {g9s:>8} {g10s:>8} {dg:>7}"
              f"  {c9s:>9} {c10s:>9} {dc:>7}")

        if w9: total_w9.append(w9)
        if w10: total_w10.append(w10)
        if g9: total_g9.append(g9)
        if g10: total_g10.append(g10)
        if c9: total_c9.append(c9)
        if c10: total_c10.append(c10)
        if cs10: cs_aucs.append((pt, cs10))

    # Averages
    if total_w9 and total_w10:
        avg_w9 = np.mean(total_w9)
        avg_w10 = np.mean(total_w10)
        avg_g9 = np.mean(total_g9) if total_g9 else 0
        avg_g10 = np.mean(total_g10) if total_g10 else 0
        avg_c9 = np.mean(total_c9) if total_c9 else 0
        avg_c10 = np.mean(total_c10) if total_c10 else 0
        print(f"  {'AVG':<4} {avg_w9:>10.4f} {avg_w10:>10.4f} {avg_w10-avg_w9:>+7.4f}"
              f"  {avg_g9:>8.4f} {avg_g10:>8.4f} {avg_g10-avg_g9:>+7.4f}"
              f"  {avg_c9:>9.4f} {avg_c10:>9.4f} {avg_c10-avg_c9:>+7.4f}")

    # Called strike AUCs (v10 only)
    if cs_aucs:
        print(f"\n  v10 Called Strike AUCs:")
        for pt, auc in cs_aucs:
            quality = "✓ useful" if auc > 0.54 else "~ marginal" if auc > 0.52 else "✗ noise"
            print(f"    {pt}: {auc:.4f} ({quality})")

    # Pitcher-level correlations comparison
    print(f"\n  Pitcher-Level Correlations (Stuff+ vs Outcomes):")
    print(f"  {'PT':<4} {'v9 whiff_r':>11} {'v10 whiff_r':>12}"
          f"  {'v9 rv100':>10} {'v10 rv100':>11}"
          f"  {'v9 xwoba':>10} {'v10 xwoba':>11}")
    print(f"  {'-'*4} {'-'*11} {'-'*12}"
          f"  {'-'*10} {'-'*11}"
          f"  {'-'*10} {'-'*11}")

    wins_v10 = 0
    wins_v9 = 0
    comparisons = 0

    for pt in SUPPORTED_PT:
        c9 = v9_corrs.get(pt, {})
        c10 = v10_corrs.get(pt, {})

        parts = []
        for outcome in ['whiff_rate', 'rv100', 'xwoba']:
            r9 = c9.get(outcome)
            r10 = c10.get(outcome)
            if r9 is not None and r10 is not None:
                # For whiff_rate, higher r = better. For rv100/xwoba, more negative r = better.
                if outcome == 'whiff_rate':
                    if r10 > r9:
                        wins_v10 += 1
                    elif r9 > r10:
                        wins_v9 += 1
                else:
                    if r10 < r9:
                        wins_v10 += 1
                    elif r9 < r10:
                        wins_v9 += 1
                comparisons += 1

        wr9 = f"{c9.get('whiff_rate', 0):>+.4f}" if 'whiff_rate' in c9 else '      N/A'
        wr10 = f"{c10.get('whiff_rate', 0):>+.4f}" if 'whiff_rate' in c10 else '       N/A'
        rv9 = f"{c9.get('rv100', 0):>+.4f}" if 'rv100' in c9 else '     N/A'
        rv10 = f"{c10.get('rv100', 0):>+.4f}" if 'rv100' in c10 else '      N/A'
        xw9 = f"{c9.get('xwoba', 0):>+.4f}" if 'xwoba' in c9 else '     N/A'
        xw10 = f"{c10.get('xwoba', 0):>+.4f}" if 'xwoba' in c10 else '      N/A'

        print(f"  {pt:<4} {wr9:>11} {wr10:>12}"
              f"  {rv9:>10} {rv10:>11}"
              f"  {xw9:>10} {xw10:>11}")

    print(f"\n  Pitcher-level correlation wins: v10={wins_v10}, v9={wins_v9} "
          f"(of {comparisons} comparisons)")

    # Overall verdict
    print(f"\n{'='*70}")
    if wins_v10 > wins_v9 * 1.5:
        print(f"  VERDICT: v10 is CLEARLY BETTER ({wins_v10}/{comparisons} wins)")
    elif wins_v10 > wins_v9:
        print(f"  VERDICT: v10 is MODERATELY BETTER ({wins_v10}/{comparisons} wins)")
    elif wins_v10 == wins_v9:
        print(f"  VERDICT: MIXED — v10 and v9 are comparable ({wins_v10}/{comparisons} wins each)")
    else:
        print(f"  VERDICT: v9 is better ({wins_v9}/{comparisons} wins) — v10 changes may not help")
    print(f"{'='*70}")


# ==========================================================================
# SAVE / LOAD ARTIFACTS
# ==========================================================================
def save_artifacts(models, league_stats, config):
    artifacts = {
        'models': models,
        'league_stats': league_stats,
        'config': {k: v for k, v in config.items() if k != 'all_features'},
        'version': 'v10',
    }

    path = os.path.join(SCRIPT_DIR, 'stuff_models_v10.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nSaved model artifacts to {path}")

    # Human-readable summary
    summary_path = os.path.join(SCRIPT_DIR, 'model_summary_v10.txt')
    with open(summary_path, 'w') as f:
        f.write("Stuff+ v10 Model Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Improvements over v9:\n")
        f.write("  1. Movement diffs from FB (ivb_diff, hb_diff)\n")
        f.write("  2. Pitch-type-specific target weights\n")
        f.write("  3. Count-aware training, neutral-count scoring\n")
        f.write("  4. Called strike probability component\n\n")
        f.write(f"Supported pitch types: {SUPPORTED_PT}\n\n")
        for pt in SUPPORTED_PT:
            if pt in models:
                m = models[pt]
                ls = league_stats[pt]
                w = config['weights'][pt]
                f.write(f"\n{pt}:\n")
                f.write(f"  Features: {m['features']}\n")
                f.write(f"  Weights: whiff={w[0]}, gb={w[1]}, contact={w[2]}, cs={w[3]}\n")
                f.write(f"  n_pitches: {ls['n_pitches']}, n_pitchers: {ls['n_pitchers']}\n")
                f.write(f"  Whiff: {'Yes' if m['whiff_model'] else 'No'}, "
                        f"GB: {'Yes' if m['gb_model'] else 'No'}, "
                        f"Contact: {'Yes' if m['contact_model'] else 'No'}, "
                        f"CS: {'Yes' if m.get('cs_model') else 'No'}\n")
                f.write(f"  raw_mean={ls['raw_mean']:.4f}, raw_std={ls['raw_std']:.4f}\n")
    print(f"Saved model summary to {summary_path}")


# ==========================================================================
# INTEGRATION: Update pitch_leaderboard.json
# ==========================================================================
def update_pitch_leaderboard(df, stuff_scores):
    df_scored = df.copy()
    df_scored['stuff_plus'] = stuff_scores

    agg = df_scored.groupby(['pitcher', 'team', 'pitch_type']).agg(
        stuff_mean=('stuff_plus', 'mean'),
    ).reset_index()

    # Try _rs suffix first, fall back to plain
    for suffix in ['_rs', '']:
        pl_path = os.path.join(DATA_DIR, f'pitch_leaderboard{suffix}.json')
        if os.path.exists(pl_path):
            break
    else:
        print("WARNING: pitch_leaderboard.json not found")
        return

    with open(pl_path) as f:
        pitch_lb = json.load(f)

    stuff_lookup = {}
    for _, row in agg.iterrows():
        key = (row['pitcher'], row['team'], row['pitch_type'])
        stuff_lookup[key] = round(row['stuff_mean'], 1)

    n_updated = 0
    for row in pitch_lb:
        key = (row['pitcher'], row['team'], row['pitchType'])
        if key in stuff_lookup:
            row['stuffScore'] = stuff_lookup[key]
            n_updated += 1
        else:
            row['stuffScore'] = None

    # Percentile ranks
    pt_groups = defaultdict(list)
    for row in pitch_lb:
        if row.get('stuffScore') is not None:
            pt_groups[row['pitchType']].append(row['stuffScore'])

    for row in pitch_lb:
        pt = row['pitchType']
        if row.get('stuffScore') is not None and pt in pt_groups:
            vals = pt_groups[pt]
            below = sum(1 for v in vals if v < row['stuffScore'])
            equal = sum(1 for v in vals if v == row['stuffScore'])
            row['stuffScore_pctl'] = round((below + 0.5 * equal) / len(vals) * 100)
        else:
            row['stuffScore_pctl'] = None

    with open(pl_path, 'w') as f:
        json.dump(pitch_lb, f)
    print(f"\nUpdated {n_updated}/{len(pitch_lb)} pitch leaderboard rows with stuffScore")

    # Also save CSV
    output_path = os.path.join(SCRIPT_DIR, 'pitcher_stuff_v10.csv')
    agg_out = agg.round(1)
    agg_out.to_csv(output_path, index=False)
    print(f"Saved pitcher × pitch type scores to {output_path}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stuff+ v10 Training & Scoring')
    parser.add_argument('--csv', help='Score a CSV file instead of reading from sheets')
    parser.add_argument('--score-only', action='store_true', help='Score using saved artifacts')
    parser.add_argument('--compare', action='store_true', help='Compare v9 vs v10 side by side')
    parser.add_argument('--no-update', action='store_true', help='Do not update pitch_leaderboard.json')
    args = parser.parse_args()

    # Load MVN models
    print("Loading MVN models...")
    mvn_models = load_mvn_models()
    print(f"  Loaded MVN models for {len(mvn_models)} pitch types")

    # Load data
    if args.csv:
        all_pitches = load_pitch_data_from_csv(args.csv)
    else:
        print("\nLoading pitch data from Google Sheets...")
        all_pitches = load_pitch_data_from_sheets()

    # Engineer features (superset for both v9 and v10)
    print("\nEngineering features...")
    df = engineer_features(all_pitches, mvn_models)

    if args.compare:
        # ============================================================
        # COMPARISON MODE: train v9, v10, v10b — compare metrics
        # ============================================================
        configs = [make_v9_config(), make_v10b_config(), make_v10_config()]
        results = {}

        for cfg in configs:
            name = cfg['name']
            models, stats, oof, cv = train_and_evaluate(df, cfg)
            scores = score_pitches(df, models, stats, cfg)
            # Post-scoring normalization
            scored = scores.dropna()
            if len(scored) > 0:
                df_tmp = df.loc[scored.index].copy()
                df_tmp['sp'] = scored
                shift = 100 - df_tmp.groupby(['pitcher', 'team', 'pitch_type'])['sp'].mean().mean()
                scores += shift
                scores.clip(40, 160, inplace=True)
            corrs = validate_and_correlate(df, scores, name)
            results[name] = {'cv': cv, 'corrs': corrs, 'scores': scores}

        # Print outcome-only comparison (rv100 + xwoba — the metrics that matter)
        print(f"\n{'='*80}")
        print(f"  OUTCOME CORRELATIONS: Stuff+ vs Run Value and xwOBA")
        print(f"  (Higher magnitude = better model. rv100 should be negative, xwoba should be negative)")
        print(f"{'='*80}")
        print(f"  {'PT':<4}  {'v9 rv100':>10} {'v10b rv100':>11} {'v10 rv100':>11}"
              f"  {'v9 xwoba':>10} {'v10b xwoba':>11} {'v10 xwoba':>11}")
        print(f"  {'-'*4}  {'-'*10} {'-'*11} {'-'*11}"
              f"  {'-'*10} {'-'*11} {'-'*11}")

        outcome_wins = {n: 0 for n in ['v9', 'v10b', 'v10']}
        n_comp = 0

        for pt in SUPPORTED_PT:
            parts = []
            for outcome in ['rv100', 'xwoba']:
                vals = {}
                for name in ['v9', 'v10b', 'v10']:
                    c = results[name]['corrs'].get(pt, {})
                    vals[name] = c.get(outcome)
                # Find best (most negative for rv100/xwoba)
                valid = {k: v for k, v in vals.items() if v is not None}
                if len(valid) >= 2:
                    best = min(valid, key=lambda k: valid[k])
                    outcome_wins[best] += 1
                    n_comp += 1

            rv = {n: results[n]['corrs'].get(pt, {}).get('rv100') for n in ['v9', 'v10b', 'v10']}
            xw = {n: results[n]['corrs'].get(pt, {}).get('xwoba') for n in ['v9', 'v10b', 'v10']}
            rv_s = {n: f"{v:>+.4f}" if v else "     N/A" for n, v in rv.items()}
            xw_s = {n: f"{v:>+.4f}" if v else "     N/A" for n, v in xw.items()}
            print(f"  {pt:<4}  {rv_s['v9']:>10} {rv_s['v10b']:>11} {rv_s['v10']:>11}"
                  f"  {xw_s['v9']:>10} {xw_s['v10b']:>11} {xw_s['v10']:>11}")

        print(f"\n  Outcome wins (rv100 + xwoba): ", end="")
        for name in ['v9', 'v10b', 'v10']:
            print(f"{name}={outcome_wins[name]}", end="  ")
        print(f"(of {n_comp})")

        # CV component comparison
        print(f"\n  CV Whiff AUC:")
        for pt in SUPPORTED_PT:
            vals = {}
            for name in ['v9', 'v10b', 'v10']:
                m = results[name]['cv'].get(pt, {})
                vals[name] = m.get('whiff_auc')
            if any(v is not None for v in vals.values()):
                s = f"  {pt:<4}"
                for name in ['v9', 'v10b', 'v10']:
                    v = vals[name]
                    s += f"  {name}: {v:.4f}" if v else f"  {name}:  N/A  "
                print(s)

        print(f"\n  CV GB AUC:")
        for pt in SUPPORTED_PT:
            vals = {}
            for name in ['v9', 'v10b', 'v10']:
                m = results[name]['cv'].get(pt, {})
                vals[name] = m.get('gb_auc')
            if any(v is not None for v in vals.values()):
                s = f"  {pt:<4}"
                for name in ['v9', 'v10b', 'v10']:
                    v = vals[name]
                    s += f"  {name}: {v:.4f}" if v else f"  {name}:  N/A  "
                print(s)

        # Top 10 for each
        for name in ['v9', 'v10b', 'v10']:
            df_tmp = df.copy()
            df_tmp['sp'] = results[name]['scores']
            pitcher_agg = df_tmp.dropna(subset=['sp']).groupby(['pitcher', 'team']).agg(
                overall=('sp', 'mean'), n=('sp', 'count')
            ).reset_index()
            print(f"\n  Top 10 Overall Stuff+ ({name}):")
            for _, r in pitcher_agg.nlargest(10, 'overall').iterrows():
                print(f"    {r['pitcher']:30s} {r['team']:4s}  {r['overall']:6.1f}  ({r['n']} pitches)")

        # Final verdict
        best = max(outcome_wins, key=outcome_wins.get)
        print(f"\n{'='*80}")
        print(f"  VERDICT: {best} wins {outcome_wins[best]}/{n_comp} outcome comparisons")
        print(f"{'='*80}")

        return

    if args.score_only:
        art_path = os.path.join(SCRIPT_DIR, 'stuff_models_v10.pkl')
        with open(art_path, 'rb') as f:
            arts = pickle.load(f)
        models = arts['models']
        league_stats = arts['league_stats']
        config = make_v10b_config()
    else:
        config = make_v10b_config()
        models, league_stats, oof_stuff, cv_metrics = train_and_evaluate(df, config)
        save_artifacts(models, league_stats, config)

    # Score
    print("\nScoring pitches...")
    stuff_scores = score_pitches(df, models, league_stats, config)
    scored = stuff_scores.dropna()

    if len(scored) > 0:
        df_tmp = df.loc[scored.index].copy()
        df_tmp['sp'] = scored
        shift = 100 - df_tmp.groupby(['pitcher', 'team', 'pitch_type'])['sp'].mean().mean()
        stuff_scores += shift
        stuff_scores = stuff_scores.clip(40, 160)
        scored = stuff_scores.dropna()
        print(f"Scored {len(scored)}/{len(df)} pitches (shifted {shift:+.1f} to center on 100)")
        print(f"Mean Stuff+: {scored.mean():.1f}, Std: {scored.std():.1f}")

    # Validate
    corrs = validate_and_correlate(df, stuff_scores, "v10b")
    print_validation(corrs, "v10b")

    # Top/bottom pitchers
    df_display = df.copy()
    df_display['sp'] = stuff_scores
    pitcher_agg = df_display.dropna(subset=['sp']).groupby(['pitcher', 'team']).agg(
        overall=('sp', 'mean'), n=('sp', 'count')
    ).reset_index()

    print(f"\n{'='*60}")
    print("TOP 20 PITCHERS BY OVERALL STUFF+")
    print(f"{'='*60}")
    for _, r in pitcher_agg.nlargest(20, 'overall').iterrows():
        print(f"  {r['pitcher']:30s} {r['team']:4s}  Stuff+: {r['overall']:6.1f}  ({r['n']} pitches)")

    print(f"\n{'='*60}")
    print("BOTTOM 20 PITCHERS BY OVERALL STUFF+")
    print(f"{'='*60}")
    for _, r in pitcher_agg.nsmallest(20, 'overall').iterrows():
        print(f"  {r['pitcher']:30s} {r['team']:4s}  Stuff+: {r['overall']:6.1f}  ({r['n']} pitches)")

    # Per-pitch-type leaders
    pt_agg = df_display.dropna(subset=['sp']).groupby(['pitcher', 'team', 'pitch_type']).agg(
        stuff_mean=('sp', 'mean'), stuff_std=('sp', 'std'), n_pitches=('sp', 'count')
    ).reset_index()

    print(f"\n{'='*60}")
    print("TOP 5 PER PITCH TYPE")
    print(f"{'='*60}")
    for pt in SUPPORTED_PT:
        pt_rows = pt_agg[(pt_agg['pitch_type'] == pt) & (pt_agg['n_pitches'] >= MIN_PITCHES_TRAIN)]
        if len(pt_rows) == 0:
            continue
        top5 = pt_rows.nlargest(5, 'stuff_mean')
        print(f"\n  {pt}:")
        for _, r in top5.iterrows():
            print(f"    {r['pitcher']:30s} {r['team']:4s}  Stuff+: {r['stuff_mean']:6.1f}  ({r['n_pitches']} pitches)")

    # Update leaderboard
    if not args.no_update:
        update_pitch_leaderboard(df, stuff_scores)


if __name__ == '__main__':
    main()
