"""train_stuff.py — Stuff+ v11 trainer.

A from-scratch rebuild validated against v10 (scripts/stuff_lab*.py). Key
changes that won the experiments:
  - ONE pooled model across pitch types (not 9 per-type models)
  - ONE learned luck-neutral xRV target (not hand-weighted whiff/GB/contact)
  - lean feature set (no SSW residuals, no location, no overfit-prone extras)

Out-of-fold it predicts future luck-neutral run prevention at ~0.23, above the
persistence ceiling (~0.195) and far above v10's architecture (~0.02-0.06 in the
same harness). Reliability ~0.87.

Offline trainer (xgboost is fine here): trains on the cleaned pitch pickle,
scores every MLB pitch OUT-OF-FOLD (pitcher-grouped, so no pitcher's own
outcomes train the model that scores him), standardizes to 100 +/- 10 PER
PITCH TYPE, aggregates to (pitcher, team, pitch_type), and (optionally, with
--inject) writes stuffScore + stuffScore_pctl into the pitch leaderboard.

Usage:
    python3 stuff_plus/train_stuff.py            # train + save bundle/CSV
    python3 stuff_plus/train_stuff.py --inject    # also write into leaderboard
"""
import os, sys, math, json, pickle, argparse, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(DATA, 'all_pitches_rs_cache.pkl')

# Live 2026 FanGraphs Guts — read from metadata_rs.json (written by
# pipeline_fetch.scrape_guts each process_data run) so the BIP target matches
# whatever Guts the leaderboard used, and refreshes daily while the season is
# live. Same source as Cards._load_guts. Fallback only on a missing/broken file.
def _load_live_guts():
    try:
        with open(os.path.join(DATA, 'metadata_rs.json')) as _f:
            _g = json.load(_f).get('gutsConstants') or {}
        _lg, _sc = _g.get('lgWOBA'), _g.get('wOBAScale')
        if _lg and _sc:
            return float(_lg), float(_sc)
    except Exception:
        pass
    return 0.3172, 1.2343

LG_WOBA, WOBA_SCALE = _load_live_guts()
sys.path.insert(0, ROOT)
try:
    from pipeline_utils import AAA_TEAMS
except Exception:
    AAA_TEAMS = {'ROC'}

SUPPORTED = {'FF', 'SI', 'SL', 'CH', 'ST', 'FC', 'CU', 'FS', 'SV', 'KC', 'KN'}
# Types whose per-type anchor pool is too thin to trust borrow a donor pool
# for (mu, sd). KN: one knuckleballer, offspeed family per Loc+ precedent.
# SV: 25 qualified units; donor is SL alone (2026-07-18, Wally) — SV's unit
# population sits on SL's (class avg 99.2 under SL anchors, sd ratio 0.94)
# while ST's runs ~0.47 SD hotter (SV class avg 94.6 under ST, 97.3 under
# SL+ST). Also makes SL<->SV retags grade-invariant (shared anchor).
ANCHOR_BORROW = {'KN': ('CH', 'FS'), 'SV': ('SL',)}
FB_TYPES = {'FF', 'SI', 'FC'}

# Prior-season training data (season-blocked scheme; see main()). The pickle
# is static (2025 is finished); 2025 FG Guts constants for its targets.
PRIOR_PKL = os.path.join(DATA, '_pitches2025_training.pkl')
PRIOR_LG_WOBA, PRIOR_WOBA_SCALE = 0.3131, 1.2317

# 2021-24 volume prior (2026-07-13): public-Statcast reconstructions
# (scripts/build_historical_training_set.py — public tags are fine, the
# model is pitch-type agnostic so labels only feed the fastball anchor,
# computed within-season). Adopted after scripts/multi_season_retrain.py
# showed FULL_21_25 beats the 2025-only prior on all three objectives
# (reliab .864→.870, pred .309→.310, desc .370→.375) and shrinks the
# off-manifold region the low-support flag guards. Per-season FG Guts
# center each season's BIP target in its own run environment.
HIST_YEARS = (2021, 2022, 2023, 2024)
HIST_PKL = os.path.join(DATA, '_pitches{year}_training.pkl')
HIST_GUTS = {
    2021: (0.314, 1.209), 2022: (0.310, 1.259),
    2023: (0.318, 1.204), 2024: (0.310, 1.242),
}


def _harmonize_tags(prior, current):
    """Map prior-season tags to the pitcher's nearest current-season tag when
    the (velo, ivb, hb) centroids say the label flipped across seasons
    (SL<->FC, CH<->FS, CU<->SV boundary churn — 85 units / 4.3% of shared
    pitcher-tags in the 2025 audit). Mutates and returns `prior`."""
    from collections import defaultdict as _dd
    import math as _m

    def cents(pitches, mv_keys):
        acc = _dd(lambda: [0.0, 0.0, 0.0, 0])
        for p in pitches:
            v = sf(p.get('Velocity'))
            iv = sf(p.get(mv_keys[0]))
            hb = sf(p.get(mv_keys[1]))
            thr = p.get('Throws')
            if None in (v, iv, hb) or thr not in ('L', 'R'):
                continue
            s = 1.0 if thr == 'R' else -1.0
            a = acc[(p.get('Pitcher'), p.get('Pitch Type'))]
            a[0] += v; a[1] += iv; a[2] += hb * s; a[3] += 1
        return {k: (a[0]/a[3], a[1]/a[3], a[2]/a[3]) for k, a in acc.items() if a[3] >= 30}

    c_p = cents(prior, ('xIndVrtBrk', 'xHorzBrk'))
    c_c = cents(current, ('xIndVrtBrk', 'xHorzBrk'))
    by_pitcher = _dd(dict)
    for (pit, tag), c in c_c.items():
        by_pitcher[pit][tag] = c
    flips = {}
    for (pit, tag_p), c in c_p.items():
        tags = by_pitcher.get(pit)
        if not tags:
            continue
        best, bd = None, 1e9
        for tag_c, c2 in tags.items():
            d = _m.sqrt(((c[0]-c2[0])/1.5)**2 + ((c[1]-c2[1])/2.5)**2 + ((c[2]-c2[2])/2.5)**2)
            if d < bd:
                bd, best = d, tag_c
        if best != tag_p and bd <= 1.5:
            flips[(pit, tag_p)] = best
    n = 0
    for p in prior:
        k = (p.get('Pitcher'), p.get('Pitch Type'))
        if k in flips:
            p['Pitch Type'] = flips[k]
            n += 1
    print(f'  tag harmonization: {len(flips)} flipped units, {n} prior pitches relabeled')
    return prior
# 2026-07-02 config (validated: scripts/phase2b_stuff_experiments.py):
# - rel_x (hand-normalized release side) added ALONE — earlier labs only
#   tested it bundled with HAA; solo it wins (rel +0.004, pred +0.007) and
#   every public reference model carries it
# - movement features are the DENSITY-ADJUSTED xIndVrtBrk/xHorzBrk (flat on
#   the harness, adopted for coherence: the model's ivb is now the same
#   quantity the site displays, and altitude pitchers stop eating a
#   systematic stuff penalty for their home air)
# - monotone velocity constraint (rel +0.008, pred flat): more velo, same
#   everything else, never grades worse — kills sparse-region artifacts
# 2026-07-07 feature-weighting audit (scripts/stuff_weight_audit.py): rel_z
# DROPPED — redundant with arm_angle (r=0.78) and actively harmful; removing
# it improves pred future xRV 0.304 -> 0.315 (8-fold, seeds 0/1/2), desc
# +0.007, reliab +0.002, and helps the no-arm ROC companion (pred 0.258 ->
# 0.272). It double-penalized low-slot arms (Kevin Kelly SI 96 -> 111; SI
# desc r 0.461 -> 0.488). RelPosZ stays in build_df rows/row-filter so the
# training sample is unchanged. All other features re-confirmed: drop-one
# hb_diff/spin_rate gains at 4-fold were noise at 8-fold; platoon_same keeps
# real descriptive value; arm_angle is tied-most load-bearing (pred -0.049
# when dropped) despite a small SHAP share.
# ── v12 config (ADOPTED 2026-08-09, per Wally) ──
# Three changes vs v11, each validated on the full-config paired 2026 harness
# (scripts/stuff_feature_battery_2026_08.py + stuff_combo_battery.py) and the
# LOSO 2021-2025 replicate gate (scripts/stuff_features_loso.py):
#   1. cross/cross_abs replace axis_dev/axis_dev_abs — the same SSW quantity
#      restated in release-axis-frame INCHES (linear in movement, stable for
#      gyro sliders where circular degrees have sd ~40).
#   2. velo_diff masked to None on FF/SI — the gap off the primary fastball
#      is real signal for cutters/breaking/offspeed (masking it there crashed
#      breaking pred -0.08) but actively harmful bookkeeping on the true
#      fastballs themselves (masking FF/SI: fastball pred 0.217 -> 0.266).
#   3. vaa/vaa_diff are nVAA — location-adjusted via frozen per-type slopes
#      (NVAA_SLOPES below). Raw VAA doubled as a pitch-height proxy, i.e.
#      location leakage in a skill-isolation model (locIndep 0.39 -> 0.29).
# Combined 2026 harness: reliab 0.8818 -> 0.9685, pred 0.2878 -> 0.3057,
# desc 0.3821 -> 0.3496 (the desc drop is the evicted location component).
# LOSO: CROSS_VD beat shipped on pitcher-level fut_r in 4/5 held-out seasons
# (rel 5/5); adding nVAA was prediction-neutral across replicates (3/5, mean
# +0.007) while gaining +0.05 reliability in all five.
BASE_FEATS = ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff',
              'spin_rate', 'extension', 'arm_angle', 'vaa', 'vaa_diff',
              'rel_x', 'cross', 'cross_abs', 'kin_eff__ff']

# v13 (2026-08-14, per Wally): kin_eff__ff = measured active-spin fraction
# (KinEff), TYPE-GATED to four-seamers — real on FF, NaN elsewhere. The
# per-type atlas (scripts/stuff_pertype_atlas.py) showed kin_eff carries
# reliance for every type but is only IRREPLACEABLE on FF (ride quality
# beyond what ivb/spin/axis imply); gated adds on CU/SI/FC and the global
# add all failed the LOSO gate, FF-gated passed (fut 4/5, mean +0.0042;
# 2026 battery: fastball-family desc .3052->.3137, overall flat). Missing
# values on FF impute like arm angle: pitcher's own FF mean, then the
# frame's FF mean, then this frozen league constant — so fresh games
# (sidecar lag) and ROC arms (no MiLB kinematics) never ride a
# near-unconstrained missing branch (the Jackson Kent debut lesson).
# Constant measured on 2021-2025 pooled FF (1.16M pitches, seasonal means
# 0.689-0.721).
KIN_FF_LEAGUE_MEAN = 0.7038

# Frozen nVAA constants: per-type OLS slope of VAA on PlateZ + league mean
# PlateZ, measured on 2021-2026 pooled (4.05M pitches; 2026-08-09). Frozen
# like build_historical_training_set's release-point calibration so every
# scoring path (trainer, score-only, ROC, Cards scratch, explain pages) is
# identical with no bundle plumbing. Types below the 2000-pitch fit floor
# (KN) keep raw VAA. Re-measure only deliberately; slopes varied < 0.03
# across 4-season subsets in the LOSO fits.
NVAA_SLOPES = {
    'CH': (1.0556, 1.8315), 'CU': (1.0010, 1.8314), 'FC': (1.1703, 2.4074),
    'FF': (1.0623, 2.8193), 'FS': (1.0604, 1.7300), 'KC': (0.9659, 1.7575),
    'SI': (1.1030, 2.3694), 'SL': (1.1117, 1.8846), 'ST': (1.1356, 1.9935),
    'SV': (1.0251, 1.8603),
}
# velo_diff is masked (None) on these types — see adoption note 2 above.
#
# WHAT THE PROBE RETURNED (scripts/stuff_velodiff_probe.py, re-run 2026-08-12).
# The question was Wally's: "a curveball should not be dinged for being thrown
# hard; separation logic belongs to changeups/splitters that mimic fastballs."
#
# STAGE 1 — +1 mph counterfactual, Stuff+ points per pitch. "joint" is the real
# within-pitcher case (same fastball, harder secondary, so the gap shrinks too):
#
#     type       n     joint(+velo,+gap)   velo-only   gap term
#     CH     54857           +0.20            +2.71      -2.51
#     FS     21137           +1.53            +3.30      -1.77
#     CU     40666           +2.64            +3.35      -0.72
#     FC     42090           +2.66            +3.54      -0.88
#     ST     49647           +2.91            +4.24      -1.34
#     SL     67282           +3.69            +5.64      -1.95
#     FF    163733           +3.73            +3.73       0.00   <- masked
#     SI     87365           +3.51            +3.51       0.00   <- masked
#
# The concern does NOT materialize: every joint value is positive, so a harder
# breaking ball still grades better, just by less than raw velocity alone would
# give. The curveball is the LEAST penalized secondary (-0.72). The pitch where
# separation nearly cancels velocity is the CHANGEUP (+0.20 net, the gap eating
# 93% of the gain) — which is the right baseball answer, since velo separation
# is most of what a changeup is. FF/SI show exactly 0.00, confirming the mask.
#
# STAGE 2 — the learned shape, mean SHAP in Stuff+ points by velo_diff bin:
#
#     family      [-18,-14) [-14,-11)  [-11,-8)   [-8,-5)   [-5,-2)   [-2,1)
#     breaking      +12.26    +12.29     +7.71     +2.82     -5.86        -
#     offspeed      +10.20    +11.61     +6.14     +1.77     -5.19        -
#     cutter             -         -     +8.51     +0.71     -6.55   -10.62
#     fastball           -         -         -         -         -        -
#
# Monotone, and worth ~18 Stuff+ points across the breaking-ball range. The
# empty fastball row is the mask. This table is also the fastest way to explain
# an individual arm's grades: a secondary's bin usually dominates its number
# (Jackson Kent 2026-08-12 — CU at -13.6 sits in the +12.29 bin and grades his
# best at 91, while his SL -6.3 and CH -6.9 sit in the +2.82/+1.77 bin at 89
# and 82).
#
# Superseded/extends the 2026-07-08 per-type earned-value check
# (scripts/velo_diff_by_type.py), which reached the same conclusion by
# permutation: FS and CH earn the most from the gap, SL almost nothing (its
# value is shape/gyro). Note that check's method warning — SHAP SHARE
# overstates slider velo_diff; confirm with permutation or a counterfactual,
# which is what Stage 1 above is.
VD_MASK_TYPES = {'FF', 'SI'}


def _nvaa(pt, vaa, plate_z):
    """Location-adjusted VAA; falls back to raw when unadjustable."""
    if vaa is None:
        return None
    sl = NVAA_SLOPES.get(pt)
    if sl is None or plate_z is None:
        return vaa
    return vaa - sl[0] * (plate_z - sl[1])
# axis_dev/axis_dev_abs (2026-07-14): the seam-shifted-wake proxy — signed,
# hand-normalized circular OTilt-RTilt deviation + its magnitude, per pitch
# (Wally's tilt conventions; see _axis_dev_deg). ADOPTED after the
# definitive 100%-coverage retest (scripts/axis_retest.py: pred .2942 ->
# .2979, desc flat, reliab -.0011). NOTE the coverage lesson: the same
# feature FAILED at 80% prior coverage (2025 season missing SpinAxis —
# scripts/v12_round2_battery.py, pred -.003) because a whole-season feature
# gap acts as an era marker; augment_2025_spinaxis.py closed the gap
# (99.6% via strict game_pk-constrained fingerprint re-join). Never test a
# feature with a season-shaped hole in it. The research loop called this
# one: persistent xRVOE overperformers (Cruz FS, Lee/Hoffman SL) live at
# extreme |axis_dev| in both directions.
# Depth history: 4 -> 6 at the ~1.08M-row scale (2026-07-04,
# scripts/stuff_hp_retune.py: pred 0.265 -> 0.298 within the -0.010
# reliability guard); 6 -> 7 at the 3.5M-row 2021-25-prior scale
# (2026-07-14, scripts/stuff_v12_battery.py + depth7_confirm.py: full-prior
# 8-fold pred 0.2823 -> 0.2942, desc 0.3815 -> 0.3922, reliability
# 0.8751 -> 0.8717 — well within guard; depth 8 decayed desc and was
# rejected). Each 3-4x of training data has bought one level of depth.
# Recency weighting of prior rows tested and rejected (uniform 1.0 stays).
TUNED = dict(max_depth=7, n_estimators=800, learning_rate=0.025, min_child_weight=10,
             reg_lambda=1.5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist')
MONO_FEAT = 'velocity'


def _params_for(X):
    """TUNED + the monotone velocity constraint mapped to X's columns."""
    p = dict(TUNED)
    p['monotone_constraints'] = tuple(-1 if c == MONO_FEAT else 0 for c in X.columns)
    return p
# Pitcher-level standardization: 10 points per between-pitcher SD, mean shrunk
# toward the qualified-pool mean by K_SHRINK pseudo-pitches, qualify at QUAL_N.
# COHERENT CANON (2026-07-18, per Wally): every displayed value is the PLAIN
# AVERAGE of per-pitch grades — no shrink. Validated empirically (split-half +
# season-blocked MAE): shrinkage HURT Stuff+ at every sample size (model
# outputs are already regularized; K=100 only added bias). Small samples are
# handled by qualification gates at render time, not by bending the numbers.
# Historical value: K_SHRINK was 100 through 2026-07-18.
K_SCALE, K_SHRINK, QUAL_N = 10, 0, 50

def sf(x):
    try: return float(x)
    except (TypeError, ValueError): return None

# Cutter-primary pitchers: their cutter IS their fastball, so anchor the
# differential features to the FC even though they throw a lightly-used
# sinker. Hand-curated by Wally 2026-07-05 from the 41 arms whose anchor the
# true-fastball rule would otherwise flip (all three have no four-seam and a
# cutter thrown far more than any sinker). Name-keyed so it applies across
# seasons and to daily/scratch cards. Extend as needed.
FC_ANCHOR_PITCHERS = {'Jansen, Kenley', 'Maton, Phil', 'Spence, Mitch'}

# ── axis deviation (SSW proxy) + spin efficiency helpers ──
# OTilt-RTilt circular difference in degrees, per pitch, Wally's tilt
# conventions (Pitcher2026.spin_axis_to_tilt / calculate_break_tilt):
# RTilt hours = spin_axis/30 - 6; OTilt = atan2(HB_raw, IVB_raw) clockwise
# from 12:00 with arm-side HB positive. Sheets carry RTilt/OTilt as "H:MM"
# strings; augmented training pickles carry SpinAxis degrees + raw movement.
def _tilt_min(val):
    """Clock value -> minutes (0-719). Accepts 'H:MM' strings or numeric
    minutes; None-safe."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) % 720.0
    if isinstance(val, str) and ':' in val:
        try:
            h, m = val.strip().split(':')[:2]
            return (int(h) % 12) * 60.0 + int(m)
        except (ValueError, IndexError):
            return None
    return None


def _axis_dev_deg(p):
    """Signed circular OTilt-RTilt in degrees, [-180, 180). NOT hand-signed
    (caller applies the throws sign)."""
    rt = _tilt_min(p.get('RTilt'))
    if rt is None:
        sa = sf(p.get('SpinAxis'))
        rt = ((sa / 30.0 - 6.0) % 12.0) * 60.0 if sa is not None else None
    ot = _tilt_min(p.get('OTilt'))
    if ot is None:
        ivb_r, hb_r = sf(p.get('IndVertBrk')), sf(p.get('HorzBrk'))
        if ivb_r is not None and hb_r is not None and not (ivb_r == 0 and hb_r == 0):
            ot = (math.degrees(math.atan2(hb_r, ivb_r)) % 360.0) * 2.0
    if rt is None or ot is None:
        return None
    d = (ot - rt) % 720.0
    if d >= 360.0:
        d -= 720.0
    return d * 0.5


def _arm_means(pitches):
    """Per-pitcher arm-angle means {(pitcher, pitch_type): deg} + {pitcher: deg}
    from a DIFFERENT frame than the one being built. Used as the debut-pitcher
    fallback in build_df (see arm_fallback there)."""
    pt = defaultdict(lambda: [0.0, 0])
    al = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        aa = sf(p.get('ArmAngle'))
        if aa is None:
            continue
        pit, pt0 = p.get('Pitcher'), p.get('Pitch Type')
        pt[(pit, pt0)][0] += aa; pt[(pit, pt0)][1] += 1
        al[pit][0] += aa; al[pit][1] += 1
    return {'pt': {k: s / n for k, (s, n) in pt.items() if n},
            'all': {k: s / n for k, (s, n) in al.items() if n}}

def build_df(pitches, prefer_true_fastball=True, arm_fallback=None):
    # pass 1: per-pitcher primary fastball reference (handedness-normalized).
    # VAA gets its own count: a pitch missing VAA must not dilute the mean
    # toward 0 by incrementing the shared n while contributing 0.0.
    fb = defaultdict(lambda: defaultdict(lambda: {'v': 0.0, 'iv': 0.0, 'hb': 0.0, 'vaa': 0.0, 'n': 0, 'n_vaa': 0}))
    for p in pitches:
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        if pt not in FB_TYPES or thr not in ('L', 'R'): continue
        v, iv, hb, vaa = sf(p.get('Velocity')), sf(p.get('IndVertBrk')), sf(p.get('HorzBrk')), sf(p.get('VAA'))
        vaa = _nvaa(pt, vaa, sf(p.get('PlateZ')))   # v12: reference is nVAA too
        if None in (v, iv, hb): continue
        s = 1.0 if thr == 'R' else -1.0
        a = fb[(p.get('Pitcher'), thr)][pt]
        a['v'] += v; a['iv'] += iv; a['hb'] += hb * s; a['n'] += 1
        if vaa is not None:
            a['vaa'] += vaa; a['n_vaa'] += 1
    primary = {}
    for k, bt in fb.items():
        # Reference = most-thrown TRUE fastball (FF/SI); the cutter (FC) is a
        # reference ONLY when the pitcher throws neither a four-seam nor a
        # sinker (2026-07-05, per Wally, validated on the season-blocked
        # harness). A secondary's separation is most meaningful vs the
        # velocity anchor hitters gear for, not vs a movement-y cutter.
        if prefer_true_fastball:
            if k[0] in FC_ANCHOR_PITCHERS and 'FC' in bt:
                cand = {'FC': bt['FC']}
            else:
                cand = {pt: d for pt, d in bt.items() if pt in ('FF', 'SI')} or bt
        else:
            cand = bt
        sel = max(cand, key=lambda pt: cand[pt]['n'])
        b = cand[sel]; n = b['n']
        primary[k] = {'v': b['v']/n, 'iv': b['iv']/n, 'hb': b['hb']/n,
                      'vaa': (b['vaa']/b['n_vaa']) if b['n_vaa'] else None}

    # Per-pitcher arm-angle averages, used as a real-time placeholder when arm
    # angle hasn't backfilled yet (it lags games ~1-2 days). Arm angle is nearly
    # constant per pitcher, so his own per-pitch-type average (or overall average
    # for a brand-new pitch) is essentially the real value; the actual number
    # replaces it on the next run after backfill. (STALE-NOTE FIXED 2026-07-28:
    # ROC arm angle is 96-97% populated since the minors Statcast backfill, so
    # ROC rows WITH arm angle score through the full model; the no-arm
    # companion is now only the fallback for the few rows still missing it —
    # see the per-pitch _arm mask in the ROC scoring block.)
    arm_pt = defaultdict(lambda: [0.0, 0]); arm_all = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        aa = sf(p.get('ArmAngle'))
        if aa is None: continue
        pit = p.get('Pitcher'); pt0 = p.get('Pitch Type')
        arm_pt[(pit, pt0)][0] += aa; arm_pt[(pit, pt0)][1] += 1
        arm_all[pit][0] += aa; arm_all[pit][1] += 1
    def _arm_placeholder(pit, pt0):
        a = arm_pt.get((pit, pt0))
        if a and a[1]: return a[0] / a[1]
        a = arm_all.get(pit)
        return a[0] / a[1] if (a and a[1]) else None

    rows = []
    for p in pitches:
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'): continue
        v, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        # density-adjusted movement (pipeline_fetch backfills these from raw
        # when the adjustment hasn't landed, so coverage matches raw)
        iv, hb_raw = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        vaa_raw, ext = sf(p.get('VAA')), sf(p.get('Extension'))
        vaa = _nvaa(pt, vaa_raw, sf(p.get('PlateZ')))   # v12: nVAA
        arm, rel_z = sf(p.get('ArmAngle')), sf(p.get('RelPosZ'))
        rel_x_raw = sf(p.get('RelPosX'))
        if arm is None:                       # real-time placeholder until backfill
            arm = _arm_placeholder(p.get('Pitcher'), pt)
        if None in (v, iv, hb_raw, vaa, ext, rel_z, rel_x_raw): continue
        s = 1.0 if thr == 'R' else -1.0
        hb = hb_raw * s
        rel_x = rel_x_raw * s
        fbref = primary.get((p.get('Pitcher'), thr))
        if fbref:
            velo_diff = v - fbref['v']; ivb_diff = iv - fbref['iv']
            hb_diff = hb - fbref['hb']
            vaa_diff = (vaa - fbref['vaa']) if fbref['vaa'] is not None else None
        else:
            velo_diff = ivb_diff = hb_diff = vaa_diff = None
        velo_diff_raw = velo_diff
        if pt in VD_MASK_TYPES:      # v12: gap masked on true fastballs
            velo_diff = None
        desc = p.get('Description', '')
        is_bip = desc == 'In Play'
        xw, re = sf(p.get('xwOBA')), sf(p.get('RunExp'))
        # NOTE: the BIP branch is deliberately NOT count-anchored (unlike the
        # displayed xRV and the SD+/CT+ cell tables). Tested 2026-07-03:
        # anchoring the target dropped reliability 0.876->0.863 and pred
        # 0.213->0.175 — same failure mode as Loc+ (the count-state term
        # correlates with pitch usage/count-mix, which is contamination for
        # a skill-isolation model). Scope rule: anchor value-accounting and
        # decision metrics; never skill-isolation models.
        if is_bip and xw is not None:
            target = (xw - LG_WOBA) / WOBA_SCALE
        elif re is not None:
            target = -re
        else:
            target = None
        # v12 candidates (emitted always; BASE_FEATS decides what trains):
        # axis_dev = hand-signed OTilt-RTilt circular deviation (SSW proxy),
        # spin_eff = total raw break per kRPM (movement the spin doesn't
        # "pay for" — seam effects / efficient spin).
        _dev = _axis_dev_deg(p)
        ivb_r = sf(p.get('IndVertBrk'))
        hb_r = sf(p.get('HorzBrk'))
        if ivb_r is None: ivb_r = iv
        if hb_r is None: hb_r = hb_raw
        spin_eff = (math.hypot(ivb_r, hb_r) / (spin / 1000.0)
                    if (spin and spin > 0 and ivb_r is not None and hb_r is not None)
                    else None)
        # 2026-08-09 battery candidates (emitted always; BASE_FEATS decides
        # what trains): release-axis frame per the xmove review — cross =
        # hand-normalized break perpendicular to the measured release axis
        # (inches; gyro-independent SSW measure, stable where axis_dev
        # degrees blow up on gyro sliders), ax_sin/ax_cos = the axis
        # direction itself, hand-mirrored. Axis from SpinAxis degrees
        # (priors) or the sheets' RTilt clock (2026). Convention matches
        # scripts/xmove_activespin_gain.py: th=0 is pure backspin.
        _sa = sf(p.get('SpinAxis'))
        if _sa is None:
            _rtm = _tilt_min(p.get('RTilt'))
            if _rtm is not None:
                _sa = ((_rtm / 60.0) + 6.0) * 30.0 % 360.0
        _cross = _ax_sin = _ax_cos = None
        if _sa is not None:
            _th = math.radians((_sa - 180.0) % 360.0) * s
            _ax_sin, _ax_cos = math.sin(_th), math.cos(_th)
            _cross = -iv * _ax_sin + hb * _ax_cos
        rows.append({
            'pitcher': p.get('Pitcher'), 'team': p.get('PTeam'), 'throws': thr,
            'date': p.get('Game Date'),
            'sheet_tab': p.get('_sheet_tab'), 'sheet_row': p.get('_sheet_row'),
            'pitch_type': pt, 'platoon_same': 1 if bats == thr else 0,
            'velocity': v, 'ivb': iv, 'hb': hb, 'velo_diff': velo_diff,
            'pid': p.get('PitchID'),   # per-pitch join key (xRVOE loc join)
            'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'spin_rate': spin,
            'extension': ext, 'arm_angle': arm, 'rel_z': rel_z, 'rel_x': rel_x,
            'vaa': vaa, 'vaa_diff': vaa_diff, 'target_xrv': target,
            'vaa_raw': vaa_raw, 'velo_diff_raw': velo_diff_raw,
            'rv_raw': (-re) if re is not None else None,  # actual RV incl. BIP luck (RVOE)
            'axis_dev': (_dev * s) if _dev is not None else None,
            'axis_dev_abs': abs(_dev) if _dev is not None else None,
            'spin_eff': spin_eff,
            'cross': _cross,
            'cross_abs': abs(_cross) if _cross is not None else None,
            'ax_sin': _ax_sin, 'ax_cos': _ax_cos,
            # passthroughs for the battery's post-hoc transforms (location-
            # adjusted approach angles, HAA reconstruction validation)
            'plate_x': sf(p.get('PlateX')), 'plate_z': sf(p.get('PlateZ')),
            'haa_meas': sf(p.get('HAA')),
            # per-pitch 9P kinematics (augment_kinematics / the 2026 sidecar;
            # None on rows not yet backfilled)
            'kin_eff': sf(p.get('KinEff')), 'kin_dev': sf(p.get('KinDev')),
            'kin_cd': sf(p.get('KinCd')),
        })
    out = pd.DataFrame(rows)
    # Arm-angle imputation (2026-07-18, per Wally): ArmAngle arrives ~2 days
    # after the game via the Savant supplement. While it's missing, use the
    # pitcher's average arm angle for that pitch type (fallback: his overall
    # average). Replaced by the actual value on the first run after the
    # supplement lands.
    #
    # arm_fallback (2026-08-13, per Wally): a DEBUT pitcher has no MLB history
    # to impute from, so his whole first game rode XGBoost's missing-value
    # branches — Jackson Kent 2026-08-12 graded CU 22 / FF 113 until the
    # supplement landed. Pass _arm_means(roc_pitches) and rows still NaN after
    # both same-frame fills use the pitcher's ROC/AAA arm angle instead (same
    # Savant measurement, r=0.991 / 1.4 deg MAD across levels — see the ROC
    # scoring block). Own-frame MLB always wins by construction: the fallback
    # only touches rows with zero MLB arm angle anywhere (Jake Bird's handful
    # of ROC pitches never override his NYY average).
    if len(out) and out['arm_angle'].isna().any():
        _n_miss = int(out['arm_angle'].isna().sum())
        by_pt = out.groupby(['pitcher', 'pitch_type'])['arm_angle'].transform('mean')
        by_p = out.groupby('pitcher')['arm_angle'].transform('mean')
        out['arm_angle'] = out['arm_angle'].fillna(by_pt).fillna(by_p)
        _n_fb = 0
        if arm_fallback and out['arm_angle'].isna().any():
            na = out['arm_angle'].isna()
            fill = [arm_fallback['pt'].get((pit, pt0),
                    arm_fallback['all'].get(pit, np.nan))
                    for pit, pt0 in zip(out.loc[na, 'pitcher'],
                                        out.loc[na, 'pitch_type'])]
            fill = pd.Series([np.nan if v is None else v for v in fill],
                             index=out.index[na], dtype='float64')
            _n_fb = int(fill.notna().sum())
            out.loc[na, 'arm_angle'] = fill
        _n_left = int(out['arm_angle'].isna().sum())
        print(f'  arm-angle impute: {_n_miss - _n_left}/{_n_miss} missing filled '
              f'from pitcher averages ({_n_fb} of those from the ROC/AAA '
              f'fallback; {_n_left} left to model-missing)')
    # kin_eff__ff (v13): typed emission + arm-angle-style imputation. Non-FF
    # rows are NaN BY DESIGN (that is the gating the battery validated); only
    # FF rows impute, so the feature never rides the missing branch where the
    # training data would leave it unconstrained.
    if len(out):
        out['kin_eff'] = pd.to_numeric(out['kin_eff'], errors='coerce')
        out['kin_eff__ff'] = np.where(out['pitch_type'] == 'FF',
                                      out['kin_eff'], np.nan)
        _ff = (out['pitch_type'] == 'FF').values
        _miss = _ff & out['kin_eff__ff'].isna().values
        if _miss.any():
            by_p = out.groupby('pitcher')['kin_eff__ff'].transform('mean')
            out.loc[_miss, 'kin_eff__ff'] = by_p[_miss]
            _frame = out.loc[_ff, 'kin_eff__ff'].mean()
            _fill = _frame if np.isfinite(_frame) else KIN_FF_LEAGUE_MEAN
            _still = _ff & out['kin_eff__ff'].isna().values
            out.loc[_still, 'kin_eff__ff'] = _fill
            print(f'  kin_eff__ff impute: {int(_miss.sum())} FF pitches missing '
                  f'({int(_still.sum())} to frame/league mean '
                  f'{_fill:.3f})')
    return out


KIN_SIDECAR = os.path.join(DATA, 'kinematics_2026_sidecar.pkl')


def apply_kin_sidecar(pitches, path=KIN_SIDECAR):
    """Fill KinEff/KinDev/KinCd on pitch dicts from the PitchID-keyed 2026
    sidecar (built by scripts/build_kinematics_2026.py, refreshed in CI).
    Prior-season pickles carry kinematics natively; this is for 2026 pitches,
    whose sheet-built cache has no kinematics columns. Returns the number of
    pitches filled, or -1 if the sidecar file is absent (callers decide how
    loudly to fail; retrain aborts, score-only degrades to imputation)."""
    if not os.path.exists(path):
        return -1
    with open(path, 'rb') as f:
        side = pickle.load(f)
    n = 0
    for p in pitches:
        if p.get('KinEff') is not None:
            continue
        v = side.get(p.get('PitchID'))
        if v:
            p['KinEff'], p['KinDev'], p['KinCd'] = v
            n += 1
    return n

def design(df, feats=BASE_FEATS):
    # PITCH-TYPE AGNOSTIC (2026-07-04, per Wally, FanGraphs-style): no pitch-
    # type dummies — every pitch is judged purely on its physics (velocity,
    # movement, release, fastball-relative diffs). Pitch type survives only
    # for display grouping / per-type standardization downstream. Validated
    # on the season-blocked harness (scripts/agnostic_stuff_experiment.py):
    # pred future xRV 0.244 -> 0.265, reliability flat (0.867 -> 0.865),
    # descriptive 0.333 -> 0.339, Rogers SI 126.5 -> 128.1. Also makes
    # training fully immune to cross-year retag drift (labels are no longer
    # model inputs).
    return pd.concat([df[feats].reset_index(drop=True),
                      df[['platoon_same']].reset_index(drop=True)], axis=1)

# Companion model trained on the same MLB data minus arm_angle. Until
# 2026-07-25 this scored ALL of ROC/AAA, whose arm angle was 0% populated. The
# minor-league Statcast Search backfill fixed that, so ROC now uses the full
# model like MLB and this companion is only the fallback for pitches that still
# have no arm angle (games Savant hasn't processed, stale caches).
NOARM_FEATS = [f for f in BASE_FEATS if f != 'arm_angle']

# ── Low-model-support indicator (2026-07-04) ──
# A Stuff+ number is only as good as the training data near the pitch. For each
# MLB (pitcher, team, pitch_type) unit we measure how close its feature centroid
# sits to the data that actually trains the model scoring it: mean z-scored
# BASE_FEATS-space Euclidean distance to the SUPPORT_K nearest pitches in a
# random <=SUPPORT_POOL_MAX subsample of the training pool. The pool EXCLUDES
# the pitcher's own current-season pitches (mirroring production OOF, where his
# own 2026 outcomes never train the model that scores him) but KEEPS his
# prior-season pitches — cross-season identity is real support (2025 Tyler
# Rogers supports 2026 Tyler Rogers, so Rogers must NOT flag). Because the
# subsample dilutes a single pitcher's rows ~20x (which falsely flagged
# Rogers's sweeper despite 249 supporting 2025 pitches), the unit pitcher's
# own prior-season pitches are ALWAYS in his neighbor pool at full density,
# appended per-unit alongside the shared subsample. The worst
# (100 - SUPPORT_FLAG_PCT)% of units flag as low-support; surfaced as
# stuffScore_lowSupport on both leaderboards, a tooltip note on the site, and
# a dagger on season cards. ROC units are NOT scored for support (never in the
# training pool or the support manifold) — always False.
SUPPORT_K = 25
SUPPORT_POOL_MAX = 60000
SUPPORT_FLAG_PCT = 98.5
SUPPORT_SEED = 20260704

def compute_support(df, df_prior):
    """Support score + low_support flag per MLB (pitcher, team, pitch_type).

    Features: BASE_FEATS only (platoon_same excluded — averaging a binary
    usage mix isn't a physical coordinate; kin_eff__ff excluded — it is NaN
    by design off FF, and a median-filled type-gated column would inject a
    constant fake dimension into every non-FF unit's centroid). z-scored by
    the FULL training pool's mean/std; NaN dims imputed at the training
    median.
    """
    feats = [f for f in BASE_FEATS if f != 'kin_eff__ff']
    parts = [df[feats].assign(_pitcher=df['pitcher'].values, _prior=0)]
    if df_prior is not None and len(df_prior):
        parts.append(df_prior[feats].assign(_pitcher=df_prior['pitcher'].values, _prior=1))
    pool = pd.concat(parts, ignore_index=True)
    mu, sd = pool[feats].mean(), pool[feats].std().replace(0, 1.0)
    med = pool[feats].median()
    rng = np.random.RandomState(SUPPORT_SEED)
    if len(pool) > SUPPORT_POOL_MAX:
        idx = rng.choice(len(pool), SUPPORT_POOL_MAX, replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
    Z = ((pool[feats].fillna(med) - mu) / sd).values.astype(np.float32)
    pool_pitcher = pool['_pitcher'].values
    # per-pitcher full-density prior-season rows (appended per unit below)
    own_prior = {}
    if df_prior is not None and len(df_prior):
        Zp = ((df_prior[feats].fillna(med) - mu) / sd).values.astype(np.float32)
        for pit, ix in df_prior.groupby('pitcher').indices.items():
            own_prior[pit] = Zp[ix]

    cent = df.groupby(['pitcher', 'team', 'pitch_type'])[feats].mean().reset_index()
    C = ((cent[feats].fillna(med) - mu) / sd).values.astype(np.float32)
    cent_pitcher = cent['pitcher'].values

    Z2 = (Z ** 2).sum(1)
    support = np.empty(len(cent))
    CH = 256
    for i0 in range(0, len(cent), CH):
        i1 = min(i0 + CH, len(cent))
        Cc = C[i0:i1]
        D = (Cc ** 2).sum(1)[:, None] + Z2[None, :] - 2.0 * (Cc @ Z.T)
        for j in range(i0, i1):
            row = D[j - i0].copy()
            # drop ALL of this pitcher's sampled rows (own-current excluded by
            # design; own-prior re-enters at full density, so drop the sampled
            # copies to avoid double counting)
            row[pool_pitcher == cent_pitcher[j]] = np.inf
            op = own_prior.get(cent_pitcher[j])
            if op is not None:
                dop = ((op - C[j][None, :]) ** 2).sum(1)
                row = np.concatenate([row, dop])
            near = np.partition(row, SUPPORT_K)[:SUPPORT_K]
            support[j] = float(np.sqrt(np.maximum(near, 0.0)).mean())
    cent = cent[['pitcher', 'team', 'pitch_type']].copy()
    cent['support'] = np.round(support, 4)
    thr = float(np.percentile(support, SUPPORT_FLAG_PCT))
    cent['low_support'] = cent['support'] > thr
    n_flag = int(cent['low_support'].sum())
    print(f'  model support: {len(cent)} units, flag threshold {thr:.3f} '
          f'({SUPPORT_FLAG_PCT} pctl), {n_flag} low-support units')
    return cent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inject', action='store_true', help='write stuffScore into the pitch leaderboard')
    ap.add_argument('--dump-pitch-grades', default=None, metavar='PATH',
                    help='write per-pitch Stuff+ grades keyed by sheet position '
                         '("tab\trow" -> grade) for the Sheets write-back '
                         '(scripts/sheets_write_grades.py). Unregressed scale-(i) '
                         'values: 100 + 10*(raw - mu_type)/sd_type, clip 40-180.')
    ap.add_argument('--score-only', action='store_true',
                    help='score with the cached fold models from the saved bundle '
                         '(no training, no prior pickles needed — seconds instead '
                         'of ~30 min). Leakage-free because the OOF is pitcher-'
                         'grouped: a fold model never saw its test pitchers\' 2026 '
                         'data, so it can score their NEW pitches too; pitchers '
                         'absent from every fold were never trained on at all. '
                         'Run a full retrain periodically to refresh the models.')
    args = ap.parse_args()

    print('loading pitches ...', flush=True)
    D = pickle.load(open(PKL, 'rb'))
    # Position-player pitching stays in the cache as of 2026-07-13 (the PAs
    # count in hitter/league data) but must not enter Stuff+ training, the
    # standardization pools, or the support manifold — an EP pitch marks the
    # whole appearance as a non-pitcher outing.
    _ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    pitches = [p for p in D if p.get('_source') == 'MLB'
               and (p.get('Pitcher'), p.get('PTeam')) not in _ep]
    # v13: 2026 kinematics ride a PitchID sidecar (the sheet cache has no
    # kinematics columns). Retrain REQUIRES it — training the FF gate on
    # imputed-only 2026 rows would quietly unlearn the feature. Score-only
    # degrades to the build_df imputation chain with a loud warning (fresh
    # grades drift toward each pitcher's own FF mean until the sidecar
    # lands, which is the designed behavior for sidecar lag).
    _n_kin = apply_kin_sidecar(pitches)
    if _n_kin < 0:
        if args.score_only:
            print('  WARNING: kinematics sidecar absent — kin_eff__ff scores '
                  'from the imputation chain only')
        else:
            sys.exit('ABORT: kinematics sidecar absent '
                     f'({KIN_SIDECAR}) — retrain would train kin_eff__ff on '
                     'imputed 2026 rows. Run scripts/build_kinematics_2026.py '
                     '(or download the release asset) first.')
    else:
        print(f'  kinematics sidecar: {_n_kin} 2026 pitches filled')
    roc_pitches = [p for p in D if p.get('_source') in ('ROC', 'AAA')
                   and (p.get('Pitcher'), p.get('PTeam')) not in _ep]
    # Scoring-only rows (the NEW tab — arms new to the org) live in their own
    # cache so the shared pickle stays exactly what generate_micro_data sees.
    # They join roc_pitches, which is a SCORE-ONLY list: build_df sets
    # target_xrv=None for it, so nothing here can enter training, the
    # standardization pools, or the support manifold. Their only effect is a
    # per-pitch grade in the dump, which the Sheets write-back then puts in
    # the NEW tab's Stuff+ column.
    _so_path = os.path.join(os.path.dirname(PKL), 'scoring_only_rs_cache.pkl')
    if os.path.exists(_so_path):
        with open(_so_path, 'rb') as _f:
            _so = pickle.load(_f)
        _so = [p for p in _so if (p.get('Pitcher'), p.get('PTeam')) not in _ep]
        roc_pitches += _so
        print(f'  + {len(_so)} scoring-only pitches from {_so_path}')
    # v13 ROC parity: the sidecar carries measured ROC kinematics from the
    # club-534 minors pull — apply it to the ROC scoring list too, or every
    # Rochester four-seamer imputes to the league constant. (Caught
    # 2026-08-14: the first parity deploy applied it to the MLB list only
    # and ROC grades moved exactly 0.0 — always verify the re-stamp moved.)
    _n_kin_roc = apply_kin_sidecar(roc_pitches)
    if _n_kin_roc > 0:
        print(f'  kinematics sidecar: {_n_kin_roc} ROC/AAA pitches filled')
    # MiLB RunExp currency correction (2026-07-28). Statcast's delta_run_exp
    # is built on each league's own RE matrix, so ROC/AAA RunExp in the raw
    # cache is MiLB-denominated (~1.27x MLB for the identical event).
    # process_data corrects this in ITS memory only — this trainer reads the
    # pickle directly and never saw the fix, so ROC xRVOE subtracted a
    # MiLB-currency actual from an MLB-currency expectation (measured bias:
    # ROC pitcher mean +0.178 -> +0.098 runs/100 once corrected;
    # scripts/xrvoe_roc_validity.py). Same per-(Description,Count) factors as
    # process_data, via the shared pipeline_utils implementation. Mutating in
    # place is safe: `pitches` (MLB) is disjoint, and no other trainer
    # consumer reads ROC RunExp before this point.
    from pipeline_utils import compute_runexp_scale, runexp_factor
    _re_scale = compute_runexp_scale(D)
    if _re_scale:
        _n_fx = 0
        for p in roc_pitches:
            _sc = _re_scale.get(p.get('_source'))
            _v = sf(p.get('RunExp'))
            if _sc and _v is not None:
                _f = runexp_factor(_sc, p.get('Description'), p.get('Count'))
                if _f:
                    p['RunExp'] = _v / _f
                    _n_fx += 1
        print(f'  MiLB RunExp -> MLB currency: {_n_fx} ROC/AAA pitches rescaled')
    # roc_pitches is final here (incl. the scoring-only NEW-tab arms), so the
    # debut fallback map sees every minor-league arm angle we hold.
    df_full = build_df(pitches, arm_fallback=_arm_means(roc_pitches))
    df = df_full[df_full['target_xrv'].notna()].reset_index(drop=True)
    # Pitches with no outcome target yet (pre-supplement, non-BIP with RunExp
    # blank) can't train or enter the rawmean aggregation, but the model can
    # still SCORE them — kept aside for the per-pitch grade dump.
    df_extra = df_full[df_full['target_xrv'].isna()].reset_index(drop=True)
    print(f'  {len(df)} training pitches, {df.pitcher.nunique()} pitchers')

    # ── SEASON-BLOCKED prior-season training data (2026-07-03) ──
    # data/_pitches2025_training.pkl: Wally's retagged 2025 season, joined to
    # public Statcast (VAA/xwOBA/RunExp recovered, density-adjusted movement;
    # built by scripts/build_2025_training_set.py; tags harmonized to 2026
    # labels via the cross-year centroid audit in the same script family).
    # It joins EVERY fold's training set below, while 2026 pitches are still
    # scored strictly out-of-fold by pitcher — so the model learns stable
    # cross-season identity (Tyler Rogers' singleton cluster: 50 -> 127) with
    # zero same-season luck leakage. Validated (scripts/
    # train_2526_experiment.py): pred future xRV 0.203 -> 0.244, descriptive
    # 0.274 -> 0.333, YoY-only model beats the 2026-only config outright.
    # 2025 targets use 2025 Guts constants (run environment differs).
    df_prior = None
    B = None
    if args.score_only:
        _bp = os.path.join(HERE, 'stuff_models.pkl')
        with open(_bp, 'rb') as f:
            B = pickle.load(f)
        if 'fold_models' not in B:
            sys.exit('--score-only needs a bundle with fold models — run a full retrain first')
        if 'kin_eff__ff' not in B.get('features', []):
            sys.exit('--score-only bundle predates the v13 config '
                     '(kin_eff__ff) — refresh the bundle from the '
                     'latest-data release or retrain')
        if 'cross' not in B.get('features', []):
            sys.exit('--score-only bundle predates the v12 config (cross/nVAA/'
                     'velo_diff-mask) — its fold models trained on different '
                     'features; run a full retrain (workflow: retrain=true) first')
        print(f'  score-only: using cached bundle ({len(B["fold_models"])} fold models, '
              f'trained {B.get("trained_through", "unknown")})')
    elif os.path.exists(PRIOR_PKL):
        global LG_WOBA, WOBA_SCALE
        _lg26, _sc26 = LG_WOBA, WOBA_SCALE

        def _build_with_guts(season_pitches, lg, sc):
            global LG_WOBA, WOBA_SCALE
            LG_WOBA, WOBA_SCALE = lg, sc
            d = build_df(season_pitches)
            LG_WOBA, WOBA_SCALE = _lg26, _sc26
            return d[d['target_xrv'].notna()].reset_index(drop=True)

        p25 = pickle.load(open(PRIOR_PKL, 'rb'))
        prior_dfs = [_build_with_guts(_harmonize_tags(p25, pitches),
                                      PRIOR_LG_WOBA, PRIOR_WOBA_SCALE)]
        print(f'  + {len(prior_dfs[0])} prior-season (2025) training pitches')
        # 2021-24 volume prior (see HIST_YEARS note above). No tag
        # harmonization: public tags never enter the agnostic model.
        for _yr in HIST_YEARS:
            _pkl = HIST_PKL.format(year=_yr)
            if not os.path.exists(_pkl):
                print(f'  ({_yr} training pickle missing — skipped)')
                continue
            _d = _build_with_guts(pickle.load(open(_pkl, 'rb')), *HIST_GUTS[_yr])
            prior_dfs.append(_d)
            print(f'  + {len(_d)} prior-season ({_yr}) training pitches')
        df_prior = pd.concat(prior_dfs, ignore_index=True)
        print(f'  prior total: {len(df_prior)} pitches across {len(prior_dfs)} seasons')
        # v12 guard: cross needs SpinAxis on the prior rows. A season-shaped
        # coverage hole acts as an era marker and silently degrades the model
        # (the axis_dev 80%-coverage lesson) — fail loudly instead.
        for _i, _d in enumerate(prior_dfs):
            _cov = _d['cross'].notna().mean() if 'cross' in _d.columns else 0.0
            if _cov < 0.90:
                sys.exit(f'ABORT: prior season index {_i} has cross coverage '
                         f'{_cov:.1%} (<90%) — training pickle likely predates '
                         f'the SpinAxis augmentation; refresh the release '
                         f'assets before retraining')
            # v13 guard: kin_eff__ff needs KinEff on the prior rows (same
            # season-shaped-hole logic). Measured on raw kin_eff before the
            # FF imputation chain masks the gap.
            _ffm = _d['pitch_type'] == 'FF'
            _kcov = (_d.loc[_ffm, 'kin_eff'].notna().mean()
                     if 'kin_eff' in _d.columns and _ffm.any() else 0.0)
            if _kcov < 0.90:
                sys.exit(f'ABORT: prior season index {_i} has FF kin_eff '
                         f'coverage {_kcov:.1%} (<90%) — training pickle '
                         f'predates the kinematics augmentation; refresh the '
                         f'release assets before retraining')
    else:
        print('  (no prior-season pickle found — training on current season only)')

    X = design(df); y = df['target_xrv'].values
    if B is not None:
        X = X.reindex(columns=B['features'], fill_value=0)

    # Production MLB scores are OUT-OF-FOLD (pitcher-grouped): a pitcher's own
    # 2026 outcomes never train the model that scores his pitches. In-sample
    # scoring inflated descriptive r from 0.22 to 0.42 by absorbing pitcher
    # luck through his unique feature cluster (median 2.7, p90 8 Stuff+ pts on
    # qualified units) — the shipped number must come from the validated path.
    groups = df['pitcher'].values
    def _oof_predict(Xd, feats=None, n_splits=8):
        """Returns (-oof, fold_models, fold_pitchers). The fitted fold models
        + their test-fold pitcher lists are saved in the bundle so
        --score-only runs can reproduce leakage-free OOF scoring without
        retraining (each model scores exactly the pitchers it never saw)."""
        oof = np.full(len(y), np.nan)
        pp = _params_for(Xd)
        Xp, yp = None, None
        if df_prior is not None:
            Xp = design(df_prior, feats or BASE_FEATS).reindex(columns=Xd.columns, fill_value=0)
            yp = df_prior['target_xrv'].values
        fold_models, fold_pitchers = [], []
        for tr, te in GroupKFold(n_splits=n_splits).split(Xd, y, groups):
            if Xp is not None:
                Xtr = pd.concat([Xd.iloc[tr], Xp], ignore_index=True)
                ytr = np.concatenate([y[tr], yp])
            else:
                Xtr, ytr = Xd.iloc[tr], y[tr]
            mm = xgb.XGBRegressor(**pp); mm.fit(Xtr, ytr); oof[te] = mm.predict(Xd.iloc[te])
            fold_models.append(mm)
            fold_pitchers.append(sorted(set(groups[te])))
        return -oof, fold_models, fold_pitchers

    def _cached_oof(Xd, models, fold_pitchers):
        """Score with the bundle's fold models: each pitcher goes to the fold
        that held him out (his data never trained that model); pitchers new
        since the retrain default to fold 0 (they were in NO fold's training
        set, so any fold model is leakage-free for them)."""
        fold_of = {p: k for k, ps in enumerate(fold_pitchers) for p in ps}
        pf = np.array([fold_of.get(p, 0) for p in groups])
        out = np.full(len(Xd), np.nan)
        for k, mm in enumerate(models):
            mask = pf == k
            if mask.any():
                out[mask] = mm.predict(Xd[mask])
        n_new = int((~np.isin(groups, list(fold_of))).sum())
        if n_new:
            print(f'  score-only: {n_new} pitches from pitchers newer than the '
                  f'bundle (scored leakage-free by fold 0)')
        return -out

    if B is not None:
        df['stuff_raw'] = _cached_oof(X, B['fold_models'], B['fold_pitchers'])
        model = B['model']
    else:
        df['stuff_raw'], fold_models, fold_pitchers = _oof_predict(X)

        # full-data model: kept in the bundle for scoring pitches outside the
        # training set (ROC is scored with this full model, and with the no-arm
        # companion only where arm angle is still missing). It no longer scores
        # MLB leaderboard rows.
        X_all = pd.concat([X, design(df_prior).reindex(columns=X.columns, fill_value=0)],
                          ignore_index=True) if df_prior is not None else X
        y_all = (np.concatenate([y, df_prior['target_xrv'].values])
                 if df_prior is not None else y)
        model = xgb.XGBRegressor(**_params_for(X)); model.fit(X_all, y_all)

    # Standardize at the PITCHER level (proper "+"-stat spread: SD=10 between
    # pitchers, not between pitches — averaging pitch-level z-scores collapses
    # the spread). The scale (mu, sd) is fixed from the qualified pool (>=QUAL_N
    # of that pitch type); each pitcher's mean is then shrunk toward the pool
    # mean by K_SHRINK pseudo-pitches so small samples fall toward 100 instead
    # of showing noise flukes. High-sample arms are essentially unaffected.
    def _standardize(grp_keys, qual_min):
        a = df.groupby(grp_keys)['stuff_raw'].agg(rawmean='mean', n='size').reset_index()
        a['stuff_mean'] = 100.0
        scale = {}
        groups = a.groupby('pitch_type') if 'pitch_type' in grp_keys else [('ALL', a)]
        for key, sub in groups:
            q = sub[sub['n'] >= qual_min]
            base = q if len(q) >= 5 else sub
            if key in ANCHOR_BORROW and 'pitch_type' in grp_keys:
                donor = a[a['pitch_type'].isin(ANCHOR_BORROW[key])]
                dq = donor[donor['n'] >= qual_min]
                base = dq if len(dq) >= 5 else donor
            mu, sd = float(base['rawmean'].mean()), float(base['rawmean'].std())
            scale[key] = {'mu': mu, 'sd': sd, 'nqual': int(len(q))}
            if sd > 0:
                # plain average, unclipped (coherent canon)
                a.loc[sub.index, 'stuff_mean'] = 100 + K_SCALE * (sub['rawmean'] - mu) / sd
        a['stuff_mean'] = a['stuff_mean'].round(1)
        return a, scale

    agg, league = _standardize(['pitcher', 'team', 'pitch_type'], QUAL_N)

    # low-model-support flag per unit (see compute_support docstring)
    if B is not None:
        # Score-only: carry the flags forward from the last full retrain's CSV
        # (recomputing needs the 3.5M-row training pool). Unit centroids move
        # slowly, so days-stale flags are fine; new units default to unflagged
        # until the next retrain measures them.
        _csv = os.path.join(HERE, 'pitcher_stuff.csv')
        if os.path.exists(_csv):
            sup = pd.read_csv(_csv)[['pitcher', 'team', 'pitch_type', 'support', 'low_support']]
            sup = sup.drop_duplicates(['pitcher', 'team', 'pitch_type'])
        else:
            sup = pd.DataFrame(columns=['pitcher', 'team', 'pitch_type', 'support', 'low_support'])
    else:
        sup = compute_support(df, df_prior)
    agg = agg.merge(sup, on=['pitcher', 'team', 'pitch_type'], how='left')
    agg['low_support'] = agg['low_support'].fillna(False).astype(bool)
    agg.to_csv(os.path.join(HERE, 'pitcher_stuff.csv'), index=False)
    # COHERENT CANON: overall Stuff+ = pitch-weighted mean of the per-type
    # atoms (100 + 10*(raw - mu_type)/sd_type per pitch), NOT a separately
    # anchored arsenal value — so averaging the sheet's Stuff+ column for a
    # pitcher reproduces his overall number exactly.
    def _atom_grade(row):
        sc = league.get(row['pitch_type'])
        if not sc or not np.isfinite(sc.get('sd', np.nan)) or sc['sd'] <= 0:
            return np.nan
        return 100 + K_SCALE * (row['stuff_raw'] - sc['mu']) / sc['sd']
    df['_atom'] = df.apply(_atom_grade, axis=1)
    overall = (df.dropna(subset=['_atom'])
                 .groupby(['pitcher', 'team', 'throws'])['_atom']
                 .agg(rawmean='mean', n='size').reset_index())
    overall['stuff_mean'] = overall['rawmean'].round(1)
    overall_scale = {'ALL': {'mu': 100.0, 'sd': 10.0, 'nqual': 0}}
    league['_overall'] = overall_scale['ALL']
    # Pitcher-level flag = USAGE-WEIGHTED (judgment call, documented): flagged
    # only when low-support units account for >=50% of his scored pitches — a
    # single fringe show-me pitch shouldn't taint the whole-arsenal grade,
    # while a pitcher whose primary offerings sit off the training manifold
    # (the actual "model can't see him" case) still flags.
    _fs = agg.assign(_fn=agg['n'] * agg['low_support']).groupby(
        ['pitcher', 'team'])[['_fn', 'n']].sum()
    _fs = (_fs['_fn'] / _fs['n'] >= 0.5).rename('low_support').reset_index()
    overall = overall.merge(_fs, on=['pitcher', 'team'], how='left')
    overall['low_support'] = overall['low_support'].fillna(False).astype(bool)

    # ── ROC/AAA scoring, against the MLB baseline. Per-pitch adaptive since
    #    the minors backfill: rows WITH arm angle (96-97%) use the FULL model,
    #    rows without fall back to the no-arm companion (the _arm mask below).
    #    This block ONLY affects ROC; MLB above is untouched. ──
    Xna = design(df, NOARM_FEATS)
    if B is not None:
        Xna = Xna.reindex(columns=B['features_na'], fill_value=0)
        model_na = B['model_na']
        # MLB anchor distribution for ROC is also OOF, so the mu/sd the ROC
        # scores are ranked against carry no in-sample luck inflation.
        df['raw_na'] = _cached_oof(Xna, B['fold_models_na'], B['fold_pitchers'])
    else:
        if df_prior is not None:
            Xna_all = pd.concat([Xna, design(df_prior, NOARM_FEATS).reindex(columns=Xna.columns, fill_value=0)],
                                ignore_index=True)
            yna_all = np.concatenate([y, df_prior['target_xrv'].values])
        else:
            Xna_all, yna_all = Xna, y
        model_na = xgb.XGBRegressor(**_params_for(Xna)); model_na.fit(Xna_all, yna_all)
        # MLB anchor distribution for ROC is also OOF, so the mu/sd the ROC scores
        # are ranked against carry no in-sample luck inflation. ROC pitches are not
        # in the training set, so model_na's predictions on ROC are truly OOS.
        df['raw_na'], fold_models_na, _fp_na = _oof_predict(Xna, feats=NOARM_FEATS)

    def _na_scale(keys, qual_min):
        a = df.groupby(keys)['raw_na'].agg(rawmean='mean', n='size').reset_index()
        out = {}
        groups = a.groupby('pitch_type') if 'pitch_type' in keys else [('ALL', a)]
        for key, sub in groups:
            q = sub[sub['n'] >= qual_min]; base = q if len(q) >= 5 else sub
            if key in ANCHOR_BORROW and 'pitch_type' in keys:
                donor = a[a['pitch_type'].isin(ANCHOR_BORROW[key])]
                dq = donor[donor['n'] >= qual_min]
                base = dq if len(dq) >= 5 else donor
            out[key] = {'mu': float(base['rawmean'].mean()), 'sd': float(base['rawmean'].std())}
        return out
    na_pt = _na_scale(['pitcher', 'team', 'pitch_type'], QUAL_N)
    na_ov = _na_scale(['pitcher', 'team', 'throws'], 2 * QUAL_N)['ALL']

    roc_df = build_df(roc_pitches)   # target_xrv is None at ROC — we only score
    if len(roc_df):
        # ── Per-pitcher arm branch (2026-07-25) ──
        # ROC/AAA now carries real arm angle, backfilled from Savant's
        # minor-league Statcast Search (~99.6% of pitches). It is the same
        # measurement MLB gets, not an estimate: across 138 pitchers who threw
        # at both levels in 2026, MLB vs MiLB means correlate r=0.991 with a
        # 1.4 deg mean absolute difference. So a ROC pitcher WITH arm angle is
        # now scored by the SAME full model against the SAME MLB per-type
        # anchors (`league`) as an MLB pitcher — mirroring MLB exactly, and
        # restoring the single most load-bearing feature in BASE_FEATS.
        #
        # build_df imputes arm angle within (pitcher, pitch_type) then pitcher,
        # so a pitcher is effectively all-or-nothing here. Anyone still without
        # any arm angle — a game Savant has not processed yet, or a stale
        # cache — falls back to the no-arm companion and its anchored baseline,
        # exactly as before. Both paths stay out-of-sample: ROC pitches never
        # enter the training pool.
        _arm = roc_df['arm_angle'].notna().values
        roc_df['_raw'] = np.nan
        if _arm.any():
            Xr = design(roc_df[_arm]).reindex(columns=X.columns, fill_value=0)
            roc_df.loc[_arm, '_raw'] = -model.predict(Xr)
        if (~_arm).any():
            Xr_na = design(roc_df[~_arm], NOARM_FEATS).reindex(
                columns=Xna.columns, fill_value=0)
            roc_df.loc[~_arm, '_raw'] = -model_na.predict(Xr_na)
        # The anchor set must follow the model that produced the raw value.
        roc_df['_use_arm'] = _arm

        def _roc_atoms(frame, rounded):
            """Per-pitch grades using each row's own anchor set: full-model rows
            against the MLB per-type anchors, no-arm fallback rows against the
            companion's. rounded=True gives the integer atoms written to the
            Sheets grade column; rounded=False the full-precision value."""
            out = pd.Series(np.nan, index=frame.index)
            for use_arm, anchors in ((True, league), (False, na_pt)):
                m = (frame['_use_arm'] == use_arm).values
                if not m.any():
                    continue
                sub = frame.loc[m]
                vals = pd.Series(np.nan, index=sub.index)
                for pt, sc in anchors.items():
                    if (not isinstance(sc, dict)
                            or not np.isfinite(sc.get('sd', np.nan)) or sc['sd'] <= 0):
                        continue
                    mm = (sub['pitch_type'] == pt).values
                    if mm.any():
                        vals[mm] = (100 + K_SCALE
                                    * (sub.loc[mm, '_raw'] - sc['mu']) / sc['sd'])
                out[m] = vals.round() if rounded else vals
            return out

        roc_df['_atom'] = _roc_atoms(roc_df, rounded=False)
        _scored = roc_df.dropna(subset=['_atom'])
        roc_agg = (_scored.groupby(['pitcher', 'team', 'pitch_type'])['_atom']
                          .agg(rawmean='mean', n='size').reset_index())
        roc_agg['stuff_mean'] = roc_agg['rawmean'].round(1)
        roc_overall = (_scored.groupby(['pitcher', 'team', 'throws'])['_atom']
                              .agg(rawmean='mean', n='size').reset_index())
        roc_overall['stuff_mean'] = roc_overall['rawmean'].round(1)
        # ROC never enters the training pool or the support manifold, so the
        # low-support diagnostic still doesn't apply (see compute_support).
        roc_agg['low_support'] = False
        roc_overall['low_support'] = False
        agg = pd.concat([agg, roc_agg], ignore_index=True)
        overall = pd.concat([overall, roc_overall], ignore_index=True)
        _n_arm = int(_arm.sum())
        print(f'  ROC vs MLB baseline: {len(roc_agg)} pitch-type rows, '
              f'{len(roc_overall)} pitchers | {_n_arm}/{len(roc_df)} pitches '
              f'by the FULL arm-angle model, {len(roc_df) - _n_arm} by the '
              f'no-arm companion')

    # ── COHERENT CANON display aggregation (2026-07-18, per Wally) ──
    # Every displayed Stuff+ — per-type unit AND pitcher overall — is the
    # plain mean of the per-pitch INTEGER atoms, the same integers written
    # to the Sheets Stuff+ column, so a sheet AVERAGEIF reproduces every
    # leaderboard value to the digit. Anchors stay fit on training-row unit
    # rawmeans (above); only the aggregation of displayed values changes.
    # Target-less pitches (df_extra) get sheet grades, so they belong in the
    # displayed averages too — scored here (OOF fold discipline) instead of
    # inside the dump gate.
    _fm = B['fold_models'] if B is not None else fold_models
    _fp = B['fold_pitchers'] if B is not None else fold_pitchers
    if len(df_extra):
        Xe = design(df_extra).reindex(columns=X.columns, fill_value=0)
        _fold_of = {pp: k for k, ps in enumerate(_fp) for pp in ps}
        _pf = np.array([_fold_of.get(pp, 0) for pp in df_extra['pitcher']])
        raw_e = np.full(len(df_extra), np.nan)
        for k, mm in enumerate(_fm):
            mask = _pf == k
            if mask.any():
                raw_e[mask] = -mm.predict(Xe[mask])
        df_extra = df_extra.assign(stuff_raw=raw_e)

    def _atom_series(frame, anchors, raw_col):
        """Per-pitch integer atoms (float dtype; NaN where ungradeable)."""
        out = pd.Series(np.nan, index=frame.index)
        for pt, sc in anchors.items():
            if (not isinstance(sc, dict)
                    or not np.isfinite(sc.get('sd', np.nan)) or sc['sd'] <= 0):
                continue
            m = (frame['pitch_type'] == pt).values
            if m.any():
                out[m] = (100 + K_SCALE
                          * (frame.loc[m, raw_col] - sc['mu']) / sc['sd']).round()
        return out

    _DISP_COLS = ['pitcher', 'team', 'throws', 'pitch_type',
                  'sheet_tab', 'sheet_row']
    _mlb_disp = pd.concat(
        [df[_DISP_COLS + ['stuff_raw']]] +
        ([df_extra[_DISP_COLS + ['stuff_raw']]] if len(df_extra) else []),
        ignore_index=True)
    _mlb_disp['_ai'] = _atom_series(_mlb_disp, league, 'stuff_raw')
    _mlb_disp = _mlb_disp.dropna(subset=['_ai'])
    _atom_frames = [_mlb_disp]
    if len(roc_df):
        # Integer atoms with each row's own anchor set (full-model rows anchor
        # to `league`, no-arm fallback rows to `na_pt`).
        _roc_disp = roc_df[_DISP_COLS + ['_raw', '_use_arm']].copy()
        _roc_disp['_ai'] = _roc_atoms(_roc_disp, rounded=True)
        _roc_disp = _roc_disp.dropna(subset=['_ai'])
        _atom_frames.append(_roc_disp)

    def _atom_stats(keys):
        parts = [f.groupby(keys)['_ai'].agg(atom_sum='sum', atom_n='size')
                 .reset_index() for f in _atom_frames]
        s = pd.concat(parts, ignore_index=True)
        s['atom_mean'] = (s['atom_sum'] / s['atom_n']).round(1)
        return s

    def _apply_atom_means(target, keys):
        m = target.merge(_atom_stats(keys), on=keys, how='left')
        has = m['atom_mean'].notna()
        m.loc[has, 'stuff_mean'] = m.loc[has, 'atom_mean']
        m['atom_sum'] = m['atom_sum'].fillna(0.0)
        m['atom_n'] = m['atom_n'].fillna(0).astype(int)
        return m.drop(columns=['atom_mean'])

    agg = _apply_atom_means(agg, ['pitcher', 'team', 'pitch_type'])
    overall = _apply_atom_means(overall, ['pitcher', 'team', 'throws'])
    # re-save the CSV so it carries the displayed (atom-mean) values
    agg.to_csv(os.path.join(HERE, 'pitcher_stuff.csv'), index=False)

    # Pitching+ atoms: pc = round(0.8·S + 0.2·L) of the two INTEGER atoms —
    # identical to the sheet's Z cells — averaged per unit / per pitcher.
    # Loc atoms come from process_data's dump (written earlier in the same
    # pipeline run); if it's absent the maps stay empty and injection falls
    # back to the blend of the two displayed components.
    pc_maps = {'pt': {}, 'ov': {}, 'pool_pt': {}, 'pool_ov': {}}
    _loc_path = os.path.join(DATA, 'pitch_loc_grades_rs.json')
    if os.path.exists(_loc_path):
        with open(_loc_path) as _f:
            _locg = json.load(_f)
        def _pc_add(mp, key, pc):
            s, n = mp.get(key, (0.0, 0))
            mp[key] = (s + pc, n + 1)
        for frame in _atom_frames:
            for tab, rownum, pitcher, team, throws, pt, sa in zip(
                    frame['sheet_tab'], frame['sheet_row'], frame['pitcher'],
                    frame['team'], frame['throws'], frame['pitch_type'],
                    frame['_ai']):
                if tab is None or rownum is None:
                    continue
                lg = _locg.get(f'{tab}\t{int(rownum)}')
                if lg is None:
                    continue
                pc = int(round(PITCHING_W_STUFF * int(sa)
                               + (1.0 - PITCHING_W_STUFF) * int(round(lg))))
                _pc_add(pc_maps['pt'], (pitcher, team, pt), pc)
                _pc_add(pc_maps['ov'], (pitcher, team), pc)
                if team not in AAA_TEAMS:
                    # pooled across MLB stints — serves combined 2TM/3TM rows
                    _pc_add(pc_maps['pool_pt'], (pitcher, pt), pc)
                    _pc_add(pc_maps['pool_ov'], (pitcher,), pc)
        print(f"  Pitching+ atoms: {sum(n for _s, n in pc_maps['ov'].values())} "
              f"pitches with both grades")

    # ── Per-pitch grade dump for the Sheets write-back ──
    # Scale (i), unregressed (2026-07-18, per Wally): each pitch graded with the
    # SAME per-type anchors as the leaderboard value, no K_SHRINK (a pitch grade
    # describes the pitch, not the pitcher's sample size). Keyed by sheet
    # position ("tab\trow", attached by pipeline_fetch). Rows without sheet
    # coordinates (stale cache) or without a valid anchor are skipped -> blank.
    if args.dump_pitch_grades:
        def _grade(raw, sc):
            # UNCLIPPED full precision: clipping is display-only (write-back),
            # so window averages and the composite stay exact.
            if raw is None or not np.isfinite(raw): return None
            if not sc or not np.isfinite(sc.get('sd', np.nan)) or sc['sd'] <= 0: return None
            return float(100 + K_SCALE * (raw - sc['mu']) / sc['sd'])
        grades = {}
        def _emit(frame, raw_col, anchors):
            for tab, rownum, pt, raw in zip(frame['sheet_tab'], frame['sheet_row'],
                                            frame['pitch_type'], frame[raw_col]):
                if tab is None or rownum is None: continue
                g = _grade(raw, anchors.get(pt))
                if g is not None:
                    grades[f'{tab}\t{int(rownum)}'] = round(g, 2)
        _emit(df, 'stuff_raw', league)
        # target-less pitches (pre-supplement): scored upstream in the canon
        # display-aggregation section with the same OOF fold discipline.
        if len(df_extra):
            _emit(df_extra, 'stuff_raw', league)
        if len(roc_df):
            # Unrounded, per-row anchors — the same values the leaderboard uses.
            _rg = _roc_atoms(roc_df, rounded=False)
            for tab, rownum, g in zip(roc_df['sheet_tab'], roc_df['sheet_row'], _rg):
                if tab is None or rownum is None or not np.isfinite(g):
                    continue
                grades[f'{tab}\t{int(rownum)}'] = round(float(g), 2)
        with open(args.dump_pitch_grades, 'w') as f:
            json.dump(grades, f, separators=(',', ':'))
        print(f'  per-pitch grade dump: {len(grades)} pitches -> {args.dump_pitch_grades}')

    # save bundle (retrain only — a score-only run must not overwrite the
    # fold models it just loaded). fold_models/fold_pitchers power the
    # --score-only path; the na fold split is identical to the main one
    # (GroupKFold is deterministic for the same groups), so one
    # fold_pitchers list serves both model sets.
    if B is None:
        with open(os.path.join(HERE, 'stuff_models.pkl'), 'wb') as f:
            pickle.dump({'model': model, 'features': list(X.columns), 'base_feats': BASE_FEATS,
                         'league': league, 'params': TUNED, 'version': 'v13',
                         'model_na': model_na, 'noarm_feats': NOARM_FEATS,
                         'features_na': list(Xna.columns),
                         'na_pt_scale': na_pt, 'na_ov_scale': na_ov,
                         'fold_models': fold_models, 'fold_pitchers': fold_pitchers,
                         'fold_models_na': fold_models_na,
                         'trained_through': max((d for d in df['date'].dropna()), default='')}, f)
    else:
        # Score-only: refresh the anchor scales in place (models untouched).
        # Cards' cell-fallback atoms read bundle['league'] — if the scales
        # went stale between retrains (new data, or an ANCHOR_BORROW change),
        # fallback grades would drift from the sheet's, breaking exact-match.
        B['league'], B['na_pt_scale'], B['na_ov_scale'] = league, na_pt, na_ov
        with open(os.path.join(HERE, 'stuff_models.pkl'), 'wb') as f:
            pickle.dump(B, f)

    # report + metric history (drift visibility: every retrain appends its
    # OOF descriptive r and split-half reliability to data/, which CI
    # commits — see "They Don't Make Pitch Models Like They Used To" for
    # why watching these decay matters)
    from numpy import corrcoef
    g = df.groupby(['pitcher', 'pitch_type'])
    recs = []
    for key, grp in g:
        if len(grp) >= 50:
            recs.append((grp['stuff_raw'].mean(), grp['target_xrv'].mean()))
    xs = np.array([r[0] for r in recs]); ys = np.array([r[1] for r in recs])
    oof_desc_r = float(corrcoef(xs, ys)[0, 1])
    print(f'  OOF stuff vs same-period xRV (descriptive): r = {oof_desc_r:+.3f}')

    # split-half reliability of OOF stuff (odd/even calendar dates)
    date_order = {d: i for i, d in enumerate(sorted(df['date'].dropna().unique()))}
    df['_half'] = df['date'].map(date_order).fillna(0).astype(int) % 2
    a0, a1 = [], []
    for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = grp[grp._half == 0], grp[grp._half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0['stuff_raw'].mean()); a1.append(h1['stuff_raw'].mean())
    rel = float(corrcoef(np.array(a0), np.array(a1))[0, 1]) if len(a0) >= 5 else float('nan')
    print(f'  OOF split-half reliability (>=40/half): r = {rel:+.3f} (n={len(a0)})')

    hist = os.path.join(DATA, 'stuff_metrics_history.csv')
    latest_date = max((d for d in df['date'].dropna()), default='')
    new_file = not os.path.exists(hist)
    with open(hist, 'a') as f:
        if new_file:
            f.write('through_date,n_pitches,n_pitchers,oof_descriptive_r,oof_splithalf_r,n_rel_units\n')
        f.write(f'{latest_date},{len(df)},{df.pitcher.nunique()},'
                f'{oof_desc_r:.4f},{rel:.4f},{len(a0)}\n')
    print('\n  top arsenals by mean Stuff+ (n>=200):')
    top = agg[agg.n >= 200].sort_values('stuff_mean', ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"    {r.pitcher:24s} {r.pitch_type:3s} {r.stuff_mean:5.1f}  (n={r.n})")
    print(f'\n  saved bundle + pitcher_stuff.csv to {HERE}')

    if args.inject:
        xrvoe_pt, xrvoe_ov = compute_xrvoe(df, pitches,
                                           roc_df=roc_df, roc_pitches=roc_pitches)
        inject(agg, overall, league, xrvoe_pt=xrvoe_pt, xrvoe_ov=xrvoe_ov,
               pc_maps=pc_maps, mlb_pitches=pitches)
    else:
        print('  (skipped leaderboard injection; run with --inject to surface)')

def _is_combined_team(t):
    return isinstance(t, str) and t.endswith('TM') and t[:-2].isdigit()


def _combo_score(parts):
    """Combined 2TM/3TM row = plain mean of the pitcher's per-pitch INTEGER
    atoms pooled across his MLB stints — the same number a sheet AVERAGEIF
    across his team tabs gives. Unclipped (coherent canon; the old version
    clipped to [40, 180], which no other surface does). `parts` is a list of
    (atom_sum, atom_n) per stint."""
    if not parts:
        return None
    tot_n = sum(n for _s, n in parts)
    if tot_n <= 0:
        return None
    return round(sum(s for s, _n in parts) / tot_n, 1)


def _pctl(sc, pool):
    below = sum(1 for x in pool if x < sc); equal = sum(1 for x in pool if x == sc)
    return round((below + 0.5 * equal) / len(pool) * 100)


# ── Pitching+ (2026-07-07) ──
# Validated composite of the two shipped scales (scripts/
# pitching_plus_experiment.py + pitching_plus_stage2.py): 0.70·z(Stuff+) +
# 0.20·z(Loc+), re-standardized so between-pitcher SD = 10 (the components
# are near-orthogonal, r=+0.04, so the raw blend has SD ~7.6).
#
# 0.70 -> 0.80 (2026-07-25, scripts/pitchingplus_loso_full.py). The old 0.70
# came from a bootstrap median on a single season (95% CI [0.51, 0.91], flat
# plateau 0.60-0.80). Re-derived leave-one-season-out across 2021-2025 — each
# season scored by a model fit on the other four, so no season sees itself —
# and measured on the SHIPPED scales rather than z-scores. That last part is
# what moved the answer: Stuff+ is the mean of per-pitch-type atoms, so
# averaging across an arsenal shrinks it to SD ~7.1, while Loc+ is z-scored to
# SD ~9.6. A nominal 0.70 was therefore delivering Stuff+ only ~63% of the
# real influence. Solving for a true 70% influence gives w ~ 0.77, adjacent to
# the measured optimum — the original 70/30 INTENT was right, the
# implementation under-weighted Stuff+ because the components were never on a
# common spread.
#
# Cross-season argmin (mean Fisher-z) is 0.82, but the objective is flat
# within 1% across [0.75, 0.90], so 0.82 beats 0.80 by 0.0002 — unmeasurable.
# 0.80 is chosen inside that flat region: near its centre (0.825), robust if
# the region drifts, and auditable by eye from the two adjacent columns, which
# 0.82/0.18 is not. What IS settled is that 0.70 sits OUTSIDE the flat region
# (mean z -0.3882 vs -0.3981) and loses to 0.80 in four of five seasons.
#
# The FG-style JOINT model (stuff + location + count in one XGBoost,
# season-blocked protocol) was built and RACED — it loses to this composite on
# both axes (best joint: pred 0.373 / rel 0.756 vs composite 0.389 / 0.764;
# adding count wrecks its reliability, same count-mix contamination as the
# Loc+ anchoring lesson). Don't revisit joint without multi-season training data.
# DEFINED IN pipeline_utils (single source of truth — process_data, Cards and
# sheets_write_grades all read the same constant without importing xgboost).
# The derivation above is why it is 0.80; that note stays here.
from pipeline_utils import PITCHING_W_STUFF  # noqa: E402

def _blend(stuff, loc):
    return (PITCHING_W_STUFF * (stuff - 100.0) / 10.0
            + (1.0 - PITCHING_W_STUFF) * (loc - 100.0) / 10.0)


def _ols_slope(xs, ys):
    """Least-squares slope of ys on xs (pure Python). None if degenerate."""
    n = len(xs)
    if n < 20:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var


def _inject_pitching(rows, key_of, qualifies, pc_lookup=None):
    """Write pitchingScore + pitchingScore_pctl into leaderboard rows.

    Pool convention matches stuffScore percentiles: MLB rows with >=25
    pitches, combined 2TM/3TM rows excluded; every row with both component
    scores gets a score AND a rank (qualification is a render-time coloring
    gate). key_of(row) -> pool bucket (pitch type at pitch level, 'ALL' at
    pitcher level); qualifies(row) -> pool membership."""
    # COHERENT CANON: pitchingScore = plain mean of the per-pitch Pitching+
    # atoms (round(0.8·S + 0.2·L) of the two INTEGER atoms — the sheet's Z
    # cells), supplied via pc_lookup. Rows the maps can't serve (loc dump
    # absent) fall back to the blend of the two displayed components, which
    # matches to within rounding since both are now atom means themselves.
    n = 0
    for row in rows:
        pc = pc_lookup(row) if pc_lookup is not None else None
        if pc is not None:
            row['pitchingScore'] = pc
            n += 1
            continue
        st, lc = row.get('stuffScore'), row.get('locPlus')
        if st is None or lc is None:
            row['pitchingScore'] = None
            continue
        row['pitchingScore'] = round(100.0 + 10.0 * _blend(st, lc), 1)
        n += 1
    pool_s = defaultdict(list)
    for row in rows:
        if row.get('pitchingScore') is not None and qualifies(row):
            pool_s[key_of(row)].append(row['pitchingScore'])
    for row in rows:
        sc = row.get('pitchingScore')
        pool = pool_s.get(key_of(row))
        row['pitchingScore_pctl'] = _pctl(sc, pool) if (sc is not None and pool) else None

    # pitchingRuns100 — a run-denominated companion for the hover tooltip.
    # Deliberately a MONOTONIC transform of pitchingScore (slope * (P+ - 100)),
    # so it can never invert against the displayed number the way a run-space
    # re-blend (0.8*stuffRuns + 0.2*locRuns) would (stuff and loc have
    # different run SDs, so that reordering lopsided arms — the exact
    # FanGraphs-style inconsistency we avoid). The slope is the pooled OLS of
    # actual xRV/100 on Pitching+ over the qualified pool: it answers "a
    # pitcher at this Pitching+ has historically prevented ~this many runs/100
    # vs average," self-calibrating each retrain. One global slope per
    # leaderboard (not per pitch-type) so the runs read on the same scale
    # across a pitcher's arsenal.
    xs = [row['pitchingScore'] for row in rows
          if row.get('pitchingScore') is not None and row.get('xRv100') is not None
          and qualifies(row)]
    ys = [row['xRv100'] for row in rows
          if row.get('pitchingScore') is not None and row.get('xRv100') is not None
          and qualifies(row)]
    slope = _ols_slope(xs, ys)
    for row in rows:
        sc = row.get('pitchingScore')
        row['pitchingRuns100'] = (round(slope * (sc - 100.0), 2)
                                  if (sc is not None and slope is not None) else None)
    return n

# ── xRVOE: per-pitch outperformance vs the stuff+location expectation ──
# Validated 2026-07-14 (scripts/xrvoe_feasibility.py + xrvoe_crossyear.py):
# within-season split-half n0 ~250-300; cross-year persistence r=+0.41
# pooled over 2021->2026 pairs (leave-two-years-out models); adding year-Y
# xRVOE to the expectation lifts next-year actual-xRV prediction r from
# .366 to .492. It is a durable pitch trait (deception / seam effects /
# tunneling / framing — everything the stuff+loc models can't see).
XRVOE_N0 = 275        # shrinkage pseudo-pitches toward 0 (measured n0)
XRVOE_MIN_PT = 150    # display floor per pitch type (below: >2/3 prior)
XRVOE_MIN_OV = 300    # display floor pitcher-level
# leaderboard field -> compute_xrvoe unit-record key. xrvoe100 is the shrunk
# skill estimate; rvoe/xrvoe/rvoe100 are raw accounting (unshrunk).
XRVOE_FIELDS = {'xrvoe100': 'v', 'rvoe100': 'rvoe100',
                'rvoe': 'rvoe', 'xrvoe': 'xrvoe'}


def compute_xrvoe(df, pitches, roc_df=None, roc_pitches=None):
    """Per-unit xRVOE/100 (pitcher-positive) from the OOF stuff predictions
    already in df['stuff_raw'] and per-pitch Loc ExpRV built here.
    Stacking is fit PER PITCH GROUP (fixes the loc-correlation calibration
    wrinkle found in feasibility). Returns (per_pitch_type, per_pitcher)
    dicts: key -> {'v': shrunk xRVOE/100, 'n': pitches}.

    ROC/AAA (2026-07-25): scored, never baseline — the same translation
    framing Stuff+ and Loc+ use. The Loc surfaces and the stacking betas are
    fit on MLB rows only; ROC pitches are then scored against them, and the
    percentile pools downstream already exclude AAA. This was previously
    impossible because both sides of the residual come from RunExp
    (target_xrv on non-BIP, rv_raw everywhere) and ROC's RunExp was 0%
    populated; the minor-league Statcast backfill filled it.
    """
    import pipeline_locplus as LOC
    baseline = [p for p in pitches if LOC.is_eligible_baseline(p)]
    S = LOC.build_surfaces(baseline, LG_WOBA, WOBA_SCALE)
    exprv = {}
    for p in baseline:
        v = LOC.score_pitch(p, S)
        if v is not None and p.get('PitchID'):
            exprv[p['PitchID']] = v
    # ROC scored against the MLB surfaces. is_eligible_baseline() is
    # MLB-only by design, so ROC never shaped them; _is_scorable is the
    # scoring-side check (documented as the one that lets ROC through).
    if roc_pitches:
        for p in roc_pitches:
            if not LOC._is_scorable(p):
                continue
            v = LOC.score_pitch(p, S)
            if v is not None and p.get('PitchID'):
                exprv[p['PitchID']] = v

    def _prep(frame):
        d = frame[['pid', 'pitcher', 'team', 'pitch_type', 'target_xrv',
                   'stuff_raw', 'rv_raw']].copy()
        d['loc_exprv'] = d['pid'].map(exprv)
        # target_xrv is non-null by construction on the MLB training frame;
        # the ROC frame carries scored-but-untargeted pitches, so filter here.
        d = d[d['loc_exprv'].notna() & d['stuff_raw'].notna()
              & d['target_xrv'].notna()]
        d['stuff_hit'] = -d['stuff_raw']      # back to hitter perspective
        d['grp'] = d['pitch_type'].map(LOC.group_of_code)
        return d

    d = _prep(df)

    # global stacking as fallback, then per-group where the sample allows.
    # Fit on MLB only — ROC is measured against this expectation, never in it.
    def _stack(sub):
        A = np.column_stack([np.ones(len(sub)), sub['stuff_hit'].values,
                             sub['loc_exprv'].values])
        beta, *_ = np.linalg.lstsq(A, sub['target_xrv'].values, rcond=None)
        return beta, A
    beta_g, _ = _stack(d)
    betas = {grp: (_stack(sub)[0] if len(sub) >= 5000 else beta_g)
             for grp, sub in d.groupby('grp')}

    def _apply(dd):
        dd = dd.copy()
        dd['expect'] = np.nan
        for grp, sub in dd.groupby('grp'):
            beta = betas.get(grp, beta_g)
            A = np.column_stack([np.ones(len(sub)), sub['stuff_hit'].values,
                                 sub['loc_exprv'].values])
            dd.loc[sub.index, 'expect'] = A @ beta
        dd['resid'] = dd['target_xrv'] - dd['expect']
        # RVOE: same expectation, but the actual side is raw RV (RunExp, luck
        # included). RVOE - xRVOE = contact luck. Accounting flavor -> unshrunk.
        dd['resid_raw'] = dd['rv_raw'] - dd['expect']
        return dd

    d = _apply(d)

    # calibration report (unit level, n>=100): should be ~0 after per-group
    # fit. MLB only — that is what the betas were fit on.
    gq = d.groupby(['pitcher', 'team', 'pitch_type']).agg(
        resid=('resid', 'mean'), n=('resid', 'size'),
        stuff=('stuff_hit', 'mean'), loc=('loc_exprv', 'mean'))
    gq = gq[gq['n'] >= 100]
    if len(gq) > 50:
        print(f"  xRVOE calibration: corr(resid, stuff) "
              f"{np.corrcoef(gq['resid'], gq['stuff'])[0,1]:+.3f}, "
              f"corr(resid, loc) {np.corrcoef(gq['resid'], gq['loc'])[0,1]:+.3f}")

    d_all = d
    if roc_df is not None and len(roc_df):
        _r = roc_df.rename(columns={'_raw': 'stuff_raw'})
        if 'stuff_raw' in _r.columns:
            d_roc = _prep(_r)
            if len(d_roc):
                d_all = pd.concat([d, _apply(d_roc)], ignore_index=True)
                print(f'  xRVOE: {len(d_roc)} ROC/AAA pitches scored against '
                      f'the MLB expectation')

    def _units(keys, min_n):
        # v (xRVOE/100) is the shrunk skill estimate; xrvoe/rvoe/rvoe100 are
        # raw accounting (unshrunk, full precision — display layer rounds),
        # matching the site's RV/xRV convention.
        g = d_all.groupby(keys).agg(resid=('resid', 'mean'), n=('resid', 'size'),
                                    resid_sum=('resid', 'sum'),
                                    raw_sum=('resid_raw', 'sum'),
                                    raw_n=('resid_raw', 'count'))
        out = {}
        for key, r in g.iterrows():
            if r['n'] < min_n:
                continue
            shrunk = (r['n'] / (r['n'] + XRVOE_N0)) * r['resid']
            has_raw = r['raw_n'] > 0
            out[key] = {'v': round(-shrunk * 100.0, 2), 'n': int(r['n']),
                        'xrvoe': -r['resid_sum'],
                        'rvoe': -r['raw_sum'] if has_raw else None,
                        'rvoe100': (-(r['raw_sum'] / r['raw_n']) * 100.0
                                    if has_raw else None)}
        return out
    return (_units(['pitcher', 'team', 'pitch_type'], XRVOE_MIN_PT),
            _units(['pitcher', 'team'], XRVOE_MIN_OV))


def inject(agg, overall, league, xrvoe_pt=None, xrvoe_ov=None, pc_maps=None,
           mlb_pitches=None):
    """Write stuffScore into BOTH leaderboards.

    Percentile convention (site standard, aligned 2026-07-02): the pool that
    DEFINES the distribution is MLB rows with >=25 pitches (per pitch type at
    pitch level; overall at pitcher level), excluding combined 2TM/3TM rows
    (their stints already represent them). EVERY row with a score gets a
    rank stored — including ROC and low-sample rows — and qualification is a
    render-only coloring gate applied by leaderboard.js. Previously the rank
    existed only where locPlus_pctl existed, which silently coupled Stuff+
    coloring to Loc+ qualification settings."""
    # ── per-pitch-type -> pitch_leaderboard ──
    pl_path = os.path.join(DATA, 'pitch_leaderboard_rs.json')
    pl = json.load(open(pl_path))
    look = {(r.pitcher, r.team, r.pitch_type): r.stuff_mean for r in agg.itertuples()}
    flag_look = {(r.pitcher, r.team, r.pitch_type): bool(getattr(r, 'low_support', False))
                 for r in agg.itertuples()}
    # Pool a pitcher's MLB per-team RAW predictions to score the combined 2TM/3TM row.
    # Combined-row low-support = ANY stint flagged (the flag is about the pitch's
    # physics vs the training manifold, which doesn't change with the uniform).
    pool_pt = defaultdict(list)
    flag_pt = defaultdict(bool)
    for r in agg.itertuples():
        if r.team not in AAA_TEAMS and not _is_combined_team(r.team):
            pool_pt[(r.pitcher, r.pitch_type)].append(
                (getattr(r, 'atom_sum', 0.0), getattr(r, 'atom_n', 0)))
            flag_pt[(r.pitcher, r.pitch_type)] |= bool(getattr(r, 'low_support', False))
    for row in pl:
        key = (row['pitcher'], row['team'], row['pitchType'])
        if key in look:
            row['stuffScore'] = look[key]
            row['stuffScore_lowSupport'] = flag_look.get(key, False)
        elif _is_combined_team(row['team']):
            row['stuffScore'] = _combo_score(
                pool_pt.get((row['pitcher'], row['pitchType'])))
            row['stuffScore_lowSupport'] = (flag_pt.get((row['pitcher'], row['pitchType']), False)
                                            if row['stuffScore'] is not None else False)
        else:
            row['stuffScore'] = None
            row['stuffScore_lowSupport'] = False
    # Pool: MLB rows with >=25 pitches of the type; combined 2TM/3TM rows
    # excluded from the POOL (their stints already represent them). Every
    # scored row gets a rank; coloring is gated at render time.
    qual_by_pt = defaultdict(list)
    for row in pl:
        if (row.get('stuffScore') is not None and (row.get('count') or 0) >= 25
                and row.get('team') not in AAA_TEAMS and not _is_combined_team(row['team'])):
            qual_by_pt[row['pitchType']].append(row['stuffScore'])
    n_pl = 0
    for row in pl:
        sc = row.get('stuffScore'); pt = row['pitchType']
        if sc is not None: n_pl += 1
        if sc is not None and qual_by_pt.get(pt):
            row['stuffScore_pctl'] = _pctl(sc, qual_by_pt[pt])
        else:
            row['stuffScore_pctl'] = None
    def _pc_mean(rec):
        return round(rec[0] / rec[1], 1) if rec and rec[1] > 0 else None

    def _pc_pt_lookup(row):
        if pc_maps is None:
            return None
        rec = pc_maps['pt'].get((row['pitcher'], row['team'], row['pitchType']))
        if rec is None and _is_combined_team(row['team']):
            rec = pc_maps['pool_pt'].get((row['pitcher'], row['pitchType']))
        return _pc_mean(rec)

    n_ps_pl = _inject_pitching(
        pl, key_of=lambda r: r['pitchType'],
        qualifies=lambda r: ((r.get('count') or 0) >= 25
                             and r.get('team') not in AAA_TEAMS
                             and not _is_combined_team(r['team'])),
        pc_lookup=_pc_pt_lookup)
    # ── xRVOE -> pitch rows. ROC/AAA now carries it too (its RunExp was
    # backfilled 2026-07-25); the pools below still exclude AAA, so ROC is
    # ranked against MLB without defining the distribution. Combined rows
    # stay None — stints carry the value. ──
    if xrvoe_pt:
        pools = {f: defaultdict(list) for f in XRVOE_FIELDS}
        for row in pl:
            k = (row['pitcher'], row['team'], row['pitchType'])
            rec = xrvoe_pt.get(k)
            for f, src in XRVOE_FIELDS.items():
                row[f] = rec[src] if rec else None
                if (rec and rec[src] is not None
                        and row.get('team') not in AAA_TEAMS
                        and not _is_combined_team(row['team'])):
                    pools[f][row['pitchType']].append(rec[src])
        for row in pl:
            for f in XRVOE_FIELDS:
                v = row.get(f)
                row[f + '_pctl'] = (_pctl(v, pools[f][row['pitchType']])
                                    if v is not None and pools[f].get(row['pitchType'])
                                    else None)
    json.dump(pl, open(pl_path, 'w'))

    # ── overall -> pitcher_leaderboard ──
    pp_path = os.path.join(DATA, 'pitcher_leaderboard_rs.json')
    pp = json.load(open(pp_path))
    olook = {(r.pitcher, r.team, r.throws): r.stuff_mean for r in overall.itertuples()}
    oflag = {(r.pitcher, r.team, r.throws): bool(getattr(r, 'low_support', False))
             for r in overall.itertuples()}
    pool_ov = defaultdict(list)
    flag_ov = defaultdict(bool)
    for r in overall.itertuples():
        if r.team not in AAA_TEAMS and not _is_combined_team(r.team):
            pool_ov[(r.pitcher, r.throws)].append(
                (getattr(r, 'atom_sum', 0.0), getattr(r, 'atom_n', 0)))
            flag_ov[(r.pitcher, r.throws)] |= bool(getattr(r, 'low_support', False))
    for row in pp:
        key = (row['pitcher'], row['team'], row.get('throws'))
        if key in olook:
            row['stuffScore'] = olook[key]
            row['stuffScore_lowSupport'] = oflag.get(key, False)
        elif _is_combined_team(row['team']):
            row['stuffScore'] = _combo_score(
                pool_ov.get((row['pitcher'], row.get('throws'))))
            row['stuffScore_lowSupport'] = (flag_ov.get((row['pitcher'], row.get('throws')), False)
                                            if row['stuffScore'] is not None else False)
        else:
            row['stuffScore'] = None
            row['stuffScore_lowSupport'] = False
    qpool = [row['stuffScore'] for row in pp
             if row.get('stuffScore') is not None and (row.get('count') or 0) >= 25
             and row.get('team') not in AAA_TEAMS and not _is_combined_team(row['team'])]
    n_pp = 0
    for row in pp:
        sc = row.get('stuffScore')
        if sc is not None: n_pp += 1
        if sc is not None and qpool:
            row['stuffScore_pctl'] = _pctl(sc, qpool)
        else:
            row['stuffScore_pctl'] = None
    def _pc_ov_lookup(row):
        if pc_maps is None:
            return None
        rec = pc_maps['ov'].get((row['pitcher'], row['team']))
        if rec is None and _is_combined_team(row['team']):
            rec = pc_maps['pool_ov'].get((row['pitcher'],))
        return _pc_mean(rec)

    n_ps_pp = _inject_pitching(
        pp, key_of=lambda r: 'ALL',
        qualifies=lambda r: ((r.get('count') or 0) >= 25
                             and r.get('team') not in AAA_TEAMS
                             and not _is_combined_team(r['team'])),
        pc_lookup=_pc_ov_lookup)
    n_x = 0
    if xrvoe_ov:
        pools_o = {f: [] for f in XRVOE_FIELDS}
        for row in pp:
            rec = xrvoe_ov.get((row['pitcher'], row['team']))
            for f, src in XRVOE_FIELDS.items():
                row[f] = rec[src] if rec else None
                if (rec and rec[src] is not None
                        and row.get('team') not in AAA_TEAMS
                        and not _is_combined_team(row['team'])):
                    pools_o[f].append(rec[src])
        for row in pp:
            if row.get('xrvoe100') is not None:
                n_x += 1
            for f in XRVOE_FIELDS:
                v = row.get(f)
                row[f + '_pctl'] = (_pctl(v, pools_o[f])
                                    if v is not None and pools_o[f] else None)
    # ── Pitcher+ (must run LAST: it consumes the fresh stuffScore/locPlus
    # written above, plus the rate stats process_data already set) ──
    from pipeline_pitcherplus import apply_pitcher_plus, serialize_baseline
    # current season = calendar year (the prior asset must be year-1); an
    # offseason run can emit a harmless staleness warning.
    from datetime import datetime as _dt
    _pp_base = apply_pitcher_plus(pp, data_dir=DATA,
                                  current_season=_dt.now().year)
    n_pplus = sum(1 for row in pp if row.get('pitcherPlus') is not None)
    n_proj = sum(1 for row in pp if row.get('pitcherPlusProj') is not None)

    # ── hdERA / hpERA (after Pitcher+: hpERA consumes the fresh
    # stuffScore + locPlusRaw + rate stats). Also sets the default
    # leaderboard row order (hpERA good -> bad).
    from pipeline_eraplus import apply_era_plus, sort_rows_default
    if mlb_pitches is not None:
        _era_const = apply_era_plus(pp, mlb_pitches)
    else:
        # No pitch list handed in (unexpected caller): keep the carried
        # hdERA/hpERA values rather than dying mid-inject.
        _era_const = None
        print('  eraplus SKIPPED: inject() called without mlb_pitches')
    sort_rows_default(pp)

    json.dump(pp, open(pp_path, 'w'))

    # Baseline goes to metadata so the client can recompute Pitcher+ under
    # filters against the season-long league scale.
    if _pp_base or _era_const:
        _md_path = os.path.join(DATA, 'metadata_rs.json')
        try:
            _md = json.load(open(_md_path))
            if _pp_base:
                _md['pitcherPlusBaseline'] = serialize_baseline(_pp_base)
            if _era_const:
                _md['eraPlusConstants'] = _era_const
                _md.setdefault('pitcherLeagueAverages', {})
                _md['pitcherLeagueAverages']['hdera'] = _era_const['anchor']
                _md['pitcherLeagueAverages']['hpera'] = _era_const['anchor']
            json.dump(_md, open(_md_path, 'w'))
        except Exception as _e:
            print(f'  (metadata baseline write skipped: {_e})')

    print(f'  injected stuffScore: pitch-level {n_pl}/{len(pl)} rows, '
          f'pitcher-level {n_pp}/{len(pp)} rows (qualified pool = {len(qpool)})')
    print(f'  injected pitchingScore: pitch-level {n_ps_pl}/{len(pl)} rows, '
          f'pitcher-level {n_ps_pp}/{len(pp)} rows')
    print(f'  injected Pitcher+: pitcher-level {n_pplus}/{len(pp)} rows'
          + ('' if _pp_base else ' (pool too thin — all None)'))
    if _pp_base:
        print(f'  injected Pitcher+ Proj: {n_proj} rows '
              f'({_pp_base.get("_projNPrior", 0)} with a '
              f'{_pp_base.get("_projPriorSeason")} prior)')
    if xrvoe_ov:
        print(f'  injected xRVOE/100: pitcher-level {n_x}/{len(pp)} rows '
              f'(floors {XRVOE_MIN_PT}/{XRVOE_MIN_OV} pitches, n0 {XRVOE_N0})')

if __name__ == '__main__':
    main()
