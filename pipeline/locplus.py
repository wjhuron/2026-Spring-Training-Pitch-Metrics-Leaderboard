"""Loc+ (Location+) — decomposition-based per-pitch location-quality metric.

The pitcher analog of a command grade: "is this pitcher putting pitches in
valuable spots, given the count, pitch type, and matchup, independent of his
stuff or what the hitter happened to do?"

MODEL (rebuilt 2026-06). For each pitch we look up the league-average
EXPECTED hitter-perspective run value of a pitch of that type/handedness at
that exact plate location, in that count, then average across the pitcher's
pitches and normalize. Lower expected RV = better location for the pitcher.

The expected value is a DECOMPOSITION over smooth league surfaces:

  ExpRV(x, z | grp, hands, count) =
      Pswing · [ Pwhiff·rvWhiff(count) + Pfoul·rvFoul(count) + Pbip·xwOBAcon(x,z) ]
    + (1-Pswing) · [ Pcs·rvCS(count) + (1-Pcs)·rvBall(count) ]

  - Pwhiff, Pfoul, xwOBAcon, Pcs (called-strike prob) and Pswing are league
    SURFACES over a (PlateX, zone-normalized PlateZ) grid, built per pitch-type
    GROUP × batter-hand × pitcher-hand. Physical surfaces (whiff/foul/contact/
    called-strike) are count-independent for sample size; swing propensity is
    count-specific. The run-value WEIGHTS (rvWhiff/rvFoul/rvCS/rvBall) are
    count-specific. Surfaces are smoothed with an anisotropic separable
    Gaussian (4.5" horizontal, 0.22 zone vertical).
  - Scoring at TRUE count level (no count-demeaning): empirically this is more
    predictive and no less stuff-independent than demeaning.
  - Contact (xwOBAcon) surface is heavily shrunk toward the group mean because
    location-driven contact suppression is mostly luck (THT command study).

Design choices were validated empirically (see scripts/research/locplus/ and scripts/archive/):
reliability (split-half), stuff-independence (low corr with whiff%/velo), and
predictive validity (first-half score vs second-half xRV allowed). This model
roughly doubles the run-prevention signal of the old 5-zone metric while
becoming markedly more stuff-independent.

Pitch-type groups (validated by clustering value surfaces):
  FF | SI | FC | SL(+ST,SW,SV) | CU(+KC,CS) | CH(+FS,KN,SC) | OTHER

Normalization: displayed Loc+ (overall and per-type) is the plain mean of
per-pitch INTEGER atoms graded against per-group anchors (coherent canon,
2026-07-18) — no reliability shrinkage: N_PRIOR_OVERALL / N_PRIOR_PT are
ZEROED below (the measured r=0.5 crossings, overall 135 + per-group dict,
are documented there but not applied). raw_loc_adj / locRuns100 / zone_loc
still flow through _normalize (mu, sigma from qualified MLB pitchers). ROC
pitchers are scored against the MLB surfaces but excluded from the
(mu, sigma) pool.
"""
import math
from collections import defaultdict

from pipeline.utils import safe_float, AAA_TEAMS
from pipeline.sdplus import classify_zone, ZONES, build_bip_count_offsets

# ── Model options (each A/B-validated on the 3-objective harness:
#    scripts/archive/phase2_locplus_eval.py — reliability / stuff-independence /
#    predictive validity). Validated 2026-07-02: ──
PCS_BY_HAND = True             # called-strike surface per batter hand (the
                               # LHH called zone sits ~2" farther outside;
                               # takes are ~half of pitches and shadow takes
                               # are where location value concentrates).
                               # WON: rel 0.568->0.575, pred 0.079->0.082.
BIP_COUNT_ANCHOR = False       # add offset(c) to the BIP value branch
                               # (pipeline_sdplus.build_bip_count_offsets).
                               # LOST decisively for Loc+ and stays OFF:
                               # velo leak 0.29->0.38, whiff leak
                               # 0.031->0.072, predictive 0.079->-0.029.
                               # Anchoring makes ExpRV strongly count-mix
                               # dependent. (The anchor is correct and ON in
                               # SD+/CT+, which score hitter decisions against
                               # the count state.)
                               # RE-TESTED 2026-07-25 under CS_COUNT_TRANSFORM
                               # (scripts/archive/locplus_phase3_eval.py) — still loses:
                               # rel 0.600->0.586, whiff leak 0.056->0.089,
                               # velo 0.305->0.382, pred 0.075->-0.025.
                               # CORRECTION to the original reasoning: count
                               # mix is NOT purely a stuff/sequencing effect.
                               # Neutralizing it directly (count-mix post-
                               # stratification: per-count means recombined at
                               # LEAGUE count weights) DOUBLED the stuff leak
                               # and killed prediction outright (0.075->-0.002),
                               # and did not rescue the anchor underneath it
                               # (0.576/-0.024). Throwing quality strikes is HOW
                               # a pitcher gets ahead, so count mix carries real
                               # LOCATION signal; stripping it discards skill and
                               # upweights sparse counts (3-0, 3-1) where
                               # per-pitcher samples are noisiest. The BIP
                               # branch's count-invariance is therefore an
                               # ACCEPTED cost, not an open item — do not
                               # re-litigate without a new mechanism.
SWING_PRIOR_COUNT_LEVEL = True # count-specific swing surfaces shrink toward
                               # collapsed-surface × league count multiplier
                               # (a sparse 3-0 surface otherwise shrinks
                               # toward a ~46% swing rate instead of ~10%).
                               # WON objective 2: whiff leak 0.031->0.019,
                               # rel +0.005, pred -0.006 (noise-level).
CS_COUNT_TRANSFORM = True      # count-transform on the called-strike surface:
                               # umpires expand the zone with more balls and
                               # shrink it with more strikes, so the SAME
                               # location is called a strike at different rates
                               # by count. One baseline CS surface + a per-
                               # (hand,count) logit intercept calibrated so the
                               # predicted called-strike count matches observed
                               # among that count's takes (BP framing-model
                               # style). WON (scripts/archive/
                               # locplus_cs_transform_test.py): rel 0.591->0.602, stuff-leak flat, pred
                               # -0.007 (noise); learned shifts monotonic and
                               # match umpire behavior (3-0 +0.32, 0-2 -0.67).

# ── Pitch-type grouping ─────────────────────────────────────────────────
GROUP = {
    'FF': 'FF', 'FA': 'FF',
    'SI': 'SI',
    'FC': 'FC', 'CF': 'FC',
    'SL': 'SL', 'ST': 'SL', 'SW': 'SL', 'SV': 'SL',
    'CU': 'CU', 'KC': 'CU', 'CS': 'CU',
    # KN/SC ride with offspeed (Wally, 2026-07-13): same speed band and
    # location logic (below-zone chase, low-heart avoidance), and it gives
    # the league's lone knuckleballer a real peer baseline instead of the
    # degenerate one-pitcher OTHER group (which returned a flat 100).
    'CH': 'CH', 'FS': 'CH', 'KN': 'CH', 'SC': 'CH',
}
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'OTHER']

def group_of(p):
    pt = p.get('Pitch Type')
    if not pt:
        return None
    return GROUP.get(pt, 'OTHER')

def group_of_code(pt):
    if not pt:
        return None
    return GROUP.get(pt, 'OTHER')

# ── Grid + smoothing ────────────────────────────────────────────────────
X_MIN, X_MAX = -1.5, 1.5            # feet (plate center = 0)
BIN_X = 2.0 / 12.0                  # 2-inch horizontal bins
NX = int(round((X_MAX - X_MIN) / BIN_X))           # 18
Z_MIN, Z_MAX = -0.6, 1.6           # zone-normalized (0 = bottom, 1 = top)
BIN_Z = 0.10
NZ = int(round((Z_MAX - Z_MIN) / BIN_Z))           # 22
PHYS_X_IN = 4.5                    # physical smoothing bandwidths
PHYS_Z_FRAC = 0.22
# Optional per-group bandwidth override, {group: (x_inches, z_frac)}. Empty
# means one bandwidth for every pitch-type group. Applies to the physical
# surfaces built per (group, bh, ph) — whiff / foul / xwOBAcon / swing. The
# called-strike surface is per hand only, so it always uses the global pair.
PHYS_BW_PT = {}

# Per-surface shrinkage pseudo-counts toward the group mean
K_WHIFF, K_FOUL, K_XWCON = 8, 8, 200
K_SWING_COLL, K_SWING_COUNT, K_CS = 6, 20, 10

# ── Count-level physical structure (2026-08-15) ─────────────────────────
# WHIFF surfaces are count-specific, built exactly like the swing surfaces:
# per-count grids from that count's own swings, shrunk toward the collapsed
# shape scaled by the count's league whiff multiplier. Won the replicate
# battery 5/5 on partial (scripts/research/locplus/locplus_countwhiff_multiseason.py),
# passed a permuted-count placebo control, the 2026 retag replicate with a
# partial|Stuff+ control, and a whiff-skill leak check. K swept {5..80}:
# flat (max spread .0008), 20 kept as the convention matching the swing
# surface. FOUL count grids were tested and REJECTED (1/5); the CONTACT
# branch takes a per-count LEVEL offset instead (below).
WH_COUNT_LEVEL = True
K_WH_COUNT = 20
# Contact-quality level by count: the BIP branch adds offset(c) = league
# mean standardized xwOBA-value on BIP in count c minus the overall mean
# (2-strike defensive contact is weaker, hitter-count contact louder).
# Won 5/5 on partial with a ~.012 rel cost (accepted: real count physics,
# prediction gain — the mirror of the CS-transform convention). This is
# NOT the rejected BIP_COUNT_ANCHOR: the anchor bundled this quality term
# with an RE-state currency term, and the currency half is what lost 0/5
# (decomposition in scripts/research/locplus/locplus_count_stability.py). In-season
# estimates need >= XW_CLEVEL_MIN_BIP per count; thinner counts fall back
# to the pooled 2021-2026 means (cross-season spread .004-.019, near-
# constants of baseball — same convention as FALLBACK_COUNT_OFFSETS).
XW_COUNT_LEVEL = True
XW_CLEVEL_MIN_BIP = 200
XW_CLEVEL_FALLBACK = {
    (0, 0): +0.012, (0, 1): -0.009, (0, 2): -0.032,
    (1, 0): +0.018, (1, 1): +0.001, (1, 2): -0.023,
    (2, 0): +0.036, (2, 1): +0.015, (2, 2): -0.014,
    (3, 0): +0.090, (3, 1): +0.039, (3, 2): +0.006,
}

# Per-pitcher regression + normalization. n_prior values are the measured
# split-half r=0.5 crossings (regression constant). Re-measured 2026-07-13
# on the full season, 10 shuffle seeds (scripts/archive/locplus_nprior_multiseed.py):
# overall mean 135 (median 134, seed range 118-155) — the early-season 117
# under-regressed. Per-group is now measurable (was "breakers unmeasurable"
# in the April measurement); values are the 10-seed medians. FF/SL stabilize
# fastest (~71-74), the cutter slowest (~117). OTHER is unmeasured → 100.
# Output is low-sensitivity here, but each group should be regressed by its
# own evidence rate.
# COHERENT CANON (2026-07-18, per Wally): displayed Loc+ is the PLAIN AVERAGE
# of per-pitch location grades — priors zeroed. The measured priors (overall
# 135; per-group FF 71 / SI 85 / FC 117 / SL 74 / CU 95 / CH 104) DO improve
# small-sample estimation (split-half MAE -2.9 to -5.4 at 25-100 pitches) but
# buy <1 point at qualified samples, while breaking the ledger property that
# sheet/card/site averages reconcile exactly. Small samples are handled by
# qualification gates at render time.
N_PRIOR_OVERALL = 0
N_PRIOR_PT = {}
N_PRIOR_PT_DEFAULT = 0

# The measured r=0.5 crossings themselves, kept live even though the priors
# above are zeroed. With an UNSHRUNK displayed mean, reliability at n pitches
# is n/(n+k) for the group's k, so these values ARE the render-time
# qualification gates the canon note defers to: a 25-pitch FF cell is only
# 25/(25+71) = 0.26 reliable, which is why pitch-type Loc+ cannot ride the flat
# 25-pitch outcome gate the other per-pitch metrics use (2026-07-25 audit: 771
# rows, 30% of all colored pitch-type Loc+ cells, sat below r=0.5).
# MIRRORED in js/aggregator.js (QUAL.MIN_PITCH_LOCPLUS) — keep the two in
# sync. (process_data imports stabilize_n from here, so Python is single-homed.)
# Re-measured 2026-07-25 (scripts/research/locplus/locplus_stabilize_celllevel.py, 10 seeds) at
# the unit the leaderboard actually renders: cells are per PITCH TYPE, surfaces
# are built once on the full season (surface estimation noise is common to all
# cells, so charging it to a pitcher understates his reliability), and each
# cell is split RANDOMLY (the displayed number estimates a season aggregate, so
# a chronological split would wrongly charge for in-season drift).
# The superseded values came from locplus_nprior_multiseed.py, which measured
# at the (pitcher, GROUP) level — a different unit that pools a pitcher's CH
# with his FS. Five of six moved <15% and the old values sit inside the new
# seed spread; CH moved 104 -> 72 with a seed range of 62-82 that excludes the
# old value, consistent with that pooling inflating k for the most
# heterogeneous group.
# Fallback for a pitch type with NO measured constant. Nothing reaches it
# today — EP was the only type that did, and it's excluded outright as a
# position-player tag (see EXCLUDE_PT) — so this is a safety net for a genuinely
# novel pitch type appearing mid-season. Policy, not a measurement: don't color
# what we can't validate. (It replaced a hardcoded 135, the old (pitcher,
# group)-level OVERALL crossing, which looked measured but was never a gate
# anyone had validated at the cell level.) To start coloring a new type, measure
# it with scripts/research/locplus/locplus_stabilize_celllevel.py and add it to STABILIZE_N_PT.
STABILIZE_N_UNVALIDATED = float('inf')
STABILIZE_N_PT = {'FF': 73, 'SI': 81, 'FC': 74, 'SL': 67, 'CU': 83, 'CH': 79}
# Re-measured 2026-08-15 on the count-aware surfaces (WH_COUNT_LEVEL +
# XW_COUNT_LEVEL): the new per-pitch atoms are less noisy, so five of six
# groups stabilize faster (FC 122 -> 74 is the big mover; CH 72 -> 79).
# Leaderboard pitch-CATEGORY rows pool several types (js/aggregator.js
# PITCH_CATEGORIES), so they take the stiffest member gate.
STABILIZE_N_CATEGORY = {'Hard': 81, 'Breaking': 83, 'Offspeed': 79}


def stabilize_n(pitch_type):
    """Minimum pitches for a pitch-type (or pitch-category) Loc+ cell to reach
    split-half r >= 0.5, i.e. at least half the variance is signal. Used as a
    render-time coloring gate; the percentile RANK is still stored for every
    row, matching the all-MLB-pool convention."""
    if pitch_type in STABILIZE_N_CATEGORY:
        return STABILIZE_N_CATEGORY[pitch_type]
    return STABILIZE_N_PT.get(GROUP.get(pitch_type), STABILIZE_N_UNVALIDATED)
LOC_SCALE_K = 10
MIN_POOL_OVERALL = 250             # min pitches to enter the (mu,sigma) pool
MIN_POOL_PT = 60                   # min pitches of a type to enter its group pool

COUNTS = [(b, s) for b in range(4) for s in range(3)]
HANDS = ('L', 'R')
SWING_DESC = {'Swinging Strike', 'Foul', 'In Play'}
TAKE_DESC = {'Ball', 'Called Strike'}
# Non-competitive pitches. Excluded from BOTH the baseline and scoring: the
# pitcher was not trying to hit a target (pitchout, intentional walk), or the
# pitch hit the batter.
EXCLUDE_DESC = {'Hit By Pitch',
                'Pitchout', 'Swinging Pitchout', 'Foul Pitchout'}
# Bunts. Excluded from the BASELINE only (Wally, 2026-08-30). A bunt's swing,
# foul and contact behaviour is nothing like normal hitting, so bunts must not
# shape the league surfaces. They ARE scored, because score_pitch reads only
# (group, hand, count, location) and never the outcome — so a bunted pitch is
# perfectly scorable, and dropping it would penalise the pitcher for the
# BATTER's choice. Loc+ grades where the pitch went, not what was done with
# it; that is the same decision-based rule the rest of the repo follows.
# Before the split, one bunt could erase a whole pitch type's Loc+ on a game
# card (Kranick 2026-08-28: his only slider was bunted, so the row was blank).
BUNT_DESC = {'Foul Bunt', 'Missed Bunt', 'Bunt Foul Tip'}
# EP tags a POSITION PLAYER on the mound, not a pitch a pitcher throws (Wally,
# 2026-07-25). The data bears it out: all 40 EP throwers this season are 100%
# EP with no other pitch type — Trevino, McCann, Higashioka, Rojas, Straw.
# So EP pitches neither define the league surfaces nor get scored. Excluding
# them in _is_scorable covers both, since is_eligible_baseline() delegates to
# it and then adds the bunt rule on top.
EXCLUDE_PT = {'EP'}
BUNT_BB = {'bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}


# ═════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════
from pipeline.utils import get_count  # single-homed count parser

def _znorm(p):
    pz = safe_float(p.get('PlateZ'))
    top = safe_float(p.get('SzTop'))
    bot = safe_float(p.get('SzBot'))
    if pz is None or top is None or bot is None or top <= bot:
        return None
    return (pz - bot) / (top - bot)

def _xbin(px):
    return min(max(int((px - X_MIN) / BIN_X), 0), NX - 1)
def _zbin(zn):
    return min(max(int((zn - Z_MIN) / BIN_Z), 0), NZ - 1)

def _is_scorable(p):
    """Valid lookup key + event exclusions. No RunExp/xwOBA requirement here
    (those are checked at surface-build time), which lets ROC pitches score."""
    if p.get('Event') == 'Intent Walk':
        return False
    if p.get('Description') in EXCLUDE_DESC:
        return False
    if p.get('Pitch Type') in EXCLUDE_PT:
        return False
    if group_of(p) is None:
        return False
    if safe_float(p.get('PlateX')) is None or _znorm(p) is None:
        return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS:
        return False
    if get_count(p) is None:
        return False
    return True

def is_bunt(p):
    """A bunt attempt, by description or by batted-ball type."""
    return (p.get('Description') in BUNT_DESC
            or p.get('BBType') in BUNT_BB)


def is_eligible_baseline(p):
    """May this pitch SHAPE the league surfaces?

    Stricter than _is_scorable on purpose: bunts are scorable but must not
    define the surfaces everything else is measured against. See BUNT_DESC.
    """
    return (p.get('_source') == 'MLB' and _is_scorable(p)
            and not is_bunt(p))


# ═════════════════════════════════════════════════════════════════════════
#  SEPARABLE ANISOTROPIC GAUSSIAN SMOOTHER
# ═════════════════════════════════════════════════════════════════════════
def _k1d(bw):
    win = max(1, int(math.ceil(3 * bw)))
    return [(d, math.exp(-0.5 * (d / bw) ** 2)) for d in range(-win, win + 1)]

_KX = _k1d(PHYS_X_IN / 2.0)        # bandwidth in cells (bins are 2", 0.10z)
_KZ = _k1d(PHYS_Z_FRAC / BIN_Z)

def _zeros():
    return [[0.0] * NZ for _ in range(NX)]

def _kernels_for(grp):
    """(_KX, _KZ) for a pitch-type group, honoring PHYS_BW_PT."""
    bw = PHYS_BW_PT.get(grp)
    if not bw:
        return _KX, _KZ
    return _k1d(bw[0] / 2.0), _k1d(bw[1] / BIN_Z)


def _smooth(num, den, prior, kprior, kx=None, kz=None):
    """Nadaraya-Watson kernel regression (num/den are NX×NZ arrays) with a
    prior pseudo-count. `prior` is a scalar or an NX×NZ array. kx/kz default to
    the global bandwidth kernels."""
    _kx = kx if kx is not None else _KX
    _kz = kz if kz is not None else _KZ
    tn, td = _zeros(), _zeros()
    for i in range(NX):
        ni, di_, tni, tdi = num[i], den[i], tn[i], td[i]
        for j in range(NZ):
            sn = sd = 0.0
            for dj, w in _kz:
                jj = j + dj
                if 0 <= jj < NZ:
                    sn += w * ni[jj]; sd += w * di_[jj]
            tni[j] = sn; tdi[j] = sd
    out = _zeros()
    pdict = not isinstance(prior, (int, float))
    for i in range(NX):
        oi = out[i]
        for j in range(NZ):
            sn = sd = 0.0
            for di2, w in _kx:
                ii = i + di2
                if 0 <= ii < NX:
                    sn += w * tn[ii][j]; sd += w * td[ii][j]
            pr = prior[i][j] if pdict else prior
            s = sd + kprior
            oi[j] = (sn + kprior * pr) / s if s > 0 else pr
    return out

def _gsum(a):
    return sum(sum(r) for r in a)

def _logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# ═════════════════════════════════════════════════════════════════════════
#  BUILD LEAGUE SURFACES
# ═════════════════════════════════════════════════════════════════════════
def build_surfaces(baseline, lg_woba, woba_scale):
    """Build all league surfaces + count value scalars from MLB baseline pitches.
    Returns a dict bundle consumed by score_pitch()."""
    has_guts = (lg_woba is not None and woba_scale not in (None, 0))

    # count value scalars (hitter perspective = -RunExp)
    cv = {k: defaultdict(lambda: [0.0, 0]) for k in ('whiff', 'foul', 'cs', 'ball')}
    for p in baseline:
        re = safe_float(p.get('RunExp'))
        if re is None:
            continue
        d = p.get('Description')
        slot = {'Swinging Strike': 'whiff', 'Foul': 'foul',
                'Called Strike': 'cs', 'Ball': 'ball'}.get(d)
        if slot:
            c = get_count(p)
            cv[slot][c][0] += -re; cv[slot][c][1] += 1
    RV = {k: {c: (s / n if n else 0.0) for c, (s, n) in dd.items()} for k, dd in cv.items()}
    # Fill counts with zero baseline events (possible on partial/backfill
    # runs — e.g. no 3-0 whiffs yet) with the slot's event-weighted overall
    # mean. Without this, score_pitch's .get(c, ...) silently valued the
    # outcome at 0.0 — e.g. a 3-0 whiff at 0.0 instead of ~-0.10 —
    # distorting every pitch scored in that count.
    for k, dd in cv.items():
        tot_s = sum(s for s, n in dd.values())
        tot_n = sum(n for _s, n in dd.values())
        overall = tot_s / tot_n if tot_n else 0.0
        for c in COUNTS:
            if c not in RV[k]:
                RV[k][c] = overall

    def acc0():
        return {k: _zeros() for k in ('swn', 'swd', 'whn', 'fln', 'bipn', 'bipd')}
    A = defaultdict(acc0)                                   # [(grp,bh,ph)]
    AC = defaultdict(lambda: {'swn': _zeros(), 'swd': _zeros(),
                              'whn': _zeros()})   # [(grp,bh,ph,count)]
    csn = {h: _zeros() for h in HANDS}
    csd = {h: _zeros() for h in HANDS}
    cnt_sw = defaultdict(lambda: [0, 0])   # count -> [swings, pitches] (league)
    cnt_wh = defaultdict(lambda: [0, 0])   # count -> [whiffs, swings] (league)
    xw_cnt = defaultdict(lambda: [0.0, 0])  # count -> [std xw-value sum, n] (BIP)
    csd_hc = defaultdict(_zeros)           # (hand,count) -> take-count grid
    cs_obs_hc = defaultdict(int)           # (hand,count) -> observed called strikes

    for p in baseline:
        bh = p['Bats']
        key = (group_of(p), bh, p['Throws'])
        c = get_count(p)
        i = _xbin(safe_float(p.get('PlateX'))); j = _zbin(_znorm(p))
        d = p.get('Description')
        a = A[key]; ac = AC[(key, c)]
        a['swd'][i][j] += 1; ac['swd'][i][j] += 1
        cnt_sw[c][1] += 1
        if d in SWING_DESC:
            a['swn'][i][j] += 1; ac['swn'][i][j] += 1
            cnt_sw[c][0] += 1
            cnt_wh[c][1] += 1
            if d == 'Swinging Strike':
                a['whn'][i][j] += 1; ac['whn'][i][j] += 1
                cnt_wh[c][0] += 1
            elif d == 'Foul':
                a['fln'][i][j] += 1
            elif d == 'In Play':
                xw = safe_float(p.get('xwOBA'))
                if has_guts and xw is not None:
                    a['bipn'][i][j] += (xw - lg_woba) / woba_scale
                    a['bipd'][i][j] += 1
                    xw_cnt[c][0] += (xw - lg_woba) / woba_scale
                    xw_cnt[c][1] += 1
        if d in TAKE_DESC:
            csd[bh][i][j] += 1
            csd_hc[(bh, c)][i][j] += 1
            if d == 'Called Strike':
                csn[bh][i][j] += 1
                cs_obs_hc[(bh, c)] += 1

    # Called-strike surface: per batter hand (PCS_BY_HAND) or pooled. Stored
    # as {hand: grid} either way so score_pitch has one lookup shape.
    if PCS_BY_HAND:
        PCS = {h: _smooth(csn[h], csd[h],
                          _gsum(csn[h]) / max(_gsum(csd[h]), 1), K_CS)
               for h in HANDS}
    else:
        pn = _zeros(); pd_ = _zeros()
        for h in HANDS:
            for i in range(NX):
                for j in range(NZ):
                    pn[i][j] += csn[h][i][j]; pd_[i][j] += csd[h][i][j]
        pooled = _smooth(pn, pd_, _gsum(pn) / max(_gsum(pd_), 1), K_CS)
        PCS = {h: pooled for h in HANDS}

    # Count-transform (CS_COUNT_TRANSFORM): keep one baseline CS surface per hand
    # and shift it per count by a single logit intercept, calibrated so the
    # predicted called-strike count matches the observed count among that
    # (hand,count)'s takes. Reshapes PCS to {hand: {count: grid}} so score_pitch
    # indexes by count. Sparse counts (< MIN_CT_TAKES) keep the base surface.
    MIN_CT_TAKES = 50
    PCS_c = {}
    for h in HANDS:
        base = PCS[h]
        PCS_c[h] = {}
        for c in COUNTS:
            delta = 0.0
            if CS_COUNT_TRANSFORM:
                tk = csd_hc.get((h, c)); obs = cs_obs_hc.get((h, c), 0)
                if tk is not None:
                    tk_n = _gsum(tk)
                    if tk_n >= MIN_CT_TAKES and 0 < obs < tk_n:
                        pred = sum(tk[i][j] * base[i][j]
                                   for i in range(NX) for j in range(NZ))
                        if pred > 0:
                            delta = _logit(obs / tk_n) - _logit(pred / tk_n)
            if delta == 0.0:
                PCS_c[h][c] = base
            else:
                PCS_c[h][c] = [[_sig(_logit(base[i][j]) + delta)
                                for j in range(NZ)] for i in range(NX)]
    PCS = PCS_c

    # League per-count swing-rate multipliers for the count-level prior.
    tot_sw = sum(v[0] for v in cnt_sw.values())
    tot_n = sum(v[1] for v in cnt_sw.values())
    overall_rate = tot_sw / tot_n if tot_n else 0.0
    cnt_mult = {c: ((v[0] / v[1]) / overall_rate if v[1] and overall_rate else 1.0)
                for c, v in cnt_sw.items()}

    # League per-count whiff multipliers for the count-level whiff prior.
    tot_wh = sum(v[0] for v in cnt_wh.values())
    tot_whsw = sum(v[1] for v in cnt_wh.values())
    ov_wh = tot_wh / tot_whsw if tot_whsw else 0.0
    wh_mult = {c: ((v[0] / v[1]) / ov_wh if v[1] and ov_wh else 1.0)
               for c, v in cnt_wh.items()}

    WH, FL, XW, SW = {}, {}, {}, {}
    for key, a in A.items():
        kx, kz = _kernels_for(key[0])
        swn = _gsum(a['swn']); swd = _gsum(a['swd']); bipd = _gsum(a['bipd'])
        wh_coll = _smooth(a['whn'], a['swn'], _gsum(a['whn']) / max(swn, 1),
                          K_WHIFF, kx, kz)
        if WH_COUNT_LEVEL:
            WH[key] = {}
            for c in COUNTS:
                m = wh_mult.get(c, 1.0)
                prior_c = [[min(1.0, wh_coll[i][j] * m) for j in range(NZ)]
                           for i in range(NX)]
                ac = AC.get((key, c))
                WH[key][c] = (prior_c if ac is None else
                              _smooth(ac['whn'], ac['swn'], prior_c,
                                      K_WH_COUNT, kx, kz))
        else:
            WH[key] = {c: wh_coll for c in COUNTS}
        FL[key] = _smooth(a['fln'], a['swn'], _gsum(a['fln']) / max(swn, 1), K_FOUL, kx, kz)
        XW[key] = _smooth(a['bipn'], a['bipd'], _gsum(a['bipn']) / max(bipd, 1), K_XWCON, kx, kz)
        coll = _smooth(a['swn'], a['swd'], swn / swd if swd else 0.0, K_SWING_COLL, kx, kz)
        if SWING_PRIOR_COUNT_LEVEL:
            SW[key] = {}
            for c in COUNTS:
                m = cnt_mult.get(c, 1.0)
                prior_c = [[min(1.0, coll[i][j] * m) for j in range(NZ)]
                           for i in range(NX)]
                SW[key][c] = _smooth(AC[(key, c)]['swn'], AC[(key, c)]['swd'],
                                     prior_c, K_SWING_COUNT, kx, kz)
        else:
            SW[key] = {c: _smooth(AC[(key, c)]['swn'], AC[(key, c)]['swd'], coll,
                                  K_SWING_COUNT, kx, kz)
                       for c in COUNTS}

    # Count-anchoring offsets for the BIP value branch (empty dict = off).
    BIPOFF = (build_bip_count_offsets(baseline, lg_woba, woba_scale)
              if (BIP_COUNT_ANCHOR and has_guts) else {})

    # Contact-quality level offsets by count (XW_COUNT_LEVEL); thin counts
    # fall back to the pooled 2021-2026 table.
    XWOFF = {}
    if XW_COUNT_LEVEL and has_guts:
        tot_s = sum(s for s, _n in xw_cnt.values())
        tot_n = sum(n for _s, n in xw_cnt.values())
        overall = tot_s / tot_n if tot_n else 0.0
        XWOFF = dict(XW_CLEVEL_FALLBACK)
        for c, (s, n) in xw_cnt.items():
            if n >= XW_CLEVEL_MIN_BIP:
                XWOFF[c] = s / n - overall

    return {'RV': RV, 'PCS': PCS, 'WH': WH, 'FL': FL, 'XW': XW, 'SW': SW,
            'BIPOFF': BIPOFF, 'XWOFF': XWOFF}


# ═════════════════════════════════════════════════════════════════════════
#  SCORE
# ═════════════════════════════════════════════════════════════════════════
def score_pitch(p, S):
    """Expected hitter-perspective RV for one pitch (lower = better for the
    pitcher). None if context missing or the (group,hand) surface is absent."""
    key = (group_of(p), p.get('Bats'), p.get('Throws'))
    if key not in S['WH']:
        return None
    c = get_count(p)
    px = safe_float(p.get('PlateX')); zn = _znorm(p)
    if c is None or px is None or zn is None:
        return None
    i = _xbin(px); j = _zbin(zn)
    psw = S['SW'][key][c][i][j]
    pwh = S['WH'][key][c][i][j]
    pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    # BIP value: contact-quality count offset (XWOFF, 2026-08-15) plus the
    # legacy anchor slot (BIPOFF, empty while BIP_COUNT_ANCHOR is off).
    vbip = (S['XW'][key][i][j] + S['XWOFF'].get(c, 0.0)
            + S['BIPOFF'].get(c, 0.0))
    pcs = S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0) + pbip * vbip
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


def _aggregate(pitches_by_key, S, want_zone=False, want_heatmap=False):
    """Mean ExpRV per key, plus optional zone rollups / heatmap grid."""
    out = {}
    for key, pitches in pitches_by_key.items():
        vals = []
        zone_acc = defaultdict(list) if want_zone else None
        cell_acc = defaultdict(lambda: [0.0, 0]) if want_heatmap else None
        for p in pitches:
            if not _is_scorable(p):
                continue
            v = score_pitch(p, S)
            if v is None:
                continue
            vals.append(v)
            if want_zone:
                z = classify_zone(p)
                if z is not None:
                    zone_acc[z].append(v)
            if want_heatmap:
                i = _xbin(safe_float(p.get('PlateX'))); j = _zbin(_znorm(p))
                cell_acc[(i, j)][0] += v; cell_acc[(i, j)][1] += 1
        if not vals:
            continue
        rec = {'raw_loc': sum(vals) / len(vals), 'n_pitches': len(vals)}
        if want_zone:
            rec['zone_loc'] = {z: (sum(vs) / len(vs) if vs else None)
                               for z, vs in zone_acc.items()}
        if want_heatmap:
            rec['heatmap'] = [[i, j, round(s / n, 4), n]
                              for (i, j), (s, n) in sorted(cell_acc.items())]
        out[key] = rec
    return out


# ═════════════════════════════════════════════════════════════════════════
#  REGRESS + NORMALIZE
# ═════════════════════════════════════════════════════════════════════════
def _is_combined_team(t):
    return isinstance(t, str) and t.endswith('TM') and t[:-2].isdigit()


def _pool_identity(k):
    # Player identity independent of team (team is k[1]); groups a combined
    # 2TM/3TM row with its per-team stint rows so we can keep only the combined
    # row in the normalization pool (matching the percentile-pool convention).
    return k[:1] + k[2:]


def _in_norm_pool(k, pool_filter, combined_ids):
    if not pool_filter(k):
        return False
    # Exclude per-team stint rows of a pitcher who also has a combined row.
    if not _is_combined_team(k[1]) and _pool_identity(k) in combined_ids:
        return False
    return True


def _normalize(raw, n_prior, min_pool, pool_filter, return_anchors=False):
    """Bayesian-regress each raw_loc toward the pool league mean, then z-score
    to locPlus = 100 - K·z. Adds 'raw_loc_adj', 'locPlus', 'locRuns100'.
    Mutates and returns the same dict. With return_anchors=True also returns
    (mu, sigma) of the pool — reused for unshrunk window averages on cards."""
    if not raw:
        return (raw, (None, None)) if return_anchors else raw
    combined_ids = {_pool_identity(k) for k in raw if _is_combined_team(k[1])}
    pool = {k: v for k, v in raw.items()
            if _in_norm_pool(k, pool_filter, combined_ids) and v['n_pitches'] >= min_pool}
    if not pool:
        for v in raw.values():
            v['raw_loc_adj'] = v['raw_loc']; v['locPlus'] = 100.0; v['locRuns100'] = 0.0
        return (raw, (None, None)) if return_anchors else raw
    lg_raw = sum(v['raw_loc'] for v in pool.values()) / len(pool)
    for v in raw.values():
        n = v['n_pitches']
        v['raw_loc_adj'] = (n * v['raw_loc'] + n_prior * lg_raw) / (n + n_prior)
    pool_adj = [raw[k]['raw_loc_adj'] for k in pool]
    mu = sum(pool_adj) / len(pool_adj)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in pool_adj) / len(pool_adj))
    for v in raw.values():
        if sigma > 1e-12:
            z = (v['raw_loc_adj'] - mu) / sigma
            v['locPlus'] = round(100.0 - LOC_SCALE_K * z, 1)
        else:
            v['locPlus'] = 100.0
        # interpretable tooltip: location runs saved per 100 pitches (pitcher persp)
        v['locRuns100'] = round(-(v['raw_loc_adj'] - lg_raw) * 100.0, 3)
    return (raw, (mu, sigma)) if return_anchors else raw


def _normalize_by_group(raw, group_fn, n_prior, min_pool, pool_filter,
                        return_anchors=False):
    """Per-pitch-type rows standardized within their pitch-type GROUP.
    n_prior may be a scalar or a per-group dict (measured per-group
    stabilization constants). With return_anchors=True also returns
    {group: (mu, sigma)} — the unit-level anchors, reused by the per-pitch
    grade dump so sheet grades share the exact leaderboard transform."""
    anchors = {}
    if not raw:
        return (raw, anchors) if return_anchors else raw
    combined_ids = {_pool_identity(k) for k in raw if _is_combined_team(k[1])}
    by_group = defaultdict(dict)
    for k, v in raw.items():
        by_group[group_fn(k)][k] = v
    for grp, rows in by_group.items():
        grp_prior = (n_prior.get(grp, N_PRIOR_PT_DEFAULT)
                     if isinstance(n_prior, dict) else n_prior)
        pool = {k: v for k, v in rows.items()
                if _in_norm_pool(k, pool_filter, combined_ids) and v['n_pitches'] >= min_pool}
        if not pool:
            for v in rows.values():
                v['raw_loc_adj'] = v['raw_loc']; v['locPlus'] = 100.0; v['locRuns100'] = 0.0
            continue
        lg_raw = sum(v['raw_loc'] for v in pool.values()) / len(pool)
        for v in rows.values():
            n = v['n_pitches']
            v['raw_loc_adj'] = (n * v['raw_loc'] + grp_prior * lg_raw) / (n + grp_prior)
        pool_adj = [rows[k]['raw_loc_adj'] for k in pool]
        mu = sum(pool_adj) / len(pool_adj)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in pool_adj) / len(pool_adj))
        anchors[grp] = (mu, sigma)
        for v in rows.values():
            if sigma > 1e-12:
                z = (v['raw_loc_adj'] - mu) / sigma
                v['locPlus'] = round(100.0 - LOC_SCALE_K * z, 1)
            else:
                v['locPlus'] = 100.0
            v['locRuns100'] = round(-(v['raw_loc_adj'] - lg_raw) * 100.0, 3)
    out = {}
    for rows in by_group.values():
        out.update(rows)
    return (out, anchors) if return_anchors else out


# ═════════════════════════════════════════════════════════════════════════
#  SERIALIZE (metadata / audit)
# ═════════════════════════════════════════════════════════════════════════
def serialize_surfaces(S):
    """Compact, JSON-friendly snapshot for metadata: config + count scalars +
    the league value surface per group×hand (the count-collapsed ExpRV proxy
    is reconstructable client-side from these if needed for a league heatmap)."""
    return {
        'config': {'binX_in': 2.0, 'binZ_frac': BIN_Z, 'nx': NX, 'nz': NZ,
                   'xMin': X_MIN, 'xMax': X_MAX, 'zMin': Z_MIN, 'zMax': Z_MAX,
                   'physX_in': PHYS_X_IN, 'physZ_frac': PHYS_Z_FRAC,
                   'scaleK': LOC_SCALE_K, 'nPriorOverall': N_PRIOR_OVERALL,
                   'nPriorPt': N_PRIOR_PT, 'groups': GROUPS,
                   'pcsByHand': PCS_BY_HAND,
                   'bipCountAnchor': BIP_COUNT_ANCHOR,
                   'swingPriorCountLevel': SWING_PRIOR_COUNT_LEVEL,
                   'csCountTransform': CS_COUNT_TRANSFORM,
                   'whCountLevel': WH_COUNT_LEVEL,
                   'xwCountLevel': XW_COUNT_LEVEL},
        'countValues': {slot: {f"{c[0]}-{c[1]}": round(v, 5) for c, v in d.items()}
                        for slot, d in S['RV'].items()},
    }


# ═════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY
# ═════════════════════════════════════════════════════════════════════════
def compute_loc_plus(all_pitches, pitches_by_pitcher, pitches_by_pitch_type,
                     lg_woba, woba_scale, dump_pitch_grades_path=None,
                     return_anchors=False):
    """Main entry point. Signature preserved for process_data.py.

    Returns:
        pitcher_results: dict[(pitcher, team, throws)] ->
            {locPlus, raw_loc_adj, n_pitches, zone_loc, heatmap, locRuns100}
        pitch_results:   dict[(pitcher, team, pitch_type, throws)] ->
            {locPlus, raw_loc_adj, n_pitches, locRuns100}  (std within group)
        weight_table_json: metadata dict
    """
    baseline = [p for p in all_pitches if is_eligible_baseline(p)]
    S = build_surfaces(baseline, lg_woba, woba_scale)

    pitcher_raw = _aggregate(pitches_by_pitcher, S, want_zone=True, want_heatmap=True)
    pitcher_results, ov_anchors = _normalize(
        pitcher_raw, N_PRIOR_OVERALL, MIN_POOL_OVERALL,
        pool_filter=lambda k: k[1] not in AAA_TEAMS,
        return_anchors=True)

    pitch_raw = _aggregate(pitches_by_pitch_type, S)
    pitch_results, group_anchors = _normalize_by_group(
        pitch_raw, group_fn=lambda k: group_of_code(k[2]),
        n_prior=N_PRIOR_PT, min_pool=MIN_POOL_PT,
        pool_filter=lambda k: k[1] not in AAA_TEAMS,
        return_anchors=True)

    # ── COHERENT CANON (2026-07-18, per Wally): every displayed Loc+ —
    # overall AND per-type — is the plain mean of the per-pitch INTEGER
    # atoms, so a window card, a filtered site view, and these leaderboard
    # values agree to the digit. (Precision caveat, 2026-08-27 audit: the
    # sheet write-back dumps round(g, 2), not the integer atom, so a sheet
    # AVERAGEIF reconciles to within ~0.005/pitch rather than exactly;
    # int(round()) is also half-to-even where Sheets ROUND is half-up.) The _normalize machinery above still
    # supplies the anchors and the raw_loc_adj / locRuns100 / heatmap
    # fields; only the displayed locPlus is overridden here. Overall uses
    # the per-GROUP anchors per pitch (not the old pitcher-population
    # standardization — that scale made overall Loc+ jump when site filters
    # engaged the atoms).
    _grade_memo = {}

    def _pitch_grade(p):
        """Full-precision per-pitch grade (same anchors as the dump)."""
        pid = id(p)
        if pid in _grade_memo:
            return _grade_memo[pid]
        g = None
        if _is_scorable(p):
            anc = group_anchors.get(group_of(p))
            if anc and anc[1] is not None and anc[1] > 1e-12:
                v = score_pitch(p, S)
                if v is not None:
                    g = 100.0 - LOC_SCALE_K * (v - anc[0]) / anc[1]
        _grade_memo[pid] = g
        return g

    def _atom_override(results, groups_by_key):
        for key, plist in groups_by_key.items():
            if key not in results:
                continue
            atoms = [int(round(g)) for g in map(_pitch_grade, plist)
                     if g is not None]
            if atoms:
                results[key]['locPlus'] = round(sum(atoms) / len(atoms), 1)

    _atom_override(pitcher_results, pitches_by_pitcher)
    _atom_override(pitch_results, pitches_by_pitch_type)

    # ── Per-pitch grade dump for the Sheets write-back (2026-07-18, per
    # Wally). Scale (i), unregressed: each pitch's raw ExpRV graded with the
    # SAME group anchors as the leaderboard Loc+ (no n_prior shrink — a pitch
    # grade describes the pitch, not the pitcher's sample). Keyed by sheet
    # position ("tab\trow", attached by pipeline_fetch); EP is excluded
    # upstream (the caller passes an EP-filtered pitch list).
    if dump_pitch_grades_path:
        import json as _json
        grades = {}
        for p in all_pitches:
            tab, rownum = p.get('_sheet_tab'), p.get('_sheet_row')
            if tab is None or rownum is None:
                continue
            # UNCLIPPED full precision: clipping (if any) is display-only.
            g = _pitch_grade(p)
            if g is None:
                continue
            grades[f'{tab}\t{int(rownum)}'] = round(g, 2)
        with open(dump_pitch_grades_path, 'w') as f:
            _json.dump(grades, f, separators=(',', ':'))
        print(f"  Loc+ per-pitch grade dump: {len(grades)} pitches -> "
              f"{dump_pitch_grades_path}")

    if return_anchors:
        # 'surfaces' = the raw surface object (NOT serialized) so callers can
        # score additional pitches per-pitch via score_pitch(p, surfaces) with
        # the exact same anchors — the card window-grade path uses this.
        return (pitcher_results, pitch_results, serialize_surfaces(S),
                {'overall': ov_anchors, 'pt': group_anchors, 'surfaces': S})
    return pitcher_results, pitch_results, serialize_surfaces(S)


# ═════════════════════════════════════════════════════════════════════════
#  STANDALONE VALIDATION  (reproduces the V3 lab leaderboard)
# ═════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import pickle, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        ALL = pickle.load(f)
    by_pitcher = defaultdict(list)
    by_pt = defaultdict(list)
    for p in ALL:
        k = (p.get('Pitcher'), p.get('PTeam'), p.get('Throws'))
        by_pitcher[k].append(p)
        by_pt[(p.get('Pitcher'), p.get('PTeam'), p.get('Pitch Type'), p.get('Throws'))].append(p)
    pr, ptr, meta = compute_loc_plus(ALL, by_pitcher, by_pt,
                                     lg_woba=0.3169, woba_scale=1.2393)
    qual = {k: v for k, v in pr.items() if v['n_pitches'] >= 400
            and k[1] not in AAA_TEAMS}
    order = sorted(qual, key=lambda k: -qual[k]['locPlus'])
    vals = [qual[k]['locPlus'] for k in order]
    print(f"qualified pitchers (>=400 pitches): {len(qual)}")
    print(f"locPlus range: {min(vals):.0f} .. {max(vals):.0f}   "
          f"mean={sum(vals)/len(vals):.1f}")
    print("\nTOP 10:")
    for k in order[:10]:
        v = qual[k]
        print(f"  {k[0]:24s} {k[1]:4s}  locPlus={v['locPlus']:5.1f}  "
              f"runs/100={v['locRuns100']:+.2f}  n={v['n_pitches']}")
    print("BOTTOM 6:")
    for k in order[-6:]:
        v = qual[k]
        print(f"  {k[0]:24s} {k[1]:4s}  locPlus={v['locPlus']:5.1f}  "
              f"runs/100={v['locRuns100']:+.2f}  n={v['n_pitches']}")
    # sanity: a multi-pitch starter's per-type rows
    print("\nexample per-pitch-type rows (Skubal if present):")
    for key, v in sorted(ptr.items(), key=lambda kv: -kv[1]['n_pitches']):
        if 'Skubal' in (key[0] or ''):
            print(f"  {key[0]} {key[2]:3s} ({key[1]}): locPlus={v['locPlus']:.1f}  n={v['n_pitches']}")
