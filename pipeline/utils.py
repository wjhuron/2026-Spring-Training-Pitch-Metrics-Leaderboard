#!/usr/bin/env python3
"""Shared utility functions and constants for the leaderboard pipeline."""

import math
import os
from collections import defaultdict
from datetime import datetime, time

# ── Paths ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')


# ── Strike zone constants ────────────────────────────────────────────────
# Pitching+ blend weight, SINGLE SOURCE OF TRUTH. Lives here rather than in
# train_stuff (where it is derived and documented) because process_data
# must read it without importing xgboost/sklearn. Every surface that renders
# a Pitching+ number imports THIS: train_stuff, process_data's per-pitch
# grade atoms, Cards, and scripts/sheets_write_grades.
#
# It is one constant with four consumers, and it has drifted before. The
# 0.70 -> 0.80 move on 2026-07-25 updated three of them; process_data's atom
# was missed and sat on 0.70 until 2026-08-12, so FILTERED Pitching+ on the
# site (which averages those atoms) disagreed with the unfiltered value,
# cards, and sheets for ~3 weeks — 41% of pitchers by a point or more, up to
# 12 points. That breaks the coherent-canon invariant (sheets = cards = site,
# filtered or not). Never hardcode the number again; import it.
#
# Derivation and the flat-region argument: train_stuff.PITCHING_W_STUFF.
# Re-audited 2026-08-12 under Stuff+ v12 (scripts/research/stuff/pitchingplus_command_loso.py):
# kept at 0.80. Re-swept 2026-08-23 on the NEXT-SEASON objective under v14
# (scripts/research/stuff/pitchingplus_nxt_sweep.py, data/_pplus_nxt_sweep.json):
# luck-neutral argmax 0.70, flat [0.65, 0.76]; actual-RV argmax 0.73, flat
# [0.67, 0.79]; 0.80 sat outside the flat region (0.70-vs-0.80 won 4/5
# pairs, mean +0.009, z ~2.2). 0.72 is a CONVENTION inside both flat
# regions, chosen to match the 2021-25 within-season argmin; any value in
# [0.67, 0.76] is defensible, 0.80 is not.
PITCHING_W_STUFF = 0.72

# ── Hitter-basis pulled-air xwOBA term (xwOBA_hb) ────────────────────────
# Savant per-pitch xwOBA is EV/LA-only and underrates pulled air balls.
# The hitter-side xwOBA applies, per non-bunt air BIP (LA >= 20):
#     xwOBA_hb = xwOBA + C * (is_pull - live league pull share)
# C measured 2026-08-24 (scripts/research/hitter/xwrc_pullair_adjust.py):
# 2021-2025 replicates, DESC LOSO 4/5, interior optima c* .12-.25;
# confirmed on the live 2026 board (xwrc_pullair_2026_check.py, +.014 r
# vs wOBA, plateau .20-.30). 0.20 is the interior consensus of the
# replicate argmaxes and sits inside the live plateau. Descriptive only —
# the predictive test failed 0/4, which is fine for a descriptive stat.
# Scope: hitter row xwOBA + hitter micro atoms + xwRC+ ONLY. Never
# pitcher xwOBA against, xwOBAcon, xwOBAsp/SACQ, or RV/xRV (see the
# xwOBA_hb block in process_data.process_game_type).
XWOBA_PULLAIR_C = 0.20
XWOBA_PULLAIR_LA = 20.0       # air-ball floor, deg (the battery's la20 set,
                              # which beat the bb_type=fly definition)
XWOBA_PULLAIR_LGSHARE = 0.28  # frozen fallback league pull share of air
                              # BIPs; measured .261-.283 (2021-2026), used
                              # only when the live pool is under 5000

BALL_RADIUS_FT = 1.45 / 12   # 1.45 inches = ~0.121 ft
ZONE_HALF_WIDTH = 0.83        # half plate (8.5") + ball radius (1.45") in feet,
                              # ROUNDED (exact is 9.95/12 = 0.82917). Kept as
                              # the zone-BAND convention for SD+/heart-shadow
                              # regions; the InZone flag itself uses the exact
                              # rounded-rect geometry in compute_in_zone.

# ── Event classification sets ────────────────────────────────────────────
SWING_DESCRIPTIONS = {'Swinging Strike', 'Foul', 'In Play'}
HIT_EVENTS = {'Single', 'Double', 'Triple', 'Home Run'}
K_EVENTS = {'Strikeout', 'Strikeout Double Play'}
BB_EVENTS = {'Walk', 'Intent Walk'}
HBP_EVENTS = {'Hit By Pitch'}
SF_EVENTS = {'Sac Fly', 'Sac Fly Double Play'}
SH_EVENTS = {'Sac Bunt', 'Sac Bunt Double Play'}
CI_EVENTS = {'Catcher Interference'}
NON_PA_EVENTS = {
    'Caught Stealing 2B', 'Caught Stealing 3B', 'Caught Stealing Home',
    'Pickoff 1B', 'Pickoff 2B', 'Pickoff 3B',
    'Pickoff Caught Stealing 2B', 'Pickoff Caught Stealing 3B',
    'Pickoff Caught Stealing Home',
    'Runner Out', 'Wild Pitch', 'Game Advisory',
    'Stolen Base 2B', 'Stolen Base 3B', 'Stolen Base Home',
    'Balk', 'Passed Ball',
}
BUNT_BB_TYPES = {'bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}


# ── No-pitch plate appearances ────────────────────────────────────────────
# Since 2017 an automatic intentional walk contains no pitches, so it is a PA
# with nothing thrown. Those rows carry PitchID `{game_pk}_{atbat}_00`; real
# pitch numbers are 1-based, so `_00` is unambiguous (verified: zero existing
# PitchIDs end in _00).
#
# The row must COUNT as a plate appearance and, for a hitter, as a walk. It
# must NOT count as a pitch. Every per-pitch denominator therefore has to
# exclude it, and the ones that do were found by measurement rather than by
# reading code — see scripts/research/misc/ibb_injection_test.py, which
# injects rows into the golden harness and diffs the whole pipeline.
#
# Most filters are already safe because they key on a column being None
# (BBType, InZone, Description), which a no-pitch row has. The ones that are
# NOT safe count rows directly, and those are the call sites of real_pitches.

def is_no_pitch(row):
    """True for a plate-appearance marker row with no pitch thrown."""
    pid = row.get('PitchID')
    return bool(pid) and str(pid).endswith('_00')


def real_pitches(rows):
    """Only rows where a pitch was actually thrown.

    Use for any denominator that counts PITCHES. Do NOT use where the
    denominator is plate appearances — a no-pitch IBB is a real PA.
    """
    return [r for r in rows if not is_no_pitch(r)]

# ── Team sets ────────────────────────────────────────────────────────────
MLB_TEAMS = {
    'ARI', 'ATH', 'ATL', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL', 'CWS',
    'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSH',
    'WBC',
}
# 'AAA' included defensively: AAA-source rows (the NLE2026 AAA tab) carry
# team 'AAA', which would otherwise slip past `team not in AAA_TEAMS`
# checks and leak into MLB normalization pools.
AAA_TEAMS = {'ROC', 'AAA'}
ALL_TEAMS = MLB_TEAMS | AAA_TEAMS

# ── Leaderboard qualification thresholds ─────────────────────────────────
# Per official MLB qualifier standards. "Team games" (TG) = games the
# player's team has played. Rookies / Spring Training / Arizona Fall League
# rows are deliberately NOT modeled (this project doesn't surface them);
# Fielding qualifiers are out of scope.
#
#   MLB hitter:        3.1 PA × TG
#   MiLB/ROC hitter:   2.7 PA × TG
#   MLB starter:       1.0 IP × TG    (GS/G > SP_GS_RATIO)
#   MLB reliever:      0.30 IP × TG   (see note below)
#   MiLB/ROC starter:  0.8 IP × TG
#   MiLB/ROC reliever: 0.24 IP × TG
#
# Reliever coefficient (0.30): MLB has no official reliever qualifier,
# so we match FanGraphs' relief-pitching leaderboard, which is the de
# facto public standard. Empirically derived: across all 30 teams,
# IP >= 0.30 × team_games reproduces FanGraphs' qualified-reliever set
# with ZERO mismatches (every qualified reliever sits at >= 0.302 IP/TG,
# every non-qualified at <= 0.296 IP/TG, so 0.30 is the exact dividing
# line). The old 0.5 value was far too strict (it qualified only ~1
# reliever per team vs FanGraphs' ~6). ROC reliever (0.24) keeps the
# same 0.80 ROC-to-MLB scaling used for starters (0.8/1.0) and hitters.
#
# These must stay in sync with the JS mirror in js/aggregator.js (QUAL)
# and the Utils.hitterPaPerGame / Utils.pitcherIpPerGame helpers.
SP_GS_RATIO = 0.5

QUAL_PA_PER_GAME_MLB     = 3.1
QUAL_PA_PER_GAME_MILB    = 2.7
QUAL_SP_IP_PER_GAME_MLB  = 1.0
QUAL_RP_IP_PER_GAME_MLB  = 0.30
QUAL_SP_IP_PER_GAME_MILB = 0.8
QUAL_RP_IP_PER_GAME_MILB = 0.24


def hitter_pa_per_game(is_roc):
    """PA-per-team-game multiplier for hitter qualification."""
    return QUAL_PA_PER_GAME_MILB if is_roc else QUAL_PA_PER_GAME_MLB


def pitcher_ip_per_game(is_starter, is_roc):
    """IP-per-team-game multiplier for pitcher qualification."""
    if is_roc:
        return QUAL_SP_IP_PER_GAME_MILB if is_starter else QUAL_RP_IP_PER_GAME_MILB
    return QUAL_SP_IP_PER_GAME_MLB if is_starter else QUAL_RP_IP_PER_GAME_MLB


def is_combined_team(team):
    """'2TM', '3TM' ... '10TM'. A multi-team LABEL, not a franchise.

    It has no schedule of its own, so it must never supply a team-games
    denominator. Mirrors Utils.isCombinedTeam in js/utils.js.
    """
    return isinstance(team, str) and team.endswith('TM') and team[:-2].isdigit()


def player_key(row, name_key):
    """Group a player's per-team and combined rows.

    Keys on mlbId when present so two players sharing a name (two 'Max Muncy')
    do not collide. Mirrors Utils.playerKey in js/utils.js and the local
    _player_key in pipeline/compute.py.
    """
    mid = row.get('mlbId')
    if mid is not None and mid != '':
        return 'id:' + str(mid)
    return 'nm:' + str(row.get(name_key) or '')


def current_team_by_player(rows, name_key, roc_teams):
    """Map player key -> the MLB club the player most recently played for.

    A multi-team player is measured against the club he is on now (Wally,
    2026-08-19). Only MLB stints count, so a player sent down still measures
    against his last MLB club rather than against Rochester. Reads the
    lastGameDate that both row builders in process_data emit.

    Mirrors the denomTeam resolution in Utils.buildQualContext (js/utils.js);
    the two must agree or the site and the shipped percentiles diverge.
    """
    best = {}
    for row in rows:
        team = row.get('team')
        if is_combined_team(team) or team in roc_teams:
            continue
        date = row.get('lastGameDate')
        if not date:
            continue
        key = player_key(row, name_key)
        prev = best.get(key)
        # A tie is impossible in practice: one player plays for one club on one
        # date. Break it on team code so the answer never depends on row order.
        if prev is None or date > prev[0] or (date == prev[0] and team < prev[1]):
            best[key] = (date, team)
    return {k: v[1] for k, v in best.items()}


def box_key(name, team, mlb_id):
    """Aggregation / lookup key for boxscore stats.

    Prefers `mlbId|team` — the MLB ID is immune to name-spelling variation
    (accents, hyphens, periods, stray spaces), which otherwise splits one
    player's boxscore across multiple buckets and leaves the leaderboard
    row matching only a partial bucket. Falls back to `name|team` only
    when no MLB ID is resolved. `|team` is retained so genuinely traded
    players keep a correct per-team split. Used identically by the
    aggregation (pipeline_fetch) and the merge (process_data)."""
    return f"{mlb_id}|{team}" if mlb_id else f"{name}|{team}"

# ── Team abbreviation → MLB API team ID ──────────────────────────────────
TEAM_ABBREV_TO_ID = {
    'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112,
    'CWS': 145, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KCR': 118, 'LAA': 108, 'LAD': 119, 'MIA': 146,
    'MIL': 158, 'MIN': 142, 'NYM': 121, 'NYY': 147, 'ATH': 133,
    'PHI': 143, 'PIT': 134, 'SDP': 135, 'SFG': 137, 'SEA': 136,
    'STL': 138, 'TBR': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120,
    # Rochester (WSH AAA) — club id, not parent org: the MLB ID lookup
    # matches ROC hitters/pitchers on currentTeam.id == 534.
    'ROC': 534,
}


# ── Pure utility functions ───────────────────────────────────────────────

def is_swing(p):
    """The swing test for every rate denominator, everywhere.

    Bunts are not swings and not chases (Wally, 2026-08-15). 'Foul Bunt'
    and 'Missed Bunt' already sit outside SWING_DESCRIPTIONS, but a bunt
    put IN PLAY reads Description='In Play' with a bunt BBType, so the
    Description test alone lets it through. Single-homed because this
    predicate has to hold identically across the leaderboard, the micro
    counters the site sums for filtered views, and the cards.
    """
    return (p.get('Description') in SWING_DESCRIPTIONS
            and p.get('BBType') not in BUNT_BB_TYPES)


def get_count(p):
    """Parse 'Count' column (e.g., '2-2') into (balls, strikes). None if invalid.

    Single home — SD+/Loc+/xwOBA3D/compute all consume this one parser so a
    count-format quirk can never be handled differently across models."""
    c = p.get('Count')
    if not isinstance(c, str) or '-' not in c:
        return None
    try:
        b_str, s_str = c.split('-', 1)
        b, s = int(b_str), int(s_str)
    except (TypeError, ValueError):
        return None
    if not (0 <= b <= 3 and 0 <= s <= 2):
        return None
    return (b, s)


def _pctl(v, pool):
    """Percentile rank of v within pool (ties averaged), 0-100, INTEGER.

    Single home for the percentile convention shared by ERA+ and Pitcher+.
    Integer to match every other leaderboard percentile (the 1-decimal form
    made hdERA/hpERA tooltips read "68.4th" while everything else is "68th")."""
    if v is None or not pool:
        return None
    below = sum(1 for x in pool if x < v)
    ties = sum(1 for x in pool if x == v)
    return round(100.0 * (below + 0.5 * ties) / len(pool))


def safe_float(val):
    """Convert a value to float, returning None if not possible."""
    if val is None or val == '':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def normalize_date(val):
    """Normalize a date value to YYYY-MM-DD string."""
    if val is None or val == '':
        return None
    if isinstance(val, datetime):
        return val.strftime('%Y-%m-%d')
    s = str(val).strip()
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return s[:10]
    parts = s.split('/')
    if len(parts) == 3:
        try:
            m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{y:04d}-{m:02d}-{d:02d}"
        except ValueError:
            pass
    return None


def _today_et():
    """Return today's date in US Eastern time (MLB schedule reference timezone)."""
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo('America/New_York')).date()
    except ImportError:
        return _dt.date.today()


def avg(values):
    """Average a list of numbers, ignoring None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def median(values):
    """Compute median, ignoring None values."""
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return None
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    return (nums[n // 2 - 1] + nums[n // 2]) / 2


def percentile(values, q):
    """Linear-interpolated percentile, ignoring None values. q is 0-100.

    Matches numpy.percentile's default ('linear') method, which is what the
    BB+ EV-ingredient derivation was run on
    (scripts/research/hitter/bbplus_ev_derivation.py).

    MIRRORED in js/aggregator.js as `percentileLinear`. BB+ is the one "+"
    the client recomputes under filters, so the two implementations must
    agree exactly or a filtered BB+ drifts from the shipped one. Change one
    and change the other in the same commit.
    """
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return None
    idx = (len(nums) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return nums[int(idx)]
    return nums[lo] + (nums[hi] - nums[lo]) * (idx - lo)


def round_metric(key, value):
    """Round a metric value according to its type."""
    if value is None:
        return None
    if key == 'Spin Rate':
        return round(value)
    if key in ('VAA', 'HAA'):
        return round(value, 2)
    return round(value, 1)


def is_barrel(ev, la):
    """Statcast barrel definition (MLB glossary / baseballr code_barrel).
    Five conditions: LA in [8,50], EV>=98, EV*1.5-LA>=117, EV+LA>=124."""
    if ev is None or la is None:
        return False
    return (la >= 8 and la <= 50 and ev >= 98 and
            ev * 1.5 - la >= 117 and
            ev + la >= 124)


# ── Spray angle functions ────────────────────────────────────────────────

def spray_angle(hc_x, hc_y):
    """Compute spray angle in degrees. 0 = center, negative = left field, positive = right field."""
    if hc_x is None or hc_y is None:
        return None
    hp_x, hp_y = 125.42, 198.27
    dx = hc_x - hp_x
    dy = hp_y - hc_y
    if dy <= 0:
        return None
    return math.atan2(dx, dy) * (180 / math.pi)


def spray_direction(angle, stands):
    """Classify spray direction into 6 equal 15° bins based on spray angle and batter side."""
    if angle is None or not stands:
        return None
    if stands == 'R':
        if angle < -30:
            return 'pull'
        elif angle < -15:
            return 'pull_side'
        elif angle < 0:
            return 'center_pull'
        elif angle < 15:
            return 'center_oppo'
        elif angle < 30:
            return 'oppo_side'
        else:
            return 'oppo'
    else:  # L
        if angle > 30:
            return 'pull'
        elif angle > 15:
            return 'pull_side'
        elif angle > 0:
            return 'center_pull'
        elif angle > -15:
            return 'center_oppo'
        elif angle > -30:
            return 'oppo_side'
        else:
            return 'oppo'


# ── Break Tilt / clock-face functions ────────────────────────────────────

def break_tilt_to_minutes(val):
    """Convert a time value (clock notation) to total minutes (0-719)."""
    if val is None:
        return None
    if isinstance(val, time):
        return val.hour * 60 + val.minute
    if isinstance(val, datetime):
        return val.hour * 60 + val.minute
    if isinstance(val, str) and ':' in val:
        try:
            parts = val.strip().split(':')
            h, m = int(parts[0]), int(parts[1])
            return h * 60 + m
        except (ValueError, IndexError):
            return None
    return None


def circular_mean_minutes(minute_values):
    """Circular mean for clock-face values (0-719 minutes = 12 hours)."""
    if not minute_values:
        return None
    angles = [m / 720.0 * 2 * math.pi for m in minute_values]
    sin_avg = sum(math.sin(a) for a in angles) / len(angles)
    cos_avg = sum(math.cos(a) for a in angles) / len(angles)
    avg_angle = math.atan2(sin_avg, cos_avg)
    if avg_angle < 0:
        avg_angle += 2 * math.pi
    avg_minutes = avg_angle / (2 * math.pi) * 720
    return round(avg_minutes)


def minutes_to_tilt_display(total_minutes):
    """Convert minutes back to H:MM display format."""
    if total_minutes is None:
        return None
    h = int(total_minutes) // 60
    m = int(total_minutes) % 60
    if h == 0:
        h = 12
    return f"{h}:{m:02d}"


# ── Strike zone ──────────────────────────────────────────────────────────

HALF_PLATE_FT = 8.5 / 12     # rulebook zone half-width (17" plate / 2)


def compute_in_zone(p):
    """'Yes' when any part of the ball intersects the rulebook strike zone.

    Exact Savant-matching rule (validated 2026-08-03 against the Statcast
    `zone` field, 3 independent days / 11,574 pitches / 0 mismatches):
    the ball CENTER must be within one ball radius of the zone RECTANGLE
    (half-plate wide, SzBot..SzTop tall) — Euclidean distance, so the
    expanded boundary has rounded corners. The previous rectangular
    approximation (|px| <= 0.83 with independent vertical margins)
    over-included the four corner regions plus a 0.0008 ft horizontal
    sliver (0.83 vs the exact 9.95/12): ~0.2%% of pitches league-wide.
    """
    px = safe_float(p.get('PlateX'))
    pz = safe_float(p.get('PlateZ'))
    top = safe_float(p.get('SzTop'))
    bot = safe_float(p.get('SzBot'))
    if any(v is None for v in [px, pz, top, bot]):
        return None
    dx = max(0.0, abs(px) - HALF_PLATE_FT)
    dz = max(0.0, bot - pz, pz - top)
    return 'Yes' if math.hypot(dx, dz) <= BALL_RADIUS_FT else 'No'


# ── IP conversion ────────────────────────────────────────────────────────

def outs_to_ip_str(outs):
    """Convert total outs to IP string notation (e.g., 19 outs -> '6.1')."""
    full = outs // 3
    remainder = outs % 3
    return f"{full}.{remainder}"


def outs_to_ip_float(outs):
    """Convert outs to float for calculations like ERA (19 outs -> 6.333...)."""
    return outs / 3.0


def ip_str_to_float(ip_str):
    """Convert baseball IP string to float. '6.1' -> 6.333, '6.2' -> 6.667, '6.0' -> 6.0."""
    if not ip_str:
        return 0.0
    parts = str(ip_str).split('.')
    full = int(parts[0])
    thirds = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return full + thirds / 3.0


# ── Name formatting ──────────────────────────────────────────────────────

def _fullname_to_lastfirst(full_name):
    """Convert 'First Last' to 'Last, First'. Simple split — handles most cases."""
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name
    suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}
    suffix = ''
    if len(parts) > 2 and parts[-1].lower().rstrip('.') in suffixes:
        suffix = ' ' + parts.pop()
    return parts[-1] + suffix + ', ' + ' '.join(parts[:-1])


def compute_runexp_scale(all_pitches, min_cell=30, min_mag=0.005):
    """Per-`_source` factors converting MiLB RunExp into MLB run currency.

    Statcast's delta_run_exp is built on each league's own run-expectancy
    matrix, so the SAME event carries a larger magnitude in MiLB.

    A SINGLE global factor is not enough — measured 2026-07-25, one WLS slope
    per source cut the weighted cell-level error only 49% -> 21%, and the
    residual was not uniform (Ball over-corrected -7%, Foul under-corrected
    +14% even after holding count mix fixed). The two leagues' RE matrices
    differ in SHAPE, not just scale, so the correction is per (Description,
    Count) cell, with fallbacks:

        cell factor   mlb_mean / milb_mean for that (Description, Count)
        desc factor   same ratio pooled over counts, for thin cells
        global factor WLS slope through the origin, for anything left

    Aligning cell means also neutralizes the leagues' different base-out
    MIX within a cell, which is context the pitcher does not control — the
    same translation framing xwOBAcon / xwOBAsp / Loc+ already use for ROC.
    Crucially it is a MULTIPLIER, not a substitution, so each pitch keeps its
    own base-out variation around that cell mean; replacing values with a
    lookup table would flatten ROC RV to context-neutral while MLB RV stayed
    context-included, making one column mean two different things by source.

    Estimated from NON-BIP cells only. 'In Play' is excluded from ESTIMATION
    because its gap is mostly real — ROC hitters genuinely produce more on
    contact (mean -0.002 vs MLB -0.048) and that must survive as signal, not
    be scaled away — but BIP still RECEIVES the global factor, since
    delta_run_exp is in league run units either way.

    Guards: a ratio is only used when both means clear `min_mag` and share a
    sign, so near-zero cells can't produce exploding or sign-flipping factors.

    Measured every run rather than hardcoded: run environments drift.
    """
    cells = defaultdict(lambda: [0.0, 0])
    descs = defaultdict(lambda: [0.0, 0])
    for p in all_pitches:
        d = p.get('Description')
        c = p.get('Count')
        if not d or not c or d == 'In Play':
            continue
        re = safe_float(p.get('RunExp'))
        if re is None:
            continue
        src = p.get('_source') or 'MLB'
        cells[(src, d, c)][0] += re; cells[(src, d, c)][1] += 1
        descs[(src, d)][0] += re; descs[(src, d)][1] += 1

    mlb_cell = {(d, c): s / n for (src, d, c), (s, n) in cells.items()
                if src == 'MLB' and n >= min_cell}
    mlb_desc = {d: s / n for (src, d), (s, n) in descs.items() if src == 'MLB'}

    def _ratio(mlb_v, milb_v):
        if (mlb_v is None or abs(mlb_v) < min_mag or abs(milb_v) < min_mag
                or (mlb_v > 0) != (milb_v > 0)):
            return None
        return milb_v / mlb_v          # divide RunExp by this to reach MLB

    out = {}
    acc = defaultdict(lambda: [0.0, 0.0])
    for (src, d, c), (s, n) in cells.items():
        if src == 'MLB' or n < min_cell:
            continue
        m = mlb_cell.get((d, c))
        if m is None:
            continue
        v = s / n
        acc[src][0] += n * m * v
        acc[src][1] += n * m * m
    for src, (num, den) in acc.items():
        g = num / den if den > 0 else 1.0
        cell_f, desc_f = {}, {}
        for (s0, d, c), (s, n) in cells.items():
            if s0 != src or n < min_cell:
                continue
            r = _ratio(mlb_cell.get((d, c)), s / n)
            if r:
                cell_f[(d, c)] = r
        for (s0, d), (s, n) in descs.items():
            if s0 != src:
                continue
            r = _ratio(mlb_desc.get(d), s / n)
            if r:
                desc_f[d] = r
        out[src] = {'cell': cell_f, 'desc': desc_f, 'global': g}
    return out


def runexp_scale_to_json(scale):
    """compute_runexp_scale() output -> JSON-safe blob.

    The cell factors are keyed by a (Description, Count) TUPLE, which JSON
    cannot represent, so they are flattened to "desc\tcount". Published in
    metadata so consumers that read the pitch data directly (Cards.py, any
    pickle reader) can apply the SAME factors the site used instead of
    re-deriving them — re-deriving needs an MLB reference set, which a
    single card's pitches don't have.
    """
    if not scale:
        return {}
    return {src: {'cell': {f'{d}\t{c}': f for (d, c), f in s['cell'].items()},
                  'desc': dict(s['desc']),
                  'global': s['global']}
            for src, s in scale.items()}


def runexp_scale_from_json(blob):
    """Inverse of runexp_scale_to_json(); {} when missing or malformed."""
    if not blob:
        return {}
    out = {}
    for src, s in blob.items():
        try:
            cell = {}
            for k, f in (s.get('cell') or {}).items():
                d, _, c = k.partition('\t')
                cell[(d, c)] = f
            out[src] = {'cell': cell,
                        'desc': dict(s.get('desc') or {}),
                        'global': s.get('global') or 1.0}
        except (AttributeError, TypeError):
            continue
    return out


def runexp_factor(scale_for_source, description, count):
    """cell -> desc -> global fallback. `scale_for_source` is one entry of
    compute_runexp_scale()'s output."""
    if not scale_for_source:
        return None
    f = scale_for_source['cell'].get((description, count))
    if f is None:
        f = scale_for_source['desc'].get(description)
    if f is None:
        f = scale_for_source['global']
    return f if f and f > 0 else None
