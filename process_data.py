#!/usr/bin/env python3
"""Process pitching and hitting data from Google Sheets into JSON files for the leaderboard website."""

import gspread
from google.oauth2.service_account import Credentials
import json
import math
import os
import pickle
import re
import sys
import time as time_module
from datetime import datetime
from collections import defaultdict

# ── Pipeline modules ─────────────────────────────────────────────────────
from pipeline_utils import (
    safe_float, normalize_date, _today_et, avg, median, round_metric,
    is_barrel, spray_angle, spray_direction,
    break_tilt_to_minutes, circular_mean_minutes, minutes_to_tilt_display,
    compute_in_zone, outs_to_ip_str, outs_to_ip_float, ip_str_to_float,
    DATA_DIR,
    SWING_DESCRIPTIONS, HIT_EVENTS, K_EVENTS, BB_EVENTS, HBP_EVENTS,
    SF_EVENTS, SH_EVENTS, CI_EVENTS, NON_PA_EVENTS, BUNT_BB_TYPES,
    MLB_TEAMS, AAA_TEAMS, ALL_TEAMS, TEAM_ABBREV_TO_ID,
    BALL_RADIUS_FT, ZONE_HALF_WIDTH, box_key,
)
from pipeline_fetch import (
    fetch_guts_constants, fetch_sprint_speed, fetch_park_factors,
    fetch_hitter_positions,
    read_pitches_from_sheet, read_all_pitches_from_sheets,
    lookup_mlb_id, load_mlb_id_cache, save_mlb_id_cache, fetch_canonical_last_first,
    fetch_and_aggregate_boxscores, fetch_and_aggregate_milb_boxscores,
    SPREADSHEET_IDS, SERVICE_ACCOUNT_FILE,
    WOBA_WEIGHTS_FALLBACK, FIP_CONSTANT_FALLBACK,
)
from pipeline_compute import (
    compute_expected_stats, compute_stats, compute_xrv,
    compute_pitcher_batted_ball, compute_hitter_stats,
    compute_percentile_ranks, compute_percentile_ranks_with_aaa,
    METRIC_COLS, METRIC_KEYS, PITCH_STAT_KEYS, STAT_KEYS,
    PITCH_PCTL_KEYS, PITCH_BB_PCTL_KEYS, PITCH_BB_INVERT,
    PITCHER_INVERT_PCTL,
    HITTER_STAT_KEYS, HITTER_INVERT_PCTL,
    PITCHER_BB_KEYS, PITCHER_BB_INVERT,
)


# Stuff-injected keys preserved across re-processing (values are computed by
# train_stuff_v11 --inject; process_data only carries them over). Includes
# Pitching+ — omitting it here once shipped an embed with a blank Pitching+
# column whenever process_data ran without a subsequent --inject (2026-07-18).
XRVOE_KEYS = ('xrvoe100', 'rvoe100', 'rvoe', 'xrvoe',
              'pitchingScore', 'pitchingRuns100',
              # Pitcher+ is computed in the inject step too (it consumes the
              # fresh stuffScore), so it must survive a process_data-only run
              # the same way Pitching+ does. The carry-over loop appends
              # '_pctl' itself — listing the rank key here would look for
              # 'pitcherPlus_pctl_pctl'.
              'pitcherPlus', 'pitcherRuns100', 'pitcherPlusProj')

# ── Runtime state (set in main) ──────────────────────────────────────────
WOBA_WEIGHTS = None
FIP_CONSTANT = None
GUTS_EXTRA = None
PARK_FACTORS = None


def _bip_woba_value(event):
    """wOBA-numerator value for a batted ball, reproducing the (now-deleted)
    Statcast 'wOBAval' column that SACQ / xwOBAsp were built on.

    Uses Statcast's woba_value weights (single 0.9, double 1.25, triple 1.6,
    HR 2.0), under which reaching on error or a fielder's choice counts as a
    single (0.9) — matching the stored values exactly so the zone-wOBA tables
    are unchanged. All outs contribute 0. The matching wOBA denominator is
    always 1 for a non-bunt batted ball.
    """
    if event in ('Single', 'Field Error', 'Fielders Choice'):
        return 0.9
    if event == 'Double':   return 1.25
    if event == 'Triple':   return 1.6
    if event == 'Home Run': return 2.0
    return 0.0


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


def generate_micro_data(all_pitches, mlb_id_cache=None, ep_pitchers=None,
                        stuff_grades=None, loc_grades=None):
    """Generate micro-aggregate data for client-side date and opponent-hand filtering.

    stuff_grades / loc_grades: per-pitch grade dumps keyed "tab\\trow"
    (full-precision floats). Their nearest-integer atoms are summed into the
    pitcher/pitch micro rows so client-side filters can reproduce windowed
    Stuff+/Loc+/Pitching+ as plain averages — the same integers the Sheets
    grade columns hold, so a filtered site view, a window card, and a sheet
    AVERAGEIF all agree to the digit (coherent canon, 2026-07-18). Pitches
    absent from a dump are excluded from that metric's atom count (the n
    fields make partial coverage explicit; scripts/refresh_micro_grades.py
    re-runs this after the Stuff+ dump lands so nothing stays stale).

    ep_pitchers: set of (Pitcher, PTeam) position-player appearances. Their
    pitches are EXCLUDED from pitcher micro rows (so no client-side filter can
    surface a position player as a pitcher) but INCLUDED in hitter micro rows
    (all PAs count in hitter stats, matching official totals).

    Groups pitches by (person, date, opponent_hand) with summable counts.
    Returns a dict with compact arrays-of-arrays format for JSON serialization.

    Filter-responsive stats (recomputed client-side when date/hand filters change):
      Pitcher: velocity, spin, movement, nVAA/nHAA, whiff%, chase%, strike%, xIVB/xHB, etc.
      Hitter: EV, barrel%, hard-hit%, GB%, swing%, chase%, contact%, bat speed, etc.

    Season-level stats (NOT recomputed by filters — use pre-agg values):
      medLA, ldPct/fbPct/puPct, pullPct/middlePct/oppoPct, izSwingPct, izSwChase,
      contactPct, izContactPct, attackAngle/attackDirection/swingPathTilt,
      twoStrikeWhiffPct, firstPitchSwingPct, sprintSpeed, nCompSwings,
      runValue/rv100 (pitchers), xBA/xSLG/xwOBA/xwOBAcon (require Statcast model).
    """
    # --- Build lookup tables ---
    pitcher_set = set()
    hitter_set = set()
    team_set = set()
    date_set = set()
    pitch_type_set = set()

    ep_pitchers = ep_pitchers or set()

    def _grade_atoms(p):
        """(stuffAtom, locAtom, pitchingAtom) ints for one pitch, or Nones.
        Pitching+ atom blends the two INTEGER atoms (matches the sheet's
        =ROUND(0.7*X+0.3*Y,0) cells and Cards' cell-first atoms)."""
        tab, rownum = p.get('_sheet_tab'), p.get('_sheet_row')
        if not tab or not rownum:
            return None, None, None
        key = f'{tab}\t{rownum}'
        sv = stuff_grades.get(key) if stuff_grades else None
        lv = loc_grades.get(key) if loc_grades else None
        sa = int(round(sv)) if sv is not None else None
        la = int(round(lv)) if lv is not None else None
        pa = (int(round(0.7 * sa + 0.3 * la))
              if sa is not None and la is not None else None)
        return sa, la, pa

    for p in all_pitches:
        if p.get('Pitcher') and (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers:
            pitcher_set.add(p['Pitcher'])
        if p.get('PTeam') and p['PTeam'] in ALL_TEAMS:
            team_set.add(p['PTeam'])
        d = normalize_date(p.get('Game Date'))
        if d:
            date_set.add(d)
        if p.get('Pitch Type'):
            pitch_type_set.add(p['Pitch Type'])

    for p in all_pitches:
        if p.get('Batter'):
            hitter_set.add(p['Batter'])
        if p.get('BTeam') and p['BTeam'] in ALL_TEAMS:
            team_set.add(p['BTeam'])
        d = normalize_date(p.get('Game Date'))
        if d:
            date_set.add(d)

    pitchers = sorted(pitcher_set)
    hitters = sorted(hitter_set)
    teams = sorted(team_set)
    dates = sorted(date_set)
    pitch_types = sorted(pitch_type_set)

    pi_idx = {name: i for i, name in enumerate(pitchers)}
    hi_idx = {name: i for i, name in enumerate(hitters)}
    tm_idx = {name: i for i, name in enumerate(teams)}
    dt_idx = {d: i for i, d in enumerate(dates)}
    pt_idx = {pt: i for i, pt in enumerate(pitch_types)}

    # ==========================================================
    #  Pitcher micro-aggs
    #  Key: (pitcherIdx, teamIdx, throws, dateIdx, batterHand)
    #  Values: 28 count fields
    #  0:n  1:iz  2:sw  3:wh  4:csw  5:ooz  6:oozSw  7:bip  8:gb
    #  9:pa  10:h  11:hr  12:k  13:bb  14:hbp  15:sf  16:sh  17:ci
    #  18:izSw  19:izWh  20:firstPitches  21:firstPitchStrikes
    #  22:fb (fly balls)  23:nHrBip (HR on BIP, for HR/FB)  24:ldHr (line-drive HRs)
    #  25:pu (popups, for HR/FB denominator)  26:nStrikes  27:ibb
    #  28:oneOneTotal  29:oneOneWins  30:earlyActionPAs
    #  31:sumStuff  32:nStuff  33:sumLoc  34:nLoc  35:sumPitching  36:nPitching
    #  (integer grade-atom sums/counts — see _grade_atoms)
    # ==========================================================
    pitcher_micro = defaultdict(lambda: [0] * 37)

    for p in all_pitches:
        pitcher = p.get('Pitcher')
        team = p.get('PTeam')
        throws = p.get('Throws')
        date = normalize_date(p.get('Game Date'))
        batter_hand = p.get('Bats')

        if not pitcher or not team or team not in ALL_TEAMS:
            continue
        if (pitcher, team) in ep_pitchers:
            continue  # position players never get pitcher micro rows
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        if not date or not batter_hand:
            continue

        key = (pi_idx[pitcher], tm_idx[team], throws or '', dt_idx[date], batter_hand)
        c = pitcher_micro[key]

        c[0] += 1  # n
        in_zone = p.get('InZone') == 'Yes'
        if in_zone:
            c[1] += 1  # iz
        desc = p.get('Description', '')
        if desc in SWING_DESCRIPTIONS:
            c[2] += 1  # sw
            if in_zone:
                c[18] += 1  # izSw
        if desc == 'Swinging Strike':
            c[3] += 1  # wh
            if in_zone:
                c[19] += 1  # izWh
        if desc in ('Called Strike', 'Swinging Strike'):
            c[4] += 1  # csw
        if p.get('InZone') == 'No':
            c[5] += 1  # ooz
            if desc in ('Swinging Strike', 'In Play', 'Foul'):
                c[6] += 1  # oozSw
        if desc not in ('Ball', 'Intent Ball', 'Hit By Pitch', 'Pitchout'):
            c[26] += 1  # nStrikes
        bb_type = p.get('BBType')
        if bb_type and bb_type not in BUNT_BB_TYPES:
            c[7] += 1  # bip
            if bb_type == 'ground_ball':
                c[8] += 1  # gb
            if bb_type == 'fly_ball':
                c[22] += 1  # fb (fly balls for HR/FB)
            if bb_type == 'popup':
                c[25] += 1  # pu (popups for HR/FB)
            if p.get('Event') == 'Home Run':
                c[23] += 1  # nHrBip (HR on BIP)
                if bb_type == 'line_drive':
                    c[24] += 1  # ldHr (line-drive HRs for HR/FB denominator)
        event = p.get('Event')
        if event and event not in NON_PA_EVENTS:
            c[9] += 1   # pa
            if event in HIT_EVENTS:      c[10] += 1  # h
            if event == 'Home Run':      c[11] += 1  # hr
            if event in K_EVENTS:        c[12] += 1  # k
            if event in BB_EVENTS:       c[13] += 1  # bb (all walks including IBB)
            if event == 'Intent Walk':   c[27] += 1  # ibb
            if event in HBP_EVENTS:      c[14] += 1  # hbp
            if event in SF_EVENTS:       c[15] += 1  # sf
            if event in SH_EVENTS:       c[16] += 1  # sh
            if event in CI_EVENTS:       c[17] += 1  # ci
        # FPS counts (first pitch of PA: count == "0-0")
        if p.get('Count') == '0-0':
            c[20] += 1  # firstPitches
            if desc in ('Called Strike', 'Swinging Strike', 'Foul', 'In Play'):
                c[21] += 1  # firstPitchStrikes

        # 1-1 Win%: everything except balls/HBP/pitchout counts as winning the 1-1 pitch
        if p.get('Count') == '1-1':
            c[28] += 1  # oneOneTotal
            if desc not in ('Ball', 'Intent Ball', 'Hit By Pitch', 'Pitchout'):
                c[29] += 1  # oneOneWins

        # Early Action: PA ended in 3 or fewer pitches
        if event and event not in NON_PA_EVENTS:
            pitch_id = p.get('PitchID') or ''
            parts = pitch_id.split('_')
            if len(parts) == 3:
                try:
                    pitch_num = int(parts[2])
                    if pitch_num <= 3:
                        c[30] += 1  # earlyActionPAs
                except ValueError:
                    pass

        # Grade-atom sums (filtered overall Stuff+/Loc+/Pitching+)
        _sa, _la, _pa = _grade_atoms(p)
        if _sa is not None:
            c[31] += _sa; c[32] += 1
        if _la is not None:
            c[33] += _la; c[34] += 1
        if _pa is not None:
            c[35] += _pa; c[36] += 1

    pitcher_rows = []
    for (pi, ti, throws, di, bh), c in pitcher_micro.items():
        pitcher_rows.append([pi, ti, throws, di, bh] + c)

    # ==========================================================
    #  Pitch micro-aggs
    #  Key: (pitcherIdx, teamIdx, throws, pitchTypeIdx, dateIdx, batterHand)
    #  Values: 22 count fields + 29 metric fields = 51 fields
    #  0:n  1:iz  2:sw  3:wh  4:csw  5:ooz  6:oozSw  7:bip  8:gb
    #  9:pa  10:h  11:hr  12:k  13:bb  14:hbp  15:sf  16:sh  17:ci
    #  18:izSw  19:izWh  20:firstPitches  21:firstPitchStrikes
    #  Metric fields (offset from 22):
    #  22:sumVelo 23:nVelo  24:sumSpin 25:nSpin  26:sumIVB 27:nIVB
    #  28:sumHB 29:nHB  30:sumRelZ 31:nRelZ  32:sumRelX 33:nRelX
    #  34:sumExt 35:nExt  36:sumArmAngle 37:nArmAngle
    #  38:sumVAA 39:nVAA  40:sumHAA 41:nHAA
    #  42:sumPlateZ 43:nPlateZ
    #  44:sumTiltSin 45:sumTiltCos 46:nTilt
    #  47:sumPlateX 48:nPlateX
    #  49:sumEffVelo 50:nEffVelo
    #  51:sumStuff 52:nStuff  53:sumLoc 54:nLoc  55:sumPitching 56:nPitching
    #  (integer grade-atom sums/counts — see _grade_atoms)
    # ==========================================================
    METRIC_OFFSETS = [
        ('Velocity', 22), ('Spin Rate', 24), ('xIndVrtBrk', 26),
        ('xHorzBrk', 28), ('RelPosZ', 30), ('RelPosX', 32),
        ('Extension', 34), ('ArmAngle', 36), ('VAA', 38), ('HAA', 40),
        ('PlateZ', 42), ('PlateX', 47),
    ]

    pitch_micro = defaultdict(lambda: [0.0] * 57)

    for p in all_pitches:
        pitcher = p.get('Pitcher')
        team = p.get('PTeam')
        throws = p.get('Throws')
        pitch_type = p.get('Pitch Type')
        date = normalize_date(p.get('Game Date'))
        batter_hand = p.get('Bats')

        if not pitcher or not team or team not in ALL_TEAMS or not pitch_type:
            continue
        if (pitcher, team) in ep_pitchers:
            continue  # position players never get pitcher micro rows
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        if not date or not batter_hand:
            continue

        key = (pi_idx[pitcher], tm_idx[team], throws or '',
               pt_idx[pitch_type], dt_idx[date], batter_hand)
        c = pitch_micro[key]

        # Same 22 count fields as pitcher (0-21), plus fly ball/HR counts don't apply at pitch level
        c[0] += 1
        in_zone = p.get('InZone') == 'Yes'
        if in_zone:
            c[1] += 1
        desc = p.get('Description', '')
        if desc in SWING_DESCRIPTIONS:
            c[2] += 1
            if in_zone:
                c[18] += 1  # izSw
        if desc == 'Swinging Strike':
            c[3] += 1
            if in_zone:
                c[19] += 1  # izWh
        if desc in ('Called Strike', 'Swinging Strike'):
            c[4] += 1
        if p.get('InZone') == 'No':
            c[5] += 1
            if desc in ('Swinging Strike', 'In Play', 'Foul'):
                c[6] += 1
        bb_type = p.get('BBType')
        if bb_type and bb_type not in BUNT_BB_TYPES:
            c[7] += 1
            if bb_type == 'ground_ball':
                c[8] += 1
        event = p.get('Event')
        if event and event not in NON_PA_EVENTS:
            c[9] += 1
            if event in HIT_EVENTS:      c[10] += 1
            if event == 'Home Run':      c[11] += 1
            if event in K_EVENTS:        c[12] += 1
            if event in BB_EVENTS:       c[13] += 1
            if event in HBP_EVENTS:      c[14] += 1
            if event in SF_EVENTS:       c[15] += 1
            if event in SH_EVENTS:       c[16] += 1
            if event in CI_EVENTS:       c[17] += 1

        # FPS counts (first pitch of PA: count == "0-0")
        if p.get('Count') == '0-0':
            c[20] += 1  # firstPitches
            if desc in ('Called Strike', 'Swinging Strike', 'Foul', 'In Play'):
                c[21] += 1  # firstPitchStrikes

        # Metric sums
        for col_name, offset in METRIC_OFFSETS:
            val = safe_float(p.get(col_name))
            if val is not None:
                c[offset] += val
                c[offset + 1] += 1

        # Break Tilt (circular sin/cos components)
        tilt_min = break_tilt_to_minutes(p.get('OTilt') or p.get('Break Tilt'))
        if tilt_min is not None:
            angle = tilt_min / 720.0 * 2 * math.pi
            c[44] += math.sin(angle)
            c[45] += math.cos(angle)
            c[46] += 1

        # Grade-atom sums (filtered per-type Stuff+/Loc+/Pitching+)
        _sa, _la, _pa = _grade_atoms(p)
        if _sa is not None:
            c[51] += _sa; c[52] += 1
        if _la is not None:
            c[53] += _la; c[54] += 1
        if _pa is not None:
            c[55] += _pa; c[56] += 1

    pitch_rows = []
    for (pi, ti, throws, pti, di, bh), c in pitch_micro.items():
        row = [pi, ti, throws, pti, di, bh]
        # 22 integer/float counts (0-21)
        for i in range(22):
            row.append(int(c[i]))
        # 13 metric sum/count pairs. Round to 4 dec so source precisions up to
        # 3 dec (PlateX/Z) are preserved through the sum — the frontend
        # divides sum/count to get the average and rounds at display time.
        for col_name, offset in METRIC_OFFSETS:
            row.append(round(c[offset], 4))       # metric sum
            row.append(int(c[offset + 1]))         # metric count
        # Tilt sin/cos
        row.append(round(c[44], 6))  # sumTiltSin
        row.append(round(c[45], 6))  # sumTiltCos
        row.append(int(c[46]))       # nTilt
        # Grade-atom sums/counts (all integers)
        for gi in range(51, 57):
            row.append(int(c[gi]))
        pitch_rows.append(row)

    # ==========================================================
    #  Pitcher BIP records (for avgEV, maxEV, hardHit%, barrel%, LD%, FB%, PU%)
    #  [pitcherIdx, dateIdx, batterHand, exitVelo, launchAngle, bbType]
    #  bbType encoded: 0=ground_ball, 1=line_drive, 2=fly_ball, 3=popup
    # ==========================================================
    BB_TYPE_CODE = {'ground_ball': 0, 'line_drive': 1, 'fly_ball': 2, 'popup': 3}
    pitcher_bip_rows = []
    for p in all_pitches:
        pitcher = p.get('Pitcher')
        team = p.get('PTeam')
        date = normalize_date(p.get('Game Date'))
        batter_hand = p.get('Bats')
        bb_type = p.get('BBType')

        if not pitcher or not team or team not in ALL_TEAMS:
            continue
        if (pitcher, team) in ep_pitchers:
            continue  # position players never get pitcher micro rows
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        if not date or not batter_hand:
            continue
        if not bb_type or bb_type in BUNT_BB_TYPES:
            continue

        ev = safe_float(p.get('ExitVelo'))
        la = safe_float(p.get('LaunchAngle'))
        # Keep BIP with null EV/LA (untracked balls that still carry a bb_type) so
        # the client LD/FB/PU%/nBip denominators count every non-bunt BIP like the
        # server. EV-based stats (avgEV/maxEV/hardHit/barrel) filter nulls separately.

        bb_code = BB_TYPE_CODE.get(bb_type, -1)
        if bb_code < 0:
            continue

        hc_x = safe_float(p.get('HC_X'))
        hc_y = safe_float(p.get('HC_Y'))
        pitcher_bip_rows.append([
            pi_idx[pitcher],
            tm_idx[team],
            dt_idx[date],
            batter_hand,
            round(ev, 1) if ev is not None else None,
            round(la, 1) if la is not None else None,
            bb_code,
            round(hc_x, 2) if hc_x is not None else None,
            round(hc_y, 2) if hc_y is not None else None,
            batter_hand,
            1 if str(p.get('Barrel', '')).strip() == '6' else 0,  # official barrel (launch_speed_angle==6)
        ])
    print(f"  Pitcher BIP records: {len(pitcher_bip_rows)}")

    # ==========================================================
    #  Hitter micro-aggs
    #  Key: (hitterIdx, teamIdx, bats, dateIdx, pitcherHand)
    #  bats = actual batting side for these pitches (R/L)
    #  Values: 50 count fields
    #  0:n  1:pa  2:h  3:db  4:tp  5:hr  6:bb  7:hbp  8:sf  9:sh  10:ci  11:k
    #  12:swings  13:whiffs  14:izPitches  15:oozPitches
    #  16:izSwings  17:oozSwings  18:contact
    #  19:izSwNonBunt  20:izContact
    #  21:bip  22:gb  23:ld  24:fb  25:pu
    #  26:barrels  27:nSpray  28:pull  29:center  30:oppo  31:airPull
    #  32:hardHit  33:nHrBip  34:ldHr
    #  35:twoStrikeSwings  36:twoStrikeWhiffs
    #  37:firstPitchAppearances  38:firstPitchSwings
    #  39:xBA_sum  40:xBA_count  41:xSLG_sum  42:xSLG_count
    #  43:xwOBA_sum  44:xwOBA_count  45:xwOBAcon_sum  46:xwOBAcon_count
    #  47:swingsNonBunt  48:contactNonBunt  49:buntAB
    # ==========================================================
    hitter_micro = defaultdict(lambda: [0.0] * 50)

    for p in all_pitches:
        batter = p.get('Batter')
        team = p.get('BTeam')
        bats = p.get('Bats')
        date = normalize_date(p.get('Game Date'))
        pitcher_hand = p.get('Throws')

        if not batter or not team or team not in ALL_TEAMS:
            continue
        if p.get('_roc_pitcher_pitch'):
            continue  # Skip AAA hitters facing ROC pitchers
        if not date or not pitcher_hand or not bats:
            continue

        key = (hi_idx[batter], tm_idx[team], bats, dt_idx[date], pitcher_hand)
        c = hitter_micro[key]

        c[0] += 1  # n (total pitches)
        desc = p.get('Description', '')
        bb_type = p.get('BBType')
        in_zone = p.get('InZone')

        # PA and event counts
        event = p.get('Event')
        if event and event not in NON_PA_EVENTS:
            c[1] += 1   # pa
            if event in HIT_EVENTS:      c[2] += 1   # h
            if event == 'Double':        c[3] += 1   # db
            if event == 'Triple':        c[4] += 1   # tp
            if event == 'Home Run':      c[5] += 1   # hr
            if event in BB_EVENTS:       c[6] += 1   # bb (all walks including IBB)
            if event in HBP_EVENTS:      c[7] += 1   # hbp
            if event in SF_EVENTS:       c[8] += 1   # sf
            if event in SH_EVENTS:       c[9] += 1   # sh
            if event in CI_EVENTS:       c[10] += 1  # ci
            if event in K_EVENTS:        c[11] += 1  # k
            # bunt at-bats (a bunt put in play that counts as an AB — not a sac
            # bunt): excluded from the xBA/xSLG denominator client-side to match
            # Savant and the server (nonbunt_ab).
            if bb_type in BUNT_BB_TYPES and event not in SH_EVENTS:
                c[49] += 1  # buntAB

        # Swing counts
        if desc in SWING_DESCRIPTIONS:
            c[12] += 1  # swings
        if desc == 'Swinging Strike':
            c[13] += 1  # whiffs

        # Zone-based counts
        if in_zone == 'Yes':
            c[14] += 1  # izPitches
            if desc in SWING_DESCRIPTIONS:
                c[16] += 1  # izSwings
                # izSwNonBunt: exclude bunt BIPs from IZ swing count
                if bb_type not in BUNT_BB_TYPES:  # None not in set → True
                    c[19] += 1
            if desc in ('Foul', 'In Play'):
                if bb_type not in BUNT_BB_TYPES:
                    c[20] += 1  # izContact
        elif in_zone == 'No':
            c[15] += 1  # oozPitches
            if desc in SWING_DESCRIPTIONS:
                c[17] += 1  # oozSwings

        # Contact (overall)
        if desc in ('Foul', 'In Play'):
            c[18] += 1

        # Contact excluding bunts (for contactPct)
        if desc in SWING_DESCRIPTIONS and bb_type not in BUNT_BB_TYPES:
            c[47] += 1  # swingsNonBunt
        if desc in ('Foul', 'In Play') and bb_type not in BUNT_BB_TYPES:
            c[48] += 1  # contactNonBunt

        # Batted ball data (non-bunt BIPs)
        if bb_type and bb_type not in BUNT_BB_TYPES:
            c[21] += 1  # bip
            if bb_type == 'ground_ball':  c[22] += 1  # gb
            if bb_type == 'line_drive':   c[23] += 1  # ld
            if bb_type == 'fly_ball':     c[24] += 1  # fb
            if bb_type == 'popup':        c[25] += 1  # pu

            # Barrel, hard-hit, HR on BIP
            ev = safe_float(p.get('ExitVelo'))
            la = safe_float(p.get('LaunchAngle'))
            barrel_val = str(p.get('Barrel', '')).strip()
            if barrel_val == '6' or (barrel_val == '' and is_barrel(ev, la)):
                c[26] += 1
            if ev is not None and ev >= 95:
                c[32] += 1  # hardHit
            if event == 'Home Run':
                c[33] += 1  # nHrBip
                if bb_type == 'line_drive':
                    c[34] += 1  # ldHr (line-drive HRs)

            # Spray direction
            hc_x = safe_float(p.get('HC_X'))
            hc_y = safe_float(p.get('HC_Y'))
            sa = spray_angle(hc_x, hc_y)
            sd = spray_direction(sa, bats)
            if sd:
                c[27] += 1  # nSpray
                if sd in ('pull', 'pull_side'):    c[28] += 1
                if sd in ('center_pull', 'center_oppo'):  c[29] += 1
                if sd in ('oppo_side', 'oppo'):    c[30] += 1
                if sd in ('pull', 'pull_side') and bb_type in ('line_drive', 'fly_ball'):
                    c[31] += 1  # airPull

            # Expected stats from Statcast per-pitch values (BIP only: xBA, xSLG, xwOBAcon)
            xba_val = safe_float(p.get('xBA'))
            xslg_val = safe_float(p.get('xSLG'))
            xwobacon_val = safe_float(p.get('xwOBA'))
            if xba_val is not None:
                c[39] += xba_val; c[40] += 1
            if xslg_val is not None:
                c[41] += xslg_val; c[42] += 1
            if xwobacon_val is not None:
                c[45] += xwobacon_val; c[46] += 1

        # xwOBA: assigned to all PA events (K, BB, HBP, BIP) EXCEPT SH (sac bunt)
        # and CI, matching the standard AB+BB+HBP+SF denominator and pipeline_compute.
        if (event and event not in NON_PA_EVENTS and event != 'Intent Walk'
                and event not in SH_EVENTS and event not in CI_EVENTS):
            xwoba_val = safe_float(p.get('xwOBA'))
            if xwoba_val is not None:
                c[43] += xwoba_val; c[44] += 1

        # Count-leverage stats (outside BIP block — applies to all pitches)
        count_str = p.get('Count', '')
        if count_str:
            strikes = count_str.split('-')[1] if '-' in count_str else ''
            if strikes == '2':
                if desc in SWING_DESCRIPTIONS and bb_type not in BUNT_BB_TYPES:
                    c[35] += 1  # twoStrikeSwings (bunt-excluded, matches whiff convention)
                if desc == 'Swinging Strike':
                    c[36] += 1  # twoStrikeWhiffs
            if count_str == '0-0':
                c[37] += 1  # firstPitchAppearances
                if desc in SWING_DESCRIPTIONS:
                    c[38] += 1  # firstPitchSwings

    hitter_rows = []
    for (hi, ti, bats, di, ph), c in hitter_micro.items():
        row = [hi, ti, bats, di, ph]
        for i in range(50):  # 0-49 (incl. buntAB at 49); matches hitterCols
            val = c[i]
            row.append(round(val, 4) if isinstance(val, float) and val != int(val) else int(val))
        hitter_rows.append(row)

    # ==========================================================
    #  Hitter BIP records (for EV, LA, spray chart, batted ball stats)
    #  [hitterIdx, dateIdx, pitcherHand, exitVelo, launchAngle, hcX, hcY, bbType, event]
    #  bbType: 0=ground_ball, 1=line_drive, 2=fly_ball, 3=popup
    #  event: 0=out, 1=single, 2=double, 3=triple, 4=hr, 5=error/fc
    # ==========================================================
    BB_TYPE_ENCODE = {'ground_ball': 0, 'line_drive': 1, 'fly_ball': 2, 'popup': 3}
    EVENT_ENCODE = {
        'Single': 1, 'Double': 2, 'Triple': 3, 'Home Run': 4,
        'Field Error': 5,
        # Fielder's choice is intentionally coded as an OUT (0), not error/fc (5):
        # on the LA×Spray tables a batter who reached on a fielder's choice is
        # shown as an out. Keys use the canonical apostrophe-free Event strings
        # (the data has no apostrophe); the wOBA-on-contact value is a separate
        # concern handled by _bip_woba_value (which keeps FC at 0.9).
        'Fielders Choice': 0, 'Fielders Choice Out': 0,
    }
    hitter_bip_rows = []
    for p in all_pitches:
        batter = p.get('Batter')
        team = p.get('BTeam')
        date = normalize_date(p.get('Game Date'))
        pitcher_hand = p.get('Throws')
        bb_type = p.get('BBType')

        if not batter or not team or team not in ALL_TEAMS:
            continue
        if p.get('_roc_pitcher_pitch'):
            continue  # Skip AAA hitters facing ROC pitchers
        if not date or not pitcher_hand:
            continue
        if not bb_type or bb_type in BUNT_BB_TYPES:
            continue

        ev = safe_float(p.get('ExitVelo'))
        la = safe_float(p.get('LaunchAngle'))
        hc_x = safe_float(p.get('HC_X'))
        hc_y = safe_float(p.get('HC_Y'))
        if ev is None and la is None and hc_x is None and hc_y is None:
            continue
        bb_enc = BB_TYPE_ENCODE.get(bb_type, 0)
        ev_enc = EVENT_ENCODE.get(p.get('Event'), 0)

        dist = safe_float(p.get('Distance'))
        woba_val = _bip_woba_value(p.get('Event'))
        bat_side = p.get('Bats')
        if not bat_side:
            bat_side = 'R'  # default to RHB if Bats field missing
        # Official barrel (launch_speed_angle==6) with is_barrel(ev, la) fallback
        # when the Barrel column is absent — identical to the hitter barrelPct
        # logic and the card's damage view, so the site's Damage tiers match.
        _barrel_raw = str(p.get('Barrel', '')).strip()
        brl_enc = 1 if (_barrel_raw == '6' or (_barrel_raw == '' and is_barrel(ev, la))) else 0
        hitter_bip_rows.append([
            hi_idx[batter],
            tm_idx[team],
            dt_idx[date],
            pitcher_hand,
            bat_side,
            round(ev, 1) if ev is not None else None,
            round(la, 1) if la is not None else None,
            round(hc_x, 2) if hc_x is not None else None,
            round(hc_y, 2) if hc_y is not None else None,
            bb_enc,
            ev_enc,
            int(round(dist)) if dist is not None else None,
            round(woba_val, 3) if woba_val is not None else None,
            brl_enc,
        ])

    # ==========================================================
    #  Hitter-Pitch micro-aggs (same counts as hitter micro, but keyed with pitch type)
    #  Key: (hitterIdx, teamIdx, bats, pitchTypeIdx, dateIdx, pitcherHand)
    #  Same 47 count fields as hitter micro
    # ==========================================================
    hitter_pitch_micro = defaultdict(lambda: [0.0] * 49)

    for p in all_pitches:
        batter = p.get('Batter')
        team = p.get('BTeam')
        bats = p.get('Bats')
        pitch_type = p.get('Pitch Type')
        date = normalize_date(p.get('Game Date'))
        pitcher_hand = p.get('Throws')

        if not batter or not team or team not in ALL_TEAMS:
            continue
        if p.get('_roc_pitcher_pitch'):
            continue  # Skip AAA hitters facing ROC pitchers
        if not date or not pitcher_hand or not bats or not pitch_type:
            continue

        key = (hi_idx[batter], tm_idx[team], bats, pt_idx[pitch_type], dt_idx[date], pitcher_hand)
        c = hitter_pitch_micro[key]

        c[0] += 1  # n
        desc = p.get('Description', '')
        bb_type = p.get('BBType')
        in_zone = p.get('InZone')

        event = p.get('Event')
        if event and event not in NON_PA_EVENTS:
            c[1] += 1   # pa
            if event in HIT_EVENTS:      c[2] += 1
            if event == 'Double':        c[3] += 1
            if event == 'Triple':        c[4] += 1
            if event == 'Home Run':      c[5] += 1
            if event in BB_EVENTS:       c[6] += 1
            if event in HBP_EVENTS:      c[7] += 1
            if event in SF_EVENTS:       c[8] += 1
            if event in SH_EVENTS:       c[9] += 1
            if event in CI_EVENTS:       c[10] += 1
            if event in K_EVENTS:        c[11] += 1

        if desc in SWING_DESCRIPTIONS:
            c[12] += 1  # swings
        if desc == 'Swinging Strike':
            c[13] += 1  # whiffs

        if in_zone == 'Yes':
            c[14] += 1  # izPitches
            if desc in SWING_DESCRIPTIONS:
                c[16] += 1  # izSwings
                if bb_type not in BUNT_BB_TYPES:
                    c[19] += 1  # izSwNonBunt
            if desc in ('Foul', 'In Play'):
                if bb_type not in BUNT_BB_TYPES:
                    c[20] += 1  # izContact
        elif in_zone == 'No':
            c[15] += 1  # oozPitches
            if desc in SWING_DESCRIPTIONS:
                c[17] += 1  # oozSwings

        if desc in ('Foul', 'In Play'):
            c[18] += 1  # contact

        # Contact excluding bunts (for contactPct)
        if desc in SWING_DESCRIPTIONS and bb_type not in BUNT_BB_TYPES:
            c[47] += 1  # swingsNonBunt
        if desc in ('Foul', 'In Play') and bb_type not in BUNT_BB_TYPES:
            c[48] += 1  # contactNonBunt

        if bb_type and bb_type not in BUNT_BB_TYPES:
            c[21] += 1  # bip
            if bb_type == 'ground_ball':  c[22] += 1
            if bb_type == 'line_drive':   c[23] += 1
            if bb_type == 'fly_ball':     c[24] += 1
            if bb_type == 'popup':        c[25] += 1

            ev = safe_float(p.get('ExitVelo'))
            la = safe_float(p.get('LaunchAngle'))
            barrel_val = str(p.get('Barrel', '')).strip()
            if barrel_val == '6' or (barrel_val == '' and is_barrel(ev, la)):
                c[26] += 1
            if ev is not None and ev >= 95:
                c[32] += 1  # hardHit
            if event == 'Home Run':
                c[33] += 1  # nHrBip
                if bb_type == 'line_drive':
                    c[34] += 1  # ldHr

            hc_x = safe_float(p.get('HC_X'))
            hc_y = safe_float(p.get('HC_Y'))
            sa = spray_angle(hc_x, hc_y)
            sd = spray_direction(sa, bats)
            if sd:
                c[27] += 1
                if sd in ('pull', 'pull_side'):    c[28] += 1
                if sd in ('center_pull', 'center_oppo'):  c[29] += 1
                if sd in ('oppo_side', 'oppo'):    c[30] += 1
                if sd in ('pull', 'pull_side') and bb_type in ('line_drive', 'fly_ball', 'popup'):
                    c[31] += 1

            # Expected stats from Statcast per-pitch values (BIP only: xBA, xSLG, xwOBAcon)
            xba_val = safe_float(p.get('xBA'))
            xslg_val = safe_float(p.get('xSLG'))
            xwobacon_val = safe_float(p.get('xwOBA'))
            if xba_val is not None:
                c[39] += xba_val; c[40] += 1
            if xslg_val is not None:
                c[41] += xslg_val; c[42] += 1
            if xwobacon_val is not None:
                c[45] += xwobacon_val; c[46] += 1

        # xwOBA: assigned to all PA events (K, BB, HBP, BIP) EXCEPT SH (sac bunt)
        # and CI, matching the standard AB+BB+HBP+SF denominator and pipeline_compute.
        if (event and event not in NON_PA_EVENTS and event != 'Intent Walk'
                and event not in SH_EVENTS and event not in CI_EVENTS):
            xwoba_val = safe_float(p.get('xwOBA'))
            if xwoba_val is not None:
                c[43] += xwoba_val; c[44] += 1

        # Count-leverage stats
        count_str = p.get('Count', '')
        if count_str:
            strikes = count_str.split('-')[1] if '-' in count_str else ''
            if strikes == '2':
                if desc in SWING_DESCRIPTIONS and bb_type not in BUNT_BB_TYPES:
                    c[35] += 1  # twoStrikeSwings (bunt-excluded, matches whiff convention)
                if desc == 'Swinging Strike':
                    c[36] += 1  # twoStrikeWhiffs
            if count_str == '0-0':
                c[37] += 1  # firstPitchAppearances
                if desc in SWING_DESCRIPTIONS:
                    c[38] += 1  # firstPitchSwings

    hitter_pitch_rows = []
    for (hi, ti, bats, pti, di, ph), c in hitter_pitch_micro.items():
        row = [hi, ti, bats, pti, di, ph]
        for i in range(49):
            val = c[i]
            row.append(round(val, 4) if isinstance(val, float) and val != int(val) else int(val))
        hitter_pitch_rows.append(row)

    # Hitter-Pitch BIP records (with pitch type)
    # [hitterIdx, pitchTypeIdx, dateIdx, pitcherHand, exitVelo, launchAngle]
    hitter_pitch_bip_rows = []
    for p in all_pitches:
        batter = p.get('Batter')
        team = p.get('BTeam')
        pitch_type = p.get('Pitch Type')
        date = normalize_date(p.get('Game Date'))
        pitcher_hand = p.get('Throws')
        bb_type = p.get('BBType')

        if not batter or not team or team not in ALL_TEAMS:
            continue
        if p.get('_roc_pitcher_pitch'):
            continue  # Skip AAA hitters facing ROC pitchers
        if not date or not pitcher_hand or not pitch_type:
            continue
        if not bb_type or bb_type in BUNT_BB_TYPES:
            continue

        ev = safe_float(p.get('ExitVelo'))
        la = safe_float(p.get('LaunchAngle'))
        if ev is None and la is None:
            continue

        hitter_pitch_bip_rows.append([
            hi_idx[batter],
            tm_idx[team],
            pt_idx[pitch_type],
            dt_idx[date],
            pitcher_hand,
            round(ev, 1) if ev is not None else None,
            round(la, 1) if la is not None else None,
        ])

    # ==========================================================
    #  Multi-team (2TM/3TM) synthesis
    #  Players on ≥2 MLB teams (ROC excluded) get synthetic combined
    #  micro records and duplicated BIP records so the "All Teams" view
    #  aggregates naturally. Per-team records are left intact so specific-
    #  team views still work.
    # ==========================================================
    aaa_team_indices = {tm_idx[t] for t in AAA_TEAMS if t in tm_idx}

    # Find multi-team pitchers (keyed by (pi_idx, throws)) and hitters (by hi_idx)
    pitcher_mlb_team_set_micro = defaultdict(set)
    for (pi, ti, throws, _di, _bh) in pitcher_micro.keys():
        if ti not in aaa_team_indices:
            pitcher_mlb_team_set_micro[(pi, throws)].add(ti)

    hitter_mlb_team_set_micro = defaultdict(set)
    for (hi, ti, _bats, _di, _ph) in hitter_micro.keys():
        if ti not in aaa_team_indices:
            hitter_mlb_team_set_micro[hi].add(ti)

    # Helper: check whether the MLB IDs match across a player's teams. If
    # different IDs appear, it's a name collision (two different players
    # with the same name) and we must NOT synthesize a combined 2TM row.
    def _ids_match_across_teams(player_name, team_indices):
        if mlb_id_cache is None:
            return True  # no ID info → assume no collision (legacy behavior)
        ids = set()
        for ti in team_indices:
            team_name = teams[ti]
            mid = mlb_id_cache.get(f"{player_name}|{team_name}")
            if mid is not None:
                ids.add(mid)
        return len(ids) <= 1

    # Extend teams + tm_idx with combined labels we'll actually need
    combined_pitcher_ti = {}  # (pi, throws) → combined tm_idx
    combined_hitter_ti = {}   # hi → combined tm_idx
    for (pi, throws), tset in pitcher_mlb_team_set_micro.items():
        if len(tset) < 2:
            continue
        if not _ids_match_across_teams(pitchers[pi], tset):
            continue
        label = f"{len(tset)}TM"
        if label not in tm_idx:
            tm_idx[label] = len(teams)
            teams.append(label)
        combined_pitcher_ti[(pi, throws)] = tm_idx[label]
    for hi, tset in hitter_mlb_team_set_micro.items():
        if len(tset) < 2:
            continue
        if not _ids_match_across_teams(hitters[hi], tset):
            continue
        label = f"{len(tset)}TM"
        if label not in tm_idx:
            tm_idx[label] = len(teams)
            teams.append(label)
        combined_hitter_ti[hi] = tm_idx[label]

    def _sum_counts(accum, src, n):
        for i in range(n):
            accum[i] += src[i]

    # --- Pitcher micro: sum counts across teams for same (di, bh) ---
    if combined_pitcher_ti:
        # Pre-index by (pi, throws) for O(1) grouping
        pmicro_by_pitcher = defaultdict(list)
        for key, c in pitcher_micro.items():
            (pi, ti, throws, di, bh) = key
            pmicro_by_pitcher[(pi, throws)].append((ti, di, bh, c))
        for (pi, throws), combined_ti in combined_pitcher_ti.items():
            teamset = pitcher_mlb_team_set_micro[(pi, throws)]
            by_dibh = defaultdict(lambda: [0] * 37)
            for (ti, di, bh, c) in pmicro_by_pitcher[(pi, throws)]:
                if ti not in teamset:
                    continue
                _sum_counts(by_dibh[(di, bh)], c, 37)
            for (di, bh), c in by_dibh.items():
                pitcher_rows.append([pi, combined_ti, throws, di, bh] + c)

    # --- Pitch micro: sum across teams for same (pt, di, bh) ---
    if combined_pitcher_ti:
        pitchmicro_by_pitcher = defaultdict(list)
        for key, c in pitch_micro.items():
            (pi, ti, throws, pti, di, bh) = key
            pitchmicro_by_pitcher[(pi, throws)].append((ti, pti, di, bh, c))
        for (pi, throws), combined_ti in combined_pitcher_ti.items():
            teamset = pitcher_mlb_team_set_micro[(pi, throws)]
            by_key = defaultdict(lambda: [0.0] * 57)
            for (ti, pti, di, bh, c) in pitchmicro_by_pitcher[(pi, throws)]:
                if ti not in teamset:
                    continue
                _sum_counts(by_key[(pti, di, bh)], c, 57)
            for (pti, di, bh), c in by_key.items():
                # Emit in the SAME reordered layout as the per-team pitch builder
                # (22 counts, then METRIC_OFFSETS sum/count pairs, then tilt). The
                # accumulator is in storage order (PlateX at 47/48, tilt at 44/45/46),
                # so a raw range(51) dump would misalign sumPlateX/tilt against
                # pitchCols and corrupt Break Tilt / nHAA for every multi-team pitcher.
                row = [pi, combined_ti, throws, pti, di, bh]
                for i in range(22):
                    row.append(int(c[i]))
                for col_name, offset in METRIC_OFFSETS:
                    row.append(round(c[offset], 4))       # metric sum
                    row.append(int(c[offset + 1]))         # metric count
                row.append(round(c[44], 6))  # sumTiltSin
                row.append(round(c[45], 6))  # sumTiltCos
                row.append(int(c[46]))       # nTilt
                # Grade-atom sums/counts (all integers)
                for gi in range(51, 57):
                    row.append(int(c[gi]))
                pitch_rows.append(row)

    # --- Hitter micro: sum across teams for same (bats, di, ph) ---
    if combined_hitter_ti:
        hmicro_by_hitter = defaultdict(list)
        for key, c in hitter_micro.items():
            (hi, ti, bats, di, ph) = key
            hmicro_by_hitter[hi].append((ti, bats, di, ph, c))
        for hi, combined_ti in combined_hitter_ti.items():
            teamset = hitter_mlb_team_set_micro[hi]
            by_key = defaultdict(lambda: [0.0] * 50)
            for (ti, bats, di, ph, c) in hmicro_by_hitter[hi]:
                if ti not in teamset:
                    continue
                _sum_counts(by_key[(bats, di, ph)], c, 50)
            for (bats, di, ph), c in by_key.items():
                row = [hi, combined_ti, bats, di, ph]
                for i in range(50):  # incl. buntAB at 49
                    v = c[i]
                    row.append(round(v, 4) if isinstance(v, float) and v != int(v) else int(v))
                hitter_rows.append(row)

    # --- Hitter-pitch micro: sum across teams for same (bats, pt, di, ph) ---
    if combined_hitter_ti:
        hpmicro_by_hitter = defaultdict(list)
        for key, c in hitter_pitch_micro.items():
            (hi, ti, bats, pti, di, ph) = key
            hpmicro_by_hitter[hi].append((ti, bats, pti, di, ph, c))
        for hi, combined_ti in combined_hitter_ti.items():
            teamset = hitter_mlb_team_set_micro[hi]
            by_key = defaultdict(lambda: [0.0] * 49)
            for (ti, bats, pti, di, ph, c) in hpmicro_by_hitter[hi]:
                if ti not in teamset:
                    continue
                _sum_counts(by_key[(bats, pti, di, ph)], c, 49)
            for (bats, pti, di, ph), c in by_key.items():
                row = [hi, combined_ti, bats, pti, di, ph]
                for i in range(49):
                    v = c[i]
                    row.append(round(v, 4) if isinstance(v, float) and v != int(v) else int(v))
                hitter_pitch_rows.append(row)

    # --- BIP records: duplicate with combined teamIdx for multi-team players ---
    if combined_pitcher_ti:
        extra_pitcher_bip = []
        for rec in pitcher_bip_rows:
            pi_v, ti_v = rec[0], rec[1]
            # Need to find matching (pi, throws) — BIP row doesn't carry throws.
            # Enumerate all throws options for this pitcher.
            for (pi2, throws), ct_ti in combined_pitcher_ti.items():
                if pi2 != pi_v:
                    continue
                if ti_v in pitcher_mlb_team_set_micro[(pi2, throws)]:
                    new_rec = rec[:]
                    new_rec[1] = ct_ti
                    extra_pitcher_bip.append(new_rec)
                    break
        pitcher_bip_rows.extend(extra_pitcher_bip)

    if combined_hitter_ti:
        extra_hitter_bip = []
        for rec in hitter_bip_rows:
            hi_v, ti_v = rec[0], rec[1]
            combined_ti = combined_hitter_ti.get(hi_v)
            if combined_ti is not None and ti_v in hitter_mlb_team_set_micro[hi_v]:
                new_rec = rec[:]
                new_rec[1] = combined_ti
                extra_hitter_bip.append(new_rec)
        hitter_bip_rows.extend(extra_hitter_bip)

        extra_hp_bip = []
        for rec in hitter_pitch_bip_rows:
            hi_v, ti_v = rec[0], rec[1]
            combined_ti = combined_hitter_ti.get(hi_v)
            if combined_ti is not None and ti_v in hitter_mlb_team_set_micro[hi_v]:
                new_rec = rec[:]
                new_rec[1] = combined_ti
                extra_hp_bip.append(new_rec)
        hitter_pitch_bip_rows.extend(extra_hp_bip)

    # ==========================================================
    #  Velocity trend sparklines (sparse time-series)
    #  Key: (pitcherIdx, pitchTypeIdx, dateIdx)
    #  Values: [sumVelo, nVelo]
    # ==========================================================
    velo_trend = defaultdict(lambda: [0.0, 0])
    for p in all_pitches:
        pitcher = p.get('Pitcher')
        team = p.get('PTeam')
        pitch_type = p.get('Pitch Type')
        date = normalize_date(p.get('Game Date'))
        velo = safe_float(p.get('Velocity'))

        if not pitcher or not team or team not in ALL_TEAMS or not pitch_type:
            continue
        if (pitcher, team) in ep_pitchers:
            continue  # position players never get pitcher micro rows
        if p.get('_roc_hitter_pitch'):
            continue
        if not date or velo is None:
            continue

        key = (pi_idx[pitcher], tm_idx[team], pt_idx[pitch_type], dt_idx[date])
        velo_trend[key][0] += velo
        velo_trend[key][1] += 1

    # Synthesize 2TM velocity-trend entries (sum across teams per pitch type/date)
    if combined_pitcher_ti:
        vt_by_pitcher = defaultdict(list)
        for key, vals in velo_trend.items():
            (pi, ti, pti, di) = key
            vt_by_pitcher[pi].append((ti, pti, di, vals))
        for (pi, throws), combined_ti in combined_pitcher_ti.items():
            teamset = pitcher_mlb_team_set_micro[(pi, throws)]
            by_key = defaultdict(lambda: [0.0, 0])
            for (ti, pti, di, vals) in vt_by_pitcher.get(pi, []):
                if ti not in teamset:
                    continue
                dst = by_key[(pti, di)]
                dst[0] += vals[0]
                dst[1] += vals[1]
            for (pti, di), vals in by_key.items():
                velo_trend[(pi, combined_ti, pti, di)] = vals

    velo_trend_rows = []
    for (pi, ti, pti, di), (s, n) in velo_trend.items():
        velo_trend_rows.append([pi, ti, pti, di, round(s, 1), n])
    print(f"  Velocity trend rows: {len(velo_trend_rows)}")

    # ==========================================================
    #  Build output
    # ==========================================================
    return {
        'lookups': {
            'pitchers': pitchers,
            'hitters': hitters,
            'teams': teams,
            'dates': dates,
            'pitchTypes': pitch_types,
        },
        'pitcherCols': [
            'pitcherIdx', 'teamIdx', 'throws', 'dateIdx', 'batterHand',
            'n', 'iz', 'sw', 'wh', 'csw', 'ooz', 'oozSw', 'bip', 'gb',
            'pa', 'h', 'hr', 'k', 'bb', 'hbp', 'sf', 'sh', 'ci',
            'izSw', 'izWh', 'firstPitches', 'firstPitchStrikes', 'fb', 'nHrBip', 'ldHr', 'pu', 'nStrikes', 'ibb',
            'oneOneTotal', 'oneOneWins', 'earlyActionPAs',
            'sumStuff', 'nStuff', 'sumLoc', 'nLoc', 'sumPitching', 'nPitching',
        ],
        'pitcherMicro': pitcher_rows,
        'pitcherBipCols': ['pitcherIdx', 'teamIdx', 'dateIdx', 'batterHand', 'exitVelo', 'launchAngle', 'bbType', 'hcX', 'hcY', 'bats', 'barrel'],
        'pitcherBip': pitcher_bip_rows,
        'pitchCols': [
            'pitcherIdx', 'teamIdx', 'throws', 'pitchTypeIdx', 'dateIdx', 'batterHand',
            'n', 'iz', 'sw', 'wh', 'csw', 'ooz', 'oozSw', 'bip', 'gb',
            'pa', 'h', 'hr', 'k', 'bb', 'hbp', 'sf', 'sh', 'ci',
            'izSw', 'izWh', 'firstPitches', 'firstPitchStrikes',
            'sumVelo', 'nVelo', 'sumSpin', 'nSpin', 'sumIVB', 'nIVB',
            'sumHB', 'nHB', 'sumRelZ', 'nRelZ', 'sumRelX', 'nRelX',
            'sumExt', 'nExt', 'sumArmAngle', 'nArmAngle',
            'sumVAA', 'nVAA', 'sumHAA', 'nHAA',
            'sumPlateZ', 'nPlateZ',
            'sumPlateX', 'nPlateX',
            'sumTiltSin', 'sumTiltCos', 'nTilt',
            'sumStuff', 'nStuff', 'sumLoc', 'nLoc', 'sumPitching', 'nPitching',
        ],
        'pitchMicro': pitch_rows,
        'hitterCols': [
            'hitterIdx', 'teamIdx', 'bats', 'dateIdx', 'pitcherHand',
            'n', 'pa', 'h', 'db', 'tp', 'hr', 'bb', 'hbp', 'sf', 'sh', 'ci', 'k',
            'swings', 'whiffs', 'izPitches', 'oozPitches', 'izSwings', 'oozSwings',
            'contact', 'izSwNonBunt', 'izContact',
            'bip', 'gb', 'ld', 'fb', 'pu',
            'barrels', 'nSpray', 'pull', 'center', 'oppo', 'airPull',
            'hardHit', 'nHrBip', 'ldHr',
            'twoStrikeSwings', 'twoStrikeWhiffs',
            'firstPitchAppearances', 'firstPitchSwings',
            'xBA_sum', 'xBA_count', 'xSLG_sum', 'xSLG_count',
            'xwOBA_sum', 'xwOBA_count', 'xwOBAcon_sum', 'xwOBAcon_count',
            'swingsNonBunt', 'contactNonBunt', 'buntAB',
        ],
        'hitterMicro': hitter_rows,
        'hitterBipCols': ['hitterIdx', 'teamIdx', 'dateIdx', 'pitcherHand', 'batSide', 'exitVelo', 'launchAngle', 'hcX', 'hcY', 'bbType', 'event', 'distance', 'wOBAval', 'barrel'],
        'hitterBip': hitter_bip_rows,
        'hitterPitchCols': [
            'hitterIdx', 'teamIdx', 'bats', 'pitchTypeIdx', 'dateIdx', 'pitcherHand',
            'n', 'pa', 'h', 'db', 'tp', 'hr', 'bb', 'hbp', 'sf', 'sh', 'ci', 'k',
            'swings', 'whiffs', 'izPitches', 'oozPitches', 'izSwings', 'oozSwings',
            'contact', 'izSwNonBunt', 'izContact',
            'bip', 'gb', 'ld', 'fb', 'pu',
            'barrels', 'nSpray', 'pull', 'center', 'oppo', 'airPull',
            'hardHit', 'nHrBip', 'ldHr',
            'twoStrikeSwings', 'twoStrikeWhiffs',
            'firstPitchAppearances', 'firstPitchSwings',
            'xBA_sum', 'xBA_count', 'xSLG_sum', 'xSLG_count',
            'xwOBA_sum', 'xwOBA_count', 'xwOBAcon_sum', 'xwOBAcon_count',
            'swingsNonBunt', 'contactNonBunt',
        ],
        'hitterPitchMicro': hitter_pitch_rows,
        'hitterPitchBipCols': ['hitterIdx', 'teamIdx', 'pitchTypeIdx', 'dateIdx', 'pitcherHand', 'exitVelo', 'launchAngle'],
        'hitterPitchBip': hitter_pitch_bip_rows,
        'veloTrendCols': ['pitcherIdx', 'teamIdx', 'pitchTypeIdx', 'dateIdx', 'sumVelo', 'nVelo'],
        'veloTrend': velo_trend_rows,
    }


def process_game_type(all_pitches, label, mlb_id_cache, mlb_id_cache_path):
    """Process a set of pitches into all leaderboard outputs.

    Args:
        all_pitches: list of pitch dicts
        label: 'ST' or 'RS' (for logging)
        mlb_id_cache: shared MLB ID cache dict (mutated in place)
        mlb_id_cache_path: path to MLB ID cache file

    Returns a dict with all outputs: pitcher_leaderboard, pitch_leaderboard,
    hitter_leaderboard, hitter_pitch_leaderboard, metadata, micro_data,
    pitch_details, hitter_pitch_details.
    """
    if not all_pitches:
        print(f"  No pitches for {label}, returning empty results")
        return {
            'pitcher_leaderboard': [],
            'pitch_leaderboard': [],
            'hitter_leaderboard': [],
            'hitter_pitch_leaderboard': [],
            'metadata': {
                'teams': [],
                'pitchTypes': [],
                'generatedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'totalPitches': 0,
                'totalPitchers': 0,
                'totalHitters': 0,
                'leagueAverages': {},
                'pitcherLeagueAverages': {},
                'hitterLeagueAverages': {},
                'vaaRegressions': {},
                'haaRegressions': {},
                'sacqZones': [],
                'sacqLaZones': [],
            },
            'micro_data': {
                'lookups': {'pitchers': [], 'hitters': [], 'teams': [], 'dates': [], 'pitchTypes': []},
                'pitcherCols': [], 'pitcherMicro': [],
                'pitcherBipCols': [], 'pitcherBip': [],
                'pitchCols': [], 'pitchMicro': [],
                'hitterCols': [], 'hitterMicro': [],
                'hitterBipCols': [], 'hitterBip': [],
                'hitterPitchCols': [], 'hitterPitchMicro': [],
                'hitterPitchBipCols': [], 'hitterPitchBip': [],
            },
            'pitch_details': {},
            'hitter_pitch_details': {},
            'hitter_swing_locations': {},
        }

    # ─── Position-player pitching (EP / Eephus): count the PAs, hide the
    #     "pitchers" (2026-07-13 policy change) ───
    # Wally tags EVERY pitch of a position player's blowout mop-up as EP, so an
    # EP pitch marks a non-pitcher appearance. Old policy dropped every EP pitch
    # here at the source, which also erased those PAs from HITTER stats and the
    # league pitch data — making pitch-derived aggregates disagree with official
    # totals (~440 PAs/season, and hitters bat ~.406 in them, so the gap was
    # hit-rich). New policy: ALL PAs count, matching official stats — EP pitches
    # stay in the pitch data (hitter metrics, SD+/CT+/xwOBAsp tables, micro
    # hitter records, league aggregates). Position players are excluded from
    # everything PITCHER-facing instead: the ep_pitchers guards below keep them
    # out of pitcher/pitch leaderboards, usage, details, and Loc+/Stuff+ scoring
    # (both consume the guarded groups), and generate_micro_data skips their
    # pitcher micro rows so no client-side filter can resurrect them.
    _ep_ids = {(p.get('Pitcher'), p.get('PTeam'))
               for p in all_pitches if p.get('Pitch Type') == 'EP'}
    if _ep_ids:
        _ep_ct = sum(1 for p in all_pitches
                     if (p.get('Pitcher'), p.get('PTeam')) in _ep_ids)
        print(f"  [{label}] Keeping {_ep_ct} pitch(es) from {len(_ep_ids)} "
              f"position-player appearance(s) in hitter/league data "
              f"(excluded from all pitcher-facing views)")

    # --- Recompute InZone from PlateX/PlateZ/SzTop/SzBot with ball-radius adjustment ---
    for p in all_pitches:
        p['InZone'] = compute_in_zone(p)

    # --- MiLB RunExp currency correction (2026-07-25) ---
    # Statcast computes delta_run_exp against EACH LEAGUE'S OWN run-expectancy
    # matrix, so MiLB RunExp is denominated in MiLB runs — measured 1.42x (ROC)
    # and 1.45x (AAA) the MLB value for the identical event. That silently
    # corrupted every run-value number for ROC/AAA:
    #   - RV summed MiLB-denominated runs against MLB-anchored league averages
    #     (+0.25 runs/100 at ROC, +0.32 at AAA of phantom credit).
    #   - xRV was WORSE than merely scaled. It values BIP via
    #     (xwOBA - lgWOBA)/wOBAScale (MLB-anchored) but falls through to
    #     -RunExp for the other ~83% of pitches, so a single ROC xRV mixed two
    #     currencies INTERNALLY and could not be interpreted at all. The
    #     count-anchoring offsets then carefully aligned the BIP branch to a
    #     currency the rest of the number wasn't in.
    # Corrected in place, once, before any consumer reads RunExp, so pitcher
    # RV/xRV, hitter xRV, the SD+/CT+ weight tables and Loc+'s count values all
    # inherit it consistently rather than each needing its own guard.
    _re_scale = compute_runexp_scale(all_pitches)
    if _re_scale:
        _n_fixed = 0
        for p in all_pitches:
            sc = _re_scale.get(p.get('_source'))
            if not sc:
                continue
            v = safe_float(p.get('RunExp'))
            if v is None:
                continue
            f = runexp_factor(sc, p.get('Description'), p.get('Count'))
            if f:
                p['RunExp'] = v / f
                _n_fixed += 1
        print(f"  [{label}] RunExp -> MLB currency: "
              + ", ".join(f"{s} (global /{d['global']:.3f}, "
                          f"{len(d['cell'])} cell factors)"
                          for s, d in sorted(_re_scale.items()))
              + f" — {_n_fixed} pitches rescaled")

    # --- Map non-MLB BTeams to MLB teams where possible ---
    mlb_hitter_teams = {}
    for p in all_pitches:
        batter = p.get('Batter')
        b_team = p.get('BTeam')
        if batter and b_team and b_team in MLB_TEAMS:
            mlb_hitter_teams[batter] = b_team

    remapped_count = 0
    for p in all_pitches:
        b_team = p.get('BTeam')
        if b_team and b_team not in MLB_TEAMS:
            # Don't remap AAA-source pitches — these hitters are actually playing in AAA
            if p.get('_source') == 'AAA':
                continue
            batter = p.get('Batter')
            if batter and batter in mlb_hitter_teams:
                p['BTeam'] = mlb_hitter_teams[batter]
                remapped_count += 1
    if remapped_count:
        print(f"  Remapped {remapped_count} non-MLB BTeam entries")

    # --- Tag ROC/AAA pitches to prevent cross-contamination ---
    # ROC tab pitches: only the pitcher side matters (batters are AAA opponents)
    # AAA tab pitches: only the hitter side matters (pitchers are AAA opponents)
    roc_pitcher_count = 0
    roc_hitter_count = 0
    roc_opp_batter_count = 0
    for p in all_pitches:
        source = p.get('_source', 'MLB')
        if source == 'ROC':
            p['_roc_pitcher_pitch'] = True  # Pitcher is ROC, batter is AAA opponent
            roc_pitcher_count += 1
        elif source == 'AAA':
            # AAA tab: the pitcher (PTeam='AAA') is always an opponent, so keep
            # it out of pitcher stats via _roc_hitter_pitch. The batter is a
            # tracked Rochester hitter ONLY when BTeam=='ROC'. Every other batter
            # (BTeam 'AAA' or blank) is an opponent that got scraped as byproduct
            # (e.g. the opposing lineup facing a moved-to-AAA arm) and must stay
            # out of hitter stats too — so it also gets _roc_pitcher_pitch, the
            # flag that excludes a pitch's batter from every hitter aggregation.
            # (Previously all 'AAA' BTeams were normalized to 'ROC', dumping ~90
            # opponent batters into the ROC hitter leaderboard.)
            p['_roc_hitter_pitch'] = True        # pitcher is an AAA opponent
            if p.get('BTeam') == 'ROC':
                roc_hitter_count += 1
            else:
                p['_roc_pitcher_pitch'] = True   # batter is an AAA opponent too
                roc_opp_batter_count += 1
    if roc_pitcher_count or roc_hitter_count or roc_opp_batter_count:
        print(f"  Tagged {roc_pitcher_count} ROC pitcher pitches, "
              f"{roc_hitter_count} ROC hitter pitches, "
              f"{roc_opp_batter_count} AAA-opponent-batter pitches excluded")

    # --- Tier 2: fill xwOBA for ROC pitches (Savant doesn't publish their
    # per-pitch xwOBA model output for AAA). The fix unlocks xwOBAcon,
    # xwOBA, BB+, and Hitter+ for ROC hitters via the existing per-hitter
    # aggregations downstream.
    #
    # BIP fill: pipeline_xwoba3d.py — joint EV x LA x spray x bats
    # empirical wOBA table with hierarchical Bayesian shrinkage to 2D
    # marginals. Table built from MLB BIP only (translation framing,
    # consistent with xwOBAsp/SACQ zones/percentile pool/wRC+ overrides
    # — ROC measured against the MLB baseline). Validated against
    # Savant's published per-pitch xwOBA on held-out MLB BIP:
    # per-BIP r=0.915, per-hitter aggregated r=0.962 at the BB+ floor
    # of 80 BIP (MAE 0.015), bias ~0. MLB pitches NEVER overwritten —
    # Savant's value stays gold standard where it exists.
    #
    # Non-BIP PA fill: BB / HBP get the FG Guts wOBA event weights;
    # K events get 0. Needed so per-hitter xwOBA (mean over all PA, not
    # just BIP) populates correctly.
    from pipeline_xwoba3d import (
        build_xwoba3d_table, shrink_xwoba3d, classify_bip as _xw3d_classify,
    )
    _mlb_bip = [p for p in all_pitches
                if p.get('_source','MLB')=='MLB'
                and _xw3d_classify(p) is not None
                and safe_float(p.get('xwOBA')) is not None]
    if _mlb_bip:
        _raw3d = build_xwoba3d_table(_mlb_bip)
        _smooth3d = shrink_xwoba3d(_raw3d, _mlb_bip)
        _bb_w  = WOBA_WEIGHTS.get('BB',  0.69)
        _hbp_w = WOBA_WEIGHTS.get('HBP', 0.72)
        _n_bip = _n_bb = _n_hbp = _n_k = 0
        for p in all_pitches:
            if p.get('_source','MLB') == 'MLB': continue
            if p.get('xwOBA') is not None:        continue   # don't overwrite
            ev = p.get('Event')
            if not ev: continue
            key = _xw3d_classify(p)
            if key is not None and key in _smooth3d:
                p['xwOBA'] = round(_smooth3d[key][0], 4)
                _n_bip += 1
                continue
            if ev in BB_EVENTS and ev != 'Intent Walk':
                p['xwOBA'] = _bb_w;  _n_bb += 1
            elif ev in HBP_EVENTS:
                p['xwOBA'] = _hbp_w; _n_hbp += 1
            elif ev in K_EVENTS:
                p['xwOBA'] = 0.0;    _n_k += 1
        # Keep the smoothed table for metadata serialization later.
        _xw3d_smoothed_table = _smooth3d
        print(f"  ROC xwOBA fill (3D EV×LA×spray×bats lookup): "
              f"{_n_bip} BIP, {_n_bb} BB, {_n_hbp} HBP, {_n_k} K")
    else:
        _xw3d_smoothed_table = None

    # --- Reclassify CF (Cut-Fastball) → FF or FC ---
    # CF is not a real Statcast classification. Remap to FF by default,
    # except specific pitchers whose "CF" is really a cutter (FC).
    CF_TO_FC_PITCHERS = {
        'Ashcraft, Graham', 'Doval, Camilo', 'Fluharty, Mason',
        'Funderburk, Kody', 'Jansen, Kenley', 'Maton, Phil',
    }
    cf_to_ff = 0
    cf_to_fc = 0
    for p in all_pitches:
        if p.get('Pitch Type') == 'CF':
            pitcher = p.get('Pitcher', '')
            if pitcher in CF_TO_FC_PITCHERS:
                p['Pitch Type'] = 'FC'
                cf_to_fc += 1
            else:
                p['Pitch Type'] = 'FF'
                cf_to_ff += 1
    if cf_to_ff or cf_to_fc:
        print(f"  Reclassified CF: {cf_to_ff} → FF, {cf_to_fc} → FC")

    # Collect unique teams (MLB + AAA) and pitch types
    all_teams = sorted(set(
        [p['PTeam'] for p in all_pitches if p.get('PTeam') and p['PTeam'] in ALL_TEAMS] +
        [p['BTeam'] for p in all_pitches if p.get('BTeam') and p['BTeam'] in ALL_TEAMS]
    ))
    all_pitch_types = sorted(set(p['Pitch Type'] for p in all_pitches if p.get('Pitch Type')))

    # --- Lookup MLB IDs for all pitchers and hitters ---
    print(f"\n--- Looking up MLB player IDs ({label}) ---")

    # Helper to get cached MLB ID
    def get_mlb_id(name, team):
        return mlb_id_cache.get(f"{name}|{team}")

    # Build unique pitcher/hitter lists
    unique_pitchers = set()
    unique_hitters = set()
    for p in all_pitches:
        pitcher = p.get('Pitcher')
        pteam = p.get('PTeam')
        if pitcher and pteam:
            unique_pitchers.add((pitcher, pteam))
        batter = p.get('Batter')
        bteam = p.get('BTeam')
        if batter and bteam:
            unique_hitters.add((batter, bteam))

    # Look up all unique players
    all_unique = unique_pitchers | unique_hitters
    new_lookups = 0
    for name, team in sorted(all_unique):
        cache_key = f"{name}|{team}"
        if cache_key not in mlb_id_cache:
            lookup_mlb_id(name, team, mlb_id_cache)
            new_lookups += 1
            if new_lookups % 20 == 0:
                time_module.sleep(0.5)  # Rate limit
                print(f"  Looked up {new_lookups} players...")

    # Save cache incrementally
    save_mlb_id_cache(mlb_id_cache, mlb_id_cache_path)
    print(f"  MLB ID cache: {len(mlb_id_cache)} entries ({new_lookups} new lookups)")

    # --- Canonicalize name-variant splits (same MLB ID, different spelling) ---
    # The feed occasionally emits two "Last, First" spellings for one player
    # (e.g. "Cauley, Cam" vs "Cauley, Cameron", "Thornton, Zac" vs "Zach").
    # Because every downstream aggregation groups on the raw Batter/Pitcher
    # string, the variants split into two half-populated rows instead of one.
    # Collapse them here — before any grouping — to the MLB Stats API's official
    # "Last, First" (lastFirstName), falling back to the most-seen spelling if
    # the API is unreachable. Self-healing: any future variant is caught the
    # same way with no per-player configuration.
    names_by_id = {}   # mlbId -> set of distinct observed "Last, First" names
    for cache_key, mid in mlb_id_cache.items():
        if not mid or '|' not in cache_key:
            continue
        names_by_id.setdefault(mid, set()).add(cache_key.rsplit('|', 1)[0])
    rename_map = {}    # variant name -> canonical name
    for mid, names in names_by_id.items():
        if len(names) < 2:
            continue
        canonical = fetch_canonical_last_first(mid)
        if canonical not in names:
            # API name not among the observed spellings (or lookup failed):
            # keep the variant carrying the most pitches so nothing is dropped.
            counts = {n: 0 for n in names}
            for p in all_pitches:
                b = p.get('Batter')
                if b in counts:
                    counts[b] += 1
                pit = p.get('Pitcher')
                if pit in counts:
                    counts[pit] += 1
            canonical = max(counts, key=counts.get)
        for n in names:
            if n != canonical:
                rename_map[n] = canonical
        print(f"  Name-variant merge (id {mid}): "
              f"{sorted(names)} -> '{canonical}'")
    if rename_map:
        for p in all_pitches:
            if p.get('Batter') in rename_map:
                p['Batter'] = rename_map[p['Batter']]
            if p.get('Pitcher') in rename_map:
                p['Pitcher'] = rename_map[p['Pitcher']]
        # Backfill cache so canonical-name lookups resolve for every stint team.
        for cache_key, mid in list(mlb_id_cache.items()):
            if '|' not in cache_key:
                continue
            nm, tm = cache_key.rsplit('|', 1)
            if nm in rename_map:
                mlb_id_cache[f"{rename_map[nm]}|{tm}"] = mid
        save_mlb_id_cache(mlb_id_cache, mlb_id_cache_path)

    # --- Exclude position players (anyone who threw EP/Eephus) ---
    # PRIMARY guard (2026-07-13): EP pitches are no longer dropped globally —
    # their PAs count in hitter/league data — so this set is what keeps
    # position players out of every pitcher-facing view (leaderboards, usage,
    # details, and Loc+/Stuff+ via the guarded groups).
    ep_pitchers = set()
    for p in all_pitches:
        if p.get('Pitch Type') == 'EP':
            ep_pitchers.add((p['Pitcher'], p['PTeam']))
    if ep_pitchers:
        print(f"  Excluding {len(ep_pitchers)} position player(s): {', '.join(n for n, _ in ep_pitchers)}")

    # --- Count total pitches per pitcher (for usage%) ---
    pitcher_total = defaultdict(int)
    for p in all_pitches:
        if (p['Pitcher'], p['PTeam']) in ep_pitchers:
            continue
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        pitcher_total[(p['Pitcher'], p['PTeam'])] += 1

    # --- Pitch Leaderboard: group by (Pitcher, PTeam, Pitch Type) ---
    pitch_groups = defaultdict(list)
    for p in all_pitches:
        if (p['Pitcher'], p['PTeam']) in ep_pitchers:
            continue
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        key = (p['Pitcher'], p['PTeam'], p['Pitch Type'], p.get('Throws'))
        pitch_groups[key].append(p)

    # ─── Synthesize multi-team (2TM/3TM) combined groups ───
    # A pitcher on ≥2 MLB teams (ROC excluded) gets synthetic combined entries.
    # The same row-building loops below then emit 2TM/3TM rows naturally.
    pitcher_mlb_teams = defaultdict(set)
    for (pitcher, team, _pt, throws) in pitch_groups.keys():
        if team not in AAA_TEAMS:
            pitcher_mlb_teams[(pitcher, throws)].add(team)

    combined_pitcher_labels = {}  # (pitcher, throws) → "2TM"/"3TM"
    pitcher_name_collisions = []
    for (pitcher, throws), teams in pitcher_mlb_teams.items():
        if len(teams) < 2:
            continue
        # Same collision guard as hitters — distinct MLB IDs across teams means
        # different players who happen to share a name.
        ids_by_team = {t: mlb_id_cache.get(f"{pitcher}|{t}") for t in teams}
        unique_ids = {mid for mid in ids_by_team.values() if mid is not None}
        if len(unique_ids) > 1:
            pitcher_name_collisions.append((pitcher, throws, ids_by_team))
            continue
        combined_team = f"{len(teams)}TM"
        combined_pitcher_labels[(pitcher, throws)] = combined_team
        if unique_ids:
            mlb_id_cache[f"{pitcher}|{combined_team}"] = next(iter(unique_ids))
    if pitcher_name_collisions:
        print(f"  Skipped 2TM synthesis for {len(pitcher_name_collisions)} pitcher name collision(s):")
        for pitcher, throws, ids in pitcher_name_collisions:
            print(f"    {pitcher} ({throws}): {ids}")

    if combined_pitcher_labels:
        # Augment pitch_groups (per pitch type)
        pitch_groups_by_ptt = defaultdict(dict)
        for (pitcher, team, pt, throws), pitches in list(pitch_groups.items()):
            pitch_groups_by_ptt[(pitcher, team, throws)][pt] = pitches

        for (pitcher, throws), combined_team in combined_pitcher_labels.items():
            teams = pitcher_mlb_teams[(pitcher, throws)]
            combined_pt_pitches = defaultdict(list)
            for team in teams:
                for pt, pitches in pitch_groups_by_ptt.get((pitcher, team, throws), {}).items():
                    combined_pt_pitches[pt].extend(pitches)
            for pt, combined in combined_pt_pitches.items():
                pitch_groups[(pitcher, combined_team, pt, throws)] = combined
            # Update pitcher_total so usagePct works for combined rows
            pitcher_total[(pitcher, combined_team)] = sum(
                pitcher_total[(pitcher, t)] for t in teams
            )

    # Count-anchoring for displayed xRV (same currency fix as SD+/CT+, adopted
    # for xRV 2026-07-03): offsets align the BIP branch with the delta-RE
    # currency of every other pitch; bip_count_means is the missing-xwOBA
    # fallback (league expected anchored BIP value per count, replacing the
    # actual-outcome fallback that leaked results into an expected stat).
    # Built from MLB pitches only; ROC rows are scored with MLB constants
    # (translation framing).
    from pipeline_sdplus import build_bip_count_offsets
    from pipeline_compute import build_bip_count_means
    _xrv_lg = GUTS_EXTRA.get('lgWOBA') if GUTS_EXTRA else None
    _xrv_scale = GUTS_EXTRA.get('wOBAScale') if GUTS_EXTRA else None
    # Position-player pitching is excluded from the BASELINE (2026-07-25). These
    # offsets/means are the league yardstick a BIP is measured against, and 40
    # catchers lobbing eephuses are not league-average pitching. This is a
    # yardstick, not a PA count, so it does not conflict with the 2026-07-13
    # policy that keeps EP PAs in hitter/league totals — those still count, they
    # are just scored against a pitcher-only baseline. Measured effect is below
    # display resolution (max offset shift 0.0007 runs, BIP count mean 0.0029),
    # so this is a correctness cleanup rather than a visible change.
    _mlb_for_xrv = [p for p in all_pitches
                    if p.get('_source', 'MLB') == 'MLB'
                    and (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers]
    XRV_COUNT_OFFSETS = build_bip_count_offsets(_mlb_for_xrv, _xrv_lg, _xrv_scale)
    XRV_BIP_COUNT_MEANS = build_bip_count_means(_mlb_for_xrv, _xrv_lg, _xrv_scale,
                                                XRV_COUNT_OFFSETS)

    pitch_leaderboard = []
    for (pitcher, team, pitch_type, throws), pitches in pitch_groups.items():
        if not pitch_type:
            continue

        total_for_pitcher = pitcher_total[(pitcher, team)]

        row = {
            'pitcher': pitcher,
            'team': team,
            'throws': throws,
            'pitchType': pitch_type,
            'count': len(pitches),
            'usagePct': round(len(pitches) / total_for_pitcher, 4) if total_for_pitcher > 0 else None,
            'mlbId': get_mlb_id(pitcher, team),
            '_isROC': team in AAA_TEAMS,
        }

        # Average metrics
        for col in METRIC_COLS:
            values = [safe_float(p.get(col)) for p in pitches]
            key_name = METRIC_KEYS[col]
            row[key_name] = round_metric(col, avg(values))

        # Max velocity
        velos = [safe_float(p.get('Velocity')) for p in pitches]
        velos = [v for v in velos if v is not None]
        row['maxVelo'] = round(max(velos), 1) if velos else None

        # Observed (Break) Tilt — circular mean of OTilt clock-notation values.
        tilt_minutes = [break_tilt_to_minutes(p.get('OTilt') or p.get('Break Tilt')) for p in pitches]
        tilt_minutes = [m for m in tilt_minutes if m is not None]
        avg_tilt = circular_mean_minutes(tilt_minutes)
        row['breakTilt'] = minutes_to_tilt_display(avg_tilt)
        row['breakTiltMinutes'] = avg_tilt

        # Release Tilt — circular mean of RTilt clock-notation values from
        # the spin-axis-derived release orientation. Sourced from the RTilt
        # column written by Pitcher2026.py (release_tilt = spin_axis_to_tilt).
        rtilt_minutes = [break_tilt_to_minutes(p.get('RTilt')) for p in pitches]
        rtilt_minutes = [m for m in rtilt_minutes if m is not None]
        avg_rtilt = circular_mean_minutes(rtilt_minutes)
        row['releaseTilt'] = minutes_to_tilt_display(avg_rtilt)
        row['releaseTiltMinutes'] = avg_rtilt

        row.update(compute_stats(pitches))
        row.update(compute_pitcher_batted_ball(pitches))
        row.update(compute_expected_stats(pitches, woba_weights=WOBA_WEIGHTS))
        row.update(compute_xrv(pitches,
                                lg_woba=_xrv_lg, woba_scale=_xrv_scale,
                                count_offsets=XRV_COUNT_OFFSETS,
                                bip_count_means=XRV_BIP_COUNT_MEANS))
        # RV/100 and xRV/100 for this pitch type (raw — rounded at final output step)
        if row.get('runValue') is not None and row.get('count', 0) > 0:
            row['rv100'] = row['runValue'] / row['count'] * 100
        else:
            row['rv100'] = None
        if row.get('xRunValue') is not None and row.get('count', 0) > 0:
            row['xRv100'] = row['xRunValue'] / row['count'] * 100
        else:
            row['xRv100'] = None

        # Per-hand splits at pitch type level (for platoon toggle)
        for hand_label, hand_val in [('_vsL', 'L'), ('_vsR', 'R')]:
            hand_pitches = [p for p in pitches if p.get('Bats') == hand_val]
            if hand_pitches:
                hand_bb = compute_pitcher_batted_ball(hand_pitches)
                hand_ex = compute_expected_stats(hand_pitches, woba_weights=WOBA_WEIGHTS)
                for sk in ['avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst',
                           'ldPct', 'fbPct', 'puPct', 'hrFbPct']:
                    if sk in hand_bb and hand_bb[sk] is not None:
                        row[sk + hand_label] = hand_bb[sk]
                for sk in ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon']:
                    if sk in hand_ex and hand_ex[sk] is not None:
                        row[sk + hand_label] = hand_ex[sk]

        pitch_leaderboard.append(row)

    # --- Regression helper functions ---
    def fit_linear_regression(pairs, label):
        """Fit y = slope*x + intercept, return dict with coefficients or None."""
        if len(pairs) < 30:
            return None
        n = len(pairs)
        sum_x = sum(p[0] for p in pairs)
        sum_y = sum(p[1] for p in pairs)
        sum_xy = sum(p[0] * p[1] for p in pairs)
        sum_x2 = sum(p[0] ** 2 for p in pairs)
        mean_x = sum_x / n
        mean_y = sum_y / n
        denom = sum_x2 - n * mean_x ** 2
        if abs(denom) < 1e-10:
            return None
        slope = (sum_xy - n * mean_x * mean_y) / denom
        intercept = mean_y - slope * mean_x
        ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in pairs)
        ss_tot = sum((p[1] - mean_y) ** 2 for p in pairs)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {label}: slope={slope:.4f}, intercept={intercept:.4f}, R²={r2:.4f} (n={n})")
        return {'slope': slope, 'intercept': intercept, 'r2': r2, 'n': n}

    def fit_within_pitcher_slope(triples, label):
        """Pitcher-demeaned (fixed-effects) OLS slope of y on x.

        triples: [(pitcher_key, x, y)]. Estimates the slope from within-pitcher
        location variance only, so between-pitcher clusters (LHP/RHP mirrors,
        shape-location selection) cannot corrupt it. The location slope is pure
        flight geometry, ~1.0-1.1 deg/ft for every pitch type — a fit far from
        that range is a red flag, not a feature."""
        if len(triples) < 30:
            return None
        sums = {}
        for k, x, y in triples:
            s = sums.setdefault(k, [0.0, 0.0, 0])
            s[0] += x; s[1] += y; s[2] += 1
        sxx = 0.0
        sxy = 0.0
        for k, x, y in triples:
            s = sums[k]
            dx = x - s[0] / s[2]
            dy = y - s[1] / s[2]
            sxx += dx * dx
            sxy += dx * dy
        if sxx < 1e-10:
            return None
        slope = sxy / sxx
        print(f"  {label}: slope={slope:.4f} (within-pitcher, n={len(triples)}, "
              f"pitchers={len(sums)})")
        return {'slope': slope, 'n': len(triples)}

    def mat_inv_general(M):
        """Invert a square matrix via Gauss-Jordan elimination with partial pivoting."""
        n = len(M)
        aug = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_row = col
            for r in range(col + 1, n):
                if abs(aug[r][col]) > abs(aug[max_row][col]):
                    max_row = r
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-12:
                return None
            for r in range(col + 1, n):
                f = aug[r][col] / aug[col][col]
                for c in range(2 * n):
                    aug[r][c] -= f * aug[col][c]
        for col in range(n - 1, -1, -1):
            piv = aug[col][col]
            for c in range(2 * n):
                aug[col][c] /= piv
            for r in range(col):
                f = aug[r][col]
                for c in range(2 * n):
                    aug[r][c] -= f * aug[col][c]
        return [aug[i][n:] for i in range(n)]

    def mvn_conditional(model_params, rel_values):
        """Compute E[IVB, HB | regressors] using MVN conditional distribution.
        model_params: dict with 'mu' (list) and 'cov' (list of lists).
        rel_values: list of regressor values (length = len(mu) - 2).
        Returns [xIVB, xHB] or None."""
        mu = model_params['mu']
        cov = model_params['cov']
        n_acc = 2  # IVB, HB
        n_rel = len(mu) - n_acc
        if len(rel_values) != n_rel:
            return None
        sigma_rel = [[cov[n_acc + i][n_acc + j] for j in range(n_rel)] for i in range(n_rel)]
        sigma_rel_inv = mat_inv_general(sigma_rel)
        if sigma_rel_inv is None:
            return None
        r_diff = [rel_values[k] - mu[n_acc + k] for k in range(n_rel)]
        sri_rdiff = [sum(sigma_rel_inv[i][j] * r_diff[j] for j in range(n_rel)) for i in range(n_rel)]
        mu_bar = []
        for a in range(n_acc):
            adj = sum(cov[a][n_acc + b] * sri_rdiff[b] for b in range(n_rel))
            mu_bar.append(mu[a] + adj)
        return mu_bar

    # Minimum pitches per (pitchType, throws) group. A 5-6 variable
    # covariance conditioned on 3-4 regressors needs real sample: at the old
    # floor of 30, Sigma22 inversion produced wildly unstable conditional
    # slopes for rare groups (early-season ST_L, FS_L, SV). Blank xIVB/xHB
    # for a rare group is honest; a number from 28 dof is not.
    MVN_MIN_N = 150

    def fit_mvn_models(all_pitches):
        """Fit MVN models per (pitchType, throws) for expected movement.
        MLB model: [IVB, HB, ArmAngle, Extension, Velocity]
        ROC model: [IVB, HB, RelPosZ, RelPosX, Extension, Velocity]
        Returns dict keyed by 'pitchType_throws' with 'mlb' and/or 'roc' sub-models.

        Both models are fit on MLB pitches ONLY: the 'roc' variant is the
        MLB baseline expressed in release-point features (arm angle is absent
        at ROC), and ROC pitchers are scored against it — so ROC pitches must
        not contribute to the baseline they're measured against.

        Interpretation note: velocity is a conditioning variable, so the OE
        residuals are "movement vs a typical arm from this slot/extension at
        this velo" — a uniqueness residual, not a velo-inclusive skill delta."""
        groups_mlb = defaultdict(list)
        groups_roc = defaultdict(list)
        for p in all_pitches:
            if p.get('_source', 'MLB') != 'MLB':
                continue
            pt = p.get('Pitch Type') or p.get('TaggedPitchType')
            throws = p.get('Throws')
            ivb = safe_float(p.get('xIndVrtBrk'))
            hb = safe_float(p.get('xHorzBrk'))
            if not pt or not throws or ivb is None or hb is None:
                continue
            key = pt + '_' + throws
            aa = safe_float(p.get('ArmAngle'))
            ext = safe_float(p.get('Extension'))
            velo = safe_float(p.get('Velocity'))
            rel_z = safe_float(p.get('RelPosZ'))
            rel_x = safe_float(p.get('RelPosX'))
            if aa is not None and ext is not None and velo is not None:
                groups_mlb[key].append([ivb, hb, aa, ext, velo])
            if rel_z is not None and rel_x is not None and ext is not None and velo is not None:
                groups_roc[key].append([ivb, hb, rel_z, rel_x, ext, velo])

        def mad_screen(data, thresh=6.0):
            """Drop rows more than thresh MADs from the median on any dim.
            Sample moments are non-robust; a handful of mistracked pitches
            (extension glitches, velocity spikes) can swing the conditional
            slopes of smaller groups."""
            n = len(data)
            k = len(data[0])
            meds, mads = [], []
            for i in range(k):
                vals = sorted(row[i] for row in data)
                med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
                devs = sorted(abs(v - med) for v in vals)
                mad = devs[n // 2] if n % 2 else (devs[n // 2 - 1] + devs[n // 2]) / 2
                meds.append(med)
                mads.append(mad)
            kept = [row for row in data
                    if all(mads[i] <= 1e-9 or abs(row[i] - meds[i]) <= thresh * mads[i]
                           for i in range(k))]
            return kept if len(kept) >= MVN_MIN_N else data

        def compute_mu_cov(data):
            n = len(data)
            k = len(data[0])
            mu = [sum(row[i] for row in data) / n for i in range(k)]
            cov = [[0.0] * k for _ in range(k)]
            for row in data:
                for i in range(k):
                    for j in range(k):
                        cov[i][j] += (row[i] - mu[i]) * (row[j] - mu[j])
            for i in range(k):
                for j in range(k):
                    cov[i][j] /= (n - 1)
            return mu, cov

        models = {}
        all_keys = set(list(groups_mlb.keys()) + list(groups_roc.keys()))
        for key in sorted(all_keys):
            model = {}
            if key in groups_mlb and len(groups_mlb[key]) >= MVN_MIN_N:
                data = mad_screen(groups_mlb[key])
                mu, cov = compute_mu_cov(data)
                model['mlb'] = {'mu': mu, 'cov': cov, 'n': len(data)}
            if key in groups_roc and len(groups_roc[key]) >= MVN_MIN_N:
                data = mad_screen(groups_roc[key])
                mu, cov = compute_mu_cov(data)
                model['roc'] = {'mu': mu, 'cov': cov, 'n': len(data)}
            if model:
                models[key] = model
        return models

    # --- Fit VAA ~ PlateZ regressions per pitch type (MLB only) ---
    # Within-pitcher (fixed-effects) fit, same estimator as HAA below: the
    # location slope is flight geometry (~1.0 deg/ft), and demeaning per
    # pitcher keeps between-pitcher shape/location selection out of it. No
    # mirroring needed (PlateZ/VAA don't flip by hand), and the league PlateZ
    # mean stays pooled (plate height doesn't differ meaningfully by hand).
    vaa_reg_by_pt = defaultdict(list)  # pitch_type -> [(pitcher_key, plateZ, vaa)]
    for p in all_pitches:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        pt = p.get('Pitch Type') or p.get('TaggedPitchType')
        vaa_val = safe_float(p.get('VAA'))
        pz_val = safe_float(p.get('PlateZ'))
        if pt and vaa_val is not None and pz_val is not None:
            vaa_reg_by_pt[pt].append(((p.get('Pitcher'), p.get('Throws')), pz_val, vaa_val))

    print("\nVAA ~ PlateZ regressions (per pitch type, within-pitcher):")
    vaa_regressions = {}  # pitch_type -> {slope, leagueAvgPlateZ}
    for pt in sorted(vaa_reg_by_pt.keys()):
        triples = vaa_reg_by_pt[pt]
        result = fit_within_pitcher_slope(triples, f"VAA~PlateZ {pt}")
        if result:
            mean_pz = sum(t[1] for t in triples) / len(triples)
            vaa_regressions[pt] = {
                'slope': result['slope'],
                'leagueAvgPlateZ': mean_pz,
            }

    # Compute nVAA for each pitch leaderboard row using per-pitch-type slope
    for row in pitch_leaderboard:
        if row.get('vaa') is not None:
            pt = row['pitchType']
            reg = vaa_regressions.get(pt)
            if reg:
                key = (row['pitcher'], row['team'], row['pitchType'], row.get('throws'))
                pitches_for_row = pitch_groups[key]
                pz_vals = [safe_float(p.get('PlateZ')) for p in pitches_for_row]
                pz_vals = [v for v in pz_vals if v is not None]
                if pz_vals:
                    avg_pz = sum(pz_vals) / len(pz_vals)
                    row['nVAA'] = round(row['vaa'] - reg['slope'] * (avg_pz - reg['leagueAvgPlateZ']), 2)
                else:
                    row['nVAA'] = None
            else:
                row['nVAA'] = None
        else:
            row['nVAA'] = None

    # --- Fit HAA ~ PlateX regressions per pitch type (MLB only) ---
    # LHP is mirrored into the RHP frame (negate PlateX AND HAA; the slope is
    # frame-invariant because both axes flip), and the slope is fit
    # within-pitcher. Fitting hand-pooled natural-frame data blended the
    # geometric slope with the LHP/RHP cluster axis (SL fit 1.93 vs true ~1.05,
    # making nHAA ANTI-correlated with location), so both guards matter.
    # League avg PlateX is hand-specific: the pooled mean injected a
    # hand-dependent constant (RHP SL -0.35 deg, LHP +0.89) that broke the
    # |nHAA| percentile-ranking fairness between hands.
    HAA_HAND_MIN_N = 10  # min pitches for a hand-specific league PlateX mean
    haa_reg_by_pt = defaultdict(list)     # pitch_type -> [(pitcher_key, mirrored plateX, mirrored haa)]
    haa_px_by_pt_hand = defaultdict(list)  # (pitch_type, throws) -> [plateX] (natural frame)
    for p in all_pitches:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        pt = p.get('Pitch Type') or p.get('TaggedPitchType')
        haa_val = safe_float(p.get('HAA'))
        px_val = safe_float(p.get('PlateX'))
        throws = p.get('Throws')
        if pt and haa_val is not None and px_val is not None and throws in ('L', 'R'):
            s = 1.0 if throws == 'R' else -1.0
            haa_reg_by_pt[pt].append(((p.get('Pitcher'), throws), px_val * s, haa_val * s))
            haa_px_by_pt_hand[(pt, throws)].append(px_val)

    print("\nHAA ~ PlateX regressions (per pitch type, mirrored + within-pitcher):")
    haa_regressions = {}  # pitch_type -> {slope, leagueAvgPlateX: {'R':…, 'L':…}}
    for pt in sorted(haa_reg_by_pt.keys()):
        result = fit_within_pitcher_slope(haa_reg_by_pt[pt], f"HAA~PlateX {pt}")
        if result:
            means = {}
            for hand in ('R', 'L'):
                vals = haa_px_by_pt_hand.get((pt, hand))
                if vals and len(vals) >= HAA_HAND_MIN_N:
                    means[hand] = sum(vals) / len(vals)
            haa_regressions[pt] = {
                'slope': result['slope'],
                'leagueAvgPlateX': means,
            }

    # Compute nHAA for each pitch leaderboard row using per-pitch-type slope
    # and the hand-specific league PlateX mean.
    for row in pitch_leaderboard:
        if row.get('haa') is not None:
            pt = row['pitchType']
            reg = haa_regressions.get(pt)
            lg_px = reg['leagueAvgPlateX'].get(row.get('throws')) if reg else None
            if reg and lg_px is not None:
                key = (row['pitcher'], row['team'], row['pitchType'], row.get('throws'))
                pitches_for_row = pitch_groups[key]
                px_vals = [safe_float(p.get('PlateX')) for p in pitches_for_row]
                px_vals = [v for v in px_vals if v is not None]
                if px_vals:
                    avg_px = sum(px_vals) / len(px_vals)
                    row['nHAA'] = round(row['haa'] - reg['slope'] * (avg_px - lg_px), 2)
                else:
                    row['nHAA'] = None
            else:
                row['nHAA'] = None
        else:
            row['nHAA'] = None

    # --- Fit MVN expected movement models per pitch type + handedness ---
    mvn_models = fit_mvn_models(all_pitches)
    print(f"\nMVN models fitted for {len(mvn_models)} pitch-type+hand groups")
    for mvn_key, mvn_sub in sorted(mvn_models.items()):
        mlb_n = mvn_sub.get('mlb', {}).get('n', 0)
        roc_n = mvn_sub.get('roc', {}).get('n', 0)
        print(f"  {mvn_key}: MLB n={mlb_n}, ROC n={roc_n}")

    def compute_expected_movement(pitch_type, throws, arm_angle, extension, velocity, rel_z, rel_x):
        """Compute xIVB and xHB using MVN conditional model per pitch type + handedness.
        Tries MLB model (ArmAngle, Extension, Velocity) first,
        falls back to ROC model (RelPosZ, RelPosX, Extension, Velocity)."""
        mvn_key = (pitch_type or '') + '_' + (throws or '')
        pt_model = mvn_models.get(mvn_key)
        if not pt_model:
            return None, None
        if pt_model.get('mlb') and arm_angle is not None and extension is not None and velocity is not None:
            result = mvn_conditional(pt_model['mlb'], [arm_angle, extension, velocity])
            if result:
                return result[0], result[1]
        if pt_model.get('roc') and rel_z is not None and rel_x is not None and extension is not None and velocity is not None:
            result = mvn_conditional(pt_model['roc'], [rel_z, rel_x, extension, velocity])
            if result:
                return result[0], result[1]
        return None, None

    # Compute xIVB/xHB (expected) and IVBOE/HBOE (residual) for each pitch leaderboard row
    for row in pitch_leaderboard:
        xivb, xhb = compute_expected_movement(
            row.get('pitchType'), row.get('throws'),
            row.get('armAngle'), row.get('extension'), row.get('velocity'),
            row.get('relPosZ'), row.get('relPosX')
        )
        if xivb is not None:
            row['xIVB'] = round(xivb, 1)
            if row.get('indVertBrk') is not None:
                row['ivbOE'] = round(row['indVertBrk'] - xivb, 1)
            else:
                row['ivbOE'] = None
        else:
            row['xIVB'] = None
            row['ivbOE'] = None
        if xhb is not None:
            row['xHB'] = round(xhb, 1)
            if row.get('horzBrk') is not None:
                row['hbOE'] = round(row['horzBrk'] - xhb, 1)
            else:
                row['hbOE'] = None
        else:
            row['xHB'] = None
            row['hbOE'] = None

    pitch_leaderboard.sort(key=lambda r: r['count'], reverse=True)
    print(f"Pitch leaderboard: {len(pitch_leaderboard)} rows")

    # --- Pitcher Leaderboard: group by (Pitcher, PTeam) ---
    pitcher_groups = defaultdict(list)
    for p in all_pitches:
        if (p['Pitcher'], p['PTeam']) in ep_pitchers:
            continue
        if p.get('_roc_hitter_pitch'):
            continue  # Skip AAA pitchers facing ROC hitters
        key = (p['Pitcher'], p['PTeam'], p.get('Throws'))
        pitcher_groups[key].append(p)

    # Synthesize combined (2TM/3TM) pitcher groups — mirror of pitch_groups synthesis above
    for (pitcher, throws), combined_team in combined_pitcher_labels.items():
        teams = pitcher_mlb_teams[(pitcher, throws)]
        combined = []
        for team in teams:
            combined.extend(pitcher_groups[(pitcher, team, throws)])
        pitcher_groups[(pitcher, combined_team, throws)] = combined

    PITCHER_METRIC_COLS = ['RelPosZ', 'RelPosX', 'Extension', 'ArmAngle', 'VAA', 'HAA']
    PITCHER_METRIC_PCTL_KEYS = [METRIC_KEYS[c] for c in PITCHER_METRIC_COLS]
    EXPECTED_KEYS = ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon']
    EXPECTED_PITCHER_INVERT = {'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon'}
    pitcher_leaderboard = []
    for (pitcher, team, throws), pitches in pitcher_groups.items():
        row = {
            'pitcher': pitcher,
            'team': team,
            'throws': throws,
            'count': len(pitches),
            'mlbId': get_mlb_id(pitcher, team),
            '_isROC': team in AAA_TEAMS,
        }
        for col in PITCHER_METRIC_COLS:
            values = [safe_float(p.get(col)) for p in pitches]
            key_name = METRIC_KEYS[col]
            row[key_name] = round_metric(col, avg(values))
        row.update(compute_stats(pitches))
        row.update(compute_pitcher_batted_ball(pitches))
        row.update(compute_expected_stats(pitches, woba_weights=WOBA_WEIGHTS))

        # Per-hand splits for stats not in micro data (2K Whiff%, plate disc, batted ball, expected)
        for hand_label, hand_val in [('_vsL', 'L'), ('_vsR', 'R')]:
            hand_pitches = [p for p in pitches if p.get('Bats') == hand_val]
            if hand_pitches:
                hand_stats = compute_stats(hand_pitches)
                hand_bb = compute_pitcher_batted_ball(hand_pitches)
                hand_ex = compute_expected_stats(hand_pitches, woba_weights=WOBA_WEIGHTS)
                for suffix_key in ['twoStrikeWhiffPct', 'fpsPct',
                                   'strikePct', 'izPct', 'swStrPct', 'cswPct',
                                   'izWhiffPct', 'chasePct', 'kPct', 'bbPct', 'kbbPct',
                                   'babip', 'gbPct']:
                    if suffix_key in hand_stats:
                        row[suffix_key + hand_label] = hand_stats[suffix_key]
                for suffix_key in ['avgEV', 'maxEV', 'hardHitPct', 'barrelPct',
                                   'gbPct_bb', 'ldPct', 'fbPct', 'puPct', 'hrFbPct']:
                    if suffix_key in hand_bb:
                        row[suffix_key + hand_label] = hand_bb[suffix_key]
                for suffix_key in ['wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon']:
                    if suffix_key in hand_ex:
                        row[suffix_key + hand_label] = hand_ex[suffix_key]

        # Fastball velo: average velo of most-used fastball (FF/SI)
        fb_types = {'FF', 'SI'}
        fb_pitches_by_type = defaultdict(list)
        for p in pitches:
            pt = p.get('Pitch Type')
            if pt in fb_types:
                v = safe_float(p.get('Velocity'))
                if v is not None:
                    fb_pitches_by_type[pt].append(v)
        if fb_pitches_by_type:
            primary_fb_type = max(fb_pitches_by_type, key=lambda t: len(fb_pitches_by_type[t]))
            fb_velos = fb_pitches_by_type[primary_fb_type]
            row['fbVelo'] = round(sum(fb_velos) / len(fb_velos), 1) if fb_velos else None
            row['primaryFbType'] = primary_fb_type
        else:
            row['fbVelo'] = None
            row['primaryFbType'] = None

        pitcher_leaderboard.append(row)

    # Recompute pitcher runValue as sum of raw (unrounded) per-pitch-type runValues.
    # Rounding only happens at the final step to avoid accumulation error.
    pitch_rv_by_pitcher = {}
    for pr in pitch_leaderboard:
        pk = pr['pitcher'] + '|' + pr['team']
        if pr.get('runValue') is not None:
            if pk not in pitch_rv_by_pitcher:
                pitch_rv_by_pitcher[pk] = 0.0
            pitch_rv_by_pitcher[pk] += pr['runValue']
    for row in pitcher_leaderboard:
        pk = row['pitcher'] + '|' + row['team']
        if pk in pitch_rv_by_pitcher:
            row['runValue'] = pitch_rv_by_pitcher[pk]

    # Recompute pitcher xRunValue as sum of raw per-pitch-type xRunValues
    pitch_xrv_by_pitcher = {}
    for pr in pitch_leaderboard:
        pk = pr['pitcher'] + '|' + pr['team']
        if pr.get('xRunValue') is not None:
            if pk not in pitch_xrv_by_pitcher:
                pitch_xrv_by_pitcher[pk] = 0.0
            pitch_xrv_by_pitcher[pk] += pr['xRunValue']
    for row in pitcher_leaderboard:
        pk = row['pitcher'] + '|' + row['team']
        if pk in pitch_xrv_by_pitcher:
            row['xRunValue'] = pitch_xrv_by_pitcher[pk]

    # Compute RV/100 and xRV/100 from raw values before rounding
    for row in pitcher_leaderboard:
        if row.get('runValue') is not None and row.get('count', 0) > 0:
            row['rv100'] = row['runValue'] / row['count'] * 100
        else:
            row['rv100'] = None
        if row.get('xRunValue') is not None and row.get('count', 0) > 0:
            row['xRv100'] = row['xRunValue'] / row['count'] * 100
        else:
            row['xRv100'] = None

    pitcher_leaderboard.sort(key=lambda r: r['count'], reverse=True)
    print(f"Pitcher leaderboard: {len(pitcher_leaderboard)} rows")

    # --- Pitch Details ---
    pitch_details = defaultdict(list)
    for p in all_pitches:
        pitcher = p.get('Pitcher')
        team = p.get('PTeam')
        if (pitcher, team) in ep_pitchers:
            continue
        pt = p.get('Pitch Type')
        ivb = safe_float(p.get('xIndVrtBrk'))
        hb = safe_float(p.get('xHorzBrk'))
        velo = safe_float(p.get('Velocity'))
        spin = safe_float(p.get('Spin Rate'))
        tilt = p.get('OTilt') or p.get('Break Tilt')
        rel_x = safe_float(p.get('RelPosX'))
        rel_z = safe_float(p.get('RelPosZ'))
        if pitcher and pt and ivb is not None and hb is not None:
            detail = {
                'pt': pt,
                'ivb': round(ivb, 1),
                'hb': round(hb, 1),
            }
            if velo is not None:
                detail['v'] = round(velo, 1)
            if spin is not None:
                detail['sp'] = int(round(spin))
            if tilt and str(tilt).strip():
                detail['tl'] = str(tilt).strip()
            # Description (pitch outcome) — short codes for space efficiency
            desc_raw = p.get('Description', '')
            DESC_MAP = {
                'Swinging Strike': 'SS', 'Called Strike': 'CS', 'Foul': 'F',
                'In Play': 'IP', 'Ball': 'B', 'Hit By Pitch': 'HBP',
                'Intent Ball': 'IB', 'Pitchout': 'PO',
            }
            desc_code = DESC_MAP.get(desc_raw, '')
            if desc_code:
                detail['d'] = desc_code
            if rel_x is not None:
                detail['rx'] = round(rel_x, 2)
            if rel_z is not None:
                detail['rz'] = round(rel_z, 2)
            ext_val_d = safe_float(p.get('Extension'))
            if ext_val_d is not None:
                detail['ext'] = round(ext_val_d, 2)
            gd_val = normalize_date(p.get('Game Date'))
            if gd_val:
                detail['gd'] = gd_val
            px_val = safe_float(p.get('PlateX'))
            pz_val = safe_float(p.get('PlateZ'))
            szt_val = safe_float(p.get('SzTop'))
            szb_val = safe_float(p.get('SzBot'))
            bh_val = p.get('Bats')
            cnt_val = p.get('Count')
            # PlateX/Z, SzTop/SzBot source is 3 dec — preserve all of it so
            # downstream zone classification matches what the pipeline used.
            if px_val is not None:
                detail['px'] = round(px_val, 3)
            if pz_val is not None:
                detail['pz'] = round(pz_val, 3)
            if szt_val is not None:
                detail['szt'] = round(szt_val, 3)
            if szb_val is not None:
                detail['szb'] = round(szb_val, 3)
            if bh_val:
                detail['bh'] = bh_val
            if cnt_val:
                detail['cnt'] = cnt_val
            aa_val = safe_float(p.get('ArmAngle'))
            if aa_val is not None:
                detail['aa'] = round(aa_val, 1)
            # Per-pitch expected movement from MVN conditional model
            ext_val = safe_float(p.get('Extension'))
            throws_val = p.get('Throws')
            xivb_val, xhb_val = compute_expected_movement(pt, throws_val, aa_val, ext_val, velo, rel_z, rel_x)
            if xivb_val is not None:
                detail['xivb'] = round(xivb_val, 1)
            if xhb_val is not None:
                detail['xhb'] = round(xhb_val, 1)
            pitch_details[pitcher + '|' + (team or '')].append(detail)

    # Synthesize combined (2TM/3TM) pitch details entries
    for (pitcher, throws), combined_team in combined_pitcher_labels.items():
        combined_details = []
        for t in pitcher_mlb_teams[(pitcher, throws)]:
            combined_details.extend(pitch_details.get(pitcher + '|' + t, []))
        if combined_details:
            pitch_details[pitcher + '|' + combined_team] = combined_details

    print(f"Pitch details: {sum(len(v) for v in pitch_details.values())} pitches for {len(pitch_details)} pitchers")

    # --- League Averages per pitch type (weighted by pitch count, MLB only) ---
    pt_groups = defaultdict(list)
    for row in pitch_leaderboard:
        pt_groups[row['pitchType']].append(row)

    league_avgs = {}
    for pt, pt_rows_all in pt_groups.items():
        pt_rows = [r for r in pt_rows_all if not r.get('_isROC') and not r.get('_isCombined')]  # Exclude ROC + combined rows
        avgs = {}
        total_count = sum(r.get('count', 0) for r in pt_rows)
        # Pitch metrics: weighted average by count
        for metric in list(METRIC_KEYS.values()):
            pairs = [(r[metric], r.get('count', 0)) for r in pt_rows if r.get(metric) is not None and r.get('count', 0) > 0]
            if pairs:
                avgs[metric] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 2)
        # Rate stats: weighted average by count
        for stat in PITCH_STAT_KEYS:
            pairs = [(r[stat], r.get('count', 0)) for r in pt_rows if r.get(stat) is not None and r.get('count', 0) > 0]
            if pairs:
                avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)
        tilts = [r['breakTiltMinutes'] for r in pt_rows if r.get('breakTiltMinutes') is not None]
        if tilts:
            avgs['breakTiltMinutes'] = circular_mean_minutes(tilts)
            avgs['breakTilt'] = minutes_to_tilt_display(avgs['breakTiltMinutes'])
        # Expected stats: weighted by PA (from compute_stats)
        for stat in ['xBA', 'xSLG', 'xwOBA']:
            pairs = [(r[stat], r.get('pa', 0)) for r in pt_rows if r.get(stat) is not None and r.get('pa', 0) > 0]
            if pairs:
                avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)
        # xwOBAcon: BIP-only stat, weighted by nBip rather than PA so bunts/etc. don't pull the avg.
        xwc_pairs = [(r['xwOBAcon'], r.get('nBip', 0)) for r in pt_rows if r.get('xwOBAcon') is not None and r.get('nBip', 0) > 0]
        if xwc_pairs:
            avgs['xwOBAcon'] = round(sum(v * w for v, w in xwc_pairs) / sum(w for _, w in xwc_pairs), 4)
        avgs['count'] = len(pt_rows)
        league_avgs[pt] = avgs

    # Flag combined (2TM/3TM) rows so league-avg math excludes them (double-count avoidance).
    # Per-team rows are the canonical league-avg source; combined rows duplicate their data.
    def _is_combined_team(label):
        return isinstance(label, str) and label.endswith('TM') and label[:-2].isdigit()

    for row in pitcher_leaderboard:
        if _is_combined_team(row.get('team')):
            row['_isCombined'] = True
    for row in pitch_leaderboard:
        if _is_combined_team(row.get('team')):
            row['_isCombined'] = True

    # League averages for pitcher leaderboard (weighted by count/TBF, MLB only; exclude combined)
    pitcher_lb_mlb = [r for r in pitcher_leaderboard if not r.get('_isROC') and not r.get('_isCombined')]
    pitcher_league_avgs = {}
    for stat in STAT_KEYS + PITCHER_METRIC_PCTL_KEYS:
        # Use TBF as weight for rate stats, count (pitches) for pitch metrics
        weight_key = 'pa' if stat in ('kPct', 'bbPct', 'kbbPct', 'babip') else 'count'
        pairs = [(r[stat], r.get(weight_key, 0)) for r in pitcher_lb_mlb if r.get(stat) is not None and r.get(weight_key, 0) > 0]
        if pairs:
            pitcher_league_avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)
    # ERA league avg computed after boxscore merge (ERA not available yet at this point)
    # Batted ball stats: weighted by nBip
    for stat in PITCHER_BB_KEYS:
        pairs = [(r[stat], r.get('nBip', 0)) for r in pitcher_lb_mlb if r.get(stat) is not None and r.get('nBip', 0) > 0]
        if pairs:
            pitcher_league_avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)
    # Expected stats: weighted by PA
    for stat in EXPECTED_KEYS:
        pairs = [(r[stat], r.get('pa', 0)) for r in pitcher_lb_mlb if r.get(stat) is not None and r.get('pa', 0) > 0]
        if pairs:
            pitcher_league_avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)
    pitcher_league_avgs['count'] = len(pitcher_lb_mlb)

    # Loc+ — pitcher location-quality index (v2 decomposition model): per
    # pitch, ExpRV = P(swing)·[whiff/foul/BIP value surfaces] + P(take)·
    # [CS/ball count values], surfaces per (pitch group × hands) on a smoothed
    # 2-inch × zone-normalized grid, count-specific RV weights; per-pitcher
    # mean Bayesian-regressed and z-scored to 100 ± 10. See pipeline_locplus.py.
    # Also computes per-pitch-type Loc+ for the Arsenal tab (each row in
    # pitch_leaderboard gets a Loc+ standardized within its pitch-type group).
    from pipeline_locplus import compute_loc_plus
    # Loc+ is a PITCHER metric end to end, so position-player pitching is
    # excluded from its league baseline surfaces too, not just from scoring
    # (pitcher_groups/pitch_groups are already EP-guarded). EP pitches stay
    # in hitter-side league tables (SD+/CT+/xwOBAsp) — those measure hitters.
    _loc_baseline = [p for p in all_pitches
                     if (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers]
    loc_results, pitch_loc_results, loc_weights = compute_loc_plus(
        _loc_baseline, pitcher_groups, pitch_groups,
        lg_woba=GUTS_EXTRA.get('lgWOBA') if GUTS_EXTRA else None,
        woba_scale=GUTS_EXTRA.get('wOBAScale') if GUTS_EXTRA else None,
        # Per-pitch Loc+ grades for the Sheets write-back
        # (scripts/sheets_write_grades.py); local artifact, gitignored.
        dump_pitch_grades_path=os.path.join(
            DATA_DIR, f'pitch_loc_grades_{label.lower()}.json'),
    )
    for row in pitch_leaderboard:
        key = (row['pitcher'], row['team'], row['pitchType'], row.get('throws'))
        r = pitch_loc_results.get(key)
        if r is not None:
            row['locPlus'] = r['locPlus']
            row['locPlusRaw'] = round(r['raw_loc_adj'], 5)
            row['locPlusN'] = r['n_pitches']
            row['locRuns100'] = r.get('locRuns100')
        else:
            row['locPlus'] = None
            row['locPlusRaw'] = None
            row['locPlusN'] = 0
            row['locRuns100'] = None
    print(f"  Loc+ per-pitch-type computed for {len(pitch_loc_results)} rows.")

    for row in pitcher_leaderboard:
        key = (row['pitcher'], row['team'], row.get('throws'))
        r = loc_results.get(key)
        if r is not None:
            row['locPlus'] = r['locPlus']
            row['locPlusRaw'] = round(r['raw_loc_adj'], 5)
            row['locPlusN'] = r['n_pitches']
            row['locRuns100'] = r.get('locRuns100')
            row['locPlusHeatmap'] = r.get('heatmap')
            zloc = r.get('zone_loc') or {}
            row['locPlusHeart']      = (round(zloc['heart'], 5)      if zloc.get('heart')      is not None else None)
            row['locPlusShadowIn']   = (round(zloc['shadow_in'], 5)  if zloc.get('shadow_in')  is not None else None)
            row['locPlusShadowOut']  = (round(zloc['shadow_out'], 5) if zloc.get('shadow_out') is not None else None)
            row['locPlusChase']      = (round(zloc['chase'], 5)      if zloc.get('chase')      is not None else None)
            row['locPlusWaste']      = (round(zloc['waste'], 5)      if zloc.get('waste')      is not None else None)
        else:
            row['locPlus'] = None
            row['locPlusRaw'] = None
            row['locPlusN'] = 0
            row['locRuns100'] = None
            row['locPlusHeatmap'] = None
            row['locPlusHeart'] = None
            row['locPlusShadowIn'] = None
            row['locPlusShadowOut'] = None
            row['locPlusChase'] = None
            row['locPlusWaste'] = None
    pitcher_league_avgs['locPlus'] = 100.0
    print(f"  Loc+ computed for {len(loc_results)} pitchers.")

    # ======================================================================
    #  HITTER LEADERBOARD
    # ======================================================================
    print(f"\n--- Hitter Leaderboard ({label}) ---")

    hitter_groups = defaultdict(list)
    for p in all_pitches:
        if p.get('_roc_pitcher_pitch'):
            continue  # Skip AAA hitters facing ROC pitchers
        batter = p.get('Batter')
        b_team = p.get('BTeam')
        if batter and b_team and b_team in ALL_TEAMS:
            hitter_groups[(batter, b_team)].append(p)

    # ─── Synthesize multi-team (2TM/3TM) combined hitter groups ───
    # Hitters on ≥2 MLB teams (ROC excluded) get a synthetic combined row.
    hitter_mlb_teams = defaultdict(set)
    for (batter, team) in hitter_groups.keys():
        if team not in AAA_TEAMS:
            hitter_mlb_teams[batter].add(team)

    combined_hitter_labels = {}  # batter → "2TM"/"3TM"
    hitter_name_collisions = []
    for batter, teams in hitter_mlb_teams.items():
        if len(teams) < 2:
            continue
        # Detect name collisions: two different players sharing a name (e.g. the
        # LAD and ATH Max Muncys) will have distinct MLB IDs per team. In that
        # case, skip the 2TM synthesis so each player keeps their own team row.
        ids_by_team = {t: mlb_id_cache.get(f"{batter}|{t}") for t in teams}
        unique_ids = {mid for mid in ids_by_team.values() if mid is not None}
        if len(unique_ids) > 1:
            hitter_name_collisions.append((batter, ids_by_team))
            continue
        combined_team = f"{len(teams)}TM"
        combined_hitter_labels[batter] = combined_team
        combined = []
        for team in teams:
            combined.extend(hitter_groups[(batter, team)])
        hitter_groups[(batter, combined_team)] = combined
        if unique_ids:
            mlb_id_cache[f"{batter}|{combined_team}"] = next(iter(unique_ids))
    if hitter_name_collisions:
        print(f"  Skipped 2TM synthesis for {len(hitter_name_collisions)} hitter name collision(s):")
        for batter, ids in hitter_name_collisions:
            print(f"    {batter}: {ids}")

    # --- Compute SACQ zone table (league-wide LA × spray → wOBA) ---
    # Negative-LA region split at -10: league wOBAcon for -10..0 (~.247,
    # near-zero choppers/low liners that sneak through) is materially
    # higher than everything below -10 (~.135, buried toppers), so the
    # old single <0 bin masked real signal. -999 low sentinel kept so the
    # serialization below (lo > -999) emits laMin=None for the catch-all.
    LA_BINS = [(-999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
               (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 999)]
    SACQ_MIN_BIP = 20
    # (SACQ% the player stat is retired — xwOBAsp keeps the full value
    # gradient a binary quality threshold threw away. The zone table below
    # survives as the xwOBAsp lookup + player-page heatmap.)

    # Collect all BIPs with spray + wOBA data (MLB only — exclude ROC/AAA pitches)
    # Build both hand-specific (spray_dir, la_bin, bats) and pooled (spray_dir, la_bin) tables.
    # Hand-specific captures L/R differences in HR-range zones (park geometry, defensive positioning).
    # Pooled serves as fallback when hand-specific bins are too thin.
    _empty_bin = lambda: {'woba_sum': 0.0, 'woba_denom': 0.0, 'xwoba_sum': 0.0, 'xwoba_count': 0, 'count': 0}
    sacq_bins_hand = {}   # (spray_dir, la_bin_idx, bats) → accumulators
    sacq_bins_pooled = {} # (spray_dir, la_bin_idx) → accumulators
    # LA-only marginals (no spray dimension): the baseline for the spray
    # residual. resid = zone wOBAcon − LA-only wOBAcon isolates WHERE the
    # ball went given how high it was hit — the sticky pull-air skill.
    la_bins_hand = {}     # (la_bin_idx, bats) → accumulators
    la_bins_pooled = {}   # (la_bin_idx,) → accumulators
    for p in all_pitches:
        if p.get('_source', 'MLB') != 'MLB':
            continue  # Exclude ROC/AAA pitches from SACQ zone computation
        bb_type = p.get('BBType')
        if not bb_type or bb_type in BUNT_BB_TYPES:
            continue
        hc_x = safe_float(p.get('HC_X'))
        hc_y = safe_float(p.get('HC_Y'))
        la = safe_float(p.get('LaunchAngle'))
        woba_val = _bip_woba_value(p.get('Event'))
        woba_dom = 1.0
        xwoba_val = safe_float(p.get('xwOBA'))
        bats = p.get('Bats')
        if la is None or hc_x is None or hc_y is None or not bats:
            continue
        angle = spray_angle(hc_x, hc_y)
        direction = spray_direction(angle, bats)
        if not direction:
            continue
        la_bin_idx = None
        for bi, (lo, hi) in enumerate(LA_BINS):
            if lo <= la < hi:
                la_bin_idx = bi
                break
        if la_bin_idx is None:
            continue
        # Accumulate into hand-specific + pooled bins (spray × LA and LA-only)
        for key, table in [((direction, la_bin_idx, bats), sacq_bins_hand),
                           ((direction, la_bin_idx), sacq_bins_pooled),
                           ((la_bin_idx, bats), la_bins_hand),
                           ((la_bin_idx,), la_bins_pooled)]:
            if key not in table:
                table[key] = _empty_bin()
            table[key]['count'] += 1
            if woba_val is not None and woba_dom is not None and woba_dom > 0:
                table[key]['woba_sum'] += woba_val
                table[key]['woba_denom'] += woba_dom
            if xwoba_val is not None:
                table[key]['xwoba_sum'] += xwoba_val
                table[key]['xwoba_count'] += 1

    def _finalize_bins(bins):
        table = {}
        for key, data in bins.items():
            woba = data['woba_sum'] / data['woba_denom'] if data['woba_denom'] > 0 else None
            xwobacon = data['xwoba_sum'] / data['xwoba_count'] if data['xwoba_count'] > 0 else None
            table[key] = {
                'woba': round(woba, 3) if woba is not None else None,
                'xwobacon': round(xwobacon, 3) if xwobacon is not None else None,
                'count': data['count'],
            }
        return table

    sacq_zone_hand = _finalize_bins(sacq_bins_hand)
    sacq_zone_pooled = _finalize_bins(sacq_bins_pooled)
    la_zone_hand = _finalize_bins(la_bins_hand)
    la_zone_pooled = _finalize_bins(la_bins_pooled)

    def la_lookup(la_bin_idx, bats_val):
        """LA-only league wOBAcon: hand-specific first, pooled fallback
        (same SACQ_MIN_BIP convention as sacq_lookup)."""
        info = la_zone_hand.get((la_bin_idx, bats_val))
        if info and info['count'] >= SACQ_MIN_BIP and info['woba'] is not None:
            return info['woba']
        info = la_zone_pooled.get((la_bin_idx,))
        if info and info['count'] >= SACQ_MIN_BIP and info['woba'] is not None:
            return info['woba']
        return None

    def sacq_lookup(direction, la_bin_idx, bats_val):
        """Look up zone wOBA: try hand-specific first, fall back to pooled."""
        hand_info = sacq_zone_hand.get((direction, la_bin_idx, bats_val))
        if hand_info and hand_info['count'] >= SACQ_MIN_BIP and hand_info['woba'] is not None:
            return hand_info['woba']
        pooled_info = sacq_zone_pooled.get((direction, la_bin_idx))
        if pooled_info and pooled_info['count'] >= SACQ_MIN_BIP and pooled_info['woba'] is not None:
            return pooled_info['woba']
        return None

    # Build serializable zone data for frontend (hand-specific + pooled).
    # The "wobacon" field is wOBAcon (sum of wOBA event values / sum of wOBA
    # denominator weights, restricted to BIPs in the zone).
    sacq_zones_output = []
    for (direction, la_bin_idx, bats_key), info in sorted(sacq_zone_hand.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        lo, hi = LA_BINS[la_bin_idx]
        sacq_zones_output.append({
            'spray': direction,
            'laMin': lo if lo > -999 else None,
            'laMax': hi if hi < 999 else None,
            'laBin': la_bin_idx,
            'bats': bats_key,
            'wobacon': info['woba'],
            'xwobacon': info['xwobacon'],
            'count': info['count'],
        })
    # Also include pooled bins (bats=null) as fallback for frontend
    for (direction, la_bin_idx), info in sorted(sacq_zone_pooled.items(), key=lambda x: (x[0][0], x[0][1])):
        lo, hi = LA_BINS[la_bin_idx]
        sacq_zones_output.append({
            'spray': direction,
            'laMin': lo if lo > -999 else None,
            'laMax': hi if hi < 999 else None,
            'laBin': la_bin_idx,
            'bats': None,
            'wobacon': info['woba'],
            'xwobacon': info['xwobacon'],
            'count': info['count'],
        })
    # LA-only marginals for the client-side sprayVal recompute (hand-specific
    # rows carry bats; pooled fallback rows carry bats=None).
    sacq_la_zones_output = []
    for (la_bin_idx, bats_key), info in sorted(la_zone_hand.items(), key=lambda x: (x[0][1], x[0][0])):
        sacq_la_zones_output.append({
            'laBin': la_bin_idx, 'bats': bats_key,
            'wobacon': info['woba'], 'count': info['count'],
        })
    for (la_bin_idx,), info in sorted(la_zone_pooled.items()):
        sacq_la_zones_output.append({
            'laBin': la_bin_idx, 'bats': None,
            'wobacon': info['woba'], 'count': info['count'],
        })
    print(f"  SACQ zones: {len(sacq_zone_hand)} hand-specific, "
          f"{len(sacq_zone_pooled)} pooled; LA-only: {len(la_zone_hand)} hand, "
          f"{len(la_zone_pooled)} pooled")

    # --- Helper: compute xwOBAsp for a list of pitches using sacq_lookup ---
    def compute_xwobasp(pitches):
        xwobasp_sum = 0.0
        xwobasp_count = 0
        for p in pitches:
            bb_type = p.get('BBType')
            if not bb_type or bb_type in BUNT_BB_TYPES:
                continue
            hc_x = safe_float(p.get('HC_X'))
            hc_y = safe_float(p.get('HC_Y'))
            la_val = safe_float(p.get('LaunchAngle'))
            bats_val = p.get('Bats')
            if la_val is None or hc_x is None or hc_y is None or not bats_val:
                continue
            angle = spray_angle(hc_x, hc_y)
            direction = spray_direction(angle, bats_val)
            if not direction:
                continue
            la_bin_idx = None
            for bi, (lo, hi) in enumerate(LA_BINS):
                if lo <= la_val < hi:
                    la_bin_idx = bi
                    break
            if la_bin_idx is None:
                continue
            zone_woba = sacq_lookup(direction, la_bin_idx, bats_val)
            if zone_woba is not None:
                xwobasp_sum += zone_woba
                xwobasp_count += 1
        return round(xwobasp_sum / xwobasp_count, 3) if xwobasp_count > 0 else None

    # --- Helper: spray residual ("pull skill") for a list of pitches ---
    # Per BIP: zone wOBAcon (spray × LA) minus LA-only league wOBAcon —
    # isolates WHERE the ball went given how high it was hit. This is the
    # sticky pull-air placement skill (public benchmarks: pulled-air%
    # stabilizes ~340 BBE; each pp above league ≈ +.005 wOBA over xwOBA),
    # near-orthogonal to xwOBAcon by construction. Returned in wOBA points.
    def compute_sprayval(pitches):
        resid_sum = 0.0
        resid_count = 0
        for p in pitches:
            bb_type = p.get('BBType')
            if not bb_type or bb_type in BUNT_BB_TYPES:
                continue
            hc_x = safe_float(p.get('HC_X'))
            hc_y = safe_float(p.get('HC_Y'))
            la_val = safe_float(p.get('LaunchAngle'))
            bats_val = p.get('Bats')
            if la_val is None or hc_x is None or hc_y is None or not bats_val:
                continue
            angle = spray_angle(hc_x, hc_y)
            direction = spray_direction(angle, bats_val)
            if not direction:
                continue
            la_bin_idx = None
            for bi, (lo, hi) in enumerate(LA_BINS):
                if lo <= la_val < hi:
                    la_bin_idx = bi
                    break
            if la_bin_idx is None:
                continue
            zone_woba = sacq_lookup(direction, la_bin_idx, bats_val)
            la_woba = la_lookup(la_bin_idx, bats_val)
            if zone_woba is not None and la_woba is not None:
                resid_sum += zone_woba - la_woba
                resid_count += 1
        return round(resid_sum / resid_count, 4) if resid_count > 0 else None

    # --- Compute xwOBAsp for each pitcher (second pass, requires sacq_zone_table) ---
    pitcher_pitch_lookup = {}
    for (pitcher, team, throws), pitches in pitcher_groups.items():
        pitcher_pitch_lookup[(pitcher, team)] = pitches

    for row in pitcher_leaderboard:
        pitches = pitcher_pitch_lookup.get((row['pitcher'], row['team']), [])
        row['xwOBAsp'] = compute_xwobasp(pitches)

    # --- Compute xwOBAsp per pitch type for pitch_leaderboard ---
    pitch_type_lookup = {}
    for (pitcher, team, pitch_type, throws), pitches in pitch_groups.items():
        pitch_type_lookup[(pitcher, team, pitch_type)] = pitches

    for row in pitch_leaderboard:
        pitches = pitch_type_lookup.get((row['pitcher'], row['team'], row['pitchType']), [])
        row['xwOBAsp'] = compute_xwobasp(pitches)
        # Per-hand splits for the pitch tab's hand filter (js/aggregator.js
        # xKeys merge). Without these the client silently fell back to the
        # both-hands value while its siblings (xwOBA etc.) switched hands.
        for _hand in ('L', 'R'):
            row['xwOBAsp_vs' + _hand] = compute_xwobasp(
                [p for p in pitches if p.get('Bats') == _hand])

    hitter_leaderboard = []
    for (hitter, team), pitches in hitter_groups.items():
        stands_set = set(p.get('Bats') for p in pitches if p.get('Bats'))
        if len(stands_set) > 1:
            stands = 'S'
        elif len(stands_set) == 1:
            stands = stands_set.pop()
        else:
            stands = None

        # Latest game date present in this hitter's pitch data. Used by
        # downstream consumers (e.g. card generator's "Through {date}" stamp)
        # so they can show a freshness stamp without needing the pitch-level
        # pickle, which is gitignored and won't propagate from CI.
        _hitter_dates = [p.get('Game Date') for p in pitches if p.get('Game Date')]
        last_game_date = max(_hitter_dates) if _hitter_dates else None

        row = {
            'hitter': hitter,
            'team': team,
            'stands': stands,
            'count': len(pitches),
            'mlbId': get_mlb_id(hitter, team),
            '_isROC': team in AAA_TEAMS,
            'lastGameDate': last_game_date,
        }
        row.update(compute_hitter_stats(pitches))
        row.update(compute_expected_stats(pitches, woba_weights=WOBA_WEIGHTS))
        row.update(compute_xrv(pitches,
                                lg_woba=_xrv_lg, woba_scale=_xrv_scale,
                                count_offsets=XRV_COUNT_OFFSETS,
                                bip_count_means=XRV_BIP_COUNT_MEANS,
                                negate=True))

        # SKILL METRICS EXCLUDE POSITION-PLAYER PITCHING (2026-07-25).
        # The rule: EP counts for the official ledger, not for skill estimation.
        # Counting stats, the slash line, wOBA and xwOBA above keep every EP PA
        # (the 2026-07-13 policy, so pitch-derived totals still reconcile to
        # official). But xwOBAcon / xwOBAsp / sprayVal are ABILITY estimates
        # with no official total to match, and a catcher's 48mph eephus is not
        # evidence about a hitter's contact quality. Measured: 445 EP batted
        # balls league-wide, median |BB+| shift 0.16 but up to 2.9 points, in
        # BOTH directions — so this removes noise rather than a bias.
        # hitter_league_avgs['xwOBAcon'] (the BB+ denominator) is derived from
        # these rows, so it inherits the exclusion automatically.
        _skill = ([p for p in pitches
                   if (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers]
                  if ep_pitchers else pitches)
        if len(_skill) != len(pitches):
            row['xwOBAcon'] = compute_expected_stats(
                _skill, woba_weights=WOBA_WEIGHTS).get('xwOBAcon')
        row['xwOBAsp'] = compute_xwobasp(_skill)
        row['sprayVal'] = compute_sprayval(_skill)

        hitter_leaderboard.append(row)

    # --- Merge Sprint Speed from Baseball Savant ---
    sprint_speeds = fetch_sprint_speed()
    sprint_merged = 0
    for row in hitter_leaderboard:
        mlb_id = row.get('mlbId')
        if mlb_id and mlb_id in sprint_speeds:
            ss = sprint_speeds[mlb_id]
            row['sprintSpeed'] = ss['speed']
            row['nCompRuns'] = ss['competitive_runs']
            row['sprintQual'] = ss['competitive_runs'] >= 10
            sprint_merged += 1
        else:
            row['sprintSpeed'] = None
            row['nCompRuns'] = 0
            row['sprintQual'] = False
    print(f"  Sprint speed merged for {sprint_merged}/{len(hitter_leaderboard)} hitters")

    # --- Determine primary MLB position per hitter (max games, MLB only) ---
    # Source: MLB Stats API per-player season fielding stats. A player with
    # games at multiple positions contributes +1 game to each position they
    # appeared at. The position with the most games is recorded as their
    # primary. Used in the hitter player-page bio line. Cached daily.
    _pos_lookup = fetch_hitter_positions(
        ((row.get('hitter'), row.get('mlbId')) for row in hitter_leaderboard if row.get('mlbId'))
    )
    for row in hitter_leaderboard:
        mlb_id = row.get('mlbId')
        row['position'] = _pos_lookup.get(mlb_id) if mlb_id else None

    # Flag hitters with sufficient BIP for batted ball percentile qualification
    for row in hitter_leaderboard:
        row['bipQual'] = (row.get('nBip') or 0) >= 25  # match QUAL.MIN_BIP_PCTL (client)

    # Per-100-PITCH run value for the hitter percentile panel's "Overall" row
    # (2026-07-03; was per 100 PA — a house deviation from the Savant/FG
    # convention that also mixed denominators with the per-pitch-type rows in
    # the same panel). Stored at full precision (no intermediate rounding)
    # per the RV memory; display layer rounds at render time.
    for row in hitter_leaderboard:
        n_pitches = row.get('count') or 0
        rv = row.get('runValue')
        xrv = row.get('xRunValue')
        row['rv100'] = (rv / n_pitches * 100) if (rv is not None and n_pitches > 0) else None
        row['xRv100'] = (xrv / n_pitches * 100) if (xrv is not None and n_pitches > 0) else None

    hitter_leaderboard.sort(key=lambda r: r.get('pa', 0), reverse=True)
    print(f"Hitter leaderboard: {len(hitter_leaderboard)} rows")

    # --- Hitter pitch details ---
    hitter_pitch_details = {}
    for (hitter, team), pitches in hitter_groups.items():
        pt_map = defaultdict(list)
        for p in pitches:
            pt = p.get('Pitch Type')
            if pt:
                pt_map[pt].append(p)

        details = []
        for pt, pt_pitches in sorted(pt_map.items()):
            entry = {
                'pitchType': pt,
                'count': len(pt_pitches),
            }
            entry.update(compute_hitter_stats(pt_pitches))
            details.append(entry)
        details.sort(key=lambda x: x['count'], reverse=True)
        hitter_pitch_details[hitter + '|' + (team or '')] = details

    # --- Hitter swing locations (for swing heat maps on player pages) ---
    # Per-pitch records for each pitch the hitter swung at. Powers the
    # Swings / Whiffs / Damage heat maps with platoon filtering. Compact
    # per-row format: [plateX, plateZ, eventCode, xwOBA, pitcherHand]
    #   eventCode: 1 = swing-other (foul, foul tip, contact-not-bip)
    #              2 = whiff (Swinging Strike)
    #              3 = BIP (In Play)
    #   xwOBA: only stored for BIPs; null otherwise
    #   pitcherHand: 'R' / 'L' single char
    # Bunts are not swings (matches SWING_DESCRIPTIONS / the leaderboard + cards).
    # 'Foul Tip' is already normalized to 'Swinging Strike' upstream, so it's dead here.
    SWING_DESC_FULL = {'Swinging Strike', 'Foul', 'In Play'}
    hitter_swing_locations = {}
    for (hitter, team), pitches in hitter_groups.items():
        sz_tops, sz_bots = [], []
        records = []
        # Total pitches faced per platoon (denominator for "% of pitches" swing
        # rate when the platoon toggle is active).
        n_all = 0; n_r = 0; n_l = 0
        for p in pitches:
            n_all += 1
            ph = p.get('Throws') or ''
            if ph == 'R': n_r += 1
            elif ph == 'L': n_l += 1
            sz_t = safe_float(p.get('SzTop'))
            sz_b = safe_float(p.get('SzBot'))
            if sz_t is not None: sz_tops.append(sz_t)
            if sz_b is not None: sz_bots.append(sz_b)
            desc = p.get('Description', '')
            if desc not in SWING_DESC_FULL:
                continue
            px = safe_float(p.get('PlateX'))
            pz = safe_float(p.get('PlateZ'))
            if px is None or pz is None:
                continue
            if desc == 'Swinging Strike':
                evt = 2  # whiff
            elif desc == 'In Play':
                evt = 3  # BIP
            else:
                evt = 1  # other swing (foul, foul tip)
            xw = safe_float(p.get('xwOBA')) if evt == 3 else None
            records.append([round(px, 3), round(pz, 3), evt, xw, ph])
        if not records:
            continue
        # SzTop/SzBot are typically constant per hitter (height-formula) but
        # average defensively in case of edge cases.
        sz_top = round(sum(sz_tops) / len(sz_tops), 3) if sz_tops else None
        sz_bot = round(sum(sz_bots) / len(sz_bots), 3) if sz_bots else None
        hitter_swing_locations[hitter + '|' + (team or '')] = {
            'szTop': sz_top,
            'szBot': sz_bot,
            'nAll': n_all,
            'nR': n_r,
            'nL': n_l,
            'records': records,
        }

    # --- Hitter pitch-type leaderboard ---
    HITTER_PITCH_PCTL_KEYS = [
        'avg', 'slg', 'iso',
        'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp',
        'ev50', 'maxEV', 'hardHitPct', 'barrelPct',
        'gbPct', 'ldPct', 'fbPct', 'hrFbPct',
        'pullPct', 'oppoPct',
        'swingPct', 'izSwingPct', 'chasePct', 'izSwChase', 'firstPitchSwingPct',
        'contactPct', 'izContactPct', 'whiffPct', 'twoStrikeWhiffPct',
        'runValue', 'rv100', 'xRunValue', 'xRv100',
    ]
    HITTER_PITCH_INVERT_PCTL = {'swingPct', 'chasePct', 'whiffPct', 'gbPct', 'twoStrikeWhiffPct'}

    PITCH_CATEGORIES = {
        'Hard': ['FF', 'SI'],
        'Breaking': ['FC', 'SL', 'ST', 'CU', 'SV'],
        'Offspeed': ['CH', 'FS', 'KN'],
    }

    hitter_pitch_leaderboard = []
    for (hitter, team), pitches in hitter_groups.items():
        total_count = len(pitches)
        stands_set = set(p.get('Bats') for p in pitches if p.get('Bats'))
        stands = 'S' if len(stands_set) > 1 else (stands_set.pop() if stands_set else None)

        pt_map = defaultdict(list)
        for p in pitches:
            pt = p.get('Pitch Type')
            if pt:
                pt_map[pt].append(p)

        is_roc = team in AAA_TEAMS
        for pt, pt_pitches in pt_map.items():
            row = {
                'hitter': hitter,
                'team': team,
                'stands': stands,
                'pitchType': pt,
                'count': len(pt_pitches),
                'seenPct': round(len(pt_pitches) / total_count, 4) if total_count else 0,
                'mlbId': get_mlb_id(hitter, team),
                '_isROC': is_roc,
            }
            row.update(compute_hitter_stats(pt_pitches))
            row.update(compute_expected_stats(pt_pitches, woba_weights=WOBA_WEIGHTS))
            row.update(compute_xrv(pt_pitches,
                                    lg_woba=_xrv_lg, woba_scale=_xrv_scale,
                                    count_offsets=XRV_COUNT_OFFSETS,
                                    bip_count_means=XRV_BIP_COUNT_MEANS,
                                    negate=True))
            hitter_pitch_leaderboard.append(row)

        row_all = {
            'hitter': hitter,
            'team': team,
            'stands': stands,
            'pitchType': 'All',
            'count': total_count,
            'seenPct': 1.0,
            'mlbId': get_mlb_id(hitter, team),
            '_isROC': is_roc,
        }
        row_all.update(compute_hitter_stats(pitches))
        row_all.update(compute_expected_stats(pitches, woba_weights=WOBA_WEIGHTS))
        row_all.update(compute_xrv(pitches,
                                    lg_woba=_xrv_lg, woba_scale=_xrv_scale,
                                    count_offsets=XRV_COUNT_OFFSETS,
                                    bip_count_means=XRV_BIP_COUNT_MEANS,
                                    negate=True))
        hitter_pitch_leaderboard.append(row_all)

        for cat_name, cat_types in PITCH_CATEGORIES.items():
            cat_pitches = []
            cat_seen = 0.0
            for ct in cat_types:
                if ct in pt_map:
                    cat_pitches.extend(pt_map[ct])
                    cat_seen += len(pt_map[ct]) / total_count if total_count else 0
            if len(cat_pitches) > 0:
                row_cat = {
                    'hitter': hitter,
                    'team': team,
                    'stands': stands,
                    'pitchType': cat_name,
                    'count': len(cat_pitches),
                    'seenPct': round(cat_seen, 4),
                    'mlbId': get_mlb_id(hitter, team),
                    '_isROC': is_roc,
                }
                row_cat.update(compute_hitter_stats(cat_pitches))
                row_cat.update(compute_expected_stats(cat_pitches, woba_weights=WOBA_WEIGHTS))
                row_cat.update(compute_xrv(cat_pitches,
                                            lg_woba=_xrv_lg, woba_scale=_xrv_scale,
                                            count_offsets=XRV_COUNT_OFFSETS,
                                            bip_count_means=XRV_BIP_COUNT_MEANS,
                                            negate=True))
                hitter_pitch_leaderboard.append(row_cat)

    # Compute rv100 and xRv100 for hitter pitch leaderboard rows
    for row in hitter_pitch_leaderboard:
        if row.get('runValue') is not None and row.get('count', 0) > 0:
            row['rv100'] = row['runValue'] / row['count'] * 100
        else:
            row['rv100'] = None
        if row.get('xRunValue') is not None and row.get('count', 0) > 0:
            row['xRv100'] = row['xRunValue'] / row['count'] * 100
        else:
            row['xRv100'] = None

    # Compute xwOBAsp per pitch type for hitter_pitch_leaderboard
    for row in hitter_pitch_leaderboard:
        pt = row['pitchType']
        hitter_pitches = hitter_groups.get((row['hitter'], row['team']), [])
        if pt == 'All':
            pt_pitches = hitter_pitches
        elif pt in PITCH_CATEGORIES:
            cat_set = set(PITCH_CATEGORIES[pt])
            pt_pitches = [p for p in hitter_pitches if p.get('Pitch Type') in cat_set]
        else:
            pt_pitches = [p for p in hitter_pitches if p.get('Pitch Type') == pt]
        row['xwOBAsp'] = compute_xwobasp(pt_pitches)

    hitter_pitch_leaderboard.sort(key=lambda r: r.get('count', 0), reverse=True)
    print(f"Hitter pitch leaderboard: {len(hitter_pitch_leaderboard)} rows")

    # Flag combined (2TM/3TM) hitter rows so league-avg math excludes them.
    for row in hitter_leaderboard:
        if _is_combined_team(row.get('team')):
            row['_isCombined'] = True
    for row in hitter_pitch_leaderboard:
        if _is_combined_team(row.get('team')):
            row['_isCombined'] = True

    # Hitter league averages (weighted by PA for rate stats, nBip for batted ball stats, MLB only; exclude combined)
    hitter_lb_mlb = [r for r in hitter_leaderboard if not r.get('_isROC') and not r.get('_isCombined')]
    hitter_league_avgs = {}
    # Rate stats weighted by PA
    pa_stats = {'avg', 'obp', 'slg', 'ops', 'iso', 'babip', 'kPct', 'bbPct', 'bbToK', 'hrFbPct',
                'wOBA', 'xBA', 'xSLG', 'xwOBA', 'rv100', 'xRv100',
                'swingPct', 'izSwingPct', 'chasePct', 'izSwChase', 'contactPct', 'izContactPct', 'whiffPct'}
    # Batted ball stats weighted by nBip
    bip_stats = {'avgEVAll', 'ev50', 'maxEV', 'medLA', 'hardHitPct', 'barrelPct',
                 'xwOBAcon', 'xwOBAsp', 'sprayVal',
                 'gbPct', 'ldPct', 'fbPct', 'puPct',
                 'pullPct', 'middlePct', 'oppoPct', 'airPullPct'}
    # Bat tracking stats weighted by nCompSwings
    comp_swing_stats = {'batSpeed', 'swingLength', 'attackAngle', 'attackDirection', 'swingPathTilt',
                        'blastPct', 'squaredUpPct', 'idealAAPct'}
    # Counting stats that should NOT appear on the league-average row (meaningless
    # weighted means of counting totals).
    hitter_no_lg_avg = {'hr', 'sb'}

    def _compute_hitter_lg_avg(stat):
        if stat in hitter_no_lg_avg:
            return
        if stat in pa_stats:
            weight_key = 'pa'
        elif stat in bip_stats:
            weight_key = 'nBip'
        elif stat in comp_swing_stats:
            weight_key = 'nCompSwings'
        else:
            weight_key = 'pa'  # default
        pairs = [(r[stat], r.get(weight_key, 0)) for r in hitter_lb_mlb if r.get(stat) is not None and r.get(weight_key, 0) > 0]
        if pairs:
            hitter_league_avgs[stat] = round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 4)

    for stat in HITTER_STAT_KEYS:
        _compute_hitter_lg_avg(stat)
    hitter_league_avgs['count'] = len(hitter_lb_mlb)

    # BB+ — batted-ball contact-quality index indexed to 100 = league avg.
    # 2026-07-13 redefinition (multi-season derivation): BB+ is PURE
    # xwOBAcon+. The 0.15 spray weight (prior-informed, 2026-07-02) was
    # retired by the 2021-2025 season-pair test (scripts/
    # derive_weights_multiseason.py, 1,181 hitter-pairs, year-N components →
    # year-N+1 wOBA): sprayVal's univariate r was -0.03 and its ridge beta
    # NEGATIVE; the pure-con BB+ beat the 85/15 mix in every leave-one-
    # pair-out fold. sprayVal stays a display/scouting stat — it's a real
    # trait (split-half r=.33), it just adds nothing to forecasting value
    # once xwOBAcon is in the composite.
    BB_PLUS_W_CON = 1.0
    BB_PLUS_W_SP  = 0.0
    # Reliability handling: Bayesian shrinkage toward league (100), matching
    # the SD+/CT+ convention, replacing the old hard 80-BIP gate. n0 = 60:
    # the same-protocol split-half study (scripts/bbplus_n0_cliff_test.py +
    # n0_remeasure_ship.py, 2021-2026) put the implied n0 at 54-66 at EVERY
    # sample size tested (59 consensus for the pure-con definition) — a
    # near-perfect rel(n)=n/(n+n0) fit. Sub-80-BIP signal is real (50-79
    # BIP band predicts next-season wOBA at r=+.33 vs +.38 for 80+), so
    # shrink it, don't hide it. MIN_BIP=30 is a display floor only (below
    # it a score is >2/3 prior — not worth a cell).
    # Mirrored in js/aggregator.js via metadata bbPlusWeights/bbPlusShrinkN0
    # (the only place BB+ is recomputed client-side).
    BB_PLUS_SHRINK_N0 = 60
    BB_PLUS_MIN_BIP = 30
    lg_xwobacon_bb = hitter_league_avgs.get('xwOBAcon')
    for row in hitter_leaderboard:
        xc = row.get('xwOBAcon')
        sv = row.get('sprayVal')
        n_bip = row.get('nBip') or 0
        sp_ok = (BB_PLUS_W_SP == 0.0 or sv is not None)
        if (xc is not None and sp_ok and lg_xwobacon_bb
                and n_bip >= BB_PLUS_MIN_BIP):
            con_plus = 100.0 * xc / lg_xwobacon_bb
            # sprayPlus: the residual re-expressed on the xwOBAcon scale so
            # the ratio-to-100 convention holds (100 + 100·resid/lg).
            # Weight is 0.0 as of 2026-07-13; term kept so the metadata
            # weight contract (and the JS mirror) stays general.
            sp_plus = (100.0 * (lg_xwobacon_bb + sv) / lg_xwobacon_bb
                       if sv is not None else 100.0)
            raw = BB_PLUS_W_CON * con_plus + BB_PLUS_W_SP * sp_plus
            row['bbPlus'] = round(
                (n_bip * raw + BB_PLUS_SHRINK_N0 * 100.0)
                / (n_bip + BB_PLUS_SHRINK_N0), 1)
        else:
            row['bbPlus'] = None
    hitter_league_avgs['bbPlus'] = 100.0

    # PD+ is retired. Superseded by SD+ (decision) and CT+ (contact-frequency).
    # Hitter+ now composites BB+, SD+, CT+ directly; see below.

    # SD+ — decision-only discipline index (xRV-weighted cells, dv_A formula,
    # Bayesian-regressed to league, ratio-to-league). See pipeline_sdplus.py.
    from pipeline_sdplus import compute_sd_plus, compute_team_games_played
    # Include MLB teams AND multi-team aggregates AND ROC. Cell weight
    # tables stay MLB-baselined (filter applied inside compute_sd_plus /
    # compute_ct_plus); per-hitter aggregation looks ROC swings up
    # against the MLB table (translation framing, same as xwOBAsp /
    # xwOBAcon). The qualified-pool re-anchor downstream is MLB-only,
    # so any minor lg_raw shift from ROC entering regress_and_normalize
    # is normalized out at the final sdPlus / ctPlus scale.
    # Position-player pitching is excluded from BOTH the league cell tables and
    # the per-hitter accumulation (2026-07-25), same rule as xwOBAcon above:
    # SD+/CT+ are skill estimates with no official total to reconcile to, and a
    # 48mph eephus is not evidence about a hitter's swing decisions. Those PAs
    # still count in the slash line / wOBA / wRC+ per the 2026-07-13 policy.
    _sd_league = ([p for p in all_pitches
                   if (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers]
                  if ep_pitchers else all_pitches)
    sd_pitches_by_hitter = {
        k: [p for p in v if (p.get('Pitcher'), p.get('PTeam')) not in ep_pitchers]
        for k, v in hitter_groups.items()} if ep_pitchers else dict(hitter_groups)
    sd_results, sd_weights = compute_sd_plus(
        _sd_league, sd_pitches_by_hitter,
        lg_woba=GUTS_EXTRA.get('lgWOBA') if GUTS_EXTRA else None,
        woba_scale=GUTS_EXTRA.get('wOBAScale') if GUTS_EXTRA else None,
    )
    for row in hitter_leaderboard:
        key = (row['hitter'], row['team'])
        r = sd_results.get(key)
        if r is not None:
            row['sdPlus'] = r['sdPlus']
            row['sdPlusRaw'] = round(r['raw_sd_adj'], 5)
            row['sdPlusN'] = r['n_decisions']
            zdv = r.get('zone_dv') or {}
            row['sdPlusHeart']      = (round(zdv['heart'], 5)      if zdv.get('heart')      is not None else None)
            row['sdPlusShadowIn']   = (round(zdv['shadow_in'], 5)  if zdv.get('shadow_in')  is not None else None)
            row['sdPlusShadowOut']  = (round(zdv['shadow_out'], 5) if zdv.get('shadow_out') is not None else None)
            row['sdPlusChase']      = (round(zdv['chase'], 5)      if zdv.get('chase')      is not None else None)
            row['sdPlusWaste']      = (round(zdv['waste'], 5)      if zdv.get('waste')      is not None else None)
        else:
            row['sdPlus'] = None
            row['sdPlusRaw'] = None
            row['sdPlusN'] = 0
            row['sdPlusHeart'] = None
            row['sdPlusShadowIn'] = None
            row['sdPlusShadowOut'] = None
            row['sdPlusChase'] = None
            row['sdPlusWaste'] = None
    hitter_league_avgs['sdPlus'] = 100.0
    print(f"  SD+ computed for {len(sd_results)} qualified hitters.")

    # CT+ — contact-execution index: leverage-weighted actual contact over
    # leverage-weighted EXPECTED contact for the same swings (league p_whiff
    # per zone × count cell), so swing-mix effects stay in SD+ where they
    # belong. See pipeline_contact.py.
    from pipeline_contact import compute_ct_plus
    ct_results, ct_weights = compute_ct_plus(
        _sd_league, sd_pitches_by_hitter,
        lg_woba=GUTS_EXTRA.get('lgWOBA') if GUTS_EXTRA else None,
        woba_scale=GUTS_EXTRA.get('wOBAScale') if GUTS_EXTRA else None,
    )
    for row in hitter_leaderboard:
        key = (row['hitter'], row['team'])
        r = ct_results.get(key)
        if r is not None:
            row['ctPlus'] = r['ctPlus']
            row['ctPlusRaw'] = round(r['raw_ct_adj'], 5)
            row['ctPlusN'] = r['n_swings']
            zdv = r.get('zone_dv') or {}
            row['ctPlusHeart']      = (round(zdv['heart'], 5)      if zdv.get('heart')      is not None else None)
            row['ctPlusShadowIn']   = (round(zdv['shadow_in'], 5)  if zdv.get('shadow_in')  is not None else None)
            row['ctPlusShadowOut']  = (round(zdv['shadow_out'], 5) if zdv.get('shadow_out') is not None else None)
            row['ctPlusChase']      = (round(zdv['chase'], 5)      if zdv.get('chase')      is not None else None)
            row['ctPlusWaste']      = (round(zdv['waste'], 5)      if zdv.get('waste')      is not None else None)
        else:
            row['ctPlus'] = None
            row['ctPlusRaw'] = None
            row['ctPlusN'] = 0
            row['ctPlusHeart'] = None
            row['ctPlusShadowIn'] = None
            row['ctPlusShadowOut'] = None
            row['ctPlusChase'] = None
            row['ctPlusWaste'] = None
    hitter_league_avgs['ctPlus'] = 100.0
    print(f"  CT+ computed for {len(ct_results)} qualified hitters.")

    # team_games_played — used for 3.1 PA × TGP leaderboard qualification
    team_games_played = compute_team_games_played(all_pitches)
    print(f"  Team games played: {dict(sorted(team_games_played.items()))}")

    # Hitter+ — composite of BB+ (contact quality), SD+ (decision quality),
    # CT+ (contact frequency), combined on z-scores of the qualified pool.
    # Weights (2026-07-13): 52/17/31, derived MULTI-SEASON out of sample
    # (scripts/derive_weights_multiseason.py + derive_weights_lopo2.py:
    # full-season year-N components → year-N+1 wOBA, pairs 2021→22 …
    # 2024→25, 1,181 hitter-pairs, components z-scored within pair). The
    # betas are stable in every pair (bb .51-.54, sd .11-.20, ct .27-.38)
    # and leave-one-pair-out prediction improves from r=.450 (the old
    # prior-informed 70/15/15) to r=.497. CT+ is a SUPPRESSOR: corr -0.52
    # with BB+ (damage swings whiff more), so its univariate r is ~0 while
    # its partial weight is large — do not "sanity check" it univariately.
    # The plateau is flat across 50-55/15-20/30, so small perturbations are
    # style, not signal. (History: 65/7/28 same-season wRC+ OLS retired
    # 2026-07-02 — collinearity artifact; 70/15/15 prior retired 2026-07-13
    # by the derivation above. The half-season derivation couldn't price
    # SD+/CT+ at n=209; four full season pairs can.)
    HITTER_PLUS_W_BB = 0.52
    HITTER_PLUS_W_SD = 0.17
    HITTER_PLUS_W_CT = 0.31
    HITTER_PLUS_TARGET_SD = 40  # interim/fallback scale. The final display
                                # scale is set AFTER the FG override fills
                                # wRC+: the wRC+ scale-match step re-scales
                                # the qualified pool's SD to the pool's
                                # measured wRC+ SD (~23 mid-season), so
                                # Hitter+ reads in wRC+'s currency. This 40
                                # only survives if wRC+ fails to populate.

    # Standardization uses the leaderboard-qualified hitter pool (ROC-aware:
    # 3.1 PA×TG for MLB, 2.7 for ROC). In practice ROC hitters have None
    # bbPlus/sdPlus/ctPlus so they're excluded anyway, but the threshold is
    # ROC-aware for consistency with the rest of the qualification logic.
    from pipeline_utils import hitter_pa_per_game as _hitter_pa_per_game
    _hplus_qual = []
    for _row in hitter_leaderboard:
        # MLB-only standardization pool: ROC hitters now have full bb/sd/ct
        # via the Tier 1 SD+/CT+ unlock and the Tier 2 xwOBAcon fill, but
        # the Hitter+ baseline must stay MLB-anchored (translation framing,
        # same convention as bbPlus re-anchor and percentile pool — ROC
        # ranks against MLB, doesn't contribute to the MLB baseline).
        if _row.get('_isROC') or _row.get('_isCombined'):
            continue
        _team_g = team_games_played.get(_row.get('team'))
        if _team_g is None and team_games_played:
            _team_g = max(team_games_played.values())
        _pa_thresh = _hitter_pa_per_game(False) * (_team_g or 0)
        if _team_g and _row.get('pa', 0) >= _pa_thresh and \
           _row.get('bbPlus') is not None and _row.get('sdPlus') is not None and _row.get('ctPlus') is not None:
            _hplus_qual.append(_row)
    if len(_hplus_qual) >= 10:
        def _mean(vals): return sum(vals)/len(vals)
        def _sd(vals):
            m = _mean(vals)
            return math.sqrt(sum((v-m)**2 for v in vals)/len(vals))
        _m_bb = _mean([h['bbPlus'] for h in _hplus_qual]); _s_bb = _sd([h['bbPlus'] for h in _hplus_qual])
        _m_sd = _mean([h['sdPlus'] for h in _hplus_qual]); _s_sd = _sd([h['sdPlus'] for h in _hplus_qual])
        _m_ct = _mean([h['ctPlus'] for h in _hplus_qual]); _s_ct = _sd([h['ctPlus'] for h in _hplus_qual])
    else:
        # Defensive fallback — shouldn't trigger in any real season
        _m_bb, _s_bb = 100.0, 15.0
        _m_sd, _s_sd = 100.0, 10.0
        _m_ct, _s_ct = 100.0,  3.0

    hitter_plus_standardization = {
        'bbPlus': {'mean': round(_m_bb, 3), 'sd': round(_s_bb, 3)},
        'sdPlus': {'mean': round(_m_sd, 3), 'sd': round(_s_sd, 3)},
        'ctPlus': {'mean': round(_m_ct, 3), 'sd': round(_s_ct, 3)},
        'weights': {'bb': HITTER_PLUS_W_BB, 'sd': HITTER_PLUS_W_SD, 'ct': HITTER_PLUS_W_CT},
        'scale': None,  # set below after the realized-SD rescale is computed
        'nQualified': len(_hplus_qual),
    }

    def _composite_z(row):
        bbp, sdp, ctp = row.get('bbPlus'), row.get('sdPlus'), row.get('ctPlus')
        if bbp is None or sdp is None or ctp is None:
            return None
        if _s_bb <= 0 or _s_sd <= 0 or _s_ct <= 0:
            return None
        return (HITTER_PLUS_W_BB * (bbp - _m_bb) / _s_bb
                + HITTER_PLUS_W_SD * (sdp - _m_sd) / _s_sd
                + HITTER_PLUS_W_CT * (ctp - _m_ct) / _s_ct)

    # Rescale so the REALIZED SD over the qualified pool is exactly
    # HITTER_PLUS_TARGET_SD (wRC+-like spread by construction, not by hope).
    _qual_zs = [z for z in (_composite_z(h) for h in _hplus_qual) if z is not None]
    _z_sd = _sd(_qual_zs) if len(_qual_zs) >= 10 else 0.0
    _scale = HITTER_PLUS_TARGET_SD / _z_sd if _z_sd > 1e-9 else HITTER_PLUS_TARGET_SD
    hitter_plus_standardization['scale'] = round(_scale, 3)

    for row in hitter_leaderboard:
        z = _composite_z(row)
        row['hitterPlus'] = round(100 + _scale * z, 1) if z is not None else None
    hitter_league_avgs['hitterPlus'] = 100.0
    print(f"  Hitter+ computed (BB+/SD+/CT+ composite, weights "
          f"{HITTER_PLUS_W_BB:.0%}/{HITTER_PLUS_W_SD:.0%}/{HITTER_PLUS_W_CT:.0%}, "
          f"scale {_scale:.1f} → SD {HITTER_PLUS_TARGET_SD}).")

    # ── All-MLB-mean re-anchor for the four "+" indices ──────────────
    # Anchor 100 to the PA-weighted MEAN of ALL MLB hitters (matches the
    # FanGraphs/Savant convention used for wRC+ and the rest of the
    # pipeline's "league average" numbers: every player contributes to
    # the mean, qualification only gates percentile coloring at render).
    # bb/sd/ct are ratio indices → rescale multiplicatively (×100/mean).
    # hitterPlus is an additive z-index (100 + 40·z) → recenter additively
    # (+100−mean) to preserve its SD spread. The bbPlus factor is published
    # in metadata because the frontend recomputes bbPlus under filters and
    # must mirror the same scale; sd/ct/hitterPlus are server-precomputed
    # pass-through. No medians used — Wally's rule.
    def _all_mlb_pa_weighted_mean(_stat):
        _pairs = []
        for _r in hitter_leaderboard:
            if _r.get('_isROC') or _r.get('_isCombined'):
                continue
            _v = _r.get(_stat)
            if _v is None:
                continue
            _w = _r.get('pa') or 0
            if _w > 0:
                _pairs.append((_v, _w))
        if not _pairs:
            return None
        _wsum = sum(_w for _, _w in _pairs)
        return sum(_v * _w for _v, _w in _pairs) / _wsum if _wsum > 0 else None

    plus_reanchor = {}
    for _stat in ('bbPlus', 'sdPlus', 'ctPlus'):
        _mean = _all_mlb_pa_weighted_mean(_stat)
        if _mean and abs(_mean) > 1e-9:
            _f = 100.0 / _mean
            plus_reanchor[_stat] = round(_f, 6)
            for _r in hitter_leaderboard:
                if _r.get(_stat) is not None:
                    _r[_stat] = round(_r[_stat] * _f, 1)
            # Keep the Hitter+ standardization metadata consistent with the
            # re-anchored component scale (mean & sd scale by the factor).
            _sd_meta = hitter_plus_standardization.get(_stat)
            if _sd_meta:
                _sd_meta['mean'] = round(_sd_meta['mean'] * _f, 3)
                _sd_meta['sd'] = round(_sd_meta['sd'] * _f, 3)
    _mean_h = _all_mlb_pa_weighted_mean('hitterPlus')
    if _mean_h is not None:
        _shift = 100.0 - _mean_h
        plus_reanchor['hitterPlusShift'] = round(_shift, 4)
        for _r in hitter_leaderboard:
            if _r.get('hitterPlus') is not None:
                _r['hitterPlus'] = round(_r['hitterPlus'] + _shift, 1)
    # 100 = PA-weighted mean of ALL MLB hitters (FG/Savant convention).
    hitter_league_avgs['bbPlus'] = 100.0
    hitter_league_avgs['sdPlus'] = 100.0
    hitter_league_avgs['ctPlus'] = 100.0
    hitter_league_avgs['hitterPlus'] = 100.0
    print(f"  Plus re-anchor (all-MLB PA-weighted mean -> 100): {plus_reanchor}")

    # --- Metadata ---
    metadata = {
        'teams': all_teams,
        'pitchTypes': all_pitch_types,
        'generatedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'totalPitches': len([p for p in all_pitches if p.get('_source', 'MLB') == 'MLB']),
        'totalPitchers': len(pitcher_lb_mlb),
        'totalHitters': len(hitter_lb_mlb),
        'rocTeams': sorted(AAA_TEAMS),
        'leagueAverages': league_avgs,
        'pitcherLeagueAverages': pitcher_league_avgs,
        'hitterLeagueAverages': hitter_league_avgs,
        'vaaRegressions': {pt: {'slope': round(r['slope'], 6),
                                  'leagueAvgPlateZ': round(r['leagueAvgPlateZ'], 6)}
                           for pt, r in vaa_regressions.items()},
        'haaRegressions': {pt: {'slope': round(r['slope'], 6),
                                  'leagueAvgPlateX': {h: round(v, 6)
                                                      for h, v in r['leagueAvgPlateX'].items()}}
                           for pt, r in haa_regressions.items()},
        'sacqZones': sacq_zones_output,
        'sacqLaZones': sacq_la_zones_output,
        'mvnModels': {
            key: {
                variant: {
                    'mu': [round(v, 6) for v in model['mu']],
                    'cov': [[round(v, 6) for v in row] for row in model['cov']],
                    'n': model['n']
                }
                for variant, model in sub.items()
            }
            for key, sub in mvn_models.items()
        },
        'teamGamesPlayed': team_games_played,
        'sdPlusWeights': sd_weights,
        'ctPlusWeights': ct_weights,
        'locPlusWeights': loc_weights,
        'hitterPlusStandardization': hitter_plus_standardization,
        'plusReanchor': plus_reanchor,
        # BB+ component weights + shrinkage — single source of truth, read by
        # js/aggregator.js so the client recompute can never drift from the
        # server again (the 0.6/0.4 → 0.585/0.415 desync shipped for months).
        'bbPlusWeights': {'con': BB_PLUS_W_CON, 'sp': BB_PLUS_W_SP},
        'bbPlusShrinkN0': BB_PLUS_SHRINK_N0,
        'bbPlusMinBip': BB_PLUS_MIN_BIP,
        # Tier 2: 3D xwOBA table used to fill ROC BIP xwOBA (no Savant
        # per-pitch xwOBA available for AAA). For transparency / audit.
        'xwOBA3DTable': (__import__('pipeline_xwoba3d').serialize_table(_xw3d_smoothed_table)
                         if _xw3d_smoothed_table else {}),
    }

    # --- Cache pitch-level data for downstream per-pitch analysis (SD+,
    # train_stuff_v11, Cards, refresh_micro_grades). Written HERE — after the
    # CF remap and every other in-place normalization — so the cache is the
    # EXACT input generate_micro_data sees; scripts/refresh_micro_grades.py
    # depends on that fidelity to rebuild micro data bit-identically.
    cache_path = os.path.join(DATA_DIR, f'all_pitches_{label.lower()}_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(all_pitches, f)
    print(f"  Cached {len(all_pitches)} pitches to {cache_path}")

    # --- Generate micro-aggregate data ---
    # Grade-atom sources: the Loc+ dump was written earlier THIS run (fresh);
    # the Stuff+ dump is the previous train_stuff_v11 run's (stale for any
    # games newer than it). scripts/refresh_micro_grades.py re-runs this
    # generation after the train step so the embedded micro data never lags.
    print(f"\n--- Generating micro-aggregate data ({label}) ---")
    _stuff_grades, _loc_grades = {}, {}
    _sgp = os.path.join(DATA_DIR, 'pitch_stuff_grades.json')
    if os.path.exists(_sgp):
        with open(_sgp) as _f:
            _stuff_grades = json.load(_f)
    _lgp = os.path.join(DATA_DIR, f'pitch_loc_grades_{label.lower()}.json')
    if os.path.exists(_lgp):
        with open(_lgp) as _f:
            _loc_grades = json.load(_f)
    micro_data = generate_micro_data(all_pitches, mlb_id_cache=mlb_id_cache,
                                     ep_pitchers=ep_pitchers,
                                     stuff_grades=_stuff_grades,
                                     loc_grades=_loc_grades)
    print(f"  micro_data: {len(micro_data['pitcherMicro'])} pitcher, "
          f"{len(micro_data['pitchMicro'])} pitch, "
          f"{len(micro_data['hitterMicro'])} hitter micro-aggs, "
          f"{len(micro_data['pitcherBip'])} pitcher BIP, "
          f"{len(micro_data['hitterBip'])} hitter BIP records")

    # --- Boxscore Data: G, GS, IP, W, L, SV, HLD, TBF, ERA, HR/9 for pitchers; G, PA, AB, TB, SB, CS for hitters ---
    mlb_game_dates = sorted(set(normalize_date(p.get('Game Date')) for p in all_pitches
                                if normalize_date(p.get('Game Date')) and p.get('_source', 'MLB') == 'MLB'))
    if mlb_game_dates:
        print(f"\n--- Fetching boxscore data ({label}) ---")
        pitcher_box, hitter_box, pitcher_id_map, hitter_id_map = fetch_and_aggregate_boxscores(mlb_game_dates)
        print(f"  Boxscore pitchers: {len(pitcher_box)}, hitters: {len(hitter_box)}")

        # Fetch MiLB boxscores for AAA teams (ROC, etc.)
        for milb_team in sorted(AAA_TEAMS):
            milb_dates = sorted(set(normalize_date(p.get('Game Date')) for p in all_pitches
                                    if normalize_date(p.get('Game Date')) and p.get('_source') in (milb_team, 'AAA')))
            if milb_dates:
                print(f"\n--- Fetching MiLB boxscore data for {milb_team} ({label}) ---")
                mp, mh, mpi, mhi = fetch_and_aggregate_milb_boxscores(milb_dates, milb_team)
                print(f"  MiLB boxscore {milb_team}: {len(mp)} pitchers, {len(mh)} hitters")
                pitcher_box.update(mp)
                hitter_box.update(mh)
                pitcher_id_map.update(mpi)
                hitter_id_map.update(mhi)

        # Synthesize combined (2TM/3TM) boxscore entries by summing per-team entries,
        # so the per-row merge below works uniformly. Boxscore dicts hold only summable
        # integers (g, gs, outs, er, hr, w, l, sv, hld, tbf, etc.).
        def _get_box(box_dict, id_map, name, team, mlb_id):
            # mlbId|team primary (variation-proof); name|team is the
            # fallback for boxscore records that resolved no MLB ID.
            b = box_dict.get(box_key(name, team, mlb_id))
            if not b:
                b = box_dict.get(f"{name}|{team}")
            if not b and mlb_id:
                alt = id_map.get(mlb_id)
                if alt:
                    b = box_dict.get(alt)
            return b

        for (pitcher, throws), combined_team in combined_pitcher_labels.items():
            mlb_id = mlb_id_cache.get(f"{pitcher}|{combined_team}")
            per_team_boxes = []
            for t in pitcher_mlb_teams[(pitcher, throws)]:
                b = _get_box(pitcher_box, pitcher_id_map, pitcher, t, mlb_id)
                if b:
                    per_team_boxes.append(b)
            if per_team_boxes:
                combined_box = {}
                keys = set()
                for b in per_team_boxes:
                    keys.update(b.keys())
                for k in keys:
                    vals = [b.get(k, 0) or 0 for b in per_team_boxes]
                    if all(isinstance(v, (int, float)) for v in vals):
                        combined_box[k] = sum(vals)
                    else:
                        combined_box[k] = per_team_boxes[0].get(k)
                pitcher_box[box_key(pitcher, combined_team, mlb_id)] = combined_box

        for batter, combined_team in combined_hitter_labels.items():
            mlb_id = mlb_id_cache.get(f"{batter}|{combined_team}")
            per_team_boxes = []
            for t in hitter_mlb_teams[batter]:
                b = _get_box(hitter_box, hitter_id_map, batter, t, mlb_id)
                if b:
                    per_team_boxes.append(b)
            if per_team_boxes:
                combined_box = {}
                keys = set()
                for b in per_team_boxes:
                    keys.update(b.keys())
                for k in keys:
                    vals = [b.get(k, 0) or 0 for b in per_team_boxes]
                    if all(isinstance(v, (int, float)) for v in vals):
                        combined_box[k] = sum(vals)
                    else:
                        combined_box[k] = per_team_boxes[0].get(k)
                hitter_box[box_key(batter, combined_team, mlb_id)] = combined_box

        # Merge pitcher boxscore stats. Primary key is mlbId|team (immune
        # to name-spelling variation); name|team + the id_map are
        # fallbacks for records with no resolved MLB ID.
        for row in pitcher_leaderboard:
            box = pitcher_box.get(box_key(row['pitcher'], row['team'], row.get('mlbId')))
            if not box:
                box = pitcher_box.get(row['pitcher'] + '|' + row['team'])
            if not box and row.get('mlbId'):
                alt_key = pitcher_id_map.get(row['mlbId'])
                if alt_key:
                    box = pitcher_box.get(alt_key)
            if box:
                row['g'] = box['g']
                row['gs'] = box['gs']
                row['ip'] = outs_to_ip_str(box['outs'])
                row['w'] = box['w']
                row['l'] = box['l']
                row['sv'] = box['sv']
                row['hld'] = box['hld']
                row['tbf'] = box['tbf']  # Override pitch-data TBF with official boxscore TBF
                ip_float = outs_to_ip_float(box['outs'])
                row['era'] = round(box['er'] * 9 / ip_float, 2) if ip_float > 0 else None
                row['hr9'] = round(box['hr'] * 9 / ip_float, 2) if ip_float > 0 else None
                row['_box_er'] = box['er']  # raw ER for league avg calc (includes 0-IP pitchers)
                # Store raw boxscore counts for FIP/xFIP/SIERA (computed below)
                row['_box'] = box
            else:
                row['g'] = None
                row['gs'] = None
                row['ip'] = None
                row['w'] = None
                row['l'] = None
                row['sv'] = None
                row['hld'] = None
                row['era'] = None
                row['hr9'] = None

        # Merge hitter boxscore stats. Primary key is mlbId|team (immune
        # to name-spelling variation); name|team + the id_map are
        # fallbacks for records with no resolved MLB ID.
        for row in hitter_leaderboard:
            box = hitter_box.get(box_key(row['hitter'], row['team'], row.get('mlbId')))
            if not box:
                box = hitter_box.get(row['hitter'] + '|' + row['team'])
            if not box and row.get('mlbId'):
                alt_key = hitter_id_map.get(row['mlbId'])
                if alt_key:
                    box = hitter_box.get(alt_key)
            if box:
                row['g'] = box['g']
                row['pa'] = box['pa']  # Override with official PA
                row['ab'] = box['ab']  # Override with official AB
                row['tb'] = box['tb']
                row['sb'] = box['sb']
                row['cs'] = box['cs']
                total_attempts = box['sb'] + box['cs']
                row['sbPct'] = round(box['sb'] / total_attempts * 100, 1) if total_attempts > 0 else None

                # Recompute batting stats using boxscore counts (fixes IBB not in pitch data)
                box_h = box.get('h', 0)
                box_bb = box.get('bb', 0)  # includes IBB
                box_ibb = box.get('ibb', 0)
                box_hbp = box.get('hbp', 0)
                box_sf = box.get('sacFlies', 0)
                box_ab = box['ab']
                box_pa = box['pa']
                box_hr = box.get('hr', 0)
                box_2b = box.get('doubles', 0)
                box_3b = box.get('triples', 0)
                box_1b = max(0, box_h - box_2b - box_3b - box_hr)
                box_tb = box['tb']
                box_so = box.get('so', 0)

                # AVG, OBP, SLG, OPS
                row['avg'] = round(box_h / box_ab, 3) if box_ab > 0 else None
                obp_denom = box_ab + box_bb + box_hbp + box_sf
                row['obp'] = round((box_h + box_bb + box_hbp) / obp_denom, 3) if obp_denom > 0 else None
                row['slg'] = round(box_tb / box_ab, 3) if box_ab > 0 else None
                row['ops'] = round(row['obp'] + row['slg'], 3) if row['obp'] is not None and row['slg'] is not None else None
                row['iso'] = round(row['slg'] - row['avg'], 3) if row['slg'] is not None and row['avg'] is not None else None

                # Doubles, triples, HR, XBH from boxscore
                row['doubles'] = box_2b
                row['triples'] = box_3b
                row['hr'] = box_hr
                row['xbh'] = box_2b + box_3b + box_hr

                # K% and BB%. Hitter BB% uses TOTAL walks (incl. IBB), matching
                # the client re-aggregation. (wOBA below still uses uBB, since IBB
                # carries no wOBA weight.) Pitchers, by contrast, use uBB.
                box_ubb = box_bb - box_ibb
                row['kPct'] = round(box_so / box_pa, 4) if box_pa > 0 else None
                row['bbPct'] = round(box_bb / box_pa, 4) if box_pa > 0 else None
                # BB/K uses the same total-walk numerator as bbPct. Stored at full
                # precision; display layer rounds to 2 decimals at render time.
                row['bbToK'] = (box_bb / box_so) if box_so > 0 else None

                # BABIP = (H - HR) / (AB - K - HR + SF)
                babip_denom = box_ab - box_so - box_hr + box_sf
                row['babip'] = round((box_h - box_hr) / babip_denom, 3) if babip_denom > 0 else None

                # wOBA from boxscore counts + FanGraphs Guts weights
                if WOBA_WEIGHTS:
                    woba_denom = box_ab + box_ubb + box_sf + box_hbp
                    if woba_denom > 0:
                        woba_num = (WOBA_WEIGHTS['BB'] * box_ubb + WOBA_WEIGHTS['HBP'] * box_hbp +
                                    WOBA_WEIGHTS['1B'] * box_1b + WOBA_WEIGHTS['2B'] * box_2b +
                                    WOBA_WEIGHTS['3B'] * box_3b + WOBA_WEIGHTS['HR'] * box_hr)
                        row['wOBA'] = round(woba_num / woba_denom, 3)
                    else:
                        row['wOBA'] = None
            else:
                row['g'] = None
                row['tb'] = None
                row['sb'] = None
                row['cs'] = None
                row['sbPct'] = None

    # Compute wRC and wRC+ for each hitter (after boxscore merge so wOBA is from official stats)
    # wRC  = (((wOBA - lgWOBA) / wOBAScale) + lgRPA) * PA
    # wRC+ = ((wRAA/PA + lgRPA) + (lgRPA - PF * lgRPA)) / lgR/PA * 100
    if GUTS_EXTRA:
        woba_scale = GUTS_EXTRA['wOBAScale']
        lg_woba = GUTS_EXTRA['lgWOBA']
        lg_rpa = GUTS_EXTRA['lgRPA']
        park_factors = PARK_FACTORS or {}
        for row in hitter_leaderboard:
            woba = row.get('wOBA')
            pa = row.get('pa') or 0
            if woba is not None and pa > 0 and woba_scale > 0:
                wraa_per_pa = (woba - lg_woba) / woba_scale
                row['wRC'] = round((wraa_per_pa + lg_rpa) * pa, 2)
                # wRC+
                pf = park_factors.get(row['team'], 1.0)
                numerator = wraa_per_pa + lg_rpa + (lg_rpa - pf * lg_rpa)
                if lg_rpa > 0:
                    row['wRCplus'] = round(numerator / lg_rpa * 100)
                else:
                    row['wRCplus'] = None
                # xWRC+ (same formula but using xwOBA instead of wOBA).
                # NO park factor here: xwOBA is an EV/LA model output mapped
                # to league-average-park outcomes, so it doesn't contain the
                # park effect wRC+'s PF term divides out — applying PF to it
                # over-corrected (e.g. docked Rockies hitters twice).
                xwoba = row.get('xwOBA')
                if xwoba is not None:
                    xwraa_per_pa = (xwoba - lg_woba) / woba_scale
                    xnumerator = xwraa_per_pa + lg_rpa
                    row['xWRCplus'] = round(xnumerator / lg_rpa * 100) if lg_rpa > 0 else None
                else:
                    row['xWRCplus'] = None
            else:
                row['wRC'] = None
                row['wRCplus'] = None
                row['xWRCplus'] = None

    # FanGraphs override: replace our pipeline-computed wRC+ with canonical
    # FG values. FG has slightly different park-factor / wOBA-weight tuning
    # and intermediate precision that produces small but visible deltas
    # (e.g. Wood reads wRC+ 151 here vs 152 on FG). Pulling FG's number
    # keeps the card aligned with fangraphs.com.
    #
    # - wRC+: overridden for both MLB and AAA hitters. AAA gap is large
    #   (~13-19 pts) because our pipeline applies MLB constants to AAA
    #   data; FG uses AAA-baseline weights + IL/PCL park factors.
    # - xwOBA / xBA / xSLG: NOT overridden (dropped in commit eab9655) —
    #   the pipeline's Statcast per-pitch values are the source of truth.
    # - wOBA / AVG / OBP / SLG / BABIP / OPS / ISO: NOT overridden —
    #   pipeline matches FG to within ±0.0005 (rounding noise), so the
    #   override would be cosmetically identical to the pipeline value.
    try:
        from fg_overrides import refresh_if_stale as _fg_refresh
        _fg = _fg_refresh(max_age_hours=24, verbose=True)
        _fg_mlb_h = _fg.get('mlbHitters', {})
        _fg_aaa_h = _fg.get('aaaHitters', {})
        n_mlb_wrc = n_mlb = 0
        n_aaa_wrc = n_aaa = 0
        for row in hitter_leaderboard:
            mid = row.get('mlbId')
            if mid is None:
                continue
            mid_str = str(int(mid))
            if row.get('_isROC'):
                n_aaa += 1
                fg_player = _fg_aaa_h.get(mid_str)
                if fg_player and fg_player.get('wRCplus') is not None:
                    row['wRCplus'] = fg_player['wRCplus']
                    n_aaa_wrc += 1
            else:
                n_mlb += 1
                fg_player = _fg_mlb_h.get(mid_str)
                if fg_player:
                    if fg_player.get('wRCplus') is not None:
                        row['wRCplus'] = fg_player['wRCplus']
                        n_mlb_wrc += 1
                    # xBA/xSLG/xwOBA are NOT overridden with FanGraphs — keep the
                    # pipeline's Statcast-computed values so they stay consistent
                    # with xwOBAcon and with the website (which re-aggregates from
                    # the same Statcast micro). wRC+ has no pipeline equivalent, so
                    # it still comes from FanGraphs.
        print(f"  FG hitter override: wRC+ {n_mlb_wrc}/{n_mlb} MLB + {n_aaa_wrc}/{n_aaa} AAA")
    except Exception as _e:
        print(f"  WARNING: FG hitter override failed ({type(_e).__name__}: {_e})")

    # Pass 2: refresh hitter league averages for stats populated by the boxscore
    # merge + wRC+ (kPct, bbPct, avg, obp, slg, ops, iso, wRCplus, xWRCplus). The
    # first pass above runs before the boxscore merge, so these are None on every
    # row at that point. Fill in anything still missing; leave the plus-metrics
    # (bbPlus/pdPlus/hitterPlus = 100) and already-computed avgs alone.
    #
    # wOBA is special-cased: pass 1 DID compute it (pitch-derived), but the
    # boxscore merge then overwrites every player's displayed wOBA with the
    # official-stats version — so without a recompute the League Avg row
    # disagrees with the rows around it (shipped .3162 vs FG's .3169 on
    # 2026-07-13; pitch data structurally misses no-pitch IBBs and any
    # source-lagged PAs). Recompute it here from the merged official values.
    for stat in HITTER_STAT_KEYS:
        if stat != 'wOBA' and hitter_league_avgs.get(stat) is not None:
            continue
        _compute_hitter_lg_avg(stat)

    # ── Hitter+ → wRC+ scale match (2026-07-13) ──
    # The composite's SD over the qualified pool is a free display parameter
    # (the exact-SD rescale in the Hitter+ block above targets 40). Re-scale
    # it here — after the FG override has populated wRC+ — to the SAME
    # pool's measured wRC+ SD, so Hitter+ and wRC+ speak one currency and a
    # gap between them reads directly as "process ahead of/behind results".
    # Dynamic rather than hard-coded because wRC+'s spread shrinks as PAs
    # accumulate (~23 mid-season, ~20 full-season); a fixed target drifts.
    # Linear around 100: ranks, percentiles, and coloring are unchanged, and
    # the PA-weighted mean stays pinned at 100 by the reanchor. If wRC+
    # didn't populate (FG override failure), the SD-40 scale stands.
    _pool_hp, _pool_wrc = [], []
    for _row in hitter_leaderboard:
        if _row.get('_isROC') or _row.get('_isCombined'):
            continue
        _team_g = team_games_played.get(_row.get('team'))
        if _team_g is None and team_games_played:
            _team_g = max(team_games_played.values())
        if not _team_g or _row.get('pa', 0) < _hitter_pa_per_game(False) * _team_g:
            continue
        if _row.get('hitterPlus') is None or _row.get('wRCplus') is None:
            continue
        _pool_hp.append(_row['hitterPlus'])
        _pool_wrc.append(_row['wRCplus'])
    if len(_pool_hp) >= 10:
        def _psd(vals):
            m = sum(vals) / len(vals)
            return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
        _sd_hp, _sd_wrc = _psd(_pool_hp), _psd(_pool_wrc)
        if _sd_hp > 1e-9 and _sd_wrc > 1e-9:
            _f = _sd_wrc / _sd_hp
            for _row in hitter_leaderboard:
                if _row.get('hitterPlus') is not None:
                    _row['hitterPlus'] = round(100.0 + (_row['hitterPlus'] - 100.0) * _f, 1)
            hitter_plus_standardization['wrcScaleMatch'] = {
                'poolWrcSd': round(_sd_wrc, 3), 'poolHpSd': round(_sd_hp, 3),
                'factor': round(_f, 4), 'n': len(_pool_hp)}
            print(f"  Hitter+ rescaled to the wRC+ spread (pool wRC+ SD {_sd_wrc:.1f}, "
                  f"factor {_f:.3f}, n={len(_pool_hp)}).")
    else:
        print("  Hitter+ wRC+ scale match skipped (wRC+ pool too small) — SD-40 scale stands.")

    # Compute total ER and outs for league ERA (needed for SIERA constant calibration)
    # Use ALL MLB pitchers from boxscore data (including EP pitchers excluded from leaderboard)
    # Exclude MiLB teams from league-wide calculations
    total_outs = 0
    total_er = 0
    for bkey, box in pitcher_box.items():
        # bkey format: "<id-or-name>|TEAM" — team is the last segment.
        box_team = bkey.split('|')[-1] if '|' in bkey else ''
        # Skip AAA and synthesized 2TM/3TM combined entries. Combined boxes are
        # the element-wise sum of the per-team boxes, so counting them alongside
        # the per-team entries would double-count traded pitchers' outs/ER.
        if box_team in AAA_TEAMS or _is_combined_team(box_team):
            continue
        total_outs += box.get('outs', 0)
        total_er += box.get('er', 0)

    # --- Compute FIP, xFIP, SIERA ---
    # FIP_CONSTANT and WOBA_WEIGHTS are set globally from FanGraphs Guts page

    # Compute league HR/FB% for xFIP. FB includes popups (fly_ball + popup).
    # Numerator (HR) and denominator (FB) must come from the SAME population — the
    # non-ROC, non-combined leaderboard pitchers — so EP (position-player) pitchers,
    # who contribute HR-allowed via their boxscore but no tracked fly balls, don't
    # inflate the ratio.
    total_hr_lg = 0
    total_fb_lg = 0
    for row in pitcher_leaderboard:
        if row.get('_isROC') or row.get('_isCombined'):
            continue
        box = row.get('_box')
        if box:
            total_hr_lg += box.get('hr', 0)
        n_bip = row.get('nBip', 0) or 0
        if n_bip > 0:
            fb_pct = row.get('fbPct') or 0
            pu_pct = row.get('puPct') or 0
            total_fb_lg += round((fb_pct + pu_pct) * n_bip)
    lg_hr_fb = total_hr_lg / total_fb_lg if total_fb_lg > 0 else 0.105  # fallback to historical avg
    print(f"  League HR/FB%: {lg_hr_fb:.3f} ({total_hr_lg} HR / {total_fb_lg} FB+PU)")

    # First pass: compute FIP, xFIP, and raw SIERA (without constant) for each pitcher
    siera_ip_pairs = []  # (raw_siera, ip_float) for constant calibration
    for row in pitcher_leaderboard:
        box = row.get('_box')
        if not box:
            row['fip'] = None
            row['xFIP'] = None
            row['_siera_raw'] = None
            continue

        ip_float = outs_to_ip_float(box['outs'])
        hr = box['hr']
        bb = box['bb']
        hbp = box['hbp']
        so = box['so']
        tbf = box['tbf']

        # FIP = ((13*HR)+(3*(BB+HBP))-(2*K))/IP + constant
        if ip_float > 0 and FIP_CONSTANT is not None:
            row['fip'] = round(((13 * hr + 3 * (bb + hbp) - 2 * so) / ip_float) + FIP_CONSTANT, 2)
        else:
            row['fip'] = None

        # xFIP: FB includes popups
        n_bip = row.get('nBip', 0) or 0
        fb_pct = row.get('fbPct') or 0
        pu_pct = row.get('puPct') or 0
        fb_count = round((fb_pct + pu_pct) * n_bip)  # fly balls + popups
        if ip_float > 0 and FIP_CONSTANT is not None:
            expected_hr = fb_count * lg_hr_fb
            row['xFIP'] = round(((13 * expected_hr + 3 * (bb + hbp) - 2 * so) / ip_float) + FIP_CONSTANT, 2)
        else:
            row['xFIP'] = None

        # SIERA (raw, without constant — constant calibrated below)
        # netGB = GB - FB (where FB includes popups)
        # -/+ 4.920 term: minus if GB >= FB, plus if FB > GB
        gb_pct_val = row.get('gbPct') or 0
        gb_count = round(gb_pct_val * n_bip)
        if tbf > 0 and ip_float > 0:
            so_pa = so / tbf
            bb_pa = bb / tbf
            net_gb_pa = (gb_count - fb_count) / tbf
            # SP/RP ratio: fraction of IP as starter
            gs = box.get('gs', 0) or 0
            g = box.get('g', 1) or 1
            ip_sp_ratio = min(gs / g, 1.0) if g > 0 else 0.0
            # Sign for 4.920 term: minus if GB >= FB, plus if FB > GB
            sign_4920 = -1.0 if gb_count >= fb_count else 1.0
            raw_siera = (
                - 15.518 * so_pa
                + 9.146 * (so_pa ** 2)
                + 8.648 * bb_pa
                + 27.252 * (bb_pa ** 2)
                - 2.298 * net_gb_pa
                + sign_4920 * 4.920 * (net_gb_pa ** 2)
                - 4.036 * so_pa * bb_pa
                + 5.155 * so_pa * net_gb_pa
                + 4.546 * bb_pa * net_gb_pa
                + 0.367 * ip_sp_ratio
            )
            row['_siera_raw'] = raw_siera
            if not row.get('_isROC') and not row.get('_isCombined'):
                siera_ip_pairs.append((raw_siera, ip_float))
        else:
            row['_siera_raw'] = None

    # Calibrate SIERA constant so league-average SIERA = league-average ERA
    # (same principle as cFIP for FIP)
    if siera_ip_pairs and total_outs > 0:
        total_ip_siera = sum(ip for _, ip in siera_ip_pairs)
        weighted_raw = sum(raw * ip for raw, ip in siera_ip_pairs) / total_ip_siera if total_ip_siera > 0 else 0
        league_era = total_er * 9 / (total_outs / 3.0) if total_outs > 0 else 4.00
        siera_constant = league_era - weighted_raw
    else:
        siera_constant = 5.77  # fallback
    print(f"  SIERA constant: {siera_constant:.3f}")
    metadata['sieraConstant'] = round(siera_constant, 4)

    # Persist live FanGraphs Guts constants so downstream tools (Cards.py)
    # can use the same values that compute_xrv used here, instead of drifting
    # against hardcoded fallbacks.
    if GUTS_EXTRA:
        metadata['gutsConstants'] = {
            'lgWOBA': GUTS_EXTRA.get('lgWOBA'),
            'wOBAScale': GUTS_EXTRA.get('wOBAScale'),
            'lgRPA': GUTS_EXTRA.get('lgRPA'),
        }

    # Second pass: apply SIERA constant and clean up
    for row in pitcher_leaderboard:
        if row.get('_siera_raw') is not None:
            row['siera'] = round(row['_siera_raw'] + siera_constant, 2)
        else:
            row['siera'] = None
        row.pop('_siera_raw', None)
        row.pop('_box', None)

    # Compute ERA league average (total_outs and total_er computed above)
    if total_outs > 0:
        total_ip = total_outs / 3.0
        metadata['pitcherLeagueAverages']['era'] = round(total_er * 9 / total_ip, 2)

    # HR/9 league average — weighted by IP (MLB only, exclude combined rows)
    hr9_pairs = [(r['hr9'], ip_str_to_float(r.get('ip'))) for r in pitcher_leaderboard
                 if r.get('hr9') is not None and r.get('ip') is not None and ip_str_to_float(r['ip']) > 0
                 and not r.get('_isROC') and not r.get('_isCombined')]
    if hr9_pairs:
        total_w = sum(w for _, w in hr9_pairs)
        metadata['pitcherLeagueAverages']['hr9'] = round(sum(v * w for v, w in hr9_pairs) / total_w, 2) if total_w > 0 else None

    # FanGraphs override: replace pipeline-computed FIP / xFIP / SIERA with
    # the canonical FG values for MLB pitchers. Same motivation as the
    # hitter wRC+ override above — pipeline values match FG approximately
    # but small precision/rounding deltas read as bugs when readers
    # cross-reference. AAA pitchers (_isROC) keep the pipeline values
    # since FG doesn't publish AAA-baseline FIP/xFIP/SIERA cleanly.
    try:
        from fg_overrides import refresh_if_stale as _fg_refresh_pit
        _fg_pit_cache = _fg_refresh_pit(max_age_hours=24, verbose=False)
        _fg_pit = _fg_pit_cache.get('mlbPitchers', {})
        n_pit_replaced = n_pit = 0
        for row in pitcher_leaderboard:
            if row.get('_isROC') or row.get('_isCombined'):
                continue
            mid = row.get('mlbId')
            if mid is None:
                continue
            fg_p = _fg_pit.get(str(int(mid)))
            if not fg_p:
                continue
            n_pit += 1
            changed = False
            if fg_p.get('fip') is not None:
                row['fip'] = fg_p['fip']
                changed = True
            if fg_p.get('xfip') is not None:
                row['xFIP'] = fg_p['xfip']
                changed = True
            if fg_p.get('siera') is not None:
                row['siera'] = fg_p['siera']
                changed = True
            if changed:
                n_pit_replaced += 1
        print(f"  FG FIP/xFIP/SIERA override: replaced "
              f"{n_pit_replaced}/{n_pit} MLB pitchers with FanGraphs values")
    except Exception as _e:
        print(f"  WARNING: FG pitcher override failed ({type(_e).__name__}: {_e})")

    # FIP, xFIP, SIERA league averages — weighted by IP (MLB only, exclude combined rows)
    # Computed AFTER the FG override so the league average reflects the
    # canonical values that ship in the JSON.
    for stat in ['fip', 'xFIP', 'siera']:
        pairs = [(r[stat], ip_str_to_float(r.get('ip'))) for r in pitcher_leaderboard
                 if r.get(stat) is not None and r.get('ip') is not None and ip_str_to_float(r['ip']) > 0
                 and not r.get('_isROC') and not r.get('_isCombined')]
        if pairs:
            total_w = sum(w for _, w in pairs)
            metadata['pitcherLeagueAverages'][stat] = round(sum(v * w for v, w in pairs) / total_w, 2) if total_w > 0 else None

    # ==========================================================
    # CONSOLIDATED PERCENTILE COMPUTATION
    # All stats are now computed, all boxscore merges done, all derived stats (FIP, wRC+, etc.) set.
    # Compute all percentiles in a single pass, then apply all inversions.
    # ==========================================================
    print("\n--- Computing percentiles (single pass) ---")

    # ── Qualified-pool helpers ──
    # Rate-stat percentile distributions are defined ONLY by qualified players
    # (3.1 PA × team_games for hitters; IP × team_games role-adjusted for
    # pitchers). All rows still get a percentile rank stored — non-qualified
    # rows have ranks for tooltip display but are not colored at render time.
    # Counting stats (hr, sb) keep the unfiltered pool.
    # Canonical qualification — ROC-aware via the shared pipeline_utils
    # helpers (MLB hitter 3.1 PA×TG, ROC 2.7; MLB SP 1.0 IP×TG / RP 0.5,
    # ROC SP 0.8 / RP 0.4). NOTE: compute_percentile_ranks_with_aaa routes
    # ROC rows to interpolation BEFORE qualifier_fn is ever called, so the
    # MLB pool is unaffected by the ROC branch here — the ROC-aware code
    # is kept for correctness/consistency with the frontend.
    from pipeline_utils import (
        hitter_pa_per_game, pitcher_ip_per_game, SP_GS_RATIO,
    )

    def _hitter_qualified_for_pctl(row):
        pa = row.get('pa', 0) or 0
        tg = team_games_played.get(row.get('team'))
        if tg is None and team_games_played:
            tg = max(team_games_played.values())
        if not tg:
            return False
        return pa >= hitter_pa_per_game(bool(row.get('_isROC'))) * tg

    def _pitcher_qualified_for_pctl(row):
        ip_str = row.get('ip')
        ip_f = ip_str_to_float(ip_str) if ip_str is not None else 0
        tg = team_games_played.get(row.get('team'))
        if tg is None and team_games_played:
            tg = max(team_games_played.values())
        if not tg:
            return False
        g = row.get('g') or 0
        gs = row.get('gs') or 0
        is_starter = g > 0 and (gs / g) > SP_GS_RATIO
        per_game = pitcher_ip_per_game(is_starter, bool(row.get('_isROC')))
        return ip_f >= tg * per_game

    # Hitter counting stats — keep unfiltered pool, do not apply qualifier_fn.
    HITTER_COUNTING_PCTL = {'hr', 'sb'}

    # Pitch-type outcome stats — non-shape per-pitch stats use min_count=25
    # so pitches thrown rarely (e.g., 5 sliders) don't pollute the per-pitch
    # percentile pool. Shape metrics (velo, IVB, HB, etc.) need no minimum.
    MIN_PITCH_TYPE_OUTCOME = 25
    # Loc+ is the one exception: it's displayed unshrunk, so its gate is the
    # pitch type's own measured split-half r=0.5 crossing, not the flat 25.
    # Mirrored in js/aggregator.js QUAL.MIN_PITCH_LOCPLUS.
    from pipeline_locplus import stabilize_n as locplus_min_pitch
    PITCH_SHAPE_KEYS = set(METRIC_KEYS.values()) | {'nVAA', 'nHAA', 'ivbOE', 'hbOE'}
    # Hand-signed shape metrics stored in an absolute frame — rank on |value| so
    # LHP/RHP aren't split by sign. Mirrors js/aggregator.js ABS_PCTL_KEYS.
    ABS_PCTL_KEYS = {'horzBrk', 'haa', 'nHAA', 'hbOE'}
    # Batted-ball stats use a BIP-count qualifier (>=25 BIPs of that pitch
    # type) instead of pitch count, since their denominator is BIPs. Includes
    # gbPct (which lives in PITCH_STAT_KEYS for historical reasons but is a
    # BIP-rate stat) and xwOBAsp (which was gated on pitch count before,
    # leaving 52% of colored pitch-type cells with <20 BIP behind them).
    PITCH_BB_QUAL_KEYS = set(PITCH_BB_PCTL_KEYS) | {'gbPct', 'xwOBAsp'}

    # 1. Pitch-type percentiles (grouped by pitch type)
    pt_groups = defaultdict(list)
    for row in pitch_leaderboard:
        pt_groups[row['pitchType']].append(row)
    for metric in PITCH_PCTL_KEYS:
        if metric in PITCH_SHAPE_KEYS:
            mc, ck = 0, 'count'
        elif metric in PITCH_BB_QUAL_KEYS:
            mc, ck = MIN_PITCH_TYPE_OUTCOME, 'nBip'
        else:
            mc, ck = MIN_PITCH_TYPE_OUTCOME, 'count'
        av = metric in ABS_PCTL_KEYS
        for pt, pt_rows in pt_groups.items():
            # Loc+ gates on its own measured stabilization constant rather than
            # the flat 25 (see locplus_min_pitch): displayed Loc+ is unshrunk,
            # so a 25-pitch cell is only ~0.26 reliable.
            mc_pt = locplus_min_pitch(pt) if metric == 'locPlus' else mc
            compute_percentile_ranks_with_aaa(pt_rows, metric, min_count=mc_pt, count_key=ck, abs_val=av)

    # 2. Pitcher percentiles (all stats including boxscore-derived).
    # Pool: ALL MLB pitchers (no qualifier). This matches the convention used
    # for the displayed league average (PA/IP-weighted mean over every MLB
    # pitcher) so "ERA below league avg" reads as "above the 50th percentile"
    # for the reader. Qualification is enforced as a render-only gate: every
    # row still gets a percentile RANK stored (for tooltip + sort), but the
    # leaderboard suppresses cell coloring on non-qualified rows. Matches
    # FanGraphs/Savant — they percentile-rank against the broader pool and
    # only display percentile chips for players who clear sample minimums.
    PITCHER_ALL_PCTL = (STAT_KEYS + PITCHER_METRIC_PCTL_KEYS + PITCHER_BB_KEYS
                        + EXPECTED_KEYS + ['fbVelo', 'runValue', 'rv100', 'xRunValue', 'xRv100', 'era', 'hr9', 'fip', 'xFIP', 'siera', 'locPlus'])
    for stat in PITCHER_ALL_PCTL:
        compute_percentile_ranks_with_aaa(pitcher_leaderboard, stat, min_count=0)

    # 3. Hitter percentiles (all stats including boxscore-derived).
    # Same pool change as pitchers: all MLB hitters define the distribution,
    # qualification is a render-only gate for coloring. Counting stats
    # (hr, sb) were already using the full pool — unchanged.
    for stat in HITTER_STAT_KEYS + EXPECTED_KEYS:
        compute_percentile_ranks_with_aaa(hitter_leaderboard, stat)

    # 4. Hitter pitch-type percentiles (grouped by pitch type) — min 25 pitches
    # of that type; pitch-type-vs-hitter has no PA-style qualifier, just the
    # per-pitch sample-size minimum.
    hpt_groups = defaultdict(list)
    for row in hitter_pitch_leaderboard:
        hpt_groups[row['pitchType']].append(row)
    for pt, pt_rows in hpt_groups.items():
        for stat in HITTER_PITCH_PCTL_KEYS:
            compute_percentile_ranks_with_aaa(pt_rows, stat, min_count=MIN_PITCH_TYPE_OUTCOME)

    # ==========================================================
    # CONSOLIDATED INVERSIONS
    # ==========================================================

    # Pitch inversions: VAA/nVAA for non-fastball, expected stats for all
    VAA_NO_INVERT_TYPES = {'FF', 'FC'}
    for pt, pt_rows in pt_groups.items():
        if pt not in VAA_NO_INVERT_TYPES:
            for row in pt_rows:
                if row.get('vaa_pctl') is not None:
                    row['vaa_pctl'] = 100 - row['vaa_pctl']
                if row.get('nVAA_pctl') is not None:
                    row['nVAA_pctl'] = 100 - row['nVAA_pctl']
        else:
            # HAA/nHAA are abs-ranked (larger magnitude = higher pctl by default).
            # For fastballs, closer to 0 is better, so invert. Mirrors the JS
            # Aggregator HAA/nHAA fastball inversion (js/aggregator.js:1751).
            for row in pt_rows:
                if row.get('haa_pctl') is not None:
                    row['haa_pctl'] = 100 - row['haa_pctl']
                if row.get('nHAA_pctl') is not None:
                    row['nHAA_pctl'] = 100 - row['nHAA_pctl']
    for row in pitch_leaderboard:
        for stat in ('wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp'):
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]
        # Batted-ball stats where lower = better for pitcher.
        for stat in PITCH_BB_INVERT:
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]

    # Pitcher inversions
    PITCHER_ALL_INVERT = PITCHER_INVERT_PCTL | PITCHER_BB_INVERT | EXPECTED_PITCHER_INVERT | {'era', 'hr9', 'fip', 'xFIP', 'siera'}
    for row in pitcher_leaderboard:
        for stat in PITCHER_ALL_INVERT:
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]

    # Hitter inversions
    for row in hitter_leaderboard:
        for stat in HITTER_INVERT_PCTL:
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]

    # Hitter pitch-type inversions
    for row in hitter_pitch_leaderboard:
        for stat in HITTER_PITCH_INVERT_PCTL:
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]

    print("  Percentiles computed and inversions applied.")

    # runValue/rv100/xRunValue/xRv100 are kept at full (float) precision in the
    # JSON output. Display rounding (1 decimal in the leaderboard, 2 decimals on
    # the player page) happens in the JS layer at render time via toFixed().
    # This avoids any intermediate rounding that could shift the displayed value
    # vs rounding from full-precision inputs (e.g., 0.236 + 0.563 = 0.799 → 0.8,
    # never 0.2 + 0.6 = 0.8). Percentile ranks are unaffected: they are computed
    # earlier in the pipeline from exact values regardless of display precision.

    return {
        'pitcher_leaderboard': pitcher_leaderboard,
        'pitch_leaderboard': pitch_leaderboard,
        'hitter_leaderboard': hitter_leaderboard,
        'hitter_pitch_leaderboard': hitter_pitch_leaderboard,
        'metadata': metadata,
        'micro_data': micro_data,
        'pitch_details': pitch_details,
        'hitter_pitch_details': hitter_pitch_details,
        'hitter_swing_locations': hitter_swing_locations,
    }


def round_floats_inplace(obj, ndigits=6):
    """Recursively round every float in a nested list/dict structure.

    The embedded payload stored full IEEE-754 float noise like
    0.6231884057971014 (18 chars) for values the UI only ever renders via
    toFixed at 1-3 decimals. Rounding to 6 decimals strips ~25-30% of the
    file with zero visible effect.

    RV-precision rule check: aggregator.js re-sums per-pitch runValue/
    xRunValue from micro-data. Rounding each per-pitch value to 6 decimals
    bounds the re-aggregation error at 5e-7 per pitch — under 0.001 runs
    across a full season of a pitcher's pitches, ~100x below the 1-decimal
    display precision. So the "sum at full precision, round at display"
    rule is honored in practice (the rule targets premature 1-2 decimal
    rounding that visibly accumulates; 6-decimal noise removal does not).

    Mutates in place and also returns obj for convenience.
    """
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if type(v) is float:
                    cur[k] = round(v, ndigits)
                elif type(v) is list or type(v) is dict:
                    stack.append(v)
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                if type(v) is float:
                    cur[i] = round(v, ndigits)
                elif type(v) is list or type(v) is dict:
                    stack.append(v)
    return obj


def write_json_outputs(result, suffix):
    """Write JSON output files with the given suffix."""
    def strip_internal_keys(rows):
        return [{k: v for k, v in row.items() if not k.startswith('_')} for row in rows]

    # Preserve Stuff+ scores from existing pitch leaderboard (injected by
    # stuff_plus_v11/train_stuff_v11.py --inject; keyed by pitcher/team/pitchType)
    pitch_json_path = os.path.join(DATA_DIR, f'pitch_leaderboard{suffix}.json')
    if os.path.exists(pitch_json_path):
        try:
            with open(pitch_json_path) as f:
                existing = json.load(f)
            stuff_map = {}
            for row in existing:
                if row.get('stuffScore') is not None:
                    key = (row.get('pitcher'), row.get('team'), row.get('pitchType'))
                    stuff_map[key] = {
                        'stuffScore': row['stuffScore'],
                        'stuffScore_pctl': row.get('stuffScore_pctl'),
                    }
                    for f in XRVOE_KEYS:
                        stuff_map[key][f] = row.get(f)
                        stuff_map[key][f + '_pctl'] = row.get(f + '_pctl')
            if stuff_map:
                n_merged = 0
                for row in result['pitch_leaderboard']:
                    key = (row.get('pitcher'), row.get('team'), row.get('pitchType'))
                    if key in stuff_map:
                        row['stuffScore'] = stuff_map[key]['stuffScore']
                        if stuff_map[key]['stuffScore_pctl'] is not None:
                            row['stuffScore_pctl'] = stuff_map[key]['stuffScore_pctl']
                        for f in XRVOE_KEYS:
                            if stuff_map[key].get(f) is not None:
                                row[f] = stuff_map[key][f]
                                row[f + '_pctl'] = stuff_map[key].get(f + '_pctl')
                        n_merged += 1
                print(f"  Preserved Stuff+ scores: {n_merged}/{len(stuff_map)} rows merged")
        except (json.JSONDecodeError, KeyError):
            print("  Warning: could not read existing Stuff+ scores")

    # Preserve OVERALL (per-pitcher) Stuff+ from existing pitcher leaderboard
    pitcher_json_path = os.path.join(DATA_DIR, f'pitcher_leaderboard{suffix}.json')
    if os.path.exists(pitcher_json_path):
        try:
            with open(pitcher_json_path) as f:
                existing = json.load(f)
            stuff_map = {}
            for row in existing:
                if row.get('stuffScore') is not None:
                    key = (row.get('pitcher'), row.get('team'), row.get('throws'))
                    stuff_map[key] = {'stuffScore': row['stuffScore'],
                                      'stuffScore_pctl': row.get('stuffScore_pctl')}
                    for f in XRVOE_KEYS:
                        stuff_map[key][f] = row.get(f)
                        stuff_map[key][f + '_pctl'] = row.get(f + '_pctl')
            if stuff_map:
                n_merged = 0
                for row in result['pitcher_leaderboard']:
                    key = (row.get('pitcher'), row.get('team'), row.get('throws'))
                    if key in stuff_map:
                        row['stuffScore'] = stuff_map[key]['stuffScore']
                        if stuff_map[key]['stuffScore_pctl'] is not None:
                            row['stuffScore_pctl'] = stuff_map[key]['stuffScore_pctl']
                        for f in XRVOE_KEYS:
                            if stuff_map[key].get(f) is not None:
                                row[f] = stuff_map[key][f]
                                row[f + '_pctl'] = stuff_map[key].get(f + '_pctl')
                        n_merged += 1
                print(f"  Preserved overall Stuff+ scores: {n_merged}/{len(stuff_map)} rows merged")
        except (json.JSONDecodeError, KeyError):
            print("  Warning: could not read existing overall Stuff+ scores")

    # Round floats in every committed artifact (these also land in git via
    # `git add data/`, so shrinking them speeds the push too). Mutating
    # result['micro_data'] in place is fine: write_embedded_js runs next and
    # wants the rounded values anyway (rounding is idempotent).
    with open(pitch_json_path, 'w') as f:
        json.dump(round_floats_inplace(strip_internal_keys(result['pitch_leaderboard'])), f)
    with open(os.path.join(DATA_DIR, f'pitcher_leaderboard{suffix}.json'), 'w') as f:
        json.dump(round_floats_inplace(strip_internal_keys(result['pitcher_leaderboard'])), f)
    with open(os.path.join(DATA_DIR, f'hitter_leaderboard{suffix}.json'), 'w') as f:
        json.dump(round_floats_inplace(strip_internal_keys(result['hitter_leaderboard'])), f)
    with open(os.path.join(DATA_DIR, f'hitter_pitch_leaderboard{suffix}.json'), 'w') as f:
        json.dump(round_floats_inplace(strip_internal_keys(result['hitter_pitch_leaderboard'])), f)
    with open(os.path.join(DATA_DIR, f'metadata{suffix}.json'), 'w') as f:
        json.dump(round_floats_inplace(result['metadata']), f, indent=2)
    with open(os.path.join(DATA_DIR, f'micro_data{suffix}.json'), 'w') as f:
        json.dump(round_floats_inplace(result['micro_data']), f, separators=(',', ':'))
    print(f"  Wrote JSON files with suffix '{suffix}'")


def write_embedded_js(rs_result):
    """Write data_embedded.js with window.RS_DATA."""
    def build_data_obj(result):
        # Strip internal _-prefixed keys from all leaderboard rows
        def strip_internal(rows):
            return [{k: v for k, v in row.items() if not k.startswith('_')} for row in rows]
        # Keep _pctl keys on hitter pitch LB rows — needed by the player-page
        # Plate Discipline / Batted Ball tables to color category rows and
        # per-pitch sub-rows (the leaderboard's hitterPitch tab recomputes
        # percentiles client-side via the aggregator, but the player page
        # reads these rows directly).
        hitter_pitch_lb_slim = []
        for row in result['hitter_pitch_leaderboard']:
            slim = {k: v for k, v in row.items() if not k.startswith('_')}
            hitter_pitch_lb_slim.append(slim)
        return {
            'pitcherData': strip_internal(result['pitcher_leaderboard']),
            'pitchData': strip_internal(result['pitch_leaderboard']),
            'hitterData': strip_internal(result['hitter_leaderboard']),
            'hitterPitchData': hitter_pitch_lb_slim,
            'metadata': result['metadata'],
            'microData': result['micro_data'],
            'pitchDetails': result['pitch_details'],
            'hitterPitchDetails': result['hitter_pitch_details'],
            'hitterSwingLocations': result.get('hitter_swing_locations', {}),
        }

    import gzip

    # Serialize the rounded payload, then gzip it. The browser fetches the
    # .gz and inflates it with DecompressionStream (see js/data.js). JSON of
    # this shape compresses ≈6-8x, so a ~97 MB payload lands at ~13-16 MB:
    # well under GitHub's 100 MB file wall (months of season headroom), a
    # far smaller git push every run, and a 6-8x smaller download for every
    # visitor (the page-load speedup).
    data_obj = round_floats_inplace(build_data_obj(rs_result))
    payload = json.dumps(data_obj, separators=(',', ':')).encode('utf-8')
    raw_mb = len(payload) / 1048576

    gz_path = os.path.join(DATA_DIR, 'data_embedded.json.gz')
    # mtime=0 → byte-identical output when the data is unchanged, so a
    # same-day re-run with no new games produces no spurious commit.
    with open(gz_path, 'wb') as f:
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))
    gz_mb = os.path.getsize(gz_path) / 1048576

    # Remove the legacy uncompressed file so the old 100 MB artifact stops
    # being committed (the workflow's `git add data/` stages this deletion).
    legacy_js = os.path.join(DATA_DIR, 'data_embedded.js')
    if os.path.exists(legacy_js):
        os.remove(legacy_js)

    print(f"  Wrote data_embedded.json.gz "
          f"({gz_mb:.1f} MB gz, {raw_mb:.1f} MB raw, "
          f"{raw_mb / gz_mb:.1f}x ratio)")

    # Guard on the COMPRESSED size (that's what git/GitHub sees). At ~15 MB
    # this never fires in normal operation; tripping it means the payload
    # grew ~3x unexpectedly and something is wrong. Fail fast with an
    # actionable message rather than at the push step.
    if os.path.getsize(gz_path) > 90 * 1048576:
        raise SystemExit(
            f"FATAL: data_embedded.json.gz is {gz_mb:.1f} MB. Even compressed "
            f"it is near GitHub's 100 MB file wall — the payload needs to be "
            f"split (move pitchDetails/microData to a Release asset like the "
            f"pickle). See write_embedded_js."
        )


def bump_asset_version(index_path=None):
    """Rewrite every `?v=...` query in index.html to the current build
    timestamp (YYYYMMDDHHMMSS). Forces browsers to bypass cached CSS/JS/data
    whenever the pipeline regenerates output. Second-resolution so two runs
    that land in the same minute still produce distinct ?v= tags — required
    because data_embedded.json.gz is served immutable with a static filename,
    so an identical ?v= on differing content would serve stale data for up to
    a year."""
    if index_path is None:
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'index.html')
    if not os.path.exists(index_path):
        print(f"  WARN: {index_path} not found; skipping version bump")
        return
    build_tag = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(index_path, 'r') as f:
        html = f.read()
    new_html, n = re.subn(r'\?v=[\w-]+', f'?v={build_tag}', html)
    if n > 0 and new_html != html:
        with open(index_path, 'w') as f:
            f.write(new_html)
        print(f"  Bumped {n} ?v= query params in index.html to {build_tag}")
    elif n > 0:
        print(f"  index.html already at ?v={build_tag} (no change)")
    else:
        print("  No ?v= query params found in index.html")


def main():
    global WOBA_WEIGHTS, FIP_CONSTANT, GUTS_EXTRA, PARK_FACTORS
    os.makedirs(DATA_DIR, exist_ok=True)

    # Fetch live wOBA weights and FIP constant from FanGraphs
    print("Fetching FanGraphs Guts constants...")
    try:
        WOBA_WEIGHTS, FIP_CONSTANT, GUTS_EXTRA = fetch_guts_constants(2026)
    except Exception as e:
        print(f"\n  *** WARNING: Could not fetch Guts data ({e}) ***")
        print(f"  *** Using 2025 FALLBACK values — wOBA weights may be inaccurate! ***\n")
        WOBA_WEIGHTS = WOBA_WEIGHTS_FALLBACK.copy()
        FIP_CONSTANT = FIP_CONSTANT_FALLBACK
        # Fallback league-level constants (2025 season estimates)
        GUTS_EXTRA = {'wOBAScale': 1.25, 'lgWOBA': 0.317, 'lgRPA': 0.119}

    # Propagate wOBA weights to pipeline_compute module
    # WOBA_WEIGHTS passed explicitly to compute_expected_stats calls

    # Fetch park factors
    print("Fetching FanGraphs park factors...")
    try:
        PARK_FACTORS = fetch_park_factors(2026)
    except Exception as e:
        print(f"  WARNING: Could not fetch park factors ({e}), defaulting to 1.0")
        PARK_FACTORS = {}

    # Read Regular Season data from the six 2026 division workbooks (Sheets, on
    # the huronalytics account). Pitcher2026 appends here and retagging happens
    # here, so this is the source of truth.
    print("\n=== Reading Regular Season data (Sheets) ===")
    rs_pitches = read_all_pitches_from_sheets()
    print(f"  Read {len(rs_pitches)} RS pitches from the 6 division workbooks")

    # Normalize player names: strip surrounding whitespace so a stray trailing/
    # leading space (e.g. "Lee, Hao-Yu ") doesn't fork one player into duplicate
    # rows. (Accents are already handled consistently upstream.)
    _name_fixed = 0
    for _p in rs_pitches:
        for _fld in ('Batter', 'Pitcher'):
            _v = _p.get(_fld)
            if isinstance(_v, str) and _v != _v.strip():
                _p[_fld] = _v.strip(); _name_fixed += 1
    if _name_fixed:
        print(f"  Normalized {_name_fixed} whitespace-padded player names")

    # Shared MLB ID cache
    mlb_id_cache_path = os.path.join(DATA_DIR, 'mlb_id_cache.json')
    mlb_id_cache = load_mlb_id_cache(mlb_id_cache_path)

    # Process Regular Season
    print("\n" + "=" * 60)
    print("=== Processing Regular Season ===")
    print("=" * 60)
    rs_result = process_game_type(rs_pitches, 'RS', mlb_id_cache, mlb_id_cache_path)

    # Save shared MLB ID cache
    save_mlb_id_cache(mlb_id_cache, mlb_id_cache_path)

    # Write output files
    print("\n--- Writing output files ---")
    write_json_outputs(rs_result, '_rs')
    write_embedded_js(rs_result)
    bump_asset_version()

    print(f"\nOutput written to {DATA_DIR}/")
    print(f"  RS: {len(rs_result['pitcher_leaderboard'])} pitchers, "
          f"{len(rs_result['pitch_leaderboard'])} pitch rows, "
          f"{len(rs_result['hitter_leaderboard'])} hitters")

    # Final integrity checks
    warnings = []
    for r in rs_result['pitcher_leaderboard']:
        if r.get('era') is not None and r['era'] < 0:
            warnings.append(f"Negative ERA: {r.get('pitcher')} = {r['era']}")
    for r in rs_result['hitter_leaderboard']:
        pa = r.get('pa') or 0
        w = r.get('wOBA')
        # wOBA can exceed 1.0 in small samples (a single HR PA = 2.091).
        # Only flag wOBA > 1.0 for hitters with enough PA that it's
        # genuinely impossible (~30+); always flag negatives or absurd
        # values regardless of sample.
        if w is not None and (w < 0 or w > 2.5 or (w > 1.0 and pa >= 30)):
            warnings.append(f"wOBA out of bounds: {r.get('hitter')} pa={pa} = {w}")
        a = r.get('avg')
        if a is not None and (a < 0 or a > 1.0):
            warnings.append(f"AVG out of bounds: {r.get('hitter')} pa={pa} = {a}")
    if warnings:
        print(f"\n*** DATA INTEGRITY WARNINGS ({len(warnings)}) ***")
        for w in warnings[:20]:
            print(f"  - {w}")
        if os.environ.get('CI'):
            print("FATAL: Data integrity checks failed in CI — aborting.")
            sys.exit(1)
    else:
        print("  Data integrity checks passed.")


if __name__ == '__main__':
    main()
