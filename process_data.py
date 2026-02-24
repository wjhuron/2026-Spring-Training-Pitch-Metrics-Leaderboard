#!/usr/bin/env python3
"""Process ST 2026 pitching and hitting data from Google Sheets into JSON files for the leaderboard website."""

import gspread
from google.oauth2.service_account import Credentials
import json
import math
import os
import time as time_module
from datetime import datetime, time
from collections import defaultdict

PITCHING_SPREADSHEET_ID = '1nIk00hnO2VlXLoApMRK2wSmnEKslqI7ybRjHO5HOG4w'
HITTING_SPREADSHEET_ID = '122pPITUxDJK0M_CyXJ4dkWOmGFodEBosAgcjE1PZ3RE'
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'service_account.json')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

METRIC_COLS = [
    'Velocity', 'Spin Rate', 'IndVertBrk', 'HorzBrk',
    'RelPosZ', 'RelPosX', 'Extension', 'VAA', 'HAA', 'VRA', 'HRA'
]

METRIC_KEYS = {
    'Velocity': 'velocity', 'Spin Rate': 'spinRate',
    'IndVertBrk': 'indVertBrk', 'HorzBrk': 'horzBrk',
    'RelPosZ': 'relPosZ', 'RelPosX': 'relPosX',
    'Extension': 'extension', 'VAA': 'vaa', 'HAA': 'haa',
    'VRA': 'vra', 'HRA': 'hra',
}

STAT_KEYS = ['izPct', 'swStrPct', 'cswPct', 'chasePct', 'gbPct']

# Metrics that get percentile ranks on the pitch leaderboard (per pitch type)
PITCH_PCTL_KEYS = list(METRIC_KEYS.values()) + STAT_KEYS

IN_ZONE = {1, 2, 3, 4, 5, 6, 7, 8, 9}
OUT_ZONE = {11, 12, 13, 14}

# --- Hitter Leaderboard constants ---
SWING_DESCRIPTIONS = {'Swinging Strike', 'Foul', 'Foul Tip', 'In Play', 'Swinging Strike (Blocked)'}
HITTER_STAT_KEYS = [
    'swingPct', 'izSwingPct', 'chasePct', 'izSwChase',
    'whiffPct', 'medEV', 'maxEV', 'barrelPct',
    'xBA', 'xSLG',
    'gbPct', 'ldPct', 'fbPct', 'medLA',
]
# Hitter stats where lower is better (invert percentile so low value = red/high pctl)
HITTER_INVERT_PCTL = {'swingPct', 'chasePct', 'whiffPct', 'gbPct'}


def break_tilt_to_minutes(val):
    """Convert a time value (clock notation) to total minutes (0-719).
    Handles time objects, datetime objects, and string formats like '12:23' or '1:17'."""
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


def safe_float(val):
    """Convert a value to float, returning None if not possible."""
    if val is None or val == '':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """Convert a value to int, returning None if not possible."""
    if val is None or val == '':
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def avg(values):
    """Average a list of numbers, ignoring None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_stats(pitches):
    """Compute IZ%, SwStr%, CSW%, Chase%, GB% from a list of pitch dicts."""
    total = len(pitches)
    if total == 0:
        return {k: None for k in STAT_KEYS}

    iz = sum(1 for p in pitches if p['Zone'] in IN_ZONE)
    swstr = sum(1 for p in pitches if p['Description'] == 'Swinging Strike')
    csw = sum(1 for p in pitches if p['Description'] in ('Called Strike', 'Swinging Strike'))

    ooz = [p for p in pitches if p['Zone'] in OUT_ZONE]
    ooz_swung = sum(1 for p in ooz if p['Description'] in ('Swinging Strike', 'In Play', 'Foul'))

    bip = [p for p in pitches if p['BB Type'] is not None]
    gb = sum(1 for p in bip if p['BB Type'] == 'ground_ball')

    return {
        'izPct': iz / total,
        'swStrPct': swstr / total,
        'cswPct': csw / total,
        'chasePct': ooz_swung / len(ooz) if ooz else None,
        'gbPct': gb / len(bip) if bip else None,
    }


def round_metric(key, value):
    """Round a metric value according to its type."""
    if value is None:
        return None
    if key == 'Spin Rate':
        return round(value)
    if key in ('VAA', 'HAA', 'VRA', 'HRA'):
        return round(value, 2)
    return round(value, 1)


def median(values):
    """Compute median, ignoring None values."""
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return None
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    return (nums[n // 2 - 1] + nums[n // 2]) / 2


def is_barrel(ev, la):
    """Statcast barrel definition.
    EV >= 98 mph, then LA must be within a range that expands with velocity."""
    if ev is None or la is None:
        return False
    if ev < 98:
        return False
    lower_la = max(8, 26 - (ev - 98))
    upper_la = min(50, 30 + 1.2 * (ev - 98))
    return lower_la <= la <= upper_la


def compute_hitter_stats(pitches):
    """Compute hitter stats from a list of pitch dicts (already Zone-converted)."""
    total = len(pitches)
    if total == 0:
        return {k: None for k in HITTER_STAT_KEYS}

    # Swings
    swings = [p for p in pitches if p['Description'] in SWING_DESCRIPTIONS]
    n_swings = len(swings)
    whiffs = sum(1 for p in pitches if p['Description'] in ('Swinging Strike', 'Swinging Strike (Blocked)'))

    # In-zone / Out-of-zone
    iz_pitches = [p for p in pitches if p['Zone'] in IN_ZONE]
    ooz_pitches = [p for p in pitches if p['Zone'] in OUT_ZONE]
    iz_swings = sum(1 for p in iz_pitches if p['Description'] in SWING_DESCRIPTIONS)
    ooz_swings = sum(1 for p in ooz_pitches if p['Description'] in SWING_DESCRIPTIONS)

    iz_swing_pct = iz_swings / len(iz_pitches) if iz_pitches else None
    chase_pct = ooz_swings / len(ooz_pitches) if ooz_pitches else None

    # Batted balls (balls in play with BB Type)
    bip = [p for p in pitches if p['BB Type'] is not None]
    n_bip = len(bip)
    gb = sum(1 for p in bip if p['BB Type'] == 'ground_ball')
    ld = sum(1 for p in bip if p['BB Type'] == 'line_drive')
    fb = sum(1 for p in bip if p['BB Type'] in ('fly_ball', 'popup'))

    # Exit Velocity & Launch Angle (only LA > 0 for EV stats)
    ev_la_pos = [(safe_float(p.get('Exit Velocity')), safe_float(p.get('Launch Angle')))
                 for p in bip
                 if safe_float(p.get('Launch Angle')) is not None and safe_float(p.get('Launch Angle')) > 0
                 and safe_float(p.get('Exit Velocity')) is not None]
    evs_pos = [ev for ev, la in ev_la_pos]

    # Barrels: need EV and LA on all batted balls
    ev_la_all = [(safe_float(p.get('Exit Velocity')), safe_float(p.get('Launch Angle')))
                 for p in bip
                 if safe_float(p.get('Exit Velocity')) is not None
                 and safe_float(p.get('Launch Angle')) is not None]
    barrels = sum(1 for ev, la in ev_la_all if is_barrel(ev, la))

    # Median launch angle on ALL batted balls
    all_la = [safe_float(p.get('Launch Angle')) for p in bip
              if safe_float(p.get('Launch Angle')) is not None]

    # xBA and xSLG on batted balls
    xba_vals = [safe_float(p.get('xBA')) for p in bip if safe_float(p.get('xBA')) is not None]
    xslg_vals = [safe_float(p.get('xSLG')) for p in bip if safe_float(p.get('xSLG')) is not None]

    return {
        'nSwings': n_swings,
        'swingPct': n_swings / total if total > 0 else None,
        'izSwingPct': iz_swing_pct,
        'chasePct': chase_pct,
        'izSwChase': round(iz_swing_pct - chase_pct, 4) if iz_swing_pct is not None and chase_pct is not None else None,
        'whiffPct': whiffs / n_swings if n_swings > 0 else None,
        'medEV': round(median(evs_pos), 1) if evs_pos else None,
        'maxEV': round(max(evs_pos), 1) if evs_pos else None,
        'barrelPct': barrels / n_bip if n_bip > 0 else None,
        'xBA': round(avg(xba_vals), 3) if xba_vals else None,
        'xSLG': round(avg(xslg_vals), 3) if xslg_vals else None,
        'gbPct': gb / n_bip if n_bip > 0 else None,
        'ldPct': ld / n_bip if n_bip > 0 else None,
        'fbPct': fb / n_bip if n_bip > 0 else None,
        'medLA': round(median(all_la), 1) if all_la else None,
    }


def read_sheet_with_retry(ws, max_retries=3):
    """Read a worksheet with retry logic for rate limiting (429 errors)."""
    for attempt in range(max_retries):
        try:
            return ws.get_all_values()
        except gspread.exceptions.APIError as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time_module.sleep(wait)
            else:
                raise


def compute_percentile_ranks(rows, metric_key):
    """Compute percentile rank (0-100) for each row's metric value.
    Uses the 'mean rank' method for ties."""
    pctl_key = metric_key + '_pctl'
    valid = [(i, rows[i][metric_key]) for i in range(len(rows))
             if rows[i].get(metric_key) is not None]

    if len(valid) < 2:
        for row in rows:
            row[pctl_key] = 50 if row.get(metric_key) is not None else None
        return

    values = [v for _, v in valid]
    n = len(values)

    for idx, val in valid:
        below = sum(1 for x in values if x < val)
        equal = sum(1 for x in values if x == val)
        pctl = (below + 0.5 * (equal - 1)) / max(1, n - 1) * 100
        rows[idx][pctl_key] = max(0, min(100, round(pctl)))

    # Set None for rows that don't have the metric
    for row in rows:
        if pctl_key not in row:
            row[pctl_key] = None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Connecting to Google Sheets...")
    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(PITCHING_SPREADSHEET_ID)
    print(f"Spreadsheet: {sh.title} ({len(sh.worksheets())} sheets)")

    # Read all pitches from all sheets
    all_pitches = []
    for i, ws in enumerate(sh.worksheets()):
        print(f"  Reading {ws.title}...")
        if i > 0:
            time_module.sleep(1.5)
        rows = read_sheet_with_retry(ws)
        if not rows:
            continue
        header = rows[0]
        col_idx = {name: i for i, name in enumerate(header) if name}

        for row in rows[1:]:
            pitcher = row[col_idx['Pitcher']] if 'Pitcher' in col_idx else None
            if not pitcher:
                continue

            pitch = {}
            for col_name, idx in col_idx.items():
                val = row[idx] if idx < len(row) else None
                # Convert empty strings to None
                if val == '':
                    val = None
                pitch[col_name] = val
            all_pitches.append(pitch)

    print(f"Read {len(all_pitches)} pitches from {len(sh.worksheets())} sheets")

    # Collect unique teams and pitch types
    all_teams = sorted(set(p['Team'] for p in all_pitches if p.get('Team')))
    all_pitch_types = sorted(set(p['Pitch Type'] for p in all_pitches if p.get('Pitch Type')))

    # --- Count total pitches per pitcher (for usage%) ---
    pitcher_total = defaultdict(int)
    for p in all_pitches:
        pitcher_total[(p['Pitcher'], p['Team'])] += 1

    # --- Pitch Leaderboard: group by (Pitcher, Team, Pitch Type) ---
    pitch_groups = defaultdict(list)
    for p in all_pitches:
        key = (p['Pitcher'], p['Team'], p['Pitch Type'], p.get('Throws'))
        pitch_groups[key].append(p)

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
        }

        # Average metrics
        for col in METRIC_COLS:
            values = [safe_float(p.get(col)) for p in pitches]
            key_name = METRIC_KEYS[col]
            row[key_name] = round_metric(col, avg(values))

        # Break Tilt (circular mean)
        tilt_minutes = [break_tilt_to_minutes(p.get('Break Tilt')) for p in pitches]
        tilt_minutes = [m for m in tilt_minutes if m is not None]
        avg_tilt = circular_mean_minutes(tilt_minutes)
        row['breakTilt'] = minutes_to_tilt_display(avg_tilt)
        row['breakTiltMinutes'] = avg_tilt

        # Stats — convert Zone to int (gspread returns strings)
        for p in pitches:
            p['Zone'] = safe_int(p.get('Zone'))

        row.update(compute_stats(pitches))
        pitch_leaderboard.append(row)

    # --- Compute percentiles per pitch type ---
    pt_groups = defaultdict(list)
    for row in pitch_leaderboard:
        pt_groups[row['pitchType']].append(row)

    for pt, pt_rows in pt_groups.items():
        for metric in PITCH_PCTL_KEYS:
            compute_percentile_ranks(pt_rows, metric)

    # --- Compute Stuff Score ---
    # Average of velocity and spin rate percentiles within pitch type
    for row in pitch_leaderboard:
        vp = row.get('velocity_pctl')
        sp = row.get('spinRate_pctl')
        if vp is not None and sp is not None:
            row['stuffScore'] = round((vp + sp) / 2)
        else:
            row['stuffScore'] = None

    # Compute percentile of stuff score within pitch type
    for pt, pt_rows in pt_groups.items():
        compute_percentile_ranks(pt_rows, 'stuffScore')

    pitch_leaderboard.sort(key=lambda r: r['count'], reverse=True)
    print(f"Pitch leaderboard: {len(pitch_leaderboard)} rows")

    # --- Pitcher Leaderboard: group by (Pitcher, Team) ---
    pitcher_groups = defaultdict(list)
    for p in all_pitches:
        key = (p['Pitcher'], p['Team'], p.get('Throws'))
        pitcher_groups[key].append(p)

    pitcher_leaderboard = []
    for (pitcher, team, throws), pitches in pitcher_groups.items():
        for p in pitches:
            p['Zone'] = safe_int(p.get('Zone'))

        row = {
            'pitcher': pitcher,
            'team': team,
            'throws': throws,
            'count': len(pitches),
        }
        row.update(compute_stats(pitches))
        pitcher_leaderboard.append(row)

    # Compute percentiles for pitcher leaderboard (across all pitchers)
    for stat in STAT_KEYS:
        compute_percentile_ranks(pitcher_leaderboard, stat)

    pitcher_leaderboard.sort(key=lambda r: r['count'], reverse=True)
    print(f"Pitcher leaderboard: {len(pitcher_leaderboard)} rows")

    # --- Pitch Details: individual pitch data for scatter plots + velo distribution ---
    pitch_details = defaultdict(list)
    for p in all_pitches:
        pitcher = p.get('Pitcher')
        pt = p.get('Pitch Type')
        ivb = safe_float(p.get('IndVertBrk'))
        hb = safe_float(p.get('HorzBrk'))
        velo = safe_float(p.get('Velocity'))
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
            if rel_x is not None:
                detail['rx'] = round(rel_x, 2)
            if rel_z is not None:
                detail['rz'] = round(rel_z, 2)
            pitch_details[pitcher].append(detail)
    print(f"Pitch details: {sum(len(v) for v in pitch_details.values())} pitches for {len(pitch_details)} pitchers")

    # --- League Averages per pitch type ---
    league_avgs = {}
    for pt, pt_rows in pt_groups.items():
        avgs = {}
        for metric in list(METRIC_KEYS.values()):
            vals = [r[metric] for r in pt_rows if r.get(metric) is not None]
            if vals:
                avgs[metric] = round(sum(vals) / len(vals), 2)
        for stat in STAT_KEYS:
            vals = [r[stat] for r in pt_rows if r.get(stat) is not None]
            if vals:
                avgs[stat] = round(sum(vals) / len(vals), 4)
        tilts = [r['breakTiltMinutes'] for r in pt_rows if r.get('breakTiltMinutes') is not None]
        if tilts:
            avgs['breakTiltMinutes'] = round(sum(tilts) / len(tilts))
            avgs['breakTilt'] = minutes_to_tilt_display(avgs['breakTiltMinutes'])
        avgs['count'] = len(pt_rows)
        league_avgs[pt] = avgs

    # League averages for pitcher leaderboard (across all pitchers)
    pitcher_league_avgs = {}
    for stat in STAT_KEYS:
        vals = [r[stat] for r in pitcher_leaderboard if r.get(stat) is not None]
        if vals:
            pitcher_league_avgs[stat] = round(sum(vals) / len(vals), 4)
    pitcher_league_avgs['count'] = len(pitcher_leaderboard)

    # ======================================================================
    #  HITTER LEADERBOARD
    # ======================================================================
    print(f"\n--- Hitter Leaderboard ---")
    hsh = gc.open_by_key(HITTING_SPREADSHEET_ID)
    print(f"Spreadsheet: {hsh.title} ({len(hsh.worksheets())} sheets)")

    all_abs = []  # each element is one pitch seen by a hitter
    for i, ws in enumerate(hsh.worksheets()):
        print(f"  Reading {ws.title}...")
        if i > 0:
            time_module.sleep(1.5)
        rows = read_sheet_with_retry(ws)
        if not rows:
            continue
        header = rows[0]
        col_idx = {name: i for i, name in enumerate(header) if name}

        for row in rows[1:]:
            hitter = row[col_idx['Hitter']] if 'Hitter' in col_idx else None
            if not hitter:
                continue
            ab = {}
            for col_name, idx in col_idx.items():
                val = row[idx] if idx < len(row) else None
                if val == '':
                    val = None
                ab[col_name] = val
            all_abs.append(ab)

    print(f"Read {len(all_abs)} pitches from {len(hsh.worksheets())} sheets (hitters)")

    # Collect unique teams from hitter data too
    hitter_teams = sorted(set(p['Team'] for p in all_abs if p.get('Team')))
    all_teams_combined = sorted(set(all_teams + hitter_teams))

    # --- Hitter Leaderboard: group by (Hitter, Team, Stands) ---
    hitter_groups = defaultdict(list)
    for p in all_abs:
        key = (p['Hitter'], p['Team'], p.get('Stands'))
        hitter_groups[key].append(p)

    hitter_leaderboard = []
    for (hitter, team, stands), pitches in hitter_groups.items():
        # Convert zones to int
        for p in pitches:
            p['Zone'] = safe_int(p.get('Zone'))

        row = {
            'hitter': hitter,
            'team': team,
            'stands': stands,
            'count': len(pitches),
        }
        row.update(compute_hitter_stats(pitches))
        hitter_leaderboard.append(row)

    # Compute percentiles across all hitters
    for stat in HITTER_STAT_KEYS:
        compute_percentile_ranks(hitter_leaderboard, stat)

    # Invert percentiles where lower is better (Swing%, Chase%, Whiff%, GB%)
    for row in hitter_leaderboard:
        for stat in HITTER_INVERT_PCTL:
            pctl_key = stat + '_pctl'
            if row.get(pctl_key) is not None:
                row[pctl_key] = 100 - row[pctl_key]

    hitter_leaderboard.sort(key=lambda r: r['count'], reverse=True)
    print(f"Hitter leaderboard: {len(hitter_leaderboard)} rows")

    # --- Hitter pitch details: per-hitter breakdown by pitch type faced ---
    hitter_pitch_details = {}
    for (hitter, team, stands), pitches in hitter_groups.items():
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
        # Sort by count desc
        details.sort(key=lambda x: x['count'], reverse=True)
        hitter_pitch_details[hitter] = details

    # Hitter league averages
    hitter_league_avgs = {}
    for stat in HITTER_STAT_KEYS:
        vals = [r[stat] for r in hitter_leaderboard if r.get(stat) is not None]
        if vals:
            hitter_league_avgs[stat] = round(sum(vals) / len(vals), 4)
    hitter_league_avgs['count'] = len(hitter_leaderboard)

    # --- Metadata ---
    metadata = {
        'teams': all_teams_combined,
        'pitchTypes': all_pitch_types,
        'generatedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'totalPitches': len(all_pitches),
        'totalPitchers': len(pitcher_leaderboard),
        'totalHitters': len(hitter_leaderboard),
        'leagueAverages': league_avgs,
        'pitcherLeagueAverages': pitcher_league_avgs,
        'hitterLeagueAverages': hitter_league_avgs,
    }

    # Write JSON files
    with open(os.path.join(DATA_DIR, 'pitch_leaderboard.json'), 'w') as f:
        json.dump(pitch_leaderboard, f)
    with open(os.path.join(DATA_DIR, 'pitcher_leaderboard.json'), 'w') as f:
        json.dump(pitcher_leaderboard, f)
    with open(os.path.join(DATA_DIR, 'hitter_leaderboard.json'), 'w') as f:
        json.dump(hitter_leaderboard, f)
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write embedded JS fallback (for file:// usage)
    with open(os.path.join(DATA_DIR, 'data_embedded.js'), 'w') as f:
        f.write('// Auto-generated — do not edit\n')
        f.write('window.PITCH_DATA = ')
        json.dump(pitch_leaderboard, f)
        f.write(';\n')
        f.write('window.PITCHER_DATA = ')
        json.dump(pitcher_leaderboard, f)
        f.write(';\n')
        f.write('window.HITTER_DATA = ')
        json.dump(hitter_leaderboard, f)
        f.write(';\n')
        f.write('window.METADATA = ')
        json.dump(metadata, f)
        f.write(';\n')
        f.write('window.PITCH_DETAILS = ')
        json.dump(pitch_details, f)
        f.write(';\n')
        f.write('window.HITTER_PITCH_DETAILS = ')
        json.dump(hitter_pitch_details, f)
        f.write(';\n')

    print(f"\nOutput written to {DATA_DIR}/")
    print(f"  pitch_leaderboard.json  ({len(pitch_leaderboard)} rows)")
    print(f"  pitcher_leaderboard.json ({len(pitcher_leaderboard)} rows)")
    print(f"  hitter_leaderboard.json  ({len(hitter_leaderboard)} rows)")
    print(f"  metadata.json")
    print(f"  data_embedded.js")


if __name__ == '__main__':
    main()
