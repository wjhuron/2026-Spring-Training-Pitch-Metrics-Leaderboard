#!/usr/bin/env python3
"""Stat computation: expected stats, pitch/hitter stats, micro-data, percentiles."""

import math
from collections import defaultdict

from pipeline.utils import (
    safe_float, median, is_barrel,
    spray_angle, spray_direction,
    SWING_DESCRIPTIONS, HIT_EVENTS, K_EVENTS, BB_EVENTS, HBP_EVENTS,
    SF_EVENTS, SH_EVENTS, CI_EVENTS, NON_PA_EVENTS, BUNT_BB_TYPES,
    ALL_TEAMS,
)

# ── Metric / leaderboard constants ───────────────────────────────────────

METRIC_COLS = [
    'Velocity', 'Spin Rate', 'xIndVrtBrk', 'xHorzBrk',
    'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle', 'VAA', 'HAA'
]

METRIC_KEYS = {
    'Velocity': 'velocity',
    'Spin Rate': 'spinRate',
    'xIndVrtBrk': 'indVertBrk', 'xHorzBrk': 'horzBrk',
    'RelPosZ': 'relPosZ', 'RelPosX': 'relPosX',
    'Extension': 'extension', 'ArmAngle': 'armAngle',
    'VAA': 'vaa', 'HAA': 'haa',
}

PITCH_STAT_KEYS = ['strikePct', 'izPct', 'swStrRate', 'swStrPct', 'cswPct', 'izWhiffPct', 'twoStrikeWhiffPct', 'chasePct', 'gbPct', 'fpsPct']
STAT_KEYS = ['strikePct', 'izPct', 'swStrRate', 'swStrPct', 'cswPct', 'izWhiffPct', 'chasePct', 'gbPct', 'kPct', 'bbPct', 'kbbPct', 'babip', 'fpsPct', 'twoStrikeWhiffPct', 'oneOneWinPct', 'earlyActionPct']

# Per-pitch-type batted-ball stats. These get percentiles computed per pitch
# type and use a BIP-count qualifier (not pitch count) since their denominator
# is BIP. Matches the leaderboard's posture for these stats.
PITCH_BB_PCTL_KEYS = ['avgEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'hrFbPct', 'ldPct', 'fbPct', 'puPct', 'babip']
# Direction: lower is better for pitcher on these. PuPct stays default (higher
# = better for pitcher; popups are auto-outs). gbPct also default (higher GB
# = good for pitcher) and is already in PITCH_STAT_KEYS.
PITCH_BB_INVERT = {'avgEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'hrFbPct', 'ldPct', 'fbPct', 'babip'}

PITCH_PCTL_KEYS = list(METRIC_KEYS.values()) + ['nVAA', 'nHAA', 'ivbOE', 'hbOE'] + PITCH_STAT_KEYS + PITCH_BB_PCTL_KEYS + ['runValue', 'rv100', 'xRunValue', 'xRv100', 'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp', 'locPlus']

PITCHER_INVERT_PCTL = {'bbPct', 'babip', 'era', 'fip', 'xFIP', 'siera'}

HITTER_STAT_KEYS = [
    'avg', 'obp', 'slg', 'ops', 'iso', 'babip', 'kPct', 'bbPct', 'bbToK',
    'wOBA', 'xBA', 'xSLG', 'xwOBA', 'xwOBAcon', 'xwOBAsp', 'sprayVal', 'bbPlus',
    'avgEVAll', 'ev50', 'maxEV', 'hardHitPct', 'barrelPct',
    'gbPct', 'ldPct', 'fbPct', 'puPct', 'hrFbPct',
    'pullPct', 'middlePct', 'oppoPct', 'airPullPct',
    'swingPct', 'izSwingPct', 'chasePct', 'izSwChase', 'contactPct', 'izContactPct', 'whiffPct', 'sdPlus', 'ctPlus',
    'batSpeed', 'swingLength', 'attackAngle', 'attackDirection', 'swingPathTilt',
    'blastPct', 'squaredUpPct', 'idealAAPct',
    'twoStrikeWhiffPct', 'firstPitchSwingPct',
    'avgFbDist', 'avgHrDist',
    'sprintSpeed',
    'wRCplus', 'xWRCplus', 'hitterPlus',
    'runValue', 'xRunValue', 'rv100', 'xRv100',
    'hr', 'sb',
]
# oppoPct inverted 2026-08-06 (per Wally): oppo contact is the anti-pull-damage
# direction, so low oppo colors well — consistent with airPullPct higher-better.
# Only the player-page batted-ball table colors it (leaderboard Oppo% is
# noPercentile), so the flip is visible exactly where requested.
# swingPct is deliberately NOT inverted (2026-08-12, per Wally): a swing rate
# has no honest good direction, so it is coloured in its natural direction and
# the cards/leaderboard carry a note saying aggression is only good when it is
# aimed well. Inverting it asserted that taking pitches is a virtue.
HITTER_INVERT_PCTL = {'chasePct', 'whiffPct', 'gbPct', 'kPct', 'puPct', 'twoStrikeWhiffPct', 'firstPitchSwingPct', 'oppoPct'}

PITCHER_BB_KEYS = ['avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'ldPct', 'fbPct', 'puPct', 'hrFbPct', 'xwOBAsp']
PITCHER_BB_INVERT = {'avgEVAgainst', 'maxEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'hrFbPct', 'xwOBAsp'}

# ── Stat computation functions ───────────────────────────────────────────

def compute_expected_stats(pitches, woba_weights=None):
    """Compute wOBA, xBA, xSLG, xwOBA, xwOBAcon from pitch-level data.

    wOBA uses FanGraphs Guts linear weights applied to actual outcomes.
    xBA/xSLG/xwOBA use Statcast per-pitch expected values from the spreadsheet.
    """
    weights = woba_weights

    ab = 0
    ubb = 0
    hbp_count = 0
    sf = 0
    singles = 0
    doubles = 0
    triples = 0
    hr = 0
    xba_sum = 0.0
    xba_denom = 0   # AB-events with a Statcast xBA value
    xslg_sum = 0.0
    xslg_denom = 0  # AB-events with a Statcast xSLG value
    xwoba_sum = 0.0
    xwoba_denom = 0
    xwobacon_sum = 0.0
    xwobacon_denom = 0
    nonbunt_ab = 0  # AB denominator for xBA/xSLG, excluding bunt ABs (Savant convention)

    for p in pitches:
        event = p.get('Event')
        if not event or event in NON_PA_EVENTS:
            continue

        if event == 'Intent Walk':
            continue

        # xwOBA denominator is AB+BB+HBP+SF (standard); exclude SH (sac bunt) and CI.
        if event not in SH_EVENTS and event not in CI_EVENTS:
            xwoba_val = safe_float(p.get('xwOBA'))
            if xwoba_val is not None:
                xwoba_sum += xwoba_val
                xwoba_denom += 1

        if event in BB_EVENTS:
            ubb += 1
            continue
        elif event in HBP_EVENTS:
            hbp_count += 1
            continue
        elif event in SF_EVENTS:
            sf += 1
            xwobacon_val = safe_float(p.get('xwOBA'))
            if xwobacon_val is not None:
                xwobacon_sum += xwobacon_val
                xwobacon_denom += 1
            continue
        elif event in SH_EVENTS or event in CI_EVENTS:
            continue

        ab += 1
        # Savant excludes bunts from expected stats entirely, so bunt at-bats are
        # dropped from BOTH the xBA/xSLG numerators AND the denominator (nonbunt_ab).
        _is_bunt = p.get('BBType') in BUNT_BB_TYPES
        if not _is_bunt:
            nonbunt_ab += 1
        if event == 'Single':
            singles += 1
        elif event == 'Double':
            doubles += 1
        elif event == 'Triple':
            triples += 1
        elif event == 'Home Run':
            hr += 1

        xba_val = safe_float(p.get('xBA'))
        xslg_val = safe_float(p.get('xSLG'))
        if not _is_bunt and xba_val is not None:
            xba_sum += xba_val
            xba_denom += 1
        if not _is_bunt and xslg_val is not None:
            xslg_sum += xslg_val
            xslg_denom += 1

        if event not in K_EVENTS and not _is_bunt:
            xwobacon_val = safe_float(p.get('xwOBA'))
            if xwobacon_val is not None:
                xwobacon_sum += xwobacon_val
                xwobacon_denom += 1

    result = {}

    woba_denom = ab + ubb + sf + hbp_count
    if woba_denom > 0 and weights:
        woba_num = (weights['BB'] * ubb + weights['HBP'] * hbp_count +
                    weights['1B'] * singles + weights['2B'] * doubles +
                    weights['3B'] * triples + weights['HR'] * hr)
        result['wOBA'] = round(woba_num / woba_denom, 3)
    else:
        result['wOBA'] = None

    # xBA/xSLG are Statcast-only — denominator is non-bunt AB (Savant drops bunts;
    # Ks contribute an implicit 0). Return None when no AB had any Statcast data,
    # otherwise the result is 0.0 for AAA-only players (which then gets a 0th
    # percentile, swamping the leaderboard).
    result['xBA'] = round(xba_sum / nonbunt_ab, 3) if (nonbunt_ab > 0 and xba_denom > 0) else None
    result['xSLG'] = round(xslg_sum / nonbunt_ab, 3) if (nonbunt_ab > 0 and xslg_denom > 0) else None
    result['xwOBA'] = round(xwoba_sum / xwoba_denom, 3) if xwoba_denom > 0 else None
    result['xwOBAcon'] = round(xwobacon_sum / xwobacon_denom, 3) if xwobacon_denom > 0 else None
    return result


def compute_stats(pitches):
    """Compute IZ%, Whiff%, CSW%, Chase%, GB%, K%, BB%, K-BB%, BABIP from a list of pitch dicts."""
    total = len(pitches)
    if total == 0:
        empty = {k: None for k in STAT_KEYS}
        empty['nSwings'] = 0
        empty['nBip'] = 0
        empty['pa'] = 0
        return empty

    iz = sum(1 for p in pitches if p.get('InZone') == 'Yes')
    swings = sum(1 for p in pitches if p['Description'] in SWING_DESCRIPTIONS)
    whiffs = sum(1 for p in pitches if p['Description'] == 'Swinging Strike')
    csw = sum(1 for p in pitches if p['Description'] in ('Called Strike', 'Swinging Strike'))

    iz_pitches = [p for p in pitches if p.get('InZone') == 'Yes']
    iz_swings = sum(1 for p in iz_pitches if p['Description'] in SWING_DESCRIPTIONS)
    iz_whiffs = sum(1 for p in iz_pitches if p['Description'] == 'Swinging Strike')
    ooz = [p for p in pitches if p.get('InZone') == 'No']
    ooz_swung = sum(1 for p in ooz if p['Description'] in ('Swinging Strike', 'In Play', 'Foul'))

    bip = [p for p in pitches if p.get('BBType') is not None and p.get('BBType') not in BUNT_BB_TYPES]
    gb = sum(1 for p in bip if p.get('BBType') == 'ground_ball')

    pa_pitches = [p for p in pitches if p.get('Event') and p['Event'] not in NON_PA_EVENTS]
    n_pa = len(pa_pitches)
    n_h = sum(1 for p in pa_pitches if p['Event'] in HIT_EVENTS)
    n_hr = sum(1 for p in pa_pitches if p['Event'] == 'Home Run')
    n_k = sum(1 for p in pa_pitches if p['Event'] in K_EVENTS)
    n_bb_all = sum(1 for p in pa_pitches if p['Event'] in BB_EVENTS)
    n_ibb = sum(1 for p in pa_pitches if p['Event'] == 'Intent Walk')
    n_bb = n_bb_all - n_ibb
    n_hbp = sum(1 for p in pa_pitches if p['Event'] in HBP_EVENTS)
    n_sf = sum(1 for p in pa_pitches if p['Event'] in SF_EVENTS)
    n_sh = sum(1 for p in pa_pitches if p['Event'] in SH_EVENTS)
    n_ci = sum(1 for p in pa_pitches if p['Event'] in CI_EVENTS)
    n_ab = n_pa - n_bb_all - n_hbp - n_sf - n_sh - n_ci
    k_pct = n_k / n_pa if n_pa > 0 else None
    bb_pct = n_bb / n_pa if n_pa > 0 else None
    kbb_pct = round(k_pct - bb_pct, 4) if k_pct is not None and bb_pct is not None else None

    babip_denom = n_ab - n_k - n_hr + n_sf
    babip = round((n_h - n_hr) / babip_denom, 3) if babip_denom > 0 else None

    BALL_DESCRIPTIONS = {'Ball', 'Intent Ball', 'Hit By Pitch', 'Pitchout'}
    n_strikes = sum(1 for p in pitches if p.get('Description') not in BALL_DESCRIPTIONS)
    strike_pct = n_strikes / total if total > 0 else None

    rv_values = [safe_float(p.get('RunExp')) for p in pitches]
    rv_values = [v for v in rv_values if v is not None]
    run_value = sum(rv_values) if rv_values else None

    first_pitches = [p for p in pitches if p.get('Count') == '0-0']
    fps_strikes = sum(1 for p in first_pitches
                      if p.get('Description') in ('Called Strike', 'Swinging Strike', 'Foul', 'In Play'))
    fps_pct = fps_strikes / len(first_pitches) if first_pitches else None

    two_strike_pitches = [p for p in pitches if '-' in p.get('Count', '') and p['Count'].split('-')[1] == '2']
    # Exclude bunt-contact swings from the 2-strike whiff denominator to match
    # the regular whiff% / contact% convention.
    two_strike_swings = sum(1 for p in two_strike_pitches
                            if p['Description'] in SWING_DESCRIPTIONS
                            and p.get('BBType') not in BUNT_BB_TYPES)
    two_strike_whiffs = sum(1 for p in two_strike_pitches if p['Description'] == 'Swinging Strike')
    two_strike_whiff_pct = two_strike_whiffs / two_strike_swings if two_strike_swings > 0 else None

    one_one_pitches = [p for p in pitches if p.get('Count') == '1-1']
    one_one_wins = sum(1 for p in one_one_pitches if p.get('Description') not in BALL_DESCRIPTIONS)
    one_one_win_pct = one_one_wins / len(one_one_pitches) if one_one_pitches else None

    early_action = 0
    for p in pitches:
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS:
            pid = p.get('PitchID') or ''
            parts = pid.split('_')
            if len(parts) == 3:
                try:
                    if int(parts[2]) <= 3:
                        early_action += 1
                except ValueError:
                    pass
    early_action_pct = early_action / n_pa if n_pa > 0 else None

    return {
        'pa': n_pa,
        'strikePct': strike_pct,
        'izPct': iz / total,
        'swStrRate': whiffs / total if total > 0 else None,
        'swStrPct': whiffs / swings if swings > 0 else None,
        'cswPct': csw / total,
        'izWhiffPct': iz_whiffs / iz_swings if iz_swings > 0 else None,
        'chasePct': ooz_swung / len(ooz) if ooz else None,
        'gbPct': gb / len(bip) if bip else None,
        'nSwings': swings,
        'nBip': len(bip),
        'kPct': k_pct,
        'bbPct': bb_pct,
        'kbbPct': kbb_pct,
        'babip': babip,
        'fpsPct': fps_pct,
        'twoStrikeWhiffPct': two_strike_whiff_pct,
        'oneOneWinPct': one_one_win_pct,
        'earlyActionPct': early_action_pct,
        'runValue': run_value,
    }


def compute_xrv(pitches, lg_woba=None, woba_scale=None, negate=False,
                count_offsets=None, bip_count_means=None):
    """Compute expected run value (xRV).

    For BIP pitches with xwOBA: uses (xwOBA - lgWOBA) / wOBAScale plus the
    per-count anchoring offset (count_offsets, from
    pipeline_sdplus.build_bip_count_offsets). The offset puts the BIP branch
    in the same count-conditional delta-RE currency as every other pitch, so
    summed xRV is league-calibrated against summed RV (without it, shipped
    xRV ran -495 league runs vs RV's +85, systematically flattering
    two-strike-contact pitchers by up to ~0.3 runs/100 — measured
    2026-07-03, predictive power identical either way).

    For BIPs MISSING xwOBA (~1.5%): uses the league's expected anchored BIP
    value for that count (bip_count_means) rather than the pitch's actual
    -RunExp, which leaked real outcomes into an "expected" stat.

    For all other pitches: uses -RunExp to convert from pitcher perspective
    (RunExp in sheets is positive = good for pitcher) to hitter perspective.

    The raw total is hitter-perspective (positive = good for hitter).
    negate=False (default, for pitchers): flips to pitcher perspective
    (positive = good for pitcher, i.e. runs saved).
    negate=True (for hitters): keeps hitter perspective (positive = good for hitter).
    """
    from pipeline.sdplus import get_count
    xrv_values = []
    has_guts = lg_woba is not None and woba_scale is not None and woba_scale != 0
    for p in pitches:
        is_bip = p.get('Description') == 'In Play'
        if is_bip and has_guts:
            c = get_count(p)
            xwoba_val = safe_float(p.get('xwOBA'))
            if xwoba_val is not None:
                v = (xwoba_val - lg_woba) / woba_scale
                if count_offsets:
                    v += count_offsets.get(c, 0.0)
                xrv_values.append(v)
                continue
            if bip_count_means and c in bip_count_means:
                xrv_values.append(bip_count_means[c])
                continue
        rv = safe_float(p.get('RunExp'))
        if rv is not None:
            xrv_values.append(-rv)
    if not xrv_values:
        return {'xRunValue': None}
    total = sum(xrv_values)
    return {'xRunValue': -total if not negate else total}


def build_bip_count_means(pitches, lg_woba, woba_scale, count_offsets):
    """League mean ANCHORED BIP xRV per count — the fallback value for BIPs
    missing xwOBA (hitter perspective)."""
    from pipeline.sdplus import get_count
    acc = {}
    if lg_woba is None or woba_scale in (None, 0):
        return {}
    for p in pitches:
        if p.get('Description') != 'In Play':
            continue
        xw = safe_float(p.get('xwOBA'))
        c = get_count(p)
        if xw is None or c is None:
            continue
        a = acc.setdefault(c, [0.0, 0])
        a[0] += (xw - lg_woba) / woba_scale + (count_offsets.get(c, 0.0) if count_offsets else 0.0)
        a[1] += 1
    return {c: s / n for c, (s, n) in acc.items() if n >= 50}


def compute_pitcher_batted_ball(pitches):
    """Compute batted-ball-against stats for a pitcher."""
    bip = [p for p in pitches if p.get('BBType') is not None and p.get('BBType') not in BUNT_BB_TYPES]
    n_bip = len(bip)
    if n_bip == 0:
        return {
            'avgEVAgainst': None, 'maxEVAgainst': None,
            'hardHitPct': None, 'barrelPctAgainst': None,
            'ldPct': None, 'fbPct': None, 'puPct': None,
            'hrFbPct': None,
        }

    ev_vals = [safe_float(p.get('ExitVelo')) for p in bip]
    ev_valid = [v for v in ev_vals if v is not None]
    avg_ev = round(sum(ev_valid) / len(ev_valid), 1) if ev_valid else None
    max_ev = round(max(ev_valid), 1) if ev_valid else None

    hard_hit = sum(1 for v in ev_valid if v >= 95)
    hard_hit_pct = hard_hit / len(ev_valid) if ev_valid else None

    has_barrel_col = any(str(p.get('Barrel', '')).strip() != '' for p in bip)
    if has_barrel_col:
        barrels = sum(1 for p in bip if str(p.get('Barrel', '')).strip() == '6')
    else:
        ev_la_pairs = [(safe_float(p.get('ExitVelo')), safe_float(p.get('LaunchAngle')))
                       for p in bip
                       if safe_float(p.get('ExitVelo')) is not None
                       and safe_float(p.get('LaunchAngle')) is not None]
        barrels = sum(1 for ev, la in ev_la_pairs if is_barrel(ev, la))
    barrel_pct = barrels / len(ev_valid) if ev_valid else None

    ld = sum(1 for p in bip if p.get('BBType') == 'line_drive')
    fb = sum(1 for p in bip if p.get('BBType') == 'fly_ball')
    pu = sum(1 for p in bip if p.get('BBType') == 'popup')

    # HR/FB (2026-08-14, per Wally): STRICT CONDITIONAL — fly-ball home runs
    # over outfield flies, so the stat reads "X% of fly balls left the yard".
    # Popups out of the denominator (structurally can't leave the yard);
    # line-drive HRs out of the numerator (barrel-quality events, already
    # priced by Barrel%/xwOBAcon). FanGraphs' published HR/FB keeps popups in
    # the denominator, so their figure reads lower by design (Nola 2026:
    # 23.6 here vs 18.9 there). One definition everywhere: cards,
    # leaderboards, aggregator mirrors.
    n_hr_fb = sum(1 for p in bip if p.get('Event') == 'Home Run'
                  and p.get('BBType') == 'fly_ball')
    hr_fb_pct = round(n_hr_fb / fb, 4) if fb > 0 else None

    return {
        'avgEVAgainst': avg_ev,
        'maxEVAgainst': max_ev,
        'hardHitPct': round(hard_hit_pct, 4) if hard_hit_pct is not None else None,
        'barrelPctAgainst': round(barrel_pct, 4) if barrel_pct is not None else None,
        'ldPct': round(ld / n_bip, 4) if n_bip > 0 else None,
        'fbPct': round(fb / n_bip, 4) if n_bip > 0 else None,
        'puPct': round(pu / n_bip, 4) if n_bip > 0 else None,
        'hrFbPct': hr_fb_pct,
    }


def compute_hitter_stats(pitches):
    """Compute hitter stats from a list of pitch dicts for all hitter leaderboard tabs."""
    total = len(pitches)
    if total == 0:
        empty = {k: None for k in HITTER_STAT_KEYS}
        empty.update({'pa': 0, 'nSwings': 0, 'nBip': 0, 'nCompSwings': 0,
                      'doubles': 0, 'triples': 0, 'hr': 0, 'xbh': 0})
        return empty

    pa_pitches = [p for p in pitches if p.get('Event') and p['Event'] not in NON_PA_EVENTS]
    n_pa = len(pa_pitches)

    n_2b = sum(1 for p in pa_pitches if p['Event'] == 'Double')
    n_3b = sum(1 for p in pa_pitches if p['Event'] == 'Triple')
    n_h = sum(1 for p in pa_pitches if p['Event'] in HIT_EVENTS)
    n_hr = sum(1 for p in pa_pitches if p['Event'] == 'Home Run')
    n_k = sum(1 for p in pa_pitches if p['Event'] in K_EVENTS)
    n_bb_all = sum(1 for p in pa_pitches if p['Event'] in BB_EVENTS)
    n_hbp = sum(1 for p in pa_pitches if p['Event'] in HBP_EVENTS)
    n_sf = sum(1 for p in pa_pitches if p['Event'] in SF_EVENTS)
    n_sh = sum(1 for p in pa_pitches if p['Event'] in SH_EVENTS)
    n_ci = sum(1 for p in pa_pitches if p['Event'] in CI_EVENTS)

    n_ab = n_pa - n_bb_all - n_hbp - n_sf - n_sh - n_ci
    xbh = n_2b + n_3b + n_hr
    babip_denom = n_ab - n_k - n_hr + n_sf
    babip = round((n_h - n_hr) / babip_denom, 3) if babip_denom > 0 else None

    n_swings = sum(1 for p in pitches if p['Description'] in SWING_DESCRIPTIONS)
    whiffs = sum(1 for p in pitches if p['Description'] == 'Swinging Strike')

    iz_pitches = [p for p in pitches if p.get('InZone') == 'Yes']
    ooz_pitches = [p for p in pitches if p.get('InZone') == 'No']
    iz_swings = sum(1 for p in iz_pitches if p['Description'] in SWING_DESCRIPTIONS)
    iz_whiffs = sum(1 for p in iz_pitches if p['Description'] == 'Swinging Strike')
    ooz_swings = sum(1 for p in ooz_pitches if p['Description'] in SWING_DESCRIPTIONS)

    iz_swing_pct = iz_swings / len(iz_pitches) if iz_pitches else None
    chase_pct = ooz_swings / len(ooz_pitches) if ooz_pitches else None

    swings_non_bunt = sum(1 for p in pitches
                          if p['Description'] in SWING_DESCRIPTIONS
                          and p.get('BBType') not in BUNT_BB_TYPES)
    contact_non_bunt = sum(1 for p in pitches
                           if p['Description'] in ('Foul', 'In Play')
                           and p.get('BBType') not in BUNT_BB_TYPES)
    contact_pct = contact_non_bunt / swings_non_bunt if swings_non_bunt > 0 else None

    iz_swings_non_bunt = sum(1 for p in iz_pitches
                             if p['Description'] in SWING_DESCRIPTIONS
                             and p.get('BBType') not in BUNT_BB_TYPES)
    iz_contact = sum(1 for p in iz_pitches
                     if p['Description'] in ('Foul', 'In Play')
                     and p.get('BBType') not in BUNT_BB_TYPES)
    iz_contact_pct = iz_contact / iz_swings_non_bunt if iz_swings_non_bunt > 0 else None

    bip = [p for p in pitches if p.get('BBType') is not None and p.get('BBType') not in BUNT_BB_TYPES]
    n_bip = len(bip)
    gb = sum(1 for p in bip if p.get('BBType') == 'ground_ball')
    ld = sum(1 for p in bip if p.get('BBType') == 'line_drive')
    fb = sum(1 for p in bip if p.get('BBType') == 'fly_ball')
    pu = sum(1 for p in bip if p.get('BBType') == 'popup')

    all_evs = [safe_float(p.get('ExitVelo')) for p in bip]
    all_evs = [v for v in all_evs if v is not None]
    ev50 = None
    if all_evs:
        sorted_evs = sorted(all_evs, reverse=True)
        top_half = sorted_evs[:max(1, len(sorted_evs) // 2)]
        ev50 = round(sum(top_half) / len(top_half), 1)

    has_barrel_col = any(str(p.get('Barrel', '')).strip() != '' for p in bip)
    if has_barrel_col:
        barrels = sum(1 for p in bip if str(p.get('Barrel', '')).strip() == '6')
    else:
        ev_la_all = [(safe_float(p.get('ExitVelo')), safe_float(p.get('LaunchAngle')))
                     for p in bip
                     if safe_float(p.get('ExitVelo')) is not None
                     and safe_float(p.get('LaunchAngle')) is not None]
        barrels = sum(1 for ev, la in ev_la_all if is_barrel(ev, la))

    all_la = [safe_float(p.get('LaunchAngle')) for p in bip
              if safe_float(p.get('LaunchAngle')) is not None]

    spray_data = []
    all_spray = []   # raw spray angles for median computation
    for p in bip:
        hc_x = safe_float(p.get('HC_X'))
        hc_y = safe_float(p.get('HC_Y'))
        angle = spray_angle(hc_x, hc_y)
        if angle is not None:
            all_spray.append(angle)
        direction = spray_direction(angle, p.get('Bats'))
        if direction:
            spray_data.append((direction, p.get('BBType')))
    n_spray = len(spray_data)
    pull = sum(1 for d, _ in spray_data if d in ('pull', 'pull_side'))
    center = sum(1 for d, _ in spray_data if d in ('center_pull', 'center_oppo'))
    oppo = sum(1 for d, _ in spray_data if d in ('oppo_side', 'oppo'))
    air_pull = sum(1 for d, bb in spray_data if d in ('pull', 'pull_side') and bb in ('line_drive', 'fly_ball'))

    ev_valid = [safe_float(p.get('ExitVelo')) for p in bip]
    ev_valid = [v for v in ev_valid if v is not None]
    hard_hit = sum(1 for v in ev_valid if v >= 95)
    hard_hit_pct = hard_hit / len(ev_valid) if ev_valid else None

    # strict-conditional HR/FB — same definition as the pitcher side (see
    # compute_pitcher_batted_ball): fly-ball HRs over outfield flies.
    n_hr_fb = sum(1 for p in bip if p.get('Event') == 'Home Run'
                  and p.get('BBType') == 'fly_ball')
    hr_fb_pct = round(n_hr_fb / fb, 4) if fb > 0 else None

    two_strike_pitches = [p for p in pitches if '-' in p.get('Count', '') and p['Count'].split('-')[1] == '2']
    # Exclude bunt-contact swings from the 2-strike whiff denominator to match
    # the regular whiff% / contact% convention.
    two_strike_swings = sum(1 for p in two_strike_pitches
                            if p['Description'] in SWING_DESCRIPTIONS
                            and p.get('BBType') not in BUNT_BB_TYPES)
    two_strike_whiffs = sum(1 for p in two_strike_pitches if p['Description'] == 'Swinging Strike')
    two_strike_whiff_pct = two_strike_whiffs / two_strike_swings if two_strike_swings > 0 else None

    first_pitches_h = [p for p in pitches if p.get('Count') == '0-0']
    first_pitch_swings = sum(1 for p in first_pitches_h if p['Description'] in SWING_DESCRIPTIONS)
    first_pitch_swing_pct = first_pitch_swings / len(first_pitches_h) if first_pitches_h else None

    fb_distances = [safe_float(p.get('Distance')) for p in bip if p.get('BBType') == 'fly_ball']
    fb_distances = [d for d in fb_distances if d is not None]
    avg_fb_dist = round(sum(fb_distances) / len(fb_distances), 0) if fb_distances else None

    hr_distances = [safe_float(p.get('Distance')) for p in bip if p.get('Event') == 'Home Run']
    hr_distances = [d for d in hr_distances if d is not None]
    avg_hr_dist = round(sum(hr_distances) / len(hr_distances), 0) if hr_distances else None

    bs_vals = []
    sl_vals = []
    aa_vals = []
    ad_vals = []
    spt_vals = []
    for p in pitches:
        bs = safe_float(p.get('BatSpeed'))
        if bs is not None and bs >= 50:
            bs_vals.append(bs)
            sl = safe_float(p.get('SwingLength'))
            if sl is not None: sl_vals.append(sl)
            aa = safe_float(p.get('AttackAngle'))
            if aa is not None: aa_vals.append(aa)
            ad = safe_float(p.get('AttackDirection'))
            if ad is not None: ad_vals.append(ad)
            spt = safe_float(p.get('SwingPathTilt'))
            if spt is not None: spt_vals.append(spt)

    n_blasts = 0
    n_squared_up = 0
    n_blast_eligible = 0
    n_ideal_aa = 0
    for p in pitches:
        bs = safe_float(p.get('BatSpeed'))
        if bs is None or bs < 50:
            continue
        aa = safe_float(p.get('AttackAngle'))
        if aa is not None and 5 <= aa <= 20:
            n_ideal_aa += 1
        ev = safe_float(p.get('ExitVelo'))
        velo = safe_float(p.get('Velocity'))
        if ev is not None and velo is not None:
            n_blast_eligible += 1
            # Statcast max EV = 1.23 * bat speed + 0.23 * pitch speed AT THE
            # PLATE. We have release velo; plate velo ≈ 0.92 * release, so
            # 0.23 * 0.92 ≈ 0.212 is the release-velo coefficient. The bat
            # coefficient is Statcast's 1.23 exactly (1.2 previously ran our
            # squaredUpPct slightly hot vs Savant).
            max_ev = 0.212 * velo + 1.23 * bs
            # Squared-up = >= 80% of max possible EV given bat speed + pitch velo.
            # Blast = Squared-up AND bat speed >= 75 (i.e. Statcast's "elite contact"
            # gate adds the swing-speed requirement on top of the energy-transfer one).
            if max_ev > 0 and ev >= 0.80 * max_ev:
                n_squared_up += 1
                if bs >= 75:
                    n_blasts += 1

    return {
        'pa': n_pa,
        'ab': n_ab,
        'nSwings': n_swings,
        'nBip': n_bip,
        'doubles': n_2b,
        'triples': n_3b,
        'hr': n_hr,
        'xbh': xbh,
        'avgEVAll': round(sum(ev_valid) / len(ev_valid), 1) if ev_valid else None,
        'ev50': ev50,
        'maxEV': round(max(ev_valid), 1) if ev_valid else None,
        'medLA': round(median(all_la), 1) if all_la else None,
        'medSpray': round(median(all_spray), 1) if all_spray else None,
        'hardHitPct': round(hard_hit_pct, 4) if hard_hit_pct is not None else None,
        'babip': babip,
        'barrelPct': barrels / len(ev_valid) if ev_valid else None,
        'gbPct': gb / n_bip if n_bip > 0 else None,
        'ldPct': ld / n_bip if n_bip > 0 else None,
        'fbPct': fb / n_bip if n_bip > 0 else None,
        'puPct': pu / n_bip if n_bip > 0 else None,
        'hrFbPct': hr_fb_pct,
        'pullPct': pull / n_spray if n_spray > 0 else None,
        'middlePct': center / n_spray if n_spray > 0 else None,
        'oppoPct': oppo / n_spray if n_spray > 0 else None,
        'airPullPct': air_pull / n_bip if n_bip > 0 else None,
        'swingPct': n_swings / total if total > 0 else None,
        'izSwingPct': iz_swing_pct,
        'chasePct': chase_pct,
        'izSwChase': round(iz_swing_pct - chase_pct, 4) if iz_swing_pct is not None and chase_pct is not None else None,
        'contactPct': contact_pct,
        'izContactPct': iz_contact_pct,
        # Whiff% denominator excludes bunt-contact swings to match contactPct,
        # so whiffPct + contactPct = 1.0 cleanly. Bunt-attempt whiffs themselves
        # are indistinguishable from regular whiffs in the data (no BBType on
        # any swinging strike), so they remain counted in the numerator —
        # negligible effect since they're rare.
        'whiffPct': whiffs / swings_non_bunt if swings_non_bunt > 0 else None,
        'izWhiffPct': iz_whiffs / iz_swings_non_bunt if iz_swings_non_bunt > 0 else None,
        'runValue': (lambda vals: -sum(vals) if vals else None)([v for v in (safe_float(p.get('RunExp')) for p in pitches) if v is not None]),
        'batSpeed': round(sum(bs_vals) / len(bs_vals), 1) if bs_vals else None,
        'swingLength': round(sum(sl_vals) / len(sl_vals), 1) if sl_vals else None,
        'attackAngle': round(sum(aa_vals) / len(aa_vals), 1) if aa_vals else None,
        'attackDirection': round(sum(ad_vals) / len(ad_vals), 1) if ad_vals else None,
        'swingPathTilt': round(sum(spt_vals) / len(spt_vals), 1) if spt_vals else None,
        'nCompSwings': len(bs_vals),
        'blastPct': round(n_blasts / n_blast_eligible, 4) if n_blast_eligible > 0 else None,
        'squaredUpPct': round(n_squared_up / n_blast_eligible, 4) if n_blast_eligible > 0 else None,
        'idealAAPct': round(n_ideal_aa / len(bs_vals), 4) if bs_vals else None,
        'twoStrikeWhiffPct': two_strike_whiff_pct,
        'firstPitchSwingPct': first_pitch_swing_pct,
        'avgFbDist': avg_fb_dist,
        'avgHrDist': avg_hr_dist,
    }


# ── Percentile computation ───────────────────────────────────────────────

def compute_percentile_ranks(rows, metric_key, min_count=0, count_key='count', qualifier_fn=None, abs_val=False):
    """Compute percentile rank (0-100) for each row's metric value.

    Distribution is built from rows that:
    - have a non-null metric value, AND
    - pass min_count threshold (if min_count > 0), AND
    - pass qualifier_fn (if qualifier_fn is provided).

    Every row with a non-null metric value is ranked against that distribution,
    regardless of whether the row itself contributes to the pool. Non-qualified
    rows therefore still get a stored percentile rank (for tooltip display);
    the rendering layer decides separately whether to apply percentile coloring.

    abs_val=True ranks on |value| instead of the signed value. Required for
    hand-signed shape metrics (haa, nHAA, horzBrk, hbOE) stored in an absolute
    catcher/pitcher frame: ranking the signed value would rank handedness rather
    than quality (LHP negative, RHP positive). Mirrors the JS Aggregator's
    ABS_PCTL_KEYS so the server-emitted percentiles match the client re-agg.

    Uses the 'mean rank' method for ties. O(n log n) via sort + bisect.
    """
    import bisect
    pctl_key = metric_key + '_pctl'
    _v = (lambda x: abs(x)) if abs_val else (lambda x: x)

    def _row_in_pool(row):
        if row.get(metric_key) is None:
            return False
        if min_count > 0 and (row.get(count_key) or 0) < min_count:
            return False
        if qualifier_fn is not None and not qualifier_fn(row):
            return False
        return True

    sorted_vals = sorted(_v(rows[i][metric_key]) for i in range(len(rows)) if _row_in_pool(rows[i]))
    n = len(sorted_vals)

    if n < 2:
        for row in rows:
            row[pctl_key] = 50 if row.get(metric_key) is not None else None
        return

    def _pctl_from_sorted(val):
        val = _v(val)
        below = bisect.bisect_left(sorted_vals, val)
        above = bisect.bisect_right(sorted_vals, val)
        equal = above - below
        return max(0, min(100, round((below + 0.5 * (equal - 1)) / max(1, n - 1) * 100)))

    for row in rows:
        val = row.get(metric_key)
        row[pctl_key] = _pctl_from_sorted(val) if val is not None else None


def compute_percentile_ranks_with_aaa(rows, metric_key, min_count=0, count_key='count', qualifier_fn=None, abs_val=False):
    """Compute percentiles from an MLB pool of one row per player (combined 2TM/3TM
    rows replace their per-team rows), then interpolate AAA rows and the per-team
    rows of multi-team players against the pool.

    qualifier_fn (optional): when provided, the MLB pool that defines the
    distribution is restricted to rows where qualifier_fn(row) returns True.
    All rows (qualified MLB, non-qualified MLB, multi-team per-team, AAA/ROC)
    are still ranked against that distribution, so non-qualified rows have a
    percentile rank stored for tooltip display even though the rendering layer
    suppresses their cell coloring.
    """
    pctl_key = metric_key + '_pctl'

    def _player_key(r):
        # Prefer mlbId so two distinct players sharing a name (e.g. two "Max
        # Muncy") aren't grouped as stints of one multi-team player. Mirrors the
        # JS Aggregator._combinedKey so the server percentile pool and the client
        # re-aggregation route the same rows to interpolation.
        mid = r.get('mlbId')
        if mid is not None and mid != '':
            return 'id:' + str(mid)
        return 'nm:' + (r.get('pitcher') or r.get('hitter') or '')

    combined_players = {_player_key(r) for r in rows if r.get('_isCombined')}

    mlb_rows = []
    interp_rows = []
    for r in rows:
        if r.get('_isROC'):
            interp_rows.append(r)
            continue
        if not r.get('_isCombined') and _player_key(r) in combined_players:
            interp_rows.append(r)
            continue
        mlb_rows.append(r)

    compute_percentile_ranks(mlb_rows, metric_key, min_count, count_key, qualifier_fn=qualifier_fn, abs_val=abs_val)
    _v = (lambda x: abs(x)) if abs_val else (lambda x: x)

    def _mlb_in_pool(row):
        if row.get(metric_key) is None:
            return False
        if min_count > 0 and (row.get(count_key) or 0) < min_count:
            return False
        if qualifier_fn is not None and not qualifier_fn(row):
            return False
        return True

    mlb_values = sorted(_v(r[metric_key]) for r in mlb_rows if _mlb_in_pool(r))
    n = len(mlb_values)

    import bisect
    for row in interp_rows:
        val = row.get(metric_key)
        if val is None:
            row[pctl_key] = None
            continue
        if n < 2:
            row[pctl_key] = 50
            continue
        val = _v(val)
        below = bisect.bisect_left(mlb_values, val)
        above = bisect.bisect_right(mlb_values, val)
        equal = above - below
        pctl = (below + 0.5 * equal) / n * 100
        row[pctl_key] = max(0, min(100, round(pctl)))


# ── Micro-data generation ────────────────────────────────────────────────
# generate_micro_data is imported from process_data.py for now, as it's
# tightly coupled with the micro-data format (800+ lines).
# It will be moved here in a follow-up refactor.
