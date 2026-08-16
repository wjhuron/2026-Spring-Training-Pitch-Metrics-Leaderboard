"""statcast_hitter_adapter.py — public Statcast cache → Wally-schema pitch dicts
for the HITTER pipelines (pipeline_sdplus / pipeline_contact / BB+ ingredients).

Mirrors the field conventions the pipelines read (see pipeline_sdplus.is_eligible,
classify_zone, make_rv_xrv; derive_hitterplus_weights.classify):
  Description  'In Play' / 'Swinging Strike' / 'Foul' / 'Called Strike' / 'Ball'
               / 'Hit By Pitch' / 'Foul Bunt' / 'Missed Bunt' / 'Pitchout'
               (foul_tip → 'Swinging Strike', matching the feed pipeline where
               tips are counted as whiffs; blocked variants fold in)
  Count        pre-pitch 'B-S' string from public balls/strikes
  RunExp       pitcher-perspective: delta_pitcher_run_exp, else -delta_run_exp
               (same convention as build_historical_training_set)
  xwOBA        estimated_woba_using_speedangle
  InZone       computed via pipeline_utils.compute_in_zone (ball-radius adj)
  Event        only the values the pipelines test: 'Intent Walk'; everything
               else passes through as the raw snake_case in 'event_raw'
  BBType       public bb_type; sac_bunt / bunt events forced to 'bunt' so
               BUNT_BB_TYPES excludes them
  Batter       str(MLBAM batter id) — pools a hitter across teams by design
  BTeam        constant 'X' (no stint splitting in experiments)

Only regular-season ('R') pitches are kept.
"""
import os, sys, pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pipeline.utils import compute_in_zone


def sf(x):
    """NA-safe float: handles pandas NA/NaN, None, ''."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None

# FanGraphs Guts (lgwOBA, wOBAScale) per season — 2021-24 from
# build_historical_training_set.GUTS (confirmed by Wally 2026-07-05),
# 2025 from train_stuff.PRIOR_* constants.
GUTS = {
    2021: (0.314, 1.209), 2022: (0.310, 1.259),
    2023: (0.318, 1.204), 2024: (0.310, 1.242),
    2025: (0.3131, 1.2317),
}

DESC_MAP = {
    'hit_into_play': 'In Play', 'hit_into_play_no_out': 'In Play',
    'hit_into_play_score': 'In Play',
    'swinging_strike': 'Swinging Strike',
    'swinging_strike_blocked': 'Swinging Strike',
    'foul_tip': 'Swinging Strike',
    'foul': 'Foul',
    'called_strike': 'Called Strike',
    'ball': 'Ball', 'blocked_ball': 'Ball', 'automatic_ball': 'Ball',
    'hit_by_pitch': 'Hit By Pitch',
    'foul_bunt': 'Foul Bunt', 'missed_bunt': 'Missed Bunt',
    'bunt_foul_tip': 'Foul Bunt',
    'pitchout': 'Pitchout',
}

# Snake-case wOBA numerators for the prediction target (Statcast woba_value
# convention, mirroring process_data._bip_woba_value / derive_hitterplus_weights)
BIP_WOBA_PUB = {'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0,
                'field_error': 0.9, 'fielders_choice': 0.9,
                'fielders_choice_out': 0.9}
PA_WOBA_PUB = {'walk': 0.7, 'hit_by_pitch': 0.72}


def season_dicts(year):
    """Load data/_statcast{year}_cache.pkl → list of Wally-schema pitch dicts."""
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    has_dpre = 'delta_pitcher_run_exp' in df.columns
    out = []
    for r in df.itertuples(index=False):
        desc = DESC_MAP.get(r.description)
        if desc is None:
            continue
        ev = getattr(r, 'events', None)
        ev = ev if isinstance(ev, str) and ev else None
        bb = getattr(r, 'bb_type', None)
        bb = bb if isinstance(bb, str) and bb else None
        if ev in ('sac_bunt', 'sac_bunt_double_play', 'bunt_groundout',
                  'bunt_pop_out', 'bunt_lineout'):
            bb = 'bunt'
        runexp = None
        if has_dpre:
            runexp = sf(r.delta_pitcher_run_exp)
        if runexp is None:
            dre = sf(r.delta_run_exp)
            runexp = -dre if dre is not None else None
        b, s = sf(r.balls), sf(r.strikes)
        p = {
            '_source': 'MLB',
            'Game Date': str(r.game_date)[:10],
            'Batter': str(int(r.batter)) if sf(r.batter) is not None else None,
            'BTeam': 'X',
            'Bats': r.stand if isinstance(r.stand, str) and r.stand in ('L', 'R') else None,
            'Throws': r.p_throws if isinstance(r.p_throws, str) and r.p_throws in ('L', 'R') else None,
            'Pitch Type': r.pitch_type if isinstance(r.pitch_type, str) else None,
            'Description': desc,
            'Event': 'Intent Walk' if ev == 'intent_walk' else (ev or None),
            'event_raw': ev,
            'BBType': bb,
            'Count': (f"{int(b)}-{int(s)}"
                      if b is not None and s is not None else None),
            'PlateX': sf(r.plate_x), 'PlateZ': sf(r.plate_z),
            'SzTop': sf(r.sz_top), 'SzBot': sf(r.sz_bot),
            'RunExp': runexp,
            'xwOBA': sf(r.estimated_woba_using_speedangle),
            'HC_X': sf(r.hc_x), 'HC_Y': sf(r.hc_y),
            'LaunchAngle': sf(r.launch_angle),
            # ExitVelo was missing until 2026-08-16 while LaunchAngle was
            # mapped, so anything in this harness keyed on EV (xwOBA3D cells,
            # hard-hit / barrel rates) silently saw None on every batted ball.
            'ExitVelo': sf(r.launch_speed),
        }
        p['InZone'] = compute_in_zone(p)
        out.append(p)
    return out


def target_y(year):
    """dict[batter_id_str] -> (woba_sum, pa_events) for season `year`,
    straight from the DataFrame (no dict conversion needed)."""
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    acc = {}
    for r in df.itertuples(index=False):
        ev = getattr(r, 'events', None)
        if not isinstance(ev, str) or not ev or ev == 'intent_walk':
            continue
        val = None
        if r.description in ('hit_into_play', 'hit_into_play_no_out',
                             'hit_into_play_score'):
            val = BIP_WOBA_PUB.get(ev, 0.0)
        elif ev in PA_WOBA_PUB:
            val = PA_WOBA_PUB[ev]
        elif 'strikeout' in ev:
            val = 0.0
        if val is None:
            continue
        bid = str(int(r.batter)) if sf(r.batter) is not None else None
        if bid is None:
            continue
        a = acc.setdefault(bid, [0.0, 0])
        a[0] += val
        a[1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}
