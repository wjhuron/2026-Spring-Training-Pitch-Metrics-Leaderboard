"""era_battery_build.py — candidate metric battery per pitcher-season.

For the ERA-estimator screen (descriptive + predictive, 2021-2026).

Sources
-------
2021-2025: data/_statcastYYYY_cache.pkl (full-season Savant DataFrames,
           game_type R only, pitcher keyed by MLB id).
2026:      data/all_pitches_rs_cache.pkl (sheet pickle, retagged, MLB rows
           only), pitcher resolved to MLB id via lastFirst match against
           data/_era_targets.json names. Ambiguous names are dropped and
           logged.

Conventions mirror the site pipeline (pipeline_utils):
  * swings = SWING_DESCRIPTIONS (bunt attempts excluded)
  * whiff = 'Swinging Strike' (feed Foul Tip is normalized into this)
  * CSW = Called Strike + Swinging Strike
  * chase = OOZ pitch with Swinging Strike / In Play / Foul
  * InZone = compute_in_zone (exact rounded-rect + ball radius)
  * BIP excludes bunt BBTypes
  * barrel = pipeline_utils.is_barrel EV/LA proxy for ALL seasons
    (official launch_speed_angle is absent from the 21-25 caches; the
    proxy undercounts ~5% but is measured the same way every season)

Two scopes per pitcher-season: 'full' and 'h1' (game_date <= that
season's All-Star date, from _era_targets.json).

Output: data/_era_battery.json
  {season: {pid: {'full': {...}, 'h1': {...}}}}
"""
import gc
import json
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline_utils import (SWING_DESCRIPTIONS, HIT_EVENTS, K_EVENTS,
                            BB_EVENTS, HBP_EVENTS, SF_EVENTS, SH_EVENTS,
                            CI_EVENTS, NON_PA_EVENTS, BUNT_BB_TYPES,
                            compute_in_zone, is_barrel,
                            _fullname_to_lastfirst)

OUT = os.path.join(ROOT, 'data', '_era_battery.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))

# fixed linear weights (FG 2024-ish); the screen uses correlations, which
# are invariant to the overall scale, and near-invariant to small
# year-to-year weight drift. Not a shipped-metric choice.
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = .69, .72, .89, 1.27, 1.61, 2.10

DESC_MAP = {  # statcast -> sheet tokens
    'ball': 'Ball', 'blocked_ball': 'Ball', 'pitchout': 'Pitchout',
    'automatic_ball': 'Ball', 'called_strike': 'Called Strike',
    'automatic_strike': 'Called Strike', 'foul': 'Foul',
    'foul_bunt': 'Foul Bunt', 'bunt_foul_tip': 'Foul Bunt',
    'missed_bunt': 'Missed Bunt', 'swinging_strike': 'Swinging Strike',
    'swinging_strike_blocked': 'Swinging Strike',
    'foul_tip': 'Swinging Strike', 'hit_by_pitch': 'Hit By Pitch',
    'hit_into_play': 'In Play',
}
EVENT_MAP = {  # statcast -> sheet-style tokens (only ones the tallies use)
    'strikeout': 'Strikeout', 'strikeout_double_play': 'Strikeout Double Play',
    'walk': 'Walk', 'intent_walk': 'Intent Walk',
    'hit_by_pitch': 'Hit By Pitch', 'single': 'Single', 'double': 'Double',
    'triple': 'Triple', 'home_run': 'Home Run', 'sac_fly': 'Sac Fly',
    'sac_fly_double_play': 'Sac Fly Double Play', 'sac_bunt': 'Sac Bunt',
    'catcher_interf': 'Catcher Interference',
    'field_out': 'Groundout', 'force_out': 'Forceout',
    'grounded_into_double_play': 'Grounded Into DP',
    'double_play': 'Double Play', 'triple_play': 'Triple Play',
    'fielders_choice': 'Fielders Choice',
    'fielders_choice_out': 'Fielders Choice Out',
    'field_error': 'Field Error',
    'truncated_pa': None,
}


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def vaa_at_plate(vy0, vz0, ay, az):
    """Vertical approach angle at the front of the plate (y = 17/12 ft)."""
    try:
        disc = vy0 * vy0 - 2.0 * ay * (50.0 - 17.0 / 12.0)
        if disc <= 0:
            return None
        vy_f = -math.sqrt(disc)
        t = (vy_f - vy0) / ay
        vz_f = vz0 + az * t
        # Savant convention: negative = descending (vy_f < 0, vz_f < 0)
        return math.atan2(vz_f, -vy_f) * 180.0 / math.pi
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def adapt_statcast(path):
    df = pickle.load(open(path, 'rb'))
    df = df[df['game_type'] == 'R']
    cols = ['pitcher', 'game_date', 'pitch_type', 'description', 'events',
            'bb_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'balls',
            'strikes', 'launch_speed', 'launch_angle',
            'estimated_woba_using_speedangle', 'delta_pitcher_run_exp',
            'release_speed', 'release_extension', 'release_spin_rate',
            'arm_angle', 'pfx_z', 'vy0', 'vz0', 'ay', 'az']
    sub = df[cols]
    out = []
    for r in sub.itertuples(index=False):
        desc = DESC_MAP.get(r.description)
        if desc is None:
            continue
        ev = EVENT_MAP.get(r.events) if isinstance(r.events, str) else None
        try:
            b, s = int(r.balls), int(r.strikes)
            count = f'{b}-{s}'
        except (TypeError, ValueError):
            count = None
        pz = _f(r.pfx_z)
        out.append({
            'Pitcher': int(r.pitcher), 'Game Date': str(r.game_date)[:10],
            'Pitch Type': r.pitch_type if isinstance(r.pitch_type, str) else None,
            'Description': desc, 'Event': ev,
            'BBType': r.bb_type if isinstance(r.bb_type, str) else None,
            'PlateX': _f(r.plate_x), 'PlateZ': _f(r.plate_z),
            'SzTop': _f(r.sz_top), 'SzBot': _f(r.sz_bot), 'Count': count,
            'ExitVelo': _f(r.launch_speed), 'LaunchAngle': _f(r.launch_angle),
            'xwOBA': _f(r.estimated_woba_using_speedangle),
            'RunExp': _f(r.delta_pitcher_run_exp),
            'Velocity': _f(r.release_speed), 'Extension': _f(r.release_extension),
            'SpinRate': _f(r.release_spin_rate), 'ArmAngle': _f(r.arm_angle),
            'IndVertBrk': None if pz is None else pz * 12.0,
            'VAA': vaa_at_plate(_f(r.vy0), _f(r.vz0), _f(r.ay), _f(r.az)),
            'StuffPlus': None, 'LocPlus': None, 'PitchingPlus': None,
        })
    del df, sub
    gc.collect()
    return out


def build_2026_name_map():
    """lastFirst -> MLB id from the 2026 targets; ambiguous names dropped."""
    m, ambig = {}, set()
    for pid, rec in TARGETS['2026']['pitchers'].items():
        full = (rec['name'] or '').strip()
        variants = {_fullname_to_lastfirst(full)}
        parts = full.split()
        if len(parts) >= 3:
            # multi-word surname variant: 'Simeon Woods Richardson' ->
            # 'Woods Richardson, Simeon' (the sheet's rendering)
            variants.add(' '.join(parts[-2:]) + ', ' + ' '.join(parts[:-2]))
        for lf in variants:
            if lf in m and m[lf] != int(pid):
                ambig.add(lf)
            m[lf] = int(pid)
    for lf in ambig:
        del m[lf]
    if ambig:
        print(f'  2026 ambiguous names dropped: {sorted(ambig)}', flush=True)
    return m


def adapt_sheet(path):
    raw = pickle.load(open(path, 'rb'))
    name_map = build_2026_name_map()
    out, unmatched = [], defaultdict(int)
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        pid = name_map.get(p.get('Pitcher'))
        if pid is None:
            unmatched[p.get('Pitcher')] += 1
            continue
        out.append({
            'Pitcher': pid, 'Game Date': p.get('Game Date'),
            'Pitch Type': p.get('Pitch Type'),
            'Description': p.get('Description'), 'Event': p.get('Event'),
            'BBType': p.get('BBType'),
            'PlateX': _f(p.get('PlateX')), 'PlateZ': _f(p.get('PlateZ')),
            'SzTop': _f(p.get('SzTop')), 'SzBot': _f(p.get('SzBot')),
            'Count': p.get('Count'),
            'ExitVelo': _f(p.get('ExitVelo')),
            'LaunchAngle': _f(p.get('LaunchAngle')),
            'xwOBA': _f(p.get('xwOBA')), 'RunExp': _f(p.get('RunExp')),
            'Velocity': _f(p.get('Velocity')),
            'Extension': _f(p.get('Extension')),
            'SpinRate': _f(p.get('Spin Rate')), 'ArmAngle': _f(p.get('ArmAngle')),
            'IndVertBrk': _f(p.get('IndVertBrk')), 'VAA': _f(p.get('VAA')),
            'StuffPlus': _f(p.get('Stuff+')), 'LocPlus': _f(p.get('Loc+')),
            'PitchingPlus': _f(p.get('Pitching+')),
        })
    del raw
    gc.collect()
    big = {n: c for n, c in unmatched.items() if c >= 50}
    if big:
        print(f'  2026 unmatched pitchers (>=50 pitches): {len(big)}',
              flush=True)
    return out


def tally(pitches):
    """One pitcher's pitch list -> aggregate dict."""
    c = defaultdict(float)
    vaas, pzs = [], []
    for p in pitches:
        c['pitches'] += 1
        desc = p['Description']
        izs = compute_in_zone(p)
        iz = izs == 'Yes'
        if izs is not None:
            c['loc_pitches'] += 1
        if iz:
            c['iz'] += 1
        if desc in SWING_DESCRIPTIONS:
            c['sw'] += 1
            if iz:
                c['izsw'] += 1
                if desc != 'Swinging Strike':
                    c['izcon'] += 1
        if desc == 'Swinging Strike':
            c['wh'] += 1
        if desc in ('Called Strike', 'Swinging Strike'):
            c['csw'] += 1
        if desc == 'Called Strike':
            c['cs'] += 1
        if izs == 'No':
            c['ooz'] += 1
            if desc in ('Swinging Strike', 'In Play', 'Foul'):
                c['chase'] += 1
        if p['Count'] == '0-0':
            c['fp'] += 1
            if desc in ('Called Strike', 'Swinging Strike', 'Foul', 'In Play'):
                c['fps'] += 1
        v = p['Velocity']
        if v is not None:
            c['velo_n'] += 1
            c['velo_sum'] += v
            if v > c.get('velo_max', 0):
                c['velo_max'] = v
        for src, key in (('Extension', 'ext'), ('ArmAngle', 'arm'),
                         ('RunExp', 'rv'), ('StuffPlus', 'stuff'),
                         ('LocPlus', 'loc'), ('PitchingPlus', 'pplus')):
            x = p[src]
            if x is not None:
                c[key + '_n'] += 1
                c[key + '_sum'] += x
        if p['Pitch Type'] == 'FF':
            c['ff'] += 1
            if v is not None:
                c['ffv_n'] += 1
                c['ffv_sum'] += v
            ivb = p['IndVertBrk']
            if ivb is not None:
                c['ffivb_n'] += 1
                c['ffivb_sum'] += ivb
            sr = p['SpinRate']
            if sr is not None:
                c['ffspin_n'] += 1
                c['ffspin_sum'] += sr
            va, pz = p['VAA'], p['PlateZ']
            if va is not None and pz is not None:
                vaas.append(va)
                pzs.append(pz)
        bbt = p['BBType']
        ev = p['Event']
        if bbt and bbt not in BUNT_BB_TYPES:
            c['bip'] += 1
            if bbt == 'ground_ball':
                c['gb'] += 1
            elif bbt == 'fly_ball':
                c['fb'] += 1
            elif bbt == 'line_drive':
                c['ld'] += 1
            elif bbt == 'popup':
                c['pu'] += 1
            x = p['ExitVelo']
            la = p['LaunchAngle']
            if x is not None:
                c['ev_n'] += 1
                c['ev_sum'] += x
                if x >= 95:
                    c['hh'] += 1
                if la is not None and is_barrel(x, la):
                    c['brl'] += 1
            xw = p['xwOBA']
            if xw is not None:
                c['xwcon_n'] += 1
                c['xwcon_sum'] += xw
            if ev == 'Home Run' and bbt == 'fly_ball':
                c['hrfb'] += 1
        if ev and ev not in NON_PA_EVENTS:
            c['pa'] += 1
            if ev in K_EVENTS:
                c['k'] += 1
            elif ev in BB_EVENTS:
                c['bb'] += 1
                if ev == 'Intent Walk':
                    c['ibb'] += 1
            elif ev in HBP_EVENTS:
                c['hbp'] += 1
            elif ev in SF_EVENTS:
                c['sf'] += 1
            elif ev in SH_EVENTS:
                c['sh'] += 1
            elif ev in CI_EVENTS:
                c['ci'] += 1
            if ev == 'Single':
                c['h1b'] += 1
            elif ev == 'Double':
                c['h2b'] += 1
            elif ev == 'Triple':
                c['h3b'] += 1
            elif ev == 'Home Run':
                c['hr'] += 1
            # PA-level xwOBA numerator: BIP -> xwOBAcon, K -> 0,
            # BB/HBP -> weights. IBB excluded from both num and den.
            if ev == 'Intent Walk':
                pass
            elif ev in BB_EVENTS:
                c['xw_num'] += W_BB
                c['xw_den'] += 1
                c['w_num'] += W_BB
            elif ev in HBP_EVENTS:
                c['xw_num'] += W_HBP
                c['xw_den'] += 1
                c['w_num'] += W_HBP
            elif ev in K_EVENTS:
                c['xw_den'] += 1
            elif ev in SH_EVENTS or ev in CI_EVENTS:
                pass
            else:
                # BIP-ending PA (incl. SF) and errors
                xw = p['xwOBA']
                c['xw_den'] += 1
                c['xw_num'] += xw if xw is not None else 0.0
                w = {'Single': W_1B, 'Double': W_2B, 'Triple': W_3B,
                     'Home Run': W_HR}.get(ev, 0.0)
                c['w_num'] += w
    c['_vaas'] = vaas
    c['_pzs'] = pzs
    return c


def finalize(c, lg_slope, lg_pz):
    """Counts -> rates. Returns the metric dict for one scope."""
    def rt(num, den):
        return None if not c.get(den) else c[num] / c[den]

    def mean(prefix):
        n = c.get(prefix + '_n')
        return None if not n else c[prefix + '_sum'] / n

    m = {
        'pa': int(c.get('pa', 0)), 'pitches': int(c.get('pitches', 0)),
        'bip': int(c.get('bip', 0)),
        'k_pct': rt('k', 'pa'), 'bb_pct': rt('bb', 'pa'),
        'hbp_pct': rt('hbp', 'pa'), 'hr_pct': rt('hr', 'pa'),
        'csw_pct': rt('csw', 'pitches'), 'swstr_pct': rt('wh', 'pitches'),
        'whiff_pct': rt('wh', 'sw'), 'zone_pct': rt('iz', 'loc_pitches'),
        'chase_pct': rt('chase', 'ooz'), 'zcon_pct': rt('izcon', 'izsw'),
        'cstr_pct': rt('cs', 'pitches'), 'fps_pct': rt('fps', 'fp'),
        'gb_pct': rt('gb', 'bip'), 'fb_pct': rt('fb', 'bip'),
        'ld_pct': rt('ld', 'bip'), 'pu_pct': rt('pu', 'bip'),
        'hr_fb': rt('hrfb', 'fb'),
        'ev': mean('ev'), 'hh_pct': rt('hh', 'ev_n'),
        'brl_pct': rt('brl', 'ev_n'),
        'xwobacon': mean('xwcon'),
        'xwoba': rt('xw_num', 'xw_den'), 'woba': rt('w_num', 'xw_den'),
        'rv100': None if not c.get('rv_n') else
                 100.0 * c['rv_sum'] / c['rv_n'],
        'velo': mean('velo'), 'velo_max': c.get('velo_max') or None,
        'ff_velo': mean('ffv'), 'ff_ivb': mean('ffivb'),
        'ff_spin': mean('ffspin'), 'ext': mean('ext'), 'arm': mean('arm'),
        'stuff_plus': mean('stuff'), 'loc_plus': mean('loc'),
        'pitching_plus': mean('pplus'),
        'k_counts': {k: int(c.get(k, 0)) for k in
                     ('k', 'bb', 'ibb', 'hbp', 'hr', 'gb', 'fb', 'ld',
                      'pu', 'sf', 'sh', 'ci', 'h1b', 'h2b', 'h3b')},
    }
    vaas, pzs = c['_vaas'], c['_pzs']
    if len(vaas) >= 20:
        vm = sum(vaas) / len(vaas)
        pm = sum(pzs) / len(pzs)
        m['ff_vaa'] = vm
        m['ff_nvaa'] = vm - lg_slope * (pm - lg_pz)
    else:
        m['ff_vaa'] = None
        m['ff_nvaa'] = None
    return m


def league_vaa_fit(tallies):
    """OLS of per-pitch FF VAA on PlateZ across the whole season."""
    xs, ys = [], []
    for c in tallies.values():
        xs.extend(c['_pzs'])
        ys.extend(c['_vaas'])
    n = len(xs)
    if n < 100:
        return 0.0, 2.5
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return (sxy / sxx if sxx else 0.0), mx


def process_season(season, pitches):
    asg = TARGETS[str(season)]['asg']
    by_p = defaultdict(list)
    for p in pitches:
        by_p[p['Pitcher']].append(p)
    tall_full = {pid: tally(ps) for pid, ps in by_p.items()}
    tall_h1 = {pid: tally([p for p in ps if p['Game Date'] and
                           p['Game Date'] <= asg])
               for pid, ps in by_p.items()}
    slope, lg_pz = league_vaa_fit(tall_full)
    print(f'  {season}: {len(by_p)} pitchers | league VAA~PlateZ slope '
          f'{slope:.3f} @ mean pz {lg_pz:.2f}', flush=True)
    out = {}
    for pid in tall_full:
        out[str(pid)] = {
            'full': finalize(tall_full[pid], slope, lg_pz),
            'h1': finalize(tall_h1[pid], slope, lg_pz),
        }
    return out


def main():
    result = {}
    for season, path in [(2021, 'data/_statcast2021_cache.pkl'),
                         (2022, 'data/_statcast2022_cache.pkl'),
                         (2023, 'data/_statcast2023_cache.pkl'),
                         (2024, 'data/_statcast2024_cache.pkl'),
                         (2025, 'data/_statcast2025_full_cache.pkl')]:
        print(f'{season}: loading {path}', flush=True)
        pitches = adapt_statcast(os.path.join(ROOT, path))
        result[str(season)] = process_season(season, pitches)
        del pitches
        gc.collect()
    print('2026: loading sheet pickle', flush=True)
    pitches = adapt_sheet(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    result['2026'] = process_season(2026, pitches)
    del pitches
    gc.collect()
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
