"""build_2025_training_set.py — assemble the 2025 season into a Stuff+
training-ready pickle from Wally's retagged 2025 sheets + public Statcast.

The 2025 AL/NL workbooks carry the RETAGGED pitch types plus most physical
features (incl. arm angle), but lack VAA, per-pitch xwOBA, RunExp, and any
game/pitch ID. All of those exist in public Statcast, so:

  1. read all 30 team tabs from the two 2025 workbooks
  2. download public 2025 Statcast (pybaseball, cached locally)
  3. join per (game date, pitcher name): fingerprint match on
     (release_speed, plate_x, plate_z) with tolerances
  4. from the matched public row take: game_pk, kinematics -> VAA,
     estimated_woba_using_speedangle -> xwOBA, delta run exp -> RunExp
     (pitcher-perspective, matching the 2026 sheets' sign convention)
  5. density-adjust movement at WEATHER_EXPONENT using per-game weather
     (cached in data/game_weather_2025.json, same sidecar pattern)
  6. write data/_pitches2025_training.pkl (gitignored) with 2026-cache-
     shaped dicts, _source='MLB2025'

Usage: python3 scripts/builders/build_2025_training_set.py
"""
import os, sys, json, math, time, pickle
from collections import defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipeline.fetch import _gspread_client, read_sheet_with_retry
from scrapers.Pitcher2026 import (compute_air_density, compute_weather_adj_factor,
                                  VENUE_ELEVATION_FT_OVERRIDE, DEFAULT_TEMP_F,
                                  ROOF_CLOSED_TEMP_F, ROOF_CLOSED_CONDITIONS)

SHEET_IDS = {
    'AL2025': '1ayBv-pQ7tJMdMTKcVils2yShCt3wDozhSW00mNQiKH8',
    'NL2025': '1xT1Knb0c5pWLXJ7zJn4ETu3D3_qsTfc9yzXEElwAgts',
}
STATCAST_CACHE = os.path.join(ROOT, 'data', '_statcast2025_cache.pkl')
WEATHER_SIDECAR = os.path.join(ROOT, 'data', 'game_weather_2025.json')
OUT = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
FEED = ('https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        '?fields=gameData,weather,temp,condition,venue,id,location,elevation')

TEAM_TABS = ['ATH', 'BAL', 'BOS', 'CLE', 'CWS', 'DET', 'HOU', 'KCR', 'LAA',
             'MIN', 'NYY', 'SEA', 'TBR', 'TEX', 'TOR',
             'ARI', 'ATL', 'CHC', 'CIN', 'COL', 'LAD', 'MIA', 'MIL', 'NYM',
             'PHI', 'PIT', 'SDP', 'SFG', 'STL', 'WSH']

# raw MLB descriptions -> 2026 pipeline vocabulary (only what the trainer
# and downstream eligibility need)
DESC_MAP = {
    'In play, out(s)': 'In Play', 'In play, no out': 'In Play',
    'In play, run(s)': 'In Play',
    'Ball In Dirt': 'Ball', 'Swinging Strike (Blocked)': 'Swinging Strike',
    'Foul Tip': 'Swinging Strike',
    'Missed Bunt': 'Missed Bunt', 'Foul Bunt': 'Foul Bunt',
}


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def read_2025_sheets():
    gc = _gspread_client()
    pitches = []
    for name, sid in SHEET_IDS.items():
        sh = gc.open_by_key(sid)
        print(f'{sh.title}:', flush=True)
        for ws in sh.worksheets():
            if ws.title not in TEAM_TABS:
                print(f'  skipping {ws.title}')
                continue
            rows = read_sheet_with_retry(ws)
            time.sleep(0.6)
            header = rows[0]
            ci = {n: i for i, n in enumerate(header) if n}
            n0 = len(pitches)
            for row in rows[1:]:
                def cell(k):
                    i = ci.get(k)
                    return row[i] if i is not None and i < len(row) else None
                if not cell('player_name'):
                    continue
                pitches.append({
                    'Game Date': cell('Game Date'), 'PTeam': ws.title,
                    'Pitcher': cell('player_name'), 'Throws': cell('throws'),
                    'Bats': cell('stands'), 'Pitch Type': cell('pitch_type'),
                    'Velocity': sf(cell('release_speed')),
                    'Spin Rate': sf(cell('release_spin_rate')),
                    'IndVertBrk': sf(cell('IndVrtBrk')),
                    'HorzBrk': sf(cell('HorzBrk')),
                    'RelPosZ': sf(cell('release_pos_z')),
                    'RelPosX': sf(cell('release_pos_x')),
                    'Extension': sf(cell('release_extension')),
                    'ArmAngle': sf(cell('arm_angle')),
                    'PlateZ': sf(cell('plate_z')), 'PlateX': sf(cell('plate_x')),
                    'Description': DESC_MAP.get(cell('description'), cell('description')),
                })
            print(f'  {ws.title}: {len(pitches) - n0} pitches')
    return pitches


def load_statcast():
    if os.path.exists(STATCAST_CACHE):
        print('loading cached public statcast 2025 ...')
        return pickle.load(open(STATCAST_CACHE, 'rb'))
    from pybaseball import statcast
    import pandas as pd
    print('downloading public statcast 2025 (chunked by pybaseball) ...', flush=True)
    df = statcast(start_dt='2025-03-18', end_dt='2025-09-28', verbose=False)
    keep = ['game_date', 'player_name', 'game_pk', 'release_speed', 'plate_x',
            'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
            'estimated_woba_using_speedangle', 'delta_run_exp',
            'delta_pitcher_run_exp', 'game_type']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    df['game_date'] = df['game_date'].astype(str).str[:10]
    pickle.dump(df, open(STATCAST_CACHE, 'wb'))
    print(f'  cached {len(df)} public pitches')
    return df


def vaa_of(vy0, vz0, ay, az):
    if None in (vy0, vz0, ay, az) or ay == 0:
        return None
    disc = vy0 * vy0 - 2 * ay * (50 - 17 / 12)
    if disc <= 0:
        return None
    vy_f = -math.sqrt(disc)
    t = (vy_f - vy0) / ay
    vz_f = vz0 + az * t
    if vy_f == 0:
        return None
    return round(-math.atan(vz_f / vy_f) * 180 / math.pi, 2)


def join(pitches, df):
    """Per (date, pitcher): greedy fingerprint match on velo + plate coords."""
    pub = defaultdict(list)
    cols = {c: i for i, c in enumerate(df.columns)}
    has_dpre = 'delta_pitcher_run_exp' in cols
    for row in df.itertuples(index=False):
        pub[(row.game_date, row.player_name)].append(row)

    matched = unmatched = 0
    for key, mine in _group(pitches).items():
        cands = pub.get(key, [])
        used = [False] * len(cands)
        for p in mine:
            best_i, best_d = None, 1e9
            v, px, pz = p['Velocity'], p['PlateX'], p['PlateZ']
            if v is None:
                unmatched += 1
                continue
            for i, c in enumerate(cands):
                if used[i]:
                    continue
                cv, cx, cz = sf(c.release_speed), sf(c.plate_x), sf(c.plate_z)
                if cv is None or abs(cv - v) > 0.25:
                    continue
                d = abs(cv - v) * 2.0
                if px is not None and cx is not None:
                    d += abs(cx - px)
                if pz is not None and cz is not None:
                    d += abs(cz - pz)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is None or best_d > 0.5:
                unmatched += 1
                continue
            used[best_i] = True
            c = cands[best_i]
            matched += 1
            p['_game_pk'] = int(c.game_pk)
            p['VAA'] = vaa_of(sf(c.vy0), sf(c.vz0), sf(c.ay), sf(c.az))
            xw = sf(c.estimated_woba_using_speedangle)
            p['xwOBA'] = xw
            if has_dpre and sf(c.delta_pitcher_run_exp) is not None:
                p['RunExp'] = sf(c.delta_pitcher_run_exp)
            else:
                dre = sf(c.delta_run_exp)
                p['RunExp'] = -dre if dre is not None else None
    print(f'join: {matched} matched, {unmatched} unmatched '
          f'({matched / max(matched + unmatched, 1):.1%} match rate)')
    return pitches


def _group(pitches):
    g = defaultdict(list)
    for p in pitches:
        g[(p['Game Date'], p['Pitcher'])].append(p)
    return g


def add_weather(pitches):
    try:
        store = json.load(open(WEATHER_SIDECAR))
    except (FileNotFoundError, ValueError):
        store = {}
    pks = sorted({p.get('_game_pk') for p in pitches if p.get('_game_pk')})
    missing = [pk for pk in pks if str(pk) not in store]
    print(f'{len(pks)} games; fetching weather for {len(missing)}', flush=True)
    s = requests.Session()
    for i, pk in enumerate(missing):
        try:
            r = s.get(FEED.format(pk=pk), timeout=15)
            gd = r.json().get('gameData', {})
            w, venue = gd.get('weather', {}), gd.get('venue', {})
            elev = venue.get('location', {}).get('elevation')
            if elev is None:
                elev = VENUE_ELEVATION_FT_OVERRIDE.get(venue.get('id'))
            store[str(pk)] = {'venueId': venue.get('id'),
                              'condition': w.get('condition'),
                              'tempF': sf(w.get('temp')), 'elevationFt': sf(elev)}
        except Exception as e:
            store[str(pk)] = {'error': str(e)}
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(missing)}', flush=True)
            json.dump(store, open(WEATHER_SIDECAR, 'w'), indent=0, sort_keys=True)
    json.dump(store, open(WEATHER_SIDECAR, 'w'), indent=0, sort_keys=True)

    def rho_of(info):
        if not info or info.get('error') or info.get('elevationFt') is None:
            return None
        temp = info.get('tempF')
        cond = (info.get('condition') or '').strip().lower()
        if cond in ROOF_CLOSED_CONDITIONS:
            temp = ROOF_CLOSED_TEMP_F
        if temp is None:
            temp = DEFAULT_TEMP_F
        return compute_air_density(info['elevationFt'], temp)

    factor_by_pk = {}
    for pk in pks:
        rho = rho_of(store.get(str(pk)))
        factor_by_pk[pk] = compute_weather_adj_factor(rho) if rho else 1.0
    for p in pitches:
        f = factor_by_pk.get(p.get('_game_pk'), 1.0)
        iv, hb = p.get('IndVertBrk'), p.get('HorzBrk')
        p['xIndVrtBrk'] = round(iv * f, 1) if iv is not None else None
        p['xHorzBrk'] = round(hb * f, 1) if hb is not None else None
    return pitches


def main():
    pitches = read_2025_sheets()
    print(f'total 2025 sheet pitches: {len(pitches)}')
    df = load_statcast()
    pitches = join(pitches, df)
    pitches = [p for p in pitches if p.get('_game_pk')]
    pitches = add_weather(pitches)
    for p in pitches:
        p['_source'] = 'MLB2025'
    with open(OUT, 'wb') as f:
        pickle.dump(pitches, f)
    n_vaa = sum(1 for p in pitches if p.get('VAA') is not None)
    n_re = sum(1 for p in pitches if p.get('RunExp') is not None)
    n_xw = sum(1 for p in pitches if p.get('Description') == 'In Play' and p.get('xwOBA') is not None)
    n_bip = sum(1 for p in pitches if p.get('Description') == 'In Play')
    print(f'saved {len(pitches)} pitches -> {OUT}')
    print(f'coverage: VAA {n_vaa/len(pitches):.1%}, RunExp {n_re/len(pitches):.1%}, '
          f'xwOBA on BIP {n_xw/max(n_bip,1):.1%}')


if __name__ == '__main__':
    main()
