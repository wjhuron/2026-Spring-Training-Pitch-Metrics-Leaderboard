"""Find and fill games that have pitches in the cache but no weather entry.

Distinct from the 373 null-rho entries in game_weather_rs.json: those are
spring-training and other-org MiLB games the scraper touched whose pitches were
never loaded into the leaderboard cache, so they affect nothing. The games that
DO matter are ones with pitches but no sidecar entry at all -- they silently
fall through to factor 1.0, so their xIndVrtBrk/xHorzBrk are raw while every
other game's are density-adjusted.

Run with --apply to fetch and write. Without it, reports only.

Filling the sidecar does NOT retroactively fix the movement columns already
stored for those games; it only makes the correct factor available. Backfilling
IVB/HB is a separate job.
"""
import os, sys, json, pickle
from collections import Counter, defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from scrapers.Pitcher2026 import (compute_air_density, compute_weather_adj_factor,
                                  VENUE_ELEVATION_FT_OVERRIDE, DEFAULT_TEMP_F,
                                  ROOF_CLOSED_TEMP_F, ROOF_CLOSED_CONDITIONS)

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
SIDECAR = os.path.join(ROOT, 'data', 'game_weather_rs.json')
FEED = ('https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        '?fields=gameData,weather,temp,condition,venue,id,location,elevation')


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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


def fetch(pk, session):
    r = session.get(FEED.format(pk=pk), timeout=20)
    gd = r.json().get('gameData', {})
    w = gd.get('weather', {}) or {}
    venue = gd.get('venue', {}) or {}
    elev = (venue.get('location', {}) or {}).get('elevation')
    if elev is None:
        elev = VENUE_ELEVATION_FT_OVERRIDE.get(venue.get('id'))
    return {'venueId': venue.get('id'), 'condition': w.get('condition'),
            'tempF': sf(w.get('temp')), 'elevationFt': sf(elev)}


def main(apply_it):
    store = json.load(open(SIDECAR))
    D = pickle.load(open(PKL, 'rb'))
    games = Counter()
    meta = {}
    for p in D:
        pid = p.get('PitchID')
        if not pid or '_' not in str(pid):
            continue
        gp = str(pid).split('_')[0]
        games[gp] += 1
        if gp not in meta:
            meta[gp] = (p.get('_source'), p.get('Game Date'), p.get('PTeam'))

    absent = [g for g in games if g not in store]
    nullrho = [g for g in games if g in store and rho_of(store[g]) is None]
    print(f'{len(games)} games with pitches in the cache')
    print(f'  no sidecar entry at all : {len(absent):>4}  '
          f'({sum(games[g] for g in absent):,} pitches)')
    print(f'  entry but no usable rho : {len(nullrho):>4}  '
          f'({sum(games[g] for g in nullrho):,} pitches)')
    need = absent + nullrho
    if not need:
        print('\nnothing to fill')
        return
    print(f'\n{"game_pk":>9} {"src":>4} {"date":>11} {"team":>4} {"pitches":>8}')
    for g in sorted(need, key=lambda x: -games[x]):
        s, dt, tm = meta[g]
        print(f'{g:>9} {str(s):>4} {str(dt):>11} {str(tm):>4} {games[g]:>8}')

    if not apply_it:
        print('\n(dry run -- pass --apply to fetch and write)')
        return

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    filled, failed = 0, []
    factors = []
    for g in need:
        try:
            info = fetch(g, session)
        except Exception as e:
            failed.append((g, str(e)))
            continue
        rho = rho_of(info)
        if rho is None:
            failed.append((g, f'no elevation (venue {info.get("venueId")})'))
            store[g] = info
            continue
        info['rho'] = round(rho, 5)
        info['factor'] = round(compute_weather_adj_factor(rho), 5)
        store[g] = info
        factors.append((g, info['factor'], games[g]))
        filled += 1
    json.dump(store, open(SIDECAR, 'w'), indent=0, sort_keys=True)
    print(f'\nfilled {filled}/{len(need)}; wrote {SIDECAR}')
    if factors:
        print(f'\n{"game_pk":>9} {"factor":>8} {"pitches":>8}  '
              f'movement shift on a 15" break')
        for g, f, n in sorted(factors, key=lambda x: -abs(x[1] - 1)):
            print(f'{g:>9} {f:>8.5f} {n:>8}  {15*(f-1):>+6.2f}"')
        w = sum(f * n for _, f, n in factors) / sum(n for _, _, n in factors)
        print(f'\npitch-weighted mean factor {w:.5f} -> a 15" break moves '
              f'{15*(w-1):.2f}" on average')
    for g, e in failed:
        print(f'  FAILED {g}: {e}')


if __name__ == '__main__':
    main('--apply' in sys.argv)
