"""weather_exponent_calibration.py — estimate the density->movement exponent
from our own cross-park data.

Model: |movement| = C(pitcher, pitch_type) x rho^e
  =>  log|mov| = e·log(rho) + const per (pitcher, throws, pitch_type)

Estimator: pooled within-group (fixed-effects) OLS of demeaned log|movement|
on demeaned log(rho), so pitcher identity and pitch shape drop out and only
cross-game density variation identifies e. Robustness pass adds calendar
month to the group key (kills any seasonal movement-drift confound; density
still varies within a month via road trips and weather).

Theory (Nathan trajectory physics) says e ~ 1.0-1.1; the shipped adjustment
uses 2/3. This script decides which the data supports.

Per-game density inputs are fetched once from the Stats API (slim fields=
query) and cached to data/game_weather_rs.json — the same sidecar
Pitcher2026.py now maintains for new games.

Usage: python3 scripts/research/misc/weather_exponent_calibration.py
"""
import os, sys, json, math, pickle
from collections import defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from scrapers.Pitcher2026 import (compute_air_density, VENUE_ELEVATION_FT_OVERRIDE,
                                  DEFAULT_TEMP_F, ROOF_CLOSED_TEMP_F, ROOF_CLOSED_CONDITIONS)

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
SIDECAR = os.path.join(ROOT, 'data', 'game_weather_rs.json')
FEED = ('https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        '?fields=gameData,weather,temp,condition,venue,id,location,elevation')


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_game_weather(pks):
    try:
        store = json.load(open(SIDECAR))
    except (FileNotFoundError, ValueError):
        store = {}
    missing = [pk for pk in pks if str(pk) not in store]
    print(f'{len(pks)} games; {len(missing)} need fetching')
    s = requests.Session()
    for i, pk in enumerate(missing):
        try:
            r = s.get(FEED.format(pk=pk), timeout=15)
            gd = r.json().get('gameData', {})
            w = gd.get('weather', {})
            venue = gd.get('venue', {})
            elev = venue.get('location', {}).get('elevation')
            if elev is None:
                elev = VENUE_ELEVATION_FT_OVERRIDE.get(venue.get('id'))
            store[str(pk)] = {'venueId': venue.get('id'),
                              'condition': w.get('condition'),
                              'tempF': sf(w.get('temp')),
                              'elevationFt': sf(elev)}
        except Exception as e:
            store[str(pk)] = {'error': str(e)}
        if (i + 1) % 200 == 0:
            print(f'  fetched {i + 1}/{len(missing)}', flush=True)
            json.dump(store, open(SIDECAR, 'w'), indent=0, sort_keys=True)
    json.dump(store, open(SIDECAR, 'w'), indent=0, sort_keys=True)
    return store


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


def fe_slope(rows, group_idx):
    """Pooled within-group OLS slope of y on x. rows: (groupkey, x, y)."""
    sums = {}
    for k, x, y in rows:
        s = sums.setdefault(k, [0.0, 0.0, 0])
        s[0] += x; s[1] += y; s[2] += 1
    sxx = sxy = 0.0
    n_used = 0
    for k, x, y in rows:
        s = sums[k]
        if s[2] < 2:
            continue
        dx = x - s[0] / s[2]; dy = y - s[1] / s[2]
        sxx += dx * dx; sxy += dx * dy
        n_used += 1
    return (sxy / sxx if sxx > 1e-12 else None), n_used


def main():
    D = pickle.load(open(PKL, 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    pks = sorted({(p.get('PitchID') or '').split('_')[0] for p in mlb} - {''})
    store = fetch_game_weather(pks)

    rho_by_pk = {pk: rho_of(store.get(str(pk))) for pk in pks}
    n_rho = sum(1 for v in rho_by_pk.values() if v)
    print(f'density available for {n_rho}/{len(pks)} games')

    rows_ivb, rows_hb, rows_tot = [], [], []
    for p in mlb:
        pk = (p.get('PitchID') or '').split('_')[0]
        rho = rho_by_pk.get(pk)
        iv, hb = sf(p.get('IndVertBrk')), sf(p.get('HorzBrk'))
        if rho is None or iv is None or hb is None:
            continue
        month = (p.get('Game Date') or '')[:7]
        key = (p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'))
        lr = math.log(rho)
        if abs(iv) >= 3.0:
            rows_ivb.append((key, lr, math.log(abs(iv))))
            rows_ivb.append(((key, month), lr, math.log(abs(iv))))  # tagged below
        if abs(hb) >= 3.0:
            rows_hb.append((key, lr, math.log(abs(hb))))
        tot = math.hypot(iv, hb)
        if tot >= 4.0:
            rows_tot.append((key, lr, math.log(tot)))
            rows_tot.append(((key, month), lr, math.log(tot)))

    # separate plain vs month-tagged (tuple-of-tuple keys)
    def split(rows):
        plain = [(k, x, y) for k, x, y in rows if not (isinstance(k, tuple) and isinstance(k[0], tuple))]
        montht = [(k, x, y) for k, x, y in rows if isinstance(k, tuple) and isinstance(k[0], tuple)]
        return plain, montht

    ivb_p, ivb_m = split(rows_ivb)
    tot_p, tot_m = split(rows_tot)

    for name, rows in [('|IVB| within (pitcher,pt)', ivb_p),
                       ('|IVB| within (pitcher,pt,month)', ivb_m),
                       ('|HB| within (pitcher,pt)', rows_hb),
                       ('total break within (pitcher,pt)', tot_p),
                       ('total break within (pitcher,pt,month)', tot_m)]:
        e, n = fe_slope(rows, 0)
        print(f'{name:38s} e = {e:+.3f}  (n={n})' if e is not None else f'{name}: n/a')

    print('\nshipped exponent: 0.667 | theory: ~1.0-1.1')
    print('NOTE: movement here is RAW (IndVertBrk/HorzBrk), so the estimate is')
    print('independent of the shipped adjustment.')


if __name__ == '__main__':
    main()
