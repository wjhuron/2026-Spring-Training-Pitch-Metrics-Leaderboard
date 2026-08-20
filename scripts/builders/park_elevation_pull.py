"""park_elevation_pull.py — game_pk -> venue elevation, MLB and Triple-A.

Needed because Stuff+ reads movement, and thin air reduces movement. There
is no MiLB weather sidecar, so any cross-level Stuff+ comparison fed raw
movement is confounded: Triple-A has FIVE parks above 3,000 ft against one
in MLB, so raw movement makes Triple-A stuff look worse than it is.

Elevations are MEASURED, not assumed. Venue coordinates come from the MLB
Stats API (which serves `elevation: null` for every venue, hence the second
hop) and the elevation from api.open-elevation.com. The Triple-A
distribution has a clean natural break rather than a chosen threshold:

    ABQ 5151, SL 4895, RNO 4491, ELP 3747, LV 3025 | then OKC 1214 and down

so "above 3,000 ft" separates five clubs from the other 25 at a gap in the
data, not at a number picked to make something pass.

Output: data/_park_elevation.json
    {'games': {game_pk: elev_ft}, 'venues': {venue_id: {name, elev_ft}}}

    python3 scripts/builders/park_elevation_pull.py
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', '_park_elevation.json')
SEASONS = (2023, 2024, 2025, 2026)
SPORTS = (1, 11)                      # MLB, Triple-A
UA = 'Huronalytics-Research/1.0 (baseball research; https://huronalytics.com)'


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    games = {}
    venues = {}
    for sport in SPORTS:
        for y in SEASONS:
            d = get(f'https://statsapi.mlb.com/api/v1/schedule'
                    f'?sportId={sport}&season={y}&gameType=R')
            n = 0
            for day in d.get('dates', []):
                for g in day.get('games', []):
                    v = g.get('venue') or {}
                    if v.get('id') is None:
                        continue
                    games[str(g['gamePk'])] = int(v['id'])
                    venues.setdefault(int(v['id']), {'name': v.get('name')})
                    n += 1
            print(f'sport {sport} {y}: {n} regular-season games', flush=True)

    need = [vid for vid, r in venues.items() if 'lat' not in r]
    for vid in need:
        try:
            d = get(f'https://statsapi.mlb.com/api/v1/venues/{vid}'
                    f'?hydrate=location', timeout=60)
            loc = (d['venues'][0].get('location') or {})
            c = loc.get('defaultCoordinates') or {}
            venues[vid]['name'] = d['venues'][0].get('name')
            venues[vid]['lat'] = c.get('latitude')
            venues[vid]['lon'] = c.get('longitude')
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as e:
            print(f'  venue {vid}: {e}')
    have = [(vid, r) for vid, r in venues.items() if r.get('lat') is not None]
    print(f'{len(have)}/{len(venues)} venues have coordinates', flush=True)

    # one batch to open-elevation; a partial answer would silently leave some
    # parks at sea level, so refuse rather than half-fill
    res = post('https://api.open-elevation.com/api/v1/lookup',
               {'locations': [{'latitude': r['lat'], 'longitude': r['lon']}
                              for _, r in have]})['results']
    if len(res) != len(have):
        sys.exit(f'ABORT: elevation service returned {len(res)} of '
                 f'{len(have)} venues. A partial answer would leave parks '
                 f'silently at sea level.')
    for (vid, r), e in zip(have, res):
        r['elev_ft'] = round(e['elevation'] * 3.28084)
    missing = [vid for vid, r in venues.items() if r.get('elev_ft') is None]
    if missing:
        print(f'WARNING: no elevation for venues {missing} — games there '
              f'carry no elevation and must be excluded, not defaulted')

    out = {'games': {gp: venues[vid].get('elev_ft')
                     for gp, vid in games.items()
                     if venues.get(vid, {}).get('elev_ft') is not None},
           'venues': {str(vid): {'name': r.get('name'),
                                 'elev_ft': r.get('elev_ft')}
                      for vid, r in venues.items()}}
    with open(OUT + '.tmp', 'w') as f:
        json.dump(out, f)
    os.replace(OUT + '.tmp', OUT)
    hi = sorted(((r['elev_ft'], r['name']) for r in venues.values()
                 if r.get('elev_ft') is not None), reverse=True)[:8]
    print(f"\n{len(out['games'])} games mapped to an elevation")
    print('highest parks:')
    for ft, nm in hi:
        print(f'   {ft:6d} ft  {nm}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
