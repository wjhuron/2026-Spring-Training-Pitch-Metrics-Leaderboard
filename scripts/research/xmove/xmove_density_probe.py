"""How wrong is the current air-density model, really?

Pitcher2026.compute_air_density estimates pressure from ELEVATION via the dry
barometric formula and takes temperature from the feed. Two things are missing:
actual barometric pressure on the day (weather systems move it a few percent)
and humidity (moist air is LESS dense than dry, since water vapour is lighter
than N2/O2).

Before changing anything, measure the size of the error. Movement scales as
rho^1.05, so a density error of x% is a movement error of ~x%, i.e. 0.15" per
1% on a 15" break. If the error is a small fraction of the 0.199" venue offsets
already measured, item 6 is not worth a pipeline dependency.

Ground truth: Open-Meteo's archive (free, no key) at the venue's own
coordinates, hourly measured temperature, relative humidity and surface
pressure.
"""
import os, sys, json, math, random, time
import urllib.request, urllib.parse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from scrapers.Pitcher2026 import compute_air_density, ROOF_CLOSED_CONDITIONS, ROOF_CLOSED_TEMP_F

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
FEED = ('https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?fields=gameData,'
        'weather,temp,condition,venue,id,name,location,elevation,'
        'defaultCoordinates,latitude,longitude,datetime,dateTime,officialDate')


def moist_density(p_hpa, temp_f, rh_pct):
    """Air density from measured pressure, temperature and relative humidity.

    Partial pressures: dry air and water vapour have different gas constants
    (287.05 vs 461.495 J/kg/K), so vapour DISPLACING dry air lowers density.
    Saturation vapour pressure via the Magnus/Alduchov-Eskridge form.
    """
    T = (temp_f - 32) * 5 / 9 + 273.15
    P = p_hpa * 100.0
    es = 610.94 * math.exp(17.625 * (T - 273.15) / (T - 30.11))
    e = max(0.0, min(rh_pct, 100.0)) / 100.0 * es
    return (P - e) / (287.05 * T) + e / (461.495 * T)


def game_meta(pk):
    with urllib.request.urlopen(FEED.format(pk=pk), timeout=45) as r:
        gd = json.load(r).get('gameData', {})
    v = gd.get('venue', {}) or {}
    loc = v.get('location', {}) or {}
    co = loc.get('defaultCoordinates') or {}
    w = gd.get('weather', {}) or {}
    dt = (gd.get('datetime') or {})
    return dict(pk=pk, venue=v.get('id'), name=v.get('name'),
                elev=loc.get('elevation'), lat=co.get('latitude'),
                lon=co.get('longitude'), temp=w.get('temp'),
                cond=w.get('condition'), date=dt.get('officialDate'),
                utc=dt.get('dateTime'))


def archive(lat, lon, utc_iso):
    """Hourly obs spanning the game's UTC instant.

    officialDate is the LOCAL date; a night game's UTC timestamp rolls to the
    next day. Querying officialDate in UTC and matching the hour therefore
    lands ~24h early, which shows up as 15-25F temperature "errors". So span
    the UTC date and the day either side, and match the full timestamp.
    """
    from datetime import datetime, timedelta
    t = datetime.strptime(utc_iso[:19], '%Y-%m-%dT%H:%M:%S')
    q = urllib.parse.urlencode(dict(
        latitude=lat, longitude=lon,
        start_date=(t - timedelta(days=1)).strftime('%Y-%m-%d'),
        end_date=(t + timedelta(days=1)).strftime('%Y-%m-%d'),
        hourly='temperature_2m,relative_humidity_2m,surface_pressure',
        temperature_unit='fahrenheit', timezone='UTC'))
    with urllib.request.urlopen(f'{ARCHIVE}?{q}', timeout=60) as r:
        h = json.load(r).get('hourly', {})
    if not h.get('time'):
        return h, None
    want = t.strftime('%Y-%m-%dT%H:00')
    if want in h['time']:
        return h, h['time'].index(want)
    key = t.strftime('%Y-%m-%dT%H')
    cands = [i for i, s_ in enumerate(h['time']) if s_[:13] == key[:13]]
    return h, (cands[0] if cands else None)


def main(n=40):
    with open(os.path.join(ROOT, 'data', 'game_weather_rs.json')) as f:
        wx = json.load(f)
    pks = [k for k, v in wx.items() if v.get('rho') is not None]
    random.Random(7).shuffle(pks)

    rows = []
    seen_venue = defaultdict(int)
    for pk in pks:
        if len(rows) >= n:
            break
        try:
            m = game_meta(pk)
        except Exception:
            continue
        if seen_venue[m['venue']] >= 2 or not m['lat'] or not m.get('utc'):
            continue
        seen_venue[m['venue']] += 1
        try:
            h, idx = archive(m['lat'], m['lon'], m['utc'])
        except Exception as e:
            print('  archive fail', pk, e)
            continue
        if not h.get('time') or idx is None:
            continue
        tF = h['temperature_2m'][idx]
        rh = h['relative_humidity_2m'][idx]
        pr = h['surface_pressure'][idx]
        if None in (tF, rh, pr):
            continue
        cond = (m['cond'] or '').strip().lower()
        closed = cond in ROOF_CLOSED_CONDITIONS
        feed_t = float(m['temp']) if m['temp'] else None
        cur = wx[pk]['rho']
        # what the pipeline WOULD use vs measured. For closed roofs the indoor
        # temperature is controlled, so keep the 72F clamp and only correct
        # pressure + an assumed indoor humidity.
        meas = moist_density(pr, ROOF_CLOSED_TEMP_F if closed else tF,
                             50.0 if closed else rh)
        # Decompose the total error into its three independent parts, so
        # "does humidity matter" gets an answer separate from "does measured
        # pressure matter" and "is the feed temperature any good".
        use_t = ROOF_CLOSED_TEMP_F if closed else tF
        use_rh = 50.0 if closed else rh
        dry_meas_pt = moist_density(pr, use_t, 0.0)        # measured P+T, bone dry
        dry_baro_obs_t = compute_air_density(m['elev'] or 0, use_t)  # baro P, measured T
        rows.append(dict(pk=pk, venue=m['name'], closed=closed, elev=m['elev'],
                         feed_t=feed_t, obs_t=tF, rh=rh, p=pr,
                         cur=cur, meas=meas, d=100 * (meas - cur) / cur,
                         d_humid=100 * (meas - dry_meas_pt) / dry_meas_pt,
                         d_press=100 * (dry_meas_pt - dry_baro_obs_t) / dry_baro_obs_t,
                         d_temp=100 * (dry_baro_obs_t - cur) / cur))
        time.sleep(0.15)

    print(f'{len(rows)} games sampled across {len({r["venue"] for r in rows})} venues\n')
    d = np.array([r['d'] for r in rows])
    print(f'density error of the current model, % (measured minus current):')
    print(f'  mean {d.mean():+.2f}   sd {d.std():.2f}   '
          f'min {d.min():+.2f}   max {d.max():+.2f}')
    print(f'  implied movement error on a 15" break: mean {15*((1/(1+d.mean()/100))**1.05-1):+.3f}"'
          f'   worst {15*((1/(1+abs(d).max()/100))**1.05-1):.3f}"')
    print(f'  sd of the movement error: '
          f'{np.std([15*((1/(1+x/100))**1.05-1) for x in d]):.3f}"')
    print(f'\n  for scale, measured venue offsets were sd 0.199" (along), '
          f'0.110" (cross)')
    print('\nWHICH PART OF THE ERROR IS WHICH (density %, and inches on a 15" break)')
    print(f"  {'component':<34} {'mean':>7} {'sd':>7} {'|max|':>7} {'sd in inches':>13}")
    for key, label in (('d_humid', 'humidity (vs bone-dry, same P,T)'),
                       ('d_press', 'measured P vs barometric estimate'),
                       ('d_temp',  'observed T vs feed-reported T'),
                       ('d',       'TOTAL current-model error')):
        v = np.array([r[key] for r in rows])
        inch = np.std([15 * ((1 / (1 + x / 100)) ** 1.05 - 1) for x in v])
        print(f'  {label:<34} {v.mean():>+7.2f} {v.std():>7.2f} '
              f'{np.abs(v).max():>7.2f} {inch:>12.3f}"')
    hum = np.array([r['d_humid'] for r in rows])
    print(f'\n  humidity alone is worth at most '
          f'{15*abs((1/(1+np.abs(hum).max()/100))**1.05-1):.3f}" on a 15" break '
          f'across this sample,')
    print(f'  and {15*abs((1/(1+np.abs(hum).mean()/100))**1.05-1):.3f}" on average. '
          f'Venue offsets, for scale, were sd 0.199".')
    print(f"\n{'venue':<26} {'roof':>5} {'RH':>4} {'P hPa':>7} {'humid%':>7} "
          f"{'press%':>7} {'temp%':>7} {'total%':>7}")
    for r in sorted(rows, key=lambda x: -abs(x['d_humid']))[:12]:
        print(f'{str(r["venue"])[:26]:<26} {"clsd" if r["closed"] else "open":>5} '
              f'{r["rh"]:>4.0f} {r["p"]:>7.1f} {r["d_humid"]:>+7.2f} '
              f'{r["d_press"]:>+7.2f} {r["d_temp"]:>+7.2f} {r["d"]:>+7.2f}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
