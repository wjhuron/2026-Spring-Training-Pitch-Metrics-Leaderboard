"""backfill_weather_exponent.py — recompute the sheet x-columns at the
calibrated density exponent (WEATHER_EXPONENT in Pitcher2026.py).

Historical xIndVrtBrk/xHorzBrk were written at append time with the old
rho^(2/3) factor. This script rewrites both columns across all six division
workbooks (plus the ROC/AAA extra tabs) as:

    x = round(raw × (STANDARD_RHO / rho_game) ** WEATHER_EXPONENT, 1)

rho_game comes from data/game_weather_rs.json (game_pk extracted from
PitchID); games missing from the sidecar are fetched from the Stats API
slim feed first (covers MiLB gamePks too, so ROC works). Rows whose game
has no density inputs (no elevation) keep x = raw, matching the pipeline's
factor-1.0 behavior.

Safety:
  - DRY RUN by default: prints per-tab change counts and the delta
    distribution without writing. Set APPLY=1 to write.
  - Before writing, the current x-columns of every tab are saved to
    data/_xcol_backup_<timestamp>.pkl (gitignored) for rollback.
  - Writes are paced (1s between tabs) to respect Sheets quotas.
  - The sidecar's cached 'factor' fields are refreshed to the new exponent.

Usage:
    python3 scripts/builders/backfill_weather_exponent.py          # dry run
    APPLY=1 python3 scripts/builders/backfill_weather_exponent.py  # write
"""
import os, sys, json, time, pickle
from collections import defaultdict
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scrapers.Pitcher2026 import (STANDARD_RHO, WEATHER_EXPONENT, compute_air_density,
                                  VENUE_ELEVATION_FT_OVERRIDE, DEFAULT_TEMP_F,
                                  ROOF_CLOSED_TEMP_F, ROOF_CLOSED_CONDITIONS)
from pipeline.fetch import (_gspread_client, DIVISION_WORKBOOK_IDS,
                            read_sheet_with_retry)
from pipeline.utils import MLB_TEAMS

SIDECAR = os.path.join(ROOT, 'data', 'game_weather_rs.json')
FEED = ('https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        '?fields=gameData,weather,temp,condition,venue,id,location,elevation')
APPLY = os.environ.get('APPLY') == '1'


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


def load_sidecar():
    try:
        return json.load(open(SIDECAR))
    except (FileNotFoundError, ValueError):
        return {}


def repair_elevations(store):
    """Fill elevation on games ALREADY cached whose venue has since been added
    to VENUE_ELEVATION_FT_OVERRIDE.

    fetch_missing() only looks at games absent from the sidecar, so without
    this a newly-listed park stays un-adjusted forever: the entry exists, it
    just has elevationFt=None, which rho_of() turns into factor 1.0. This is
    how 373 MiLB games sat at factor 1.0 while carrying a valid temperature.
    """
    fixed = 0
    for info in store.values():
        if not isinstance(info, dict) or info.get('error'):
            continue
        if info.get('elevationFt') is not None:
            continue
        elev = VENUE_ELEVATION_FT_OVERRIDE.get(info.get('venueId'))
        if elev is not None:
            info['elevationFt'] = float(elev)
            fixed += 1
    print(f'repaired elevation on {fixed} cached games from the override table')
    return store


def fetch_missing(store, pks):
    def stale(pk):
        info = store.get(str(pk))
        if info is None:
            return True
        if not isinstance(info, dict):
            return True
        # a prior failure, or a record with no venueId — the override can't
        # repair those from cache, so go back to the feed for them
        return bool(info.get('error')) or (
            info.get('elevationFt') is None and info.get('venueId') is None)

    missing = [pk for pk in pks if pk and stale(pk)]
    if not missing:
        return store
    print(f'fetching density inputs for {len(missing)} games not in sidecar...')
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
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(missing)}', flush=True)
    return store


def col_letter(idx0):
    """0-based column index -> A1 letter(s)."""
    s = ''
    n = idx0 + 1
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def main():
    gc = _gspread_client()
    store = load_sidecar()

    # pass 1: read every tab, collect game pks, plan changes
    plans = []       # (ws, tab, n_rows, ivb_col, hb_col, new_ivb, new_hb, old_ivb, old_hb)
    all_pks = set()
    tab_rows = []    # cache raw reads to avoid a second pass

    for name, wid in DIVISION_WORKBOOK_IDS.items():
        # NEW is the scratch tab for arms new to the org — its rows come from
        # AAA/complex/college parks all over, which is exactly where the
        # elevation gaps live, so it needs the backfill more than most.
        extra = {'ROC', 'AAA', 'NEW'} if name == 'NLE2026' else set()
        sh = gc.open_by_key(wid)
        print(f'{sh.title}:')
        for ws in sh.worksheets():
            tab = ws.title
            if tab not in MLB_TEAMS and tab not in extra:
                continue
            rows = read_sheet_with_retry(ws)
            time.sleep(0.6)
            if not rows:
                continue
            header = rows[0]
            ci = {n: i for i, n in enumerate(header) if n}
            need = ('PitchID', 'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk')
            if any(k not in ci for k in need):
                print(f'  {tab}: missing columns, SKIPPED ({[k for k in need if k not in ci]})')
                continue
            for row in rows[1:]:
                pid = row[ci['PitchID']] if ci['PitchID'] < len(row) else ''
                if pid and '_' in pid:
                    all_pks.add(pid.split('_')[0])
            tab_rows.append((ws, tab, rows, ci))
            print(f'  {tab}: {len(rows) - 1} rows read')

    store = repair_elevations(store)
    store = fetch_missing(store, sorted(all_pks))
    rho_by_pk = {pk: rho_of(store.get(str(pk))) for pk in all_pks}
    # refresh cached factor fields at the new exponent
    for pk, info in store.items():
        if isinstance(info, dict) and not info.get('error'):
            r = rho_of(info)
            info['rho'] = round(r, 5) if r else None
            info['factor'] = round((STANDARD_RHO / r) ** WEATHER_EXPONENT, 5) if r else 1.0
    json.dump(store, open(SIDECAR, 'w'), indent=0, sort_keys=True)

    total_changed = 0
    deltas = []
    backup = {}
    for ws, tab, rows, ci in tab_rows:
        n = len(rows) - 1
        new_ivb = [['xIndVrtBrk']]
        new_hb = [['xHorzBrk']]
        old_ivb = []
        old_hb = []
        changed = 0
        for row in rows[1:]:
            def cell(k):
                i = ci[k]
                return row[i] if i < len(row) else ''
            pid = cell('PitchID')
            pk = pid.split('_')[0] if pid and '_' in pid else None
            rho = rho_by_pk.get(pk)
            factor = (STANDARD_RHO / rho) ** WEATHER_EXPONENT if rho else 1.0
            for raw_key, out_list, old_list, old_key in (
                    ('IndVertBrk', new_ivb, old_ivb, 'xIndVrtBrk'),
                    ('HorzBrk', new_hb, old_hb, 'xHorzBrk')):
                raw = sf(cell(raw_key))
                old = cell(old_key)
                old_list.append(old)
                if raw is None:
                    out_list.append([''])
                    continue
                new = round(raw * factor, 1)
                out_list.append([new])
                old_f = sf(old)
                if old_f is None or abs(old_f - new) > 0.049:
                    changed += 1
                    if old_f is not None:
                        deltas.append(new - old_f)
        total_changed += changed
        backup[f'{ws.spreadsheet.title}|{tab}'] = {'xIndVrtBrk': old_ivb, 'xHorzBrk': old_hb}
        plans.append((ws, tab, n, ci, new_ivb, new_hb, changed))
        print(f'  planned {tab}: {changed} changed cells / {2 * n}')

    print(f'\nTOTAL: {total_changed} changed cells across {len(plans)} tabs')
    if deltas:
        ds = sorted(deltas)
        print(f'delta distribution (new − old, inches): min={ds[0]:+.1f} '
              f'p5={ds[len(ds)//20]:+.1f} med={ds[len(ds)//2]:+.1f} '
              f'p95={ds[-len(ds)//20]:+.1f} max={ds[-1]:+.1f}')

    if not APPLY:
        print('\nDRY RUN — no writes. Set APPLY=1 to write.')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bpath = os.path.join(ROOT, 'data', f'_xcol_backup_{stamp}.pkl')
    with open(bpath, 'wb') as f:
        pickle.dump(backup, f)
    print(f'backup of old x-columns saved: {bpath}')

    for ws, tab, n, ci, new_ivb, new_hb, changed in plans:
        if changed == 0:
            print(f'  {tab}: no changes, skipped')
            continue
        for col_key, values in (('xIndVrtBrk', new_ivb), ('xHorzBrk', new_hb)):
            letter = col_letter(ci[col_key])
            rng = f'{letter}1:{letter}{n + 1}'
            ws.update(values=values, range_name=rng,
                      value_input_option='USER_ENTERED')
            time.sleep(1.0)
        print(f'  wrote {tab} ({changed} changed cells)', flush=True)

    print('\nBackfill complete. Re-run process_data.py + the Stuff+ trainer.')


if __name__ == '__main__':
    main()
