#!/usr/bin/env python3
"""gf_plate_stability_check.py — is a T+1min `gf` plate coordinate final?

Background: the scrape merges Savant plate coords from the `gf` endpoint
(pitcher2026.download_savant_gf), because `gf` is fully populated one minute
after a game goes Final while the Statcast Search CSV is still empty. That
answers "are the values THERE"; it does not answer "are they FINAL".

This re-pulls the snapshotted game and compares every coordinate against
what `gf` served at T+1min, then against the Statcast Search CSV now that
the game has settled. Two clean results mean the scrape-time merge needs no
correction pass; any drift tells us how long to wait, or that the nightly
backfill_supplement run is doing real work on these columns after all.

  python3 scripts/audits/gf_plate_stability_check.py
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import requests

SNAPSHOT = os.path.join(ROOT, 'data', '_gf_822688_T1_snapshot.json')
GAME_PK = 822688
GAME_DATE = '2026-08-30'
TEAMS = 'WSH%7CMIA%7C'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def gf_now(game_pk):
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_pk}',
                     headers=UA, timeout=60)
    r.raise_for_status()
    d = r.json()
    out = {}
    for side in ('team_home', 'team_away'):
        for p in d.get(side) or []:
            ab, pn = p.get('ab_number'), p.get('pitch_number')
            if ab is None or pn is None or p.get('plate_z') is None:
                continue
            out[f'{int(ab)}_{int(pn)}'] = [float(p['plate_x']), float(p['plate_z'])]
    return out


def search_csv_now():
    import csv
    import io
    url = ('https://baseballsavant.mlb.com/statcast_search/csv?all=true'
           f'&player_type=pitcher&game_date_gt={GAME_DATE}&game_date_lt={GAME_DATE}'
           f'&hfTeam={TEAMS}&type=details')
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    out = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        if row.get('game_pk') != str(GAME_PK) or not row.get('plate_z'):
            continue
        out[f"{int(row['at_bat_number'])}_{int(row['pitch_number'])}"] = [
            float(row['plate_x']), float(row['plate_z'])]
    return out


def compare(label, a, b):
    keys = set(a) & set(b)
    if not keys:
        print(f'{label}: NO OVERLAP (a={len(a)}, b={len(b)})')
        return
    dz = [a[k][1] - b[k][1] for k in keys]
    dx = [a[k][0] - b[k][0] for k in keys]
    changed = sum(1 for v in dz if abs(v) > 0.0005)
    print(f'{label}: n={len(keys)}  dPlateZ mean {st.mean(dz):+.5f} '
          f'max|d| {max(abs(v) for v in dz):.5f}  '
          f'dPlateX max|d| {max(abs(v) for v in dx):.5f}  '
          f'changed>0.0005ft: {changed}')
    if len(a) != len(keys) or len(b) != len(keys):
        print(f'    coverage: snapshot {len(a)}, other {len(b)}, overlap {len(keys)}')


def main():
    snap = json.load(open(SNAPSHOT))
    print(f'T+1min snapshot: {len(snap)} pitches (game {GAME_PK}, {GAME_DATE})\n')
    compare('gf T+1min  vs  gf now       ', snap, gf_now(GAME_PK))
    try:
        compare('gf T+1min  vs  search CSV  ', snap, search_csv_now())
    except requests.HTTPError as e:
        print(f'search CSV unavailable: {e}')
    print('\nZero changed cells in both = a T+1min gf value is already final, '
          'so the scrape-time merge needs no correction pass.')


if __name__ == '__main__':
    main()
