#!/usr/bin/env python3
"""gate_v2_height_map.py — rebuild data/_gate_v2/pitcher_height.json, the
name-keyed height lookup stuff_gate_v2.height_map() reads.

Schema (what height_map() expects):
  {"height_in": {"<mlb id>": inches, ...},
   "name_to_ids": {"Last, First": [id, ...], ...}}

Sources, in order:
  ids     2021-2025 Statcast caches (data/_statcast{Y}_cache.pkl and
          _statcast2025_full_cache.pkl: pitcher id + player_name),
          data/mlb_id_cache.json (name|team -> id), and the shipped
          data/pitcher_heights.json ids_by_name.
  height  data/pitcher_heights.json by_name for names that map to ONE id;
          the MLB Stats API /people endpoint (batched personIds, same call
          as scripts/ci/build_pitcher_heights.py) for every other id.
          Any HTTP/network error aborts before writing (fail closed).

Coverage is reported per gate season frame (data/_gate_v2/season_*.pkl),
pitch-weighted, with the unresolved names listed.

Usage: python3 scripts/research/stuff/gate_v2_height_map.py [--no-api]
"""
import argparse
import json
import os
import pickle
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(DATA, '_gate_v2')
OUT = os.path.join(CACHE, 'pitcher_height.json')
API = 'https://statsapi.mlb.com/api/v1/'
STATCAST = [os.path.join(DATA, f'_statcast{y}_cache.pkl')
            for y in (2021, 2022, 2023, 2024, 2025)] + \
           [os.path.join(DATA, '_statcast2025_full_cache.pkl')]


def parse_height(h):
    m = re.match(r"(\d+)' ?(\d+)", h or '')
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def fetch_people(ids):
    """id -> inches. Raises on any HTTP / network failure (fail closed)."""
    out = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        url = API + 'people?personIds=' + ','.join(map(str, chunk))
        with urllib.request.urlopen(url, timeout=30) as r:
            if r.status != 200:
                raise urllib.error.HTTPError(url, r.status, 'bad status',
                                             r.headers, None)
            j = json.load(r)
        for p in j.get('people', []):
            h = parse_height(p.get('height'))
            if h:
                out[int(p['id'])] = h
        print(f'    people {i + len(chunk)}/{len(ids)}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-api', action='store_true',
                    help='asset-only map; ids without a shipped height stay '
                         'unresolved')
    a = ap.parse_args()

    name_to_ids = defaultdict(set)
    for p in STATCAST:
        if not os.path.exists(p):
            print(f'  missing {p}')
            continue
        with open(p, 'rb') as f:
            df = pickle.load(f)
        pairs = df[['pitcher', 'player_name']].dropna().drop_duplicates()
        for pid, nm in zip(pairs['pitcher'], pairs['player_name']):
            name_to_ids[str(nm)].add(int(pid))
        print(f'  {os.path.basename(p)}: {len(pairs)} (id, name) pairs')
        del df
    with open(os.path.join(DATA, 'mlb_id_cache.json')) as f:
        for k, v in json.load(f).items():
            name_to_ids[k.split('|')[0]].add(int(v))
    with open(os.path.join(DATA, 'pitcher_heights.json')) as f:
        asset = json.load(f)
    for nm, ids in asset['ids_by_name'].items():
        for i in ids:
            name_to_ids[nm].add(int(i))
    all_ids = {i for s in name_to_ids.values() for i in s}
    print(f'  {len(name_to_ids)} names, {len(all_ids)} ids')

    # heights from the shipped asset where the name maps to exactly one id
    height_in = {}
    for nm, ids in asset['ids_by_name'].items():
        if len(ids) == 1 and nm in asset['by_name']:
            height_in[int(ids[0])] = float(asset['by_name'][nm])
    need = sorted(i for i in all_ids if i not in height_in)
    print(f'  {len(height_in)} ids with a shipped height, {len(need)} to fetch')
    if need and not a.no_api:
        try:
            got = fetch_people(need)
        except (urllib.error.URLError, OSError, ValueError) as e:
            sys.exit(f'MLB API failed ({e}); no map written (fail closed)')
        height_in.update({k: float(v) for k, v in got.items()})
        print(f'  API resolved {len(got)}/{len(need)}')
    still = [i for i in need if i not in height_in]

    out = {'height_in': {str(k): v for k, v in sorted(height_in.items())},
           'name_to_ids': {nm: sorted(ids) for nm, ids in
                           sorted(name_to_ids.items())},
           'source': 'gate_v2_height_map.py 2026-09-02: ids from Statcast '
                     '2021-25 caches + mlb_id_cache + pitcher_heights.json; '
                     'heights from pitcher_heights.json (single-id names) '
                     'and the MLB Stats API people endpoint',
           'n_ids_unresolved': len(still)}
    os.makedirs(CACHE, exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=0, sort_keys=True)
    os.replace(tmp, OUT)
    print(f'  wrote {OUT}: {len(height_in)} heights, {len(still)} ids '
          f'unresolved')

    # coverage per gate season frame, pitch-weighted
    name_h = {}
    for nm, ids in name_to_ids.items():
        hs = [height_in[i] for i in ids if i in height_in]
        if hs:
            name_h[nm] = sum(hs) / len(hs)
    multi = sum(1 for nm, ids in name_to_ids.items()
                if len({height_in[i] for i in ids if i in height_in}) > 1)
    print(f'  {len(name_h)} names with a height; {multi} names whose ids '
          f'disagree on height (mean taken)')
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        p = os.path.join(CACHE, f'season_{y}.pkl')
        if not os.path.exists(p):
            continue
        s = pd.read_pickle(p)['pitcher']
        cnt = s.value_counts()
        hit = cnt[cnt.index.isin(name_h)].sum()
        miss = cnt[~cnt.index.isin(name_h)]
        print(f'  season {y}: {hit / cnt.sum():.4%} of pitches covered, '
              f'{len(miss)} of {len(cnt)} pitchers unresolved'
              + (f': {list(miss.index[:6])}' if len(miss) else ''))
        del s


if __name__ == '__main__':
    main()
