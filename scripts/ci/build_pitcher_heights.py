#!/usr/bin/env python3
"""build_pitcher_heights.py — refresh data/pitcher_heights.json (v14 Stuff+
'height' feature).

Pitcher height in inches, keyed by the "Last, First" name the pitch caches
carry. Sources, in order: the existing artifact, data/mlb_id_cache.json
(name|team -> MLB id), then an MLB Stats API name search for anything still
unresolved. Heights come from /api/v1/people. Ambiguous names (several ids)
take the mean.

Names are collected from data/all_pitches_rs_cache.pkl (MLB + ROC/AAA).
The NEW tab is deliberately excluded — see collect_names. The artifact is written to a temp path and moved, never
partially. A network failure leaves the previous artifact in place and
prints how many names stay unresolved; the trainer imputes those to the
frozen league mean and says so.

Runs in CI before stuff_plus/train_stuff.py. Also safe to run locally.

Usage: python3 scripts/ci/build_pitcher_heights.py [--seed-from data/_gate_v2/pitcher_height.json]
"""
import argparse
import json
import os
import pickle
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA, 'pitcher_heights.json')
API = 'https://statsapi.mlb.com/api/v1/'


def parse_height(h):
    m = re.match(r"(\d+)' ?(\d+)", h or '')
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def fetch_people(ids):
    out = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        url = API + 'people?personIds=' + ','.join(map(str, chunk))
        with urllib.request.urlopen(url, timeout=30) as r:
            j = json.load(r)
        for p in j.get('people', []):
            h = parse_height(p.get('height'))
            if h:
                out[int(p['id'])] = h
    return out


def search_name(name):
    """'Last, First' -> candidate ids via the people search endpoint."""
    last, _, first = name.partition(', ')
    q = urllib.parse.quote(f'{first} {last}'.strip())
    url = API + f'people/search?names={q}&sportIds=1,11,12,13,14,16'
    with urllib.request.urlopen(url, timeout=30) as r:
        j = json.load(r)
    return [int(p['id']) for p in j.get('people', [])
            if p.get('lastName', '').lower() == last.lower()
            and p.get('primaryPosition', {}).get('code') in ('1', 'Y')]


def collect_names():
    """MLB + ROC/AAA arms only. The NEW tab (scoring_only_rs_cache.pkl) is
    deliberately excluded (per Wally 2026-08-23): its arms are often college
    or indie pitchers the MLB API cannot resolve, they only ever receive a
    NEW-tab grade, and they impute to the league mean in the trainer with a
    log line. When one of them reaches a real MLB/ROC cache, this builder
    resolves them then."""
    names = set()
    p = os.path.join(DATA, 'all_pitches_rs_cache.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            for x in pickle.load(f):
                if x.get('Pitch Type') != 'EP' and x.get('Pitcher'):
                    names.add(x['Pitcher'])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-from', default=None,
                    help='research lookup to fold in (name_to_ids/height_in)')
    a = ap.parse_args()

    art = {'by_name': {}, 'ids_by_name': {}}
    if os.path.exists(OUT):
        with open(OUT) as f:
            art = json.load(f)
    if a.seed_from and os.path.exists(a.seed_from):
        with open(a.seed_from) as f:
            j = json.load(f)
        H = {int(k): v for k, v in j['height_in'].items()}
        for n, ids in j['name_to_ids'].items():
            hs = [H[i] for i in ids if i in H]
            if hs and n not in art['by_name']:
                art['by_name'][n] = round(sum(hs) / len(hs), 1)
                art['ids_by_name'][n] = ids

    names = collect_names()
    todo = sorted(n for n in names if n not in art['by_name'])
    print(f'  {len(names)} pitchers in caches, {len(art["by_name"])} known, '
          f'{len(todo)} to resolve')
    unresolved = []
    if todo:
        idc = {}
        p = os.path.join(DATA, 'mlb_id_cache.json')
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    idc.setdefault(k.split('|')[0], set()).add(int(v))
        want = {}
        try:
            for n in todo:
                ids = idc.get(n) or set(search_name(n))
                if ids:
                    want[n] = sorted(ids)
            H = fetch_people([i for v in want.values() for i in v])
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f'  WARNING: MLB API failed ({e}); artifact left as-is')
            H, want = {}, {}
        for n in todo:
            hs = [H[i] for i in want.get(n, []) if i in H]
            if hs:
                art['by_name'][n] = round(sum(hs) / len(hs), 1)
                art['ids_by_name'][n] = want[n]
            else:
                unresolved.append(n)
    vals = list(art['by_name'].values())
    art['league_mean'] = round(sum(vals) / len(vals), 2) if vals else None
    art['n'] = len(vals)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(art, f, indent=0, sort_keys=True)
    os.replace(tmp, OUT)
    print(f'  wrote {OUT}: {art["n"]} heights, league mean {art["league_mean"]}, '
          f'{len(unresolved)} unresolved' + (f': {unresolved}' if unresolved else ''))


if __name__ == '__main__':
    main()
