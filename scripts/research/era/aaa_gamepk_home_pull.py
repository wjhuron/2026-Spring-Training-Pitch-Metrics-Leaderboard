"""aaa_gamepk_home_pull.py — home club for every gamePk in the AAA caches.

The MLB-shaped AAA season caches (data/_aaa_statcast{y}_cache.pkl) carry no
team columns, so the hpERA park-channel validation cannot attribute a
pitcher's home park without this map. One schedule call resolves up to 40
gamePks (gamePks= accepts a comma list), so the full corpus resolves in a
few hundred requests.

Output: data/_aaa_gamepk_home.json  {gamePk: {"home": "Rochester Red Wings",
"homeId": 534, "date": "2025-05-01"}}

Usage: python3 scripts/research/era/aaa_gamepk_home_pull.py
"""
import json
import os
import pickle
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', '_aaa_gamepk_home.json')
YEARS = (2023, 2024, 2025, 2026)
BATCH = 40


def main():
    pks = set()
    for y in YEARS:
        p = os.path.join(ROOT, 'data', f'_aaa_statcast{y}_cache.pkl')
        if not os.path.exists(p):
            print(f'  missing {p} — skipped')
            continue
        df = pickle.load(open(p, 'rb'))
        pks.update(int(v) for v in df['game_pk'].dropna().unique())
        print(f'  {y}: cumulative {len(pks)} gamePks')
    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT))
        print(f'  resuming: {len(done)} already resolved')
    todo = sorted(pk for pk in pks if str(pk) not in done)
    print(f'  to resolve: {len(todo)}')
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        url = ('https://statsapi.mlb.com/api/v1/schedule?sportId=11&gamePks='
               + ','.join(map(str, chunk)))
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            print(f'  batch {i}: {e} — retrying once in 5s')
            time.sleep(5)
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
        for day in data.get('dates', []):
            for g in day.get('games', []):
                home = g.get('teams', {}).get('home', {}).get('team', {})
                done[str(g['gamePk'])] = {
                    'home': home.get('name'), 'homeId': home.get('id'),
                    'date': g.get('officialDate'),
                }
        if (i // BATCH) % 20 == 0:
            # checkpoint: partial progress survives an interrupt, and the
            # temp-then-move keeps the artifact whole if this dies mid-write
            tmp = OUT + '.tmp'
            json.dump(done, open(tmp, 'w'))
            os.replace(tmp, OUT)
            print(f'  {min(i + BATCH, len(todo))}/{len(todo)} resolved',
                  flush=True)
        time.sleep(0.4)
    unresolved = [pk for pk in todo if str(pk) not in done]
    tmp = OUT + '.tmp'
    json.dump(done, open(tmp, 'w'))
    os.replace(tmp, OUT)
    print(f'done: {len(done)} resolved, {len(unresolved)} unresolved')
    if unresolved:
        print(f'  unresolved sample: {unresolved[:10]}')


if __name__ == '__main__':
    main()
