"""era_targets_pull.py — pitcher-season run-prevention targets 2021-2026.

Pulls from the MLB Stats API, one pass per season:
  * full-season pitching line (playerPool=ALL)
  * first-half line  (byDateRange: 03-01 .. All-Star date)
  * second-half line (byDateRange: All-Star date + 1 .. 11-30)

The h1/h2 sitCodes are not supported for pitching (situationCodes says
pitching: False), so halves come from byDateRange at the season's actual
All-Star date (schedule API, gameTypes=A).

Traded pitchers can appear as several player-team rows; rows are summed
from raw components (outs, ER, R, BF, GS, G) per player id, and rates are
recomputed: ERA = ER * 27 / outs, RA9 = R * 27 / outs.

Output: data/_era_targets.json
  {season: {pitcher_id: {name, teams:[ids], gs, g, outs, er, r, bf,
                         h1: {...}, h2: {...}}}}
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'data', '_era_targets.json')
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
BASE = 'https://statsapi.mlb.com/api/v1'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def asg_date(season):
    d = get(f'{BASE}/schedule?sportId=1&season={season}&gameTypes=A')
    dates = [dd['date'] for dd in d.get('dates', []) if dd.get('games')]
    if not dates:
        raise RuntimeError(f'no ASG date for {season}')
    return sorted(dates)[0]


def pull_rows(url):
    d = get(url)
    stats = d.get('stats', [])
    if not stats:
        return []
    return stats[0].get('splits', [])


def accumulate(rows):
    """player-team rows -> per-player summed components."""
    out = {}
    for sp in rows:
        pl = sp.get('player') or {}
        pid = pl.get('id')
        if pid is None:
            continue
        st = sp.get('stat') or {}
        outs = st.get('outs')
        if outs is None:
            ip = st.get('inningsPitched') or '0.0'
            whole, _, frac = str(ip).partition('.')
            outs = int(whole or 0) * 3 + int(frac or 0)
        rec = out.setdefault(pid, {
            'name': pl.get('fullName'), 'teams': [],
            'outs': 0, 'er': 0, 'r': 0, 'bf': 0, 'gs': 0, 'g': 0,
            'so': 0, 'bb': 0, 'ibb': 0, 'hbp': 0, 'hr': 0, 'h': 0,
            'd2': 0, 'd3': 0, 'ab': 0, 'sf': 0, 'go': 0, 'ao': 0})
        rec['outs'] += int(outs or 0)
        rec['er'] += int(st.get('earnedRuns') or 0)
        rec['r'] += int(st.get('runs') or 0)
        rec['bf'] += int(st.get('battersFaced') or 0)
        rec['gs'] += int(st.get('gamesStarted') or 0)
        rec['g'] += int(st.get('gamesPitched') or st.get('gamesPlayed') or 0)
        rec['so'] += int(st.get('strikeOuts') or 0)
        rec['bb'] += int(st.get('baseOnBalls') or 0)
        rec['ibb'] += int(st.get('intentionalWalks') or 0)
        rec['hbp'] += int(st.get('hitBatsmen') or 0)
        rec['hr'] += int(st.get('homeRuns') or 0)
        rec['h'] += int(st.get('hits') or 0)
        rec['d2'] += int(st.get('doubles') or 0)
        rec['d3'] += int(st.get('triples') or 0)
        rec['ab'] += int(st.get('atBats') or 0)
        rec['sf'] += int(st.get('sacFlies') or 0)
        rec['go'] += int(st.get('groundOuts') or 0)
        rec['ao'] += int(st.get('airOuts') or 0)
        tid = (sp.get('team') or {}).get('id')
        if tid and tid not in rec['teams']:
            rec['teams'].append(tid)
    return out


def main():
    result = {}
    for season in SEASONS:
        asg = asg_date(season)
        print(f'{season}: ASG {asg}', flush=True)
        common = f'group=pitching&season={season}&sportId=1&limit=5000&playerPool=ALL'
        full = accumulate(pull_rows(f'{BASE}/stats?stats=season&{common}'))
        h1 = accumulate(pull_rows(
            f'{BASE}/stats?stats=byDateRange&{common}'
            f'&startDate={season}-03-01&endDate={asg}'))
        y, m, d = asg.split('-')
        # day after the ASG; the break has no games so +1 on the date string
        # is safe even without calendar math only if day < 28 (always true
        # for July dates).
        nxt = f'{y}-{m}-{int(d) + 1:02d}'
        h2 = accumulate(pull_rows(
            f'{BASE}/stats?stats=byDateRange&{common}'
            f'&startDate={nxt}&endDate={season}-11-30'))
        srec = {}
        keys = ('outs', 'er', 'r', 'bf', 'gs', 'g', 'so', 'bb', 'ibb',
                'hbp', 'hr', 'h', 'd2', 'd3', 'ab', 'sf', 'go', 'ao')
        for pid, rec in full.items():
            rec['h1'] = {k: h1.get(pid, {}).get(k, 0) for k in keys}
            rec['h2'] = {k: h2.get(pid, {}).get(k, 0) for k in keys}
            srec[str(pid)] = rec
        result[str(season)] = {'asg': asg, 'pitchers': srec}
        print(f'  full {len(full)} | h1 {len(h1)} | h2 {len(h2)}', flush=True)
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
