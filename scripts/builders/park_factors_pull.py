"""park_factors_pull.py — per-team runs park factors 2021-2026 from Savant.

Savant's statcast-park-factors leaderboard, rolling 3-year window, runs
index (100 = neutral). The page embeds the table as `var data = [...]`;
the csv=true param does not work on this endpoint, so parse the HTML.
curl-style urllib with a browser UA is fine for Savant (unlike FanGraphs).

Output: data/park_factors.json  {season: {team_id: index_runs}}
"""
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'park_factors.json')
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
URL = ('https://baseballsavant.mlb.com/leaderboard/statcast-park-factors'
       '?type=year&year={y}&batSide=&stat=index_wOBA&condition=All&rolling=3')


def main():
    result = {}
    for y in SEASONS:
        req = urllib.request.Request(URL.format(y=y),
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode()
        m = re.search(r'var data = (\[.*?\]);', html, re.S)
        rows = json.loads(m.group(1))
        season = {}
        for row in rows:
            tid = row.get('main_team_id')
            pf = row.get('index_runs')
            if tid is None or pf is None:
                continue
            season[str(tid)] = float(pf)
        result[str(y)] = season
        print(f'{y}: {len(season)} teams, '
              f'min {min(season.values()):.0f} max {max(season.values()):.0f}',
              flush=True)
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
