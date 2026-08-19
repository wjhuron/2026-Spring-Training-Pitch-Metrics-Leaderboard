"""park_factors_pull.py — per-team runs park factors 2021-2026 from Savant.

Savant's statcast-park-factors leaderboard, runs index (100 = neutral).
The page embeds the table as `var data = [...]`; the csv=true param does
not work on this endpoint, so parse the HTML. curl-style urllib with a
browser UA is fine for Savant (unlike FanGraphs).

WINDOW CASCADE (2026-08-19). The rolling-3 window is the default, but
Savant OMITS a club entirely when its current venue has less than three
seasons of history — it does not fall back on its own. That silently
dropped TEX 2021 (Globe Life Field opened 2020), TBR 2025 (Steinbrenner
Field) and ATH 2025-2026 (Sutter Health Park), and every consumer read
the gap as a neutral park. Sutter Health Park is not neutral: its 2026
runs index is 123 over two years and 130 over one.

So each club takes the WIDEST window that actually carries a value,
3 -> 2 -> 1. This is a stated convention, not a measured optimum: a
wider window is steadier, so the rule is "as much history as the venue
has". The window used is recorded per club under the '_window' key.

Output: data/park_factors.json
    {season: {team_id: index_runs}, '_window': {season: {team_id: 1|2|3}}}
Consumers read PF[str(season)][str(club_id)]; the '_window' key is
provenance and is skipped by an ordinary season lookup.
"""
import json
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'park_factors.json')
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
WINDOWS = (3, 2, 1)          # widest first
URL = ('https://baseballsavant.mlb.com/leaderboard/statcast-park-factors'
       '?type=year&year={y}&batSide=&stat=index_wOBA&condition=All&rolling={r}')


def pull(year, rolling):
    req = urllib.request.Request(URL.format(y=year, r=rolling),
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode()
    m = re.search(r'var data = (\[.*?\]);', html, re.S)
    if not m:
        raise RuntimeError(f'Savant park-factor page structure changed '
                           f'({year}, rolling={rolling}) — no `var data`')
    out = {}
    for row in json.loads(m.group(1)):
        tid = row.get('main_team_id')
        pf = row.get('index_runs')
        if tid is None or pf is None:
            continue
        out[str(tid)] = float(pf)
    return out


def main():
    result = {}
    windows = {}
    for y in SEASONS:
        season = {}
        used = {}
        for w in WINDOWS:
            got = pull(y, w)
            for tid, pf in got.items():
                if tid not in season:
                    season[tid] = pf
                    used[tid] = w
        narrow = sorted(t for t, w in used.items() if w != WINDOWS[0])
        result[str(y)] = season
        windows[str(y)] = used
        note = (f'  (narrower window: '
                + ', '.join(f'{t}=r{used[t]}' for t in narrow) + ')') if narrow else ''
        print(f'{y}: {len(season)} teams, '
              f'min {min(season.values()):.0f} max {max(season.values()):.0f}{note}',
              flush=True)
    result['_window'] = windows
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f)
    os.replace(tmp, OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
