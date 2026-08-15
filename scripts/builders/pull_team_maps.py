"""pull_team_maps.py — pitcher team per season, for the changed-teams holdout.

The main pull didn't keep team columns. Re-pull just what's needed (keyed by the
same player_name format as the caches, so no fuzzy name matching) and reduce to a
compact per-(year, pitcher) team map: primary team + full team set + n_teams.
Pitcher's team on a pitch = home_team if inning_topbot=='Top' else away_team.
2026 comes from Wally's all_pitches cache (PTeam), no pull needed.

Usage: python3 scripts/builders/pull_team_maps.py
"""
import os, sys, pickle, warnings
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', '_team_maps.pkl')
SEASONS = {2021: ('2021-04-01', '2021-10-03'), 2022: ('2022-04-07', '2022-10-05'),
           2023: ('2023-03-30', '2023-10-01'), 2024: ('2024-03-20', '2024-09-30'),
           2025: ('2025-03-18', '2025-09-28')}


def reduce_year(df):
    counts = defaultdict(Counter)
    for r in df.itertuples(index=False):
        name = r.player_name
        team = r.home_team if r.inning_topbot == 'Top' else r.away_team
        if name and isinstance(team, str):
            counts[name][team] += 1
    out = {}
    for name, c in counts.items():
        primary = c.most_common(1)[0][0]
        out[name] = {'primary': primary, 'teams': sorted(c), 'n_teams': len(c)}
    return out


def main():
    from pybaseball import statcast
    maps = {}
    for yr, (start, end) in SEASONS.items():
        print(f'{yr}: pulling team cols ...', flush=True)
        df = statcast(start_dt=start, end_dt=end, verbose=False)
        df = df[['player_name', 'inning_topbot', 'home_team', 'away_team', 'game_type']]
        df = df[df['game_type'] == 'R']
        maps[yr] = reduce_year(df)
        multi = sum(1 for v in maps[yr].values() if v['n_teams'] > 1)
        print(f'  {yr}: {len(maps[yr])} pitchers, {multi} multi-team', flush=True)

    # 2026 from Wally's cache (PTeam)
    mlb = [p for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
           if p.get('_source') == 'MLB']
    c26 = defaultdict(Counter)
    for p in mlb:
        if p.get('Pitcher') and p.get('PTeam'):
            c26[p['Pitcher']][p['PTeam']] += 1
    maps[2026] = {n: {'primary': c.most_common(1)[0][0], 'teams': sorted(c), 'n_teams': len(c)}
                  for n, c in c26.items()}
    print(f'  2026: {len(maps[2026])} pitchers (from PTeam)', flush=True)

    pickle.dump(maps, open(OUT, 'wb'))
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
