"""air_density_screen.py — is there air-density signal in the wOBA-vs-xwOBA
residual that xwRC+ and hdERA inherit?

Physics prior (Nathan): about 4 ft of fly-ball carry per 10 degrees F,
and thin air at elevation carries further. Savant xwOBA is trained
pooled across environments, so a given EV/LA is worth more real runs at
Coors and less at sea level; the residual (wOBA - xwOBA) should then
correlate with elevation exposure if the bias is material.

This is a SCREEN, not an adjustment battery: it measures whether the
signal exists before anything is built. Elevation only — the caches
carry no game-time temperature, so the temperature half of air density
is out of scope and the screen understates the full effect.

  Pitcher side (hdERA): per pitcher-season, residual = wOBA against
  minus xwOBA against (era battery, full scope), against the HOME club's
  park elevation. r per season 2021-2026, and again with Colorado rows
  excluded (is it Coors or a gradient?).

  Hitter side (xwRC+): per hitter-season 2021-2025 (public caches),
  residual = (wOBA - xwOBA) per denominator, against the hitter's
  BIP-weighted mean venue elevation (schedule API game -> venue map).
  r per season, and again with hitters whose mean exposure exceeds
  3000 ft excluded.

Venue elevations come from data/_park_elevation.json (venues block) plus
a fixed fill for pre-2026 home venues the 2026 file lacks (Oakland
Coliseum 43 ft, Sutter Health Park 30 ft, Steinbrenner Field 26 ft —
physical facts, not tuning constants).

Usage: python3 scripts/research/era/air_density_screen.py
Output: console + data/_air_density_screen.json
"""
import json
import math
import os
import pickle
import sys
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from era_estimator_screen import BATTERY, pearson

TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
ELEV = json.load(open(os.path.join(ROOT, 'data', '_park_elevation.json')))
VENUE_ELEV = {int(k): v['elev_ft'] for k, v in ELEV['venues'].items()}
VENUE_ELEV.update({2395: 43,        # Oakland Coliseum
                   2603: 30,        # Sutter Health Park (2025+ ATH)
                   2523: 26})       # George M. Steinbrenner Field (2025 TB)
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
HITTER_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PA = 300
BASE = 'https://statsapi.mlb.com/api/v1'
COL_ID = 115

WOBA_W = {'walk': .69, 'hit_by_pitch': .72, 'single': .89, 'double': 1.27,
          'triple': 1.61, 'home_run': 2.10}
NON_DENOM = {'intent_walk', 'sac_bunt', 'catcher_interf',
             'sac_bunt_double_play'}
NON_PA_TOKENS = ('stealing', 'pickoff', 'stolen', 'wild_pitch',
                 'passed_ball', 'truncated', 'game_advisory')
BIP_DESC = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def team_venue_map(season):
    d = get(f'{BASE}/teams?sportId=1&season={season}')
    return {t['id']: (t.get('venue') or {}).get('id')
            for t in d.get('teams', [])}


def game_venue_map(season):
    path = os.path.join(ROOT, 'data', '_feed_cache',
                        f'_sched_venues_{season}.json')
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    d = get(f'{BASE}/schedule?sportId=1&season={season}&gameTypes=R'
            f'&hydrate=venue')
    out = {}
    for dd in d.get('dates', []):
        for g in dd.get('games', []):
            vid = (g.get('venue') or {}).get('id')
            if vid is not None:
                out[g['gamePk']] = vid
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f)
    return out


def pitcher_screen():
    print('=== PITCHER SIDE: (wOBA - xwOBA) against vs home elevation ===')
    res = {}
    for y in SEASONS:
        tv = team_venue_map(y)
        rows, rows_nocol = [], []
        miss = set()
        for pid, brec in BATTERY.get(str(y), {}).items():
            m = brec['full']
            line = TARGETS[str(y)]['pitchers'].get(pid)
            if (line is None or line['outs'] < 180
                    or m.get('woba') is None or m.get('xwoba') is None):
                continue
            club = line['teams'][0] if line.get('teams') else None
            vid = tv.get(club)
            elev = VENUE_ELEV.get(vid)
            if elev is None:
                if vid is not None:
                    miss.add(vid)
                continue
            resid = m['woba'] - m['xwoba']
            rows.append((elev, resid))
            if club != COL_ID:
                rows_nocol.append((elev, resid))
        r_all = pearson([e for e, _ in rows], [x for _, x in rows])
        r_noc = pearson([e for e, _ in rows_nocol],
                        [x for _, x in rows_nocol])
        if miss:
            print(f'  {y}: WARNING unmapped venues {sorted(miss)}')
        print(f'  {y}: n {len(rows):3d}  r {r_all:+.3f}   '
              f'no-COL n {len(rows_nocol):3d}  r {r_noc:+.3f}')
        res[str(y)] = {'n': len(rows), 'r': r_all, 'r_nocol': r_noc}
    return res


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def hitter_screen():
    print('\n=== HITTER SIDE: (wOBA - xwOBA) vs BIP elevation exposure ===')
    res = {}
    for y in HITTER_SEASONS:
        gv = game_venue_map(y)
        df = pickle.load(open(os.path.join(
            ROOT, 'data', f'_statcast{y}_cache.pkl'), 'rb'))
        df = df[df['game_type'] == 'R']
        ev = df[df['events'].notna() & (df['events'] != '')]
        acc = {}
        for row in ev[['batter', 'events', 'description', 'game_pk',
                       'estimated_woba_using_speedangle']].itertuples(
                index=False):
            e = row.events
            if not isinstance(e, str) or any(t in e for t in NON_PA_TOKENS):
                continue
            bid = str(int(row.batter))
            r = acc.setdefault(bid, {'den': 0, 'wnum': 0.0, 'xnum': 0.0,
                                     'esum': 0.0, 'en': 0, 'pa': 0})
            r['pa'] += 1
            if e in NON_DENOM:
                continue
            r['den'] += 1
            r['wnum'] += WOBA_W.get(e, 0.0)
            is_bip = isinstance(row.description, str) \
                and row.description in BIP_DESC
            if is_bip:
                xw = _f(row.estimated_woba_using_speedangle)
                r['xnum'] += xw if xw is not None else WOBA_W.get(e, 0.0)
                elev = VENUE_ELEV.get(gv.get(int(row.game_pk)))
                if elev is not None:
                    r['esum'] += elev
                    r['en'] += 1
            else:
                r['xnum'] += WOBA_W.get(e, 0.0)
        rows = []
        for bid, r in acc.items():
            if r['pa'] < MIN_PA or r['den'] == 0 or r['en'] == 0:
                continue
            rows.append((r['esum'] / r['en'],
                         (r['wnum'] - r['xnum']) / r['den']))
        low = [(e, x) for e, x in rows if e <= 3000]
        r_all = pearson([e for e, _ in rows], [x for _, x in rows])
        r_low = pearson([e for e, _ in low], [x for _, x in low])
        print(f'  {y}: n {len(rows):3d}  r {r_all:+.3f}   '
              f'sub-3000ft n {len(low):3d}  r {r_low:+.3f}')
        res[str(y)] = {'n': len(rows), 'r': r_all, 'r_sub3000': r_low}
    return res


def main():
    out = {'pitcher': pitcher_screen(), 'hitter': hitter_screen()}
    tmp = os.path.join(ROOT, 'data', '_air_density_screen.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_air_density_screen.json'))
    print('\nwrote data/_air_density_screen.json')


if __name__ == '__main__':
    main()
