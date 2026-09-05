"""era_targets_build.py — official pitcher lines per season and half, from the
MLB Stats API, for the ERA-estimator replicates (2026-09-05 rebuild; the
original data/_era_targets.json had no builder in the repo and was lost).

Schema (consumed by era_estimator_screen / era_weights_final / the battery,
cmdloc, xrv and stuff builders):
  {season: {'asg': 'YYYY-MM-DD',
            'pitchers': {pid: {'name', 'hand', 'teams': [club ids],
                               'outs','er','r','bf','so','bb','g','gs','ip',
                               'h1': {outs, er, r, bf, so, bb, g, gs},
                               'h2': {...}}}}}
full  = stats?stats=season (one row per pitcher, regular season)
h1/h2 = stats?stats=byDateRange, season start..ASG date and ASG+1..end
teams = person hydrate season splits (one per club stint); hand = pitchHand
Usage: python3 scripts/research/era/era_targets_build.py [--seasons 2021,...]
"""
import argparse, datetime as dt, json, os, sys, time
import requests
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'data', '_era_targets.json')
B = 'https://statsapi.mlb.com/api/v1'
KEYS = {'outs': 'outs', 'er': 'earnedRuns', 'r': 'runs', 'bf': 'battersFaced', 'so': 'strikeOuts',
        'bb': 'baseOnBalls', 'g': 'gamesPlayed', 'gs': 'gamesStarted',
        # 2026-09-05 hWAR battery: running game, HR and HBP for FIP, IBB, balks, wild pitches
        'sb': 'stolenBases', 'cs': 'caughtStealing', 'hr': 'homeRuns', 'hbp': 'hitBatsmen',
        'ibb': 'intentionalWalks', 'bk': 'balks', 'wp': 'wildPitches'}

def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f'{B}/{path}', params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))

def asg_date(season):
    d = get('schedule', sportId=1, gameType='A', season=season)
    ds = [x['date'] for x in d.get('dates', []) if any(g.get('gameType') == 'A' for g in x.get('games', []))]
    if not ds:
        raise SystemExit(f'{season}: no All-Star game on the schedule')
    return ds[0]

def line(stat):
    out = {k: int(stat.get(v) or 0) for k, v in KEYS.items()}
    out['ip'] = stat.get('inningsPitched')
    return out

def bulk(season, **extra):
    p = dict(group='pitching', season=season, gameType='R', sportId=1, limit=5000, playerPool='ALL', **extra)
    d = get('stats', **p)
    s = d['stats'][0] if d.get('stats') else {}
    if s.get('totalSplits', 0) != len(s.get('splits', [])):
        raise SystemExit(f'{season} {extra}: totalSplits {s.get("totalSplits")} != returned {len(s.get("splits", []))}')
    return {int(sp['player']['id']): (sp['player'].get('fullName'), line(sp['stat'])) for sp in s.get('splits', [])}

def stints(season, pids):
    """pid -> (hand, [club ids with a regular-season pitching stint])"""
    out = {}
    pids = list(pids)
    for i in range(0, len(pids), 100):
        batch = pids[i:i + 100]
        d = get('people', personIds=','.join(map(str, batch)),
                hydrate=f'stats(group=[pitching],type=[season],season={season},gameType=[R])')
        for p in d.get('people', []):
            teams = []
            for st in p.get('stats', []):
                for sp in st.get('splits', []):
                    t = (sp.get('team') or {}).get('id')
                    if t is not None and (sp['stat'].get('outs') or 0) > 0 and t not in teams:
                        teams.append(int(t))
            out[int(p['id'])] = ((p.get('pitchHand') or {}).get('code'), teams)
        time.sleep(0.25)
    return out

def build(season):
    asg = asg_date(season)
    d_asg = dt.date.fromisoformat(asg)
    end = min(dt.date(season, 11, 15), dt.date.today())
    full = bulk(season, stats='season')
    h1 = bulk(season, stats='byDateRange', startDate=f'{season}-03-01', endDate=asg)
    h2 = bulk(season, stats='byDateRange', startDate=(d_asg + dt.timedelta(days=1)).isoformat(), endDate=end.isoformat())
    st = stints(season, full.keys())
    zero = {k: 0 for k in KEYS}
    pitchers = {}
    bad = 0
    for pid, (name, ln) in full.items():
        hand, teams = st.get(pid, (None, []))
        rec = dict(ln, name=name, hand=hand, teams=teams,
                   h1=h1.get(pid, (None, dict(zero)))[1], h2=h2.get(pid, (None, dict(zero)))[1])
        if rec['h1']['outs'] + rec['h2']['outs'] != rec['outs']:
            bad += 1
        pitchers[str(pid)] = rec
    n_l = sum(1 for r in pitchers.values() if r['hand'] == 'L')
    print(f'{season}: asg {asg}, {len(pitchers)} pitchers ({n_l} LHP), h1 rows {len(h1)}, h2 rows {len(h2)}, '
          f'halves != full outs: {bad}, no stint: {sum(1 for r in pitchers.values() if not r["teams"])}', flush=True)
    return {'asg': asg, 'pitchers': pitchers}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seasons', default='2021,2022,2023,2024,2025,2026')
    a = ap.parse_args()
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for y in [int(s) for s in a.seasons.split(',')]:
        out[str(y)] = build(y)
        time.sleep(0.5)
    tmp = OUT + '.tmp'
    json.dump(out, open(tmp, 'w'))
    os.replace(tmp, OUT)
    print(f'wrote {OUT}')

if __name__ == '__main__':
    main()
