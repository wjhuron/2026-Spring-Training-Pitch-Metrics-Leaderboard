"""era_team_outs_pull.py — per-team outs per pitcher-season, 2021-2026.

Sidecar to era_targets_pull.py. That pull uses the BULK stats endpoint,
which returns ONE season-combined row per pitcher with a single team id
(the final club) — so _era_targets.json 'teams' never holds two clubs and
park_exposure scored every traded pitcher against his LAST park only.
Verified 2026-08-24 on Scherzer 2023: bulk returns (TEX, 152.2 IP), the
person hydrate returns (NYM 107.2, TEX 45.0).

This pull batches person ids through /people?hydrate=stats(...), which
DOES split stints by team, for two scopes that mirror the weight-fit
harness: full season, and first half (03-01 .. ASG date, what the ROS fit
z-scores). ASG dates are reused from _era_targets.json, so run
era_targets_pull.py first.

Rows with team None are the API's own combined line; skipped. Duplicate
splits (the API sometimes repeats one) collapse by (pid, team) assignment.

Output: data/_era_team_outs.json
  {season: {pid: {'full': {team_id: outs}, 'h1': {team_id: outs}}}}
"""
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'data', '_era_team_outs.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
BASE = 'https://statsapi.mlb.com/api/v1'
BATCH = 100


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _outs(st):
    outs = st.get('outs')
    if outs is None:
        ip = st.get('inningsPitched') or '0.0'
        whole, _, frac = str(ip).partition('.')
        outs = int(whole or 0) * 3 + int(frac or 0)
    return int(outs or 0)


def hydrate_outs(pids, hyd):
    """{pid: {tid: outs}} via batched person hydrate; combined (team None)
    rows skipped, duplicate splits collapsed by assignment."""
    out = {}
    for i in range(0, len(pids), BATCH):
        chunk = ','.join(str(p) for p in pids[i:i + BATCH])
        d = get(f'{BASE}/people?personIds={chunk}&hydrate={hyd}')
        for p in d.get('people', []):
            pid = p.get('id')
            for st in p.get('stats', []):
                for sp in st.get('splits', []):
                    tid = (sp.get('team') or {}).get('id')
                    if tid is None:
                        continue
                    out.setdefault(pid, {})[str(tid)] = _outs(
                        sp.get('stat') or {})
    return out


def main():
    result = {}
    for season in SEASONS:
        asg = TARGETS[str(season)]['asg']
        pids = sorted(int(p) for p in TARGETS[str(season)]['pitchers'])
        y, m, d = asg.split('-')
        full = hydrate_outs(pids, ('stats(group=[pitching],type=[season],'
                                   f'season={season})'))
        h1 = hydrate_outs(pids, ('stats(group=[pitching],'
                                 f'type=[byDateRange],'
                                 f'startDate=03/01/{season},'
                                 f'endDate={m}/{d}/{season},'
                                 f'season={season})'))
        srec = {}
        for pid in pids:
            ft = full.get(pid)
            if not ft:
                continue
            srec[str(pid)] = {'full': ft, 'h1': h1.get(pid, {})}
        result[str(season)] = srec
        n_multi = sum(1 for v in srec.values() if len(v['full']) > 1)
        print(f'{season}: {len(srec)} pitchers, {n_multi} multi-team',
              flush=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f)
    os.replace(tmp, OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
