#!/usr/bin/env python3
"""enumerate_missing_ibb.py — list every no-pitch intentional walk that is
absent from the pitch data. READ-ONLY: writes a CSV, touches nothing else.

Step 1 of the IBB backfill. Since 2017 an automatic IBB contains no pitches,
so it cannot appear in pitch-level data. completeness_audit.py already treats
that as a by-design absence; this turns the excuse into a work list.

Approach, cheapest path first:
  1. data/boxscore_cache.json already carries per-game, per-hitter `ibb`, so
     the candidate games are known offline - 326 of 1,885, not all of them.
     NOTE the cache lists 23 gamePks under two date keys, three of them with
     disagreeing IBB totals, so games are de-duplicated by MAX before counting.

ROC/AAA (--source roc) is covered too, on a different path, because the MLB
boxscore cache holds no MiLB games. Rochester's game_pks come from the pitch
cache instead, and each one needs a boxscore fetch as well as a playByPlay so
the pitcher's TEAM NAME is known. Routing follows the scraper's own rule
(pitcher2026.normalize_aaa_labels): "Rochester Red Wings" becomes ROC and
every other club becomes AAA, so a Rochester pitcher's row lands in the ROC
tab and every opposing pitcher's row lands in AAA.
  2. Anything already present as a PITCHED intent walk is subtracted, using
     the pitch cache keyed on (game_pk, batter).
  3. Only the remaining games get a playByPlay fetch. A play qualifies when
     result.eventType == 'intent_walk' AND no playEvent has isPitch.

The boxscore cache also supplies batter and pitcher names ALREADY in the
sheets' "Last, First" form plus team abbreviations, so no name canonicalization
is needed here.

Output columns are what the write step needs to build a sheet row. Note two
deliberate blanks:
  * Runners - filled from the Statcast supplement in the normal path, which
    has no row for a pitch that was never thrown. Nothing a no-pitch row
    reaches consumes it.
  * every pitch measurement - blank is what keeps these rows out of the
    batted-ball and zone filters (fetch.py turns blanks into None).

Rows are routed to the PITCHER's team tab: every row in the cache has
_sheet_tab == PTeam.

Usage:
    python3 scripts/audits/enumerate_missing_ibb.py [--out PATH] [--limit N]
"""
import argparse
import csv
import json
import os
import pickle
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')
PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
BOX = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
ROCHESTER = 'Rochester Red Wings'      # pitcher2026.normalize_aaa_labels
_NAME_CACHE = {}


def _fetch(pk):
    try:
        req = urllib.request.Request(PBP.format(pk=pk),
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return pk, json.loads(r.read())
    except Exception as e:
        return pk, {'_error': f'{type(e).__name__}: {e}'}


def _get(url, pk):
    try:
        req = urllib.request.Request(url.format(pk=pk),
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {'_error': f'{type(e).__name__}: {e}'}


def _lastfirst(pid, fullname=None):
    """Canonical "Last, First" for a player id. MiLB boxscores omit
    lastFirstName, so fall back to the people API (cached)."""
    if pid in _NAME_CACHE:
        return _NAME_CACHE[pid]
    out = None
    d = _get("https://statsapi.mlb.com/api/v1/people/{pk}", pid)
    if not d.get('_error'):
        people = d.get('people') or []
        if people:
            out = people[0].get('lastFirstName')
    if not out and fullname and ' ' in fullname:
        first, _, last = fullname.partition(' ')
        out = f"{last}, {first}"
    _NAME_CACHE[pid] = out
    return out


def _roc_rows(have, verbose=True):
    """No-pitch IBBs in Rochester games. Game list comes from the pitch cache,
    since the MLB boxscore cache has no MiLB games."""
    with open(os.path.join(DATA, 'all_pitches_rs_cache.pkl'), 'rb') as f:
        pks = sorted({int(str(p['PitchID']).split('_')[0])
                      for p in pickle.load(f)
                      if p.get('_sheet_tab') in ('ROC', 'AAA') and p.get('PitchID')})
    print(f"ROC/AAA: {len(pks)} games from the pitch cache")
    rows, errors = [], []
    for n, pk in enumerate(pks, 1):
        if verbose and n % 25 == 0:
            print(f"    {n}/{len(pks)}...")
        pbp = _get(PBP, pk)
        if pbp.get('_error'):
            errors.append((pk, pbp['_error']))
            continue
        iw = [p for p in pbp.get('allPlays', [])
              if (p.get('result') or {}).get('eventType') == 'intent_walk'
              and not any(e.get('isPitch') for e in (p.get('playEvents') or []))]
        if not iw:
            continue
        box = _get(BOX, pk)                      # only fetched when needed
        if box.get('_error'):
            errors.append((pk, box['_error']))
            continue
        team_of, date = {}, None
        for side in ('away', 'home'):
            td = (box.get('teams') or {}).get(side) or {}
            tname = ((td.get('team') or {}).get('name'))
            tag = 'ROC' if tname == ROCHESTER else 'AAA'
            for key, pl in (td.get('players') or {}).items():
                pid = (pl.get('person') or {}).get('id')
                if pid:
                    team_of[pid] = tag
        for play in iw:
            about, m = play.get('about') or {}, play.get('matchup') or {}
            bid = (m.get('batter') or {}).get('id')
            pid = (m.get('pitcher') or {}).get('id')
            ab = about.get('atBatIndex')
            rows.append({
                'gamePk': pk,
                'gameDate': (about.get('startTime') or '')[:10] or None,
                'atBatIndex': ab,
                'PitchID': f"{pk}_{ab:03d}_00" if ab is not None else '',
                'sheetTab': team_of.get(pid),
                'PTeam': team_of.get(pid),
                'Pitcher': _lastfirst(pid, (m.get('pitcher') or {}).get('fullName')),
                'pitcherMlbId': pid,
                'Throws': (m.get('pitchHand') or {}).get('code'),
                'BTeam': team_of.get(bid),
                'Batter': _lastfirst(bid, (m.get('batter') or {}).get('fullName')),
                'batterMlbId': bid,
                'Bats': (m.get('batSide') or {}).get('code'),
                'Outs': (play.get('count') or {}).get('outs'),
                'inning': about.get('inning'),
                'halfInning': about.get('halfInning'),
                'Event': 'Intent Walk',
                'alreadyInData': int(bool(have.get((pk, None)))),
            })
    if errors:
        print(f"  *** {len(errors)} ROC games failed to fetch: {errors[:3]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['mlb', 'roc', 'both'], default='both')
    ap.add_argument('--out', default=os.path.join(DATA, '_missing_ibb.csv'))
    ap.add_argument('--limit', type=int, default=0, help='cap games, for a smoke test')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    with open(os.path.join(DATA, 'boxscore_cache.json')) as f:
        box = json.load(f)
    do_mlb = a.source in ('mlb', 'both')
    do_roc = a.source in ('roc', 'both')

    # ── candidate games + the identity maps the sheet row needs ──
    # 23 gamePks appear under two date keys, and three of those disagree on
    # their IBB total (one copy reads 0). Summing naively double-counts by 5,
    # so take the MAX per gamePk and count each game once.
    cand, date_of, hitter_of, pitcher_team = {}, {}, {}, {}
    for date, games in box.items():
        for g in games:
            pk = g.get('gamePk')
            for p in g.get('pitchers', []):
                if p.get('mlbId'):
                    pitcher_team[(pk, p['mlbId'])] = p.get('team')
            ibb_hitters = [h for h in g.get('hitters', []) if h.get('ibb')]
            if not ibb_hitters:
                continue
            total = sum(h['ibb'] for h in ibb_hitters)
            if total > cand.get(pk, 0):
                cand[pk] = total
                date_of[pk] = date
            for h in ibb_hitters:
                hitter_of[(pk, h['mlbId'])] = (h.get('name'), h.get('team'))
    official = sum(cand.values())
    n_games = len({g['gamePk'] for gs in box.values() for g in gs})
    print(f"boxscores: {official} official IBB across {len(cand)} games "
          f"(of {n_games} distinct games)")

    # ── subtract the IBBs that WERE pitched and are already in the data ──
    have = defaultdict(int)
    with open(os.path.join(DATA, 'all_pitches_rs_cache.pkl'), 'rb') as f:
        for p in pickle.load(f):
            if p.get('Event') == 'Intent Walk' and p.get('PitchID'):
                have[(int(str(p['PitchID']).split('_')[0]), p.get('Batter'))] += 1
    print(f"pitch cache already holds {sum(have.values())} pitched intent walks")

    if not do_mlb:
        cand = {}
    pks = sorted(cand)
    if a.limit:
        pks = pks[:a.limit]
        print(f"  --limit {a.limit}: fetching {len(pks)} games only")
    print(f"fetching playByPlay for {len(pks)} games ({a.workers} workers)...")

    rows, errors, pitched_seen = [], [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for pk, pbp in ex.map(_fetch, pks):
            if pbp.get('_error'):
                errors.append((pk, pbp['_error']))
                continue
            for play in pbp.get('allPlays', []):
                if (play.get('result') or {}).get('eventType') != 'intent_walk':
                    continue
                events = play.get('playEvents') or []
                if any(e.get('isPitch') for e in events):
                    pitched_seen += 1
                    continue            # already representable, and present
                about = play.get('about') or {}
                m = play.get('matchup') or {}
                bid = (m.get('batter') or {}).get('id')
                pid = (m.get('pitcher') or {}).get('id')
                bname, bteam = hitter_of.get((pk, bid), (None, None))
                pteam = pitcher_team.get((pk, pid))
                ab = about.get('atBatIndex')
                rows.append({
                    'gamePk': pk,
                    'gameDate': date_of.get(pk),
                    'atBatIndex': ab,
                    'PitchID': f"{pk}_{ab:03d}_00" if ab is not None else '',
                    'sheetTab': pteam,          # rows live in the pitcher's tab
                    'PTeam': pteam,
                    'Pitcher': (m.get('pitcher') or {}).get('fullName'),
                    'pitcherMlbId': pid,
                    'Throws': (m.get('pitchHand') or {}).get('code'),
                    'BTeam': bteam,
                    'Batter': bname,
                    'batterMlbId': bid,
                    'Bats': (m.get('batSide') or {}).get('code'),
                    'Outs': (play.get('count') or {}).get('outs'),
                    'inning': about.get('inning'),
                    'halfInning': about.get('halfInning'),
                    'Event': 'Intent Walk',
                    'alreadyInData': int(bool(have.get((pk, bname)))),
                })

    n_mlb = len(rows)
    if do_roc:
        rows.extend(_roc_rows(have))
        print(f"  ROC/AAA no-pitch intent walks: {len(rows) - n_mlb}")
    rows.sort(key=lambda r: (r['gameDate'] or '', r['gamePk'], r['atBatIndex'] or 0))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['gamePk'])
        w.writeheader()
        w.writerows(rows)

    unresolved = [r for r in rows if not r['sheetTab'] or not r['Batter']]
    n_roc = len(rows) - n_mlb
    print(f"\nno-pitch intent walks found: {len(rows)}"
          f"  (MLB {n_mlb}, ROC/AAA {n_roc})")
    print(f"  pitched intent walks skipped (already representable): {pitched_seen}")
    if do_mlb:
        ok = (n_mlb + pitched_seen) == official
        print(f"  MLB reconciliation: {n_mlb} + {pitched_seen} = "
              f"{n_mlb + pitched_seen} against {official} official "
              f"{'OK' if ok else '*** MISMATCH ***'}")
    if do_roc:
        print(f"  ROC/AAA has no official IBB source to reconcile against "
              f"(the boxscore cache is MLB-only), so {n_roc} is feed-derived only")
    if unresolved:
        print(f"  *** {len(unresolved)} rows missing a team or batter name — "
              f"these need resolving before any write")
    if errors:
        print(f"  *** {len(errors)} games failed to fetch: {errors[:3]}")
    from collections import Counter
    per_tab = Counter(r['sheetTab'] for r in rows)
    print(f"  spread over {len(per_tab)} tabs, "
          f"max {max(per_tab.values()) if per_tab else 0} rows in one tab")
    print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
