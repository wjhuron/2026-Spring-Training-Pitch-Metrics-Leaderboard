"""aaa_pitch_pull.py — every Triple-A pitch, 2023-2026, from Savant minors.

The calibration corpus for the AAA-to-MLB channel translation. Pitchers
carry their AAA rate stats into an MLB-scored metric unchanged today, and
the raw outcome channels (K%, in-zone whiff%, GB%, xwOBA, xRV/100) are not
level-neutral: measured on 2026 Rochester arms who pitched at both levels,
K% falls about 5 points and xwOBA rises about .06 on promotion. Stuff+ and
Loc+ do NOT move, which is why they need no translation. The offsets are
fit on pitcher-seasons that appear at both levels in the same year.

WHY 2023 AND NOT 2022. Hawk-Eye reached Triple-A parks for 2023. A 2022
probe returns rows, but arm_angle, release_spin_rate and release_extension
are 0% populated and xwOBA covers 8% of pitches, against 87-96% arm angle
from 2023 on. Only delta_run_exp survives, because it comes from the
play-by-play rather than the tracking. 2022 therefore cannot score Stuff+,
Loc+ or the xwOBA channel, so it is excluded rather than half-used.

SCOPE. player_type=pitcher with team=<club id> returns that club's OWN
pitchers and nobody else (verified: a Rochester-at-Lehigh-Valley window
came back 864/864 ROC pitching). So 30 clubs covers every Triple-A pitcher
exactly once, with no opponent duplication and no batter-side pull.

OUTPUT. One gzipped CSV per (season, club, window) under
data/_aaa_raw/<season>/. The pull is therefore resumable: a present,
non-empty window file is skipped, so an interrupted run picks up where it
stopped. Consolidation into a scoring frame is a separate step.

Savant caps a response at MILB_ROW_CAP rows with no error, so a capped
window is SUBDIVIDED rather than accepted. A window that still caps at one
day is reported as a hole, never passed off as complete.

    python3 scripts/builders/aaa_pitch_pull.py                  # everything
    python3 scripts/builders/aaa_pitch_pull.py --seasons 2024   # one season
    python3 scripts/builders/aaa_pitch_pull.py --clubs 534      # one club
"""
import argparse
import datetime as dt
import gzip
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, 'data', '_aaa_raw')

SEASONS = (2023, 2024, 2025, 2026)
SEASON_START, SEASON_END = '03-01', '10-15'
CHUNK_DAYS = 30              # matches scrapers/backfill_supplement.MILB_CHUNK_DAYS
ROW_CAP = 25000              # matches MILB_ROW_CAP; Savant truncates silently
MIN_CHUNK_DAYS = 1           # below this a cap is a hole, not a subdivision
SEARCH = 'https://baseballsavant.mlb.com/statcast-search-minors/csv'
TEAMS_API = 'https://statsapi.mlb.com/api/v1/teams?sportId=11&season={y}'
UA = 'Huronalytics-Research/1.0 (baseball research; https://huronalytics.com)'


def _get(url, params=None, timeout=180):
    """GET with a browser UA. Returns (status, body-text)."""
    if params:
        url = url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, ''


def fetch_window(club_id, start, end, label):
    """CSV text for one club-window, or None when the window is genuinely
    empty. Retries 5xx, transport errors, and the HTML-error-page-with-200
    that Savant intermittently serves."""
    params = {
        'all': 'true', 'type': 'details',
        'game_date_gt': start, 'game_date_lt': end,
        'team': str(club_id), 'player_type': 'pitcher',
        'min_pitches': '0', 'min_results': '0',
        'sort_col': 'pitches', 'sort_order': 'desc', 'minors': 'true',
    }
    for attempt in range(4):
        try:
            status, body = _get(SEARCH, params)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == 3:
                print(f'    FAIL {label}: {e}')
                return False, None
            time.sleep(5 * (2 ** attempt))
            continue
        if status != 200:
            if status >= 500 and attempt < 3:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f'    FAIL {label}: HTTP {status}')
            return False, None
        if not body or not body.strip() or 'No Results' in body[:200]:
            return True, None
        if body.lstrip()[:1] == '<':
            # HTML error page served with a 200. Not data.
            if attempt < 3:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f'    FAIL {label}: HTML body, not CSV')
            return False, None
        return True, body
    return False, None


def _rows(body):
    return max(0, body.count('\n') - 1)


def _windows(start, end, days):
    out = []
    cur = start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=days - 1), end)
        out.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    return out


def pull_club_season(season, club_id, abbr, holes):
    """Write one gz per window. Returns rows written this call."""
    d = os.path.join(OUT_DIR, str(season))
    os.makedirs(d, exist_ok=True)
    start = dt.date.fromisoformat(f'{season}-{SEASON_START}')
    end = dt.date.fromisoformat(f'{season}-{SEASON_END}')
    total = 0
    queue = list(_windows(start, end, CHUNK_DAYS))
    while queue:
        w0, w1 = queue.pop(0)
        span = (w1 - w0).days + 1
        path = os.path.join(d, f'{club_id}_{w0}_{w1}.csv.gz')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        label = f'{season} {abbr} {w0}..{w1}'
        ok, body = fetch_window(club_id, w0.isoformat(), w1.isoformat(), label)
        if not ok:
            holes.append(label)
            continue
        if body is None:
            # Genuinely empty (off days / out of season). Stamp it so a
            # resumed run does not re-request it.
            with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as f:
                f.write('')
            os.replace(path + '.tmp', path)
            continue
        n = _rows(body)
        if n >= ROW_CAP:
            if span <= MIN_CHUNK_DAYS:
                print(f'    HOLE {label}: capped at {n} rows on a single day')
                holes.append(label + ' (capped)')
                continue
            half = span // 2
            queue.insert(0, (w0 + dt.timedelta(days=half), w1))
            queue.insert(0, (w0, w0 + dt.timedelta(days=half - 1)))
            print(f'    {label}: {n} rows hit the cap, subdividing')
            continue
        with gzip.open(path + '.tmp', 'wt', encoding='utf-8') as f:
            f.write(body)
        os.replace(path + '.tmp', path)
        total += n
    return total


def clubs_for(season):
    status, body = _get(TEAMS_API.format(y=season), timeout=60)
    if status != 200:
        sys.exit(f'ABORT: Triple-A club list for {season} returned HTTP {status}')
    teams = json.loads(body)['teams']
    return sorted((t['id'], t.get('abbreviation') or str(t['id'])) for t in teams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', nargs='*', type=int, default=list(SEASONS))
    ap.add_argument('--clubs', nargs='*', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    holes = []
    t0 = time.time()
    for season in a.seasons:
        clubs = clubs_for(season)
        if a.clubs:
            clubs = [c for c in clubs if c[0] in a.clubs]
        print(f'== {season}: {len(clubs)} Triple-A clubs', flush=True)
        got = 0
        for i, (cid, abbr) in enumerate(clubs, 1):
            n = pull_club_season(season, cid, abbr, holes)
            got += n
            print(f'  [{i:2d}/{len(clubs)}] {abbr:4s} ({cid}) +{n:6d} rows'
                  f'   season total {got:8d}   {time.time()-t0:6.0f}s',
                  flush=True)
        print(f'== {season} done: {got} new rows\n', flush=True)
    if holes:
        print(f'HOLES ({len(holes)}) — these windows have NO data on disk:')
        for h in holes:
            print(f'   {h}')
        print('Re-run to retry them; present windows are skipped.')
    else:
        print('No holes.')
    print(f'total wall clock {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
