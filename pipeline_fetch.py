#!/usr/bin/env python3
"""Data fetching: Google Sheets, MLB Stats API, FanGraphs, Savant, boxscore caching."""

import gspread
import json
import os
import re
import time as time_module
import urllib.request
import urllib.parse
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from guts import scrape_guts
from pipeline_utils import (
    DATA_DIR, MLB_TEAMS, ALL_TEAMS, TEAM_ABBREV_TO_ID,
    _today_et, _fullname_to_lastfirst, box_key, is_barrel,
)

# ── Config ───────────────────────────────────────────────────────────────
SPREADSHEET_IDS = {
    'AL': os.environ.get('SPREADSHEET_ID_AL', '1hzAtZ_Wqi8ZuUHaGvgjJcQMU5jj5CzGXuBtjYmPOj9U'),
    'NL': os.environ.get('SPREADSHEET_ID_NL', '1DH3NI-3bSXW7dl98tdg5uFgJ4O6aWRvRB_XnVb340YE'),
}
SERVICE_ACCOUNT_FILE = os.environ.get(
    'GOOGLE_SERVICE_ACCOUNT_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_account.json')
)

# ── Fallback constants ───────────────────────────────────────────────────
WOBA_WEIGHTS_FALLBACK = {
    'BB': 0.692, 'HBP': 0.723, '1B': 0.884, '2B': 1.256, '3B': 1.591, 'HR': 2.048,
}
FIP_CONSTANT_FALLBACK = 3.102

# ── MLB ID lookup ────────────────────────────────────────────────────────
MANUAL_MLB_IDS = {
    'Kayfus, CJ|CLE': 692216,
}

# ── Boxscore cache paths ────────────────────────────────────────────────
BOXSCORE_CACHE_PATH = os.path.join(DATA_DIR, 'boxscore_cache.json')
MILB_BOXSCORE_CACHE_PATH = os.path.join(DATA_DIR, 'milb_boxscore_cache.json')

# Boxscores on/after (today - this many days) are refetched every run so late
# scoring changes (hits/errors, earned/unearned runs) get picked up; older dates
# are served from the committed cache instead of refetching the whole season.
# MLB scoring corrections settle within days, so 14 is comfortably safe.
BOXSCORE_REFRESH_WINDOW_DAYS = 14

# A cached hitter position stays valid this many days. Position is "most games
# at a spot this season" (stable midseason), so a few days' staleness avoids
# refetching ~600 hitters every day. New (uncached) players are always fetched.
HITTER_POSITION_CACHE_DAYS = 7

# Concurrency for MLB Stats API fetches (boxscores, positions). _fetch_with_retry
# backs off on transient throttling; lower this if the API starts rate-limiting.
FETCH_MAX_WORKERS = 8

# ── MiLB team configuration ─────────────────────────────────────────────
MILB_TEAMS_CONFIG = {
    'ROC': {
        'sport_id': 11,
        'search_name': 'Rochester',
        'api_name': 'Rochester Red Wings',
    },
}
MILB_TEAM_NAME_TO_ABBREV = {
    'Rochester Red Wings': 'ROC',
}

# ── Full team name → abbreviation ────────────────────────────────────────
TEAM_NAME_TO_ABBREV = {
    'Arizona Diamondbacks': 'ARI', 'Athletics': 'ATH', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS', 'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET', 'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR', 'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM', 'New York Yankees': 'NYY', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}


# ── HTTP helpers ─────────────────────────────────────────────────────────

def _fetch_with_retry(url, headers=None, timeout=15, retries=3):
    """Fetch URL with retry logic and exponential backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time_module.sleep(2 ** attempt)
    raise last_err


# ── FanGraphs / Savant data ─────────────────────────────────────────────

def fetch_guts_constants(year=2026):
    """Scrape wOBA weights and cFIP from FanGraphs Guts page."""
    row = scrape_guts(year)
    weights = {
        'BB': round(row['wBB'], 3),
        'HBP': round(row['wHBP'], 3),
        '1B': round(row['w1B'], 3),
        '2B': round(row['w2B'], 3),
        '3B': round(row['w3B'], 3),
        'HR': round(row['wHR'], 3),
    }
    cfip = round(row['cFIP'], 3)
    guts_extra = {
        'wOBAScale': round(row['wOBAScale'], 4),
        'lgWOBA': round(row['wOBA'], 4),
        'lgRPA': round(row['R/PA'], 4),
    }
    print(f"  FanGraphs Guts {year}: wBB={weights['BB']}, wHBP={weights['HBP']}, "
          f"w1B={weights['1B']}, w2B={weights['2B']}, w3B={weights['3B']}, "
          f"wHR={weights['HR']}, cFIP={cfip}")
    print(f"  wOBA Scale={guts_extra['wOBAScale']}, League wOBA={guts_extra['lgWOBA']}, "
          f"League R/PA={guts_extra['lgRPA']}")
    return weights, cfip, guts_extra


def fetch_sprint_speed(year=2026):
    """Fetch sprint speed leaderboard from Baseball Savant."""
    import csv
    import io
    url = (f'https://baseballsavant.mlb.com/leaderboard/sprint_speed'
           f'?type=raw&year={year}&position=&team=&min=1&csv=true')
    try:
        raw = _fetch_with_retry(url, headers={
            'User-Agent': 'Huronalytics-Leaderboard/1.0 (baseball research; https://huronalytics.com)',
            'Accept': 'text/csv',
        }, timeout=30)
        data = raw.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(data))
        result = {}
        for row in reader:
            try:
                mlb_id = int(row.get('player_id') or 0)
                speed_str = row.get('sprint_speed') or row.get('hp_to_1b') or ''
                speed = float(speed_str) if speed_str else 0
                comp_runs = int(row.get('competitive_runs') or 0)
                if mlb_id and speed > 0:
                    result[mlb_id] = {'speed': round(speed, 1), 'competitive_runs': comp_runs}
            except (ValueError, TypeError):
                continue
        qualified = sum(1 for v in result.values() if v['competitive_runs'] >= 10)
        print(f"  Sprint speed: fetched {len(result)} players from Savant ({year}), {qualified} qualified (≥10 runs)")
        return result
    except Exception as e:
        print(f"  WARNING: Could not fetch sprint speed data ({type(e).__name__}): {e}")
        return {}


def fetch_park_factors(year=2026):
    """Scrape park factors from FanGraphs."""
    import re as _re
    url = f'https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season={year}'
    html = _fetch_with_retry(url, headers={
        'User-Agent': 'Huronalytics-Leaderboard/1.0 (baseball research; https://huronalytics.com)',
        'Accept': 'text/html',
    }, timeout=15).decode('utf-8')
    match = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, _re.DOTALL)
    if not match:
        raise RuntimeError('Could not find __NEXT_DATA__ on FanGraphs park factors page')
    data = json.loads(match.group(1))
    props = data.get('props', {})
    page_props = props.get('pageProps', {})
    dehydrated = page_props.get('dehydratedState', {})
    queries = dehydrated.get('queries', [])
    if not queries:
        raise RuntimeError('FanGraphs park factors page structure changed — no queries in __NEXT_DATA__')
    FG_TEAM_MAP = {
        'Angels': 'LAA', 'Orioles': 'BAL', 'Red Sox': 'BOS', 'White Sox': 'CWS',
        'Guardians': 'CLE', 'Tigers': 'DET', 'Royals': 'KCR', 'Twins': 'MIN',
        'Yankees': 'NYY', 'Athletics': 'ATH', 'Mariners': 'SEA', 'Rays': 'TBR',
        'Rangers': 'TEX', 'Blue Jays': 'TOR',
        'Diamondbacks': 'ARI', 'Braves': 'ATL', 'Cubs': 'CHC', 'Reds': 'CIN',
        'Rockies': 'COL', 'Marlins': 'MIA', 'Astros': 'HOU', 'Dodgers': 'LAD',
        'Brewers': 'MIL', 'Nationals': 'WSH', 'Mets': 'NYM', 'Phillies': 'PHI',
        'Pirates': 'PIT', 'Cardinals': 'STL', 'Padres': 'SDP', 'Giants': 'SFG',
    }
    park_factors = {}
    for q in queries:
        rows = q.get('state', {}).get('data', [])
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and 'Team' in rows[0]:
            for row in rows:
                team_name = row.get('Team')
                if not team_name:
                    continue
                abbr = FG_TEAM_MAP.get(team_name)
                basic = row.get('Basic (5yr)')
                if abbr and basic is not None:
                    park_factors[abbr] = round(basic / 100, 4)
    print(f"  Park factors: {len(park_factors)} teams fetched")
    return park_factors


# ── Google Sheets reading ────────────────────────────────────────────────

def sheets_call_with_retry(fn, max_retries=5):
    """Run any gspread call with retry logic for rate-limit (429) and
    transient backend errors (500, 502, 503, 504). Backs off exponentially.
    Google returns sporadic 503s on metadata calls too (open_by_key killed a
    CI run on 2026-07-13), so every Sheets round-trip goes through this."""
    transient_codes = ('429', '500', '502', '503', '504')
    for attempt in range(max_retries):
        try:
            return fn()
        except gspread.exceptions.APIError as e:
            msg = str(e)
            code = next((c for c in transient_codes if c in msg), None)
            if code and attempt < max_retries - 1:
                wait = min(60, 5 * (2 ** attempt))  # 5, 10, 20, 40, 60 s
                label = 'Rate limited' if code == '429' else f'Transient {code}'
                print(f"    {label}, waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries - 1})...")
                time_module.sleep(wait)
            else:
                raise


def read_sheet_with_retry(ws, max_retries=5):
    """Read a worksheet's values through the transient-error retry wrapper."""
    return sheets_call_with_retry(ws.get_all_values, max_retries)


def read_pitches_from_sheet(gc, sheet_id, extra_tabs=None):
    """Read all pitches from a single Google Sheets spreadsheet. Returns a list of pitch dicts."""
    pitches = []
    extra_tabs = extra_tabs or set()
    sh = sheets_call_with_retry(lambda: gc.open_by_key(sheet_id))
    tabs = sheets_call_with_retry(sh.worksheets)
    print(f"  {sh.title} ({len(tabs)} tabs)")
    for i, ws in enumerate(tabs):
        tab_name = ws.title
        is_extra = tab_name in extra_tabs
        if tab_name not in MLB_TEAMS and not is_extra:
            print(f"    Skipping {tab_name} (not a team sheet)")
            continue
        print(f"    Reading {ws.title}...")
        if i > 0:
            time_module.sleep(0.5)
        rows = read_sheet_with_retry(ws)
        if not rows:
            continue
        header = rows[0]
        col_idx = {name: idx for idx, name in enumerate(header) if name}

        for row_num, row in enumerate(rows[1:], start=2):
            pitcher = row[col_idx['Pitcher']] if 'Pitcher' in col_idx else None
            if not pitcher:
                continue
            pitch = {}
            for col_name, idx in col_idx.items():
                val = row[idx] if idx < len(row) else None
                if val == '':
                    val = None
                pitch[col_name] = val
            pitch['_source'] = tab_name if is_extra else 'MLB'
            # Sheet coordinates — join key for the per-pitch Stuff+/Loc+
            # write-back (scripts/sheets_write_grades.py). Underscore keys are
            # pipeline-internal (like _source) and never serialized to JSON.
            pitch['_sheet_tab'] = tab_name
            pitch['_sheet_row'] = row_num
            # Fallback: use raw movement if adjusted values not yet backfilled
            if pitch.get('xIndVrtBrk') is None and pitch.get('IndVertBrk') is not None:
                pitch['xIndVrtBrk'] = pitch['IndVertBrk']
            if pitch.get('xHorzBrk') is None and pitch.get('HorzBrk') is not None:
                pitch['xHorzBrk'] = pitch['HorzBrk']
            # Barrel IS stored in the sheet again (official launch_speed_angle,
            # col 48, re-added 2026-06-29) — this branch only fills BLANK cells
            # (pre-supplement rows, AAA gaps) with the EV/LA code_barrel
            # recompute, which undercounts ~5% vs official. `_barrelSource`
            # marks which path produced the flag so recomputed barrels are
            # distinguishable downstream.
            if not pitch.get('Barrel'):
                try:
                    _ev = float(pitch['ExitVelo']) if pitch.get('ExitVelo') not in (None, '') else None
                    _la = float(pitch['LaunchAngle']) if pitch.get('LaunchAngle') not in (None, '') else None
                except (ValueError, TypeError):
                    _ev = _la = None
                pitch['Barrel'] = '6' if is_barrel(_ev, _la) else ''
                pitch['_barrelSource'] = 'recomputed'
            else:
                pitch['_barrelSource'] = 'official'
            pitches.append(pitch)
    return pitches


def read_pitches_from_supabase(teams=None):
    """Read all RS pitches from the Supabase per-team tables (the database
    mirror of the AL/NL 2026 Sheets). Returns the SAME list-of-pitch-dicts shape
    as read_pitches_from_sheet: column-name -> string value, blanks -> None,
    plus a `_source` tag ('MLB' | 'ROC' | 'AAA') and a recomputed Barrel.

    Post-cutover source of truth: pitches are pulled by Pitcher2026 and retagged
    in Postico/Supabase, so the website reads them here instead of the Sheets.
    Reads the same team set the Sheets path did (30 MLB + ROC + AAA; FCL excluded).
    """
    import datetime as _dt
    import psycopg2

    if teams is None:
        teams = sorted(MLB_TEAMS | {'ROC', 'AAA'})

    url = os.environ.get('SUPABASE_DB_URL')
    if not url:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('SUPABASE_DB_URL'):
                        url = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        raise RuntimeError('SUPABASE_DB_URL not set (environment or repo .env)')
    kw = {'connect_timeout': 30}
    if 'sslmode=' not in url:
        kw['sslmode'] = 'require'

    conn = psycopg2.connect(url, **kw)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM all_pitches WHERE "PTeam" = ANY(%s)', (list(teams),))
            colnames = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    print(f"  Supabase all_pitches: {len(rows)} pitches across {len(teams)} teams")

    pitches = []
    for row in rows:
        pitch = {}
        for ci, name in enumerate(colnames):
            v = row[ci]
            if v is None:
                pitch[name] = None
            elif isinstance(v, _dt.date):           # date -> 'YYYY-MM-DD'
                pitch[name] = v.isoformat()
            else:                                   # numeric/int -> str (matches Sheet text)
                pitch[name] = str(v)
        pteam = pitch.get('PTeam')
        pitch['_source'] = pteam if pteam in ('ROC', 'AAA') else 'MLB'
        # identical fallbacks to read_pitches_from_sheet
        if pitch.get('xIndVrtBrk') is None and pitch.get('IndVertBrk') is not None:
            pitch['xIndVrtBrk'] = pitch['IndVertBrk']
        if pitch.get('xHorzBrk') is None and pitch.get('HorzBrk') is not None:
            pitch['xHorzBrk'] = pitch['HorzBrk']
        if not pitch.get('Barrel'):
            try:
                _ev = float(pitch['ExitVelo']) if pitch.get('ExitVelo') not in (None, '') else None
                _la = float(pitch['LaunchAngle']) if pitch.get('LaunchAngle') not in (None, '') else None
            except (ValueError, TypeError):
                _ev = _la = None
            pitch['Barrel'] = '6' if is_barrel(_ev, _la) else ''
        pitches.append(pitch)
    return pitches


# Six 2026 per-division workbooks (replaced the two AL/NL books, on the
# huronalytics account). NLE2026 also holds ROC/AAA/FCL.
DIVISION_WORKBOOK_IDS = {
    'ALE2026': '1YbgAliQzXePiFan-ruwJ50G80l4AjeyTGN8cO3KJ1XI',
    'ALC2026': '14gglESfgJoT90crQb5hHoEZNUFDZ5chPLbUIV9mlm4E',
    'ALW2026': '1eSFfKRo5kSImjP0SZ1SMssGrOhrKSZM9GOHiwntIlhs',
    'NLE2026': '1BypxxlWgQAltETOLqccOYigeo8nXX-FIuVv6rhT4anA',
    'NLC2026': '1-I8BVEw9bR9rzGVYJao_Ar0bjYZF54pi5pm3YEluB9w',
    'NLW2026': '1vm257A676FORcSRzXcNj6txgehGhYI7k5mnmsgQCYH0',
}


def _gspread_client():
    """Authorize gspread to read the division workbooks.

    In CI (the update-leaderboard workflow) the service-account key arrives in
    the GOOGLE_SERVICE_ACCOUNT_JSON env var (GitHub secret SERVICE_ACCOUNT_JSON);
    locally it falls back to the default gspread file (~/.config/gspread/
    service_account.json — the huronalytics account the six books are shared
    with). Full spreadsheets scope (not readonly): the same client serves
    scripts/sheets_write_grades.py in CI, which 403'd on readonly scopes
    (2026-07-19). Actual access is still governed by the books' sharing.
    """
    sa_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if sa_json:
        from google.oauth2.service_account import Credentials
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=scopes)
        return gspread.authorize(creds)
    return gspread.service_account()


def read_all_pitches_from_sheets():
    """Read all RS pitches from the six 2026 division workbooks (the Sheets the
    site reads). NLE2026's ROC/AAA tabs come in via extra_tabs; FCL is skipped
    (not in MLB_TEAMS)."""
    gc = _gspread_client()
    pitches = []
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        extra = {'ROC', 'AAA'} if name == 'NLE2026' else None
        pitches += read_pitches_from_sheet(gc, wid, extra_tabs=extra)
    return pitches


# ── MLB ID lookup ────────────────────────────────────────────────────────

def load_mlb_id_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def save_mlb_id_cache(cache, cache_path):
    # Persist only successful lookups — an mlbId is stable, so caching it
    # forever is correct. None entries (failed lookups, e.g. a callup MLB
    # hadn't listed yet) are dropped on save so the next run re-attempts
    # them instead of freezing the player ID-less permanently. None is
    # still kept in-memory during a run for within-run dedup.
    with open(cache_path, 'w') as f:
        json.dump({k: v for k, v in cache.items() if v is not None},
                  f, indent=2)


# ── Hitter primary position (max games per position, MLB only) ──────────

HITTER_POSITION_CACHE_FILE = os.path.join(DATA_DIR, 'hitter_position_cache.json')


def load_hitter_position_cache():
    if os.path.exists(HITTER_POSITION_CACHE_FILE):
        try:
            with open(HITTER_POSITION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_hitter_position_cache(cache):
    os.makedirs(os.path.dirname(HITTER_POSITION_CACHE_FILE), exist_ok=True)
    with open(HITTER_POSITION_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def _fetch_player_position(mlb_id, season=2026):
    """Determine a player's primary MLB position for the given season by
    games played per position. MLB only (sportId=1); MiLB games excluded.
    A game where the player appeared at multiple positions counts +1 for
    each position (matches MLB's per-position games stat directly).

    Position-player-pitching games ('P', e.g. blowout outings by a position
    player like Dylan Moore) are excluded from the resolution since this
    function is called from the hitter pipeline — the defensive position
    is what matters here, not pitcher appearances.

    Returns: position abbreviation string ('3B', 'LF', 'CF', 'RF', 'DH', etc.)
    or None if no fielding records exist (rare; e.g., not yet called up)."""
    url = (f'https://statsapi.mlb.com/api/v1/people/{mlb_id}/stats'
           f'?stats=season&group=fielding&season={season}&sportId=1')
    try:
        raw = _fetch_with_retry(url, timeout=15)
        data = json.loads(raw)
        stats_arr = data.get('stats') or []
        if not stats_arr:
            return None
        splits = stats_arr[0].get('splits') or []
        # Aggregate games per position (multi-team players have one split per
        # team per position; sum across teams). Pitcher appearances dropped
        # for hitter-side position resolution.
        EXCLUDE_FROM_HITTER = {'P', 'TWP'}
        games_by_pos = {}
        for split in splits:
            pos_abbr = (split.get('position') or {}).get('abbreviation')
            games = (split.get('stat') or {}).get('games') or 0
            if pos_abbr and games and pos_abbr not in EXCLUDE_FROM_HITTER:
                games_by_pos[pos_abbr] = games_by_pos.get(pos_abbr, 0) + games
        if not games_by_pos:
            return None
        # Resolve ties: when a fielding position is tied with DH, prefer the
        # fielding position. Other ties (e.g., LF vs RF) fall through to
        # whichever max() returns first — not strictly determined but rare.
        max_games = max(games_by_pos.values())
        top = [p for p in games_by_pos if games_by_pos[p] == max_games]
        if len(top) > 1 and 'DH' in top:
            non_dh = [p for p in top if p != 'DH']
            if non_dh:
                return non_dh[0]
        return max(games_by_pos, key=games_by_pos.get)
    except Exception:
        return None


def fetch_hitter_positions(hitters, season=2026):
    """For each hitter, return primary MLB position (max games per position)
    using a daily-refresh cache.

    hitters: iterable of (name, mlb_id) tuples. Names are informational only;
    mlb_id is the lookup key.

    Returns: dict mlb_id (int) -> position abbreviation (str) or None.
    """
    cache = load_hitter_position_cache()
    today = _today_et().strftime('%Y-%m-%d')
    cutoff = (_today_et() - timedelta(days=HITTER_POSITION_CACHE_DAYS)).strftime('%Y-%m-%d')
    n_cache_hit = 0

    # Collect the unique players whose cached position is older than the window
    # (or absent); a player faced by many pitchers is only fetched once.
    to_fetch, seen = [], set()
    for _name, mlb_id in hitters:
        if not mlb_id:
            continue
        key = str(mlb_id)
        if key in seen:
            continue
        seen.add(key)
        cached = cache.get(key)
        if cached and cached.get('fetched', '') >= cutoff:
            n_cache_hit += 1
        else:
            to_fetch.append(key)

    # Fetch the stale/new positions concurrently (independent MLB API calls).
    if to_fetch:
        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as ex:
            futures = {ex.submit(_fetch_player_position, int(k), season): k
                       for k in to_fetch}
            for fut in as_completed(futures):
                cache[futures[fut]] = {'position': fut.result(), 'fetched': today}

    save_hitter_position_cache(cache)
    print(f"  Hitter positions: {n_cache_hit} fresh in cache "
          f"(<= {HITTER_POSITION_CACHE_DAYS}d), {len(to_fetch)} fetched")
    return {int(k): (v.get('position') if v else None) for k, v in cache.items()}


_CANONICAL_NAME_MEMO = {}


def fetch_canonical_last_first(mlb_id):
    """Return the MLB Stats API canonical "Last, First" name for a player ID.

    Uses the person record's `lastFirstName` field, which already matches the
    leaderboard's "Last, First" display format (e.g. 695508 -> "Cauley, Cam").
    Result is memoized per process run. Returns None on any lookup failure so
    callers can fall back to a locally-observed spelling.
    """
    if mlb_id in _CANONICAL_NAME_MEMO:
        return _CANONICAL_NAME_MEMO[mlb_id]
    name = None
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{int(mlb_id)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            people = json.loads(resp.read().decode()).get('people', [])
        if people:
            name = people[0].get('lastFirstName') or None
    except Exception as e:
        print(f"  Warning: canonical-name lookup failed for id {mlb_id}: {e}")
    _CANONICAL_NAME_MEMO[mlb_id] = name
    return name


def lookup_mlb_id(player_name, team_abbrev, mlb_id_cache):
    """Look up MLB player ID using the MLB Stats API, matching by name and team."""
    cache_key = f"{player_name}|{team_abbrev}"
    if cache_key in MANUAL_MLB_IDS:
        mlb_id_cache[cache_key] = MANUAL_MLB_IDS[cache_key]
        return MANUAL_MLB_IDS[cache_key]
    if cache_key in mlb_id_cache:
        return mlb_id_cache[cache_key]

    parts = player_name.split(', ')
    if len(parts) == 2:
        search_name = f"{parts[1]} {parts[0]}"
    else:
        search_name = player_name

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={urllib.parse.quote(search_name)}&sportIds=1,11,12,13,14&hydrate=currentTeam&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
        people = data.get('people', [])

        if team_id and people:
            for person in people:
                ct = person.get('currentTeam', {})
                parent = ct.get('parentOrgId') or ct.get('id')
                if parent == team_id or ct.get('id') == team_id:
                    mlb_id = person['id']
                    mlb_id_cache[cache_key] = mlb_id
                    return mlb_id

        if people:
            last_name = parts[0] if len(parts) == 2 else player_name.split()[-1]
            # Sheet names can carry a generational suffix ("Romero Jr.")
            # that the API's lastName field ("Romero") never does.
            last_name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', last_name,
                               flags=re.IGNORECASE)
            for person in people:
                if person.get('lastName', '').lower() == last_name.lower():
                    mlb_id = person['id']
                    mlb_id_cache[cache_key] = mlb_id
                    return mlb_id
            # Do NOT fall back to people[0] — wrong-player matches are worse than no match
            print(f"  Warning: MLB ID lookup found no team/name match for {player_name} ({team_abbrev})")

    except Exception as e:
        print(f"  Warning: MLB ID lookup failed for {player_name} ({team_abbrev}): {e}")

    mlb_id_cache[cache_key] = None
    return None


# ── Boxscore fetching and caching ────────────────────────────────────────

def load_boxscore_cache():
    if os.path.exists(BOXSCORE_CACHE_PATH):
        with open(BOXSCORE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_boxscore_cache(cache):
    with open(BOXSCORE_CACHE_PATH, 'w') as f:
        json.dump(cache, f)


def load_milb_boxscore_cache():
    if os.path.exists(MILB_BOXSCORE_CACHE_PATH):
        with open(MILB_BOXSCORE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_milb_boxscore_cache(cache):
    with open(MILB_BOXSCORE_CACHE_PATH, 'w') as f:
        json.dump(cache, f)


def fetch_milb_game_pks_for_date(date_str, sport_id=11, team_filter=None):
    """Fetch MiLB game PKs for a given date, optionally filtered by team name substring."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId={sport_id}&gameType=R"
    try:
        body = _fetch_with_retry(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        data = json.loads(body)
        game_pks = []
        for date_data in data.get('dates', []):
            for game in date_data.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                if team_filter:
                    away_name = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                    home_name = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                    if team_filter not in away_name and team_filter not in home_name:
                        continue
                game_pks.append(game['gamePk'])
        return game_pks
    except Exception as e:
        print(f"    Error fetching MiLB schedule for {date_str}: {e}")
        return []


def fetch_game_pks_for_date(date_str):
    """Fetch all MLB game PKs for a given date from the schedule API."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1&gameType=R,F,D,L,W"
    try:
        body = _fetch_with_retry(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        data = json.loads(body)
        game_pks = []
        for date_data in data.get('dates', []):
            for game in date_data.get('games', []):
                if game.get('status', {}).get('abstractGameState') == 'Final':
                    game_pks.append(game['gamePk'])
        return game_pks
    except Exception as e:
        print(f"    Error fetching schedule for {date_str}: {e}")
        return []


def fetch_boxscore(game_pk):
    """Fetch boxscore data for a single game. Returns dict with pitcher and hitter stats."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        body = _fetch_with_retry(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        box = json.loads(body)
    except Exception as e:
        print(f"    Error fetching boxscore for game {game_pk}: {e}")
        return None

    result = {'gamePk': game_pk, 'pitchers': [], 'hitters': []}

    for side in ['away', 'home']:
        team_data = box.get('teams', {}).get(side, {})
        team_name = team_data.get('team', {}).get('name', '')
        team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name, team_name)
        pitcher_ids = team_data.get('pitchers', [])
        batter_ids = team_data.get('batters', [])
        players = team_data.get('players', {})

        for idx, pid in enumerate(pitcher_ids):
            p = players.get(f'ID{pid}', {})
            full_name = p.get('person', {}).get('fullName', '')
            stats = p.get('stats', {}).get('pitching', {})
            if not stats:
                continue
            last_first = p.get('person', {}).get('lastFirstName', '')
            if not last_first and full_name:
                last_first = _fullname_to_lastfirst(full_name)
            result['pitchers'].append({
                'name': last_first,
                'mlbId': pid,
                'team': team_abbrev,
                'g': 1,
                'gs': 1 if idx == 0 else 0,
                'outs': stats.get('outs', 0),
                'w': stats.get('wins', 0),
                'l': stats.get('losses', 0),
                'sv': stats.get('saves', 0),
                'hld': stats.get('holds', 0),
                'er': stats.get('earnedRuns', 0),
                'r': stats.get('runs', 0),
                'h': stats.get('hits', 0),
                'hr': stats.get('homeRuns', 0),
                'so': stats.get('strikeOuts', 0),
                'bb': stats.get('baseOnBalls', 0),
                'hbp': stats.get('hitByPitch', 0),
                'ibb': stats.get('intentionalWalks', 0),
                'tbf': stats.get('battersFaced', 0),
                'pitchesThrown': stats.get('pitchesThrown', 0),
                'balls': stats.get('balls', 0),
                'strikes': stats.get('strikes', 0),
                'doubles': stats.get('doubles', 0),
                'triples': stats.get('triples', 0),
                'groundOuts': stats.get('groundOuts', 0),
                'flyOuts': stats.get('flyOuts', 0),
                'popOuts': stats.get('popOuts', 0),
                'lineOuts': stats.get('lineOuts', 0),
                'airOuts': stats.get('airOuts', 0),
                'wp': stats.get('wildPitches', 0),
                'bk': stats.get('balks', 0),
                'ir': stats.get('inheritedRunners', 0),
                'irs': stats.get('inheritedRunnersScored', 0),
            })

        for pid in batter_ids:
            p = players.get(f'ID{pid}', {})
            batting = p.get('stats', {}).get('batting', {})
            if not batting or batting.get('plateAppearances', 0) == 0:
                continue
            full_name = p.get('person', {}).get('fullName', '')
            last_first = p.get('person', {}).get('lastFirstName', '')
            if not last_first and full_name:
                last_first = _fullname_to_lastfirst(full_name)
            result['hitters'].append({
                'name': last_first,
                'mlbId': pid,
                'team': team_abbrev,
                'g': 1,
                'pa': batting.get('plateAppearances', 0),
                'ab': batting.get('atBats', 0),
                'h': batting.get('hits', 0),
                'r': batting.get('runs', 0),
                'doubles': batting.get('doubles', 0),
                'triples': batting.get('triples', 0),
                'hr': batting.get('homeRuns', 0),
                'rbi': batting.get('rbi', 0),
                'tb': batting.get('totalBases', 0),
                'sb': batting.get('stolenBases', 0),
                'cs': batting.get('caughtStealing', 0),
                'bb': batting.get('baseOnBalls', 0),
                'ibb': batting.get('intentionalWalks', 0),
                'hbp': batting.get('hitByPitch', 0),
                'so': batting.get('strikeOuts', 0),
                'sacBunts': batting.get('sacBunts', 0),
                'sacFlies': batting.get('sacFlies', 0),
                'groundOuts': batting.get('groundOuts', 0),
                'flyOuts': batting.get('flyOuts', 0),
                'popOuts': batting.get('popOuts', 0),
                'lineOuts': batting.get('lineOuts', 0),
            })

    return result


def fetch_and_aggregate_milb_boxscores(game_dates, team_abbrev):
    """Fetch MiLB boxscores for a specific AAA team. Returns aggregated pitcher and hitter stats."""
    config = MILB_TEAMS_CONFIG.get(team_abbrev)
    if not config:
        return {}, {}, {}, {}

    cache = load_milb_boxscore_cache()
    new_fetches = 0
    cache_key_prefix = team_abbrev + '|'

    # Refetch only recent dates so late scoring changes are picked up; serve
    # older dates from the committed cache. An older date missing from the cache
    # is still fetched so a gap never silently drops data.
    cutoff = (_today_et() - timedelta(days=BOXSCORE_REFRESH_WINDOW_DAYS)).strftime('%Y-%m-%d')
    dates_to_fetch = sorted(d for d in set(game_dates)
                            if d >= cutoff or (cache_key_prefix + d) not in cache)

    if dates_to_fetch:
        n_cached = len(set(game_dates)) - len(dates_to_fetch)
        print(f"  Refreshing MiLB boxscores for {team_abbrev}: {len(dates_to_fetch)} "
              f"recent/missing date(s) (window {BOXSCORE_REFRESH_WINDOW_DAYS}d), {n_cached} from cache")
        def _milb_pks(d):
            return fetch_milb_game_pks_for_date(d, sport_id=config['sport_id'],
                                                team_filter=config['search_name'])
        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as ex:
            pks_by_date = dict(zip(dates_to_fetch, ex.map(_milb_pks, dates_to_fetch)))
        for d in dates_to_fetch:
            cache[cache_key_prefix + d] = []
        work = [(d, gpk) for d in dates_to_fetch for gpk in pks_by_date.get(d, [])]
        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_boxscore, gpk): d for (d, gpk) in work}
            for fut in as_completed(futures):
                box = fut.result()
                if box:
                    cache[cache_key_prefix + futures[fut]].append(box)
                    new_fetches += 1
        save_milb_boxscore_cache(cache)
        print(f"  Fetched {new_fetches} MiLB boxscores for {team_abbrev}")
    else:
        print(f"  All {len(game_dates)} MiLB game dates for {team_abbrev} already cached")

    pitcher_agg = {}
    hitter_agg = {}
    pitcher_id_map = {}
    hitter_id_map = {}
    seen_game_pks = set()

    accepted_names = set()
    for full_name, abbrev in MILB_TEAM_NAME_TO_ABBREV.items():
        if abbrev == team_abbrev:
            accepted_names.add(full_name)

    for d in game_dates:
        ck = cache_key_prefix + d
        if ck not in cache:
            continue
        for box in cache[ck]:
            # Dedup by gamePk: a suspended/resumed game can be cached under two
            # dates, which would otherwise double-count every player's stats.
            gpk = box.get('gamePk')
            if gpk and gpk in seen_game_pks:
                continue
            if gpk:
                seen_game_pks.add(gpk)
            for p in box.get('pitchers', []):
                p_team = MILB_TEAM_NAME_TO_ABBREV.get(p['team'], p['team'])
                if p_team != team_abbrev:
                    continue
                key = box_key(p['name'], team_abbrev, p.get('mlbId'))
                if key not in pitcher_agg:
                    pitcher_agg[key] = {
                        'g': 0, 'gs': 0, 'outs': 0, 'w': 0, 'l': 0, 'sv': 0, 'hld': 0,
                        'er': 0, 'r': 0, 'h': 0, 'hr': 0, 'so': 0, 'bb': 0, 'hbp': 0, 'ibb': 0,
                        'tbf': 0, 'pitchesThrown': 0, 'balls': 0, 'strikes': 0,
                        'doubles': 0, 'triples': 0,
                        'groundOuts': 0, 'flyOuts': 0, 'popOuts': 0, 'lineOuts': 0, 'airOuts': 0,
                        'wp': 0, 'bk': 0, 'ir': 0, 'irs': 0,
                    }
                a = pitcher_agg[key]
                for k in a:
                    a[k] += p.get(k, 0)
                if p.get('mlbId'):
                    pitcher_id_map[p['mlbId']] = key

            for h in box.get('hitters', []):
                h_team = MILB_TEAM_NAME_TO_ABBREV.get(h['team'], h['team'])
                if h_team != team_abbrev:
                    continue
                key = box_key(h['name'], team_abbrev, h.get('mlbId'))
                if key not in hitter_agg:
                    hitter_agg[key] = {
                        'g': 0, 'pa': 0, 'ab': 0, 'h': 0, 'r': 0,
                        'doubles': 0, 'triples': 0, 'hr': 0, 'rbi': 0,
                        'tb': 0, 'sb': 0, 'cs': 0,
                        'bb': 0, 'ibb': 0, 'hbp': 0, 'so': 0,
                        'sacBunts': 0, 'sacFlies': 0,
                        'groundOuts': 0, 'flyOuts': 0, 'popOuts': 0, 'lineOuts': 0,
                    }
                a = hitter_agg[key]
                for k in a:
                    a[k] += h.get(k, 0)
                if h.get('mlbId'):
                    hitter_id_map[h['mlbId']] = key

    return pitcher_agg, hitter_agg, pitcher_id_map, hitter_id_map


def fetch_and_aggregate_boxscores(game_dates):
    """Fetch boxscores for all game dates, using cache. Returns aggregated pitcher and hitter stats."""
    cache = load_boxscore_cache()
    new_fetches = 0

    # Refetch only recent dates so late scoring changes (hits/errors,
    # earned/unearned) are picked up; serve older dates from the committed
    # cache. An older date missing from the cache is still fetched so a gap
    # never silently drops data.
    cutoff = (_today_et() - timedelta(days=BOXSCORE_REFRESH_WINDOW_DAYS)).strftime('%Y-%m-%d')
    dates_to_fetch = sorted(d for d in game_dates if d >= cutoff or d not in cache)

    if dates_to_fetch:
        n_cached = len(game_dates) - len(dates_to_fetch)
        print(f"  Refreshing {len(dates_to_fetch)} recent/missing boxscore date(s) "
              f"(window {BOXSCORE_REFRESH_WINDOW_DAYS}d), {n_cached} served from cache")
        # Schedules first (one call per date), then all boxscores concurrently.
        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as ex:
            pks_by_date = dict(zip(dates_to_fetch,
                                   ex.map(fetch_game_pks_for_date, dates_to_fetch)))
        for d in dates_to_fetch:
            cache[d] = []
        work = [(d, gpk) for d in dates_to_fetch for gpk in pks_by_date.get(d, [])]
        with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_boxscore, gpk): d for (d, gpk) in work}
            for fut in as_completed(futures):
                box = fut.result()
                if box:
                    cache[futures[fut]].append(box)
                    new_fetches += 1
        save_boxscore_cache(cache)
        print(f"  Fetched {new_fetches} boxscores, cache now has {len(cache)} dates")
    else:
        print(f"  All {len(game_dates)} game dates already cached")

    pitcher_agg = {}
    hitter_agg = {}
    pitcher_id_map = {}
    hitter_id_map = {}
    seen_game_pks = set()

    for d in game_dates:
        if d not in cache:
            continue
        for box in cache[d]:
            gpk = box.get('gamePk')
            if gpk and gpk in seen_game_pks:
                continue
            if gpk:
                seen_game_pks.add(gpk)
            for p in box.get('pitchers', []):
                key = box_key(p['name'], p['team'], p.get('mlbId'))
                if key not in pitcher_agg:
                    pitcher_agg[key] = {
                        'g': 0, 'gs': 0, 'outs': 0, 'w': 0, 'l': 0, 'sv': 0, 'hld': 0,
                        'er': 0, 'r': 0, 'h': 0, 'hr': 0, 'so': 0, 'bb': 0, 'hbp': 0, 'ibb': 0,
                        'tbf': 0, 'pitchesThrown': 0, 'balls': 0, 'strikes': 0,
                        'doubles': 0, 'triples': 0,
                        'groundOuts': 0, 'flyOuts': 0, 'popOuts': 0, 'lineOuts': 0, 'airOuts': 0,
                        'wp': 0, 'bk': 0, 'ir': 0, 'irs': 0,
                    }
                a = pitcher_agg[key]
                for k in a:
                    a[k] += p.get(k, 0)
                if p.get('mlbId'):
                    pitcher_id_map[p['mlbId']] = key

            for h in box.get('hitters', []):
                key = box_key(h['name'], h['team'], h.get('mlbId'))
                if key not in hitter_agg:
                    hitter_agg[key] = {
                        'g': 0, 'pa': 0, 'ab': 0, 'h': 0, 'r': 0,
                        'doubles': 0, 'triples': 0, 'hr': 0, 'rbi': 0,
                        'tb': 0, 'sb': 0, 'cs': 0,
                        'bb': 0, 'ibb': 0, 'hbp': 0, 'so': 0,
                        'sacBunts': 0, 'sacFlies': 0,
                        'groundOuts': 0, 'flyOuts': 0, 'popOuts': 0, 'lineOuts': 0,
                    }
                a = hitter_agg[key]
                for k in a:
                    a[k] += h.get(k, 0)
                if h.get('mlbId'):
                    hitter_id_map[h['mlbId']] = key

    return pitcher_agg, hitter_agg, pitcher_id_map, hitter_id_map
