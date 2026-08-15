#!/usr/bin/env python3
"""fg_overrides.py — Scrape canonical FanGraphs values for hitters and
pitchers and cache them locally for the pipeline.

Why this exists
---------------
Our pipeline computes wRC+, FIP, xFIP, and SIERA from sheet data. These
match the public FanGraphs values *approximately* but not exactly because
of small differences in:

- Rounding and intermediate precision
- Park-factor versioning (FG updates park factors mid-season)
- League-average wOBA weights (FG re-anchors as the season progresses)
- For AAA hitters specifically: FG uses AAA-baseline wOBA + IL/PCL park
  factors, while our pipeline applies MLB constants by default

When the card shows wRC+ = 151 and FanGraphs shows wRC+ = 152, that
1-point gap reads as a bug to anyone cross-referencing. Pulling FG's
authoritative numbers and overriding the pipeline's computed values
keeps the card aligned with what readers see on FanGraphs.

Cache structure (data/fg_overrides.json):

    {
        "fetchedAt": "2026-05-14T...",
        "season": 2026,
        "mlbHitters":  { "<mlbId>": {"wRCplus": 152, "pa": 204, "name": "James Wood"} },
        "mlbPitchers": { "<mlbId>": {"fip": 3.42, "xfip": 3.55,
                                       "siera": 3.45, "ip": 50.1, "name": "..."} },
        "aaaHitters":  { "<mlbId>": {"wRCplus": 94, "pa": 156, "name": "Dylan Crews"} }
    }

Usage:
    python3 fg_overrides.py                 # refresh all three groups
    python3 fg_overrides.py --year 2026

From other modules:
    from pipeline.fg_overrides import refresh_if_stale
    cache = refresh_if_stale(max_age_hours=24)
    wood_wrc = cache['mlbHitters'].get('695578', {}).get('wRCplus')
"""

import argparse
import datetime
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'fg_overrides.json')

FG_API = 'https://www.fangraphs.com/api/leaders/major-league/data'
FG_MILB_API = 'https://www.fangraphs.com/api/leaders/minor-league/data'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/605.1.15 (KHTML, like Gecko) '
                  'Version/17.0 Safari/605.1.15',
    'Accept': 'application/json',
    'Referer': 'https://www.fangraphs.com/leaders/major-league',
}


def _http_get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    body = urllib.request.urlopen(req, timeout=timeout).read().decode('utf-8')
    return json.loads(body)


def fetch_mlb_hitters(year=2026):
    """Returns dict keyed by xMLBAMID with wRC+, xwOBA, xBA, xSLG, PA, name.

    Override scope:
    - wRC+, xwOBA, xBA, xSLG — pulled because the pipeline's rounded
      values flip the displayed third decimal (or whole number for wRC+)
      vs FG for a substantial fraction of hitters.
    - wOBA / AVG / OBP / SLG / BABIP / OPS / ISO — NOT pulled. Pipeline
      matches FG to within ±0.0005 (rounding noise) since we already use
      FG's published wOBA linear weights from the Guts page. Override
      would be cosmetically identical.
    - HardHit%, Barrel% — not pulled yet. The deltas (~0.003) hint at
      methodology differences (EV cutoffs / denominators), so an
      override would paper over an underlying mismatch rather than fix
      it. Worth auditing the pipeline definitions before deciding.

    FG field naming quirks:
    - "xAVG" on the API → exposed as `xBA` here to match pipeline naming.
    - `xSLG` and `xwOBA` use FG's names directly (same as pipeline).
    """
    params = (
        f'pos=all&stats=bat&lg=all&qual=0&type=1'
        f'&season={year}&seasonEnd={year}'
        f'&ind=0&pageitems=5000&pagenum=1'
    )
    payload = _http_get_json(f'{FG_API}?{params}')
    rows = payload.get('data', [])
    out = {}
    for r in rows:
        mid = r.get('xMLBAMID')
        wrc = r.get('wRC+')
        if mid is None or wrc is None:
            continue
        xwoba = r.get('xwOBA')
        xba = r.get('xAVG')
        xslg = r.get('xSLG')
        out[str(int(mid))] = {
            'wRCplus': round(float(wrc)),
            'xwOBA':   round(float(xwoba), 3) if xwoba is not None else None,
            'xBA':     round(float(xba),   3) if xba   is not None else None,
            'xSLG':    round(float(xslg),  3) if xslg  is not None else None,
            'pa':      int(r.get('PA') or 0),
            'name':    r.get('PlayerName') or r.get('Name'),
        }
    return out


def fetch_mlb_pitchers(year=2026):
    """Returns dict keyed by xMLBAMID with FIP, xFIP, SIERA, IP, name."""
    params = (
        f'pos=all&stats=pit&lg=all&qual=0&type=1'
        f'&season={year}&seasonEnd={year}'
        f'&ind=0&pageitems=5000&pagenum=1'
    )
    payload = _http_get_json(f'{FG_API}?{params}')
    rows = payload.get('data', [])
    out = {}
    for r in rows:
        mid = r.get('xMLBAMID')
        if mid is None:
            continue
        fip = r.get('FIP')
        xfip = r.get('xFIP')
        siera = r.get('SIERA')
        # Skip rows with no rate stats at all (relievers with 0 IP, etc.)
        if fip is None and xfip is None and siera is None:
            continue
        out[str(int(mid))] = {
            'fip':   round(float(fip), 2) if fip is not None else None,
            'xfip':  round(float(xfip), 2) if xfip is not None else None,
            'siera': round(float(siera), 2) if siera is not None else None,
            'ip':    float(r.get('IP') or 0),
            'name':  r.get('PlayerName') or r.get('Name'),
        }
    return out


def fetch_aaa_hitters(year=2026):
    """Returns dict keyed by xMLBAMID with wRC+, PA, name. Uses the
    minor-league endpoint with level=1 (AAA) and org= empty (all orgs)."""
    params = (
        f'pos=all&level=1&lg=&stats=bat&qual=0&type=1'
        f'&season={year}&seasonEnd={year}'
        f'&org=&ind=0&splitTeam=false'
        f'&pageitems=5000&pagenum=1'
    )
    rows = _http_get_json(f'{FG_MILB_API}?{params}', timeout=30)
    if not isinstance(rows, list):
        raise RuntimeError(f'Unexpected response from FG minor-league API: {type(rows).__name__}')
    out = {}
    for r in rows:
        mid = r.get('xMLBAMID')
        wrc = r.get('wRC+')
        if mid is None or wrc is None:
            continue
        out[str(int(mid))] = {
            'wRCplus': round(float(wrc)),
            'pa': int(r.get('PA') or 0),
            'name': r.get('PlayerName') or r.get('Name'),
        }
    return out


def build_cache(year=2026, verbose=False):
    """Fetch all three groups and shape the cache."""
    if verbose:
        print(f'  FG override: fetching MLB hitters for {year}...')
    mlb_h = fetch_mlb_hitters(year)
    if verbose:
        print(f'  FG override: fetched {len(mlb_h)} MLB hitters')
        print(f'  FG override: fetching MLB pitchers for {year}...')
    mlb_p = fetch_mlb_pitchers(year)
    if verbose:
        print(f'  FG override: fetched {len(mlb_p)} MLB pitchers')
        print(f'  FG override: fetching AAA hitters for {year}...')
    aaa_h = fetch_aaa_hitters(year)
    if verbose:
        print(f'  FG override: fetched {len(aaa_h)} AAA hitters')
    return {
        'fetchedAt': datetime.datetime.now().isoformat(timespec='seconds'),
        'season': year,
        'mlbHitters': mlb_h,
        'mlbPitchers': mlb_p,
        'aaaHitters': aaa_h,
    }


def save_cache(cache, path=CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    return path


def load_cache(path=CACHE_PATH):
    """Return cached dict or None if missing/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_stale(cache, max_age_hours=24):
    """True if cache is missing, malformed, or older than max_age_hours."""
    if not cache or 'fetchedAt' not in cache:
        return True
    # Also stale if any of the three groups is missing (older cache shape)
    for k in ('mlbHitters', 'mlbPitchers', 'aaaHitters'):
        if k not in cache:
            return True
    try:
        fetched = datetime.datetime.fromisoformat(cache['fetchedAt'])
    except (ValueError, TypeError):
        return True
    age = datetime.datetime.now() - fetched
    return age.total_seconds() > max_age_hours * 3600


RELEASE_CACHE_URL = ('https://github.com/wjhuron/Huronalytics/releases/download/'
                     'latest-data/fg_overrides.json.enc')


def _stuff_data_key():
    """STUFF_DATA_KEY from .env, else the environment (same source the
    pitch pickle's release asset uses)."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    try:
        for line in open(env_path):
            if line.startswith('STUFF_DATA_KEY='):
                return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return os.environ.get('STUFF_DATA_KEY')


def fetch_from_release(timeout=30):
    """The cache as CI last published it, from the latest-data release.

    FanGraphs sits behind Cloudflare bot scoring, which challenges some
    client IPs while letting others (GitHub's runners among them) straight
    through — so a local run can 403 on every endpoint at the same moment
    CI fetches all three cleanly. The cache is gitignored, so without this
    a blocked machine has no route to fresh values at all and silently
    serves an ever-older cache. Same pattern the pitch pickle already uses,
    including AES-256 at rest: the repo is public and this is a third
    party's leaderboard data, so it does not ship in the clear.
    """
    import subprocess
    import tempfile
    key = _stuff_data_key()
    if not key:
        raise RuntimeError('STUFF_DATA_KEY missing from .env — cannot '
                           'decrypt the release cache')
    req = urllib.request.Request(RELEASE_CACHE_URL, headers=HEADERS)
    blob = urllib.request.urlopen(req, timeout=timeout).read()
    enc_path = dec_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.enc') as f:
            f.write(blob)
            enc_path = f.name
        dec_path = enc_path + '.dec'
        subprocess.run(['openssl', 'enc', '-d', '-aes-256-cbc', '-pbkdf2',
                        '-pass', 'env:STUFF_DATA_KEY',
                        '-in', enc_path, '-out', dec_path],
                       check=True, env={**os.environ, 'STUFF_DATA_KEY': key})
        with open(dec_path) as f:
            return json.load(f)
    finally:
        for p in (enc_path, dec_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _fetched_at(cache):
    """Cache timestamp as a datetime; datetime.min when absent/unparseable."""
    try:
        return datetime.datetime.fromisoformat((cache or {}).get('fetchedAt') or '')
    except (TypeError, ValueError):
        return datetime.datetime.min


def refresh_if_stale(year=2026, max_age_hours=24, path=CACHE_PATH, verbose=False):
    """Refresh the cache if it's older than max_age_hours. Returns the
    cache dict (refreshed or current). A failed direct fetch falls back to
    the release copy, then to the existing cache, then to an empty-but-valid
    shape."""
    cache = load_cache(path)
    if not is_stale(cache, max_age_hours):
        return cache
    try:
        if verbose:
            print(f'  FG overrides cache stale — refreshing for season {year}')
        cache = build_cache(year=year, verbose=verbose)
        save_cache(cache, path)
        if verbose:
            print(f'  -> wrote cache to {path}')
    except Exception as e:
        if verbose:
            print(f'  WARNING: FG overrides refresh failed ({type(e).__name__}: {e})')
        try:
            rel = fetch_from_release()
        except Exception as e2:
            rel = None
            if verbose:
                print(f'  release fallback also failed ({type(e2).__name__}: {e2})')
        # Only adopt it if it actually beats what we already have.
        if rel and _fetched_at(rel) > _fetched_at(cache):
            save_cache(rel, path)
            cache = rel
            if verbose:
                print(f"  -> using CI-published cache from the release "
                      f"(fetched {rel.get('fetchedAt')})")
        if cache is None:
            return {
                'fetchedAt': '', 'season': year,
                'mlbHitters': {}, 'mlbPitchers': {}, 'aaaHitters': {},
            }
    return cache


def main():
    parser = argparse.ArgumentParser(description='Refresh FG overrides cache')
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--out', default=CACHE_PATH)
    parser.add_argument('--max-age-hours', type=float, default=0,
                        help='skip the refresh when the cache is younger than '
                             'this (default 0 = always refresh). Unlike the '
                             'old direct-fetch-only main(), this goes through '
                             'refresh_if_stale\'s full ladder — direct fetch '
                             '-> release asset -> existing cache — so a '
                             'Cloudflare-blocked morning degrades to the best '
                             'available cache instead of crashing the cron.')
    args = parser.parse_args()

    print(f'Refreshing FanGraphs overrides for {args.year}...')
    cache = refresh_if_stale(year=args.year, max_age_hours=args.max_age_hours,
                             path=args.out, verbose=True)
    print(f'\nCache at {args.out}')
    print(f'  MLB hitters:  {len(cache["mlbHitters"])}')
    print(f'  MLB pitchers: {len(cache["mlbPitchers"])}')
    print(f'  AAA hitters:  {len(cache["aaaHitters"])}')

    # Show a few samples
    print('\nSample MLB hitters:')
    for mid in list(cache['mlbHitters'].keys())[:3]:
        p = cache['mlbHitters'][mid]
        print(f"  mlbId={mid}  {p['name']:25s}  PA={p['pa']:3d}  wRC+={p['wRCplus']}")
    print('\nSample MLB pitchers:')
    for mid in list(cache['mlbPitchers'].keys())[:3]:
        p = cache['mlbPitchers'][mid]
        print(f"  mlbId={mid}  {p['name']:25s}  IP={p['ip']:5.1f}  "
              f"FIP={p['fip']}  xFIP={p['xfip']}  SIERA={p['siera']}")


if __name__ == '__main__':
    main()
