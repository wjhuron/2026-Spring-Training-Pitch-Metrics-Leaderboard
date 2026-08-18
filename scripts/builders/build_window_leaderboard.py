#!/usr/bin/env python3
"""Build a full leaderboard for ONE date window, using the season code path.

Why this exists
---------------
A hitter card for a date window cannot read the season leaderboard row, and it
cannot honestly draw percentile bubbles either: a percentile needs an all-MLB
pool measured over the SAME window. This script produces that pool.

It does NOT reimplement any of the metric math. It filters the Sheets pitches
by date and hands them to `pipeline.process_data.process_game_type`, the same
function the shipped build calls. Everything downstream of that call is
therefore identical by construction to a season run:

  * the boxscore merge, whose dates are derived from the pitches passed in, so
    PA / AVG / OBP / SLG / wOBA come from the window's own official boxscores
  * every league average, weighted the same way
  * the SD+ and CT+ cell weight tables, rebuilt over the window
  * the BB+ league xwOBAcon anchor
  * the Hitter+ composite, its standardization, and the plus re-anchor
  * every percentile pool

`window_mode=True` suppresses only the three merges that reach OUTSIDE the
pitch set for season-scoped numbers: Savant sprint speed and the two
FanGraphs overrides. See the docstring on process_game_type.

Output
------
data/_window_<slug>_hitter_leaderboard.json  (scratch, underscore-prefixed)
data/_window_<slug>_metadata.json
data/_window_index.json                      (slug -> window bounds + counts)

Nothing here can reach the site. The shipped `_rs` artifacts are never
written, the gzipped embeds are never rebuilt, and the asset version is never
bumped.

Usage
-----
    python3 scripts/builders/build_window_leaderboard.py START END [--cached]

    # any range works; the pool is always measured over exactly that range
    python3 scripts/builders/build_window_leaderboard.py 2026-05-01 2026-06-15
    python3 scripts/builders/build_window_leaderboard.py 2026-07-16 2026-08-17 --cached

Budget about 8 minutes per window: roughly 2m30s to read the six division
workbooks, then the build itself.

--cached reuses data/_window_pitches_cache.pkl, the frozen Sheets read from a
previous run, and skips the 2m30s read. The cache records the date it was
taken. Use it when you are sweeping several ranges in one sitting; drop it
when the Sheets have been appended to since, because Sheets is the source of
truth and the cache is only a copy of it.
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_DIR = os.path.join(_ROOT, 'data')
INDEX_PATH = os.path.join(DATA_DIR, '_window_index.json')
PITCH_CACHE_PATH = os.path.join(DATA_DIR, '_window_pitches_cache.pkl')


def window_slug(start_date, end_date):
    """Filename component for a window. Must match cards/hitter.py."""
    return f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"


def window_paths(start_date, end_date):
    slug = window_slug(start_date, end_date)
    return (slug,
            os.path.join(DATA_DIR, f'_window_{slug}_hitter_leaderboard.json'),
            os.path.join(DATA_DIR, f'_window_{slug}_metadata.json'))


def _write_atomic(path, obj, **dump_kw):
    """Build to a temp path and move. Never leave a partial artifact."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, **dump_kw)
    os.replace(tmp, path)


def main():
    argv = [a for a in sys.argv[1:] if a != '--cached']
    use_cache = '--cached' in sys.argv
    if len(argv) != 2:
        print(__doc__)
        print("ERROR: need exactly two dates: START END (YYYY-MM-DD)")
        sys.exit(2)
    start_date, end_date = argv[0], argv[1]
    from datetime import datetime
    for d in (start_date, end_date):
        try:
            datetime.strptime(d, '%Y-%m-%d')
        except ValueError:
            print(f"ERROR: '{d}' is not YYYY-MM-DD")
            sys.exit(2)
    if end_date < start_date:
        print(f"ERROR: end {end_date} is before start {start_date}")
        sys.exit(2)

    slug, lb_path, md_path = window_paths(start_date, end_date)
    # Guard the shipped artifacts. A window build must never be able to land
    # on an `_rs` name, whatever gets passed in.
    for p in (lb_path, md_path):
        if '_rs.json' in os.path.basename(p):
            print(f"ERROR: refusing to write a shipped artifact name: {p}")
            sys.exit(1)

    import pipeline.process_data as pd_mod
    from pipeline.fetch import read_all_pitches_from_sheets
    from pipeline.process_data import (process_game_type, load_mlb_id_cache,
                                       save_mlb_id_cache, round_floats_inplace,
                                       _load_fg_manual)

    # Same rule the shipped writer applies: underscore keys are pipeline
    # internals and never reach a JSON artifact. It is a nested helper inside
    # write_json_outputs, so the one-liner is repeated here rather than
    # refactor a function the shipped build depends on.
    def strip_internal_keys(rows):
        return [{k: v for k, v in row.items() if not k.startswith('_')}
                for row in rows]
    from pipeline.fetch import fetch_guts_constants, fetch_park_factors

    print(f"=== Window leaderboard: {start_date} -> {end_date} (slug {slug}) ===")

    # Season Guts constants and park factors. These are deliberately SEASON
    # values: FanGraphs publishes no half-season wOBA weights or park factors,
    # and a split is conventionally scored on the season's run environment.
    # Same fail-closed contract as the shipped build — a degraded constant set
    # produces numbers that look normal, so refuse rather than guess.
    manual = _load_fg_manual(2026)
    print("Fetching FanGraphs Guts constants...")
    try:
        pd_mod.WOBA_WEIGHTS, pd_mod.FIP_CONSTANT, pd_mod.GUTS_EXTRA = \
            fetch_guts_constants(2026)
    except Exception as e:
        mg = (manual or {}).get('guts') or {}
        if not mg:
            print(f"ERROR: Guts fetch failed ({e}) and data/fg_manual.json has "
                  f"no 'guts' block. Refusing to build a window on fallback "
                  f"weights. Re-run (the block is intermittent), or fill in "
                  f"fg_manual.json from https://www.fangraphs.com/tools/guts")
            sys.exit(1)
        pd_mod.WOBA_WEIGHTS = {'BB': mg['wBB'], 'HBP': mg['wHBP'], '1B': mg['w1B'],
                               '2B': mg['w2B'], '3B': mg['w3B'], 'HR': mg['wHR']}
        pd_mod.FIP_CONSTANT = mg['cFIP']
        pd_mod.GUTS_EXTRA = {'wOBAScale': mg['wOBAScale'], 'lgWOBA': mg['lgWOBA'],
                             'lgRPA': mg['lgRPA']}
        print(f"  Live fetch failed ({e}) -> using data/fg_manual.json")

    print("Fetching FanGraphs park factors...")
    try:
        pd_mod.PARK_FACTORS = fetch_park_factors(2026)
    except Exception as e:
        mp = (manual or {}).get('parkFactors') or {}
        if len(mp) < 30:
            print(f"ERROR: park factor fetch failed ({e}) and fg_manual.json "
                  f"has {len(mp)}/30 teams. Refusing to build with parks "
                  f"neutral: wRC+ would ship with no park adjustment.")
            sys.exit(1)
        pd_mod.PARK_FACTORS = dict(mp)
        print(f"  Live fetch failed ({e}) -> using data/fg_manual.json")
    if pd_mod.PARK_FACTORS is not None and 0 < len(pd_mod.PARK_FACTORS) < 30:
        print(f"ERROR: park factors returned only {len(pd_mod.PARK_FACTORS)}/30 "
              f"teams. Refusing to build: the rest would default to 1.0.")
        sys.exit(1)

    import pickle
    rs_pitches = None
    if use_cache:
        if os.path.exists(PITCH_CACHE_PATH):
            print("\n=== Reading Regular Season data (cached Sheets read) ===")
            with open(PITCH_CACHE_PATH, 'rb') as f:
                _blob = pickle.load(f)
            rs_pitches = _blob['pitches']
            print(f"  Reused {len(rs_pitches)} RS pitches captured "
                  f"{_blob.get('capturedAt', '?')} from {PITCH_CACHE_PATH}")
            print(f"  Sheets is the source of truth. Re-run without --cached "
                  f"if rows were appended since.")
        else:
            print(f"\n  --cached given but {PITCH_CACHE_PATH} does not exist; "
                  f"reading Sheets and writing the cache for next time.")
    if rs_pitches is None:
        print("\n=== Reading Regular Season data (Sheets) ===")
        rs_pitches = read_all_pitches_from_sheets()
        print(f"  Read {len(rs_pitches)} RS pitches from the 6 division workbooks")

        # Same whitespace normalization the shipped build does, so a stray
        # padded name cannot fork one player into two rows. Done BEFORE the
        # cache write so a cached run gets normalized rows too.
        for _p in rs_pitches:
            for _fld in ('Batter', 'Pitcher'):
                _v = _p.get(_fld)
                if isinstance(_v, str) and _v != _v.strip():
                    _p[_fld] = _v.strip()

        _tmp = PITCH_CACHE_PATH + '.tmp'
        with open(_tmp, 'wb') as f:
            pickle.dump({'capturedAt': datetime.now().isoformat(timespec='seconds'),
                         'pitches': rs_pitches}, f, protocol=4)
        os.replace(_tmp, PITCH_CACHE_PATH)
        print(f"  Cached the read to {PITCH_CACHE_PATH} "
              f"(reuse with --cached)")

    from pipeline.utils import normalize_date
    window = []
    for p in rs_pitches:
        d = normalize_date(p.get('Game Date'))
        if d and start_date <= d <= end_date:
            window.append(p)
    if not window:
        print(f"ERROR: no pitches between {start_date} and {end_date}")
        sys.exit(1)
    _wd = sorted({normalize_date(p.get('Game Date')) for p in window
                  if normalize_date(p.get('Game Date'))})
    print(f"  Window keeps {len(window)} of {len(rs_pitches)} pitches "
          f"across {len(_wd)} game dates ({_wd[0]} to {_wd[-1]})")

    mlb_id_cache_path = os.path.join(DATA_DIR, 'mlb_id_cache.json')
    mlb_id_cache = load_mlb_id_cache(mlb_id_cache_path)

    print("\n" + "=" * 60)
    print(f"=== Processing window {slug} (window_mode=True) ===")
    print("=" * 60)
    result = process_game_type(window, f'WINDOW {slug}', mlb_id_cache,
                               mlb_id_cache_path, window_mode=True)
    save_mlb_id_cache(mlb_id_cache, mlb_id_cache_path)

    hitters = round_floats_inplace(strip_internal_keys(result['hitter_leaderboard']))
    _write_atomic(lb_path, hitters)
    _write_atomic(md_path, round_floats_inplace(result['metadata']), indent=2)

    index = {}
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH) as f:
                index = json.load(f)
        except (OSError, ValueError):
            index = {}
    index[slug] = {
        'startDate': start_date, 'endDate': end_date,
        'gameDates': len(_wd), 'pitches': len(window),
        'hitters': len(hitters),
        'firstGameDate': _wd[0], 'lastGameDate': _wd[-1],
    }
    _write_atomic(INDEX_PATH, index, indent=2)

    _qual = sum(1 for r in hitters if (r.get('pa') or 0) >= 50)
    print(f"\n--- Wrote window artifacts ---")
    print(f"  {lb_path}")
    print(f"  {md_path}")
    print(f"  {len(hitters)} hitter rows, {_qual} with 50+ PA")
    print(f"  Index: {INDEX_PATH}")
    print(f"\nRender a card against this window:")
    print(f"  python3 -m cards.hitter --hitters \"Wood, James\" "
          f"--start {start_date} --end {end_date}")
    print(f"\nSweep another range without re-reading Sheets:")
    print(f"  python3 scripts/builders/build_window_leaderboard.py "
          f"START END --cached")


if __name__ == '__main__':
    main()
