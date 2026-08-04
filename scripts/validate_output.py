#!/usr/bin/env python3
"""Validate pipeline outputs before publish.

Single source of truth for output integrity — run locally after
process_data.py (+ Stuff+ inject / rebuild_embed) or from CI, where it
gates the commit-and-push step:

    python3 scripts/validate_output.py

Checks the leaderboard JSONs and both embed chunks of the 2026-07-29
split (data_core.json.gz / data_heavy.json.gz), and that the legacy
combined artifacts are gone. Exits 1 with a FAIL list on any problem.

Replaces the root test_pipeline.py (which predated the embed split) and
the inline heredoc that used to live in .github/workflows/update-leaderboard.yml.
"""
import gzip
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
ERRORS = []


def fail(msg):
    ERRORS.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        fail(f"{filename} missing")
        return None
    with open(path) as f:
        return json.load(f)


def check_leaderboard(filename, min_rows, required_keys, name_key):
    data = load_json(filename)
    if data is None:
        return
    if not isinstance(data, list):
        fail(f"{filename} is not a list")
        return
    if len(data) < min_rows:
        fail(f"{filename} has only {len(data)} rows (expected {min_rows}+)")
    else:
        ok(f"{filename}: {len(data)} rows")
    # Every row, not just row 0 — a field dropped partway through the build
    # (e.g. only for one team) should fail here, not on the live site.
    for key in required_keys:
        n_missing = sum(1 for r in data if key not in r)
        if n_missing:
            fail(f"{filename}: {n_missing} rows missing '{key}'")
    n_null = sum(1 for r in data if not r.get(name_key))
    if n_null:
        fail(f"{filename}: {n_null} rows with null {name_key}")


def check_woba_bounds():
    """Regression guard on box_1b = box_h - box_2b - box_3b - box_hr.

    A 2-batter outing can legitimately carry wOBA > 1 (single-event weights
    top out at the HR weight ~2.0), so the hard bound is [0, 2.2]; the
    strict [0, 1] bound applies once the sample (TBF >= 20) makes a higher
    value impossible without a counting bug."""
    data = load_json('pitcher_leaderboard_rs.json') or []
    bad = [r for r in data if r.get('wOBA') is not None
           and not 0 <= r['wOBA'] <= (2.2 if (r.get('tbf') or 0) < 20 else 1.0)]
    if bad:
        r = bad[0]
        fail(f"{len(bad)} pitchers with impossible wOBA "
             f"(e.g. {r.get('pitcher')}: {r['wOBA']}, tbf {r.get('tbf')})")
    else:
        ok("all pitcher wOBA values within sample-feasible bounds")


def check_gz_chunk(name, min_bytes, required_keys, row_probe):
    """Inflate the chunk for real — a truncated or corrupt .gz would break
    the live site for every visitor, so gzip integrity is the whole point."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        fail(f"{name} is missing")
        return
    size = os.path.getsize(path)
    if size < min_bytes:
        fail(f"{name} is only {size:,} bytes (expected {min_bytes:,}+)")
        return
    try:
        obj = json.loads(gzip.decompress(open(path, 'rb').read()))
    except Exception as e:
        fail(f"{name} did not decompress/parse: {e}")
        return
    missing = [k for k in required_keys if k not in obj]
    if missing:
        fail(f"{name} inflated but missing keys: {missing}")
        return
    probe_key, probe_fn, probe_min, probe_desc = row_probe
    n = probe_fn(obj[probe_key])
    if n < probe_min:
        fail(f"{name}: {probe_desc} has only {n} rows")
    else:
        ok(f"{name}: {size:,} bytes, inflates to JSON ({n} {probe_desc})")


def check_core_stays_lean():
    """data_core is the only thing first paint waits on, so guard the two ways
    that quietly regresses: a big table drifting back into it, and homeCounts
    going missing (js/app.js falls back to rendering no count at all, which
    looks fine locally and would tempt someone to re-add hitterData)."""
    name = 'data_core lean'
    try:
        with gzip.open(os.path.join(DATA_DIR, 'data_core.json.gz'), 'rt', encoding='utf-8') as f:
            core = json.load(f)
    except Exception as e:
        fail(f"{name}: could not read data_core ({e})")
        return

    strays = [k for k in ('pitchData', 'hitterData', 'hitterPitchData') if k in core]
    if strays:
        fail(f"{name}: {strays} are back in data_core — they belong in "
             f"data_tables; first paint now parses them again")
    meta = core.get('metadata') or {}
    counts = meta.get('homeCounts') or {}
    if not counts.get('pitchers') or not counts.get('hitters'):
        fail(f"{name}: metadata.homeCounts missing or zero {counts} — the "
             f"home page renders its headline counts from this")
    # Without teamGames every 'Qualified' threshold is zero until data_heavy
    # lands, so the leaderboard shows all ~733 pitchers while the control still
    # reads "Qualified". Silent and wrong, so assert it is present and sane.
    tg = meta.get('teamGames') or {}
    if len(tg) < 30 or min(tg.values(), default=0) < 1:
        fail(f"{name}: metadata.teamGames has {len(tg)} teams — qualification "
             f"thresholds collapse to zero without it (expect ~30+)")
    raw = len(json.dumps(core).encode('utf-8'))
    if raw > 12_000_000:
        fail(f"{name}: inflates to {raw:,} bytes; first paint parses all of "
             f"it, so this should stay near pitcherData + metadata")
    if not strays and counts.get('pitchers') and raw <= 12_000_000:
        ok(f"{name}: {raw:,} bytes inflated, "
           f"{counts['pitchers']}P/{counts['hitters']}H home counts")


def check_pitch_detail_shards():
    """Pitch details ship as one gzipped shard per pitcher, indexed by
    metadata.pitchDetailsIndex in data_core (2026-08-03). A missing shard
    silently degrades a player page to its no-data state, so verify the index
    and the directory agree rather than trusting them to stay in sync."""
    name = 'pitch-detail shards'
    shard_dir = os.path.join(DATA_DIR, 'pitchdetails')
    if not os.path.isdir(shard_dir):
        fail(f"{name}: data/pitchdetails/ missing")
        return
    try:
        with gzip.open(os.path.join(DATA_DIR, 'data_core.json.gz'), 'rt', encoding='utf-8') as f:
            index = json.load(f)['metadata'].get('pitchDetailsIndex', {})
    except Exception as e:
        fail(f"{name}: could not read the index out of data_core ({e})")
        return

    if len(index) < 100:
        fail(f"{name}: index has only {len(index)} pitchers")
        return

    on_disk = {n[:-len('.json.gz')] for n in os.listdir(shard_dir) if n.endswith('.json.gz')}
    missing = [k for k, sid in index.items() if sid not in on_disk]
    orphans = on_disk - set(index.values())
    if missing:
        fail(f"{name}: {len(missing)} indexed pitchers have no shard "
             f"(e.g. {missing[:3]})")
    if orphans:
        fail(f"{name}: {len(orphans)} shard files are not in the index")
    if not missing and not orphans:
        total = sum(os.path.getsize(os.path.join(shard_dir, n))
                    for n in os.listdir(shard_dir))
        ok(f"{name}: {len(index)} shards, {total:,} bytes, index matches disk")


def main():
    print("=== Output validation ===")

    print("Leaderboard JSONs:")
    check_leaderboard('pitch_leaderboard_rs.json', 100,
                      ['pitcher', 'team', 'pitchType', 'velocity', 'count'], 'pitcher')
    check_leaderboard('pitcher_leaderboard_rs.json', 50,
                      ['pitcher', 'team', 'era', 'kPct'], 'pitcher')
    check_leaderboard('hitter_leaderboard_rs.json', 50,
                      ['hitter', 'team', 'pa', 'avg'], 'hitter')
    check_woba_bounds()

    print("Embed chunks:")
    check_gz_chunk('data_core.json.gz', 500_000,
                   ['pitcherData', 'metadata'],
                   ('pitcherData', len, 50, 'pitchers'))
    check_gz_chunk('data_tables.json.gz', 500_000,
                   ['pitchData', 'hitterData', 'hitterPitchData'],
                   ('hitterData', len, 50, 'hitters'))
    check_core_stays_lean()
    check_gz_chunk('data_heavy.json.gz', 5_000_000,
                   ['microData', 'hitterPitchDetails', 'hitterSwingLocations'],
                   ('microData', lambda m: len(m.get('pitchMicro', [])), 1000, 'pitch micro rows'))
    check_pitch_detail_shards()

    print("Legacy artifacts:")
    for legacy in ('data_embedded.js', 'data_embedded.json.gz'):
        if os.path.exists(os.path.join(DATA_DIR, legacy)):
            fail(f"legacy data/{legacy} still present (replaced by the core/heavy split)")
        else:
            ok(f"data/{legacy} absent")

    print("=" * 40)
    if ERRORS:
        print(f"FAILED: {len(ERRORS)} error(s)")
        for e in ERRORS:
            print(f"  - {e}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
