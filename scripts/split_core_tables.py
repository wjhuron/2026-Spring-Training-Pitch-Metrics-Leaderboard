#!/usr/bin/env python3
"""One-off migration: split pitchData/hitterData/hitterPitchData out of
data_core.json.gz into data_tables.json.gz, and precompute the home-page
counts into metadata.

process_data.write_embedded_js emits this layout natively now, so the next
pipeline run reproduces it. This exists so the committed data matches the new
client immediately; pushing the JS without it would leave the live site with
an empty Arsenal/Hitters/vs-Pitches tab until the next run.

Idempotent. Delete after the first pipeline run under the new layout.

    python3 scripts/split_core_tables.py
"""
import gzip
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

MOVED = ('pitchData', 'hitterData', 'hitterPitchData')


def _read_gz(name):
    with gzip.open(os.path.join(DATA_DIR, name), 'rt', encoding='utf-8') as f:
        return json.load(f)


def _write_gz(name, obj):
    payload = json.dumps(obj, separators=(',', ':')).encode('utf-8')
    with open(os.path.join(DATA_DIR, name), 'wb') as f:
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))
    path = os.path.join(DATA_DIR, name)
    return os.path.getsize(path) / 1048576, len(payload) / 1048576


def _count_distinct_mlb_players(rows, name_key, roc_teams):
    """Mirrors process_data._count_distinct_mlb_players, which in turn mirrors
    the countDistinctMlbPlayers() this replaced in js/app.js."""
    seen = set()
    for i, r in enumerate(rows):
        if r.get('team') in roc_teams:
            continue
        mlb_id = r.get('mlbId')
        seen.add(f'id:{mlb_id}' if mlb_id is not None
                 else 'nm:' + str(r.get(name_key) or i))
    return len(seen)


def main():
    core = _read_gz('data_core.json.gz')
    if not any(k in core for k in MOVED):
        print('data_core already split — nothing to do.')
        return

    tables = {k: core.pop(k, []) for k in MOVED}

    roc = set((core.get('metadata') or {}).get('rocTeams') or [])
    core.setdefault('metadata', {})['homeCounts'] = {
        'pitchers': _count_distinct_mlb_players(core['pitcherData'], 'pitcher', roc),
        'hitters': _count_distinct_mlb_players(tables['hitterData'], 'hitter', roc),
    }
    print('  home counts:', core['metadata']['homeCounts'])

    for name, obj in (('data_core.json.gz', core), ('data_tables.json.gz', tables)):
        gz_mb, raw_mb = _write_gz(name, obj)
        print(f'  {name}: {gz_mb:.2f} MB gz, {raw_mb:.1f} MB raw')


if __name__ == '__main__':
    main()
