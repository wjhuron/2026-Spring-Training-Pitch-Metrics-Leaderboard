#!/usr/bin/env python3
"""One-off migration: split pitchDetails out of data_heavy.json.gz into
per-pitcher shards, and add the shard index to data_core's metadata.

process_data.write_embedded_js now emits this layout natively, so the next
pipeline run reproduces it. This script exists so the committed data matches
the new client immediately rather than only after the next run — pushing the
JS without it would leave the live site with no pitch details until then.

Idempotent: safe to re-run. Delete after the first pipeline run under the new
layout, when it has no remaining purpose.

    python3 scripts/shard_pitch_details.py
"""
import gzip
import hashlib
import json
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def _read_gz(name):
    with gzip.open(os.path.join(DATA_DIR, name), 'rt', encoding='utf-8') as f:
        return json.load(f)


def _write_gz(name, obj):
    payload = json.dumps(obj, separators=(',', ':')).encode('utf-8')
    with open(os.path.join(DATA_DIR, name), 'wb') as f:
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))
    return os.path.getsize(os.path.join(DATA_DIR, name)) / 1048576, len(payload) / 1048576


def main():
    heavy = _read_gz('data_heavy.json.gz')
    details = heavy.pop('pitchDetails', None)
    if details is None:
        print('data_heavy has no pitchDetails — already migrated.')
        return

    shard_dir = os.path.join(DATA_DIR, 'pitchdetails')
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir)

    index, raw_total = {}, 0
    for key, pitches in details.items():
        shard_id = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
        payload = json.dumps(pitches, separators=(',', ':')).encode('utf-8')
        with open(os.path.join(shard_dir, shard_id + '.json.gz'), 'wb') as f:
            f.write(gzip.compress(payload, compresslevel=9, mtime=0))
        index[key] = shard_id
        raw_total += len(payload)

    shard_mb = sum(os.path.getsize(os.path.join(shard_dir, n))
                   for n in os.listdir(shard_dir)) / 1048576
    print(f'  {len(index)} shards ({shard_mb:.1f} MB gz, {raw_total / 1048576:.1f} MB raw)')

    core = _read_gz('data_core.json.gz')
    core.setdefault('metadata', {})['pitchDetailsIndex'] = index

    for name, obj in (('data_core.json.gz', core), ('data_heavy.json.gz', heavy)):
        gz_mb, raw_mb = _write_gz(name, obj)
        print(f'  {name}: {gz_mb:.1f} MB gz, {raw_mb:.1f} MB raw')


if __name__ == '__main__':
    main()
