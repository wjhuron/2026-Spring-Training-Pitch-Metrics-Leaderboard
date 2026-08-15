#!/usr/bin/env python3
"""Golden-output harness for the Huronalytics pipeline.

Runs process_data.main() with the network layer frozen:
  - Sheets rows replayed from golden_input_rs.pkl / golden_input_new.pkl
  - FanGraphs constants pinned to fixed values

Commands:
  prestate            snapshot data/, js/, index.html into prestate/
  run <label>         frozen pipeline run; outputs copied to runs/<label>/;
                      touched files restored from prestate afterwards
  compare <A> <B>     byte-compare two run snapshots; prints differing files

Invoke via: PYTHONHASHSEED=0 python3 golden_run.py <cmd> ...
(hash seed pinned so set-iteration order cannot fake a diff)
"""
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time

REPO = '/Users/wallyhuron/Huronalytics'
HERE = os.path.dirname(os.path.abspath(__file__))
PRESTATE = os.path.join(HERE, 'prestate')
RUNS = os.path.join(HERE, 'runs')
COPY_LIMIT = 100 * 1024 * 1024  # copy files under 100 MB; hash-only above

WATCH = ['data', 'js', 'index.html']  # repo-relative trees the pipeline writes


def _walk_watched():
    for top in WATCH:
        p = os.path.join(REPO, top)
        if os.path.isfile(p):
            yield top
        else:
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    full = os.path.join(root, fn)
                    yield os.path.relpath(full, REPO)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prestate():
    if os.path.isdir(PRESTATE):
        shutil.rmtree(PRESTATE)
    manifest = {}
    n_copied = 0
    for rel in _walk_watched():
        full = os.path.join(REPO, rel)
        st = os.stat(full)
        manifest[rel] = {'size': st.st_size, 'mtime': st.st_mtime}
        if st.st_size < COPY_LIMIT:
            dst = os.path.join(PRESTATE, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(full, dst)
            n_copied += 1
    with open(os.path.join(PRESTATE, '_manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f"prestate: {len(manifest)} files tracked, {n_copied} copied")


INPUT_CACHES = {
    # input-side caches, not outputs: excluded from comparison, and the
    # weather gap-fill is left in place (not restored) so it converges
    'data/boxscore_cache.json', 'data/milb_boxscore_cache.json',
    'data/game_weather_rs.json',
}
NO_RESTORE = {'data/game_weather_rs.json'}


def run(label):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    try:                       # post-reorg layout
        from pipeline import process_data, fetch as pipeline_fetch
        from pipeline.utils import MLB_TEAMS
    except ImportError:        # pre-reorg layout
        import process_data
        import pipeline_fetch
        from pipeline_utils import MLB_TEAMS
    # freeze the boxscore refresh window: serve purely from cache so live
    # in-progress games cannot leak nondeterminism into the run
    pipeline_fetch.BOXSCORE_REFRESH_WINDOW_DAYS = -10**6
    for name in ('MILB_BOXSCORE_REFRESH_WINDOW_DAYS',):
        if hasattr(pipeline_fetch, name):
            setattr(pipeline_fetch, name, -10**6)

    with open(os.path.join(HERE, 'golden_input_rs.pkl'), 'rb') as f:
        rs_rows = pickle.load(f)
    with open(os.path.join(HERE, 'golden_input_new.pkl'), 'rb') as f:
        new_rows = pickle.load(f)

    import copy
    process_data.read_all_pitches_from_sheets = lambda: copy.deepcopy(rs_rows)
    process_data.read_new_tab_pitches = lambda: copy.deepcopy(new_rows)
    process_data.fetch_guts_constants = lambda year=2026: (
        {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.25, '3B': 1.59, 'HR': 2.05},
        3.15,
        {'wOBAScale': 1.25, 'lgWOBA': 0.317, 'lgRPA': 0.119},
    )
    process_data.fetch_park_factors = (
        lambda year=2026: {t: 1.0 for t in sorted(MLB_TEAMS)})

    marker = time.time()
    t0 = time.time()
    process_data.main()
    print(f"pipeline ran in {time.time() - t0:.0f}s")

    out_dir = os.path.join(RUNS, label)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    manifest = {}
    for rel in _walk_watched():
        full = os.path.join(REPO, rel)
        st = os.stat(full)
        if st.st_mtime <= marker:
            continue
        entry = {'size': st.st_size, 'sha': _sha(full)}
        manifest[rel] = entry
        if st.st_size < COPY_LIMIT:
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(full, dst)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, '_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"run '{label}': {len(manifest)} files written by pipeline")

    # restore every touched file from prestate so the next run starts clean
    restored, missing = 0, []
    for rel in manifest:
        if rel in NO_RESTORE:
            continue
        src = os.path.join(PRESTATE, rel)
        dst = os.path.join(REPO, rel)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            restored += 1
        else:
            missing.append(rel)
    print(f"restored {restored} files from prestate")
    if missing:
        print(f"NO PRESTATE COPY (left as run output): {missing}")


NOISE_RES = [
    (re.compile(rb'\?v=[A-Za-z0-9_-]+'), b'?v=X'),
    (re.compile(rb'"generatedAt":\s*"[^"]*"'), b'"generatedAt": "X"'),
]


def _load_bytes(path):
    with open(path, 'rb') as f:
        data = f.read()
    if path.endswith('.gz'):
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    for rx, sub in NOISE_RES:
        data = rx.sub(sub, data)
    return data


def compare(a, b):
    da, db = os.path.join(RUNS, a), os.path.join(RUNS, b)
    ma = json.load(open(os.path.join(da, '_manifest.json')))
    mb = json.load(open(os.path.join(db, '_manifest.json')))
    ma = {k: v for k, v in ma.items() if k not in INPUT_CACHES}
    mb = {k: v for k, v in mb.items() if k not in INPUT_CACHES}
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))
    if only_a:
        print(f"only in {a}: {only_a}")
    if only_b:
        print(f"only in {b}: {only_b}")
    diffs = []
    for rel in sorted(set(ma) & set(mb)):
        fa, fb = os.path.join(da, rel), os.path.join(db, rel)
        if os.path.isfile(fa) and os.path.isfile(fb):
            if _load_bytes(fa) != _load_bytes(fb):
                diffs.append(rel)
        elif ma[rel]['sha'] != mb[rel]['sha']:
            diffs.append(rel + ' (hash-only)')
    if diffs:
        print(f"DIFFERING FILES ({len(diffs)}):")
        for d in diffs:
            print(f"  {d}")
    else:
        print("IDENTICAL (modulo gzip container bytes)")
    return diffs


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prestate':
        prestate()
    elif cmd == 'run':
        run(sys.argv[2])
    elif cmd == 'compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command {cmd}")
