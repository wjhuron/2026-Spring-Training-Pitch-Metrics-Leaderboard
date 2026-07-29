#!/usr/bin/env python3
"""Smoke tests for leaderboard pipeline outputs.
Run after process_data.py to validate JSON/JS output integrity.
Usage: python3 test_pipeline.py
"""
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ERRORS = []

def check(condition, msg):
    if not condition:
        ERRORS.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK: {msg}")

def test_json_file(filename, min_rows, required_keys):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        ERRORS.append(f"{filename} missing")
        print(f"  FAIL: {filename} missing")
        return
    with open(path) as f:
        data = json.load(f)
    check(isinstance(data, list), f"{filename} is a list")
    check(len(data) >= min_rows, f"{filename} has {len(data)} rows (need {min_rows}+)")
    if data:
        for key in required_keys:
            check(key in data[0], f"{filename}[0] has key '{key}'")

def test_embedded_js():
    path = os.path.join(DATA_DIR, 'data_embedded.js')
    if not os.path.exists(path):
        ERRORS.append("data_embedded.js missing")
        return
    size = os.path.getsize(path)
    check(size > 100_000, f"data_embedded.js is {size:,} bytes (need 100KB+)")
    with open(path) as f:
        content = f.read(200)
    check('RS_DATA' in content or 'PITCHER_DATA' in content, "data_embedded.js contains expected globals")

def test_no_null_names():
    for filename in ['pitcher_leaderboard_rs.json', 'hitter_leaderboard_rs.json']:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        name_key = 'pitcher' if 'pitcher' in filename else 'hitter'
        nulls = [r for r in data if not r.get(name_key)]
        check(len(nulls) == 0, f"{filename}: {len(nulls)} rows with null {name_key}")

def test_no_negative_singles():
    """Catch the box_1b = box_h - box_2b - box_3b - box_hr issue."""
    path = os.path.join(DATA_DIR, 'pitcher_leaderboard_rs.json')
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    for r in data:
        woba = r.get('wOBA')
        if woba is not None:
            check(0 <= woba <= 1.0, f"pitcher {r.get('pitcher')}: wOBA={woba} in [0,1]")
            if woba < 0 or woba > 1.0:
                break  # One failure is enough

if __name__ == '__main__':
    print("=== Pipeline Smoke Tests ===\n")

    print("1. Pitch leaderboard:")
    test_json_file('pitch_leaderboard_rs.json', 100, ['pitcher', 'team', 'pitchType', 'velocity', 'count'])

    print("\n2. Pitcher leaderboard:")
    test_json_file('pitcher_leaderboard_rs.json', 50, ['pitcher', 'team', 'era', 'kPct'])

    print("\n3. Hitter leaderboard:")
    test_json_file('hitter_leaderboard_rs.json', 50, ['hitter', 'team', 'pa', 'avg'])

    print("\n4. Embedded JS:")
    test_embedded_js()

    print("\n5. No null names:")
    test_no_null_names()

    print("\n6. Data bounds:")
    test_no_negative_singles()

    print(f"\n{'=' * 40}")
    if ERRORS:
        print(f"FAILED: {len(ERRORS)} error(s)")
        for e in ERRORS:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
