#!/usr/bin/env python3
"""Hold back ONLY the scratch date filters in the R report scripts while
committing everything else (reformat + badge fixes). `revert` rewrites the
start/end_date_filter default lines to their HEAD values (saving the working
versions); `restore` puts the working versions back after the commit. Matches
lines by variable name (not index), so it's safe if other edits shift lines.
"""
import subprocess, json, sys

FILES = ['R-scripts/Daily.R', 'R-scripts/OnePitcher.R', 'R-scripts/Season.R']
VARS = ['start_date_filter', 'end_date_filter']
SAVE = '/tmp/r_scratch_restore.json'


def is_default(line, v):
    s = line.lstrip()
    return s.startswith(v) and '<-' in line and 'cli_args' not in line and not s.startswith('if')


def find_default(lines, v):
    for i, l in enumerate(lines):
        if is_default(l, v):
            return i
    return None


def revert():
    saved = {}
    for f in FILES:
        head = subprocess.run(['git', 'show', f'HEAD:{f}'], capture_output=True, text=True).stdout.splitlines()
        with open(f) as fh:
            lines = fh.readlines()
        saved[f] = {}
        for v in VARS:
            wi = find_default(lines, v)
            hi = find_default(head, v)
            if wi is None or hi is None:
                continue
            if lines[wi].rstrip('\n') != head[hi]:
                saved[f][v] = lines[wi].rstrip('\n')
                lines[wi] = head[hi] + '\n'
        with open(f, 'w') as fh:
            fh.writelines(lines)
        print(f, 'held back:', list(saved[f].keys()))
    json.dump(saved, open(SAVE, 'w'))


def restore():
    saved = json.load(open(SAVE))
    for f, m in saved.items():
        with open(f) as fh:
            lines = fh.readlines()
        for v, content in m.items():
            wi = find_default(lines, v)
            if wi is not None:
                lines[wi] = content + '\n'
        with open(f, 'w') as fh:
            fh.writelines(lines)
        print(f, 'restored:', list(m.keys()))


{'revert': revert, 'restore': restore}[sys.argv[1]]()
