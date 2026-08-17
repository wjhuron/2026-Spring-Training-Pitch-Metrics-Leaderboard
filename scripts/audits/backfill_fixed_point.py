#!/usr/bin/env python3
"""Fixed-point test: apply the accepted changes in memory, re-diff, demand silence.

An oscillation is a cell the sweep proposes, writes, and then proposes again. Five
of the six bugs found on 2026-08-17 were exactly that, and none of them could be
seen by testing a component on its own. This applies the write set to a copy of the
sheet and re-runs the whole diff against it.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import sys, collections, requests, gspread
from pipeline.fetch import DIVISION_WORKBOOK_IDS, read_sheet_with_retry
from scrapers.backfill_supplement import _retry_sheets_call, download_statcast
from scrapers import backfill_full as bf

if len(sys.argv) < 2:
    raise SystemExit('usage: python3 scripts/audits/backfill_fixed_point.py <TAB>\n'
                     'Run it on a tab that has NOT been applied yet. A tab already\n'
                     'written to passes trivially, because there is nothing to write.')
TAB = sys.argv[1].upper()
gc = gspread.service_account()
ws = None
for label, sid in DIVISION_WORKBOOK_IDS.items():
    sh = _retry_sheets_call(lambda: gc.open_by_key(sid), 'open')
    ws = next((w for w in _retry_sheets_call(sh.worksheets, 'tabs')
               if w.title.upper() == TAB), None)
    if ws:
        break
rows = read_sheet_with_retry(ws)
hdr = rows[0]
jp, col_idx = hdr.index('PitchID'), {n: j for j, n in enumerate(hdr) if n}
s = requests.Session(); s.headers.update({'User-Agent': 'Mozilla/5.0'})
feed = {}
for g in sorted({r[jp].split('_')[0] for r in rows[1:] if r[jp] and '_' in r[jp]}):
    feed.update(bf.feed_rows(g, s))
dts = {r[hdr.index('Game Date')] for r in rows[1:] if r[hdr.index('Game Date')]}
sav = download_statcast(TAB, min(dts), max(dts), s) or {}
led = bf.DecisionLedger.load()
KINDS = {'new', 'drift', 'drift_sub', 'drift_small', 'precision',
         'zone_fix', 'zone_fix_nodonor'}

def sweep(grid):
    zone = []
    ch, miss = bf.diff_tab(TAB, grid, hdr, feed, sav, led, zone)
    zc, unex = bf.zone_outlier_changes(zone)
    allc = ch + zc
    return allc, [c for c in allc if bf._wanted(c, None, KINDS)], miss

def apply_to(grid, accepted):
    out = [list(r) for r in grid]
    for c in accepted:
        j = col_idx[c.col]
        while len(out[c.row - 1]) <= j:
            out[c.row - 1].append('')
        out[c.row - 1][j] = c.new
    return out

print(f'\n=== {TAB}: {len(rows)-1} rows ===')
p1, w1, m1 = sweep(rows)
print(f'pass 1: {len(p1)} changes, {len(w1)} accepted for write, {len(m1)} missing')

# one cell must never be written twice with different values in a single pass
dupe = collections.defaultdict(set)
for c in w1:
    dupe[(c.row, c.col)].add(c.new)
conflict = {k: v for k, v in dupe.items() if len(v) > 1}
print(f'cells written twice with different values: {len(conflict)}  (must be 0)')
for k, v in list(conflict.items())[:3]:
    print(f'   {k} -> {v}')

p2, w2, m2 = sweep(apply_to(rows, w1))
print(f'pass 2 after applying: {len(w2)} accepted for write  (must be 0)')
if w2:
    by = collections.Counter(c.col for c in w2)
    print(f'   OSCILLATING columns: {dict(by)}')
    for c in w2[:6]:
        print(f'   {c.col:9} row={c.row} {c.pitch_id} {c.old!r} -> {c.new!r} '
              f'kind={c.kind} rec={c.rec}')
p3, w3, _ = sweep(apply_to(apply_to(rows, w1), w2))
print(f'pass 3: {len(w3)} accepted for write  (must be 0)')
print('\n' + ('FIXED POINT REACHED' if not w2 and not w3 and not conflict
              else 'NOT A FIXED POINT — oscillation present'))
