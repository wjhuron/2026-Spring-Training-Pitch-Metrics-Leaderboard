#!/usr/bin/env python3
"""write_missing_ibb.py — append no-pitch intentional-walk marker rows to the
division workbooks. Step 2 of the backfill that enumerate_missing_ibb.py
starts.

Since 2017 an automatic intentional walk contains no pitches, so it cannot
appear in pitch-level data on its own. The sheets record each one as a MARKER
row: PitchID ends `_00`, Event is 'Intent Walk', the identity columns are
filled and every pitch measurement is blank. `pipeline.utils.is_no_pitch`
recognises the row and `read_all_pitches_from_sheets` drops it by default, so
the marker completes the source of truth without moving a single per-pitch
denominator. Only PA-level callers opt in with include_no_pitch=True.

The first sweep ran 2026-08-18 and wrote 320 rows. This script is what keeps
the set current: run the enumerator, then run this, and only the games since
the last sweep are appended.

Ten columns carry a value, matching the 320 rows already in the sheets
exactly (verified field by field, 2026-08-26):

    Game Date, PTeam, Pitcher, Throws, BTeam, Batter, Bats, Outs,
    Event, PitchID

Everything else stays blank on purpose. `pipeline.fetch` turns a blank into
None, which is what keeps these rows out of the batted-ball and zone filters.
Runners is blank too: it comes from the Statcast supplement in the normal
path, and no supplement row exists for a pitch that was never thrown.

Rows are routed to the PITCHER's tab, because that is where the scraper puts
every other row for that pitcher.

Safety: the write goes through scrapers.sheets_append.push_team_data, which
refuses to append when the tab header does not match the frame (positional
append, so a mismatch would corrupt every row), and which drops any PitchID
already present. Re-running this is therefore a no-op, not a duplicate.

Usage:
    python3 scripts/audits/enumerate_missing_ibb.py     # refresh the work list
    python3 scripts/ops/write_missing_ibb.py            # DRY RUN, writes nothing
    python3 scripts/ops/write_missing_ibb.py --apply    # append for real
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

import pandas as pd

from scrapers.sheets_append import (
    _get_client, _workbook_id_for_team, push_team_data, _sheets_retry,
)

# CSV column -> sheet column. The sheet name is the key the frame is built on.
FIELD_MAP = {
    'Game Date': 'gameDate',
    'PTeam': 'PTeam',
    'Pitcher': 'Pitcher',
    'Throws': 'Throws',
    'BTeam': 'BTeam',
    'Batter': 'Batter',
    'Bats': 'Bats',
    'Outs': 'Outs',
    'Event': 'Event',
    'PitchID': 'PitchID',
}


def _load(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows
           if not r.get('sheetTab') or not r.get('PitchID')
           or not r.get('Pitcher') or not r.get('Batter')]
    if bad:
        raise SystemExit(
            f"{len(bad)} row(s) in {csv_path} have no tab, PitchID, pitcher or "
            f"batter. Re-run enumerate_missing_ibb.py; a row that cannot be "
            f"routed must never be guessed at. First: {bad[0]}")
    off = [r for r in rows if not str(r['PitchID']).endswith('_00')]
    if off:
        raise SystemExit(
            f"{len(off)} row(s) have a PitchID that does not end '_00', so "
            f"pipeline.utils.is_no_pitch would NOT recognise them and they "
            f"would count as thrown pitches. First: {off[0]['PitchID']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(DATA, '_missing_ibb_fresh.csv'))
    ap.add_argument('--apply', action='store_true',
                    help='append for real; without it nothing is written')
    a = ap.parse_args()

    rows = _load(a.csv)
    by_tab = defaultdict(list)
    for r in rows:
        by_tab[r['sheetTab']].append(r)
    print(f"{len(rows)} marker row(s) in {os.path.basename(a.csv)}, "
          f"{len(by_tab)} tab(s)")

    gc = _get_client()
    total_new = total_present = 0
    for tab in sorted(by_tab):
        wb_id = _workbook_id_for_team(tab)
        if wb_id is None:
            print(f"  {tab}: not in the workbook mapping — SKIPPED")
            continue
        ws = _sheets_retry(lambda: gc.open_by_key(wb_id).worksheet(tab),
                           label=f'{tab} open')
        header = _sheets_retry(lambda: ws.row_values(1),
                               label=f'{tab} header read')
        if not header:
            print(f"  {tab}: empty header row — SKIPPED")
            continue
        missing_cols = [c for c in FIELD_MAP if c not in header]
        if missing_cols:
            print(f"  {tab}: header lacks {missing_cols} — SKIPPED")
            continue

        # Positional append: the frame must carry the tab's columns, in the
        # tab's order. push_team_data re-checks and refuses on any drift.
        existing = set(_sheets_retry(
            lambda: ws.col_values(header.index('PitchID') + 1),
            label=f'{tab} PitchID read')[1:])
        fresh = [r for r in by_tab[tab] if r['PitchID'] not in existing]
        total_present += len(by_tab[tab]) - len(fresh)
        total_new += len(fresh)
        if not fresh:
            print(f"  {tab}: all {len(by_tab[tab])} already present")
            continue

        df = pd.DataFrame([
            {col: (r[FIELD_MAP[col]] if col in FIELD_MAP else None)
             for col in header}
            for r in fresh
        ], columns=header)

        if not a.apply:
            print(f"  {tab}: would append {len(fresh)} row(s) "
                  f"({', '.join(r['PitchID'] for r in fresh[:3])}"
                  f"{'...' if len(fresh) > 3 else ''})")
            continue
        push_team_data(df, tab, gc=gc)
        print(f"  {tab}: appended {len(fresh)} row(s)")

    print(f"\n{total_new} row(s) to append, {total_present} already present")
    if not a.apply:
        print("DRY RUN — nothing was written. Re-run with --apply.")


if __name__ == '__main__':
    main()
