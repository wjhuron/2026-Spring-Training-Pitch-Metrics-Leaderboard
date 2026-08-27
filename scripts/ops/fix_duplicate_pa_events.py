#!/usr/bin/env python3
"""fix_duplicate_pa_events.py — blank the plate-appearance outcome on rows
that carry it but did not end the plate appearance.

`pipeline.process_data` prints a WARNING naming this script when it finds a
group. Exactly one pitch ends a plate appearance, so exactly one row per
at-bat may carry a PA Event.

The defect comes from a live scrape. A game pulled while an at-bat is in
progress stamps the outcome on what is then the last pitch. The re-pull
appends the real final pitch, and `push_team_data`'s PitchID dedupe cannot
tell that the earlier row is now wrong, so both rows keep the outcome.

It costs more than a duplicated plate appearance. The stale row carries
BBType and the batted-ball columns too, so ONE ball in play is counted TWICE:
GB%/FB%/PU% denominators, BABIP, hard-hit rate and average exit velocity all
move. Measured 2026-08-26: three at-bats league-wide, one each in CIN, PIT
and COL, and Kyle Freeland's stale row carried a full 95.0 mph exit velocity.

WHAT IS CLEARED — `pipeline.utils.PA_OUTCOME_COLUMNS` only:

    Event, BBType, ExitVelo, LaunchAngle, Distance, HC_X, HC_Y,
    Barrel, xBA, xSLG, xwOBA

WHAT IS KEPT, and why: the stale row is a REAL PITCH that was really thrown.
Description, Count, Runners, Outs and RunExp are per-pitch facts and are
correct on it, as is every pitch measurement. RunExp especially: the stale
rows read -0.020, -0.029 and 0.000, which are the right per-pitch run values
for a ball at 1-2, a ball at 0-1 and a foul at 3-2. Clearing it would delete
real run value.

Which row is stale: within an at-bat the terminal pitch is the one with the
highest pitch number. Every other row in the group is stale. Before a cell is
touched the script re-reads that sheet row and refuses unless the PitchID in
it matches, so a row-index drift can never blank the wrong row.

Detection reads `data/all_pitches_rs_cache.pkl`, the pickle the last
pipeline run wrote, because a live read of all 33 tabs costs minutes and
burns the Sheets read quota to find a handful of rows. A stale cache cannot
cause a wrong write: the script re-reads each target row and refuses unless
the PitchID matches. Pass --from-sheets to detect against the live workbooks
instead.

Only the tabs that actually hold a stale row are opened.

Usage:
    python3 scripts/ops/fix_duplicate_pa_events.py            # DRY RUN
    python3 scripts/ops/fix_duplicate_pa_events.py --apply    # blank for real
    python3 scripts/ops/fix_duplicate_pa_events.py --from-sheets
"""
import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pipeline.fetch import read_all_pitches_from_sheets
from pipeline.utils import duplicate_pa_events, PA_OUTCOME_COLUMNS, DATA_DIR
from scrapers.sheets_append import (
    _get_client, _workbook_id_for_team, _col_letter, _sheets_retry,
)


def _pitch_no(row):
    """Trailing field of the PitchID: the pitch's number within the at-bat."""
    return int(str(row['PitchID']).split('_')[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='blank the cells; without it nothing is written')
    ap.add_argument('--from-sheets', action='store_true',
                    help='detect against the live workbooks instead of the '
                         'pitch cache (minutes, and heavy on the read quota)')
    a = ap.parse_args()

    if a.from_sheets:
        print("Reading the six division workbooks...")
        rows = read_all_pitches_from_sheets(include_no_pitch=True)
    else:
        cache = os.path.join(DATA_DIR, 'all_pitches_rs_cache.pkl')
        if not os.path.exists(cache):
            raise SystemExit(
                f"{cache} is absent. Run the pipeline once, or pass "
                f"--from-sheets to read the workbooks directly.")
        print(f"Reading {os.path.basename(cache)} "
              f"(pass --from-sheets to read the workbooks instead)...")
        with open(cache, 'rb') as f:
            rows = pickle.load(f)
    groups = duplicate_pa_events(rows)
    if not groups:
        print("No at-bat carries a PA Event on more than one row. Nothing to do.")
        return
    print(f"\n{len(groups)} at-bat(s) with a duplicated plate-appearance outcome\n")

    gc = _get_client()
    _tabs = {}

    def _tab(name):
        """Open a tab once and cache it; only tabs with a stale row are opened."""
        if name not in _tabs:
            wb_id = _workbook_id_for_team(name)
            if wb_id is None:
                _tabs[name] = (None, None)
            else:
                ws = _sheets_retry(lambda: gc.open_by_key(wb_id).worksheet(name),
                                   label=f'{name} open')
                header = _sheets_retry(lambda: ws.row_values(1),
                                       label=f'{name} header')
                _tabs[name] = (ws, header)
        return _tabs[name]

    n_cleared = n_refused = 0
    for key, group in sorted(groups.items()):
        group = sorted(group, key=_pitch_no)
        terminal, stale = group[-1], group[:-1]
        tab = terminal.get('_sheet_tab')
        print(f"game {key[0]} atBat {key[1]}  {tab}  {terminal.get('Pitcher')} "
              f"vs {terminal.get('Batter')}  ({terminal.get('Event')})")
        print(f"  keep  {terminal['PitchID']}  {terminal.get('Description')!r} "
              f"(pitch {_pitch_no(terminal)}, sheet row {terminal.get('_sheet_row')})")

        ws, header = _tab(tab)
        if ws is None:
            print(f"  {tab} is not in the workbook mapping — SKIPPED")
            continue
        if 'PitchID' not in header:
            print(f"  {tab} header has no PitchID column — SKIPPED")
            continue
        pid_col = header.index('PitchID') + 1

        for row in stale:
            r1 = row.get('_sheet_row')
            present = _sheets_retry(
                lambda: ws.cell(r1, pid_col).value, label=f'{tab} row {r1} PitchID')
            if present != row['PitchID']:
                # The cached row index is stale: rows shift when anyone
                # inserts, deletes or re-sorts. PitchID is the real key, so
                # look the row up by it rather than trusting the index or
                # blanking whatever now sits there. One column read.
                print(f"  note: sheet row {r1} now holds {present!r}; "
                      f"locating {row['PitchID']} by PitchID")
                col = _sheets_retry(lambda: ws.col_values(pid_col),
                                    label=f'{tab} PitchID column')
                found = [i for i, v in enumerate(col, start=1)
                         if v == row['PitchID']]
                if len(found) != 1:
                    print(f"  *** REFUSED {row['PitchID']}: found {len(found)} "
                          f"row(s) with that PitchID in {tab}. Expected exactly "
                          f"one. Resolve by hand.")
                    n_refused += 1
                    continue
                r1 = found[0]
                print(f"  found at sheet row {r1}")
            targets = [c for c in PA_OUTCOME_COLUMNS
                       if c in header and row.get(c) not in (None, '')]
            print(f"  clear {row['PitchID']}  {row.get('Description')!r} "
                  f"(pitch {_pitch_no(row)}, sheet row {r1}) -> {targets}")
            if not a.apply:
                continue
            ranges = [f"{_col_letter(header.index(c) + 1)}{r1}" for c in targets]
            _sheets_retry(lambda: ws.batch_clear(ranges),
                          label=f'{tab} row {r1} clear')
            n_cleared += 1

    print(f"\n{n_cleared} row(s) cleared, {n_refused} refused")
    if not a.apply:
        print("DRY RUN — nothing was written. Re-run with --apply.")


if __name__ == '__main__':
    main()
