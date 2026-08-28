#!/usr/bin/env python3
"""Write per-pitch Stuff+ / Loc+ grades into the Sheets grade columns.

Reads the two grade dumps keyed by sheet position ("tab\trow" -> float grade):
  - data/pitch_stuff_grades.json      (train_stuff.py --dump-pitch-grades)
  - data/pitch_loc_grades_rs.json     (process_data.py -> pipeline_locplus)

and overwrites the Stuff+ / Loc+ columns (X:Y, positions 24-25 after HAA;
the Pitching+ column Z retired 2026-08-28 with the season blend) in
every migrated tab of the six division workbooks — full-column overwrite each
run, so retags, late-arriving arm angles, and model retrains all self-heal.

Display rule (2026-07-18, per Wally): sheet cells hold NEAREST-INTEGER grades
(99.6 -> 100); all aggregation everywhere uses the full-precision values (the
rounded cells are never read back by anything).

Cells with no grade (EP, unscorable rows, tabs the pipeline doesn't read like
FCL/NEW) are written as blanks. Tabs whose header row isn't the migrated
50-column schema are skipped with a warning (see sheets_append's schema guard).

If EITHER dump is absent the script writes nothing at all. Because each write
is a full-column overwrite, a partial payload would blank the missing column
rather than leave it alone — see the guard in main().
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
from pipeline.fetch import _gspread_client, DIVISION_WORKBOOK_IDS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, '..', 'data')
STUFF_DUMP = os.path.join(DATA, 'pitch_stuff_grades.json')
LOC_DUMP = os.path.join(DATA, 'pitch_loc_grades_rs.json')

GRADE_COL_RANGE = 'X{first}:Y{last}'   # cols 24-25 = Stuff+ / Loc+
HEADER_SLICE = ['Stuff+', 'Loc+']   # 1-based cols 24-25


def _load(path, label):
    if not os.path.exists(path):
        print(f"  {label} dump missing ({path}) — this run will write nothing")
        return {}
    with open(path) as f:
        d = json.load(f)
    print(f"  {label}: {len(d)} per-pitch grades loaded")
    return d


def _cell(val):
    # nearest-integer display; blank when no grade
    return int(round(val)) if val is not None else ''


def _status(exc):
    """HTTP status from a gspread APIError, or None."""
    resp = getattr(exc, 'response', None)
    code = getattr(resp, 'status_code', None)
    if code:
        return int(code)
    for c in (429, 500, 502, 503, 504):
        if f'[{c}]' in str(exc):
            return c
    return None


def _retry(fn, what, attempts=5):
    """Run a Sheets call, retrying the transient failures.

    429 (quota) waits out the per-minute window; 5xx are Google-side
    hiccups (a 500 on open_by_key killed the 2026-08-03 19:23 run after
    one workbook) and back off exponentially. Anything else raises at
    once: a schema or auth error should not be retried.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            code = _status(e)
            last = attempt == attempts - 1
            if code == 429 and not last:
                print(f"  [quota] {what}: waiting 70s ...")
                time.sleep(70)
            elif code in (500, 502, 503, 504) and not last:
                wait = 5 * 2 ** attempt
                print(f"  [{code}] {what}: retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise


def _write_tab(ws, name, stuff, loc):
    """Overwrite one tab's grade columns; returns (rows, n_stuff, n_loc)
    or None when the tab isn't on the migrated schema / has no data."""
    tab = ws.title
    header = _retry(lambda: ws.row_values(1), f'{name}/{tab} header')
    if len(header) < 25 or header[23:25] != HEADER_SLICE:
        print(f"  {name}/{tab}: not on migrated schema — skip")
        return None
    n_rows = len(_retry(lambda: ws.col_values(1), f'{name}/{tab} rows'))
    if n_rows < 2:                              # data through last used row
        return None
    values, ns, nl = [], 0, 0
    for r in range(2, n_rows + 1):
        key = f'{tab}\t{r}'
        sv, lv = stuff.get(key), loc.get(key)
        ns += sv is not None
        nl += lv is not None
        sc, lc = _cell(sv), _cell(lv)
        values.append([sc, lc])
    rng = GRADE_COL_RANGE.format(first=2, last=n_rows)
    _retry(lambda: ws.update(rng, values, value_input_option='USER_ENTERED'),
           f'{name}/{tab} write')
    return len(values), ns, nl


def main():
    stuff = _load(STUFF_DUMP, 'Stuff+')
    loc = _load(LOC_DUMP, 'Loc+')
    # FAIL CLOSED when ANY dump is missing. Every write below is a
    # full-column overwrite of X:Y, so a run missing one dump does not merely
    # skip that column — it BLANKS it. This is exactly how the 2026-08-12
    # 21:08 run wiped Stuff+ across all six workbooks: a transient failure
    # downloading the model bundle left the stuff dump unwritten, and the old
    # guard only bailed when BOTH dumps were absent.
    # Refusing to write costs nothing — the columns keep their last-good
    # values, and the next healthy run rewrites every cell anyway.
    missing = [n for n, d in (('Stuff+', stuff), ('Loc+', loc)) if not d]
    if missing:
        print(f"refusing to write: {' and '.join(missing)} dump(s) missing — "
              f"a full-column overwrite would blank those columns. "
              f"Columns keep their current values; re-run once the dump exists.")
        return

    gc = _gspread_client()
    total_rows = total_stuff = total_loc = 0
    failed = []
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        # Each workbook and tab is isolated: one that dies mid-run must
        # not cost the others their writes (every write is a full-column
        # overwrite, so a partial pass is always safe to repeat).
        try:
            sh = _retry(lambda: gc.open_by_key(wid), f'{name} open')
            sheets = _retry(sh.worksheets, f'{name} worksheets')
        except Exception as e:
            print(f"  {name}: FAILED to open ({e}) — skipping workbook")
            failed.append(name)
            continue
        for ws in sheets:
            try:
                res = _write_tab(ws, name, stuff, loc)
            except Exception as e:
                print(f"  {name}/{ws.title}: FAILED ({e}) — skipping tab")
                failed.append(f'{name}/{ws.title}')
                continue
            if res is None:
                continue
            rows, ns, nl = res
            print(f"  {name}/{ws.title}: {rows} rows written "
                  f"(Stuff+ {ns}, Loc+ {nl})")
            total_rows += rows; total_stuff += ns; total_loc += nl
            time.sleep(1.2)
        time.sleep(1.0)
    print(f"\nwrote {total_rows} rows across all tabs "
          f"({total_stuff} Stuff+ grades, {total_loc} Loc+ grades)")
    if failed:
        # Non-zero so the run annotates truthfully, but only after every
        # other workbook has had its chance.
        print(f"FAILED after retries: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
