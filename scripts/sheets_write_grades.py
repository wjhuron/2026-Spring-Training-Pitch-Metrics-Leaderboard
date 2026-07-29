#!/usr/bin/env python3
"""Write per-pitch Stuff+ / Loc+ / Pitching+ grades into the Sheets grade columns.

Reads the two grade dumps keyed by sheet position ("tab\trow" -> float grade):
  - data/pitch_stuff_grades.json      (train_stuff_v11.py --dump-pitch-grades)
  - data/pitch_loc_grades_rs.json     (process_data.py -> pipeline_locplus)

and overwrites the Stuff+ / Loc+ / Pitching+ columns (X:Z, positions 24-26 after HAA) in
every migrated tab of the six division workbooks — full-column overwrite each
run, so retags, late-arriving arm angles, and model retrains all self-heal.

Display rule (2026-07-18, per Wally): sheet cells hold NEAREST-INTEGER grades
(99.6 -> 100); all aggregation everywhere uses the full-precision values (the
rounded cells are never read back by anything).

Cells with no grade (EP, unscorable rows, tabs the pipeline doesn't read like
FCL/NEW) are written as blanks. Tabs whose header row isn't the migrated
50-column schema are skipped with a warning (see sheets_append's schema guard).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pipeline_fetch import _gspread_client, DIVISION_WORKBOOK_IDS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
STUFF_DUMP = os.path.join(DATA, 'pitch_stuff_grades.json')
LOC_DUMP = os.path.join(DATA, 'pitch_loc_grades_rs.json')

GRADE_COL_RANGE = 'X{first}:Z{last}'   # cols 24-26 = Stuff+ / Loc+ / Pitching+
HEADER_SLICE = ['Stuff+', 'Loc+', 'Pitching+']   # 1-based cols 24-26


def _load(path, label):
    if not os.path.exists(path):
        print(f"  {label} dump missing ({path}) — its column will be blank")
        return {}
    with open(path) as f:
        d = json.load(f)
    print(f"  {label}: {len(d)} per-pitch grades loaded")
    return d


def _cell(val):
    # nearest-integer display; blank when no grade
    return int(round(val)) if val is not None else ''


def main():
    stuff = _load(STUFF_DUMP, 'Stuff+')
    loc = _load(LOC_DUMP, 'Loc+')
    if not stuff and not loc:
        print('no grade dumps present — nothing to write')
        return

    gc = _gspread_client()
    total_rows = total_stuff = total_loc = 0
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        sh = gc.open_by_key(wid)
        for ws in sh.worksheets():
            header = ws.row_values(1)
            if len(header) < 26 or header[23:26] != HEADER_SLICE:
                print(f"  {name}/{ws.title}: not on migrated schema — skip")
                continue
            n_rows = len(ws.col_values(1))          # data through last used row
            if n_rows < 2:
                continue
            tab = ws.title
            values, ns, nl = [], 0, 0
            for r in range(2, n_rows + 1):
                key = f'{tab}\t{r}'
                sv, lv = stuff.get(key), loc.get(key)
                ns += sv is not None
                nl += lv is not None
                sc, lc = _cell(sv), _cell(lv)
                # Pitching+ cell = blend of the two VISIBLE integer cells
                # (auditable in-sheet: =ROUND(0.7*X+0.3*Y,0))
                pc = int(round(0.7 * sc + 0.3 * lc)) if (sc != '' and lc != '') else ''
                values.append([sc, lc, pc])
            rng = GRADE_COL_RANGE.format(first=2, last=n_rows)
            for attempt in range(4):
                try:
                    ws.update(rng, values, value_input_option='USER_ENTERED')
                    break
                except Exception as e:
                    if '429' in str(e) and attempt < 3:
                        print(f"  [quota] {tab}: waiting 70s ...")
                        time.sleep(70)
                    else:
                        raise
            print(f"  {name}/{tab}: {len(values)} rows written "
                  f"(Stuff+ {ns}, Loc+ {nl})")
            total_rows += len(values); total_stuff += ns; total_loc += nl
            time.sleep(1.2)
        time.sleep(1.0)
    print(f"\nwrote {total_rows} rows across all tabs "
          f"({total_stuff} Stuff+ grades, {total_loc} Loc+ grades)")


if __name__ == '__main__':
    main()
