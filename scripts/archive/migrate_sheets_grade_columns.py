#!/usr/bin/env python3
"""One-time schema migration: insert Stuff+ / Loc+ columns after HAA.

Inserts two columns at position 24-25 (1-based, right after HAA) in every tab
of the six division workbooks whose header row matches the pre-migration
48-column schema exactly, then writes the two header cells. Tabs already on
the new 50-column schema are skipped (idempotent); tabs with any OTHER header
are reported and left untouched.

Rollback: --rollback deletes columns 24-25 wherever their headers are exactly
Stuff+ / Loc+ (the columns only ever hold pipeline-derived values, so this is
lossless).

Run in a quiet window: no Pitcher2026 / backfill runs in flight. The schema
guard in sheets_append.py turns any missed-tab / stale-code combination into
a loud refusal rather than silent misalignment.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pipeline_fetch import _gspread_client, DIVISION_WORKBOOK_IDS

# Pre-migration schema — Pitcher2026 final_columns as of commit 60f125d.
OLD_HEADER = [
    'Game Date', 'PTeam', 'Pitcher', 'Throws', 'Pitch Type',
    'Velocity', 'Spin Rate', 'RTilt', 'OTilt', 'IndVertBrk', 'HorzBrk',
    'xIndVrtBrk', 'xHorzBrk',
    'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle',
    'PlateZ', 'PlateX', 'SzTop', 'SzBot',
    'VAA', 'HAA',
    'BTeam', 'Batter', 'Bats', 'Count', 'Runners', 'Outs',
    'Description', 'Event',
    'ExitVelo', 'LaunchAngle', 'Distance', 'BBType',
    'HC_X', 'HC_Y', 'xBA', 'xSLG', 'xwOBA', 'RunExp',
    'BatSpeed', 'SwingLength',
    'AttackAngle', 'AttackDirection', 'SwingPathTilt',
    'PitchID', 'Barrel',
]
HAA_IDX0 = OLD_HEADER.index('HAA')            # 22 (0-based)
GRADE_COLS = ['Stuff+', 'Loc+', 'Pitching+']
HEADER_50 = OLD_HEADER[:HAA_IDX0 + 1] + ['Stuff+', 'Loc+'] + OLD_HEADER[HAA_IDX0 + 1:]
NEW_HEADER = OLD_HEADER[:HAA_IDX0 + 1] + GRADE_COLS + OLD_HEADER[HAA_IDX0 + 1:]
INS_START0 = HAA_IDX0 + 1                     # 23: insert before current BTeam
INS_END0 = INS_START0 + 3                     # 26 (three grade columns)


def migrate(dry_run=False):
    gc = _gspread_client()
    done = skipped = untouched = 0
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        sh = gc.open_by_key(wid)
        for ws in sh.worksheets():
            header = ws.row_values(1)
            if header == NEW_HEADER:
                print(f"  {name}/{ws.title}: already migrated — skip")
                skipped += 1
                continue
            if header == HEADER_50:
                # phase-2: 50-col tabs need only the Pitching+ column at 26
                ins_s, ins_e, hdr_rng, hdr_vals = INS_START0 + 2, INS_START0 + 3, 'Z1', [['Pitching+']]
            elif header == OLD_HEADER:
                ins_s, ins_e, hdr_rng, hdr_vals = INS_START0, INS_END0, 'X1:Z1', [GRADE_COLS]
            else:
                print(f"  {name}/{ws.title}: UNRECOGNIZED header "
                      f"({len(header)} cols) — left untouched")
                untouched += 1
                continue
            if dry_run:
                print(f"  {name}/{ws.title}: would migrate ({ws.row_count} rows)")
                done += 1
                continue
            sh.batch_update({'requests': [{
                'insertDimension': {
                    'range': {'sheetId': ws.id, 'dimension': 'COLUMNS',
                              'startIndex': ins_s, 'endIndex': ins_e},
                    'inheritFromBefore': True,
                }}]})
            ws.update(hdr_rng, hdr_vals)
            print(f"  {name}/{ws.title}: migrated ({ws.row_count} rows)")
            done += 1
            time.sleep(1.0)
        time.sleep(1.0)
    print(f"\nmigrated {done}, already-done {skipped}, untouched {untouched}")
    if untouched:
        print("NOTE: untouched tabs will fail sheets_append's schema guard "
              "if they receive appends — inspect them.")


def rollback():
    gc = _gspread_client()
    for name, wid in DIVISION_WORKBOOK_IDS.items():
        sh = gc.open_by_key(wid)
        for ws in sh.worksheets():
            header = ws.row_values(1)
            if len(header) >= INS_END0 and \
                    header[INS_START0:INS_END0] == GRADE_COLS:
                sh.batch_update({'requests': [{
                    'deleteDimension': {
                        'range': {'sheetId': ws.id, 'dimension': 'COLUMNS',
                                  'startIndex': INS_START0, 'endIndex': INS_END0},
                    }}]})
                print(f"  {name}/{ws.title}: rolled back")
                time.sleep(1.0)
            else:
                print(f"  {name}/{ws.title}: no Stuff+/Loc+ at cols 24-25 — skip")
        time.sleep(1.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen, change nothing')
    ap.add_argument('--rollback', action='store_true',
                    help='delete the two grade columns wherever present')
    args = ap.parse_args()
    if args.rollback:
        rollback()
    else:
        migrate(dry_run=args.dry_run)
