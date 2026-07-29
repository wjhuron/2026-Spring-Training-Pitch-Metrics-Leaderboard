#!/usr/bin/env python3
"""Unhide column AC on every tab of the six 2026 division workbooks.

AC is the 29th column (0-indexed: 28). Unhiding an already-visible column
is a no-op on the Sheets side, so this is safe to re-run.

Usage: python3 scripts/unhide_column_ac.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gspread
from pipeline_fetch import DIVISION_WORKBOOK_IDS

COL_AC_INDEX = 28  # zero-indexed; AC is the 29th column

def main():
    gc = gspread.service_account()

    total_tabs = 0
    for label, sheet_id in DIVISION_WORKBOOK_IDS.items():
        sh = gc.open_by_key(sheet_id)
        tabs = sh.worksheets()
        print(f"{label}: {sh.title} ({len(tabs)} tabs)")

        requests = [
            {
                'updateDimensionProperties': {
                    'range': {
                        'sheetId': ws.id,
                        'dimension': 'COLUMNS',
                        'startIndex': COL_AC_INDEX,
                        'endIndex': COL_AC_INDEX + 1,
                    },
                    'properties': {'hiddenByUser': False},
                    'fields': 'hiddenByUser',
                }
            }
            for ws in tabs
        ]
        sh.batch_update({'requests': requests})
        tab_names = ', '.join(ws.title for ws in tabs)
        print(f"  unhid AC on: {tab_names}")
        total_tabs += len(tabs)

    print(f"\nDone. Column AC unhidden across {total_tabs} tabs.")


if __name__ == '__main__':
    main()
