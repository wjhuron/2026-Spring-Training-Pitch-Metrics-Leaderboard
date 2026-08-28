#!/usr/bin/env python3
"""Append specific missing pitches to their team tab, by PitchID.

Deliberately NOT part of the sweep. Every other write in this codebase updates an
existing cell; this one adds a row, which is the only operation that can change the
shape of the sheet. So it takes an explicit list of PitchIDs, refuses anything it
cannot fully verify, and prints the row it will write before writing it.

    python3 scripts/ops/append_missing_pitches.py 822700_066_06 823755_037_08
    python3 scripts/ops/append_missing_pitches.py --apply <ids...>

A new row needs the columns the sweep excludes — Game Date, PTeam, BTeam, Pitch Type
and PitchID all identify the pitch and must be present. Stuff+ and Loc+
are left blank: they are model output written by scripts/ci/sheets_write_grades.py,
not values any feed carries.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import gspread
import requests

from pipeline.fetch import DIVISION_WORKBOOK_IDS, read_sheet_with_retry
from scrapers.backfill_supplement import (_retry_sheets_call, download_statcast,
                                          MLB_TEAMS)
from scrapers.backfill_full import feed_rows, SAVANT_COLS, fetch_game_json
from scrapers.sheet_precision import fmt, as_float, STRING_COLS

# Written by the model layer, never by a feed. Blank on a new row.
GRADE_COLS = {'Stuff+', 'Loc+'}


def main(pitch_ids, apply=False):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                            'Accept': 'application/json, text/csv'})
    gc = gspread.service_account()

    # Full-precision feed row for each pitch, plus the game date from the payload.
    wanted, dates = {}, {}
    for pid in pitch_ids:
        gpk = pid.split('_')[0]
        if gpk not in dates:
            payload = fetch_game_json(gpk, session)
            dates[gpk] = payload['gameData'].get('datetime', {}).get('officialDate')
            wanted.update({k: v for k, v in feed_rows(gpk, session).items()})
        if pid not in wanted:
            raise SystemExit(f"{pid}: the feed for game {gpk} does not contain it. "
                             f"Refusing to invent a row.")

    # Group by the tab that should hold each pitch, which is its PTeam.
    by_tab = {}
    for pid in pitch_ids:
        tab = (wanted[pid].get('_PTeam') or '').upper()
        if tab not in MLB_TEAMS and tab not in ('ROC', 'AAA'):
            raise SystemExit(f"{pid}: PTeam '{tab}' is not a tracked tab. "
                             f"Refusing to route it.")
        by_tab.setdefault(tab, []).append(pid)

    for label, sid in DIVISION_WORKBOOK_IDS.items():
        sh = _retry_sheets_call(lambda: gc.open_by_key(sid), 'workbook open')
        for ws in _retry_sheets_call(sh.worksheets, 'tab list'):
            tab = ws.title.upper()
            if tab not in by_tab:
                continue
            rows = read_sheet_with_retry(ws)
            header = rows[0]
            jp = header.index('PitchID')
            existing = {r[jp] for r in rows[1:] if jp < len(r) and r[jp]}
            last = max((i for i, r in enumerate(rows) if any(c for c in r)),
                       default=0) + 1        # 0-based index of the last used row

            for pid in by_tab[tab]:
                if pid in existing:
                    print(f"  {tab} {pid}: ALREADY PRESENT, skipping")
                    continue
                src = wanted[pid]
                gpk = pid.split('_')[0]
                # Savant supplement for this one game's date.
                d = dates[gpk]
                sav = download_statcast(ws.title, d, d, session) or {}
                parts = pid.split('_')
                skey = (parts[0], str(int(parts[1])), str(int(parts[2])))
                supp = sav.get(skey) or {}

                # The identity columns the sweep deliberately never writes. A new
                # row has to carry them or it is not attributable to a pitch.
                identity = {'Game Date': d or '', 'PitchID': pid,
                            'PTeam': src.get('_PTeam', ''),
                            'BTeam': src.get('_BTeam', ''),
                            'Pitch Type': src.get('_PitchType', '')}
                row = []
                for col in header:
                    if col in identity:
                        row.append(identity[col])
                    elif col in GRADE_COLS:
                        row.append('')
                    elif col in src:
                        row.append(src[col])
                    elif col in SAVANT_COLS and col in supp:
                        # A string column must never go through as_float: Runners
                        # '0' came back as '0.0' because fmt() was handed a float
                        # for a column it treats as text.
                        v = supp[col]
                        row.append(v if col in STRING_COLS else fmt(col, as_float(v)))
                    else:
                        row.append('')

                filled = {h: v for h, v in zip(header, row) if v != ''}
                print(f"\n  {tab} row {last + 1} <- {pid}  ({len(filled)} of "
                      f"{len(header)} columns populated)")
                for h, v in filled.items():
                    print(f"      {h:16} {v}")
                blank = [h for h, v in zip(header, row) if v == '']
                print(f"      blank: {', '.join(blank)}")

                if apply:
                    _retry_sheets_call(
                        lambda: ws.update(f'A{last + 1}', [row],
                                          value_input_option='USER_ENTERED'),
                        'append row')
                    print(f"      WROTE row {last + 1}")
                    last += 1
                else:
                    print(f"      DRY RUN, nothing written")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pitch_ids', nargs='+')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    main(a.pitch_ids, apply=a.apply)
