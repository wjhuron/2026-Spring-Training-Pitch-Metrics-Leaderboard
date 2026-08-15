#!/usr/bin/env python3
"""Verify the 6 division workbooks contain every pitch the original AL/NL 2026
Sheets do, comparing by PitchID per team.

This is the cutover safety gate: before deleting the original "AL 2026" / "NL
2026" workbooks, every PitchID in them must already exist in the corresponding
division-workbook tab. Any PitchID present in an original but missing from a copy
is real data loss waiting to happen.

Read-only. Run:  python3 scripts/verify_sheets_parity.py
With --sync it appends the missing rows to the copies (full row, preserving the
existing cell formatting of the tab).
"""
import os
import sys
import time
import gspread

# Run from anywhere: put the repo root (parent of scripts/) on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.sheets_append import (
    SHEETS_AL, SHEETS_NL,
    AL_TEAMS, NL_TEAMS, ROC_AAA_TEAMS,
    WORKBOOKS, TEAM_DIVISION,
)


def _origin_sheet_id(team):
    return SHEETS_AL if team in AL_TEAMS else SHEETS_NL


def _copy_sheet_id(team):
    div = TEAM_DIVISION.get(team)
    return WORKBOOKS.get(div) if div else None


def _read_tab(ws):
    """Return (header, rows, pitchid_idx, ids_set) for a worksheet (1 API call)."""
    values = ws.get_all_values()
    if not values:
        return [], [], None, set()
    header = values[0]
    try:
        idx = header.index("PitchID")
    except ValueError:
        idx = len(header) - 1  # fall back to last column
    rows = values[1:]
    ids = set()
    for r in rows:
        if idx < len(r):
            v = r[idx].strip()
            if v:
                ids.add(v)
    return header, rows, idx, ids


def main():
    do_sync = "--sync" in sys.argv
    gc = gspread.service_account()

    teams = list(AL_TEAMS) + list(NL_TEAMS) + list(ROC_AAA_TEAMS)
    # de-dup, keep order
    seen = set()
    teams = [t for t in teams if not (t in seen or seen.add(t))]

    # cache opened spreadsheets
    sheet_cache = {}

    def open_sheet(sid):
        if sid not in sheet_cache:
            sheet_cache[sid] = gc.open_by_key(sid)
        return sheet_cache[sid]

    total_missing = 0
    total_extra = 0
    problems = []

    print(f"{'TEAM':5} {'ORIG':>7} {'COPY':>7} {'MISSING':>8} {'EXTRA':>7}  status")
    print("-" * 50)
    for team in teams:
        osid = _origin_sheet_id(team)
        csid = _copy_sheet_id(team)
        if csid is None:
            print(f"{team:5} {'?':>7} {'?':>7} {'?':>8} {'?':>7}  NO DIVISION MAP")
            continue
        try:
            ows = open_sheet(osid).worksheet(team)
        except gspread.WorksheetNotFound:
            print(f"{team:5} {'-':>7} {'-':>7} {'-':>8} {'-':>7}  orig tab missing")
            continue
        try:
            cws = open_sheet(csid).worksheet(team)
        except gspread.WorksheetNotFound:
            print(f"{team:5} {'-':>7} {'-':>7} {'-':>8} {'-':>7}  COPY TAB MISSING")
            problems.append(team)
            continue

        _, o_rows, o_idx, o_ids = _read_tab(ows)
        _, _, _, c_ids = _read_tab(cws)

        missing = o_ids - c_ids   # in original, absent from copy → DATA LOSS RISK
        extra = c_ids - o_ids     # in copy only → newer pulls, fine
        total_missing += len(missing)
        total_extra += len(extra)
        status = "OK" if not missing else f"*** {len(missing)} MISSING ***"
        if missing:
            problems.append(team)
        print(f"{team:5} {len(o_ids):>7} {len(c_ids):>7} {len(missing):>8} {len(extra):>7}  {status}")

        if do_sync and missing:
            # append the full original rows whose PitchID is missing
            add = [r for r in o_rows if o_idx < len(r) and r[o_idx].strip() in missing]
            if add:
                resp = cws.append_rows(add, value_input_option="USER_ENTERED")
                print(f"      synced {len(add)} rows -> {team}")
                # USER_ENTERED parses "2026-07-24" into a date serial and
                # "12:56" into a day fraction; without a number format the
                # cells render as 46227 / 0.045138. append_rows applies no
                # formatting of its own, so mirror what sheets_append does.
                try:
                    from scrapers.sheets_append import apply_block_format
                    rng = resp['updates']['updatedRange'].split('!')[1]
                    start = int(''.join(c for c in rng.split(':')[0] if c.isdigit()))
                    apply_block_format(cws, cws.row_values(1),
                                       start, start + len(add) - 1)
                except Exception as e:
                    print(f"      WARNING: could not format synced rows "
                          f"({type(e).__name__}: {e}); run "
                          f"scripts/ops/fix_unformatted_blocks.py --apply")
        time.sleep(0.4)  # be gentle on the read quota

    print("-" * 50)
    print(f"TOTAL missing (orig not in copy): {total_missing}")
    print(f"TOTAL extra   (copy not in orig): {total_extra}")
    if total_missing == 0:
        print("\n✅ SAFE: every original PitchID exists in the copies. Cutover is non-destructive.")
    else:
        print(f"\n⚠️  {total_missing} pitches missing from copies across: {', '.join(problems)}")
        print("   Run again with --sync to append them, then re-verify.")
    return 0 if total_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
