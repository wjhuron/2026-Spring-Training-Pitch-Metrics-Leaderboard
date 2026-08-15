"""Backfill (and re-sync) the AL 2026 / NL 2026 Google Sheets into the Supabase
`pitches` table.

Reads every team tab from both workbooks using the FORMATTED/displayed cell
values — exactly what R's CSV export and pipeline_fetch's get_all_values() read
today — maps each tab to the canonical 47-column schema by header name, and
upserts into Supabase keyed on PitchID.

Re-running is safe and idempotent (ON CONFLICT updates), so this script doubles
as the retag-sync during the parallel-run phase: corrections you make in the
Sheets flow into Supabase the next time it runs.

Usage:
    python scripts/tools/backfill_sheets_to_supabase.py              # full backfill
    python scripts/tools/backfill_sheets_to_supabase.py --team PIT   # one tab (test)
    python scripts/tools/backfill_sheets_to_supabase.py --dry-run    # read + diagnose, no write
"""
import os
import sys
import time
import argparse

import pandas as pd
import gspread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import scrapers.supabase_append as sa
from scrapers.sheets_append import SHEETS_NL, SHEETS_AL

WORKBOOKS = [('NL 2026', SHEETS_NL), ('AL 2026', SHEETS_AL)]


def read_sheet_with_retry(ws, max_retries=5):
    """get_all_values with exponential backoff on transient API errors."""
    delay = 2
    for attempt in range(max_retries):
        try:
            return ws.get_all_values()
        except gspread.exceptions.APIError as e:
            if attempt == max_retries - 1:
                raise
            print(f'    [retry {attempt + 1}] {ws.title}: {e}; waiting {delay}s')
            time.sleep(delay)
            delay *= 2


def tab_to_df(values):
    """Build a DataFrame mapped to the canonical 47 columns by header NAME.
    Handles ragged rows (pads/truncates to header width), duplicate/blank
    header cells, and extra columns (ignored). Returns (df, header) or None."""
    if not values or len(values) < 2:
        return None
    header = values[0]
    width = len(header)
    data = [(row + [''] * width)[:width] for row in values[1:]]
    raw = pd.DataFrame(data, columns=header)
    raw = raw.loc[:, ~pd.Index(raw.columns).duplicated()]  # keep first of any dup header
    out = pd.DataFrame(index=raw.index)
    for col in sa.COLUMNS:
        out[col] = raw[col] if col in raw.columns else None
    return out, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', help='restrict to a single tab (for testing)')
    ap.add_argument('--dry-run', action='store_true',
                    help='read + diagnose but do not write to Supabase')
    ap.add_argument('--chunk', type=int, default=5000, help='upsert batch size')
    args = ap.parse_args()

    gc = gspread.service_account()

    frames = []
    header_issues = []
    for label, wid in WORKBOOKS:
        sh = gc.open_by_key(wid)
        print(f'{label}:')
        for ws in sh.worksheets():
            tab = ws.title
            if args.team and tab != args.team:
                continue
            values = read_sheet_with_retry(ws)
            res = tab_to_df(values)
            if res is None:
                print(f'  {tab:<6} empty, skipping')
                continue
            df, header = res
            missing = [c for c in sa.COLUMNS if c not in header]
            extra = [c for c in header if c and c not in sa.COLUMNS]
            if missing or extra:
                header_issues.append((label, tab, missing, extra))
            print(f'  {tab:<6} {len(df):>7} rows')
            frames.append(df)

    if not frames:
        print('No data read; nothing to do.')
        return

    allp = pd.concat(frames, ignore_index=True)
    print(f'\nTotal rows read: {len(allp):,}')

    blank_mask = allp['PitchID'].apply(sa._is_blank)
    n_blank = int(blank_mask.sum())
    nonblank = allp[~blank_mask]
    n_dup = int(nonblank['PitchID'].duplicated().sum())
    print(f'  PitchID: {n_blank} blank, {n_dup} duplicate (of {len(nonblank):,} non-blank)')
    if header_issues:
        print('  HEADER ISSUES (mapped by name; FYI):')
        for label, tab, missing, extra in header_issues:
            print(f'    {label}/{tab}: missing={missing} extra={extra}')

    if args.dry_run:
        print('\n[dry-run] no write performed.')
        return

    # Keyed upsert needs unique, non-blank PitchIDs.
    before = len(allp)
    allp = allp[~blank_mask].drop_duplicates(subset='PitchID', keep='last')
    dropped = before - len(allp)
    if dropped:
        print(f'  dropped {dropped} blank/duplicate-PitchID rows before upsert')

    conn = sa.get_conn()
    try:
        sa.setup_team_tables(conn)                          # ensure all 33 tables exist
        total = sa.push_csv_to_supabase(allp, conn=conn)    # routes each PTeam -> its table
        sa.create_union_view(conn)                          # (re)build the all_pitches view
        with conn.cursor() as cur:
            cur.execute(f'select count(*) from "{sa.VIEW}"')
            db_count = cur.fetchone()[0]
        print(f'Done. Upserted {total:,} rows across per-team tables; '
              f'{sa.VIEW} now spans {db_count:,} rows.')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
