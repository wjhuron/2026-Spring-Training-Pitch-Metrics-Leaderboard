"""Validate that the Supabase `pitches` table holds EXACTLY what R reads today.

For a given team it reads the live Sheet tab's formatted/displayed values (what
the CSV export contains) and the stored Supabase rows, aligns them by PitchID,
and compares every cell. Any mismatch is reported with examples.

Usage:
    python scripts/validate_supabase_parity.py PIT
"""
import os
import sys
import datetime
from decimal import Decimal

import gspread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import supabase_append as sa
from sheets_append import SHEETS_NL, SHEETS_AL, NL_TEAMS, AL_TEAMS, ROC_AAA_TEAMS
from backfill_sheets_to_supabase import tab_to_df, read_sheet_with_retry


def workbook_for(team):
    if team in AL_TEAMS:
        return SHEETS_AL
    return SHEETS_NL  # NL + ROC/AAA


def norm_sheet(v):
    if v is None:
        return ''
    s = str(v).strip()
    return s


def norm_db(v):
    if v is None:
        return ''
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()[:10] if isinstance(v, datetime.datetime) else v.isoformat()
    return str(v)


def main():
    if len(sys.argv) < 2:
        print('usage: validate_supabase_parity.py <TEAM>')
        sys.exit(1)
    team = sys.argv[1].strip().upper()

    gc = gspread.service_account()
    ws = gc.open_by_key(workbook_for(team)).worksheet(team)
    sheet_df, _ = tab_to_df(read_sheet_with_retry(ws))
    pid_i = sa.COLUMNS.index('PitchID')

    # Sheet rows keyed by PitchID (formatted strings)
    sheet_by_id = {}
    for _, r in sheet_df.iterrows():
        pid = r['PitchID']
        if pid is None or str(pid).strip() == '':
            continue
        sheet_by_id[str(pid).strip()] = r

    # DB rows keyed by PitchID (typed Python values), read from the team's table
    conn = sa.get_conn()
    db_table = sa.table_for_team(team)
    cols_sql = ', '.join(f'"{c}"' for c in sa.COLUMNS)
    with conn.cursor() as cur:
        cur.execute(f'select {cols_sql} from "{db_table}"')
        db_rows = cur.fetchall()
    conn.close()
    db_by_id = {row[pid_i]: row for row in db_rows}

    print(f'{team}: sheet rows={len(sheet_by_id):,}  db rows={len(db_by_id):,}')

    only_sheet = set(sheet_by_id) - set(db_by_id)
    only_db = set(db_by_id) - set(sheet_by_id)
    common = set(sheet_by_id) & set(db_by_id)
    print(f'  only in sheet: {len(only_sheet)}   only in db: {len(only_db)}   common: {len(common):,}')
    if only_sheet:
        print('   e.g. only-in-sheet PitchIDs:', list(only_sheet)[:5])
    if only_db:
        print('   e.g. only-in-db PitchIDs:', list(only_db)[:5])

    def values_equal(sv, dv):
        """True if the two cells mean the same thing. Numeric cells are compared
        as floats so 1.72==1.720 and -0.0==0.0 (representation-only diffs)."""
        if sv == dv:
            return True
        if sv == '' and dv == '':
            return True
        try:
            return abs(float(sv) - float(dv)) < 1e-9
        except (ValueError, TypeError):
            return False

    real = {c: [] for c in sa.COLUMNS}     # genuinely different values
    repr_only = {c: 0 for c in sa.COLUMNS}  # same number, different text
    for pid in common:
        srow = sheet_by_id[pid]
        drow = db_by_id[pid]
        for ci, col in enumerate(sa.COLUMNS):
            sv = norm_sheet(srow[col])
            dv = norm_db(drow[ci])
            if sv == dv:
                continue
            if values_equal(sv, dv):
                repr_only[col] += 1
            else:
                real[col].append((pid, sv, dv))

    n_real = sum(len(v) for v in real.values())
    n_repr = sum(repr_only.values())
    print(f'\n  REAL value differences: {n_real:,}')
    print(f'  representation-only differences (same number, e.g. 1.72 vs 1.720): {n_repr:,}')
    if n_repr:
        print('    by column: ' +
              ', '.join(f'{c}={n}' for c, n in repr_only.items() if n))
    if n_real == 0:
        print('  ✅ ZERO real differences — every stored value equals what R reads today.')
    else:
        for col, ms in real.items():
            if ms:
                print(f'    REAL {col}: {len(ms)}; e.g. ' +
                      '; '.join(f'{pid} sheet={sv!r} db={dv!r}' for pid, sv, dv in ms[:4]))


if __name__ == '__main__':
    main()
