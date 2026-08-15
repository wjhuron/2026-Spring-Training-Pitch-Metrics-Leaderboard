"""Append Pitcher2026 download output into Supabase: one table per team
(`PIT`, `NYY`, `AAA`, `ROC`, `FCL`, ...), mirroring the per-team tabs of the
AL 2026 / NL 2026 Google Sheets. A cross-team UNION view (`all_pitches`) spans
all of them for queries that need every pitch at once.

This is the Supabase counterpart to sheets_append.py. It is used two ways:
  1. Pitcher2026.py calls push_csv_to_supabase(combined_df) to append new
     pitches (dual-write alongside the Sheets push during the parallel-run
     migration phase).
  2. scripts/tools/backfill_sheets_to_supabase.py imports the schema + upsert helpers
     to load the full Sheet history and to sync retag corrections.

Design goals:
  - The stored value of every column matches EXACTLY what the Sheets currently
    DISPLAY (and therefore what R's CSV export and pipeline_fetch's
    get_all_values() read today). Numeric columns are rounded to the same
    decimal places Pitcher2026.py already applies before pushing to Sheets
    (see Pitcher2026.download_game_data: numeric_round_1/2/3 + numeric_int).
  - Column names are the EXACT Sheet headers ("Game Date", "PlateX", "Count",
    ...) stored as quoted Postgres identifiers, so neither R nor Python needs
    any column renaming.
  - PitchID (gamePk_atbat_pitch) is the natural unique key. Upserts on PitchID
    make re-running a game idempotent (no duplicate rows) — a strict
    improvement over the Sheets' blind-append.

Connection: reads SUPABASE_DB_URL from the environment, or from a `.env` file
in the repo root (gitignored). Use the Supabase "Session pooler" URI.
"""

import os
import re
import math
import datetime
from decimal import Decimal

import pandas as pd

# Team -> league/workbook routing. Reuse the single source of truth in
# sheets_append so the two writers can never drift. (gspread is imported
# lazily inside sheets_append, so this import stays cheap.)
from scrapers.sheets_append import NL_TEAMS, AL_TEAMS, ROC_AAA_TEAMS

VIEW = 'all_pitches'   # cross-team UNION ALL of the per-team tables

# One table per Sheets tab: 30 MLB teams + ROC + AAA + FCL = 33.
ALL_TEAMS = sorted(NL_TEAMS | AL_TEAMS | {'ROC', 'AAA', 'FCL'})

# ---------------------------------------------------------------------------
# Canonical schema: the 47 columns Pitcher2026 writes, in order, each with its
# Postgres type and a "kind" that drives type coercion + rounding on write.
#   kind: 'text' | 'date' | 'int' | 'num1' | 'num2' | 'num3'
#   num1/num2/num3 -> rounded to 1 / 2 / 3 decimals (mirrors Pitcher2026's
#   numeric_round_1 / numeric_round_2 / numeric_round_3).
# ---------------------------------------------------------------------------
COLUMN_SPEC = [
    ('Game Date',        'date',    'date'),
    ('PTeam',            'text',    'text'),
    ('Pitcher',          'text',    'text'),
    ('Throws',           'text',    'text'),
    ('Pitch Type',       'text',    'text'),
    ('Velocity',         'numeric', 'num1'),
    ('Spin Rate',        'integer', 'int'),
    ('RTilt',            'text',    'text'),   # clock "h:mm" string, e.g. "1:31"
    ('OTilt',            'text',    'text'),   # clock "h:mm" string, e.g. "1:24"
    ('IndVertBrk',       'numeric', 'num1'),
    ('HorzBrk',          'numeric', 'num1'),
    ('xIndVrtBrk',       'numeric', 'num1'),
    ('xHorzBrk',         'numeric', 'num1'),
    ('RelPosZ',          'numeric', 'num2'),
    ('RelPosX',          'numeric', 'num2'),
    ('Extension',        'numeric', 'num2'),
    ('ArmAngle',         'numeric', 'num1'),
    ('PlateZ',           'numeric', 'num3'),
    ('PlateX',           'numeric', 'num3'),
    ('SzTop',            'numeric', 'num3'),
    ('SzBot',            'numeric', 'num3'),
    ('VAA',              'numeric', 'num2'),
    ('HAA',              'numeric', 'num2'),
    ('BTeam',            'text',    'text'),
    ('Batter',           'text',    'text'),
    ('Bats',             'text',    'text'),
    ('Count',            'text',    'text'),   # "1-0" — NOT a date
    ('Runners',          'text',    'text'),   # "0", "1+3", "1+2+3"
    ('Outs',             'integer', 'int'),
    ('Description',      'text',    'text'),
    ('Event',            'text',    'text'),
    ('ExitVelo',         'numeric', 'num1'),
    ('LaunchAngle',      'integer', 'int'),
    ('Distance',         'integer', 'int'),
    ('BBType',           'text',    'text'),
    ('HC_X',             'numeric', 'num2'),
    ('HC_Y',             'numeric', 'num2'),
    ('xBA',              'numeric', 'num3'),
    ('xSLG',             'numeric', 'num3'),
    ('xwOBA',            'numeric', 'num3'),
    ('RunExp',           'numeric', 'num3'),
    ('BatSpeed',         'numeric', 'num1'),
    ('SwingLength',      'numeric', 'num1'),
    ('AttackAngle',      'numeric', 'num1'),
    ('AttackDirection',  'numeric', 'num1'),
    ('SwingPathTilt',    'numeric', 'num1'),
    ('PitchID',          'text',    'text'),   # natural unique key
]

COLUMNS = [name for name, _, _ in COLUMN_SPEC]
_KIND = {name: kind for name, _, kind in COLUMN_SPEC}
_DECIMALS = {'num1': 1, 'num2': 2, 'num3': 3}

# Plate-coordinate columns. MLB tabs display these at 3 decimals (their stored
# precision); the AAA/ROC tabs display them at 2. To keep the DB value equal to
# what R reads today, these 4 are rounded per-row: 2 decimals for AAA/ROC, 3 for
# MLB. (Cosmetic only — R reads them as doubles either way.)
_PLATE_COORDS = {'PlateZ', 'PlateX', 'SzTop', 'SzBot'}

# Numeric columns use a FIXED scale so DB clients (Postico, etc.) display the
# right number of decimals (78.0 not 78, -9.0 not -9, 5.10 not 5.1). Each maps
# to its Sheet precision: num1 -> 1dp, num2 -> 2dp, num3 -> 3dp.
_FIXED_NUM_TYPE = {'num1': 'numeric(8,1)', 'num2': 'numeric(8,2)', 'num3': 'numeric(8,3)'}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def _load_db_url():
    """Return SUPABASE_DB_URL from the environment, falling back to a `.env`
    file in this module's directory. Returns None if not found."""
    url = os.environ.get('SUPABASE_DB_URL')
    if url:
        return url.strip()
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('SUPABASE_DB_URL'):
                    val = line.split('=', 1)[1].strip()
                    # strip optional surrounding quotes
                    if len(val) >= 2 and val[0] in '"\'' and val[-1] == val[0]:
                        val = val[1:-1]
                    return val
    return None


def get_conn(connect_timeout=15):
    """Return a psycopg2 connection to the Supabase Postgres, TLS enforced.

    connect_timeout bounds the connect attempt so a wrong host / blocked IPv6
    route fails fast instead of hanging.
    """
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError(
            'psycopg2 is required. Install with: pip install psycopg2-binary'
        )
    url = _load_db_url()
    if not url:
        raise RuntimeError(
            'SUPABASE_DB_URL not set. Add it to the environment or to a `.env` '
            'file in the repo root (see .env.example).'
        )
    kwargs = {'connect_timeout': connect_timeout}
    # Supabase requires TLS; make it explicit unless the URL already says so.
    if 'sslmode=' not in url:
        kwargs['sslmode'] = 'require'
    conn = psycopg2.connect(url, **kwargs)
    conn.autocommit = False
    return conn


def table_for_team(team):
    """Per-team table name = the (sanitized) uppercase team code: 'PIT', 'NYY',
    'AAA', 'ROC', 'FCL', ... — one table per Sheets tab."""
    t = ('' if team is None else str(team)).strip().upper()
    return re.sub(r'[^A-Z0-9_]', '_', t)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def create_table_sql(table):
    """One table per team: the 47 Sheet columns at their Sheet decimal scale
    (numeric(8,N) so clients display the right precision), PitchID primary key."""
    lines = [f'  "{name}" {_FIXED_NUM_TYPE.get(kind, pgtype)}'
             for name, pgtype, kind in COLUMN_SPEC]
    body = ',\n'.join(lines)
    return (f'CREATE TABLE IF NOT EXISTS "{table}" (\n{body},\n'
            f'  PRIMARY KEY ("PitchID")\n)')


def ensure_team_table(conn, table):
    """Create one team's table (PitchID primary key for idempotent upserts, plus
    read indexes) if absent. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(create_table_sql(table))
        cur.execute(f'CREATE INDEX IF NOT EXISTS "{table}_pitcher_idx" '
                    f'ON "{table}" ("Pitcher")')
        cur.execute(f'CREATE INDEX IF NOT EXISTS "{table}_gamedate_idx" '
                    f'ON "{table}" ("Game Date")')
    conn.commit()


def setup_team_tables(conn, teams=None):
    """Create all per-team tables (empty if new). Idempotent."""
    for t in (teams or ALL_TEAMS):
        ensure_team_table(conn, t)


def _existing_tables(conn, teams):
    """Subset of `teams` whose base table currently exists in public schema."""
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
        present = {r[0] for r in cur.fetchall()}
    return [t for t in teams if t in present]


def create_union_view(conn, view=VIEW, teams=None):
    """(Re)create the cross-team UNION view over whichever per-team tables exist,
    so cross-team queries / the website pipeline can read every pitch in one
    statement: SELECT ... FROM all_pitches WHERE ..."""
    teams = _existing_tables(conn, teams or ALL_TEAMS)
    if not teams:
        return
    collist = ', '.join(f'"{c}"' for c in COLUMNS)
    selects = '\nUNION ALL\n'.join(f'  SELECT {collist} FROM "{t}"' for t in teams)
    with conn.cursor() as cur:
        cur.execute(f'CREATE OR REPLACE VIEW "{view}" AS\n{selects}')
    conn.commit()


# ---------------------------------------------------------------------------
# Row preparation: DataFrame (Sheet-header columns) -> typed Python tuples
# ---------------------------------------------------------------------------
def _is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == '':
        return True
    return False


def prepare_rows(df):
    """Coerce a Pitcher2026-style DataFrame to typed rows matching the table.

    Returns (columns, rows) where columns is the 47 data columns and rows is a
    list of tuples ready for execute_values. Every numeric column is rounded to
    its canonical precision (mirroring the Sheet display), blanks become None,
    dates become datetime.date.
    """
    df = df.copy()
    for name in COLUMNS:
        if name not in df.columns:
            df[name] = None

    out_cols = COLUMNS
    series = {}
    # Per-row AAA/ROC flag, for the plate-coordinate 2-vs-3 decimal rule.
    is_aaa = [(('' if t is None else str(t)).strip().upper() in ROC_AAA_TEAMS)
              for t in df['PTeam']]

    for name in COLUMNS:
        kind = _KIND[name]
        col = df[name]
        if kind == 'text':
            series[name] = [None if _is_blank(v) else str(v).strip() for v in col]
        elif kind == 'date':
            parsed = pd.to_datetime(col, errors='coerce')
            series[name] = [None if pd.isna(v) else v.date() for v in parsed]
        elif kind == 'int':
            num = pd.to_numeric(col, errors='coerce')
            series[name] = [None if pd.isna(v) else int(round(float(v))) for v in num]
        elif name in _PLATE_COORDS:
            # AAA/ROC display 2 decimals, MLB 3 — match each row's source.
            num = pd.to_numeric(col, errors='coerce')
            series[name] = [None if pd.isna(v)
                            else Decimal(f'{float(v):.{2 if aaa else 3}f}')
                            for v, aaa in zip(num, is_aaa)]
        else:  # num1 / num2 / num3
            d = _DECIMALS[kind]
            num = pd.to_numeric(col, errors='coerce')
            series[name] = [None if pd.isna(v) else Decimal(f'{float(v):.{d}f}') for v in num]

    rows = list(zip(*[series[c] for c in out_cols]))
    return out_cols, rows


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_rows(conn, columns, rows, table, on_conflict='PitchID', page_size=1000):
    """Insert rows into `table`, upserting on `on_conflict` (or plain insert if
    None). Commits on success. Returns the number of rows sent."""
    if not rows:
        return 0
    from psycopg2.extras import execute_values
    collist = ', '.join(f'"{c}"' for c in columns)
    if on_conflict:
        updates = ', '.join(f'"{c}" = EXCLUDED."{c}"'
                            for c in columns if c != on_conflict)
        sql = (f'INSERT INTO "{table}" ({collist}) VALUES %s '
               f'ON CONFLICT ("{on_conflict}") DO UPDATE SET {updates}')
    else:
        sql = f'INSERT INTO "{table}" ({collist}) VALUES %s'
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=page_size)
    conn.commit()
    return len(rows)


def push_csv_to_supabase(df, conn=None, verbose=True):
    """Append `df` to Supabase, routing each PTeam's rows to that team's table
    (upsert on PitchID). Mirrors sheets_append.push_csv_to_sheets, which groups
    by PTeam and pushes each group to its tab. Returns total rows upserted.
    """
    if 'PitchID' not in df.columns or 'PTeam' not in df.columns:
        if verbose:
            print('  [supabase] dataframe missing PitchID/PTeam column; skipping push')
        return 0

    # Rows with no usable PitchID can't be upserted idempotently — drop them.
    has_id = df['PitchID'].apply(lambda v: not _is_blank(v))
    df = df[has_id]
    if len(df) == 0:
        if verbose:
            print('  [supabase] no rows with a PitchID; skipping push')
        return 0

    own = conn is None
    if own:
        conn = get_conn()
    try:
        total = 0
        for team, group in df.groupby('PTeam', sort=False):
            table = table_for_team(team)
            if not table:
                if verbose:
                    print(f'  [supabase] blank team for {len(group)} rows; skipping')
                continue
            ensure_team_table(conn, table)
            cols, rows = prepare_rows(group)
            n = upsert_rows(conn, cols, rows, table=table, on_conflict='PitchID')
            total += n
            if verbose:
                print(f'  [supabase] {team}: upserted {n} rows into "{table}"')
        return total
    finally:
        if own:
            conn.close()
