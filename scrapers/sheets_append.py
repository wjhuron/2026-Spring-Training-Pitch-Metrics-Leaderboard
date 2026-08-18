"""Append Pitcher2026 download output directly into the AL 2026 / NL 2026
Google Sheets workbooks.

Usage:
    from scrapers.sheets_append import push_csv_to_sheets
    push_csv_to_sheets(combined_df)

The dataframe is grouped by `PTeam`, and each group is appended to the tab
matching that team in the appropriate workbook (NL 2026 or AL 2026). For ROC
downloads the data already has PTeam normalized to {'ROC', 'AAA'} by
Pitcher2026.py, so it splits cleanly across the ROC and AAA tabs in NL 2026.

Behavior:
- The dataframe's header row is NOT pushed. The destination tabs already have
  headers in row 1; new rows go into the first blank line below the existing
  data.
- Each appended block is formatted Helvetica Neue, size 8, centered, to match
  the existing data in the sheet.

One-time auth setup (service account — preferred, headless, never expires):
    1. Google Cloud Console → pick/create a project.
    2. Enable the Google Sheets API for that project.
    3. IAM & Admin → Service Accounts → Create service account.
    4. On the new account: Keys → Add key → JSON → download.
    5. Save the JSON to ~/.config/gspread/service_account.json.
    6. Open the downloaded JSON, copy the "client_email" value, and share
       BOTH the "AL 2026" and "NL 2026" workbooks with that email as
       Editor (the same way you'd share with a person).

    The service account has no token expiry and needs no browser, so the
    pipeline runs unattended forever.

Legacy fallback (interactive OAuth — used only if no service_account.json):
    Desktop OAuth client at ~/.config/gspread/credentials.json, browser
    consent cached at ~/.config/gspread/authorized_user.json. NOTE: if the
    OAuth consent screen is in "Testing" status, Google revokes the refresh
    token after 7 days ("invalid_grant: Token has been expired or revoked"),
    which is exactly why the service-account path above is preferred.
"""

# Runnable as a file from any directory (IDE run buttons included):
# put the repo root on sys.path before the intra-repo package imports.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import os
import sys
import math

import pandas as pd

# Legacy 2-workbook IDs (kept for old imports; no longer written to).
SHEETS_NL = '1DH3NI-3bSXW7dl98tdg5uFgJ4O6aWRvRB_XnVb340YE'
SHEETS_AL = '1hzAtZ_Wqi8ZuUHaGvgjJcQMU5jj5CzGXuBtjYmPOj9U'

NL_TEAMS = {'ARI', 'ATL', 'CHC', 'CIN', 'COL', 'LAD', 'MIA', 'MIL',
            'NYM', 'PHI', 'PIT', 'SDP', 'SFG', 'STL', 'WSH'}
AL_TEAMS = {'BAL', 'BOS', 'CWS', 'CLE', 'DET', 'HOU', 'KCR', 'LAA',
            'MIN', 'NYY', 'ATH', 'SEA', 'TBR', 'TEX', 'TOR'}
ROC_AAA_TEAMS = {'ROC', 'AAA'}

# 2026 layout: six per-division workbooks (replaces the two AL/NL books to stay
# under the Sheets cell limit). Each team appends to its division's workbook;
# ROC/AAA/FCL live in NL East.
WORKBOOKS = {
    'ALE2026': '1YbgAliQzXePiFan-ruwJ50G80l4AjeyTGN8cO3KJ1XI',
    'ALC2026': '14gglESfgJoT90crQb5hHoEZNUFDZ5chPLbUIV9mlm4E',
    'ALW2026': '1eSFfKRo5kSImjP0SZ1SMssGrOhrKSZM9GOHiwntIlhs',
    'NLE2026': '1BypxxlWgQAltETOLqccOYigeo8nXX-FIuVv6rhT4anA',
    'NLC2026': '1-I8BVEw9bR9rzGVYJao_Ar0bjYZF54pi5pm3YEluB9w',
    'NLW2026': '1vm257A676FORcSRzXcNj6txgehGhYI7k5mnmsgQCYH0',
}
TEAM_DIVISION = {
    'BAL': 'ALE2026', 'BOS': 'ALE2026', 'NYY': 'ALE2026', 'TBR': 'ALE2026', 'TOR': 'ALE2026',
    'CLE': 'ALC2026', 'CWS': 'ALC2026', 'DET': 'ALC2026', 'KCR': 'ALC2026', 'MIN': 'ALC2026',
    'ATH': 'ALW2026', 'HOU': 'ALW2026', 'LAA': 'ALW2026', 'SEA': 'ALW2026', 'TEX': 'ALW2026',
    'ATL': 'NLE2026', 'MIA': 'NLE2026', 'NYM': 'NLE2026', 'PHI': 'NLE2026', 'WSH': 'NLE2026',
    'ROC': 'NLE2026', 'AAA': 'NLE2026', 'FCL': 'NLE2026',
    'CHC': 'NLC2026', 'CIN': 'NLC2026', 'MIL': 'NLC2026', 'PIT': 'NLC2026', 'STL': 'NLC2026',
    'ARI': 'NLW2026', 'COL': 'NLW2026', 'LAD': 'NLW2026', 'SDP': 'NLW2026', 'SFG': 'NLW2026',
}
WORKBOOK_LABEL = {wid: name for name, wid in WORKBOOKS.items()}


def _workbook_id_for_team(team):
    div = TEAM_DIVISION.get(('' if team is None else str(team)).strip())
    return WORKBOOKS.get(div) if div else None


def _get_client():
    """Return an authorized gspread client.

    Prefers a service account (headless, no token expiry) at gspread's
    conventional path ~/.config/gspread/service_account.json. Falls back to
    interactive OAuth only if no service-account key is present, so an
    in-progress migration never hard-breaks the pipeline.
    """
    try:
        import gspread
    except ImportError:
        raise RuntimeError(
            "gspread is required for sheets push. Install with: pip install gspread"
        )

    sa_path = os.path.expanduser('~/.config/gspread/service_account.json')
    if os.path.exists(sa_path):
        # Service account: no browser, no 7-day refresh-token expiry. The
        # AL 2026 / NL 2026 workbooks must each be shared (Editor) with the
        # account's client_email or gspread raises APIError 403 / 404.
        return gspread.service_account(filename=sa_path)

    # Fallback: interactive OAuth (Desktop client). Breaks every 7 days if
    # the OAuth consent screen is in "Testing" status — see module docstring.
    cred_path = os.path.expanduser('~/.config/gspread/credentials.json')
    if not os.path.exists(cred_path):
        raise RuntimeError(
            "No Google Sheets credentials found.\n"
            "Preferred (headless, never expires) — save a service-account "
            f"key to {sa_path}:\n"
            "  1. Google Cloud Console → enable the Google Sheets API.\n"
            "  2. IAM & Admin → Service Accounts → Create; add a JSON key.\n"
            "  3. Save the JSON to the path above.\n"
            "  4. Share the AL 2026 and NL 2026 Sheets (Editor) with the\n"
            "     service account's client_email.\n"
            f"Or legacy OAuth — save a Desktop OAuth client to {cred_path}."
        )
    return gspread.oauth()  # uses ~/.config/gspread/{credentials,authorized_user}.json


def _col_letter(n):
    """1-indexed column number to A1 letter (1 → 'A', 27 → 'AA', etc.)."""
    s = ''
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# Google returns sporadic 429s (write quota is 60 requests/min/user) and
# transient 5xx on any Sheets endpoint — metadata reads included. Historically
# only ws.update() here was retried, and only on 429, so a single blip on any
# of the ~10 follow-up formatting calls left the just-appended rows raw:
# Game Date rendering as the date serial (46227) and RTilt/OTilt as decimals.
# Every round-trip in this module now goes through this wrapper, mirroring
# pipeline_fetch.sheets_call_with_retry.
_TRANSIENT_CODES = ('429', '500', '502', '503', '504')


def _sheets_retry(fn, label='sheets call', max_retries=5, verbose=True):
    """Run a gspread call, retrying rate-limit and transient backend errors."""
    import time as _time
    import gspread as _gspread
    for attempt in range(max_retries):
        try:
            return fn()
        except _gspread.exceptions.APIError as e:
            msg = str(e)
            code = next((c for c in _TRANSIENT_CODES if c in msg), None)
            if code is None or attempt >= max_retries - 1:
                raise
            # Write quota is a per-minute bucket, so a 429 needs to outwait the
            # whole window; 5xx clears much faster.
            wait = 70 if code == '429' else min(60, 5 * (2 ** attempt))
            if verbose:
                kind = 'rate limited (429)' if code == '429' else f'transient {code}'
                print(f"  [sheets] {label}: {kind}; waiting {wait}s "
                      f"(retry {attempt + 1}/{max_retries - 1})...")
            _time.sleep(wait)


def _format_requests(ws, columns, start_row_1idx, end_row_1idx, src_row_1idx=2):
    """Build the full batch_update request list that makes an appended block
    look like the rest of the sheet.

    Returned as ONE list so the caller can issue a single atomic API call:
    the Sheets API applies the requests in order and rejects the batch as a
    whole, so the block can never end up half-formatted. This replaces the old
    sequence of ~10 independent calls (copyPaste + 2 ws.format + 7 per-column
    ws.format), which was both the quota hog and the corruption window.

    Order matters and mirrors the original sequence:
      1. copyPaste PASTE_FORMAT from row 2 — inherits every per-column number
         format Wally set up (Velocity `0.0`, Extension `0.00`, PlateX
         `0.000`, Game Date `yyyy-mm-dd`, RTilt/OTilt `h:mm`).
      2. Base cell format — Helvetica Neue 8, centered, top, 1px solid borders.
      3. Column A (Game Date) bold on top of the base.
      4. EXPLICIT_NUMBER_FORMATS, which don't trust row 2's state.
    """
    n_cols = len(columns)
    start_0 = start_row_1idx - 1
    end_0 = end_row_1idx          # exclusive
    grid = {'sheetId': ws.id, 'startRowIndex': start_0, 'endRowIndex': end_0,
            'startColumnIndex': 0, 'endColumnIndex': n_cols}
    reqs = []

    # 1. Inherit per-column formats from the canonical data row. Only valid
    #    when there's an existing data row above the new block.
    if start_row_1idx > src_row_1idx:
        reqs.append({'copyPaste': {
            'source': {'sheetId': ws.id,
                       'startRowIndex': src_row_1idx - 1,
                       'endRowIndex': src_row_1idx,
                       'startColumnIndex': 0, 'endColumnIndex': n_cols},
            'destination': dict(grid),
            'pasteType': 'PASTE_FORMAT',
            'pasteOrientation': 'NORMAL',
        }})

    # 2. Base look. `fields` deliberately omits numberFormat so this never
    #    clobbers what step 1 inherited.
    _SOLID = {'style': 'SOLID', 'width': 1}
    base_fmt = {
        'textFormat': {'fontFamily': 'Helvetica Neue', 'fontSize': 8, 'bold': False},
        'horizontalAlignment': 'CENTER',
        'verticalAlignment': 'TOP',
        'borders': {'top': _SOLID, 'bottom': _SOLID,
                    'left': _SOLID, 'right': _SOLID},
    }
    fmt_fields = ('userEnteredFormat.textFormat,'
                  'userEnteredFormat.horizontalAlignment,'
                  'userEnteredFormat.verticalAlignment,'
                  'userEnteredFormat.borders')
    reqs.append({'repeatCell': {'range': dict(grid),
                                'cell': {'userEnteredFormat': base_fmt},
                                'fields': fmt_fields}})

    # 3. Bold pass on column A (Game Date).
    reqs.append({'repeatCell': {
        'range': {**grid, 'endColumnIndex': 1},
        'cell': {'userEnteredFormat': {
            **base_fmt,
            'textFormat': {**base_fmt['textFormat'], 'bold': True}}},
        'fields': fmt_fields}})

    # 4. Number formats we pin by name rather than inherit.
    cols = list(columns)
    for col_name, fmt_spec in EXPLICIT_NUMBER_FORMATS.items():
        if col_name not in cols:
            continue
        idx0 = cols.index(col_name)
        reqs.append({'repeatCell': {
            'range': {**grid, 'startColumnIndex': idx0, 'endColumnIndex': idx0 + 1},
            'cell': {'userEnteredFormat': {'numberFormat': fmt_spec}},
            'fields': 'userEnteredFormat.numberFormat'}})

    return reqs


def _column_a_is_dated(ws, row_1idx):
    """True if column A of `row_1idx` carries a DATE number format.

    The single cheapest tell that formatting landed: an unformatted Game Date
    cell falls back to Automatic and renders (and CSV-exports) as the raw
    serial, e.g. 46227 instead of 2026-07-24.
    """
    rng = f"'{ws.title}'!A{row_1idx}"
    meta = _sheets_retry(
        lambda: ws.spreadsheet.fetch_sheet_metadata({
            'includeGridData': True, 'ranges': [rng],
            'fields': 'sheets(data(rowData(values(effectiveFormat(numberFormat/type)))))',
        }),
        label=f'{ws.title} format check')
    try:
        cell = meta['sheets'][0]['data'][0]['rowData'][0]['values'][0]
        return cell['effectiveFormat']['numberFormat']['type'] == 'DATE'
    except (KeyError, IndexError, TypeError):
        return False


def apply_block_format(ws, columns, start_row_1idx, end_row_1idx, verbose=True):
    """Format an appended block atomically, then verify and re-apply once.

    Raises if the block is still unformatted after the retry — leaving raw
    date serials in the sheet silently is the failure this whole path exists
    to prevent, so it must be loud.
    """
    reqs = _format_requests(ws, columns, start_row_1idx, end_row_1idx)
    label = f'{ws.title} rows {start_row_1idx}-{end_row_1idx} format'
    _sheets_retry(lambda: ws.spreadsheet.batch_update({'requests': reqs}), label=label)

    # Cheap read-back on the first appended row. Catches the case where the
    # batch reported success but the formatting didn't stick.
    if _column_a_is_dated(ws, start_row_1idx):
        return True
    if verbose:
        print(f"  [sheets] {ws.title}: format verify failed on row "
              f"{start_row_1idx}; re-applying...")
    _sheets_retry(lambda: ws.spreadsheet.batch_update({'requests': reqs}),
                  label=label + ' (retry)')
    if not _column_a_is_dated(ws, start_row_1idx):
        raise RuntimeError(
            f"[sheets] {ws.title}: rows {start_row_1idx}-{end_row_1idx} were "
            f"appended but could NOT be formatted — Game Date will render as a "
            f"raw serial. Re-run scripts/ops/fix_unformatted_blocks.py to repair.")
    return True


# Columns whose string values would otherwise be misparsed by USER_ENTERED.
# Baseball counts like "1-0" get parsed as the date "January 0" and stored
# as date serials (46022, 46023, …) — the cell then displays as "46023"
# instead of "1-0". Prefixing each value with a leading apostrophe forces
# Sheets to store the cell as text; the apostrophe is hidden in the rendered
# display and downstream readers still see "1-0".
TEXT_FORCE_COLUMNS = {'Count'}


# Columns where we explicitly enforce a number format on every appended block,
# independent of what row 2 happens to look like. The format-paste-from-row-2
# step inherits most column formats correctly, but some tabs have row 2 cells
# with the format missing or wrong — e.g. AAA's row 2 RTilt was empty, which
# caused the whole RTilt column to render as raw decimals (0.0986) instead of
# clock times (2:22). Enforcing these by name makes the push robust to any
# tab's row 2 state.
EXPLICIT_NUMBER_FORMATS = {
    'RTilt':           {'type': 'TIME',   'pattern': 'h:mm'},
    'OTilt':           {'type': 'TIME',   'pattern': 'h:mm'},
    # Supplemental columns whose row-2 cells are AUTOMATIC-formatted. Under
    # USER_ENTERED, a whole-number write like "41.0" is parsed to the number 41
    # and the trailing zero is dropped; a later backfill then reads "41", sees a
    # mismatch against its freshly-formatted "41.0", and phantom-overwrites the
    # cell. Pinning one decimal makes the stored value display (and read back) as
    # "41.0" regardless of any tab's row-2 state.
    'ArmAngle':        {'type': 'NUMBER', 'pattern': '0.0'},
    'SwingLength':     {'type': 'NUMBER', 'pattern': '0.0'},
    'AttackAngle':     {'type': 'NUMBER', 'pattern': '0.0'},
    'AttackDirection': {'type': 'NUMBER', 'pattern': '0.0'},
    'SwingPathTilt':   {'type': 'NUMBER', 'pattern': '0.0'},
}


def _force_text_for_columns(df, cols):
    """Prefix non-empty string values in `cols` with `'` so Sheets'
    USER_ENTERED parser treats them as text and skips number/date coercion.
    Returns a copy of `df`; never mutates the caller's dataframe.
    """
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return df
    df = df.copy()
    for c in cols_present:
        df[c] = df[c].apply(lambda v: f"'{v}" if isinstance(v, str) and v else v)
    return df


def _df_to_rows(df):
    """Convert a dataframe to a list-of-lists suitable for gspread. NaN /
    NaT / pd.NA become empty strings (Sheets shows blank rather than '#N/A').
    numpy scalar types (int64/float64/bool_) are converted to Python natives
    so the result is JSON-serializable for gspread's API call."""
    out = []
    for row in df.itertuples(index=False, name=None):
        cleaned = []
        for v in row:
            if v is None:
                cleaned.append('')
            elif isinstance(v, float) and math.isnan(v):
                cleaned.append('')
            elif pd.isna(v):
                cleaned.append('')
            elif hasattr(v, 'item'):
                # numpy scalar (int64, float64, bool_, etc.) → Python native
                cleaned.append(v.item())
            else:
                cleaned.append(v)
        out.append(cleaned)
    return out


def push_team_data(df, team, gc=None, verbose=True, dedupe=True):
    """Append `df` (already filtered to one team's rows) to the sheet tab
    named `team` in whichever workbook owns that team.

    dedupe=True (the default) drops any incoming row whose PitchID is already
    in the tab, so a push can never duplicate what is already there. That makes
    every pull idempotent: pull a game at the 5th inning, pull it again when it
    is final, and only the pitches that were not there before get appended.

    This is not hypothetical. On 2026-07-13 a backfill appended 1,187 duplicate
    rows because nothing checked. Before this guard existed the ONLY protection
    was remembering not to re-pull.

    Returns (next_row, n_appended) on success, or None if skipped or if
    everything was already present.
    """
    wb_id = _workbook_id_for_team(team)
    if wb_id is None:
        if verbose:
            print(f"  [sheets] team {team!r} not in NL/AL/ROC mapping; skipping")
        return None

    if gc is None:
        gc = _get_client()
    wb = gc.open_by_key(wb_id)

    try:
        ws = wb.worksheet(team)
    except Exception as e:
        if verbose:
            print(f"  [sheets] tab {team!r} not found in workbook ({e}); skipping")
        return None

    # Force text storage for columns whose values would be misparsed by
    # USER_ENTERED (e.g. Count "1-0" → date serial 46023).
    df = _force_text_for_columns(df, TEXT_FORCE_COLUMNS)

    rows = _df_to_rows(df)
    if not rows:
        return None

    # ── Schema guard (2026-07-18) ── The appended rows are POSITIONAL: if the
    # tab's header row doesn't match the DataFrame's columns exactly, every
    # value after the divergence point lands in the wrong column — silent row
    # corruption. Refuse loudly instead. This is the tripwire that makes
    # schema migrations (e.g. the Stuff+/Loc+ insert after HAA) safe: a tab
    # missed by the migration, or an old Pitcher2026 running against a
    # migrated tab, fails here with an explicit diff instead of appending
    # misaligned rows. An entirely empty header row (brand-new tab) is
    # allowed through — there is nothing to misalign against.
    sheet_header = _sheets_retry(lambda: ws.row_values(1),
                                 label=f'{team} header read')
    expected = [str(c) for c in df.columns]
    if sheet_header and sheet_header != expected:
        diffs = [f"col {i+1}: sheet={sh!r} != df={ex!r}"
                 for i, (sh, ex) in enumerate(
                     __import__('itertools').zip_longest(sheet_header, expected))
                 if sh != ex][:6]
        raise RuntimeError(
            f"[sheets] SCHEMA MISMATCH on tab {team!r} — refusing to append "
            f"({len(sheet_header)} sheet cols vs {len(expected)} df cols). "
            f"First diffs: {'; '.join(diffs)}. If the schema just changed, "
            f"run the column migration (scripts/archive/migrate_sheets_grade_columns.py) "
            f"or update Pitcher2026's final_columns to match.")

    # ── PitchID dedupe ── Read the tab's existing PitchIDs and drop anything
    # already present. One extra column read per push, and it is what makes a
    # re-pull of a partially-scraped game safe.
    if dedupe:
        if 'PitchID' not in expected:
            if verbose:
                print(f"  [sheets] {team}: no PitchID column — dedupe SKIPPED, "
                      f"a re-push of the same rows would duplicate them")
        else:
            pid_col = expected.index('PitchID') + 1     # gspread is 1-based
            existing = set(_sheets_retry(
                lambda: ws.col_values(pid_col),
                label=f'{team} PitchID read')[1:])      # drop the header cell
            pid_i = pid_col - 1
            before = len(rows)
            rows = [r for r in rows
                    if pid_i < len(r) and str(r[pid_i]) not in existing]
            skipped = before - len(rows)
            if verbose and skipped:
                print(f"  [sheets] {team}: {skipped} of {before} rows already "
                      f"in the tab — appending {len(rows)}")
            if not rows:
                if verbose:
                    print(f"  [sheets] {team}: all {before} rows already "
                          f"present, nothing to append")
                return None

    # First blank row in column A. Header row 1 stays untouched. Using
    # col_values('A') is more reliable than get_all_values() because trailing
    # blank rows are correctly trimmed.
    col_a = _sheets_retry(lambda: ws.col_values(1), label=f'{team} col A read')
    next_row = len(col_a) + 1
    if next_row < 2:
        next_row = 2  # never overwrite a (possibly missing) header row

    end_row = next_row + len(rows) - 1
    end_col = _col_letter(len(df.columns))
    range_name = f"A{next_row}:{end_col}{end_row}"

    # Grow the tab's grid if the new block would run past its last row.
    # Sheets tabs have a fixed row count (new tabs default to 1000), and
    # ws.update() with a range beyond that ceiling fails with "exceeds grid
    # limits" — the same wall that forces the manual "add 1000 rows" step.
    # Resize first, with a 1000-row buffer so back-to-back pushes don't have
    # to resize every single time.
    if end_row > ws.row_count:
        _sheets_retry(lambda: ws.add_rows(end_row - ws.row_count + 1000),
                      label=f'{team} add_rows')

    # Grow the tab's grid width too. Tabs are provisioned at the exact schema
    # width, so a push whose block is wider than the current grid (e.g. after a
    # new column like Barrel is added to the schema) would otherwise fail with
    # "exceeds grid limits" just like the row ceiling above.
    n_cols = len(df.columns)
    if n_cols > ws.col_count:
        _sheets_retry(lambda: ws.add_cols(n_cols - ws.col_count),
                      label=f'{team} add_cols')

    _sheets_retry(
        lambda: ws.update(range_name, rows, value_input_option='USER_ENTERED'),
        label=f'{team} rows {next_row}-{end_row} write')

    # Format the block. This is NOT cosmetic: without the number formats the
    # USER_ENTERED-parsed values render as their raw storage — Game Date as a
    # date serial (46227), RTilt/OTilt as day fractions (0.045138) — and that
    # is exactly what a CSV export then contains. apply_block_format issues a
    # single atomic batch_update, verifies it stuck, and raises if it didn't.
    apply_block_format(ws, list(df.columns), next_row, end_row, verbose=verbose)

    if verbose:
        wb_label = WORKBOOK_LABEL.get(wb_id, wb_id)
        print(f"  [sheets] {team}: appended {len(rows)} rows to {wb_label} → "
              f"{team} (rows {next_row}-{end_row})")
    return next_row, len(rows)


def push_csv_to_sheets(df, verbose=True):
    """Group `df` by PTeam and push each group to its sheet tab.

    Most downloads have a single PTeam — one tab gets one update. ROC
    downloads (after Pitcher2026.py normalization) have PTeam ∈ {ROC, AAA},
    so they split cleanly across the ROC and AAA tabs of NL 2026.

    Teams with no NL/AL/ROC mapping (FCL / complex-league or international
    games) can't be routed to a workbook. They're collected and returned so
    the caller can fall back to a local CSV instead of silently dropping the
    data. Auth is deferred until a mappable team is found, so an all-unmapped
    push needs no Google credentials at all.

    Returns the list of unmapped (un-pushed) team names.
    """
    if 'PTeam' not in df.columns:
        if verbose:
            print("  [sheets] dataframe has no PTeam column; skipping push")
        return []

    # Drop rows with no PTeam — they can't be routed
    valid = df.dropna(subset=['PTeam'])
    valid = valid[valid['PTeam'].astype(str).str.strip() != '']
    if len(valid) == 0:
        if verbose:
            print("  [sheets] no rows with a PTeam value; skipping push")
        return []

    gc = None  # lazy: only authenticate once we hit a mappable team
    unmapped = []
    failures = []
    for team, group in valid.groupby('PTeam', sort=False):
        team = str(team)
        if _workbook_id_for_team(team) is None:
            if verbose:
                print(f"  [sheets] team {team!r} not in NL/AL/ROC mapping; "
                      f"skipping push (will write CSV)")
            unmapped.append(team)
            continue
        if gc is None:
            gc = _get_client()  # auth once, shared across the mappable groups
        # Isolate each team. Previously a raise mid-loop (e.g. a formatting
        # failure on the first team) aborted the whole push, silently dropping
        # every remaining team's rows — a game pull would land one side of the
        # matchup and lose the other.
        try:
            push_team_data(group, team, gc=gc, verbose=verbose)
        except Exception as e:
            failures.append((team, e))
            print(f"  [sheets] {team}: PUSH FAILED ({type(e).__name__}: {e}) "
                  f"— continuing with remaining teams")

    if failures:
        raise RuntimeError(
            "[sheets] push failed for " +
            ", ".join(f"{t} ({type(e).__name__}: {e})" for t, e in failures))
    return unmapped
