#!/usr/bin/env python3
"""
Full-column backfill: reconcile every Google Sheet column against its source.

`backfill_supplement.py` fills the 12 Savant-served columns and nothing else.
This module reconciles all 43 in-scope columns, 31 of which come from the MLB
Stats API game feed and have never been backfillable. It also adds the two
things the supplement path cannot do: it finds pitches that are missing from the
sheet entirely, and it rewrites stored values at full source precision.

Out of scope by instruction (Wally, 2026-08-17): Game Date, PTeam, BTeam,
PitchID, Pitch Type, Stuff+, Loc+, Pitching+.

DRY RUN IS THE DEFAULT. Nothing is written to any sheet without --apply.

    python3 -m scrapers.backfill_full                    # dry run, whole season
    python3 -m scrapers.backfill_full --teams ARI,WSH    # dry run, two tabs
    python3 -m scrapers.backfill_full --apply            # write

Design notes that are not obvious:

  * The feed is parsed by Pitcher2026.download_game_data, not by a second copy
    of the parser here. It is called with raw_precision=True and a disk-cached
    payload. One parser means the backfill can never disagree with the scraper
    that wrote the row in the first place.

  * "Precision" changes and "drift" changes are separated, because they need
    different review. A cell is a precision change only when the stored value is
    exactly what you get by rounding today's source value to the stored value's
    own decimal depth. Anything else is real movement in the source.

  * Blank cells are not all the same thing. Wally deliberately deleted a small
    number of Velocity, Spin Rate, RTilt and movement values that were obvious
    misreads (a 55 mph slip, a 200 rpm curveball). Refilling those with the same
    number would undo the decision, so a blank-fill candidate is suppressed when
    the source still offers the value that was rejected. See DecisionLedger.
"""

# Runnable as a file from any directory (IDE run buttons included):
# put the repo root on sys.path before the intra-repo package imports.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import collections
import gzip
import json
import math
import os
import time
from datetime import datetime

import gspread
import pandas as pd
import requests

from pipeline.fetch import DIVISION_WORKBOOK_IDS as SPREADSHEET_IDS
from scrapers.backfill_supplement import (
    MLB_TEAMS, MILB_TEAMS, ALL_TRACKED_TEAMS,
    _retry_sheets_call, read_sheet_with_retry, update_cells_with_retry,
    download_statcast,
)
from scrapers.sheet_precision import (
    PRECISION, NUMBER_FORMATS, STRING_COLS, TIME_COLS,
    fmt, as_float, stored_decimals,
)

DATA = os.path.join(_ROOT, 'data')
FEED_CACHE = os.path.join(DATA, '_feed_cache')
LEDGER_PATH = os.path.join(DATA, 'backfill_decisions.json')
REPORT_DIR = os.path.join(os.path.expanduser('~'), 'Downloads')


# ── Column scope ─────────────────────────────────────────────────────────────
# Wally's exclusion list. Game Date/PTeam/BTeam/PitchID/Pitch Type are identity
# and must never move under a row. The three grade columns are written by
# scripts/ci/sheets_write_grades.py from the models, not by any feed.
EXCLUDED_COLS = {
    'Game Date', 'PTeam', 'BTeam', 'PitchID', 'Pitch Type',
    'Stuff+', 'Loc+', 'Pitching+',
}

# Columns whose authority is the MLB Stats API game feed. Six of these are
# computed from feed values rather than read from it (RTilt, OTilt, VAA, HAA,
# xIndVrtBrk, xHorzBrk) — Pitcher2026 owns that math.
FEED_COLS = [
    'Pitcher', 'Throws', 'Velocity', 'Spin Rate', 'RTilt', 'OTilt',
    'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk',
    'RelPosZ', 'RelPosX', 'Extension', 'PlateZ', 'PlateX', 'SzTop', 'SzBot',
    'VAA', 'HAA', 'Batter', 'Bats', 'Count', 'Outs', 'Description', 'Event',
    'ExitVelo', 'LaunchAngle', 'Distance', 'BBType', 'HC_X', 'HC_Y',
]

# Columns whose authority is Baseball Savant. Unchanged from the supplement path.
SAVANT_COLS = [
    'ArmAngle', 'xBA', 'xSLG', 'xwOBA', 'RunExp', 'Barrel', 'Runners',
    'BatSpeed', 'SwingLength', 'AttackAngle', 'AttackDirection', 'SwingPathTilt',
]

IN_SCOPE = FEED_COLS + SAVANT_COLS
SAVANT_COL_SET = set(SAVANT_COLS)
assert not (set(IN_SCOPE) & EXCLUDED_COLS), 'scope lists disagree'
assert not (set(FEED_COLS) & SAVANT_COL_SET), 'a column has two authorities'

# The precision policy and the formatter live in scrapers/sheet_precision.py so
# this module and backfill_supplement.py cannot disagree about a column's depth.


# ── Feed cache ───────────────────────────────────────────────────────────────
FEED_URL = 'https://statsapi.mlb.com/api/v1.1/game/{}/feed/live'


def feed_path(game_pk):
    return os.path.join(FEED_CACHE, f'{game_pk}.json.gz')


def fetch_game_json(game_pk, session, refresh=False):
    """Return one game's feed payload, from disk when cached.

    Fails closed on every path. A non-200, a short body or unparseable JSON
    raises rather than returning a partial game, because a partial game reads
    downstream as "these pitches do not exist" and would be reported as rows to
    delete or refill. Writes to a temp path and moves, so an interrupted run
    never leaves a truncated cache entry behind.
    """
    path = feed_path(game_pk)
    if os.path.exists(path) and not refresh:
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, EOFError, json.JSONDecodeError, gzip.BadGzipFile) as e:
            print(f"    cached feed for {game_pk} is unreadable "
                  f"({type(e).__name__}); re-fetching")

    last = None
    for attempt in range(4):
        try:
            r = session.get(FEED_URL.format(game_pk), timeout=(15, 90))
            if r.status_code != 200:
                last = f'HTTP {r.status_code}'
                if r.status_code >= 500 and attempt < 3:
                    time.sleep(5 * (2 ** attempt))
                    continue
                raise RuntimeError(f'feed {game_pk}: {last}')
            data = r.json()
            if not data or 'liveData' not in data:
                raise RuntimeError(f'feed {game_pk}: payload has no liveData')
            os.makedirs(FEED_CACHE, exist_ok=True)
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, path)
            return data
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                json.JSONDecodeError) as e:
            last = f'{type(e).__name__}: {e}'
            if attempt < 3:
                time.sleep(5 * (2 ** attempt))
                continue
            raise RuntimeError(f'feed {game_pk}: {last}') from e
    raise RuntimeError(f'feed {game_pk}: {last}')


_SCRAPER = None


def _scraper():
    """One Pitcher2026 instance for the process. Importing it is not free."""
    global _SCRAPER
    if _SCRAPER is None:
        from scrapers.pitcher2026 import BaseballSavantFocusedDownloader
        _SCRAPER = BaseballSavantFocusedDownloader()
    return _SCRAPER


_FEED_ROWS_CACHE = {}


def feed_rows(game_pk, session, refresh=False):
    """PitchID -> {column: formatted string} for one game, at full precision.

    Parsed by Pitcher2026.download_game_data so this module cannot drift from
    the scraper. Memoised because a game is walked once per team tab.
    """
    if game_pk in _FEED_ROWS_CACHE:
        return _FEED_ROWS_CACHE[game_pk]
    payload = fetch_game_json(game_pk, session, refresh=refresh)
    df = _scraper().download_game_data(game_pk, game_json=payload,
                                       raw_precision=True)
    out = {}
    if df is not None and len(df):
        df = _scraper().normalize_aaa_labels(df)
        cols = [c for c in FEED_COLS if c in df.columns]
        for rec in df.to_dict('records'):
            pid = rec.get('PitchID')
            if not pid:
                continue
            out[pid] = {c: fmt(c, rec.get(c)) for c in cols}
            out[pid]['_PTeam'] = str(rec.get('PTeam') or '')
    _FEED_ROWS_CACHE[game_pk] = out
    return out


# ── Prior-decision ledger ────────────────────────────────────────────────────
# Wally deleted a handful of values on purpose because they were obvious
# misreads. No ledger was ever written, but two 2026-07-06 snapshots survive and
# reconstruct what the source was offering at the time he looked:
#
#   data/_tracking_pitches.pkl   396,214 pitches, sheet value AND Savant value
#                                side by side
#   data/_statcast2026_full.pkl  435,374 Savant rows, incl. spin_axis (RTilt)
#
# Measured coverage of today's blanks on rows that do have tracking:
#   Velocity   22 blank, 17 with a record (3 held a value on Jul 6)
#   Spin Rate  2,782 blank, 1,239 with a record (56 held a value on Jul 6)
#   RTilt      1,139 blank, 1,118 with a record
#
# The snapshot cannot prove a blank was a deletion rather than a feed gap, and
# it does not need to. The test is not "was this deleted", it is "is the number
# on offer today the number that was already rejected". Exact match only —
# Wally's call 2026-08-17: no tolerance band, any difference gets reviewed.
LEDGER_COLS = {'Velocity', 'Spin Rate', 'RTilt',
               'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk'}

_SNAPSHOT_DATE = '2026-07-06'


class DecisionLedger:
    """Values Wally has already seen and left out. Blank-fill only."""

    def __init__(self, entries=None):
        # (PitchID, column) -> {'rejected': [str, ...], 'asof': str, 'origin': str}
        self.entries = entries or {}
        self.new_rejections = {}

    # -- persistence ----------------------------------------------------------
    @classmethod
    def load(cls):
        if os.path.exists(LEDGER_PATH):
            with open(LEDGER_PATH, encoding='utf-8') as f:
                blob = json.load(f)
            entries = {tuple(k.split('|', 1)): v
                       for k, v in blob.get('decisions', {}).items()}
            print(f"  decision ledger: {len(entries)} entries from "
                  f"{os.path.relpath(LEDGER_PATH, _ROOT)}")
            return cls(entries)
        print("  decision ledger: absent, bootstrapping from the 2026-07-06 "
              "snapshots")
        return cls(cls._bootstrap())

    def save(self):
        blob = {
            'version': 1,
            'note': ('Values deliberately left out of the sheet. A blank-fill '
                     'candidate matching one of these exactly is suppressed. '
                     'Bootstrapped from data/_tracking_pitches.pkl and '
                     'data/_statcast2026_full.pkl (both 2026-07-06); every run '
                     'since appends what the review rejected.'),
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'decisions': {f'{pid}|{col}': v
                          for (pid, col), v in sorted(self.entries.items())},
        }
        tmp = LEDGER_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(blob, f, indent=1, sort_keys=True)
        os.replace(tmp, LEDGER_PATH)

    # -- bootstrap ------------------------------------------------------------
    @staticmethod
    def _bootstrap():
        """Reconstruct prior decisions from the July 6 snapshots.

        Records a rejection for every LEDGER_COLS cell that is blank in the
        current sheet cache while the July 6 snapshot shows a value for it, on
        either the sheet side or the Savant side. Both numbers are stored: a
        deliberate deletion leaves the sheet-side value, and a blank that
        predates the snapshot leaves only the Savant-side proxy.
        """
        import pickle
        entries = {}
        cur_path = os.path.join(DATA, 'all_pitches_rs_cache.pkl')
        trk_path = os.path.join(DATA, '_tracking_pitches.pkl')
        sav_path = os.path.join(DATA, '_statcast2026_full.pkl')
        for p in (cur_path, trk_path, sav_path):
            if not os.path.exists(p):
                print(f"    WARNING: {os.path.relpath(p, _ROOT)} is missing, so "
                      f"the ledger bootstrap is incomplete. Every prior "
                      f"decision it covered will be offered again for review.")
                return entries

        cur = pickle.load(open(cur_path, 'rb'))
        trk = {r['pid']: r for r in pickle.load(open(trk_path, 'rb'))}
        sav = pickle.load(open(sav_path, 'rb'))
        sav['_pid'] = (sav['game_pk'].astype(int).astype(str) + '_' +
                       sav['at_bat_number'].astype(int).astype(str).str.zfill(3)
                       + '_' +
                       sav['pitch_number'].astype(int).astype(str).str.zfill(2))
        spin_axis = sav.set_index('_pid')['spin_axis'].to_dict()

        # Savant field that stands in for each sheet column in the snapshot.
        SAV_KEY = {'Velocity': 'release_speed', 'Spin Rate': 'release_spin_rate',
                   'IndVertBrk': 'pfx_z', 'HorzBrk': 'pfx_x',
                   'xIndVrtBrk': 'pfx_z', 'xHorzBrk': 'pfx_x'}

        def filled(v):
            return v not in (None, '')

        for row in cur:
            pid = row.get('PitchID')
            if not pid or pid not in trk and pid not in spin_axis:
                continue
            # Only rows that were tracked at all. A pitch with no tracking is a
            # feed gap, not a decision.
            if not (filled(row.get('PlateZ')) and filled(row.get('IndVertBrk'))):
                continue
            snap = trk.get(pid) or {}
            snap_sav = snap.get('sav') or {}
            for col in LEDGER_COLS:
                if filled(row.get(col)):
                    continue
                rejected = []
                if col == 'RTilt':
                    ax = spin_axis.get(pid)
                    if ax is not None and not pd.isna(ax):
                        t = _scraper().spin_axis_to_tilt(float(ax))
                        if t:
                            rejected.append(t)
                else:
                    sheet_then = snap.get(col)
                    if sheet_then is not None and not pd.isna(sheet_then):
                        rejected.append(fmt(col, sheet_then))
                    sk = SAV_KEY.get(col)
                    if sk and snap_sav.get(sk) is not None:
                        v = snap_sav[sk]
                        if not pd.isna(v):
                            rejected.append(fmt(col, v))
                rejected = sorted({r for r in rejected if r})
                if rejected:
                    entries[(pid, col)] = {
                        'rejected': rejected,
                        'asof': _SNAPSHOT_DATE,
                        'origin': 'bootstrap_2026_07_06_snapshot',
                    }
        print(f"    bootstrapped {len(entries)} prior decisions")
        by_col = collections.Counter(c for _, c in entries)
        for c, n in by_col.most_common():
            print(f"      {c:12} {n}")
        return entries

    # -- use ------------------------------------------------------------------
    def suppresses(self, pid, col, candidate):
        """True when `candidate` is a value already rejected for this cell."""
        e = self.entries.get((pid, col))
        return bool(e) and candidate in e['rejected']

    def record(self, pid, col, value):
        """Note that `value` was offered for a blank cell and declined."""
        key = (pid, col)
        e = self.entries.setdefault(
            key, {'rejected': [], 'asof': datetime.now().strftime('%Y-%m-%d'),
                  'origin': 'review'})
        if value not in e['rejected']:
            e['rejected'] = sorted(e['rejected'] + [value])
            self.new_rejections[key] = value


# ── Diff ─────────────────────────────────────────────────────────────────────
# One row per proposed cell change.
Change = collections.namedtuple(
    'Change', 'tab row col pitcher batter game_date pitch_id '
              'old new kind delta source')

# kind:
#   new         sheet cell is blank, the source has a value
#   drift       both present and they disagree beyond stored precision
#   precision   both present, same number, the source carries more digits
#   suppressed  blank cell, but the source still offers a rejected value
#   missing     the whole pitch is absent from the sheet


def classify(col, old, new):
    """Return 'new' | 'drift' | 'drift_sub' | 'precision' | None for one cell.

    The three change classes exist because they need different review, and
    lumping them together buries the ~15k cells that carry real information
    under ~200k that do not.

      precision  The stored value is EXACTLY today's source value rounded to the
                 stored value's own decimal depth. Nothing moved; digits are
                 being added. LaunchAngle 24 -> 24.3, SzTop 3.19 -> 3.1900.
      drift_sub  The source moved, but by less than half a unit of the stored
                 depth, so the sheet was never displaying the difference. VAA
                 stored at 2 decimals moving 0.003. Reported as a count only.
      drift      The source moved by more than the sheet was showing. This is
                 the class worth a human decision: a re-tagged Description, a
                 corrected Spin Rate, a revised xwOBA.

    `precision` is a strict subset of `drift_sub`, so it is tested first.
    """
    if new == '':
        return None                      # never blank a populated cell
    if old == '':
        return 'new'
    if old == new:
        return None
    if col in STRING_COLS or col in TIME_COLS:
        return 'drift'
    o, n = as_float(old), as_float(new)
    if o is None or n is None:
        return 'drift'
    d = stored_decimals(old)
    if d is None:
        return 'drift'
    if round(n, d) == round(o, d):
        return 'precision'
    # The epsilon is not a tuning knob, it is a float-comparison guard. Without
    # it, HAA 1.90 -> 1.905 computes a delta of 0.0050000000000001 and lands in
    # `drift` instead of `drift_sub`. That one artifact accounted for 1,826 of
    # ARI's 20,224 drift rows: HAA 662, PlateZ 654, VAA 328, RelPosX 87, RelPosZ
    # 25, SzTop 24, Extension 22, every one of them at exactly the boundary.
    if abs(n - o) <= 0.5 * (10 ** -d) + 1e-9:
        return 'drift_sub'
    return 'drift'


def modal_zone_changes(zone_obs):
    """Optional: force one SzTop/SzBot per hitter, using the modal feed value.

    OFF by default, and it should stay off unless Wally decides otherwise.
    Measured 2026-08-17 on 33 cells across 6 flagged at-bats: the sheet already
    matches the feed on 25 of them. The odd values are in MLB's own data, and
    MLB really does re-measure a stance mid-plate-appearance — Altuve's at-bat
    824170_065 reads 3.161/1.595 for pitches 1 to 3 and 2.926/1.477 for pitches
    4 to 6. Normalising those would replace faithful source data with a number
    MLB never reported for that pitch.

    Keyed on (Batter, BTeam) rather than Batter alone, because two players share
    a name: Max Muncy reads 3.128/1.579 for LAD and 3.228/1.629 for ATH, and
    both are correct. A name-only key would corrupt one of them.
    """
    by_hitter = collections.defaultdict(lambda: collections.defaultdict(int))
    for o in zone_obs:
        key = (o['batter'], o['bteam'])
        by_hitter[key][(o['sztop'], o['szbot'])] += 1

    modal = {}
    for key, counts in by_hitter.items():
        # Ties broken by the higher count then the value, so the result does not
        # depend on dict order and a re-run cannot flip.
        modal[key] = max(sorted(counts), key=lambda v: counts[v])

    out = []
    for o in zone_obs:
        key = (o['batter'], o['bteam'])
        want_top, want_bot = modal[key]
        for col, have, want in (('SzTop', o['sztop'], want_top),
                                ('SzBot', o['szbot'], want_bot)):
            if not want or have == want:
                continue
            sheet_val = o['sheet_' + col]
            if classify(col, sheet_val, want) is None:
                continue
            o_f, n_f = as_float(sheet_val), as_float(want)
            out.append(Change(
                tab=o['tab'], row=o['row'], col=col, pitcher=o['pitcher'],
                batter=o['batter'], game_date=o['game_date'],
                pitch_id=o['pid'], old=sheet_val, new=want, kind='zone_modal',
                delta=(n_f - o_f) if (o_f is not None and n_f is not None) else None,
                source='modal'))
    return out


def diff_tab(tab, rows, header, feed_by_pid, savant_lookup, ledger, zone_obs):
    """Compare one team tab against its sources. Returns (changes, missing).

    Appends one record per row to `zone_obs` so the optional modal-zone pass can
    run across every tab at once, which it must: a hitter appears in the tab of
    each team that pitched to him.
    """
    col_idx = {name: j for j, name in enumerate(header) if name}
    if 'PitchID' not in col_idx:
        return [], []
    pid_c = col_idx['PitchID']
    present = set()
    changes = []

    live = [c for c in IN_SCOPE if c in col_idx]
    for r_idx, row in enumerate(rows[1:], start=2):
        pid = row[pid_c] if pid_c < len(row) else ''
        if not pid or '_' not in pid:
            continue
        present.add(pid)
        expected = dict(feed_by_pid.get(pid) or {})
        # Savant keys on unpadded ints; the sheet pads.
        parts = pid.split('_')
        if len(parts) == 3:
            try:
                skey = (parts[0], str(int(parts[1])), str(int(parts[2])))
            except ValueError:
                skey = None
            if skey:
                for c, v in (savant_lookup.get(skey) or {}).items():
                    # download_statcast also returns Outs and Event, which the
                    # FEED owns here. Letting them through would undo the main
                    # reason for the feed pass: Savant's generic `field_out`
                    # code cannot distinguish Groundout from Lineout, so a
                    # scoring correction could never land. Authority is one
                    # source per column, enforced by the assert above.
                    if c not in SAVANT_COL_SET:
                        continue
                    expected[c] = v if c in STRING_COLS else fmt(c, as_float(v))
        if not expected:
            continue

        def cell(name):
            j = col_idx.get(name)
            return (row[j].strip() if j is not None and j < len(row) and row[j]
                    else '')

        pitcher, batter, gdate = cell('Pitcher'), cell('Batter'), cell('Game Date')

        # Feed truth for the zone, recorded whether or not it differs, because
        # the modal pass needs every observation to find the mode.
        if batter and ('SzTop' in expected or 'SzBot' in expected):
            zone_obs.append({
                'tab': tab, 'row': r_idx, 'pid': pid, 'pitcher': pitcher,
                'batter': batter, 'bteam': cell('BTeam'), 'game_date': gdate,
                'sztop': expected.get('SzTop', ''),
                'szbot': expected.get('SzBot', ''),
                'sheet_SzTop': cell('SzTop'), 'sheet_SzBot': cell('SzBot'),
            })

        for col in live:
            c_idx = col_idx[col]
            old = (row[c_idx] if c_idx < len(row) else '') or ''
            old = old.strip()
            new = expected.get(col)
            if new is None:
                continue
            kind = classify(col, old, new)
            if kind is None:
                continue
            if kind == 'new' and col in LEDGER_COLS and ledger.suppresses(pid, col, new):
                kind = 'suppressed'
            o, n = as_float(old), as_float(new)
            delta = (n - o) if (o is not None and n is not None) else None
            changes.append(Change(
                tab=tab, row=r_idx, col=col, pitcher=pitcher, batter=batter,
                game_date=gdate, pitch_id=pid, old=old, new=new, kind=kind,
                delta=delta,
                source='feed' if col in FEED_COLS else 'savant'))

    missing = []
    for pid, exp in feed_by_pid.items():
        if pid in present:
            continue
        if exp.get('_PTeam', '').upper() != tab.upper():
            continue          # belongs to the other team's tab
        missing.append((pid, exp))
    return changes, missing


# ── Report ───────────────────────────────────────────────────────────────────
# Which player column each metric is reviewed by. Zone and batted-ball columns
# read naturally by hitter; everything else by pitcher.
BATTER_SIDE = {'SzTop', 'SzBot', 'Batter', 'Bats'}


def write_report(changes, missing, ledger, path):
    """One workbook. A summary tab, then one tab per column, then extras."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)

    def sheet(title, headers):
        ws = wb.create_sheet(title=title[:31])
        for i, h in enumerate(headers, start=1):
            ws.cell(row=1, column=i, value=h).font = bold
        ws.freeze_panes = 'A2'
        return ws

    def autosize(ws, widths):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    by_col = collections.defaultdict(list)
    for ch in changes:
        by_col[ch.col].append(ch)

    # ---- Summary -----------------------------------------------------------
    ws = sheet('SUMMARY', ['Column', 'Source', 'Decimals', 'New fills',
                           'Drift (reviewed)', 'Drift below precision',
                           'Precision rewrites', 'Suppressed',
                           'Median |drift|', 'Max |drift|', 'Reviewable tab?'])
    order = sorted(by_col, key=lambda c: (c not in FEED_COLS, c))
    for c in order:
        rows_ = by_col[c]
        kinds = collections.Counter(r.kind for r in rows_)
        # Magnitudes describe the REVIEWED drift only. Mixing in the
        # sub-precision class would drag every median toward zero and make a
        # real correction look like noise.
        deltas = sorted(abs(r.delta) for r in rows_
                        if r.delta is not None and r.kind == 'drift')
        med = deltas[len(deltas) // 2] if deltas else None
        reviewable = kinds['new'] + kinds['drift'] + kinds['suppressed']
        ws.append([c, 'feed' if c in FEED_COLS else 'savant',
                   PRECISION.get(c, 'string/time'),
                   kinds['new'], kinds['drift'], kinds['drift_sub'],
                   kinds['precision'], kinds['suppressed'],
                   round(med, 6) if med is not None else '',
                   round(deltas[-1], 6) if deltas else '',
                   'yes' if reviewable else 'summary only'])
    total = collections.Counter(r.kind for r in changes)
    ws.append([])
    ws.append(['TOTAL', '', '', total['new'], total['drift'],
               total['drift_sub'], total['precision'], total['suppressed'],
               '', '', ''])
    ws.append(['Missing pitches', '', '', len(missing)])
    autosize(ws, [18, 8, 10, 11, 17, 21, 19, 11, 15, 13, 16])

    # ---- One tab per column ------------------------------------------------
    # `precision` and `drift_sub` are counted on SUMMARY and deliberately NOT
    # listed: Wally's call 2026-08-17, summary only. Together they run to
    # millions of rows and carry no judgement, because neither one moves the
    # value by as much as the sheet was already displaying.
    SUMMARY_ONLY = {'precision', 'drift_sub'}
    for c in order:
        rows_ = [r for r in by_col[c] if r.kind not in SUMMARY_ONLY]
        if not rows_:
            continue
        keyer = ((lambda r: (r.tab, r.batter, r.game_date, r.pitch_id))
                 if c in BATTER_SIDE else
                 (lambda r: (r.tab, r.pitcher, r.game_date, r.pitch_id)))
        rows_.sort(key=keyer)
        ws = sheet(c, ['Team', 'Pitcher', 'Batter', 'Game Date', 'PitchID',
                       'Sheet row', 'Change', 'Current', 'Proposed', 'Delta',
                       'Previously rejected'])
        for r in rows_:
            prior = ledger.entries.get((r.pitch_id, r.col))
            ws.append([r.tab, r.pitcher, r.batter, r.game_date, r.pitch_id,
                       r.row, r.kind, r.old, r.new,
                       round(r.delta, 6) if r.delta is not None else '',
                       ', '.join(prior['rejected']) if prior else ''])
        autosize(ws, [7, 24, 24, 12, 18, 10, 11, 12, 12, 11, 22])

    # ---- Missing pitches ---------------------------------------------------
    if missing:
        ws = sheet('_MISSING PITCHES', ['Team', 'PitchID', 'Game Date',
                                        'Pitcher', 'Batter', 'Pitch Type',
                                        'Count', 'Description'])
        rows_ = sorted(missing, key=lambda m: (m[1].get('_PTeam', ''),
                                               m[1].get('Pitcher', ''), m[0]))
        for pid, exp in rows_:
            ws.append([exp.get('_PTeam', ''), pid, exp.get('Game Date', ''),
                       exp.get('Pitcher', ''), exp.get('Batter', ''),
                       exp.get('Pitch Type', ''), exp.get('Count', ''),
                       exp.get('Description', '')])
        autosize(ws, [7, 18, 12, 24, 24, 11, 8, 22])

    # ---- Modal-zone forcing, when it was requested -------------------------
    zc = [r for r in changes if r.kind == 'zone_modal']
    if zc:
        ws = sheet('_ZONE NORMALIZE', ['Team', 'Column', 'Batter', 'Game Date',
                                       'PitchID', 'Sheet row', 'Current',
                                       'Modal', 'Delta'])
        zc.sort(key=lambda r: (r.batter, r.col, r.tab, r.pitch_id))
        for r in zc:
            ws.append([r.tab, r.col, r.batter, r.game_date, r.pitch_id, r.row,
                       r.old, r.new,
                       round(r.delta, 6) if r.delta is not None else ''])
        autosize(ws, [7, 9, 24, 12, 18, 10, 12, 12, 11])

    # ---- Suppressed, gathered in one place ---------------------------------
    supp = [r for r in changes if r.kind == 'suppressed']
    if supp:
        ws = sheet('_SUPPRESSED', ['Team', 'Column', 'Pitcher', 'Game Date',
                                   'PitchID', 'Proposed', 'Previously rejected',
                                   'Recorded'])
        supp.sort(key=lambda r: (r.col, r.tab, r.pitcher, r.pitch_id))
        for r in supp:
            e = ledger.entries.get((r.pitch_id, r.col)) or {}
            ws.append([r.tab, r.col, r.pitcher, r.game_date, r.pitch_id, r.new,
                       ', '.join(e.get('rejected', [])), e.get('asof', '')])
        autosize(ws, [7, 12, 24, 12, 18, 12, 22, 12])

    wb.save(path)
    return path


# ── Apply ────────────────────────────────────────────────────────────────────
def pin_formats(ws, header, cols):
    """Pin NUMBER_FORMATS over each column's full data range. Idempotent."""
    reqs = []
    for name in cols:
        f = NUMBER_FORMATS.get(name)
        if f is None or name not in header:
            continue
        i = header.index(name)
        reqs.append({'repeatCell': {
            'range': {'sheetId': ws.id, 'startRowIndex': 1,
                      'startColumnIndex': i, 'endColumnIndex': i + 1},
            'cell': {'userEnteredFormat': {'numberFormat': f}},
            'fields': 'userEnteredFormat.numberFormat'}})
    if reqs:
        _retry_sheets_call(lambda: ws.spreadsheet.batch_update({'requests': reqs}),
                           'number-format pin')


def apply_changes(ws, header, changes, chunk=40000):
    """Write the accepted changes. USER_ENTERED so numbers sort in the sheet."""
    col_idx = {n: j for j, n in enumerate(header) if n}
    cells = [gspread.Cell(row=c.row, col=col_idx[c.col] + 1, value=c.new)
             for c in changes if c.col in col_idx]
    for i in range(0, len(cells), chunk):
        batch = cells[i:i + chunk]
        print(f"      writing cells {i + 1}..{i + len(batch)} of {len(cells)}",
              flush=True)
        update_cells_with_retry(ws, batch, value_input_option='USER_ENTERED')
        time.sleep(2)
    pin_formats(ws, header, {c.col for c in changes})
    return len(cells)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(filter_teams=None, apply=False, refresh_feed=False, kinds=None,
         report=True, force_modal_zone=False):
    kinds = kinds or {'new', 'drift', 'drift_sub', 'precision'}
    print(f"Mode: {'APPLY' if apply else 'DRY RUN (nothing is written)'}")
    print(f"Change classes in scope: {', '.join(sorted(kinds))}")
    print(f"Columns: {len(IN_SCOPE)} ({len(FEED_COLS)} feed, "
          f"{len(SAVANT_COLS)} savant)")
    if force_modal_zone:
        print("SzTop/SzBot will be forced to each hitter's modal feed value. "
              "This overwrites values the feed still reports — see "
              "modal_zone_changes().")

    ledger = DecisionLedger.load()
    gc = gspread.service_account()
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Accept': 'application/json, text/csv',
    })

    all_changes, all_missing, zone_obs = [], [], []
    feed_failures = []
    # tab -> (worksheet, header) so the optional modal-zone pass can write after
    # the cross-tab mode is known.
    tab_handles = {}

    for label, sheet_id in SPREADSHEET_IDS.items():
        sh = _retry_sheets_call(lambda: gc.open_by_key(sheet_id), 'workbook open')
        print(f"\n{'=' * 62}\n[{label}] {sh.title}\n{'=' * 62}")

        for i, ws in enumerate(_retry_sheets_call(sh.worksheets, 'tab list')):
            tab = ws.title.upper()
            if tab not in ALL_TRACKED_TEAMS:
                continue
            if filter_teams and tab not in filter_teams:
                continue
            print(f"\n[{ws.title}]")
            if i > 0:
                time.sleep(1.5)

            rows = read_sheet_with_retry(ws)
            if not rows or len(rows) < 2:
                print("  empty tab")
                continue
            header = rows[0]
            if 'PitchID' not in header:
                print("  no PitchID column, skipping")
                continue

            pid_c = header.index('PitchID')
            game_pks, dates = set(), set()
            d_c = header.index('Game Date') if 'Game Date' in header else None
            for row in rows[1:]:
                pid = row[pid_c] if pid_c < len(row) else ''
                if pid and '_' in pid:
                    game_pks.add(pid.split('_')[0])
                if d_c is not None and d_c < len(row) and row[d_c]:
                    dates.add(row[d_c])
            if not game_pks:
                print("  no PitchIDs, skipping")
                continue
            print(f"  {len(rows) - 1} rows, {len(game_pks)} games")

            # --- feed pass ---
            feed_by_pid = {}
            for n, gpk in enumerate(sorted(game_pks), 1):
                try:
                    feed_by_pid.update(feed_rows(gpk, session,
                                                 refresh=refresh_feed))
                except RuntimeError as e:
                    # Fail closed per game, not per tab: a game we could not
                    # read must not be reported as missing pitches.
                    print(f"    FEED FAILED {gpk}: {e}")
                    feed_failures.append((tab, gpk, str(e)))
                if n % 25 == 0:
                    print(f"    feed {n}/{len(game_pks)} games", flush=True)
            print(f"    feed gave {len(feed_by_pid)} pitches")

            # --- savant pass ---
            savant_lookup = {}
            if dates:
                time.sleep(3)
                savant_lookup = download_statcast(ws.title, min(dates),
                                                  max(dates), session) or {}

            tab_handles[tab] = (ws, header)
            changes, missing = diff_tab(tab, rows, header, feed_by_pid,
                                        savant_lookup, ledger, zone_obs)
            if feed_failures and any(f[0] == tab for f in feed_failures):
                bad = {f[1] for f in feed_failures if f[0] == tab}
                missing = [m for m in missing if m[0].split('_')[0] not in bad]
                print(f"    NOTE: {len(bad)} unreadable games excluded from the "
                      f"missing-pitch list for this tab")

            k = collections.Counter(c.kind for c in changes)
            print(f"  new {k['new']} | drift {k['drift']} | "
                  f"sub-precision {k['drift_sub']} | precision {k['precision']} "
                  f"| suppressed {k['suppressed']} | missing {len(missing)}")

            all_changes.extend(changes)
            all_missing.extend(missing)

            if apply:
                todo = [c for c in changes if c.kind in kinds]
                if todo:
                    n = apply_changes(ws, header, todo)
                    print(f"  wrote {n} cells")
                else:
                    print("  nothing to write")

    # ---- optional modal-zone pass, after every tab is in ----
    if force_modal_zone and zone_obs:
        zc = modal_zone_changes(zone_obs)
        print(f"\nmodal-zone pass: {len(zc)} SzTop/SzBot cells would be forced "
              f"to the hitter's modal value")
        all_changes.extend(zc)
        if apply and 'zone_modal' in kinds:
            for tab, group in collections.groupby(
                    sorted(zc, key=lambda c: c.tab), key=lambda c: c.tab):
                ws, header = tab_handles[tab]
                n = apply_changes(ws, header, list(group))
                print(f"  {tab}: wrote {n} zone cells")

    # ---- summary ----
    k = collections.Counter(c.kind for c in all_changes)
    print(f"\n{'=' * 62}")
    print(f"new fills             {k['new']}")
    print(f"drift (reviewed)      {k['drift']}")
    print(f"drift below precision {k['drift_sub']}   (summary only)")
    print(f"precision rewrites    {k['precision']}   (summary only)")
    print(f"suppressed            {k['suppressed']}")
    print(f"missing pitches       {len(all_missing)}")
    print(f"cells to write        "
          f"{k['new'] + k['drift'] + k['drift_sub'] + k['precision']}")
    if feed_failures:
        print(f"\nWARNING: {len(feed_failures)} games could not be read. Their "
              f"pitches were left out of the missing-pitch list rather than "
              f"reported as absent. Re-run to retry:")
        for tab, gpk, err in feed_failures[:20]:
            print(f"  {tab} {gpk}: {err}")

    if report:
        os.makedirs(REPORT_DIR, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(REPORT_DIR, f'backfill_full_{stamp}.xlsx')
        write_report(all_changes, all_missing, ledger, path)
        print(f"\nReport: {path}")

    ledger.save()
    print(f"Ledger: {os.path.relpath(LEDGER_PATH, _ROOT)} "
          f"({len(ledger.entries)} entries)")
    return all_changes, all_missing


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Reconcile every sheet column against its source.')
    ap.add_argument('--teams', default=None,
                    help='comma-separated tabs, e.g. ARI,WSH')
    ap.add_argument('--apply', action='store_true',
                    help='write to the sheets. Omit for a dry run.')
    ap.add_argument('--refresh-feed', action='store_true',
                    help='re-pull every game instead of using data/_feed_cache')
    ap.add_argument('--kinds', default='new,drift,drift_sub,precision',
                    help='which change classes --apply writes')
    ap.add_argument('--force-modal-zone', action='store_true',
                    help='force one SzTop/SzBot per hitter. Off by default: the '
                         'sheet already matches the feed on most flagged cells, '
                         'and MLB does re-measure a stance mid-plate-appearance.')
    ap.add_argument('--no-report', action='store_true')
    a = ap.parse_args()
    main(filter_teams=[t.strip().upper() for t in a.teams.split(',')] if a.teams else None,
         apply=a.apply,
         refresh_feed=a.refresh_feed,
         kinds={k.strip() for k in a.kinds.split(',') if k.strip()},
         report=not a.no_report,
         force_modal_zone=a.force_modal_zone)
