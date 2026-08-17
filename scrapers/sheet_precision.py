#!/usr/bin/env python3
"""
Per-column decimal depth for the Google Sheet columns, and the one formatter.

Single home for the precision policy. `backfill_supplement.py` and
`backfill_full.py` both write the same columns, so a depth that lived in two
places would have them overwrite each other's cells on every run: one writes
53.6, the other writes 53.600, and each sees the other's value as a change
forever. `pipeline/utils.py:32` carries the same warning about PITCHING_W_STUFF,
which once ran 70/30 in one file and 80/20 in another for about three weeks.

Depths were measured 2026-08-17 by reading every field of 5 MLB feed games and
one Savant day as raw text, before any float parsing. That split the columns in
two:

CLASS 1, the source is genuinely quantized. The depth here IS the source depth
and there is nothing to decide.
    Velocity, IndVertBrk, HorzBrk, ExitVelo, ArmAngle, BatSpeed, SwingLength  1
    LaunchAngle, Distance                                                     1
    HC_X, HC_Y                                                                2
    xBA, xSLG, xwOBA, RunExp                                                  3
    Spin Rate, Outs, Barrel                                                   0
LaunchAngle and Distance were being cast to integers, so they gain a digit that
the feed was always sending.

CLASS 2, the source is a raw IEEE-754 double. The feed and Savant report 12 to
20 decimals of binary expansion, which is not measurement. There is no objective
that could pick a depth here, so these are a CONVENTION, not a measured optimum,
and they are labelled as such per the tuning rule in ~/.claude/CLAUDE.md.
Wally set them 2026-08-17: 4 decimals for feet, 3 for degrees. Reasoning on
record: 0.0001 ft is 0.0012 inch and Hawkeye resolves to roughly 0.1 inch, so 4
decimals keeps every real digit with about 100x margin, and digit 5 onward is
noise.
    PlateZ, PlateX, RelPosZ, RelPosX, Extension, SzTop, SzBot                 4
    VAA, HAA, AttackAngle, AttackDirection, SwingPathTilt                     3

xIndVrtBrk and xHorzBrk stay at 1 decimal by instruction. They are IndVertBrk
times a weather factor, and IndVertBrk itself carries only 1 decimal.

RTilt and OTilt are Sheets time values rendered h:mm and are deliberately absent
from PRECISION. Wally's call 2026-08-17: leave the representation alone.
"""

import math

import pandas as pd

PRECISION = {
    # feet — convention, 4
    'PlateZ': 4, 'PlateX': 4, 'RelPosZ': 4, 'RelPosX': 4, 'Extension': 4,
    # SzTop and SzBot are 3, not 4. Measured 2026-08-17 over 57,371 feed values:
    # the 4th decimal is zero on 100.000% of them, while every other place on
    # every other column runs about 10% zeros, which is what a digit carrying
    # information looks like. The feed simply never resolves the strike zone finer
    # than a thousandth of a foot, so a 4th digit is decoration. Same rule that
    # sent LaunchAngle and Distance back to integers.
    'SzTop': 3, 'SzBot': 3,
    # degrees — convention, 3
    'VAA': 3, 'HAA': 3,
    'AttackAngle': 3, 'AttackDirection': 3, 'SwingPathTilt': 3,
    # source-quantized — measured
    'Velocity': 1, 'IndVertBrk': 1, 'HorzBrk': 1,
    'xIndVrtBrk': 1, 'xHorzBrk': 1,
    'ExitVelo': 1,
    'ArmAngle': 1, 'BatSpeed': 1, 'SwingLength': 1,
    'HC_X': 2, 'HC_Y': 2,
    'xBA': 3, 'xSLG': 3, 'xwOBA': 3, 'RunExp': 3,
    # LaunchAngle and Distance are INTEGERS, and stay integers. An earlier pass
    # here read them as carrying one decimal and gave them a 0.0 pattern. That
    # measurement was wrong: it counted the decimal places in repr(13.0), which
    # is 1, rather than asking whether the fraction is ever non-zero. Re-checked
    # 2026-08-17 over 6,183 batted balls in 120 cached games — the fractional
    # part is .0 on 100% of both fields, so a decimal adds a digit that carries
    # no information. ExitVelo is the genuine contrast and keeps its decimal:
    # only 9.8% of its values are integral and the other nine digits all appear.
    'LaunchAngle': 0, 'Distance': 0,
    'Spin Rate': 0, 'Outs': 0, 'Barrel': 0,
}

# Free-form strings. Compared verbatim, never coerced to a number.
STRING_COLS = {'Pitcher', 'Throws', 'Batter', 'Bats', 'Count', 'Description',
               'Event', 'BBType', 'Runners'}

# Sheets time values, rendered h:mm.
TIME_COLS = {'RTilt', 'OTilt'}

# The number format pinned on each column so the stored number and the rendered
# string agree. This is load-bearing, not cosmetic: both backfills decide
# "already correct" by comparing against get_all_values(), which returns the
# FORMATTED string. Under Sheets' Automatic format trailing zeros are stripped,
# so 94.0 reads back as "94" and every later run sees a mismatch and rewrites
# the cell forever.
NUMBER_FORMATS = {c: {'type': 'NUMBER', 'pattern': ('0.' + '0' * d) if d else '0'}
                  for c, d in PRECISION.items()}


def fmt(col, value):
    """Render a source value as the exact string that belongs in the sheet.

    Returns '' for a missing value. Never returns '0' for a NaN: a blank arm
    angle, a zero arm angle and a missing pitch type are three different
    failures, and none of them is 0.
    """
    if value is None or value is pd.NA:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass

    if col in STRING_COLS or col in TIME_COLS:
        return str(value).strip()

    d = PRECISION.get(col)
    if d is None:
        return str(value).strip()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ''
    s = f"{f:.{d}f}"

    # Negative zero. A small negative that rounds to nothing renders with a sign
    # that means nothing: a launch angle of -0.4 at 0 decimals becomes "-0", and an
    # HAA of -0.0004 at 3 decimals becomes "-0.000". Caught 2026-08-17 by Wally
    # asking what HOU's 18 remaining precision rewrites were — all 18 were this.
    #
    # It is two faults at once. "-0" is not a number anyone wants in a cell, and
    # Sheets stores it as 0, so the sweep read back "0", saw a difference, and
    # would have re-proposed those cells on every future run forever.
    #
    # A negative zero is not a negative number, so dropping the sign does not
    # collide with the display rule that negatives keep theirs.
    if s.startswith('-') and float(s) == 0.0:
        s = s[1:]
    return s


# Columns read as UNFORMATTED values rather than as the displayed string.
#
# Google will not let this account change the number format on eight of these
# columns. Verified 2026-08-17 on the HOU tab: repeatCell over the whole column,
# repeatCell with an explicit endRowIndex, gspread's ws.format() on three cells,
# and a deliberately different pattern were all accepted with HTTP 200 and empty
# replies, and none of them changed anything. There are no protected ranges, no
# banded ranges and no conditional formats on the tab, and a value write to the
# same cell lands fine.
#
# The values in the sheet ARE correct. Extension stores 6.6053 and displays 6.61.
# So the fix is to stop reading the display: get_all_values() returns the
# formatted string, and UNFORMATTED_VALUE returns the stored number.
#
# It has to be per column, not a blanket switch. Read unformatted, Game Date comes
# back as 46241 and the tilts come back as a fraction of a day — 1:03 becomes
# 0.04375. So the set is every column with a declared decimal depth, MINUS the
# tilts, which leaves dates, times and strings on the formatted read where they
# belong.
UNFORMATTED_COLS = set(PRECISION) - TIME_COLS


def merge_rendered(header, formatted_rows, unformatted_rows):
    """Merge a formatted and an unformatted read into one grid of strings.

    UNFORMATTED_COLS take their value from the unformatted read and are then put
    back through fmt(), so both sides of any later comparison are rendered by the
    same function and a display format cannot make a matching value look changed.
    Every other column keeps the formatted string.

    Row 0 is the header and is passed through untouched. It must be: running the
    header through fmt() turns every substituted column name into '', because
    float('Extension') raises, and the caller then cannot find its own columns.
    """
    idx = [(j, name) for j, name in enumerate(header)
           if name in UNFORMATTED_COLS]
    out = [list(formatted_rows[0])]
    for r in range(1, len(formatted_rows)):
        row = list(formatted_rows[r])
        urow = unformatted_rows[r] if r < len(unformatted_rows) else []
        for j, name in idx:
            if j >= len(row):
                continue
            raw = urow[j] if j < len(urow) else ''
            row[j] = '' if raw == '' or raw is None else fmt(name, raw)
        out.append(row)
    return out


def as_float(text):
    """Parse a sheet string to float, or None. Blank and junk both give None."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


TILT_MINUTES = 720          # a tilt clock is 12 hours, and it wraps


def tilt_minutes(text):
    """Parse an h:mm tilt to minutes past 12:00, or None.

    RTilt and OTilt are stored as clock readings, so they need a numeric view
    before any size-of-change question can be asked about them. 12:00 is 0 and
    the reading advances clockwise, so 1:19 is 79 and 11:37 is 697.
    """
    if text is None:
        return None
    s = str(text).strip()
    if ':' not in s:
        return None
    h, _, m = s.partition(':')
    try:
        hh, mm = int(h), int(m)
    except ValueError:
        return None
    if not (0 <= mm < 60) or not (1 <= hh <= 12):
        return None
    return ((hh % 12) * 60 + mm) % TILT_MINUTES


def tilt_gap(a, b):
    """Shortest distance between two tilt readings, in minutes.

    Must be circular. 11:59 and 12:01 are two minutes apart, not 718, and a
    linear subtraction would report every wrap across 12:00 as a huge change.
    """
    ma, mb = tilt_minutes(a), tilt_minutes(b)
    if ma is None or mb is None:
        return None
    d = abs(ma - mb) % TILT_MINUTES
    return min(d, TILT_MINUTES - d)


def stored_decimals(text):
    """Decimal places in a stored sheet string. None when it is not a number."""
    if text is None:
        return None
    s = str(text).strip()
    if not s or ':' in s:
        return None
    try:
        float(s)
    except ValueError:
        return None
    return len(s.split('.')[1]) if '.' in s else 0


# ── Domain validation for the per-event columns ──────────────────────────────
# These columns describe a single event, so no player median is a yardstick for
# them, and until 2026-08-17 that meant they were adopted with NOTHING checking
# them: 1,293 rows across the season set. Wally asked for a real test.
#
# What they DO have is a domain. Some are mathematically fixed — a code, an angle,
# a probability-like quantity — and those are asserted without inventing anything.
# The closed vocabularies are read off 583,619 real rows, and every one of them is
# a small enumerated set that a feed cannot legitimately extend mid-season.
#
# The soft-bounded quantities (ExitVelo, Distance, HC_X, HC_Y, BatSpeed,
# SwingLength, RunExp) have no fixed limit, so their envelopes are a CONVENTION:
# the observed range across the season widened by roughly a third, chosen only to
# catch a garbage value while never rejecting a real extreme. Observed ranges at
# the time of writing, for reference when one of these ever trips:
#   ExitVelo 4.0..119.0   Distance 0..474   HC_X 1.0..254.8   HC_Y 6.9..254.0
#   BatSpeed 50.0..88.0   SwingLength 3.7..15.7   RunExp -2.702..0.637
HARD_DOMAIN = {
    # code sets and counts — exact, not conventional
    'Barrel': (1, 6), 'Outs': (0, 2),
    # angles in degrees — geometry fixes these
    'LaunchAngle': (-90, 90), 'AttackAngle': (-90, 90),
    'AttackDirection': (-180, 180), 'SwingPathTilt': (0, 90),
    # expected-stat scales — a rate cannot leave its own scale
    'xBA': (0, 1), 'xSLG': (0, 4), 'xwOBA': (0, 2.1),
}
SOFT_DOMAIN = {
    'ExitVelo': (0, 130), 'Distance': (0, 600),
    'HC_X': (0, 300), 'HC_Y': (0, 300),
    'BatSpeed': (40, 110), 'SwingLength': (2, 20),
    'RunExp': (-3.5, 1.5),
}
VOCABULARY = {
    'BBType': {'bunt', 'fly_ball', 'ground_ball', 'line_drive', 'popup'},
    'Count': {f'{b}-{s}' for b in range(4) for s in range(3)},
    'Runners': {'0', '1', '2', '3', '1+2', '1+3', '2+3', '1+2+3'},
}


def domain_problem(col, value):
    """Why `value` cannot be right for `col`, or None if it is admissible.

    Only knows about the per-event columns. Returns None for anything it has no
    opinion on, so a column absent from these tables is unaffected.
    """
    if value in (None, ''):
        return None
    if col in VOCABULARY:
        if str(value).strip() not in VOCABULARY[col]:
            return (f'{value!r} is not one of the {len(VOCABULARY[col])} values '
                    f'{col} can take')
        return None
    lo_hi = HARD_DOMAIN.get(col)
    hard = lo_hi is not None
    if lo_hi is None:
        lo_hi = SOFT_DOMAIN.get(col)
    if lo_hi is None:
        return None
    v = as_float(value)
    if v is None:
        return f'{value!r} is not a number'
    lo, hi = lo_hi
    if not (lo <= v <= hi):
        return (f'{v} is outside the {"fixed" if hard else "plausible"} range '
                f'{lo} to {hi} for {col}')
    return None
