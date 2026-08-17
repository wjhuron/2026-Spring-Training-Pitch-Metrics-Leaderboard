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
    'SzTop': 4, 'SzBot': 4,
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
        return f"{float(value):.{d}f}"
    except (TypeError, ValueError):
        return ''


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
