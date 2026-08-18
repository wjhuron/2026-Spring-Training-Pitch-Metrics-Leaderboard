#!/usr/bin/env python3
"""HitterCards.py — Seasonal hitter cards.

Mirrors the visual grammar of Cards.py (pitcher cards) but reframed for
a hitter audience. Sections, top to bottom:

1. Pitch type color stripe (header band).
2. Identity zone (left): photo, name, "LHH | TEAM | Age", season label,
   headline stats strip
   (PA, AVG, OBP, SLG, BB%, K%, wRC+, SD+, CT+, BB+, Hitter+).
3. LA x Spray scatter (right): MLB wOBAcon zone heatmap, EV-colored dots,
   cyan Avg Placement marker, xwOBAsp annotation.
4. BIP donut + per-pitch-group composition bars (Hard / Breaking / Offspeed).
5. Per-pitch-group performance bars (vs RHP, vs LHP): bar length = usage,
   color = xwOBA performance.
6. Zone heat maps: 4-panel grid (Whiffs / Damage x vs RHP / vs LHP),
   Gaussian KDE density.
7. Contact Profile strip (7 cells): Bat Speed, Avg EV, Max EV,
   Squared-Up%, Blast%, IdealAtkAngle%, Air Pull%.
8. Hitter Performance Table:
   Pitch Group | Count | Usage | Swing% | Chase% | Whiff% | Hard-Hit%
   | Barrel% | xwOBAcon | xwOBAsp | RV/100 | xRV/100, plus Total row.
9. Huronalytics watermark.

Reuses helpers from Cards.py where the math is identical (xRV per pitch,
percentile cell coloring, headshot fetch).
"""

# Runnable as a file from any directory (IDE run buttons included):
# put the repo root on sys.path before the intra-repo package imports.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import json
import time as _time
import os
import pickle
import sys
from collections import defaultdict
from math import atan2, pi

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PICKLE_PATH = os.path.join(_REPO, 'data', 'all_pitches_rs_cache.pkl')
_HITTER_LB_PATH = os.path.join(_REPO, 'data', 'hitter_leaderboard_rs.json')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Reuse helpers from Cards.py. BG/ACCENT/DARK_CELL/DARKER imported but
# immediately overridden below — Cards.py defines them for the dark-theme
# pitcher cards, but the hitter card now defaults to a warm-paper theme.
from cards.pitcher import (
    PITCH_COLORS, PITCH_NAMES,
    sf, fmt_fi, fetch_headshot, badge_text_color,
    pct_cell_color as _default_pct_cell_color,
    raw_cell_color as _default_raw_cell_color,
    OUTPUT_DIR, METADATA_PATH, _load_guts,
    is_barrel, compute_iz, _compute_pitch_xrv,
    GUTS_LG_WOBA, GUTS_WOBA_SCALE,
    SWING_DESC, STRIKE_DESC,
)

# ─────────────────────────────────────────────────────────────────────
# WARM PAPER THEME — the canonical hitter-card look.
#
# Magazine-paper aesthetic: cream background, deep-terracotta accent,
# warm near-black text. Distinctive from the cool dark-dashboard genre
# that every other analytics site uses.
# ─────────────────────────────────────────────────────────────────────
BG          = '#f0e8d8'   # warm cream paper background
DARK_CELL   = '#e2d8c4'   # slightly darker cream for cells
DARKER      = '#d8ccb4'   # deepest tan for headers and Total row
ACCENT      = '#9f3026'   # deep terracotta red


# Module-level wrappers — theme wrappers (e.g. warm_paper) can monkey-patch
# these with their own tinting functions to use lower-saturation palettes
# on light themes. Default behavior delegates to Cards.py.
#
# Both wrappers accept an optional `max_alpha` (default None → use Cards.py
# default of 0.55). Set lower (e.g. 0.40) to dampen specific strips like
# Contact Profile / Pitch Group so they don't compete with the heat maps
# and LA × Spray for visual attention.
def pct_cell_color(value_str, league_avg, row_bg_hex,
                    higher_is_better=True, max_alpha=None):
    if max_alpha is None:
        return _default_pct_cell_color(value_str, league_avg, row_bg_hex,
                                         higher_is_better)
    if league_avg is None or not value_str or value_str == '—':
        return None
    try:
        val = float(value_str.replace('%', ''))
    except (ValueError, AttributeError):
        return None
    avg_pct = league_avg * 100
    diff = val - avg_pct
    if not higher_is_better:
        diff = -diff
    intensity = max(-1.0, min(1.0, diff / 8.0))
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    if intensity >= 0:
        target = (0, 180, 0)
    else:
        target = (180, 0, 0); intensity = abs(intensity)
    alpha = intensity * max_alpha
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


def raw_cell_color(value_str, league_avg, scale, higher_is_better,
                    row_bg_hex, max_alpha=None):
    if max_alpha is None:
        return _default_raw_cell_color(value_str, league_avg, scale,
                                         higher_is_better, row_bg_hex)
    if league_avg is None or not value_str or value_str == '—':
        return None
    # Re-use Cards.py's _parse_fi via the default function would be nice but
    # it's a private helper; reimplement value parsing inline.
    try:
        val = float(value_str)
    except (ValueError, AttributeError):
        return None
    diff = val - league_avg
    if not higher_is_better:
        diff = -diff
    intensity = max(-1.0, min(1.0, diff / scale))
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    if intensity >= 0:
        target = (0, 180, 0)
    else:
        target = (180, 0, 0); intensity = abs(intensity)
    alpha = intensity * max_alpha
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


# Avg Placement marker color — deep purple. Cyan washes on cream paper;
# deep purple contrasts strongly without competing with the warm/red
# palette. Theme wrappers override for dark themes (cyan reads better).
MARKER_ACCENT = '#5d3b8e'

# ─────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────
FIG_W = 22
FIG_H = 18
DPI = 100
SAVE_DPI = 150
PLATE_HALF = 17 / 12 / 2  # half-plate width in feet

# Pitch grouping (Hard / Breaking / Offspeed). KN included in Offspeed.
PITCH_GROUPS = {
    'Hard':     ['FF', 'SI'],
    'Breaking': ['FC', 'SL', 'ST', 'CU', 'SV'],
    'Offspeed': ['CH', 'FS', 'KN'],
}
GROUP_ORDER = ['Hard', 'Breaking', 'Offspeed']
GROUP_COLORS = {
    'Hard':     '#3b82f6',  # blue
    'Breaking': '#a78bfa',  # purple
    'Offspeed': '#f59e0b',  # orange
}

# Bat-ball type constants (mirrors Cards.py)
BB_COLORS = {
    'ground_ball': '#06b6d4',
    'line_drive':  '#f97316',
    'fly_ball':    '#a78bfa',
    'popup':       '#fbbf24',
}
BB_LABELS = {
    'ground_ball': 'Ground Ball',
    'line_drive':  'Line Drive',
    'fly_ball':    'Fly Ball',
    'popup':       'Popup',
}
BB_TYPES = ['ground_ball', 'line_drive', 'fly_ball', 'popup']

# Headline stats coloring rule:
#   key  -> (hitter-row metric key, type, higher_is_better, scale)
# 'pct' = stored as decimal (.234), displayed as 23.4%; pct_cell_color picks scale.
# 'raw' = displayed as raw number; scale below sets ±intensity.
HITTER_STAT_LINE_COLOR = {
    'AVG':     ('avg',        'raw', True,  0.05),
    'OBP':     ('obp',        'raw', True,  0.05),
    'SLG':     ('slg',        'raw', True,  0.10),
    'OPS':     ('ops',        'raw', True,  0.15),   # OBP+SLG combined range
    'wOBA':    ('wOBA',       'raw', True,  0.05),
    'wRC+':    ('wRCplus',    'raw', True,  20),
    # Legacy entries — no longer in the canonical headline strip but kept
    # in case a theme wrapper or alternative layout references them.
    'BB%':     ('bbPct',      'pct', True),
    'K%':      ('kPct',       'pct', False),
    # Tint distances track each metric's SHIPPED spread (run-truth rescale,
    # 2026-08-15): SD+/CT+ are pinned at pool SD 10 (skill z-ruler), BB+ at
    # 0.66 x wRC+'s SD (~14), Hitter+ at 0.82 x (~17) — full colour lands at
    # ~2 SD for each, matching the wRC+ cell's visual intensity per SD.
    'SD+':     ('sdPlus',     'raw', True,  10),
    'CT+':     ('ctPlus',     'raw', True,  10),
    'BB+':     ('bbPlus',     'raw', True,  14),
    'Hitter+': ('hitterPlus', 'raw', True,  17),
}

# Headline stat order — MLB default. ROC overrides this since the + family
# and bat-tracking metrics aren't computed for AAA.
# Headline strip — non-redundant with the percentile bubble panel below.
# Bubbles already carry BB%, K%, SD+, CT+, BB+, Hitter+ and the QoC family,
# so the strip focuses on the slash line and the bottom-line production
# stats (wOBA = true linear-weighted runs, wRC+ = park/league-adjusted).
HEADLINE_STATS_MLB = ['PA', 'HR', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+']
HEADLINE_STATS_ROC = ['PA', 'HR', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+']
HEADLINE_STATS = HEADLINE_STATS_MLB  # default; render switches based on team

# ─────────────────────────────────────────────────────────────────────
# Theme variables — warm-paper defaults. (The light-gray and vintage
# theme wrappers that used to override these were deleted 2026-08-15;
# a new theme can still monkey-patch these globals before rendering.)
# ─────────────────────────────────────────────────────────────────────
TEXT_PRIMARY       = '#1a1612'      # warm near-black (name, headline values)
TEXT_SECONDARY     = '#3a3530'      # deep warm gray (section titles, headers)
TEXT_MUTED         = '#6a5f55'      # mid warm gray (annotations, axes)
TEXT_FAINT         = '#8a7f75'      # light warm gray (bipnote, fine print)
TEXT_DIMMED        = '#9f9890'      # very light warm gray (legend captions)
SUBTLE_BORDER      = '#c5b89f'      # light tan border (cell edges)
TOTAL_BORDER       = '#6a5f55'      # mid-dark for Total row distinction
ALT_ROW_BG         = '#e8dfcb'      # slightly off DARK_CELL (alt rows)
GRID_COLOR         = '#3a3530'      # dark grid on cream bg
TICK_COLOR         = '#6a5f55'      # spray-chart axis ticks
SPINE_COLOR        = '#8a7f75'      # axis spine
PHOTO_BORDER       = '#6a5f55'      # photo edge
PERCENTILE_NEUTRAL = '#8a7f75'      # fallback when xwOBAsp_pctl is None

# Heat-map colormap — saturated diverging blue → red with NO white middle.
# Skipping the white midpoint means neutral data still shows color (light
# blue / soft red) instead of fading to bg. Heat maps stay bold across
# their entire data region — extremes pop AND neutral zones stay visible.
HEAT_BLUE_DARK    = '#08306b'   # cold extreme
HEAT_BLUE_MID     = '#3a6fa8'   # mid blue
HEAT_BLUE_LIGHT   = '#7eaad6'   # light blue (no white)
HEAT_RED_LIGHT    = '#f0a89c'   # soft red (no white)
HEAT_RED_MID      = '#c43d2e'   # mid red
HEAT_RED_DARK     = '#67000d'   # hot extreme


def make_heat_cmap(_bg_hex_unused=None):
    """Build a saturated blue → red diverging colormap with no white
    transition zone. Neutral data renders as light blue / soft red
    instead of fading away."""
    return LinearSegmentedColormap.from_list(
        'heatmap',
        [(0.00, HEAT_BLUE_DARK),
         (0.20, HEAT_BLUE_MID),
         (0.45, HEAT_BLUE_LIGHT),
         (0.55, HEAT_RED_LIGHT),
         (0.80, HEAT_RED_MID),
         (1.00, HEAT_RED_DARK)],
        N=256,
    )


HEAT_CMAP = make_heat_cmap()

# Zone-wOBAcon colormap (LA x Spray underlay) — EXACT match to hitter page
# js/player-page.js sacqZones plugin's wobaColorRGB function:
#   t=0.00  -> rgb(  8, 48,107)
#   t=0.35  -> rgb( 38,148,147)
#   t=0.55  -> rgb(238,228, 47)
#   t=1.00  -> rgb(215, 48, 39)
WOBA_CMAP = LinearSegmentedColormap.from_list(
    'woba',
    [(0.00, (8/255,   48/255, 107/255)),
     (0.35, (38/255, 148/255, 147/255)),
     (0.55, (238/255, 228/255,  47/255)),
     (1.00, (215/255,  48/255,  39/255))],
    N=256,
)


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
# The release asset is AES-256 ciphertext (2026-07-03: the retagged pitch
# data is no longer published in plaintext). STUFF_DATA_KEY from .env
# decrypts after download.
RELEASE_PICKLE_URL = (
    'https://github.com/wjhuron/Huronalytics/releases/download/'
    'latest-data/all_pitches_rs_cache.pkl.enc'
)


def _stuff_data_key():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    try:
        for line in open(env_path):
            if line.startswith('STUFF_DATA_KEY='):
                return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return os.environ.get('STUFF_DATA_KEY')


def fetch_pickle_from_release(out_path, verbose=True):
    """Download the latest pickle from the GitHub Release that the CI
    pipeline pushes to on every successful run. Streams in 4MB chunks so
    we don't blow memory on a ~100MB download. Returns True on success."""
    import urllib.request
    if verbose:
        print(f"  Fetching latest pickle from GitHub Release…")
    req = urllib.request.Request(
        RELEASE_PICKLE_URL,
        headers={'User-Agent': 'HitterCards/auto-refresh'},
    )
    tmp_path = out_path + '.tmp'
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            written = 0
            with open(tmp_path, 'wb') as f:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if verbose and total:
                        pct = 100.0 * written / total
                        print(f"\r    downloaded {written // (1024 * 1024)} / "
                              f"{total // (1024 * 1024)} MB  ({pct:.0f}%)", end='')
            if verbose and total:
                print()
        key = _stuff_data_key()
        if not key:
            raise RuntimeError('STUFF_DATA_KEY missing from .env — cannot decrypt pickle')
        import subprocess
        dec_path = tmp_path + '.dec'
        subprocess.run(['openssl', 'enc', '-d', '-aes-256-cbc', '-pbkdf2',
                        '-pass', 'env:STUFF_DATA_KEY', '-in', tmp_path,
                        '-out', dec_path],
                       check=True, env={**os.environ, 'STUFF_DATA_KEY': key})
        os.remove(tmp_path)
        os.replace(dec_path, out_path)
        return True
    except Exception as e:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        if verbose:
            print(f"  Auto-refresh failed: {type(e).__name__}: {e}")
        return False


def load_pitch_data(path=_PICKLE_PATH,
                     auto_refresh=True):
    """Load the pitch-level pickle. When auto_refresh is True (default),
    the pickle is automatically refreshed from the latest GitHub Release
    if it's stale or missing. This makes the local pickle effectively
    "never stuck" — the CI run uploads a fresh pickle every day, and this
    function pulls it as needed."""
    needs_refresh = False
    refresh_reason = None
    if not os.path.exists(path):
        needs_refresh = True
        refresh_reason = 'missing'
    else:
        try:
            is_stale, pickle_dt, json_dt = check_pickle_freshness(path)
            if is_stale:
                needs_refresh = True
                refresh_reason = f'stale ({pickle_dt} → {json_dt})'
        except Exception:
            pass  # If check fails, just load whatever we have

    if needs_refresh and auto_refresh:
        print(f"  Pickle {refresh_reason}; downloading latest from release…")
        ok = fetch_pickle_from_release(path)
        if ok:
            print(f"  ✓ Pickle refreshed.")
        else:
            print(f"  ✗ Auto-refresh failed; using whatever's on disk.")
            print(f"     Manual fix: python3 refresh_pickle.py (Sheets→pickle)")
            print(f"                  python3 process_data.py    (full pipeline)")

    with open(path, 'rb') as f:
        pitches = pickle.load(f)
    _apply_runexp_currency(pitches)
    return pitches


def _apply_runexp_currency(pitches):
    """Rescale MiLB RunExp into MLB run currency, in place.

    The pickle stores RunExp exactly as Statcast served it, and Statcast
    builds delta_run_exp on each league's own run-expectancy matrix — so
    ROC/AAA values are ~1.25x MLB for the identical event. process_data
    corrects this in memory for the site and never writes it back, so any
    script reading the pickle inherits the raw MiLB values; here that fed
    both PitchRV/100 and (via _compute_pitch_xrv's non-BIP fallthrough)
    xPitchRV/100 on every ROC hitter card.

    Done once at load, before any consumer, mirroring process_data. Level
    comes from the pitch's own `_source`, which the pickle carries, so no
    inference is needed (the pitcher-card path reads worksheets instead,
    which have no _source, and has to detect level from bat tracking).
    """
    try:
        from pipeline.utils import compute_runexp_scale, runexp_factor
    except ImportError:
        return
    scale = compute_runexp_scale(pitches)
    if not scale:
        return
    n_fixed = 0
    for p in pitches:
        sc = scale.get(p.get('_source'))
        if not sc:
            continue
        v = sf(p.get('RunExp'))
        if v is None:
            continue
        f = runexp_factor(sc, p.get('Description'), p.get('Count'))
        if f:
            p['RunExp'] = v / f
            n_fixed += 1
    # Only report a real rescale. The CI-released pickle is written AFTER
    # process_data's in-place correction, so its factors measure ~1.000 and
    # this is a no-op; a locally refreshed pickle (refresh_pickle.py goes
    # Sheets -> pickle with no correction) is the case that actually moves.
    if n_fixed and any(abs(d['global'] - 1.0) > 0.005 for d in scale.values()):
        print(f"  RunExp -> MLB currency: {n_fixed} MiLB pitches rescaled "
              + ", ".join(f"{s} /{d['global']:.3f}" for s, d in sorted(scale.items())))


def load_hitter_leaderboard(path=_HITTER_LB_PATH):
    with open(path) as f:
        return json.load(f)


def check_pickle_freshness(pickle_path=_PICKLE_PATH,
                            leaderboard_path=_HITTER_LB_PATH):
    """Compare pickle mtime against the leaderboard JSON's most-recent
    lastGameDate. Returns (is_stale, pickle_latest_date_str, json_latest_date_str).

    The pickle holds pitch-level data (gitignored, doesn't propagate from
    CI); the JSON holds aggregated stats (git-tracked, always fresh). When
    the JSON's most recent game date is newer than the latest date present
    in the pickle, the card will plot a stale snapshot. This function
    surfaces that drift.
    """
    import datetime
    # Get the most recent Game Date present in the pickle
    try:
        with open(pickle_path, 'rb') as f:
            pitches = pickle.load(f)
    except Exception:
        return True, None, None
    dates = [p.get('Game Date') for p in pitches if p.get('Game Date')]
    pickle_latest = max(dates) if dates else None

    # Get the most recent lastGameDate in the JSON across all hitters
    try:
        with open(leaderboard_path) as f:
            lb = json.load(f)
    except Exception:
        return False, pickle_latest, None
    json_dates = [r.get('lastGameDate') for r in lb if r.get('lastGameDate')]
    json_latest = max(json_dates) if json_dates else None

    is_stale = bool(json_latest and pickle_latest and json_latest > pickle_latest)
    return is_stale, pickle_latest, json_latest


def load_metadata(path=METADATA_PATH):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Spray angle / SACQ zones
# ─────────────────────────────────────────────────────────────────────
HP_X, HP_Y = 125.42, 198.27  # home-plate origin in HC space (matches website)


def spray_angle(hc_x, hc_y):
    if hc_x is None or hc_y is None:
        return None
    dx = hc_x - HP_X
    dy = HP_Y - hc_y
    if dy <= 0:
        return None
    return atan2(dx, dy) * 180 / pi


def spray_direction(angle, bats):
    if angle is None or not bats:
        return None
    if bats == 'R':
        if angle < -30: return 'pull'
        if angle < -15: return 'pull_side'
        if angle < 0:   return 'center_pull'
        if angle < 15:  return 'center_oppo'
        if angle < 30:  return 'oppo_side'
        return 'oppo'
    else:
        if angle > 30:  return 'pull'
        if angle > 15:  return 'pull_side'
        if angle > 0:   return 'center_pull'
        if angle > -15: return 'center_oppo'
        if angle > -30: return 'oppo_side'
        return 'oppo'


# Negative-LA region split at -10 (mirrors process_data.py LA_BINS):
# -10..0 (~.247 wOBAcon) is materially more productive than below -10
# (~.135), so they get distinct SACQ zones.
LA_BINS = [(-9999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
           (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 9999)]


def la_bin_idx(la):
    for i, (lo, hi) in enumerate(LA_BINS):
        if lo <= la < hi:
            return i
    return None


def build_sacq_lookup(metadata, bats):
    """Returns a lookup function (spray_dir, la_bin) -> wOBAcon, hand-specific
    with pooled fallback. Mirrors the website's sacqLookup."""
    zones = metadata.get('sacqZones', [])
    hand_map, pool_map = {}, {}
    for z in zones:
        key = (z['spray'], z['laBin'])
        if z.get('bats') == bats:
            hand_map[key] = z
        elif z.get('bats') is None:
            pool_map[key] = z
    MIN_BIP = 20

    def lookup(direction, lb):
        if direction is None or lb is None: return None
        z = hand_map.get((direction, lb))
        if z and z.get('count', 0) >= MIN_BIP:
            v = z.get('wobacon')
            if v is None: v = z.get('woba')
            if v is not None: return v
        z = pool_map.get((direction, lb))
        if z and z.get('count', 0) >= MIN_BIP:
            v = z.get('wobacon')
            if v is None: v = z.get('woba')
            if v is not None: return v
        return None

    return lookup, hand_map, pool_map


# ─────────────────────────────────────────────────────────────────────
# Per-pitch stat computation helpers
# ─────────────────────────────────────────────────────────────────────
def compute_group_stats(group_pitches, sacq_lookups, bats):
    """Compute the per-pitch-group row for the bottom table.
    Returns dict with: count, usagePct (caller fills in), swingPct, chasePct,
    whiffPct, hardHitPct, barrelPct, xwOBAcon, xwOBAsp, rv100, xRv100.

    sacq_lookups is a dict of per-hand lookup fns ({'L':..., 'R':...}); each BIP
    is scored by its own Bats so switch hitters are not mirror-flipped."""
    n = len(group_pitches)
    if n == 0:
        return None

    def _sacq_for(p):
        pb = p.get('Bats') or bats
        return pb, (sacq_lookups.get(pb) or sacq_lookups.get(bats)
                    or next(iter(sacq_lookups.values())))
    # Bunts are not swings and not chases (2026-08-15) — one predicate,
    # shared with the leaderboard (pipeline.utils.is_swing), so Swing%,
    # Chase% and Whiff% here cannot drift from the columns they mirror.
    # Bunt-attempt whiffs are indistinguishable from regular whiffs (no
    # BBType on any swinging strike), so they stay in the numerator.
    from pipeline.utils import is_swing as _is_swing
    swings = [p for p in group_pitches if _is_swing(p)]
    whiffs = [p for p in group_pitches if p.get('Description') == 'Swinging Strike']
    swings_non_bunt = swings
    iz = [p for p in group_pitches if compute_iz(p) is True]
    ooz_swings = [p for p in swings if compute_iz(p) is False]
    ooz_total = sum(1 for p in group_pitches if compute_iz(p) is False)

    # Contact-quality rates use BIP (excluding bunts) with valid EV as denom
    bip = [p for p in group_pitches
           if p.get('Description') == 'In Play'
           and p.get('BBType') and not str(p.get('BBType', '')).startswith('bunt')]
    evs = [(sf(p.get('ExitVelo')), p) for p in bip]
    evs_valid = [(v, p) for v, p in evs if v is not None]
    n_ev = len(evs_valid)
    n_hh = sum(1 for v, _ in evs_valid if v >= 95.0)
    # Barrels — prefer Statcast `Barrel` column code '6' (matches the canonical
    # leaderboard); fall back to is_barrel(ev, la) heuristic only when the
    # column is empty for the whole group. See pipeline_compute.py ~L329-338.
    has_barrel_col = any(str(p.get('Barrel', '')).strip() != '' for p in bip)
    if has_barrel_col:
        n_brl = sum(1 for p in bip if str(p.get('Barrel', '')).strip() == '6')
    else:
        n_brl = sum(1 for v, p in evs_valid if is_barrel(v, sf(p.get('LaunchAngle'))))

    # Avg EV / Max EV — population-wide mean / max ExitVelo for the group.
    # Used directly for ROC (where xwOBAcon is null) and as cross-check
    # context for MLB.
    avg_ev = sum(v for v, _p in evs_valid) / n_ev if n_ev else None
    max_ev = max((v for v, _p in evs_valid), default=None)

    # Air Pull% — pulled line drives + fly balls / total BIPs.
    # Matches pipeline_compute.py exactly (canonical leaderboard formula):
    #   air_pull = (pull|pull_side spray) AND (line_drive|fly_ball BBType)
    #   airPullPct = air_pull / n_bip
    n_airpull = 0
    for p in bip:
        bb_type = p.get('BBType')
        ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
        pb, _lk = _sacq_for(p)
        sd = spray_direction(ang, pb)
        if sd is None:
            continue
        if sd in ('pull', 'pull_side') and bb_type in ('line_drive', 'fly_ball'):
            n_airpull += 1
    air_pull_pct = (n_airpull / len(bip)) if bip else None

    # xwOBAcon: average xwOBA on BIP with non-null xwOBA
    bip_xw = [sf(p.get('xwOBA')) for p in bip]
    bip_xw = [v for v in bip_xw if v is not None]
    xwobacon = sum(bip_xw) / len(bip_xw) if bip_xw else None

    # xwOBAsp: average SACQ-zone wOBA across this group's BIPs.
    xw_sp_vals = []
    for p in bip:
        la = sf(p.get('LaunchAngle'))
        ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
        pb, lk = _sacq_for(p)
        sd = spray_direction(ang, pb)
        lb = la_bin_idx(la) if la is not None else None
        v = lk(sd, lb)
        if v is not None:
            xw_sp_vals.append(v)
    xwobasp = sum(xw_sp_vals) / len(xw_sp_vals) if xw_sp_vals else None

    # Run values: PitchRV/100 (actual outcomes) + xPitchRV/100 (xwOBA model on BIP)
    actual_rvs = [sf(p.get('RunExp')) for p in group_pitches]
    actual_rvs = [v for v in actual_rvs if v is not None]
    rv100 = -round(sum(actual_rvs) / n * 100, 1) if actual_rvs and n else None
    # Note: RunExp is from pitcher perspective in the sheet. Negate for hitter.
    x_rvs = _compute_pitch_xrv(group_pitches)
    # xRV from _compute_pitch_xrv is pitcher-perspective (positive = good for
    # pitcher). Flip sign so hitter card displays hitter-positive values.
    xRv100 = -round(sum(x_rvs) / n * 100, 1) if x_rvs and n else None

    return {
        'count': n,
        'swingPct': len(swings) / n if n else None,
        'chasePct': len(ooz_swings) / ooz_total if ooz_total else None,
        'whiffPct': len(whiffs) / len(swings_non_bunt) if swings_non_bunt else None,
        'hardHitPct': n_hh / n_ev if n_ev else None,
        'barrelPct': n_brl / n_ev if n_ev else None,
        'xwOBAcon': xwobacon,
        'xwOBAsp':  xwobasp,
        'rv100':    rv100,
        'xRv100':   xRv100,
        'avgEV':       avg_ev,
        'maxEV':       max_ev,
        'airPullPct':  air_pull_pct,
        # Carry forward for downstream charts
        '_bip':     bip,
        '_swings':  swings,
        '_whiffs':  whiffs,
    }


# ─────────────────────────────────────────────────────────────────────
# Hand-rolled Gaussian KDE for zone heat maps (no scipy)
# ─────────────────────────────────────────────────────────────────────
def kde_grid(points, weights, x_min, x_max, z_min, z_max,
             n_grid=120, bandwidth=0.15, normalize=True):
    """Return a (n_grid, n_grid) density array.
    points: list of (x, z) tuples. weights: list of floats or None for uniform.
    normalize: if True, scale grid max to 1.0 (per-panel relative density).
    Set normalize=False to keep raw weighted-sum values for ratio math."""
    if not points:
        return None
    bw2 = 2 * bandwidth ** 2
    pts_x = np.array([p[0] for p in points])
    pts_z = np.array([p[1] for p in points])
    pts_w = np.array(weights) if weights is not None else np.ones(len(points))
    xs = np.linspace(x_min, x_max, n_grid)
    zs = np.linspace(z_min, z_max, n_grid)
    grid = np.zeros((n_grid, n_grid))
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            dx = x - pts_x
            dz = z - pts_z
            grid[iz, ix] = float(np.sum(pts_w * np.exp(-(dx * dx + dz * dz) / bw2)))
    if normalize and grid.max() > 0:
        grid /= grid.max()
    return grid


def render_rate_kde_to_axes(ax, num_points, num_weights,
                             den_points, den_weights,
                             x_min, x_max, z_min, z_max,
                             sz_top, sz_bot, label, sample_label,
                             scale_max=1.0, scale_min=0.0,
                             mask_threshold=0.05, min_n=5):
    """Render a SMOOTHED RATE field via two KDEs.
    rate(x, z) = K_num(x, z) / K_den(x, z)

    For DAMAGE: num=BIPs weighted by xwOBA, den=BIPs unweighted, scale_max=1.0
                → mean xwOBA per location (absolute scale anchored at MLB avg
                via WOBA_CMAP — weak hitters render mostly cool, strong hitters red)
    For WHIFFS: num=whiffs (weight=1), den=swings (weight=1), scale_max=0.6
                → whiff rate per location (cool below MLB ~25%, red above ~36%)

    Alpha is gated by the denominator KDE so regions with too few events fade
    out — the panel only "speaks" where there's enough signal.
    """
    # Transparent panel — lets the figure background (with subtle radial
    # gradient) show through. Avoids visible rectangle frames where the
    # panel's flat color doesn't perfectly match the gradient.
    ax.set_facecolor('none')
    if num_points and den_points and len(num_points) >= min_n:
        # Compute both KDEs raw (unnormalized so the ratio is meaningful)
        num_grid = kde_grid(num_points, num_weights,
                             x_min, x_max, z_min, z_max, normalize=False)
        den_grid = kde_grid(den_points, den_weights,
                             x_min, x_max, z_min, z_max, normalize=False)
        # Ratio = smoothed local rate (mean weight per event in that area)
        # Guard division-by-zero — set rate to 0 where denominator is essentially nil
        eps = 1e-9
        rate = np.where(den_grid > eps, num_grid / den_grid, 0.0)
        # Map rate → colormap input [0, 1] using absolute scale_max
        # WOBA_CMAP is anchored: 0=blue, 0.35=teal, 0.55=yellow, 1.0=red.
        # For DAMAGE (scale_max=1.0): rate=.37 (MLB) → 0.37 input → teal/yellow boundary
        # For WHIFFS (scale_max=0.6): rate=.252 (MLB) → 0.42 → teal-yellow boundary
        # Use HEAT_CMAP (navy → light blue → white → red) — matches the
        # website's swing-heat-map color scheme. With absolute scaling, weak
        # hitters' bins stay in the cool blue range (no white/red).
        # Normalize: (rate - scale_min) / (scale_max - scale_min). For most
        # metrics scale_min=0; for ROC's Avg-EV damage panel, scale_min=50
        # so the cmap input maps EV ∈ [50, 110] → [0, 1].
        scale_range = max(scale_max - scale_min, 1e-9)
        normalized_rate = np.clip((rate - scale_min) / scale_range, 0.0, 1.0)
        rgba = HEAT_CMAP(normalized_rate)
        # Alpha mask — fade regions where denominator is below threshold (low data)
        den_peak = den_grid.max() if den_grid.max() > 0 else 1.0
        den_rel = den_grid / den_peak  # [0, 1]
        # Linear ramp above threshold, smoothed with gamma
        alpha = np.clip((den_rel - mask_threshold) / (1.0 - mask_threshold), 0, 1)
        # Gamma 0.15 = very aggressive ramp — colors hit full saturation
        # quickly so almost the entire data region shows bold color
        # instead of feathering out toward bg.
        alpha = np.power(alpha, 0.15)
        rgba[..., 3] = alpha
        # Smooth bilinear interpolation with tight bandwidth (0.18) and high
        # resolution (120×120 grid). This produces the TruMedia "cloud of
        # detail" look — many small hot/cold atoms with feathered edges
        # rather than one big wavy blob.
        ax.imshow(rgba, origin='lower', extent=[x_min, x_max, z_min, z_max],
                  aspect='auto', interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, 'n = ' + str(len(num_points or [])) + ' (insufficient)',
                transform=ax.transAxes, color=TEXT_DIMMED, ha='center',
                fontstyle='italic', fontfamily='IBM Plex Sans')
    # Strike-zone overlay — alpha 0.85 (was 0.95) softens the outline
    # slightly so it doesn't read as harsh against the saturated heat blobs.
    ax.add_patch(Rectangle((-PLATE_HALF, sz_bot), PLATE_HALF * 2, sz_top - sz_bot,
                            fill=False, edgecolor=GRID_COLOR, linewidth=1.8,
                            alpha=0.85, zorder=10))
    ax.set_xlim(x_min, x_max); ax.set_ylim(z_min, z_max)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # Title — bigger, brighter for visibility against saturated panels
    ax.set_title(label, color=TEXT_PRIMARY, fontsize=13, fontweight='700',
                 fontfamily='IBM Plex Sans', pad=8)
    # Sample-size annotation — backdrop alpha 0.65 (was 0.85) is more
    # subtle, less visible-pill feel.
    ax.text(0.02, 0.97, sample_label, transform=ax.transAxes,
            color=TEXT_MUTED, fontsize=11, fontweight='600',
            fontfamily='IBM Plex Sans', va='top', ha='left',
            bbox=dict(facecolor=BG, edgecolor='none', alpha=0.65,
                       boxstyle='round,pad=0.25'),
            zorder=11)


# ─────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────
def fmt_3dec(v):
    if v is None: return '—'
    s = f"{v:.3f}"
    if s.startswith('0.'): s = s[1:]
    if s.startswith('-0.'): s = '-' + s[2:]
    return s


def fmt_pct(v, decimals=1):
    if v is None: return '—'
    return f"{v * 100:.{decimals}f}%"


def fmt_int(v):
    if v is None: return '—'
    return str(int(round(v)))


def fmt_signed_decimal(v, decimals=1):
    if v is None: return '—'
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────
# Percentile bubble grid (layout='bubbles' variant)
# ─────────────────────────────────────────────────────────────────────
#
# Replaces heat maps + contact profile + pitch group table with a 4-column
# percentile grid. Each cell shows:
#     [label]                                          [value]
#     [██████████████████████████████████]  [○ pctl]
# colored on a blue→red gradient matching the player page (low = blue/bad,
# high = red/good — the percentile rank is already directionally normalized
# for inverted stats like K%, Whiff%, Chase%).
#
# Column spec format: (display_label, value_key, pctl_key, format_spec)
# format_spec: one of '3dec' (.425), 'pct1' (62.0%), 'int' (160),
#              'dec1' (96.3), 'dec1+' (signed, e.g. +8.7), 'mph' (96.3 mph)

# 2026-08-12 restructure (per Wally): RESULT leads with the "+" family so the
# Hitter+ decomposition reads top-down (composite, then its three components),
# with Bat Speed folded in as the physical input. The standalone BAT TRACKING
# section is gone — Squared-Up% and Blast% dropped, Bat Speed promoted here.
# Directionality is handled by the stored percentiles, which are already
# inverted for Swing% and GB% (HITTER_INVERT_PCTL in pipeline_compute.py), so
# "lower is better" needs no special casing at render time.
BUBBLE_COLUMNS = [
    ('RESULT', [
        ('Hitter+',           'hitterPlus',   'hitterPlus_pctl',   'int'),
        ('Batted Ball+',      'bbPlus',       'bbPlus_pctl',       'int'),
        ('Contact+',          'ctPlus',       'ctPlus_pctl',       'int'),
        ('Swing Decisions+',  'sdPlus',       'sdPlus_pctl',       'int'),
        ('Bat Speed',         'batSpeed',     'batSpeed_pctl',     'mph'),
        ('xwOBA',             'xwOBA',        'xwOBA_pctl',        '3dec'),
    ]),
    ('QUALITY OF CONTACT', [
        ('xwOBAcon',    'xwOBAcon',     'xwOBAcon_pctl',     '3dec'),
        # BABIP directly under xwOBAcon (2026-08-03, per Wally): the pair
        # reads realized-vs-expected on contact — mismatched tints flag
        # batted-ball fortune. Hitter side is uninverted: high BABIP = red.
        ('BABIP',       'babip',        'babip_pctl',        '3dec'),
        ('Max EV',      'maxEV',        'maxEV_pctl',        'mph'),
        ('Hard-Hit%',   'hardHitPct',   'hardHitPct_pctl',   'pct1'),
        ('Barrel%',     'barrelPct',    'barrelPct_pctl',    'pct1'),
        ('Air Pull%',   'airPullPct',   'airPullPct_pctl',   'pct1'),
        ('GB%',         'gbPct',        'gbPct_pctl',        'pct1'),
    ]),
    ('PLATE DISCIPLINE', [
        # Outcomes first, then the swing itself in the order it happens:
        # DECIDE (which pitches he offers at, then the differential those two
        # rates feed) and then EXECUTE (whether the swing connected).
        ('BB%',         'bbPct',        'bbPct_pctl',        'pct1'),
        ('K%',          'kPct',         'kPct_pctl',         'pct1'),
        # -- decision --
        # Z-Swing% replaced Swing% on 2026-08-18 per Wally. Overall Swing% is
        # directionally ambiguous (more swings only helps if they are aimed
        # well), which is why it needed its own "higher is not better" note on
        # the card. Restricting the denominator to strikes removes the
        # ambiguity: swinging at a strike is the good outcome, so the row
        # reads higher = better and colors that way with no caveat. izSwingPct
        # is NOT in HITTER_INVERT_PCTL, so the shipped percentile already runs
        # in that direction. Same label as the site column (js/leaderboard.js).
        ('Z-Swing%',    'izSwingPct',   'izSwingPct_pctl',   'pct1'),
        ('Chase%',      'chasePct',     'chasePct_pctl',     'pct1'),
        # The differential sits directly under the two rates it subtracts, so
        # a reader can see both inputs and the result without hunting.
        # Same label as the site column. Short enough that 'Swing Decisions+'
        # stays the longest label, so it does not eat into the bar width the
        # way the fully spelled-out version did.
        ('Z-Sw% - Chase%', 'izSwChase', 'izSwChase_pctl', 'pct1'),
        # -- execution --
        # Whiff% restored 2026-08-12 per Wally. Statistically it is close to a
        # triplicate of things already here (r = -0.92 with Z-Contact%, -0.92
        # with Contact+, +0.91 with K%), which is why the 2026-07-20 prune
        # dropped it. It earns the row on readability rather than on new
        # information: swing-and-miss is the number a scout asks for by name.
        ('Whiff%',      'whiffPct',     'whiffPct_pctl',     'pct1'),
        # Whiff% over every swing, then the same idea over in-zone swings
        # only: the pair reads "misses" then "misses on the pitches he should
        # not miss".
        ('Z-Contact%',  'izContactPct', 'izContactPct_pctl', 'pct1'),
    ]),
]


def _measure_text_axis_w(fig, strings, fontsize, weight, family='IBM Plex Sans'):
    """Width of the widest string, as a fraction of figure width.

    Measured off the real renderer rather than guessed as a fraction of the
    column, so the label and value gutters are sized to the text actually
    drawn and the pill bar can claim everything left over. Returns None if the
    backend can't measure yet, in which case callers keep their fixed
    fallback fractions.
    """
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        return None
    widest = 0.0
    for s in strings:
        if not s:
            continue
        t = fig.text(0, 0, s, fontsize=fontsize, fontfamily=family,
                     fontweight=weight)
        try:
            widest = max(widest, t.get_window_extent(renderer=renderer).width)
        except Exception:
            t.remove()
            return None
        t.remove()
    fig_px = fig.get_size_inches()[0] * fig.dpi
    return (widest / fig_px) if fig_px else None


def _format_bubble_value(v, spec):
    if v is None:
        return '—'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    if spec == '3dec':
        s = f'{v:.3f}'
        # Match site convention: no leading 0 on rate stats like .425
        return s[1:] if s.startswith('0.') else (f'-{s[2:]}' if s.startswith('-0.') else s)
    if spec == 'pct1':
        return f'{v * 100:.1f}%' if abs(v) <= 1 else f'{v:.1f}%'
    if spec == 'int':
        return f'{int(round(v))}'
    if spec == 'dec1':
        return f'{v:.1f}'
    if spec == 'dec1+':
        # Per memory: never prefix positives with '+'. Negatives still get '-'.
        return f'{v:.1f}'
    if spec == 'mph':
        return f'{v:.1f} mph'
    return str(v)


def _percentile_color(pctl):
    """PRINT-IDENTITY percentile scale — matches the redesigned website's BUBBLE
    scale (Utils.percentileBubbleColor). Blends from a VISIBLE warm-greige floor
    at the 50th percentile toward slate (low) or brick (high), so mid-percentile
    bubbles read as filled discs on cream instead of vanishing into the paper.
    Endpoints kept light enough for ink text. Returns (fill_rgb01, ring_rgb01)."""
    if pctl is None:
        return (0.796, 0.722, 0.612), (0.757, 0.682, 0.573)  # greige neutral
    p = max(0, min(100, pctl))
    neutral = (203 / 255, 184 / 255, 156 / 255)       # warm greige, visible on cream
    target = (168 / 255, 54 / 255, 40 / 255) if p >= 50 else (86 / 255, 118 / 255, 152 / 255)
    t = (abs(p - 50) / 50.0) ** 0.72
    fill = tuple(neutral[i] + (target[i] - neutral[i]) * t for i in range(3))
    # Ring: a touch deeper so the circle reads distinct from the bar fill.
    tr = min(1.0, t * 1.10 + 0.05)
    ring = tuple(neutral[i] + (target[i] - neutral[i]) * tr for i in range(3))
    return fill, ring


def _hitter_stat_cell_color(value_str, league_avg, scale, higher_is_better,
                            row_bg_hex, is_pct):
    """Headline-strip cell tint, in the SAME hue family as the percentile
    bubbles below it: red = above average (good), blue = below average (bad).

    Deliberately NOT Cards.py's pct_cell_color/raw_cell_color, which tint
    green/red — that palette is shared with the pitcher cards and must stay
    put. The intensity math (deviation from league average, normalized and
    clamped) is identical to Cards.py so only the hue changes; the extreme
    anchors are pulled straight from _percentile_color so the strip can
    never drift out of sync with the bubble gradient.
    """
    if league_avg is None or not value_str or value_str == '—':
        return None
    try:
        if is_pct:
            val = float(value_str.replace('%', ''))
            diff = val - league_avg * 100
            denom = 8.0                      # ±8 pp → full intensity (Cards.py)
        else:
            val = float(value_str)
            diff = val - league_avg
            denom = scale
    except (ValueError, AttributeError):
        return None
    if not higher_is_better:
        diff = -diff
    intensity = max(-1.0, min(1.0, diff / denom))
    # Extreme red (pctl 100 = good) / extreme blue (pctl 0 = bad) — exactly
    # the bubble gradient endpoints.
    anchor = _percentile_color(100 if intensity >= 0 else 0)[0]
    target = tuple(int(round(ch * 255)) for ch in anchor)
    alpha = abs(intensity) * 0.55           # match Cards.py saturation
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


def _render_percentile_bubbles(fig, h_row):
    """Single-column percentile panel matching the website's PERCENTILE
    RANKINGS sidebar. Vertical stack of section sub-headers + pill-bar rows:

        RESULT  ──────────────────────────────────
          Run Value      [──────fill──────(○)]   8.7
          xwOBA          [──────fill──────(○)]   .425
          ...
        QUALITY OF CONTACT  ──────────────────────
          xwOBAcon       [──────fill──────(○)]   .603
          ...

    Anchored to the left side of the card (clear of LA × Spray at x ≥ 0.43).
    """
    from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch

    # GRID_RIGHT 0.385 -> 0.390 (2026-08-12). The binding neighbour is NOT the
    # LA x Spray plot area (spray_axes_left = 9.5/22 = 0.432) but its rotated
    # "Launch Angle" y-label, which hangs ~0.015 further left at ~0.417. 0.405
    # put the widest value ("116.3 mph") right up against it.
    GRID_LEFT, GRID_RIGHT = 0.020, 0.390
    # GRID_BOT raised 0.030 -> 0.098 to clear the reader's notes below, with
    # a visible gap: at 0.088 the last bubble row sat almost on top of them.
    GRID_TOP, GRID_BOT = 0.715, 0.098
    col_w = GRID_RIGHT - GRID_LEFT

    # ROC (AAA) hitters: hide the BAT TRACKING section entirely. Bat
    # tracking (BatSpeed/SwingLength/AttackAngle/Squared-Up%/Blast%) is
    # MLB-only Statcast hardware — the data will never exist for AAA,
    # so showing the section as five "—" rows is just dead space.
    # _isROC is pipeline-internal (stripped from JSON output), so detect
    # ROC the same way the rest of the card does: by team string.
    _is_roc = (h_row.get('team') == 'ROC')
    _columns = []
    for name, metrics in BUBBLE_COLUMNS:
        if _is_roc and name == 'RESULT':
            _m = []
            for spec in metrics:
                if spec[1] != 'batSpeed':
                    _m.append(spec)
                elif h_row.get('_mlbBatSpeed') is not None:
                    # ROC hitter with an MLB stint: show that bat speed,
                    # labeled as MLB-sourced (injected by render_hitter_card —
                    # the only MLB field allowed onto a ROC card).
                    _m.append(('Bat Speed (MLB)', '_mlbBatSpeed',
                               '_mlbBatSpeed_pctl', 'mph'))
                # else: dropped. Bat tracking is MLB-only Statcast hardware,
                # so the AAA value will never exist — a permanent "—" row is
                # dead space, not missing data.
            _columns.append((name, _m))
        else:
            _columns.append((name, metrics))
    total_rows = sum(len(metrics) for _h, metrics in _columns)
    n_sections = len(_columns)

    grid_h = GRID_TOP - GRID_BOT
    SECTION_HEADER_H = 0.024
    SECTION_TOP_GAP  = 0.008
    SECTION_GAP      = 0.022
    fixed_overhead = (n_sections * (SECTION_HEADER_H + SECTION_TOP_GAP)
                       + (n_sections - 1) * SECTION_GAP)
    row_h = (grid_h - fixed_overhead) / total_rows

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_zorder(5)

    # Row layout: [label][gap][pill bar with circle on right end][gap][value]
    #
    # The label and value gutters are MEASURED from the strings this card will
    # actually draw, rather than reserved as a fixed fraction of the column.
    # The old fixed 0.38/0.13 split was sized for the pitcher card's
    # "Run Value (All Pitches)"; hitter labels top out at "Swing Decisions+",
    # so it was donating a large dead margin to every row. Reclaiming it
    # lengthens the bar, and bar length is what makes percentiles legible:
    # travel per percentile point is (effective_bar_w - MIN_VISIBLE) / 100, so
    # a longer bar is exactly what separates a 75th from a 90th.
    ROW_FS = 13
    _labels = [m[0] for _s, ms in _columns for m in ms]
    _values = [_format_bubble_value(h_row.get(m[1]), m[3])
               for _s, ms in _columns for m in ms]
    _lw = _measure_text_axis_w(fig, _labels, ROW_FS, '500')
    _vw = _measure_text_axis_w(fig, _values, ROW_FS, '600')
    LABEL_W = _lw if _lw else col_w * 0.38
    VALUE_W = _vw if _vw else col_w * 0.13
    # Breathing room so the bar never crowds the text on either side.
    LABEL_BAR_GAP = 0.012
    BAR_VALUE_GAP = 0.012

    # Bar height is now ~85% of the circle diameter so the bar fully enters
    # the circle at the meeting point — no empty curve above/below the
    # tangent. Combined with the fill-extends-into-circle trick below, the
    # bar and circle read as one connected glyph instead of "bar, gap,
    # bubble, gap, value".
    BAR_HEIGHT_IN  = 0.34
    bar_h_axis     = BAR_HEIGHT_IN / fig.get_size_inches()[1]
    CIRCLE_DIAM_IN = 0.40
    ellipse_w = CIRCLE_DIAM_IN / fig.get_size_inches()[0]
    ellipse_h = CIRCLE_DIAM_IN / fig.get_size_inches()[1]
    CIRCLE_CLEARANCE_AXIS_X = (CIRCLE_DIAM_IN / fig.get_size_inches()[0]) * 0.55

    x_label_left  = GRID_LEFT
    x_label_right = GRID_LEFT + LABEL_W
    x_bar_left    = x_label_right + LABEL_BAR_GAP
    x_bar_zone_right = GRID_RIGHT - VALUE_W - BAR_VALUE_GAP
    x_bar_right   = x_bar_zone_right - CIRCLE_CLEARANCE_AXIS_X
    x_value_right = GRID_RIGHT
    bar_total_w   = x_bar_right - x_bar_left

    rounding = bar_h_axis / 2  # fully rounded pill ends

    y_cursor = GRID_TOP
    for sec_idx, (section, metrics) in enumerate(_columns):
        if sec_idx > 0:
            y_cursor -= SECTION_GAP

        header_y = y_cursor
        ax.text(GRID_LEFT, header_y, section,
                ha='left', va='top',
                fontsize=13, fontfamily='IBM Plex Sans Condensed', fontweight='700',
                color=TEXT_SECONDARY)
        rule_y = header_y - SECTION_HEADER_H + 0.002
        ax.add_patch(Rectangle((GRID_LEFT, rule_y),
                                col_w, 0.0010,
                                facecolor=TEXT_FAINT, edgecolor='none', alpha=0.5))
        y_cursor = header_y - SECTION_HEADER_H - SECTION_TOP_GAP

        for label, val_key, pctl_key, fmt_spec in metrics:
            row_top = y_cursor
            row_bot = y_cursor - row_h
            row_mid = (row_top + row_bot) / 2
            y_cursor = row_bot

            val = h_row.get(val_key)
            pctl = h_row.get(pctl_key)
            val_str = _format_bubble_value(val, fmt_spec)
            fill_color, ring_color = _percentile_color(pctl)

            # Label
            ax.text(x_label_left, row_mid, label,
                    ha='left', va='center',
                    fontsize=13, fontfamily='IBM Plex Sans', fontweight='500',
                    color=TEXT_PRIMARY)

            # Pill track — full rounded pill in gray
            track_y = row_mid - bar_h_axis / 2
            track = FancyBboxPatch(
                (x_bar_left + rounding, track_y),
                bar_total_w - 2 * rounding, bar_h_axis,
                boxstyle=f'round,pad=0,rounding_size={rounding}',
                facecolor=TEXT_FAINT, edgecolor='none', alpha=0.20,
                linewidth=0, zorder=8,
            )
            ax.add_patch(track)

            # Pill fill — a plain rectangle CLIPPED to the track's rounded
            # pill shape. The visible-fill width controls where the circle
            # sits; the rendered fill overlaps INTO the circle so the bar
            # and bubble read as one continuous glyph (the circle is drawn
            # at higher zorder, so the overlap is hidden but the seam
            # disappears).
            #
            # Why visible_fill_w isn't just `effective_bar_w * p`:
            # A pure-linear mapping puts a 6th-percentile circle at ~6% of
            # the bar width, which is so close to x_bar_left that the
            # bubble visually collapses to "at zero". We add a baseline
            # MIN_VISIBLE so low percentiles get a clear non-zero offset.
            # The numeric percentile inside the circle stays the source of
            # truth; this just gives the eye a fairer visual cue. The
            # cost is a small compression of the linear range — 99th
            # percentile sits a hair shorter than it would otherwise.
            radius_x = ellipse_w / 2
            # Reserve a circle's diameter at the right end so the circle
            # never overshoots the bar.
            effective_bar_w = bar_total_w - 2 * radius_x
            p = max(0, min(100, pctl)) / 100.0 if pctl is not None else 0
            MIN_VISIBLE = radius_x * 1.5   # ~0.6 in baseline at pctl=0
            visible_fill_w = MIN_VISIBLE + p * (effective_bar_w - MIN_VISIBLE)
            FILL_INTO_CIRCLE = radius_x * 0.85
            fill_render_w = visible_fill_w + FILL_INTO_CIRCLE
            if pctl is not None and fill_render_w > 0:
                fill = Rectangle(
                    (x_bar_left, track_y),
                    fill_render_w, bar_h_axis,
                    facecolor=fill_color, edgecolor='none', alpha=0.95,
                    zorder=9,
                )
                ax.add_patch(fill)
                fill.set_clip_path(track)

            # Circle's LEFT edge sits at x_bar_left + visible_fill_w, so the
            # full visible fill is always to the left of the circle. With
            # the MIN_VISIBLE baseline, low-percentile bubbles sit a clear
            # circle-width to the right of x_bar_left, never collapsing
            # to "at zero". For pctl=100 the circle's right edge aligns
            # with the bar's right (because visible_fill_w = effective_bar_w
            # at p=1, and circle_x + radius_x = x_bar_left + bar_total_w).
            circle_x = x_bar_left + visible_fill_w + radius_x
            ell = Ellipse((circle_x, row_mid),
                           ellipse_w, ellipse_h,
                           facecolor=ring_color, edgecolor='none',
                           linewidth=0, zorder=12)
            ax.add_patch(ell)
            label_pctl = f'{int(round(pctl))}' if pctl is not None else '—'
            ax.text(circle_x, row_mid, label_pctl,
                    ha='center', va='center',
                    fontsize=11, fontfamily='IBM Plex Sans', fontweight='700',
                    color=TEXT_PRIMARY, zorder=13)

            # Value
            ax.text(x_value_right, row_mid, val_str,
                    ha='right', va='center',
                    fontsize=13, fontfamily='IBM Plex Sans', fontweight='600',
                    color=TEXT_PRIMARY)


# ─────────────────────────────────────────────────────────────────────
# Date-window rows
# ─────────────────────────────────────────────────────────────────────
# The season leaderboard row is not valid inside a date window, so a
# windowed card rebuilds every number it can from the window's own pitches.
# The pipeline functions do the arithmetic, so a window value cannot drift
# from the shipped season definition of the same stat.
#
# Three families stay out of the row, and therefore render as "—":
#   * wRC+ — the shipped value comes from FanGraphs, and our fg_overrides
#     cache is fetched season-scoped, so there is no window-matched FG value
#     to use. (FanGraphs DOES serve custom date ranges; we just do not ask
#     it for them yet.) Our own formula would be a different quantity under
#     the same name.
#   * SD+, CT+, BB+, Hitter+ — each needs a whole-league cell table (SD+,
#     CT+) or the league xwOBAcon anchor (BB+) measured over the SAME
#     window, which only a pipeline run can produce.
#   * every *_pctl — a percentile needs an all-MLB pool over the same
#     window. Without it the bubbles render in the neutral greige.
#
# Known gap: a no-pitch intentional walk leaves no pitch, so it is absent
# from the window PA count. The season card takes PA from the official
# boxscore and does not have this gap. Measured on Wood 2026 (the full
# season fed through this function, compared against the shipped row):
# 529 PA against 535, all six the walks. Every other key matched exactly,
# and OBP/OPS/BB%/K% moved only by that denominator.
WINDOW_UNAVAILABLE = ('wRCplus', 'xWRCplus', 'wRC',
                      'sdPlus', 'ctPlus', 'bbPlus', 'hitterPlus')


class _SkipFGOverride(Exception):
    """Sentinel: the FanGraphs season override does not apply to this card."""


def _build_window_hitter_row(season_row, window_pitches, metadata):
    """Rebuild one hitter's leaderboard row from a date window's pitches."""
    from pipeline.compute import compute_hitter_stats, compute_expected_stats
    from pipeline.utils import (NON_PA_EVENTS, HIT_EVENTS, K_EVENTS, BB_EVENTS,
                                HBP_EVENTS, SF_EVENTS, SH_EVENTS, CI_EVENTS)

    _dates = [p.get('Game Date') for p in window_pitches if p.get('Game Date')]
    row = {
        'hitter': season_row.get('hitter'),
        'team': season_row.get('team'),
        'stands': season_row.get('stands'),
        'mlbId': season_row.get('mlbId'),
        'age': season_row.get('age'),
        'throws': season_row.get('throws'),
        'position': season_row.get('position'),
        '_isROC': season_row.get('_isROC'),
        'count': len(window_pitches),
        'lastGameDate': max(_dates) if _dates else None,
    }
    row.update(compute_hitter_stats(window_pitches))
    row.update(compute_expected_stats(
        window_pitches, woba_weights=metadata.get('wobaWeights')))

    # Slash line. compute_hitter_stats owns pa/ab/hr/babip; it does not
    # return AVG/OBP/SLG, because the season pipeline takes those from the
    # official boxscore. A window has no boxscore, so count the events.
    pa_pitches = [p for p in window_pitches
                  if p.get('Event') and p['Event'] not in NON_PA_EVENTS]
    n_pa = len(pa_pitches)
    n_h = sum(1 for p in pa_pitches if p['Event'] in HIT_EVENTS)
    n_2b = sum(1 for p in pa_pitches if p['Event'] == 'Double')
    n_3b = sum(1 for p in pa_pitches if p['Event'] == 'Triple')
    n_hr = sum(1 for p in pa_pitches if p['Event'] == 'Home Run')
    n_k = sum(1 for p in pa_pitches if p['Event'] in K_EVENTS)
    n_bb = sum(1 for p in pa_pitches if p['Event'] in BB_EVENTS)   # incl. IBB
    n_hbp = sum(1 for p in pa_pitches if p['Event'] in HBP_EVENTS)
    n_sf = sum(1 for p in pa_pitches if p['Event'] in SF_EVENTS)
    n_sh = sum(1 for p in pa_pitches if p['Event'] in SH_EVENTS)
    n_ci = sum(1 for p in pa_pitches if p['Event'] in CI_EVENTS)
    n_ab = n_pa - n_bb - n_hbp - n_sf - n_sh - n_ci
    n_tb = n_h + n_2b + 2 * n_3b + 3 * n_hr
    obp_denom = n_ab + n_bb + n_hbp + n_sf

    row['tb'] = n_tb
    row['avg'] = round(n_h / n_ab, 3) if n_ab > 0 else None
    row['obp'] = round((n_h + n_bb + n_hbp) / obp_denom, 3) if obp_denom > 0 else None
    row['slg'] = round(n_tb / n_ab, 3) if n_ab > 0 else None
    row['ops'] = (round(row['obp'] + row['slg'], 3)
                  if row['obp'] is not None and row['slg'] is not None else None)
    row['iso'] = (round(row['slg'] - row['avg'], 3)
                  if row['slg'] is not None and row['avg'] is not None else None)
    # Hitter BB% uses TOTAL walks, matching the season row.
    row['kPct'] = round(n_k / n_pa, 4) if n_pa > 0 else None
    row['bbPct'] = round(n_bb / n_pa, 4) if n_pa > 0 else None
    row['bbToK'] = (n_bb / n_k) if n_k > 0 else None

    for k in WINDOW_UNAVAILABLE:
        row[k] = None
    # xwOBAsp is deliberately ABSENT rather than None: render_hitter_card
    # reads it with a default, and the default is the value it computed from
    # this window's own batted balls.
    row.pop('xwOBAsp', None)
    return row


# ─────────────────────────────────────────────────────────────────────
# Card render
# ─────────────────────────────────────────────────────────────────────
def render_hitter_card(hitter_name, team_abbrev=None, year_label='2026 Season',
                       output_dir=OUTPUT_DIR, layout='bubbles', la_view='damage',
                       date_filter=None, date_display=None, date_slug=None):
    """Render a seasonal hitter card, or a card for one date window.

    date_filter:
        None (default) renders the full season from the leaderboard row.
        A ('YYYY-MM-DD', 'YYYY-MM-DD') pair keeps only the pitches inside
        that window, then rebuilds every stat from those pitches. See
        _build_window_hitter_row for what a window cannot rebuild.
    date_display:
        Label drawn under the player name. Replaces the season "Through
        {date}" stamp when a window is set.
    date_slug:
        Filename component. Replaces the year_label slug when set.

    layout:
        'bubbles' — default. Single-column percentile bubble grid
            (Result / Quality of Contact / Plate Discipline / Bat Tracking)
            on the left, LA × Spray chart on the right, headline strip on top.
        'classic' — legacy variant. Heat maps (Whiff/Damage × RHP/LHP) on the
            left, LA × Spray on the right, Contact Profile + Pitch Group table
            at the bottom. Still selectable via --layout classic.
    """
    _span = date_display or year_label
    print(f"Generating hitter card: {hitter_name} ({team_abbrev or 'auto'}) — {_span} [layout={layout}]")

    # Load data — load_pitch_data auto-refreshes from the CI Release if the
    # local pickle is stale or missing (the pipeline uploads it on every run,
    # so this keeps the card aligned with whatever date is in the JSON).
    print("  Loading pitch cache + leaderboard + metadata...")
    all_pitches = load_pitch_data()
    hitter_lb = load_hitter_leaderboard()
    metadata = load_metadata()

    # Find hitter row
    hitter_rows = [r for r in hitter_lb if r.get('hitter') == hitter_name]
    if team_abbrev:
        hitter_rows = [r for r in hitter_rows if r.get('team') == team_abbrev]
    if not hitter_rows:
        print(f"  ERROR: {hitter_name} not found in leaderboard")
        return False
    if len(hitter_rows) > 1:
        # Prefer single-team row over multi-team aggregate
        single = [r for r in hitter_rows if not (str(r.get('team', '')).endswith('TM'))]
        hitter_rows = single or hitter_rows
    h_row = hitter_rows[0]
    _season_row = dict(h_row)   # identity fields a window row cannot derive
    team = h_row.get('team') or team_abbrev or '???'
    bats = h_row.get('stands') or 'R'

    # ROC cards: supplement the ONE level-independent bat-tracking number —
    # Bat Speed — from the hitter's MLB stint row when one exists. Bat speed
    # is a physical tool (Statcast hardware is MLB-only), so an MLB reading
    # is valid context on an AAA card; nothing else from the MLB row is
    # merged. Rendered as its own "BAT TRACKING (MLB)" bubble section.
    if team == 'ROC' and date_filter is None:
        _mid = h_row.get('mlbId')
        _mlb_rows = [r for r in hitter_lb
                     if r.get('hitter') == hitter_name
                     and (_mid is None or r.get('mlbId') == _mid)
                     and r.get('team') not in ('ROC', 'AAA')
                     and r.get('batSpeed') is not None]
        if _mlb_rows:
            _mlb = max(_mlb_rows, key=lambda r: r.get('pa') or 0)
            h_row = dict(h_row)
            h_row['_mlbBatSpeed'] = _mlb.get('batSpeed')
            h_row['_mlbBatSpeed_pctl'] = _mlb.get('batSpeed_pctl')
            print(f"  MLB Bat Speed supplement: {_mlb.get('batSpeed')} mph "
                  f"({_mlb.get('team')}, {_mlb.get('pa')} PA)")

    # Filter pitches to this hitter (and team if specified)
    hitter_pitches = [p for p in all_pitches
                      if p.get('Batter') == hitter_name
                      and (not team_abbrev or p.get('BTeam') == team_abbrev)]
    if not hitter_pitches:
        print(f"  ERROR: no pitches found for {hitter_name}")
        return False

    # Date window. Every stat below reads hitter_pitches or h_row, so the
    # filter and the row rebuild together move the whole card onto the window.
    window_pool = False
    if date_filter is not None:
        _lo, _hi = date_filter
        _season_n = len(hitter_pitches)
        hitter_pitches = [p for p in hitter_pitches
                          if p.get('Game Date')
                          and _lo <= p['Game Date'] <= _hi]
        if not hitter_pitches:
            print(f"  ERROR: no pitches for {hitter_name} between {_lo} and {_hi}")
            return False
        print(f"  Date window {_lo} → {_hi}: "
              f"{len(hitter_pitches)} of {_season_n} season pitches")

        # Values from the window; percentiles from the SEASON pool, for every
        # stat, with no sample-size gate. A hot three-week stretch is allowed
        # to read as an elite percentile — that is the whole point, and it is
        # what makes two windows comparable to each other.
        try:
            from pipeline.window_pool import score_window_against_season
            _t0 = _time.time()
            h_row = score_window_against_season(
                (hitter_name, h_row.get('team') or team),
                hitter_pitches, all_pitches, hitter_lb, metadata,
                date_range=date_filter,
                identity_mlb_id=_season_row.get('mlbId'))
            h_row.update({k: v for k, v in _season_row.items()
                          if k in ('mlbId', 'age', 'throws', 'position',
                                   'stands', 'sprintSpeed', 'sprintSpeed_pctl')
                          and v is not None})
            window_pool = True
            print(f"  Scored in {_time.time() - _t0:.1f}s. "
                  f"{h_row.get('pa')} PA, Hitter+ {h_row.get('hitterPlus')}, "
                  f"wRC+ {h_row.get('wRCplus')}. Percentiles rank this window "
                  f"against the full-season MLB pool.")
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"  WARNING: window scoring failed ({_e}) — "
                  f"falling back to a pitch-rebuilt row with no percentiles")
            h_row = _build_window_hitter_row(_season_row, hitter_pitches, metadata)

    print(f"  {len(hitter_pitches)} pitches faced")

    # Compute SzTop/SzBot (already constant per hitter in current data)
    sz_tops = [v for v in (sf(p.get('SzTop')) for p in hitter_pitches) if v is not None]
    sz_bots = [v for v in (sf(p.get('SzBot')) for p in hitter_pitches) if v is not None]
    sz_top = float(np.mean(sz_tops)) if sz_tops else 3.5
    sz_bot = float(np.mean(sz_bots)) if sz_bots else 1.5

    # Headshot
    mlb_id = h_row.get('mlbId')
    headshot = fetch_headshot(mlb_id) if mlb_id else None
    if headshot is None:
        from PIL import Image
        headshot = Image.new('RGB', (180, 180), (50, 50, 50))

    # Display name (FIRST LAST)
    parts = hitter_name.split(', ')
    display_name = (parts[1] + ' ' + parts[0]).upper() if len(parts) == 2 else hitter_name.upper()

    # Age + throws — fetch from MLB API. fetch_player_metadata returns
    # 'hand' = pitchHand.code (the player's throwing hand). For a hitter
    # this is what they throw with — i.e. the "throws" side. The function
    # also returns 'age'. Both are used in the identity line below.
    age = h_row.get('age')
    throws = h_row.get('throws')
    if (not age or not throws) and mlb_id:
        try:
            from cards.pitcher import fetch_player_metadata
            meta = fetch_player_metadata(mlb_id)
            if not age:
                age = meta.get('age', '—')
            if not throws:
                throws = meta.get('hand', 'R')
        except Exception:
            pass
    if not age:    age = '—'
    if not throws: throws = 'R'

    # Position from h_row (resolved daily into hitter_position_cache.json
    # at pipeline time). Fallback to '—' if missing.
    position = h_row.get('position') or '—'

    # ─── Set up the figure ─────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)

    # Subtle radial-gradient background for depth — slightly lighter at center,
    # darker at edges. Painted at the very back so all other axes draw on top.
    # BG rgb derived from the BG hex constant so theme overrides flow through.
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-100)
    ax_bg.axis('off')
    n_bg = 200
    _xs, _ys = np.meshgrid(np.linspace(-1, 1, n_bg), np.linspace(-1, 1, n_bg))
    _d = np.sqrt(_xs ** 2 + _ys ** 2) / np.sqrt(2)
    _lift = (1.0 - _d) * 0.045
    _bg_hex = BG.lstrip('#')
    bg_rgb = (int(_bg_hex[0:2], 16) / 255,
              int(_bg_hex[2:4], 16) / 255,
              int(_bg_hex[4:6], 16) / 255)
    bg_grad = np.zeros((n_bg, n_bg, 3))
    for ci in range(3):
        bg_grad[..., ci] = np.clip(bg_rgb[ci] + _lift, 0, 1)
    ax_bg.imshow(bg_grad, extent=[0, 1, 0, 1], aspect='auto',
                  origin='lower', interpolation='bilinear')

    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, FIG_W); ax_main.set_ylim(0, FIG_H)
    ax_main.axis('off')
    # Don't fill ax_main background — let the radial gradient show through

    # Layout anchors.
    stripe_left = 0.01 * FIG_W
    stripe_right = 0.99 * FIG_W
    stripe_y = FIG_H - 0.20

    # ─── Brand stripe (PRINT-IDENTITY) ──────────────────────────────
    # Fixed Okabe-Ito pitch palette, spanning the card width — the shared
    # brand mark used on the website header and the pitcher cards. Unlike the
    # pitcher card's usage-ordered arsenal stripe, the hitter has no arsenal,
    # so this is the fixed 5-colour brand bar (SI/CH/FF/SL/FC).
    _BRAND_STRIPE = ['#E0A81E', '#009E73', '#0072B2', '#D55E00', '#8B5A2B']
    _seg_w = (stripe_right - stripe_left) / len(_BRAND_STRIPE)
    _sx = stripe_left
    for _c in _BRAND_STRIPE:
        ax_main.add_patch(Rectangle((_sx, FIG_H - 0.18), _seg_w, 0.16,
                                     facecolor=_c, edgecolor='none', zorder=6))
        _sx += _seg_w

    # ─── Identity zone (top-left) ───────────────────────────────────
    photo_left = stripe_left
    photo_top = stripe_y - 0.05      # nudged up; gap below photo widened
                                      # via stat_y_header offset below
    photo_w = 2.0
    photo_h = photo_w * headshot.size[1] / headshot.size[0]
    photo_bottom = photo_top - photo_h
    ax_main.imshow(np.array(headshot),
                    extent=[photo_left, photo_left + photo_w, photo_bottom, photo_top],
                    aspect='auto', zorder=2, interpolation='antialiased')
    # Photo edge — alpha 0.80, linewidth 1.5: actually frames the photo
    # without screaming. Reads as intentional, not decorative.
    ax_main.add_patch(Rectangle((photo_left, photo_bottom), photo_w, photo_h,
                                 fill=False, edgecolor=PHOTO_BORDER, linewidth=1.5,
                                 alpha=0.80, zorder=3))

    # Inverted typography hierarchy: name down, context up, season label
    # demoted from cyan display font to a quiet off-white sans subline.
    text_x = photo_left + photo_w + 0.4
    ax_main.text(text_x, photo_top - 0.15, display_name,
                  fontsize=31, fontfamily='Bitter', color=TEXT_PRIMARY,
                  va='top', fontweight='black')
    # Identity line: "POS | B/T | TEAM | Age: N"
    # Bats/throws are shown as "L/R" (single-letter codes joined by slash)
    # so the line stays compact. Switch hitters show "S/<throws>".
    ax_main.text(text_x, photo_top - 1.05,
                  f"{position}  |  {bats}/{throws}  |  {team}  |  Age: {age}",
                  fontsize=16, fontfamily='IBM Plex Sans', color=TEXT_PRIMARY, va='top',
                  fontweight='600')
    # Year label + "Through {date}" inline data freshness stamp.
    # Resolution order:
    #   1. h_row['lastGameDate'] — written by process_data.py from the JSON
    #      leaderboard, always fresh (the JSON is git-tracked, the pickle
    #      is not, so this is the most reliable source).
    #   2. max Game Date in hitter_pitches — fallback when the leaderboard
    #      row doesn't yet have lastGameDate (older pipeline run).
    #   3. metadata.generatedAt — last-resort fallback.
    # A window card already names its own span, so it takes no "Through"
    # stamp — the stamp would repeat the window's end date.
    _date_suffix = ''
    if date_display is None:
        try:
            from datetime import datetime
            _latest = h_row.get('lastGameDate')
            if not _latest:
                _hitter_dates = [p.get('Game Date') for p in hitter_pitches
                                  if p.get('Game Date')]
                _latest = max(_hitter_dates) if _hitter_dates else None
            if not _latest:
                _gen_at = metadata.get('generatedAt', '')
                _latest = _gen_at[:10] if _gen_at else None
            if _latest:
                _dt = datetime.strptime(_latest[:10], '%Y-%m-%d')
                _date_suffix = f"  ·  Through {_dt.strftime('%B %-d').lstrip('0')}"
        except (ValueError, TypeError):
            _date_suffix = ''
    ax_main.text(text_x, photo_top - 1.80, (date_display or year_label) + _date_suffix,
                  fontsize=17, fontfamily='IBM Plex Sans', color=TEXT_SECONDARY,
                  va='top', fontweight='600')

    # ─── Headline stats strip ────────────────────────────────────────
    # ROC players have a reduced stat set (no + family, no bat-tracking)
    is_roc = (h_row.get('team') == 'ROC')
    headline_stats = HEADLINE_STATS_ROC if is_roc else HEADLINE_STATS_MLB

    # FG override: pull canonical wRC+ (and xwOBA/xBA/xSLG for MLB) so
    # the card matches fangraphs.com. The pipeline writes these values
    # directly into the JSON, so this fallback only matters when the
    # card is rendered against an older leaderboard snapshot (pre-FG-
    # override pipeline run). MLB hitters get wRC+ + xwOBA + xBA + xSLG;
    # AAA hitters get wRC+ only (FG doesn't publish the Statcast
    # expected stats for AAA).
    # A window card skips this: our fg_overrides cache is fetched
    # season-scoped, so the override would put four SEASON values back onto a
    # window card. FanGraphs itself does serve custom date ranges.
    try:
        if date_filter is not None:
            # A window pool already ran with window_mode=True, which skipped
            # the FanGraphs override on purpose. A pitch-rebuilt row has no
            # season values to protect either. Both cases: stay off.
            raise _SkipFGOverride
        from pipeline.fg_overrides import refresh_if_stale as _fg_refresh
        _fg_cache = _fg_refresh(max_age_hours=24, verbose=True)
        _group_key = 'aaaHitters' if is_roc else 'mlbHitters'
        _mid = h_row.get('mlbId')
        if _mid is not None:
            _fg_player = _fg_cache.get(_group_key, {}).get(str(int(_mid)))
            if _fg_player:
                _h_row_modified = False
                # Fields to copy from the FG cache to the in-memory row.
                # Each entry is (cache_key, row_key); silently skip when
                # the cache doesn't have a value for that key (AAA only
                # has wRCplus, MLB has all four).
                for cache_k, row_k in (
                    ('wRCplus', 'wRCplus'),
                    ('xwOBA',   'xwOBA'),
                    ('xBA',     'xBA'),
                    ('xSLG',    'xSLG'),
                ):
                    if _fg_player.get(cache_k) is not None:
                        if not _h_row_modified:
                            h_row = dict(h_row); _h_row_modified = True
                        h_row[row_k] = _fg_player[cache_k]
    except _SkipFGOverride:
        pass
    except Exception as _e:
        # Never block the card render on FG scraper failure.
        print(f'  WARNING: FG override unavailable ({type(_e).__name__}: {_e})')
    stat_values = []
    for k in headline_stats:
        if k == 'PA':
            stat_values.append(str(h_row.get('pa', '—')))
        # HR uncolored like PA: counting stats tinted against a league mean
        # mostly reflect playing time, not quality (2026-07-28, per Wally's
        # card review — the one number every casual reader asks for).
        elif k == 'HR':   stat_values.append(fmt_int(h_row.get('hr')))
        elif k == 'AVG':  stat_values.append(fmt_3dec(h_row.get('avg')))
        elif k == 'OBP':  stat_values.append(fmt_3dec(h_row.get('obp')))
        elif k == 'SLG':  stat_values.append(fmt_3dec(h_row.get('slg')))
        elif k == 'OPS':  stat_values.append(fmt_3dec(h_row.get('ops')))
        elif k == 'wOBA': stat_values.append(fmt_3dec(h_row.get('wOBA')))
        elif k == 'wRC+': stat_values.append(fmt_int(h_row.get('wRCplus')))
        # Legacy stats kept as fallbacks in case a theme wrapper or older
        # caller still references them — the canonical strip doesn't show
        # any of these anymore (they're all in the percentile bubbles).
        elif k == 'BB%':  stat_values.append(fmt_pct(h_row.get('bbPct')))
        elif k == 'K%':   stat_values.append(fmt_pct(h_row.get('kPct')))
        elif k == 'SD+':  stat_values.append(fmt_int(h_row.get('sdPlus')))
        elif k == 'CT+':  stat_values.append(fmt_int(h_row.get('ctPlus')))
        elif k == 'BB+':  stat_values.append(fmt_int(h_row.get('bbPlus')))
        elif k == 'Hitter+': stat_values.append(fmt_int(h_row.get('hitterPlus')))
        else: stat_values.append('—')

    # Proportional column widths — every cell sized to its content + padding.
    # All cells (slash line + rate stats + + stats) render at the same scale.
    HL_PAD_CHARS = 4
    HL_MIN_CHARS = 5
    hl_char_widths = []
    for i, k in enumerate(headline_stats):
        widest = max(len(k), len(stat_values[i]))
        hl_char_widths.append(max(HL_MIN_CHARS, widest + HL_PAD_CHARS))
    hl_total_chars = sum(hl_char_widths)
    # Total horizontal budget: identity column edge → just before LA × Spray
    hl_total_w = (9.5 / FIG_W * FIG_W) - photo_left - 0.4   # in inches
    hl_widths = [hl_total_w * c / hl_total_chars for c in hl_char_widths]

    cell_h = 0.55
    stat_y_header = photo_bottom - 0.70   # was 0.55 — extra breathing room
                                            # between the photo and headline
    stat_y_value = stat_y_header - cell_h
    hitter_la = metadata.get('hitterLeagueAverages', {})
    cur_x = photo_left
    for i, k in enumerate(headline_stats):
        cw = hl_widths[i]
        val_str = stat_values[i]
        # Header cell
        ax_main.add_patch(Rectangle((cur_x, stat_y_header), cw, cell_h,
                                     facecolor=DARKER, edgecolor=SUBTLE_BORDER,
                                     linewidth=0.8))
        ax_main.text(cur_x + cw / 2, stat_y_header + cell_h / 2, k,
                      fontsize=12, ha='center', va='center', color=TEXT_SECONDARY,
                      fontweight='bold', fontfamily='IBM Plex Sans Condensed')
        # Value cell — percentile-colored bg
        cell_bg = DARK_CELL
        sl_cfg = HITTER_STAT_LINE_COLOR.get(k)
        if sl_cfg and hitter_la and val_str and val_str != '—':
            la_val = hitter_la.get(sl_cfg[0])
            if la_val is not None:
                if sl_cfg[1] == 'pct':
                    tinted = _hitter_stat_cell_color(val_str, la_val, None,
                                                     sl_cfg[2], DARK_CELL, is_pct=True)
                else:
                    scale = sl_cfg[3] if len(sl_cfg) > 3 else 1.0
                    tinted = _hitter_stat_cell_color(val_str, la_val, scale,
                                                     sl_cfg[2], DARK_CELL, is_pct=False)
                if tinted:
                    cell_bg = tinted
        ax_main.add_patch(Rectangle((cur_x, stat_y_value), cw, cell_h,
                                     facecolor=cell_bg, edgecolor=SUBTLE_BORDER,
                                     linewidth=0.8))
        # All values render at the same size — no slash-line emphasis
        ax_main.text(cur_x + cw / 2, stat_y_value + cell_h / 2, val_str,
                      fontsize=15, ha='center', va='center',
                      color=TEXT_PRIMARY, fontweight='bold',
                      fontfamily='IBM Plex Sans')
        cur_x += cw

    # Subtle group dividers — thin vertical hairlines between stat groups
    # so the headline strip reads as three semantic groups without changing
    # cell sizes:
    #   PA  |  AVG OBP SLG OPS  |  wOBA wRC+
    #   (sample size)   (traditional production)   (modern summary)
    GROUP_BOUNDARIES = {'AVG', 'wOBA'}
    div_x = photo_left
    for i, k in enumerate(headline_stats):
        if i > 0 and k in GROUP_BOUNDARIES:
            ax_main.plot([div_x, div_x],
                          [stat_y_value, stat_y_header + cell_h],
                          color=PHOTO_BORDER, linewidth=1.2, alpha=0.60,
                          zorder=6)
        div_x += hl_widths[i]

    # ─── LA × Spray scatter (top-right) ────────────────────────────
    # Mirrors the hitter page panel: title centered above, annotations in
    # top-left, big tall chart, legend below.
    sacq_lookup, hand_zones, pool_zones = build_sacq_lookup(metadata, bats)
    # Per-hand lookups so a switch hitter's BIP are each scored by their actual
    # side (matching process_data.compute_xwobasp). The overlay zones above stay
    # in the single `bats` orientation — that's a visual-axis choice only.
    _sacq_full = {'L': build_sacq_lookup(metadata, 'L'),
                  'R': build_sacq_lookup(metadata, 'R')}
    sacq_lookups = {s2: t[0] for s2, t in _sacq_full.items()}
    # Shift the entire LA × Spray block so the title's TOP edge (not its
    # center) sits at the top of the headline stats strip cells. The title
    # text is drawn with va='center' so the center sits ~0.0070 figrel
    # below the top edge (half of the title's ~0.0140 figrel text height
    # at the bumped fontsize). The shift accounts for that half-height
    # offset. Bumped from 12pt to 18pt — title was reading too small at
    # the card's actual print/share size.
    _LA_TITLE_FONTSIZE = 18
    _headline_top_y_figrel = (stat_y_header + cell_h) / FIG_H
    _TITLE_HALF_HEIGHT_FIGREL = _LA_TITLE_FONTSIZE * 1.2 / 72 / FIG_H / 2
    _LA_SHIFT = 0.985 - (_headline_top_y_figrel - _TITLE_HALF_HEIGHT_FIGREL)

    spray_axes_left = 9.5 / FIG_W      # left edge (clear of headline stats)
    spray_axes_right = 0.985
    spray_axes_bottom = 0.295 - _LA_SHIFT
                                        # the legend block can breathe
    spray_axes_top = 0.910 - _LA_SHIFT
    # Switch hitters get two half-width panels (one per batting side).
    # A single panel is wrong twice over for them: the wOBAcon overlay can
    # only be oriented for one side (an 'S' hitter fell into the RHB
    # branch, valuing his majority-LHB pulls as oppo), and the two swings
    # pull to opposite fields on the shared axis. Mirrors the pitcher
    # card's VS RHH / VS LHH plates. Side detection is data-driven (both
    # sides >= 10%% of pitches), so an S-listed hitter who has abandoned a
    # side keeps the single full-width panel.
    from collections import Counter as _SideCounter
    _side_counts = _SideCounter(p.get('Bats') for p in hitter_pitches
                                if p.get('Bats') in ('L', 'R'))
    is_switch = (len(_side_counts) == 2 and
                 min(_side_counts.values()) >= 0.10 * sum(_side_counts.values()))
    _spray_panel_h = spray_axes_top - spray_axes_bottom
    if is_switch:
        _panel_gap = 0.014
        _panel_w = (spray_axes_right - spray_axes_left - _panel_gap) / 2
        _ax_l = fig.add_axes([spray_axes_left, spray_axes_bottom,
                              _panel_w, _spray_panel_h])
        _ax_r = fig.add_axes([spray_axes_left + _panel_w + _panel_gap,
                              spray_axes_bottom, _panel_w, _spray_panel_h])
        # RHH panel on the left, LHH on the right (2026-08-03, per Wally)
        spray_panels = [(_ax_l, 'R', _sacq_full['R'][1], _sacq_full['R'][2]),
                        (_ax_r, 'L', _sacq_full['L'][1], _sacq_full['L'][2])]
        ax_spray = _ax_l          # annotation/legend anchors keep working
    else:
        ax_spray = fig.add_axes([spray_axes_left, spray_axes_bottom,
                                  spray_axes_right - spray_axes_left,
                                  _spray_panel_h])
        spray_panels = [(ax_spray, bats if bats in ('L', 'R') else 'R',
                         hand_zones, pool_zones)]
    for _axp, _s2, _hz2, _pz2 in spray_panels:
        _axp.set_facecolor(BG)
        _axp.set_xlim(-50, 50); _axp.set_ylim(-20, 60)

    LA_RANGES = LA_BINS
    # Damage-view threshold: zones at/above the cmap's yellow anchor count as
    # "damage" (kept warm); cooler zones are left unpainted in the damage view.
    _DAMAGE_THRESH = 0.55
    # Zone overlay / grid / scatter / median all render per panel inside
    # _draw_spray_panel below (once for a one-sided hitter, twice for a
    # switch hitter — each side with its own orientation + zone values).

    # BIP scatter — capture Event so we can outline hits / extra-base hits
    bip_pts = []
    for p in hitter_pitches:
        if p.get('Description') != 'In Play': continue
        bbt = str(p.get('BBType', '')).strip()
        if not bbt or bbt.startswith('bunt'): continue
        la = sf(p.get('LaunchAngle'))
        ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
        ev = sf(p.get('ExitVelo'))
        if la is None or ang is None: continue
        event = p.get('Event', '') or ''
        _bcol = str(p.get('Barrel', '')).strip()
        _brl = (_bcol == '6') if _bcol else bool(is_barrel(ev, la))
        _pside = p.get('Bats') if p.get('Bats') in ('L', 'R') else None
        bip_pts.append((ang, max(-20, min(60, la)), ev, la, event, _brl, _pside))

    # Outcome-based dot coloring (replaces the old EV gradient). Warm-paper
    # palette: gray for non-hits, amber for singles, purple for doubles,
    # teal for triples, crimson for home runs. Size still encodes EV so the
    # chart shows BOTH outcome AND contact quality.
    OUTCOME_COLORS = {
        'Out': '#6e6557',      # muted warm gray (Out/E/FC)
        '1B':  '#e0892b',      # rich amber
        '2B':  '#9a4eaf',      # medium purple (distinct from #5d3b8e Avg Placement)
        '3B':  '#188a8a',      # teal
        'HR':  '#a8261e',      # deep crimson (matches the warm-theme accent family)
    }
    OUTCOME_ALPHA = {'Out': 0.62, '1B': 0.95, '2B': 0.95, '3B': 0.95, 'HR': 0.95}

    def _outcome_category(event):
        if event == 'Single':   return '1B'
        if event == 'Double':   return '2B'
        if event == 'Triple':   return '3B'
        if event == 'Home Run': return 'HR'
        # Everything else in-play: Out / Error / Fielders Choice / Sac
        return 'Out'

    def outcome_color(event):
        cat = _outcome_category(event)
        hex_c = OUTCOME_COLORS[cat]
        a = OUTCOME_ALPHA[cat]
        r = int(hex_c[1:3], 16) / 255
        g = int(hex_c[3:5], 16) / 255
        b = int(hex_c[5:7], 16) / 255
        return (r, g, b, a)

    def ev_size(ev):
        # Bumped up across the board so high-EV dots really pop. The ratio
        # between min (slow grounder) and max (115+ rocket) is wider too,
        # so the eye reads contact quality without needing to check colors.
        # Each tier is roughly 1.7× the previous (was 1.3×).
        if ev is None: return 110
        if ev < 80:  return 110
        if ev < 90:  return 175
        if ev < 95:  return 250
        if ev < 100: return 340
        if ev < 105: return 430
        return 540

    def _draw_spray_panel(ax, side, pts, hand_zones_p, pool_zones_p,
                          med_pair, first_panel=True, header=None):
        """One LA x Spray panel: zone overlay + grid + BIP scatter + median
        marker + axis styling, oriented and valued for `side`. Called once
        for a one-sided hitter, once per side for a switch hitter."""
        if side == 'L':
            bounds = {'pull': (30, 50), 'pull_side': (15, 30), 'center_pull': (0, 15),
                      'center_oppo': (-15, 0), 'oppo_side': (-30, -15), 'oppo': (-50, -30)}
        else:
            bounds = {'pull': (-50, -30), 'pull_side': (-30, -15), 'center_pull': (-15, 0),
                      'center_oppo': (0, 15), 'oppo_side': (15, 30), 'oppo': (30, 50)}
        # Zone fills — hand-specific value if qualified, else pooled
        def _paint(sd, lb, z):
            v = z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
            if v is None or sd not in bounds or lb >= len(LA_RANGES):
                return
            bx = bounds[sd]
            ly = LA_RANGES[lb]
            lo = max(-20, ly[0]); hi = min(60, ly[1])
            if la_view == 'damage':
                if v < _DAMAGE_THRESH:
                    return
                col = WOBA_CMAP(min(1.0, v / 1.0))
                alpha = 0.14 if z.get('count', 0) < 20 else 0.40
            else:
                col = WOBA_CMAP(min(1.0, v / 1.0))
                alpha = 0.22 if z.get('count', 0) < 20 else 0.70
            ax.add_patch(Rectangle((min(bx), lo), abs(bx[1] - bx[0]), hi - lo,
                                    facecolor=col, alpha=alpha,
                                    edgecolor=GRID_COLOR, linewidth=0.3))
        for (sd, lb), z_hand in hand_zones_p.items():
            z_pool = pool_zones_p.get((sd, lb))
            z = z_hand if z_hand.get('count', 0) >= 20 else z_pool
            if z:
                _paint(sd, lb, z)
        for (sd, lb), z in pool_zones_p.items():
            if (sd, lb) not in hand_zones_p:
                _paint(sd, lb, z)

        # Zone-boundary grid overlay (matches the true zone edges)
        _GRID_LW, _GRID_ALPHA = 1.0, 0.75
        for _x in sorted({e for b in bounds.values() for e in b}):
            if -50 < _x < 50:
                ax.axvline(_x, color=GRID_COLOR, linewidth=_GRID_LW,
                           alpha=_GRID_ALPHA, zorder=2)
        for _y in sorted({e for rng in LA_BINS for e in rng}):
            if -20 < _y < 60:
                ax.axhline(_y, color=GRID_COLOR, linewidth=_GRID_LW,
                           alpha=_GRID_ALPHA, zorder=2)

        if la_view == 'damage':
            # DAMAGE VIEW: three contact tiers — barrel (crimson), hard-hit
            # 95+ non-barrel (amber), everything else gray.
            _BARREL_RED  = (0.659, 0.149, 0.118, 0.96)
            _HARDHIT_AMB = (0.878, 0.529, 0.118, 0.93)
            _OTHER_GRAY  = (0.43, 0.40, 0.35, 0.38)
            def _tier(ev, brl):
                if brl:
                    return 2
                if ev is not None and ev >= 95.0:
                    return 1
                return 0
            _TIER_STYLE = {2: (_BARREL_RED, 5), 1: (_HARDHIT_AMB, 4), 0: (_OTHER_GRAY, 3)}
            from matplotlib.patheffects import withStroke
            for x, y, ev, _real, event, _brl, _ps in sorted(pts, key=lambda r: _tier(r[2], r[5])):
                _c, _z = _TIER_STYLE[_tier(ev, _brl)]
                ax.scatter([x], [y], s=ev_size(ev), c=[_c],
                           edgecolors='#f0e8d8', linewidths=1.1, zorder=_z,
                           path_effects=[withStroke(linewidth=1.9, foreground='#1a1612')])
            # Quality-contact concentration ellipse (barrels + hard-hit 95+)
            from matplotlib.patches import Ellipse as _Ellipse
            _qx = [q[0] for q in pts if q[5] or (q[2] is not None and q[2] >= 95)]
            _qy = [q[1] for q in pts if q[5] or (q[2] is not None and q[2] >= 95)]
            if len(_qx) >= 6:
                _cvm = np.cov(_qx, _qy)
                _vals, _vecs = np.linalg.eigh(_cvm)
                _o = _vals.argsort()[::-1]
                _vals, _vecs = _vals[_o], _vecs[:, _o]
                if _vals[0] > 0 and _vals[1] > 0:
                    _th = np.degrees(np.arctan2(_vecs[1, 0], _vecs[0, 0]))
                    _w, _h = 2.0 * 1.51 * np.sqrt(_vals)   # 1.51 sigma ~ 68%
                    ax.add_patch(_Ellipse((np.mean(_qx), np.mean(_qy)), _w, _h,
                                          angle=_th, facecolor='none',
                                          edgecolor='#9a3b1e', linewidth=2.0,
                                          linestyle=(0, (5, 3)), zorder=2.6))
        else:
            # OUTCOME VIEW: outs first (back), then hits by weight
            _OUTCOME_Z = {'Out': 0, '1B': 1, '2B': 2, '3B': 3, 'HR': 4}
            for x, y, ev, _real, event, _brl, _ps in sorted(pts,
                                                  key=lambda r: _OUTCOME_Z[_outcome_category(r[4])]):
                cat = _outcome_category(event)
                ax.scatter([x], [y], s=ev_size(ev), c=[outcome_color(event)],
                           edgecolors='#1a1612', linewidths=0.6,
                           zorder=3 + _OUTCOME_Z[cat])

        # Median placement marker
        _ms, _ml = med_pair
        if _ms is not None and _ml is not None:
            _mlp = max(-20, min(60, _ml))
            ax.scatter([_ms], [_mlp], s=420 if first_panel or not is_switch else 300,
                       c='white', zorder=10, alpha=0.95,
                       edgecolors='black', linewidths=0.5)
            ax.scatter([_ms], [_mlp], s=240 if first_panel or not is_switch else 170,
                       c=MARKER_ACCENT, edgecolors='black',
                       linewidths=2, zorder=11)

        # Axis styling — orientation labels follow the panel's side
        ax.set_xticks([-30, -15, 0, 15, 30] if is_switch
                      else [-50, -30, -15, 0, 15, 30, 50])
        ax.set_yticks(range(-20, 61, 10))
        ax.tick_params(colors=TICK_COLOR, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(SPINE_COLOR)
        ax.grid(False)
        leftL = ('Oppo' if side == 'L' else 'Pull')
        rightL = ('Pull' if side == 'L' else 'Oppo')
        # Arrows point the way each direction actually runs on the axis, so the
        # reader never has to work out which side is pull for this handedness
        # (leftL/rightL already swap by side, the arrows just follow them).
        # Arrows point the way each direction actually runs on the axis, so the
        # reader never has to work out which side is pull for this handedness
        # (leftL/rightL already swap by side, the arrows just follow them).
        # Drawn as mathtext: IBM Plex Sans has no U+2190/U+2192 glyph, so a
        # literal arrow renders as tofu.
        ax.set_xlabel(rf'$\leftarrow$ {leftL}   •   Spray Angle   •   {rightL} $\rightarrow$',
                      color=TEXT_MUTED, fontsize=10, fontfamily='IBM Plex Sans')
        if first_panel:
            ax.set_ylabel('Launch Angle', color=TEXT_MUTED, fontsize=10,
                          fontfamily='IBM Plex Sans')
        else:
            ax.set_yticklabels([])
        if header:
            ax.set_title(header, color=TEXT_MUTED, fontsize=12,
                         fontweight='700', fontfamily='IBM Plex Sans', pad=5)

    def _xwobasp_of(pts, side):
        """xwOBAsp for one panel's batted balls, scored in that side's zones.

        Every point in `pts` already has Bats == side, so this is the same
        arithmetic as the combined loop below with the side held fixed. The
        two per-side values reblend to the combined value by qualifying-BIP
        count. Verified on the four highest-PA switch hitters of 2026: each
        side's pair reblended to the card's combined value exactly, and that
        combined value matched the shipped leaderboard xwOBAsp to within
        display rounding (max delta .0005).

        There is no leaderboard equivalent to fall back on: the hitter rows
        carry xwOBAsp and xwOBAsp_pctl only, with no per-side split. So this
        is computed from the pickle, and it carries no percentile.
        """
        _sum = 0.0
        _n = 0
        for _ang, _yc2, _ev2, _la2, _evt2, _brl2, _ps2 in pts:
            _sd2 = spray_direction(_ang, side)
            _lb2 = la_bin_idx(_la2) if _la2 is not None else None
            _v2 = sacq_lookups[side](_sd2, _lb2)
            if _v2 is not None:
                _sum += _v2
                _n += 1
        return (_sum / _n) if _n else None

    def _median_of(pts):
        if not pts:
            return (None, None)
        ss = sorted(q[0] for q in pts)
        ls = sorted(q[3] for q in pts)
        n2 = len(pts)
        m2 = n2 // 2
        if n2 % 2 == 0:
            return ((ss[m2 - 1] + ss[m2]) / 2, (ls[m2 - 1] + ls[m2]) / 2)
        return (ss[m2], ls[m2])

    # Median placement marker — prefer the values stored on the leaderboard
    # row (h_row['medLA'] and h_row['medSpray']). Those are computed by the
    # pipeline from the source-of-truth sheet data and stay in sync with the
    # website's "Avg Placement" annotation. Falls back to recomputing from
    # the local pickle ONLY for whichever value is missing (older pipeline
    # output) — never overwrite a value that already came from the JSON.
    med_spray = sf(h_row.get('medSpray'))
    med_la_real = sf(h_row.get('medLA'))
    if (med_spray is None or med_la_real is None) and bip_pts:
        sorted_sprays = sorted(p[0] for p in bip_pts)
        sorted_las = sorted(p[3] for p in bip_pts)
        n_pts = len(bip_pts)
        mid = n_pts // 2
        if n_pts % 2 == 0:
            pickle_spray = (sorted_sprays[mid - 1] + sorted_sprays[mid]) / 2
            pickle_la = (sorted_las[mid - 1] + sorted_las[mid]) / 2
        else:
            pickle_spray = sorted_sprays[mid]
            pickle_la = sorted_las[mid]
        if med_spray is None:    med_spray = pickle_spray
        if med_la_real is None:  med_la_real = pickle_la
    # Render the panel(s). Switch hitters: each side gets its own BIP, its
    # own hand-oriented + hand-valued overlay, its own median marker
    # (leaderboard medSpray/medLA mix mirrored sprays for an S hitter, so
    # per-side medians are recomputed from the panel's own points).
    for _pi, (_axp, _s2, _hz2, _pz2) in enumerate(spray_panels):
        if is_switch:
            _pts2 = [q for q in bip_pts if q[6] == _s2]
            _med2 = _median_of(_pts2)
            # Each panel carries its OWN xwOBAsp. The combined value in the
            # top-left annotation blends the two sides, which can hide a wide
            # split: Marte 2026 read .356 as LHH against .412 as RHH behind a
            # blended .370. The value sits next to the BIP count it came from.
            _sp2 = _xwobasp_of(_pts2, _s2)
            _side2 = 'AS LHH (VS RHP)' if _s2 == 'L' else 'AS RHH (VS LHP)'
            _hdr2 = (f"{_side2}  ·  xwOBAsp {fmt_3dec(_sp2)}  ·  {len(_pts2)} BIP"
                     if _sp2 is not None
                     else f"{_side2}  ·  {len(_pts2)} BIP")
        else:
            _pts2 = bip_pts
            _med2 = (med_spray, med_la_real)
            _hdr2 = None
        _draw_spray_panel(_axp, _s2, _pts2, _hz2, _pz2, _med2,
                          first_panel=(_pi == 0), header=_hdr2)

    # xwOBAsp (hitter overall, hand-specific zones with pooled fallback).
    # Each BIP is scored by its ACTUAL side — matters for switch hitters.
    xwobasp_sum = 0; xwobasp_n = 0; total_bip = 0
    for ang, _yc, _ev, la_r, _event, _brl, _ps in bip_pts:
        total_bip += 1
        _b2 = _ps if (_ps and is_switch) else bats
        sd = spray_direction(ang, _b2)
        lb = la_bin_idx(la_r) if la_r is not None else None
        v = (sacq_lookups[_b2] if is_switch and _ps else sacq_lookup)(sd, lb)
        if v is not None:
            xwobasp_sum += v
            xwobasp_n += 1
    xwobasp = xwobasp_sum / xwobasp_n if xwobasp_n else None

    # Annotations placed TOP-LEFT of chart panel (matches hitter page).
    # Each line is one consolidated fig.text() so spacing is correct regardless
    # of font metrics. Color emphasis applied via second-pass overlay.
    annot_x = spray_axes_left + 0.005   # left edge, just inside panel
    # Use leaderboard's xwOBAsp value + percentile for consistency with website
    xwobasp_display = h_row.get('xwOBAsp', xwobasp)
    pctl = h_row.get('xwOBAsp_pctl')

    # PRINT-IDENTITY annotation color. The light paper-blend percentile tint is
    # a background color — unreadable as TEXT on cream — so this value/placement
    # text instead blends from warm-gray ink toward a DEEP version of the
    # percentile hue (deep terracotta for good, deep slate for bad). Stays
    # readable on paper while carrying the same good/bad hue as the bubbles.
    def _pctl_color_dark(p):
        if p is None: return TEXT_MUTED
        neutral = (0x6a, 0x5f, 0x55)                 # TEXT_MUTED (warm ink-gray)
        target = (0x9f, 0x30, 0x26) if p >= 50 else (0x2f, 0x55, 0x73)
        t = (abs(p - 50) / 50.0) ** 1.1
        r = int(round(neutral[0] + (target[0] - neutral[0]) * t))
        g = int(round(neutral[1] + (target[1] - neutral[1]) * t))
        b = int(round(neutral[2] + (target[2] - neutral[2]) * t))
        return f'#{r:02x}{g:02x}{b:02x}'

    pctl_color = _pctl_color_dark(pctl)

    if xwobasp_display is not None:
        # Note format: "(95th percentile, 58 qualifying BIP)"
        # If qualifying < total, show "X of Y qualifying BIP"
        bip_str = (f"{xwobasp_n} of {total_bip} qualifying BIP"
                   if total_bip > xwobasp_n else f"{xwobasp_n} qualifying BIP")
        if pctl is not None:
            pctl_int = int(round(pctl))
            # Pick suffix (1st, 2nd, 3rd, 4th-9th, 10th-19th)
            if 11 <= (pctl_int % 100) <= 13:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(pctl_int % 10, 'th')
            bipnote = (f"({pctl_int}{suffix} percentile, both sides, {bip_str})"
                       if is_switch
                       else f"({pctl_int}{suffix} percentile, {bip_str})")
        else:
            bipnote = (f"(both sides, {bip_str})" if is_switch
                       else f"({bip_str})")
        l1_y = 0.955 - _LA_SHIFT
        # Approach: render label first (gray), measure its width via the
        # renderer, place value right after, measure that, place bipnote.
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()

        def _place_text(x, y, txt, fontsize, color, fontweight):
            t = fig.text(x, y, txt, color=color, fontsize=fontsize,
                          fontweight=fontweight, fontfamily='IBM Plex Sans',
                          va='center', ha='left')
            bbox = t.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(inv)
            return bbox_fig.x1  # right edge in figure fraction coords

        right = _place_text(annot_x, l1_y, 'xwOBAsp: ', 18, TEXT_MUTED, '600')
        # Value uses percentile-derived color (matches hitter page)
        right = _place_text(right, l1_y, fmt_3dec(xwobasp_display), 18, pctl_color, '700')
        right += 0.005  # tiny gap before bipnote
        _place_text(right, l1_y, bipnote, 11, TEXT_FAINT, '600')

    if not is_switch and med_spray is not None and med_la_real is not None:
        sd = ('Pull' if med_spray > 0 else 'Oppo') if bats == 'L' else \
             ('Pull' if med_spray < 0 else 'Oppo')
        l2_y = 0.930 - _LA_SHIFT
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()

        def _place_text2(x, y, txt, fontsize, color, fontweight):
            t = fig.text(x, y, txt, color=color, fontsize=fontsize,
                          fontweight=fontweight, fontfamily='IBM Plex Sans',
                          va='center', ha='left')
            bbox = t.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(inv)
            return bbox_fig.x1

        right = _place_text2(annot_x, l2_y, 'Avg Placement: ', 14, TEXT_MUTED, '600')
        # Both spray-direction value and LA value share xwOBAsp percentile color
        # (matches hitter page — links Avg Placement visually to xwOBAsp)
        right = _place_text2(right, l2_y, f"{abs(med_spray):.1f}° {sd}",
                              14, pctl_color, '700')
        right += 0.006
        right = _place_text2(right, l2_y, '|', 13, TEXT_FAINT, '600')
        right += 0.006
        _place_text2(right, l2_y, f"{med_la_real:.1f}° LA",
                       14, pctl_color, '700')

    # Section title (centered ABOVE the annotation block, matches hitter page)
    # Editorial-style title: letterspaced uppercase, off-white, with thin
    # underline rule. Same treatment used for all section titles below.
    title_x = (spray_axes_left + spray_axes_right) / 2
    fig.text(title_x, 0.985 - _LA_SHIFT, 'L A U N C H   A N G L E   ×   S P R A Y   A N G L E',
              color=TEXT_SECONDARY, fontsize=_LA_TITLE_FONTSIZE, fontweight='700',
              fontfamily='IBM Plex Sans', va='center', ha='center')
    # Thin underline rule — slightly wider with the bigger title
    rule_w = 0.27
    ax_rule = fig.add_axes([title_x - rule_w / 2, 0.978 - _LA_SHIFT, rule_w, 0.0008])
    ax_rule.axis('off')
    ax_rule.add_patch(Rectangle((0, 0), 1, 1, facecolor=TEXT_SECONDARY,
                                  edgecolor='none', alpha=0.30,
                                  transform=ax_rule.transAxes))

    # ─── Heat map + per-group bar geometry (computed up front so both
    #     sections can reference shared boundaries) ─────────────────
    # Heat maps fill the middle band; top lifted to give panels more height.
    # Panels rendered ~square so the 4×4 ft data range stays proportional.
    # Bottom raised to 0.20 to make room for the heat-map color-scale legend
    # (single compact row) in the band between heat maps and Contact Profile.
    zhm_y_bottom = 0.20
    zhm_y_top = 0.70
    panel_h = (zhm_y_top - zhm_y_bottom) / 2 - 0.025
    panel_h_in = panel_h * FIG_H
    panel_w_in = panel_h_in
    panel_w = panel_w_in / FIG_W
    zhm_x_left = 0.012
    zhm_x_right = zhm_x_left + 2 * panel_w + 0.014

    # ─── LA × Spray legend (matches hitter page exactly) ───────────
    # Two rows beneath the LA × Spray chart:
    #   Row 1: 70 mph (blue) · 95 mph (gray) · 115 mph (red) · Size = EV
    #   Row 2: MLB wOBAcon: .000 [gradient bar] 1.000+
    from matplotlib.patches import Circle
    legend_center_x = (spray_axes_left + spray_axes_right) / 2
    # Larger gaps below xlabel and between rows so the legend doesn't
    # feel crammed together with the chart's axis label.
    legend_y_dots = spray_axes_bottom - 0.045    # clear of xlabel
    legend_y_grad = spray_axes_bottom - 0.090    # generous gap from dots

    # Dot colors now encode outcome (not EV). Legend uses the same warm-paper
    # palette as the BIP scatter above. Each tuple is (label, color, cat-key)
    # so we can filter out categories the hitter has zero of (e.g. no 3B for
    # a player with no triples) — no point listing what isn't on the chart.
    if la_view == 'damage':
        OUTCOME_LEGEND = [
            ('Barrel',       '#a8261e', 'brl'),
            ('Hard-Hit 95+', '#e0871e', 'hh'),
            ('Other BIP',    '#6e6557', 'oth'),
        ]
    else:
        OUTCOME_LEGEND_ALL = [
            ('Out / E / FC', OUTCOME_COLORS['Out'], 'Out'),
            ('1B',           OUTCOME_COLORS['1B'],  '1B'),
            ('2B',           OUTCOME_COLORS['2B'],  '2B'),
            ('3B',           OUTCOME_COLORS['3B'],  '3B'),
            ('HR',           OUTCOME_COLORS['HR'],  'HR'),
        ]
        _present_cats = {_outcome_category(p[4]) for p in bip_pts}
        OUTCOME_LEGEND = [item for item in OUTCOME_LEGEND_ALL
                           if item[2] in _present_cats]
    LEGEND_FONTSIZE = 13
    DOT_RADIUS = 0.0065
    ITEM_GAP = 0.026

    renderer_l = fig.canvas.get_renderer()
    inv_l = fig.transFigure.inverted()

    def _measure_text(txt, fontsize, fontweight='600'):
        t = fig.text(0, 0, txt, color='#000', fontsize=fontsize,
                      fontweight=fontweight, fontfamily='IBM Plex Sans',
                      va='center', ha='left')
        w = t.get_window_extent(renderer=renderer_l).transformed(inv_l).width
        t.remove()
        return w

    def _draw_text(x, y, txt, fontsize, color, fontweight='600'):
        t = fig.text(x, y, txt, color=color, fontsize=fontsize,
                      fontweight=fontweight, fontfamily='IBM Plex Sans',
                      va='center', ha='left')
        return t.get_window_extent(renderer=renderer_l).transformed(inv_l).x1

    def _draw_dot(x, y, color, radius=DOT_RADIUS):
        c = Circle((x, y), radius,
                    facecolor=color, edgecolor='none',
                    transform=fig.transFigure, figure=fig)
        fig.add_artist(c)

    def _draw_outlined_dot(x, y, edge_color, edge_width, radius=DOT_RADIUS):
        # Neutral gray fill so the outline reads clearly; matches the visual
        # treatment of a typical mid-EV BIP dot.
        c = Circle((x, y), radius,
                    facecolor=PERCENTILE_NEUTRAL,
                    edgecolor=edge_color, linewidth=edge_width,
                    transform=fig.transFigure, figure=fig)
        fig.add_artist(c)

    # Row 1 — pre-measure entire row to center it. Includes:
    #   N outcome color dots + Size = EV + Avg Placement
    # The Avg Placement swatch is a halo-style two-circle marker (white
    # outer + purple inner) so we use a larger effective radius for its
    # slot in the layout. Matches the chart's median marker.
    AVG_HALO_RADIUS = DOT_RADIUS * 1.35

    # Vertical separator between the outcome swatches (Out/1B/2B/3B/HR —
    # what the dot's color means) and the visual-encoding swatches
    # (Size = EV, Avg Placement — what the dot's size/shape means). The
    # gap on each side of the divider is half the regular ITEM_GAP to
    # signal "section break" rather than "another item in the list".
    DIV_HALF_GAP = ITEM_GAP * 0.55
    DIV_W = 0.0012   # axis-x width of the divider (thin)
    DIV_HEIGHT = 0.018   # axis-y; matches roughly 1.5x the dot diameter

    row1_w = 0.0
    for label, _, _cat in OUTCOME_LEGEND:
        row1_w += 2 * DOT_RADIUS + 0.006 + _measure_text(label, LEGEND_FONTSIZE) + ITEM_GAP
    # Replace the trailing ITEM_GAP after the last outcome with the
    # half-gap + divider + half-gap pattern.
    row1_w -= ITEM_GAP
    row1_w += DIV_HALF_GAP + DIV_W + DIV_HALF_GAP
    row1_w += 2 * DOT_RADIUS + 0.006 + _measure_text('Size = EV', LEGEND_FONTSIZE) + ITEM_GAP
    row1_w += 2 * AVG_HALO_RADIUS + 0.006 + _measure_text('Avg Placement', LEGEND_FONTSIZE)
    cur_x = legend_center_x - row1_w / 2

    for i, (label, hexcolor, _cat) in enumerate(OUTCOME_LEGEND):
        _draw_dot(cur_x + DOT_RADIUS, legend_y_dots, hexcolor)
        cur_x += 2 * DOT_RADIUS + 0.006
        cur_x = _draw_text(cur_x, legend_y_dots, label, LEGEND_FONTSIZE, PERCENTILE_NEUTRAL)
        # Trailing gap after each outcome — but the LAST outcome uses the
        # half-gap pattern around the divider instead.
        if i < len(OUTCOME_LEGEND) - 1:
            cur_x += ITEM_GAP

    # Divider: thin vertical rule centered on the legend row.
    # Added to fig (not an axes) so the figure-relative transform is
    # consistent with the dots and text in this row.
    cur_x += DIV_HALF_GAP
    divider = Rectangle(
        (cur_x, legend_y_dots - DIV_HEIGHT / 2),
        DIV_W, DIV_HEIGHT,
        facecolor=TEXT_FAINT, edgecolor='none', alpha=0.6,
        transform=fig.transFigure, figure=fig, zorder=10,
    )
    fig.add_artist(divider)
    cur_x += DIV_W + DIV_HALF_GAP

    _draw_dot(cur_x + DOT_RADIUS * 0.85, legend_y_dots, TEXT_DIMMED,
              radius=DOT_RADIUS * 0.85)
    cur_x += 2 * DOT_RADIUS + 0.006
    cur_x = _draw_text(cur_x, legend_y_dots, 'Size = EV', LEGEND_FONTSIZE, TEXT_DIMMED)
    cur_x += ITEM_GAP

    # Avg Placement marker — two concentric circles matching the chart's
    # median marker: outer white halo with thin black edge, inner purple
    # core with heavier black edge. Center positioned so the OUTER circle
    # occupies the same slot width as a regular dot would at AVG_HALO_RADIUS.
    avg_cx = cur_x + AVG_HALO_RADIUS
    halo = Circle((avg_cx, legend_y_dots), AVG_HALO_RADIUS,
                   facecolor='white', edgecolor='black', linewidth=0.6,
                   transform=fig.transFigure, figure=fig, zorder=10)
    fig.add_artist(halo)
    core = Circle((avg_cx, legend_y_dots), DOT_RADIUS,
                   facecolor=MARKER_ACCENT, edgecolor='black', linewidth=1.4,
                   transform=fig.transFigure, figure=fig, zorder=11)
    fig.add_artist(core)
    cur_x += 2 * AVG_HALO_RADIUS + 0.006
    _draw_text(cur_x, legend_y_dots, 'Avg Placement', LEGEND_FONTSIZE, TEXT_DIMMED)

    # Row 2 — MLB wOBAcon gradient (label + .000 + bar + 1.000+)
    grad_label = 'MLB wOBAcon: '
    grad_low = '.000'
    grad_high = '1.000+'
    grad_bar_w = 0.22
    grad_bar_h = 0.014
    lbl_w = _measure_text(grad_label, LEGEND_FONTSIZE)
    low_w = _measure_text(grad_low, LEGEND_FONTSIZE)
    high_w = _measure_text(grad_high, LEGEND_FONTSIZE)
    total_w = lbl_w + 0.005 + low_w + 0.006 + grad_bar_w + 0.006 + high_w
    cur = legend_center_x - total_w / 2
    cur = _draw_text(cur, legend_y_grad, grad_label, LEGEND_FONTSIZE, TEXT_DIMMED)
    cur += 0.005
    cur = _draw_text(cur, legend_y_grad, grad_low, LEGEND_FONTSIZE, TEXT_DIMMED)
    cur += 0.006
    bar_left_x = cur
    ax_grad = fig.add_axes([bar_left_x, legend_y_grad - grad_bar_h / 2,
                              grad_bar_w, grad_bar_h])
    ax_grad.set_facecolor(BG)
    grad_arr = np.linspace(0, 1, 256).reshape(1, -1)
    ax_grad.imshow(grad_arr, aspect='auto', cmap=WOBA_CMAP,
                    extent=[0, 1, 0, 1], origin='lower')
    ax_grad.set_xticks([]); ax_grad.set_yticks([])
    for s in ax_grad.spines.values(): s.set_visible(False)
    cur = bar_left_x + grad_bar_w + 0.006
    _draw_text(cur, legend_y_grad, grad_high, LEGEND_FONTSIZE, TEXT_DIMMED)

    # ═════════════════════════════════════════════════════════════════
    # LAYOUT BRANCH: 'bubbles' variant
    # Drops heat maps, contact profile, and pitch group table; renders a
    # 4-column percentile bubble grid in the left half. Keeps headline +
    # LA × Spray exactly as the classic layout. Early-returns after save
    # so the classic-layout code below is untouched.
    # ═════════════════════════════════════════════════════════════════
    if layout == 'bubbles':
        _render_percentile_bubbles(fig, h_row)

        # Watermark — bottom-right corner.
        # Reader's notes — the same job the pitcher card's below-table block
        # does. Each line exists because the number above it is either
        # directionally ambiguous, freshly rescaled, or luck-laden, and a
        # reader has no way to know that from the bubble alone.
        # Ordered to follow the rail above: RESULT's "+" family, then Quality of
        # Contact's GB%, then Plate Discipline's Swing%. Bulleted so each note
        # reads as its own item rather than one running paragraph.
        if window_pool:
            # Pooled window card: everything the season card shows is present,
            # measured over the window. Say what the percentiles rank against,
            # because that is the one thing a reader cannot infer.
            _notes = (
                '•  Hitter+, Batted Ball+, Contact+ and Swing Decisions+: 100 is league '
                'average and each point is 1% better or worse, the way wRC+ reads.\n'
                '•  GB% is colored so that lower = better.\n'
                '•  Values are for this date window. Percentiles rank them against '
                'the full-season MLB pool, with no minimum sample.'
            )
        elif date_filter is not None:
            # Window card with no pool. Say plainly which numbers are missing
            # and why the bubbles are grey, so a blank cell is not read as zero.
            _notes = (
                '•  Every value on this card is computed from this date '
                'window\'s pitches only.\n'
                '•  Hitter+, Batted Ball+, Contact+, Swing Decisions+ and '
                'wRC+ need the window pool, so they read "—".\n'
                '•  Percentile bars are grey: a percentile needs an all-MLB '
                'pool measured over the same window.'
            )
        else:
            _notes = (
                '•  Hitter+, Batted Ball+, Contact+ and Swing Decisions+: 100 is league '
                'average and each point is 1% better or worse, the way wRC+ reads.\n'
                '•  GB% is colored so that lower = better.\n'
                '•  Z-Swing% and Z-Contact% count in-zone pitches only: swings at '
                'strikes, and contact on those swings.'
            )
        fig.text(0.020, 0.068, _notes, fontsize=13, color=TEXT_MUTED, va='top',
                 ha='left', fontfamily='IBM Plex Sans', fontweight='600',
                 linespacing=1.5)

        fig.text(0.99, 0.012, 'huronalytics.vercel.app', color=TEXT_PRIMARY,
                  fontsize=9, fontfamily='IBM Plex Sans', ha='right',
                  va='bottom', style='italic')

        # Save with _bubbles suffix so the two layouts can coexist.
        safe_name = display_name.replace(' ', '_').replace('.', '').replace(',', '')
        out_path = os.path.join(output_dir,
                                  f'HitterCard_{safe_name}_'
                                  f'{date_slug or year_label.replace(" ", "_")}'
                                  f'_bubbles.png')
        plt.savefig(out_path, dpi=SAVE_DPI, facecolor=BG, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return True

    # No section title for heat maps — the four panel titles
    # ("WHIFFS vs RHP — 33.9%", etc.) already label the section.

    # ─── Zone heat maps: 2 rows × 2 cols (geometry already computed) ─
    ZONE_X_MIN, ZONE_X_MAX = -2.0, 2.0
    ZONE_Z_MIN, ZONE_Z_MAX = 0.5, 4.5

    SWING_ALL = set(SWING_DESC)
    for hi, hand in enumerate(['R', 'L']):  # row index
        # Build the two underlying point sets per handedness, used for both
        # the panel subtitles AND the rate-KDE numerator/denominator.
        swing_pts = []          # all swing locations (denominator for WHIFFS)
        whiff_pts = []          # swinging-strike locations (numerator for WHIFFS)
        bip_pts_loc = []        # BIP locations (denominator for DAMAGE)
        bip_xw_weights = []     # BIP xwOBA values (MLB DAMAGE weights)
        bip_ev_weights = []     # BIP ExitVelo values (ROC DAMAGE weights)
        for p in hitter_pitches:
            if p.get('Throws') != hand: continue
            px = sf(p.get('PlateX')); pz = sf(p.get('PlateZ'))
            if px is None or pz is None: continue
            desc = p.get('Description', '')
            if desc in SWING_ALL:
                swing_pts.append((px, pz))
                if desc == 'Swinging Strike':
                    whiff_pts.append((px, pz))
            if desc == 'In Play' and not str(p.get('BBType', '')).startswith('bunt'):
                if is_roc:
                    # ROC: weight by ExitVelo (xwOBA isn't tagged for AAA)
                    ev = sf(p.get('ExitVelo'))
                    if ev is not None:
                        bip_pts_loc.append((px, pz))
                        bip_ev_weights.append(ev)
                else:
                    xw = sf(p.get('xwOBA'))
                    if xw is not None:
                        bip_pts_loc.append((px, pz))
                        bip_xw_weights.append(xw)

        # Aggregate metrics for the panel subtitle
        n_swings_hand = len(swing_pts)
        n_whiffs_hand = len(whiff_pts)
        whiff_pct_hand = (n_whiffs_hand / n_swings_hand) if n_swings_hand else None
        xwobacon_hand = (sum(bip_xw_weights) / len(bip_xw_weights)) if bip_xw_weights else None
        avg_ev_hand = (sum(bip_ev_weights) / len(bip_ev_weights)) if bip_ev_weights else None

        for ci, mode in enumerate(['whiffs', 'damage']):  # col index
            ax = fig.add_axes([zhm_x_left + ci * (panel_w + 0.01),
                                zhm_y_top - (hi + 1) * panel_h - hi * 0.02,
                                panel_w, panel_h])
            hand_label = 'RHP' if hand == 'R' else 'LHP'
            if mode == 'whiffs':
                # WHIFFS rate KDE — numerator = whiffs, denominator = swings.
                # Color anchored to MLB whiff% via scale_max=0.6.
                if whiff_pct_hand is not None:
                    label = (f"WHIFFS vs {hand_label} · "
                             f"{whiff_pct_hand * 100:.1f}%")
                    sample = f"{n_whiffs_hand} of {n_swings_hand} swings"
                else:
                    label = f"WHIFFS vs {hand_label}"
                    sample = f"n = {n_whiffs_hand}"
                render_rate_kde_to_axes(ax,
                    whiff_pts, None,           # num: whiffs (uniform weight)
                    swing_pts, None,           # den: all swings (uniform weight)
                    ZONE_X_MIN, ZONE_X_MAX, ZONE_Z_MIN, ZONE_Z_MAX,
                    sz_top, sz_bot, label, sample,
                    scale_max=0.6, mask_threshold=0.02, min_n=5)
            else:
                # DAMAGE rate KDE — for MLB: xwOBA-weighted BIPs gives mean
                # xwOBAcon per location. For ROC: ExitVelo-weighted gives mean
                # Avg EV per location (xwOBA isn't tagged for AAA).
                if is_roc:
                    if avg_ev_hand is not None:
                        label = (f"DAMAGE vs {hand_label} · "
                                 f"Avg EV {avg_ev_hand:.1f} mph")
                        sample = f"{len(bip_pts_loc)} BIP"
                    else:
                        label = f"DAMAGE vs {hand_label}"
                        sample = f"n = {len(bip_pts_loc)}"
                    # ExitVelo: clamp range [50, 110] mph. MLB avg EV ~88 mph
                    # falls at (88-50)/60 = 0.63 along the cmap → warm.
                    render_rate_kde_to_axes(ax,
                        bip_pts_loc, bip_ev_weights,
                        bip_pts_loc, None,
                        ZONE_X_MIN, ZONE_X_MAX, ZONE_Z_MIN, ZONE_Z_MAX,
                        sz_top, sz_bot, label, sample,
                        scale_max=110.0, scale_min=50.0,
                        mask_threshold=0.05, min_n=5)
                else:
                    if xwobacon_hand is not None:
                        label = (f"DAMAGE vs {hand_label} · "
                                 f"xwOBAcon {fmt_3dec(xwobacon_hand)}")
                        sample = f"{len(bip_pts_loc)} BIP"
                    else:
                        label = f"DAMAGE vs {hand_label}"
                        sample = f"n = {len(bip_pts_loc)}"
                    render_rate_kde_to_axes(ax,
                        bip_pts_loc, bip_xw_weights,
                        bip_pts_loc, None,
                        ZONE_X_MIN, ZONE_X_MAX, ZONE_Z_MIN, ZONE_Z_MAX,
                        sz_top, sz_bot, label, sample,
                        scale_max=1.0, mask_threshold=0.05, min_n=5)

    # ─── Heat-map color-scale legend ──────────────────────────────
    # Compact side-by-side scales sharing HEAT_CMAP but different scale_max:
    # red on WHIFFS = 60%+ whiff rate, red on DAMAGE = 1.000+ xwOBAcon.
    # Each scale takes ~half the heat-maps width so they sit on a single line.
    HM_BAR_H = 0.010
    HM_FONT  = 9

    renderer_hm = fig.canvas.get_renderer()
    inv_hm = fig.transFigure.inverted()

    def _hm_measure(txt, fontsize=HM_FONT, fontweight='600'):
        t = fig.text(0, 0, txt, color='#000', fontsize=fontsize,
                      fontweight=fontweight, fontfamily='IBM Plex Sans',
                      va='center', ha='left')
        w = t.get_window_extent(renderer=renderer_hm).transformed(inv_hm).width
        t.remove()
        return w

    # NOTE: 700 is the ceiling for IBM Plex Sans / Plex Sans Condensed —
    # the family ships no 800 face (assets/fonts/ has 400-700). Asking for
    # 800 made matplotlib fall back to 700 anyway and print a findfont
    # warning per call; the weights below are the heaviest available, so
    # this is the same rendering without the noise. Bitter is the only
    # bundled family with a heavier face (900) if a card ever needs one.
    def _hm_text(x, y, txt, color, fontsize=HM_FONT, fontweight='600'):
        t = fig.text(x, y, txt, color=color, fontsize=fontsize,
                      fontweight=fontweight, fontfamily='IBM Plex Sans',
                      va='center', ha='left')
        return t.get_window_extent(renderer=renderer_hm).transformed(inv_hm).x1

    def _hm_legend_row(x_left, x_right, y_center, label, low_str, high_str,
                       mlb_pos=None):
        """Inline legend: 'LABEL low [bar] high' within [x_left, x_right].
        mlb_pos: optional [0, 1] position on the bar to draw a white tick +
        small 'MLB' label, anchoring the absolute scale."""
        x = x_left
        x = _hm_text(x, y_center, label, TEXT_MUTED, fontweight='700')
        x += 0.005
        x = _hm_text(x, y_center, low_str, TEXT_SECONDARY)
        x += 0.003
        high_w = _hm_measure(high_str)
        bar_right = x_right - high_w - 0.004
        bar_w = bar_right - x
        if bar_w > 0.02:
            ax_g = fig.add_axes([x, y_center - HM_BAR_H / 2,
                                  bar_w, HM_BAR_H])
            ax_g.set_facecolor(BG)
            grad_arr = np.linspace(0, 1, 256).reshape(1, -1)
            ax_g.imshow(grad_arr, aspect='auto', cmap=HEAT_CMAP,
                         extent=[0, 1, 0, 1], origin='lower')
            ax_g.set_xticks([]); ax_g.set_yticks([])
            for s in ax_g.spines.values(): s.set_visible(False)
            # MLB anchor: thin white tick across the bar + tiny 'MLB' label below
            if mlb_pos is not None:
                ax_g.axvline(mlb_pos, color=GRID_COLOR, linewidth=1.1,
                              alpha=0.95, zorder=5)
                tick_fig_x = x + bar_w * mlb_pos
                fig.text(tick_fig_x, y_center - HM_BAR_H / 2 - 0.006, 'MLB',
                          color=TEXT_SECONDARY, fontsize=8, fontweight='700',
                          fontfamily='IBM Plex Sans', va='top', ha='center')
        _hm_text(bar_right + 0.003, y_center, high_str, TEXT_SECONDARY)

    # Side-by-side: WHIFFS (left) + DAMAGE (right), with MLB-avg tick on each.
    # MLB whiff rate ≈ 25.2% → 0.252/0.6 = 0.420 of bar
    # MLB xwOBAcon  ≈ .370   → 0.370/1.0 = 0.370 of bar
    hm_mid = (zhm_x_left + zhm_x_right) / 2
    legend_y = 0.215
    _hm_legend_row(zhm_x_left + 0.005, hm_mid - 0.008, legend_y,
                    'WHIFF RATE', '0%', '60%+', mlb_pos=0.420)
    # ROC: DAMAGE legend shows EV (50-110 mph) with MLB tick at 88 mph
    # → (88-50)/60 = 0.633 along the bar. MLB: xwOBAcon (.000-1.000+) with
    # MLB tick at .370 → 0.370.
    if is_roc:
        _hm_legend_row(hm_mid + 0.008, zhm_x_right - 0.005, legend_y,
                        'Avg EV', '50', '110+ mph', mlb_pos=0.633)
    else:
        _hm_legend_row(hm_mid + 0.008, zhm_x_right - 0.005, legend_y,
                        'xwOBAcon', '.000', '1.000+', mlb_pos=0.370)

    # ─── Contact Profile strip (7 cells, proportional widths) ──────
    # Skipped entirely for ROC — bat-tracking metrics aren't recorded for
    # AAA, and the three available (Avg EV / Max EV / Air Pull%) move into
    # the Pitch Group table instead.
    if not is_roc:
        cp_y_top = 0.155
        cp_y_bot = 0.120
        cp_x_left = 0.01
        cp_x_right = 0.99
        cp_height = cp_y_top - cp_y_bot
        cp_cells = [
            ('Bat Speed',        h_row.get('batSpeed'),      'mph', None),
            ('Avg EV',           h_row.get('avgEVAll'),      'mph', None),
            ('Max EV',           h_row.get('maxEV'),         'mph', None),
            ('Squared-Up%',      h_row.get('squaredUpPct'),  'pct', None),
            ('Blast%',           h_row.get('blastPct'),      'pct', None),
            ('IdealAtkAngle%',   h_row.get('idealAAPct'),    'pct', None),
            ('Air Pull%',        h_row.get('airPullPct'),    'pct', None),
        ]
        def _format_cp_val(val, kind):
            if val is None: return '—'
            if kind == 'pct': return f'{val * 100:.1f}%'
            return f'{val:.1f} {kind}' if kind else f'{val:.1f}'
        cp_val_strs = [_format_cp_val(v, kd) for (_l, v, kd, _) in cp_cells]
        CP_PAD_CHARS = 4
        CP_MIN_CHARS = 9
        cp_char_widths = []
        for i, (lbl, _v, _kd, _) in enumerate(cp_cells):
            widest = max(len(lbl), len(cp_val_strs[i]))
            cp_char_widths.append(max(CP_MIN_CHARS, widest + CP_PAD_CHARS))
        cp_total = sum(cp_char_widths)
        cp_widths_frac = [(cp_x_right - cp_x_left) * c / cp_total
                          for c in cp_char_widths]
        cp_x_offsets = []
        _cx = cp_x_left
        for w in cp_widths_frac:
            cp_x_offsets.append(_cx)
            _cx += w

        for i, (lbl, val, kind, _) in enumerate(cp_cells):
            cx = cp_x_offsets[i]
            cw = cp_widths_frac[i]
            val_str = cp_val_strs[i]
            ax_main.add_patch(Rectangle((cx * FIG_W, (cp_y_bot + cp_height / 2) * FIG_H),
                                          cw * FIG_W, cp_height / 2 * FIG_H,
                                          facecolor=DARKER, edgecolor=SUBTLE_BORDER,
                                          linewidth=0.6))
            ax_main.text((cx + cw / 2) * FIG_W,
                           (cp_y_bot + cp_height * 0.75) * FIG_H, lbl,
                           fontsize=11, ha='center', va='center', color=ACCENT,
                           fontweight='bold', fontfamily='IBM Plex Sans')
            pctl_key_map = {
                'Bat Speed': 'batSpeed_pctl', 'Avg EV': 'avgEVAll_pctl',
                'Max EV': 'maxEV_pctl', 'Squared-Up%': 'squaredUpPct_pctl',
                'Blast%': 'blastPct_pctl', 'IdealAtkAngle%': 'idealAAPct_pctl',
                'Air Pull%': 'airPullPct_pctl',
            }
            pctl = h_row.get(pctl_key_map.get(lbl, ''))
            cell_bg = DARK_CELL
            if pctl is not None:
                if pctl <= 50:
                    t = pctl / 50.0
                    r_t = int(0 + t * 140); g_t = int(100 + t * 40); b_t = int(255 - t * 115)
                else:
                    t = (pctl - 50) / 50.0
                    r_t = int(140 + t * 115); g_t = int(140 - t * 120); b_t = int(140 - t * 120)
                cp_alpha = 0.55
                rb = int(DARK_CELL[1:3], 16)
                rg = int(DARK_CELL[3:5], 16)
                rbb = int(DARK_CELL[5:7], 16)
                r = int(rb * (1 - cp_alpha) + r_t * cp_alpha)
                g = int(rg * (1 - cp_alpha) + g_t * cp_alpha)
                b = int(rbb * (1 - cp_alpha) + b_t * cp_alpha)
                cell_bg = f'#{r:02x}{g:02x}{b:02x}'
            ax_main.add_patch(Rectangle((cx * FIG_W, cp_y_bot * FIG_H),
                                          cw * FIG_W, cp_height / 2 * FIG_H,
                                          facecolor=cell_bg, edgecolor=SUBTLE_BORDER,
                                          linewidth=0.6))
            ax_main.text((cx + cw / 2) * FIG_W,
                           (cp_y_bot + cp_height * 0.25) * FIG_H, val_str,
                           fontsize=14, ha='center', va='center', color=TEXT_PRIMARY,
                           fontweight='bold', fontfamily='IBM Plex Sans')
        # Section title — same letterspaced editorial style as LA × Spray title
        fig.text(0.5, cp_y_top + 0.008, 'C O N T A C T   P R O F I L E',
                  fontsize=12, ha='center', va='bottom', color=TEXT_SECONDARY,
                  fontweight='700', fontfamily='IBM Plex Sans')
        rule_w_cp = 0.13
        ax_rule_cp = fig.add_axes([0.5 - rule_w_cp / 2, cp_y_top + 0.005,
                                    rule_w_cp, 0.0008])
        ax_rule_cp.axis('off')
        ax_rule_cp.add_patch(Rectangle((0, 0), 1, 1, facecolor=TEXT_SECONDARY,
                                         edgecolor='none', alpha=0.30,
                                         transform=ax_rule_cp.transAxes))

    # ─── Bottom table: per-pitch-group performance ──────────────────
    # Compute per-group stats
    group_rows = []
    n_total_seen = len(hitter_pitches)
    for g in GROUP_ORDER:
        gp = [p for p in hitter_pitches if p.get('Pitch Type') in PITCH_GROUPS[g]]
        stats = compute_group_stats(gp, sacq_lookups, bats)
        if stats:
            stats['group'] = g
            stats['usagePct'] = stats['count'] / n_total_seen if n_total_seen else None
            group_rows.append(stats)
    # Total row
    total_stats = compute_group_stats(hitter_pitches, sacq_lookups, bats) or {}
    total_stats['group'] = 'Total'
    total_stats['usagePct'] = 1.0

    # Build table data — different columns for ROC (no xwOBA-based metrics
    # but adds Avg EV / Max EV / Air Pull% in their place since the Contact
    # Profile strip is dropped for ROC).
    def _fmt_ev(v):
        return f'{v:.1f}' if v is not None else '—'

    if is_roc:
        BOTTOM_HEADERS = ['Pitch Group', 'Count', 'Usage', 'Swing%', 'Chase%',
                          'Whiff%', 'Hard-Hit%', 'Barrel%', 'xwOBAsp',
                          'Avg EV', 'Max EV', 'Air Pull%']
        rows = []
        for r in group_rows + [total_stats]:
            rows.append([
                r['group'],
                str(r.get('count', 0)),
                fmt_pct(r.get('usagePct')),
                fmt_pct(r.get('swingPct')),
                fmt_pct(r.get('chasePct')),
                fmt_pct(r.get('whiffPct')),
                fmt_pct(r.get('hardHitPct')),
                fmt_pct(r.get('barrelPct')),
                fmt_3dec(r.get('xwOBAsp')),
                _fmt_ev(r.get('avgEV')),
                _fmt_ev(r.get('maxEV')),
                fmt_pct(r.get('airPullPct')),
            ])
    else:
        BOTTOM_HEADERS = ['Pitch Group', 'Count', 'Usage', 'Swing%', 'Chase%',
                          'Whiff%', 'Hard-Hit%', 'Barrel%', 'xwOBAcon',
                          'xwOBAsp', 'RV/100', 'xRV/100']
        rows = []
        for r in group_rows + [total_stats]:
            rows.append([
                r['group'],
                str(r.get('count', 0)),
                fmt_pct(r.get('usagePct')),
                fmt_pct(r.get('swingPct')),
                fmt_pct(r.get('chasePct')),
                fmt_pct(r.get('whiffPct')),
                fmt_pct(r.get('hardHitPct')),
                fmt_pct(r.get('barrelPct')),
                fmt_3dec(r.get('xwOBAcon')),
                fmt_3dec(r.get('xwOBAsp')),
                fmt_signed_decimal(r.get('rv100')),
                fmt_signed_decimal(r.get('xRv100')),
            ])

    # PITCH GROUP BREAKDOWN section title removed — the table's own header
    # row already labels the section (Pitch Group | Count | Usage | ...).
    # Render bottom table
    table_y_top = 0.112
    table_y_bot = 0.010
    ax_table = fig.add_axes([0.01, table_y_bot, 0.98, table_y_top - table_y_bot])
    ax_table.axis('off'); ax_table.set_facecolor(BG)

    # Proportional column widths — smart padding: each column fits its widest
    # entry plus a generous padding floor so headers + values breathe.
    PAD_CHARS = 4
    MIN_CHARS = 6
    col_char_w = []
    for ci in range(len(BOTTOM_HEADERS)):
        max_len = len(BOTTOM_HEADERS[ci])
        for r in rows:
            if len(r[ci]) > max_len: max_len = len(r[ci])
        col_char_w.append(max(MIN_CHARS, max_len + PAD_CHARS))
    total_chars = sum(col_char_w)
    col_widths = [c / total_chars for c in col_char_w]

    table = ax_table.table(cellText=rows, colLabels=BOTTOM_HEADERS,
                            loc='upper center', cellLoc='center',
                            colWidths=col_widths)
    table.auto_set_font_size(False); table.set_fontsize(13); table.scale(1, 2.4)

    # Style cells — header off-white (no cyan), body lighter weight, Total row
    # gets architectural distinction (heavier weight + brighter color + thicker
    # border on top edge of Total row to anchor it as the summary)
    for (rr, cc), cell in table.get_celld().items():
        cell.set_edgecolor(SUBTLE_BORDER); cell.set_linewidth(0.5)
        if rr == 0:
            # Header row: off-white, heavy weight, slightly bigger
            cell.set_facecolor(DARKER)
            cell.set_text_props(color=TEXT_PRIMARY, fontweight='700', fontsize=13, fontfamily='IBM Plex Sans Condensed')
        elif rr == len(rows):
            # Total row: heavier weight + brighter, edge thickened all-around
            # to read as a distinct summary band
            cell.set_facecolor(DARKER)
            cell.set_edgecolor(TOTAL_BORDER); cell.set_linewidth(1.4)
            cell.set_text_props(fontweight='700', color=TEXT_PRIMARY, fontsize=13, fontfamily='IBM Plex Sans')
        else:
            # Body rows: lighter weight to differentiate from header/total
            cell.set_facecolor(DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
            cell.set_text_props(color=TEXT_PRIMARY, fontweight='600', fontsize=13, fontfamily='IBM Plex Sans')
        if cc == 0 and rr > 0:
            row_idx = rr - 1
            grp = group_rows[row_idx]['group'] if row_idx < len(group_rows) else None
            if grp and grp in GROUP_COLORS:
                # Dampened badge — blend GROUP_COLORS with DARKER at 0.70 so
                # the category labels harmonize with the dampened rate cells
                # in the rest of the table instead of being the loudest thing
                # in the lower half of the card.
                gc = GROUP_COLORS[grp]
                gr = int(gc[1:3], 16); gg = int(gc[3:5], 16); gb = int(gc[5:7], 16)
                dr = int(DARKER[1:3], 16); dg = int(DARKER[3:5], 16); db_ = int(DARKER[5:7], 16)
                a = 0.70
                blend_r = int(dr * (1 - a) + gr * a)
                blend_g = int(dg * (1 - a) + gg * a)
                blend_b = int(db_ * (1 - a) + gb * a)
                blended = f'#{blend_r:02x}{blend_g:02x}{blend_b:02x}'
                cell.set_facecolor(blended)
                cell.set_text_props(color=badge_text_color(blended),
                                    fontweight='700', fontsize=13, fontfamily='IBM Plex Sans')

    # Color rate-stat cells based on hitter direction
    hl_a = metadata.get('hitterLeagueAverages', {})

    def col_idx(name):
        return BOTTOM_HEADERS.index(name) if name in BOTTOM_HEADERS else None

    # higher_is_better convention for hitter:
    HITTER_DIR_COLS = {
        'Swing%':    ('swingPct',   'pct', False),  # lower = better (selectivity)
        'Chase%':    ('chasePct',   'pct', False),
        'Whiff%':    ('whiffPct',   'pct', False),
        'Hard-Hit%': ('hardHitPct', 'pct', True),
        'Barrel%':   ('barrelPct',  'pct', True),
    }
    for hdr, (k, kind, hib) in HITTER_DIR_COLS.items():
        ci = col_idx(hdr)
        if ci is None: continue
        la = hl_a.get(k)
        if la is None: continue
        for rr in range(1, len(rows) + 1):
            val_str = rows[rr - 1][ci]
            row_bg = DARKER if rr == len(rows) else (DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
            # Dampened: max_alpha=0.40 (default 0.55) so bottom table reads
            # as supporting context, not competing with hero visuals above.
            tinted = pct_cell_color(val_str, la, row_bg, hib, max_alpha=0.40)
            if tinted: table.get_celld()[(rr, ci)].set_facecolor(tinted)

    # xwOBAcon, xwOBAsp: higher_is_better=True, raw scale ~0.05
    for hdr, (k, scale) in [('xwOBAcon', ('xwOBAcon', 0.05)), ('xwOBAsp', ('xwOBAsp', 0.05))]:
        ci = col_idx(hdr)
        if ci is None: continue
        la = hl_a.get(k)
        if la is None: continue
        for rr in range(1, len(rows) + 1):
            val_str = rows[rr - 1][ci]
            row_bg = DARKER if rr == len(rows) else (DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
            tinted = raw_cell_color(val_str, la, scale, True, row_bg, max_alpha=0.40)
            if tinted: table.get_celld()[(rr, ci)].set_facecolor(tinted)

    # RV/100 and xRV/100: centered at 0, higher=better for hitter
    for hdr in ('RV/100', 'xRV/100'):
        ci = col_idx(hdr)
        if ci is None: continue
        for rr in range(1, len(rows) + 1):
            val_str = rows[rr - 1][ci]
            row_bg = DARKER if rr == len(rows) else (DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
            tinted = raw_cell_color(val_str, 0.0, 2.0, True, row_bg, max_alpha=0.40)
            if tinted: table.get_celld()[(rr, ci)].set_facecolor(tinted)

    # ROC-only columns: Avg EV / Max EV (mph, MLB avg ~88/110, higher=better)
    # and Air Pull% (pct, MLB avg ~13%, higher=better).
    if is_roc:
        for hdr, (k, scale) in [('Avg EV', ('avgEVAll', 4.0)),
                                  ('Max EV', ('maxEV', 4.0))]:
            ci = col_idx(hdr)
            if ci is None: continue
            la = hl_a.get(k)
            if la is None: continue
            for rr in range(1, len(rows) + 1):
                val_str = rows[rr - 1][ci]
                row_bg = DARKER if rr == len(rows) else (DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
                tinted = raw_cell_color(val_str, la, scale, True, row_bg, max_alpha=0.40)
                if tinted: table.get_celld()[(rr, ci)].set_facecolor(tinted)
        # Air Pull% (percentage column, higher=better)
        ci = col_idx('Air Pull%')
        if ci is not None:
            la = hl_a.get('airPullPct')
            if la is not None:
                for rr in range(1, len(rows) + 1):
                    val_str = rows[rr - 1][ci]
                    row_bg = DARKER if rr == len(rows) else (DARK_CELL if rr % 2 == 1 else ALT_ROW_BG)
                    tinted = pct_cell_color(val_str, la, row_bg, True, max_alpha=0.40)
                    if tinted: table.get_celld()[(rr, ci)].set_facecolor(tinted)

    # ─── Watermark ─────────────────────────────────────────────────
    # Watermark — bottom-right corner.
    fig.text(0.99, 0.012, 'huronalytics.vercel.app', color=TEXT_PRIMARY,
              fontsize=9, fontfamily='IBM Plex Sans', ha='right',
              va='bottom', style='italic')

    # Save
    safe_name = display_name.replace(' ', '_').replace('.', '').replace(',', '')
    out_path = os.path.join(output_dir,
                              f'HitterCard_{safe_name}_'
                              f'{date_slug or year_label.replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=SAVE_DPI, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return True


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── Settings (edit these directly or override via command line) ──
    team           = "WSH"                   # Team filter (e.g., "NYY"), or None for all teams
    start_date     = None                         # Set to None for full season
    end_date       = None                         # Set to a date for date range, or None for single day
    filter_hitters = ""       # Semicolon-separated "Last, First" names, or "" for all
    year_label     = "2026 Season"        # Display label on the card
    output_dir     = OUTPUT_DIR

    # ── CLI overrides (optional — values above are used if no args passed) ──
    parser = argparse.ArgumentParser(description='Generate hitter stat cards')
    parser.add_argument('--team', default=None,
                         help='Team abbreviation — only render hitters on this team')
    parser.add_argument('--start', default=None,
                         help='Start date YYYY-MM-DD, or "none" for full season')
    parser.add_argument('--end', default=None,
                         help='End date YYYY-MM-DD. Omit for a single day.')
    parser.add_argument('--hitters', default=None,
                         help='Semicolon-separated "Last, First" names; empty string = all qualified hitters')
    parser.add_argument('--year-label', default=None,
                         help=f'Display label on the card (default: "{year_label}")')
    parser.add_argument('--output-dir', default=None,
                         help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--layout', default='bubbles',
                         choices=['classic', 'bubbles'],
                         help="'bubbles' (default) = single-column percentile grid "
                              "(Result / QoC / Plate Discipline / Bat Tracking). "
                              "'classic' = legacy heat maps + contact profile + pitch group table.")
    parser.add_argument('--la-view', default='damage', choices=['damage', 'outcome'],
                         help="LA x Spray mode: 'damage' (default) = softened damage-zone "
                              "gradient + barrel/hard-hit/other dots + quality-contact "
                              "concentration ellipse; 'outcome' = full wOBAcon zones + "
                              "1B/2B/3B/HR/out dots.")
    args = parser.parse_args()

    if args.team is not None: team = args.team
    if args.start is not None: start_date = None if args.start.lower() == 'none' else args.start
    if args.end is not None: end_date = None if args.end.lower() == 'none' else args.end
    if args.hitters is not None: filter_hitters = args.hitters
    if args.year_label is not None: year_label = args.year_label
    if args.output_dir is not None: output_dir = args.output_dir
    layout = args.layout
    la_view = args.la_view

    # Parse filter_hitters string into a list (empty string → render all)
    if filter_hitters:
        hitter_names = [h.strip() for h in filter_hitters.split(';') if h.strip()]
    else:
        hitter_names = None

    # Resolve the date window. Same rules as cards/pitcher.py: a same-day
    # range collapses to a single date, and no dates means the full season.
    from datetime import datetime as _dt_cls
    # "--start none" means full season, so it clears end_date too. Without
    # this, an end_date left in the settings block above sends a start-less
    # range into strptime(None).
    if start_date is None and end_date is not None:
        print("  --start none: ignoring end_date "
              f"{end_date} and rendering the full season.")
        end_date = None
    if end_date is not None and end_date == start_date:
        end_date = None
    if start_date is None and end_date is None:
        date_filter = None
        date_display = None
        date_slug = None
        date_label = 'full season'
    elif end_date is None:
        date_filter = (start_date, start_date)
        _d = _dt_cls.strptime(start_date, '%Y-%m-%d')
        date_display = _d.strftime('%B %d, %Y').replace(' 0', ' ')
        date_slug = _d.strftime('%m%d%Y')
        date_label = start_date
    else:
        date_filter = (start_date, end_date)
        _s = _dt_cls.strptime(start_date, '%Y-%m-%d')
        _e = _dt_cls.strptime(end_date, '%Y-%m-%d')
        date_display = (f"{_s.strftime('%b %d').replace(' 0', ' ')} – "
                        f"{_e.strftime('%b %d, %Y').replace(' 0', ' ')}")
        date_slug = f"{_s.strftime('%m%d')}-{_e.strftime('%m%d%Y')}"
        date_label = f"{start_date} to {end_date}"
    # ──────────────────────────────────────────────────────────

    span_label = date_display or year_label
    if date_filter is not None:
        print(f"Date window: {date_label}. Every stat is rebuilt from the "
              f"window's pitches; wRC+ and the + family read '—'.")

    if hitter_names:
        team_label = team if team else 'auto-resolve team'
        print(f"═══ Generating hitter cards for {', '.join(hitter_names)} "
              f"({team_label}) — {span_label} ═══\n")
        for name in hitter_names:
            render_hitter_card(name, team_abbrev=team,
                                year_label=year_label, output_dir=output_dir,
                                layout=layout, la_view=la_view,
                                date_filter=date_filter,
                                date_display=date_display, date_slug=date_slug)
    else:
        # No specific names: render every (qualified) hitter, optionally filtered by team
        leaderboard = load_hitter_leaderboard()
        targets = [r for r in leaderboard if r.get('bipQual')]
        if team:
            targets = [r for r in targets if r.get('team') == team]
        team_label = team if team else 'all teams'
        print(f"═══ Generating hitter cards for {len(targets)} hitters "
              f"({team_label}) — {span_label} ═══\n")
        for r in targets:
            render_hitter_card(r.get('hitter'), team_abbrev=r.get('team'),
                                year_label=year_label, output_dir=output_dir,
                                layout=layout, la_view=la_view,
                                date_filter=date_filter,
                                date_display=date_display, date_slug=date_slug)


if __name__ == '__main__':
    main()
