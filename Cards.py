"""
Batch Pitcher Card Generator
Generates dark-themed pitcher stat cards for all pitchers on a team for a given date.

Usage:
    1. Edit the Settings block at the top of main()
    2. python3 Cards.py
"""

import argparse
import math
import sys
import os
import json
import urllib.request
import urllib.parse
import time as time_module
from datetime import datetime
from collections import defaultdict
from math import atan2, sin, cos

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

# Register the bundled print-identity fonts (Bitter, IBM Plex Sans / Condensed)
# so cards render in the correct typefaces on any machine, independent of a
# system font install or a stale matplotlib cache. HitterCards.py imports from
# this module, so it inherits the registration too.
import os as _os
import matplotlib.font_manager as _fm
_FONT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'assets', 'fonts')
if _os.path.isdir(_FONT_DIR):
    for _fn in sorted(_os.listdir(_FONT_DIR)):
        if _fn.lower().endswith(('.ttf', '.otf')):
            try:
                _fm.fontManager.addfont(_os.path.join(_FONT_DIR, _fn))
            except Exception:
                pass
from PIL import Image
from io import BytesIO
import numpy as np
import gspread
from sheets_append import _workbook_id_for_team

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
# Sheet routing comes from sheets_append._workbook_id_for_team (per team).

AL_TEAMS = {'ATH','BAL','BOS','CLE','CWS','DET','HOU','KCR','LAA','MIN','NYY','SEA','TBR','TEX','TOR'}
NL_TEAMS = {'ARI','ATL','CHC','CIN','COL','LAD','MIA','MIL','NYM','PHI','PIT','SDP','SFG','STL','WSH'}

# MiLB teams — data lives as extra tabs in the NL spreadsheet
MILB_TEAMS = {
    'ROC': {
        'sheet_key': 'NL',
        'sport_id': 11,         # AAA = sportId 11
        'search_name': 'Rochester',
    },
}
MILB_TEAM_NAME_TO_ABBREV = {
    'Rochester Red Wings': 'ROC',
}

TEAM_ABBREV_TO_ID = {
    'ARI':109,'ATL':144,'BAL':110,'BOS':111,'CHC':112,'CWS':145,'CIN':113,'CLE':114,
    'COL':115,'DET':116,'HOU':117,'KCR':118,'LAA':108,'LAD':119,'MIA':146,'MIL':158,
    'MIN':142,'NYM':121,'NYY':147,'ATH':133,'PHI':143,'PIT':134,'SDP':135,'SFG':137,
    'SEA':136,'STL':138,'TBR':139,'TEX':140,'TOR':141,'WSH':120,
    'ROC':120,  # Rochester Red Wings — parent org is WSH
}

TEAM_NAME_TO_ABBREV = {
    'Arizona Diamondbacks':'ARI','Athletics':'ATH','Atlanta Braves':'ATL',
    'Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago Cubs':'CHC',
    'Chicago White Sox':'CWS','Cincinnati Reds':'CIN','Cleveland Guardians':'CLE',
    'Colorado Rockies':'COL','Detroit Tigers':'DET','Houston Astros':'HOU',
    'Kansas City Royals':'KCR','Los Angeles Angels':'LAA','Los Angeles Dodgers':'LAD',
    'Miami Marlins':'MIA','Milwaukee Brewers':'MIL','Minnesota Twins':'MIN',
    'New York Mets':'NYM','New York Yankees':'NYY','Philadelphia Phillies':'PHI',
    'Pittsburgh Pirates':'PIT','San Diego Padres':'SDP','San Francisco Giants':'SFG',
    'Seattle Mariners':'SEA','St. Louis Cardinals':'STL','Tampa Bay Rays':'TBR',
    'Texas Rangers':'TEX','Toronto Blue Jays':'TOR','Washington Nationals':'WSH',
}

# Pitch colors. The light hues (sinker, slider-gray, splitter-teal) are toned
# DOWN from pure Statcast brights so they're readable on the warm cream cards
# everywhere — the old #FFD700 sinker / #DDDDDD slider washed out. SI is a warm
# gold (not yellow/mustard); the dark hues keep their saturated brand values.
PITCH_COLORS = {
    'FF':'#0072B2','SI':'#E0A81E','FC':'#8B5A2B','SL':'#D55E00',
    'ST':'#56B4E9','CU':'#332288','SV':'#882255','CH':'#009E73','FS':'#CC79A7','KN':'#9A9A9A',
    'EP':'#888888'
}
PITCH_NAMES = {
    'FF':'Fastball','SI':'Sinker','FC':'Cutter','SL':'Slider',
    'ST':'Sweeper','CU':'Curveball','SV':'Slurve','CH':'Changeup','FS':'Splitter','KN':'Knuckleball',
    'EP':'Eephus'
}
PITCH_ORDER = ['FF','SI','FC','SL','ST','CU','SV','CH','FS','KN']
# Bunts are not swings (matches pipeline_utils.SWING_DESCRIPTIONS) so card
# Whiff%/Chase%/Swing% use the same swing set as the leaderboard they're colored
# against. STRIKE_DESC still counts Foul Bunt — a foul bunt is a strike.
SWING_DESC = ['Swinging Strike','Foul','In Play']

def _opponent_label(pitches):
    """Opposing team for a single-game card, off the pitches themselves.

    BTeam is the batting side on every pitch, so the modal value is the
    opponent. No extra API call, unlike the score/decision, which would need
    the linescore and is left off.
    """
    counts = defaultdict(int)
    for p in pitches:
        bt = str(p.get('BTeam', '') or '').strip()
        if bt:
            counts[bt] += 1
    return max(counts, key=counts.get) if counts else None


def _season_pitch_lb_for(pitcher_name, eff_team, pitch_lb_by_pitcher):
    """The pitcher's OWN season per-pitch-type row, for daily cards.

    Cannot reuse config['pitch_lb']: on a daily card scratch_ctx overwrites it
    with values computed from THIS START's pitches, which would make every
    delta identically zero. The team key can also disagree (card header says
    WSH, the leaderboard row is filed under ROC), so fall back to the largest
    by-name row set rather than returning nothing.

    Caveat: if the pipeline has already run today, the season row INCLUDES
    this start, which shrinks the delta slightly. Worst case is a fastball
    thrown 34 times against a 386-pitch season, so ~9% self-reference.
    """
    # Largest row set by total pitches, NOT the exact (name, team) match.
    # Preferring the team match looked right and was wrong for anyone traded
    # mid-season: Jake Bird's WSH rows are 8 sweepers and 4 sinkers, while his
    # actual season lives under the combined 2TM key at 208 and 200. The exact
    # match succeeded, returned 12 pitches, and every pitch type then failed
    # the sample floor, so the whole block rendered unshaded with no error.
    #
    # Caveat: this can cross levels for a prospect with a big AAA sample and a
    # short MLB stint. That is still the best baseline available for "what does
    # he normally do", and the SE below widens on thin samples either way.
    cands = [d for (nm, _tm), d in pitch_lb_by_pitcher.items() if nm == pitcher_name]
    if not cands:
        return {}
    return max(cands, key=lambda d: sum((v or {}).get('count') or 0 for v in d.values()))


def _normalize_name(name):
    """Case-fold for name matching (handles 'de Oca' vs 'De Oca')."""
    return name.strip().lower()
STRIKE_DESC = ['Called Strike','Swinging Strike','Foul','Foul Bunt','In Play']

# Batted ball type colors (for distribution chart)
BB_COLORS = {
    'ground_ball': '#2E8FA8',   # steel teal
    'line_drive':  '#FF6B6B',   # soft coral
    'fly_ball':    '#7B68EE',   # medium slate blue
    'popup':       '#FF9F43',   # warm amber
}
BB_LABELS = {
    'ground_ball': 'Ground Ball',
    'line_drive':  'Line Drive',
    'fly_ball':    'Fly Ball',
    'popup':       'Popup',
}
BB_TYPES = ['ground_ball', 'line_drive', 'fly_ball', 'popup']

# Layout constants (v30)
TABLE_LEFT_FIG=0.01; TABLE_RIGHT_FIG=0.99; PLOT_LEFT=0.585; PLOT_RIGHT=0.99
USAGE_SHIFT=0.18; DIVIDER_COL=14; PLATE_HALF=17/12/2
FIG_W=16; FIG_H=17.5; DPI=100; SAVE_DPI=150

# WARM PAPER THEME — matches the hitter cards (HitterCards.py). The pitcher
# card was historically a dark "command console"; this brings it into the same
# light, editorial identity. Constant NAMES are kept (BG/ACCENT/DARK_CELL/
# DARKER) so existing render_card references pick up the new values; the extra
# TEXT_*/border constants below replace the old hardcoded '#888'/'white'/
# '#333840' literals.
BG          = '#f0e8d8'   # warm cream paper background
DARK_CELL   = '#e2d8c4'   # slightly darker cream for cells / alt rows
DARKER      = '#d8ccb4'   # deepest tan for headers and Total row
ACCENT      = '#9f3026'   # deep terracotta red (borders, accents, dates)

TEXT_PRIMARY   = '#1a1612'  # warm near-black (name, headline values, table)
TEXT_SECONDARY = '#3a3530'  # deep warm gray (section titles, headers)
TEXT_MUTED     = '#6a5f55'  # mid warm gray (subtitle, annotations, axes)
TEXT_FAINT     = '#8a7f75'  # light warm gray (fine print, legend)
SUBTLE_BORDER  = '#c5b89f'  # light tan border (cell edges, grid)
ALT_ROW_BG     = '#e8dfcb'  # alternating table row / plot panel
PLOT_PANEL     = '#e8dfcb'  # light panel for movement / location plots
GRID_COLOR     = '#c5b89f'  # subtle grid on cream
PHOTO_BORDER   = '#6a5f55'  # photo edge

MLB_ID_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlb_id_cache.json')
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'Downloads', '')
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'metadata_rs.json')


def _apply_runexp_currency(rows):
    """Rescale MiLB-sourced RunExp into MLB run currency, in place.

    Statcast builds delta_run_exp on each league's own run-expectancy matrix,
    so the identical event carries ~1.25x the magnitude at AAA/ROC. process_data
    corrects this for the site, but the correction is never written back to the
    sheets, so every card built straight from a worksheet inherited the raw
    MiLB values — inflating PitchRV/xPitchRV (and, through xRV/100, Pitcher+)
    on ROC/AAA and cross-level cards.

    LEVEL comes from bat tracking, not from a team label: BatSpeed/SwingLength
    are MLB-only, so a game with any tracked swing is MLB and everything else
    is MiLB. That matters because a cross-level card (--all-levels, or a
    scratch tab holding a full MLB+AAA season) mixes both currencies inside
    one pitcher, and its team columns cannot be trusted to say which is which.
    The ROC-vs-AAA factor set is picked off BTeam, which only chooses between
    two factors within ~2% of each other.

    Factors come from metadata (published by the pipeline run) rather than
    being re-derived here: deriving them needs an MLB reference set that a
    single card's pitches don't contain.
    """
    from pipeline_utils import runexp_factor, runexp_scale_from_json
    try:
        with open(METADATA_PATH) as f:
            scale = runexp_scale_from_json(json.load(f).get('runexpScale'))
    except (OSError, ValueError):
        scale = None
    if not scale:
        print("  [WARN] metadata has no runexpScale — MiLB RunExp stays in "
              "MiLB currency (RV/xRV/Pitcher+ will read high on ROC/AAA "
              "pitches). Re-run process_data.py to publish it.")
        return

    # Keyed per PITCHER-date, not per tab-date: a scratch tab pools arms who
    # pitched at different levels on the same day (24 such dates in the NEW
    # tab), so a tab-wide date key would inherit one MLB arm's bat tracking
    # and silently skip the correction for everyone else that day.
    SWINGS = ('Swinging Strike', 'Foul', 'In Play')
    mlb_outings, swings_seen, mlb_dates = set(), set(), set()
    for r in rows:
        key = (r.get('Pitcher'), r.get('Game Date'))
        if any(str(r.get('Description', '')).startswith(s) for s in SWINGS):
            swings_seen.add(key)
        if str(r.get('BatSpeed', '')).strip():
            mlb_outings.add(key)
            mlb_dates.add((r.get('_card_team'), r.get('Game Date')))

    n_fixed = 0
    for r in rows:
        key = (r.get('Pitcher'), r.get('Game Date'))
        if key in mlb_outings:
            continue
        # No swing in the outing means no bat tracking could appear either
        # way (a one-batter called-strike inning), so fall back to whether
        # anything else on that tab-date was tracked.
        if key not in swings_seen and (r.get('_card_team'), r.get('Game Date')) in mlb_dates:
            continue
        src = r.get('BTeam') if r.get('BTeam') in scale else None
        sc = scale.get(src) or scale.get('AAA') or scale.get('ROC')
        if not sc:
            continue
        v = sf(r.get('RunExp'))
        if v is None:
            continue
        f = runexp_factor(sc, r.get('Description'), r.get('Count'))
        if f:
            r['RunExp'] = v / f
            n_fixed += 1
    if n_fixed:
        print(f"  RunExp -> MLB currency: {n_fixed} MiLB pitches rescaled "
              f"({len(mlb_outings)} MLB outings left as-is)")

# Guts constants for xRV computation. Read live from metadata_rs.json so the
# values match whatever process_data.py used on its last run; otherwise a
# Cards-vs-leaderboard mismatch creeps back in as FanGraphs updates Guts.
def _load_guts():
    try:
        with open(METADATA_PATH) as _f:
            g = json.load(_f).get('gutsConstants') or {}
        lg, sc = g.get('lgWOBA'), g.get('wOBAScale')
        if lg and sc:
            return float(lg), float(sc)
    except Exception:
        pass
    # Fallback only if metadata is missing/incomplete (first run, network issue).
    return 0.320, 1.252

GUTS_LG_WOBA, GUTS_WOBA_SCALE = _load_guts()

# Mapping: card column header → metadata league average key
PCT_COLOR_COLS = {
    'CSW%': 'cswPct',
    'Zone%':   'izPct',
    'Whiff%':  'swStrPct',
    'Chase%':  'chasePct',
}

# Raw-value columns that get percentile coloring (not percentages)
# Maps column header → (metadata key, scale, higher_is_better)
# scale = deviation in raw units that maps to full color intensity
RAW_COLOR_COLS = {
    'Ext': ('extension', 0.5, True),
}

# DAILY CARDS — minimum season pitches of a type before that type gets a
# baseline at all. Deliberately LOW, because the standard error below already
# prices a thin baseline: se = sd * sqrt(1/n_today + 1/n_season), so a 14-pitch
# season sample widens the error bar and shrinks the z on its own. A hard floor
# on top of that double-penalises, and at 50 it silently blanked every
# secondary pitch of every reliever (Cosgrove's fastball at 29, Cruz's at 42).
# This floor now only guards against a baseline built on a handful of pitches.
SEASON_DELTA_MIN = 10
# Pitch types where extra RIDE works against the pitch, so more IVB than his
# own norm reads as worse. Sinkers included (2026-08-13, Wally): a sinker's job
# is to stay down, so extra ride cuts against it exactly as it does on a
# changeup, even though the sinker is not a separation pitch.
LOW_IVB_PITCH_TYPES = {'CH', 'FS', 'SI'}
# Narrower set: pitches that also want LESS spin than his norm. Changeups and
# splitters work by separating from the fastball and extra spin fights that;
# a sinker's raw spin rate is not the same lever (efficiency is), so it is
# deliberately NOT in here.
LOW_SPIN_PITCH_TYPES = {'CH', 'FS'}
# Approach-angle direction. On a four-seamer or cutter a FLATTER angle (closer
# to zero) plays; on everything else steeper does. Same set as process_data's
# VAA_NO_INVERT_TYPES so the card and the site agree on which pitches are the
# flat-is-better ones — note that excludes sinkers, which want steep.
FLAT_APPROACH_TYPES = {'FF', 'FC'}
# Shading scale for the self-baseline block (Usage, Avg Velo). The z the
# tint ramps on is the delta in SDs of the OBSERVED start-vs-season delta:
# sqrt(day-to-day component + this start's own sampling noise). The original
# scale was sampling SE alone — (within-start SD)/sqrt(n) — which answers
# "is this delta measurable?", not "is this start unusual for him?": past
# n≈20 the SE is so small that perfectly ordinary starts saturate (+0.3 mph
# on 44 four-seamers rendered full red, Gausman 2026-08-13, per Wally).
# Day-to-day components measured on 2021-2025, 187k pitcher-game-pitchtype
# starts (scripts/daily_delta_scale.py), stable within ±0.05 mph / ±0.005 c
# across all five seasons as independent replicates:
#   velo: SD_day in mph, the sampling-free spread of a start's true mean
#         around his season mean;
#   usage: game-plan variance modeled as c*u(1-u) — strategic mix choice is
#         overdispersed ~2-5x vs the binomial term u(1-u)/tc, and c genuinely
#         differs by type, so per-type constants.
DAILY_VELO_SD_DAY = {'FF': 0.67, 'SI': 0.65, 'FC': 0.81, 'SL': 0.92,
                     'ST': 0.80, 'CU': 0.80, 'CH': 0.77, 'FS': 0.80}
DAILY_VELO_SD_DAY_DEFAULT = 0.80   # unlisted types: median of the measured 8
DAILY_USAGE_C = {'FF': 0.0215, 'SI': 0.0301, 'FC': 0.0279, 'SL': 0.0241,
                 'ST': 0.0266, 'CU': 0.0155, 'CH': 0.0160, 'FS': 0.0213}
DAILY_USAGE_C_DEFAULT = 0.0227     # pooled across types
# Full colour at 2 SD of that observed-delta scale = his own ~5% tails
# (measured share of real starts with |z|>=2: 4.7-4.9% by type). NOT a
# cutoff — smaller gaps still tint faintly, which is what makes
# "faint = inside normal game-to-game noise" readable.
DELTA_MIN_SE = 2.0
# Deltas render inside the parent cell — "92.7 (-0.9)" — so there are no
# separate delta columns and the table gains no width.

# Stat-line coloring for multi-game cards
# Maps header → (pitcherLeagueAverages key, type, higher_is_better)
# type: 'pct' = stored as decimal (0.23), displayed as '23.0%'; 'raw' = displayed as raw number
STAT_LINE_COLOR = {
    'ERA':    ('era',      'raw', False, 1.5),
    'FIP':    ('fip',      'raw', False, 1.5),
    'SIERA':  ('siera',    'raw', False, 1.5),
    # hdERA/hpERA tint scales match ERA's visual intensity per SD: their
    # spreads are compressed (SD ~0.95 and ~0.75 runs vs ERA ~1.4), so the
    # full-color distance shrinks proportionally. Convention, not a fit.
    'hdERA':  ('hdera',    'raw', False, 1.0),
    'hpERA':  ('hpera',    'raw', False, 0.8),
    'K%':     ('kPct',     'pct', True),
    'BB%':    ('bbPct',    'pct', False),
    'Zone%':  ('izPct',    'pct', True),
    'Whiff%': ('swStrPct', 'pct', True),
    'GB%':    ('gbPct',    'pct', True),
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def sf(v):
    if v is None or v == '': return None
    try: return float(v)
    except Exception: return None

def _compute_pitch_xrv(pitches_list):
    """Per-pitch xRV in pitcher perspective (positive = good for pitcher).

    BIP w/ xwOBA: -(xwOBA-lgWOBA)/scale  — flip from hitter to pitcher view.
    Else: RunExp                         — already pitcher perspective in the sheet.

    Matches pipeline_compute.compute_xrv (web leaderboard).
    """
    vals = []
    for p in pitches_list:
        is_bip = p.get('Description') == 'In Play'
        xw = sf(p.get('xwOBA')) if is_bip else None
        if is_bip and xw is not None:
            vals.append(-(xw - GUTS_LG_WOBA) / GUTS_WOBA_SCALE)
        else:
            rv = sf(p.get('RunExp'))
            if rv is not None:
                vals.append(rv)
    return vals

# wOBA weights (FanGraphs Guts 2026, matching pipeline_fetch fallback) for the
# actual-outcome run value. Outs / SF / etc. map to 0.
_EVENT_WOBA = {'Single': 0.884, 'Double': 1.256, 'Triple': 1.591, 'Home Run': 2.048}

# Pitch-type qualification minimum (matches MIN_PITCH_TYPE_OUTCOME in
# process_data.py): per-pitch RV needs >= 25 pitches of that type. Below it the
# per-100 rate is noise (e.g. a 1-pitch changeup), so the RV cells show '—'.
PITCH_QUAL_MIN = 25

# ── Season-card coloring gates (2026-07-30, Wally's package) ──────────────
# Values ALWAYS render; these gate only the percentile tint, season and
# date-range cards only (daily cards keep their historical behavior).
#
# Flat 50 for the outcome row (Zone%, Whiff%, Chase%, RV pair): measured as
# the optimal single compromise over real arsenal row sizes — the
# misclassification curve vs per-metric measured gates bottoms at 50-55
# (flat 45-60). 50 IS Whiff%'s measured constant; Zone% k=29, Chase% k=96;
# the RV pair colors as a ledger convention (its own k would be 359/1234).
CARD_COLOR_MIN_PITCHES = 50
# Stuff+ is shape-family: measured per-type k = 13 (seeds 12.9-13.8), so 15
# colors only cells that are >=half signal without hiding real information.
STUFF_COLOR_MIN_PITCHES = 15
# Loc+ rides the site's measured per-group gates (retested 2026-07-30, all
# six inside fresh seed spreads) — keeps card/Arsenal-leaderboard parity.
# Mirrors js/aggregator.js QUAL.MIN_PITCH_LOCPLUS + LOCPLUS_GROUP.
LOCPLUS_COLOR_MIN = {'FF': 73, 'SI': 81, 'FC': 74, 'SL': 67, 'CU': 83, 'CH': 79}
LOCPLUS_COLOR_GROUP = {
    'FF': 'FF', 'FA': 'FF', 'SI': 'SI', 'FC': 'FC', 'CF': 'FC',
    'SL': 'SL', 'ST': 'SL', 'SW': 'SL', 'SV': 'SL',
    'CU': 'CU', 'KC': 'CU', 'CS': 'CU',
    'CH': 'CH', 'FS': 'CH', 'KN': 'CH', 'SC': 'CH'}
LOCPLUS_COLOR_DEFAULT = float('inf')   # unmapped types stay uncolored —
                                       # matches pipeline_locplus
                                       # STABILIZE_N_UNVALIDATED ("don't
                                       # color what we can't validate");
                                       # the old 135 fallback was the
                                       # retired overall crossing

# BIP-denominated coloring gates: GB% 25 BIP is the measured reliability-0.5
# crossing (== site MIN_BIP_PCTL); xwOBAcon 25 BIP is an accepted convention
# (its measured k is ~175 BIP — the tint describes the sample, not stable skill).
GB_COLOR_MIN_BIP = 25
XWC_COLOR_MIN_BIP = 25


def _locplus_color_min(pt):
    g = LOCPLUS_COLOR_GROUP.get(pt)
    return LOCPLUS_COLOR_MIN.get(g, LOCPLUS_COLOR_DEFAULT)

def _compute_pitch_rv(pitches_list):
    """Per-pitch ACTUAL run value, pitcher perspective. The actual-outcome twin of
    _compute_pitch_xrv: for each BIP use the outcome's wOBA weight (hit value, else
    0 for outs) in place of xwOBA. Non-BIP fall back to RunExp (empty for ROC, so
    ROC is contact-only — apples-to-apples with the BIP-only xPitchRV)."""
    vals = []
    for p in pitches_list:
        is_bip = p.get('Description') == 'In Play'
        if is_bip:
            w = _EVENT_WOBA.get(p.get('Event'), 0.0)
            vals.append(-(w - GUTS_LG_WOBA) / GUTS_WOBA_SCALE)
        else:
            rv = sf(p.get('RunExp'))
            if rv is not None:
                vals.append(rv)
    return vals

def pct_cell_color(value_str, league_avg, row_bg_hex, higher_is_better=True):
    """Return cell background color based on how a percentage compares to league average.
    value_str: cell text like '65.3%'
    league_avg: league average as decimal (e.g. 0.6587)
    row_bg_hex: base row background color (e.g. '#1e2127')
    higher_is_better: if False, above-average is red (bad) instead of green
    """
    if league_avg is None or not value_str or value_str == '—':
        return None
    try:
        val = float(value_str.replace('%', ''))
    except (ValueError, AttributeError):
        return None
    avg_pct = league_avg * 100
    diff = val - avg_pct  # positive = above average
    if not higher_is_better:
        diff = -diff
    # Scale: ±8 pp maps to full intensity
    intensity = max(-1.0, min(1.0, diff / 8.0))
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    if intensity >= 0:
        target = (0, 180, 0)
    else:
        target = (180, 0, 0)
        intensity = abs(intensity)
    alpha = intensity * 0.55
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'

def _parse_fi(s):
    """Parse feet-inches string like 6'3\" to float (6.25). Returns None on failure."""
    import re
    m = re.match(r"(-?)(\d+)'(\d+)\"", s)
    if not m:
        return None
    sign = -1 if m.group(1) == '-' else 1
    return sign * (int(m.group(2)) + int(m.group(3)) / 12.0)

def raw_cell_color(value_str, league_avg, scale, higher_is_better, row_bg_hex):
    """Return cell background color for a raw (non-percentage) value vs league average.
    scale: deviation in raw units that maps to full color intensity (e.g. 0.5 ft for extension).
    higher_is_better: True if above-average is green, False if below-average is green.
    """
    if league_avg is None or not value_str or value_str == '—':
        return None
    val = _parse_fi(value_str)
    if val is None:
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
        target = (180, 0, 0)
        intensity = abs(intensity)
    alpha = intensity * 0.55
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'

def avg_tilt(tilts):
    valid = [t for t in tilts if t and t != '']
    if not valid: return '—'
    sins, coss = [], []
    for t in valid:
        parts = str(t).split(':')
        if len(parts) != 2: continue
        h, m = int(parts[0]), int(parts[1])
        if h == 12: h = 0
        a = (h*60+m)/720*2*3.14159
        sins.append(sin(a)); coss.append(cos(a))
    if not sins: return '—'
    am = (atan2(sum(sins)/len(sins), sum(coss)/len(coss))*720/(2*3.14159)) % 720
    h, m = int(am//60), int(am%60)
    if h == 0: h = 12
    return f'{h}:{m:02d}'

def _mat_inv(M):
    """Invert a square matrix via Gauss-Jordan elimination with partial pivoting."""
    n = len(M)
    aug = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        max_row = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[max_row][col]):
                max_row = r
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for c in range(2 * n):
                aug[r][c] -= f * aug[col][c]
    for col in range(n - 1, -1, -1):
        piv = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= piv
        for r in range(col):
            f = aug[r][col]
            for c in range(2 * n):
                aug[r][c] -= f * aug[col][c]
    return [aug[i][n:] for i in range(n)]

def _mvn_conditional(model_params, rel_values):
    """Compute E[IVB, HB | regressors] using MVN conditional distribution."""
    mu = model_params['mu']
    cov = model_params['cov']
    n_acc = 2  # IVB, HB
    n_rel = len(mu) - n_acc
    if len(rel_values) != n_rel:
        return None
    sigma_rel = [[cov[n_acc + i][n_acc + j] for j in range(n_rel)] for i in range(n_rel)]
    sigma_rel_inv = _mat_inv(sigma_rel)
    if sigma_rel_inv is None:
        return None
    r_diff = [rel_values[k] - mu[n_acc + k] for k in range(n_rel)]
    sri_rdiff = [sum(sigma_rel_inv[i][j] * r_diff[j] for j in range(n_rel)) for i in range(n_rel)]
    mu_bar = []
    for a in range(n_acc):
        adj = sum(cov[a][n_acc + b] * sri_rdiff[b] for b in range(n_rel))
        mu_bar.append(mu[a] + adj)
    return mu_bar

def compute_expected_movement(mvn_models, pitch_type, throws, arm_angle, extension, velocity, rel_z, rel_x):
    """Compute xIVB and xHB using MVN conditional model. Returns (xIVB, xHB) or (None, None)."""
    if not mvn_models:
        return None, None
    mvn_key = (pitch_type or '') + '_' + (throws or '')
    pt_model = mvn_models.get(mvn_key)
    if not pt_model:
        return None, None
    if pt_model.get('mlb') and arm_angle is not None and extension is not None and velocity is not None:
        result = _mvn_conditional(pt_model['mlb'], [arm_angle, extension, velocity])
        if result:
            return result[0], result[1]
    if pt_model.get('roc') and rel_z is not None and rel_x is not None and extension is not None and velocity is not None:
        result = _mvn_conditional(pt_model['roc'], [rel_z, rel_x, extension, velocity])
        if result:
            return result[0], result[1]
    return None, None

def load_mvn_models():
    """Load MVN models from metadata_rs.json."""
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return meta.get('mvnModels', {})
    except Exception as e:
        print(f"  WARNING: Could not load MVN models from {METADATA_PATH}: {e}")
        return {}

def fmt_fi(v):
    if v is None: return '—'
    neg = v < 0; av = abs(v); ft = int(av); inc = round((av-ft)*12)
    if inc == 12: ft += 1; inc = 0
    s = f"{ft}'{inc}\""; return f"-{s}" if neg else s

def compute_iz(p):
    px, pz, st, sb = sf(p.get('PlateX')), sf(p.get('PlateZ')), sf(p.get('SzTop')), sf(p.get('SzBot'))
    if any(v is None for v in [px, pz, st, sb]): return None
    return abs(px) <= 0.83 and pz >= sb-0.121 and pz <= st+0.121

def luminance(hc):
    r, g, b = int(hc[1:3],16)/255, int(hc[3:5],16)/255, int(hc[5:7],16)/255
    r = r/12.92 if r<=0.03928 else ((r+0.055)/1.055)**2.4
    g = g/12.92 if g<=0.03928 else ((g+0.055)/1.055)**2.4
    b = b/12.92 if b<=0.03928 else ((b+0.055)/1.055)**2.4
    return 0.2126*r + 0.7152*g + 0.0722*b

def badge_text_color(hc):
    return 'black' if luminance(hc) > 0.25 else 'white'

def _darken(hexc, factor):
    """Multiply an #rrggbb color's channels by factor (<1 = darker)."""
    r = int(int(hexc[1:3], 16) * factor)
    g = int(int(hexc[3:5], 16) * factor)
    b = int(int(hexc[5:7], 16) * factor)
    return f'#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}'

def _rgba(hexc, a):
    """#rrggbb -> (r,g,b,a) float tuple for independent fill/edge alphas."""
    return (int(hexc[1:3],16)/255.0, int(hexc[3:5],16)/255.0, int(hexc[5:7],16)/255.0, a)

def is_barrel(ev, la):
    """Statcast barrel definition from baseballr code_barrel (EV >= 98 per MLB glossary)."""
    if ev is None or la is None:
        return False
    return (la >= 8 and la <= 50 and ev >= 98 and
            ev * 1.5 - la >= 117 and
            ev + la >= 124)

def outs_to_ip_str(outs):
    return f"{outs//3}.{outs%3}"

def compute_siera(so, bb, tbf, gb_count, fb_count, gs, g, siera_constant):
    """Compute SIERA for a single pitcher. Returns rounded value or None."""
    if tbf <= 0 or g <= 0:
        return None
    so_pa = so / tbf
    bb_pa = bb / tbf
    net_gb_pa = (gb_count - fb_count) / tbf
    ip_sp_ratio = min(gs / g, 1.0)
    sign_4920 = -1.0 if gb_count >= fb_count else 1.0
    raw = (
        - 15.518 * so_pa
        + 9.146 * (so_pa ** 2)
        + 8.648 * bb_pa
        + 27.252 * (bb_pa ** 2)
        - 2.298 * net_gb_pa
        + sign_4920 * 4.920 * (net_gb_pa ** 2)
        - 4.036 * so_pa * bb_pa
        + 5.155 * so_pa * net_gb_pa
        + 4.546 * bb_pa * net_gb_pa
        + 0.367 * ip_sp_ratio
    )
    return round(raw + siera_constant, 2)

def _fullname_to_lastfirst(full_name):
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name
    suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}
    suffix = ''
    if len(parts) > 2 and parts[-1].lower().rstrip('.') in {s.rstrip('.') for s in suffixes}:
        suffix = ' ' + parts.pop()
    # Handle surname particles (de, del, la, van, von, etc.)
    surname_particles = {'de', 'del', 'la', 'las', 'los', 'van', 'von', 'der', 'den', 'di', 'da', 'do', 'dos', 'das'}
    last_name_start = len(parts) - 1
    for i in range(len(parts) - 2, 0, -1):
        if parts[i].lower() in surname_particles:
            last_name_start = i
        else:
            break
    last_name = ' '.join(parts[last_name_start:]) + suffix
    first_name = ' '.join(parts[:last_name_start])
    if not first_name or not last_name:
        return full_name
    return f"{last_name}, {first_name}"


# ═══════════════════════════════════════════════════════════════
# MLB API FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def load_mlb_id_cache():
    if os.path.exists(MLB_ID_CACHE_PATH):
        with open(MLB_ID_CACHE_PATH) as f:
            return json.load(f)
    return {}

def save_mlb_id_cache(cache):
    with open(MLB_ID_CACHE_PATH, 'w') as f:
        json.dump(cache, f)

def lookup_mlb_id(player_name, team_abbrev, cache):
    cache_key = f"{player_name}|{team_abbrev}"
    if cache_key in cache:
        return cache[cache_key]
    parts = player_name.split(', ')
    search_name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else player_name
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={urllib.parse.quote(search_name)}&sportIds=1,11,12,13,14&hydrate=currentTeam&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
        people = data.get('people', [])
        if team_id and people:
            for person in people:
                ct = person.get('currentTeam', {})
                parent = ct.get('parentOrgId') or ct.get('id')
                if parent == team_id or ct.get('id') == team_id:
                    cache[cache_key] = person['id']; return person['id']
        if people:
            last_name = parts[0] if len(parts) == 2 else player_name.split()[-1]
            for person in people:
                if person.get('lastName','').lower() == last_name.lower():
                    cache[cache_key] = person['id']; return person['id']
            cache[cache_key] = people[0]['id']; return people[0]['id']
    except Exception as e:
        print(f"  Warning: MLB ID lookup failed for {player_name}: {e}")
    cache[cache_key] = None
    return None

def fetch_player_metadata(mlb_id):
    """Fetch age and hand from MLB API."""
    if not mlb_id: return {'age': '??', 'hand': 'R'}
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}"
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        p = data.get('people', [{}])[0]
        return {
            'age': str(p.get('currentAge', '??')),
            'hand': p.get('pitchHand', {}).get('code', 'R'),
        }
    except Exception:
        return {'age': '??', 'hand': 'R'}

def fetch_headshot(mlb_id):
    """Fetch and process MLB headshot."""
    try:
        url = f'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{mlb_id}/headshot/67/current'
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        img = Image.open(BytesIO(urllib.request.urlopen(req, timeout=10).read()))
        ha = np.array(img.convert('RGBA'))
        gm = ((np.abs(ha[:,:,0].astype(int)-ha[:,:,1].astype(int))<15) &
              (np.abs(ha[:,:,1].astype(int)-ha[:,:,2].astype(int))<15) &
              (ha[:,:,0]>170) & (ha[:,:,0]<230))
        ha[gm] = [255,255,255,255]
        return Image.fromarray(ha)
    except Exception:
        # Return a placeholder
        img = Image.new('RGBA', (213, 320), (50, 50, 50, 255))
        return img

def fetch_game_pks_for_date(date_str, include_live=False, sport_id=1, team_filter=None):
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId={sport_id}&gameType=R,F,D,L,W"
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        game_pks = []
        for date_data in data.get('dates', []):
            for game in date_data.get('games', []):
                state = game.get('status', {}).get('abstractGameState', '')
                if state == 'Final' or (include_live and state == 'Live'):
                    if team_filter:
                        away = game.get('teams',{}).get('away',{}).get('team',{}).get('name','')
                        home = game.get('teams',{}).get('home',{}).get('team',{}).get('name','')
                        if team_filter not in away and team_filter not in home:
                            continue
                    game_pks.append(game['gamePk'])
        return game_pks
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return []

_person_name_cache = {}

def _lookup_person_lastfirst(person_id):
    """Fetch canonical lastFirstName from MLB people API (cached)."""
    if person_id in _person_name_cache:
        return _person_name_cache[person_id]
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{person_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            people = data.get('people', [])
            if people:
                name = people[0].get('lastFirstName', '')
                _person_name_cache[person_id] = name
                return name
    except Exception:
        pass
    _person_name_cache[person_id] = None
    return None

def fetch_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            box = json.loads(resp.read())
    except Exception as e:
        print(f"  Error fetching boxscore for {game_pk}: {e}")
        return None
    result = {'pitchers': []}
    for side in ['away', 'home']:
        team_data = box.get('teams', {}).get(side, {})
        team_name = team_data.get('team', {}).get('name', '')
        team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name) or MILB_TEAM_NAME_TO_ABBREV.get(team_name, team_name)
        pitcher_ids = team_data.get('pitchers', [])
        players = team_data.get('players', {})
        for idx, pid in enumerate(pitcher_ids):
            p = players.get(f'ID{pid}', {})
            person = p.get('person', {})
            full_name = person.get('fullName', '')
            stats = p.get('stats', {}).get('pitching', {})
            if not stats: continue
            last_first = person.get('lastFirstName', '')
            if not last_first:
                # MiLB boxscores omit lastFirstName — fetch from people API
                person_id = person.get('id')
                if person_id:
                    last_first = _lookup_person_lastfirst(person_id) or ''
                if not last_first and full_name:
                    last_first = _fullname_to_lastfirst(full_name)
            result['pitchers'].append({
                'name': last_first, 'team': team_abbrev,
                'outs': stats.get('outs', 0),
                'r': stats.get('runs', 0), 'er': stats.get('earnedRuns', 0),
                'h': stats.get('hits', 0), 'hr': stats.get('homeRuns', 0),
                'so': stats.get('strikeOuts', 0), 'bb': stats.get('baseOnBalls', 0),
                'tbf': stats.get('battersFaced', 0),
                'is_starter': idx == 0,
            })
    return result

def fetch_boxscores_for_team(date_str, team_abbrev, include_live=False, game_pk=None, per_game=False):
    """Fetch boxscore stats for all pitchers on a team for a given date.

    per_game=True returns a LIST of per-game {name: pbox} dicts (so a
    reliever pitching both ends of a doubleheader counts twice); the default
    merged dict is kept for the single-game (game_pk) caller."""
    milb_config = MILB_TEAMS.get(team_abbrev)
    if game_pk:
        game_pks = [int(game_pk)]
        print(f"  Using game PK: {game_pk}")
    else:
        print(f"  Fetching boxscores for {date_str}...")
        sport_id = milb_config['sport_id'] if milb_config else 1
        team_filter = milb_config['search_name'] if milb_config else None
        game_pks = fetch_game_pks_for_date(date_str, include_live=include_live,
                                            sport_id=sport_id, team_filter=team_filter)
        status = "games (including live)" if include_live else "completed games"
        print(f"  Found {len(game_pks)} {status}")
    pitcher_stats = {}
    games = []
    for gpk in game_pks:
        box = fetch_boxscore(gpk)
        if not box: continue
        game = {}
        for p in box['pitchers']:
            if p['team'] == team_abbrev:
                game[p['name']] = p
        games.append(game)
        pitcher_stats.update(game)
        time_module.sleep(0.1)
    return games if per_game else pitcher_stats


# ── Fast multi-date boxscore path (2026-07-30) ────────────────────────────
# The per-date loop above fetches the day's FULL league schedule and every
# boxscore in it (~16 requests/date, one team kept) — a season card burned
# ~1,700 sequential requests. This path is request-minimal with identical
# aggregation semantics: ONE ranged schedule call filtered to the team, a
# disk cache of parsed boxscores (Final games are immutable), and a small
# thread pool for the misses. Returns a LIST of per-GAME {name: pbox} dicts —
# per game, not per date, so a reliever pitching both ends of a doubleheader
# counts both appearances (the old per-date merge dropped one).

BOXSCORE_CACHE_PATH = os.path.join(os.path.dirname(METADATA_PATH), '_boxscore_cache.pkl')


def _load_boxscore_cache():
    try:
        import pickle
        with open(BOXSCORE_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_boxscore_cache(cache):
    try:
        import pickle
        with open(BOXSCORE_CACHE_PATH, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"  WARNING: could not save boxscore cache: {e}")


def fetch_boxscores_for_team_dates(dates, team_abbrev, include_live=False):
    """Boxscores for a team across many dates: one schedule request, cached +
    parallel boxscore fetches. Only dates in `dates` are used (parity with the
    sheet-driven per-date loop)."""
    milb_config = MILB_TEAMS.get(team_abbrev)
    sport_id = milb_config['sport_id'] if milb_config else 1
    date_set = set(dates)
    lo, hi = min(date_set), max(date_set)
    url = (f"https://statsapi.mlb.com/api/v1/schedule?startDate={lo}&endDate={hi}"
           f"&sportId={sport_id}&gameType=R,F,D,L,W")
    if not milb_config:
        team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
        if team_id:
            url += f"&teamId={team_id}"
    print(f"  Fetching {team_abbrev} schedule {lo} → {hi} (one request)...")
    games_by_date = {}   # date -> [(game_pk, is_final), ...] in schedule order
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        for date_data in data.get('dates', []):
            gd = date_data.get('date', '')
            if gd not in date_set:
                continue
            for game in date_data.get('games', []):
                state = game.get('status', {}).get('abstractGameState', '')
                if state != 'Final' and not (include_live and state == 'Live'):
                    continue
                if milb_config:
                    away = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                    home = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                    if milb_config['search_name'] not in away and milb_config['search_name'] not in home:
                        continue
                games_by_date.setdefault(gd, []).append(
                    (game['gamePk'], state == 'Final'))
    except Exception as e:
        print(f"  Error fetching ranged schedule: {e} — falling back to per-date fetch")
        return [g for gd in sorted(date_set)
                for g in fetch_boxscores_for_team(gd, team_abbrev,
                                                  include_live=include_live, per_game=True)]

    cache = _load_boxscore_cache()
    # Live games always refetch; Final games come from / land in the cache.
    to_fetch = sorted({pk for pks in games_by_date.values()
                       for (pk, final) in pks if not (final and str(pk) in cache)})
    fetched = {}
    if to_fetch:
        print(f"  Fetching {len(to_fetch)} boxscores "
              f"({sum(len(v) for v in games_by_date.values()) - len(to_fetch)} cached)...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            for pk, box in zip(to_fetch, ex.map(fetch_boxscore, to_fetch)):
                fetched[pk] = box
    else:
        print(f"  All {sum(len(v) for v in games_by_date.values())} boxscores cached")
    cache_dirty = False
    for pks in games_by_date.values():
        for (pk, final) in pks:
            if final and pk in fetched and fetched[pk] is not None:
                cache[str(pk)] = fetched[pk]
                cache_dirty = True
    if cache_dirty:
        _save_boxscore_cache(cache)

    out = []
    for gd in sorted(games_by_date):
        for (pk, final) in games_by_date[gd]:
            box = fetched.get(pk) if pk in fetched else cache.get(str(pk))
            if not box:
                continue
            game = {}
            for p in box['pitchers']:
                if p['team'] == team_abbrev:
                    game[p['name']] = p
            if game:
                out.append(game)
    return out


# ═══════════════════════════════════════════════════════════════
# CARD RENDERING (v30)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# PERCENTILE BUBBLE PANEL — ported from HitterCards.py so the pitcher
# card speaks the same visual language: a left-column stack of titled
# sections, each row a label + pill-bar-with-bubble + value. Bubbles are
# colored on the website's blue(bad)→red(good) percentile gradient. The
# ranks come from the pitcher leaderboard row (config['pctl_row']).
# ═══════════════════════════════════════════════════════════════
# (label, value_key, pctl_key, format_spec). Organized GM-style:
# outcome → swing-and-miss → contact suppression → stuff & command.
BUBBLE_COLUMNS = [
    ('RESULT', [
        # xRV (cumulative) above xRV/100 (rate): volume then efficiency, so a
        # reliever's strong rate and a starter's larger total both read.
        ('xRV',           'xRunValue', 'xRunValue_pctl', 'dec1+'),
        ('xRV/100',       'xRv100',    'xRv100_pctl',   'dec1+'),
        # Pitcher+ (2026-07-30): flagship composite right after the outcome
        # stat — mirrors the hitter card, where Hitter+ sits second in RESULT.
        ('Pitcher+',      'pitcherPlus', 'pitcherPlus_pctl', 'int'),
        # xwOBA dropped 2026-08-12 (scripts/bubble_redundancy.py): r = -0.98
        # with xRV/100 across 842 arms, so it was the same row twice in a
        # currency the card does not lead with.
        ('K%',            'kPct',      'kPct_pctl',     'pct1'),
        ('BB%',           'bbPct',     'bbPct_pctl',    'pct1'),
        ('K-BB%',         'kbbPct',    'kbbPct_pctl',   'pct1'),
    ]),
    # 2026-07-20 bubble prune (page-card parity, battery-backed): dropped
    # Z-Whiff% (r=.83 w/ Whiff%), 2K Whiff% (r=.86), xwOBAcon (pitcher-side
    # rel .26), Zone% (pred ~0), FPS% (pred ~0) — all still in page tables.
    ('SWING & MISS', [
        ('Whiff%',     'swStrPct',          'swStrPct_pctl',          'pct1'),
        # Z-Whiff% (2026-07-30, scripts/zwhiff_incremental.py): predicts
        # next-season xwOBA-against beyond Whiff%+Chase%+K% in 8/8 replicates
        # (mean partial r .14); the unique bat-missing signal lives in-zone.
        ('Z-Whiff%',   'izWhiffPct',        'izWhiffPct_pctl',        'pct1'),
        ('Chase%',     'chasePct',          'chasePct_pctl',          'pct1'),
    ]),
    ('CONTACT MGMT', [
        ('xwOBAcon',   'xwOBAcon',         'xwOBAcon_pctl',         '3dec'),
        # BABIP directly under xwOBAcon (2026-08-03, per Wally): the pair
        # reads realized-vs-expected on contact — a BABIP tint that fights
        # the xwOBAcon tint flags batted-ball fortune at a glance. Pctl is
        # pipeline-inverted (PITCHER_INVERT_PCTL): lower BABIP = red.
        ('BABIP',      'babip',            'babip_pctl',            '3dec'),
        ('Hard-Hit%',  'hardHitPct',       'hardHitPct_pctl',       'pct1'),
        ('Barrel%',    'barrelPctAgainst', 'barrelPctAgainst_pctl', 'pct1'),
        # HR/FB% under Barrel% (2026-08-14, per Wally): damage quality vs
        # damage REALIZED — the pair reads HR fortune the way xwOBAcon/BABIP
        # reads batted-ball fortune. Measured split-half reliability in 2026
        # is ~0 (r=-0.09 at 20+ FB/half; the xFIP thesis), so this bubble is
        # a fortune flag by design, never a skill grade. Lower = better.
        ('HR/FB%',     'hrFbPct',          'hrFbPct_pctl',          'pct1'),
        ('GB%',        'gbPct',            'gbPct_pctl',            'pct1'),
        # PU% under GB% (2026-08-14, per Wally): the other free out. Real
        # skill: split-half r=0.43-0.45 vs GB%'s 0.59 on the same protocol.
        # Higher = better. (FanGraphs calls the cousin stat IFFB%; this is
        # popups per BIP, matching the sheets' BBType currency.)
        ('PU%',        'puPct',            'puPct_pctl',            'pct1'),
    ]),
    ('COMMAND & SHAPE', [
        ('Velocity',   'fbVelo',    'fbVelo_pctl',    'mph'),
        # Extension sits with Velocity as the second physical/release input:
        # r = +0.011 with velo and 0.11 against anything else on the card
        # (scripts/bubble_redundancy.py), and perceived velo is a function of
        # both, so the pair reads together.
        ('Extension',  'extension', 'extension_pctl', 'ftin'),
        ('Stuff+',     'stuffScore', 'stuffScore_pctl', 'int'),
        ('Loc+',       'locPlus',   'locPlus_pctl',   'int'),
        # Pitching+ bubble removed 2026-08-13 (per Wally): it is the
        # deterministic 0.8/0.2 blend of the two bubbles directly above it —
        # same reasoning as dropping the per-pitch table column.
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
        # Site convention: no leading 0 on rate stats like .425
        return s[1:] if s.startswith('0.') else (f'-{s[2:]}' if s.startswith('-0.') else s)
    if spec == 'dec2':
        return f'{v:.2f}'           # ERA/SIERA/xFIP keep the leading number
    if spec == 'pct1':
        return f'{v * 100:.1f}%' if abs(v) <= 1 else f'{v:.1f}%'
    if spec == 'int':
        return f'{int(round(v))}'
    if spec == 'dec1':
        v = v + 0.0 if abs(v) >= 0.05 else 0.0   # avoid '-0.0'
        return f'{v:.1f}'
    if spec == 'dec1+':
        # Per memory: never prefix positives with '+'. Negatives still get '-'.
        v = v if abs(v) >= 0.05 else 0.0          # avoid '-0.0'
        return f'{v:.1f}'
    if spec == 'mph':
        return f'{v:.1f} mph'
    if spec == 'ft':
        return f'{v:.1f} ft'
    if spec == 'ftin':
        # Feet and inches (6'10"), same as the pitch table's Ext column, so the
        # bubble and the table below it read in the same units.
        return fmt_fi(v)
    if spec == 'deg':
        return f'{v:.1f}°'
    return str(v)


def _percentile_color(pctl):
    """PRINT-IDENTITY percentile scale — matches the redesigned website's BUBBLE
    scale (Utils.percentileBubbleColor) and the hitter cards. Blends from a
    VISIBLE warm-greige floor at the 50th percentile toward slate (low) or brick
    (high), so mid-percentile bubbles read as filled discs on cream instead of
    vanishing into the paper. Endpoints kept light enough for ink text.
    pctl is 0-100, already directionally normalized (high = good for pitcher).
    Returns (fill_rgb01, ring_rgb01)."""
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


def _pitcher_stat_cell_color(value_str, league_avg, scale, higher_is_better,
                             row_bg_hex, is_pct):
    """Headline-strip / table cell tint in the SAME blue→red hue family as the
    percentile bubbles: red = better than league avg (good for pitcher), blue =
    worse. Ported from HitterCards._hitter_stat_cell_color so the whole pitcher
    card speaks one color language (replaces the old green/red pct_cell_color).
    """
    if league_avg is None or not value_str or value_str == '—':
        return None
    try:
        if is_pct:
            val = float(value_str.replace('%', ''))
            diff = val - league_avg * 100
            denom = 8.0                      # ±8 pp → full intensity
        else:
            # Handle feet-inches (Ext, e.g. 6'3"), then plain numbers with any
            # trailing unit glyphs (", °, ' ft').
            val = _parse_fi(str(value_str))
            if val is None:
                val = float(str(value_str).replace('"', '').replace('°', '').replace(' ft', ''))
            diff = val - league_avg
            denom = scale
    except (ValueError, AttributeError):
        return None
    if not higher_is_better:
        diff = -diff
    intensity = max(-1.0, min(1.0, diff / denom))
    anchor = _percentile_color(100 if intensity >= 0 else 0)[0]
    target = tuple(int(round(ch * 255)) for ch in anchor)
    alpha = abs(intensity) * 0.72
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


def _z_cell_color(z, row_bg_hex, full_at=2.0):
    """Cell tint from a SIGNED z-score, same blue->red family as the rest of
    the card. Used only on daily cards, where the comparison is the pitcher's
    OWN season baseline rather than the league. z>0 (better/more) -> red.

    full_at is in standard errors: the ramp saturates at 2 SE, so a faint cell
    is a gap inside normal game-to-game noise and a saturated one is not.
    """
    if z is None:
        return None
    intensity = max(-1.0, min(1.0, z / full_at))
    anchor = _percentile_color(100 if intensity >= 0 else 0)[0]
    target = tuple(int(round(ch * 255)) for ch in anchor)
    alpha = abs(intensity) * 0.72
    rb, rg, rbb = (int(row_bg_hex[1:3], 16), int(row_bg_hex[3:5], 16),
                   int(row_bg_hex[5:7], 16))
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


def _blom_expected_max(n):
    """E[max of n standard normals], Blom's approximation.

    A max is not a mean. The max of 34 pitches sits systematically BELOW the
    max of 386 purely from sample size (~0.75 mph on a fastball), so comparing
    a start's Max Velo to a season Max Velo without this correction paints
    almost every start blue. Bisects Phi rather than pulling in scipy for the
    one call: math has erf but no erfinv.
    """
    if n is None or n < 2:
        return None
    q = (n - 0.375) / (n + 0.25)
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _pctl_cell_color(pctl, row_bg_hex):
    """Table-cell tint driven directly by an already-directional percentile
    (0-100, high = good), in the same blue→red family as _pitcher_stat_cell_color.
    Used for nVAA, whose 'good' direction flips by pitch type and is baked into
    the precomputed nVAA_pctl (FF: flatter/closer-to-zero better; SI: steeper)."""
    if pctl is None:
        return None
    intensity = max(-1.0, min(1.0, (pctl - 50) / 50.0))
    anchor = _percentile_color(100 if intensity >= 0 else 0)[0]
    target = tuple(int(round(ch * 255)) for ch in anchor)
    alpha = abs(intensity) * 0.72
    rb = int(row_bg_hex[1:3], 16); rg = int(row_bg_hex[3:5], 16); rbb = int(row_bg_hex[5:7], 16)
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'


def _velo_pctl_vs_ff(velo, config):
    """Approximate percentile of a velo vs the league FF velo distribution. Used
    only for the rare FC-only fallback bubble (a pitcher with no FF and no SI).
    Normal-CDF approx with the league FF mean and an assumed ~2.2 mph SD."""
    import math
    la = (config.get('league_avgs') or {}).get('FF') or {}
    mean = la.get('velocity')
    if mean is None or velo is None:
        return None
    sd = 2.2
    z = (velo - mean) / (sd * math.sqrt(2))
    return max(0.0, min(100.0, 100 * 0.5 * (1 + math.erf(z))))


def _bubble_columns_for(config, p_row):
    """Split the single 'Velocity' bubble into Fastball/Sinker velo bubbles
    (graded vs MLB same-pitch-type velo). A pitcher with neither falls back to a
    Cutter velo bubble graded vs league FF velo. Applies to MLB and ROC."""
    pitch_lb = config.get('pitch_lb') or {}
    def _vel(pt):
        d = pitch_lb.get(pt) or {}
        return d.get('velocity'), d.get('velocity_pctl')
    velo_rows = []
    # ONE fastball bubble (2026-07-28, per Wally): count-weighted average velo
    # across every fastball variant (FF/FA/SI) instead of separate FF and SI
    # rows that almost always read within half a tick of each other. The
    # percentile is the leaderboard's fbVelo rank (primary-fastball
    # definition) — the combined average differs by <0.2 mph in practice and
    # keeps the bubble on the same pool as the site. Falls back to the plain
    # per-type value when counts are missing (older leaderboard JSONs).
    _fbs = []
    for _t in ('FF', 'FA', 'SI'):
        _d = pitch_lb.get(_t) or {}
        if _d.get('velocity') is not None:
            _fbs.append((_d['velocity'], _d.get('count') or 0, _d.get('velocity_pctl')))
    if _fbs:
        _wsum = sum(n for _v, n, _p in _fbs)
        if _wsum > 0:
            p_row['fbCombVelo'] = round(sum(v * n for v, n, _p in _fbs) / _wsum, 1)
        else:
            p_row['fbCombVelo'] = round(sum(v for v, _n, _p in _fbs) / len(_fbs), 1)
        p_row['fbCombVelo_pctl'] = (p_row.get('fbVelo_pctl')
                                    if p_row.get('fbVelo_pctl') is not None
                                    else _fbs[0][2])
        velo_rows.append(('Fastball Velo', 'fbCombVelo', 'fbCombVelo_pctl', 'mph'))
    if not velo_rows:
        fc_v, _ = _vel('FC')
        if fc_v is not None:
            p_row['fcVelo'] = fc_v
            p_row['fcVelo_pctl'] = _velo_pctl_vs_ff(fc_v, config)
            velo_rows.append(('Cutter Velo', 'fcVelo', 'fcVelo_pctl', 'mph'))
    if not velo_rows:
        return BUBBLE_COLUMNS
    # Rebuild columns, swapping the single 'Velocity' row for the velo rows.
    new_cols = []
    for name, metrics in BUBBLE_COLUMNS:
        new_metrics = []
        for m in metrics:
            if m[1] == 'fbVelo':
                new_metrics.extend(velo_rows)
            else:
                new_metrics.append(m)
        new_cols.append((name, new_metrics))
    return new_cols


def _render_percentile_bubbles(fig, p_row, grid_left, grid_right, grid_top, grid_bot, columns=None):
    """Left-column percentile panel (mirrors the website PERCENTILE RANKINGS
    sidebar). Vertical stack of section sub-headers + pill-bar rows. Grid bounds
    are passed in (fig coords) so the layout can be tuned from the call site.
    ROC pitchers: sections whose every metric is missing are dropped."""
    from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch

    col_w = grid_right - grid_left

    # Drop a section if the pitcher has no data for ANY metric in it (ROC/AAA
    # pitchers are missing the Statcast-derived bubbles — show nothing rather
    # than a column of dashes).
    _columns = []
    for name, metrics in (columns or BUBBLE_COLUMNS):
        if any(p_row.get(vk) is not None for _l, vk, _pk, _f in metrics):
            _columns.append((name, metrics))
    total_rows = sum(len(m) for _h, m in _columns)
    n_sections = len(_columns)
    if total_rows == 0:
        return

    grid_h = grid_top - grid_bot
    # Vertical spacing is fixed in INCHES (converted to fig fractions here) so
    # the rail renders identically whatever the figure's total height (the
    # season card grew 0.7in for the velo sparkline). Inch values = the
    # original fractions x the classic 17.5in frame.
    _fh_in = fig.get_size_inches()[1]
    SECTION_HEADER_H = 0.350 / _fh_in   # 0.020 * 17.5
    SECTION_TOP_GAP  = 0.105 / _fh_in   # 0.006 * 17.5
    SECTION_GAP      = 0.280 / _fh_in   # 0.016 * 17.5
    fixed_overhead = (n_sections * (SECTION_HEADER_H + SECTION_TOP_GAP)
                       + (n_sections - 1) * SECTION_GAP)
    row_h = (grid_h - fixed_overhead) / total_rows

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off'); ax.set_zorder(5)

    # Label and value gutters MEASURED from the strings this card will actually
    # draw, rather than reserved as a fixed fraction of the column. The old
    # 0.34/0.17 split left a large dead margin on every row; reclaiming it
    # lengthens the bar, and bar length is what makes percentiles legible —
    # travel per percentile point is (effective_bar_w - MIN_VISIBLE) / 100.
    ROW_FS = 12.5
    _labels = [m[0] for _s, ms in _columns for m in ms]
    _values = [_format_bubble_value(p_row.get(m[1]), m[3])
               for _s, ms in _columns for m in ms]
    _lw = _measure_text_axis_w(fig, _labels, ROW_FS, '500')
    _vw = _measure_text_axis_w(fig, _values, ROW_FS, '600')
    LABEL_W = _lw if _lw else col_w * 0.34
    VALUE_W = _vw if _vw else col_w * 0.17
    # Breathing room so the bar never crowds the text on either side.
    LABEL_BAR_GAP = 0.012
    BAR_VALUE_GAP = 0.012

    BAR_HEIGHT_IN  = 0.34
    bar_h_axis     = BAR_HEIGHT_IN / fig.get_size_inches()[1]
    CIRCLE_DIAM_IN = 0.40
    ellipse_w = CIRCLE_DIAM_IN / fig.get_size_inches()[0]
    ellipse_h = CIRCLE_DIAM_IN / fig.get_size_inches()[1]
    CIRCLE_CLEARANCE_AXIS_X = (CIRCLE_DIAM_IN / fig.get_size_inches()[0]) * 0.55

    x_label_left  = grid_left
    x_label_right = grid_left + LABEL_W
    x_bar_left    = x_label_right + LABEL_BAR_GAP
    x_bar_zone_right = grid_right - VALUE_W - BAR_VALUE_GAP
    x_bar_right   = x_bar_zone_right - CIRCLE_CLEARANCE_AXIS_X
    x_value_right = grid_right
    bar_total_w   = x_bar_right - x_bar_left
    rounding = bar_h_axis / 2

    y_cursor = grid_top
    for sec_idx, (section, metrics) in enumerate(_columns):
        if sec_idx > 0:
            y_cursor -= SECTION_GAP
        header_y = y_cursor
        ax.text(grid_left, header_y, section, ha='left', va='top',
                fontsize=12.5, fontfamily='IBM Plex Sans Condensed', fontweight='700',
                color=TEXT_SECONDARY)
        rule_y = header_y - SECTION_HEADER_H + 0.035 / _fh_in   # 0.002 * 17.5
        ax.add_patch(Rectangle((grid_left, rule_y), col_w, 0.0175 / _fh_in,
                                facecolor=TEXT_FAINT, edgecolor='none', alpha=0.5))
        y_cursor = header_y - SECTION_HEADER_H - SECTION_TOP_GAP

        for label, val_key, pctl_key, fmt_spec in metrics:
            row_top = y_cursor
            row_bot = y_cursor - row_h
            row_mid = (row_top + row_bot) / 2
            y_cursor = row_bot

            val = p_row.get(val_key)
            pctl = p_row.get(pctl_key)
            val_str = _format_bubble_value(val, fmt_spec)
            fill_color, ring_color = _percentile_color(pctl)

            ax.text(x_label_left, row_mid, label, ha='left', va='center',
                    fontsize=12.5, fontfamily='IBM Plex Sans', fontweight='500',
                    color=TEXT_PRIMARY)

            track_y = row_mid - bar_h_axis / 2
            track = FancyBboxPatch(
                (x_bar_left + rounding, track_y),
                bar_total_w - 2 * rounding, bar_h_axis,
                boxstyle=f'round,pad=0,rounding_size={rounding}',
                facecolor=TEXT_FAINT, edgecolor='none', alpha=0.20,
                linewidth=0, zorder=8)
            ax.add_patch(track)

            radius_x = ellipse_w / 2
            effective_bar_w = bar_total_w - 2 * radius_x
            p = max(0, min(100, pctl)) / 100.0 if pctl is not None else 0
            MIN_VISIBLE = radius_x * 1.5
            visible_fill_w = MIN_VISIBLE + p * (effective_bar_w - MIN_VISIBLE)
            FILL_INTO_CIRCLE = radius_x * 0.85
            fill_render_w = visible_fill_w + FILL_INTO_CIRCLE
            if pctl is not None and fill_render_w > 0:
                fill = Rectangle((x_bar_left, track_y), fill_render_w, bar_h_axis,
                                 facecolor=fill_color, edgecolor='none',
                                 alpha=0.95, zorder=9)
                ax.add_patch(fill)
                fill.set_clip_path(track)

            circle_x = x_bar_left + visible_fill_w + radius_x
            ell = Ellipse((circle_x, row_mid), ellipse_w, ellipse_h,
                           facecolor=ring_color, edgecolor='none',
                           linewidth=0, zorder=12)
            ax.add_patch(ell)
            label_pctl = f'{int(round(pctl))}' if pctl is not None else '—'
            ax.text(circle_x, row_mid, label_pctl, ha='center', va='center',
                    fontsize=10.5, fontfamily='IBM Plex Sans', fontweight='700',
                    color=TEXT_PRIMARY, zorder=13)

            ax.text(x_value_right, row_mid, val_str, ha='right', va='center',
                    fontsize=12.5, fontfamily='IBM Plex Sans', fontweight='600',
                    color=TEXT_PRIMARY)


# Batted-ball palette tuned for the warm-paper theme.
_BB_TYPES = ['ground_ball', 'line_drive', 'fly_ball', 'popup']
_BB_COLORS = {'ground_ball': '#2E8FA8', 'line_drive': '#FF6B6B',
              'fly_ball': '#7B68EE', 'popup': '#FF9F43'}
_BB_LABELS = {'ground_ball': 'Ground Ball', 'line_drive': 'Line Drive',
              'fly_ball': 'Fly Ball', 'popup': 'Popup'}


def _render_single_game_panel(fig, pitches, config=None):
    """Single-game extras in the old layout (warm-paper palette): a batted-ball
    donut + per-pitch stacked bars top-left (below the stat strip), and per-hand
    usage bars on the right (below the movement plot). Location plots are placed
    on the left by render_card; the percentile-bubble panel is skipped."""
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.patches as mpatches
    TRACK = '#d8ccb4'   # warm bar track

    # ── VELOCITY BY PITCH ──────────────────────────────────────────────
    # Replaces the batted-ball donut AND the per-pitch-type stacked bars. Both
    # described 8 batted balls across 2 categories, which is closer to
    # decoration than information; the freed width is spent on the one thing a
    # 78-pitch sample supports, which is the shape of his velocity through the
    # start. Dashed lines are his own season average for that pitch, so a tired
    # start is visible rather than inferred.
    #
    # ORDER MATTERS. `pitches` arrives grouped by pitch type, not
    # chronological, so plotting it as received produced a tidy monotonic slide
    # that was an artifact of the sort and not of his arm. PitchID is
    # game_pk_atbat_pitch, zero-padded, so a lexicographic sort is exact game
    # order.
    _sl = (config or {}).get('season_pitch_lb') or {}
    ax_v = fig.add_axes([0.012, 0.596, 0.458, 0.140]); ax_v.set_facecolor(PLOT_PANEL)
    _ord = sorted(pitches, key=lambda q: str(q.get('PitchID') or ''))
    _seq = [(i2 + 1, sf(q.get('Velocity')), q.get('Pitch Type', ''))
            for i2, q in enumerate(_ord)]
    _seq = [(i2, v, pt) for i2, v, pt in _seq if v is not None and pt]
    if _seq:
        _n_by_pt = defaultdict(int)
        for _, _, pt in _seq:
            _n_by_pt[pt] += 1
        _xmax = max(i2 for i2, _, _ in _seq) + 1
        # Reference line for every pitch type he threw today whose SEASON
        # sample supports a baseline. There is deliberately no floor on today's
        # count: an 8-pitch curveball still has a real season average, and
        # suppressing the line just left the reader wondering where it went.
        for pt, cnt in _n_by_pt.items():
            _sb = _sl.get(pt) or {}
            if (_sb.get('count') or 0) < SEASON_DELTA_MIN:
                continue
            if _sb.get('velocity') is None:
                continue
            ax_v.axhline(_sb['velocity'], color=PITCH_COLORS.get(pt, '#999'),
                         linewidth=1.1, linestyle='--', alpha=0.6, zorder=1)
        for pt in PITCH_ORDER:
            _pp2 = [(i2, v) for i2, v, q in _seq if q == pt]
            if not _pp2:
                continue
            ax_v.scatter([i2 for i2, _ in _pp2], [v for _, v in _pp2],
                         c=PITCH_COLORS.get(pt, '#999'), s=30, alpha=1.0,
                         edgecolors=PLOT_PANEL, linewidths=0.4, zorder=3)
        ax_v.set_xlim(0, _xmax)
        ax_v.tick_params(labelsize=8, colors=TEXT_MUTED, length=2.5, pad=2)
        ax_v.grid(True, alpha=0.4, color=GRID_COLOR, linewidth=0.6)
        for _sp in ax_v.spines.values():
            _sp.set_color(TEXT_FAINT)
        ax_v.set_xlabel('pitch number', fontsize=8.5, color=TEXT_MUTED,
                        fontweight='bold', fontfamily='IBM Plex Sans', labelpad=1.5)
        ax_v.set_ylabel('MPH', fontsize=8.5, color=TEXT_MUTED, fontweight='bold',
                        fontfamily='IBM Plex Sans', labelpad=2)
        # Above the axes: inside, the pitch dots ran straight through it.
        fig.text(0.012, 0.742, 'VELOCITY BY PITCH', fontsize=11,
                 fontweight='bold', color=TEXT_SECONDARY,
                 fontfamily='IBM Plex Sans', va='bottom', ha='left')
        # Notes render BLACK on every layout (2026-08-13, per Wally — the
        # muted gray read too faint at card scale).
        ax_v.text(0.992, 0.04, 'dashed = his season average',
                  transform=ax_v.transAxes, fontsize=7.5, color='#000000',
                  fontfamily='IBM Plex Sans', va='bottom', ha='right', zorder=6)
    else:
        ax_v.axis('off')

    # ── usage bars (right, below the movement plot) ──
    usage = {'L': defaultdict(int), 'R': defaultdict(int)}
    tot = {'L': 0, 'R': 0}
    for p in pitches:
        bh, pt = p.get('Bats', ''), p.get('Pitch Type', '')
        if bh in ('L', 'R') and pt:
            usage[bh][pt] += 1; tot[bh] += 1

    def _usage(rect, data, total, title):
        ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off'); ax.set_facecolor(BG)
        # Pitch count in the title: the bars are percentages, so without it a
        # 55.6% slider off 18 pitches looks the same as one off 180.
        _ttl = title if not total else f'{title} ({total} pitch{"" if total == 1 else "es"})'
        ax.text(0.5, 0.99, _ttl, fontsize=12, fontweight='bold', ha='center', va='top',
                color=TEXT_SECONDARY, fontfamily='IBM Plex Sans')
        spts = sorted(data, key=lambda x: (-data[x], PITCH_ORDER.index(x) if x in PITCH_ORDER else 99))
        if not spts:
            return
        n = len(spts); rh = min(0.13, 0.78 / n); gap = 0.02
        for i, pt in enumerate(spts):
            y = 0.80 - i * (rh + gap); pct = data[pt] / total if total else 0
            color = PITCH_COLORS.get(pt, '#999'); tcb = badge_text_color(color)
            ax.add_patch(FancyBboxPatch((0.04, y - rh * 0.4), 0.10, rh * 0.8,
                         boxstyle="round,pad=0.006", facecolor=color, edgecolor='none'))
            ax.text(0.09, y, pt, fontsize=8, ha='center', va='center', color=tcb, fontweight='bold')
            ax.add_patch(Rectangle((0.17, y - rh * 0.28), 0.58, rh * 0.56, facecolor=TRACK, edgecolor='none'))
            if pct > 0:
                ax.add_patch(Rectangle((0.17, y - rh * 0.28), 0.58 * pct, rh * 0.56, facecolor=color, edgecolor='none'))
            ax.text(0.78, y, ("< 1%" if 0 < pct*100 < 1 else f'{pct*100:.1f}%'), fontsize=10, va='center', ha='left',
                    color=TEXT_PRIMARY, fontweight='bold', fontfamily='IBM Plex Sans')

    _usage([0.55, 0.32, 0.22, 0.17], usage['R'], tot['R'], 'VS RHH')
    _usage([0.77, 0.32, 0.22, 0.17], usage['L'], tot['L'], 'VS LHH')


def render_card(config, pitches, output_file):
    """Render a single pitcher card. config has display_name, hand, team, age, game_date, stat_headers, stat_values, headshot, mlb_id."""
    headshot = config['headshot']

    # Compute pitch data
    locations = {'L': defaultdict(list), 'R': defaultdict(list)}
    sz_tops, sz_bots = [], []
    groups = defaultdict(list)

    for p in pitches:
        pt = p.get('Pitch Type', '')
        # Adjusted movement only (sheet cols L/M) — no raw fallback.
        hb = p.get('xHorzBrk')
        ivb = p.get('xIndVrtBrk')
        if pt and hb is not None and hb != '' and ivb is not None and ivb != '':
            try: groups[pt].append((float(hb), float(ivb)))
            except Exception: pass
        bh = p.get('Bats', '')
        px, pz = p.get('PlateX'), p.get('PlateZ')
        szt, szb = p.get('SzTop'), p.get('SzBot')
        if bh in ('L','R') and pt and px is not None and px != '' and pz is not None and pz != '':
            try:
                desc = p.get('Description', '')
                is_b = str(p.get('Barrel', '')).strip() == '6'
                locations[bh][pt].append((float(px), float(pz), desc, is_b))
            except Exception: pass
        if szt is not None and szt != '' and szb is not None and szb != '':
            try: sz_tops.append(float(szt)); sz_bots.append(float(szb))
            except Exception: pass

    avg_top = np.mean(sz_tops) if sz_tops else 3.5
    avg_bot = np.mean(sz_bots) if sz_bots else 1.5
    sorted_types = [pt for pt in PITCH_ORDER if pt in groups]

    # Batted ball distribution per pitch type
    bb_by_pitch = defaultdict(lambda: {'ground_ball': 0, 'line_drive': 0, 'fly_ball': 0, 'popup': 0, 'hh': 0, 'brl': 0})
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if not pt or p.get('Description') != 'In Play':
            continue
        bbt = str(p.get('BBType', '')).strip()
        if not bbt or bbt.startswith('bunt'):
            continue
        if bbt in BB_TYPES:
            bb_by_pitch[pt][bbt] += 1
        ev = sf(p.get('ExitVelo'))
        if ev is not None and ev >= 95:
            bb_by_pitch[pt]['hh'] += 1
        if str(p.get('Barrel', '')).strip() == '6':
            bb_by_pitch[pt]['brl'] += 1

    if not sorted_types:
        print(f"  WARNING: No pitch type data for {config['display_name']}, skipping")
        return False

    # Single-game keeps the old wider frame (~1.22 ratio). Season cards grew
    # 0.7in taller than the classic 17.5in frame to make room for the velo
    # sparkline; everything below the header block is re-anchored in INCHES so
    # it renders pixel-identical to the classic card (the extra height is
    # absorbed between the boxscore strip and the percentile rail).
    fig_h = 14.3 if not config.get('mvn_models') else FIG_H + 0.7
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax_main = fig.add_axes([0,0,1,1])
    ax_main.set_xlim(0, FIG_W); ax_main.set_ylim(0, fig_h)
    ax_main.axis('off'); ax_main.set_facecolor(BG)

    # Stripe — usage-ordered, equal widths, aligned with photo. Anchored near
    # the top of the (taller) figure.
    photo_left = TABLE_LEFT_FIG * FIG_W
    stripe_bottom = fig_h - 0.20
    stripe_height = 0.22
    stripe_x = photo_left
    total_w = FIG_W * TABLE_RIGHT_FIG - photo_left
    stripe_counts = {pt: sum(1 for p in pitches if p.get('Pitch Type') == pt) for pt in sorted_types}
    stripe_order = sorted(sorted_types,
        key=lambda pt: (-stripe_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999))
    for pt in stripe_order:
        w = total_w / len(sorted_types)
        ax_main.add_patch(Rectangle((stripe_x, stripe_bottom), w, stripe_height,
            facecolor=PITCH_COLORS.get(pt, '#999'), edgecolor='none', zorder=6))
        stripe_x += w

    # Photo
    photo_w = 1.4; photo_h = photo_w * headshot.size[1] / headshot.size[0]
    photo_top = fig_h - 0.25; photo_bottom = photo_top - photo_h
    ax_main.imshow(np.array(headshot), extent=[photo_left, photo_left+photo_w, photo_bottom, photo_top], aspect='auto', zorder=2, interpolation='antialiased')
    ax_main.add_patch(Rectangle((photo_left, photo_bottom), photo_w, photo_h, fill=False, edgecolor=PHOTO_BORDER, linewidth=1.5, alpha=0.8, zorder=3))

    photo_right = photo_left + photo_w; text_x = photo_right + 0.3
    ax_main.text(text_x, photo_top-0.1, config['display_name'], fontsize=28, fontfamily='Bitter', color=TEXT_PRIMARY, va='top', fontweight='black')
    hand_code = 'LHP' if config['hand'] == 'L' else 'RHP'
    ax_main.text(text_x, photo_top-0.85, f"{hand_code}  |  {config['team']}  |  Age: {config['age']}", fontsize=12, fontfamily='IBM Plex Sans', color=TEXT_MUTED, va='top')
    # Date line — single-game cards append the opponent to the SAME string, so
    # it renders in one font, one size, one colour. A daily report that never
    # says who he faced is missing the most basic frame. Score/decision would
    # need the linescore endpoint and is deliberately left off.
    _date_txt = config['game_date']
    _opp = config.get('opponent')
    if _opp and not bool(config.get('mvn_models')):
        _date_txt = f"{_date_txt} vs. {_opp}"
    ax_main.text(text_x, photo_top-1.5, _date_txt, fontsize=24, fontfamily='IBM Plex Sans', color=ACCENT, va='top')

    # Stat line — season cards widen the 5-cell strip so it spans the bubble
    # column beneath it. Single-game cards have no bubble column, so they use
    # a much tighter column so the (now 8-cell) strip doesn't run too wide.
    is_season_strip = bool(config.get('mvn_models'))
    # 5-cell strips (hdERA/hpERA season layout) widen to 1.33 so the strip
    # still spans the bubble column; 6-cell (range cards: ERA/FIP/SIERA)
    # keep 1.25 for the same total width.
    col_w = ((1.33 if len(config['stat_headers']) == 5 else 1.25)
             if is_season_strip else 0.72)
    cell_h = 0.46
    hdr_fs = 11 if is_season_strip else 10
    val_fs = 14 if is_season_strip else 13
    stat_y_header = photo_bottom - 0.5; stat_y_value = stat_y_header - cell_h
    pitcher_la = config.get('pitcher_league_avgs', {})
    for i in range(len(config['stat_headers'])):
        x = photo_left + i * col_w
        hdr = config['stat_headers'][i]
        val_str = config['stat_values'][i]
        ax_main.add_patch(Rectangle((x, stat_y_header), col_w, cell_h, facecolor=DARKER, edgecolor=SUBTLE_BORDER, linewidth=0.8))
        ax_main.text(x+col_w/2, stat_y_header+cell_h/2, hdr, fontsize=hdr_fs, ha='center', va='center', color=TEXT_SECONDARY, fontweight='bold', fontfamily='IBM Plex Sans Condensed')
        # Determine cell color — blue→red percentile hue (matches the bubbles).
        cell_bg = DARK_CELL
        sl_cfg = STAT_LINE_COLOR.get(hdr)
        if sl_cfg and pitcher_la:
            la_val = pitcher_la.get(sl_cfg[0])
            if la_val is not None and val_str and val_str != '—':
                is_pct = (sl_cfg[1] == 'pct')
                scale = sl_cfg[3] if len(sl_cfg) > 3 else 1.0
                tinted = _pitcher_stat_cell_color(val_str, la_val, scale, sl_cfg[2],
                                                  DARK_CELL, is_pct)
                if tinted:
                    cell_bg = tinted
        ax_main.add_patch(Rectangle((x, stat_y_value), col_w, cell_h, facecolor=cell_bg, edgecolor=SUBTLE_BORDER, linewidth=0.8))
        ax_main.text(x+col_w/2, stat_y_value+cell_h/2, val_str, fontsize=val_fs, ha='center', va='center', color=TEXT_PRIMARY, fontweight='bold', fontfamily='IBM Plex Sans')
    ax_main.add_patch(Rectangle((photo_left, stat_y_value), len(config['stat_headers'])*col_w, stat_y_header+cell_h-stat_y_value, fill=False, edgecolor=ACCENT, linewidth=2, zorder=5))

    # ── FB velo-by-outing sparkline — season cards only, thin strip directly
    # under the boxscore line. Combined fastball pool (FF/FA/SI) average velo
    # per game date — the same count-weighted definition as the Fastball Velo
    # bubble, so the strip's avg matches the bubble's mph. Muted dots on a
    # thin line, dotted season-average reference, "last · avg · max" above,
    # first/mid/last date labels below. Skips gracefully under 3 games. Lives
    # in the 0.7in of extra card height, so nothing below has to yield.
    if config.get('mvn_models'):
        _FB_POOL = ('FF', 'FA', 'SI')
        _velo_by_start = defaultdict(list)
        _fb_counts = defaultdict(int)
        for p in pitches:
            _pt_ = p.get('Pitch Type')
            if _pt_ not in _FB_POOL:
                continue
            _v = sf(p.get('Velocity')); _gd = p.get('Game Date')
            if _v is not None and _gd:
                _velo_by_start[_gd].append(_v)
                _fb_counts[_pt_] += 1
        # dominant type just picks the accent color for the season-high dot
        _fb_type = max(_fb_counts, key=_fb_counts.get) if _fb_counts else 'FF'
        _sdates = sorted(_velo_by_start)
        if len(_sdates) >= 3:
            _svelos = [float(np.mean(_velo_by_start[d])) for d in _sdates]
            _savg = float(np.mean([v for d in _sdates for v in _velo_by_start[d]]))
            strip_w_in = len(config['stat_headers']) * col_w   # same width as the stat strip
            strip_h_in = 0.27                                  # ~40 px at save scale
            strip_top = stat_y_value - 0.36
            ax_spark = fig.add_axes([photo_left / FIG_W, (strip_top - strip_h_in) / fig_h,
                                     strip_w_in / FIG_W, strip_h_in / fig_h])
            ax_spark.set_facecolor(BG)
            _sxs = np.arange(len(_sdates))
            ax_spark.set_xlim(-0.6, len(_sdates) - 0.4)
            _spad = 0.6
            ax_spark.set_ylim(min(_svelos) - _spad, max(_svelos) + _spad)
            ax_spark.axhline(_savg, color=TEXT_FAINT, lw=0.8, ls=(0, (2, 3)), alpha=0.8, zorder=1)
            ax_spark.plot(_sxs, _svelos, color=TEXT_MUTED, lw=1.1, alpha=0.85, zorder=2)
            ax_spark.scatter(_sxs, _svelos, s=16, c=TEXT_MUTED, zorder=3)
            # accent the season high + latest start
            _shi = int(np.argmax(_svelos))
            ax_spark.scatter([_shi], [_svelos[_shi]], s=22,
                             c=PITCH_COLORS.get(_fb_type, '#0072B2'), zorder=4)
            ax_spark.scatter([_sxs[-1]], [_svelos[-1]], s=22, c=ACCENT, zorder=4)
            ax_spark.axis('off')

            _label_y = strip_top + 0.07
            ax_main.text(photo_left, _label_y, 'FB VELO BY OUTING', fontsize=8.5,
                         color=TEXT_SECONDARY, fontweight='bold',
                         fontfamily='IBM Plex Sans', va='bottom')
            ax_main.text(photo_left + strip_w_in, _label_y,
                         f'{_svelos[-1]:.1f} last  ·  {_savg:.1f} avg  ·  {max(_svelos):.1f} max',
                         fontsize=8.5, color=TEXT_MUTED, fontweight='bold',
                         fontfamily='IBM Plex Sans', va='bottom', ha='right')

            def _fmt_spark_date(d):
                try:
                    return datetime.strptime(d, '%Y-%m-%d').strftime('%b %-d')
                except Exception:
                    return str(d)

            _date_y = strip_top - strip_h_in - 0.12
            ax_main.text(photo_left, _date_y, _fmt_spark_date(_sdates[0]), fontsize=7.5,
                         color=TEXT_FAINT, fontfamily='IBM Plex Sans', va='top', ha='left')
            ax_main.text(photo_left + strip_w_in / 2, _date_y,
                         _fmt_spark_date(_sdates[len(_sdates) // 2]), fontsize=7.5,
                         color=TEXT_FAINT, fontfamily='IBM Plex Sans', va='top', ha='center')
            ax_main.text(photo_left + strip_w_in, _date_y, _fmt_spark_date(_sdates[-1]),
                         fontsize=7.5, color=TEXT_FAINT, fontfamily='IBM Plex Sans',
                         va='top', ha='right')

    # Movement plot — right-upper, near-square (movement is read to-scale). Season
    # centers over the location block beneath it; single-game uses the old wider
    # frame that fills to the right edge.
    if config.get('mvn_models'):
        # Season: classic-frame geometry (fractions of the 17.5in card, fixed
        # in inches) anchored to the TOP of the taller card so the plot keeps
        # its exact size/position relative to the header; the sparkline's
        # extra height falls into the gap below the legend.
        # Enlarged 2026-07-28 (per Wally): the plot claims the dead gutter to
        # its left plus the slack above/below — 6.48x6.21in -> 7.75x7.43in
        # (+43% area), near-square kept so movement still reads to-scale.
        # The pitch legend moves ABOVE the plot (below the title): with the
        # taller box there is no longer room for xlabel + legend beneath it
        # before the VS RHH/LHH titles at 0.480*FIG_H.
        _mv_h_in = 7.43
        _mv_y0_in = 9.37
        ax_plot = fig.add_axes([0.501, _mv_y0_in / fig_h, 0.484, _mv_h_in / fig_h])
        _mv_cx, _mv_ty = 0.743, 0.956
    else:
        ax_plot = fig.add_axes([0.585, 0.575, 0.405, 0.385]); _mv_cx, _mv_ty = 0.7875, 0.975
    ax_plot.set_xlim(-25,25); ax_plot.set_ylim(-25,25)
    # Title — parity with the hitter card's titled hero viz.
    fig.text(_mv_cx, _mv_ty, 'PITCH MOVEMENT', ha='center', va='center',
             fontsize=15, fontweight='bold', color=TEXT_SECONDARY,
             fontfamily='IBM Plex Sans')
    ax_plot.axhline(y=0, color=GRID_COLOR, linestyle='--', linewidth=0.6)
    ax_plot.axvline(x=0, color=GRID_COLOR, linestyle='--', linewidth=0.6)
    _mv_big = bool(config.get('mvn_models'))
    ax_plot.set_xlabel('Horizontal Break (in)', fontsize=12 if _mv_big else 10, color=TEXT_MUTED, fontweight='bold', fontfamily='IBM Plex Sans')
    ax_plot.set_ylabel('Induced Vertical Break (in)', fontsize=12 if _mv_big else 10, color=TEXT_MUTED, fontweight='bold', fontfamily='IBM Plex Sans')
    ax_plot.tick_params(labelsize=9.5 if _mv_big else 8, colors=TEXT_MUTED)
    ax_plot.set_xticks(range(-25,26,5)); ax_plot.set_yticks(range(-25,26,5))
    ax_plot.grid(True, alpha=0.5, color=GRID_COLOR); ax_plot.set_facecolor(PLOT_PANEL)
    for spine in ax_plot.spines.values(): spine.set_color(TEXT_FAINT)

    # Compute expected movement per pitch type from MVN model
    mvn_models = config.get('mvn_models', {})
    throws = config['hand']
    exp_movement = {}  # pt -> {'sum_ivb': ..., 'sum_hb': ..., 'n': ...}
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if not pt:
            continue
        aa = sf(p.get('ArmAngle'))
        ext = sf(p.get('Extension'))
        velo = sf(p.get('Velocity'))
        rz = sf(p.get('RelPosZ'))
        rx = sf(p.get('RelPosX'))
        xivb, xhb = compute_expected_movement(mvn_models, pt, throws, aa, ext, velo, rz, rx)
        if xivb is not None and xhb is not None:
            if pt not in exp_movement:
                exp_movement[pt] = {'sum_ivb': 0, 'sum_hb': 0, 'n': 0}
            exp_movement[pt]['sum_ivb'] += xivb
            exp_movement[pt]['sum_hb'] += xhb
            exp_movement[pt]['n'] += 1

    # Drop the shaded expected-movement ellipses + caption on all cards.
    exp_movement = {}

    # Draw expected movement ellipses first (behind scatter points, min 6 pitches)
    for pt in PITCH_ORDER:
        if pt not in exp_movement or pt not in groups:
            continue
        if len(groups[pt]) < 6:
            continue
        em = exp_movement[pt]
        cx = em['sum_hb'] / em['n']
        cy = em['sum_ivb'] / em['n']
        color = PITCH_COLORS.get(pt, '#999')
        ax_plot.add_patch(Ellipse((cx, cy), 7.0, 7.0,
            fill=True, facecolor=color, edgecolor=color,
            linewidth=1.2, alpha=0.24, zorder=1,
            hatch='///'))

    for pt in PITCH_ORDER:
        if pt not in groups: continue
        xs, ys = zip(*groups[pt]); color = PITCH_COLORS[pt]
        ax_plot.scatter(xs, ys, c=color, s=65, alpha=1.0, edgecolors=PLOT_PANEL, linewidths=0.5, zorder=3)
        if len(groups[pt]) >= 6:
            cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
            if vals[0] <= 0 or vals[1] <= 0:
                continue
            ax_plot.add_patch(Ellipse((np.mean(xs), np.mean(ys)), 2*1.5*np.sqrt(vals[1]), 2*1.5*np.sqrt(vals[0]),
                angle=np.degrees(np.arctan2(vecs[1,1], vecs[0,1])), fill=False, edgecolor=color, linewidth=1.2, linestyle='--', alpha=0.7))

    # DAILY ONLY — open marker at his SEASON centroid for the same pitch type,
    # joined to today's centroid by a hairline. Same idea as the delta columns:
    # on one start the readable quantity is drift, not level.
    _season_lb = config.get('season_pitch_lb') or {}
    _ghosted = False
    if not config.get('mvn_models') and _season_lb:
        for pt in PITCH_ORDER:
            if pt not in groups:
                continue
            _sb = _season_lb.get(pt) or {}
            if (_sb.get('count') or 0) < SEASON_DELTA_MIN:
                continue
            sx, sy = _sb.get('horzBrk'), _sb.get('indVertBrk')
            if sx is None or sy is None:
                continue
            _xs, _ys = zip(*groups[pt])
            _cx, _cy = float(np.mean(_xs)), float(np.mean(_ys))
            _c = PITCH_COLORS[pt]
            ax_plot.plot([sx, _cx], [sy, _cy], color=_c, linewidth=1.1,
                         linestyle=':', alpha=0.85, zorder=2)
            ax_plot.scatter([sx], [sy], s=110, facecolors='none', edgecolors=_c,
                            linewidths=1.8, zorder=4)
            _ghosted = True

    legend_handles = [mpatches.Patch(color=PITCH_COLORS[pt], label=f'{pt} - {PITCH_NAMES[pt]}') for pt in sorted_types]
    if config.get('mvn_models'):
        # Season: legend rides just above the plot, under the title.
        leg = ax_plot.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.004), ncol=min(len(sorted_types),5), fontsize=9, frameon=False, handlelength=1.2, columnspacing=1.2)
    else:
        leg = ax_plot.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5,-0.09), ncol=min(len(sorted_types),5), fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.2)
    for t in leg.get_texts(): t.set_color(TEXT_SECONDARY)
    # Add movement plot annotations
    if exp_movement:
        ax_plot.text(0.02, 0.035, 'Shaded = expected movement', transform=ax_plot.transAxes,
                     fontsize=7, color='#000000', fontfamily='IBM Plex Sans', va='bottom')
    # Notes render BLACK on every layout (2026-08-13, per Wally — the muted
    # gray read too faint at card scale).
    ax_plot.text(0.02, 0.005, 'Min. 6 pitches for ellipse', transform=ax_plot.transAxes,
                 fontsize=6.5, color='#000000',
                 fontfamily='IBM Plex Sans', va='bottom', fontstyle='italic')
    if _ghosted:
        ax_plot.text(0.02, 0.035, 'Open ring = his season average for that pitch',
                     transform=ax_plot.transAxes, fontsize=7, color='#000000',
                     fontfamily='IBM Plex Sans', va='bottom')

    # Location plots. Season: lower-right quadrant under the movement plot (left
    # column holds the bubbles). Single-game: left side (old layout), with the
    # donut/bars above and the usage bars on the right.
    is_season_loc = bool(config.get('mvn_models'))
    if is_season_loc:
        # Classic-frame inches: titles + top edge unchanged (top at 0.480 of
        # the 17.5in card), bottom edge pulled DOWN to align with the
        # percentile rail's bottom (0.235 * 17.5in). The panels are ~8.9%
        # taller than the classic 0.225 height; draw_zone shrinks the x-span
        # by the same factor so the zone/plate/ellipses enlarge uniformly.
        LOC_TITLE_Y = (0.498 * FIG_H) / fig_h
        LOC_BOTTOM  = (0.235 * FIG_H) / fig_h
        LOC_HEIGHT  = ((0.480 - 0.235) * FIG_H) / fig_h
        LOC_L_X=0.445; LOC_R_X=0.720; LOC_W=0.265
    else:
        LOC_TITLE_Y=0.555; LOC_BOTTOM=0.25; LOC_HEIGHT=0.29
        LOC_L_X=0.01; LOC_R_X=0.26; LOC_W=0.245

    # Per-hand pitch usage (for the small mix readout in each plot corner).
    hand_usage = {'L': defaultdict(int), 'R': defaultdict(int)}
    hand_tot = {'L': 0, 'R': 0}
    for p in pitches:
        bh = p.get('Bats', ''); pt = p.get('Pitch Type', '')
        if bh in ('L', 'R') and pt:
            hand_usage[bh][pt] += 1; hand_tot[bh] += 1
    # Single-game zone plots use a lower 6-pitch ellipse minimum (matches the
    # movement scatter rule). Season cards keep 10 to suppress ellipse noise
    # when many pitch types are crammed into the same plot.
    # Season: the >=10%-usage gate (below) decides WHICH pitches get
    # ellipses; the count floor is only the covariance-fit minimum, so a
    # >=10% pitch in a thin platoon split still draws (was 10, which
    # silently hid qualifying pitches vs the rarer hand).
    zone_ellipse_min = 6

    # Fixed zone bounds — same size for every pitcher, every card.
    # Season cards TRANSLATE the window so the plate/zone sits middle-right
    # (the lateral shift opens a gutter on the left where the pitch-mix legend
    # lives with clear separation from the ellipses, while preserving
    # glove-side coverage on the right) and SHRINK the x-span by the panels'
    # height-growth factor (4.2875/3.9375 vs the classic frame) so the taller
    # panel enlarges zone/plate/ellipses uniformly — no distortion. The plate
    # center stays at the same horizontal fraction as the classic (-2.3, 1.5)
    # window.
    def draw_zone(ax, hand):
        ax.set_facecolor(PLOT_PANEL)
        if is_season_loc:
            ax.set_xlim(-2.112, 1.378); ax.set_ylim(0.5, 4.2)
        else:
            ax.set_xlim(-1.9, 1.9); ax.set_ylim(0.5, 4.2)
        ax.add_patch(Rectangle((-PLATE_HALF, avg_bot), PLATE_HALF*2, avg_top-avg_bot, fill=False, edgecolor=TEXT_SECONDARY, linewidth=1.5, zorder=2))
        tw = PLATE_HALF*2/3; th = (avg_top-avg_bot)/3
        for i in range(1,3):
            ax.plot([-PLATE_HALF+i*tw, -PLATE_HALF+i*tw], [avg_bot, avg_top], color=GRID_COLOR, linewidth=0.6, zorder=2)
            ax.plot([-PLATE_HALF, PLATE_HALF], [avg_bot+i*th, avg_bot+i*th], color=GRID_COLOR, linewidth=0.6, zorder=2)
        pt_y = avg_bot - 0.15
        ax.plot([-PLATE_HALF,-PLATE_HALF,0,PLATE_HALF,PLATE_HALF,-PLATE_HALF], [pt_y,pt_y-0.10,pt_y-0.20,pt_y-0.10,pt_y,pt_y], color=TEXT_SECONDARY, linewidth=1.2, zorder=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values(): spine.set_color(TEXT_FAINT)

        # Per-hand pitch-mix legend — opaque paper panel in the reserved LEFT
        # GUTTER (the zone sits middle-right), drawn ON TOP of any ellipse
        # spill (zorder 8+) so it is always fully legible. Each row is a solid
        # pitch-color chip + the usage % in dark bold text.
        # Season cards show the pitch-mix legend in-plot; single-game cards have
        # dedicated usage bars right beside the locations, so skip it there.
        _u = hand_usage[hand]; _tot = hand_tot[hand]
        if is_season_loc and _tot > 0:
            _mix = sorted(_u.items(), key=lambda kv: -kv[1])
            _row_h = 0.072
            _panel_w = 0.215
            _panel_h = len(_mix) * _row_h + 0.035
            _px0, _py1 = 0.022, 0.978
            ax.add_patch(FancyBboxPatch(
                (_px0, _py1 - _panel_h), _panel_w, _panel_h,
                boxstyle='round,pad=0.008,rounding_size=0.012',
                transform=ax.transAxes, facecolor=BG,
                edgecolor=SUBTLE_BORDER, linewidth=1.0, zorder=8))
            _cy = _py1 - 0.033
            for _pt, _cnt in _mix:
                _col = PITCH_COLORS.get(_pt, TEXT_SECONDARY)
                ax.add_patch(Rectangle((_px0 + 0.014, _cy - _row_h * 0.34), 0.095, _row_h * 0.68,
                                       transform=ax.transAxes, facecolor=_col,
                                       edgecolor='none', zorder=9))
                ax.text(_px0 + 0.0615, _cy, _pt, transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color=badge_text_color(_col), zorder=10, fontfamily='IBM Plex Sans')
                # A pitch that is in this legend was thrown at least once vs this
                # hand, so a sub-1% share reads "< 1%" rather than a misleading
                # "0%" (matches the metrics table + usage-list convention).
                ax.text(_px0 + 0.135, _cy,
                        ("< 1%" if 0 < _cnt / _tot * 100 < 1 else f'{_cnt / _tot * 100:.0f}%'),
                        transform=ax.transAxes, ha='left', va='center',
                        fontsize=9.5, fontweight='bold', color=TEXT_PRIMARY,
                        zorder=10, fontfamily='IBM Plex Sans')
                _cy -= _row_h

        is_season = bool(config.get('mvn_models'))
        # Location ellipses (1.0σ covariance). Season cards: outline-only, for
        # every pitch type thrown >= 10% of the time vs this handedness (plus
        # the pitch-count minimum below). Single-game keeps the filled look.
        _ellipse_types = {pt for pt, cnt in _u.items()
                          if _tot > 0 and cnt / _tot >= 0.10}
        for pt in PITCH_ORDER:
            if pt not in locations[hand]: continue
            if is_season and pt not in _ellipse_types: continue
            pts = locations[hand][pt]
            if len(pts) >= zone_ellipse_min:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                cov = np.cov(xs, ys)
                vals, vecs = np.linalg.eigh(cov)
                if vals[0] > 0 and vals[1] > 0:
                    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                    mx, my = np.mean(xs), np.mean(ys)
                    _pc = PITCH_COLORS[pt]
                    # Outline-only ellipse + center dot at the per-type mean —
                    # one format on both layouts. The center dot is the ONLY
                    # dot: per-pitch dots and W/B marks were removed
                    # 2026-08-13, per Wally.
                    ax.add_patch(Ellipse(
                        (mx, my),
                        2 * 1.0 * np.sqrt(vals[1]), 2 * 1.0 * np.sqrt(vals[0]),
                        angle=angle, fill=False,
                        edgecolor=_rgba(_pc, 0.95),
                        linewidth=2.2, zorder=1
                    ))
                    ax.scatter([mx], [my], c=_pc, s=32, alpha=1.0,
                               edgecolors=TEXT_PRIMARY, linewidths=0.6, zorder=4)

    # loc_hands restricts which zone panels are drawn. A platoon-split card
    # holds pitches to one side only, so the other panel would render as an
    # empty strike zone; a single hand draws one panel, centered across both
    # slots. Default keeps the two-panel layout every existing caller expects.
    _loc_hands = tuple(config.get('loc_hands') or ('R', 'L'))
    _hand_title = {'R': 'VS RHH', 'L': 'VS LHH'}

    def _zone_title(h):
        """'VS RHH (Loc+ 82)' when the per-hand grade is available.

        Daily cards carry the split in config['pitch_locplus'] (_vsR/_vsL from
        the scratch context). Season cards get theirs computed here from the
        window's own per-pitch integer Loc+ atoms filtered by batter side —
        same coherent-canon atoms as every other displayed grade (added
        2026-08-13, per Wally, to match the daily card). Falls back to the
        bare title when no atoms exist (e.g. grade columns not yet stamped).
        """
        v = (config.get('pitch_locplus') or {}).get(f'_vs{h}')
        if v is None:
            _lh = [l for l in (sf(p.get('Loc+'))
                               for p in pitches if p.get('Bats') == h)
                   if l is not None]
            v = (sum(_lh) / len(_lh)) if _lh else None
        return _hand_title[h] if v is None else f'{_hand_title[h]} (Loc+ {int(round(v))})'

    if len(_loc_hands) == 1:
        _h = _loc_hands[0]
        _x = (LOC_L_X + LOC_R_X + LOC_W) / 2.0 - LOC_W / 2.0
        draw_zone(fig.add_axes([_x, LOC_BOTTOM, LOC_W, LOC_HEIGHT]), _h)
        fig.text(_x + LOC_W / 2, LOC_TITLE_Y, _zone_title(_h), fontsize=14,
                 fontweight='bold', color=TEXT_SECONDARY,
                 fontfamily='IBM Plex Sans', ha='center', va='center')
    else:
        ax_loc_l = fig.add_axes([LOC_L_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
        ax_loc_r = fig.add_axes([LOC_R_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
        draw_zone(ax_loc_l, 'R'); draw_zone(ax_loc_r, 'L')
        fig.text(LOC_L_X+LOC_W/2, LOC_TITLE_Y, _zone_title('R'), fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='IBM Plex Sans', ha='center', va='center')
        fig.text(LOC_R_X+LOC_W/2, LOC_TITLE_Y, _zone_title('L'), fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='IBM Plex Sans', ha='center', va='center')

    # Footnote — single-game only: ellipse minimum under the bottom-left of
    # the first (VS RHH) plot (moved from right of the plots 2026-08-13, per
    # Wally; the W/B legend left with the per-pitch marks the same day).
    # Season panels carry no footnote (the 10-pitch ellipse minimum still
    # applies, just unlabeled).
    if not is_season_loc:
        _fx = _x if len(_loc_hands) == 1 else LOC_L_X
        fig.text(_fx, LOC_BOTTOM - 0.006, f'Min. {zone_ellipse_min} pitches for ellipse',
                 fontsize=8, color='#000000', va='top', ha='left',
                 fontfamily='IBM Plex Sans', fontweight='bold')

    # ── Left column: season cards get the percentile bubble panel; single-game
    # cards (no season pool) get the batted-ball donut + stacked bars + usage. ──
    p_row = config.get('pctl_row') or {}

    def _batted_ball_line(pp):
        """Fly-ball / FB-HR / popup / BIP tallies from pitch dicts. HR/FB is
        the STRICT CONDITIONAL (2026-08-14, per Wally): fly-ball home runs
        over outfield flies — numerator a subset of the denominator, so the
        bubble reads "X% of his fly balls left the yard". Popups excluded
        (structurally can't leave), line-drive HRs excluded (barrel-quality
        events, already priced by Barrel%/xwOBAcon); bunts excluded."""
        fb = hr = pu = bip = 0
        for p in pp:
            if p.get('Description') != 'In Play':
                continue
            bb = str(p.get('BBType') or '')
            if not bb or bb.startswith('bunt'):
                continue
            bip += 1
            if bb == 'fly_ball':
                fb += 1
                if p.get('Event') == 'Home Run':
                    hr += 1
            elif bb == 'popup':
                pu += 1
        return {'fb': fb, 'hr': hr, 'pu': pu, 'bip': bip} if bip else None
    if config.get('mvn_models'):
        if p_row:
            # Extension comes off the leaderboard rounded to 0.1 ft, which is
            # 1.2 inches — too coarse once it renders as feet and inches, and
            # it disagreed with the pitch table's Ext column (7'1" vs 7'2" for
            # the same arm). Recompute from this card's pitches, the same
            # source the table uses, so the two always agree. The percentile
            # still comes from the leaderboard.
            _ext_vals = [v for v in (sf(p.get('Extension')) for p in pitches)
                         if v is not None]
            if _ext_vals:
                p_row = dict(p_row)
                p_row['extension'] = sum(_ext_vals) / len(_ext_vals)
            # HR/FB% + PU% (2026-08-14): computed from THIS card's pitches
            # (same source as the table) against the all-MLB pickle pools.
            # Leaderboard plumbing can replace this once the stats graduate;
            # display floors keep micro-denominator garbage off the rail.
            _bb = _batted_ball_line(pitches)
            if _bb:
                p_row = dict(p_row)
                _pools = _hrfb_pu_pools()
                if _bb['fb'] >= 10 and _pools.get('hrfb'):
                    p_row['hrFbPct'] = _bb['hr'] / _bb['fb'] * 100
                    # lower = better: percentile is share of pool ABOVE us
                    p_row['hrFbPct_pctl'] = _pctl_of(
                        p_row['hrFbPct'], _pools['hrfb'], invert=True)
                if _bb['bip'] >= 25 and _pools.get('pu'):
                    p_row['puPct'] = _bb['pu'] / _bb['bip'] * 100
                    p_row['puPct_pctl'] = _pctl_of(
                        p_row['puPct'], _pools['pu'], invert=False)
            bubble_cols = _bubble_columns_for(config, p_row)
            # Classic-frame inches (0.790/0.235 of the 17.5in card) so the
            # rail's physical geometry is untouched by the taller figure; the
            # sparkline lives entirely in the extra height above the rail.
            _render_percentile_bubbles(fig, p_row,
                                       # grid_right 0.405 -> 0.425 (2026-08-12).
                                       # The binding neighbour is NOT the
                                       # movement plot at 0.501 but the VS RHH
                                       # zone panel below it (LOC_L_X = 0.445),
                                       # so this keeps a 0.020 gutter.
                                       grid_left=0.015, grid_right=0.425,
                                       grid_top=(0.790 * FIG_H) / fig_h,
                                       grid_bot=(0.235 * FIG_H) / fig_h,
                                       columns=bubble_cols)
    else:
        _render_single_game_panel(fig, pitches, config)

    # Metrics table — full-width bottom band. Season: classic-frame inches so
    # the band is physically identical on the taller card.
    if is_season_loc:
        ax_table = fig.add_axes([TABLE_LEFT_FIG, (0.015 * FIG_H) / fig_h,
                                 TABLE_RIGHT_FIG - TABLE_LEFT_FIG,
                                 (0.205 * FIG_H) / fig_h])
    else:
        ax_table = fig.add_axes([TABLE_LEFT_FIG, 0.015, TABLE_RIGHT_FIG-TABLE_LEFT_FIG, 0.205])
    ax_table.axis('off'); ax_table.set_facecolor(BG)

    tc = len(pitches)
    pitch_stats = []
    is_season = bool(config.get('mvn_models'))
    # Per-pitch-type Loc+ (location quality, 100 = pitch-type group avg). Comes
    # from the pitch-level leaderboard via config; the card can't recompute it
    # (needs the league zone-quality tables). Empty dict → column auto-drops.
    locplus_by_pt = config.get('pitch_locplus') or {}
    pitch_lb = config.get('pitch_lb') or {}
    is_milb = config.get('team') in MILB_TEAMS
    nvaa_by_pt = {pt: d.get('nVAA') for pt, d in pitch_lb.items()}
    nvaa_pctl_by_pt = {pt: d.get('nVAA_pctl') for pt, d in pitch_lb.items()}
    # nHAA is hand-signed — display as-is (no |value| flip, no leading '+').
    nhaa_by_pt = {pt: d.get('nHAA') for pt, d in pitch_lb.items()}
    xrv_by_pt = {pt: d.get('xRunValue') for pt, d in pitch_lb.items()}
    xrv100_by_pt = {pt: d.get('xRv100') for pt, d in pitch_lb.items()}
    rv100_by_pt = {pt: d.get('rv100') for pt, d in pitch_lb.items()}
    rv_by_pt = {pt: d.get('runValue') for pt, d in pitch_lb.items()}
    # xRVOE/100 — per-type outperformance vs the stuff+location expectation.
    # MLB-only (leaderboard-computed, 150-pitch floor); ROC rows carry None
    # so the column auto-drops via the keep-check.
    xrvoe100_by_pt = {pt: d.get('xrvoe100') for pt, d in pitch_lb.items()}
    # Leaderboard per-type xwOBAcon — fallback for ROC cards, whose sheet
    # pitches carry no per-pitch xwOBA (the Tier-2 fill is pipeline-only).
    xwc_by_pt = {pt: d.get('xwOBAcon') for pt, d in pitch_lb.items()}
    stuff_by_pt = {pt: d.get('stuffScore') for pt, d in pitch_lb.items()}
    pitching_by_pt = {pt: d.get('pitchingScore') for pt, d in pitch_lb.items()}
    # RV columns: season cards default to the actual + expected per-100 pair
    # (PitchRV/100 + xPitchRV/100); --rv-mode totals swaps in the cumulative
    # pair (PitchRV + xPitchRV), --rv-mode both shows all four. PitchRV is
    # the real RunExp-based value for MLB and the contact-wOBA proxy for ROC.
    # Single-game keeps the cumulative xPitchRV.
    if is_season:
        rv_cols = {'per100': ['PitchRV/100', 'xPitchRV/100'],
                   'totals': ['PitchRV', 'xPitchRV'],
                   'both':   ['PitchRV', 'xPitchRV', 'PitchRV/100', 'xPitchRV/100'],
                   }[config.get('rv_mode') or 'per100']
    else:
        rv_cols = ['xPitchRV']
    _pt_qual_min = config.get('pitch_qual') or CARD_COLOR_MIN_PITCHES
    rv_qual_by_pt = {}   # pitch-type RV coloring gate (values always render)
    bip_by_pt = {}       # pitch-type BIP counts for the BIP coloring gates
    self_z_by_pt = {}    # DAILY: per-type z vs HIS OWN season, drives shading
    self_z_total = {}    # DAILY: same, for the Total row (release block only)

    # Sort pitch types by usage (descending), with PITCH_ORDER as tiebreaker
    pitch_counts = {}
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if pt:
            pitch_counts[pt] = pitch_counts.get(pt, 0) + 1
    table_pitch_order = sorted(
        pitch_counts.keys(),
        key=lambda pt: (-pitch_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999)
    )

    for pt in table_pitch_order:
        pp = [p for p in pitches if p.get('Pitch Type') == pt]
        if not pp: continue
        n = len(pp)
        velos=[v for v in (sf(p.get('Velocity')) for p in pp) if v]
        spins=[v for v in (sf(p.get('Spin Rate')) for p in pp) if v]
        ivbs=[v for v in (sf(p.get('xIndVrtBrk')) for p in pp) if v is not None]
        hbs=[v for v in (sf(p.get('xHorzBrk')) for p in pp) if v is not None]
        relzs=[v for v in (sf(p.get('RelPosZ')) for p in pp) if v is not None]
        relxs=[v for v in (sf(p.get('RelPosX')) for p in pp) if v is not None]
        exts=[v for v in (sf(p.get('Extension')) for p in pp) if v is not None]
        armangles=[v for v in (sf(p.get('ArmAngle')) for p in pp) if v is not None]
        swings=[p for p in pp if p.get('Description') in SWING_DESC]
        whiffs=[p for p in pp if p.get('Description')=='Swinging Strike']
        iz_n=0
        for p in pp:
            r = compute_iz(p)
            if r is None: continue
            if r: iz_n+=1
        # Run value. Season cards show the per-100 pair: xPitchRV/100 (expected,
        # xwOBA-based) from the leaderboard for all; PitchRV/100 (actual) is the real
        # RunExp-based rv100 for MLB and the contact-wOBA proxy for ROC (no RunExp).
        xrv_100 = xrv100_by_pt.get(pt)
        if is_milb:
            _prv = _compute_pitch_rv(pp)
            prv_100 = (sum(_prv) / len(pp) * 100) if _prv else None
        else:
            prv_100 = rv100_by_pt.get(pt)
        xrv_100 = (round(xrv_100, 1) + 0.0) if xrv_100 is not None else None
        prv_100 = (round(prv_100, 1) + 0.0) if prv_100 is not None else None
        # Cumulative pair. Season cards: expected from the leaderboard's
        # stored xRunValue (full precision, rounded once here); actual from
        # the leaderboard runValue for MLB or the contact-proxy sum for ROC.
        # Single-game keeps the in-card cumulative xPitchRV.
        prv_cum = None
        if is_season:
            xrv_cum = xrv_by_pt.get(pt)
            xrv_cum = (round(xrv_cum, 1) + 0.0) if xrv_cum is not None else None
            if is_milb:
                _prv_c = _compute_pitch_rv(pp)
                prv_cum = (round(sum(_prv_c), 1) + 0.0) if _prv_c else None
            else:
                prv_cum = rv_by_pt.get(pt)
                prv_cum = (round(prv_cum, 1) + 0.0) if prv_cum is not None else None
        else:
            rvs_x = _compute_pitch_xrv(pp)
            xrv_cum = (round(sum(rvs_x), 1) + 0.0) if rvs_x else None   # +0.0 kills -0.0
        # Qualification rework (2026-07-30, Wally): RV values always render;
        # on season/date-range cards the pitch-type minimum now gates only the
        # percentile COLORING below (site convention — qualification is a
        # render-only coloring gate). Daily cards were never gated here.
        rv_qual_by_pt[pt] = n >= _pt_qual_min
        _rvmap = {'PitchRV': prv_cum, 'xPitchRV': xrv_cum,
                  'PitchRV/100': prv_100, 'xPitchRV/100': xrv_100,
                  'xRVOE/100': ((round(xrvoe100_by_pt[pt], 1) + 0.0)
                                if xrvoe100_by_pt.get(pt) is not None else None)}
        # Chase% — swings on out-of-zone pitches over OoZ pitches.
        oop_swings_n = sum(1 for p in pp if p.get('Description') in SWING_DESC and compute_iz(p) == False)
        oop_pitches_n = sum(1 for p in pp if compute_iz(p) == False)
        chase_pct = oop_swings_n / oop_pitches_n if oop_pitches_n else None
        # xwOBAcon — average xwOBA on BIPs only. ROC sheet pitches carry no
        # per-pitch xwOBA, so fall back to the leaderboard's per-type value
        # (pipeline-computed via the Tier-2 3D fill) so the column renders
        # on ROC cards too.
        bip_xw = [v for v in (sf(p.get('xwOBA')) for p in pp if p.get('Description') == 'In Play' and not str(p.get('BBType', '')).startswith('bunt')) if v is not None]
        xwobacon = sum(bip_xw) / len(bip_xw) if bip_xw else xwc_by_pt.get(pt)
        # GB% — ground balls over BIP (BBType-based, bunts excluded; the site's
        # definition). Value always renders; coloring gates at GB_COLOR_MIN_BIP.
        _bips_n = sum(1 for p in pp if p.get('BBType') and not str(p.get('BBType')).startswith('bunt'))
        _gb_n = sum(1 for p in pp if p.get('BBType') == 'ground_ball')
        gb_pct = _gb_n / _bips_n if _bips_n else None
        bip_by_pt[pt] = _bips_n
        pt_name='Fastball' if pt=='FF' else PITCH_NAMES.get(pt,pt)
        # DAILY ONLY — z-scores of this start against HIS OWN season baseline,
        # for the Usage..HB block. These drive cell SHADING, not text, so the
        # table gains no width and every cell keeps its plain value.
        # season_pitch_lb is populated for every card, hence the is_season guard.
        _sb = (config.get('season_pitch_lb') or {}).get(pt) or {}
        _sb_ok = (not is_season) and (_sb.get('count') or 0) >= SEASON_DELTA_MIN
        _zs = {}

        def _z_mean(vals, key, flip_by_sign=False, invert=False, _sb=_sb, _ok=_sb_ok):
            """(start mean - season mean) in SEs of that difference.

            SE combines both samples, sqrt(1/n_today + 1/n_season), though the
            season term is small at 95-386 pitches. flip_by_sign is for IVB/HB:
            'better' there is MORE break in the direction the pitch already
            moves, so a curveball dropping 2" more than usual must read red,
            not blue, which a raw higher-is-red rule would get backwards.
            """
            base = _sb.get(key)
            if not _ok or base is None or len(vals) < 2:
                return None
            sd = float(np.std(vals, ddof=1))
            n_s = _sb.get('count') or 0
            if not (sd > 0) or n_s < 2:
                return None
            se = sd * math.sqrt(1.0 / len(vals) + 1.0 / n_s)
            z = (sum(vals) / len(vals) - base) / se
            if flip_by_sign and base < 0:
                z = -z
            return -z if invert else z

        # Sinkers, changeups and splitters run the OTHER WAY on ride: extra IVB
        # works against a pitch whose job is to stay down or to separate from
        # the fastball, so it has to shade blue. Changeups and splitters invert
        # on spin as well; sinkers do not (see LOW_SPIN_PITCH_TYPES).
        # IVB uses a plain invert rather than the sign flip: on these pitches
        # less ride is better whether the baseline IVB is positive or negative,
        # whereas on a breaking ball 'better' really is more break in whatever
        # direction the pitch already moves.
        _low_ivb = pt in LOW_IVB_PITCH_TYPES
        # Avg Velo — day-to-day scale, not sampling SE (see DAILY_VELO_SD_DAY):
        # z = delta / sqrt(SD_day^2 + s^2/n), so a small start still needs a
        # bigger delta to color while a long one bottoms out at the measured
        # day-to-day floor instead of saturating on ordinary variation.
        _v_base = _sb.get('velocity')
        if _sb_ok and _v_base is not None and len(velos) >= 2:
            _sd_day = DAILY_VELO_SD_DAY.get(pt, DAILY_VELO_SD_DAY_DEFAULT)
            _s2 = float(np.std(velos, ddof=1)) ** 2
            _zs['Avg Velo'] = ((sum(velos) / len(velos) - _v_base)
                               / math.sqrt(_sd_day ** 2 + _s2 / len(velos)))
        _zs['Spin Rate'] = _z_mean(spins, 'spinRate',
                                   invert=(pt in LOW_SPIN_PITCH_TYPES))
        _zs['IVB'] = (_z_mean(ivbs, 'indVertBrk', invert=True) if _low_ivb
                      else _z_mean(ivbs, 'indVertBrk', flip_by_sign=True))
        _zs['HB'] = _z_mean(hbs, 'horzBrk', flip_by_sign=True)
        # Release block — no valence, so red simply reads higher (Wally).
        _zs['RelZ'] = _z_mean(relzs, 'relPosZ')
        _zs['RelX'] = _z_mean(relxs, 'relPosX')
        _zs['Ext'] = _z_mean(exts, 'extension')

        # Approach angles. Both are signed, and 'better' is about MAGNITUDE:
        # flatter (toward zero) on a four-seam/cutter, steeper (away from zero)
        # on everything else. Scale off the per-pitch VAA/HAA spread in this
        # start -- nVAA is VAA shifted by a per-type constant, so their
        # within-start SDs are the same.
        def _z_abs(cur, key, vals, better_closer, _sb=_sb, _ok=_sb_ok):
            base = _sb.get(key)
            if not _ok or cur is None or base is None or len(vals) < 2:
                return None
            sd = float(np.std(vals, ddof=1))
            n_s = _sb.get('count') or 0
            if not (sd > 0) or n_s < 2:
                return None
            se = sd * math.sqrt(1.0 / len(vals) + 1.0 / n_s)
            z = (abs(cur) - abs(base)) / se
            return -z if better_closer else z

        _flat = pt in FLAT_APPROACH_TYPES
        _vaas = [v for v in (sf(q.get('VAA')) for q in pp) if v is not None]
        _haas = [v for v in (sf(q.get('HAA')) for q in pp) if v is not None]
        _zs['nVAA'] = _z_abs(nvaa_by_pt.get(pt), 'nVAA', _vaas, _flat)
        _zs['nHAA'] = _z_abs(nhaa_by_pt.get(pt), 'nHAA', _haas, _flat)

        # Max Velo — sample-size corrected (see _blom_expected_max), and scaled
        # by the SD of a MAXIMUM, sd / (n * phi(b_n)), not sd/sqrt(n).
        _mx_base = _sb.get('maxVelo')
        if _sb_ok and _mx_base is not None and len(velos) >= 2:
            _sd_v = float(np.std(velos, ddof=1))
            _n_s = _sb.get('count') or 0
            _b_t, _b_s = _blom_expected_max(len(velos)), _blom_expected_max(_n_s)
            if _sd_v > 0 and _b_t is not None and _b_s is not None:
                def _mx_f(_b, _n):
                    _phi = math.exp(-0.5 * _b * _b) / math.sqrt(2 * math.pi)
                    return 1.0 / (_n * _phi) if _n * _phi > 0 else None
                _f_t, _f_s = _mx_f(_b_t, len(velos)), _mx_f(_b_s, _n_s)
                if _f_t and _f_s:
                    _se_mx = _sd_v * math.sqrt(_f_t ** 2 + _f_s ** 2)
                    if _se_mx > 0:
                        _zs['Max Velo'] = ((max(velos) - (_mx_base - _sd_v * (_b_s - _b_t)))
                                           / _se_mx)

        # Usage — against his season share, on the OBSERVED game-to-game
        # scale: var = u(1-u) * (c + 1/tc). The binomial term alone treats
        # pitch mix as iid coin flips, but mix is a game-plan choice and the
        # measured day variance is ~2-5x binomial (see DAILY_USAGE_C).
        _u_base = _sb.get('usagePct')
        if _sb_ok and _u_base is not None and tc:
            _u_c = DAILY_USAGE_C.get(pt, DAILY_USAGE_C_DEFAULT)
            _u_se = math.sqrt(max(_u_base * (1 - _u_base), 1e-9)
                              * (_u_c + 1.0 / tc))
            _zs['Usage'] = ((n / tc) - _u_base) / _u_se if _u_se > 0 else None

        self_z_by_pt[pt] = _zs

        _c_velo = sum(velos)/len(velos) if velos else None
        _c_spin = sum(spins)/len(spins) if spins else None
        _called_n = sum(1 for q in pp if q.get('Description') == 'Called Strike')
        _c_ivb = sum(ivbs)/len(ivbs) if ivbs else None
        _c_hb = sum(hbs)/len(hbs) if hbs else None
        _nvaa = nvaa_by_pt.get(pt)
        _nhaa = nhaa_by_pt.get(pt)
        row=[pt_name,str(n),("< 1%" if 0 < n/tc*100 < 1 else f"{n/tc*100:.1f}%"),
            f"{_c_velo:.1f}" if velos else '—',
            f"{max(velos):.1f}" if velos else '—',
            f"{int(_c_spin)}" if spins else '—',
            f'{_c_ivb:.1f}"' if ivbs else '—',
            f'{_c_hb:.1f}"' if hbs else '—',
            f"{_nvaa:.2f}" if _nvaa is not None else '—',
            f"{_nhaa:.2f}" if _nhaa is not None else '—',
            fmt_fi(sum(relzs)/len(relzs)) if relzs else '—',fmt_fi(sum(relxs)/len(relxs)) if relxs else '—',
            fmt_fi(sum(exts)/len(exts)) if exts else '—',
            f"{sum(armangles)/len(armangles):.1f}°" if armangles else '—',
            (f"{int(round(stuff_by_pt[pt]))}"
             if stuff_by_pt.get(pt) is not None else '—'),
            (f"{int(round(locplus_by_pt[pt]))}" if locplus_by_pt.get(pt) is not None else '—'),
            (f"{int(round(pitching_by_pt[pt]))}" if pitching_by_pt.get(pt) is not None else '—'),
            f"{iz_n/n*100:.1f}%" if n else '—',
            # CSW% — called strikes plus whiffs over PITCHES (not swings), so
            # unlike Whiff% it is defined on the full per-type sample.
            f"{(_called_n + len(whiffs))/n*100:.1f}%" if n else '—',
            f"{len(whiffs)/len(swings)*100:.1f}%" if swings else '—',
            f"{chase_pct*100:.1f}%" if chase_pct is not None else '—',
            f"{xwobacon:.3f}".replace('0.', '.') if xwobacon is not None else '—',
            f"{gb_pct*100:.1f}%" if gb_pct is not None else '—']
        for _h in rv_cols:
            _v = _rvmap.get(_h)
            row.append(str(_v) if _v is not None else '—')
        pitch_stats.append((pt, row))

    t_sw=[p for p in pitches if p.get('Description') in SWING_DESC]
    t_wh=[p for p in pitches if p.get('Description')=='Swinging Strike']
    t_iz=sum(1 for p in pitches if compute_iz(p)==True)
    t_called=sum(1 for p in pitches if p.get('Description')=='Called Strike')
    # Expected run value for the Total row — cumulative + per-100.
    t_rvs_x = _compute_pitch_xrv(pitches)
    # Overall averages for RelZ, RelX, Ext
    t_relzs=[v for v in (sf(p.get('RelPosZ')) for p in pitches) if v is not None]
    t_relxs=[v for v in (sf(p.get('RelPosX')) for p in pitches) if v is not None]
    t_exts=[v for v in (sf(p.get('Extension')) for p in pitches) if v is not None]
    t_armangles=[v for v in (sf(p.get('ArmAngle')) for p in pitches) if v is not None]
    # Chase% total
    t_oop_sw = sum(1 for p in pitches if p.get('Description') in SWING_DESC and compute_iz(p) == False)
    t_oop_n = sum(1 for p in pitches if compute_iz(p) == False)
    t_chase = t_oop_sw / t_oop_n if t_oop_n else None
    # xwOBAcon total — average xwOBA on BIPs.
    t_bip_xw = [v for v in (sf(p.get('xwOBA')) for p in pitches if p.get('Description') == 'In Play' and not str(p.get('BBType', '')).startswith('bunt')) if v is not None]
    t_xwobacon = (sum(t_bip_xw) / len(t_bip_xw) if t_bip_xw
                  else (config.get('pctl_row') or {}).get('xwOBAcon'))
    # GB% total — same BBType-based definition as the per-type rows.
    t_bips_n = sum(1 for p in pitches if p.get('BBType') and not str(p.get('BBType')).startswith('bunt'))
    t_gb_n = sum(1 for p in pitches if p.get('BBType') == 'ground_ball')
    t_gb_pct = t_gb_n / t_bips_n if t_bips_n else None
    # DAILY — release baselines for the TOTAL row. Averaging velocity or break
    # across pitch types is meaningless (those Total cells render '—'), but
    # release point and extension are one delivery, so the Total row carries a
    # real number and deserves the same self-comparison. Baseline = his season
    # per-type values weighted by season pitch counts, which reconstructs his
    # season overall mean from data already in hand.
    _slb = config.get('season_pitch_lb') or {}
    if not is_season and _slb:
        def _tot_z(vals, key, _slb=_slb):
            num = den = 0.0
            for _d in _slb.values():
                _v, _c = (_d or {}).get(key), (_d or {}).get('count') or 0
                if _v is not None and _c > 0:
                    num += _v * _c
                    den += _c
            if den <= 0 or len(vals) < 2:
                return None
            sd = float(np.std(vals, ddof=1))
            if not (sd > 0):
                return None
            se = sd * math.sqrt(1.0 / len(vals) + 1.0 / den)
            return (sum(vals) / len(vals) - num / den) / se
        self_z_total['RelZ'] = _tot_z(t_relzs, 'relPosZ')
        self_z_total['RelX'] = _tot_z(t_relxs, 'relPosX')
        self_z_total['Ext'] = _tot_z(t_exts, 'extension')

    # Pitcher-level Loc+ for the Total row (from the bubble's leaderboard row).
    _total_locplus = (config.get('pctl_row') or {}).get('locPlus')
    _total_stuff = (config.get('pctl_row') or {}).get('stuffScore')
    _total_pitching = (config.get('pctl_row') or {}).get('pitchingScore')
    # RV totals. xPitchRV/100 from the leaderboard (expected) for all. PitchRV/100:
    # the real rv100 (MLB) or contact-wOBA proxy (ROC). Single-game keeps cumulative.
    _pr = config.get('pctl_row') or {}
    total_xrv_100 = (round(_pr['xRv100'], 1) + 0.0) if _pr.get('xRv100') is not None else None
    if is_milb:
        _tprv = _compute_pitch_rv(pitches)
        total_prv_100 = (round(sum(_tprv) / tc * 100, 1) + 0.0) if (_tprv and tc) else None
    else:
        total_prv_100 = (round(_pr['rv100'], 1) + 0.0) if _pr.get('rv100') is not None else None
    total_prv_cum = None
    if is_season:
        total_xrv_cum = _pr.get('xRunValue')
        total_xrv_cum = (round(total_xrv_cum, 1) + 0.0) if total_xrv_cum is not None else None
        if is_milb:
            _tprv_c = _compute_pitch_rv(pitches)
            total_prv_cum = (round(sum(_tprv_c), 1) + 0.0) if _tprv_c else None
        else:
            total_prv_cum = _pr.get('runValue')
            total_prv_cum = (round(total_prv_cum, 1) + 0.0) if total_prv_cum is not None else None
    else:
        total_xrv_cum = (round(sum(t_rvs_x), 1) + 0.0) if t_rvs_x else None
    _trvmap = {'PitchRV': total_prv_cum, 'xPitchRV': total_xrv_cum,
               'PitchRV/100': total_prv_100, 'xPitchRV/100': total_xrv_100,
               'xRVOE/100': ((round(_pr['xrvoe100'], 1) + 0.0)
                             if _pr.get('xrvoe100') is not None else None)}
    total_row=['Total',str(tc),'100.0%','—','—','—','—','—','—','—',
        fmt_fi(sum(t_relzs)/len(t_relzs)) if t_relzs else '—',
        fmt_fi(sum(t_relxs)/len(t_relxs)) if t_relxs else '—',
        fmt_fi(sum(t_exts)/len(t_exts)) if t_exts else '—',
        f"{sum(t_armangles)/len(t_armangles):.1f}°" if t_armangles else '—',
        (f"{int(round(_total_stuff))}" if _total_stuff is not None else '—'),
        (f"{int(round(_total_locplus))}" if _total_locplus is not None else '—'),
        (f"{int(round(_total_pitching))}" if _total_pitching is not None else '—'),
        f"{t_iz/tc*100:.1f}%" if tc else '—',
        f"{(t_called + len(t_wh))/tc*100:.1f}%" if tc else '—',
        f"{len(t_wh)/len(t_sw)*100:.1f}%" if t_sw else '—',
        f"{t_chase*100:.1f}%" if t_chase is not None else '—',
        f"{t_xwobacon:.3f}".replace('0.', '.') if t_xwobacon is not None else '—',
        f"{t_gb_pct*100:.1f}%" if t_gb_pct is not None else '—']
    for _h in rv_cols:
        _v = _trvmap.get(_h)
        total_row.append(str(_v) if _v is not None else '—')

    # Source-data presence check — RV needs RunExp on at least one pitch.
    has_pitchrv_data = any(p.get('RunExp') is not None and str(p.get('RunExp','')).strip() != '' for p in pitches)

    all_col_headers=['Pitch Type','Count','Usage','Avg Velo','Max Velo','Spin Rate','IVB','HB','nVAA','nHAA','RelZ','RelX','Ext','Arm Angle','Stuff+','Loc+','Pitching+','Zone%','CSW%','Whiff%','Chase%','xwOBAcon','GB%'] + rv_cols
    all_cell_data=[r[1] for r in pitch_stats]+[total_row]

    # Daily cards use a different column ORDER than season (Wally's layout):
    # nVAA/nHAA sit after the release block (Ext/Arm Angle) rather than after
    # HB, and Stuff+/Loc+ form their own section AFTER Chase% rather than
    # leading the outcomes block. Reorder headers + cells together (a pure
    # permutation of the same column set); the downstream keep/color logic is
    # name-indexed so it follows. Season layout is unchanged.
    if not is_season:
        _daily_order = ['Pitch Type','Count','Usage','Avg Velo','Max Velo','Spin Rate',
                        'IVB','HB','RelZ','RelX','Ext','Arm Angle','nVAA','nHAA',
                        'Zone%','CSW%','Whiff%','Chase%','xwOBAcon','Stuff+','Loc+','Pitching+'] + rv_cols
        _perm = [all_col_headers.index(h) for h in _daily_order]
        all_col_headers = _daily_order
        all_cell_data = [[row[i] for i in _perm] for row in all_cell_data]

    # Columns to force-exclude based on data availability and card type.
    force_exclude = set()
    # Per-pitch Pitching+ dropped EVERYWHERE (season 2026-07-30, daily
    # 2026-08-12, both Wally): it is the deterministic 0.8/0.2 blend of the
    # adjacent Stuff+/Loc+ columns — zero new information.
    # GB% is season/date-range only (not in the daily column order).
    force_exclude.add('Pitching+')
    if is_season:
        force_exclude.add('CSW%')
    _have_xrv = any(v is not None for v in xrv100_by_pt.values()) if is_season else has_pitchrv_data
    if not _have_xrv:
        for _h in ('PitchRV', 'xPitchRV', 'PitchRV/100', 'xPitchRV/100'):
            force_exclude.add(_h)
    # If no xwOBA on any BIP AND no leaderboard fallback, xwOBAcon drops.
    has_xwoba_bip = (any(sf(p.get('xwOBA')) is not None and p.get('Description') == 'In Play' and not str(p.get('BBType', '')).startswith('bunt') for p in pitches)
                     or any(v is not None for v in xwc_by_pt.values()))
    if not has_xwoba_bip: force_exclude.add('xwOBAcon')
    # Conditional RelZ/RelX: always exclude on season cards. On single-game cards,
    # exclude only when Arm Angle data exists (Arm Angle conveys the same release
    # info more compactly); keep RelZ/RelX as a fallback when Arm Angle is missing.
    has_arm_angle = any(sf(p.get('ArmAngle')) is not None for p in pitches)
    # Drop RelZ/RelX once Arm Angle is present — it conveys the same release
    # information more compactly. ROC/AAA used to be special-cased here because
    # AAA had no arm angle at all; the minor-league Statcast backfill
    # (2026-07-25) ended that, so the rule is now purely data-driven. The
    # is_season shortcut stays MLB-only: a MiLB card for a pitcher whose games
    # Savant hasn't processed yet keeps RelZ/RelX rather than losing release
    # info entirely (the all-'—' Arm Angle column gets dropped below).
    if has_arm_angle or (is_season and not is_milb):
        force_exclude.add('RelZ')
        force_exclude.add('RelX')

    # Drop columns where ALL pitch-type rows have '—' OR source data is missing.
    # Derive from all_cell_data (NOT pitch_stats) so the keep-check stays aligned
    # to all_col_headers after the daily reorder above.
    pitch_rows_only = all_cell_data[:-1]  # all but the Total row
    cols_to_keep = []
    for ci in range(len(all_col_headers)):
        col_name = all_col_headers[ci]
        # Always keep Pitch Type, Count, Usage
        if ci < 3:
            cols_to_keep.append(ci)
        elif col_name in force_exclude:
            continue  # skip — source data not available
        else:
            # Keep if at least one pitch-type row has a real value (not '—')
            if any(row[ci] != '—' for row in pitch_rows_only):
                cols_to_keep.append(ci)

    col_headers = [all_col_headers[i] for i in cols_to_keep]
    cell_data = [[row[i] for i in cols_to_keep] for row in all_cell_data]
    pt_codes=[r[0] for r in pitch_stats]+[None]

    # Divider sits at the boundary between physical traits (left) and outcomes
    # (right). Stuff+ leads the outcomes block (Stuff+, Loc+, Zone%, ...).
    divider_col = col_headers.index('Stuff+') if 'Stuff+' in col_headers else (
        col_headers.index('Zone%') if 'Zone%' in col_headers else None)

    # Column autosizing happens after styling (bold text measures wider):
    # start from equal widths, then shrink each column to its widest rendered
    # cell (header or value) plus a fixed padding.
    n_cols = len(col_headers)
    table = ax_table.table(cellText=cell_data, colLabels=col_headers,
                            loc='upper center', cellLoc='center',
                            colWidths=[1.0 / n_cols] * n_cols)
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)

    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor(SUBTLE_BORDER); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(DARKER); cell.set_text_props(color=TEXT_SECONDARY, fontweight='bold', fontsize=10, fontfamily='IBM Plex Sans Condensed')
        elif r == len(cell_data):
            cell.set_facecolor(DARKER); cell.set_text_props(fontweight='bold', color=TEXT_PRIMARY, fontfamily='IBM Plex Sans')
        else:
            bg = DARK_CELL if r%2==1 else ALT_ROW_BG
            cell.set_facecolor(bg); cell.set_text_props(color=TEXT_PRIMARY, fontweight='bold', fontfamily='IBM Plex Sans')
        if c == 0 and r > 0:
            pc = pt_codes[r-1]
            if pc:
                cell.set_facecolor(PITCH_COLORS.get(pc,'#999'))
                cell.set_text_props(color=badge_text_color(PITCH_COLORS.get(pc,'#999')), fontweight='bold', fontfamily='IBM Plex Sans')

    # Gated cells render FADED (site parity: #8a7f75 = the unqualified-circle
    # text color) so "full ink, no tint" reads as league-average and "faded"
    # reads as sample-too-small. Values always render either way.
    def _fade_cell(r_, c_):
        table.get_celld()[(r_, c_)].set_text_props(color=TEXT_FAINT)

    # Percentile-based coloring for Zone%, Whiff%, Chase%
    league_avgs = config.get('league_avgs', {})
    overall_avgs = config.get('overall_avgs', {})
    for c, col_name in enumerate(col_headers):
        meta_key = PCT_COLOR_COLS.get(col_name)
        if not meta_key:
            continue
        for r in range(1, len(cell_data) + 1):
            if r == len(cell_data):
                # Total row — use overall league averages
                la = overall_avgs.get(meta_key)
                row_bg = DARKER
            else:
                pc = pt_codes[r - 1]
                if not pc or pc not in league_avgs:
                    continue
                # Flat outcome coloring gate (season/date-range cards).
                # Deliberately NOT applied to daily cards: at one start almost
                # nothing clears 50 pitches, so gating faded the whole table
                # and the card lost more than the false precision was worth.
                if is_season and pitch_counts.get(pc, 0) < CARD_COLOR_MIN_PITCHES:
                    _fade_cell(r, c)
                    continue
                la = league_avgs[pc].get(meta_key)
                row_bg = DARK_CELL if r % 2 == 1 else ALT_ROW_BG
            if la is None:
                continue
            val_str = cell_data[r - 1][c]
            tinted = _pitcher_stat_cell_color(val_str, la, 1.0, True, row_bg, True)
            if tinted:
                table.get_celld()[(r, c)].set_facecolor(tinted)

    # Raw-value coloring (Extension, etc.)
    for c, col_name in enumerate(col_headers):
        raw_cfg = RAW_COLOR_COLS.get(col_name)
        if not raw_cfg:
            continue
        # Ext tints on season/date-range cards only (2026-08-12, Wally). On a
        # daily card the Total-row extension is one start's mean, and tinting
        # it asserted a league judgement that sample cannot carry.
        if not is_season:
            continue
        meta_key, scale, higher_is_better = raw_cfg
        # Total row only
        r = len(cell_data)
        wsum, wn = 0.0, 0
        for pt_key, pt_data in league_avgs.items():
            v = pt_data.get(meta_key)
            n = pt_data.get('count', 0)
            if v is not None and n > 0:
                wsum += v * n; wn += n
        la = wsum / wn if wn > 0 else None
        if la is not None:
            val_str = cell_data[r - 1][c]
            tinted = _pitcher_stat_cell_color(val_str, la, scale, higher_is_better, DARKER, False)
            if tinted:
                table.get_celld()[(r, c)].set_facecolor(tinted)

    # xwOBAcon coloring — lower is better for pitcher; scale ±0.05 from league avg.
    if 'xwOBAcon' in col_headers:
        xwc_col_idx = col_headers.index('xwOBAcon')
        overall_xwc = overall_avgs.get('xwOBAcon')
        for r in range(1, len(cell_data) + 1):
            if r == len(cell_data):
                la = overall_xwc
                row_bg = DARKER
            else:
                pc = pt_codes[r - 1]
                la = league_avgs.get(pc, {}).get('xwOBAcon') if pc else None
                row_bg = DARK_CELL if r % 2 == 1 else ALT_ROW_BG
            if la is None:
                continue
            # BIP coloring gate (season/date-range): xwOBAcon tints at 25 BIP.
            if is_season and r < len(cell_data) and bip_by_pt.get(pc, 0) < XWC_COLOR_MIN_BIP:
                _fade_cell(r, xwc_col_idx)
                continue
            val_str = cell_data[r - 1][xwc_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, la, 0.05, False, row_bg, False)
            if tinted:
                table.get_celld()[(r, xwc_col_idx)].set_facecolor(tinted)

    # GB% coloring — higher is better for pitcher, vs the per-type league
    # average (same pct tint family as Zone%/Whiff%/Chase%). The value always
    # renders; tint gates at GB_COLOR_MIN_BIP (measured reliability crossing,
    # aligned with the site's MIN_BIP_PCTL). Total row colors vs overall avg.
    if 'GB%' in col_headers:
        gb_col_idx = col_headers.index('GB%')
        for r in range(1, len(cell_data) + 1):
            if r == len(cell_data):
                la = overall_avgs.get('gbPct')
                row_bg = DARKER
                qual = True
            else:
                pc = pt_codes[r - 1]
                la = league_avgs.get(pc, {}).get('gbPct') if pc else None
                row_bg = DARK_CELL if r % 2 == 1 else ALT_ROW_BG
                qual = bip_by_pt.get(pc, 0) >= GB_COLOR_MIN_BIP
            if not qual:
                _fade_cell(r, gb_col_idx)
                continue
            if la is None:
                continue
            val_str = cell_data[r - 1][gb_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, la, 1.0, True, row_bg, True)
            if tinted:
                table.get_celld()[(r, gb_col_idx)].set_facecolor(tinted)

    # xPitchRV coloring — higher is better for pitcher, centered at 0. The
    # per-100 rate gets scale 2.0; the cumulative column spans wider (a full
    # season of one pitch type), so it uses scale 3.0.
    RV_COL_NAMES = ('PitchRV', 'xPitchRV', 'PitchRV/100', 'xPitchRV/100')
    for c, col_name in enumerate(col_headers):
        if col_name not in RV_COL_NAMES:
            continue
        rv_scale = 2.0 if col_name.endswith('/100') else 3.0
        for r in range(1, len(cell_data) + 1):
            # Coloring gate (2026-07-30): below the pitch-type minimum the
            # value renders untinted. Season/date-range cards only; the Total
            # row always colors.
            if is_season and r < len(cell_data):
                _pc = pt_codes[r - 1]
                if _pc and not rv_qual_by_pt.get(_pc, True):
                    _fade_cell(r, c)
                    continue
            row_bg = DARKER if r == len(cell_data) else (DARK_CELL if r % 2 == 1 else ALT_ROW_BG)
            val_str = cell_data[r - 1][c]
            tinted = _pitcher_stat_cell_color(val_str, 0.0, rv_scale, True, row_bg, False)
            if tinted:
                table.get_celld()[(r, c)].set_facecolor(tinted)

    # DAILY ONLY — Usage and Avg Velo shade against HIS OWN season baseline
    # rather than the league (2026-08-12, Wally). Max Velo through nHAA ran
    # self-shaded too until 2026-08-13, when Wally dropped their coloring:
    # those columns now render untinted and the outcome columns to the right
    # keep league shading, so the note under the table stays two-part.
    # Total row is skipped, having no per-type baseline.
    SELF_BASELINE_COLS = ('Usage', 'Avg Velo')
    _self_shaded = False
    if not is_season and self_z_by_pt:
        for c, col_name in enumerate(col_headers):
            if col_name not in SELF_BASELINE_COLS:
                continue
            for r in range(1, len(cell_data) + 1):
                if r == len(cell_data):
                    # Total row: release block only — self_z_total carries no
                    # key for the others, so they fall through untinted.
                    _z, _bg = self_z_total.get(col_name), DARKER
                else:
                    _pc = pt_codes[r - 1]
                    _z = (self_z_by_pt.get(_pc) or {}).get(col_name)
                    _bg = DARK_CELL if r % 2 == 1 else ALT_ROW_BG
                if _z is None:
                    continue
                _tint = _z_cell_color(_z, _bg, full_at=DELTA_MIN_SE)
                if _tint:
                    table.get_celld()[(r, c)].set_facecolor(_tint)
                    _self_shaded = True

    # Loc+ coloring — index centered at 100 (group avg), higher is better,
    # scale 10 (≈1 SD). Matches the Loc+ bubble's blue→red direction.
    if 'Loc+' in col_headers:
        lp_col_idx = col_headers.index('Loc+')
        for r in range(1, len(cell_data) + 1):
            # Loc+ colors at its measured per-group gate (site parity).
            if is_season and r < len(cell_data):
                _pc = pt_codes[r - 1]
                if _pc and pitch_counts.get(_pc, 0) < _locplus_color_min(_pc):
                    _fade_cell(r, lp_col_idx)
                    continue
            row_bg = DARKER if r == len(cell_data) else (DARK_CELL if r % 2 == 1 else ALT_ROW_BG)
            val_str = cell_data[r - 1][lp_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, 100.0, 10.0, True, row_bg, False)
            if tinted:
                table.get_celld()[(r, lp_col_idx)].set_facecolor(tinted)

    # Stuff+ coloring — index centered at 100 (group avg), higher is better,
    # scale 10 (≈1 SD). Matches the Stuff+ bubble's blue→red direction.
    if 'Stuff+' in col_headers:
        sp_col_idx = col_headers.index('Stuff+')
        for r in range(1, len(cell_data) + 1):
            # Stuff+ is shape-fast (measured k=13): colors from 15 pitches.
            if is_season and r < len(cell_data):
                _pc = pt_codes[r - 1]
                if _pc and pitch_counts.get(_pc, 0) < STUFF_COLOR_MIN_PITCHES:
                    _fade_cell(r, sp_col_idx)
                    continue
            row_bg = DARKER if r == len(cell_data) else (DARK_CELL if r % 2 == 1 else ALT_ROW_BG)
            val_str = cell_data[r - 1][sp_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, 100.0, 10.0, True, row_bg, False)
            if tinted:
                table.get_celld()[(r, sp_col_idx)].set_facecolor(tinted)

    # Pitching+ coloring — index centered at 100, higher is better, scale 10
    # (≈1 SD). Same convention as the Stuff+/Loc+ columns it blends.
    if 'Pitching+' in col_headers:
        pp_col_idx = col_headers.index('Pitching+')
        for r in range(1, len(cell_data) + 1):
            row_bg = DARKER if r == len(cell_data) else (DARK_CELL if r % 2 == 1 else ALT_ROW_BG)
            val_str = cell_data[r - 1][pp_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, 100.0, 10.0, True, row_bg, False)
            if tinted:
                table.get_celld()[(r, pp_col_idx)].set_facecolor(tinted)

    # nVAA coloring vs the LEAGUE — FF and SI only (per spec). nVAA_pctl is
    # already directional (FF: flatter/closer-to-zero better; SI: steeper).
    # SEASON/DATE-RANGE ONLY: this pass runs after the self-baseline block
    # above, so on a daily card it silently overwrote FF and SI, leaving nVAA
    # league-relative for two pitch types and self-relative for the rest while
    # the note under the table claimed the whole block was self-relative.
    if is_season and 'nVAA' in col_headers:
        nvaa_col_idx = col_headers.index('nVAA')
        for r in range(1, len(cell_data)):   # pitch rows only; skip Total
            pc = pt_codes[r - 1]
            if pc not in ('FF', 'SI'):
                continue
            pctl = nvaa_pctl_by_pt.get(pc)
            if pctl is None:
                continue
            row_bg = DARK_CELL if r % 2 == 1 else ALT_ROW_BG
            tinted = _pctl_cell_color(pctl, row_bg)
            if tinted:
                table.get_celld()[(r, nvaa_col_idx)].set_facecolor(tinted)

    # Content-fit column widths: measure each column's widest rendered text
    # (header or any cell, styling already applied) and set the column width to
    # that plus a fixed padding. The table shrinks to content — loc='upper
    # center' keeps it centered in the band — instead of stretching short
    # columns (Count, IVB) to fill the full card width. If the fitted widths
    # would overflow the band (many pitch types + all 20 columns), scale all
    # columns down proportionally so the table still fits.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _ax_w_px = ax_table.get_window_extent(renderer).width
    COL_PAD_IN = 0.22   # fixed padding per column (total, inches)
    _pad_px = COL_PAD_IN * fig.dpi
    _col_px = [0.0] * n_cols
    for (_r, _c), _cell in table.get_celld().items():
        _txt = _cell.get_text()
        if not _txt.get_text():
            continue
        _col_px[_c] = max(_col_px[_c], _txt.get_window_extent(renderer).width)
    _fit_fracs = [(_w + _pad_px) / _ax_w_px for _w in _col_px]
    # Normalize to FILL the band: with fewer columns the table keeps its
    # full span (cells widen) instead of centering narrower.
    _shrink = 1.0 / sum(_fit_fracs)
    for (_r, _c), _cell in table.get_celld().items():
        _cell.set_width(_fit_fracs[_c] * _shrink)

    # Divider + borders
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer(); fig_bbox = fig.get_window_extent(renderer)
    if divider_col is not None:
        strike_cell = table.get_celld()[0, divider_col]
        x_fig = strike_cell.get_window_extent(renderer).x0 / fig_bbox.width
        top_y = strike_cell.get_window_extent(renderer).y1 / fig_bbox.height - 0.001
        bot_cell = table.get_celld()[len(cell_data), divider_col]
        bot_y = bot_cell.get_window_extent(renderer).y0 / fig_bbox.height
        fig.add_artist(plt.Line2D([x_fig,x_fig], [bot_y,top_y], transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))

    tl = table.get_celld()[0,0].get_window_extent(renderer)
    br = table.get_celld()[len(cell_data), len(col_headers)-1].get_window_extent(renderer)
    l = tl.x0/fig_bbox.width; r_ = br.x1/fig_bbox.width
    t = tl.y1/fig_bbox.height - 0.001; b = br.y0/fig_bbox.height
    for x1,y1,x2,y2 in [(l,b,r_,b),(l,t,r_,t),(l,b,l,t),(r_,b,r_,t)]:
        fig.add_artist(plt.Line2D([x1,x2],[y1,y2], transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))

    # Stuff+ footnote — season cards only, just below the table's bottom border.
    # Left edge aligned under the Stuff+ column (the outcomes-block divider);
    # two lines so it never runs past the card's right edge.
    # Below-table drop: classic-frame inches on season cards (0.008 * 17.5in).
    _below_off = (0.008 * FIG_H) / fig_h if is_season else 0.008
    if is_season and 'Stuff+' in col_headers:
        _sp_cell = table.get_celld()[(0, col_headers.index('Stuff+'))]
        _sp_x = _sp_cell.get_window_extent(renderer).x0 / fig_bbox.width
        _sp_note = ('PitchRV/100 actual · xPitchRV/100 expected (luck-neutral) runs saved per 100 pitches: the gap is fortune\n'
                    'Per-pitch Stuff+ and Loc+ graded vs same pitch type (100 = average for that type)\n'
                    'Overall Stuff+ = pitch-weighted average of per-pitch grades\n'
                    'Faded values: sample too small to grade; they color in as pitches accumulate')
        fig.text(_sp_x, b - _below_off, _sp_note,
                 fontsize=8, color='#000000', va='top', ha='left', fontfamily='IBM Plex Sans', fontweight='bold', linespacing=1.5)

    # Watermark — just below the table border. Bottom-LEFT on season cards
    # (unchanged); bottom-RIGHT on daily, where the left of that band is now
    # the shading note.
    _wm_x, _wm_ha = ((l, 'left') if is_season else (r_, 'right'))
    fig.text(_wm_x, b - _below_off, 'huronalytics.vercel.app', fontsize=9, ha=_wm_ha, va='top',
             color='#000000', style='italic', fontfamily='IBM Plex Sans')
    if _self_shaded:
        # Left edge aligned under the Usage column — the first column the note
        # describes — so the note visually claims the block it explains.
        _nx = l
        if 'Usage' in col_headers:
            _nx = (table.get_celld()[(0, col_headers.index('Usage'))]
                   .get_window_extent(renderer).x0 / fig_bbox.width)
        fig.text(_nx, b - _below_off,
                 'Usage and Avg Velo are shaded against HIS OWN season average for that pitch; '
                 'Zone% through Loc+ against LEAGUE average. Red = better, blue = worse.\n'
                 'Usage simply reads red = higher. '
                 'Full color on Usage/Avg Velo = 2 standard errors, so a faint cell there is a gap inside normal game-to-game noise.',
                 fontsize=8.5, ha='left', va='top', color='#000000',
                 fontfamily='IBM Plex Sans', linespacing=1.5)
    plt.savefig(output_file, dpi=SAVE_DPI, bbox_inches='tight', facecolor=BG, pad_inches=0.1)
    plt.close()

    # Crop bottom dead space from saved PNG
    card_img = Image.open(output_file)
    pixels = np.array(card_img)
    bg_rgb = (240, 232, 216)  # BG=#f0e8d8 (warm cream)
    # Scan from bottom up to find last non-background row
    for y in range(pixels.shape[0]-1, 0, -1):
        row = pixels[y, :, :3]
        if not np.all(np.abs(row.astype(int) - np.array(bg_rgb)) < 10):
            # Found content — add small padding below
            crop_y = min(y + 30, pixels.shape[0])
            card_img = card_img.crop((0, 0, card_img.width, crop_y))
            card_img.save(output_file)
            break
    return True


# ═══════════════════════════════════════════════════════════════
# SCRATCH-TAB MLB-STYLE CONTEXT (computed, not looked up)
#
# Scratch-tab pitchers (Pitcher2026 player_id pulls into a non-team tab of
# NLE2026) never enter the leaderboards, so a full MLB-style card can't look
# anything up. Instead we follow the ROC translation pattern: every derived
# quantity is COMPUTED from the scratch pitches against MLB baselines, then
# RANKED into the MLB leaderboard pools:
#   Stuff+  — stuff_plus bundle, v12 config (full model when the pitcher has ArmAngle
#             data, else the no-arm companion + its MLB anchor scales)
#   Loc+    — pipeline_locplus.compute_loc_plus with MLB pickle pitches as the
#             baseline/pool and the scratch pitchers keyed under 'AAA' (scored
#             against MLB surfaces, excluded from the (mu, sigma) pool)
#   RV/xRV  — pipeline_compute.compute_stats / compute_xrv with the same
#             count-anchoring offsets the leaderboard uses
#   nVAA/nHAA — metadata vaaRegressions / haaRegressions applied to the
#             scratch pitches' mean VAA/HAA + plate coords per pitch type
#   bubbles — pitcher-level stats ranked against pitcher_leaderboard_rs.json
#             (interpolation: fraction below + half ties, same as
#             compute_percentile_ranks_with_aaa's interp path)
# ═══════════════════════════════════════════════════════════════

# Pitcher-level stats whose percentile is inverted (lower = better for the
# pitcher). Mirrors PITCHER_ALL_INVERT in process_data for the bubble stats.
_SCRATCH_INVERT_PITCHER = {'bbPct', 'xwOBA', 'xwOBAcon', 'hardHitPct', 'barrelPctAgainst', 'babip'}
# Bubble-panel stats we compute and rank (everything BUBBLE_COLUMNS reads).
_SCRATCH_POOL_STATS = ['xRunValue', 'xRv100', 'xwOBA', 'kPct', 'bbPct', 'kbbPct',
                       'swStrPct', 'chasePct', 'izWhiffPct', 'twoStrikeWhiffPct',
                       'xwOBAcon', 'hardHitPct', 'barrelPctAgainst', 'gbPct', 'babip',
                       'fbVelo', 'extension', 'stuffScore', 'locPlus', 'pitchingScore',
                       'izPct', 'fpsPct']


def _pitching_blend(stuff, loc):
    """Stuff+/Loc+ blend in z units — the trainer's _blend (single source of
    truth for the 70/30 weight; falls back to 0.70 if the import fails)."""
    _sv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stuff_plus')
    if _sv_dir not in sys.path:
        sys.path.insert(0, _sv_dir)
    try:
        from train_stuff import _blend
        return _blend(stuff, loc)
    except Exception:
        return 0.80 * (stuff - 100.0) / 10.0 + 0.20 * (loc - 100.0) / 10.0


def _pitching_scale(rows, min_pitches=25):
    """League (mu, sd) of the blend from leaderboard pool rows with >=25
    pitches — mirrors train_stuff._inject_pitching's pool convention."""
    bs = [_pitching_blend(r['stuffScore'], r['locPlus']) for r in rows
          if r.get('stuffScore') is not None and r.get('locPlus') is not None
          and (r.get('count') or 0) >= min_pitches]
    if len(bs) < 5:
        return None
    mu = sum(bs) / len(bs)
    sd = (sum((b - mu) ** 2 for b in bs) / len(bs)) ** 0.5
    return (mu, sd) if sd > 1e-9 else None


def _pitching_score(stuff, loc, scale=None):
    # COHERENT CANON (2026-07-18): Pitching+ = the shared PITCHING_W_STUFF
    # blend of Stuff+ and Loc+ exactly (pipeline_utils holds the constant;
    # train_stuff.PITCHING_W_STUFF carries the derivation) — no
    # restandardization, no clip. `scale` is accepted for call-site compat
    # and ignored.
    from pipeline_utils import PITCHING_W_STUFF
    if stuff is None or loc is None:
        return None
    return round(PITCHING_W_STUFF * stuff + (1.0 - PITCHING_W_STUFF) * loc, 1)


def _normalize_scratch_pitch(row):
    """Sheet row → pipeline-format pitch dict. Mirrors
    pipeline_fetch.read_pitches_from_sheet: blanks → None, Barrel recompute
    fallback, plus the InZone recompute the pipeline applies in process_data
    (the scratch tab has no InZone column). Movement has NO fallback — see
    read_pitches_from_sheet."""
    from pipeline_utils import compute_in_zone, is_barrel
    p = {k: (None if v == '' else v) for k, v in row.items()}
    if not p.get('Barrel'):
        p['Barrel'] = '6' if is_barrel(sf(p.get('ExitVelo')), sf(p.get('LaunchAngle'))) else ''
    p['InZone'] = compute_in_zone(p)
    return p


def _rank_in_mlb_pool(val, sorted_pool, invert=False):
    """Percentile of val against a sorted MLB value pool — the interpolation
    path of pipeline_compute.compute_percentile_ranks_with_aaa (fraction
    below + half ties, rounded, clamped)."""
    if val is None or len(sorted_pool) < 2:
        return None
    import bisect
    below = bisect.bisect_left(sorted_pool, val)
    above = bisect.bisect_right(sorted_pool, val)
    pctl = max(0, min(100, round((below + 0.5 * (above - below)) / len(sorted_pool) * 100)))
    return 100 - pctl if invert else pctl


def _scratch_mlb_pool_rows(rows):
    """MLB percentile pool from leaderboard rows: one row per player (a
    combined 2TM/3TM row replaces its per-team stints), ROC/AAA rows excluded.
    Mirrors compute_percentile_ranks_with_aaa's pool construction."""
    from pipeline_utils import AAA_TEAMS

    def _pkey(r):
        mid = r.get('mlbId')
        if mid is not None and mid != '':
            return 'id:' + str(mid)
        return 'nm:' + (r.get('pitcher') or '')

    combined = {_pkey(r) for r in rows if str(r.get('team', '')).endswith('TM')}
    out = []
    for r in rows:
        t = str(r.get('team', ''))
        if t in AAA_TEAMS:
            continue
        if not t.endswith('TM') and _pkey(r) in combined:
            continue
        out.append(r)
    return out


# Stuff+ shrinkage for DAILY (single-game) cards. Season/scratch-season cards
# use train_stuff.K_SHRINK (=100) because they estimate a stable between-
# pitcher grade over hundreds of pitches. On one game that would compress every
# pitch type toward 100. Stuff+ grades pitch SHAPES, which stabilize in ~10
# pitches, so a daily card can grade the shapes he actually threw with light
# shrinkage — the number moves game-to-game (Wally's "grade today's shapes").
# Window cards (single game / date range / scratch) show the PLAIN AVERAGE
# of per-pitch grades — no shrink (2026-07-18, per Wally: 'card should be
# average'). k=0 makes _scratch_stuff_scores' unit value exactly the mean
# of the per-pitch grades on the same scale as the Sheets grade columns.
K_SHRINK_DAILY = 0


def _scratch_stuff_scores(norm_by_pitcher, k_shrink=None):
    """Stuff+ (v12 config since 2026-08-09) for scratch pitchers. Returns
    ({pitcher: overall}, {(pitcher, pitch_type): score}). The v12 features
    (nVAA, release-axis cross, FF/SI velo_diff mask) live inside
    train_stuff.build_df, so this path picks them up automatically —
    the guard below only enforces that the LOCAL bundle is a v12 retrain
    (a stale v11 bundle would score transformed features with models
    trained on raw ones, silently drifting every grade).
    Full model when the pitcher has any
    ArmAngle data (build_df fills gaps with his own average); otherwise the
    no-arm companion model anchored to its MLB (no-arm) scales — exactly the
    ROC path in train_stuff.main(). k_shrink overrides the season K_SHRINK
    (used lightly for daily cards)."""
    import pickle as _pickle
    _sv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stuff_plus')
    if _sv_dir not in sys.path:
        sys.path.insert(0, _sv_dir)
    import train_stuff as _sv

    with open(os.path.join(_sv_dir, 'stuff_models.pkl'), 'rb') as f:
        bundle = _pickle.load(f)
    if 'kin_eff__ff' not in bundle.get('features', []) \
            or 'cross' not in bundle.get('features', []):
        raise RuntimeError(
            'stuff_models.pkl predates the v13 config — its models '
            'trained on different features than build_df now emits. Refresh '
            'it from the latest-data release:\n  curl -sL https://github.com/'
            'wjhuron/Huronalytics/releases/download/latest-data/'
            'stuff_models.pkl.gz | gunzip -c > '
            'stuff_plus/stuff_models.pkl')

    all_pitches = [p for pl in norm_by_pitcher.values() for p in pl]
    # v13: 2026 kinematics ride the PitchID sidecar; daily/scratch pitches
    # come off the sheets without KinEff. Absent sidecar degrades to the
    # build_df imputation chain (pitcher FF mean -> frame -> league const).
    _n_kin = _sv.apply_kin_sidecar(all_pitches)
    if _n_kin >= 0:
        print(f"  [ctx] kinematics sidecar: {_n_kin} pitches filled")
    else:
        print("  [ctx] WARNING: kinematics sidecar absent — kin_eff__ff "
              "scores from the imputation chain")
    df = _sv.build_df(all_pitches)
    overall, per_pt, atoms_by_pid = {}, {}, {}
    if not len(df):
        return overall, per_pt, atoms_by_pid

    # COHERENT CANON (2026-07-18): window values are plain averages of the
    # per-pitch INTEGER atoms — the same integers written to the Sheets grade
    # columns — so AVERAGEIF over the sheet reproduces the card to the digit.
    # k_shrink is accepted for call-site compat and ignored (always 0).
    import numpy as _np
    def _atom(r, scale):
        if not scale:
            return None
        mu, sd = scale.get('mu'), scale.get('sd')
        if mu is None or sd is None or not sd > 0:
            return None
        return int(round(100 + _sv.K_SCALE * (r - mu) / sd))

    # OOF fold-model discipline — the SAME model choice the pipeline's
    # per-pitch grade dump uses (each pitcher scored by the fold that held him
    # out; pitchers newer than the bundle default to fold 0). Without this the
    # card would score with the full-data model and drift ~1 point from the
    # sheet atoms, breaking the exact-match guarantee.
    _folds = bundle.get('fold_pitchers') or []
    _fold_of = {pp: k for k, ps in enumerate(_folds) for pp in ps}
    for pitcher, sub in df.groupby('pitcher'):
        _fi = _fold_of.get(pitcher, 0)
        has_arm = any(sf(p.get('ArmAngle')) is not None
                      for p in norm_by_pitcher.get(pitcher, []))
        if has_arm:
            _fms = bundle.get('fold_models') or []
            model = _fms[_fi] if _fi < len(_fms) else bundle['model']
            X = _sv.design(sub).reindex(columns=bundle['features'], fill_value=0)
            pt_scale = bundle['league']
        else:
            _fms = bundle.get('fold_models_na') or []
            model = _fms[_fi] if _fi < len(_fms) else bundle['model_na']
            na_cols = model.get_booster().feature_names
            X = _sv.design(sub, bundle['noarm_feats']).reindex(columns=na_cols, fill_value=0)
            pt_scale = bundle['na_pt_scale']
        raw = -model.predict(X)
        sub = sub.reset_index(drop=True)
        atoms_all = []
        for pt in sub['pitch_type'].unique():
            mask = (sub['pitch_type'] == pt).values
            atoms = [a for a in (_atom(float(r), pt_scale.get(pt)) for r in raw[mask])
                     if a is not None]
            if atoms:
                per_pt[(pitcher, pt)] = round(float(_np.mean(atoms)), 1)
                atoms_all.extend(atoms)
        # per-pitch atoms by PitchID — the fallback source when a sheet row's
        # grade cells are still blank (fresh game before the pipeline ran)
        for _pid, _pt3, _r in zip(sub['pid'], sub['pitch_type'], raw):
            _a = _atom(float(_r), pt_scale.get(_pt3))
            if _pid and _a is not None:
                atoms_by_pid[str(_pid)] = _a
        overall[pitcher] = (round(float(_np.mean(atoms_all)), 1)
                            if atoms_all else None)
    return overall, per_pt, atoms_by_pid


_HRFB_PU_POOLS = None      # lazy all-MLB pools for the HR/FB% + PU% bubbles


def _hrfb_pu_pools():
    """All-MLB per-pitcher HR/FB% and PU% value pools for the bubble
    percentiles (min 20 FB / 50 BIP; the all-MLB pool convention). Reuses
    the scratch context's pickle when it is already loaded."""
    global _HRFB_PU_POOLS
    if _HRFB_PU_POOLS is not None:
        return _HRFB_PU_POOLS
    import pickle as _pickle
    from collections import defaultdict as _dd
    try:
        if _MLB_PICKLE_CACHE is not None:
            _mlb = _MLB_PICKLE_CACHE
        else:
            with open(os.path.join(os.path.dirname(METADATA_PATH),
                                   'all_pitches_rs_cache.pkl'), 'rb') as f:
                _all = _pickle.load(f)
            _mlb = [p for p in _all if p.get('_source') == 'MLB']
        acc = _dd(lambda: [0, 0, 0, 0])   # fb, hr, pu, bip
        for p in _mlb:
            if p.get('Description') != 'In Play':
                continue
            bb = str(p.get('BBType') or '')
            if not bb or bb.startswith('bunt'):
                continue
            a = acc[p.get('Pitcher')]
            a[3] += 1
            if bb == 'fly_ball':
                a[0] += 1
                # strict conditional: fly-ball HRs only (matches
                # _batted_ball_line)
                if p.get('Event') == 'Home Run':
                    a[1] += 1
            elif bb == 'popup':
                a[2] += 1
        _HRFB_PU_POOLS = {
            'hrfb': sorted(a[1] / a[0] * 100 for a in acc.values()
                           if a[0] >= 20),
            'pu': sorted(a[2] / a[3] * 100 for a in acc.values()
                         if a[3] >= 50),
        }
    except Exception as _e:
        print(f"  WARNING: HR/FB-PU pools unavailable ({_e})")
        _HRFB_PU_POOLS = {}
    return _HRFB_PU_POOLS


def _pctl_of(value, pool, invert=False):
    """Percentile of value in a sorted pool; invert for lower-is-better."""
    import bisect
    if not pool:
        return None
    pct = bisect.bisect_left(pool, value) / len(pool) * 100
    return round(100 - pct if invert else pct)


_MLB_PICKLE_CACHE = None   # module-level: load the 382k-pitch pickle once per process
_ARM_LOOKUP_CACHE = None   # {(pitcher, pitch type) -> mean AA}, ALL levels


def _build_arm_lookup(all_pitches):
    """Per-pitcher arm-angle averages from the FULL pitch cache — MLB *and*
    ROC/AAA. Built before the MLB filter below precisely because the MiLB rows
    are the point: a debut arm has no MLB history to average."""
    pt_acc, all_acc = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for p in all_pitches:
        aa = sf(p.get('ArmAngle'))
        if aa is None:
            continue
        nm, pt = p.get('Pitcher'), p.get('Pitch Type')
        if not nm:
            continue
        if pt:
            a = pt_acc[(nm, pt)]; a[0] += aa; a[1] += 1
        a = all_acc[nm]; a[0] += aa; a[1] += 1
    return {'pt': {k: v[0] / v[1] for k, v in pt_acc.items() if v[1]},
            'all': {k: v[0] / v[1] for k, v in all_acc.items() if v[1]}}


def _backfill_arm_angle(norm_by_pitcher, lookup):
    """Fill missing ArmAngle from the pitcher's own history, per pitch type
    first, then his overall average.

    WHY THIS EXISTS. ArmAngle arrives ~2 days after a game via the Savant
    supplement, and train_stuff.build_df already fills gaps from the
    pitcher's own average — but only from the frame it is handed. A daily card
    hands it ONE game, so a debut or callup has nothing to average and drops to
    the no-arm companion model. His MiLB arm angle is sitting right there in
    the cache (ROC is 96-97% populated since the minors Statcast backfill), and
    arm angle is near-constant per pitcher, so his own ROC average is
    essentially the real value. Replaced by the actual number once the
    supplement lands.

    Per pitch type matters: a pitcher can vary a lot by type (Jackson Kent's
    ROC CU sits at 50.3 degrees against 33.7 for his CH), so his overall
    average would misstate both ends.
    """
    if not lookup:
        return 0
    filled, by_src = 0, defaultdict(int)
    for nm, pl in norm_by_pitcher.items():
        for p in pl:
            if sf(p.get('ArmAngle')) is not None:
                continue
            v = lookup['pt'].get((nm, p.get('Pitch Type')))
            src = 'pitch type'
            if v is None:
                v = lookup['all'].get(nm)
                src = 'pitcher avg'
            if v is not None:
                p['ArmAngle'] = round(v, 2)
                filled += 1
                by_src[(nm, src)] += 1
    if filled:
        print(f"  [ctx] arm-angle backfill: {filled} pitches filled from cached "
              f"history (incl. MiLB)")
        for (nm, src), n in sorted(by_src.items()):
            print(f"        {nm}: {n} from his {src} average")
    return filled


def _build_scratch_league_context(norm_by_pitcher, stuff_k_shrink=None):
    """Heavy one-time setup for scratch-tab / daily cards: MLB pickle baselines
    (Loc+ surfaces + norm pool, xRV count anchoring), Stuff+ scoring (v12
    config via the bundle),
    leaderboard percentile pools, nVAA/nHAA regressions. stuff_k_shrink is
    passed through to Stuff+ scoring (light for daily cards)."""
    global _MLB_PICKLE_CACHE, _ARM_LOOKUP_CACHE
    import pickle as _pickle
    from pipeline_compute import build_bip_count_means
    from pipeline_sdplus import build_bip_count_offsets
    from pipeline_locplus import compute_loc_plus

    t0 = time_module.time()
    ctx = {'norm_by_pitcher': norm_by_pitcher}

    if _MLB_PICKLE_CACHE is None or _ARM_LOOKUP_CACHE is None:
        print("  [ctx] Loading MLB pitch pickle for league baselines...")
        with open(os.path.join(os.path.dirname(METADATA_PATH), 'all_pitches_rs_cache.pkl'), 'rb') as f:
            _all = _pickle.load(f)
        # Built from ALL sources before the MLB filter — see _build_arm_lookup.
        _ARM_LOOKUP_CACHE = _build_arm_lookup(_all)
        _MLB_PICKLE_CACHE = [p for p in _all if p.get('_source') == 'MLB']
    mlb = _MLB_PICKLE_CACHE
    print(f"  [ctx] {len(mlb)} MLB pitches ready ({time_module.time()-t0:.0f}s)")

    # Must run BEFORE Stuff+ scoring: build_df only sees this window's pitches,
    # so a debut arm would otherwise fall to the no-arm companion model.
    _backfill_arm_angle(norm_by_pitcher, _ARM_LOOKUP_CACHE)

    # xRV count anchoring — same currency as the leaderboard's xRV.
    ctx['count_offsets'] = build_bip_count_offsets(mlb, GUTS_LG_WOBA, GUTS_WOBA_SCALE)
    ctx['bip_count_means'] = build_bip_count_means(mlb, GUTS_LG_WOBA, GUTS_WOBA_SCALE,
                                                   ctx['count_offsets'])

    # Loc+ — the pipeline's own entry point. Scratch pitchers are keyed under
    # team 'AAA' so they score against the MLB surfaces but stay OUT of the
    # normalization (mu, sigma) pool, exactly like ROC pitchers.
    print("  [scratch] Building Loc+ surfaces + scoring MLB pool...")
    by_pitcher, by_pt = defaultdict(list), defaultdict(list)
    for p in mlb:
        k = (p.get('Pitcher'), p.get('PTeam'), p.get('Throws'))
        by_pitcher[k].append(p)
        by_pt[(k[0], k[1], p.get('Pitch Type'), k[2])].append(p)
    for name, plist in norm_by_pitcher.items():
        for p in plist:
            by_pitcher[(name, 'AAA', p.get('Throws'))].append(p)
            by_pt[(name, 'AAA', p.get('Pitch Type'), p.get('Throws'))].append(p)
    from pipeline_locplus import (group_of, LOC_SCALE_K, score_pitch,
                                  _is_scorable as _loc_scorable)
    loc_pr, loc_ptr, _, _loc_anchors = compute_loc_plus(
        mlb, by_pitcher, by_pt, GUTS_LG_WOBA, GUTS_WOBA_SCALE,
        return_anchors=True)
    # COHERENT CANON: window Loc+ = plain average of the per-pitch INTEGER
    # atoms (group-anchored — identical to the Sheets Loc+ cells), for both
    # the per-type rows and the overall value. No priors, no clip.
    _loc_S = _loc_anchors['surfaces']
    _loc_pt_anc = _loc_anchors['pt']
    def _loc_atom(p):
        if not _loc_scorable(p):
            return None
        anc = _loc_pt_anc.get(group_of(p))
        if not anc or anc[0] is None or not anc[1] or anc[1] <= 1e-12:
            return None
        v = score_pitch(p, _loc_S)
        if v is None:
            return None
        return int(round(100.0 - LOC_SCALE_K * (v - anc[0]) / anc[1]))
    ctx['loc_atom_fn'] = _loc_atom   # per-pitch fallback for blank sheet cells
    ctx['loc_overall'], ctx['loc_pt'] = {}, {}
    for _nm, _plist in norm_by_pitcher.items():
        _by_type = defaultdict(list)
        for _p in _plist:
            _a = _loc_atom(_p)
            if _a is not None and _p.get('Pitch Type'):
                _by_type[_p['Pitch Type']].append(_a)
        _all = [a for arr in _by_type.values() for a in arr]
        if _all:
            ctx['loc_overall'][_nm] = round(sum(_all) / len(_all), 1)
        for _pt2, _arr in _by_type.items():
            ctx['loc_pt'][(_nm, _pt2)] = round(sum(_arr) / len(_arr), 1)
    print(f"  [scratch] Loc+ done ({time_module.time()-t0:.0f}s)")

    # Stuff+ (v12 config; version enforced by the bundle guard in
    # _scratch_stuff_scores)
    print("  [scratch] Scoring Stuff+ v12...")
    try:
        (ctx['stuff_overall'], ctx['stuff_pt'],
         ctx['stuff_atoms_by_pid']) = _scratch_stuff_scores(norm_by_pitcher, stuff_k_shrink)
    except Exception as _e:
        print(f"  WARNING: Stuff+ scoring failed for scratch pitches: {_e}")
        ctx['stuff_overall'], ctx['stuff_pt'], ctx['stuff_atoms_by_pid'] = {}, {}, {}

    # Percentile pools from the leaderboard JSONs.
    _data_dir = os.path.dirname(METADATA_PATH)
    ctx['pitcher_pools'], ctx['pitch_pools'] = {}, {}
    try:
        with open(os.path.join(_data_dir, 'pitcher_leaderboard_rs.json')) as f:
            _raw_prows = json.load(f)
        _prows = _scratch_mlb_pool_rows(_raw_prows)
        for s in _SCRATCH_POOL_STATS:
            vals = [r.get(s) for r in _prows]
            ctx['pitcher_pools'][s] = sorted(v for v in vals if v is not None)
        # Pitcher+ percentile pool — the pipeline's own convention (qualified
        # MLB baseline rows, apply_pitcher_plus), NOT the all-MLB pool the
        # bubble stats rank in.
        from pipeline_pitcherplus import _is_baseline as _pp_is_baseline
        ctx['pitcher_plus_pool'] = sorted(
            v for v in (r.get('pitcherPlus') for r in _raw_prows
                        if _pp_is_baseline(r, ('ROC', 'AAA')))
            if v is not None)
        # Pitching+ blend scale (overall + per type) from the same MLB pools,
        # so scratch pitchers score on the exact league standardization.
        ctx['pitching_scale'] = _pitching_scale(_prows)
        with open(os.path.join(_data_dir, 'pitch_leaderboard_rs.json')) as f:
            _plrows = _scratch_mlb_pool_rows(json.load(f))
        pt_pools = defaultdict(dict)
        by_type = defaultdict(list)
        for r in _plrows:
            by_type[r.get('pitchType')].append(r)
        ctx['pitching_scale_pt'] = {}
        for pt, rows in by_type.items():
            for s in ('velocity', 'nVAA', 'stuffScore', 'locPlus'):
                pt_pools[pt][s] = sorted(v for v in (r.get(s) for r in rows) if v is not None)
            ctx['pitching_scale_pt'][pt] = _pitching_scale(rows)
        ctx['pitch_pools'] = dict(pt_pools)
    except Exception as _e:
        print(f"  WARNING: could not build scratch percentile pools: {_e}")

    # nVAA / nHAA regressions from metadata.
    ctx['vaa_reg'], ctx['haa_reg'] = {}, {}
    ctx['pitcher_plus_base'] = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _m = json.load(f)
        ctx['vaa_reg'] = _m.get('vaaRegressions') or {}
        ctx['haa_reg'] = _m.get('haaRegressions') or {}
        # Pitcher+ (mu, sigma) baseline published by pipeline_pitcherplus —
        # reshaped into the {key: (mu, sd), '_composite': (mu, sd)} form
        # score_row consumes, so scratch pitchers score on the exact league
        # standardization the leaderboard used.
        try:
            from pipeline_pitcherplus import COMPONENTS as _PP_COMPONENTS
            _ppb = _m.get('pitcherPlusBaseline') or {}
            _base = {c['key']: (c['mu'], c['sd'])
                     for c in _ppb.get('components', [])}
            _base['_composite'] = (_ppb['composite']['mu'],
                                   _ppb['composite']['sd'])
            if all(k in _base for k, _w, _k2 in _PP_COMPONENTS):
                ctx['pitcher_plus_base'] = _base
        except (KeyError, TypeError):
            print("  WARNING: metadata pitcherPlusBaseline missing/unreadable "
                  "— scratch cards will show Pitcher+ as —")

    print(f"  [scratch] League context ready ({time_module.time()-t0:.0f}s)")
    return ctx


def _compute_scratch_pitcher_context(pitcher_name, ctx):
    """One scratch pitcher's MLB-style card context, computed from his scratch
    pitches. Returns (pctl_row, pitch_lb, locplus_by_pt) shaped exactly like
    the leaderboard-sourced structures the regular MLB render path consumes."""
    from pipeline_compute import (compute_stats, compute_expected_stats,
                                  compute_pitcher_batted_ball, compute_xrv)

    pitches = ctx['norm_by_pitcher'].get(pitcher_name) or []
    n = len(pitches)
    if n == 0:
        return None, {}, {}
    throws = pitches[0].get('Throws')

    row = {'count': n}
    row.update(compute_stats(pitches))
    row.update(compute_expected_stats(pitches, None))
    row.update(compute_pitcher_batted_ball(pitches))
    rv = row.get('runValue')
    row['rv100'] = rv / n * 100 if rv is not None else None
    xrv = compute_xrv(pitches, GUTS_LG_WOBA, GUTS_WOBA_SCALE,
                      count_offsets=ctx.get('count_offsets'),
                      bip_count_means=ctx.get('bip_count_means'))['xRunValue']
    row['xRunValue'] = xrv
    row['xRv100'] = xrv / n * 100 if xrv is not None else None

    # fbVelo — avg velo of the most-used fastball (FF/SI), matching process_data.
    fb_by_type = defaultdict(list)
    for p in pitches:
        if p.get('Pitch Type') in ('FF', 'SI'):
            v = sf(p.get('Velocity'))
            if v is not None:
                fb_by_type[p['Pitch Type']].append(v)
    if fb_by_type:
        fbv = fb_by_type[max(fb_by_type, key=lambda t: len(fb_by_type[t]))]
        row['fbVelo'] = round(sum(fbv) / len(fbv), 1)
    else:
        row['fbVelo'] = None

    # extension — mean over all pitches, matching process_data's pitcher-level
    # METRIC_COLS aggregation, so a scratch card's Extension sits on the same
    # footing as the leaderboard values its percentile pool is built from.
    _exts = [v for v in (sf(p.get('Extension')) for p in pitches) if v is not None]
    row['extension'] = round(sum(_exts) / len(_exts), 1) if _exts else None

    # COHERENT CANON: atoms come from the sheet's integer grade cells when
    # present (so card values equal AVERAGEIF over the sheet exactly), with
    # model-computed fallbacks only for rows whose cells are still blank.
    def _pitch_atoms(p):
        s_c, l_c, p_c = sf(p.get('Stuff+')), sf(p.get('Loc+')), sf(p.get('Pitching+'))
        S = (int(round(s_c)) if s_c is not None
             else ctx.get('stuff_atoms_by_pid', {}).get(str(p.get('PitchID') or '')))
        L = (int(round(l_c)) if l_c is not None
             else (ctx['loc_atom_fn'](p) if ctx.get('loc_atom_fn') else None))
        if p_c is not None:
            P = int(round(p_c))
        elif S is not None and L is not None:
            # Canon blend — same shared constant as _pitching_score.
            from pipeline_utils import PITCHING_W_STUFF as _PW
            P = int(round(_PW * S + (1.0 - _PW) * L))
        else:
            P = None
        return S, L, P

    _Sa, _La, _Pa = [], [], []
    for _p in pitches:
        _s, _l, _pp = _pitch_atoms(_p)
        if _s is not None: _Sa.append(_s)
        if _l is not None: _La.append(_l)
        if _pp is not None: _Pa.append(_pp)
    row['stuffScore'] = round(sum(_Sa) / len(_Sa), 1) if _Sa else None
    row['locPlus'] = round(sum(_La) / len(_La), 1) if _La else None
    row['pitchingScore'] = round(sum(_Pa) / len(_Pa), 1) if _Pa else None

    # Percentile bubbles — rank each computed stat into the MLB pool.
    for s in _SCRATCH_POOL_STATS:
        row[s + '_pctl'] = _rank_in_mlb_pool(row.get(s), ctx['pitcher_pools'].get(s) or [],
                                             invert=(s in _SCRATCH_INVERT_PITCHER))

    # Pitcher+ — the six-component shrunk-z composite, scored on the
    # published season baseline (metadata pitcherPlusBaseline) exactly as
    # apply_pitcher_plus scores leaderboard rows, then ranked in the
    # qualified-MLB pool that column ships with. All six components
    # (stuffScore, locPlus, kPct, izWhiffPct, xRv100, gbPct) are already in
    # `row`; a missing one shrinks to league average inside score_row.
    if ctx.get('pitcher_plus_base'):
        from pipeline_pitcherplus import score_row as _pp_score_row
        row['pitcherPlus'] = _pp_score_row(row, ctx['pitcher_plus_base'])
        row['pitcherPlus_pctl'] = _rank_in_mlb_pool(
            row.get('pitcherPlus'), ctx.get('pitcher_plus_pool') or [])

    # Per-pitch-type rows (nVAA/nHAA, velo, RV rates, Stuff+, Loc+).
    by_pt = defaultdict(list)
    for p in pitches:
        if p.get('Pitch Type'):
            by_pt[p['Pitch Type']].append(p)
    pitch_lb, locplus_by_pt = {}, {}
    for pt, pp in by_pt.items():
        npt = len(pp)
        d = {}
        d['count'] = npt
        velos = [v for v in (sf(x.get('Velocity')) for x in pp) if v is not None]
        d['velocity'] = round(sum(velos) / len(velos), 1) if velos else None

        # nVAA — mean VAA normalized to the league-average plate height.
        vaas = [v for v in (sf(x.get('VAA')) for x in pp) if v is not None]
        pzs = [v for v in (sf(x.get('PlateZ')) for x in pp) if v is not None]
        reg = ctx['vaa_reg'].get(pt)
        if vaas and pzs and reg:
            d['nVAA'] = round(sum(vaas) / len(vaas)
                              - reg['slope'] * (sum(pzs) / len(pzs) - reg['leagueAvgPlateZ']), 2)
        else:
            d['nVAA'] = None

        # nHAA — mean HAA normalized to the hand-specific league plate side.
        haas = [v for v in (sf(x.get('HAA')) for x in pp) if v is not None]
        pxs = [v for v in (sf(x.get('PlateX')) for x in pp) if v is not None]
        hreg = ctx['haa_reg'].get(pt)
        lg_px = (hreg or {}).get('leagueAvgPlateX', {}).get(throws)
        if haas and pxs and hreg and lg_px is not None:
            d['nHAA'] = round(sum(haas) / len(haas)
                              - hreg['slope'] * (sum(pxs) / len(pxs) - lg_px), 2)
        else:
            d['nHAA'] = None

        rvs = [v for v in (sf(x.get('RunExp')) for x in pp) if v is not None]
        d['rv100'] = sum(rvs) / npt * 100 if rvs else None
        xrv_pt = compute_xrv(pp, GUTS_LG_WOBA, GUTS_WOBA_SCALE,
                             count_offsets=ctx.get('count_offsets'),
                             bip_count_means=ctx.get('bip_count_means'))['xRunValue']
        d['xRunValue'] = xrv_pt
        d['xRv100'] = xrv_pt / npt * 100 if xrv_pt is not None else None

        _Spt, _Lpt, _Ppt = [], [], []
        for _p2 in pp:
            _s2, _l2, _pp2 = _pitch_atoms(_p2)
            if _s2 is not None: _Spt.append(_s2)
            if _l2 is not None: _Lpt.append(_l2)
            if _pp2 is not None: _Ppt.append(_pp2)
        d['stuffScore'] = round(sum(_Spt) / len(_Spt), 1) if _Spt else None
        lp = round(sum(_Lpt) / len(_Lpt), 1) if _Lpt else None
        if lp is not None:
            locplus_by_pt[pt] = lp
        d['pitchingScore'] = round(sum(_Ppt) / len(_Ppt), 1) if _Ppt else None

        pools = ctx['pitch_pools'].get(pt, {})
        d['velocity_pctl'] = _rank_in_mlb_pool(d['velocity'], pools.get('velocity') or [])
        nvp = _rank_in_mlb_pool(d['nVAA'], pools.get('nVAA') or [])
        # nVAA direction flips by pitch type (matches process_data's
        # VAA_NO_INVERT_TYPES): steeper is better except for FF/FC.
        if nvp is not None and pt not in ('FF', 'FC'):
            nvp = 100 - nvp
        d['nVAA_pctl'] = nvp
        d['stuffScore_pctl'] = _rank_in_mlb_pool(d['stuffScore'], pools.get('stuffScore') or [])
        pitch_lb[pt] = d

    # Per-hand Loc+ for the location-plot titles. Same integer atoms as every
    # other displayed grade (coherent canon), just filtered by the batter's
    # side. Rides on locplus_by_pt under keys no pitch type can collide with,
    # so no call signature changes; consumers that don't know about them (the
    # metrics table, which only ever does .get(pitch_type)) never see them.
    for _h in ('R', 'L'):
        _Lh = []
        for _p3 in pitches:
            if _p3.get('Bats') != _h:
                continue
            _s3, _l3, _pp3 = _pitch_atoms(_p3)
            if _l3 is not None:
                _Lh.append(_l3)
        if _Lh:
            locplus_by_pt[f'_vs{_h}'] = round(sum(_Lh) / len(_Lh), 1)

    return row, pitch_lb, locplus_by_pt


# ═══════════════════════════════════════════════════════════════
# MAIN BATCH LOGIC
# ═══════════════════════════════════════════════════════════════
def _resolve_pitcher_teams(names, include_non_mlb=False):
    """Auto whole-season team resolution for --pitchers-without-a---team.

    Looks each requested pitcher up in the season pitcher leaderboard and returns
    (union_of_team_tabs_to_read, {pitcher_name: leaderboard_team_label}).

    A pitcher with MLB stints on more than one team resolves to all of those
    tabs with a 2TM/3TM label (matching the pipeline's combined leaderboard row),
    so a traded arm's full season is stitched together automatically. A pitcher
    with a single MLB stint resolves to that team; an arm with no MLB rows (e.g.
    AAA-only) resolves to its non-MLB tab(s). Aggregate 2TM/3TM pseudo-teams are
    never read directly (they are not real worksheets).

    include_non_mlb=True (the --all-levels flag) also folds a pitcher's AAA (and
    any other non-MLB) stints into the combine, so an arm that split the year
    between MLB and AAA gets ONE cross-level card. Such an arm is labeled
    'MLB+AAA'; because no combined leaderboard row exists for it, the caller
    computes its grades/bubbles from the combined pitches instead."""
    MLB = AL_TEAMS | NL_TEAMS
    lb_path = os.path.join(os.path.dirname(METADATA_PATH), 'pitcher_leaderboard_rs.json')
    by_name = defaultdict(set)
    try:
        with open(lb_path) as f:
            for r in json.load(f):
                nm, tm = r.get('pitcher'), r.get('team')
                if nm and tm:
                    by_name[nm].add(tm)
    except Exception as e:
        print(f"  WARNING: could not load leaderboard for team auto-resolve: {e}")
    union, labels = set(), {}
    for nm in names:
        real = {t for t in by_name.get(nm, set()) if t not in ('2TM', '3TM')}
        mlb_stints = {t for t in real if t in MLB}
        if include_non_mlb:
            used = sorted(real)
        else:
            used = sorted(mlb_stints) if mlb_stints else sorted(real)
        if not used:
            print(f"  WARNING: no leaderboard team found for '{nm}' — cannot auto-resolve")
            continue
        union.update(used)
        if mlb_stints and set(used) - MLB:
            labels[nm] = 'MLB+AAA'   # cross-level combine → grades from pitches
        else:
            labels[nm] = f"{len(used)}TM" if len(used) > 1 else used[0]
    return sorted(union), labels


def main():
    # ── Settings (edit these directly or override via command line) ──
    team            = "NEW"
    start_date      = None    # Set to None for full season
    end_date        = None             # Set to a date for date range, or None for single day
    filter_pitchers = "Martinez, Seth"                 # Semicolon-separated "Last, First" names, or "" for all
    game_pk         = ""                 # Optional game PK for live/in-progress games
    display_team    = None               # Header team label override (display only)
    output_dir      = OUTPUT_DIR

    # ── CLI overrides (optional — values above are used if no args passed) ──
    parser = argparse.ArgumentParser(description='Generate pitcher stat cards')
    parser.add_argument('--team', default=None, help='Team abbreviation')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD, or "none" for full season')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--pitchers', default=None, help='Semicolon-separated "Last, First" names')
    parser.add_argument('--game-pk', default=None, help='Game PK for live/in-progress games')
    parser.add_argument('--output-dir', default=None, help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--rv-mode', default='per100', choices=['per100', 'totals', 'both'],
                        help='Season-card RV columns: per-100 rates (default), cumulative '
                             'totals (PitchRV/xPitchRV), or both pairs. Single-game cards '
                             'always show cumulative xPitchRV.')
    parser.add_argument('--pitch-qual', type=int, default=None,
                        help='Min pitches for a pitch type\'s RV COLORING '
                             f'(default {CARD_COLOR_MIN_PITCHES}; values always render)')
    parser.add_argument('--display-team', default=None,
                        help='Team label shown in the card header (display '
                             'only — does not change which tab/team the '
                             'pitches are read from). Useful for scratch-tab '
                             'pulls, e.g. --team NEW --display-team WSH.')
    parser.add_argument('--tab', default=None,
                        help='Read pitches from this scratch tab in the NLE2026 '
                             'workbook (e.g. Sheet2) instead of a team tab. '
                             'Scratch data never touches the leaderboards; cards '
                             'render MiLB-style (no percentile bubbles).')
    parser.add_argument('--all-levels', action='store_true',
                        help='Auto whole-season mode only (no --team): also fold in '
                             'a pitcher\'s AAA/non-MLB stints so an arm split between '
                             'MLB and AAA gets ONE combined card. No combined '
                             'leaderboard row exists for a cross-level arm, so its '
                             'grades and percentile bubbles are computed from the '
                             'combined pitches vs the MLB pool; the card is labeled '
                             'MLB+AAA. Shape grades (Stuff+/Loc+/velo/movement) are '
                             'competition-agnostic; outcome bubbles (K%%, xwOBA, '
                             'Whiff%%...) blend AAA and MLB hitters.')
    args = parser.parse_args()

    if args.team is not None: team = args.team
    if args.start is not None: start_date = None if args.start.lower() == 'none' else args.start
    if args.end is not None: end_date = args.end
    if args.pitchers is not None: filter_pitchers = args.pitchers
    if args.game_pk is not None: game_pk = args.game_pk
    if args.display_team is not None: display_team = args.display_team
    if args.output_dir is not None: output_dir = args.output_dir
    rv_mode = args.rv_mode
    pitch_qual = args.pitch_qual

    # Parse filter_pitchers string into list
    if filter_pitchers:
        filter_pitchers = [p.strip() for p in filter_pitchers.split(';') if p.strip()]
    # ──────────────────────────────────────────────────────────

    # Teams: a comma-separated --team (e.g. TOR,LAD) combines a multi-team
    # pitcher's full season. Pitch data is read from each team's worksheet; the
    # bubbles use the pipeline's synthetic 2TM/3TM combined leaderboard row.
    # team = "NEW" selects the scratch tab of the same name in the NLE2026
    # workbook (never read by the pipeline, so it cannot reach the site);
    # --tab overrides for other scratch tab names.
    scratch_tab = args.tab or ('NEW' if str(team).strip().upper() == 'NEW' else None)
    # Auto whole-season mode: pitchers given but no team. Resolve each
    # pitcher's stint team(s) from the leaderboard so multi-team arms combine
    # their full season automatically (see _resolve_pitcher_teams). Per-pitcher
    # lookups then use each arm's own leaderboard label via pitcher_team_label.
    #
    # WHO decides the team depends on HOW the script was invoked (2026-08-13,
    # per Wally). A CLI run (`--pitchers` without `--team`) auto-resolves even
    # though the Settings block always carries some leftover team. A
    # settings-block run (no CLI args at all) treats the Settings team as
    # authoritative: team="ROC" + filter_pitchers renders the ROC season, with
    # auto mode only when team is left empty. Keying purely on args.team broke
    # the day Jackson Kent debuted: team="ROC" in Settings was silently
    # discarded because args.team was None, auto-resolution found his new MLB
    # leaderboard row, and the "ROC season" run rendered his one WSH start.
    _cli_run = len(sys.argv) > 1
    auto_team = (not scratch_tab and bool(filter_pitchers)
                 and (args.team is None if _cli_run
                      else not str(team or '').strip()))
    # Cross-level combine (--all-levels): grades/bubbles must be computed from the
    # combined pitches (no leaderboard row spans MLB+AAA), via the scratch context.
    compute_from_pitches = bool(auto_team and args.all_levels)
    pitcher_team_label = {}   # pitcher_name -> leaderboard team label for per-pitcher lookups
    if scratch_tab:
        # Scratch-tab mode: pitch data comes from a non-team tab (never read
        # by the pipeline, so it can't leak to the site). MiLB-style render.
        teams = [scratch_tab]
        team = scratch_tab
        league = 'MiLB'
    elif auto_team:
        teams, pitcher_team_label = _resolve_pitcher_teams(
            filter_pitchers, include_non_mlb=args.all_levels)
        if not teams:
            print("Error: could not resolve any team for: " + ', '.join(filter_pitchers))
            sys.exit(1)
        team = 'Season'    # display placeholder; each card uses its own team label
        league = 'MLB'
        print("Auto whole-season teams: " + ", ".join(teams) + "  "
              + " ".join(f"[{n} -> {lbl}]" for n, lbl in pitcher_team_label.items()))
    else:
        teams = [t.strip() for t in str(team).split(',') if t.strip()]
        if not teams:
            print("Error: no team specified")
            sys.exit(1)
        for t in teams:
            if t not in AL_TEAMS and t not in NL_TEAMS and t not in MILB_TEAMS:
                print(f"Error: Unknown team '{t}'")
                sys.exit(1)
        if len(teams) > 1:
            league = 'MLB'
            team = f"{len(teams)}TM"   # combined label = leaderboard 2TM/3TM key
        else:
            team = teams[0]
            league = 'AL' if team in AL_TEAMS else ('NL' if team in NL_TEAMS else 'MiLB')
    # A same-day range (--start X --end X) is one game, not a multi-game span:
    # collapse it to a single date so it renders the daily card format.
    if end_date is not None and end_date == start_date:
        end_date = None

    # Resolve date range
    if start_date is None and end_date is None:
        # Full season — no date filter
        date_filter = None
        display_date = f"{datetime.now().year} Season"
        date_label = "full season"
        date_slug = "Season"
    elif end_date is None:
        # Single date
        date_filter = (start_date, start_date)
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        display_date = date_obj.strftime('%B %d, %Y').replace(' 0', ' ')
        date_label = start_date
        date_slug = date_obj.strftime('%m%d%Y')
    else:
        # Date range
        date_filter = (start_date, end_date)
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        display_date = f"{start_obj.strftime('%b %d').replace(' 0', ' ')} – {end_obj.strftime('%b %d, %Y').replace(' 0', ' ')}"
        date_label = f"{start_date} to {end_date}"
        date_slug = f"{start_obj.strftime('%m%d')}-{end_obj.strftime('%m%d%Y')}"

    if filter_pitchers:
        print(f"═══ Generating cards for {', '.join(filter_pitchers)} ({team}) — {date_label} ({league}) ═══\n")
    else:
        print(f"═══ Generating cards for {team} — {date_label} ({league}) ═══\n")

    # Load league averages for percentile coloring
    league_avgs = {}
    overall_avgs = {}
    siera_constant = 5.77  # fallback
    # pipeline_fetch owns the FIP constant fallback; import it rather than
    # re-typing the number so the two cannot drift apart.
    from pipeline_fetch import FIP_CONSTANT_FALLBACK
    fip_constant = FIP_CONSTANT_FALLBACK
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        league_avgs = meta.get('leagueAverages', {})
        overall_avgs = meta.get('pitcherLeagueAverages', {})
        siera_constant = meta.get('sieraConstant', 5.77)
        fip_constant = meta.get('fipConstant', FIP_CONSTANT_FALLBACK)

    # Pitcher leaderboard — source of the season percentile ranks (_pctl) that
    # feed the bubble panel. Indexed by mlbId (primary) and (pitcher, team).
    pctl_by_id, pctl_by_name = {}, {}
    _lb_path = os.path.join(os.path.dirname(METADATA_PATH), 'pitcher_leaderboard_rs.json')
    if os.path.exists(_lb_path):
        try:
            with open(_lb_path) as f:
                for _r in json.load(f):
                    if _r.get('mlbId') is not None:
                        pctl_by_id[str(int(_r['mlbId']))] = _r
                    pctl_by_name[(_r.get('pitcher'), _r.get('team'))] = _r
        except Exception as _e:
            print(f"  WARNING: could not load pitcher leaderboard for bubbles: {_e}")

    # Per-pitch-type Loc+ for the metrics table — from the pitch-level
    # leaderboard, keyed (pitcher, team) -> {pitchType: locPlus}.
    locplus_by_pitcher = defaultdict(dict)
    pitch_lb_by_pitcher = defaultdict(dict)   # ROC cards: nVAA, per-type velo, xRV from leaderboard
    _pl_path = os.path.join(os.path.dirname(METADATA_PATH), 'pitch_leaderboard_rs.json')
    if os.path.exists(_pl_path):
        try:
            with open(_pl_path) as f:
                for _r in json.load(f):
                    _lbkey = (_r.get('pitcher'), _r.get('team'))
                    _lbpt = _r.get('pitchType')
                    if _r.get('locPlus') is not None:
                        locplus_by_pitcher[_lbkey][_lbpt] = _r['locPlus']
                    pitch_lb_by_pitcher[_lbkey][_lbpt] = {
                        'nVAA': _r.get('nVAA'), 'nVAA_pctl': _r.get('nVAA_pctl'),
                        'nHAA': _r.get('nHAA'),
                        'velocity': _r.get('velocity'), 'velocity_pctl': _r.get('velocity_pctl'),
                        'count': _r.get('count'),
                        'xRunValue': _r.get('xRunValue'), 'xRv100': _r.get('xRv100'),
                        'rv100': _r.get('rv100'), 'runValue': _r.get('runValue'),
                        'xwOBAcon': _r.get('xwOBAcon'),
                        'stuffScore': _r.get('stuffScore'), 'stuffScore_pctl': _r.get('stuffScore_pctl'),
                        'pitchingScore': _r.get('pitchingScore'),
                        'xrvoe100': _r.get('xrvoe100'),
                        # Season baselines the daily card reads back: the
                        # self-baseline cell shading, the ghost movement
                        # centroid, and the velocity reference lines.
                        'indVertBrk': _r.get('indVertBrk'),
                        'horzBrk': _r.get('horzBrk'),
                        'spinRate': _r.get('spinRate'),
                        'usagePct': _r.get('usagePct'),
                        'maxVelo': _r.get('maxVelo'),
                        'relPosZ': _r.get('relPosZ'),
                        'relPosX': _r.get('relPosX'),
                        'extension': _r.get('extension'),
                    }
        except Exception as _e:
            print(f"  WARNING: could not load pitch leaderboard for Loc+: {_e}")

    # Multi-game mode: date range or full season
    is_multi_game = (start_date is None) or (end_date is not None)

    # Step 1: Load pitch data from Google Sheets (one worksheet per team)
    print("Step 1: Loading pitch data from Google Sheets...")
    gc = gspread.service_account()
    all_rows = []
    for t in teams:
        _NLE2026 = '1BypxxlWgQAltETOLqccOYigeo8nXX-FIuVv6rhT4anA'
        ws = gc.open_by_key(_NLE2026 if scratch_tab else _workbook_id_for_team(t)).worksheet(t)
        for attempt in range(3):
            try:
                t_rows = ws.get_all_records()
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Sheets API error ({t}), retrying ({attempt+1}/3): {e}")
                    time_module.sleep(2 ** attempt)
                else:
                    raise
        for r in t_rows:
            r['_card_team'] = t      # tag source team for per-team boxscore fetch
        all_rows.extend(t_rows)
        print(f"  {t}: {len(t_rows)} rows")

    _apply_runexp_currency(all_rows)

    # Filter by date range (and optionally by pitcher name)
    pitches_by_pitcher = defaultdict(list)
    game_dates_seen = set()
    team_dates = defaultdict(set)   # source team -> game dates (per-team boxscores)
    for row in all_rows:
        row_date = row.get('Game Date', '')
        if date_filter is not None:
            if row_date < date_filter[0] or row_date > date_filter[1]:
                continue
        pitcher_name = row.get('Pitcher', '')
        if pitcher_name:
            if filter_pitchers and pitcher_name not in filter_pitchers:
                continue
            pitches_by_pitcher[pitcher_name].append(row)
            if row_date:
                game_dates_seen.add(row_date)
                team_dates[row.get('_card_team', team)].add(row_date)

    pitcher_names = sorted(pitches_by_pitcher.keys())
    print(f"  Found {len(pitcher_names)} pitchers across {len(game_dates_seen)} game dates: {', '.join(pitcher_names)}")

    # Season cards: stamp the latest game date for freshness (matches the
    # hitter card's "Through May 31"). game_dates_seen hold 'YYYY-MM-DD'.
    if start_date is None and end_date is None and game_dates_seen:
        try:
            _ld = datetime.strptime(max(game_dates_seen), '%Y-%m-%d')
            display_date = f"{display_date}  ·  Through {_ld.strftime('%b %d').replace(' 0', ' ')}"
        except Exception:
            pass

    if not pitcher_names:
        print(f"  No pitch data found for {team} — {date_label}")
        if filter_pitchers:
            print(f"  (filter_pitchers was set to: {filter_pitchers})")
        sys.exit(0)

    # Scratch-tab season cards: build the computed MLB-style context (Stuff+,
    # Loc+, xRV anchoring, percentile pools, nVAA/nHAA regressions) once for
    # all card pitchers. Heavy (loads the MLB pitch pickle) — scratch only.
    scratch_ctx = None
    if scratch_tab and is_multi_game:
        print("\nStep 1b: Computing scratch-tab MLB-style context...")
        try:
            _norm = {nm: [_normalize_scratch_pitch(r) for r in pl]
                     for nm, pl in pitches_by_pitcher.items()}
            scratch_ctx = _build_scratch_league_context(_norm, stuff_k_shrink=K_SHRINK_DAILY)
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"  WARNING: scratch context failed ({_e}) — rendering MiLB-style")
    elif start_date is not None or compute_from_pitches:
        # WINDOW cards — single game OR partial date range (2026-07-18, per
        # Wally): Stuff+/Loc+/Pitching+ are computed from JUST the window's
        # pitches as the plain average of per-pitch grades (no shrink), scored
        # against the season league anchors. Full-season cards (start_date
        # None) keep the season leaderboard values — EXCEPT --all-levels
        # cross-level cards (compute_from_pitches), which have no combined
        # leaderboard row and so are graded from their combined pitches too.
        print("\nStep 1b: Computing Stuff+/Loc+ context from pitches...")
        try:
            _norm = {nm: [_normalize_scratch_pitch(r) for r in pl]
                     for nm, pl in pitches_by_pitcher.items()}
            scratch_ctx = _build_scratch_league_context(_norm, stuff_k_shrink=K_SHRINK_DAILY)
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"  WARNING: daily Stuff+/Loc+ context failed ({_e}) — omitting those columns")

    # Step 2: Fetch boxscore stats (per source team, aggregated across game dates)
    print("\nStep 2: Fetching boxscore stats from MLB API...")
    box_stats = {}
    _single_date = len(game_dates_seen) == 1
    if scratch_tab:
        # Scratch-tab mode: the rows' PitchIDs embed the game_pks the data
        # came from (Pitcher2026 player_id pulls), so fetch exactly those
        # boxscores — works for MLB and MiLB feeds alike.
        _pks = sorted({str(r.get('PitchID', '')).split('_')[0]
                       for r in all_rows if r.get('PitchID')} - {''})
        print(f"  Fetching {len(_pks)} boxscores from scratch-tab game IDs...")
        for _pk in _pks:
            _bx = fetch_boxscore(_pk) or {}
            for pbox in _bx.get('pitchers', []):
                nk = _normalize_name(pbox.get('name', ''))
                if not nk:
                    continue
                if nk not in box_stats:
                    box_stats[nk] = {k: 0 for k in ('outs', 'r', 'er', 'h', 'so', 'bb', 'hr', 'tbf', 'g', 'gs')}
                for k in ('outs', 'r', 'er', 'h', 'so', 'bb', 'hr', 'tbf'):
                    box_stats[nk][k] += pbox.get(k, 0)
                box_stats[nk]['g'] += 1
                if pbox.get('is_starter'):
                    box_stats[nk]['gs'] += 1
    for t in ([] if scratch_tab else teams):
      _t_dates = sorted(team_dates.get(t, ()))
      if not _t_dates:
          continue
      if game_pk and _single_date:
          # Explicit game PK (live/in-progress): the per-date path handles it.
          day_boxes = [fetch_boxscores_for_team(_t_dates[0], t, include_live=True, game_pk=game_pk)]
      else:
          day_boxes = fetch_boxscores_for_team_dates(_t_dates, t, include_live=bool(game_pk))
      for day_box in day_boxes:
        for pname, pbox in day_box.items():
            nk = _normalize_name(pname)
            if nk not in box_stats:
                box_stats[nk] = {k: 0 for k in ('outs', 'r', 'er', 'h', 'so', 'bb', 'hr', 'tbf', 'g', 'gs')}
            for k in ('outs', 'r', 'er', 'h', 'so', 'bb', 'hr', 'tbf'):
                box_stats[nk][k] += pbox.get(k, 0)
            box_stats[nk]['g'] += 1
            if pbox.get('is_starter'):
                box_stats[nk]['gs'] += 1
    print(f"  Found boxscore data for: {', '.join(box_stats.keys())}")

    # Step 3: Look up MLB IDs and metadata
    print("\nStep 3: Looking up MLB player IDs...")
    mlb_cache = load_mlb_id_cache()

    # Step 4: Generate cards
    # Load MVN models for expected movement ellipses
    mvn_models = load_mvn_models()
    print(f"  Loaded MVN models for {len(mvn_models)} pitch-type+hand groups")

    print("\nStep 4: Generating cards...")
    generated = []

    for pitcher_name in pitcher_names:
        pitches = pitches_by_pitcher[pitcher_name]
        print(f"\n  --- {pitcher_name} ({len(pitches)} pitches) ---")

        # Per-pitcher leaderboard key. In explicit / scratch mode this is the
        # single global team label (unchanged behavior); in auto whole-season
        # mode it is this arm's own resolved label (e.g. 2TM for a traded arm).
        eff_team = pitcher_team_label.get(pitcher_name, team)

        # Get hand from pitch data
        hand = pitches[0].get('Throws', 'R') if pitches else 'R'

        # Look up MLB ID
        mlb_id = lookup_mlb_id(pitcher_name, teams[0], mlb_cache)
        print(f"  MLB ID: {mlb_id}")

        # Get age from MLB API
        meta = fetch_player_metadata(mlb_id)
        age = meta['age']
        # Use hand from sheet data (more reliable for current game)
        if not hand: hand = meta['hand']

        # Get boxscore stats
        box = box_stats.get(_normalize_name(pitcher_name))
        if not box:
            if scratch_tab:
                # Scratch-tab data has no official boxscore trail — render the
                # card from pitch-level data with zeroed line-score fields.
                box = {'outs': 0, 'g': 0, 'gs': 0, 'w': 0, 'l': 0, 'sv': 0,
                       'er': 0, 'r': 0, 'h': 0, 'hr': 0, 'so': 0, 'bb': 0,
                       'hbp': 0, 'tbf': 0}
            else:
                print(f"  WARNING: No boxscore data found for {pitcher_name}, skipping")
                continue

        ip_str = outs_to_ip_str(box['outs'])
        ip_float = box['outs'] / 3.0
        pitch_count = len(pitches_by_pitcher[pitcher_name])

        if is_multi_game:
            # Season/range stat line: G, IP, ERA, SIERA, K%, BB%, Zone%, Whiff%, GB%
            era_val = round(box['er'] * 9 / ip_float, 2) if ip_float > 0 else None

            # Compute Zone%, Whiff%, GB% from pitch data
            pp = pitches
            iz_results = [compute_iz(p) for p in pp]
            iz_count = sum(1 for r in iz_results if r is True)
            total_p = sum(1 for r in iz_results if r is not None)
            swings = [p for p in pp if p.get('Description') in SWING_DESC]
            whiffs = [p for p in pp if p.get('Description') == 'Swinging Strike']
            bip_all = [p for p in pp if p.get('BBType') and not str(p.get('BBType', '')).startswith('bunt')]
            gb_count = sum(1 for p in bip_all if p.get('BBType') == 'ground_ball')
            fb_count = sum(1 for p in bip_all if p.get('BBType') in ('fly_ball', 'popup'))
            n_bip = len(bip_all)

            zone_pct = iz_count / total_p if total_p > 0 else None
            whiff_pct = len(whiffs) / len(swings) if swings else None
            gb_pct = gb_count / n_bip if n_bip > 0 else None

            siera_val = compute_siera(box['so'], box['bb'], box['tbf'],
                                      gb_count, fb_count, box.get('gs', 0), box.get('g', 1),
                                      siera_constant)
            # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + cFIP. Computed from the
            # box like ERA and SIERA, so it honours the card's date range
            # instead of pulling a season figure off the leaderboard.
            # It earns the slot because ERA and SIERA sit far apart (r = .44);
            # FIP is between them (.73 with ERA, .63 with SIERA).
            fip_val = (((13 * box.get('hr', 0)
                         + 3 * (box.get('bb', 0) + box.get('hbp', 0))
                         - 2 * box.get('so', 0)) / ip_float) + fip_constant
                       ) if ip_float > 0 else None

            # Headline strip = context (G/GS/IP) + the two rate stats that are
            # NOT bubbles (ERA/SIERA). Everything else (K%, BB%, Zone%, Whiff%,
            # GB%) lives only in the percentile bubbles — no duplication.
            stat_headers = ['G', 'GS', 'IP', 'ERA', 'FIP', 'SIERA']
            stat_values = [
                str(box.get('g', len(game_dates_seen))),
                str(box.get('gs', 0)),
                ip_str,
                f"{era_val:.2f}" if era_val is not None else '—',
                f"{fip_val:.2f}" if fip_val is not None else '—',
                f"{siera_val:.2f}" if siera_val is not None else '—',
            ]
        else:
            # Single-game stat line — xRV is now shown per-pitch-type as
            # PitchRV/xPitchRV in the metrics table; no need to duplicate it
            # in the box-score header.
            whiff_count = sum(1 for p in pitches if p.get('Description') == 'Swinging Strike')
            # CSW% lives in the metrics table (per pitch type), not here.
            stat_headers = ['IP', 'P', 'TBF', 'R', 'ER', 'K', 'BB', 'Whiffs']
            stat_values = [ip_str, str(pitch_count), str(box['tbf']), str(box['r']),
                           str(box['er']), str(box['so']), str(box['bb']), str(whiff_count)]

        print(f"  Stat line: {' | '.join(f'{h}:{v}' for h,v in zip(stat_headers, stat_values))}")

        # Fetch headshot
        headshot = fetch_headshot(mlb_id)

        # Format display name
        parts = pitcher_name.split(', ')
        if len(parts) == 2:
            display_name = f"{parts[1]} {parts[0]}".upper()
            last_name = parts[0]
        else:
            display_name = pitcher_name.upper()
            last_name = pitcher_name

        # Percentile row for the bubble panel — match the exact (name, team)
        # FIRST, then fall back to mlbId. (name, team) must win: a pitcher with
        # rows on multiple teams (e.g. a ROC arm with an MLB call-up) shares ONE
        # mlbId, so a by-id lookup returns whichever row hashed last — often the
        # wrong team's tiny sample. Season cards only (single-game cards have no
        # season percentile context); pass None otherwise so the panel is empty.
        pctl_row = None
        scratch_pitch_lb, scratch_locplus = {}, {}
        if scratch_ctx is not None:
            # WINDOW context (single game, date range, or scratch tab): all
            # grades computed from just these pitches — plain averages of
            # per-pitch values vs the season league anchors.
            pctl_row, scratch_pitch_lb, scratch_locplus = \
                _compute_scratch_pitcher_context(pitcher_name, scratch_ctx)
            print(f"  Window context: Stuff+ {pctl_row.get('stuffScore')} | "
                  f"Loc+ {pctl_row.get('locPlus')}")
        elif is_multi_game:
            # (name, eff_team) wins over mlbId: a traded/called-up arm shares one
            # mlbId across team rows, so the by-name key targets the right row
            # (the combined 2TM/3TM row in auto whole-season mode).
            pctl_row = pctl_by_name.get((pitcher_name, eff_team)) \
                       or (pctl_by_id.get(str(int(mlb_id))) if mlb_id is not None else None)

        # Season cards: the headline strip shows hdERA/hpERA off the
        # leaderboard row (they are season-scope metrics: shrunk inputs,
        # 30+ IP z-pools — not reconstructable from a pitch window).
        # Explicit date-range cards keep the box-derived ERA/FIP/SIERA
        # trio, which honours the range.
        if (start_date is None and end_date is None
                and pctl_row is not None and is_multi_game):
            _dh = pctl_row.get('hdERA')
            _ph = pctl_row.get('hpERA')
            if _dh is not None or _ph is not None:
                stat_headers = ['G', 'GS', 'IP', 'hdERA', 'hpERA']
                stat_values = [
                    str(box.get('g', len(game_dates_seen))),
                    str(box.get('gs', 0)),
                    ip_str,
                    f"{_dh:.2f}" if _dh is not None else '—',
                    f"{_ph:.2f}" if _ph is not None else '—',
                ]

        # Build config
        config = {
            'display_name': display_name,
            'hand': hand,
            # display_team overrides the header label only; every data lookup
            # above stays keyed on eff_team (the tab/leaderboard identity).
            'team': display_team or eff_team,
            'age': age,
            'game_date': display_date,
            'stat_headers': stat_headers,
            'stat_values': stat_values,
            'headshot': headshot,
            'mlb_id': mlb_id,
            'league_avgs': league_avgs,
            'overall_avgs': overall_avgs,
            'pitcher_league_avgs': overall_avgs,
            # mvn_models stays empty for daily → is_season False → daily layout
            # (RelZ/RelX kept, no RV pair). Stuff+/Loc+/nVAA/nHAA come
            # from the computed context maps below regardless.
            'mvn_models': mvn_models if is_multi_game else {},
            'pctl_row': pctl_row,
            'pitch_locplus': (scratch_locplus if scratch_ctx is not None
                              else locplus_by_pitcher.get((pitcher_name, eff_team), {})),
            'pitch_lb': (scratch_pitch_lb if scratch_ctx is not None
                         else pitch_lb_by_pitcher.get((pitcher_name, eff_team), {})),
            'rv_mode': rv_mode,
            'pitch_qual': pitch_qual,
            # DAILY ONLY — his own season baselines (see _season_pitch_lb_for).
            'season_pitch_lb': _season_pitch_lb_for(pitcher_name, eff_team,
                                                    pitch_lb_by_pitcher),
            'opponent': _opponent_label(pitches),
        }

        # Output file — DateSlug-LastFirst format
        # Build LastFirst from pitcher_name ("Last, First" -> "LastFirst")
        if len(parts) == 2:
            name_slug = f"{parts[0]}{parts[1]}".replace(' ', '')
        else:
            name_slug = pitcher_name.replace(' ', '').replace(',', '')
        output_file = os.path.join(output_dir, f"{date_slug}-{name_slug}.png")

        # Render
        success = render_card(config, pitches, output_file)
        if success:
            print(f"  ✅ Saved: {output_file}")
            generated.append(output_file)
        else:
            print(f"  ❌ Failed to render")

    # Save MLB ID cache
    save_mlb_id_cache(mlb_cache)

    # Summary
    print(f"\n{'═'*60}")
    print(f"Generated {len(generated)} cards for {team} — {date_label}:")
    for f in generated:
        print(f"  {os.path.basename(f)}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
