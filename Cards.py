"""
Batch Pitcher Card Generator
Generates dark-themed pitcher stat cards for all pitchers on a team for a given date.

Usage:
    1. Edit the Settings block at the top of main()
    2. python3 Cards.py
"""

import argparse
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
OUTPUT_DIR = '/Users/wallyhuron/Downloads/'
METADATA_PATH = '/Users/wallyhuron/Huronalytics/data/metadata_rs.json'

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

# Stat-line coloring for multi-game cards
# Maps header → (pitcherLeagueAverages key, type, higher_is_better)
# type: 'pct' = stored as decimal (0.23), displayed as '23.0%'; 'raw' = displayed as raw number
STAT_LINE_COLOR = {
    'ERA':    ('era',      'raw', False, 1.5),
    'SIERA':  ('siera',    'raw', False, 1.5),
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

def fetch_boxscores_for_team(date_str, team_abbrev, include_live=False, game_pk=None):
    """Fetch boxscore stats for all pitchers on a team for a given date."""
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
    for gpk in game_pks:
        box = fetch_boxscore(gpk)
        if not box: continue
        for p in box['pitchers']:
            if p['team'] == team_abbrev:
                pitcher_stats[p['name']] = p
        time_module.sleep(0.1)
    return pitcher_stats


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
        ('xRV/100',       'xRv100',    'xRv100_pctl',   'dec1+'),
        ('xwOBA',         'xwOBA',     'xwOBA_pctl',    '3dec'),
        ('K%',            'kPct',      'kPct_pctl',     'pct1'),
        ('BB%',           'bbPct',     'bbPct_pctl',    'pct1'),
        ('K-BB%',         'kbbPct',    'kbbPct_pctl',   'pct1'),
    ]),
    # 2026-07-20 bubble prune (page-card parity, battery-backed): dropped
    # Z-Whiff% (r=.83 w/ Whiff%), 2K Whiff% (r=.86), xwOBAcon (pitcher-side
    # rel .26), Zone% (pred ~0), FPS% (pred ~0) — all still in page tables.
    ('SWING & MISS', [
        ('Whiff%',     'swStrPct',          'swStrPct_pctl',          'pct1'),
        ('Chase%',     'chasePct',          'chasePct_pctl',          'pct1'),
    ]),
    ('CONTACT MGMT', [
        ('xwOBAcon',   'xwOBAcon',         'xwOBAcon_pctl',         '3dec'),
        ('Hard-Hit%',  'hardHitPct',       'hardHitPct_pctl',       'pct1'),
        ('Barrel%',    'barrelPctAgainst', 'barrelPctAgainst_pctl', 'pct1'),
        ('GB%',        'gbPct',            'gbPct_pctl',            'pct1'),
    ]),
    ('COMMAND & SHAPE', [
        ('Velocity',   'fbVelo',    'fbVelo_pctl',    'mph'),
        ('Stuff+',     'stuffScore', 'stuffScore_pctl', 'int'),
        ('Loc+',       'locPlus',   'locPlus_pctl',   'int'),
        ('Pitching+',  'pitchingScore', 'pitchingScore_pctl', 'int'),
    ]),
]


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

    LABEL_W = col_w * 0.34
    VALUE_W = col_w * 0.17
    LABEL_BAR_GAP = 0.006
    BAR_VALUE_GAP = 0.008

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


def _render_single_game_panel(fig, pitches):
    """Single-game extras in the old layout (warm-paper palette): a batted-ball
    donut + per-pitch stacked bars top-left (below the stat strip), and per-hand
    usage bars on the right (below the movement plot). Location plots are placed
    on the left by render_card; the percentile-bubble panel is skipped."""
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.patches as mpatches
    TRACK = '#d8ccb4'   # warm bar track

    # Batted-ball counts per pitch type
    bb = defaultdict(lambda: {**{k: 0 for k in _BB_TYPES}, 'brl': 0})
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if not pt or p.get('Description') != 'In Play':
            continue
        bbt = str(p.get('BBType', '')).strip()
        if not bbt or bbt.startswith('bunt'):
            continue
        if bbt in _BB_TYPES:
            bb[pt][bbt] += 1
        if str(p.get('Barrel', '')).strip() == '6':
            bb[pt]['brl'] += 1
    overall = {t: sum(bb[pt][t] for pt in bb) for t in _BB_TYPES}
    total_bip = sum(overall.values())
    order = sorted([pt for pt in bb if sum(bb[pt][t] for t in _BB_TYPES) > 0],
                   key=lambda pt: -sum(bb[pt][t] for t in _BB_TYPES))

    # ── donut (top-left) ──
    ax_d = fig.add_axes([0.012, 0.625, 0.115, 0.13]); ax_d.set_facecolor(BG)
    if total_bip > 0:
        vals = [overall[t] for t in _BB_TYPES]
        ax_d.pie(vals, colors=[_BB_COLORS[t] for t in _BB_TYPES], startangle=90,
                 counterclock=False, wedgeprops=dict(width=0.32, edgecolor=BG, linewidth=2.0))
        ax_d.text(0, 0, f'{total_bip}\nBIP', ha='center', va='center', fontsize=10,
                  fontweight='bold', color=TEXT_PRIMARY, linespacing=1.1)
        ang = 90
        for t, v in zip(_BB_TYPES, vals):
            if not v:
                continue
            span = v / total_bip * 360
            mid = np.radians(ang - span / 2)
            ax_d.text(0.84 * np.cos(mid), 0.84 * np.sin(mid), str(v), ha='center',
                      va='center', fontsize=7.5, fontweight='bold',
                      color=badge_text_color(_BB_COLORS[t]))
            ang -= span
    else:
        ax_d.axis('off')

    # ── stacked bars (right of donut) ──
    _bb_btm, _bb_h = 0.625, 0.13
    ax_b = fig.add_axes([0.135, _bb_btm, 0.335, _bb_h])
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis('off')
    if order:
        n = len(order); gap = 0.04
        rh = min(0.16, (0.94 - (n - 1) * gap) / n)
        for i, pt in enumerate(order):
            y = 0.94 - i * (rh + gap) - rh / 2
            color = PITCH_COLORS.get(pt, '#999'); tcb = badge_text_color(color)
            tot_pt = sum(bb[pt][t] for t in _BB_TYPES); brl = bb[pt]['brl']
            ax_b.add_patch(FancyBboxPatch((0.02, y - rh * 0.42), 0.085, rh * 0.84,
                           boxstyle="round,pad=0.006", facecolor=color, edgecolor='none'))
            ax_b.text(0.0625, y, pt, fontsize=8, ha='center', va='center', color=tcb, fontweight='bold')
            tl, tw = 0.135, 0.50
            ax_b.add_patch(Rectangle((tl, y - rh * 0.30), tw, rh * 0.60, facecolor=TRACK, edgecolor='none'))
            left = tl
            for t in _BB_TYPES:
                cnt = bb[pt][t]
                if cnt:
                    w = tw * cnt / tot_pt
                    ax_b.add_patch(Rectangle((left, y - rh * 0.30), w, rh * 0.60,
                                   facecolor=_BB_COLORS[t], edgecolor=BG, linewidth=0.5))
                    ax_b.text(left + w / 2, y, str(cnt), ha='center', va='center', fontsize=7,
                              color=badge_text_color(_BB_COLORS[t]), fontweight='bold')
                    left += w
            lbl = str(tot_pt) + (f'  ({brl} Brl)' if brl else '')
            ax_b.text(0.66, y, lbl, fontsize=8, va='center', ha='left', color=TEXT_PRIMARY, fontweight='bold')
        # Bottom edge of the lowest bar in figure fraction, so the legend hugs
        # the stack regardless of how many pitch types are shown.
        y_last = 0.94 - (n - 1) * (rh + gap) - rh / 2
        bars_bottom_fig = _bb_btm + (y_last - rh * 0.30) * _bb_h
    else:
        bars_bottom_fig = _bb_btm + _bb_h

    # ── batted-ball legend (right below the stacked bars) ──
    if total_bip > 0:
        axl = fig.add_axes([0, 0, 1, 1]); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
        axl.axis('off'); axl.set_zorder(6)
        pat = [mpatches.Patch(color=_BB_COLORS[t], label=f'{_BB_LABELS[t]} ({overall[t]})')
               for t in _BB_TYPES if overall[t] > 0]
        axl.legend(handles=pat, loc='upper left', bbox_to_anchor=(0.135, bars_bottom_fig - 0.012), ncol=2,
                   fontsize=7.5, frameon=False, labelcolor=TEXT_MUTED, handlelength=1.0, columnspacing=0.8)

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
        ax.text(0.5, 0.99, title, fontsize=12, fontweight='bold', ha='center', va='top',
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
        # `is None` (not `or`): a numeric 0.0 adjusted value must not fall
        # through to raw. Safe today only because cache values are strings.
        hb = p.get('xHorzBrk') if p.get('xHorzBrk') is not None else p.get('HorzBrk')
        ivb = p.get('xIndVrtBrk') if p.get('xIndVrtBrk') is not None else p.get('IndVertBrk')
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
    ax_main.text(text_x, photo_top-1.5, config['game_date'], fontsize=24, fontfamily='IBM Plex Sans', color=ACCENT, va='top')

    # Stat line — season cards widen the 5-cell strip so it spans the bubble
    # column beneath it. Single-game cards have no bubble column, so they use
    # a much tighter column so the (now 8-cell) strip doesn't run too wide.
    is_season_strip = bool(config.get('mvn_models'))
    col_w = 1.25 if is_season_strip else 0.72
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

    # ── FB velo-by-start sparkline — season cards only, thin strip directly
    # under the boxscore line. Fastball (FF; SI fallback when no FF) average
    # velo per game date: muted dots on a thin line, dotted season-average
    # reference, "last · avg · max" annotation right-aligned above, first/mid/
    # last date labels below. Skips gracefully with fewer than 3 starts. Lives
    # in the 0.7in of extra card height, so nothing below has to yield.
    if config.get('mvn_models'):
        _fb_type = 'FF' if any(p.get('Pitch Type') == 'FF' for p in pitches) else 'SI'
        _velo_by_start = defaultdict(list)
        for p in pitches:
            if p.get('Pitch Type') != _fb_type:
                continue
            _v = sf(p.get('Velocity')); _gd = p.get('Game Date')
            if _v is not None and _gd:
                _velo_by_start[_gd].append(_v)
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
            ax_main.text(photo_left, _label_y, 'FB VELO BY START', fontsize=8.5,
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
                     fontsize=7, color=TEXT_MUTED, fontfamily='IBM Plex Sans', va='bottom')
    ax_plot.text(0.02, 0.005, 'Min. 6 pitches for ellipse', transform=ax_plot.transAxes,
                 fontsize=6.5, color=TEXT_FAINT, fontfamily='IBM Plex Sans', va='bottom', fontstyle='italic')

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
                    if is_season:
                        # Outline-only ellipse + center dot at the mean.
                        ax.add_patch(Ellipse(
                            (mx, my),
                            2 * 1.0 * np.sqrt(vals[1]), 2 * 1.0 * np.sqrt(vals[0]),
                            angle=angle, fill=False,
                            edgecolor=_rgba(_pc, 0.95),
                            linewidth=2.2, zorder=1
                        ))
                        ax.scatter([mx], [my], c=_pc, s=32, alpha=1.0,
                                   edgecolors=TEXT_PRIMARY, linewidths=0.6, zorder=4)
                    else:
                        # Faint fill + a bold darker edge so the ellipse is
                        # defined even when the fill is light. Separate
                        # fill/edge alphas, so no single `alpha=`.
                        ax.add_patch(Ellipse(
                            (mx, my),
                            2 * 1.0 * np.sqrt(vals[1]), 2 * 1.0 * np.sqrt(vals[0]),
                            angle=angle, fill=True,
                            facecolor=_rgba(_pc, 0.28),
                            edgecolor=_rgba(_darken(_pc, 0.6), 0.9),
                            linewidth=1.3, zorder=1
                        ))
        # Pitch dots and W/B annotations — single-game cards only. Season
        # panels show just the outline ellipses, per-type center dots,
        # zone/plate, and the legend panel (no per-pitch marks).
        if not is_season:
            for pt in PITCH_ORDER:
                if pt not in locations[hand]: continue
                color = PITCH_COLORS[pt]
                for px_val, pz_val, desc, barrel_flag in locations[hand][pt]:
                    if desc == 'Swinging Strike':
                        ax.text(px_val, pz_val, 'W', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
                    elif barrel_flag:
                        ax.text(px_val, pz_val, 'B', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
                    else:
                        ax.scatter([px_val], [pz_val], c=[color], s=55, alpha=1.0, edgecolors='none', zorder=3)

    ax_loc_l = fig.add_axes([LOC_L_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
    ax_loc_r = fig.add_axes([LOC_R_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
    draw_zone(ax_loc_l, 'R'); draw_zone(ax_loc_r, 'L')
    fig.text(LOC_L_X+LOC_W/2, LOC_TITLE_Y, 'VS RHH', fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='IBM Plex Sans', ha='center', va='center')
    fig.text(LOC_R_X+LOC_W/2, LOC_TITLE_Y, 'VS LHH', fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='IBM Plex Sans', ha='center', va='center')

    # Footnote — single-game only (old layout): W/B legend + ellipse minimum
    # stacked to the right of the location plots. Season panels carry no
    # footnote (the 10-pitch ellipse minimum still applies, just unlabeled).
    if not is_season_loc:
        _wx = LOC_R_X + LOC_W + 0.012
        for _dy, _txt in [(0.055, 'W = Whiff'), (0.033, 'B = Barrel'),
                          (0.011, f'Min. {zone_ellipse_min} pitches for ellipse')]:
            fig.text(_wx, LOC_BOTTOM + _dy, _txt, fontsize=8, color=TEXT_MUTED,
                     va='bottom', ha='left', fontfamily='IBM Plex Sans', fontweight='bold')

    # ── Left column: season cards get the percentile bubble panel; single-game
    # cards (no season pool) get the batted-ball donut + stacked bars + usage. ──
    p_row = config.get('pctl_row') or {}
    if config.get('mvn_models'):
        if p_row:
            bubble_cols = _bubble_columns_for(config, p_row)
            # Classic-frame inches (0.790/0.235 of the 17.5in card) so the
            # rail's physical geometry is untouched by the taller figure; the
            # sparkline lives entirely in the extra height above the rail.
            _render_percentile_bubbles(fig, p_row,
                                       grid_left=0.015, grid_right=0.405,
                                       grid_top=(0.790 * FIG_H) / fig_h,
                                       grid_bot=(0.235 * FIG_H) / fig_h,
                                       columns=bubble_cols)
    else:
        _render_single_game_panel(fig, pitches)

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
    # Low-model-support daggers (season cards only): the pitch sits far from
    # the Stuff+ model's training data (worst ~1.5% of units league-wide), so
    # its per-type Stuff+ is an extrapolation — marked with a superscript †.
    stuff_lowsup_by_pt = {pt: bool(d.get('stuffScore_lowSupport')) for pt, d in pitch_lb.items()}
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
    _pt_qual_min = config.get('pitch_qual') or PITCH_QUAL_MIN

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
        ivbs=[v for v in (sf(p.get('xIndVrtBrk') if p.get('xIndVrtBrk') is not None else p.get('IndVertBrk')) for p in pp) if v is not None]
        hbs=[v for v in (sf(p.get('xHorzBrk') if p.get('xHorzBrk') is not None else p.get('HorzBrk')) for p in pp) if v is not None]
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
        # Qualification gate: below the pitch-type minimum the RV cells are
        # noise (rates especially, but totals blank too so cards read
        # consistently). Single-game cumulative stays ungated as before.
        if n < _pt_qual_min:
            prv_100 = xrv_100 = None
            if is_season:
                prv_cum = xrv_cum = None
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
        pt_name='Fastball' if pt=='FF' else PITCH_NAMES.get(pt,pt)
        _nvaa = nvaa_by_pt.get(pt)
        _nhaa = nhaa_by_pt.get(pt)
        row=[pt_name,str(n),("< 1%" if 0 < n/tc*100 < 1 else f"{n/tc*100:.1f}%"),
            f"{sum(velos)/len(velos):.1f}" if velos else '—',f"{max(velos):.1f}" if velos else '—',
            f"{int(sum(spins)/len(spins))}" if spins else '—',
            f'{sum(ivbs)/len(ivbs):.1f}"' if ivbs else '—',f'{sum(hbs)/len(hbs):.1f}"' if hbs else '—',
            f"{_nvaa:.2f}" if _nvaa is not None else '—',
            f"{_nhaa:.2f}" if _nhaa is not None else '—',
            fmt_fi(sum(relzs)/len(relzs)) if relzs else '—',fmt_fi(sum(relxs)/len(relxs)) if relxs else '—',
            fmt_fi(sum(exts)/len(exts)) if exts else '—',
            f"{sum(armangles)/len(armangles):.1f}°" if armangles else '—',
            ((f"{int(round(stuff_by_pt[pt]))}" +
              ('†' if is_season and stuff_lowsup_by_pt.get(pt) else ''))
             if stuff_by_pt.get(pt) is not None else '—'),
            (f"{int(round(locplus_by_pt[pt]))}" if locplus_by_pt.get(pt) is not None else '—'),
            (f"{int(round(pitching_by_pt[pt]))}" if pitching_by_pt.get(pt) is not None else '—'),
            f"{iz_n/n*100:.1f}%" if n else '—',
            f"{len(whiffs)/len(swings)*100:.1f}%" if swings else '—',
            f"{chase_pct*100:.1f}%" if chase_pct is not None else '—',
            f"{xwobacon:.3f}".replace('0.', '.') if xwobacon is not None else '—']
        for _h in rv_cols:
            _v = _rvmap.get(_h)
            row.append(str(_v) if _v is not None else '—')
        pitch_stats.append((pt, row))

    t_sw=[p for p in pitches if p.get('Description') in SWING_DESC]
    t_wh=[p for p in pitches if p.get('Description')=='Swinging Strike']
    t_iz=sum(1 for p in pitches if compute_iz(p)==True)
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
        f"{len(t_wh)/len(t_sw)*100:.1f}%" if t_sw else '—',
        f"{t_chase*100:.1f}%" if t_chase is not None else '—',
        f"{t_xwobacon:.3f}".replace('0.', '.') if t_xwobacon is not None else '—']
    for _h in rv_cols:
        _v = _trvmap.get(_h)
        total_row.append(str(_v) if _v is not None else '—')

    # Source-data presence check — RV needs RunExp on at least one pitch.
    has_pitchrv_data = any(p.get('RunExp') is not None and str(p.get('RunExp','')).strip() != '' for p in pitches)

    all_col_headers=['Pitch Type','Count','Usage','Avg Velo','Max Velo','Spin Rate','IVB','HB','nVAA','nHAA','RelZ','RelX','Ext','Arm Angle','Stuff+','Loc+','Pitching+','Zone%','Whiff%','Chase%','xwOBAcon'] + rv_cols
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
                        'Zone%','Whiff%','Chase%','Stuff+','Loc+','Pitching+','xwOBAcon'] + rv_cols
        _perm = [all_col_headers.index(h) for h in _daily_order]
        all_col_headers = _daily_order
        all_cell_data = [[row[i] for i in _perm] for row in all_cell_data]

    # Columns to force-exclude based on data availability and card type.
    force_exclude = set()
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
            val_str = cell_data[r - 1][xwc_col_idx]
            tinted = _pitcher_stat_cell_color(val_str, la, 0.05, False, row_bg, False)
            if tinted:
                table.get_celld()[(r, xwc_col_idx)].set_facecolor(tinted)

    # xPitchRV coloring — higher is better for pitcher, centered at 0. The
    # per-100 rate gets scale 2.0; the cumulative column spans wider (a full
    # season of one pitch type), so it uses scale 3.0.
    RV_COL_NAMES = ('PitchRV', 'xPitchRV', 'PitchRV/100', 'xPitchRV/100')
    for c, col_name in enumerate(col_headers):
        if col_name not in RV_COL_NAMES:
            continue
        rv_scale = 2.0 if col_name.endswith('/100') else 3.0
        for r in range(1, len(cell_data) + 1):
            row_bg = DARKER if r == len(cell_data) else (DARK_CELL if r % 2 == 1 else ALT_ROW_BG)
            val_str = cell_data[r - 1][c]
            tinted = _pitcher_stat_cell_color(val_str, 0.0, rv_scale, True, row_bg, False)
            if tinted:
                table.get_celld()[(r, c)].set_facecolor(tinted)

    # Loc+ coloring — index centered at 100 (group avg), higher is better,
    # scale 10 (≈1 SD). Matches the Loc+ bubble's blue→red direction.
    if 'Loc+' in col_headers:
        lp_col_idx = col_headers.index('Loc+')
        for r in range(1, len(cell_data) + 1):
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

    # nVAA coloring — FF and SI only (per spec). nVAA_pctl is already directional
    # (FF: flatter/closer-to-zero better; SI: steeper better), computed vs MLB.
    if 'nVAA' in col_headers:
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
                    'Per-pitch Stuff+ graded vs same pitch type (100 = average for that type)\n'
                    'Overall Stuff+ = pitch-weighted average of per-pitch grades')
        if any(stuff_lowsup_by_pt.get(_pt) and stuff_by_pt.get(_pt) is not None
               for _pt, _ in pitch_stats):
            _sp_note += '\n† = low model support (unusual pitch profile, score less certain)'
        fig.text(_sp_x, b - _below_off, _sp_note,
                 fontsize=8, color=TEXT_MUTED, va='top', ha='left', fontfamily='IBM Plex Sans', fontweight='bold', linespacing=1.5)

    # Watermark — bottom-left of the card, just below the table border.
    fig.text(l, b - _below_off, 'Huronalytics', fontsize=9, ha='left', va='top', color=TEXT_FAINT, style='italic', fontfamily='IBM Plex Sans')
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
#   Stuff+  — stuff_plus_v11 bundle (full model when the pitcher has ArmAngle
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
_SCRATCH_INVERT_PITCHER = {'bbPct', 'xwOBA', 'xwOBAcon', 'hardHitPct', 'barrelPctAgainst'}
# Bubble-panel stats we compute and rank (everything BUBBLE_COLUMNS reads).
_SCRATCH_POOL_STATS = ['xRv100', 'xwOBA', 'kPct', 'bbPct', 'kbbPct',
                       'swStrPct', 'chasePct', 'izWhiffPct', 'twoStrikeWhiffPct',
                       'xwOBAcon', 'hardHitPct', 'barrelPctAgainst', 'gbPct',
                       'fbVelo', 'stuffScore', 'locPlus', 'pitchingScore',
                       'izPct', 'fpsPct']


def _pitching_blend(stuff, loc):
    """Stuff+/Loc+ blend in z units — the trainer's _blend (single source of
    truth for the 70/30 weight; falls back to 0.70 if the import fails)."""
    _sv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stuff_plus_v11')
    if _sv_dir not in sys.path:
        sys.path.insert(0, _sv_dir)
    try:
        from train_stuff_v11 import _blend
        return _blend(stuff, loc)
    except Exception:
        return 0.80 * (stuff - 100.0) / 10.0 + 0.20 * (loc - 100.0) / 10.0


def _pitching_scale(rows, min_pitches=25):
    """League (mu, sd) of the blend from leaderboard pool rows with >=25
    pitches — mirrors train_stuff_v11._inject_pitching's pool convention."""
    bs = [_pitching_blend(r['stuffScore'], r['locPlus']) for r in rows
          if r.get('stuffScore') is not None and r.get('locPlus') is not None
          and (r.get('count') or 0) >= min_pitches]
    if len(bs) < 5:
        return None
    mu = sum(bs) / len(bs)
    sd = (sum((b - mu) ** 2 for b in bs) / len(bs)) ** 0.5
    return (mu, sd) if sd > 1e-9 else None


def _pitching_score(stuff, loc, scale=None):
    # COHERENT CANON (2026-07-18): Pitching+ = 0.8*Stuff+ + 0.2*Loc+ exactly
    # (weight re-derived LOSO across 2021-2025 on 2026-07-25; see
    # train_stuff_v11.PITCHING_W_STUFF) —
    # no restandardization, no clip. `scale` is accepted for call-site compat
    # and ignored.
    if stuff is None or loc is None:
        return None
    return round(0.8 * stuff + 0.2 * loc, 1)


def _normalize_scratch_pitch(row):
    """Sheet row → pipeline-format pitch dict. Mirrors
    pipeline_fetch.read_pitches_from_sheet: blanks → None, adjusted-movement
    fallback, Barrel recompute fallback, plus the InZone recompute the
    pipeline applies in process_data (the scratch tab has no InZone column)."""
    from pipeline_utils import compute_in_zone, is_barrel
    p = {k: (None if v == '' else v) for k, v in row.items()}
    if p.get('xIndVrtBrk') is None and p.get('IndVertBrk') is not None:
        p['xIndVrtBrk'] = p['IndVertBrk']
    if p.get('xHorzBrk') is None and p.get('HorzBrk') is not None:
        p['xHorzBrk'] = p['HorzBrk']
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
# use train_stuff_v11.K_SHRINK (=100) because they estimate a stable between-
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
    """Stuff+ v11 for scratch pitchers. Returns ({pitcher: overall},
    {(pitcher, pitch_type): score}). Full model when the pitcher has any
    ArmAngle data (build_df fills gaps with his own average); otherwise the
    no-arm companion model anchored to its MLB (no-arm) scales — exactly the
    ROC path in train_stuff_v11.main(). k_shrink overrides the season K_SHRINK
    (used lightly for daily cards)."""
    import pickle as _pickle
    _sv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stuff_plus_v11')
    if _sv_dir not in sys.path:
        sys.path.insert(0, _sv_dir)
    import train_stuff_v11 as _sv

    with open(os.path.join(_sv_dir, 'stuff_models_v11.pkl'), 'rb') as f:
        bundle = _pickle.load(f)

    all_pitches = [p for pl in norm_by_pitcher.values() for p in pl]
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


_MLB_PICKLE_CACHE = None   # module-level: load the 382k-pitch pickle once per process


def _build_scratch_league_context(norm_by_pitcher, stuff_k_shrink=None):
    """Heavy one-time setup for scratch-tab / daily cards: MLB pickle baselines
    (Loc+ surfaces + norm pool, xRV count anchoring), Stuff+ v11 scoring,
    leaderboard percentile pools, nVAA/nHAA regressions. stuff_k_shrink is
    passed through to Stuff+ scoring (light for daily cards)."""
    global _MLB_PICKLE_CACHE
    import pickle as _pickle
    from pipeline_compute import build_bip_count_means
    from pipeline_sdplus import build_bip_count_offsets
    from pipeline_locplus import compute_loc_plus

    t0 = time_module.time()
    ctx = {'norm_by_pitcher': norm_by_pitcher}

    if _MLB_PICKLE_CACHE is None:
        print("  [ctx] Loading MLB pitch pickle for league baselines...")
        with open(os.path.join(os.path.dirname(METADATA_PATH), 'all_pitches_rs_cache.pkl'), 'rb') as f:
            _all = _pickle.load(f)
        _MLB_PICKLE_CACHE = [p for p in _all if p.get('_source') == 'MLB']
    mlb = _MLB_PICKLE_CACHE
    print(f"  [ctx] {len(mlb)} MLB pitches ready ({time_module.time()-t0:.0f}s)")

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

    # Stuff+ v11
    print("  [scratch] Scoring Stuff+ v11...")
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
            _prows = _scratch_mlb_pool_rows(json.load(f))
        for s in _SCRATCH_POOL_STATS:
            vals = [r.get(s) for r in _prows]
            ctx['pitcher_pools'][s] = sorted(v for v in vals if v is not None)
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
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _m = json.load(f)
        ctx['vaa_reg'] = _m.get('vaaRegressions') or {}
        ctx['haa_reg'] = _m.get('haaRegressions') or {}

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
            P = int(round(0.7 * S + 0.3 * L))
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
    team            = "ROC"
    start_date      = None    # Set to None for full season
    end_date        = None              # Set to a date for date range, or None for single day
    filter_pitchers = "Williams, Trevor"                 # Semicolon-separated "Last, First" names, or "" for all
    game_pk         = ""                 # Optional game PK for live/in-progress games
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
                        help=f'Min pitches for a pitch type\'s RV cells (default {PITCH_QUAL_MIN})')
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
    # Auto whole-season mode: --pitchers given but no --team. Resolve each
    # pitcher's stint team(s) from the leaderboard so multi-team arms combine
    # their full season automatically (see _resolve_pitcher_teams). Per-pitcher
    # lookups then use each arm's own leaderboard label via pitcher_team_label.
    auto_team = (args.team is None and not scratch_tab and bool(filter_pitchers))
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
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        league_avgs = meta.get('leagueAverages', {})
        overall_avgs = meta.get('pitcherLeagueAverages', {})
        siera_constant = meta.get('sieraConstant', 5.77)

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
                        'stuffScore_lowSupport': _r.get('stuffScore_lowSupport'),
                        'pitchingScore': _r.get('pitchingScore'),
                        'xrvoe100': _r.get('xrvoe100'),
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
      for gd in sorted(team_dates.get(t, ())):
        print(f"  Fetching boxscores for {gd} ({t})...")
        day_box = fetch_boxscores_for_team(gd, t, include_live=bool(game_pk), game_pk=game_pk if _single_date else None)
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

            # Headline strip = context (G/GS/IP) + the two rate stats that are
            # NOT bubbles (ERA/SIERA). Everything else (K%, BB%, Zone%, Whiff%,
            # GB%) lives only in the percentile bubbles — no duplication.
            stat_headers = ['G', 'GS', 'IP', 'ERA', 'SIERA']
            stat_values = [
                str(box.get('g', len(game_dates_seen))),
                str(box.get('gs', 0)),
                ip_str,
                f"{era_val:.2f}" if era_val is not None else '—',
                f"{siera_val:.2f}" if siera_val is not None else '—',
            ]
        else:
            # Single-game stat line — xRV is now shown per-pitch-type as
            # PitchRV/xPitchRV in the metrics table; no need to duplicate it
            # in the box-score header.
            whiff_count = sum(1 for p in pitches if p.get('Description') == 'Swinging Strike')
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

        # Build config
        config = {
            'display_name': display_name,
            'hand': hand,
            'team': eff_team,
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
            # (RelZ/RelX kept, no RV pair, † off). Stuff+/Loc+/nVAA/nHAA come
            # from the computed context maps below regardless.
            'mvn_models': mvn_models if is_multi_game else {},
            'pctl_row': pctl_row,
            'pitch_locplus': (scratch_locplus if scratch_ctx is not None
                              else locplus_by_pitcher.get((pitcher_name, eff_team), {})),
            'pitch_lb': (scratch_pitch_lb if scratch_ctx is not None
                         else pitch_lb_by_pitcher.get((pitcher_name, eff_team), {})),
            'rv_mode': rv_mode,
            'pitch_qual': pitch_qual,
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
