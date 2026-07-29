"""
Pitcher Card Generator — v30 (Dark + Data-viz Hybrid)
Generates a dark-themed visual stat card for a pitcher's game performance.
Uses Google Sheets data + MLB headshot API.

Usage:
    cd ~/Huronalytics && python3 generate_pitcher_card.py

To modify for a different pitcher/game, change the CONFIGURATION section below.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from PIL import Image
from io import BytesIO
import urllib.request
import os
import numpy as np
from collections import defaultdict
from math import atan2, sin, cos
import gspread
from sheets_append import _workbook_id_for_team

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — change these for different pitchers/games
# ═══════════════════════════════════════════════════════════════
PITCHER_NAME = 'Irvin, Jake'
WORKSHEET_NAME = 'WSH'   # team tab; its division workbook is resolved automatically
PLAYER_ID = '663623'
DISPLAY_NAME = 'JAKE IRVIN'
HAND = 'RHP'
TEAM = 'WSH'
AGE = '29'
GAME_DATE = 'March 29, 2026'
STAT_HEADERS = ['IP', 'TBF', 'R', 'ER', 'H', 'K', 'BB', 'HR']
STAT_VALUES = ['5.0', '20', '2', '2', '3', '7', '1', '2']
GAME_DATE_FILTER = '2026-03-29'  # YYYY-MM-DD format matching sheet's Game Date column
OUTPUT_FILE = '/Users/wallyhuron/Downloads/03292026-IrvinJake.png'

# ═══════════════════════════════════════════════════════════════
# PITCH CONSTANTS
# ═══════════════════════════════════════════════════════════════
PITCH_COLORS = {
    'FF': '#0072B2', 'SI': '#FFD700', 'FC': '#8B5A2B',
    'SL': '#D55E00', 'ST': '#56B4E9', 'CU': '#332288', 'SV': '#882255',
    'CH': '#009E73', 'FS': '#CC79A7', 'KN': '#AAAAAA'
}
PITCH_NAMES = {
    'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter',
    'SL': 'Slider', 'ST': 'Sweeper', 'CU': 'Curveball', 'SV': 'Slurve',
    'CH': 'Changeup', 'FS': 'Splitter', 'KN': 'Knuckleball'
}
PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS', 'KN']
# Bunts are not swings (matches Cards.SWING_DESC / pipeline_utils.SWING_DESCRIPTIONS)
# so Whiff%/Chase%/Swing% match the leaderboard. STRIKE_DESC still counts foul bunt.
SWING_DESC = ['Swinging Strike', 'Foul', 'In Play']
STRIKE_DESC = ['Called Strike', 'Swinging Strike', 'Foul', 'Foul Bunt', 'In Play']

# Batted ball type colors (for distribution chart)
BB_COLORS = {
    'ground_ball': '#00d4ff',   # cyan (matches card accent)
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

# ═══════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ═══════════════════════════════════════════════════════════════
TABLE_LEFT_FIG = 0.01
TABLE_RIGHT_FIG = 0.99
PLOT_LEFT = 0.585       # ylabel left edge aligns with VS RHH badge left
PLOT_RIGHT = 0.99       # spine at PitchRV right edge (no aspect='equal')
USAGE_SHIFT = 0.18      # right-shift for pitch usage content within axes
DIVIDER_COL = 14        # Strike% column index (divider between metrics and rate stats)
PLATE_HALF = 17 / 12 / 2  # half plate width in feet

FIG_W, FIG_H = 16, 15
DPI = 100
SAVE_DPI = 150

# Theme colors
BG = '#141619'
ACCENT = '#00d4ff'
DARK_CELL = '#1e2127'
DARKER = '#0d0f12'


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def sf(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except:
        return None


def avg_tilt(tilts):
    valid = [t for t in tilts if t and t != '']
    if not valid:
        return '—'
    sins, coss = [], []
    for t in valid:
        parts = str(t).split(':')
        if len(parts) != 2:
            continue
        h, m = int(parts[0]), int(parts[1])
        if h == 12:
            h = 0
        a = (h * 60 + m) / 720 * 2 * 3.14159
        sins.append(sin(a))
        coss.append(cos(a))
    if not sins:
        return '—'
    am = (atan2(sum(sins) / len(sins), sum(coss) / len(coss)) * 720 / (2 * 3.14159)) % 720
    h, m = int(am // 60), int(am % 60)
    if h == 0:
        h = 12
    return f'{h}:{m:02d}'


def fmt_fi(v):
    if v is None:
        return '—'
    neg = v < 0
    av = abs(v)
    ft = int(av)
    inc = round((av - ft) * 12)
    if inc == 12:
        ft += 1
        inc = 0
    s = f"{ft}'{inc}\""
    return f"-{s}" if neg else s


def compute_iz(p):
    px, pz, st, sb = sf(p.get('PlateX')), sf(p.get('PlateZ')), sf(p.get('SzTop')), sf(p.get('SzBot'))
    if any(v is None for v in [px, pz, st, sb]):
        return None
    return abs(px) <= 0.83 and pz >= sb - 0.121 and pz <= st + 0.121


def luminance(hc):
    r, g, b = int(hc[1:3], 16) / 255, int(hc[3:5], 16) / 255, int(hc[5:7], 16) / 255
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def badge_text_color(hc):
    return 'black' if luminance(hc) > 0.25 else 'white'


def is_barrel(ev, la):
    """Statcast barrel definition (LA in [8,50], EV>=98, EV*1.5-LA>=117, EV+LA>=124)."""
    if ev is None or la is None:
        return False
    return (la >= 8 and la <= 50 and ev >= 98 and
            ev * 1.5 - la >= 117 and
            ev + la >= 124)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
gc = gspread.service_account()
sheet_key = _workbook_id_for_team(WORKSHEET_NAME)
fried = [r for r in gc.open_by_key(sheet_key).worksheet(WORKSHEET_NAME).get_all_records()
         if r.get('Pitcher') == PITCHER_NAME and r.get('Game Date') == GAME_DATE_FILTER]

print("Fetching headshot...")
headshot_raw = Image.open(BytesIO(urllib.request.urlopen(urllib.request.Request(
    f'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{PLAYER_ID}/headshot/67/current',
    headers={'User-Agent': 'Mozilla/5.0'})).read()))
ha = np.array(headshot_raw.convert('RGBA'))
gm = ((np.abs(ha[:, :, 0].astype(int) - ha[:, :, 1].astype(int)) < 15) &
      (np.abs(ha[:, :, 1].astype(int) - ha[:, :, 2].astype(int)) < 15) &
      (ha[:, :, 0] > 170) & (ha[:, :, 0] < 230))
ha[gm] = [255, 255, 255, 255]
headshot = Image.fromarray(ha)


# ═══════════════════════════════════════════════════════════════
# COMPUTE DATA
# ═══════════════════════════════════════════════════════════════
print("Computing stats...")
locations = {'L': defaultdict(list), 'R': defaultdict(list)}
sz_tops = []
sz_bots = []
groups = defaultdict(list)

for p in fried:
    pt = p.get('Pitch Type', '')
    hb, ivb = p.get('HorzBrk'), p.get('IndVertBrk')
    if pt and hb != '' and ivb != '':
        try:
            groups[pt].append((float(hb), float(ivb)))
        except:
            pass
    bh = p.get('Bats', '')
    px, pz = p.get('PlateX'), p.get('PlateZ')
    szt, szb = p.get('SzTop'), p.get('SzBot')
    if bh in ('L', 'R') and pt and px != '' and pz != '':
        try:
            desc = p.get('Description', '')
            is_b = str(p.get('Barrel', '')).strip() == '6'
            locations[bh][pt].append((float(px), float(pz), desc, is_b))
        except:
            pass
    if szt != '' and szb != '':
        try:
            sz_tops.append(float(szt))
            sz_bots.append(float(szb))
        except:
            pass

# Guard the empty case (e.g. a name/date CONFIG typo matches no rows): np.mean([])
# is NaN and would draw the strike zone at NaN coordinates. Mirror Cards.py.
avg_top = np.mean(sz_tops) if sz_tops else 3.5
avg_bot = np.mean(sz_bots) if sz_bots else 1.5
sorted_types = [pt for pt in PITCH_ORDER if pt in groups]

# Batted ball distribution per pitch type
bb_by_pitch = defaultdict(lambda: {'ground_ball': 0, 'line_drive': 0, 'fly_ball': 0, 'popup': 0, 'hh': 0, 'brl': 0})
for p in fried:
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


# ═══════════════════════════════════════════════════════════════
# FIGURE SETUP
# ═══════════════════════════════════════════════════════════════
print("Rendering card...")
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor(BG)
ax_main = fig.add_axes([0, 0, 1, 1])
ax_main.set_xlim(0, FIG_W)
ax_main.set_ylim(0, FIG_H)
ax_main.axis('off')
ax_main.set_facecolor(BG)


# ═══════════════════════════════════════════════════════════════
# COLORED PITCH STRIPE BANNER
# ═══════════════════════════════════════════════════════════════
photo_left = TABLE_LEFT_FIG * FIG_W
stripe_bottom = 14.90  # small gap above photo_top (14.85)
stripe_height = 0.22
stripe_x = photo_left  # left-align with photo
total_w = FIG_W * TABLE_RIGHT_FIG - photo_left  # from photo_left to right edge
# Order by usage (descending), PITCH_ORDER as tiebreaker; equal widths
stripe_counts = {pt: sum(1 for p in fried if p.get('Pitch Type') == pt) for pt in sorted_types}
stripe_order = sorted(sorted_types,
    key=lambda pt: (-stripe_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999))
for pt in stripe_order:
    w = total_w / len(sorted_types)
    ax_main.add_patch(Rectangle((stripe_x, stripe_bottom), w, stripe_height,
        facecolor=PITCH_COLORS.get(pt, '#999'), edgecolor='none', zorder=6))
    stripe_x += w


# ═══════════════════════════════════════════════════════════════
# PLAYER INFO PANEL
# ═══════════════════════════════════════════════════════════════
photo_w = 1.4
photo_h = photo_w * headshot.size[1] / headshot.size[0]
photo_top = 14.85
photo_bottom = photo_top - photo_h

ax_main.imshow(np.array(headshot),
    extent=[photo_left, photo_left + photo_w, photo_bottom, photo_top],
    aspect='auto', zorder=2, interpolation='antialiased')
ax_main.add_patch(Rectangle((photo_left, photo_bottom), photo_w, photo_h,
    fill=False, edgecolor=ACCENT, linewidth=1.5, alpha=0.5, zorder=3))

photo_right = photo_left + photo_w
text_x = photo_right + 0.3
ax_main.text(text_x, photo_top - 0.1, DISPLAY_NAME, fontsize=32,
    fontfamily='DIN Condensed', color='white', va='top', fontweight='bold')
ax_main.text(text_x, photo_top - 0.85, f'{HAND}  |  {TEAM}  |  Age: {AGE}',
    fontsize=12, fontfamily='Avenir Next', color='#888', va='top')
ax_main.text(text_x, photo_top - 1.5, GAME_DATE, fontsize=24,
    fontfamily='DIN Condensed', color=ACCENT, va='top')


# ═══════════════════════════════════════════════════════════════
# STAT LINE TABLE
# ═══════════════════════════════════════════════════════════════
col_w = 0.6
cell_h = 0.42
stat_y_header = photo_bottom - 0.5
stat_y_value = stat_y_header - cell_h

for i in range(len(STAT_HEADERS)):
    x = photo_left + i * col_w
    ax_main.add_patch(Rectangle((x, stat_y_header), col_w, cell_h,
        facecolor=DARKER, edgecolor='#333840', linewidth=0.8))
    ax_main.text(x + col_w / 2, stat_y_header + cell_h / 2, STAT_HEADERS[i],
        fontsize=10, ha='center', va='center', color=ACCENT, fontweight='bold',
        fontfamily='Avenir Next')
    ax_main.add_patch(Rectangle((x, stat_y_value), col_w, cell_h,
        facecolor=DARK_CELL, edgecolor='#333840', linewidth=0.8))
    ax_main.text(x + col_w / 2, stat_y_value + cell_h / 2, STAT_VALUES[i],
        fontsize=12, ha='center', va='center', color='white', fontweight='bold',
        fontfamily='Avenir Next')

# Cyan border around stat line
ax_main.add_patch(Rectangle((photo_left, stat_y_value),
    len(STAT_HEADERS) * col_w, stat_y_header + cell_h - stat_y_value,
    fill=False, edgecolor=ACCENT, linewidth=2, zorder=5))


# ═══════════════════════════════════════════════════════════════
# PITCH MOVEMENT PLOT
# ═══════════════════════════════════════════════════════════════
ax_plot = fig.add_axes([PLOT_LEFT, 0.58, PLOT_RIGHT - PLOT_LEFT, 0.40])
ax_plot.set_xlim(-25, 25)
ax_plot.set_ylim(-25, 25)
# No aspect='equal' — lets plot fill axes, spine = axes edge
ax_plot.axhline(y=0, color='#333840', linestyle='--', linewidth=0.5)
ax_plot.axvline(x=0, color='#333840', linestyle='--', linewidth=0.5)
ax_plot.set_xlabel('Horizontal Break (in)', fontsize=10, color='#ccc',
    fontweight='bold', fontfamily='Avenir Next')
ax_plot.set_ylabel('Induced Vertical Break (in)', fontsize=10, color='#ccc',
    fontweight='bold', fontfamily='Avenir Next')
ax_plot.tick_params(labelsize=8, colors='#999')
ax_plot.set_xticks(range(-25, 26, 5))
ax_plot.set_yticks(range(-25, 26, 5))
ax_plot.grid(True, alpha=0.2, color='#333840')
ax_plot.set_facecolor('#1a1d21')
for spine in ax_plot.spines.values():
    spine.set_color('#444')

for pt in PITCH_ORDER:
    if pt not in groups:
        continue
    xs, ys = zip(*groups[pt])
    color = PITCH_COLORS[pt]
    ax_plot.scatter(xs, ys, c=color, s=65, alpha=1.0, edgecolors='none', zorder=3)
    if len(groups[pt]) >= 4:
        cov = np.cov(xs, ys)
        vals, vecs = np.linalg.eigh(cov)
        ax_plot.add_patch(Ellipse(
            (np.mean(xs), np.mean(ys)),
            2 * 1.5 * np.sqrt(vals[1]), 2 * 1.5 * np.sqrt(vals[0]),
            angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
            fill=False, edgecolor=color, linewidth=1.2, linestyle='--', alpha=0.7))

legend_handles = [mpatches.Patch(color=PITCH_COLORS[pt],
    label=f'{pt} - {"Fastball" if pt == "FF" else PITCH_NAMES[pt]}')
    for pt in sorted_types]
leg = ax_plot.legend(handles=legend_handles, loc='upper center',
    bbox_to_anchor=(0.5, -0.07), ncol=min(len(sorted_types), 4),
    fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.2)
for t in leg.get_texts():
    t.set_color('#ccc')


# ═══════════════════════════════════════════════════════════════
# PITCH LOCATION PLOTS (RHH left, LHH right)
# ═══════════════════════════════════════════════════════════════
LOC_TITLE_Y = 0.565
LOC_TOP = 0.55
LOC_BOTTOM = 0.29
LOC_HEIGHT = LOC_TOP - LOC_BOTTOM
LOC_L_X = 0.01
LOC_R_X = 0.24
LOC_W = 0.22


# Fixed zone bounds — same size for every pitcher, every card
def draw_zone(ax, hand):
    ax.set_facecolor('#1a1d21')
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(0.5, 4.2)
    # No aspect='equal'
    # Strike zone
    ax.add_patch(Rectangle((-PLATE_HALF, avg_bot), PLATE_HALF * 2, avg_top - avg_bot,
        fill=False, edgecolor='#888', linewidth=1.5, zorder=2))
    # 9-zone grid
    tw = PLATE_HALF * 2 / 3
    th = (avg_top - avg_bot) / 3
    for i in range(1, 3):
        ax.plot([-PLATE_HALF + i * tw, -PLATE_HALF + i * tw], [avg_bot, avg_top],
            color='#444', linewidth=0.5, zorder=2)
        ax.plot([-PLATE_HALF, PLATE_HALF], [avg_bot + i * th, avg_bot + i * th],
            color='#444', linewidth=0.5, zorder=2)
    # Closed home plate
    pt_y = avg_bot - 0.15
    plate_x = [-PLATE_HALF, -PLATE_HALF, 0, PLATE_HALF, PLATE_HALF, -PLATE_HALF]
    plate_y = [pt_y, pt_y - 0.10, pt_y - 0.20, pt_y - 0.10, pt_y, pt_y]
    ax.plot(plate_x, plate_y, color='#888', linewidth=1.2, zorder=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color('#444')
    # Dots
    for pt in PITCH_ORDER:
        if pt not in locations[hand]:
            continue
        color = PITCH_COLORS[pt]
        for px_val, pz_val, desc, barrel_flag in locations[hand][pt]:
            if desc == 'Swinging Strike':
                ax.text(px_val, pz_val, 'W', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
            elif barrel_flag:
                ax.text(px_val, pz_val, 'B', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
            else:
                ax.scatter([px_val], [pz_val], c=color, s=55, alpha=1.0, edgecolors='none', zorder=3)


ax_loc_l = fig.add_axes([LOC_L_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
ax_loc_r = fig.add_axes([LOC_R_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
draw_zone(ax_loc_l, 'R')  # RHH on left
draw_zone(ax_loc_r, 'L')  # LHH on right

fig.text(LOC_L_X + LOC_W / 2, LOC_TITLE_Y, 'VS RHH', fontsize=14,
    fontweight='bold', color='white', fontfamily='DIN Condensed', ha='center', va='center')
fig.text(LOC_R_X + LOC_W / 2, LOC_TITLE_Y, 'VS LHH', fontsize=14,
    fontweight='bold', color='white', fontfamily='DIN Condensed', ha='center', va='center')

# W/B legend below pitch location plots
fig.text(LOC_R_X + LOC_W + 0.005, LOC_BOTTOM + 0.015, 'W = Whiff',
    fontsize=7, color='#ccc', va='bottom', ha='left', fontfamily='Avenir Next')
fig.text(LOC_R_X + LOC_W + 0.005, LOC_BOTTOM, 'B = Barrel',
    fontsize=7, color='#ccc', va='bottom', ha='left', fontfamily='Avenir Next')


# ═══════════════════════════════════════════════════════════════
# BATTED BALL DISTRIBUTION (donut + stacked bars)
# ═══════════════════════════════════════════════════════════════
BB_CHART_BOTTOM = 0.58
BB_CHART_HEIGHT = 0.17

# Overall totals
overall_bb = {bb: sum(bb_by_pitch[pt][bb] for pt in bb_by_pitch) for bb in BB_TYPES}
total_bip = sum(overall_bb.values())

# Donut (left side) — top aligned with top of first badge row
# First badge y in axes coords = 0.90, badge top = 0.90 + rh_bb*0.4
# where rh_bb = min(0.12, 0.85 / n_bb_pitches)
_n_bb_tmp = max(len([pt for pt in sorted_types if pt in bb_by_pitch and
    sum(bb_by_pitch[pt][bb] for bb in BB_TYPES) > 0]), 1)
_rh_bb_tmp = min(0.12, 0.85 / _n_bb_tmp)
_badge_top_axes = 0.90 + _rh_bb_tmp * 0.4  # top of first badge in axes coords
_badge_top_fig = BB_CHART_BOTTOM + BB_CHART_HEIGHT * _badge_top_axes  # in figure coords
_donut_bottom = _badge_top_fig - BB_CHART_HEIGHT + 0.02  # keep same height, nudge up
ax_bb_donut = fig.add_axes([0.01, _donut_bottom, 0.16, BB_CHART_HEIGHT])
ax_bb_donut.set_facecolor(BG)

if total_bip > 0:
    bb_vals = [overall_bb[bb] for bb in BB_TYPES]
    bb_cols = [BB_COLORS[bb] for bb in BB_TYPES]
    bb_wedges, _ = ax_bb_donut.pie(
        bb_vals, colors=bb_cols, startangle=90,
        wedgeprops=dict(width=0.30, edgecolor=BG, linewidth=2.0),
        counterclock=False
    )
    ax_bb_donut.text(0, 0, f'{total_bip}\nBIP', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='#e8e8e8', linespacing=1.15)
    # Add count labels on each wedge
    angle = 90  # startangle
    for bb, val in zip(BB_TYPES, bb_vals):
        if val == 0:
            continue
        ang_span = val / total_bip * 360
        mid_angle = angle - ang_span / 2
        mid_rad = np.radians(mid_angle)
        r = 0.85  # radius to place text (middle of donut ring)
        x = r * np.cos(mid_rad)
        y_pos = r * np.sin(mid_rad)
        tc = badge_text_color(BB_COLORS[bb])
        ax_bb_donut.text(x, y_pos, str(val), ha='center', va='center',
                         fontsize=7, fontweight='bold', color=tc)
        angle -= ang_span

# Stacked bars (right of donut)
ax_bb_bars = fig.add_axes([0.17, BB_CHART_BOTTOM, 0.30, BB_CHART_HEIGHT])
ax_bb_bars.set_xlim(0, 1)
ax_bb_bars.set_ylim(0, 1)
ax_bb_bars.set_xticks([])
ax_bb_bars.set_yticks([])
ax_bb_bars.axis('off')
ax_bb_bars.set_facecolor(BG)

# Sort pitch types by BIP count (descending), matching usage order
bb_pitch_order = sorted(
    [pt for pt in sorted_types if pt in bb_by_pitch and
     sum(bb_by_pitch[pt][bb] for bb in BB_TYPES) > 0],
    key=lambda pt: sum(bb_by_pitch[pt][bb] for bb in BB_TYPES),
    reverse=True
)

if bb_pitch_order:
    n_bb = len(bb_pitch_order)
    rh_bb = min(0.12, 0.85 / n_bb)
    gap_bb = 0.015

    for i, pt in enumerate(bb_pitch_order):
        y = 0.90 - i * (rh_bb + gap_bb)
        color = PITCH_COLORS.get(pt, '#999')
        tc = badge_text_color(color)
        total_pt = sum(bb_by_pitch[pt][bb] for bb in BB_TYPES)
        hh = bb_by_pitch[pt]['hh']
        brl = bb_by_pitch[pt]['brl']

        # Badge
        ax_bb_bars.add_patch(FancyBboxPatch(
            (0.02, y - rh_bb * 0.4), 0.08, rh_bb * 0.8,
            boxstyle="round,pad=0.008", facecolor=color, edgecolor='none'))
        ax_bb_bars.text(0.06, y, pt, fontsize=8, ha='center', va='center',
                        color=tc, fontweight='bold')

        # Gray track
        ax_bb_bars.add_patch(Rectangle(
            (0.13, y - rh_bb * 0.275), 0.48, rh_bb * 0.55,
            facecolor='#333840', edgecolor='none'))

        # Stacked segments with counts
        left = 0.13
        for bb in BB_TYPES:
            cnt = bb_by_pitch[pt][bb]
            pct = cnt / total_pt if total_pt > 0 else 0
            if pct > 0:
                seg_w = 0.48 * pct
                ax_bb_bars.add_patch(Rectangle(
                    (left, y - rh_bb * 0.275), seg_w, rh_bb * 0.55,
                    facecolor=BB_COLORS[bb], edgecolor=BG, linewidth=0.5))
                # Always show count centered in segment (text color based on bg luminance)
                seg_tc = badge_text_color(BB_COLORS[bb])
                ax_bb_bars.text(left + seg_w / 2, y, str(cnt),
                                ha='center', va='center', fontsize=7,
                                color=seg_tc, fontweight='bold')
                left += seg_w

        # Count label: "4 (1 Brl)" — barrels only
        label = str(total_pt)
        if brl > 0:
            label += f'  ({brl} Brl)'
        ax_bb_bars.text(0.63, y, label, fontsize=8, va='center', ha='left',
                        color='#e8e8e8', fontweight='bold')

    # Legend at bottom of bars area (inside axes)
    bb_legend_patches = [mpatches.Patch(color=BB_COLORS[bb], label=f'{BB_LABELS[bb]} ({overall_bb[bb]})')
                         for bb in BB_TYPES]
    ax_bb_bars.legend(handles=bb_legend_patches, loc='lower left',
                      bbox_to_anchor=(0.02, 0.12), ncol=2, fontsize=7,
                      frameon=False, labelcolor='#bbb', handlelength=1.0,
                      columnspacing=0.8)


# ═══════════════════════════════════════════════════════════════
# PITCH USAGE BARS (RHH left, LHH right)
# ═══════════════════════════════════════════════════════════════
usage = {'L': defaultdict(int), 'R': defaultdict(int)}
tots = {'L': 0, 'R': 0}
for p in fried:
    bh, pt = p.get('Bats', ''), p.get('Pitch Type', '')
    if bh in ('L', 'R') and pt:
        usage[bh][pt] += 1
        tots[bh] += 1


def draw_usage(ax, data, total, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(BG)
    S = USAGE_SHIFT
    content_center = (0.04 + S + 0.80 + S) / 2
    ax.text(content_center, 0.96, title, fontsize=13, fontweight='bold',
        ha='center', va='top', color='white', fontfamily='DIN Condensed')
    spts = sorted(data.keys(),
        key=lambda x: (-data[x], PITCH_ORDER.index(x) if x in PITCH_ORDER else 999))
    n = len(spts)
    if n == 0:
        return
    rh = min(0.10, 0.78 / n)
    gap = 0.012
    for i, pt in enumerate(spts):
        y = 0.84 - i * (rh + gap)
        pct = data[pt] / total if total > 0 else 0
        color = PITCH_COLORS.get(pt, '#999')
        tc = badge_text_color(color)
        ax.add_patch(FancyBboxPatch((0.04 + S, y - rh * 0.4), 0.08, rh * 0.8,
            boxstyle="round,pad=0.008", facecolor=color, edgecolor='none'))
        ax.text(0.08 + S, y, pt, fontsize=8, ha='center', va='center',
            color=tc, fontweight='bold')
        ax.add_patch(Rectangle((0.15 + S, y - rh * 0.275), 0.55, rh * 0.55,
            facecolor='#333840', edgecolor='none'))
        if pct > 0:
            ax.add_patch(Rectangle((0.15 + S, y - rh * 0.275), 0.55 * pct, rh * 0.55,
                facecolor=color, edgecolor='none'))
        ax.text(0.74 + S, y, f'{pct * 100:.0f}%', fontsize=10, va='center', ha='left',
            color='white', fontweight='bold', fontfamily='Avenir Next')


usage_left = 0.50
usage_mid = (usage_left + TABLE_RIGHT_FIG) / 2
ax_ul = fig.add_axes([usage_left, 0.32, usage_mid - usage_left, 0.17])
ax_ur = fig.add_axes([usage_mid, 0.32, TABLE_RIGHT_FIG - usage_mid, 0.17])
draw_usage(ax_ul, usage['R'], tots['R'], 'VS RHH')
draw_usage(ax_ur, usage['L'], tots['L'], 'VS LHH')


# ═══════════════════════════════════════════════════════════════
# PITCH METRICS TABLE
# ═══════════════════════════════════════════════════════════════
ax_table = fig.add_axes([TABLE_LEFT_FIG, 0.01, TABLE_RIGHT_FIG - TABLE_LEFT_FIG, 0.27])
ax_table.axis('off')
ax_table.set_facecolor(BG)

tc = len(fried)
pitch_stats = []
all_iz_wh = 0; all_iz_sw = 0; all_ooz_sw = 0; all_ooz_n = 0; all_bip = []; all_hard = 0; all_barrels = 0

# Sort pitch types by usage (descending), with PITCH_ORDER as tiebreaker
pitch_counts = {}
for p in fried:
    pt = p.get('Pitch Type', '')
    if pt:
        pitch_counts[pt] = pitch_counts.get(pt, 0) + 1
table_pitch_order = sorted(
    pitch_counts.keys(),
    key=lambda pt: (-pitch_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999)
)

for pt in table_pitch_order:
    pp = [p for p in fried if p.get('Pitch Type') == pt]
    if not pp:
        continue
    n = len(pp)
    velos = [v for v in (sf(p.get('Velocity')) for p in pp) if v]
    spins = [v for v in (sf(p.get('Spin Rate')) for p in pp) if v]
    ivbs = [v for v in (sf(p.get('IndVertBrk')) for p in pp) if v is not None]
    hbs = [v for v in (sf(p.get('HorzBrk')) for p in pp) if v is not None]
    relzs = [v for v in (sf(p.get('RelPosZ')) for p in pp) if v is not None]
    relxs = [v for v in (sf(p.get('RelPosX')) for p in pp) if v is not None]
    exts = [v for v in (sf(p.get('Extension')) for p in pp) if v is not None]
    vaas = [v for v in (sf(p.get('VAA')) for p in pp) if v is not None]
    haas = [v for v in (sf(p.get('HAA')) for p in pp) if v is not None]
    tilts = [p.get('OTilt', '') for p in pp]
    swings = [p for p in pp if p.get('Description') in SWING_DESC]
    whiffs = [p for p in pp if p.get('Description') == 'Swinging Strike']
    strikes = [p for p in pp if p.get('Description') in STRIKE_DESC]
    csw = [p for p in pp if p.get('Description') in ['Called Strike', 'Swinging Strike']]
    iz_n = iz_sw = ooz_n = ooz_sw = iz_wh = 0
    for p in pp:
        r = compute_iz(p)
        if r is None:
            continue
        if r:
            iz_n += 1
            if p.get('Description') in SWING_DESC: iz_sw += 1
            if p.get('Description') == 'Swinging Strike': iz_wh += 1
        else:
            ooz_n += 1
            if p.get('Description') in SWING_DESC: ooz_sw += 1
    all_iz_wh += iz_wh; all_iz_sw += iz_sw; all_ooz_sw += ooz_sw; all_ooz_n += ooz_n
    bip = [p for p in pp if p.get('Description') == 'In Play' and p.get('BBType')
           and not str(p.get('BBType', '')).startswith('bunt')]
    hard = sum(1 for p in bip if sf(p.get('ExitVelo')) and float(p['ExitVelo']) >= 95)
    barrels = sum(1 for p in bip if str(p.get('Barrel','')).strip()=='6')
    all_bip.extend(bip); all_hard += hard; all_barrels += barrels
    rvs = [v for v in (sf(p.get('RunExp')) for p in pp) if v is not None]
    rv = round(sum(rvs), 1) if rvs else None
    pt_name = 'Fastball' if pt == 'FF' else PITCH_NAMES.get(pt, pt)
    row = [
        pt_name, str(n), f"{n / tc * 100:.1f}%",
        f"{sum(velos) / len(velos):.1f}" if velos else '—',
        f"{max(velos):.1f}" if velos else '—',
        f"{int(sum(spins) / len(spins))}" if spins else '—',
        avg_tilt(tilts),
        f'{sum(ivbs) / len(ivbs):.1f}"' if ivbs else '—',
        f'{sum(hbs) / len(hbs):.1f}"' if hbs else '—',
        fmt_fi(sum(relzs) / len(relzs)) if relzs else '—',
        fmt_fi(sum(relxs) / len(relxs)) if relxs else '—',
        fmt_fi(sum(exts) / len(exts)) if exts else '—',
        f"{sum(vaas) / len(vaas):.2f}°" if vaas else '—',
        f"{sum(haas) / len(haas):.2f}°" if haas else '—',
        f"{len(strikes) / n * 100:.1f}%",
        f"{iz_n / n * 100:.1f}%" if n else '—',
        f"{len(csw) / n * 100:.1f}%",
        f"{len(whiffs) / len(swings) * 100:.1f}%" if swings else '—',
        f"{iz_wh / iz_sw * 100:.1f}%" if iz_sw else '—',
        f"{ooz_sw / ooz_n * 100:.1f}%" if ooz_n else '—',
        f"{hard / len(bip) * 100:.1f}%" if bip else '—',
        f"{barrels / len(bip) * 100:.1f}%" if bip else '—',
        str(rv) if rv is not None else '—',
    ]
    pitch_stats.append((pt, row))

t_str = [p for p in fried if p.get('Description') in STRIKE_DESC]
t_csw = [p for p in fried if p.get('Description') in ['Called Strike', 'Swinging Strike']]
t_sw = [p for p in fried if p.get('Description') in SWING_DESC]
t_wh = [p for p in fried if p.get('Description') == 'Swinging Strike']
t_iz = sum(1 for p in fried if compute_iz(p) == True)
t_rv = [v for v in (sf(p.get('RunExp')) for p in fried) if v is not None]

total_row = [
    'Total', str(tc), '100%', '—', '—', '—', '—', '—', '—', '—', '—', '—', '—', '—',
    f"{len(t_str) / tc * 100:.1f}%",
    f"{t_iz / tc * 100:.1f}%",
    f"{len(t_csw) / tc * 100:.1f}%",
    f"{len(t_wh) / len(t_sw) * 100:.1f}%" if t_sw else '—',
    f"{all_iz_wh / all_iz_sw * 100:.1f}%" if all_iz_sw else '—',
    f"{all_ooz_sw / all_ooz_n * 100:.1f}%" if all_ooz_n else '—',
    f"{all_hard / len(all_bip) * 100:.1f}%" if all_bip else '—',
    f"{all_barrels / len(all_bip) * 100:.1f}%" if all_bip else '—',
    str(round(sum(t_rv), 1)) if t_rv else '—',
]

col_headers = [
    'Pitch Type', 'Count', 'Usage', 'Avg Velo', 'Max Velo', 'Spin Rate', 'OTilt',
    'IVB', 'HB', 'RelZ', 'RelX', 'Ext', 'VAA', 'HAA',
    'Strike%', 'Zone%', 'CSW%', 'Whiff%', 'IZWhiff%', 'Chase%', 'HardHit%', 'Barrel%', 'PitchRV',
]

cell_data = [r[1] for r in pitch_stats] + [total_row]
pt_codes = [r[0] for r in pitch_stats] + [None]

table = ax_table.table(cellText=cell_data, colLabels=col_headers,
    loc='upper center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#333840')
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor(DARKER)
        cell.set_text_props(color=ACCENT, fontweight='bold', fontsize=8)
    elif r == len(cell_data):
        cell.set_facecolor(DARKER)
        cell.set_text_props(fontweight='bold', color='white')
    else:
        bg = DARK_CELL if r % 2 == 1 else '#252930'
        cell.set_facecolor(bg)
        cell.set_text_props(color='#e8e8e8', fontweight='bold')
    if c == 0 and r > 0:
        pc = pt_codes[r - 1]
        if pc:
            cell.set_facecolor(PITCH_COLORS.get(pc, '#999'))
            cell.set_text_props(color=badge_text_color(PITCH_COLORS.get(pc, '#999')),
                fontweight='bold')


# ═══════════════════════════════════════════════════════════════
# DIVIDER LINE + CYAN BORDERS
# ═══════════════════════════════════════════════════════════════
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_bbox = fig.get_window_extent(renderer)

# Cyan divider between HAA and Strike%
strike_cell = table.get_celld()[0, DIVIDER_COL]
x_fig = strike_cell.get_window_extent(renderer).x0 / fig_bbox.width
top_y = strike_cell.get_window_extent(renderer).y1 / fig_bbox.height - 0.001
bot_cell = table.get_celld()[len(cell_data), DIVIDER_COL]
bot_y = bot_cell.get_window_extent(renderer).y0 / fig_bbox.height
fig.add_artist(plt.Line2D([x_fig, x_fig], [bot_y, top_y],
    transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))

# Cyan border around entire metrics table
tl = table.get_celld()[0, 0].get_window_extent(renderer)
br = table.get_celld()[len(cell_data), len(col_headers) - 1].get_window_extent(renderer)
l = tl.x0 / fig_bbox.width
r_ = br.x1 / fig_bbox.width
t = tl.y1 / fig_bbox.height - 0.001
b = br.y0 / fig_bbox.height
for x1, y1, x2, y2 in [(l, b, r_, b), (l, t, r_, t), (l, b, l, t), (r_, b, r_, t)]:
    fig.add_artist(plt.Line2D([x1, x2], [y1, y2],
        transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))


# ═══════════════════════════════════════════════════════════════
# WATERMARK & SAVE
# ═══════════════════════════════════════════════════════════════
# Place watermark just below the metrics table border
fig.text(0.99, b - 0.008, 'Huronalytics', fontsize=9, ha='right', va='top', color='#555',
    style='italic', fontfamily='DIN Condensed')

plt.savefig(OUTPUT_FILE, dpi=SAVE_DPI, bbox_inches='tight', facecolor=BG, pad_inches=0.1)
plt.close()

# Crop bottom dead space
card_img = Image.open(OUTPUT_FILE)
pixels = np.array(card_img)
bg_rgb = (20, 22, 25)  # BG=#141619
for y in range(pixels.shape[0]-1, 0, -1):
    row = pixels[y, :, :3]
    if not np.all(np.abs(row.astype(int) - np.array(bg_rgb)) < 10):
        crop_y = min(y + 30, pixels.shape[0])
        card_img = card_img.crop((0, 0, card_img.width, crop_y))
        card_img.save(OUTPUT_FILE)
        break

print(f'Done — saved to {OUTPUT_FILE}')
