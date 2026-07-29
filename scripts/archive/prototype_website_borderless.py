#!/usr/bin/env python3
"""Before/after for the website movement plot: current (Okabe fills + the stale
old-palette borders) vs borderless (solid dots). Dark theme to match the site,
web palette (bright sinker), real Lugo data (all 9 pitch types).
Output: ~/Downloads/website_dots_borderless_example.png
"""
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Dark site theme
BG = '#0e0e10'; PANEL = '#15171c'; GRID = '#33363d'; TXT = '#c8ccd4'; TXT_DIM = '#8a8f99'

# Web palette: Okabe-Ito with bright sinker
FILL = {'FF': '#0072B2', 'SI': '#FFD700', 'FC': '#8B5A2B', 'SL': '#D55E00', 'ST': '#56B4E9',
        'CU': '#332288', 'SV': '#882255', 'CH': '#009E73', 'FS': '#CC79A7'}
# The stale PITCH_BORDER_COLORS still in js/utils.js (old palette) — the mismatch
BORDER = {'FF': '#3366CC', 'SI': '#CCB000', 'FC': '#CC8400', 'SL': '#BBBBBB', 'ST': '#CC1076',
          'CU': '#B32626', 'SV': '#28A428', 'CH': '#A352BE', 'FS': '#33B3A6'}
NAMES = {'FF': 'Four-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider', 'ST': 'Sweeper',
         'CU': 'Curveball', 'SV': 'Slurve', 'CH': 'Changeup', 'FS': 'Splitter'}
ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS']

with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
groups = defaultdict(list)
for p in P:
    if p.get('Pitcher') != 'Lugo, Seth':
        continue
    try:
        hb = float(p['HorzBrk']); ivb = float(p['IndVertBrk'])
    except (TypeError, ValueError):
        continue
    if p.get('Pitch Type') in FILL:
        groups[p['Pitch Type']].append((hb, ivb))


def draw(ax, bordered, title):
    ax.set_facecolor(PANEL); ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.axhline(0, color=GRID, lw=0.8, ls='--'); ax.axvline(0, color=GRID, lw=0.8, ls='--')
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.tick_params(colors=TXT_DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel('Horizontal Break (in)', color=TXT_DIM, fontsize=10, fontweight='bold')
    ax.set_ylabel('Induced Vertical Break (in)', color=TXT_DIM, fontsize=10, fontweight='bold')
    ax.set_title(title, color=TXT, fontsize=13, fontweight='bold', pad=8)
    for pt in ORDER:
        xs, ys = zip(*groups[pt])
        if bordered:
            ax.scatter(xs, ys, c=FILL[pt], s=58, edgecolors=BORDER[pt], linewidths=1.3, zorder=3)
        else:
            ax.scatter(xs, ys, c=FILL[pt], s=58, edgecolors='none', zorder=3)


fig, ax = plt.subplots(1, 2, figsize=(17, 8.8), dpi=150)
fig.patch.set_facecolor(BG)
fig.suptitle('Website movement plot — current borders vs borderless (Lugo, dark theme)',
             color=TXT, fontsize=16, fontweight='bold', y=0.97)
draw(ax[0], True, 'Current — Okabe fills + stale old-palette borders')
draw(ax[1], False, 'Borderless — just the dots')
handles = [mpatches.Patch(color=FILL[pt], label=f'{pt} · {NAMES[pt]}') for pt in ORDER]
leg = fig.legend(handles=handles, loc='lower center', ncol=9, frameon=False,
                 fontsize=9, bbox_to_anchor=(0.5, 0.005))
for t in leg.get_texts():
    t.set_color(TXT)
fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.11, wspace=0.16)
out = os.path.expanduser('~/Downloads/website_dots_borderless_example.png')
fig.savefig(out, dpi=150, facecolor=BG); print('Saved:', out)
