#!/usr/bin/env python3
"""Preview the website movement plot after matching the cards: borderless dots +
dashed/unfilled/pitch-colored covariance ellipses (1.5 sigma), dark theme, web
palette (bright sinker). Real Lugo data. Mirrors the new player-page.js logic.
Output: ~/Downloads/website_cards_style_example.png
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from collections import defaultdict

BG = '#0e0e10'; PANEL = '#15171c'; GRID = '#33363d'; TXT = '#c8ccd4'; TXT_DIM = '#8a8f99'

FILL = {'FF': '#0072B2', 'SI': '#FFD700', 'FC': '#8B5A2B', 'SL': '#D55E00', 'ST': '#56B4E9',
        'CU': '#332288', 'SV': '#882255', 'CH': '#009E73', 'FS': '#CC79A7'}
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
        groups[p['Pitch Type']].append((float(p['HorzBrk']), float(p['IndVertBrk'])))
    except (TypeError, ValueError):
        continue

fig, ax = plt.subplots(figsize=(10.5, 10), dpi=150)
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
ax.axhline(0, color=GRID, lw=0.9, ls='--'); ax.axvline(0, color=GRID, lw=0.9, ls='--')
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.tick_params(colors=TXT_DIM, labelsize=8)
for s in ax.spines.values():
    s.set_color(GRID)
ax.set_xlabel('Horizontal Break (in)', color=TXT_DIM, fontsize=10, fontweight='bold')
ax.set_ylabel('Induced Vertical Break (in)', color=TXT_DIM, fontsize=10, fontweight='bold')
ax.set_title('Website movement plot — matched to the cards (borderless + dashed spread ellipses)',
             color=TXT, fontsize=12.5, fontweight='bold', pad=10)

for pt in ORDER:
    xs, ys = zip(*groups[pt]); c = FILL[pt]
    ax.scatter(xs, ys, c=c, s=42, edgecolors='none', zorder=3)
    if len(groups[pt]) >= 6:                     # same gate as the cards
        cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
        if vals[0] > 0 and vals[1] > 0:
            ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)),
                2 * 1.5 * np.sqrt(vals[1]), 2 * 1.5 * np.sqrt(vals[0]),
                angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
                fill=False, edgecolor=c, lw=1.3, ls='--', alpha=0.7, zorder=2))

handles = [mpatches.Patch(color=FILL[pt], label=f'{pt} · {NAMES[pt]}') for pt in ORDER]
leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                ncol=5, frameon=False, fontsize=8.5)
for t in leg.get_texts():
    t.set_color(TXT)
fig.subplots_adjust(bottom=0.13, top=0.94, left=0.09, right=0.97)
out = os.path.expanduser('~/Downloads/website_cards_style_example.png')
fig.savefig(out, dpi=150, facecolor=BG); print('Saved:', out)
