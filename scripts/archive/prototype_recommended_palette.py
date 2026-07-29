#!/usr/bin/env python3
"""Render the RECOMMENDED pitch palette full-size on Seth Lugo's season.

Okabe-Ito (colorblind-safe, pops on warm paper), with two anchors Wally already
locked kept in place: amber sinker (#E0A81E) and brown cutter (#8B5A2B). Also
appends the 'recommended' palette to ~/Downloads/pitch_palettes.json.
Output: ~/Downloads/lugo_recommended_palette.png
"""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from collections import defaultdict

BG = '#f0e8d8'
PLOT_PANEL = '#e8dfcb'
GRID = '#cfc4ad'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

# Okabe-Ito assignment, keeping amber sinker + brown cutter.
REC = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'CH': '#009E73',
       'FS': '#CC79A7', 'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}
NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Changeup', 'FS': 'Splitter',
         'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball'}
ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']

with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
groups = defaultdict(list)
for p in P:
    if p.get('Pitcher') != 'Lugo, Seth':
        continue
    try:
        hb = float(p.get('HorzBrk')); ivb = float(p.get('IndVertBrk'))
    except (TypeError, ValueError):
        continue
    if p.get('Pitch Type') in NAMES:
        groups[p['Pitch Type']].append((hb, ivb))

fig, ax = plt.subplots(figsize=(11, 11.4), dpi=150)
fig.patch.set_facecolor(BG); ax.set_facecolor(PLOT_PANEL)
ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
ax.grid(True, alpha=0.5, color=GRID)
ax.tick_params(labelsize=9, colors=TEXT_MUTED)
ax.set_xlabel('Horizontal Break (in)', fontsize=11, color=TEXT_MUTED, fontweight='bold')
ax.set_ylabel('Induced Vertical Break (in)', fontsize=11, color=TEXT_MUTED, fontweight='bold')
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.set_title('Seth Lugo · 2026 — recommended palette (Okabe-Ito, amber sinker + brown cutter kept)',
             fontsize=14, fontweight='bold', color=TEXT_PRIMARY, pad=12)

for pt in sorted(groups, key=lambda k: -len(groups[k])):
    xs, ys = zip(*groups[pt]); c = REC[pt]
    ax.scatter(xs, ys, c=c, s=58, alpha=1.0, edgecolors='none', zorder=3)
    if len(groups[pt]) >= 6:
        cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
        if vals[0] > 0 and vals[1] > 0:
            ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)),
                2 * 1.5 * np.sqrt(vals[1]), 2 * 1.5 * np.sqrt(vals[0]),
                angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
                fill=False, edgecolor=c, lw=1.3, ls='--', alpha=0.75, zorder=2))

handles = [mpatches.Patch(color=REC[pt], label=f'{pt} · {NAMES[pt]}') for pt in ORDER if pt in groups]
leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                ncol=5, fontsize=9, frameon=False, handlelength=1.2, columnspacing=1.4)
for t in leg.get_texts():
    t.set_color(TEXT_PRIMARY)
fig.subplots_adjust(bottom=0.13, top=0.93, left=0.09, right=0.96)

out = os.path.expanduser('~/Downloads/lugo_recommended_palette.png')
fig.savefig(out, dpi=150, facecolor=BG)
print('Saved:', out)

pj = os.path.expanduser('~/Downloads/pitch_palettes.json')
data = {}
if os.path.exists(pj):
    with open(pj) as f:
        data = json.load(f)
data['recommended'] = REC
with open(pj, 'w') as f:
    json.dump(data, f, indent=2)
print('Updated:', pj, '(added "recommended")')
