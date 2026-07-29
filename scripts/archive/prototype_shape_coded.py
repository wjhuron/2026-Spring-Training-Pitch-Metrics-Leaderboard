#!/usr/bin/env python3
"""Mock up color + marker-shape (redundant) pitch encoding.

Redundant encoding is the accessibility gold standard: color for normal vision,
shape as the backup channel that survives colorblindness AND density. This
renders a realistic all-9-pitch plot (real RHP movement profiles, ~140 each) as
a 2x2: {Normal, Deuteranopia} x {color only, color + shape}. The payoff is the
bottom row — under deuteranopia the arm-side colors converge, but circle/square/
triangle still separate sinker/changeup/splitter.
Output: ~/Downloads/shape_coded_mockup.png   (prototype only — not committed)
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

np.random.seed(7)
BG = '#f0e8d8'; PLOT_PANEL = '#e8dfcb'; GRID = '#cfc4ad'
TEXT_PRIMARY = '#1a1612'; TEXT_MUTED = '#6a5f55'

NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Changeup', 'FS': 'Splitter',
         'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball'}
ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']
COLOR = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'CH': '#009E73', 'FS': '#CC79A7',
         'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}
# arm-side trio SI/CH/FS = circle/square/triangle (the 3 most distinct shapes)
SHAPE = {'FF': 'D', 'SI': 'o', 'FC': 'P', 'CH': 's', 'FS': '^',
         'SL': 'v', 'ST': 'X', 'SV': '*', 'CU': 'p'}
N_PER = 140

DEUT = np.array([[0.367322, 0.860646, -0.227968],
                 [0.280085, 0.672501, 0.047413],
                 [-0.011820, 0.042940, 0.968881]])
hx = lambda h: np.array([int(h[i:i+2], 16) / 255 for i in (1, 3, 5)])
s2l = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
l2s = lambda c: np.where(np.clip(c, 0, 1) <= 0.0031308, np.clip(c, 0, 1) * 12.92,
                         1.055 * np.clip(c, 0, 1) ** (1 / 2.4) - 0.055)
deut = lambda h: tuple(l2s(DEUT @ s2l(hx(h))))

with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
pts = defaultdict(list)
for p in P:
    if p.get('Throws') != 'R' or p.get('Pitch Type') not in NAMES:
        continue
    try:
        pts[p['Pitch Type']].append((float(p['HorzBrk']), float(p['IndVertBrk'])))
    except (TypeError, ValueError):
        continue
samp = {pt: np.random.multivariate_normal(np.array(pts[pt]).mean(0),
        np.cov(np.array(pts[pt]).T) * 0.45, N_PER) for pt in ORDER}


def draw(ax, cmap, shapes, title):
    ax.set_facecolor(PLOT_PANEL); ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
    ax.grid(True, alpha=0.5, color=GRID); ax.tick_params(labelsize=7, colors=TEXT_MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, fontsize=12.5, fontweight='bold', color=TEXT_PRIMARY, pad=6)
    for pt in ORDER:
        xy = samp[pt]
        ax.scatter(xy[:, 0], xy[:, 1], color=cmap[pt], marker=(shapes[pt] if shapes else 'o'),
                   s=30, alpha=0.85, edgecolors='none', zorder=3)
    handles = [Line2D([0], [0], marker=(shapes[pt] if shapes else 'o'), color='none',
               markerfacecolor=cmap[pt], markeredgecolor='none', markersize=8,
               label=f'{pt}·{NAMES[pt]}') for pt in ORDER]
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.11),
                    ncol=5, fontsize=7.5, frameon=False, columnspacing=0.9, handletextpad=0.3)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)


fig, ax = plt.subplots(2, 2, figsize=(16, 16.6), dpi=145)
fig.patch.set_facecolor(BG)
fig.suptitle('Redundant color + shape encoding — does shape rescue the colorblind arm-side?',
             fontsize=16.5, fontweight='bold', color=TEXT_PRIMARY, y=0.985)
draw(ax[0, 0], COLOR, None, 'Normal vision — color only')
draw(ax[0, 1], COLOR, SHAPE, 'Normal vision — color + shape')
draw(ax[1, 0], {pt: deut(COLOR[pt]) for pt in ORDER}, None, 'Deuteranopia — color only')
draw(ax[1, 1], {pt: deut(COLOR[pt]) for pt in ORDER}, SHAPE, 'Deuteranopia — color + shape')
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06, hspace=0.20, wspace=0.14)
out = os.path.expanduser('~/Downloads/shape_coded_mockup.png')
fig.savefig(out, dpi=145, facecolor=BG); print('Saved:', out)
