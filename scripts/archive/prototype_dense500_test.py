#!/usr/bin/env python3
"""Stress-test the recommended palette at HIGH VOLUME.

Real life: some pitchers throw 500+ splitters (Gausman, Cruz) sitting right on
top of their changeup and sinker. This pulls each pitch type's real RHP movement
profile (mean + covariance) from the data, samples 500 of EACH, and renders the
recommended palette in normal vision + simulated deuteranopia, so we can see
whether the colors hold up when every cluster is fully populated and overlapping.
Output: ~/Downloads/dense500_palette_test.png
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

np.random.seed(7)

BG = '#f0e8d8'
PLOT_PANEL = '#e8dfcb'
GRID = '#cfc4ad'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

REC = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'CH': '#009E73',
       'FS': '#CC79A7', 'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}
NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Changeup', 'FS': 'Splitter',
         'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball'}
ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']
N_PER = 500

# ── deuteranopia simulation (Machado 2009, linear RGB) ──
DEUT = np.array([[0.367322, 0.860646, -0.227968],
                 [0.280085, 0.672501, 0.047413],
                 [-0.011820, 0.042940, 0.968881]])


def hex2rgb(h):
    return np.array([int(h[i:i+2], 16) / 255 for i in (1, 3, 5)])


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def deut(hexc):
    return tuple(np.clip(lin2srgb(DEUT @ srgb2lin(hex2rgb(hexc))), 0, 1))

# ── real RHP per-type movement profiles ──
with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
pts = defaultdict(list)
for p in P:
    if p.get('Throws') != 'R':
        continue
    pt = p.get('Pitch Type')
    if pt not in NAMES:
        continue
    try:
        pts[pt].append((float(p.get('HorzBrk')), float(p.get('IndVertBrk'))))
    except (TypeError, ValueError):
        continue

# sample 500 per type from each type's gaussian (cov scaled to a single
# pitcher's tighter command, so clusters are realistic, not league-wide blobs)
SCALE = 0.5
samples = {}
for pt in ORDER:
    arr = np.array(pts[pt])
    mean = arr.mean(axis=0)
    cov = np.cov(arr.T) * SCALE
    samples[pt] = np.random.multivariate_normal(mean, cov, N_PER)
    print(f'{pt} {NAMES[pt]:9s} n={len(arr):5d}  mean HB={mean[0]:+.1f} IVB={mean[1]:+.1f}')


def draw(ax, colormap, title):
    ax.set_facecolor(PLOT_PANEL)
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
    ax.grid(True, alpha=0.5, color=GRID)
    ax.tick_params(labelsize=8, colors=TEXT_MUTED)
    ax.set_xlabel('Horizontal Break (in)', fontsize=10, color=TEXT_MUTED, fontweight='bold')
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=10, color=TEXT_MUTED, fontweight='bold')
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT_PRIMARY, pad=8)
    for pt in ORDER:
        xy = samples[pt]
        ax.scatter(xy[:, 0], xy[:, 1], color=colormap[pt], s=9, alpha=0.55,
                   edgecolors='none', zorder=3)
    handles = [mpatches.Patch(color=colormap[pt], label=f'{pt} · {NAMES[pt]}') for pt in ORDER]
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                    ncol=5, fontsize=8, frameon=False, handlelength=1.1, columnspacing=1.1)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)


fig, axes = plt.subplots(1, 2, figsize=(17, 9), dpi=150)
fig.patch.set_facecolor(BG)
fig.suptitle('Recommended palette · 500 of every pitch type (real RHP movement profiles)',
             fontsize=17, fontweight='bold', color=TEXT_PRIMARY, y=0.98)
draw(axes[0], REC, 'Normal vision')
draw(axes[1], {pt: deut(REC[pt]) for pt in ORDER}, 'Deuteranopia (~6% of men)')
fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.14, wspace=0.16)

out = os.path.expanduser('~/Downloads/dense500_palette_test.png')
fig.savefig(out, dpi=150, facecolor=BG)
print('Saved:', out)
