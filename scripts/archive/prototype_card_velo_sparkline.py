#!/usr/bin/env python3
"""Mockup: fastball-velo-by-start sparkline strip under the season boxscore line.

Renders the top-left corner of the season card (name block + G/GS/IP/ERA/SIERA
stat strip) at true card proportions, with a thin (~40px at card scale) muted
sparkline mocked directly beneath the stat strip: one dot per start, game dates
on the x-axis, season-average dashes for reference.

Real Lugo (KCR) data from data/all_pitches_rs_cache.pkl. Run from repo root.
Output: ~/Downloads/mockup_velo_sparkline.png
"""
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Card theme (Cards.py warm paper) ──
BG = '#f0e8d8'; DARK_CELL = '#e2d8c4'; DARKER = '#d8ccb4'; ACCENT = '#9f3026'
TEXT_PRIMARY = '#1a1612'; TEXT_SECONDARY = '#3a3530'; TEXT_MUTED = '#6a5f55'
TEXT_FAINT = '#8a7f75'; SUBTLE_BORDER = '#c5b89f'; PLOT_PANEL = '#e8dfcb'
FF_COLOR = '#0072B2'


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)

velo_by_start = defaultdict(list)
for p in P:
    if p.get('Pitcher') != 'Lugo, Seth' or p.get('PTeam') != 'KCR':
        continue
    if p.get('Pitch Type') != 'FF':
        continue
    v = sf(p.get('Velocity'))
    if v is not None and p.get('Game Date'):
        velo_by_start[p['Game Date']].append(v)
dates = sorted(velo_by_start)
velos = [np.mean(velo_by_start[d]) for d in dates]
season_avg = np.mean([v for d in dates for v in velo_by_start[d]])
print(f'{len(dates)} starts, FF velo {min(velos):.1f}-{max(velos):.1f}, avg {season_avg:.1f}')

# ── Card-corner mock at true proportions (card is 16in wide) ──
FIG_W, FIG_H = 16, 5.0
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.axis('off'); ax.set_facecolor(BG)

photo_left = 0.16
# Headshot placeholder (keeps proportions honest without a network fetch)
photo_w, photo_h = 1.4, 2.1
photo_top = FIG_H - 0.25; photo_bottom = photo_top - photo_h
ax.add_patch(Rectangle((photo_left, photo_bottom), photo_w, photo_h,
                       facecolor=DARKER, edgecolor=TEXT_MUTED, linewidth=1.5))
ax.text(photo_left + photo_w/2, photo_bottom + photo_h/2, 'PHOTO', ha='center',
        va='center', fontsize=10, color=TEXT_FAINT, fontfamily='Avenir Next')

text_x = photo_left + photo_w + 0.3
ax.text(text_x, photo_top - 0.1, 'SETH LUGO', fontsize=32, fontfamily='DIN Condensed',
        color=TEXT_PRIMARY, va='top', fontweight='bold')
ax.text(text_x, photo_top - 0.85, 'RHP  |  KCR  |  Age: 36', fontsize=12,
        fontfamily='Avenir Next', color=TEXT_MUTED, va='top')
ax.text(text_x, photo_top - 1.5, '2026 Season  ·  Through Jul 1', fontsize=24,
        fontfamily='DIN Condensed', color=ACCENT, va='top')

# Boxscore stat strip (same geometry as render_card)
col_w, cell_h = 1.25, 0.46
headers = ['G', 'GS', 'IP', 'ERA', 'SIERA']
values = ['17', '17', '96.1', '4.20', '4.64']
stat_y_header = photo_bottom - 0.5; stat_y_value = stat_y_header - cell_h
for i, (hdr, val) in enumerate(zip(headers, values)):
    x = photo_left + i * col_w
    ax.add_patch(Rectangle((x, stat_y_header), col_w, cell_h, facecolor=DARKER,
                           edgecolor=SUBTLE_BORDER, linewidth=0.8))
    ax.text(x + col_w/2, stat_y_header + cell_h/2, hdr, fontsize=11, ha='center',
            va='center', color=TEXT_SECONDARY, fontweight='bold', fontfamily='Avenir Next')
    ax.add_patch(Rectangle((x, stat_y_value), col_w, cell_h, facecolor=DARK_CELL,
                           edgecolor=SUBTLE_BORDER, linewidth=0.8))
    ax.text(x + col_w/2, stat_y_value + cell_h/2, val, fontsize=14, ha='center',
            va='center', color=TEXT_PRIMARY, fontweight='bold', fontfamily='Avenir Next')
ax.add_patch(Rectangle((photo_left, stat_y_value), len(headers)*col_w,
                       stat_y_header + cell_h - stat_y_value, fill=False,
                       edgecolor=ACCENT, linewidth=2, zorder=5))

# ── Sparkline strip — thin (~0.27in ≈ 40px at save scale), under the boxscore ──
strip_w_in = len(headers) * col_w                 # same width as the stat strip
strip_h_in = 0.27                                 # ~40 px at 150 dpi
strip_top = stat_y_value - 0.42
axs = fig.add_axes([photo_left/FIG_W, (strip_top - strip_h_in)/FIG_H,
                    strip_w_in/FIG_W, strip_h_in/FIG_H])
axs.set_facecolor(BG)
xs = np.arange(len(dates))
axs.set_xlim(-0.6, len(dates) - 0.4)
pad = 0.6
axs.set_ylim(min(velos) - pad, max(velos) + pad)
axs.axhline(season_avg, color=TEXT_FAINT, lw=0.8, ls=(0, (2, 3)), alpha=0.8, zorder=1)
axs.plot(xs, velos, color=TEXT_MUTED, lw=1.1, alpha=0.85, zorder=2)
axs.scatter(xs, velos, s=16, c=TEXT_MUTED, zorder=3)
# accent the season high + latest start
hi = int(np.argmax(velos))
axs.scatter([hi], [velos[hi]], s=22, c=FF_COLOR, zorder=4)
axs.scatter([xs[-1]], [velos[-1]], s=22, c=ACCENT, zorder=4)
axs.axis('off')

# Label + endpoint annotations (drawn on the main axes, tiny & muted)
label_y = strip_top + 0.10
ax.text(photo_left, label_y, 'FB VELO BY START', fontsize=8.5, color=TEXT_SECONDARY,
        fontweight='bold', fontfamily='Avenir Next', va='bottom')
ax.text(photo_left + strip_w_in, label_y,
        f'{velos[-1]:.1f} last  ·  {season_avg:.1f} avg  ·  {max(velos):.1f} max',
        fontsize=8.5, color=TEXT_MUTED, fontweight='bold', fontfamily='Avenir Next',
        va='bottom', ha='right')


def _fmt(d):
    return datetime.strptime(d, '%Y-%m-%d').strftime('%b %-d')


date_y = strip_top - strip_h_in - 0.16
ax.text(photo_left, date_y, _fmt(dates[0]), fontsize=7.5, color=TEXT_FAINT,
        fontfamily='Avenir Next', va='top', ha='left')
ax.text(photo_left + strip_w_in/2, date_y, _fmt(dates[len(dates)//2]), fontsize=7.5,
        color=TEXT_FAINT, fontfamily='Avenir Next', va='top', ha='center')
ax.text(photo_left + strip_w_in, date_y, _fmt(dates[-1]), fontsize=7.5,
        color=TEXT_FAINT, fontfamily='Avenir Next', va='top', ha='right')

fig.text(0.985, 0.035, 'mockup — velo sparkline under the boxscore line',
         fontsize=9, ha='right', color=TEXT_FAINT, style='italic',
         fontfamily='DIN Condensed')

out = os.path.expanduser('~/Downloads/mockup_velo_sparkline.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.12)
print('saved', out)
