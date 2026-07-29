#!/usr/bin/env python3
"""Mockup: redesigned vs RHH / vs LHH location panels for the season pitcher card.

Top row = CURRENT (filled ellipses for every qualifying pitch type, W/B letter
glyphs, floating chip legend). Bottom row = PROPOSED (outline-only ellipses for
the top-4 usage pitch types, small x glyphs for whiffs, open diamond glyphs for
barrels, legend on an opaque paper panel that data marks clip out of).

Real Lugo (KCR) data from data/all_pitches_rs_cache.pkl. Run from repo root.
Output: ~/Downloads/mockup_location_panels.png
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch

# ── Card theme (Cards.py warm paper) ──
BG = '#f0e8d8'; PLOT_PANEL = '#e8dfcb'; GRID_COLOR = '#c5b89f'
TEXT_PRIMARY = '#1a1612'; TEXT_SECONDARY = '#3a3530'; TEXT_MUTED = '#6a5f55'
TEXT_FAINT = '#8a7f75'; SUBTLE_BORDER = '#c5b89f'; ACCENT = '#9f3026'
PITCH_COLORS = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#D55E00',
                'ST': '#56B4E9', 'CU': '#332288', 'SV': '#882255', 'CH': '#009E73',
                'FS': '#CC79A7', 'KN': '#9A9A9A'}
PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS', 'KN']
PLATE_HALF = 17 / 12 / 2


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def luminance(hc):
    r, g, b = int(hc[1:3], 16)/255, int(hc[3:5], 16)/255, int(hc[5:7], 16)/255
    def lin(c):
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)


def badge_text_color(hc):
    return 'black' if luminance(hc) > 0.25 else 'white'


def _darken(hexc, f):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(int(hexc[i:i+2], 16)*f)))
                                   for i in (1, 3, 5))


def _rgba(hexc, a):
    return (int(hexc[1:3], 16)/255, int(hexc[3:5], 16)/255, int(hexc[5:7], 16)/255, a)


# ── Load Lugo pitches ──
with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
lugo = [p for p in P if p.get('Pitcher') == 'Lugo, Seth' and p.get('PTeam') == 'KCR']
print(f'{len(lugo)} Lugo pitches')

locations = {'L': defaultdict(list), 'R': defaultdict(list)}
usage = {'L': defaultdict(int), 'R': defaultdict(int)}
tot = {'L': 0, 'R': 0}
szt, szb = [], []
for p in lugo:
    bh, pt = p.get('Bats'), p.get('Pitch Type')
    px, pz = sf(p.get('PlateX')), sf(p.get('PlateZ'))
    t, b = sf(p.get('SzTop')), sf(p.get('SzBot'))
    if t is not None and b is not None:
        szt.append(t); szb.append(b)
    if bh not in ('L', 'R') or not pt or px is None or pz is None:
        continue
    is_whiff = p.get('Description') == 'Swinging Strike'
    is_brl = str(p.get('Barrel', '')).strip() == '6'
    locations[bh][pt].append((px, pz, is_whiff, is_brl))
    usage[bh][pt] += 1; tot[bh] += 1
avg_top, avg_bot = np.mean(szt), np.mean(szb)

ELLIPSE_MIN = 10


def zone_frame(ax):
    ax.set_facecolor(PLOT_PANEL)
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(0.5, 4.2)
    ax.add_patch(Rectangle((-PLATE_HALF, avg_bot), PLATE_HALF*2, avg_top-avg_bot,
                           fill=False, edgecolor=TEXT_SECONDARY, linewidth=1.5, zorder=2))
    tw = PLATE_HALF*2/3; th = (avg_top-avg_bot)/3
    for i in (1, 2):
        ax.plot([-PLATE_HALF+i*tw]*2, [avg_bot, avg_top], color=GRID_COLOR, lw=0.6, zorder=2)
        ax.plot([-PLATE_HALF, PLATE_HALF], [avg_bot+i*th]*2, color=GRID_COLOR, lw=0.6, zorder=2)
    pty = avg_bot - 0.15
    ax.plot([-PLATE_HALF, -PLATE_HALF, 0, PLATE_HALF, PLATE_HALF, -PLATE_HALF],
            [pty, pty-0.10, pty-0.20, pty-0.10, pty, pty], color=TEXT_SECONDARY, lw=1.2, zorder=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_color(TEXT_FAINT)


def cov_ellipse(pts):
    xs = np.array([q[0] for q in pts]); ys = np.array([q[1] for q in pts])
    cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
    if vals[0] <= 0 or vals[1] <= 0:
        return None
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return (np.mean(xs), np.mean(ys), 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), ang)


def draw_current(ax, hand):
    zone_frame(ax)
    # Filled ellipses for EVERY qualifying pitch type + center dots (current card)
    for pt in PITCH_ORDER:
        pts = locations[hand].get(pt, [])
        if len(pts) < ELLIPSE_MIN:
            continue
        e = cov_ellipse(pts)
        if not e:
            continue
        mx, my, w, h, ang = e
        c = PITCH_COLORS[pt]
        ax.add_patch(Ellipse((mx, my), w, h, angle=ang, fill=True,
                             facecolor=_rgba(c, 0.42), edgecolor=_rgba(_darken(c, 0.6), 0.9),
                             linewidth=2.0, zorder=1))
        ax.scatter([mx], [my], c=c, s=32, edgecolors=TEXT_PRIMARY, linewidths=0.6, zorder=4)
    # W/B letter glyphs
    for pt in PITCH_ORDER:
        c = PITCH_COLORS.get(pt, '#999')
        for px, pz, wf, brl in locations[hand].get(pt, []):
            if wf:
                ax.text(px, pz, 'W', fontsize=8, fontweight='bold', color=c,
                        ha='center', va='center', zorder=3)
            elif brl:
                ax.text(px, pz, 'B', fontsize=8, fontweight='bold', color=c,
                        ha='center', va='center', zorder=3)
    # Floating chip legend (no backing panel) — the current look
    mix = sorted(usage[hand].items(), key=lambda kv: -kv[1])
    x0, row_h, cy = 0.035, 0.072, 0.945
    for pt, cnt in mix:
        c = PITCH_COLORS.get(pt, TEXT_SECONDARY)
        ax.add_patch(Rectangle((x0, cy-row_h*0.34), 0.095, row_h*0.68,
                               transform=ax.transAxes, facecolor=c, edgecolor='none', zorder=6))
        ax.text(x0+0.0475, cy, pt, transform=ax.transAxes, ha='center', va='center',
                fontsize=8, fontweight='bold', color=badge_text_color(c), zorder=7,
                fontfamily='Avenir Next')
        ax.text(x0+0.125, cy, f'{cnt/tot[hand]*100:.0f}%', transform=ax.transAxes,
                ha='left', va='center', fontsize=9.5, fontweight='bold',
                color=TEXT_PRIMARY, zorder=7, fontfamily='Avenir Next')
        cy -= row_h


def draw_proposed(ax, hand):
    zone_frame(ax)
    mix = sorted(usage[hand].items(), key=lambda kv: -kv[1])
    top4 = {pt for pt, _ in mix[:4]}
    # Outline-only ellipses, top-4 usage types only
    for pt in PITCH_ORDER:
        if pt not in top4:
            continue
        pts = locations[hand].get(pt, [])
        if len(pts) < ELLIPSE_MIN:
            continue
        e = cov_ellipse(pts)
        if not e:
            continue
        mx, my, w, h, ang = e
        c = PITCH_COLORS[pt]
        ax.add_patch(Ellipse((mx, my), w, h, angle=ang, fill=False,
                             edgecolor=_rgba(c, 0.95), linewidth=2.2, zorder=1))
        ax.scatter([mx], [my], c=c, s=30, edgecolors=TEXT_PRIMARY, linewidths=0.6, zorder=4)
    # Whiffs = small x glyphs, barrels = open diamonds
    for pt in PITCH_ORDER:
        c = PITCH_COLORS.get(pt, '#999')
        pts = locations[hand].get(pt, [])
        wx = [(px, pz) for px, pz, wf, brl in pts if wf]
        bx = [(px, pz) for px, pz, wf, brl in pts if (not wf) and brl]
        if wx:
            ax.scatter(*zip(*wx), marker='x', s=26, c=c, linewidths=1.4, zorder=3)
        if bx:
            ax.scatter(*zip(*bx), marker='D', s=34, facecolors='none',
                       edgecolors=c, linewidths=1.3, zorder=3)
    # Legend on an OPAQUE paper panel (reserved zone) — marks render under it
    mix_rows = len(mix)
    row_h = 0.072
    panel_w, panel_h = 0.235, mix_rows*row_h + 0.035
    px0, py1 = 0.022, 0.978
    ax.add_patch(FancyBboxPatch((px0, py1-panel_h), panel_w, panel_h,
                                boxstyle='round,pad=0.008,rounding_size=0.012',
                                transform=ax.transAxes, facecolor=BG,
                                edgecolor=SUBTLE_BORDER, linewidth=1.0, zorder=8))
    cy = py1 - 0.033
    for pt, cnt in mix:
        c = PITCH_COLORS.get(pt, TEXT_SECONDARY)
        ax.add_patch(Rectangle((px0+0.014, cy-row_h*0.34), 0.095, row_h*0.68,
                               transform=ax.transAxes, facecolor=c, edgecolor='none', zorder=9))
        ax.text(px0+0.0615, cy, pt, transform=ax.transAxes, ha='center', va='center',
                fontsize=8, fontweight='bold', color=badge_text_color(c), zorder=10,
                fontfamily='Avenir Next')
        ax.text(px0+0.135, cy, f'{cnt/tot[hand]*100:.0f}%', transform=ax.transAxes,
                ha='left', va='center', fontsize=9.5, fontweight='bold',
                color=TEXT_PRIMARY, zorder=10, fontfamily='Avenir Next')
        cy -= row_h


fig = plt.figure(figsize=(13.2, 15.2), dpi=100)
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.978, 'LOCATION PANELS — CURRENT vs PROPOSED', ha='center', va='top',
         fontsize=24, fontweight='bold', color=TEXT_PRIMARY, fontfamily='DIN Condensed')
fig.text(0.5, 0.955, 'Seth Lugo · KCR · 2026 Season', ha='center', va='top',
         fontsize=12, color=TEXT_MUTED, fontfamily='Avenir Next')

ROW_H = 0.36; COL_W = 0.44
positions = {('cur', 'R'): [0.055, 0.530, COL_W, ROW_H],
             ('cur', 'L'): [0.545, 0.530, COL_W, ROW_H],
             ('pro', 'R'): [0.055, 0.068, COL_W, ROW_H],
             ('pro', 'L'): [0.545, 0.068, COL_W, ROW_H]}
for (mode, hand), rect in positions.items():
    ax = fig.add_axes(rect)
    (draw_current if mode == 'cur' else draw_proposed)(ax, hand)
    fig.text(rect[0] + rect[2]/2, rect[1] + rect[3] + 0.006, f'VS {hand}HH',
             ha='center', va='bottom', fontsize=15, fontweight='bold',
             color=TEXT_SECONDARY, fontfamily='DIN Condensed')

fig.text(0.5, 0.923, 'CURRENT — filled ellipses (all pitch types) · W / B letter glyphs · floating legend',
         ha='center', va='bottom', fontsize=11, fontweight='bold', color=ACCENT,
         fontfamily='Avenir Next')
fig.text(0.5, 0.461, 'PROPOSED — outline-only ellipses (top-4 usage) · × = whiff · open diamond = barrel · legend on reserved paper panel',
         ha='center', va='bottom', fontsize=11, fontweight='bold', color=ACCENT,
         fontfamily='Avenir Next')
fig.text(0.5, 0.028, f'W = Whiff · B = Barrel · Min. {ELLIPSE_MIN} pitches for ellipse',
         ha='center', fontsize=9, color=TEXT_MUTED, fontfamily='Avenir Next', fontweight='bold')

out = os.path.expanduser('~/Downloads/mockup_location_panels.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.15)
print('saved', out)
