#!/usr/bin/env python3
"""Render SD+/CT+ 5-zone attack-region PNGs for the ROC article hitters.

First consumer of the shipped per-zone splits (sdPlusHeart/ShadowIn/ShadowOut/
Chase/Waste and the ctPlus* equivalents) in data/hitter_leaderboard_rs.json.
Each hitter gets one PNG with two panels — Swing Decisions and Contact — where
each zone band is colored and labeled on the house 100±10 scale: z-scored
against the sample-weighted MLB distribution for that zone, so 100 = MLB
average, 110 = one SD above. Zone geometry mirrors pipeline_sdplus.py
(heart ±6.7" / middle 67% of zone height; shadow ±13.3" / ±17% height;
chase ±20" / ±50% height; waste beyond).

Usage: python3 scripts/render_decision_zones.py
Outputs to ~/Downloads/ArticleVisuals/<Last>/.
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HITTER_LB = os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')
OUT_ROOT = os.path.expanduser('~/Downloads/ArticleVisuals')

for f in os.listdir(os.path.join(ROOT, 'assets', 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(ROOT, 'assets', 'fonts', f))
plt.rcParams['font.family'] = 'IBM Plex Sans'
TITLE_FONT = {'fontfamily': 'Bitter', 'fontweight': 700}

PAPER = (240 / 255, 232 / 255, 216 / 255)
CREAM = '#e8dfcb'
INK = (58 / 255, 48 / 255, 38 / 255)
BRICK = (176 / 255, 64 / 255, 47 / 255)
SLATE = (66 / 255, 100 / 255, 138 / 255)

HITTERS = ['Ortiz, Abimelec', 'Morales, Yohandy', 'King, Seaver',
           'Glasser, Phillip', 'Pinckney, Andrew']
AAA_TEAMS = {'ROC', 'AAA'}
ZONES = ['Heart', 'ShadowIn', 'ShadowOut', 'Chase', 'Waste']
PANELS = [('sdPlus', 'Swing Decisions'), ('ctPlus', 'Contact')]

# Zone-normalized display geometry (x in feet, z 0..1 = strike zone).
ZONE_X, HEART_X, SHADOW_X, CHASE_X = 0.83, 6.7 / 12, 13.3 / 12, 20.0 / 12
DX, DZ0, DZ1 = 1.95, -0.75, 1.75


def heat_color(t):
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def wstats(pairs):
    """Sample-weighted mean and SD over (value, weight) pairs."""
    wsum = sum(w for _, w in pairs)
    mu = sum(v * w for v, w in pairs) / wsum
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / wsum
    return mu, math.sqrt(var)


def zone_rects():
    """(zone -> list of display rectangles) + border specs, painted
    outermost-first so inner bands sit on top."""
    return [
        ('Waste', [(-DX, DZ0, 2 * DX, DZ1 - DZ0)]),
        ('Chase', [(-CHASE_X, -0.5, 2 * CHASE_X, 2.0)]),
        ('ShadowOut', [(-SHADOW_X, -1 / 6, 2 * SHADOW_X, 1 + 2 / 6)]),
        ('ShadowIn', [(-ZONE_X, 0, 2 * ZONE_X, 1.0)]),
        ('Heart', [(-HEART_X, 1 / 6, 2 * HEART_X, 4 / 6)]),
    ]


LABEL_POS = {  # (x, z) anchor per zone label
    'Heart': (0, 0.5),
    'ShadowIn': (0, 0.115),
    'ShadowOut': (0, 1.085),
    'Chase': (0, 1.335),
    'Waste': (0, 1.60),
}
ZONE_TITLES = {'Heart': 'HEART', 'ShadowIn': 'SHADOW·IN',
               'ShadowOut': 'SHADOW·OUT', 'Chase': 'CHASE', 'Waste': 'WASTE'}


def main():
    with open(HITTER_LB) as f:
        rows = json.load(f)

    # League anchors: sample-weighted mean/SD per (metric, zone) over MLB rows.
    anchors = {}
    for prefix, _ in PANELS:
        for z in ZONES:
            pairs = [(r[prefix + z], r.get(prefix + 'N') or 0)
                     for r in rows
                     if r.get('team') not in AAA_TEAMS
                     and r.get(prefix + z) is not None
                     and (r.get(prefix + 'N') or 0) > 0]
            anchors[(prefix, z)] = wstats(pairs)

    targets = {r['hitter']: r for r in rows
               if r.get('hitter') in HITTERS and r.get('team') == 'ROC'}

    for name in HITTERS:
        r = targets[name]
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), dpi=200)
        fig.patch.set_facecolor(CREAM)
        for ax, (prefix, panel_title) in zip(axes, PANELS):
            ax.set_xlim(-DX, DX)
            ax.set_ylim(DZ0, DZ1)
            ax.set_aspect('equal')
            ax.axis('off')
            for zone, rects in zone_rects():
                mu, sd = anchors[(prefix, zone)]
                val = r.get(prefix + zone)
                zscore = (val - mu) / sd if (val is not None and sd > 0) else 0.0
                idx = 100 + 10 * zscore
                t = 0.5 + max(-0.5, min(0.5, zscore / 4))  # ±2 SD color span
                for (x, z, w, h) in rects:
                    ax.add_patch(Rectangle((x, z), w, h, facecolor=heat_color(t),
                                           edgecolor=(*INK, 0.35),
                                           linewidth=0.8, zorder=1))
                lx, lz = LABEL_POS[zone]
                halo = dict(boxstyle='round,pad=0.15', facecolor=CREAM,
                            edgecolor='none', alpha=0.55)
                ax.text(lx, lz + 0.055, ZONE_TITLES[zone], ha='center',
                        fontsize=6.5, color=(*INK, 0.8),
                        fontfamily='IBM Plex Sans', fontweight=600,
                        bbox=halo, zorder=4)
                ax.text(lx, lz - 0.05, f'{idx:.0f}', ha='center', fontsize=12.5,
                        color=INK, fontweight=700, bbox=halo, zorder=4)
            # Strike-zone border on top for orientation.
            ax.add_patch(Rectangle((-ZONE_X, 0), 2 * ZONE_X, 1, fill=False,
                                   edgecolor=(*INK, 0.8), linewidth=2, zorder=3))
            overall = r.get(prefix)
            pct = r.get(prefix + '_pctl')
            sub = f'{panel_title}  ·  {prefix.replace("Plus", "").upper()}+ ' \
                  f'{overall:.0f}' + (f' ({pct} pctl)' if pct is not None else '')
            ax.set_title(sub, fontsize=11, color=INK, pad=10, **TITLE_FONT)

        fig.suptitle(f'{first} {last} — Zone Profile', fontsize=15, color=INK,
                     y=0.985, **TITLE_FONT)
        fig.text(0.5, 0.025,
                 'per-zone grades on the 100 scale: 100 = MLB average, '
                 '10 = one SD across MLB hitters · catcher view',
                 ha='center', fontsize=7.5, color=(*INK, 0.6))
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = os.path.join(outdir, f'ZoneProfile_{last}{first}_SD_CT.png')
        fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        print(name, '->', out)


if __name__ == '__main__':
    main()
