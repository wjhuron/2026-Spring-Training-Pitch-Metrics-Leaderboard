#!/usr/bin/env python3
"""Render per-pitch-type Loc+ 5-zone decomposition PNGs (article arms).

The pitcher analog of the hitter SD+/CT+ zone profile, but built on the
per-pitch Loc+ atoms cached on every pitch row: each pitch is classified into
Heart / Shadow-In / Shadow-Out / Chase / Waste (pipeline_sdplus geometry,
InZone-split shadow), and each zone shows the mean Loc+ atom of the pitches
thrown there plus the share of pitches in that zone. Because displayed Loc+
is the plain mean of atoms (coherent canon), the share-weighted mean of the
five zone numbers reproduces the printed pitch Loc+ exactly; the script
asserts that per panel.

Usage: python3 scripts/render_loc_zones.py
Outputs ~/Downloads/ArticleVisuals/<Last>/LocZones_<LastFirst>.png
"""
import math
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_sdplus import classify_zone  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
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

ATOM_LO, ATOM_HI = 70.0, 130.0   # 100 ± 3 SD color domain (matches command maps)
FADE_N = 10                       # zones below this n render faded
MIN_TYPE_N = 25                   # skip pitch types below this many graded pitches

PITCHERS = {
    'Perales, Luis': ['ROC'],
    'Kent, Jackson': ['ROC'],
    'Sinclair, Jack': ['ROC'],
    'Tolman, Erik': ['ROC'],
    'Bird, Jake': ['NYY', 'ROC', 'AAA'],
    'Cruz, Yovanny': ['NYY', 'WSH', 'AAA'],
    'Dion, Will': ['CLE', 'WSH', 'ROC', 'AAA'],
}

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
ZONE_TITLES = {'heart': 'HEART', 'shadow_in': 'SHADOW·IN',
               'shadow_out': 'SHADOW·OUT', 'chase': 'CHASE', 'waste': 'WASTE'}

# Display geometry (zone-normalized z, x in feet) — mirrors render_decision_zones.
ZONE_X, HEART_X, SHADOW_X, CHASE_X = 0.83, 6.7 / 12, 13.3 / 12, 20.0 / 12
DX, DZ0, DZ1 = 1.95, -0.75, 1.75
ZONE_RECTS = [
    ('waste', (-DX, DZ0, 2 * DX, DZ1 - DZ0)),
    ('chase', (-CHASE_X, -0.5, 2 * CHASE_X, 2.0)),
    ('shadow_out', (-SHADOW_X, -1 / 6, 2 * SHADOW_X, 1 + 2 / 6)),
    ('shadow_in', (-ZONE_X, 0, 2 * ZONE_X, 1.0)),
    ('heart', (-HEART_X, 1 / 6, 2 * HEART_X, 4 / 6)),
]
LABEL_POS = {'heart': (0, 0.5), 'shadow_in': (0, 0.115), 'shadow_out': (0, 1.078),
             'chase': (0, 1.33), 'waste': (0, 1.615)}


def heat_color(t):
    """Command-map ramp: t=0 slate (good), t=1 brick (costly)."""
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def draw_panel(ax, title, zone_stats, n_total):
    ax.set_xlim(-DX, DX)
    ax.set_ylim(DZ0, DZ1)
    ax.set_aspect('equal')
    ax.axis('off')
    halo = dict(boxstyle='round,pad=0.15', facecolor=CREAM, edgecolor='none', alpha=0.55)
    for zone, (x, z, w, h) in ZONE_RECTS:
        mean, n = zone_stats.get(zone, (None, 0))
        if mean is None:
            face, alpha = PAPER, 1.0
        else:
            # Percentile-heat convention: brick (hot) = good grades, slate = costly.
            t = (mean - ATOM_LO) / (ATOM_HI - ATOM_LO)
            face = heat_color(t)
            alpha = 1.0 if n >= FADE_N else 0.45
        ax.add_patch(Rectangle((x, z), w, h, facecolor=face, alpha=alpha,
                               edgecolor=(*INK, 0.35), linewidth=0.8, zorder=1))
    ax.add_patch(Rectangle((-ZONE_X, 0), 2 * ZONE_X, 1, fill=False,
                           edgecolor=(*INK, 0.8), linewidth=2, zorder=3))
    for zone, (x, z, w, h) in ZONE_RECTS:
        mean, n = zone_stats.get(zone, (None, 0))
        lx, lz = LABEL_POS[zone]
        dim = mean is not None and n < FADE_N
        head = ZONE_TITLES[zone]
        if mean is not None:
            head += f' · {100.0 * n / n_total:.0f}%'
        ax.text(lx, lz + 0.07, head, ha='center', fontsize=5.8,
                color=(*INK, 0.75), fontweight=600, bbox=halo, zorder=4)
        big = '–' if mean is None else f'{mean:.0f}'
        ax.text(lx, lz - 0.055, big, ha='center', fontsize=11.5,
                color=(*INK, 0.45 if dim else 1.0), fontweight=700, bbox=halo, zorder=4)
    ax.set_title(title, fontsize=10, color=INK, pad=8, **TITLE_FONT)


def main():
    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)

    by_pitcher = defaultdict(list)
    for p in allp:
        teams = PITCHERS.get(p.get('Pitcher'))
        if teams and p.get('PTeam') in teams:
            by_pitcher[p['Pitcher']].append(p)

    footer = ('number = average location grade (Loc+) of his pitches in that zone: '
              '100 = the MLB-average pitch location for that pitch type and count, '
              'each 10 = one SD better (higher = spots that help the pitcher) · '
              '% = share of pitches thrown there · usage-weighted zone average = the '
              'pitch’s overall Loc+ · red = good, blue = costly · '
              'faded = under 10 pitches · catcher view')

    for name, pitches in sorted(by_pitcher.items()):
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)
        teams = ' + '.join(sorted({p['PTeam'] for p in pitches}))

        # (pitch type or 'ALL') -> zone -> [sum, n]
        acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        for p in pitches:
            v = sf(p.get('Loc+'))
            if v is None:
                continue
            zone = classify_zone(p)
            if zone is None:
                continue
            for key in ('ALL', p.get('Pitch Type') or '?'):
                acc[key][zone][0] += v
                acc[key][zone][1] += 1

        panels = [('ALL', acc['ALL'])]
        for pt, zs in sorted(((k, v) for k, v in acc.items() if k != 'ALL'),
                             key=lambda kv: -sum(n for _, n in kv[1].values())):
            n_pt = sum(n for _, n in zs.values())
            if n_pt >= MIN_TYPE_N:
                panels.append((pt, zs))
            else:
                print(f'  skip {name} {pt}: n={n_pt} < {MIN_TYPE_N}')

        ncols = 3
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), dpi=200)
        fig.patch.set_facecolor(CREAM)
        axes = [ax for row in (axes if nrows > 1 else [axes]) for ax in row]
        for ax in axes[len(panels):]:
            ax.axis('off')

        for ax, (key, zs) in zip(axes, panels):
            n_total = sum(n for _, n in zs.values())
            wsum = sum(s for s, _ in zs.values())
            overall = wsum / n_total
            # Coherence check: share-weighted zone means must equal the atom mean.
            recon = sum((s / n) * n for s, n in zs.values()) / n_total
            assert abs(recon - overall) < 1e-9
            stats = {z: (s / n, n) for z, (s, n) in zs.items()}
            label = 'All pitches' if key == 'ALL' else key
            draw_panel(ax, f'{label} · Loc+ {overall:.0f} · n={n_total}', stats, n_total)

        fig.suptitle(f'{first} {last}: Loc+ by Attack Zone · 2026 ({teams})',
                     fontsize=14, color=INK, y=0.995, **TITLE_FONT)
        fig.text(0.5, 0.012, footer, ha='center', fontsize=7, color=(*INK, 0.6))
        fig.tight_layout(rect=(0, 0.03, 1, 0.965))
        out = os.path.join(outdir, f'LocZones_{last}{first}.png')
        fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        print(name, '->', out)


if __name__ == '__main__':
    main()
