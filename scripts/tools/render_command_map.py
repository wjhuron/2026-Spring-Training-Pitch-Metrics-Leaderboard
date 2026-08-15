#!/usr/bin/env python3
"""Render Loc+ command-map PNGs (season + per pitch type) for article use.

Port of js/player-page.js _renderLocValueCanvas onto matplotlib, except cells
are colored by the mean per-pitch Loc+ ATOM (the integer grades cached on
every pitch row) instead of raw league ExpRV. Within a pitch-type group the
atom is an affine transform of ExpRV, so the picture is the same shape as the
site's command map while carrying the same units as the printed Loc+ — and by
the coherent-canon rule the usage-weighted cell means tie out to the
leaderboard Loc+ exactly.

Usage:
    python3 scripts/tools/render_command_map.py                 # all 7 article arms
    python3 scripts/tools/render_command_map.py --parity        # + ExpRV parity map
                                                          #   for Perales
Outputs to ~/Downloads/ArticleVisuals/<Last>/.
"""
import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PITCHER_LB = os.path.join(ROOT, 'data', 'pitcher_leaderboard_rs.json')
PITCH_LB = os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')
OUT_ROOT = os.path.expanduser('~/Downloads/ArticleVisuals')

for f in os.listdir(os.path.join(ROOT, 'assets', 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(ROOT, 'assets', 'fonts', f))
plt.rcParams['font.family'] = 'IBM Plex Sans'
TITLE_FONT = {'fontfamily': 'Bitter', 'fontweight': 700}

# Grid mirrors pipeline_locplus.py.
X_MIN, X_MAX = -1.5, 1.5
BIN_X = 2.0 / 12.0
Z_MIN, Z_MAX = -0.6, 1.6
BIN_Z = 0.10
# Display bounds mirror the JS canvas.
DX0, DX1 = -1.7, 1.7
DZ0, DZ1 = -0.5, 1.6

PAPER = (240 / 255, 232 / 255, 216 / 255)
CREAM = '#e8dfcb'
INK = (58 / 255, 48 / 255, 38 / 255)
BRICK = (176 / 255, 64 / 255, 47 / 255)
SLATE = (66 / 255, 100 / 255, 138 / 255)

# Atom color domain: 100 ± 30 (three Loc+ SDs) onto the print heat ramp.
ATOM_LO, ATOM_HI = 70.0, 130.0

# Article arms: teams whose 2026 pitches count for each (deadline pickups
# carry their MLB clubs; PTeam 'AAA' = AAA-level pitches logged as the
# opponent in ROC games, e.g. Bird with SWB).
PITCHERS = {
    'Perales, Luis': ['ROC'],
    'Kent, Jackson': ['ROC'],
    'Sinclair, Jack': ['ROC'],
    'Tolman, Erik': ['ROC'],
    'Bird, Jake': ['NYY', 'ROC', 'AAA'],
    'Cruz, Yovanny': ['NYY', 'WSH', 'AAA'],
    'Dion, Will': ['CLE', 'WSH', 'ROC', 'AAA'],
}
MIN_TYPE_N = 25  # skip per-type maps below this many graded pitches


def heat_color(t):
    """Print scale: slate (good/cold) -> paper -> brick (costly/hot)."""
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def draw_map(ax, cells, n_max):
    """cells: {(i, j): (color_t, n)} on the pipeline grid."""
    ax.set_xlim(DX0, DX1)
    ax.set_ylim(DZ0, DZ1)
    ax.set_aspect((DX1 - DX0) / (DZ1 - DZ0) / (250 / 300))  # JS canvas aspect
    ax.axis('off')
    ax.add_patch(Rectangle((DX0, DZ0), DX1 - DX0, DZ1 - DZ0,
                           facecolor=CREAM, edgecolor='none', zorder=0))
    for (i, j), (t, n) in cells.items():
        xc = X_MIN + (i + 0.5) * BIN_X
        zc = Z_MIN + (j + 0.5) * BIN_Z
        alpha = 0.2 + 0.8 * math.sqrt(n / n_max)
        ax.add_patch(Rectangle((xc - BIN_X / 2, zc - BIN_Z / 2), BIN_X, BIN_Z,
                               facecolor=heat_color(t), alpha=alpha,
                               edgecolor='none', zorder=1))
    # Strike zone: zone-normalized 0..1, plate half-width 0.83 ft.
    ax.add_patch(Rectangle((-0.83, 0), 1.66, 1, fill=False,
                           edgecolor=(*INK, 0.65), linewidth=2, zorder=3))
    for k in (1, 2):
        ax.plot([-0.83 + k * 1.66 / 3] * 2, [0, 1],
                color=(*INK, 0.28), linewidth=0.8, zorder=3)
        ax.plot([-0.83, 0.83], [k / 3] * 2,
                color=(*INK, 0.28), linewidth=0.8, zorder=3)
    ax.add_patch(Polygon([(-8 / 60, -0.4), (8 / 60, -0.4), (5 / 60, -0.44),
                          (0, -0.465), (-5 / 60, -0.44)],
                         closed=True, facecolor=(*INK, 0.18),
                         edgecolor='none', zorder=3))


def bin_pitches(pitches, value_key='Loc+'):
    """Aggregate atoms onto the grid.

    Returns ({(i,j): (mean, n)}, n_graded, atom_mean). atom_mean covers ALL
    graded pitches, including those outside the display grid — clipping the
    far-off-plate pitches (the worst atoms) would inflate a grid-only mean.
    """
    acc = defaultdict(lambda: [0.0, 0])
    tot, tot_n = 0.0, 0
    for p in pitches:
        v = sf(p.get(value_key))
        px, pz = sf(p.get('PlateX')), sf(p.get('PlateZ'))
        top, bot = sf(p.get('SzTop')), sf(p.get('SzBot'))
        if v is None or px is None or pz is None or not top or not bot or top <= bot:
            continue
        tot += v
        tot_n += 1
        zn = (pz - bot) / (top - bot)
        i = int(math.floor((px - X_MIN) / BIN_X))
        j = int(math.floor((zn - Z_MIN) / BIN_Z))
        if not (0 <= i < round((X_MAX - X_MIN) / BIN_X)):
            continue
        if not (0 <= j < round((Z_MAX - Z_MIN) / BIN_Z)):
            continue
        acc[(i, j)][0] += v
        acc[(i, j)][1] += 1
    return ({k: (s / n, n) for k, (s, n) in acc.items()},
            tot_n, (tot / tot_n) if tot_n else None)


def render_png(path, title, meta, cells_mean, footer):
    if not cells_mean:
        return False
    n_max = max(n for _, n in cells_mean.values())
    cells = {k: ((ATOM_HI - m) / (ATOM_HI - ATOM_LO), n)
             for k, (m, n) in cells_mean.items()}
    fig, ax = plt.subplots(figsize=(5, 6), dpi=200)
    fig.patch.set_facecolor(CREAM)
    draw_map(ax, cells, n_max)
    ax.set_title(title, fontsize=13, color=INK, pad=14, **TITLE_FONT)
    ax.text(0.5, 1.015, meta, transform=ax.transAxes, ha='center',
            fontsize=8.5, color=(*INK, 0.75))
    ax.text(0.5, -0.03, footer, transform=ax.transAxes, ha='center',
            fontsize=7.5, color=(*INK, 0.6))
    fig.tight_layout()
    fig.savefig(path, facecolor=CREAM, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parity', action='store_true',
                    help='also render Perales from the shipped ExpRV grid')
    args = ap.parse_args()

    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    with open(PITCH_LB) as f:
        pitch_lb = json.load(f)
    with open(PITCHER_LB) as f:
        pitcher_lb = json.load(f)

    by_pitcher = defaultdict(list)
    for p in allp:
        teams = PITCHERS.get(p.get('Pitcher'))
        if teams and p.get('PTeam') in teams:
            by_pitcher[p['Pitcher']].append(p)

    # Printed Loc+ per (pitcher, type) for labels — prefer the row with the
    # largest sample (2TM > single club for the deadline pickups).
    printed = {}
    for r in pitch_lb:
        key = (r.get('pitcher'), r.get('pitchType'))
        if r.get('pitcher') in PITCHERS and r.get('locPlus') is not None:
            if key not in printed or r.get('count', 0) > printed[key][1]:
                printed[key] = (r['locPlus'], r.get('count', 0))

    footer = 'blue = good location · red = costly · opacity = usage'
    for name, pitches in sorted(by_pitcher.items()):
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)
        teams = ' + '.join(sorted({p['PTeam'] for p in pitches}))

        cells, n, season_loc = bin_pitches(pitches)
        render_png(os.path.join(outdir, f'CommandMap_{last}{first}_Season.png'),
                   f'{first} {last} — Loc+ Command Map',
                   f'2026 season ({teams}) · {n} graded pitches'
                   + (f' · Loc+ {season_loc:.0f}' if season_loc else ''),
                   cells, footer)

        by_type = defaultdict(list)
        for p in pitches:
            by_type[p.get('Pitch Type')].append(p)
        for pt, plist in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            if not pt:
                continue
            cells, n, atom_mean = bin_pitches(plist)
            if n < MIN_TYPE_N:
                print(f'  skip {name} {pt}: n={n} < {MIN_TYPE_N}')
                continue
            lp = printed.get((name, pt))
            meta = f'2026 season ({teams}) · {n} graded {pt}'
            if lp:
                meta += f' · {pt} Loc+ {lp[0]:.0f}'
            elif atom_mean is not None:
                meta += f' · {pt} Loc+ {atom_mean:.0f}'
            render_png(os.path.join(outdir, f'CommandMap_{last}{first}_{pt}.png'),
                       f'{first} {last} — {pt} Loc+ Command Map',
                       meta, cells, footer)
        print(f'{name}: season n={n_total(by_type)} -> {outdir}')

    if args.parity:
        row = next(r for r in pitcher_lb
                   if r.get('pitcher') == 'Perales, Luis' and r.get('team') == 'ROC')
        hm = row['locPlusHeatmap']
        n_max = max(c[3] for c in hm)
        LO, HI = -0.06, 0.06
        cells = {(c[0], c[1]): ((c[2] - LO) / (HI - LO), c[3]) for c in hm}
        fig, ax = plt.subplots(figsize=(5, 6), dpi=200)
        fig.patch.set_facecolor(CREAM)
        draw_map(ax, cells, n_max)
        ax.set_title('Perales — site ExpRV parity check', fontsize=13,
                     color=INK, pad=14, **TITLE_FONT)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_ROOT, 'Perales', 'parity_exprv.png'),
                    facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        print('parity map written')


def n_total(by_type):
    return sum(len(v) for v in by_type.values())


if __name__ == '__main__':
    sys.exit(main())
