#!/usr/bin/env python3
"""Render per-pitch-type Loc+ attack-zone pages (article arms).

Layman-readable redesign: a small labeled diagram of the five attack zones
(Heart / Shadow-In / Shadow-Out / Chase / Waste) sits in the header, and each
pitch type gets a five-row bar panel:

  - bar length  = share of his pitches landing in that zone
  - black tick  = the MLB share for that pitch group (the "normal" mix)
  - bar color + number = the average location grade (Loc+ atom) of his
    pitches there, vs the league's average grade in small text

Grades come from the per-pitch Loc+ atoms cached on every pitch row, zones
from pipeline_sdplus.classify_zone. Coherent canon: the share-weighted mean
of the five zone grades reproduces the printed pitch Loc+ exactly (asserted).

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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_sdplus import classify_zone  # noqa: E402
from pipeline_locplus import group_of_code  # noqa: E402

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

ATOM_LO, ATOM_HI = 70.0, 130.0   # 100 ± 3 SD color domain
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
ZONE_NAMES = {'heart': 'Heart', 'shadow_in': 'Shadow-In',
              'shadow_out': 'Shadow-Out', 'chase': 'Chase', 'waste': 'Waste'}
PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball',
               'KC': 'Knuckle-Curve', 'CH': 'Changeup', 'FS': 'Splitter'}

SHARE_MAX = 66   # x-axis span; bars use 0-50%, grades live at the right edge


def heat_color(t):
    """Percentile-heat ramp: t=1 brick (good grades), t=0 slate (costly)."""
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def draw_zone_legend(ax):
    """Small labeled map of the five zones — the only place geometry appears."""
    ZONE_X, HEART_X, SHADOW_X, CHASE_X = 0.83, 6.7 / 12, 13.3 / 12, 20.0 / 12
    DXl, Z0, Z1 = 1.95, -0.75, 1.75
    ax.set_xlim(-DXl, DXl)
    ax.set_ylim(Z0, Z1)
    ax.set_aspect('equal')
    ax.axis('off')
    shades = {'waste': (0.78, 0.73, 0.62), 'chase': (0.85, 0.80, 0.69),
              'shadow_out': (0.90, 0.86, 0.76), 'shadow_in': (0.94, 0.90, 0.81),
              'heart': (0.97, 0.94, 0.86)}
    rects = [('waste', (-DXl, Z0, 2 * DXl, Z1 - Z0)),
             ('chase', (-CHASE_X, -0.5, 2 * CHASE_X, 2.0)),
             ('shadow_out', (-SHADOW_X, -1 / 6, 2 * SHADOW_X, 1 + 2 / 6)),
             ('shadow_in', (-ZONE_X, 0, 2 * ZONE_X, 1.0)),
             ('heart', (-HEART_X, 1 / 6, 2 * HEART_X, 4 / 6))]
    for z, (x, y, w, h) in rects:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=shades[z],
                               edgecolor=(*INK, 0.4), linewidth=0.8))
    ax.add_patch(Rectangle((-ZONE_X, 0), 2 * ZONE_X, 1, fill=False,
                           edgecolor=(*INK, 0.85), linewidth=1.8))
    labels = [('Heart', 0, 0.5), ('Shadow-In', 0, 0.08),
              ('Shadow-Out', 0, 1.075), ('Chase', 0, 1.33), ('Waste', 0, 1.615)]
    for txt, x, y in labels:
        ax.text(x, y, txt, ha='center', va='center', fontsize=6.5,
                color=INK, fontweight=600)
    ax.set_title('the five zones (catcher view)', fontsize=7.5,
                 color=(*INK, 0.75), pad=4)


def draw_panel(ax, title, zone_stats, n_total, lg_stats):
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, SHARE_MAX)
    ax.set_ylim(-0.85, 4.65)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    for gx in (10, 20, 30, 40, 50):
        ax.axvline(gx, color=(*INK, 0.10), linewidth=0.7, zorder=1)
    ax.set_yticks(range(5))
    ax.set_yticklabels([ZONE_NAMES[z] for z in ZONES], fontsize=8, color=INK)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(['0%', '10', '20', '30', '40', '50'],
                       fontsize=6.5, color=(*INK, 0.6))
    ax.tick_params(axis='x', length=0)

    # Column headers so neither number can be misread.
    ax.text(0, -0.68, '% OF PITCHES THROWN THERE', fontsize=5.6,
            color=(*INK, 0.55), fontweight=600)
    ax.text(SHARE_MAX - 0.5, -0.68, 'GRADE · LG', fontsize=5.6, ha='right',
            color=(*INK, 0.55), fontweight=600)

    for i, z in enumerate(ZONES):
        mean, n = zone_stats.get(z, (None, 0))
        lg_share, lg_grade = lg_stats.get(z, (None, None))
        if mean is None:
            ax.text(1.0, i, 'none thrown', fontsize=6.5, va='center',
                    color=(*INK, 0.45), style='italic')
            continue
        share = 100.0 * n / n_total
        t = (mean - ATOM_LO) / (ATOM_HI - ATOM_LO)
        alpha = 1.0 if n >= FADE_N else 0.4
        ax.barh(i, share, height=0.62, color=heat_color(t), alpha=alpha,
                edgecolor=(*INK, 0.35), linewidth=0.7, zorder=2)
        if lg_share is not None:
            ax.plot([lg_share, lg_share], [i - 0.40, i + 0.40],
                    color=INK, linewidth=1.4, zorder=3)
            ax.text(lg_share, i + 0.47, f'lg {lg_share:.1f}%', fontsize=5.3,
                    ha='center', va='center', color=(*INK, 0.6), zorder=3)
        ax.text(max(share, lg_share or 0) + 1.2, i, f'{share:.1f}%',
                fontsize=7, va='center', fontweight=600,
                color=(*INK, 0.5 if n < FADE_N else 0.95), zorder=4)
        ax.text(59, i, f'{mean:.0f}', fontsize=8.5, ha='right', va='center',
                fontweight=700, color=(*INK, 0.5 if n < FADE_N else 1.0), zorder=4)
        if lg_grade is not None:
            ax.text(SHARE_MAX - 0.5, i, f'lg {lg_grade:.0f}', fontsize=6.3,
                    ha='right', va='center', color=(*INK, 0.55), zorder=4)
    ax.set_title(title, fontsize=10.5, color=INK, pad=9, loc='left', **TITLE_FONT)


def main():
    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)

    by_pitcher = defaultdict(list)
    # League zone shares + mean grades per (batter hand, pitch-type group),
    # MLB pitches only. hand 'ALL' pools both sides.
    lg_acc = {h: defaultdict(dict) for h in ('ALL', 'L', 'R')}
    for p in allp:
        teams = PITCHERS.get(p.get('Pitcher'))
        if teams and p.get('PTeam') in teams:
            by_pitcher[p['Pitcher']].append(p)
        v = sf(p.get('Loc+'))
        if p.get('_source') == 'MLB' and v is not None:
            zone = classify_zone(p)
            if zone is not None:
                hands = ['ALL'] + ([p.get('Bats')] if p.get('Bats') in ('L', 'R') else [])
                for h in hands:
                    for g in ('ALL', group_of_code(p.get('Pitch Type'))):
                        s, c = lg_acc[h][g].setdefault(zone, [0.0, 0])
                        lg_acc[h][g][zone] = [s + v, c + 1]
    lg_stats = {}   # hand -> grp -> zone -> (lg share %, lg mean grade)
    for h, groups in lg_acc.items():
        lg_stats[h] = {}
        for grp, zc in groups.items():
            tot = sum(c for _, c in zc.values())
            lg_stats[h][grp] = {z: (100.0 * c / tot, s / c) for z, (s, c) in zc.items()}

    how_to = ('How to read this: each bar is the percent of this pitch\u2019s throws\n'
              'that land in that zone; the black tick, labeled "lg __%", marks\n'
              'the MLB-average percent for that pitch family.\n'
              'The GRADE column scores the quality of those spots: 100 = MLB-\n'
              'average location, higher is better for the pitcher; "lg" = the\n'
              'league\u2019s own grade in that zone. Bar color mirrors the grade\n'
              '(red = good spots, blue = costly). Multiply each zone\u2019s percent\n'
              'by its grade and add them up: that is the pitch\u2019s Loc+.\n'
              'Faded bar = under 10 pitches.')

    PAGES = [('ALL', '', ''), ('L', '_vsLHH', ' vs LHH'), ('R', '_vsRHH', ' vs RHH')]

    for name, all_pitches in sorted(by_pitcher.items()):
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)
        teams = ' + '.join(sorted({p['PTeam'] for p in all_pitches}))

        for hand, suffix, hand_label in PAGES:
            pitches = (all_pitches if hand == 'ALL'
                       else [p for p in all_pitches if p.get('Bats') == hand])
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
            if not acc.get('ALL'):
                continue

            panels = [('ALL', acc['ALL'])]
            for pt, zs in sorted(((k, v) for k, v in acc.items() if k != 'ALL'),
                                 key=lambda kv: -sum(n for _, n in kv[1].values())):
                n_pt = sum(n for _, n in zs.values())
                if n_pt >= MIN_TYPE_N:
                    panels.append((pt, zs))
                else:
                    print(f'  skip {name}{hand_label} {pt}: n={n_pt} < {MIN_TYPE_N}')

            ncols = 3
            nrows = math.ceil(len(panels) / ncols)
            fig = plt.figure(figsize=(11.5, 2.7 + 2.35 * nrows), dpi=200)
            fig.patch.set_facecolor(CREAM)
            gs = GridSpec(nrows + 1, ncols, figure=fig,
                          height_ratios=[1.55] + [1.3] * nrows,
                          hspace=0.55, wspace=0.42,
                          left=0.075, right=0.97, top=0.90, bottom=0.075)

            ax_head = fig.add_subplot(gs[0, :2])
            ax_head.axis('off')
            vs_title = {'L': ' vs LHH', 'R': ' vs RHH'}
            ax_head.text(0, 0.98, f'{first} {last}{vs_title.get(hand, "")}: Where His '
                         'Pitches Land, and What Those Spots Are Worth',
                         fontsize=14, color=INK, va='top', **TITLE_FONT)
            sub = f'2026 season ({teams}) \u00b7 Loc+ by attack zone'
            if hand != 'ALL':
                sub += f'{hand_label} \u00b7 league ticks/grades = MLB{hand_label} only'
            ax_head.text(0, 0.60, sub, fontsize=8.5, color=(*INK, 0.7), va='top')
            ax_head.text(0, 0.42, how_to, fontsize=7.3, color=(*INK, 0.85),
                         va='top', linespacing=1.55)
            ax_leg = fig.add_subplot(gs[0, 2])
            draw_zone_legend(ax_leg)

            for i, (key, zs) in enumerate(panels):
                n_total = sum(n for _, n in zs.values())
                overall = sum(s for s, _ in zs.values()) / n_total
                recon = sum((s / n) * n for s, n in zs.values()) / n_total
                assert abs(recon - overall) < 1e-9
                stats = {z: (s / n, n) for z, (s, n) in zs.items()}
                label = 'All pitches' if key == 'ALL' else PITCH_NAMES.get(key, key)
                grp = 'ALL' if key == 'ALL' else group_of_code(key)
                r, c = divmod(i, ncols)
                ax = fig.add_subplot(gs[1 + r, c])
                draw_panel(ax, f'{label} \u00b7 Loc+ {overall:.0f} \u00b7 {n_total} pitches',
                           stats, n_total, lg_stats[hand].get(grp, {}))

            out = os.path.join(outdir, f'LocZones_{last}{first}{suffix}.png')
            fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
            plt.close(fig)
            print(name, hand_label or 'season', '->', out)


if __name__ == '__main__':
    main()
