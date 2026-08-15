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
from pipeline.sdplus.classify_zone. Coherent canon: the share-weighted mean
of the five zone grades reproduces the printed pitch Loc+ exactly (asserted).

Usage: python3 scripts/tools/render_loc_zones.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pipeline.sdplus import classify_zone  # noqa: E402
from pipeline.locplus import group_of_code  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT_ROOT = os.path.expanduser('~/Downloads/ArticleVisuals')

for f in os.listdir(os.path.join(ROOT, 'assets', 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(ROOT, 'assets', 'fonts', f))
plt.rcParams['font.family'] = 'IBM Plex Sans'
TITLE_FONT = {'fontfamily': 'Bitter', 'fontweight': 700}

PAPER = (240 / 255, 232 / 255, 216 / 255)
CREAM = '#e8dfcb'
CREAM_RGB = (232 / 255, 223 / 255, 203 / 255)
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

COUNT_STATES = ['ahead', 'even', 'behind']
COUNT_NAMES = {'ahead': 'Ahead in count', 'even': 'Even count',
               'behind': 'Behind in count'}


def count_state(p):
    """Pitcher's count state from 'balls-strikes'."""
    try:
        b, s = (int(x) for x in str(p.get('Count', '')).split('-'))
    except ValueError:
        return None
    return 'ahead' if s > b else ('behind' if b > s else 'even')

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
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                color=INK, fontweight=600)
    ax.set_title('the five zones (catcher view)', fontsize=9,
                 color=(*INK, 0.75), pad=5)


ROW = 1.65   # vertical spacing between zone rows (bar height stays 0.62)


def draw_panel(ax, title, zone_stats, count_stats, n_total, lg_zone, lg_count):
    n_rows_z, n_rows_c = 5, 3
    cbase = n_rows_z * ROW + 1.9          # y of first count row
    y_max = cbase + (n_rows_c - 1) * ROW + 0.85
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, SHARE_MAX)
    ax.set_ylim(-1.0, y_max)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    for gx in (10, 20, 30, 40, 50):
        ax.axvline(gx, color=(*INK, 0.10), linewidth=0.7, zorder=1)
    ys = [i * ROW for i in range(n_rows_z)] + [cbase + j * ROW for j in range(n_rows_c)]
    ax.set_yticks(ys)
    ax.set_yticklabels([ZONE_NAMES[z] for z in ZONES]
                       + [COUNT_NAMES[c] for c in COUNT_STATES],
                       fontsize=8, color=INK)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(['0%', '10', '20', '30', '40', '50'],
                       fontsize=6.5, color=(*INK, 0.6))
    ax.tick_params(axis='x', length=0)

    ax.text(0, -0.80, '% OF PITCHES · BY ZONE (WHERE)', fontsize=5.6,
            color=(*INK, 0.55), fontweight=600)
    ax.text(SHARE_MAX - 0.5, -0.80, 'GRADE · LG', fontsize=5.6, ha='right',
            color=(*INK, 0.55), fontweight=600)
    div_y = n_rows_z * ROW + 0.45
    ax.axhline(div_y, xmin=0.0, xmax=1.0, color=(*INK, 0.18), linewidth=0.8)
    ax.text(0, div_y + 0.62, '% OF PITCHES · BY COUNT (WHEN)', fontsize=5.6,
            color=(*INK, 0.55), fontweight=600)
    ax.text(SHARE_MAX - 0.5, div_y + 0.62, 'GRADE · LG', fontsize=5.6, ha='right',
            color=(*INK, 0.55), fontweight=600)

    def row(y, key, stats, lg_stats):
        mean, n = stats.get(key, (None, 0))
        lg_share, lg_grade = lg_stats.get(key, (None, None))
        if mean is None:
            ax.text(1.0, y, 'none thrown', fontsize=6.5, va='center',
                    color=(*INK, 0.45), style='italic')
            return
        share = 100.0 * n / n_total
        t = (mean - ATOM_LO) / (ATOM_HI - ATOM_LO)
        alpha = 1.0 if n >= FADE_N else 0.4
        ax.barh(y, share, height=0.62, color=heat_color(t), alpha=alpha,
                edgecolor=(*INK, 0.35), linewidth=0.7, zorder=2)
        if lg_share is not None:
            ax.plot([lg_share, lg_share], [y - 0.40, y + 0.40],
                    color=INK, linewidth=1.4, zorder=3)
            ax.text(lg_share, y + 0.62, f'lg {lg_share:.1f}%', fontsize=5.5,
                    ha='center', va='center', color=(*INK, 0.9), zorder=3)
        if share > 40:
            # Long bar: label inside its right end so it can't hit the grades.
            txt_color = CREAM if abs(t - 0.5) > 0.28 else INK
            ax.text(share - 0.9, y, f'{share:.1f}%', fontsize=7, va='center',
                    ha='right', fontweight=600, color=txt_color, zorder=4)
        else:
            ax.text(max(share, lg_share or 0) + 1.2, y, f'{share:.1f}%',
                    fontsize=7, va='center', fontweight=600,
                    color=(*INK, 0.5 if n < FADE_N else 0.95), zorder=4)
        ax.text(59, y, f'{mean:.0f}', fontsize=8.5, ha='right', va='center',
                fontweight=700, color=(*INK, 0.5 if n < FADE_N else 1.0), zorder=4)
        if lg_grade is not None:
            ax.text(SHARE_MAX - 0.5, y, f'lg {lg_grade:.0f}', fontsize=6.3,
                    ha='right', va='center', color=(*INK, 0.55), zorder=4)

    for i, z in enumerate(ZONES):
        row(i * ROW, z, zone_stats, lg_zone)
    for j, c in enumerate(COUNT_STATES):
        row(cbase + j * ROW, c, count_stats, lg_count)
    ax.set_title(title, fontsize=10.5, color=INK, pad=9, loc='left', **TITLE_FONT)



def draw_matrix_panel(ax, title, cell_stats, n_total, lg_cells):
    """Zone x count matrix. Cell: usage% (lg avg) over Loc+ (lg avg)."""
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.75, 5)
    ax.invert_yaxis()
    ax.axis('off')
    for j, cs in enumerate(COUNT_STATES):
        ax.text(j + 0.5, -0.28, COUNT_NAMES[cs].replace(' in count', '').replace(' count', '').upper(),
                ha='center', fontsize=7.5, color=(*INK, 0.7), fontweight=600)
    for i, z in enumerate(ZONES):
        ax.text(-0.06, i + 0.5, ZONE_NAMES[z], ha='right', va='center',
                fontsize=8.5, color=INK)
        for j, cs in enumerate(COUNT_STATES):
            mean, n = cell_stats.get((z, cs), (None, 0))
            if mean is None or n == 0:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=PAPER,
                                       edgecolor=(*INK, 0.25), linewidth=0.6))
                ax.text(j + 0.5, i + 0.5, '\u2013', ha='center', va='center',
                        fontsize=8, color=(*INK, 0.4))
                continue
            share = 100.0 * n / n_total
            lg_share, lg_grade = lg_cells.get((z, cs), (None, None))
            t = (mean - ATOM_LO) / (ATOM_HI - ATOM_LO)
            face = heat_color(t)
            dim = n < FADE_N
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                   alpha=0.45 if dim else 1.0,
                                   edgecolor=(*INK, 0.25), linewidth=0.6))
            txt = CREAM_RGB if (abs(t - 0.5) > 0.28 and not dim) else INK
            main_c = (*txt, 0.55) if dim else txt
            sub_c = (*txt, 0.5) if dim else (*txt, 0.85)
            lg_s = f' (lg avg {lg_share:.1f}%)' if lg_share is not None else ''
            ax.text(j + 0.5, i + 0.38, f'{share:.1f}%{lg_s}', ha='center',
                    va='center', fontsize=6.9, fontweight=700, color=main_c)
            lg_g = f' (lg avg {lg_grade:.0f})' if lg_grade is not None else ''
            ax.text(j + 0.5, i + 0.62, f'Loc+ {mean:.0f}{lg_g}', ha='center',
                    va='center', fontsize=6.1, color=sub_c)
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left', **TITLE_FONT)


def main():
    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)

    by_pitcher = defaultdict(list)
    # League zone shares + mean grades per (batter hand, pitch-type group),
    # MLB pitches only. hand 'ALL' pools both sides.
    lg_acc = {h: defaultdict(dict) for h in ('ALL', 'L', 'R')}
    lg_cacc = {h: defaultdict(dict) for h in ('ALL', 'L', 'R')}
    lg_macc = {h: defaultdict(dict) for h in ('ALL', 'L', 'R')}
    for p in allp:
        teams = PITCHERS.get(p.get('Pitcher'))
        if teams and p.get('PTeam') in teams:
            by_pitcher[p['Pitcher']].append(p)
        v = sf(p.get('Loc+'))
        if p.get('_source') == 'MLB' and v is not None:
            zone = classify_zone(p)
            state = count_state(p)
            if zone is not None:
                hands = ['ALL'] + ([p.get('Bats')] if p.get('Bats') in ('L', 'R') else [])
                for h in hands:
                    for g in ('ALL', group_of_code(p.get('Pitch Type'))):
                        s, c = lg_acc[h][g].setdefault(zone, [0.0, 0])
                        lg_acc[h][g][zone] = [s + v, c + 1]
                        if state is not None:
                            s2, c2 = lg_cacc[h][g].setdefault(state, [0.0, 0])
                            lg_cacc[h][g][state] = [s2 + v, c2 + 1]
                            s3, c3 = lg_macc[h][g].setdefault((zone, state), [0.0, 0])
                            lg_macc[h][g][(zone, state)] = [s3 + v, c3 + 1]

    def finalize(accs):
        out = {}
        for h, groups in accs.items():
            out[h] = {}
            for grp, kc in groups.items():
                tot = sum(c for _, c in kc.values())
                out[h][grp] = {k: (100.0 * c / tot, s / c) for k, (s, c) in kc.items()}
        return out

    lg_stats = finalize(lg_acc)     # hand -> grp -> zone -> (share %, grade)
    lg_cstats = finalize(lg_cacc)   # hand -> grp -> count state -> (share %, grade)
    lg_mstats = finalize(lg_macc)   # hand -> grp -> (zone, state) -> (share %, grade)

    how_to = ('How to read this: each bar is the percent of this pitch\u2019s throws\n'
              'that land in that zone; the black tick, labeled "lg __%", marks\n'
              'the MLB-average percent for that pitch family.\n'
              'The GRADE column scores the quality of those spots: 100 = MLB-\n'
              'average location, higher is better for the pitcher; "lg" = the\n'
              'league\u2019s own grade in that zone. Bar color mirrors the grade\n'
              '(red = good spots, blue = costly). Multiply each zone\u2019s percent\n'
              'by its grade and add them up: that is the pitch\u2019s Loc+.\n'
              'The BY COUNT rows split the SAME pitches by count instead of\n'
              'zone. Falling behind makes every location worth less, so a\n'
              'bloated "Behind" bar with a blue grade is a location problem\n'
              'all by itself. Faded bar = under 10 pitches.')

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
            cacc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
            macc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
            for p in pitches:
                v = sf(p.get('Loc+'))
                if v is None:
                    continue
                zone = classify_zone(p)
                if zone is None:
                    continue
                state = count_state(p)
                for key in ('ALL', p.get('Pitch Type') or '?'):
                    acc[key][zone][0] += v
                    acc[key][zone][1] += 1
                    if state is not None:
                        cacc[key][state][0] += v
                        cacc[key][state][1] += 1
                        macc[key][(zone, state)][0] += v
                        macc[key][(zone, state)][1] += 1
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
            fig = plt.figure(figsize=(11.5, 2.9 + 4.9 * nrows), dpi=200)
            fig.patch.set_facecolor(CREAM)
            gs = GridSpec(nrows + 1, ncols, figure=fig,
                          height_ratios=[0.85] + [1.3] * nrows,
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
                cstats = {c: (s / n, n) for c, (s, n) in cacc[key].items()}
                label = 'All pitches' if key == 'ALL' else PITCH_NAMES.get(key, key)
                grp = 'ALL' if key == 'ALL' else group_of_code(key)
                r, c = divmod(i, ncols)
                ax = fig.add_subplot(gs[1 + r, c])
                draw_panel(ax, f'{label} \u00b7 Loc+ {overall:.0f} \u00b7 {n_total} pitches',
                           stats, cstats, n_total, lg_stats[hand].get(grp, {}),
                           lg_cstats[hand].get(grp, {}))

            out = os.path.join(outdir, f'LocZones_{last}{first}{suffix}.png')
            fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
            plt.close(fig)
            print(name, hand_label or 'season', '->', out)

            mm_cols = 2
            mm_rows = math.ceil(len(panels) / mm_cols)
            f2 = plt.figure(figsize=(12.0, 2.6 + 6.6 * mm_rows), dpi=200)
            f2.patch.set_facecolor(CREAM)
            g2 = GridSpec(mm_rows + 1, mm_cols + 1, figure=f2,
                          height_ratios=[0.42] + [1.3] * mm_rows,
                          width_ratios=[1.0, 1.0, 0.72],
                          hspace=0.28, wspace=0.4,
                          left=0.075, right=0.985, top=0.94, bottom=0.04)
            h2 = f2.add_subplot(g2[0, :2])
            h2.axis('off')
            h2.text(0, 0.95, f'{first} {last}{vs_title.get(hand, "")}: The Same Spot '
                    'Changes Value With the Count', fontsize=15, color=INK,
                    va='top', **TITLE_FONT)
            sub2 = f'2026 season ({teams}) \u00b7 zone \u00d7 count matrix'
            if hand != 'ALL':
                sub2 += f'{hand_label} \u00b7 league averages = MLB{hand_label} only'
            h2.text(0, 0.62, sub2, fontsize=8.5, color=(*INK, 0.7), va='top')
            h2.text(0, 0.42,
                    'Each cell: how often the pitch is thrown to that zone in that count '
                    'state, and the Loc+ of those pitches, each with the\nMLB average for '
                    'that pitch family in parentheses. Background color follows the Loc+ '
                    '(red = good spots for that count,\nblue = costly \u00b7 100 = MLB-average '
                    'location \u00b7 faded = under 10 pitches). Compare a row across its three columns.',
                    fontsize=7.4, color=(*INK, 0.85), va='top', linespacing=1.55)
            l2 = f2.add_subplot(g2[0, 2])
            draw_zone_legend(l2)
            for i, (key, zs) in enumerate(panels):
                n_total = sum(n for _, n in zs.values())
                overall = sum(s for s, _ in zs.values()) / n_total
                cells = {k: (s / n, n) for k, (s, n) in macc[key].items()}
                label = 'All pitches' if key == 'ALL' else PITCH_NAMES.get(key, key)
                grp = 'ALL' if key == 'ALL' else group_of_code(key)
                r, c = divmod(i, mm_cols)
                axm2 = f2.add_subplot(g2[1 + r, c])
                draw_matrix_panel(axm2, f'{label} \u00b7 Loc+ {overall:.0f} \u00b7 '
                                  f'{n_total} pitches', cells, n_total,
                                  lg_mstats[hand].get(grp, {}))
            out2 = os.path.join(outdir, f'LocZones_{last}{first}_matrix{suffix}.png')
            f2.savefig(out2, facecolor=CREAM, bbox_inches='tight')
            plt.close(f2)
            print(name, 'matrix', hand_label or 'season', '->', out2)


if __name__ == '__main__':
    main()
