#!/usr/bin/env python3
"""PROTOTYPE: zone x count Loc+ matrix recolored by IMPACT vs league.

Implements the four-part redesign:
  1. Every cell shows its impact in Loc+ points: his joint share times his
     (shrunken) grade there, minus the league's share times the league grade.
     Cell impacts sum to (his Loc+ - league grade) for the pitch, and the
     background diverging color encodes impact (red helps him, blue costs).
  2. Usage percentages are CONDITIONAL within each count column (columns sum
     to 100%); the count state's own share vs league moves to the column head.
  3. Automated callouts: the two most helpful and two most costly cells on
     the page, written as sentences, tagged by whether usage or execution
     drives them.
  4. Cell grades are shrunk toward the league cell grade with k = 8 before
     impacts are computed. k was tuned by split-half predictive sweep
     (scripts/loczone_impact_shrink_sweep.py, interior optimum bracketed:
     r 0.3137 at k=0, 0.3169 at k=8, 0.3109 at k=110; flat 5-12).

Season pages only (prototype). Usage: python3 scripts/render_loc_zones_impact.py
Outputs ~/Downloads/ArticleVisuals/<Last>/LocZones_<LastFirst>_impact.png
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
CREAM_RGB = (232 / 255, 223 / 255, 203 / 255)
INK = (58 / 255, 48 / 255, 38 / 255)
BRICK = (176 / 255, 64 / 255, 47 / 255)
SLATE = (66 / 255, 100 / 255, 138 / 255)

SHRINK_K = 8          # swept: scripts/loczone_impact_shrink_sweep.py
IMPACT_SPAN = 3.0     # color domain: +-3 Loc+ points
FADE_N = 10
MIN_TYPE_N = 25

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
STATES = ['ahead', 'even', 'behind']
STATE_NAMES = {'ahead': 'AHEAD', 'even': 'EVEN', 'behind': 'BEHIND'}
STATE_PROSE = {'ahead': 'while ahead', 'even': 'in even counts',
               'behind': 'while behind'}
PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball',
               'KC': 'Knuckle-Curve', 'CH': 'Changeup', 'FS': 'Splitter'}


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def count_state(p):
    try:
        b, s = (int(x) for x in str(p.get('Count', '')).split('-'))
    except ValueError:
        return None
    return 'ahead' if s > b else ('behind' if b > s else 'even')


def heat_color(t):
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def draw_zone_legend(ax):
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
    for txt, x, y in [('Heart', 0, 0.5), ('Shadow-In', 0, 0.08),
                      ('Shadow-Out', 0, 1.075), ('Chase', 0, 1.33),
                      ('Waste', 0, 1.615)]:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                color=INK, fontweight=600)
    ax.set_title('the five zones (catcher view)', fontsize=9,
                 color=(*INK, 0.75), pad=5)


def cell_impacts(acc, n_total, lg):
    """Per-cell dict -> impact stats. Returns {cell: dict} plus totals."""
    out = {}
    for z in ZONES:
        for st in STATES:
            cell = (z, st)
            ls, lgg = lg['joint'].get(cell, (0.0, 0.0))
            s_sum, n = acc.get(cell, [0.0, 0])
            share = n / n_total if n_total else 0.0
            g_raw = (s_sum / n) if n else None
            g_shr = ((n * (s_sum / n) + SHRINK_K * lgg) / (n + SHRINK_K)
                     if n else lgg)
            impact = share * g_shr - ls * lgg
            mix = lgg * (share - ls)
            execu = share * (g_shr - lgg)
            out[cell] = {'n': n, 'share': share, 'g_raw': g_raw,
                         'g_shr': g_shr, 'lg_share': ls, 'lg_grade': lgg,
                         'impact': impact, 'mix': mix, 'exec': execu}
    return out


def driver_phrase(c):
    m, e = abs(c['mix']), abs(c['exec'])
    if m > 2 * e:
        return 'mostly usage'
    if e > 2 * m:
        return 'mostly execution'
    return 'usage and execution'


def draw_panel(ax, title, cells, state_n, lg, n_total, lg_tot):
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.95, 5.45)
    ax.invert_yaxis()
    ax.axis('off')
    for j, st in enumerate(STATES):
        hs = 100.0 * state_n.get(st, 0) / n_total if n_total else 0.0
        ls = 100.0 * lg['state'].get(st, 0.0)
        ax.text(j + 0.5, -0.52, STATE_NAMES[st], ha='center', fontsize=8,
                color=INK, fontweight=700)
        ax.text(j + 0.5, -0.18, f'{hs:.1f}% (lg {ls:.1f}%)',
                ha='center', fontsize=6.2, color=(*INK, 0.85))
    for i, z in enumerate(ZONES):
        ax.text(-0.06, i + 0.5, ZONE_NAMES[z], ha='right', va='center',
                fontsize=9, color=INK, fontweight=600)
        for j, st in enumerate(STATES):
            c = cells[(z, st)]
            t = 0.5 + c['impact'] / (2 * IMPACT_SPAN)
            face = heat_color(t)
            dim = 0 < c['n'] < FADE_N
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                   alpha=0.45 if dim else 1.0,
                                   edgecolor=(*INK, 0.25), linewidth=0.6))
            txt = CREAM_RGB if (abs(t - 0.5) > 0.28 and not dim) else INK
            star = '*' if dim else ''
            ax.text(j + 0.5, i + 0.24, f'{c["impact"]:.1f} pts{star}',
                    ha='center', va='center', fontsize=8.6, fontweight=700,
                    color=txt)
            sn = state_n.get(st, 0)
            cshare = 100.0 * c['n'] / sn if sn else 0.0
            lg_state = lg['state'].get(st, 0.0)
            lgc = (100.0 * c['lg_share'] / lg_state) if lg_state else 0.0
            sub_c = (*txt, 0.92)
            ax.text(j + 0.5, i + 0.53, f'{cshare:.1f}% (lg {lgc:.1f}%)',
                    ha='center', va='center', fontsize=6.2, color=sub_c)
            g_txt = f'{c["g_raw"]:.0f}' if c['g_raw'] is not None else '–'
            ax.text(j + 0.5, i + 0.78, f'Loc+ {g_txt} (lg {c["lg_grade"]:.0f})',
                    ha='center', va='center', fontsize=6.2, color=sub_c)
    total = sum(c['impact'] for c in cells.values())
    ax.text(0, 5.32, f'cells sum to {total:.1f} pts vs a league-average map '
            f'(league overall grade {lg_tot:.0f}) · * = under 10 pitches',
            fontsize=6.4, color=(*INK, 0.8))
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left', **TITLE_FONT)



def fit_font(text, base=11.0, max_chars=116):
    """Shrink fontsize when a header line would overflow its text area."""
    n = len(text)
    return base if n <= max_chars else max(7.2, base * max_chars / n)


def main():
    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)

    by_pitcher = defaultdict(list)
    lg_acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
    lg_state_n = defaultdict(lambda: defaultdict(int))
    lg_n = defaultdict(int)
    for p in allp:
        teams = PITCHERS.get(p.get('Pitcher'))
        if teams and p.get('PTeam') in teams:
            by_pitcher[p['Pitcher']].append(p)
        if p.get('_source') != 'MLB':
            continue
        v = sf(p.get('Loc+'))
        if v is None:
            continue
        z = classify_zone(p)
        st = count_state(p)
        if z is None or st is None:
            continue
        hands = ['ALL'] + ([p.get('Bats')] if p.get('Bats') in ('L', 'R') else [])
        for h in hands:
            for g in ('ALL', group_of_code(p.get('Pitch Type'))):
                cell = lg_acc[(h, g)][(z, st)]
                cell[0] += v
                cell[1] += 1
                cell[2] += v * v
                lg_state_n[(h, g)][st] += 1
                lg_n[(h, g)] += 1

    lg = {h: {} for h in ('ALL', 'L', 'R')}
    for (h, grp), cells in lg_acc.items():
        tot = lg_n[(h, grp)]
        joint = {c: (n / tot, s / n) for c, (s, n, _) in cells.items()}
        sd = {c: math.sqrt(max(ss / n - (s / n) ** 2, 0.0))
              for c, (s, n, ss) in cells.items() if n}
        lg[h][grp] = {'joint': joint, 'sd': sd,
                      'state': {st: lg_state_n[(h, grp)][st] / tot for st in STATES},
                      'total': sum(sh * g for sh, g in joint.values())}

    PAGES = [('ALL', '', ''), ('L', '_vsLHH', ' vs LHH'), ('R', '_vsRHH', ' vs RHH')]

    for name, all_pitches in sorted(by_pitcher.items()):
      last, first = [s.strip() for s in name.split(',')]
      outdir = os.path.join(OUT_ROOT, last)
      os.makedirs(outdir, exist_ok=True)
      teams = ' + '.join(sorted({p['PTeam'] for p in all_pitches}))
      for hand, suffix, hand_label in PAGES:
        pitches = (all_pitches if hand == 'ALL'
                   else [p for p in all_pitches if p.get('Bats') == hand])
        if not pitches:
            continue
        lgh = lg[hand]

        acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        state_n = defaultdict(lambda: defaultdict(int))
        for p in pitches:
            v = sf(p.get('Loc+'))
            if v is None:
                continue
            z = classify_zone(p)
            st = count_state(p)
            if z is None or st is None:
                continue
            for key in ('ALL', p.get('Pitch Type') or '?'):
                acc[key][(z, st)][0] += v
                acc[key][(z, st)][1] += 1
                state_n[key][st] += 1

        panels = [('ALL', acc['ALL'])]
        for pt, zs in sorted(((k, v) for k, v in acc.items() if k != 'ALL'),
                             key=lambda kv: -sum(n for _, n in kv[1].values())):
            if sum(n for _, n in zs.values()) >= MIN_TYPE_N:
                panels.append((pt, zs))

        # Page-level callouts from per-type cells (ALL excluded), ranked by
        # impact discounted 0.6 standard errors (split-half tuned; see
        # scripts/loczone_callout_sweep.py -- realized-spread optimum at
        # lam=0.6 with the existing 25-pitch panel gate as the floor).
        CALLOUT_LAM = 0.6
        ranked = []
        type_sums = []
        for key, zs in panels[1:]:
            n_total = sum(n for _, n in zs.values())
            grp = group_of_code(key)
            cells = cell_impacts(zs, n_total, lgh[grp])
            type_sums.append((sum(c['impact'] for c in cells.values()), key, n_total))
            for (z, st), c in cells.items():
                sd = lgh[grp]['sd'].get((z, st), 0.0)
                se_g = sd / math.sqrt(c['n']) if c['n'] else 0.0
                se_s = math.sqrt(max(c['share'] * (1 - c['share']), 1e-9) / n_total)
                se = math.sqrt((c['share'] * se_g) ** 2
                               + (c['lg_grade'] * se_s) ** 2)
                d = abs(c['impact']) - CALLOUT_LAM * se
                score = math.copysign(max(0.0, d), c['impact']) if d > 0 else 0.0
                ranked.append((score, c['impact'], key, z, st, c))
        ranked.sort(key=lambda r: r[0])
        costs = [r for r in ranked[:2] if r[0] < 0]
        helps = [r for r in ranked[-2:][::-1] if r[0] > 0]

        def sentence(r):
            _, imp, key, z, st, c = r
            verb = 'adds' if imp > 0 else 'costs'
            return (f'{PITCH_NAMES.get(key, key)} to {ZONE_NAMES[z]} '
                    f'{STATE_PROSE[st]}: {verb} {abs(imp):.1f} pts '
                    f'({driver_phrase(c)})')

        mm_cols = 3
        mm_rows = math.ceil(len(panels) / mm_cols)
        fig = plt.figure(figsize=(17.0, 3.6 + 5.2 * mm_rows), dpi=200)
        fig.patch.set_facecolor(CREAM)
        gs = GridSpec(mm_rows + 1, mm_cols, figure=fig,
                      height_ratios=[0.78] + [1.3] * mm_rows,
                      hspace=0.30, wspace=0.35,
                      left=0.055, right=0.985, top=0.955, bottom=0.04)
        hd = fig.add_subplot(gs[0, :2])
        hd.axis('off')
        hd.text(0, 0.97, f'{first} {last}{hand_label}: What His Locations Add and Cost',
                fontsize=23, color=INK, va='top', **TITLE_FONT)
        sub = (f'2026 season ({teams}){hand_label} · zone × count impact '
               f'matrix · shrinkage k={SHRINK_K}, callouts impact minus 0.6 SE (both split-half tuned)')
        if hand != 'ALL':
            sub += f' · league ={hand_label} only'
        hd.text(0, 0.78, sub, fontsize=fit_font(sub, base=11.5), color=(*INK, 0.75), va='top')
        hd.text(0, 0.68,
                'Each cell: its IMPACT in Loc+ points, his usage times his grade '
                'there minus the league’s usage times the league grade; red adds '
                'value, blue costs it,\nand the 15 cells sum to the gap between his '
                'Loc+ and league. Usage percents are within-column: of his pitches '
                'in that count state, how often that zone\n(league in parentheses); '
                'the column header shows how often he is in that count at all. '
                'Loc+ line shows the unshrunken grade of those pitches.',
                fontsize=9.8, color=INK, va='top', linespacing=1.6)
        helps_line = 'Helps him most:  ' + '   |   '.join(sentence(r) for r in helps)
        hd.text(0, 0.28, helps_line, fontsize=fit_font(helps_line),
                color=(140 / 255, 52 / 255, 38 / 255), va='top', fontweight=700)
        costs_line = 'Costs him most:  ' + '   |   '.join(sentence(r) for r in costs)
        hd.text(0, 0.09, costs_line, fontsize=fit_font(costs_line),
                color=(52 / 255, 80 / 255, 110 / 255), va='top', fontweight=700)
        if type_sums:
            best_map = max(type_sums, key=lambda t: t[0])
            worst_map = min(type_sums, key=lambda t: t[0])
            maps_line = (f'Whole pitch maps: best {PITCH_NAMES.get(best_map[1], best_map[1])} '
                         f'{best_map[0]:.1f} pts ({best_map[2]} pitches) · worst '
                         f'{PITCH_NAMES.get(worst_map[1], worst_map[1])} {worst_map[0]:.1f} pts '
                         f'({worst_map[2]} pitches)')
            hd.text(0, -0.07, maps_line, fontsize=fit_font(maps_line, base=10.5),
                    color=INK, va='top', fontweight=600)
        lg_ax = fig.add_subplot(gs[0, 2])  # tall header row -> big legend
        draw_zone_legend(lg_ax)

        for i, (key, zs) in enumerate(panels):
            n_total = sum(n for _, n in zs.values())
            overall = sum(s for s, _ in zs.values()) / n_total
            grp = 'ALL' if key == 'ALL' else group_of_code(key)
            cells = cell_impacts(zs, n_total, lgh[grp])
            label = 'All pitches' if key == 'ALL' else PITCH_NAMES.get(key, key)
            r, c = divmod(i, mm_cols)
            ax = fig.add_subplot(gs[1 + r, c])
            draw_panel(ax, f'{label} · Loc+ {overall:.0f} · {n_total} pitches',
                       cells, state_n[key], lgh[grp], n_total, lgh[grp]['total'])

        out = os.path.join(outdir, f'LocZones_{last}{first}_impact{suffix}.png')
        fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        print(name, hand_label or 'season', '->', out)


if __name__ == '__main__':
    main()
