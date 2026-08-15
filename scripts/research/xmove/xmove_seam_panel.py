#!/usr/bin/env python3
"""PROTOTYPE — cross-axis break: how much of a pitch spin cannot explain.

Cross-axis break is the component of the observed break perpendicular to the
MEASURED release spin axis. Gyro spin scales break magnitude without rotating
it, and IVB is already gravity-removed, so anything perpendicular to the axis is
non-Magnus: seam-shifted wake.

Why it is worth its own panel rather than another column on the expected-movement
plate:

  * No model. No fitted expectation, no league baseline, no regression. Two
    measured columns and trigonometry, so there is nothing to be wrong.
  * No pitch type. It never reads the tag, so retagging cannot move it. That is
    the property the expected-movement work spent all day failing to get.
  * It answers a question the plate cannot: not "did the ball go where the
    release predicted" but "how much of this break is seams".

Every class sits on ONE shared axis, because the finding this was built to show
is that a pitch can sit in another class's territory. Medina's slider carries
6.7" of cross-axis break where league sliders average 0.1" and sweepers 7.3":
by seam behaviour it is a sweeper, whatever the tag says.

League distributions: 2021-2025, pitcher x hand x type x season, >= 50 pitches.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_seam_panel.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from cards.pitcher import (BG, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY,  # noqa: E402
                   TEXT_MUTED, TEXT_FAINT, PITCH_COLORS)
from xmove_compare import load_np  # noqa: E402
from xmove_medina_plot import PITCH_NAMES  # noqa: E402


def load_pitcher(name, team):
    """Any pitcher's season from the retagged sheet, not just Medina."""
    import gspread
    from scrapers.sheets_append import _workbook_id_for_team
    from pipeline.utils import break_tilt_to_minutes
    gc = gspread.service_account()
    ws = gc.open_by_key(_workbook_id_for_team(team)).worksheet(team)
    rows = [r for r in ws.get_all_records()
            if str(r.get('Pitcher', '')).strip() == name]
    if not rows:
        raise SystemExit(f'no rows for {name} on {team}')
    d = pd.DataFrame(rows)
    mins = d['RTilt'].map(break_tilt_to_minutes)
    d['SpinAxis'] = (180.0 + (mins % 720) * 0.5) % 360.0
    for c in ['xIndVrtBrk', 'xHorzBrk', 'SpinAxis']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'SpinAxis'])


def diverging(v, lo=-6.0, hi=8.0):
    """Glove side to arm side, neutral at zero. Okabe-Ito blue / vermillion so
    the ramp states what the number measures instead of asserting good or bad."""
    import matplotlib.colors as mc
    neg, mid, pos = mc.to_rgb('#0072B2'), mc.to_rgb('#cfc4ae'), mc.to_rgb('#D55E00')
    if v < 0:
        t = max(-1.0, v / lo * -1.0) if lo else 0.0
        t = min(1.0, abs(v) / abs(lo))
        return tuple(mid[i] + (neg[i] - mid[i]) * t for i in range(3))
    t = min(1.0, v / hi)
    return tuple(mid[i] + (pos[i] - mid[i]) * t for i in range(3))

def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


PITCHER = _arg('--pitcher', 'Medina, Luis')
TEAM = _arg('--team', 'ATH')
DIVERGING = '--diverging' in sys.argv
OUT = os.path.expanduser(_arg('--out', '~/Downloads/seam_break.png'))
DISP, BODY, COND = 'Bitter', 'IBM Plex Sans', 'IBM Plex Sans Condensed'
PTS = ['FC', 'FF', 'CU', 'SL', 'CH', 'FS', 'SI', 'ST']
UNIT_MIN = 50


def darken(col, f=0.30):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(col)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def cross_of(ivb, hb, spin_axis, sign):
    """Break perpendicular to the measured release axis, hand-signed."""
    th = np.radians(((spin_axis - 180.0) % 360.0) * sign)
    hb_s = hb * sign
    return -ivb * np.sin(th) + hb_s * np.cos(th)


def main():
    print('Loading 2021-2025...', file=sys.stderr)
    A = load_np()
    lg = pd.DataFrame({
        'pitcher': A['pitcher'], 'thr': A['thr'], 'pt': A['pt'],
        'season': A['season'], 'cross': A['cross'],
    }).groupby(['pitcher', 'thr', 'pt', 'season']).agg(
        n=('cross', 'size'), cross=('cross', 'mean')).reset_index()
    lg = lg[lg.n >= UNIT_MIN]

    d = load_pitcher(PITCHER, TEAM)
    hand = str(d['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0
    d = d.assign(cross=cross_of(d['xIndVrtBrk'].values, d['xHorzBrk'].values,
                                d['SpinAxis'].values, sign))
    his = {pt: g['cross'].mean() for pt, g in d.groupby('Pitch Type') if len(g) >= 25}

    order = [pt for pt in PTS if len(lg[lg.pt == pt]) >= 50]
    order.sort(key=lambda pt: lg[lg.pt == pt]['cross'].median())

    fig = plt.figure(figsize=(13.5, 8.8), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.text(0.045, 0.950, 'BREAK THAT SPIN CANNOT EXPLAIN', fontsize=26,
             fontfamily=DISP, fontweight='black', color=TEXT_PRIMARY, va='top')
    fig.text(0.045, 0.893,
             'Break measured perpendicular to the release axis. No model, no pitch type, '
             'nothing to retag.',
             fontsize=12, fontfamily=BODY, color=TEXT_MUTED, va='top')
    _last, _first = (PITCHER.split(',') + [''])[:2]
    fig.text(0.955, 0.950, f'{_first.strip()} {_last.strip()}'.strip().upper() +
             f'  ·  {TEAM}  ·  2026', fontsize=12.5,
             fontfamily=COND, fontweight='700', color=ACCENT, va='top', ha='right')
    fig.add_artist(Rectangle((0.045, 0.870), 0.910, 0.0016,
                             facecolor=ACCENT, edgecolor='none', alpha=0.85))

    ax = fig.add_axes([0.150, 0.185, 0.800, 0.640])
    ax.set_facecolor(BG)

    for i, pt in enumerate(order):
        g = lg[lg.pt == pt]['cross']
        p10, p25, p50, p75, p90 = np.percentile(g, [10, 25, 50, 75, 90])
        # Diverging mode colours by the VALUE, not by pitch identity: the ramp
        # then states glove side / arm side rather than asserting good / bad.
        c = diverging(p50) if DIVERGING else PITCH_COLORS.get(pt, '#777777')
        cd = darken(c, 0.18 if DIVERGING else 0.30)
        # box plot without the box: whisker, thick bar for the middle half, tick
        # at the median. Less ink than a violin and exact where it matters.
        ax.plot([p10, p90], [i, i], color=c, lw=1.4, alpha=0.55, zorder=3,
                solid_capstyle='round')
        ax.plot([p25, p75], [i, i], color=c, lw=7.5, alpha=0.40, zorder=4,
                solid_capstyle='round')
        ax.plot([p50, p50], [i - 0.16, i + 0.16], color=cd, lw=2.2, alpha=0.9,
                zorder=5)
        # Name and league median both live in the left gutter, so nothing
        # floats inside the plot where it can land on another row's whisker.
        ax.text(-0.014, i + 0.13, PITCH_NAMES.get(pt, pt),
                transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=12, fontfamily=BODY, fontweight='600', color=TEXT_PRIMARY)
        ax.text(-0.014, i - 0.20, f'league {p50:.1f}"',
                transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=8.5, fontfamily=BODY, color=TEXT_FAINT)

        if pt in his:
            v = his[pt]
            mk = darken(diverging(v), 0.18) if DIVERGING else cd
            ax.plot([v], [i], marker='o', ms=13, mfc=mk, mec=BG, mew=2.2, zorder=9)
            ax.annotate(f'{v:.1f}"   {(g < v).mean() * 100:.0f}th',
                        xy=(v, i), xytext=(0, 17), textcoords='offset points',
                        ha='center', fontsize=11, fontfamily=BODY,
                        fontweight='700', color=mk if DIVERGING else cd, zorder=10)

    ax.axvline(0, color=TEXT_FAINT, lw=1.0, alpha=0.6, zorder=1)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks([])
    ax.grid(True, axis='x', color=TEXT_FAINT, alpha=0.13, lw=0.7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5, length=0)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(BODY)
    ax.set_xlabel('cross-axis break (inches)', fontsize=10.5, fontfamily=BODY,
                  color=TEXT_MUTED, labelpad=10)
    ax.text(0.0, 1.022, 'glove side of the axis', transform=ax.transAxes,
            fontsize=9.5, fontfamily=BODY, color=TEXT_FAINT, ha='left', va='bottom')
    ax.text(1.0, 1.022, 'arm side of the axis', transform=ax.transAxes,
            fontsize=9.5, fontfamily=BODY, color=TEXT_FAINT, ha='right', va='bottom')

    # The finding this panel exists for.
    if 'SL' in his and 'ST' in order and 'SL' in order:
        sl_v = his['SL']
        st_med = lg[lg.pt == 'ST']['cross'].median()
        sl_med = lg[lg.pt == 'SL']['cross'].median()
        i_sl, i_st = order.index('SL'), order.index('ST')
        ax.annotate('', xy=(sl_v, i_st - 0.28), xytext=(sl_v, i_sl + 0.22),
                    arrowprops=dict(arrowstyle='-', color=ACCENT, lw=1.1,
                                    ls=(0, (3, 3)), alpha=0.75), zorder=8)
        fig.text(0.150, 0.075,
                 f'His slider carries {sl_v:.1f}" of cross-axis break. League sliders sit at '
                 f'{sl_med:.1f}", sweepers at {st_med:.1f}". By the physics it is a sweeper, '
                 f'whatever the tag says,\nand nothing here read the tag to find that out.',
                 fontsize=10.5, fontfamily=BODY, color=TEXT_SECONDARY, va='bottom',
                 ha='left', linespacing=1.7)
    fig.text(0.150, 0.032,
             'Percentile is the share of that pitch with LESS arm-side cross-axis break, so a '
             'four-seam sits low by being further to the glove side.',
             fontsize=9, fontfamily=BODY, color=TEXT_FAINT, va='bottom', ha='left')

    fig.savefig(OUT, facecolor=BG, dpi=150)
    print(f'saved {OUT}')
    print(f'\n{"pt":<5}{"league p50":>12}{"Medina":>10}{"pctl":>7}')
    for pt in order:
        g = lg[lg.pt == pt]['cross']
        v = his.get(pt)
        cell = f'{v:>10.2f}{(g < v).mean() * 100:>7.0f}' if v is not None else f'{"-":>10}{"-":>7}'
        print(f'{pt:<5}{np.median(g):>12.2f}{cell}')


if __name__ == '__main__':
    main()
