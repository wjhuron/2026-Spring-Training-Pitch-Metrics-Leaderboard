#!/usr/bin/env python3
"""PROTOTYPE — Luis Medina (ATH) movement plot with expected-movement ellipses.

Shows, per pitch type, where the ball ACTUALLY moved against where a
pitch-type-AGNOSTIC model expects it to move given only the release: arm angle,
extension, velocity, spin rate and measured release spin axis. Nothing in the
expectation knows what the pitch is called.

Model: pooled by hand (no pitch type in the grouping key), harmonics to 3 with
spin and velo tensors, per scripts/research/xmove/xmove_agnostic_basis.py. Fit on 2021-2025 and
scored on Medina's 2026 pitches, so his data is out of sample in time.

The ellipse is deliberately ONE shape for every pitch type: +/- 1 SD of the
league distribution of pitcher-level residuals, pooled across pitch types. A
per-type ellipse would smuggle pitch type back into a metric whose whole point
is not to use it. So the ellipse reads "a pitch released like this usually lands
in here," and a pitch type sitting outside it is doing something its release
parameters do not account for.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_medina_plot.py [--out PATH]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402

PITCH_COLORS = {  # Okabe-Ito, matching Cards.py
    'FF': '#0072B2', 'SI': '#E69F00', 'FC': '#8B5A2B', 'SL': '#56B4E9',
    'ST': '#56B4E9', 'CU': '#3B2E8C', 'CH': '#009E73', 'FS': '#CC79A7',
}
PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'CU': 'Curveball', 'CH': 'Changeup', 'FS': 'Splitter'}
BG, INK, MUTED = '#F5EFE1', '#2A2723', '#8A8378'
FEATS = form(3, True, True)


def medina_2026():
    """Pitches from the retagged sheet, same source the cards read."""
    import gspread
    gc = gspread.service_account()
    sys.path.insert(0, ROOT)
    from cards.pitcher import _workbook_id_for_team
    ws = gc.open_by_key(_workbook_id_for_team('ATH')).worksheet('ATH')
    rows = [r for r in ws.get_all_records()
            if str(r.get('Pitcher', '')).strip() == 'Medina, Luis']
    d = pd.DataFrame(rows)
    missing = [c for c in ('Velocity', 'Spin Rate', 'RTilt', 'xIndVrtBrk',
                           'xHorzBrk', 'Extension', 'ArmAngle', 'Throws',
                           'Pitch Type') if c not in d.columns]
    if missing:
        raise SystemExit(f'sheet is missing {missing}; has {sorted(d.columns)[:40]}')
    # The sheet stores release tilt as clock notation; the training data uses
    # Savant's SpinAxis in degrees. Verified empirically on RHP four-seams:
    # mean 71.6 clock-minutes -> 215.8 deg against a training mean of 214.3.
    from pipeline.utils import break_tilt_to_minutes
    mins = d['RTilt'].map(break_tilt_to_minutes)
    d['SpinAxis'] = (180.0 + (mins % 720) * 0.5) % 360.0
    for c in ['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk', 'xHorzBrk',
              'Extension', 'ArmAngle']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk',
                            'xHorzBrk', 'Extension', 'ArmAngle'])


def build_feats(velo, spin, axis, ext, aa, sign):
    """Same feature construction as xmove_compare.load_np / add_harmonics."""
    th = np.radians(((axis - 180.0) % 360.0) * sign)
    A = {'aa': aa, 'ext': ext, 'velo': velo, 'spin': spin,
         'spin_v': spin / velo, 'st': np.sin(th), 'ct': np.cos(th)}
    return add_harmonics(A, 3)


def draw_plot(d26, hand, sd_i, sd_h, subtitle, note, out, title='LUIS MEDINA  ·  ATH  ·  2026'):
    """Scatter of actual movement, expected marker + ellipse, and the OE arrow.

    d26 needs columns: Pitch Type, ivb, hb (actual) and xivb, xhb (expected),
    all in catcher's perspective inches.
    """
    order = d26.groupby('Pitch Type').size().sort_values(ascending=False).index.tolist()
    order = [p for p in order if (d26['Pitch Type'] == p).sum() >= 25]

    fig, ax = plt.subplots(figsize=(11.5, 10.2), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for pt in order:
        g = d26[d26['Pitch Type'] == pt]
        c = PITCH_COLORS.get(pt, '#777777')
        ax.scatter(g.hb, g.ivb, s=26, c=c, alpha=0.30, edgecolors='none', zorder=3)
        ax_, ay = g.hb.mean(), g.ivb.mean()
        ex, ey = g.xhb.mean(), g.xivb.mean()
        ax.add_patch(Ellipse((ex, ey), 2 * sd_h, 2 * sd_i, facecolor='none',
                             edgecolor=c, lw=1.8, ls=(0, (5, 3)), alpha=0.95, zorder=5))
        ax.plot([ex], [ey], marker='o', ms=7, mfc=BG, mec=c, mew=2.0, zorder=6)
        ax.annotate('', xy=(ax_, ay), xytext=(ex, ey), zorder=7,
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4,
                                    shrinkA=7, shrinkB=0))
        ax.plot([ax_], [ay], marker='o', ms=10, mfc=c, mec=BG, mew=1.6, zorder=8)
        ax.annotate(f'{PITCH_NAMES.get(pt, pt)}  {np.hypot(ay - ey, ax_ - ex):.1f}"',
                    xy=(ax_, ay), xytext=(6, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=c, zorder=9)

    lim = 26
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Horizontal Break (in)   ' + ('glove side  <-  ->  arm side'
                  if hand == 'R' else 'arm side  <-  ->  glove side'),
                  fontsize=11, color=INK)
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=11, color=INK)
    ax.set_title(f'{title}\n{subtitle}', fontsize=15, fontweight='bold',
                 color=INK, pad=14)
    ax.grid(True, color=MUTED, alpha=0.18, lw=0.7)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_alpha(0.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.annotate(note, xy=(0.015, 0.015), xycoords='axes fraction', fontsize=9,
                color=MUTED, va='bottom', ha='left', linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f'saved {out}')

    print(f'\nper pitch type: actual vs expected (inches)  [{subtitle}]')
    print(f'{"pt":<5}{"n":>5}{"IVB":>8}{"xIVB":>8}{"dIVB":>8}{"HB":>8}{"xHB":>8}{"dHB":>8}')
    for pt in order:
        g = d26[d26['Pitch Type'] == pt]
        print(f'{pt:<5}{len(g):>5}{g.ivb.mean():>8.1f}{g.xivb.mean():>8.1f}'
              f'{g.ivb.mean()-g.xivb.mean():>8.1f}{g.hb.mean():>8.1f}'
              f'{g.xhb.mean():>8.1f}{g.hb.mean()-g.xhb.mean():>8.1f}')


def main():
    out = (sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv
           else os.path.expanduser('~/Downloads/Medina_expected_movement.png'))

    print('Loading 2021-2025 training data...', file=sys.stderr)
    A = add_harmonics(load_np())
    d26 = medina_2026()
    hand = str(d26['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0
    print(f'  Medina 2026: {len(d26)} pitches, throws {hand}', file=sys.stderr)

    # Fit pooled on this hand, all five seasons.
    tr = np.where(A['thr'] == hand)[0]
    Xt = _design(A, FEATS, tr)
    b_i = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
    b_h = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]

    # League residual spread at the pitcher x type unit -> the ellipse.
    xi_tr, xh_tr = Xt @ b_i, Xt @ b_h
    lg = pd.DataFrame({
        'pitcher': A['pitcher'][tr], 'pt': A['pt'][tr], 'season': A['season'][tr],
        'oe_i': A['ivb'][tr] - xi_tr, 'oe_h': A['hb_s'][tr] - xh_tr,
    }).groupby(['pitcher', 'pt', 'season']).agg(
        n=('oe_i', 'size'), oe_i=('oe_i', 'mean'), oe_h=('oe_h', 'mean')).reset_index()
    lg = lg[lg.n >= 50]
    sd_i, sd_h = lg.oe_i.std(), lg.oe_h.std()
    print(f'  league pitcher-level OE sd: IVB {sd_i:.2f}"  HB {sd_h:.2f}"', file=sys.stderr)

    # Score Medina's pitches.
    F = build_feats(d26['Velocity'].values, d26['Spin Rate'].values,
                    d26['SpinAxis'].values, d26['Extension'].values,
                    d26['ArmAngle'].values, sign)
    idx = np.arange(len(d26))
    X = _design(F, FEATS, idx)
    d26 = d26.assign(xivb=X @ b_i, xhb_s=(X @ b_h))
    d26['xhb'] = d26['xhb_s'] * sign          # back to catcher's perspective
    d26['ivb'] = d26['xIndVrtBrk']
    d26['hb'] = d26['xHorzBrk']

    order = (d26.groupby('Pitch Type').size().sort_values(ascending=False).index.tolist())
    order = [p for p in order if (d26['Pitch Type'] == p).sum() >= 25]

    fig, ax = plt.subplots(figsize=(11.5, 10.2), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for pt in order:
        g = d26[d26['Pitch Type'] == pt]
        c = PITCH_COLORS.get(pt, '#777777')
        ax.scatter(g.hb, g.ivb, s=26, c=c, alpha=0.30, edgecolors='none', zorder=3)
        ax_, ay = g.hb.mean(), g.ivb.mean()
        ex, ey = g.xhb.mean(), g.xivb.mean()
        # expectation: ellipse at +/- 1 SD of league pitcher-level OE
        ax.add_patch(Ellipse((ex, ey), 2 * sd_h, 2 * sd_i, facecolor='none',
                             edgecolor=c, lw=1.8, ls=(0, (5, 3)), alpha=0.95, zorder=5))
        ax.plot([ex], [ey], marker='o', ms=7, mfc=BG, mec=c, mew=2.0, zorder=6)
        # OE vector: expected -> actual
        ax.annotate('', xy=(ax_, ay), xytext=(ex, ey), zorder=7,
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4,
                                    shrinkA=7, shrinkB=0))
        ax.plot([ax_], [ay], marker='o', ms=10, mfc=c, mec=BG, mew=1.6, zorder=8)
        d_i, d_h = ay - ey, ax_ - ex
        ax.annotate(f'{PITCH_NAMES.get(pt, pt)}  {np.hypot(d_i, d_h):.1f}"',
                    xy=(ax_, ay), xytext=(6, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=c, zorder=9)

    lim = 26
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Horizontal Break (in)   ' + ('glove side  <-  ->  arm side'
                  if hand == 'R' else 'arm side  <-  ->  glove side'),
                  fontsize=11, color=INK)
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=11, color=INK)
    ax.set_title('LUIS MEDINA  ·  ATH  ·  2026\n'
                 'expected movement from release only, vs actual',
                 fontsize=15, fontweight='bold', color=INK, pad=14)
    ax.grid(True, color=MUTED, alpha=0.18, lw=0.7)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_alpha(0.5)
    ax.tick_params(colors=MUTED, labelsize=9)

    note = ('open circle + dashed ellipse = expected from arm angle, extension, velocity,\n'
            'spin rate and release axis alone (pitch type NOT used).  ellipse = +/-1 SD of\n'
            'league pitcher-level deviation.  arrow = what the seams and everything\n'
            'unmeasured added on top.')
    ax.annotate(note, xy=(0.015, 0.015), xycoords='axes fraction', fontsize=9,
                color=MUTED, va='bottom', ha='left', linespacing=1.5)

    fig.tight_layout()
    fig.savefig(out, facecolor=BG, dpi=150)
    print(f'saved {out}')

    print('\nper pitch type: actual vs expected (inches)')
    print(f'{"pt":<5}{"n":>5}{"IVB":>8}{"xIVB":>8}{"dIVB":>8}{"HB":>8}{"xHB":>8}{"dHB":>8}')
    for pt in order:
        g = d26[d26['Pitch Type'] == pt]
        print(f'{pt:<5}{len(g):>5}{g.ivb.mean():>8.1f}{g.xivb.mean():>8.1f}'
              f'{g.ivb.mean()-g.xivb.mean():>8.1f}{g.hb.mean():>8.1f}'
              f'{g.xhb.mean():>8.1f}{g.hb.mean()-g.xhb.mean():>8.1f}')


if __name__ == '__main__':
    main()
