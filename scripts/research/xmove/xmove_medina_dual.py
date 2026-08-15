#!/usr/bin/env python3
"""PROTOTYPE — the design that keeps the number pitch-type agnostic and still
shows the class comparison.

Three marks per pitch type:

  1. OPEN CIRCLE at the expectation from the PITCH-TYPE-AGNOSTIC model: arm
     angle, extension, velocity, spin rate and measured release axis. Nothing
     here knows what the pitch is called, so nothing here moves if the pitch is
     retagged.

  2. FAINT ARROW from that circle to the LEAGUE-TYPICAL deviation for pitches
     of this class. This is what the seams normally add to a sinker, a slider,
     and so on. It is drawn rather than modelled, which is the whole point:
     retagging changes which faint arrow gets drawn and changes nothing else.

  3. SOLID ARROW from the same circle to where HIS pitches actually land, with
     an ellipse at +/- 1 SD of pitcher-level deviation WITHIN that class,
     centred on the class-typical tip. Inside the ellipse means ordinary for
     someone throwing this pitch; outside means genuinely unusual.

So the number (his arrow length) is physics and is retag-proof, while the
comparison the viewer actually wants (is he unusual for a sinker?) is read off
the picture by comparing two arrows and a ring.

Model fit on 2021-2025, scored on Medina's 2026, so his data is out of sample.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_medina_dual.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402
from xmove_medina_plot import (medina_2026, build_feats, PITCH_COLORS,  # noqa: E402
                               PITCH_NAMES, BG, INK, MUTED)

def darken(hexcol, f=0.42):
    """Blend a colour toward black by fraction f. Used for arrows so they
    contrast against a scatter cloud of the same hue."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexcol)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def lighten(hexcol, f=0.62):
    """Blend toward the page background. Used for the shadow arrow: a solid
    pale stroke reads better than a translucent one, which washes out against
    the opaque halo beneath it."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexcol)
    br, bg_, bb = mc.to_rgb(BG)
    return (r + (br - r) * f, g + (bg_ - g) * f, b + (bb - b) * f)


FEATS = form(3, True, True)
OUT = os.path.expanduser('~/Downloads/Medina_expected_movement_dual.png')


def main():
    print('Loading 2021-2025...', file=sys.stderr)
    A = add_harmonics(load_np())
    d26 = medina_2026()
    hand = str(d26['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0

    # Agnostic fit: pooled by hand, no pitch type anywhere.
    tr = np.where(A['thr'] == hand)[0]
    Xt = _design(A, FEATS, tr)
    b_i = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
    b_h = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]

    # League deviation per pitcher x type, under that same agnostic model.
    lg = pd.DataFrame({
        'pitcher': A['pitcher'][tr], 'pt': A['pt'][tr], 'season': A['season'][tr],
        'oe_i': A['ivb'][tr] - Xt @ b_i, 'oe_h': A['hb_s'][tr] - Xt @ b_h,
    }).groupby(['pitcher', 'pt', 'season']).agg(
        n=('oe_i', 'size'), oe_i=('oe_i', 'mean'), oe_h=('oe_h', 'mean')).reset_index()
    lg = lg[lg.n >= 50]
    cls = lg.groupby('pt').agg(n=('oe_i', 'size'),
                               mi=('oe_i', 'mean'), mh=('oe_h', 'mean'),
                               si=('oe_i', 'std'), sh=('oe_h', 'std'))
    print(cls.round(2), file=sys.stderr)

    # Score Medina.
    F = build_feats(d26['Velocity'].values, d26['Spin Rate'].values,
                    d26['SpinAxis'].values, d26['Extension'].values,
                    d26['ArmAngle'].values, sign)
    X = _design(F, FEATS, np.arange(len(d26)))
    d26 = d26.assign(xivb=X @ b_i, xhb=(X @ b_h) * sign,
                     ivb=d26['xIndVrtBrk'], hb=d26['xHorzBrk'])

    order = d26.groupby('Pitch Type').size().sort_values(ascending=False).index.tolist()
    order = [p for p in order if (d26['Pitch Type'] == p).sum() >= 25]

    fig, ax = plt.subplots(figsize=(12, 10.6), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    rows, exp_pos = [], {}
    for pt in order:
        g = d26[d26['Pitch Type'] == pt]
        c = PITCH_COLORS.get(pt, '#777777')
        # Dots go very light: the arrows have to cross their own cloud, and at
        # matching hue and weight the arrow simply disappears into it (the
        # curveball's was invisible). Light wash for the cloud, dark ink for
        # the arrow, so the two never compete.
        cd = darken(c, 0.42)
        ax.scatter(g.hb, g.ivb, s=16, c=c, alpha=0.10, edgecolors='none', zorder=3)

        ex, ey = g.xhb.mean(), g.xivb.mean()          # agnostic expectation
        axx, ayy = g.hb.mean(), g.ivb.mean()          # his actual
        if pt not in cls.index:
            continue
        r = cls.loc[pt]
        nx, ny = ex + r.mh * sign, ey + r.mi          # league-typical for the class

        def arrow(x0, y0, x1, y1, lw, col, alpha, z, halo=5.0):
            """Opaque background halo underneath, then the stroke on top."""
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=z - 0.5,
                        arrowprops=dict(arrowstyle='-|>', color=BG, lw=lw + halo,
                                        alpha=1.0, shrinkA=8, shrinkB=0,
                                        mutation_scale=18))
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=z,
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=lw,
                                        alpha=alpha, shrinkA=8, shrinkB=0,
                                        mutation_scale=18))

        # 2. shadow arrow: what the class typically adds
        arrow(ex, ey, nx, ny, 7.5, lighten(c, 0.55), 1.0, 6, halo=3.0)
        # peer spread at the 68% contour. In 2D that is 1.51 SD, NOT 1.0 —
        # a +/-1 SD ellipse holds only 39% of a bivariate normal.
        ax.add_patch(Ellipse((nx, ny), 2 * 1.51 * r.sh, 2 * 1.51 * r.si,
                             facecolor='none', edgecolor=c, lw=1.5,
                             ls=(0, (4, 3)), alpha=0.60, zorder=5))
        ax.plot([nx], [ny], marker='x', ms=7, mec=cd, mew=2.0, alpha=0.85, zorder=7)
        # 3. his arrow, in the darkened hue so it reads against its own cloud
        arrow(ex, ey, axx, ayy, 3.0, cd, 1.0, 8)
        # 1. expectation, drawn last so nothing overlaps its centre
        ax.plot([ex], [ey], marker='o', ms=8, mfc=BG, mec=cd, mew=2.4, zorder=10)
        ax.plot([axx], [ayy], marker='o', ms=11, mfc=cd, mec=BG, mew=1.8, zorder=11)
        exp_pos[pt] = (ex, ey, axx, ayy)

        his = np.hypot(ayy - ey, axx - ex)
        typ = np.hypot(r.mi, r.mh)
        # inside the peer ellipse?
        z = np.hypot((ayy - ny) / r.si, (axx - nx) / r.sh)
        # 2D radius -> percentile among that class's throwers (Rayleigh):
        # P(R < z) = 1 - exp(-z^2 / 2). Median is 1.18, not 1.0.
        pctl = (1 - np.exp(-z * z / 2)) * 100
        ax.annotate(f'{PITCH_NAMES.get(pt, pt)}  {his:.1f}"',
                    xy=(axx, ayy), xytext=(7, 9), textcoords='offset points',
                    fontsize=11.5, fontweight='bold', color=cd, zorder=12)
        rows.append((pt, len(g), his, typ, pctl))

    lim = 26
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Horizontal Break (in)   ' + ('glove side  <-  ->  arm side'
                  if hand == 'R' else 'arm side  <-  ->  glove side'),
                  fontsize=11, color=INK)
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=11, color=INK)
    ax.set_title('LUIS MEDINA  ·  ATH  ·  2026\n'
                 'movement vs what his release alone predicts',
                 fontsize=15, fontweight='bold', color=INK, pad=14)
    ax.grid(True, color=MUTED, alpha=0.16, lw=0.7)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_alpha(0.5)
    ax.tick_params(colors=MUTED, labelsize=9)

    handles = [
        Line2D([], [], marker='o', ls='none', mfc=BG, mec=INK, mew=2, ms=8,
               label='expected from release alone (pitch type not used)'),
        Line2D([], [], color=INK, lw=7, alpha=0.32,
               label='what this pitch type typically adds (x = its centre)'),
        Line2D([], [], color=INK, lw=2.4, label='what HE adds'),
        Line2D([], [], color=INK, lw=1.5, ls=(0, (4, 3)), alpha=0.6,
               label='68% of pitchers throwing this pitch (1.51 SD in 2D)'),
    ]
    # Legend into the empty upper-left quadrant; the caption becomes a figure
    # footer so the two can never collide again.
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9.5,
              labelcolor=MUTED, handlelength=2.6, borderaxespad=1.0)
    fig.text(0.5, 0.012,
             'His arrow never moves if the pitch is retagged: the expectation uses no label. '
             'Only the shadow arrow and the ring would change, because those ARE the comparison.',
             fontsize=9.5, color=MUTED, ha='center', va='bottom')

    # FF/SI callout — the single most interesting fact on the plot, and it
    # reads as clutter without being named.
    if 'FF' in exp_pos and 'SI' in exp_pos:
        fx, fy, fax, fay = exp_pos['FF']
        sx, sy, sax, say = exp_pos['SI']
        gap = np.hypot(fay - say, fax - sax)
        mx, my = (fx + sx) / 2, (fy + sy) / 2
        ax.annotate('', xy=(fx, fy), xytext=(sx, sy), zorder=9,
                    arrowprops=dict(arrowstyle='-', color=INK, lw=1.2, alpha=0.55))
        ax.annotate(f'four-seam and sinker:\none release, one expectation.\n'
                    f'the {gap:.1f}" gap is seams.',
                    xy=(mx, my + 1.4), xytext=(19.0, 22.0),
                    fontsize=10, color=INK, ha='center', va='center', linespacing=1.5,
                    zorder=13,
                    bbox=dict(boxstyle='round,pad=0.5', fc=BG, ec=MUTED,
                              lw=0.8, alpha=0.92),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3,
                                    alpha=0.85, shrinkA=6, shrinkB=8,
                                    connectionstyle='arc3,rad=-0.18'))

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUT, facecolor=BG, dpi=150)
    print(f'saved {OUT}')

    print(f'\n{"pt":<5}{"n":>5}{"his dev":>10}{"typical":>10}{"pctl":>7}  read')
    for pt, n, his, typ, pctl in sorted(rows, key=lambda r: -r[4]):
        read = ('extreme' if pctl >= 90 else 'notable' if pctl >= 70
                else 'ordinary for this pitch')
        print(f'{pt:<5}{n:>5}{his:>9.1f}"{typ:>9.1f}"{pctl:>7.0f}  {read}')


if __name__ == '__main__':
    main()
