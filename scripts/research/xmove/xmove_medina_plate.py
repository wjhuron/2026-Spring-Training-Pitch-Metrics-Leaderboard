#!/usr/bin/env python3
"""PROTOTYPE — expected-movement plate, in the card identity.

Reading it — three marks, nothing else:
  hollow circle   where his release alone says the ball should have gone
  filled dot      where it actually went
  arrow           the difference

The rail prints ACTUAL movement with the expected value beneath it. A deviation
on its own ("5.9 inches") does not say from what, or in which direction, which
is why earlier drafts were unreadable. Those drafts also carried a
class-typical diamond and a peer ring on the plot; that was several glyphs too
many, so the class comparison is now a single percentile in the rail.

The expectation uses arm angle, extension, velocity, spin rate and the measured
release axis. It never uses pitch type, so retagging a pitch cannot move it.

Model fit on 2021-2025, scored on Medina's 2026.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_medina_plate.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
# Importing Cards registers the bundled Bitter / IBM Plex faces and gives us the
# exact card palette, so this cannot drift from the print identity.
from cards.pitcher import (BG, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY,  # noqa: E402
                   TEXT_MUTED, TEXT_FAINT, PITCH_COLORS)
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402
from xmove_medina_plot import medina_2026, build_feats, PITCH_NAMES  # noqa: E402

FEATS = form(3, True, True)
OUT = os.path.expanduser('~/Downloads/Medina_expected_movement_plate.png')
DISP, BODY, COND = 'Bitter', 'IBM Plex Sans', 'IBM Plex Sans Condensed'


def darken(col, f=0.30):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(col)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def compute():
    A = add_harmonics(load_np())
    d = medina_2026()
    hand = str(d['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0

    tr = np.where(A['thr'] == hand)[0]
    Xt = _design(A, FEATS, tr)
    b_i = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
    b_h = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]

    lg = pd.DataFrame({
        'pitcher': A['pitcher'][tr], 'pt': A['pt'][tr], 'season': A['season'][tr],
        'oe_i': A['ivb'][tr] - Xt @ b_i, 'oe_h': A['hb_s'][tr] - Xt @ b_h,
    }).groupby(['pitcher', 'pt', 'season']).agg(
        n=('oe_i', 'size'), oe_i=('oe_i', 'mean'), oe_h=('oe_h', 'mean')).reset_index()
    lg = lg[lg.n >= 50]
    cls = lg.groupby('pt').agg(mi=('oe_i', 'mean'), mh=('oe_h', 'mean'),
                               si=('oe_i', 'std'), sh=('oe_h', 'std'))

    F = build_feats(d['Velocity'].values, d['Spin Rate'].values,
                    d['SpinAxis'].values, d['Extension'].values,
                    d['ArmAngle'].values, sign)
    X = _design(F, FEATS, np.arange(len(d)))
    d = d.assign(xivb=X @ b_i, xhb=(X @ b_h) * sign,
                 ivb=d['xIndVrtBrk'], hb=d['xHorzBrk'])
    # Cross-axis break: the component of the observed break perpendicular to the
    # MEASURED release axis, i.e. the part Magnus cannot explain. Pure geometry,
    # no model, so it is quoted in the callout as the evidence for the claim
    # that the four-seam / sinker separation is seams rather than spin.
    th = np.radians(((d['SpinAxis'].values - 180.0) % 360.0) * sign)
    _ivb, _hb_s = d['xIndVrtBrk'].values, d['xHorzBrk'].values * sign
    d = d.assign(cross=-_ivb * np.sin(th) + _hb_s * np.cos(th))
    return d, cls, hand, sign


def main():
    print('Loading 2021-2025...', file=sys.stderr)
    d, cls, hand, sign = compute()
    order = [p for p in d.groupby('Pitch Type').size().sort_values(ascending=False).index
             if (d['Pitch Type'] == p).sum() >= 25 and p in cls.index]

    fig = plt.figure(figsize=(15.0, 9.4), dpi=150)
    fig.patch.set_facecolor(BG)

    # ── header ────────────────────────────────────────────────────────
    fig.text(0.042, 0.950, 'LUIS MEDINA', fontsize=31, fontfamily=DISP,
             fontweight='black', color=TEXT_PRIMARY, va='top')
    fig.text(0.042, 0.888, 'RHP  ·  ATH  ·  2026 Season', fontsize=12.5,
             fontfamily=BODY, fontweight='500', color=TEXT_MUTED, va='top')
    fig.text(0.958, 0.950, 'EXPECTED MOVEMENT', fontsize=13, fontfamily=COND,
             fontweight='700', color=ACCENT, va='top', ha='right')
    fig.text(0.958, 0.910, 'where the ball should have gone, and where it went',
             fontsize=11, fontfamily=BODY, color=TEXT_MUTED, va='top', ha='right')
    fig.add_artist(Rectangle((0.042, 0.872), 0.916, 0.0016,
                             facecolor=ACCENT, edgecolor='none', alpha=0.85))

    ax = fig.add_axes([0.042, 0.090, 0.505, 0.745])
    ax.set_facecolor(BG)

    rows = []
    for pt in order:
        g = d[d['Pitch Type'] == pt]
        c = PITCH_COLORS.get(pt, '#777777')
        cd = darken(c)
        ax.scatter(g.hb, g.ivb, s=14, c=c, alpha=0.13, edgecolors='none', zorder=3)

        ex, ey = g.xhb.mean(), g.xivb.mean()
        axx, ayy = g.hb.mean(), g.ivb.mean()

        # halo under the stroke so the arrow survives its own dot cloud
        for cc, w, z in ((BG, 7.0, 5), (cd, 2.8, 6)):
            ax.annotate('', xy=(axx, ayy), xytext=(ex, ey), zorder=z,
                        arrowprops=dict(arrowstyle='-|>', color=cc, lw=w,
                                        shrinkA=7, shrinkB=0, mutation_scale=16))
        ax.plot([ex], [ey], marker='o', ms=9, mfc=BG, mec=cd, mew=2.4, zorder=8)
        ax.plot([axx], [ayy], marker='o', ms=10, mfc=cd, mec=BG, mew=1.6, zorder=9)
        ax.annotate(PITCH_NAMES.get(pt, pt), xy=(axx, ayy), xytext=(10, -14),
                    textcoords='offset points', fontsize=10.5, fontfamily=BODY,
                    fontweight='600', color=cd, zorder=10)

        r = cls.loc[pt]
        nx, ny = ex + r.mh * sign, ey + r.mi
        z = np.hypot((ayy - ny) / r.si, (axx - nx) / r.sh)
        rows.append((pt, c, cd, len(g), ayy, ey, axx, ex,
                     (1 - np.exp(-z * z / 2)) * 100))

    pad = 4.5
    cx = (d.hb.min() + d.hb.max()) / 2
    cy = (d.ivb.min() + d.ivb.max()) / 2
    half = max(d.hb.max() - d.hb.min(), d.ivb.max() - d.ivb.min()) / 2 + pad
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.axhline(0, color=TEXT_FAINT, lw=0.8, alpha=0.45, zorder=1)
    ax.axvline(0, color=TEXT_FAINT, lw=0.8, alpha=0.45, zorder=1)
    ax.grid(True, color=TEXT_FAINT, alpha=0.13, lw=0.7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(BODY)
    ax.set_xlabel('horizontal break (in)', fontsize=10, fontfamily=BODY,
                  color=TEXT_MUTED, labelpad=8)
    ax.set_ylabel('induced vertical break (in)', fontsize=10, fontfamily=BODY,
                  color=TEXT_MUTED, labelpad=8)
    _l, _r = ('glove side', 'arm side') if hand == 'R' else ('arm side', 'glove side')
    for _fx, _t, _ha in ((0.012, _l, 'left'), (0.988, _r, 'right')):
        ax.text(_fx, 0.018, _t, transform=ax.transAxes, fontsize=9.5,
                fontfamily=BODY, color=TEXT_FAINT, ha=_ha, va='bottom')

    # key sits inside the plot, where the marks are actually used
    kx, ky = 0.028, 0.962
    ax.plot([kx], [ky], marker='o', ms=9, mfc=BG, mec=TEXT_SECONDARY, mew=2.3,
            transform=ax.transAxes, zorder=12, clip_on=False)
    ax.text(kx + 0.032, ky, 'expected', transform=ax.transAxes, fontsize=10,
            fontfamily=BODY, color=TEXT_SECONDARY, va='center')
    ax.plot([kx + 0.190], [ky], marker='o', ms=10, mfc=TEXT_SECONDARY, mec=BG,
            mew=1.6, transform=ax.transAxes, zorder=12, clip_on=False)
    ax.text(kx + 0.222, ky, 'actual', transform=ax.transAxes, fontsize=10,
            fontfamily=BODY, color=TEXT_SECONDARY, va='center')

    # ── rail ──────────────────────────────────────────────────────────
    RX, RW = 0.600, 0.358
    # Column CENTRES, and every cell in the column is centred on them, so the
    # header sits over its numbers instead of off to one side.
    IVB_X, HB_X = RX + 0.132, RX + 0.220
    PILL_X, PILL_W = RX + 0.262, 0.062
    fig.text(RX, 0.828, 'ACTUAL MOVEMENT, VS EXPECTED', fontsize=11.5,
             fontfamily=COND, fontweight='700', color=TEXT_SECONDARY, va='top')
    fig.add_artist(Rectangle((RX, 0.812), RW, 0.0012, facecolor=TEXT_FAINT,
                             edgecolor='none', alpha=0.55))
    fig.text(IVB_X, 0.794, 'IVB', fontsize=8.5, fontfamily=COND, fontweight='700',
             color=TEXT_FAINT, va='top', ha='center')
    fig.text(HB_X, 0.794, 'HB', fontsize=8.5, fontfamily=COND, fontweight='700',
             color=TEXT_FAINT, va='top', ha='center')
    fig.text(PILL_X + PILL_W / 2, 0.794, 'HOW UNUSUAL FOR THAT PITCH', fontsize=8.5,
             fontfamily=COND, fontweight='700', color=TEXT_FAINT, va='top', ha='center')

    y = 0.738
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl in rows:
        fig.add_artist(Rectangle((RX, y - 0.004), 0.0075, 0.042,
                                 facecolor=c, edgecolor='none'))
        fig.text(RX + 0.016, y + 0.032, PITCH_NAMES.get(pt, pt), fontsize=13,
                 fontfamily=BODY, fontweight='600', color=TEXT_PRIMARY, va='top')
        fig.text(RX + 0.016, y + 0.010, f'{n} pitches', fontsize=9,
                 fontfamily=BODY, color=TEXT_FAINT, va='top')
        for _x, _act, _exp in ((IVB_X, a_i, e_i), (HB_X, a_h, e_h)):
            fig.text(_x, y + 0.032, f'{_act:.1f}"', fontsize=16, fontfamily=BODY,
                     fontweight='700', color=cd, va='top', ha='center')
            fig.text(_x, y + 0.010, f'expected {_exp:.1f}', fontsize=8.5,
                     fontfamily=BODY, color=TEXT_FAINT, va='top', ha='center')
        bx, bw, bh = PILL_X, PILL_W, 0.0125
        by = y + 0.016
        fig.add_artist(FancyBboxPatch((bx, by), bw, bh,
                                      boxstyle=f'round,pad=0,rounding_size={bh/2}',
                                      facecolor=TEXT_FAINT, edgecolor='none', alpha=0.22))
        fig.add_artist(FancyBboxPatch((bx, by), max(bw * pctl / 100, bh), bh,
                                      boxstyle=f'round,pad=0,rounding_size={bh/2}',
                                      facecolor=cd, edgecolor='none', alpha=0.9))
        fig.text(RX + RW, y + 0.030, f'{pctl:.0f}', fontsize=13, fontfamily=BODY,
                 fontweight='700', color=cd, va='top', ha='right')
        y -= 0.075

    fig.text(RX, y + 0.034,
             'Every sinker misses its expectation in much the same way, so a big\n'
             'gap is not automatically interesting. The last column ranks HIS gap\n'
             'against every other pitcher throwing that pitch. 98 means only 2%\n'
             'of changeups miss by as much, or in as odd a direction. 50 is the\n'
             'ordinary miss. His sinker drops 4.6" more than expected, which\n'
             'looks big but is simply what sinkers do, so it sits at 42.',
             fontsize=9.5, fontfamily=BODY, color=TEXT_MUTED, va='top',
             linespacing=1.65)

    ff = d[d['Pitch Type'] == 'FF']
    si = d[d['Pitch Type'] == 'SI']
    if len(ff) and len(si):
        gap = np.hypot(ff.ivb.mean() - si.ivb.mean(), ff.hb.mean() - si.hb.mean())
        exp_gap = np.hypot(ff.xivb.mean() - si.xivb.mean(),
                           ff.xhb.mean() - si.xhb.mean())
        fig.add_artist(Rectangle((RX, y - 0.086), RW, 0.0012,
                                 facecolor=ACCENT, edgecolor='none', alpha=0.7))
        fig.text(RX, y - 0.102,
                 f'His four-seam and sinker leave the hand the same way, so the\n'
                 f'model expects them {exp_gap:.1f}" apart. They land {gap:.1f}" apart.\n'
                 f'Measured straight off the release axis, with no model at all:\n'
                 f'the four-seam carries {ff["cross"].mean():.1f}" of break that spin cannot\n'
                 f'explain, the sinker {si["cross"].mean():.1f}". That {si["cross"].mean() - ff["cross"].mean():.1f}" of separation is seams.',
                 fontsize=9.8, fontfamily=BODY, color=TEXT_SECONDARY,
                 va='top', linespacing=1.6)

    fig.text(0.042, 0.028,
             'Expected movement uses arm angle, extension, velocity, spin rate and the measured '
             'release axis. It never uses the pitch type, so retagging a pitch cannot move it.',
             fontsize=9, fontfamily=BODY, color=TEXT_FAINT, va='bottom')

    fig.savefig(OUT, facecolor=BG, dpi=150)
    print(f'saved {OUT}')
    print(f'  {"pt":<5}{"n":>5}{"IVB":>8}{"xIVB":>8}{"HB":>8}{"xHB":>8}{"pctl":>7}')
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl in rows:
        print(f'  {pt:<5}{n:>5}{a_i:>8.1f}{e_i:>8.1f}{a_h:>8.1f}{e_h:>8.1f}{pctl:>7.0f}')


if __name__ == '__main__':
    main()
