#!/usr/bin/env python3
"""PROTOTYPE — expected-movement plate, v2.

v1 (xmove_medina_plate.py) showed the last two links of a four-link chain:
release inputs -> Magnus expectation -> actual movement -> seam residual. It
named the inputs in a footnote and never showed one, so the expectation arrived
as a black box. Five changes:

  1. RELEASE BLOCK. Velocity, spin rate and release tilt per pitch, plus arm
     angle and extension. The expectation stops being an assertion, and the
     four-seam / sinker case becomes self-evident: 97.6 vs 97.2 mph, 2406 vs
     2424 rpm, 1:15 vs 1:09 tilt. Nearly the same hand action, nine inches
     apart on the plot.
  2. It fills the dead bottom third of the rail, so 1 and 2 solve each other.
  3. DEVIATION PANEL. All five expected->actual vectors redrawn from a shared
     origin with 3" and 6" reference rings. On the main plot these render at
     about a tenth of the width, which is small for the thing the card is about.
  4. The percentile pill gets a tick at 50, and the six-line paragraph
     explaining it collapses to one line. If a column needs a paragraph, the
     column is wrong.
  5. Tighter frame: movement lives on a diagonal band, so a square always
     wastes two corners.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_plate_v2.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from cards.pitcher import (BG, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY,  # noqa: E402
                   TEXT_MUTED, TEXT_FAINT, PITCH_COLORS)
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402
from xmove_medina_plot import build_feats, PITCH_NAMES  # noqa: E402


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


PITCHER = _arg('--pitcher', 'Medina, Luis')
TEAM = _arg('--team', 'ATH')
# --per-class swaps the expectation for the option-2 form (fit per pitch type,
# so each class's typical seam deflection lands in its own intercept). Layout
# and every other number are untouched, so the two renders differ ONLY by the
# model and can be read side by side.
PER_CLASS = '--per-class' in sys.argv


def load_pitcher(name, team):
    """Any pitcher's season off the retagged sheet."""
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
    d['_rmin'] = d['RTilt'].map(break_tilt_to_minutes)
    d['_omin'] = d['OTilt'].map(break_tilt_to_minutes)
    d['SpinAxis'] = (180.0 + (d['_rmin'] % 720) * 0.5) % 360.0
    for c in ['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk', 'xHorzBrk',
              'Extension', 'ArmAngle']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk',
                            'xHorzBrk', 'Extension', 'ArmAngle'])

FEATS = form(3, True, True)
OUT = os.path.expanduser(_arg('--out', '~/Downloads/plate_v2.png'))
DISP, BODY, COND = 'Bitter', 'IBM Plex Sans', 'IBM Plex Sans Condensed'


def darken(col, f=0.30):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(col)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def _circ(minutes):
    """Circular mean of clock minutes."""
    a = np.radians(np.asarray(minutes.dropna(), dtype=float) / 720.0 * 360.0)
    return (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) / 360.0 * 720.0


def tilt_str(minutes):
    """Circular mean of clock-notation tilts, back to h:mm."""
    a = np.radians(np.asarray(minutes, dtype=float) / 720.0 * 360.0)
    m = (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) / 360.0 * 720.0
    h, mm = int(m // 60) % 12, int(round(m % 60))
    if mm == 60:
        h, mm = (h + 1) % 12, 0
    return f'{12 if h == 0 else h}:{mm:02d}'


def compute():
    A = add_harmonics(load_np())
    d = load_pitcher(PITCHER, TEAM)
    hand = str(d['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0

    F = build_feats(d['Velocity'].values, d['Spin Rate'].values,
                    d['SpinAxis'].values, d['Extension'].values,
                    d['ArmAngle'].values, sign)
    X_all = _design(F, FEATS, np.arange(len(d)))

    def _fit(idx):
        Xt = _design(A, FEATS, idx)
        return (Xt,
                np.linalg.lstsq(Xt, A['ivb'][idx], rcond=None)[0],
                np.linalg.lstsq(Xt, A['hb_s'][idx], rcond=None)[0])

    hand_idx = np.where(A['thr'] == hand)[0]
    oe_i = np.full(len(hand_idx), np.nan)
    oe_h = np.full(len(hand_idx), np.nan)
    xi = np.full(len(d), np.nan)
    xh = np.full(len(d), np.nan)

    if PER_CLASS:
        for pt in pd.unique(A['pt'][hand_idx]):
            sel = np.where(A['pt'][hand_idx] == pt)[0]
            if len(sel) < 2000:
                continue
            Xt, b_i, b_h = _fit(hand_idx[sel])
            oe_i[sel] = A['ivb'][hand_idx[sel]] - Xt @ b_i
            oe_h[sel] = A['hb_s'][hand_idx[sel]] - Xt @ b_h
            m = (d['Pitch Type'] == pt).values
            if m.any():
                Xs = _design(F, FEATS, np.where(m)[0])
                xi[m], xh[m] = Xs @ b_i, Xs @ b_h
    else:
        Xt, b_i, b_h = _fit(hand_idx)
        oe_i = A['ivb'][hand_idx] - Xt @ b_i
        oe_h = A['hb_s'][hand_idx] - Xt @ b_h
        xi, xh = X_all @ b_i, X_all @ b_h

    ok = np.isfinite(oe_i)
    lg = pd.DataFrame({
        'pitcher': A['pitcher'][hand_idx][ok], 'pt': A['pt'][hand_idx][ok],
        'season': A['season'][hand_idx][ok],
        'oe_i': oe_i[ok], 'oe_h': oe_h[ok],
    }).groupby(['pitcher', 'pt', 'season']).agg(
        n=('oe_i', 'size'), oe_i=('oe_i', 'mean'), oe_h=('oe_h', 'mean')).reset_index()
    lg = lg[lg.n >= 50]
    cls = lg.groupby('pt').agg(mi=('oe_i', 'mean'), mh=('oe_h', 'mean'),
                               si=('oe_i', 'std'), sh=('oe_h', 'std'))

    d = d.assign(xivb=xi, xhb=xh * sign,
                 ivb=d['xIndVrtBrk'], hb=d['xHorzBrk'])
    th = np.radians(((d['SpinAxis'].values - 180.0) % 360.0) * sign)
    d = d.assign(cross=-d['xIndVrtBrk'].values * np.sin(th)
                 + (d['xHorzBrk'].values * sign) * np.cos(th))
    return d, cls, hand, sign


def main():
    print('Loading 2021-2025...', file=sys.stderr)
    d, cls, hand, sign = compute()
    order = [p for p in d.groupby('Pitch Type').size().sort_values(ascending=False).index
             if (d['Pitch Type'] == p).sum() >= 25 and p in cls.index]

    # Height tracks the arsenal: at six pitch types the fixed 10.2 clipped the
    # closing note off the bottom.
    _n = max(len(order), 1)
    _f = min(1.0, 5.0 / _n)          # fraction budget per row
    fig = plt.figure(figsize=(15.5, 10.2 + max(0, _n - 5) * 0.45), dpi=150)
    fig.patch.set_facecolor(BG)

    # ── header ────────────────────────────────────────────────────────
    _last, _first = (PITCHER.split(',') + [''])[:2]
    fig.text(0.040, 0.958, f'{_first.strip()} {_last.strip()}'.strip().upper(),
             fontsize=31, fontfamily=DISP,
             fontweight='black', color=TEXT_PRIMARY, va='top')
    fig.text(0.040, 0.902, f'{hand}HP  ·  {TEAM}  ·  2026 Season', fontsize=12.5,
             fontfamily=BODY, fontweight='500', color=TEXT_MUTED, va='top')
    fig.text(0.960, 0.958, 'EXPECTED MOVEMENT', fontsize=13, fontfamily=COND,
             fontweight='700', color=ACCENT, va='top', ha='right')
    fig.text(0.960, 0.921,
             ('OPTION 2: expectation fit per pitch class'
              if PER_CLASS else
              'OPTION 1: expectation ignores the pitch label'),
             fontsize=11, fontfamily=BODY, color=TEXT_MUTED, va='top', ha='right')
    fig.add_artist(Rectangle((0.040, 0.884), 0.920, 0.0016,
                             facecolor=ACCENT, edgecolor='none', alpha=0.85))

    ax = fig.add_axes([0.040, 0.335, 0.440, 0.505])
    ax.set_facecolor(BG)

    rows = []
    for pt in order:
        g = d[d['Pitch Type'] == pt]
        c = PITCH_COLORS.get(pt, '#777777')
        cd = darken(c)
        ax.scatter(g.hb, g.ivb, s=13, c=c, alpha=0.12, edgecolors='none', zorder=3)
        ex, ey = g.xhb.mean(), g.xivb.mean()
        axx, ayy = g.hb.mean(), g.ivb.mean()
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
                     (1 - np.exp(-z * z / 2)) * 100,
                     g.Velocity.mean(), g['Spin Rate'].mean(),
                     tilt_str(g._rmin.dropna()), tilt_str(g._omin.dropna()),
                     g['cross'].mean()))

    # 5. tighter frame: pad the data box rather than forcing a square on a
    #    diagonal band, but keep equal aspect so the geometry stays honest.
    padx, pady = 5.0, 4.0
    x0, x1 = d.hb.min() - padx, d.hb.max() + padx
    y0, y1 = d.ivb.min() - pady, d.ivb.max() + pady
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal', adjustable='box')
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
    kx, ky = 0.028, 0.962
    ax.plot([kx], [ky], marker='o', ms=9, mfc=BG, mec=TEXT_SECONDARY, mew=2.3,
            transform=ax.transAxes, zorder=12, clip_on=False)
    ax.text(kx + 0.036, ky, 'expected', transform=ax.transAxes, fontsize=10,
            fontfamily=BODY, color=TEXT_SECONDARY, va='center')
    ax.plot([kx + 0.215], [ky], marker='o', ms=10, mfc=TEXT_SECONDARY, mec=BG,
            mew=1.6, transform=ax.transAxes, zorder=12, clip_on=False)
    ax.text(kx + 0.251, ky, 'actual', transform=ax.transAxes, fontsize=10,
            fontfamily=BODY, color=TEXT_SECONDARY, va='center')

    # ── 3. deviation panel: every vector from a shared origin ─────────
    dv = fig.add_axes([0.120, 0.038, 0.152, 0.222])
    dv.set_facecolor(BG)
    lim = max(np.hypot(r[4] - r[5], r[6] - r[7]) for r in rows) * 1.42
    for ring in (3, 6):
        if ring < lim:
            dv.add_patch(Circle((0, 0), ring, facecolor='none', edgecolor=TEXT_FAINT,
                                lw=0.9, ls=(0, (3, 3)), alpha=0.55, zorder=2))
            dv.text(ring * 0.7071 + 0.25, ring * 0.7071 + 0.25, f'{ring}"',
                    fontsize=8, fontfamily=BODY, color=TEXT_FAINT, zorder=3)
    dv.axhline(0, color=TEXT_FAINT, lw=0.7, alpha=0.4, zorder=1)
    dv.axvline(0, color=TEXT_FAINT, lw=0.7, alpha=0.4, zorder=1)
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl, v, sp_, rt, ot, cx in rows:
        vx, vy = a_h - e_h, a_i - e_i
        dv.annotate('', xy=(vx, vy), xytext=(0, 0), zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=cd, lw=2.4,
                                    shrinkA=0, shrinkB=0, mutation_scale=13))
        mag = math.hypot(vx, vy)
        lx, ly = vx / mag * (mag + lim * 0.16), vy / mag * (mag + lim * 0.16)
        dv.text(lx, ly, PITCH_NAMES.get(pt, pt), fontsize=8.5,
                fontfamily=BODY, fontweight='700', color=cd, ha='center',
                va='center', zorder=8)
    def _dir(vx, vy):
        ud = 'down' if vy < 0 else 'up'
        lr = 'arm-side' if vx * sign > 0 else 'glove-side'
        return f'{ud} and {lr}'

    _big = sorted(rows, key=lambda r: -math.hypot(r[6] - r[7], r[4] - r[5]))[:2]
    _dev_sentence = ' '.join(
        f'The {PITCH_NAMES.get(b[0], b[0]).lower()} misses '
        f'{_dir(b[6] - b[7], b[4] - b[5])}.' for b in _big)
    dv.set_xlim(-lim, lim); dv.set_ylim(-lim, lim); dv.set_aspect('equal')
    dv.set_xticks([]); dv.set_yticks([])
    for sp2 in dv.spines.values():
        sp2.set_visible(False)
    fig.text(0.196, 0.276, 'HOW FAR EACH MISSED, FROM A COMMON ORIGIN',
             fontsize=9.5, fontfamily=COND, fontweight='700',
             color=TEXT_SECONDARY, va='bottom', ha='center')
    # the empty band right of the panel earns its keep
    import textwrap
    _cap_text = '\n'.join(textwrap.wrap(
        'Every arrow above, redrawn from one origin. On the main plot these are '
        'a tenth of the width, too small to compare against each other. Here the '
        f'lengths and directions are directly readable. {_dev_sentence} '
        'Rings are 3" and 6".', width=44))
    fig.text(0.300, 0.235,
             _cap_text,
             fontsize=9.5, fontfamily=BODY, color=TEXT_MUTED, va='top',
             ha='left', linespacing=1.65)

    # ── rail ──────────────────────────────────────────────────────────
    RX, RW = 0.525, 0.435
    IVB_X, HB_X = RX + 0.150, RX + 0.245
    PILL_X, PILL_W = RX + 0.288, 0.086
    fig.text(RX, 0.828, 'ACTUAL MOVEMENT, VS EXPECTED', fontsize=11.5,
             fontfamily=COND, fontweight='700', color=TEXT_SECONDARY, va='top')
    fig.add_artist(Rectangle((RX, 0.812), RW, 0.0012, facecolor=TEXT_FAINT,
                             edgecolor='none', alpha=0.55))
    for _x, _t in ((IVB_X, 'IVB'), (HB_X, 'HB')):
        fig.text(_x, 0.796, _t, fontsize=8.5, fontfamily=COND, fontweight='700',
                 color=TEXT_FAINT, va='top', ha='center')
    fig.text(PILL_X + PILL_W / 2, 0.796, 'HOW UNUSUAL FOR THAT PITCH', fontsize=8.5,
             fontfamily=COND, fontweight='700', color=TEXT_FAINT, va='top', ha='center')

    y = 0.744
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl, v, sp_, rt, ot, cx in rows:
        fig.add_artist(Rectangle((RX, y - 0.004), 0.0075, 0.040,
                                 facecolor=c, edgecolor='none'))
        fig.text(RX + 0.015, y + 0.030, PITCH_NAMES.get(pt, pt), fontsize=13,
                 fontfamily=BODY, fontweight='600', color=TEXT_PRIMARY, va='top')
        fig.text(RX + 0.015, y + 0.009, f'{n} pitches', fontsize=9,
                 fontfamily=BODY, color=TEXT_FAINT, va='top')
        for _x, _act, _exp in ((IVB_X, a_i, e_i), (HB_X, a_h, e_h)):
            fig.text(_x, y + 0.030, f'{_act:.1f}"', fontsize=16, fontfamily=BODY,
                     fontweight='700', color=cd, va='top', ha='center')
            fig.text(_x, y + 0.009, f'expected {_exp:.1f}', fontsize=8.5,
                     fontfamily=BODY, color=TEXT_FAINT, va='top', ha='center')
        bh = 0.0125
        by = y + 0.015
        fig.add_artist(FancyBboxPatch((PILL_X, by), PILL_W, bh,
                                      boxstyle=f'round,pad=0,rounding_size={bh/2}',
                                      facecolor=TEXT_FAINT, edgecolor='none', alpha=0.22))
        fig.add_artist(FancyBboxPatch((PILL_X, by), max(PILL_W * pctl / 100, bh), bh,
                                      boxstyle=f'round,pad=0,rounding_size={bh/2}',
                                      facecolor=cd, edgecolor='none', alpha=0.9))
        # 4. tick at 50 so the bar has a reference instead of a paragraph
        fig.add_artist(Rectangle((PILL_X + PILL_W / 2 - 0.0007, by - 0.003),
                                 0.0014, bh + 0.006, facecolor=BG,
                                 edgecolor='none', zorder=5))
        fig.text(RX + RW, y + 0.029, f'{pctl:.0f}', fontsize=13, fontfamily=BODY,
                 fontweight='700', color=cd, va='top', ha='right')
        y -= 0.070 * _f

    fig.text(RX, y + 0.030,
             'Ranked against everyone else throwing that pitch. The tick is 50, the ordinary miss.',
             fontsize=9.5, fontfamily=BODY, color=TEXT_MUTED, va='top')

    # ── 1. release block ──────────────────────────────────────────────
    ry = y - 0.010
    fig.text(RX, ry, 'HOW IT LEFT THE HAND, AND WHAT THE BALL DID WITH IT',
             fontsize=11.5, fontfamily=COND, fontweight='700',
             color=TEXT_SECONDARY, va='top')
    fig.add_artist(Rectangle((RX, ry - 0.016), RW, 0.0012, facecolor=TEXT_FAINT,
                             edgecolor='none', alpha=0.55))
    V_X, S_X, T_X = RX + 0.150, RX + 0.228, RX + 0.296
    DIV_X = RX + 0.312
    O_X, C_X = RX + 0.376, RX + 0.435
    # The divider is the point: everything left of it is the release, everything
    # right of it is what came out. RTilt against OTilt is seam-shifted wake
    # stated in the units a coach already uses.
    for _x, _t in ((V_X, 'VELO'), (S_X, 'SPIN'), (T_X, 'RTILT'),
                   (O_X, 'OTILT'), (C_X, 'SEAM')):
        fig.text(_x, ry - 0.032, _t, fontsize=8.5, fontfamily=COND,
                 fontweight='700', color=TEXT_FAINT, va='top', ha='right')
    # Sits under RTILT, mirroring 'came out like this' under OTILT, right-aligned
    # so it stops short of the divider rather than straddling it.
    fig.text(T_X, ry - 0.049, 'released like this', fontsize=8.5,
             fontfamily=BODY, style='italic', color=TEXT_FAINT, va='top', ha='right')
    fig.text((DIV_X + C_X) / 2, ry - 0.049, 'came out like this', fontsize=8.5,
             fontfamily=BODY, style='italic', color=TEXT_FAINT, va='top', ha='center')
    ty = ry - 0.088
    fig.add_artist(Rectangle((DIV_X, ty - 0.030 * _f * _n + 0.012), 0.0010,
                             0.030 * _f * _n + 0.020,
                             facecolor=TEXT_FAINT, edgecolor='none', alpha=0.45))
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl, v, sp_, rt, ot, cx in rows:
        fig.add_artist(Rectangle((RX, ty - 0.001), 0.0055, 0.019,
                                 facecolor=c, edgecolor='none'))
        fig.text(RX + 0.013, ty + 0.014, PITCH_NAMES.get(pt, pt), fontsize=10.5,
                 fontfamily=BODY, fontweight='500', color=TEXT_PRIMARY, va='top')
        for _x, _v, _col in ((V_X, f'{v:.1f}', TEXT_SECONDARY),
                             (S_X, f'{sp_:.0f}', TEXT_SECONDARY),
                             (T_X, rt, TEXT_SECONDARY),
                             (O_X, ot, TEXT_PRIMARY),
                             (C_X, f'{cx:.1f}"', cd)):
            fig.text(_x, ty + 0.014, _v, fontsize=11, fontfamily=BODY,
                     fontweight='700' if _col is cd else '600', color=_col,
                     va='top', ha='right')
        ty -= 0.030 * _f
    aa, ext = d['ArmAngle'].mean(), d['Extension'].mean()
    fig.text(RX, ty + 0.010, f'arm angle {aa:.1f}°     extension {ext:.1f} ft'
             '     (the other two inputs, shared by every pitch)',
             fontsize=9.5, fontfamily=BODY, color=TEXT_FAINT, va='top')
    # The three right-hand columns state one fact three ways and the card never
    # said so, which is exactly what a reader asked about first.
    fig.text(RX, ty - 0.014,
             'SEAM is that rotation in inches: how far the break points away from where the '
             'spin alone would send it.\n'
             'Arm-side of the release axis is positive. It is the only column here computed '
             'with no model at all.',
             fontsize=9.5, fontfamily=BODY, color=TEXT_MUTED, va='top',
             linespacing=1.6)

    ff, si = d[d['Pitch Type'] == 'FF'], d[d['Pitch Type'] == 'SI']
    if len(ff) and len(si):
        gap = np.hypot(ff.ivb.mean() - si.ivb.mean(), ff.hb.mean() - si.hb.mean())

        def _cmin(a, b):
            """Signed clock difference in minutes, shortest way round."""
            return (b - a + 360) % 720 - 360

        _rg = abs(_cmin(_circ(ff._rmin), _circ(si._rmin)))
        _og = abs(_cmin(_circ(ff._omin), _circ(si._omin)))
        _dv_ = abs(ff.Velocity.mean() - si.Velocity.mean())
        _ds_ = abs(ff['Spin Rate'].mean() - si['Spin Rate'].mean())
        _same = ('at the same speed and spin' if _dv_ < 1.0 and _ds_ < 100
                 else f'{_dv_:.1f} mph and {_ds_:.0f} rpm apart')

        def _clock(mins):
            """Minutes read naturally: nobody says '110 minutes on the clock'."""
            h, m = int(mins // 60), int(round(mins % 60))
            if h == 0:
                return f'{m} minutes'
            return f'{h}h{m:02d}m' if m else f'{h} hour' + ('s' if h > 1 else '')
        cy2 = ty - 0.066   # clears the two-line SEAM note above
        fig.add_artist(Rectangle((RX, cy2), RW, 0.0012, facecolor=ACCENT,
                                 edgecolor='none', alpha=0.7))
        fig.text(RX, cy2 - 0.016,
                 f'Four-seam and sinker leave the hand {_rg:.0f} minutes of tilt apart, '
                 f'{_same}.\n'
                 f'They come out {_clock(_og)} apart on the clock and land {gap:.1f}" '
                 f'apart. That rotation is the seams.',
                 fontsize=9.8, fontfamily=BODY, color=TEXT_SECONDARY, va='top',
                 linespacing=1.6)

    fig.savefig(OUT, facecolor=BG, dpi=150)
    print(f'saved {OUT}')
    print(f'  {"pt":<5}{"IVB":>7}{"xIVB":>7}{"HB":>7}{"xHB":>7}{"pctl":>6}'
          f'{"velo":>7}{"spin":>7}{"RTilt":>7}{"OTilt":>7}{"seam":>7}')
    for pt, c, cd, n, a_i, e_i, a_h, e_h, pctl, v, sp_, rt, ot, cx in rows:
        print(f'  {pt:<5}{a_i:>7.1f}{e_i:>7.1f}{a_h:>7.1f}{e_h:>7.1f}{pctl:>6.0f}'
              f'{v:>7.1f}{sp_:>7.0f}{rt:>7}{ot:>7}{cx:>7.2f}')


if __name__ == '__main__':
    main()
