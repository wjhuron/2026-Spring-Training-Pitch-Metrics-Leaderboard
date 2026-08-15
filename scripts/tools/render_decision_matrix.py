#!/usr/bin/env python3
"""Render SD+ (and CT+) decision-matrix PNGs for the ROC article hitters.

Hitter-side analog of scripts/tools/render_command_map.py: a fine location grid
(same bins, bounds, fonts and cell grammar as the pitcher Loc+ command
maps) colored by decision quality at each spot. The pitcher map can bin
raw per-pitch atoms because a Loc+ atom is a deterministic function of
location; a per-pitch SD+ decision value is a signed, near-binary variable
(median 2 decisions per cell for these hitters), so raw cell means are
checkerboard noise. Instead each map colors the LEAGUE-RELATIVE smoothed
surface:

    dv_i  = RV(chosen) - RV(opposite) from the SHIPPED sdPlusWeights cell
            table in data/metadata_rs.json (exact SD+ currency; ROC hitters
            are scored against the MLB table, same translation framing as
            the shipped metric)
    L(x,z) = Gaussian-smoothed mean dv of ALL MLB decisions at that spot
    H(x,z) = (smoothed hitter dv mass + K_SHRINK * L) / (mass + K_SHRINK)
    excess = H - L   ... runs per decision above/below the MLB-average
                         decision-maker seeing the same pitch there

Only cells where the hitter actually saw pitches are drawn; opacity =
pitch count (command-map grammar), so the smoothing stabilizes color
without inventing coverage. Color polarity follows the HITTER convention
used by the card bubbles and the ZoneProfile renders: red = better than
MLB average, blue = costlier (the pitcher command map is the reverse).

CT+ panel: same machinery on swings only, with
    e_i = I[contact] - (1 - p_whiff[cell])   from ctPlusWeights,
i.e. contact made minus league expected contact for that (zone x count)
cell — the numerator/denominator atoms of shipped CT+ without the
leverage weights (leverage would make the map read as "value" rather
than "where does he beat the expected-contact rate", and BB+ already
owns damage). League-relative subtraction removes the within-zone
structure the cell table can't see.

Display conventions (not fitted constants): smoothing bandwidths reuse the
Loc+ locked surface smoothing (4.5in / 0.22 zone units — tuned for this
exact location domain in pipeline_locplus.py); K_SHRINK and the color
spans are display conventions chosen so dense regions keep ~70% of their
own signal and the span brackets the p5-p95 of drawn-cell excess (printed
per hitter at render time as a check).

Usage: python3 scripts/tools/render_decision_matrix.py
Outputs to ~/Downloads/ArticleVisuals/<Last>/.
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pipeline.sdplus import (  # noqa: E402
    is_eligible, classify_zone, classify_decision, get_count, cat_of,
)
from pipeline.contact import is_ct_eligible, classify_contact_outcome  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
HITTER_LB = os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')
METADATA = os.path.join(ROOT, 'data', 'metadata_rs.json')
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

HITTERS = ['Ortiz, Abimelec', 'Morales, Yohandy', 'King, Seaver',
           'Glasser, Phillip', 'Pinckney, Andrew']

# Computation grid: command-map bins, extended past the display window so
# the kernel never clips at the edges.
BIN_X = 2.0 / 12.0
BIN_Z = 0.10
GX_MIN, GX_MAX = -2.0, 2.0
GZ_MIN, GZ_MAX = -1.1, 2.1
NX = round((GX_MAX - GX_MIN) / BIN_X)
NZ = round((GZ_MAX - GZ_MIN) / BIN_Z)
# Display bounds mirror render_command_map.py / the JS canvas.
DX0, DX1 = -1.7, 1.7
DZ0, DZ1 = -0.5, 1.6

# Smoothing bandwidths: Loc+ locked surface smoothing (see module docstring).
SIGMA_X_BINS = (4.5 / 12.0) / BIN_X   # 2.25 bins
SIGMA_Z_BINS = 0.22 / BIN_Z           # 2.2 bins
K_SHRINK = 25.0   # pseudo-decisions toward the league surface (convention)

# Color spans (value units per DISPLAY convention, checked vs printed
# p5-p95 diagnostics): SD in runs/decision of excess, CT in contact-rate
# points of excess.
SPAN_SD = 0.030
SPAN_CT = 0.12


def heat_color(t):
    """Hitter polarity: t > 0.5 -> brick (better than league),
    t < 0.5 -> slate (costlier)."""
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def grid_index(p):
    """(i, j) on the computation grid, or None."""
    px, pz = sf(p.get('PlateX')), sf(p.get('PlateZ'))
    top, bot = sf(p.get('SzTop')), sf(p.get('SzBot'))
    if px is None or pz is None or not top or not bot or top <= bot:
        return None
    zn = (pz - bot) / (top - bot)
    i = int(math.floor((px - GX_MIN) / BIN_X))
    j = int(math.floor((zn - GZ_MIN) / BIN_Z))
    if 0 <= i < NX and 0 <= j < NZ:
        return (i, j)
    return None


def accumulate(pitches, value_fn):
    """Sum/count grids of value_fn over pitches."""
    s = np.zeros((NX, NZ))
    n = np.zeros((NX, NZ))
    for p in pitches:
        idx = grid_index(p)
        if idx is None:
            continue
        v = value_fn(p)
        if v is None:
            continue
        s[idx] += v
        n[idx] += 1
    return s, n


def _kernel(sigma, radius=7):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return k / k.sum()


# L2 mass of the normalized 2D kernel. The blurred count grid is a per-bin
# weighted DENSITY, not a neighborhood total: for locally uniform density n
# per bin, the effective number of decisions behind a smoothed cell mean is
# n_eff = blurred_n / W2 (variance-of-weighted-mean identity). K_SHRINK is
# specified in effective decisions, so it enters the bin-unit blend as
# K_SHRINK * W2.
W2 = float(np.sum(_kernel(SIGMA_X_BINS) ** 2) * np.sum(_kernel(SIGMA_Z_BINS) ** 2))


def gauss_blur(a, radius=7):
    """Separable truncated-Gaussian blur (pure numpy, no scipy)."""
    kx, kz = _kernel(SIGMA_X_BINS, radius), _kernel(SIGMA_Z_BINS, radius)
    pad = np.pad(a, radius, mode='constant')
    out = np.apply_along_axis(lambda m: np.convolve(m, kx, mode='same'), 0, pad)
    out = np.apply_along_axis(lambda m: np.convolve(m, kz, mode='same'), 1, out)
    return out[radius:-radius, radius:-radius]


def excess_surface(h_sum, h_cnt, lg_sum, lg_cnt):
    """League surface L, shrunk hitter surface H, and H - L."""
    bl_s, bl_n = gauss_blur(lg_sum), gauss_blur(lg_cnt)
    L = np.divide(bl_s, bl_n, out=np.zeros_like(bl_s), where=bl_n > 0)
    bh_s, bh_n = gauss_blur(h_sum), gauss_blur(h_cnt)
    k_bin = K_SHRINK * W2
    H = (bh_s + k_bin * L) / (bh_n + k_bin)
    return H - L


def draw_matrix(ax, excess, raw_cnt, span):
    """Command-map cell grammar: draw only cells the hitter actually saw
    pitches in; color = smoothed league-relative excess, opacity = count."""
    ax.set_xlim(DX0, DX1)
    ax.set_ylim(DZ0, DZ1)
    ax.set_aspect((DX1 - DX0) / (DZ1 - DZ0) / (250 / 300))
    ax.axis('off')
    ax.add_patch(Rectangle((DX0, DZ0), DX1 - DX0, DZ1 - DZ0,
                           facecolor=CREAM, edgecolor='none', zorder=0))
    n_max = raw_cnt.max()
    for i in range(NX):
        for j in range(NZ):
            n = raw_cnt[i, j]
            if n <= 0:
                continue
            xc = GX_MIN + (i + 0.5) * BIN_X
            zc = GZ_MIN + (j + 0.5) * BIN_Z
            if not (DX0 - BIN_X < xc < DX1 + BIN_X
                    and DZ0 - BIN_Z < zc < DZ1 + BIN_Z):
                continue
            t = 0.5 + max(-0.5, min(0.5, excess[i, j] / (2 * span)))
            alpha = 0.2 + 0.8 * math.sqrt(n / n_max)
            ax.add_patch(Rectangle((xc - BIN_X / 2, zc - BIN_Z / 2),
                                   BIN_X, BIN_Z, facecolor=heat_color(t),
                                   alpha=alpha, edgecolor='none', zorder=1))
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


def render_png(path, title, meta_line, excess, raw_cnt, span, footer):
    fig, ax = plt.subplots(figsize=(5, 6), dpi=200)
    fig.patch.set_facecolor(CREAM)
    draw_matrix(ax, excess, raw_cnt, span)
    ax.set_title(title, fontsize=13, color=INK, pad=14, **TITLE_FONT)
    ax.text(0.5, 1.015, meta_line, transform=ax.transAxes, ha='center',
            fontsize=8.5, color=(*INK, 0.75))
    ax.text(0.5, -0.03, footer, transform=ax.transAxes, ha='center',
            fontsize=7.0, color=(*INK, 0.6))
    fig.tight_layout()
    fig.savefig(path, facecolor=CREAM, bbox_inches='tight')
    plt.close(fig)


def diag(name, tag, excess, raw_cnt):
    vals = excess[raw_cnt > 0]
    print(f'  {name} {tag}: drawn cells {vals.size}, excess '
          f'p5 {np.percentile(vals, 5):+.4f}  p50 {np.percentile(vals, 50):+.4f}  '
          f'p95 {np.percentile(vals, 95):+.4f}')


def main():
    print('Loading pitch cache ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    meta = json.load(open(METADATA))
    sdw = meta['sdPlusWeights']
    ctw = meta['ctPlusWeights']
    lb = json.load(open(HITTER_LB))
    lb_row = {r['hitter']: r for r in lb
              if r.get('hitter') in HITTERS and r.get('team') == 'ROC'}

    def sd_dv(p):
        z, c, cat = classify_zone(p), get_count(p), cat_of(p)
        s = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|swing']['rv']
        t = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|take']['rv']
        return (s - t) if classify_decision(p) == 'swing' else (t - s)

    def ct_excess(p):
        z, c = classify_zone(p), get_count(p)
        cell = ctw.get(f'{z}|{c[0]}-{c[1]}')
        if cell is None:
            return None
        made = 1.0 if classify_contact_outcome(p) == 'contact' else 0.0
        return made - (1.0 - cell['p_whiff'])

    print('Building MLB league surfaces ...')
    mlb_dec = [p for p in allp
               if p.get('_source', 'MLB') == 'MLB' and is_eligible(p)]
    lg_sd = accumulate(mlb_dec, sd_dv)
    lg_ct = accumulate([p for p in mlb_dec if is_ct_eligible(p)], ct_excess)
    print(f'  {int(lg_sd[1].sum())} MLB decisions, '
          f'{int(lg_ct[1].sum())} MLB swings')

    by_hitter = defaultdict(list)
    for p in allp:
        if p.get('Batter') in HITTERS and p.get('BTeam') == 'ROC':
            by_hitter[p['Batter']].append(p)

    foot_sd = ('red = better than MLB-average decisions here · blue = '
               'costlier · opacity = pitches seen · catcher view')
    foot_ct = ('red = more contact than expected here · blue = more whiffs '
               '· opacity = swings · catcher view')

    for name in HITTERS:
        row = lb_row.get(name, {})
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)
        pitches = by_hitter[name]

        dec = [p for p in pitches if is_eligible(p)]
        h_sd = accumulate(dec, sd_dv)
        ex_sd = excess_surface(*h_sd, *lg_sd)
        diag(name, 'SD', ex_sd, h_sd[1])
        m = f'2026 season (ROC) · {int(h_sd[1].sum())} decisions'
        if row.get('sdPlus') is not None:
            m += f' · SD+ {row["sdPlus"]:.0f}'
            if row.get('sdPlus_pctl') is not None:
                m += f' ({row["sdPlus_pctl"]} pctl)'
        render_png(os.path.join(outdir, f'DecisionMatrix_{last}{first}_SD.png'),
                   f'{first} {last} — SD+ Decision Matrix', m,
                   ex_sd, h_sd[1], SPAN_SD, foot_sd)

        swings = [p for p in pitches if is_ct_eligible(p)]
        h_ct = accumulate(swings, ct_excess)
        ex_ct = excess_surface(*h_ct, *lg_ct)
        diag(name, 'CT', ex_ct, h_ct[1])
        m = f'2026 season (ROC) · {int(h_ct[1].sum())} swings'
        if row.get('ctPlus') is not None:
            m += f' · CT+ {row["ctPlus"]:.0f}'
            if row.get('ctPlus_pctl') is not None:
                m += f' ({row["ctPlus_pctl"]} pctl)'
        render_png(os.path.join(outdir, f'ContactMatrix_{last}{first}_CT.png'),
                   f'{first} {last} — CT+ Contact Matrix', m,
                   ex_ct, h_ct[1], SPAN_CT, foot_ct)
        print(f'{name} -> {outdir}')


if __name__ == '__main__':
    main()
