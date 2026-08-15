"""Sonny Gray 2026 pitch-movement plot with expected-movement ellipses.

Renders the card's movement panel with, per pitch type:
  * the actual pitches (scatter) and their 1.5-sigma spread ellipse, dashed,
    exactly as Cards.py draws today
  * an EXPECTED ellipse at (xHB, xIVB) from the rebuilt model
  * an arrow from expected to actual -- the OE vector, whose length and
    direction are the whole story in one mark

Model: the sweep winner from scripts/research/xmove/xmove_sweep.py -- release-axis frame,
linear in spin, two axis harmonics, spin x axis tensor, fit per (pitch type,
hand) on 2026 MLB pitches only, which is how production refits each season.

Ellipse size is NOT the old arbitrary 7-inch disc. It is the 1-sigma covariance
of the LEAGUE distribution of pitcher-level OE for that pitch type and hand
(>=50 pitches), so "outside the ellipse" is falsifiable: roughly the outer
third of pitchers.

Usage:  python3 scripts/research/xmove/xmove_gray_plot.py ["Last, First"] [--season-tag 2026]
"""
import os, sys, math, pickle, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FONT_DIR = os.path.join(ROOT, 'assets', 'fonts')
if os.path.isdir(FONT_DIR):
    for fn in sorted(os.listdir(FONT_DIR)):
        if fn.lower().endswith(('.ttf', '.otf')):
            fm.fontManager.addfont(os.path.join(FONT_DIR, fn))

CACHE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT_DIR = os.path.join(os.path.expanduser('~'), 'Downloads')

# card palette (Cards.py)
PITCH_COLORS = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#D55E00',
                'ST': '#56B4E9', 'CU': '#332288', 'SV': '#882255', 'CH': '#009E73',
                'FS': '#CC79A7', 'KN': '#9A9A9A'}
PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'CU': 'Curveball', 'SV': 'Slurve', 'CH': 'Changeup',
               'FS': 'Splitter', 'KN': 'Knuckleball'}
PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS', 'KN']
BG, PLOT_PANEL, GRID_COLOR = '#f0e8d8', '#e8dfcb', '#c5b89f'
TEXT_PRIMARY, TEXT_SECONDARY = '#1a1612', '#3a3530'
TEXT_MUTED, TEXT_FAINT, ACCENT = '#6a5f55', '#8a7f75', '#9f3026'

MIN_GROUP = 150     # matches MVN_MIN_N in process_data.py
MIN_UNIT = 50       # pitcher-level floor for the league OE spread


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tilt_to_axis(rtilt):
    """RTilt clock string -> spin axis degrees. Inverse of
    Pitcher2026.spin_axis_to_tilt: hours = axis/30 - 6."""
    if not isinstance(rtilt, str) or ':' not in rtilt:
        return None
    try:
        h, m = rtilt.strip().split(':')[:2]
        hours = (int(h) % 12) + int(m) / 60.0
    except (ValueError, IndexError):
        return None
    return ((hours + 6.0) * 30.0) % 360.0


def load(season_source='MLB'):
    with open(CACHE, 'rb') as f:
        raw = pickle.load(f)
    rows = []
    for p in raw:
        if p.get('_source') != season_source:
            continue
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        ivb, hb = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        velo, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        ext, aa = sf(p.get('Extension')), sf(p.get('ArmAngle'))
        axis = tilt_to_axis(p.get('RTilt'))
        if None in (ivb, hb, velo, spin, ext, aa, axis) or thr not in ('L', 'R'):
            continue
        if pt not in PITCH_COLORS:
            continue
        rows.append((p.get('Pitcher'), thr, pt, ivb, hb, velo, spin, ext, aa, axis))
    d = {}
    arr = list(zip(*rows))
    d['pitcher'] = np.array(arr[0], dtype=object)
    d['thr'] = np.array(arr[1], dtype=object)
    d['pt'] = np.array(arr[2], dtype=object)
    s = np.where(d['thr'] == 'R', 1.0, -1.0)
    d['ivb'] = np.array(arr[3], dtype='f8')
    d['hb'] = np.array(arr[4], dtype='f8')          # natural frame, for plotting
    d['hb_s'] = d['hb'] * s                          # hand-mirrored, for modelling
    d['velo'] = np.array(arr[5], dtype='f8')
    d['spin'] = np.array(arr[6], dtype='f8')
    d['ext'] = np.array(arr[7], dtype='f8')
    d['aa'] = np.array(arr[8], dtype='f8')
    d['s'] = s
    th = np.radians(((np.array(arr[9], dtype='f8') - 180.0) % 360.0) * s)
    d['th'] = th
    d['ct'], d['st'] = np.cos(th), np.sin(th)
    d['along'] = d['ivb'] * d['ct'] + d['hb_s'] * d['st']
    d['cross'] = -d['ivb'] * d['st'] + d['hb_s'] * d['ct']
    return d


def design(d, idx):
    """Sweep winner: linear spin, 2 axis harmonics, spin/velo x axis tensor."""
    th = d['th'][idx]
    sv = d['spin'][idx] / d['velo'][idx]
    cols = [np.ones(len(idx)), d['ext'][idx], d['aa'][idx], d['aa'][idx] ** 2,
            d['spin'][idx], d['velo'][idx]]
    for h in (1, 2):
        cols += [np.sin(h * th), np.cos(h * th)]
    cols += [sv * np.sin(th), sv * np.cos(th),
             d['aa'][idx] * np.sin(th), d['aa'][idx] * np.cos(th)]
    return np.column_stack(cols)


def fit_and_score(d):
    """Per (pitch type, hand) fit on the whole league; return xIVB/xHB per pitch
    in the NATURAL frame so it can be plotted directly."""
    n = len(d['ivb'])
    x_ivb = np.full(n, np.nan)
    x_hb = np.full(n, np.nan)
    for pt in np.unique(d['pt']):
        for thr in ('L', 'R'):
            idx = np.flatnonzero((d['pt'] == pt) & (d['thr'] == thr))
            if len(idx) < MIN_GROUP:
                continue
            X = design(d, idx)
            ba = np.linalg.lstsq(X, d['along'][idx], rcond=1e-8)[0]
            bc = np.linalg.lstsq(X, d['cross'][idx], rcond=1e-8)[0]
            a, c = X @ ba, X @ bc
            xi = a * d['ct'][idx] - c * d['st'][idx]
            xh_s = a * d['st'][idx] + c * d['ct'][idx]
            x_ivb[idx] = xi
            x_hb[idx] = xh_s * d['s'][idx]      # back to the natural frame
    return x_ivb, x_hb


def league_oe_cov(d, x_ivb, x_hb):
    """1-sigma covariance of the LEAGUE distribution of pitcher-level OE, per
    (pitch type, hand). This is what sizes the expected ellipse -- a measured
    spread, not a constant radius."""
    out = {}
    ok = np.isfinite(x_ivb)
    oe_i = d['ivb'] - x_ivb
    oe_h = (d['hb'] - x_hb) * d['s']      # hand-signed so L and R pool
    for pt in np.unique(d['pt']):
        for thr in ('L', 'R'):
            m = ok & (d['pt'] == pt) & (d['thr'] == thr)
            if m.sum() < MIN_GROUP:
                continue
            pit = d['pitcher'][m]
            names, inv = np.unique(pit, return_inverse=True)
            cnt = np.bincount(inv)
            mi = np.bincount(inv, weights=oe_i[m]) / cnt
            mh = np.bincount(inv, weights=oe_h[m]) / cnt
            keep = cnt >= MIN_UNIT
            if keep.sum() < 20:
                continue
            out[(pt, thr)] = np.cov(np.vstack([mh[keep], mi[keep]]))
    return out


def plot(d, x_ivb, x_hb, covs, name, tag, out_path):
    m = d['pitcher'] == name
    if m.sum() == 0:
        print(f'no pitches for {name}')
        return
    thr = d['thr'][np.flatnonzero(m)[0]]
    fig = plt.figure(figsize=(9.5, 9.8), facecolor=BG)
    ax = fig.add_axes([0.10, 0.085, 0.86, 0.80])
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.axhline(0, color=GRID_COLOR, ls='--', lw=0.6)
    ax.axvline(0, color=GRID_COLOR, ls='--', lw=0.6)
    ax.set_xlabel('Horizontal Break (in)', fontsize=12, color=TEXT_MUTED,
                  fontweight='bold', fontfamily='IBM Plex Sans')
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=12, color=TEXT_MUTED,
                  fontweight='bold', fontfamily='IBM Plex Sans')
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.tick_params(labelsize=9.5, colors=TEXT_MUTED)
    ax.grid(True, alpha=0.5, color=GRID_COLOR)
    ax.set_facecolor(PLOT_PANEL)
    for sp in ax.spines.values():
        sp.set_color(TEXT_FAINT)

    rows = []
    present = [pt for pt in PITCH_ORDER if (m & (d['pt'] == pt)).sum() >= 6]
    for pt in present:
        k = m & (d['pt'] == pt) & np.isfinite(x_ivb)
        if k.sum() < 6:
            continue
        col = PITCH_COLORS[pt]
        ax_, ay_ = d['hb'][k], d['ivb'][k]
        ex, ey = np.nanmean(x_hb[k]), np.nanmean(x_ivb[k])

        # EXPECTED: filled ellipse at 1 sigma of the league pitcher-level OE spread
        C = covs.get((pt, thr))
        if C is not None:
            sgn = 1.0 if thr == 'R' else -1.0
            Cp = C.copy()
            Cp[0, 1] *= sgn; Cp[1, 0] *= sgn      # mirror the HB axis for LHP
            vals, vecs = np.linalg.eigh(Cp)
            if vals.min() > 0:
                ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
                ax.add_patch(Ellipse((ex, ey), 2 * math.sqrt(vals[1]),
                                     2 * math.sqrt(vals[0]), angle=ang,
                                     facecolor=col, edgecolor=col, alpha=0.20,
                                     lw=1.4, zorder=1))
        ax.plot([ex], [ey], marker='x', ms=9, mew=2.2, color=col, zorder=4)

        # ACTUAL: scatter + 1.5 sigma spread ellipse (as Cards.py draws it)
        ax.scatter(ax_, ay_, c=col, s=26, alpha=0.85, edgecolors=PLOT_PANEL,
                   linewidths=0.4, zorder=3)
        cv = np.cov(ax_, ay_)
        vals, vecs = np.linalg.eigh(cv)
        if vals.min() > 0:
            ax.add_patch(Ellipse((ax_.mean(), ay_.mean()), 2 * 1.5 * math.sqrt(vals[1]),
                                 2 * 1.5 * math.sqrt(vals[0]),
                                 angle=math.degrees(math.atan2(vecs[1, 1], vecs[0, 1])),
                                 fill=False, edgecolor=col, lw=1.3, ls='--',
                                 alpha=0.85, zorder=2))

        # the OE vector, expected -> actual
        dx, dy = ax_.mean() - ex, ay_.mean() - ey
        if math.hypot(dx, dy) > 0.6:
            ax.annotate('', xy=(ax_.mean(), ay_.mean()), xytext=(ex, ey),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8,
                                        alpha=0.95, shrinkA=3, shrinkB=3), zorder=5)
        # split the OE vector into the two physically distinct pieces:
        # ALONG the measured release axis = Magnus magnitude miss, which is
        # dominated by spin efficiency (gyro spin is not measured per pitch);
        # CROSS the axis = seam-shifted wake, which gyro cannot produce.
        ct, st = d['ct'][k].mean(), d['st'][k].mean()
        sgn = d['s'][k][0]
        dxs = dx * sgn                      # hand-signed HB deviation
        along_oe = dy * ct + dxs * st
        cross_oe = -dy * st + dxs * ct
        rtilt = math.degrees(math.atan2(d['st'][k].mean(), d['ct'][k].mean())) % 360
        otilt = math.degrees(math.atan2(d['hb_s'][k].mean(), d['ivb'][k].mean())) % 360
        rows.append((pt, k.sum(), ex, ey, ax_.mean(), ay_.mean(), dx, dy,
                     along_oe, cross_oe, d['cross'][k].mean(),
                     d['spin'][k].mean(), d['velo'][k].mean(), rtilt, otilt,
                     (otilt - rtilt + 180) % 360 - 180))

    handles = [mpatches.Patch(color=PITCH_COLORS[pt], label=f'{pt} — {PITCH_NAMES[pt]}')
               for pt in present]
    leg = ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.003),
                    ncol=min(len(present), 6), fontsize=9.5, frameon=False,
                    handlelength=1.2, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    fig.text(0.53, 0.965, f'{name.split(", ")[1]} {name.split(", ")[0]} — {tag} PITCH MOVEMENT',
             ha='center', fontsize=16, fontweight='bold', color=TEXT_SECONDARY,
             fontfamily='Bitter')
    fig.text(0.53, 0.936,
             'shaded + X = expected movement (release axis, spin, slot)   |   '
             'dashed = actual spread   |   arrow = over/under expected',
             ha='center', fontsize=8.6, color=TEXT_MUTED, fontfamily='IBM Plex Sans')
    fig.text(0.10, 0.022,
             'Expected ellipse = 1 SD of the league spread of pitcher-level '
             'deviations for that pitch type. Min. 6 pitches.',
             fontsize=7.6, color=TEXT_FAINT, fontfamily='IBM Plex Sans', style='italic')
    fig.savefig(out_path, dpi=150, facecolor=BG)
    print(f'wrote {out_path}\n')
    print(f"{'pt':>4} {'n':>5} {'velo':>5} {'spin':>6} {'RTilt':>6} {'OTilt':>6} "
          f"{'dev':>6} | {'xHB':>6} {'xIVB':>6} {'HB':>6} {'IVB':>6} | {'HBOE':>6} "
          f"{'IVBOE':>6} | {'alongOE':>8} {'crossOE':>8} {'SSWin':>6}")
    for (pt, n, ex, ey, ah, av, dx, dy, ao, co, cr, sp, vl, rt, ot, dv) in rows:
        print(f'{pt:>4} {n:>5} {vl:>5.1f} {sp:>6.0f} {rt:>6.1f} {ot:>6.1f} {dv:>6.1f} | '
              f'{ex:>6.1f} {ey:>6.1f} {ah:>6.1f} {av:>6.1f} | {dx:>6.1f} {dy:>6.1f} | '
              f'{ao:>8.1f} {co:>8.1f} {cr:>6.1f}')
    print('\n  alongOE = Magnus-magnitude miss along the measured release axis.'
          '\n            Dominated by SPIN EFFICIENCY: total spin is measured, the'
          '\n            gyro/transverse split is not, so a gyro-heavy pitch reads'
          '\n            as under-breaking. This is NOT seam-shifted wake.'
          '\n  crossOE = deviation PERPENDICULAR to the release axis. Gyro spin'
          '\n            scales break magnitude but cannot rotate it, so this is'
          '\n            gyro-independent and is the real SSW skill number.'
          '\n  SSWin   = raw cross-axis break (level, not vs expectation).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='Gray, Sonny')
    ap.add_argument('--tag', default='2026')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = load()
    print(f'{len(d["ivb"]):,} MLB pitches in the {a.tag} cache', flush=True)
    xi, xh = fit_and_score(d)
    covs = league_oe_cov(d, xi, xh)
    out = a.out or os.path.join(OUT_DIR, f'{a.name.split(", ")[0].lower()}_{a.tag}_movement_expected.png')
    plot(d, xi, xh, covs, a.name, a.tag, out)
