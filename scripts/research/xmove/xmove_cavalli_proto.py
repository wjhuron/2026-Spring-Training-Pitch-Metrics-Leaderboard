"""Cade Cavalli 2026: the shipped expected-movement model against the spin + axis one.

Three panels in the site's convention (disc = actual mean, hollow dashed ghost =
model expectation, connector = the OE vector):
  1. CURRENT   per (type, hand) MVN on [ArmAngle, Extension, Velocity], scored
               from the shipped metadata_rs.json mvnModels, per pitch.
  2. PROPOSED  per (type, hand) OLS on the same three plus Spin Rate and the
               release axis (2 harmonics + spin x axis), review Finding 5's
               interior optimum. Fit on the same 2026 MLB pitches the shipped
               model is fit on, so the only difference is the inputs.
  3. SHIFT     old ghost -> new ghost per type, with the decomposition of WHY:
               the part that comes from his release tilt, and the part from his
               spin rate, each measured by re-scoring at the league-typical
               value for that pitch type and hand.

Also prints the league SD of pitcher-level OE per type under both models, so
"unusual" has a ruler.

Usage: python3 scripts/research/xmove/xmove_cavalli_proto.py [--pitcher 'Cavalli, Cade']
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
from cards.pitcher import (PITCH_COLORS, PITCH_NAMES, PITCH_ORDER, BG, PLOT_PANEL,  # noqa: E402
                           GRID_COLOR, TEXT_MUTED, TEXT_SECONDARY, TEXT_FAINT)

PITCHER = 'Cavalli, Cade'
if '--pitcher' in sys.argv:
    PITCHER = sys.argv[sys.argv.index('--pitcher') + 1]
OUT = os.path.expanduser(f'~/Downloads/{PITCHER.split(",")[0].lower()}_xmove_proto.png')
MIN_N = 150          # matches MVN_MIN_N in process_data
MIN_SHOW = 10        # matches the player page's modeled-pitch floor


def clock_to_deg(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v)
    if ':' not in s:
        return pd.to_numeric(s, errors='coerce')
    h, m = s.split(':')
    return (float(h) % 12) * 30.0 + float(m) * 0.5


def load_2026():
    with open(f'{ROOT}/data/all_pitches_rs_cache.pkl', 'rb') as f:
        rows = pickle.load(f)
    keep = ['Pitcher', 'Throws', 'Pitch Type', '_source', '_game_pk', 'PitchID', 'Velocity',
            'Spin Rate', 'RTilt', 'ArmAngle', 'Extension', 'xIndVrtBrk', 'xHorzBrk']
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
    df = df[df['_source'].fillna('MLB') == 'MLB']
    for c in ['Velocity', 'Spin Rate', 'ArmAngle', 'Extension', 'xIndVrtBrk', 'xHorzBrk']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['rtilt'] = df['RTilt'].map(clock_to_deg)
    df = df.dropna(subset=['Velocity', 'Spin Rate', 'ArmAngle', 'Extension',
                           'xIndVrtBrk', 'xHorzBrk', 'rtilt', 'Throws', 'Pitch Type'])
    s = np.where(df['Throws'] == 'R', 1.0, -1.0)
    df['s'] = s
    df['ivb'] = df['xIndVrtBrk']
    df['hb_s'] = df['xHorzBrk'] * s          # hand-signed: positive = arm side
    th = np.radians(df['rtilt'].values * s)
    df['th'] = th
    df['spin_v'] = df['Spin Rate'] / df['Velocity']
    for k in (1, 2):
        df[f'h{k}s'], df[f'h{k}c'] = np.sin(k * th), np.cos(k * th)
        df[f'sv{k}s'], df[f'sv{k}c'] = df['spin_v'] * df[f'h{k}s'], df['spin_v'] * df[f'h{k}c']
    return df


OLD = ['ArmAngle', 'Extension', 'Velocity']
NEW = OLD + ['Spin Rate', 'h1s', 'h1c', 'h2s', 'h2c', 'sv1s', 'sv1c', 'sv2s', 'sv2c']


def X(df, feats):
    return np.column_stack([np.ones(len(df))] + [df[f].values.astype('f8') for f in feats])


def fit_per_group(df, feats):
    coefs = {}
    for (pt, thr), g in df.groupby(['Pitch Type', 'Throws']):
        if len(g) < MIN_N:
            continue
        A = X(g, feats)
        bi = np.linalg.lstsq(A, g['ivb'].values, rcond=None)[0]
        bh = np.linalg.lstsq(A, g['hb_s'].values, rcond=None)[0]
        coefs[(pt, thr)] = (bi, bh)
    return coefs


def score(df, feats, coefs):
    xi, xh = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for (pt, thr), (bi, bh) in coefs.items():
        m = ((df['Pitch Type'] == pt) & (df['Throws'] == thr)).values
        if m.any():
            A = X(df[m], feats)
            xi[m], xh[m] = A @ bi, A @ bh
    return xi, xh


def mvn_shipped(df):
    """Score with the shipped metadata mvnModels, per pitch (the site's numbers)."""
    meta = json.load(open(f'{ROOT}/data/metadata_rs.json'))['mvnModels']
    xi, xh = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for (pt, thr), idx in df.groupby(['Pitch Type', 'Throws']).indices.items():
        m = meta.get(f'{pt}_{thr}', {}).get('mlb')
        if not m:
            continue
        mu, cov = np.array(m['mu']), np.array(m['cov'])
        s22 = cov[2:, 2:]
        s12 = cov[:2, 2:]
        r = df.iloc[idx][OLD].values.astype('f8') - mu[2:]
        e = mu[:2] + r @ np.linalg.solve(s22, s12.T)
        xi[idx], xh[idx] = e[:, 0], e[:, 1] * df['s'].values[idx]
    return xi, xh


def unit_sd(df, xi, xh):
    """League SD of pitcher-level OE per (type, hand), 50+ pitches."""
    d = df.assign(oi=df['ivb'] - xi, oh=df['hb_s'] - xh).dropna(subset=['oi', 'oh'])
    u = d.groupby(['Pitcher', 'Pitch Type', 'Throws']).agg(n=('oi', 'size'), oi=('oi', 'mean'),
                                                             oh=('oh', 'mean')).reset_index()
    u = u[u.n >= 50]
    return u.groupby(['Pitch Type', 'Throws']).agg(sd_i=('oi', 'std'), sd_h=('oh', 'std'))


def draw_panel(ax, title, rows, label_ghost=True, arrows=None):
    ax.set_facecolor(PLOT_PANEL)
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25); ax.set_aspect('equal')
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.grid(True, alpha=0.5, color=GRID_COLOR)
    ax.axhline(0, color=GRID_COLOR, linestyle='--', linewidth=0.6)
    ax.axvline(0, color=GRID_COLOR, linestyle='--', linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color(TEXT_FAINT)
    ax.tick_params(labelsize=8, colors=TEXT_MUTED)
    ax.set_xlabel('Horizontal Break (in)', fontsize=9, color=TEXT_MUTED, fontweight='bold')
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=9, color=TEXT_MUTED, fontweight='bold')
    ax.set_title(title, fontsize=11, color=TEXT_SECONDARY, fontweight='bold', pad=8)
    for r in rows:
        c = PITCH_COLORS[r['pt']]
        if r.get('cloud') is not None:
            cx, cy = r['cloud']
            ax.scatter(cx, cy, s=12, color=c, alpha=0.18, linewidths=0, zorder=1)
        ax.scatter([r['aX']], [r['aY']], s=140, color=c, edgecolors=BG, linewidths=1.5, zorder=5)
        ax.annotate(r['pt'], (r['aX'], r['aY']), ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', zorder=6)
        if r.get('eX') is not None:
            ax.plot([r['eX'], r['aX']], [r['eY'], r['aY']], color=c, linewidth=1.2,
                    linestyle='--', alpha=0.9, zorder=3)
            ax.scatter([r['eX']], [r['eY']], s=140, facecolors='none', edgecolors=c,
                       linewidths=1.6, linestyle='--', zorder=4)
    if arrows:
        for a in arrows:
            c = PITCH_COLORS[a['pt']]
            ax.annotate('', xy=(a['x1'], a['y1']), xytext=(a['x0'], a['y0']),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.8), zorder=4)
            ax.scatter([a['x0']], [a['y0']], s=90, facecolors='none', edgecolors=c,
                       linewidths=1.0, linestyle=':', zorder=3, alpha=0.7)
            ax.scatter([a['x1']], [a['y1']], s=140, facecolors='none', edgecolors=c,
                       linewidths=1.6, linestyle='--', zorder=4)


def main():
    df = load_2026()
    print(f'{len(df):,} MLB 2026 pitches with every input', file=sys.stderr)
    old_fit = fit_per_group(df, OLD)
    new_fit = fit_per_group(df, NEW)
    xi_old, xh_old = mvn_shipped(df)
    xi_ref, xh_ref = score(df, OLD, old_fit)     # same form refit locally: reproduction check
    xi_new, xh_new = score(df, NEW, new_fit)
    ok = np.isfinite(xi_old) & np.isfinite(xi_ref)
    print(f'shipped-vs-local reproduction of the current model: max |dIVB| '
          f'{np.nanmax(np.abs(xi_old[ok] - xi_ref[ok])):.2f}", median '
          f'{np.nanmedian(np.abs(xi_old[ok] - xi_ref[ok])):.2f}" '
          f'(nonzero = metadata was fit on a different pitch set than this cache)')

    sd_old, sd_new = unit_sd(df, xi_old, xh_old), unit_sd(df, xi_new, xh_new)

    p = df[df['Pitcher'] == PITCHER].copy()
    if p.empty:
        sys.exit(f'no pitches for {PITCHER}')
    thr = p['Throws'].iloc[0]
    sgn = 1.0 if thr == 'R' else -1.0
    p['xi_old'], p['xh_old'] = xi_old[p.index.map(df.index.get_loc)], xh_old[p.index.map(df.index.get_loc)]
    p['xi_new'], p['xh_new'] = xi_new[p.index.map(df.index.get_loc)], xh_new[p.index.map(df.index.get_loc)]

    # league-typical tilt and spin per (type, hand), for the WHY decomposition
    # circular mean for tilt: clock degrees wrap at 12:00, a plain mean is wrong
    def circ_mean(deg):
        r = np.radians(deg)
        return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0
    lg = df.groupby(['Pitch Type', 'Throws']).agg(rtilt=('rtilt', circ_mean), spin=('Spin Rate', 'mean'),
                                                   aa=('ArmAngle', 'mean'))

    rows_old, rows_new, arrows, table = [], [], [], []
    for pt in PITCH_ORDER:
        g = p[p['Pitch Type'] == pt]
        if len(g) < MIN_SHOW:
            continue
        # HB back to the plot's frame (as the site shows it, not hand-signed)
        aX, aY = g['xHorzBrk'].mean(), g['ivb'].mean()
        oX, oY = (g['xh_old'].mean() * sgn if g['xh_old'].notna().sum() >= MIN_SHOW else None,
                  g['xi_old'].mean() if g['xi_old'].notna().sum() >= MIN_SHOW else None)
        nX, nY = (g['xh_new'].mean() * sgn if g['xh_new'].notna().sum() >= MIN_SHOW else None,
                  g['xi_new'].mean() if g['xi_new'].notna().sum() >= MIN_SHOW else None)
        cloud = (g['xHorzBrk'].values, g['ivb'].values)
        rows_old.append(dict(pt=pt, aX=aX, aY=aY, eX=oX, eY=oY, cloud=cloud))
        rows_new.append(dict(pt=pt, aX=aX, aY=aY, eX=nX, eY=nY, cloud=cloud))
        if oX is None or nX is None:
            continue
        arrows.append(dict(pt=pt, x0=oX, y0=oY, x1=nX, y1=nY))

        # WHY: re-score the new model with his tilt replaced by the league mean
        # for this type/hand, and again with his spin replaced. Each difference is
        # that input's contribution to the shift, at his own slot/ext/velo.
        key = (pt, thr)
        bi, bh = new_fit[key]
        def new_at(tilt_deg, spin):
            q = g.copy()
            th = np.radians(tilt_deg * sgn)
            q['Spin Rate'] = spin
            q['spin_v'] = spin / q['Velocity']
            for k in (1, 2):
                q[f'h{k}s'], q[f'h{k}c'] = np.sin(k * th), np.cos(k * th)
                q[f'sv{k}s'], q[f'sv{k}c'] = q['spin_v'] * q[f'h{k}s'], q['spin_v'] * q[f'h{k}c']
            A = X(q, NEW)
            return (A @ bi).mean(), (A @ bh).mean() * sgn
        base_i, base_h = nY, nX
        t_i, t_h = new_at(lg.loc[key, 'rtilt'], g['Spin Rate'])          # league tilt, his spin
        s_i, s_h = new_at(g['rtilt'], lg.loc[key, 'spin'])               # his tilt, league spin
        table.append(dict(
            pt=pt, n=len(g), velo=g['Velocity'].mean(), spin=g['Spin Rate'].mean(),
            lg_spin=lg.loc[key, 'spin'], tilt=circ_mean(g['rtilt']), lg_tilt=lg.loc[key, 'rtilt'],
            aa=g['ArmAngle'].mean(),
            aY=aY, aX=aX, oY=oY, oX=oX, nY=nY, nX=nX,
            tilt_i=base_i - t_i, tilt_h=base_h - t_h,
            spin_i=base_i - s_i, spin_h=base_h - s_h,
            sd_old=sd_old.loc[key].values, sd_new=sd_new.loc[key].values,
        ))

    # ---- print the table
    print(f'\n{PITCHER} 2026, {thr}HP. IVB / HB in inches, plot frame. OE = actual - expected.')
    print(f'{"pt":<4}{"n":>5}{"velo":>6}{"spin":>6}{"lg":>6}{"tilt":>7}{"lg":>6}'
          f'{"actual":>13}{"CURRENT x":>13}{"OE":>12}{"PROPOSED x":>13}{"OE":>12}'
          f'{"from tilt":>12}{"from spin":>12}')
    for t in table:
        print(f'{t["pt"]:<4}{t["n"]:>5}{t["velo"]:>6.1f}{t["spin"]:>6.0f}{t["lg_spin"]:>6.0f}'
              f'{t["tilt"]:>7.0f}{t["lg_tilt"]:>6.0f}'
              f'{t["aY"]:>7.1f}/{t["aX"]:>5.1f}'
              f'{t["oY"]:>7.1f}/{t["oX"]:>5.1f}'
              f'{t["aY"]-t["oY"]:>6.1f}/{t["aX"]-t["oX"]:>5.1f}'
              f'{t["nY"]:>7.1f}/{t["nX"]:>5.1f}'
              f'{t["aY"]-t["nY"]:>6.1f}/{t["aX"]-t["nX"]:>5.1f}'
              f'{t["tilt_i"]:>6.1f}/{t["tilt_h"]:>5.1f}'
              f'{t["spin_i"]:>6.1f}/{t["spin_h"]:>5.1f}')
    print('\nLeague SD of pitcher-level OE (50+ pitches), the ruler for "unusual":')
    print(f'{"pt":<4}{"current IVB/HB":>16}{"proposed IVB/HB":>17}{"his OE in SDs, current":>24}{"proposed":>10}')
    for t in table:
        so, sn = t['sd_old'], t['sd_new']
        zo = ((t['aY'] - t['oY']) / so[0], (t['aX'] - t['oX']) / so[1])
        zn = ((t['aY'] - t['nY']) / sn[0], (t['aX'] - t['nX']) / sn[1])
        print(f'{t["pt"]:<4}{so[0]:>8.2f}/{so[1]:>6.2f}{sn[0]:>9.2f}/{sn[1]:>6.2f}'
              f'{zo[0]:>14.1f}/{zo[1]:>5.1f}{zn[0]:>6.1f}/{zn[1]:>4.1f}')

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), facecolor=BG)
    draw_panel(axes[0], 'CURRENT: slot, extension, velocity', rows_old)
    draw_panel(axes[1], 'PROPOSED: + spin rate + release axis', rows_new)
    draw_panel(axes[2], 'SHIFT: where the ghost moves, and why',
               [dict(r, eX=None, eY=None, cloud=None) for r in rows_new], arrows=arrows)
    for t in table:
        dx, dy = t['nX'] - t['oX'], t['nY'] - t['oY']
        axes[2].annotate(f'{t["pt"]}  tilt {t["tilt"]:.0f}° vs {t["lg_tilt"]:.0f}°  '
                         f'spin {t["spin"]:.0f} vs {t["lg_spin"]:.0f}',
                         xy=(t['nX'], t['nY']), xytext=(6, 6), textcoords='offset points',
                         fontsize=6.5, color=PITCH_COLORS[t['pt']], zorder=7)
    handles = [Line2D([], [], marker='o', color='w', markerfacecolor=TEXT_MUTED, markersize=9,
                      linestyle='', label='disc = his actual mean'),
               Line2D([], [], marker='o', color=TEXT_MUTED, markerfacecolor='none', markersize=9,
                      linestyle='--', label='ghost = model expectation; dashed line = over expectation'),
               Line2D([], [], color=TEXT_MUTED, lw=1.8, label='arrow (right panel) = current ghost to proposed ghost')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'{PITCHER}  2026  expected movement, current model vs spin + release-axis model',
                 fontsize=14, color=TEXT_SECONDARY, fontweight='bold', y=0.98)
    fig.text(0.5, 0.925, 'Both models fit per pitch type and hand on the same 2026 MLB pitches. '
             'Right panel: each label gives his release tilt and spin against the league mean for that pitch type.',
             ha='center', fontsize=8.5, color=TEXT_MUTED)
    plt.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT, dpi=170, facecolor=BG)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
