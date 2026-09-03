"""locplus_bw_prototype.py — visual prototype of the per-group bandwidth
options for Loc+, on Grant Taylor (CWS, 475 FF / 274 CU in 2026).

Three configs side by side, FF and CU rows:
  current   all groups at 4.5" horizontal
  moderate  FF/SI/CU at 9.0"
  full      FF/CU x-flat (200"), SI 9.0"

Each panel: league ExpRV surface (vs RHB, RHP, 0-0 count, the most common
slice), Taylor's pitch locations overlaid, and his group percentile among
MLB pitchers under that config (group-restricted mean score, min 150 FF /
100 CU pitches). Output: ~/Downloads/locplus_bw_prototype_taylor.png

Usage: python3 scripts/research/locplus/locplus_bw_prototype.py
"""
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

import pipeline.locplus as lp
import locplus_constants_multiseason as base

LG, SCALE = 0.3172, 1.2343
PITCHER = 'Taylor, Grant'
CONFIGS = [
    ('Current (4.5″)', {}),
    ('Moderate (9″ FF/SI/CU)', {'FF': (9.0, 0.22), 'SI': (9.0, 0.22), 'CU': (9.0, 0.22)}),
    ('Full (FF/CU x-flat, SI 9″)', {'FF': (200.0, 0.22), 'SI': (9.0, 0.22), 'CU': (200.0, 0.22)}),
]
GROUPS = [('FF', 150), ('CU', 100)]

CREAM = '#F7F1E5'
INK = '#2B2320'
CMAP = LinearSegmentedColormap.from_list(
    'loc_div', ['#2E5F6E', '#7FA3AC', CREAM, '#D89B7E', '#C0533B'])


def league_map(S, grp, count=(0, 0), bh='R', ph='R'):
    """ExpRV over the grid for one (group, hands, count) slice."""
    key = (grp, bh, ph)
    M = np.full((lp.NZ, lp.NX), np.nan)
    RV = S['RV']
    for i in range(lp.NX):
        for j in range(lp.NZ):
            psw = S['SW'][key][count][i][j]
            # 2026-09-02: live S['WH'][key] is per-count; base.wh_at reads either shape.
            pwh = base.wh_at(S, key, count, i, j)
            pfl = S['FL'][key][i][j]
            pbip = max(0.0, 1.0 - pwh - pfl)
            vbip = S['XW'][key][i][j] + S['BIPOFF'].get(count, 0.0)
            pcs = S['PCS'][bh][count][i][j]
            sw = (pwh * RV['whiff'].get(count, 0.0) + pfl * RV['foul'].get(count, 0.0)
                  + pbip * vbip)
            tk = pcs * RV['cs'].get(count, 0.0) + (1 - pcs) * RV['ball'].get(count, 0.0)
            M[j, i] = psw * sw + (1 - psw) * tk
    return M


def group_scores(baseline, S, grp, min_n):
    """Per-pitcher mean score over their pitches of group grp."""
    acc = defaultdict(list)
    for p in baseline:
        if lp.group_of(p) != grp:
            continue
        s = lp.score_pitch(p, S)
        if s is not None:
            acc[(p.get('Pitcher'), p.get('Throws'))].append(s)
    return {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= min_n}


def main():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    base = [p for p in D if p.get('_source', 'MLB') == 'MLB'
            and lp.is_eligible_baseline(p)]
    del D
    print(f"baseline: {len(base)} pitches", flush=True)

    results = []
    for name, bwpt in CONFIGS:
        lp.PHYS_BW_PT = bwpt
        try:
            S = lp.build_surfaces(base, LG, SCALE)
            cfg = {'name': name, 'maps': {}, 'pctl': {}, 'taylor': {}}
            for grp, min_n in GROUPS:
                cfg['maps'][grp] = league_map(S, grp)
                sc = group_scores(base, S, grp, min_n)
                tk = (PITCHER, 'R')
                if tk in sc:
                    vals = sorted(sc.values(), reverse=True)  # high = bad
                    below = sum(1 for v in vals if v > sc[tk])
                    cfg['pctl'][grp] = round(100.0 * below / (len(vals) - 1), 0)
                    cfg['taylor'][grp] = sc[tk]
            results.append(cfg)
            print(f"{name}: done "
                  + '  '.join(f"{g} pctl {cfg['pctl'].get(g)}" for g, _ in GROUPS),
                  flush=True)
        finally:
            lp.PHYS_BW_PT = {}

    tay = defaultdict(list)
    P2 = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    for p in P2:
        if p.get('Pitcher') == PITCHER and p.get('_source', 'MLB') == 'MLB':
            g = lp.group_of(p)
            if g in ('FF', 'CU'):
                px = lp.safe_float(p.get('PlateX'))
                zn = lp._znorm(p)
                if px is not None and zn is not None:
                    tay[g].append((px, zn))
    del P2

    lo = min(np.nanmin(c['maps'][g]) for c in results for g, _ in GROUPS)
    hi = max(np.nanmax(c['maps'][g]) for c in results for g, _ in GROUPS)
    mid = float(np.nanmean(results[0]['maps']['FF']))
    norm = TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2), facecolor=CREAM)
    ext = [lp.X_MIN, lp.X_MAX, lp.Z_MIN, lp.Z_MAX]
    for r, (grp, _) in enumerate(GROUPS):
        for c, cfg in enumerate(results):
            ax = axes[r][c]
            ax.imshow(cfg['maps'][grp], origin='lower', extent=ext,
                      aspect='auto', cmap=CMAP, norm=norm,
                      interpolation='bilinear')
            ax.add_patch(Rectangle((-0.83, 0.0), 1.66, 1.0, fill=False,
                                   edgecolor=INK, linewidth=1.4))
            if tay[grp]:
                xs, zs = zip(*tay[grp])
                ax.scatter(xs, zs, s=7, c=INK, alpha=0.45, linewidths=0)
            pct = cfg['pctl'].get(grp)
            ax.set_title(f"{cfg['name']}\n{grp} Loc percentile: "
                         f"{int(pct) if pct is not None else '--'}",
                         fontsize=10.5, color=INK)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.6, 1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(INK)
        axes[r][0].set_ylabel(f"{grp}  (catcher view)", fontsize=11, color=INK)

    fig.suptitle(f"Loc+ bandwidth options — league value surfaces (vs RHB, 0-0) "
                 f"with {PITCHER}'s pitches", fontsize=13, color=INK)
    fig.text(0.5, 0.015,
             'Teal = pitcher-favorable location value, terracotta = hitter-favorable, cream = league mean. '
             'Dots: Grant Taylor 2026 pitch locations. Percentile: group-restricted Loc among MLB pitchers under that config.',
             ha='center', fontsize=8.5, color=INK)
    out = os.path.expanduser('~/Downloads/locplus_bw_prototype_taylor.png')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out, dpi=150, facecolor=CREAM)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
