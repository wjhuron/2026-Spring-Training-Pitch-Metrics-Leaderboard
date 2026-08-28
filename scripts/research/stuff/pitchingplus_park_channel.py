#!/usr/bin/env python3
"""pitchingplus_park_channel.py — park adjustment for the Pitching+ xRV
slot (approved by Wally 2026-08-28).

The outing grade's xRV/100 is venue-blind: a Coors outing grades low and a
pitcher-park outing grades high for reasons that are not the pitcher. This
measures the PASS-THROUGH of the park run factor into per-outing
pitcher-perspective xRV/100 (it is not full: xRV's non-BIP majority is
RunExp deltas that barely see the park, and the BIP channel is a
league-wide xwOBA model), then evaluates the adjusted grade.

  pass-through  WLS outing xrv100 ~ pfdev, weight = pitches, per season
                (5 replicates) + pooled; pfdev = PF/100 - 1 from
                data/park_factors.json via data/_mlb_gamepk_home.json
  adjustment    xrv100_adj = xrv100 - b * pfdev   (b = pooled slope)
  grade         frozen Pitching+ weights, pool params rebuilt from the
                adjusted component (everything else unchanged)
  validation    1. venue bias: per-venue mean grade SD, before vs after
                2. next-outing 2x2: {raw, adj} grade -> {raw, adj} next
                   outing xrv100
                3. within-outing split-half r (reported only: park is
                   shared across halves, so this objective PENALIZES
                   removing shared park variance by construction)

Usage: PYTHONHASHSEED=0 python3 scripts/research/stuff/pitchingplus_park_channel.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pitcherplus_outing_grade import load, MIN_N  # noqa: E402

KMAP = {'stuffRaw': 42.0, 'locRaw': 185.0, 'cswPct': 398.0, 'xrv100': 1581.0}
W = {'stuffRaw': .205, 'locRaw': .169, 'cswPct': .252, 'xrv100': .374}


def add_park(t):
    pf = json.load(open(os.path.join(DATA, 'park_factors.json')))
    gh = json.load(open(os.path.join(DATA, '_mlb_gamepk_home.json')))
    t = t.copy()
    t['home_id'] = t['game_pk'].astype(int).astype(str).map(gh)
    t['pfdev'] = [
        (pf.get(str(int(s)), {}).get(str(int(h)), 100.0) / 100.0 - 1.0)
        if pd.notna(h) else np.nan
        for s, h in zip(t['season'], t['home_id'])]
    print(f'venue matched {t["home_id"].notna().mean():.4f}, '
          f'pfdev range {t.pfdev.min():.2f}..{t.pfdev.max():.2f}')
    return t


def wls_slope(x, y, w):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[m], y[m], w[m]
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    b = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
    return b, len(x)


def grade_all(t, xcol):
    """Frozen-weight grade for every pooled outing, using xcol as the xRV
    component. Returns the frame with 'grade_<xcol>'."""
    t = t.copy()
    comp = np.zeros(len(t))
    feats = {'stuffRaw': 'stuffRaw', 'locRaw': 'locRaw',
             'cswPct': 'cswPct', 'xrv100': xcol}
    for season, g in t.groupby('season'):
        seas = np.zeros(len(g))
        for f, col in feats.items():
            vals = g[col].dropna()
            mu, sd = float(vals.mean()), float(vals.std())
            z = ((g[col] - mu) / sd).fillna(0.0)
            seas += W[f] * (z * (g['n'] / (g['n'] + KMAP[f]))).to_numpy()
        comp[t.index.get_indexer(g.index)] = seas
    mu, sd = comp.mean(), comp.std(ddof=1)
    t['grade_' + xcol] = 100 + 10 * (comp - mu) / sd
    return t


def main():
    t = load()
    t = t[t['n'] >= MIN_N].reset_index(drop=True)
    t = add_park(t)

    print('\n══ pass-through: outing xrv100 ~ pfdev (WLS by pitches) ══')
    slopes = []
    for season, g in t.groupby('season'):
        b, n = wls_slope(g['pfdev'].to_numpy(float),
                         g['xrv100'].to_numpy(float),
                         g['n'].to_numpy(float))
        slopes.append(b)
        print(f'  {season}: b = {b:+.3f} runs/100 per park unit (n {n})')
    b_pool, n_pool = wls_slope(t['pfdev'].to_numpy(float),
                               t['xrv100'].to_numpy(float),
                               t['n'].to_numpy(float))
    print(f'  pooled: b = {b_pool:+.3f} (n {n_pool}); season spread '
          f'{np.std(slopes):.3f}')

    t['xrv100_adj'] = t['xrv100'] - b_pool * t['pfdev']
    t = grade_all(t, 'xrv100')
    t = grade_all(t, 'xrv100_adj')

    print('\n══ venue bias (per-venue mean grade, outings >= 50) ══')
    vm = t.groupby('home_id').agg(raw=('grade_xrv100', 'mean'),
                                  adj=('grade_xrv100_adj', 'mean'),
                                  n=('n', 'size'))
    vm = vm[vm['n'] >= 50]
    print(f'  SD of venue means: raw {vm.raw.std():.3f} -> adj '
          f'{vm.adj.std():.3f}')
    ex = vm.loc[vm.raw.idxmin()], vm.loc[vm.raw.idxmax()]
    print(f'  lowest venue mean: raw {ex[0].raw:.2f} -> {ex[0].adj:.2f} '
          f'(club {vm.raw.idxmin()})')
    print(f'  highest venue mean: raw {ex[1].raw:.2f} -> {ex[1].adj:.2f} '
          f'(club {vm.raw.idxmax()})')

    print('\n══ next-outing 2x2 (same season, r) ══')
    t = t.sort_values(['pid', 'season', 'date'])
    for tgt in ('xrv100', 'xrv100_adj'):
        t['_next_' + tgt] = t.groupby(['pid', 'season'])[tgt].shift(-1)
    for pred in ('grade_xrv100', 'grade_xrv100_adj'):
        row = []
        for tgt in ('xrv100', 'xrv100_adj'):
            a = t[pred].to_numpy(float)
            b = t['_next_' + tgt].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            row.append(f'{np.corrcoef(a[m], b[m])[0, 1]:.4f} (n {m.sum()})')
        lab = 'raw grade' if pred == 'grade_xrv100' else 'adj grade'
        print(f'  {lab:10s} -> next raw {row[0]}   -> next adj {row[1]}')

    # split-half r, reported only
    hp = t[(t['n_o'] >= 8) & (t['n_e'] >= 8)]
    for col, lab in (('xrv100', 'raw'), ('xrv100_adj', 'adj')):
        # adjusted halves share the outing's pfdev
        if col == 'xrv100_adj':
            o = hp['xrv100_o'] - b_pool * hp['pfdev']
            e = hp['xrv100_e'] - b_pool * hp['pfdev']
        else:
            o, e = hp['xrv100_o'], hp['xrv100_e']
        m = o.notna() & e.notna()
        print(f'  (report only) half-half xrv r, {lab}: '
              f'{np.corrcoef(o[m], e[m])[0, 1]:.4f}')


if __name__ == '__main__':
    main()
