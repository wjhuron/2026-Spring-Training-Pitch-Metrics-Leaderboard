#!/usr/bin/env python3
"""Is cross-axis break a pitcher skill, or mostly a pitch-type detector?

Cross-axis break (observed break perpendicular to the measured release axis)
looked like the best seam metric available: model-free, retag-proof, and it
flagged Medina's slider as a sweeper by physics. But its league means order
almost perfectly by pitch class (cutter -3.9" through sweeper +7.4"), which
raises the obvious worry: if the number is mostly "which pitch is this", it is
a classifier and not an evaluation metric, and per-pitcher differences would be
swamped exactly as they were for the pitch-type-agnostic residual.

Four tests, all WITHIN pitch class, because between-class variation is the part
we already know about:

  V1 VARIANCE   share of variance between classes vs within. High between-share
                means the headline number is mostly telling you the pitch type.
  V2 RELIABILITY split-half by game parity within a pitcher-type-season,
                Spearman-Brown corrected. Does the same arm repeat?
  V3 PERSISTENCE same pitcher x type, season N vs N+1. Is it a trait?
  V4 DISTINCTNESS corr(cross, raw HB) and corr(cross, raw IVB) within class. If
                cross just restates the movement column it is decoration.

Only V2 and V3 can tell us whether the WITHIN-class part is signal or noise,
which is the whole question for a leaderboard.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_cross_validate.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, corr  # noqa: E402

PTS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']
UNIT_MIN = 50
HALF_MIN = 25


def sb(r):
    """Spearman-Brown: a split-half r corrected to full length."""
    return (2 * r) / (1 + r) if r is not None and r > -1 else None


def main():
    A = load_np()
    d = pd.DataFrame({
        'pitcher': A['pitcher'], 'thr': A['thr'], 'pt': A['pt'],
        'season': A['season'], 'game': A['game'], 'cross': A['cross'],
        'ivb': A['ivb'], 'hb': A['hb_s'],
    })
    d['par'] = d.game % 2

    u = d.groupby(['pitcher', 'thr', 'pt', 'season']).agg(
        n=('cross', 'size'), cross=('cross', 'mean'),
        ivb=('ivb', 'mean'), hb=('hb', 'mean')).reset_index()
    u = u[u.n >= UNIT_MIN]
    print(f'{len(u):,} pitcher-hand-type-seasons\n')

    # V1 — variance decomposition
    print('V1 VARIANCE — where does cross-axis variance live?')
    gm = u['cross'].mean()
    between = ((u.groupby('pt')['cross'].transform('mean') - gm) ** 2).mean()
    within = ((u['cross'] - u.groupby('pt')['cross'].transform('mean')) ** 2).mean()
    tot = between + within
    print(f'  between pitch classes  {between / tot:6.1%}')
    print(f'  within  pitch classes  {within / tot:6.1%}')
    print(f'  (sd within class: {np.sqrt(within):.2f}", between: {np.sqrt(between):.2f}")')

    # V2 — split-half reliability within class
    h = d.groupby(['pitcher', 'thr', 'pt', 'season', 'par']).agg(
        n=('cross', 'size'), cross=('cross', 'mean')).reset_index()
    h = h[h.n >= HALF_MIN]
    w = h.pivot_table(index=['pitcher', 'thr', 'pt', 'season'], columns='par',
                      values='cross').dropna()
    w.columns = ['a', 'b']
    w = w.reset_index()

    # V3 — year over year
    nxt = u.copy()
    nxt['season'] = nxt['season'] - 1
    yoy = u.merge(nxt, on=['pitcher', 'thr', 'pt', 'season'],
                  suffixes=('', '_next'))

    print(f'\n{"pt":<5}{"n":>6}{"V2 rel":>9}{"n yoy":>7}{"V3 yoy":>9}'
          f'{"V4 vs HB":>10}{"V4 vs IVB":>11}')
    print('-' * 57)
    rows = []
    for pt in PTS:
        g = u[u.pt == pt]
        gw = w[w.pt == pt]
        gy = yoy[yoy.pt == pt]
        if len(g) < 100:
            continue
        rel = sb(corr(gw.a.values, gw.b.values)) if len(gw) >= 50 else None
        yo = corr(gy['cross'].values, gy['cross_next'].values) if len(gy) >= 50 else None
        chb = corr(g['cross'].values, g.hb.values)
        civ = corr(g['cross'].values, g.ivb.values)
        rows.append((pt, len(g), rel, len(gy), yo, chb, civ))
        print(f'{pt:<5}{len(g):>6}'
              f'{rel:>9.3f}' if rel is not None else f'{pt:<5}{len(g):>6}{"-":>9}', end='')
        print(f'{len(gy):>7}'
              f'{yo:>9.3f}' if yo is not None else f'{len(gy):>7}{"-":>9}', end='')
        print(f'{abs(chb):>10.3f}{abs(civ):>11.3f}')

    ok = [r for r in rows if r[2] is not None]
    if ok:
        print(f'\nmean within-class reliability {np.mean([r[2] for r in ok]):.3f}, '
              f'persistence {np.mean([r[4] for r in ok if r[4] is not None]):.3f}')
    print('\nRead: V2 and V3 are the ones that matter. Cross-axis is guaranteed to '
          'look\nimpressive across classes; the question is whether the within-class '
          'spread is\na trait the same arm repeats, or noise.')


if __name__ == '__main__':
    main()
