#!/usr/bin/env python3
"""Can option 1 + rank-within-type recover option 2, without the label?

Wally's question: why can option 2's benefit (residual = the pitcher, not his
pitch class) not be had pitch-type-agnostically?

The claim to test. Option 2's expectation is option 1's plus, roughly, a
per-class offset for that class's typical seam deflection. If that offset were
exactly constant within a class, it would cancel out of any within-class
comparison: a sinker ranked against other sinkers would land in the same place
under either model, and you could use the retag-proof agnostic number and do
the class comparison at DISPLAY time instead of inside the model. A retag would
then move only which pool a pitch is ranked in, never the number itself.

The catch: option 2 refits every coefficient per class, not just the intercept,
so the difference is not exactly a constant. This measures how close it is.

  corr    per pitch type, corr(OE_agnostic, OE_perclass) across pitcher-seasons.
          ~1.0 means the rankings are interchangeable and the label is only
          needed to choose the comparison pool.
  rank    Spearman, since ranking is what a percentile column actually does.
  sd      spread of (OE_agnostic - OE_perclass) within the class, in inches.
          If the offset were a pure constant this is 0.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_agnostic_vs_class.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, corr  # noqa: E402
from xmove_agnostic import fit_cv, unit_frame, PTS  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402

FEATS = form(3, True, True)


def spearman(a, b):
    return corr(pd.Series(a).rank().values, pd.Series(b).rank().values)


def main():
    A = add_harmonics(load_np())
    gid_pool = pd.factorize(pd.Series(A['thr']) + '_' +
                            pd.Series(A['season']).astype(str))[0]
    print('fitting agnostic (pooled by hand x season)...', file=sys.stderr)
    xi_a, xh_a = fit_cv(A, FEATS, gid_pool)
    print('fitting per-class (pitch type x hand x season)...', file=sys.stderr)
    xi_c, xh_c = fit_cv(A, FEATS, A['gid'])

    ua = unit_frame(A, xi_a, xh_a).set_index(['pitcher', 'thr', 'pt', 'season'])
    uc = unit_frame(A, xi_c, xh_c).set_index(['pitcher', 'thr', 'pt', 'season'])
    j = ua.join(uc, lsuffix='_a', rsuffix='_c', how='inner').reset_index()
    print(f'{len(j):,} pitcher-hand-type-seasons\n')

    print(f'{"pt":<5}{"n":>6}{"corr i":>9}{"rank i":>9}{"sd(diff) i":>12}'
          f'{"corr h":>9}{"rank h":>9}{"sd(diff) h":>12}')
    print('-' * 71)
    for pt in PTS:
        g = j[j.pt == pt]
        if len(g) < 100:
            continue
        di = g.ivb_oe_a.values - g.ivb_oe_c.values
        dh = g.hb_oe_a.values - g.hb_oe_c.values
        print(f'{pt:<5}{len(g):>6}'
              f'{corr(g.ivb_oe_a.values, g.ivb_oe_c.values):>9.3f}'
              f'{spearman(g.ivb_oe_a.values, g.ivb_oe_c.values):>9.3f}'
              f'{di.std():>12.2f}'
              f'{corr(g.hb_oe_a.values, g.hb_oe_c.values):>9.3f}'
              f'{spearman(g.hb_oe_a.values, g.hb_oe_c.values):>9.3f}'
              f'{dh.std():>12.2f}')

    print('\nFor scale, the SD of the per-class residual itself (what a')
    print('within-class comparison is actually spreading pitchers over):')
    print(f'{"pt":<5}{"sd OE_perclass i":>19}{"sd OE_perclass h":>19}')
    for pt in PTS:
        g = j[j.pt == pt]
        if len(g) < 100:
            continue
        print(f'{pt:<5}{g.ivb_oe_c.std():>19.2f}{g.hb_oe_c.std():>19.2f}')


if __name__ == '__main__':
    main()
