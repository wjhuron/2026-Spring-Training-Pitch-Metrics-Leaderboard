#!/usr/bin/env python3
"""Does conditioning on ARM ANGLE specifically make the movement residual more
valuable? (companion to xmove_value_test.py)

Wally's prior: what matters most is how different the movement is relative to
the pitcher's arm angle. xmove_value_test.py contrasts the shipped form against
S3b, but both contain arm angle, so neither isolates it. This does.

Four residuals, each scored the same way: incremental R^2 on run value per 100
at the rendered unit (pitcher x hand x pitch type x season, >= UNIT_MIN
pitches), over a base of velo + IVB + HB + spin.

  AA-ONLY   arm angle alone. The purest form of the prior: movement relative
            to slot, nothing else conditioned out.
  S1        arm angle + extension + velo (what ships today).
  S4        everything except arm angle (ext, velo, spin, axis, spin x axis).
  S3b       S4 plus arm angle. S3b minus S4 is arm angle's marginal value
            contribution, holding the rest of the model fixed.

Read S3b vs S4: if the prior holds, dropping arm angle should cost value. If
AA-ONLY beats the richer forms, the prior holds strongly. If S4 matches or beats
S3b, arm angle is not where the value lives, whatever it does for fit.

NULL BENCHMARK: k added regressors raise R^2 by about k/(n-k-1) on average even
when they are pure noise, so every increment is printed against that null
rather than against zero. An increment below its null row is nothing.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_value_armangle.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, run_linear, corr  # noqa: E402
from xmove_value_test import rv_lookup, units, r2_of  # noqa: E402

FORMS_AA = {
    'AA-ONLY': ['aa'],
    'S1 ship (aa,ext,v)': ['aa', 'ext', 'velo'],
    'S4 no arm angle': ['ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin', 'sv_cos'],
    'S3b with arm angle': ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin', 'sv_cos'],
}
PTS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']


def main():
    A = load_np()
    rv = rv_lookup()

    tables = {}
    for name, form in FORMS_AA.items():
        xi, xh = run_linear(A, form)
        u = units(A, xi, xh).merge(rv, on=['season', 'pitcher', 'thr', 'pt'])
        tables[name] = u
        print(f'  fitted {name:<20} units={len(u)}', file=sys.stderr)

    print('\nINCREMENTAL R^2 ON RUN VALUE/100 over velo+IVB+HB+spin')
    print('(residual pair added as 2 regressors; "null" = E[dR2] from 2 noise columns)\n')
    hdr = f'{"pt":>4}{"units":>7}{"base":>8}{"null":>8}'
    for name in FORMS_AA:
        hdr += f'{name.split()[0]:>10}'
    print(hdr)
    print('-' * len(hdr))

    agg = {name: [] for name in FORMS_AA}
    for pt in PTS:
        row_n = len(tables['S3b with arm angle'].query('pt == @pt'))
        if row_n < 150:
            continue
        null = 2.0 / (row_n - 3.0)
        b = tables['S3b with arm angle'].query('pt == @pt')
        base_cols = np.column_stack([b.velo, b.ivb, b.hb, b.spin])
        r0 = r2_of(b.rv100.values, base_cols)
        line = f'{pt:>4}{row_n:>7}{r0:>8.3f}{null:>8.4f}'
        for name in FORMS_AA:
            u = tables[name].query('pt == @pt')
            y = u.rv100.values
            bc = np.column_stack([u.velo, u.ivb, u.hb, u.spin])
            d = r2_of(y, np.column_stack([bc, u.ivb_oe, u.hb_oe])) - r2_of(y, bc)
            agg[name].append(d)
            line += f'{d:>10.4f}'
        print(line)

    print(f'\n{"MEAN across pitch types":<28}', end='')
    for name in FORMS_AA:
        print(f'{np.mean(agg[name]):>10.4f}', end='')
    print()
    print(f'{"beats its null in N types":<28}', end='')
    for name in FORMS_AA:
        wins = sum(1 for pt, d in zip([p for p in PTS
                                       if len(tables[name].query('pt == @p')) >= 150], agg[name])
                   if d > 2.0 / (len(tables[name].query('pt == @pt')) - 3.0))
        print(f'{wins:>10d}', end='')
    print(f'   (of {len(agg["S3b with arm angle"])})')

    print('\nARM ANGLE MARGINAL: S3b minus S4, per pitch type')
    for pt, a, b_ in zip([p for p in PTS if len(tables['S3b with arm angle'].query('pt == @p')) >= 150],
                         agg['S3b with arm angle'], agg['S4 no arm angle']):
        print(f'  {pt:>4}  {a - b_:+.4f}')


if __name__ == '__main__':
    main()
