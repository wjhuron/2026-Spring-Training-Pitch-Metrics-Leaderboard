#!/usr/bin/env python3
"""How much does a RETAG move the expected movement under option 2?

Wally's objection, and it is the right one: under a per-pitch-class model the
expectation depends on the label, so retagging a pitcher's sliders as sweepers
moves his expected movement even though not one pitch changed. Since hand
retagging is the differentiator of this dataset, a metric that moves when the
tag moves is measuring the tagger.

This quantifies it instead of arguing about it. For every pitch currently
tagged SL, score it under the SL-fitted model and again under the ST-fitted
model, and report how far the expectation shifts. Same for the FF/SI pair,
which is the other boundary Wally retags most.

A shift well inside the league residual spread means the objection is real but
small. A shift comparable to the residual itself means option 2 is measuring
the label.

Reported league-wide and for Medina specifically.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_retag_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402
from xmove_medina_plot import medina_2026, build_feats  # noqa: E402

FEATS = form(3, True, True)
PAIRS = [('SL', 'ST'), ('FF', 'SI'), ('SI', 'FF'), ('ST', 'SL'), ('FC', 'SL')]


def fit_for(A, hand, pt):
    tr = np.where((A['thr'] == hand) & (A['pt'] == pt))[0]
    if len(tr) < 2000:
        return None
    X = _design(A, FEATS, tr)
    bi = np.linalg.lstsq(X, A['ivb'][tr], rcond=None)[0]
    bh = np.linalg.lstsq(X, A['hb_s'][tr], rcond=None)[0]
    resid = np.hypot(A['ivb'][tr] - X @ bi, A['hb_s'][tr] - X @ bh)
    return bi, bh, np.median(resid)


def main():
    A = add_harmonics(load_np())
    hand = 'R'
    models = {}
    for pt in ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS'):
        m = fit_for(A, hand, pt)
        if m:
            models[pt] = m

    print(f'\nRETAG SHIFT — {hand}HP, how far the EXPECTATION moves when the '
          f'label changes\n(the pitches are identical; only the model used to '
          f'score them changes)\n')
    print(f'{"pitches tagged":<16}{"scored as":<12}{"n":>8}{"dIVB":>8}{"dHB":>8}'
          f'{"shift":>8}{"vs resid":>10}')
    print('-' * 70)
    for src, dst in PAIRS:
        if src not in models or dst not in models:
            continue
        idx = np.where((A['thr'] == hand) & (A['pt'] == src))[0]
        X = _design(A, FEATS, idx)
        bi_s, bh_s, med_resid = models[src]
        bi_d, bh_d, _ = models[dst]
        d_i = (X @ bi_d) - (X @ bi_s)
        d_h = (X @ bh_d) - (X @ bh_s)
        shift = np.median(np.hypot(d_i, d_h))
        print(f'{src:<16}{dst:<12}{len(idx):>8}{np.median(d_i):>8.2f}'
              f'{np.median(d_h):>8.2f}{shift:>8.2f}"{shift / med_resid:>9.2f}x')
    print('\n"vs resid" = shift divided by the median residual of the correct '
          'model.\n1.00x means a retag moves the expectation as much as the '
          'metric itself is worth.')

    # Medina, the concrete case.
    d26 = medina_2026()
    sign = 1.0
    F = build_feats(d26['Velocity'].values, d26['Spin Rate'].values,
                    d26['SpinAxis'].values, d26['Extension'].values,
                    d26['ArmAngle'].values, sign)
    print(f'\n\nMEDINA — his actual pitches, rescored under a different label')
    print(f'{"tagged":<10}{"n":>5}{"as":<8}{"xIVB":>8}{"xHB":>8}'
          f'{"-> dIVB":>10}{"dHB":>8}{"shift":>8}')
    print('-' * 66)
    for src, dst in (('SL', 'ST'), ('SI', 'FF'), ('FF', 'SI')):
        m = (d26['Pitch Type'] == src).values
        if m.sum() < 20 or src not in models or dst not in models:
            continue
        X = _design(F, FEATS, np.where(m)[0])
        bi_s, bh_s, _ = models[src]
        bi_d, bh_d, _ = models[dst]
        xi_s, xh_s = (X @ bi_s).mean(), (X @ bh_s).mean()
        xi_d, xh_d = (X @ bi_d).mean(), (X @ bh_d).mean()
        print(f'{src:<10}{int(m.sum()):>5}{dst:<8}{xi_s:>8.1f}{xh_s:>8.1f}'
              f'{xi_d - xi_s:>10.1f}{xh_d - xh_s:>8.1f}'
              f'{np.hypot(xi_d - xi_s, xh_d - xh_s):>8.1f}"')
        act_i = d26.loc[m, 'xIndVrtBrk'].mean()
        act_h = d26.loc[m, 'xHorzBrk'].mean()
        print(f'{"":10}{"":5}{"":8}actual {act_i:.1f} / {act_h:.1f}  ->  '
              f'OE as {src}: {np.hypot(act_i - xi_s, act_h - xh_s):.1f}"   '
              f'OE as {dst}: {np.hypot(act_i - xi_d, act_h - xh_d):.1f}"')


if __name__ == '__main__':
    main()
