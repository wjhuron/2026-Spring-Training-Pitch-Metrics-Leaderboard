#!/usr/bin/env python3
"""Does OTilt minus RTilt identify the most seam-shifted wake?

Wally's proposition: the pitcher whose sinker's observed break direction sits
furthest from its measured release axis has the most SSW.

The mechanism is right. Gyro spin scales break magnitude but does not rotate
it, and IVB is already gravity-removed, so a rotation of the break away from
the release axis is a genuine non-Magnus signature.

The question is the UNIT. Deviation in degrees and deflection in inches are not
the same ranking, because

    cross (inches) = total break x sin(tilt gap)

so a pitch with little break needs only a small sideways force to swing its
angle a long way. The review already recorded the symptom: slider axis
deviation has sd 40.6 degrees, by far the widest class, precisely because gyro
sliders barely break at all. If degrees and inches disagree, degrees is
measuring the denominator.

Tests, at the pitcher x hand x pitch type x season unit (>= 50 pitches):
  1. Per pitch type: tilt gap in degrees vs cross-axis deflection in inches.
  2. Within each type, corr(|tilt gap|, |cross|). If ~1 the shortcut is safe.
  3. Top 5 sinkers by each measure. Do the two lists name the same arms?

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_tilt_gap.py
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


def main():
    A = load_np()
    # load_np already builds the release-axis frame, hand-signed:
    #   along = break along the measured release axis (Magnus direction)
    #   cross = break perpendicular to it (the non-Magnus, seam part)
    # so the tilt gap is just the angle of the break vector in that frame.
    gap_deg = np.degrees(np.arctan2(A['cross'], A['along']))
    total = np.hypot(A['ivb'], A['hb_s'])

    d = pd.DataFrame({
        'pitcher': A['pitcher'], 'thr': A['thr'], 'pt': A['pt'],
        'season': A['season'], 'gap': gap_deg, 'cross': A['cross'],
        'along': A['along'], 'total': total,
    })
    u = d.groupby(['pitcher', 'thr', 'pt', 'season']).agg(
        n=('gap', 'size'), gap=('gap', 'mean'), cross=('cross', 'mean'),
        along=('along', 'mean'), total=('total', 'mean')).reset_index()
    u = u[u.n >= UNIT_MIN]
    print(f'{len(u):,} pitcher-hand-type-seasons\n')

    print('PER PITCH TYPE — tilt gap (deg) against cross-axis break (in)')
    print(f'{"pt":<5}{"n":>6}{"gap deg":>10}{"sd":>7}{"cross in":>10}{"sd":>7}'
          f'{"break in":>10}')
    print('-' * 55)
    for pt in PTS:
        g = u[u.pt == pt]
        if len(g) < 50:
            continue
        print(f'{pt:<5}{len(g):>6}{g.gap.mean():>10.1f}{g.gap.std():>7.1f}'
              f'{g["cross"].mean():>10.2f}{g["cross"].std():>7.2f}'
              f'{g.total.mean():>10.1f}')

    print('\nDOES THE DEGREE RANKING MATCH THE INCH RANKING?')
    print('corr within pitch type, and Spearman since this is about ordering')
    print(f'{"pt":<5}{"n":>6}{"corr":>9}{"rank":>9}')
    print('-' * 29)
    for pt in PTS:
        g = u[u.pt == pt]
        if len(g) < 50:
            continue
        a, b = g.gap.abs().values, g['cross'].abs().values
        rk = corr(pd.Series(a).rank().values, pd.Series(b).rank().values)
        print(f'{pt:<5}{len(g):>6}{corr(a, b):>9.3f}{rk:>9.3f}')

    print('\nSINKERS — top 5 by each measure (RHP, 2025)')
    si = u[(u.pt == 'SI') & (u.thr == 'R') & (u.season == 2025)].copy()
    si['abs_gap'] = si.gap.abs()
    si['abs_cross'] = si['cross'].abs()
    by_deg = si.nlargest(5, 'abs_gap')
    by_in = si.nlargest(5, 'abs_cross')
    print(f'\n{"by TILT GAP (deg)":<34}{"by CROSS (in)":<34}')
    for (_, r1), (_, r2) in zip(by_deg.iterrows(), by_in.iterrows()):
        left = f'{r1.pitcher[:20]:<22}{r1.gap:>5.1f}d{r1["cross"]:>5.1f}"'
        right = f'{r2.pitcher[:20]:<22}{r2.gap:>5.1f}d{r2["cross"]:>5.1f}"'
        print(f'{left:<34}{right:<34}')
    overlap = len(set(by_deg.pitcher) & set(by_in.pitcher))
    print(f'\noverlap between the two top-5 lists: {overlap} of 5')

    # Why they can disagree: the small-break arms
    print('\nSMALL-BREAK ARMS INFLATE THE ANGLE')
    print('sliders, split by total break, to show the denominator effect')
    sl = u[u.pt == 'SL'].copy()
    lo = sl[sl.total < sl.total.quantile(0.25)]
    hi = sl[sl.total > sl.total.quantile(0.75)]
    for lab, g in (('least break (bottom 25%)', lo), ('most break (top 25%)', hi)):
        print(f'  {lab:<26} break {g.total.mean():>5.1f}"   '
              f'|gap| {g.gap.abs().mean():>5.1f}deg   '
              f'|cross| {g["cross"].abs().mean():>5.2f}"')


if __name__ == '__main__':
    main()
