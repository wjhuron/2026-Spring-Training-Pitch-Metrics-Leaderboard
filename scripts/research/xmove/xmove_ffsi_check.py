"""Check Wally's premise directly: for a pitcher who throws both a four-seam
and a sinker, is the RELEASE axis really near-identical while the OBSERVED
break direction diverges?

If true, an expected-movement model built on RTilt alone predicts the same
shape for both pitches -- which is the failure mode to design around. If false,
RTilt carries the FF/SI distinction on its own and no pitch-class seam term is
needed. Measured, not assumed.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_audit import load, add_axis_trig
from xmove_polar import prep

MIN = 50


def circ_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


if __name__ == '__main__':
    df = prep(add_axis_trig(load()))
    g = (df.groupby(['season', 'pitcher', 'thr', 'pt'])
           .agg(n=('ivb', 'size'), rt=('rtilt', 'mean'), ot=('otilt', 'mean'),
                ivb=('ivb', 'mean'), hb=('hb_s', 'mean'), cross=('cross', 'mean'),
                spin=('spin', 'mean'), velo=('velo', 'mean'))
           .reset_index())
    g = g[g.n >= MIN]
    for a, b in (('FF', 'SI'), ('FF', 'FC'), ('SL', 'ST'), ('FF', 'CH')):
        A = g[g.pt == a]; B = g[g.pt == b]
        m = A.merge(B, on=['season', 'pitcher', 'thr'], suffixes=('_a', '_b'))
        if len(m) < 30:
            continue
        s = np.where(m.thr == 'R', 1.0, -1.0)
        drt = circ_diff(m.rt_b, m.rt_a) * s
        dot = circ_diff(m.ot_b, m.ot_a) * s
        print(f'\n{a} -> {b}   n={len(m)} pitcher-seasons with both (>= {MIN} each)')
        print(f'  release-axis shift   RTilt_{b} - RTilt_{a}:  '
              f'mean {drt.mean():>6.1f} deg   sd {drt.std():>5.1f}   '
              f'median |d| {np.median(abs(drt)):>5.1f}')
        print(f'  observed-break shift OTilt_{b} - OTilt_{a}:  '
              f'mean {dot.mean():>6.1f} deg   sd {dot.std():>5.1f}   '
              f'median |d| {np.median(abs(dot)):>5.1f}')
        print(f'  => {abs(dot.mean()) - abs(drt.mean()):>5.1f} deg of the '
              f'{abs(dot.mean()):.1f} deg shape separation is NOT in the release axis')
        print(f'  cross-axis (non-Magnus) break, inches: '
              f'{a} {m.cross_a.mean():>5.2f}   {b} {m.cross_b.mean():>5.2f}   '
              f'delta {m.cross_b.mean() - m.cross_a.mean():>5.2f}')
        print(f'  corr(RTilt_{a}, RTilt_{b}) = {np.corrcoef(m.rt_a, m.rt_b)[0,1]:.3f}')
