"""Sonny Gray FF vs SI: what the shipped model says vs the release-axis model.

This is the picture Wally described. Gray throws a four-seam and a sinker off a
near-identical release axis; the sinker's extra arm-side run is seam-shifted
wake, not a different spin orientation. A model that ignores the release axis
(the shipped one) buries that; a model that uses it names it.

Usage:  python3 scripts/research/xmove/xmove_case_study.py [ "Last, First" ... ]
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, run_linear, FORMS, UNIT_MIN

DEFAULT = ['Gray, Sonny', 'Ohtani, Shohei', 'Bassitt, Chris', 'Kirby, George',
           'Cease, Dylan', 'Webb, Logan']
SEASON = 2025


def table(A, name, season, xi_ship, xh_ship, xi_new, xh_new):
    m = (A['pitcher'] == name) & (A['season'] == season)
    if m.sum() == 0:
        return
    print(f'\n{name}  {season}')
    print(f"  {'pt':>3} {'n':>5} {'velo':>5} {'spin':>6} {'RTilt':>7} {'OTilt':>7} "
          f"{'dev':>6} | {'IVB':>5} {'HB':>5} | {'xIVB':>6} {'xHB':>6} {'IVBOE':>6} {'HBOE':>6}"
          f" | {'xIVB':>6} {'xHB':>6} {'IVBOE':>6} {'HBOE':>6} | {'SSW in':>6}")
    print(f"  {'':>3} {'':>5} {'':>5} {'':>6} {'':>7} {'':>7} {'':>6} | {'':>5} {'':>5} "
          f"| {'------ shipped -------------':>27} | {'------ release-axis --------':>27} |")
    for pt in ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'SV']:
        k = m & (A['pt'] == pt) & np.isfinite(xi_new)
        if k.sum() < UNIT_MIN:
            continue
        # circular mean of the release axis (already hand-mirrored in load_np)
        rtilt = np.degrees(np.arctan2(A['ax_sin'][k].mean(), A['ax_cos'][k].mean())) % 360
        ot = np.degrees(np.arctan2(A['hb_s'][k].mean(), A['ivb'][k].mean())) % 360
        dev = (ot - rtilt + 180) % 360 - 180
        print(f'  {pt:>3} {k.sum():>5} {A["velo"][k].mean():>5.1f} {A["spin"][k].mean():>6.0f} '
              f'{rtilt:>7.1f} {ot:>7.1f} {dev:>6.1f} | '
              f'{A["ivb"][k].mean():>5.1f} {A["hb_s"][k].mean()*1:>5.1f} | '
              f'{xi_ship[k].mean():>6.1f} {xh_ship[k].mean():>6.1f} '
              f'{(A["ivb"][k]-xi_ship[k]).mean():>6.1f} {(A["hb_s"][k]-xh_ship[k]).mean():>6.1f} | '
              f'{xi_new[k].mean():>6.1f} {xh_new[k].mean():>6.1f} '
              f'{(A["ivb"][k]-xi_new[k]).mean():>6.1f} {(A["hb_s"][k]-xh_new[k]).mean():>6.1f} | '
              f'{A["cross"][k].mean():>6.1f}')
    print('  RTilt/OTilt in clock-degrees (0 = 12:00, clockwise +). HB, dev, SSW '
          'hand-signed: + = arm side.')
    print('  SSW in = observed break perpendicular to the measured release axis, inches.')


if __name__ == '__main__':
    names = sys.argv[1:] or DEFAULT
    A = load_np()
    xi_s, xh_s = run_linear(A, FORMS['S1 shipped aa,ext,v'])
    xi_n, xh_n = run_linear(A, FORMS['S3b +spin x axis'])
    for n in names:
        table(A, n, SEASON, xi_s, xh_s, xi_n, xh_n)
