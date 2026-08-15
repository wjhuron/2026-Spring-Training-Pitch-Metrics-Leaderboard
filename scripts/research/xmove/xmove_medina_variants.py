#!/usr/bin/env python3
"""PROTOTYPE — Medina's movement plot under design options 2 and 3.

Option 1 (pitch-type agnostic, empirical) is scripts/research/xmove/xmove_medina_plot.py. The
two alternatives it raised:

  OPTION 2 — pitch-class seam term.
    Fit per (pitch type, hand), so each class's TYPICAL seam deflection lands
    in that group's intercept for free. What survives into the residual is the
    pitcher's seam behaviour on top of what a sinker normally gets, which is
    the version that belongs on a leaderboard: under option 1 every sinker in
    baseball lights up, because the residual is dominated by what sinkers do.

  OPTION 3 — physics baseline.
    Direction comes from the MEASURED release axis, not from a fit. Magnitude
    comes from transverse (Magnus-effective) spin:

        S_t   = spin rate x active spin fraction
        along = M(S_t, velocity)          Magnus-direction break
        cross = 0                          a pure Magnus ball has none

    so xIVB = M cos(theta), xHB = M sin(theta). The only fitted part is the
    scalar M, which is the lift curve, not a per-class offset. Everything that
    is not Magnus lands in the residual, which is the seam-shifted wake.

    M is fit on 2025 (the season the active-spin prior is built from) pooled by
    hand, with no pitch type in the fit.

  COVERAGE WARNING, and it is not incidental: option 3 needs a PRIOR-season
  active spin per pitcher and pitch. Medina missed 2025, so he has no entry and
  falls back to leagueMeanByPitchType. His physics plot is therefore the
  degenerate case — league-average efficiency by class — which is exactly where
  option 3 is weakest. Any returning-from-injury arm or rookie has this problem.

Usage: XMOVE_DIR=<scratch> python3 scripts/research/xmove/xmove_medina_variants.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_compare import load_np, _design  # noqa: E402
from xmove_agnostic_basis import add_harmonics, form  # noqa: E402
from xmove_medina_plot import medina_2026, build_feats, draw_plot  # noqa: E402

FEATS = form(3, True, True)
OUT2 = os.path.expanduser('~/Downloads/Medina_expected_movement_opt2_seamterm.png')
OUT3 = os.path.expanduser('~/Downloads/Medina_expected_movement_opt3_physics.png')
# Magnus magnitude basis: lift saturates in spin factor, so a quadratic in S_t
# with a velocity interaction is enough curvature for a prototype.
M_FEATS = ['st_spin', 'st_spin2', 'velo', 'st_v']


def league_oe_sd(pitcher, pt, season, oe_i, oe_h, min_n=50):
    u = pd.DataFrame({'p': pitcher, 'pt': pt, 's': season,
                      'i': oe_i, 'h': oe_h}).groupby(['p', 'pt', 's']).agg(
        n=('i', 'size'), i=('i', 'mean'), h=('h', 'mean')).reset_index()
    u = u[u.n >= min_n]
    return u.i.std(), u.h.std()


def option2(A, d26, hand, sign):
    """Per (pitch type, hand) fit — the seam term lands in the intercept."""
    xi = np.full(len(d26), np.nan)
    xh = np.full(len(d26), np.nan)
    oe_i_all, oe_h_all, pit_all, pt_all, se_all = [], [], [], [], []
    F = build_feats(d26['Velocity'].values, d26['Spin Rate'].values,
                    d26['SpinAxis'].values, d26['Extension'].values,
                    d26['ArmAngle'].values, sign)
    for pt in d26['Pitch Type'].unique():
        tr = np.where((A['thr'] == hand) & (A['pt'] == pt))[0]
        if len(tr) < 2000:
            continue
        Xt = _design(A, FEATS, tr)
        b_i = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
        b_h = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]
        oe_i_all.append(A['ivb'][tr] - Xt @ b_i)
        oe_h_all.append(A['hb_s'][tr] - Xt @ b_h)
        pit_all.append(A['pitcher'][tr]); pt_all.append(A['pt'][tr])
        se_all.append(A['season'][tr])
        m = (d26['Pitch Type'] == pt).values
        Xs = _design(F, FEATS, np.where(m)[0])
        xi[m], xh[m] = Xs @ b_i, Xs @ b_h
    sd_i, sd_h = league_oe_sd(np.concatenate(pit_all), np.concatenate(pt_all),
                              np.concatenate(se_all), np.concatenate(oe_i_all),
                              np.concatenate(oe_h_all))
    return xi, xh * sign, sd_i, sd_h


def _mag_feats(spin_t, velo):
    return {'st_spin': spin_t, 'st_spin2': spin_t ** 2,
            'velo': velo, 'st_v': spin_t * velo}


def option3(A, d26, hand, sign):
    """Physics: measured axis sets direction, transverse spin sets magnitude."""
    pri = json.load(open(os.path.join(ROOT, 'data/active_spin_prior.json')))
    ent, lg = pri['entries'], pri['leagueMeanByPitchType']

    def active_for(pitcher, thr, pt):
        e = ent.get(f'{pitcher}|{thr}|{pt}')
        return (e['active'] if e else lg.get(pt, 80.0)) / 100.0

    # Fit M on 2025 pooled by hand. No pitch type in the design.
    tr = np.where((A['thr'] == hand) & (A['season'] == 2025))[0]
    act = np.array([active_for(p, hand, t)
                    for p, t in zip(A['pitcher'][tr], A['pt'][tr])])
    S_t = A['spin'][tr] * act
    Ftr = _mag_feats(S_t, A['velo'][tr])
    Xt = _design(Ftr, M_FEATS, np.arange(len(tr)))
    b_m = np.linalg.lstsq(Xt, A['along'][tr], rcond=None)[0]
    print(f'  option3: M fit on {len(tr):,} pitches ({hand}HP, 2025)', file=sys.stderr)

    # League residual spread under the physics model.
    M_tr = Xt @ b_m
    oe_i = A['ivb'][tr] - (M_tr * A['ct'][tr])
    oe_h = A['hb_s'][tr] - (M_tr * A['st'][tr])
    sd_i, sd_h = league_oe_sd(A['pitcher'][tr], A['pt'][tr], A['season'][tr],
                              oe_i, oe_h)

    # Score Medina.
    act26 = np.array([active_for('Medina, Luis', hand, t)
                      for t in d26['Pitch Type'].values])
    n_direct = sum(1 for t in d26['Pitch Type'].unique()
                   if f'Medina, Luis|{hand}|{t}' in ent)
    print(f'  option3: pitcher-specific active spin for {n_direct} of '
          f'{d26["Pitch Type"].nunique()} pitch types (rest = league mean)',
          file=sys.stderr)
    S26 = d26['Spin Rate'].values * act26
    F26 = _mag_feats(S26, d26['Velocity'].values)
    M26 = _design(F26, M_FEATS, np.arange(len(d26))) @ b_m
    th = np.radians(((d26['SpinAxis'].values - 180.0) % 360.0) * sign)
    return M26 * np.cos(th), (M26 * np.sin(th)) * sign, sd_i, sd_h


def main():
    print('Loading 2021-2025...', file=sys.stderr)
    A = add_harmonics(load_np())
    d26 = medina_2026()
    hand = str(d26['Throws'].iloc[0])
    sign = 1.0 if hand == 'R' else -1.0
    d26 = d26.assign(ivb=d26['xIndVrtBrk'], hb=d26['xHorzBrk'])

    xi, xh, sd_i, sd_h = option2(A, d26, hand, sign)
    draw_plot(d26.assign(xivb=xi, xhb=xh), hand, sd_i, sd_h,
              'OPTION 2 — expected per pitch class (seam term in the intercept)',
              'expected = what a pitch of THIS CLASS, released this way, normally does.\n'
              'arrow = his seam behaviour on top of the class norm. ellipse = +/-1 SD of\n'
              'league pitcher-level deviation under this model.', OUT2)

    xi3, xh3, sd3i, sd3h = option3(A, d26, hand, sign)
    draw_plot(d26.assign(xivb=xi3, xhb=xh3), hand, sd3i, sd3h,
              'OPTION 3 — Magnus physics baseline (measured axis + transverse spin)',
              'expected = pure Magnus: direction from the MEASURED release axis, magnitude\n'
              'from spin rate x active spin. arrow = everything non-Magnus, i.e. seam-shifted\n'
              'wake. NOTE: Medina missed 2025, so active spin falls back to the league mean\n'
              'for his class — this is the degenerate case for option 3.', OUT3)


if __name__ == '__main__':
    main()
