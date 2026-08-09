#!/usr/bin/env python3
"""stuff_kinematics_probe.py — can per-pitch spin efficiency be measured from
the public 9-parameter kinematics? (Feasibility gate for the Stuff+ 3b
candidate: per-pitch active spin + per-pitch SSW from ax/ay/az.)

Physics: the 9P fit gives velocity and constant acceleration at y=50.
Subtract gravity; split the remaining aerodynamic acceleration into the
component along the (mid-flight) velocity (drag) and perpendicular to it
(lift = Magnus + seam). Lift magnitude -> lift coefficient CL -> spin factor
S via Nathan's parameterization CL = 1/(2.32 + 0.4/S) -> transverse spin
omega_T = S*v/R. Efficiency = omega_T / measured total spin rate.

Validation (2025, three June weeks, ~60k pitches):
  1. unit-level (pitcher x throws x pitch type, n>=50) corr of estimated
     efficiency vs Savant's published 2025 active spin
     (data/active_spin_prior.json, via='direct' entries)
  2. league mean efficiency by pitch type vs asp leagueMeanByPitchType
  3. per-pitch movement-axis-vs-spin-axis deviation distribution vs the
     xmove review's OTilt-RTilt table (FF ~ -9, SI ~ +18, ST ~ +31)
  4. drag coefficient sanity (league mean Cd ~ 0.33)

Air density is held at the sea-level standard here (feasibility only; the
production version would use the venue density work). Coors-sized density
error is ~8% on CL, ~5-6% on efficiency — visible in per-park splits but not
fatal to a pooled rank validation.

Usage: python3 scripts/stuff_kinematics_probe.py [--refetch]
"""
import argparse
import json
import math
import os
import subprocess
import sys
from io import StringIO

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'data', '_kinematics_probe_2025.csv')
ASP = os.path.join(ROOT, 'data', 'active_spin_prior.json')

CHUNKS = [('2025-06-02', '2025-06-04'), ('2025-06-05', '2025-06-07'),
          ('2025-06-09', '2025-06-11'), ('2025-06-12', '2025-06-14'),
          ('2025-06-16', '2025-06-18'), ('2025-06-19', '2025-06-21')]

URL = ('https://baseballsavant.mlb.com/statcast_search/csv?all=true'
       '&hfSea=2025%7C&player_type=pitcher&type=details'
       '&game_date_gt={gt}&game_date_lt={lt}')

# ball constants (imperial): rho lb/ft^3, radius ft, mass oz->lb, g ft/s^2
RHO = 0.0740
R_BALL = 0.121
MASS = 5.125 / 16.0
G = 32.174
KAPPA = RHO * math.pi * R_BALL ** 2 / (2.0 * MASS)   # 1/ft


def fetch():
    frames = []
    for gt, lt in CHUNKS:
        cmd = ['curl', '-s', '--fail', '-A', 'Mozilla/5.0',
               URL.format(gt=gt, lt=lt)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            print(f'  fetch {gt}..{lt} FAILED (curl {out.returncode})')
            continue
        d = pd.read_csv(StringIO(out.stdout), low_memory=False)
        print(f'  {gt}..{lt}: {len(d)} pitches')
        if len(d) >= 25000:
            print('  *** chunk at the 25k row cap — narrow the window ***')
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(CACHE, index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    args = ap.parse_args()
    if args.refetch or not os.path.exists(CACHE):
        print('fetching 2025 sample from Savant ...')
        d = fetch()
    else:
        d = pd.read_csv(CACHE, low_memory=False)
    cols = ['player_name', 'p_throws', 'pitch_type', 'release_spin_rate',
            'spin_axis', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
            'release_speed', 'home_team']
    d = d.dropna(subset=[c for c in cols if c != 'home_team'])
    d = d[d.pitch_type.isin(['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS'])]
    d = d[d.release_spin_rate > 500]
    print(f'{len(d)} usable pitches')

    v0 = d[['vx0', 'vy0', 'vz0']].values
    a = d[['ax', 'ay', 'az']].values
    a_aero = a + np.array([0.0, 0.0, G])          # remove gravity
    # mid-flight velocity: y=50 -> plate front (y=17/12)
    y0, yf = 50.0, 17.0 / 12.0
    vy0 = v0[:, 1]
    disc = vy0 ** 2 - 2.0 * a[:, 1] * (y0 - yf)
    ok = disc > 0
    d, v0, a, a_aero, disc = d[ok], v0[ok], a[ok], a_aero[ok], disc[ok]
    t_f = (-np.sqrt(disc) - v0[:, 1]) / a[:, 1]
    v_mid = v0 + a * (t_f[:, None] / 2.0)
    speed = np.linalg.norm(v_mid, axis=1)
    vhat = v_mid / speed[:, None]

    a_par = (a_aero * vhat).sum(1)                # signed, along flight
    drag = -a_par                                 # positive = decelerating
    a_perp_vec = a_aero - a_par[:, None] * vhat
    lift = np.linalg.norm(a_perp_vec, axis=1)

    cd = drag / (KAPPA * speed ** 2)
    cl = lift / (KAPPA * speed ** 2)
    # invert CL = 1/(2.32 + 0.4/S)  ->  S = 0.4 / (1/CL - 2.32)
    inv = 1.0 / np.clip(cl, 1e-6, None) - 2.32
    S = np.where(inv > 1e-6, 0.4 / inv, np.nan)
    omega_t = S * speed / R_BALL * 60.0 / (2.0 * math.pi)   # rpm
    eff = omega_t / d.release_spin_rate.values
    d = d.assign(cd=cd, cl=cl, eff=np.clip(eff, 0, 1.3),
                 eff_raw=eff, lift=lift)

    # per-pitch movement-axis deviation: lift direction in the (z, x) plane
    # perpendicular to flight vs the measured release spin axis
    # (both mapped to Savant's spin_axis convention: 180 = pure backspin)
    # x negated: the tilt/spin_axis clock runs mirrored vs the raw +x frame
    # (same flip the HAA regression measured on breakHorizontal)
    ax_move = (np.degrees(np.arctan2(-a_perp_vec[:, 0], a_perp_vec[:, 2]))
               + 180.0) % 360.0
    dev = (ax_move - d.spin_axis.values + 540.0) % 360.0 - 180.0
    s = np.where(d.p_throws.values == 'R', 1.0, -1.0)
    d = d.assign(axis_dev_pp=dev * s)

    print(f'\nleague mean Cd {d.cd.mean():.3f} (expect ~0.33), '
          f'CL {d.cl.mean():.3f}')

    asp = json.load(open(ASP))
    lg_asp = asp['leagueMeanByPitchType']
    # entries keyed by Wally tag; entry["savant"] is the savant type to join on
    ent = {}
    for k, v in asp['entries'].items():
        name, thr, _pt = k.split('|')
        ent[(name, thr, v.get('savant'))] = v['active']

    print(f'\n{"pt":>4} {"n_units":>8} {"eff_est":>8} {"asp_lg":>7} '
          f'{"corr":>6} {"axis_dev":>9} {"|dev|":>6}')
    all_x, all_y = [], []
    g = d.groupby(['player_name', 'p_throws', 'pitch_type'])
    units = g.agg(eff=('eff', 'mean'), n=('eff', 'size'),
                  dev=('axis_dev_pp', 'mean')).reset_index()
    units = units[units.n >= 50]
    for pt in ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS'):
        sub = units[units.pitch_type == pt]
        xs, ys = [], []
        for r in sub.itertuples():
            v = ent.get((r.player_name, r.p_throws, pt))
            if v is not None:
                xs.append(r.eff * 100.0); ys.append(v)
        c = (float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 15
             else float('nan'))
        all_x += xs; all_y += ys
        dd = d[d.pitch_type == pt]
        print(f'{pt:>4} {len(sub):>8} {dd.eff.mean()*100:>7.1f}% '
              f'{lg_asp.get(pt, float("nan")):>6.1f}% {c:>6.3f} '
              f'{dd.axis_dev_pp.mean():>8.1f} {dd.axis_dev_pp.abs().mean():>6.1f}')
    pooled = float(np.corrcoef(all_x, all_y)[0, 1])
    # pooled-within: remove per-type means so the corr is not carried by
    # between-type structure
    ax2, ay2 = np.array(all_x), np.array(all_y)
    print(f'\npooled unit corr(eff_est, savant active spin): {pooled:.3f} '
          f'(n={len(all_x)} joined units)')
    print('xmove-review anchors for axis_dev: FF ~ -9, SI ~ +18, FC ~ -24, '
          'ST ~ +31, CH ~ +10')


if __name__ == '__main__':
    main()
