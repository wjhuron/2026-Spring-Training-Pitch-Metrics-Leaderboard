#!/usr/bin/env python3
"""kinematics_lib.py — per-pitch aerodynamic decomposition from the public
9-parameter fit (vx0/vy0/vz0 + ax/ay/az at y=50).

Validated 2026-08-09 (scripts/research/stuff/stuff_kinematics_probe.py, 66k 2025 pitches):
league mean Cd 0.337 (textbook ~0.33); unit-level spin-efficiency estimate
correlates 0.926 with Savant's published active spin; per-pitch axis
deviation league means land on the xmove review's OTilt-RTilt anchors
(FF -7 vs -9, SI +24 vs +18, FC -29 vs -24).

Outputs per pitch:
  kin_cd    drag coefficient (deceleration along flight / kappa v^2)
  kin_eff   transverse-spin fraction: lift -> CL -> spin factor S via
            Nathan's CL = 1/(2.32 + 0.4/S), omega_T = S*v/R, divided by the
            measured total spin rate. UNCALIBRATED scale (runs ~0.7-0.8x
            Savant's active spin level); rank order is what validates, and a
            monotone map to Savant scale can be fit downstream if a
            calibrated display value is ever needed.
  kin_dev   hand-signed per-pitch deviation (degrees) of the measured LIFT
            direction from the measured release spin axis — the per-pitch,
            force-based SSW measure (positive = arm side).

Air density is the sea-level standard: venue density variation (~8% CL at
Coors) is deliberately left in for now; the venue-density work can refine
this when a kinematics feature ships.
"""
import math

import numpy as np
import pandas as pd

RHO = 0.0740                       # lb/ft^3, sea-level standard
R_BALL = 0.121                     # ft
MASS = 5.125 / 16.0                # lb
G = 32.174                         # ft/s^2
KAPPA = RHO * math.pi * R_BALL ** 2 / (2.0 * MASS)   # 1/ft
Y0, YF = 50.0, 17.0 / 12.0


def compute_kinematics(df):
    """df needs vx0, vy0, vz0, ax, ay, az, release_spin_rate, spin_axis,
    p_throws. Returns a DataFrame (same index) with kin_cd, kin_eff, kin_dev
    (NaN where inputs are missing/degenerate)."""
    def num(cols):
        return np.column_stack([
            pd.to_numeric(df[c], errors='coerce').astype(float).values
            for c in cols])

    v0 = num(['vx0', 'vy0', 'vz0'])
    a = num(['ax', 'ay', 'az'])
    spin = num(['release_spin_rate'])[:, 0]
    sa = num(['spin_axis'])[:, 0]
    thr = df['p_throws'].values

    n_all = len(df)
    ok = np.isfinite(v0).all(1) & np.isfinite(a).all(1) & np.isfinite(spin) \
        & np.isfinite(sa) & (spin > 500)
    disc = np.where(ok, v0[:, 1] ** 2 - 2.0 * a[:, 1] * (Y0 - YF), np.nan)
    ok &= disc > 0

    cd_full = np.full(n_all, np.nan)
    eff_full = np.full(n_all, np.nan)
    dev_full = np.full(n_all, np.nan)
    if not ok.any():
        return pd.DataFrame({'kin_cd': cd_full, 'kin_eff': eff_full,
                             'kin_dev': dev_full})
    v0, a, spin, sa, thr = v0[ok], a[ok], spin[ok], sa[ok], thr[ok]
    disc = disc[ok]

    a_aero = a + np.array([0.0, 0.0, G])
    t_f = (-np.sqrt(disc) - v0[:, 1]) / a[:, 1]
    v_mid = v0 + a * (t_f[:, None] / 2.0)
    speed = np.linalg.norm(v_mid, axis=1)
    vhat = v_mid / speed[:, None]

    a_par = (a_aero * vhat).sum(1)
    a_perp = a_aero - a_par[:, None] * vhat
    lift = np.linalg.norm(a_perp, axis=1)

    cd = -a_par / (KAPPA * speed ** 2)
    cl = lift / (KAPPA * speed ** 2)
    inv = 1.0 / np.clip(cl, 1e-6, None) - 2.32
    S = np.where(inv > 1e-6, 0.4 / inv, np.nan)
    omega_t = S * speed / R_BALL * 60.0 / (2.0 * math.pi)
    eff = np.clip(omega_t / spin, 0.0, 1.3)

    # lift direction in the catcher plane, mapped to the spin_axis clock
    # (x negated: the tilt convention runs mirrored vs the raw +x frame)
    ax_move = (np.degrees(np.arctan2(-a_perp[:, 0], a_perp[:, 2]))
               + 180.0) % 360.0
    dev = (ax_move - sa + 540.0) % 360.0 - 180.0
    s = np.where(thr == 'R', 1.0, -1.0)

    cd_full[ok] = np.round(cd, 4)
    eff_full[ok] = np.round(eff, 4)
    dev_full[ok] = np.round(dev * s, 2)
    return pd.DataFrame({'kin_cd': cd_full, 'kin_eff': eff_full,
                         'kin_dev': dev_full})
