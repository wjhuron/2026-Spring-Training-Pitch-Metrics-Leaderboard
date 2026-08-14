#!/usr/bin/env python3
"""augment_haa.py — reconstruct HAA into the 2021-2025 training pickles.

HAA was never built in the historical training set (the 2021-24 rebuild
computed VAA only; the 2025 sheets join carried VAA only), which left nHAA
unmeasurable in the LOSO replicates. The public Statcast caches hold the
same kinematics VAA came from (vx0/vy0/ax/ay), so HAA reconstructs exactly
like production computes it from the feed (Pitcher2026.calculate_approach_
angles, FanGraphs primer formula):

    vy_f = -sqrt(vy0^2 - 2*ay*(50 - 17/12))
    t    = (vy_f - vy0) / ay
    vx_f = vx0 + ax*t
    HAA  = -atan(vx_f / vy_f) * 180/pi

Join strategy mirrors scripts/augment_kinematics.py: 2021-2024 pickles are
POSITIONAL descendants of data/_statcast{year}_cache.pkl (asserted with
500-row velocity/name/date spot checks); 2025 uses the game_pk-constrained
fingerprint re-join. Idempotent: re-running overwrites the same key.

Validated against 2026 before injection (scripts caller): recomputing HAA
from a Savant pull of a 2026 game reproduces the sheet's HAA column.

Usage: python3 scripts/augment_haa.py
"""
import math
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.augment_kinematics import atomic_dump, sf   # noqa: E402


def haa_of(vx0, vy0, ax, ay):
    if any(v is None for v in (vx0, vy0, ax, ay)) or not ay:
        return None
    disc = vy0 ** 2 - 2 * ay * (50 - 17 / 12)
    if disc < 0:
        return None
    vy_f = -math.sqrt(disc)
    t = (vy_f - vy0) / ay
    vx_f = vx0 + ax * t
    return -math.degrees(math.atan(vx_f / vy_f))


def haa_of_cache(year):
    df = pickle.load(open(
        os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    vx0 = pd.to_numeric(df['vx0'], errors='coerce').astype('float64').values
    vy0 = pd.to_numeric(df['vy0'], errors='coerce').astype('float64').values
    ax = pd.to_numeric(df['ax'], errors='coerce').astype('float64').values
    ay = pd.to_numeric(df['ay'], errors='coerce').astype('float64').values
    haa = np.full(len(df), np.nan)
    ok = np.isfinite(vx0) & np.isfinite(vy0) & np.isfinite(ax) \
        & np.isfinite(ay) & (ay != 0)
    disc = np.where(ok, vy0 ** 2 - 2 * ay * (50 - 17 / 12), np.nan)
    ok &= disc > 0
    vy_f = -np.sqrt(np.where(ok, disc, np.nan))
    t = (vy_f - vy0) / np.where(ay == 0, np.nan, ay)
    vx_f = vx0 + ax * t
    haa[ok] = (-np.degrees(np.arctan(vx_f / vy_f)))[ok]
    print(f'  {year}: cache {len(df)} rows, HAA on '
          f'{np.isfinite(haa).mean() * 100:.1f}%')
    return df, haa


def positional(year):
    path = os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl')
    pitches = pickle.load(open(path, 'rb'))
    df, haa = haa_of_cache(year)
    assert len(pitches) == len(df), \
        f'{year}: pickle {len(pitches)} vs cache {len(df)} — not positional'
    rnd = random.Random(11)
    vel = df['release_speed'].values
    nam = df['player_name'].values
    dat = df['game_date'].values
    for i in rnd.sample(range(len(pitches)), 500):
        pv, cv = pitches[i].get('Velocity'), sf(vel[i])
        assert (pv is None and cv is None) or abs((pv or 0) - (cv or 0)) < 1e-6, \
            f'{year} row {i}: velocity mismatch {pv!r} vs {cv!r}'
        assert pitches[i].get('Pitcher') == nam[i], f'{year} row {i}: name'
        assert str(pitches[i].get('Game Date'))[:10] == str(dat[i])[:10], \
            f'{year} row {i}: date'
    n = 0
    for i, p in enumerate(pitches):
        if np.isfinite(haa[i]):
            p['HAA'] = round(float(haa[i]), 2)
            n += 1
        else:
            p['HAA'] = None
    atomic_dump(pitches, path)
    print(f'  {year}: injected positionally on {n}/{len(pitches)} rows '
          f'({n / len(pitches) * 100:.1f}%) — saved')


def fingerprint_2025():
    path = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
    pitches = pickle.load(open(path, 'rb'))
    df, haa = haa_of_cache(2025)
    if 'game_type' in df.columns:
        keep = (df['game_type'] == 'R').values
        df = df[keep].reset_index(drop=True)
        haa = haa[keep]
    pub = defaultdict(list)
    cols = dict(gd=df['game_date'].astype(str).str[:10].values,
                nm=df['player_name'].values,
                gpk=df['game_pk'].values,
                v=df['release_speed'].values,
                px=df['plate_x'].values, pz=df['plate_z'].values)
    for i in range(len(df)):
        pub[(cols['gd'][i], cols['nm'][i])].append(i)

    groups = defaultdict(list)
    for p in pitches:
        groups[(str(p.get('Game Date'))[:10], p.get('Pitcher'))].append(p)

    matched = unmatched = 0
    for key, mine in groups.items():
        cand = pub.get(key, [])
        used = [False] * len(cand)
        for p in mine:
            v, px, pz = p.get('Velocity'), p.get('PlateX'), p.get('PlateZ')
            gpk = p.get('_game_pk')
            best_i, best_d = None, 1e9
            if v is not None:
                for j, i in enumerate(cand):
                    if used[j]:
                        continue
                    if (gpk is not None and sf(cols['gpk'][i]) is not None
                            and int(cols['gpk'][i]) != gpk):
                        continue
                    cv = sf(cols['v'][i])
                    if cv is None or abs(cv - v) > 0.25:
                        continue
                    dist = abs(cv - v) * 2.0
                    cx, cz = sf(cols['px'][i]), sf(cols['pz'][i])
                    if px is not None and cx is not None:
                        dist += abs(cx - px)
                    if pz is not None and cz is not None:
                        dist += abs(cz - pz)
                    if dist < best_d:
                        best_d, best_i = dist, j
            if best_i is None or best_d > 0.5:
                p.setdefault('HAA', None)
                unmatched += 1
                continue
            used[best_i] = True
            i = cand[best_i]
            p['HAA'] = (round(float(haa[i]), 2)
                        if np.isfinite(haa[i]) else None)
            matched += 1
    n = sum(1 for p in pitches if p.get('HAA') is not None)
    atomic_dump(pitches, path)
    print(f'  2025: re-join {matched} matched / {unmatched} unmatched '
          f'({matched / max(matched + unmatched, 1):.1%}); HAA on '
          f'{n}/{len(pitches)} rows ({n / len(pitches) * 100:.1f}%) — saved')


if __name__ == '__main__':
    for y in (2021, 2022, 2023, 2024):
        positional(y)
    fingerprint_2025()
    print('HAA augmentation complete.')
