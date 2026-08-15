#!/usr/bin/env python3
"""augment_kinematics.py — attach KinEff/KinDev/KinCd to the 2021-2025
training pickles from the local full-season Statcast caches (no network).

2021-24: the pickles were built positionally from _statcast{year}_cache.pkl
(one dict per cache row, no filtering — build_historical_training_set), so
kinematics computed on the cache inject by position after verifying sampled
row identity (velocity + pitcher + date), the augment_priors_spinaxis
pattern.

2025: the pickle came from the sheets-statcast fingerprint join, so the same
strict game_pk-constrained fingerprint re-join used by augment_2025_spinaxis
carries the three fields (velo +/-0.25, plate-coord distance <= 0.5).

Idempotent: re-running overwrites the same keys.

Usage: python3 scripts/builders/augment_kinematics.py
"""
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np


def atomic_dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ci'))  # kinematics_lib moved in 2026-08 reorg
from kinematics_lib import compute_kinematics

KEYS = ('KinEff', 'KinDev', 'KinCd')


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kin_of_cache(year):
    df = pickle.load(open(
        os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    kin = compute_kinematics(df)
    print(f'  {year}: cache {len(df)} rows, kinematics on '
          f'{kin.kin_eff.notna().mean() * 100:.1f}%')
    return df, kin


def positional(year):
    path = os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl')
    pitches = pickle.load(open(path, 'rb'))
    df, kin = kin_of_cache(year)
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
    e = kin['kin_eff'].values
    d = kin['kin_dev'].values
    c = kin['kin_cd'].values
    n = 0
    for i, p in enumerate(pitches):
        if np.isfinite(e[i]):
            p['KinEff'], p['KinDev'], p['KinCd'] = \
                float(e[i]), float(d[i]), float(c[i])
            n += 1
        else:
            p['KinEff'] = p['KinDev'] = p['KinCd'] = None
    atomic_dump(pitches, path)
    print(f'  {year}: injected positionally on {n}/{len(pitches)} rows '
          f'({n / len(pitches) * 100:.1f}%) — saved')


def fingerprint_2025():
    path = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
    pitches = pickle.load(open(path, 'rb'))
    df, kin = kin_of_cache(2025)
    if 'game_type' in df.columns:
        keep = (df['game_type'] == 'R').values
        df = df[keep].reset_index(drop=True)
        kin = kin[keep].reset_index(drop=True)
    pub = defaultdict(list)
    cols = dict(gd=df['game_date'].astype(str).str[:10].values,
                nm=df['player_name'].values,
                gpk=df['game_pk'].values,
                v=df['release_speed'].values,
                px=df['plate_x'].values, pz=df['plate_z'].values,
                e=kin['kin_eff'].values, d=kin['kin_dev'].values,
                c=kin['kin_cd'].values)
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
                for k in KEYS:
                    p.setdefault(k, None)
                unmatched += 1
                continue
            used[best_i] = True
            i = cand[best_i]
            if np.isfinite(cols['e'][i]):
                p['KinEff'] = float(cols['e'][i])
                p['KinDev'] = float(cols['d'][i])
                p['KinCd'] = float(cols['c'][i])
            else:
                p['KinEff'] = p['KinDev'] = p['KinCd'] = None
            matched += 1
    n = sum(1 for p in pitches if p.get('KinEff') is not None)
    atomic_dump(pitches, path)
    print(f'  2025: re-join {matched} matched / {unmatched} unmatched '
          f'({matched / max(matched + unmatched, 1):.1%}); KinEff on '
          f'{n}/{len(pitches)} rows ({n / len(pitches) * 100:.1f}%) — saved')


if __name__ == '__main__':
    for yr in (2021, 2022, 2023, 2024):
        positional(yr)
    fingerprint_2025()
