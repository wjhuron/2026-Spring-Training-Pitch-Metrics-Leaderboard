#!/usr/bin/env python3
"""build_kinematics_2026.py — season-to-date per-pitch kinematics sidecar.

Fetches 2026 regular-season Statcast (chunked 3-day windows, well under the
25k row cap), computes KinEff/KinDev/KinCd (scripts/kinematics_lib.py), and
joins them to the 2026 pitch cache by (game_pk, at_bat_number) + velocity /
plate-coordinate fingerprint. Savant pitch_number counts auto balls and the
sheets renumber real pitches (see the numbering memory), so pitch_number is
deliberately NOT a join key — the fingerprint inside the PA is.

Outputs:
  data/_statcast2026_kin_cache.pkl   raw fetched columns (incremental: only
                                     missing dates are fetched on re-run)
  data/kinematics_2026_sidecar.pkl   {PitchID: (KinEff, KinDev, KinCd)}

Usage: python3 scripts/build_kinematics_2026.py
"""
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from kinematics_lib import compute_kinematics

RAW = os.path.join(ROOT, 'data', '_statcast2026_kin_cache.pkl')
SIDE = os.path.join(ROOT, 'data', 'kinematics_2026_sidecar.pkl')
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')

URL = ('https://baseballsavant.mlb.com/statcast_search/csv?all=true'
       '&hfSea=2026%7C&hfGT=R%7C&player_type=pitcher&type=details'
       '&game_date_gt={gt}&game_date_lt={lt}')

# ROC/AAA (v13 parity, 2026-08-14): the minors Statcast Search serves the
# full 9P kinematics on real ROC games (vx0..az/spin_axis 100%, verified
# 918 pitches 2026-08-06..12), so ROC four-seamers get MEASURED kin_eff
# instead of the league-constant imputation. Both player_types per the
# supplement's quirk 1: pitcher-side = ROC pitchers' PAs, batter-side = the
# opponent pitchers facing ROC bats (the AAA-source rows); the two PA sets
# are disjoint so the concat cannot duplicate. team must be the NUMERIC
# club id and minors=true is load-bearing (without it the CSV is empty).
RAW_MILB = os.path.join(ROOT, 'data', '_statcast2026_roc_kin_cache.pkl')
MILB_URL = ('https://baseballsavant.mlb.com/statcast-search-minors/csv'
            '?all=true&type=details&team=534&player_type={ptype}'
            '&min_pitches=0&min_results=0&sort_col=pitches'
            '&sort_order=desc&minors=true'
            '&game_date_gt={gt}&game_date_lt={lt}')
COLS = ['game_pk', 'at_bat_number', 'pitch_number', 'game_date',
        'player_name', 'p_throws', 'pitch_type', 'release_speed',
        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
        'release_spin_rate', 'spin_axis']


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def daterange_windows(d0, d1, step=3):
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=step - 1), d1)
        yield cur.isoformat(), end.isoformat()
        cur = end + timedelta(days=1)


def fetch_missing(pitch_dates, urls=(URL,), raw_path=RAW, step=3):
    """Incrementally fetch the given date set for each URL template into
    raw_path. urls is a tuple so the minors pull can hit both player_type
    sides for the same window (disjoint PA sets, concat-safe)."""
    old = pd.read_pickle(raw_path) if os.path.exists(raw_path) else None
    have = (set(old['game_date'].astype(str).str[:10].unique())
            if old is not None else set())
    todo = sorted(d for d in pitch_dates if d not in have)
    if not todo:
        print(f'raw cache current ({0 if old is None else len(old)} rows)')
        return old
    d0 = date.fromisoformat(todo[0])
    d1 = date.fromisoformat(todo[-1])
    frames = [] if old is None else [old]
    for gt, lt in daterange_windows(d0, d1, step):
        got = []
        for u in urls:
            out = subprocess.run(['curl', '-s', '--fail', '-A', 'Mozilla/5.0',
                                  u.format(gt=gt, lt=lt)],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                print(f'  {gt}..{lt}: FETCH FAILED (curl {out.returncode})')
                continue
            try:
                d = pd.read_csv(StringIO(out.stdout), low_memory=False)
            except Exception as e:
                print(f'  {gt}..{lt}: parse failed ({e})')
                continue
            if len(d) >= 25000:
                print(f'  {gt}..{lt}: *** at 25k cap — data lost, '
                      f'narrow step ***')
            got.append(d[[c for c in COLS if c in d.columns]])
        if not got:
            continue
        d = pd.concat(got, ignore_index=True)
        # drop dates already cached (window edges can overlap)
        d = d[~d['game_date'].astype(str).str[:10].isin(have)]
        print(f'  {gt}..{lt}: +{len(d)} rows')
        if len(d):
            frames.append(d)
            have |= set(d['game_date'].astype(str).str[:10].unique())
    alld = pd.concat(frames, ignore_index=True)
    tmp = raw_path + '.tmp'
    alld.to_pickle(tmp)
    os.replace(tmp, raw_path)
    print(f'raw cache: {len(alld)} rows -> {raw_path}')
    return alld


def join_population(pitches, raw, kin, side, label):
    """Fingerprint-join a pitch population to the raw kinematics frame and
    add matches to the PitchID-keyed sidecar dict."""
    gpk = pd.to_numeric(raw['game_pk'], errors='coerce').values
    ab = pd.to_numeric(raw['at_bat_number'], errors='coerce').values
    v = pd.to_numeric(raw['release_speed'], errors='coerce').values
    px = pd.to_numeric(raw['plate_x'], errors='coerce').values
    pz = pd.to_numeric(raw['plate_z'], errors='coerce').values
    e = kin['kin_eff'].values
    dv = kin['kin_dev'].values
    cd = kin['kin_cd'].values
    cands = defaultdict(list)
    for i in range(len(raw)):
        if np.isfinite(gpk[i]) and np.isfinite(ab[i]):
            cands[(int(gpk[i]), int(ab[i]))].append(i)

    by_pa = defaultdict(list)
    for p in pitches:
        pid = p['PitchID']
        try:
            g, a, _ = pid.split('_')
            by_pa[(int(g), int(a))].append(p)
        except ValueError:
            continue

    matched = unmatched = 0
    for key, mine in by_pa.items():
        cl = cands.get(key, [])
        used = [False] * len(cl)
        for p in mine:
            mv, mx, mz = sf(p.get('Velocity')), sf(p.get('PlateX')), sf(p.get('PlateZ'))
            best_j, best_d = None, 1e9
            if mv is not None:
                for j, i in enumerate(cl):
                    if used[j] or not np.isfinite(v[i]) or abs(v[i] - mv) > 0.25:
                        continue
                    dist = abs(v[i] - mv) * 2.0
                    if mx is not None and np.isfinite(px[i]):
                        dist += abs(px[i] - mx)
                    if mz is not None and np.isfinite(pz[i]):
                        dist += abs(pz[i] - mz)
                    if dist < best_d:
                        best_d, best_j = dist, j
            if best_j is None or best_d > 0.5:
                unmatched += 1
                continue
            used[best_j] = True
            i = cl[best_j]
            if np.isfinite(e[i]):
                side[p['PitchID']] = (float(e[i]), float(dv[i]), float(cd[i]))
            matched += 1
    print(f'{label} join: {matched} matched / {unmatched} unmatched '
          f'({matched / max(matched + unmatched, 1):.1%})')


def main():
    print('loading 2026 pitch cache ...')
    allp = pickle.load(open(PKL, 'rb'))
    mlb = [p for p in allp if p.get('_source') == 'MLB' and p.get('PitchID')]
    pitch_dates = sorted({str(p.get('Game Date'))[:10] for p in mlb
                          if p.get('Game Date')})
    print(f'{len(mlb)} MLB pitches, {len(pitch_dates)} game dates '
          f'({pitch_dates[0]} .. {pitch_dates[-1]})')

    side = {}
    raw = fetch_missing(pitch_dates)
    kin = compute_kinematics(raw)
    print(f'kinematics on {kin.kin_eff.notna().mean() * 100:.1f}% of raw rows')
    join_population(mlb, raw, kin, side, 'MLB')

    # ROC/AAA (v13 parity): both sources ride the same PitchID convention,
    # so the minors rows join with the identical fingerprint machinery and
    # land in the same sidecar — no consumer changes anywhere.
    roc = [p for p in allp
           if p.get('_source') in ('ROC', 'AAA') and p.get('PitchID')]
    if roc:
        roc_dates = sorted({str(p.get('Game Date'))[:10] for p in roc
                            if p.get('Game Date')})
        print(f'{len(roc)} ROC/AAA pitches, {len(roc_dates)} game dates')
        raw_m = fetch_missing(
            roc_dates,
            urls=(MILB_URL.replace('{ptype}', 'pitcher'),
                  MILB_URL.replace('{ptype}', 'batter')),
            raw_path=RAW_MILB)
        if raw_m is not None and len(raw_m):
            kin_m = compute_kinematics(raw_m)
            print(f'minors kinematics on '
                  f'{kin_m.kin_eff.notna().mean() * 100:.1f}% of raw rows')
            join_population(roc, raw_m, kin_m, side, 'ROC/AAA')

    tmp = SIDE + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(side, f)
    os.replace(tmp, SIDE)
    print(f'sidecar {len(side)} PitchIDs -> {SIDE}')


if __name__ == '__main__':
    main()
