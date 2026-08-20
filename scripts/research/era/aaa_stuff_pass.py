"""aaa_stuff_pass.py — Stuff+ for Triple-A and MLB, scored the same way.

hpERA's heaviest channel is Stuff+ at weight .297, and the AAA battery
leaves it None because era_battery_build's adapter does not carry the
physical features the model reads. Without it the hpERA translation cannot
be tested at all. This pass scores BOTH levels with the SHIPPED bundle so
the two sides are commensurable, which matters more here than matching the
production value to the digit.

MODEL. stuff_plus/stuff_models.pkl, the v13 bundle. Triple-A pitches are
SCORED against it and never enter it, and the per-pitch atom uses the MLB
per-type anchors in B['league'] (or B['na_pt_scale'] for the no-arm
companion), exactly as train_stuff.py scores Rochester today.

THREE DOCUMENTED APPROXIMATIONS. All three apply to BOTH levels, so they
shift the absolute Stuff+ without biasing the AAA-minus-MLB difference,
except where noted.

1. RAW MOVEMENT, NOT DENSITY-ADJUSTED. build_df reads xIndVrtBrk/xHorzBrk,
   which pipeline/fetch.py computes from venue elevation and weather. There
   is no MiLB weather sidecar, so raw pfx is fed on both sides — the same
   fallback production uses when the adjustment has not landed.

   THIS ONE IS NOT SYMMETRIC AND THE DIRECTION MATTERS. Triple-A carries
   about five genuinely high-altitude parks (Reno, Albuquerque, El Paso,
   Salt Lake, Las Vegas) against one in MLB, and thin air reduces movement.
   So raw movement makes Triple-A stuff look WORSE than it is. A result of
   'Stuff+ needs no translation' is therefore CONSERVATIVE: the true AAA
   value can only be higher than measured here. A result of 'AAA Stuff+ is
   lower' cannot be separated from altitude by this pass.

2. kin_eff__ff IS IMPUTED at the frozen 0.7038. The kinematics sidecar
   covers MLB 2026 and Rochester only, so no season here can be fed the
   measured value on both sides.

3. FC_ANCHOR_PITCHERS DOES NOT FIRE. It is keyed by name and this pass
   keys pitchers by numeric MLB id, which is what makes the AAA-to-MLB
   join exact. Three pitchers lose a cutter fastball-reference exception.

Output: data/_aaa_stuff.json   {level: {season: {pid: {v, n}}}}
        level is 'AAA' or 'MLB'.

    python3 scripts/research/era/aaa_stuff_pass.py
    python3 scripts/research/era/aaa_stuff_pass.py --levels AAA --seasons 2024
"""
import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from era_battery_build import vaa_at_plate, _f, DESC_MAP
import train_stuff as TS

OUT = os.path.join(ROOT, 'data', '_aaa_stuff.json')
BUNDLE = os.path.join(ROOT, 'stuff_plus', 'stuff_models.pkl')
PATHS = {
    'AAA': {y: f'data/_aaa_statcast{y}_cache.pkl' for y in (2023, 2024, 2025, 2026)},
    'MLB': {2023: 'data/_statcast2023_cache.pkl',
            2024: 'data/_statcast2024_cache.pkl',
            2025: 'data/_statcast2025_full_cache.pkl',
            2026: 'data/_statcast2026_full.pkl'},
}
MIN_N = 50


ELEV_PATH = os.path.join(ROOT, 'data', '_park_elevation.json')


def _elev_map():
    try:
        with open(ELEV_PATH) as f:
            return {int(k): v for k, v in json.load(f)['games'].items()
                    if v is not None}
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def adapt_stuff(path, max_elev=None):
    """Statcast DataFrame -> the sheet-shaped dicts build_df reads.

    max_elev drops every pitch thrown at a park above that elevation. It
    exists because raw movement is altitude-sensitive and Triple-A carries
    five parks above 3,000 ft against MLB's one, which would otherwise read
    as a stuff difference between the levels rather than between the parks.

    A game with NO elevation mapping is DROPPED, never kept. Keeping it
    would let an unmapped park slip through the filter unnoticed, which is
    the exact failure the filter exists to prevent.
    """
    df = pickle.load(open(path, 'rb'))
    df = df[df['game_type'] == 'R']
    if max_elev is not None:
        em = _elev_map()
        if not em:
            sys.exit(f'ABORT: --max-elev given but {ELEV_PATH} is missing '
                     f'or empty. Run scripts/builders/park_elevation_pull.py')
        gp = pd.to_numeric(df['game_pk'], errors='coerce')
        elev = gp.map(em)
        keep = elev.notna() & (elev <= max_elev)
        n_hi = int((elev.notna() & (elev > max_elev)).sum())
        n_un = int(elev.isna().sum())
        print(f'    elevation filter <= {max_elev} ft: dropped {n_hi} '
              f'high-altitude pitches and {n_un} with no mapping, '
              f'kept {int(keep.sum())}', flush=True)
        df = df[keep]
    cols = ['pitcher', 'game_pk', 'game_date', 'pitch_type', 'description', 'p_throws',
            'stand', 'release_speed', 'release_spin_rate', 'release_extension',
            'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_z',
            'arm_angle', 'vy0', 'vz0', 'ay', 'az', 'spin_axis',
            'estimated_woba_using_speedangle', 'delta_pitcher_run_exp']
    sub = df[cols]
    out = []
    for r in sub.itertuples(index=False):
        desc = DESC_MAP.get(r.description)
        if desc is None:
            continue
        try:
            pid = int(r.pitcher)
        except (TypeError, ValueError):
            continue
        ivb = _f(r.pfx_z)
        hb = _f(r.pfx_x)
        ivb = None if ivb is None else ivb * 12.0
        hb = None if hb is None else hb * 12.0
        out.append({
            'Pitcher': pid, 'Game Date': str(r.game_date)[:10],
            'Pitch Type': r.pitch_type if isinstance(r.pitch_type, str) else None,
            'Throws': r.p_throws if isinstance(r.p_throws, str) else None,
            'Bats': r.stand if isinstance(r.stand, str) else None,
            'Description': desc,
            'Velocity': _f(r.release_speed), 'Spin Rate': _f(r.release_spin_rate),
            'Extension': _f(r.release_extension),
            'RelPosX': _f(r.release_pos_x), 'RelPosZ': _f(r.release_pos_z),
            'IndVertBrk': ivb, 'HorzBrk': hb,
            # raw movement stands in for the density-adjusted columns; see
            # approximation 1 in the module docstring
            'xIndVrtBrk': ivb, 'xHorzBrk': hb,
            'PlateZ': _f(r.plate_z), 'ArmAngle': _f(r.arm_angle),
            # cross / cross_abs are BASE_FEATS and both need the release
            # axis. Omitting SpinAxis left them all-None, which pandas types
            # as object and xgboost refuses outright -- a loud failure, but
            # only because the column was ENTIRELY empty. A partly-populated
            # axis would have trained silently on a degraded feature.
            'SpinAxis': _f(r.spin_axis),
            'VAA': vaa_at_plate(_f(r.vy0), _f(r.vz0), _f(r.ay), _f(r.az)),
            'xwOBA': _f(r.estimated_woba_using_speedangle),
            'RunExp': _f(r.delta_pitcher_run_exp),
        })
    del df, sub
    gc.collect()
    return out


def score(pitches, B):
    """pid -> (mean per-pitch Stuff+ atom, n). Mirrors the ROC block in
    train_stuff.py: full model where arm angle exists, no-arm companion
    otherwise, each against its own anchor set."""
    df = TS.build_df(pitches)
    if not len(df):
        return {}
    arm = df['arm_angle'].notna().values
    df = df.copy()
    df['_raw'] = np.nan
    if arm.any():
        X = TS.design(df[arm]).reindex(columns=B['features'], fill_value=0)
        df.loc[arm, '_raw'] = -B['model'].predict(X)
    if (~arm).any():
        Xn = TS.design(df[~arm], TS.NOARM_FEATS).reindex(
            columns=B['features_na'], fill_value=0)
        df.loc[~arm, '_raw'] = -B['model_na'].predict(Xn)
    df['_use_arm'] = arm
    atom = pd.Series(np.nan, index=df.index)
    for use_arm, anchors in ((True, B['league']), (False, B['na_pt_scale'])):
        m = (df['_use_arm'] == use_arm).values
        if not m.any():
            continue
        s = df.loc[m]
        vals = pd.Series(np.nan, index=s.index)
        for pt, sc in anchors.items():
            if (not isinstance(sc, dict)
                    or not np.isfinite(sc.get('sd', np.nan)) or sc['sd'] <= 0):
                continue
            mm = (s['pitch_type'] == pt).values
            if mm.any():
                vals[mm] = (100 + TS.K_SCALE
                            * (s.loc[mm, '_raw'] - sc['mu']) / sc['sd'])
        atom[m] = vals
    df['_atom'] = atom
    g = df.dropna(subset=['_atom']).groupby('pitcher')['_atom'].agg(
        v='mean', n='size')
    print(f'    {int(arm.sum())}/{len(df)} pitches by the full arm model',
          flush=True)
    return {int(pid): (float(r.v), int(r.n)) for pid, r in g.iterrows()
            if r.n >= MIN_N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', nargs='*', default=['AAA', 'MLB'])
    ap.add_argument('--seasons', nargs='*', type=int,
                    default=[2023, 2024, 2025, 2026])
    ap.add_argument('--max-elev', type=int, default=None,
                    help='drop pitches thrown above this elevation (ft)')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    out_path = a.out
    with open(BUNDLE, 'rb') as f:
        B = pickle.load(f)
    print(f"bundle v{B.get('version')} trained through "
          f"{B.get('trained_through')}", flush=True)
    result = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            result = json.load(f)
    for lvl in a.levels:
        result.setdefault(lvl, {})
        for y in a.seasons:
            rel = PATHS[lvl].get(y)
            p = os.path.join(ROOT, rel) if rel else None
            if not p or not os.path.exists(p):
                print(f'{lvl} {y}: {rel} absent — skipped', flush=True)
                continue
            print(f'{lvl} {y}: {rel}', flush=True)
            pitches = adapt_stuff(p, a.max_elev)
            sc = score(pitches, B)
            del pitches
            gc.collect()
            result[lvl][str(y)] = {str(k): {'v': v, 'n': n}
                                   for k, (v, n) in sc.items()}
            vals = [v for v, _ in sc.values()]
            print(f'    {len(sc)} pitchers | mean Stuff+ '
                  f'{sum(vals)/len(vals):.2f}' if vals else '    none',
                  flush=True)
            with open(out_path + '.tmp', 'w') as f:
                json.dump(result, f)
            os.replace(out_path + '.tmp', out_path)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
