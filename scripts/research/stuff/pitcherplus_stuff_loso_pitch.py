#!/usr/bin/env python3
"""pitcherplus_stuff_loso_pitch.py — PER-PITCH leave-one-season-out v14
Stuff+ raw predictions, 2021-2025.

Why (2026-09-02): the outing Pitching+ weights (cards/pitcher.py
PP_OUTING_W) were fit with stuffRaw taken from the FULL-GAME LOSO mean on
BOTH odd/even-PA halves (pitcherplus_outing_grade.load), so stuff's
half-reliability was 1.0 by construction and its weight is biased up. A
true per-half stuff needs the per-pitch prediction, which the existing
pitcherplus_stuff_loso_v14.py aggregates away. This script keeps it.

Same machinery as pitcherplus_stuff_loso_v14.py: season Y is scored by a
model fit on the other four seasons (train_stuff.build_df / design /
_params_for verbatim, per-season FG Guts for targets). Cross-season
same-pitcher rows remain in training (the v11/v14 leakage standard).

Memory: the machine has 8 GB. Stage `features` builds ONE season at a time
and saves a slim float32 frame to the scratch dir; stage `loso` loads four
of those per fold. Each fold's predictions are written as soon as the fold
finishes, so a partial run is usable.

Join key: the training pickles carry no pitch id and no within-game order
(the 2025 pickle is re-sorted and rounded to 2 decimals), so every row keeps
a fingerprint (game_pk, pitcher, pitch_type, velocity, plate_x, plate_z)
for a per-pitch join to the Savant cache rows that pitcherplus_outing_tables
builds the PA index from. The join itself lives in
pitcherplus_outing_refit.py.

Usage:
  python3 scripts/research/stuff/pitcherplus_stuff_loso_pitch.py features
  python3 scripts/research/stuff/pitcherplus_stuff_loso_pitch.py loso
  python3 scripts/research/stuff/pitcherplus_stuff_loso_pitch.py all
Output: data/_pplus_stuff_loso_pitch.pkl (DataFrame: season, pitcher, date,
        game_pk, pitch_type, fp_velocity, fp_plate_x, fp_plate_z, target_xrv,
        stuff_raw [pitcher perspective, higher = better]; the refit script
        adds pid / pa_idx / pitch_ord / atom after the cache join)
"""
import gc
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import stuff_plus.train_stuff as tv                    # noqa: E402

SCRATCH = os.environ.get(
    'PPLUS_SCRATCH',
    '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/'
    '757704f0-0078-43ef-a650-75186bd117dd/scratchpad')
OUT_PKL = os.path.join(ROOT, 'data', '_pplus_stuff_loso_pitch.pkl')
SEASONS = [2021, 2022, 2023, 2024, 2025]
GUTS = dict(tv.HIST_GUTS)
GUTS[2025] = (tv.PRIOR_LG_WOBA, tv.PRIOR_WOBA_SCALE)
DESIGN_COLS = list(tv.BASE_FEATS) + ['platoon_same']
KEEP = ['pitcher', 'date', 'game_pk', 'pitch_type', 'velocity',
        'plate_x', 'plate_z', 'target_xrv']
# 'velocity' is also a design feature (and the monotone one, so its design
# name must stay); the fingerprint copies carry an fp_ prefix.
FP_RENAME = {'velocity': 'fp_velocity', 'plate_x': 'fp_plate_x',
             'plate_z': 'fp_plate_z'}
KEEP_FP = [FP_RENAME.get(c, c) for c in KEEP]


def feat_path(year):
    return os.path.join(SCRATCH, f'_pplus_stuff_feat_{year}.pkl')


def fold_path(year):
    return os.path.join(SCRATCH, f'_pplus_stuff_loso_fold_{year}.pkl')


def build_year(year):
    """build_df on the season pickle with that season's Guts; keep the design
    columns (float32) plus the fingerprint/target columns."""
    t0 = time.time()
    pitches = pickle.load(open(
        os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl'), 'rb'))
    # build_df emits p['PitchID'] as 'pid'; the pickles carry none, so use
    # the pickle row index and map game_pk back through it.
    gpk = np.empty(len(pitches), dtype=np.int64)
    for i, p in enumerate(pitches):
        p['PitchID'] = i
        gpk[i] = int(p.get('_game_pk') or -1)
    lg, sc = GUTS[year]
    _lg0, _sc0 = tv.LG_WOBA, tv.WOBA_SCALE
    tv.LG_WOBA, tv.WOBA_SCALE = lg, sc
    d = tv.build_df(pitches)
    tv.LG_WOBA, tv.WOBA_SCALE = _lg0, _sc0
    del pitches
    gc.collect()
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    d['game_pk'] = gpk[d['pid'].astype(int).to_numpy()]
    n_nogpk = int((d['game_pk'] < 0).sum())
    if n_nogpk:
        print(f'  {year}: {n_nogpk} rows without game_pk', flush=True)
    X = tv.design(d).astype('float32')
    X.columns = DESIGN_COLS
    out = pd.concat([d[KEEP].rename(columns=FP_RENAME).reset_index(drop=True),
                     X], axis=1)
    out['season'] = year
    with open(feat_path(year), 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  {year}: {len(out)} training pitches, '
          f'{out.pitcher.nunique()} pitchers, '
          f'{time.time() - t0:.0f}s -> {feat_path(year)}', flush=True)
    del d, X, out
    gc.collect()


def stage_features():
    for y in SEASONS:
        if os.path.exists(feat_path(y)):
            print(f'  {y}: features exist, skip', flush=True)
            continue
        build_year(y)


def stage_loso(seasons=None):
    seasons = seasons or SEASONS
    for year in seasons:
        if os.path.exists(fold_path(year)):
            print(f'── LOSO {year}: fold exists, skip', flush=True)
            continue
        t0 = time.time()
        print(f'── LOSO {year}', flush=True)
        frames = [pickle.load(open(feat_path(y), 'rb'))
                  for y in SEASONS if y != year]
        train = pd.concat(frames, ignore_index=True)
        del frames
        X = train[DESIGN_COLS]
        y = train['target_xrv'].to_numpy(dtype='float32')
        model = XGBRegressor(**tv._params_for(X))
        model.fit(X, y)
        del train, X, y
        gc.collect()
        test = pickle.load(open(feat_path(year), 'rb'))
        pred = model.predict(test[DESIGN_COLS])
        out = test[KEEP_FP + ['season']].copy()
        out['stuff_raw'] = -pred.astype('float64')   # pitcher perspective
        with open(fold_path(year), 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        r = np.corrcoef(out['stuff_raw'], -out['target_xrv'])[0, 1]
        print(f'  {year}: {len(out)} pitches scored, per-pitch r vs target '
              f'{r:.4f}, {time.time() - t0:.0f}s', flush=True)
        del model, test, out
        gc.collect()


def assemble():
    parts = []
    for y in SEASONS:
        if os.path.exists(fold_path(y)):
            parts.append(pickle.load(open(fold_path(y), 'rb')))
        else:
            print(f'  WARNING fold {y} missing, output is partial', flush=True)
    out = pd.concat(parts, ignore_index=True)
    tmp = OUT_PKL + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, OUT_PKL)
    print(f'saved {OUT_PKL} ({len(out)} rows, seasons '
          f'{sorted(out.season.unique().tolist())})', flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(SCRATCH, exist_ok=True)
    if stage in ('features', 'all'):
        stage_features()
    if stage in ('loso', 'all'):
        stage_loso()
        assemble()
    if stage == 'assemble':
        assemble()


if __name__ == '__main__':
    main()
