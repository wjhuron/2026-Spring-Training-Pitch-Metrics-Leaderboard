#!/usr/bin/env python3
"""Regenerate micro_data_rs.json with CURRENT grade-atom dumps.

process_data.py builds micro data mid-run, when the Stuff+ dump on disk is
still the PREVIOUS train_stuff run's — so games newer than that run
would carry no stuff atoms in the embedded micro data (filtered site views
would silently omit them from windowed Stuff+/Pitching+). This script runs
AFTER train_stuff --dump-pitch-grades and rebuilds micro data from the
pitch cache (written by process_data at the exact point micro generation
reads it, so inputs are bit-identical) with both dumps fresh.

Pipeline order: process_data.py -> train_stuff --score-only --inject
--dump-pitch-grades -> refresh_micro_grades.py -> rebuild_embed.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import process_data as PD
from pipeline_fetch import load_mlb_id_cache

LABEL = 'rs'
DATA = PD.DATA_DIR


def main():
    cache_path = os.path.join(DATA, f'all_pitches_{LABEL}_cache.pkl')
    with open(cache_path, 'rb') as f:
        all_pitches = pickle.load(f)
    print(f'  loaded {len(all_pitches)} pitches from {cache_path}')

    # Same EP derivation as process_game_type (anyone who threw an Eephus)
    ep_pitchers = {(p['Pitcher'], p['PTeam']) for p in all_pitches
                   if p.get('Pitch Type') == 'EP'}

    grades = {}
    for name, fn in (('stuff', 'pitch_stuff_grades.json'),
                     ('loc', f'pitch_loc_grades_{LABEL}.json')):
        path = os.path.join(DATA, fn)
        if os.path.exists(path):
            with open(path) as f:
                grades[name] = json.load(f)
            print(f'  {name} dump: {len(grades[name])} grades')
        else:
            grades[name] = {}
            print(f'  {name} dump missing ({path}) — its atoms will be empty')

    mlb_id_cache = load_mlb_id_cache(os.path.join(DATA, 'mlb_id_cache.json'))
    micro = PD.generate_micro_data(all_pitches, mlb_id_cache=mlb_id_cache,
                                   ep_pitchers=ep_pitchers,
                                   stuff_grades=grades['stuff'],
                                   loc_grades=grades['loc'])

    out_path = os.path.join(DATA, f'micro_data_{LABEL}.json')
    with open(out_path, 'w') as f:
        json.dump(PD.round_floats_inplace(micro), f, separators=(',', ':'))
    n_stuff = sum(r[-5] for r in micro['pitchMicro'])
    n_loc = sum(r[-3] for r in micro['pitchMicro'])
    print(f'  wrote {out_path}: {len(micro["pitchMicro"])} pitch micro rows '
          f'({n_stuff} stuff atoms, {n_loc} loc atoms)')


if __name__ == '__main__':
    main()
