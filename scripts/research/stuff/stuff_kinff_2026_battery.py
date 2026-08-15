#!/usr/bin/env python3
"""stuff_kinff_2026_battery.py — 2026-harness confirmation for KIN_FF, the
one survivor of the per-type LOSO gate (fut 4/5, mean +0.0042; see
scripts/research/stuff/stuff_pertype_loso_gate.py and data/_stuff_pertype_gate.json).

KIN_FF = type-gated kinematic efficiency: a new column kin_eff__FF equal to
kin_eff on four-seamers, NaN elsewhere. Motivated by the per-type atlas
(scripts/research/stuff/stuff_pertype_atlas.py): kin_eff carries its largest out-of-sample
permutation importance on FF (16.1 x1000, 5/5 seasons) while the GLOBAL
kin_eff add was already tested and rejected 2026-08-09 (prediction-neutral;
data/_kin_battery_results.json).

Protocol: paired variants on the production 2026 harness (8-fold pitcher-
grouped OOF, 2021-2025 priors in every fold), exactly the battery that
adopted v12. Metrics: rel / pred / desc / indep via BAT.unit_metrics.
Adoption needs this AND the LOSO gate to agree.

Usage: python3 scripts/research/stuff/stuff_kinff_2026_battery.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.stuff_feature_battery_2026_08 as BAT

BAT.KEEP = list(dict.fromkeys(
    BAT.KEEP + ['kin_eff', 'kin_dev', 'kin_cd', 'pid']))
SIDE = os.path.join(ROOT, 'data', 'kinematics_2026_sidecar.pkl')
OUT = os.path.join(ROOT, 'data', '_stuff_kinff_battery.json')


def main():
    print('loading 2026 ...', flush=True)
    p26 = BAT.load_mlb_2026()
    df26 = BAT.build_season(p26, (T.LG_WOBA, T.WOBA_SCALE))
    side = pickle.load(open(SIDE, 'rb'))
    vals = df26['pid'].map(side)
    has = vals.notna()
    for j, col in enumerate(('kin_eff', 'kin_dev', 'kin_cd')):
        df26.loc[has, col] = vals[has].str[j]
        df26[col] = pd.to_numeric(df26[col], errors='coerce')
    print(f'  2026: {len(df26)} rows, sidecar filled {has.mean()*100:.1f}%',
          flush=True)

    print('loading priors ...', flush=True)
    priors = []
    for yr in BAT.YEARS:
        pk = pickle.load(open(
            os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        d = BAT.build_season(pk, BAT.GUTS[yr],
                             harmonize_against=p26 if yr == 2025 else None)
        d['_yr'] = yr
        priors.append(d)
        print(f'  {yr}: {len(d)} rows', flush=True)
    prior = pd.concat(priors, ignore_index=True)
    del priors, p26
    for col in ('kin_eff',):
        prior[col] = pd.to_numeric(prior[col], errors='coerce')

    # typed column, both frames
    for dd in (df26, prior):
        dd['kin_eff__FF'] = np.where(dd['pitch_type'] == 'FF',
                                     dd['kin_eff'], np.nan)
    cov26 = df26.loc[df26.pitch_type == 'FF', 'kin_eff__FF'].notna().mean()
    covpr = prior.loc[prior.pitch_type == 'FF', 'kin_eff__FF'].notna().mean()
    print(f'  kin_eff__FF coverage on FF: 2026 {cov26*100:.1f}%, '
          f'prior {covpr*100:.1f}%', flush=True)

    with open(BAT.PITCH_LB) as f:
        lb = json.load(f)
    loc_map = {(r['pitcher'], r['team'], r['pitchType']): -r['locPlusRaw']
               for r in lb if r.get('locPlusRaw') is not None}
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')

    BASE = list(T.BASE_FEATS)
    results = {}
    for name, feats in (('SHIPPED', BASE),
                        ('KIN_FF', BASE + ['kin_eff__FF'])):
        d26, dt = BAT.run_variant(name, df26, prior, feats)
        rel, pred, desc, indep, fams, n_rel, n_pred = \
            BAT.unit_metrics(d26, loc_map)
        results[name] = dict(rel=rel, pred=pred, desc=desc, indep=indep,
                             fams={k: v for k, v in fams.items()})
        print(f'  {name:<8} rel {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}'
              f'  indep {indep:.4f}  (n_rel {n_rel}, n_pred {n_pred})'
              f'  [{dt/60:.1f} min]', flush=True)
        if 'fastball' in fams:
            print(f'           fastball family: '
                  f'desc {fams["fastball"]["desc"]:.4f}  '
                  f'pred {fams["fastball"]["pred"]:.4f}', flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
