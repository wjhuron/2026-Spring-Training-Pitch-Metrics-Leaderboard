#!/usr/bin/env python3
"""stuff_combo_battery.py — refine the velo_diff mask and race the combined
candidate configs, full production harness (imported from
stuff_feature_battery_2026_08), paired folds.

From the two prior batteries:
  VDMASK_FB (mask FF/SI/FC) won big on pred (+0.0176) but crashed cutter
  desc (0.267 -> 0.128): cutters live off their gap to the fastball.
Variants here:
  BASE            shipped
  VDMASK_FFSI     velo_diff masked on FF/SI only (cutters keep theirs)
  CROSS_VD        CROSS feature swap + VDMASK_FFSI
  NVAA_CROSS_VD   the full candidate config: nVAA rebuild + CROSS + mask
                  (family B frames)

Usage: python3 scripts/stuff_combo_battery.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
import scripts.stuff_feature_battery_2026_08 as BAT


def load_family(transform=None):
    p26 = BAT.load_mlb_2026()
    slopes = None
    if transform:
        # nVAA slopes must come from untransformed frames
        slopes = transform
        p26t = BAT.transform_nvaa(p26, slopes)
    else:
        p26t = p26
    df26 = BAT.build_season(p26t, (T.LG_WOBA, T.WOBA_SCALE))
    priors = []
    for yr in BAT.YEARS:
        pk = pickle.load(open(
            os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        if yr == 2025:
            T._harmonize_tags(pk, p26)
        if transform:
            pk = BAT.transform_nvaa(pk, slopes)
        d = BAT.build_season(pk, BAT.GUTS[yr])
        priors.append(d)
    prior = pd.concat(priors, ignore_index=True)
    return df26, prior


def mask_ffsi(df26, prior):
    a, b = df26.copy(), prior.copy()
    for dd in (a, b):
        dd.loc[dd['pitch_type'].isin({'FF', 'SI'}), 'velo_diff'] = np.nan
    return a, b


def main():
    print('family A frames ...', flush=True)
    df26, prior = load_family()
    with open(BAT.PITCH_LB) as f:
        lb = json.load(f)
    loc_map = {(r['pitcher'], r['team'], r['pitchType']): -r['locPlusRaw']
               for r in lb if r.get('locPlusRaw') is not None}
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}

    def stamp(d):
        d['half'] = d['date'].map(order).fillna(0).astype(int) % 2
        d['period'] = np.where(d['date'] < '2026-05-01', 'early', 'late')
        return d

    df26 = stamp(df26)
    BASE = list(T.BASE_FEATS)
    CROSS_FEATS = [f for f in BASE if f not in ('axis_dev', 'axis_dev_abs')] \
        + ['cross', 'cross_abs']
    df26_m, prior_m = mask_ffsi(df26, prior)

    results = {}

    def go(name, a26, apr, feats):
        d26, dt = BAT.run_variant(name, a26, apr, feats)
        rel, pred, desc, indep, fams, n_rel, n_pred = \
            BAT.unit_metrics(d26, loc_map)
        results[name] = dict(rel=rel, pred=pred, desc=desc, indep=indep)
        fam_s = '  '.join(f'{f[:4]} d{v["desc"]:.3f}/p{v["pred"]:.3f}'
                          for f, v in sorted(fams.items()))
        print(f'{name:<14} reliab {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}'
              f'  locIndep {indep:+.4f}  [{dt:.0f}s]\n'
              f'               {fam_s}', flush=True)

    go('BASE', df26, prior, BASE)
    go('VDMASK_FFSI', df26_m, prior_m, BASE)
    go('CROSS_VD', df26_m, prior_m, CROSS_FEATS)

    print('family B frames (nVAA) ...', flush=True)
    slopes = BAT.fit_nvaa_slopes([df26, prior])
    del df26_m, prior_m
    df26_b, prior_b = load_family(transform=slopes)
    df26_b = stamp(df26_b)
    df26_bm, prior_bm = mask_ffsi(df26_b, prior_b)
    del df26_b, prior_b
    go('NVAA_CROSS_VD', df26_bm, prior_bm, CROSS_FEATS)

    print('\n=== PAIRED DELTAS vs BASE ===')
    b = results['BASE']
    for name, r in results.items():
        if name != 'BASE':
            print(f'{name:<14} d_rel {r["rel"]-b["rel"]:+.4f}  '
                  f'd_pred {r["pred"]-b["pred"]:+.4f}  '
                  f'd_desc {r["desc"]-b["desc"]:+.4f}  '
                  f'd_|locIndep| {abs(r["indep"])-abs(b["indep"]):+.4f}')
    with open(os.path.join(ROOT, 'data', '_combo_battery_results.json'),
              'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
