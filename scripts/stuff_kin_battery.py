#!/usr/bin/env python3
"""stuff_kin_battery.py — do the per-pitch 9P kinematics features earn a spot
in Stuff+? Full production config, paired folds, same harness as
stuff_feature_battery_2026_08 (imported).

Candidates (backfilled 2021-2025 into the training pickles by
augment_kinematics.py, 2026 via kinematics_2026_sidecar.pkl):
  kin_eff  per-pitch transverse-spin fraction (measured active spin —
           validated 0.926 unit corr vs Savant's published values)
  kin_dev  per-pitch force-based SSW deviation (deg, hand-signed) — the
           instantaneous version of axis_dev, which stays in BASE
  kin_cd   per-pitch drag coefficient

Variants: BASE / KINEFF / KINDEV / KINCD / KINALL.

Usage: python3 scripts/stuff_kin_battery.py
"""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.stuff_feature_battery_2026_08 as BAT

BAT.KEEP = list(dict.fromkeys(
    BAT.KEEP + ['kin_eff', 'kin_dev', 'kin_cd', 'pid']))
SIDE = os.path.join(ROOT, 'data', 'kinematics_2026_sidecar.pkl')


def main():
    print('loading 2026 ...', flush=True)
    p26 = BAT.load_mlb_2026()
    df26 = BAT.build_season(p26, (T.LG_WOBA, T.WOBA_SCALE))
    import pandas as pd
    side = pickle.load(open(SIDE, 'rb'))
    vals = df26['pid'].map(side)
    has = vals.notna()
    for j, col in enumerate(('kin_eff', 'kin_dev', 'kin_cd')):
        df26.loc[has, col] = vals[has].str[j]
        df26[col] = pd.to_numeric(df26[col], errors='coerce')
    print(f'  2026: {len(df26)} rows, sidecar filled {has.mean()*100:.1f}%')

    print('loading priors ...', flush=True)
    priors = []
    import pandas as pd
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
    for col in ('kin_eff', 'kin_dev', 'kin_cd'):
        prior[col] = pd.to_numeric(prior[col], errors='coerce')

    print('\ncandidate coverage by season (%):')
    for col in ('kin_eff', 'kin_dev', 'kin_cd'):
        row = [f'2026:{df26[col].notna().mean()*100:.1f}']
        for yr in BAT.YEARS:
            sub = prior[prior._yr == yr]
            row.append(f'{yr}:{sub[col].notna().mean()*100:.1f}')
        print(f'  {col:<8} ' + '  '.join(row))

    with open(BAT.PITCH_LB) as f:
        lb = json.load(f)
    loc_map = {(r['pitcher'], r['team'], r['pitchType']): -r['locPlusRaw']
               for r in lb if r.get('locPlusRaw') is not None}
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')

    BASE = list(T.BASE_FEATS)
    # VDMASK_FB: velo_diff masked on FASTBALLS ONLY (FF/SI/FC), keeping it
    # for breaking + offspeed — the main battery's full mask improved
    # fastball pred (+0.056) while crashing breaking (-0.079), so the
    # surgical version tests whether both families can win at once.
    FB = {'FF', 'SI', 'FC'}
    df26_fb = df26.copy()
    prior_fb = prior.copy()
    for dd in (df26_fb, prior_fb):
        dd.loc[dd['pitch_type'].isin(FB), 'velo_diff'] = np.nan

    results = {}
    for name, feats, frames in (
            ('BASE', BASE, None),
            ('KINEFF', BASE + ['kin_eff'], None),
            ('KINDEV', BASE + ['kin_dev'], None),
            ('KINCD', BASE + ['kin_cd'], None),
            ('KINALL', BASE + ['kin_eff', 'kin_dev', 'kin_cd'], None),
            ('VDMASK_FB', BASE, (df26_fb, prior_fb))):
        a26, apr = frames if frames is not None else (df26, prior)
        d26, dt = BAT.run_variant(name, a26, apr, feats)
        rel, pred, desc, indep, fams, n_rel, n_pred = \
            BAT.unit_metrics(d26, loc_map)
        results[name] = dict(rel=rel, pred=pred, desc=desc, indep=indep)
        fam_s = '  '.join(f'{f[:4]} d{v["desc"]:.3f}/p{v["pred"]:.3f}'
                          for f, v in sorted(fams.items()))
        print(f'{name:<8} reliab {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}  '
              f'locIndep {indep:+.4f}  (n_rel {n_rel}, n_pred {n_pred})  '
              f'[{dt:.0f}s]\n         {fam_s}', flush=True)

    print('\n=== PAIRED DELTAS vs BASE ===')
    b = results['BASE']
    for name, r in results.items():
        if name != 'BASE':
            print(f'{name:<8} d_rel {r["rel"]-b["rel"]:+.4f}  '
                  f'd_pred {r["pred"]-b["pred"]:+.4f}  '
                  f'd_desc {r["desc"]-b["desc"]:+.4f}  '
                  f'd_|locIndep| {abs(r["indep"])-abs(b["indep"]):+.4f}')
    with open(os.path.join(ROOT, 'data', '_kin_battery_results.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
