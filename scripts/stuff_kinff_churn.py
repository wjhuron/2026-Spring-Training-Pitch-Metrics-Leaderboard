#!/usr/bin/env python3
"""stuff_kinff_churn.py — displayed-scale score-churn distribution for
KIN_FF vs SHIPPED (companion to stuff_kinff_2026_battery.py; run before any
ship decision so the site impact is known).

Reruns the paired 2026 battery variants (same folds — GroupKFold on the
same row order, so per-pitch predictions are paired), then converts each
config's raw OOF to the DISPLAYED convention: per-(pitcher, pitch_type)
rawmean, standardized 100 +/- 10 against the qualified per-type pool
(n >= 50), shrunk toward the pool mean by K_SHRINK=100 pseudo-pitches —
the production season-card scale. Reports the per-unit and per-pitcher
delta distribution.

Usage: python3 scripts/stuff_kinff_churn.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus.train_stuff as T
import scripts.stuff_feature_battery_2026_08 as BAT

BAT.KEEP = list(dict.fromkeys(
    BAT.KEEP + ['kin_eff', 'kin_dev', 'kin_cd', 'pid']))
SIDE = os.path.join(ROOT, 'data', 'kinematics_2026_sidecar.pkl')
OUT = os.path.join(ROOT, 'data', '_stuff_kinff_churn.json')
K_SHRINK = 100
QUAL_N = 50


def displayed_scores(d26):
    """(pitcher, pitch_type) -> displayed score; plus pitcher -> overall."""
    unit = (d26.groupby(['pitcher', 'pitch_type'])['stuff']
            .agg(['mean', 'size']).reset_index())
    scores = {}
    for pt, sub in unit.groupby('pitch_type'):
        qual = sub[sub['size'] >= QUAL_N]
        if len(qual) < 20:
            continue
        mu, sd = qual['mean'].mean(), qual['mean'].std()
        if not sd > 0:
            continue
        shr = (sub['size'] * sub['mean'] + K_SHRINK * mu) / (sub['size'] + K_SHRINK)
        val = 100 + 10 * (shr - mu) / sd
        for p, v, n in zip(sub['pitcher'], val, sub['size']):
            scores[(p, pt)] = (float(v), int(n))
    ov = (d26.groupby('pitcher')['stuff'].agg(['mean', 'size']).reset_index())
    qual = ov[ov['size'] >= 100]
    mu, sd = qual['mean'].mean(), qual['mean'].std()
    shr = (ov['size'] * ov['mean'] + K_SHRINK * mu) / (ov['size'] + K_SHRINK)
    val = 100 + 10 * (shr - mu) / sd
    overall = {p: (float(v), int(n))
               for p, v, n in zip(ov['pitcher'], val, ov['size'])}
    return scores, overall


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

    print('loading priors ...', flush=True)
    priors = []
    for yr in BAT.YEARS:
        pk = pickle.load(open(
            os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        d = BAT.build_season(pk, BAT.GUTS[yr],
                             harmonize_against=p26 if yr == 2025 else None)
        priors.append(d)
    prior = pd.concat(priors, ignore_index=True)
    del priors, p26
    prior['kin_eff'] = pd.to_numeric(prior['kin_eff'], errors='coerce')

    for dd in (df26, prior):
        dd['kin_eff__FF'] = np.where(dd['pitch_type'] == 'FF',
                                     dd['kin_eff'], np.nan)

    BASE = list(T.BASE_FEATS)
    per = {}
    for name, feats in (('SHIPPED', BASE),
                        ('KIN_FF', BASE + ['kin_eff__FF'])):
        d26, dt = BAT.run_variant(name, df26, prior, feats)
        per[name] = displayed_scores(d26)
        print(f'  {name}: scored [{dt/60:.1f} min]', flush=True)

    (s_unit, s_ov), (k_unit, k_ov) = per['SHIPPED'], per['KIN_FF']

    def dist(deltas, label):
        a = np.array(deltas)
        out = dict(n=len(a), mean_abs=round(float(np.abs(a).mean()), 3),
                   p50=round(float(np.percentile(np.abs(a), 50)), 2),
                   p90=round(float(np.percentile(np.abs(a), 90)), 2),
                   p99=round(float(np.percentile(np.abs(a), 99)), 2),
                   max_abs=round(float(np.abs(a).max()), 2),
                   ge1=int((np.abs(a) >= 1).sum()),
                   ge2=int((np.abs(a) >= 2).sum()),
                   ge3=int((np.abs(a) >= 3).sum()))
        print(f'{label}: n={out["n"]}  mean|d|={out["mean_abs"]}  '
              f'p50={out["p50"]}  p90={out["p90"]}  p99={out["p99"]}  '
              f'max={out["max_abs"]}  >=1: {out["ge1"]}  >=2: {out["ge2"]}  '
              f'>=3: {out["ge3"]}', flush=True)
        return out

    res = {}
    common = [k for k in s_unit if k in k_unit]
    res['units_all'] = dist([k_unit[k][0] - s_unit[k][0] for k in common],
                            'ALL units')
    ff = [k for k in common if k[1] == 'FF']
    res['units_ff'] = dist([k_unit[k][0] - s_unit[k][0] for k in ff],
                           'FF units')
    non = [k for k in common if k[1] != 'FF']
    res['units_nonff'] = dist([k_unit[k][0] - s_unit[k][0] for k in non],
                              'non-FF units')
    ovc = [p for p in s_ov if p in k_ov]
    res['overall'] = dist([k_ov[p][0] - s_ov[p][0] for p in ovc],
                          'OVERALL pitcher')

    movers = sorted(((k, k_unit[k][0] - s_unit[k][0], s_unit[k][1])
                     for k in common if s_unit[k][1] >= QUAL_N),
                    key=lambda x: -abs(x[1]))[:15]
    print('\nbiggest qualified unit movers:')
    res['movers'] = []
    for (p, pt), dv, n in movers:
        print(f'  {p:<22} {pt}  {dv:+.1f}  (n={n})', flush=True)
        res['movers'].append([p, pt, round(dv, 1), n])

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
