"""era_stuff_loso_v2.py — per-pitcher-season LOSO Stuff+ (raw, xRV scale,
pitcher-positive) for the ERA-estimator replicates, 2021-2025, built on the
REPAIRED gate v2 machinery (stuff_gate_v2: single nVAA adjustment from
vaa_raw, clipped height, production params and monotone velocity).

Replaces era_stuff_loso_scores.py, whose v1 import chain
(stuff_features_loso -> stuff_feature_battery_2026_08 ->
scripts.build_historical_training_set) no longer resolves after the
2026-08 reorg. Protocol is the same: for each held-out season Y, fit the
SHIPPED feature set on the other four seasons of 2021-2025 and score Y;
pitcher score = mean per-pitch stuff, 'full' = all pitches, 'h1' = pitches
on/before the All-Star date. MIN_N 50. Keys are MLB ids resolved from
data/_era_targets.json names.
Output: data/_era_internal_stuff.json {season: {pid: {stuff_full, stuff_h1, n_full, n_h1}}}
"""
import gc, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'stuff'))
import stuff_gate_v2 as G
from pipeline.utils import _fullname_to_lastfirst
OUT = os.path.join(ROOT, 'data', '_era_internal_stuff.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
SEASONS = (2021, 2022, 2023, 2024, 2025)
MIN_N = 50

def name_map(season):
    m, ambig = {}, set()
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        full = (rec['name'] or '').strip()
        variants = {_fullname_to_lastfirst(full).lower()}
        parts = full.split()
        if len(parts) >= 3:
            variants.add((' '.join(parts[-2:]) + ', ' + ' '.join(parts[:-2])).lower())
        for lf in variants:
            if lf in m and m[lf] != int(pid):
                ambig.add(lf)
            m[lf] = int(pid)
    for lf in ambig:
        del m[lf]
    return m

def main():
    frames = {y: G.load_frame(y) for y in SEASONS}
    G.set_arm_side_sign(list(frames.values()))
    feats = G.variant_feats({})
    print('feats', feats, flush=True)
    result = {}
    for Y in SEASONS:
        train = [y for y in SEASONS if y != Y]
        print(f'=== held-out {Y} (train {train}) ===', flush=True)
        slopes = G.fit_vaa_slopes([frames[y] for y in train])
        P = {y: G.prepare(frames[y], slopes, 'true') for y in train + [Y]}
        Xtr = pd.concat([G.design(P[y], {}, feats) for y in train], ignore_index=True)
        ytr = np.concatenate([P[y]['target_xrv'].values for y in train])
        keep = np.isfinite(ytr)
        dY = P[Y]
        XY = G.design(dY, {}, feats)
        pred = G.fit_predict(Xtr[keep], ytr[keep], XY, {}, 0)
        dY = dY.assign(stuff=-pred.astype('float32'))
        asg = TARGETS[str(Y)]['asg']; nm = name_map(Y)
        rec, unmatched = {}, 0
        dates = dY['date'].astype(str).str[:10]
        for name, g in dY.groupby('pitcher'):
            pid = nm.get(str(name).lower())
            if pid is None:
                unmatched += 1; continue
            gh = g[dates.loc[g.index] <= asg]
            r = {}
            if len(g) >= MIN_N:
                r['stuff_full'] = float(g['stuff'].mean()); r['n_full'] = int(len(g))
            if len(gh) >= MIN_N:
                r['stuff_h1'] = float(gh['stuff'].mean()); r['n_h1'] = int(len(gh))
            if r:
                rec[str(pid)] = r
        result[str(Y)] = rec
        print(f'  {Y}: {len(rec)} pitchers, {unmatched} unmatched names, train rows {int(keep.sum())}', flush=True)
        del P, Xtr, XY, dY, pred; gc.collect()
    tmp = OUT + '.tmp'
    json.dump(result, open(tmp, 'w')); os.replace(tmp, OUT)
    print(f'wrote {OUT}')

if __name__ == '__main__':
    main()
