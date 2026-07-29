"""stuff_v10_compare.py — apply v10's ACTUAL pickled model to 2026 pitches and
evaluate it through the same harness as the new model, for an exact head-to-head.
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v10'))
os.chdir(ROOT)
import train_stuff_v10 as v10  # noqa

LG_WOBA, WOBA_SCALE = 0.3169, 1.2393
MIN_HALF, MIN_PERIOD = 40, 50

def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5: return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0: return None
    return float(np.corrcoef(xs, ys)[0, 1])

print("loading pitches + v10 model ...", flush=True)
D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
pitches = [p for p in D if p.get('_source') == 'MLB']
bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v10', 'stuff_models_v10.pkl'), 'rb'))
import json
_meta = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
mvn = _meta.get('mvnModels', {})
print(f"  MVN models loaded for {len(mvn)} groups (for real IVBOE/HBOE)", flush=True)

print("engineering v10 features (with MVN) ...", flush=True)
df = v10.engineer_features(pitches, mvn)
print(f"  v10 df: {len(df)} rows", flush=True)

# reproduce engineer_features' filter to align rows -> recover date/PitchID
kept = []
for p in pitches:
    pitcher = p.get('Pitcher', '').strip().strip('"'); pt = p.get('Pitch Type', '')
    if not pitcher or not pt or pt not in v10.SUPPORTED_PT:
        continue
    if v10.safe_float(p.get('Velocity')) is None or v10.safe_float(p.get('IndVertBrk')) is None \
       or v10.safe_float(p.get('HorzBrk')) is None:
        continue
    kept.append(p)
assert len(kept) == len(df), f"alignment mismatch {len(kept)} vs {len(df)}"
df = df.reset_index(drop=True)
df['date'] = [p.get('Game Date') for p in kept]
df['throws'] = [p.get('Throws') for p in kept]

print("scoring with v10 ...", flush=True)
df['stuff'] = v10.score_pitches(df, bundle['models'], bundle['league_stats'], bundle['config']).values

# luck-neutral target + half/period (same as the new harness)
xw = df['xwoba']; rv = df['run_value']; isbip = df['is_bip'].astype(bool)
df['target_xrv'] = np.where(isbip & xw.notna(), (xw - LG_WOBA) / WOBA_SCALE, -rv)
order = {d: i for i, d in enumerate(sorted(df['date'].dropna().unique()))}
df['half'] = df['date'].map(order).fillna(0).astype(int) % 2
df['period'] = np.where(df['date'] < '2026-05-01', 'early', 'late')
df = df[df['target_xrv'].notna() & df['stuff'].notna()].reset_index(drop=True)

late, latew = {}, {}
for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
    e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
    if len(e) >= MIN_PERIOD and len(l) >= MIN_PERIOD:
        late[key] = l['target_xrv'].dropna().mean()
        sw = l['is_swing'].sum(); latew[key] = l['is_whiff'].sum()/sw if sw >= 20 else np.nan

a0, a1, est = [], [], {}
for key, grp in df.groupby(['pitcher', 'throws', 'pitch_type']):
    h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
    if len(h0) >= MIN_HALF and len(h1) >= MIN_HALF:
        a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    if key in late:
        est[key] = grp[grp.period == 'early'].stuff.mean()
ks = list(est)
rel = pearson(a0, a1)
pr = -pearson([est[k] for k in ks], [late[k] for k in ks])
pw = pearson([est[k] for k in ks], [latew[k] for k in ks])
print("\n" + "=" * 56)
print(f"  v10 (actual pickled model) on 2026 pitches")
print(f"  reliability = {rel:.3f}   pred_xRV = {pr:.3f}   pred_whiff = {pw:.3f}")
print(f"  (n_rel={len(a0)}, n_pred={len(ks)})")
print(f"  new (lean+vaa): reliability 0.867  pred_xRV 0.230  pred_whiff 0.340")
print("=" * 56)
