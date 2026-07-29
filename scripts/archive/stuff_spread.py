"""stuff_spread.py — show FF Stuff+ spread under different scale factors and
standardization choices, to pick a wider spread."""
import os, sys, pickle
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v11'))
from train_stuff_v11 import build_df, design, BASE_FEATS  # noqa

D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
pitches = [p for p in D if p.get('_source') == 'MLB']
df = build_df(pitches)
df = df[df['target_xrv'].notna()].reset_index(drop=True)
bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11', 'stuff_models_v11.pkl'), 'rb'))
model = bundle['model']
X = design(df).reindex(columns=bundle['features'], fill_value=0)
df['raw'] = -model.predict(X)

ff = df[df['pitch_type'] == 'FF']
mu, sd = ff['raw'].mean(), ff['raw'].std()
g = ff.groupby(['pitcher', 'team'])['raw'].agg(['mean', 'size']).reset_index()
g = g[g['size'] >= 50].copy()
print(f"FF: {len(ff)} pitches, {len(g)} pitchers (>=50 FF)")
print(f"pitch-level FF raw: mu={mu:.4f} sd={sd:.4f}")

def show(colname, label):
    v = g[colname]
    p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    print(f"\n=== {label} ===  range {v.min():.0f}..{v.max():.0f}  sd={v.std():.1f}")
    print("  pctiles 1/5/25/50/75/95/99: " + " ".join(f"{x:.0f}" for x in p))
    top = g.sort_values(colname, ascending=False).head(6)
    bot = g.sort_values(colname).head(4)
    print("  TOP:  " + ", ".join(f"{r.pitcher.split(',')[0]} {r[colname]:.0f}" for _, r in top.iterrows()))
    print("  BOT:  " + ", ".join(f"{r.pitcher.split(',')[0]} {r[colname]:.0f}" for _, r in bot.iterrows()))

# A) current: pitch-level std, K=10 (then averaged to pitcher)
for K in (10, 15, 20):
    g[f'pitchK{K}'] = 100 + K * (g['mean'] - mu) / sd
    show(f'pitchK{K}', f"A) pitch-level std, scale {K}/SD  (current uses 10)")

# B) pitcher-level std: standardize the per-pitcher FF means directly, K=10
gm, gs = g['mean'].mean(), g['mean'].std()
g['pitcherK10'] = 100 + 10 * (g['mean'] - gm) / gs
show('pitcherK10', "B) pitcher-level std, scale 10/SD (pitcher distribution has SD=10)")
g['pitcherK12'] = 100 + 12 * (g['mean'] - gm) / gs
show('pitcherK12', "B) pitcher-level std, scale 12/SD")
