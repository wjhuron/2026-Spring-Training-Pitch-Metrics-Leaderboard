"""stuff_shrink.py — does sample-size shrinkage matter for pitcher-level Stuff+?
Compares no-shrink vs shrink (mean pulled toward league by k pseudo-pitches)
for FF (high volume) and for low-count per-type rows (where noise bites)."""
import os, sys, pickle
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus_v11'))
from train_stuff_v11 import build_df, design  # noqa

D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
pitches = [p for p in D if p.get('_source') == 'MLB']
df = build_df(pitches); df = df[df['target_xrv'].notna()].reset_index(drop=True)
bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11', 'stuff_models_v11.pkl'), 'rb'))
X = design(df).reindex(columns=bundle['features'], fill_value=0)
df['raw'] = -bundle['model'].predict(X)

K_SCALE = 10
K_SHRINK = 100  # pseudo-pitches toward league mean

def analyze(sub, label, qual_n=50):
    g = sub.groupby(['pitcher', 'team'])['raw'].agg(mean='mean', n='size').reset_index()
    # Scale (mu, sd) comes from the QUALIFIED pool only, and is FIXED — shrinkage
    # pulls a pitcher's mean toward mu but never rescales, so real outliers keep
    # their value while small samples fall toward 100.
    qual = g[g['n'] >= qual_n]
    mu, sd = qual['mean'].mean(), qual['mean'].std()
    g['ns'] = 100 + K_SCALE * (g['mean'] - mu) / sd
    g['adj'] = (g['n'] * g['mean'] + K_SHRINK * mu) / (g['n'] + K_SHRINK)
    g['sh'] = 100 + K_SCALE * (g['adj'] - mu) / sd
    g['diff'] = (g['sh'] - g['ns']).abs()
    print(f"\n=== {label}: {len(g)} pitchers, count median={int(g['n'].median())}, "
          f"5th pctl={int(np.percentile(g['n'],5))}, min={int(g['n'].min())} ===")
    print(f"  no-shrink range {g['ns'].min():.0f}..{g['ns'].max():.0f} | "
          f"shrink range {g['sh'].min():.0f}..{g['sh'].max():.0f}")
    print(f"  |change| from shrink: mean {g['diff'].mean():.1f}, median {g['diff'].median():.1f}, max {g['diff'].max():.1f}")
    # how extreme are the LOW-count pitchers without shrink?
    lowcut = np.percentile(g['n'], 20)
    low = g[g['n'] <= lowcut]
    print(f"  low-count group (n<={int(lowcut)}, {len(low)} pitchers): "
          f"no-shrink extremes {low['ns'].min():.0f}..{low['ns'].max():.0f}, "
          f"shrink {low['sh'].min():.0f}..{low['sh'].max():.0f}")
    # biggest movers
    mov = g.sort_values('diff', ascending=False).head(4)
    print("  biggest movers (no-shrink -> shrink, n):")
    for _, r in mov.iterrows():
        print(f"    {r.pitcher.split(',')[0]:14s} {r.ns:.0f} -> {r.sh:.0f}  (n={int(r.n)})")

analyze(df[df['pitch_type'] == 'FF'], "FF (high volume)")
analyze(df[df['pitch_type'] == 'CU'], "CU (lower volume)")
# overall (all pitches per pitcher)
analyze(df, "OVERALL (all pitch types pooled per pitcher)")
