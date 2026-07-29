#!/usr/bin/env python3
"""Preview: pitch-type-AGNOSTIC Stuff+ anchors vs the current per-type anchors.

Re-anchors the existing v11 output (stuff_plus_v11/pitcher_stuff_v11.csv —
rawmean/n per pitcher×team×pitch_type unit) against ONE global qualified pool
instead of per-type pools, mirroring _standardize's math exactly
(QUAL_N=50, K_SHRINK=100, K_SCALE=10, clip 40–180, fixed anchors from the
qualified units' rawmean mean/SD).

Read-only: writes a comparison CSV to ~/Downloads, changes nothing on the site.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'stuff_plus_v11', 'pitcher_stuff_v11.csv')
OUT = os.path.expanduser('~/Downloads/stuffplus_agnostic_preview.csv')

K_SCALE, K_SHRINK, QUAL_N = 10, 100, 50

df = pd.read_csv(CSV)

# ── global anchors: one qualified pool across ALL pitch types ──
q = df[df['n'] >= QUAL_N]
mu_g, sd_g = float(q['rawmean'].mean()), float(q['rawmean'].std())

adj = (df['n'] * df['rawmean'] + K_SHRINK * mu_g) / (df['n'] + K_SHRINK)
df['agnostic'] = (100 + K_SCALE * (adj - mu_g) / sd_g).clip(40, 180).round(1)
df['delta'] = (df['agnostic'] - df['stuff_mean']).round(1)

# ── per-type summary (qualified units; usage-weighted league mean) ──
print(f"Global anchors: {len(q)} qualified units pooled  "
      f"(mu={mu_g:.5f}, sd={sd_g:.5f})\n")
print(f"{'Type':5} {'units':>5} {'qual':>5} | {'cur mean':>8} "
      f"{'agn mean':>8} {'agn wtd':>8} | {'agn p10':>7} {'agn p90':>7}")
for pt, sub in df.groupby('pitch_type'):
    qs = sub[sub['n'] >= QUAL_N]
    if len(qs) == 0:
        continue
    wtd = (qs['agnostic'] * qs['n']).sum() / qs['n'].sum()
    print(f"{pt:5} {len(sub):>5} {len(qs):>5} | {qs['stuff_mean'].mean():>8.1f} "
          f"{qs['agnostic'].mean():>8.1f} {wtd:>8.1f} | "
          f"{qs['agnostic'].quantile(.10):>7.1f} {qs['agnostic'].quantile(.90):>7.1f}")

# ── familiar pitchers, before/after ──
NAMES = ['Misiorowski, Jacob', 'Skenes, Paul', 'Sale, Chris', 'Wheeler, Zack',
         'deGrom, Jacob', 'Cease, Dylan', 'Miller, Mason', 'Duran, Jhoan']
print('\n── Familiar arms: per-type Stuff+  current → agnostic ──')
for name in NAMES:
    sub = df[(df['pitcher'] == name) & (df['n'] >= 20)].sort_values('n', ascending=False)
    if sub.empty:
        continue
    parts = [f"{r.pitch_type} {r.stuff_mean:.0f}→{r.agnostic:.0f}"
             for r in sub.itertuples()]
    print(f"  {name:22} {'  '.join(parts)}")

# ── biggest movers among qualified units ──
qd = df[df['n'] >= QUAL_N].copy()
print('\n── Largest shifts (qualified units) ──')
for r in qd.reindex(qd['delta'].abs().sort_values(ascending=False).index).head(8).itertuples():
    print(f"  {r.pitcher:22} {r.pitch_type:3} n={r.n:<5} "
          f"{r.stuff_mean:.0f} → {r.agnostic:.0f}  ({r.delta:+.1f})")

df.sort_values(['pitcher', 'pitch_type']).to_csv(OUT, index=False)
print(f"\nFull comparison written to {OUT}")
