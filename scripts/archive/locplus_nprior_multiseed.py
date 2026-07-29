"""locplus_nprior_multiseed.py — multi-seed n_prior for Loc+ (overall + per group).

locplus_stabilize_pt.py measures the split-half r=0.5 crossing with a single
random shuffle (seed 17). The per-group curves are jagged (small n_pitchers at
high N), so this reuses its exact surface build (exec'd once) and repeats the
crossing measurement over 10 shuffle seeds, reporting mean / median / spread.

Usage: python3 scripts/locplus_nprior_multiseed.py
"""
import os, sys, random, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'locplus_stabilize_pt.py')

src = open(SRC).read()
prefix = src.split('# overall')[0]
ns = {'__file__': SRC, '__name__': '_locplus_stab_pt'}
exec(compile(prefix, SRC, 'exec'), ns)

P = ns['P']
crossing = ns['crossing']
SEEDS = range(10)

print("\nMULTI-SEED n_prior (10 shuffle seeds)")

res = collections.defaultdict(list)
for s in SEEDS:
    random.seed(s)
    bp = collections.defaultdict(list)
    for p in P:
        bp[(p['Pitcher'], p['Throws'])].append(p['_v3'])
    _, cross = crossing(bp, [50, 75, 100, 125, 150, 200, 250, 300, 400])
    if cross:
        res['OVERALL'].append(cross)
    for G in ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']:
        bpg = collections.defaultdict(list)
        for p in P:
            if p['_g'] == G:
                bpg[(p['Pitcher'], p['Throws'])].append(p['_v3'])
        _, cross = crossing(bpg, [25, 50, 75, 100, 125, 150, 200, 300])
        if cross:
            res[G].append(cross)

print(f"\n  {'group':8s} {'n_seeds':>7s} {'mean':>6s} {'median':>7s} {'min':>6s} {'max':>6s}")
for g in ['OVERALL', 'FF', 'SI', 'FC', 'SL', 'CU', 'CH']:
    v = res.get(g, [])
    if not v:
        print(f"  {g:8s}    none")
        continue
    print(f"  {g:8s} {len(v):7d} {statistics.mean(v):6.0f} "
          f"{statistics.median(v):7.0f} {min(v):6.0f} {max(v):6.0f}")
