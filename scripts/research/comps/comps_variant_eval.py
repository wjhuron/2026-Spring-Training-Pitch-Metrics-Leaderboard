"""comps_variant_eval.py — fixed fingerprint variants scored on BOTH splits.

The greedy sets from comps_feature_selection.py do not transfer across
splits (interleaved-selected .250 on temporal vs the shipped fingerprint's
.283 there), so the decision comes down to a small number of fixed,
hypothesis-driven variants evaluated identically on both splits. A variant
ships only if it wins or ties on both.
"""
import os, sys, math, pickle
from array import array
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from comps_feature_selection import (agg_pitchers, agg_hitters, halves,
                                     pearson, zscale, K_NN, BATTERY_P,
                                     BATTERY_H, AAA)

# shipped fingerprints (ford_comps FEATS_P / FEATS_H, cache-computable names)
SHIP_P = ['fbVelo', 'extension', 'armAngle', 'vaa', 'haa', 'kPct', 'bbPct',
          'whiffPct', 'izWhiffPct', 'chasePct', 'izPct', 'gbPct', 'puPct',
          'avgEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'xwOBAcon']
PRUNED_P = [f for f in SHIP_P if f not in
            ('bbPct', 'avgEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'xwOBAcon')]
VARIANTS_P = [
    ('shipped-17', SHIP_P),
    ('pruned-12', PRUNED_P),
    ('pruned+grades-14', PRUNED_P + ['stuffScore', 'locPlus']),
    ('greedy-interleaved-5', ['swStrRate', 'gbPct', 'stuffScore', 'locPlus', 'ldPct']),
    ('greedy-temporal-10', ['twoStrikeWhiffPct', 'rv100', 'swStrRate', 'gbPct',
                            'fbVelo', 'kPct', 'kbbPct', 'puPct', 'fpsPct',
                            'izWhiffPct']),
]

SHIP_H = ['avgEVAll', 'hardHitPct', 'barrelPct', 'xwOBAcon', 'gbPct', 'puPct',
          'pullPct', 'airPullPct', 'swingPct', 'izSwingPct', 'chasePct',
          'whiffPct', 'izWhiffPct', 'kPct', 'bbPct', 'batSpeed', 'swingLength',
          'attackAngle']
CORE_H = ['barrelPct', 'kPct', 'bbPct', 'chasePct', 'batSpeed', 'whiffPct',
          'xwOBAcon', 'p90EV', 'swingLength', 'swingPathTilt']
VARIANTS_H = [
    ('shipped-18', SHIP_H),
    ('core-10', CORE_H),
    ('shipped+track-20', SHIP_H + ['p90EV', 'swingPathTilt']),
    ('greedy-interleaved-8', ['barrelPct', 'kPct', 'bbPct', 'batSpeed', 'rv100',
                              'xwOBAcon', 'whiffPct', 'chasePct']),
    ('greedy-temporal-11', ['barrelPct', 'izWhiffPct', 'chasePct', 'p90EV',
                            'swingLength', 'swingPathTilt', 'kPct', 'izSwingPct',
                            'avgEVAll', 'pullPct', 'batSpeed']),
]


def score_set(names, A_rows, B_rows, feats, battery):
    pool = [A_rows[n] for n in names]
    zs = zscale(pool, feats)
    feats = [f for f in feats if zs[f] is not None]
    N = len(names)
    zvals = {f: [(A_rows[n].get(f) - zs[f][0]) / zs[f][1]
                 if A_rows[n].get(f) is not None else None for n in names]
             for f in feats}
    need = max(1, int(0.7 * len(feats)))
    preds = defaultdict(list)
    for i in range(N):
        ds = []
        for j in range(N):
            if j == i:
                continue
            s = c = 0
            for f in feats:
                vi, vj = zvals[f][i], zvals[f][j]
                if vi is not None and vj is not None:
                    s += abs(vi - vj)
                    c += 1
            if c >= need:
                ds.append((s / c, j))
        ds.sort()
        nn = [names[j] for _, j in ds[:K_NN]]
        if len(nn) < K_NN:
            continue
        for b in battery:
            vs = [B_rows[n][b] for n in nn if B_rows[n].get(b) is not None]
            tv = B_rows[names[i]].get(b)
            if vs and tv is not None:
                preds[b].append((sum(vs) / len(vs), tv))
    rs = [pearson(*zip(*preds[b])) for b in battery if preds[b]]
    rs = [r for r in rs if r is not None]
    return sum(rs) / len(rs) if rs else float('nan')


def main():
    print("Loading 2026 cache (MLB only)...")
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    for role, who, tkey, aggfn, battery, variants in (
            ('pitcher', 'Pitcher', 'PTeam', agg_pitchers, BATTERY_P, VARIANTS_P),
            ('hitter', 'Batter', 'BTeam', agg_hitters, BATTERY_H, VARIANTS_H)):
        mlb = [p for p in D if p.get(who) and p.get(tkey) not in AAA]
        results = defaultdict(dict)
        for mode in ('interleaved', 'temporal'):
            A, B = halves(mlb, mode, who)
            A_rows, B_rows = aggfn(A), aggfn(B)
            names = sorted(set(A_rows) & set(B_rows))
            for lab, feats in variants:
                results[lab][mode] = score_set(names, A_rows, B_rows, feats, battery)
        print(f"\n=== {role.upper()} fixed-variant head-to-head (kNN battery mean r) ===")
        print(f"   {'variant':24s} {'interleaved':>12s} {'temporal':>10s}")
        for lab, _ in variants:
            print(f"   {lab:24s} {results[lab]['interleaved']:+12.3f} "
                  f"{results[lab]['temporal']:+10.3f}")


if __name__ == '__main__':
    main()
