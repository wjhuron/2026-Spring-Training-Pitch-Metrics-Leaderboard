"""Addendum to comps_validation.py (same objective, two follow-ups):

1. out_w was monotone to the grid edge (1.0) in arsenal-only mode on both
   2026 splits -> extend the grid ({0.5, 1, 1.5, 2, 3}) until the optimum
   is bracketed or the curve is flat.
2. The contact-quality block (xwOBAcon / Barrel% / HardHit% / EV) is LESS
   split-half reliable than BB% (r .10-.34 vs .37) -> test fingerprint
   pruning: full vs -bb vs -bb-contact at mix_w=1/3, out_w=0.5.
"""
import sys, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import ford_comps as fc
from comps_validation import (load_2026_mlb, halves, fingerprint_half, add_rv,
                              pearson, BATTERY, K_NN)

OUT_W_EXT = [0.5, 1.0, 1.5, 2.0, 3.0]
DROP_CONTACT = {'avgEVAgainst', 'hardHitPct', 'barrelPctAgainst', 'xwOBAcon'}


def knn_score(names, A_rows, B_rows, dist_fn):
    preds = defaultdict(list)
    for i, nm in enumerate(names):
        ds = [(dist_fn(i, j), j) for j in range(len(names)) if j != i]
        ds = [(d, j) for d, j in ds if d is not None]
        ds.sort()
        nn = [names[j] for _, j in ds[:K_NN]]
        if len(nn) < K_NN:
            continue
        for b in BATTERY:
            vs = [B_rows[n][b] for n in nn if B_rows[n].get(b) is not None]
            tv = B_rows[nm].get(b)
            if vs and tv is not None:
                preds[b].append((sum(vs) / len(vs), tv))
    rs = {b: pearson(*zip(*preds[b])) for b in BATTERY if preds[b]}
    ok = [v for v in rs.values() if v is not None]
    return sum(ok) / len(ok), rs


def main():
    print("Loading 2026 cache...")
    mlb, _ = load_2026_mlb()
    for mode in ('interleaved', 'temporal'):
        A, B = halves(mlb, mode)
        A_rows, B_rows = fingerprint_half(A), fingerprint_half(B)
        add_rv(A_rows, A)
        add_rv(B_rows, B)
        names = sorted(set(A_rows) & set(B_rows))
        pool = [A_rows[n] for n in names]
        N = len(names)
        print(f"\n=== 2026 {mode}: {N} pitchers ===")

        # 1: arsenal-only out_w extension
        fc.MIX_W, fc.USE_MIX_OUTCOMES = 1.0, True
        print("arsenal-only, extended out_w grid (battery mean r):")
        for w in OUT_W_EXT:
            fc.MIX_OUTCOME = (('whiff', w), ('zone', w), ('gb', w))
            mz = fc.mix_zstats(pool)
            M = [[None] * N for _ in range(N)]
            for i in range(N):
                ai = A_rows[names[i]].get('arsenal')
                for j in range(i + 1, N):
                    aj = A_rows[names[j]].get('arsenal')
                    if ai and aj:
                        M[i][j] = M[j][i] = fc.arsenal_dist(ai, aj, mz)
            mean_r, _ = knn_score(names, A_rows, B_rows, lambda i, j: M[i][j])
            print(f"   out_w={w:4.2f}   mean r={mean_r:+.3f}")

        # 2: fingerprint pruning at mix_w=1/3, out_w=0.5
        fc.MIX_OUTCOME = (('whiff', 0.5), ('zone', 0.5), ('gb', 0.5))
        mz = fc.mix_zstats(pool)
        EMD = [[None] * N for _ in range(N)]
        for i in range(N):
            ai = A_rows[names[i]].get('arsenal')
            for j in range(i + 1, N):
                aj = A_rows[names[j]].get('arsenal')
                if ai and aj:
                    EMD[i][j] = EMD[j][i] = fc.arsenal_dist(ai, aj, mz)
        print("fingerprint pruning (mix_w=1/3, out_w=0.5):")
        variants = [('full', fc.FEATS_P),
                    ('-bb', [f for f in fc.FEATS_P if f != 'bbPct']),
                    ('-bb-contact', [f for f in fc.FEATS_P
                                     if f != 'bbPct' and f not in DROP_CONTACT])]
        for lab, feats in variants:
            stats = fc.zstats(pool, feats)
            zs = {n: {f: (A_rows[n][f] - stats[f][0]) / stats[f][1]
                      for f in feats if A_rows[n].get(f) is not None}
                  for n in names}
            min_feats = max(6, int(len(feats) * 0.7))

            def dist_fn(i, j, zs=zs, feats=feats, min_feats=min_feats):
                za, zb = zs[names[i]], zs[names[j]]
                ds = [abs(za[f] - zb[f]) for f in feats if f in za and f in zb]
                if len(ds) < min_feats:
                    return None
                fd = sum(ds) / len(ds)
                md = EMD[i][j]
                return (2 / 3) * fd + (1 / 3) * md if md is not None else fd
            mean_r, rs = knn_score(names, A_rows, B_rows, dist_fn)
            print(f"   {lab:12s} mean r={mean_r:+.3f}   "
                  + ' '.join(f"{b}={rs.get(b):+.3f}" for b in BATTERY
                             if rs.get(b) is not None))


if __name__ == '__main__':
    main()
