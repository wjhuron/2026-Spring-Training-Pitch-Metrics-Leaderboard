"""derive_weights_lopo2.py — final LOPO decision table for BB+/Hitter+ weights.

Extends derive_weights_multiseason.py: caches per-pair component rows, then
evaluates composite candidates leave-one-pair-out, crossing BB+ variants
(current 85/15 con/sp vs pure-con) with Hitter+ weight vectors (current
70/15/15 vs multi-season-derived ~52/17/31 and neighbors).

Usage: python3 scripts/derive_weights_lopo2.py
"""
import os, sys, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from derive_weights_multiseason import (PAIRS, W_CON, W_SP, build_pair,
                                        zscore, pear, ridge_k)

CACHE = os.path.join(os.path.dirname(HERE), 'data', '_hitterplus_pair_rows.pkl')


def get_rows():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    out = {}
    for yn, yn1 in PAIRS:
        out[(yn, yn1)] = build_pair(yn, yn1)
        print(f"pair {yn}->{yn1}: {len(out[(yn, yn1)])} hitters", flush=True)
    pickle.dump(out, open(CACHE, 'wb'))
    return out


def main():
    pair_rows = get_rows()

    # per-pair z-scored components under both BB+ variants
    per_pair = {}
    for pk, rows in pair_rows.items():
        con = zscore([r[0] for r in rows])
        bb_mix = zscore([W_CON * r[0] + W_SP * r[1] for r in rows])
        sdz = zscore([r[2] for r in rows])
        ctz = zscore([r[3] for r in rows])
        y = [r[4] for r in rows]
        per_pair[pk] = {'con': con, 'mix': bb_mix, 'sd': sdz, 'ct': ctz, 'y': y}

    VECS = {
        'current 70/15/15': (0.70, 0.15, 0.15),
        'derived 52/17/31': (0.52, 0.17, 0.31),
        '55/15/30':         (0.55, 0.15, 0.30),
        '50/20/30':         (0.50, 0.20, 0.30),
        '60/15/25':         (0.60, 0.15, 0.25),
    }
    print("\nLOPO r of composite vs next-season wOBA (rows: Hitter+ weights; "
          "cols: BB+ variant):")
    print(f"  {'weights':18s} {'bb=85/15 con/sp':>16s} {'bb=pure con':>13s}")
    for name, w in VECS.items():
        cells = []
        for bbk in ('mix', 'con'):
            rs = []
            for hold in per_pair:
                d = per_pair[hold]
                comp = [w[0] * d[bbk][i] + w[1] * d['sd'][i] + w[2] * d['ct'][i]
                        for i in range(len(d['y']))]
                rs.append(pear(comp, d['y']))
            cells.append(sum(rs) / len(rs))
        print(f"  {name:18s} {cells[0]:16.4f} {cells[1]:13.4f}")

    # also: fully LOPO-derived weights on pure-con bb (honest number)
    rs = []
    for hold in per_pair:
        tb, ts, tc, ty = [], [], [], []
        for pk, d in per_pair.items():
            if pk == hold:
                continue
            tb += d['con']; ts += d['sd']; tc += d['ct']; ty += d['y']
        b = ridge_k([tb, ts, tc], ty, 0.05)
        tot = sum(abs(x) for x in b)
        w = [abs(x) / tot for x in b]
        d = per_pair[hold]
        comp = [w[0] * d['con'][i] + w[1] * d['sd'][i] + w[2] * d['ct'][i]
                for i in range(len(d['y']))]
        rs.append(pear(comp, d['y']))
    print(f"\n  LOPO-derived on pure-con bb: mean r {sum(rs)/len(rs):+.4f}   "
          + ' '.join(f"{r:+.3f}" for r in rs))


if __name__ == '__main__':
    main()
