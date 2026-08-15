"""Is xBB%cmd worth a column? Test it against the free baseline: more BB% data.

The .517 baseline was ONE season of walk rate. A reader with two seasons of
BB% has a better baseline and needs no model at all. If BB%(Y) + BB%(Y-1)
already reaches what BB% + xBB%cmd reaches, the command signal is standing in
for information the site can get for nothing.
"""
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, '/Users/wallyhuron/Huronalytics/scripts')
sys.path.insert(0, '/Users/wallyhuron/Huronalytics')
from commandplus_xbb_build import PANEL, fit2, mean, pearson


def fitk(rows, k):
    """least squares with k predictors via gaussian elimination on the
    normal equations; rows = [(x1..xk, y)]."""
    n = len(rows)
    mx = [sum(r[i] for r in rows) / n for i in range(k)]
    my = sum(r[k] for r in rows) / n
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for r in rows:
        d = [r[i] - mx[i] for i in range(k)]
        dy = r[k] - my
        for i in range(k):
            b[i] += d[i] * dy
            for j in range(k):
                A[i][j] += d[i] * d[j]
    for i in range(k):
        p = max(range(i, k), key=lambda t: abs(A[t][i]))
        if abs(A[p][i]) < 1e-12:
            return None
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        for t in range(i + 1, k):
            f = A[t][i] / A[i][i]
            for j in range(i, k):
                A[t][j] -= f * A[i][j]
            b[t] -= f * b[i]
    c = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = b[i] - sum(A[i][j] * c[j] for j in range(i + 1, k))
        c[i] = s / A[i][i]
    return [my - sum(c[i] * mx[i] for i in range(k))] + c


def main():
    panel = [r for r in json.load(open(PANEL)) if r['n'] >= 300]
    by = defaultdict(dict)
    for r in panel:
        by[r['season']][(r['pitcher'], r['throws'])] = r
    seasons = sorted(by)

    for y in seasons:
        v = [r['tgt'] for r in by[y].values()]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))
        for r in by[y].values():
            r['tgtz'] = (r['tgt'] - mu) / sd

    coef = {}
    for y in seasons:
        rows = [(r['miss'], r['tgtz'], r['bb'])
                for s in seasons if s != y for r in by[s].values()]
        coef[y] = fit2(rows)

    def xbb(r):
        b0, b1, b2 = coef[r['season']]
        return b0 + b1 * r['miss'] + b2 * r['tgtz']

    # year-pairs where the pitcher ALSO has the prior season, so every model
    # is compared on identical rows
    trips = []
    for i in range(1, len(seasons) - 1):
        p, a, b = seasons[i - 1], seasons[i], seasons[i + 1]
        ks = [k for k in by[a] if k in by[b] and k in by[p]]
        trips.append((p, a, b, ks))

    MODELS = {
        'BB%(Y)': [lambda r, pr: r['bb']],
        'BB%(Y) + BB%(Y-1)': [lambda r, pr: r['bb'], lambda r, pr: pr['bb']],
        'BB%(Y) + xBB%cmd': [lambda r, pr: r['bb'], lambda r, pr: xbb(r)],
        'BB%(Y) + BB%(Y-1) + xBB%cmd': [lambda r, pr: r['bb'],
                                        lambda r, pr: pr['bb'],
                                        lambda r, pr: xbb(r)],
    }
    print(f'{"model":<30}' + ''.join(f'{f"{a%100}->{b%100}":>9}'
                                     for _p, a, b, _k in trips) + f'{"mean":>9}')
    got = {}
    for lbl, feats in MODELS.items():
        k = len(feats)
        vs = []
        for pi, a, b, ks in trips:
            train = []
            for pj, x, yy, kk in trips:
                if x == a:
                    continue
                train += [tuple(f(by[x][t], by[pj][t]) for f in feats)
                          + (by[yy][t]['bb'], ) for t in kk]
            m = fitk(train, k)
            pred = [m[0] + sum(m[i + 1] * f(by[a][t], by[pi][t])
                               for i, f in enumerate(feats)) for t in ks]
            vs.append(pearson(pred, [by[b][t]['bb'] for t in ks]))
        got[lbl] = vs
        print(f'{lbl:<30}' + ''.join(f'{v:>9.3f}' for v in vs)
              + f'{mean(vs):>9.3f}')
    n = sum(len(k) for _p, _a, _b, k in trips)
    print(f'\n  {n} pitcher-season rows with two prior seasons of walk data')
    two = got['BB%(Y) + BB%(Y-1)']
    cmd = got['BB%(Y) + xBB%cmd']
    full = got['BB%(Y) + BB%(Y-1) + xBB%cmd']
    w = sum(1 for a, b in zip(cmd, two) if a > b)
    print(f'  command model vs a second year of BB%: {mean(cmd):.3f} vs '
          f'{mean(two):.3f}, command wins {w}/{len(two)}')
    w2 = sum(1 for a, b in zip(full, two) if a > b)
    print(f'  does command add ON TOP of two years of BB%? '
          f'{mean(full):.3f} vs {mean(two):.3f}, wins {w2}/{len(two)}')


if __name__ == '__main__':
    main()
