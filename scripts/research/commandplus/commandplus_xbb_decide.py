"""xBB%cmd — what is actually worth shipping, given the decisive test failed.

scripts/research/commandplus/commandplus_xbb_build.py established:
  - xBB%cmd fits same-season walks better than miss alone (.614 vs .568)
  - it does NOT out-forecast a pitcher's own walk rate (.478 vs .517,
    winning 1/5 year-pairs), so the xwOBA-beats-wOBA framing is dead
  - it DOES carry more incremental signal beyond actual BB% than miss alone
    (.239 vs .198, winning 5/5)

Two questions decide the product:

  1. SENSITIVITY. The panel gate is loose (100 PA / 300 scored pitches) and
     BB% at 100 PA is noisy. Does the verdict hold at a real qualification?
     If xBB%cmd only loses to actual BB% because actual BB% is measured on
     thin samples, the conclusion would flip at a stricter cut.

  2. THE COMBINATION. If the honest role of xBB%cmd is "adds to actual BB%",
     then the thing worth building is the combination, not the component.
     Fit next-season BB% ~ actual BB% + xBB%cmd, LOSO, and see how much it
     beats actual BB% alone — and whether it beats actual BB% + miss, which
     is what Command+ already offers.

Reads the cached panel, so it is fast to iterate.

Usage: python3 scripts/research/commandplus/commandplus_xbb_decide.py
"""
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from commandplus_xbb_build import PANEL, f3, fit1, fit2, mean, pearson

GATES = [300, 600, 900]


def loso_pred(by_season, seasons, feats):
    """LOSO forecast of next-season BB% from `feats` (list of extractors).
    Returns per-year-pair correlation of prediction vs actual next BB%."""
    pairs = list(zip(seasons, seasons[1:]))
    out = []
    for a, b in pairs:
        train = []
        for x, y in pairs:
            if x == a:
                continue
            ks = [k for k in by_season[x] if k in by_season[y]]
            train += [tuple(f(by_season[x][k]) for f in feats)
                      + (by_season[y][k]['bb'], ) for k in ks]
        if len(feats) == 1:
            m = fit1([(t[0], t[1]) for t in train])
            pred = lambda r: m[0] + m[1] * feats[0](r)
        else:
            m = fit2(train)
            pred = lambda r: m[0] + m[1] * feats[0](r) + m[2] * feats[1](r)
        ks = [k for k in by_season[a] if k in by_season[b]]
        out.append(pearson([pred(by_season[a][k]) for k in ks],
                           [by_season[b][k]['bb'] for k in ks]))
    return out


def main():
    panel = json.load(open(PANEL))
    print(f'panel: {len(panel)} pitcher-seasons\n')

    print('=' * 78)
    print('1. SENSITIVITY — does the verdict survive a stricter sample gate?')
    print('=' * 78)
    for gate in GATES:
        sub = [r for r in panel if r['n'] >= gate]
        by_season = defaultdict(dict)
        for r in sub:
            by_season[r['season']][(r['pitcher'], r['throws'])] = r
        seasons = sorted(by_season)
        pairs = list(zip(seasons, seasons[1:]))

        # LOSO xBB%cmd coefficients on this subset
        coef = {}
        for y in seasons:
            rows = [(r['miss'], r['tgt'], r['bb'])
                    for s in seasons if s != y for r in by_season[s].values()]
            coef[y] = fit2(rows)

        def mk(y):
            b0, b1, b2 = coef[y]
            return lambda r: b0 + b1 * r['miss'] + b2 * r['tgt']

        act, xb = [], []
        for a, b in pairs:
            ks = [k for k in by_season[a] if k in by_season[b]]
            nxt = [by_season[b][k]['bb'] for k in ks]
            act.append(pearson([by_season[a][k]['bb'] for k in ks], nxt))
            f = mk(a)
            xb.append(pearson([f(by_season[a][k]) for k in ks], nxt))
        w = sum(1 for p, q in zip(xb, act) if p and q and p > q)
        n_ps = sum(len(v) for v in by_season.values())
        print(f'  n>={gate:<5} ({n_ps:>4} pitcher-seasons)   '
              f'actual BB% {mean(act):.3f}   xBB%cmd {mean(xb):.3f}   '
              f'xBB%cmd wins {w}/{len(pairs)}')
    print('\n  A stricter gate does not rescue it if the gap holds at every cut.')

    # ── 2. the combination ──
    by_season = defaultdict(dict)
    for r in panel:
        if r['n'] >= 300:
            by_season[r['season']][(r['pitcher'], r['throws'])] = r
    seasons = sorted(by_season)
    pairs = list(zip(seasons, seasons[1:]))

    coef = {}
    for y in seasons:
        rows = [(r['miss'], r['tgt'], r['bb'])
                for s in seasons if s != y for r in by_season[s].values()]
        coef[y] = fit2(rows)

    def xbb_of(r):
        b0, b1, b2 = coef[r['season']]
        return b0 + b1 * r['miss'] + b2 * r['tgt']

    print('\n' + '=' * 78)
    print('2. THE COMBINATION — forecasting next-season BB%, LOSO-fit')
    print('=' * 78)
    ph = ''.join(f'{f"{a % 100}->{b % 100}":>8}' for a, b in pairs)
    print(f'{"model":<30}{ph}{"mean":>8}')
    models = (
        ('actual BB% alone', [lambda r: r['bb']]),
        ('actual BB% + miss', [lambda r: r['bb'], lambda r: r['miss']]),
        ('actual BB% + xBB%cmd', [lambda r: r['bb'], xbb_of]),
    )
    got = {}
    for lbl, feats in models:
        vs = loso_pred(by_season, seasons, feats)
        got[lbl] = vs
        print(f'{lbl:<30}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))
    base = got['actual BB% alone']
    for lbl in ('actual BB% + miss', 'actual BB% + xBB%cmd'):
        w = sum(1 for p, q in zip(got[lbl], base) if p and q and p > q)
        print(f'  {lbl} beats BB% alone in {w}/{len(base)}')
    a = got['actual BB% + xBB%cmd']
    b = got['actual BB% + miss']
    w = sum(1 for p, q in zip(a, b) if p and q and p > q)
    print(f'  two-variable command beats Command+ alone in {w}/{len(b)} '
          f'({mean(a):.3f} vs {mean(b):.3f})')


if __name__ == '__main__':
    main()
