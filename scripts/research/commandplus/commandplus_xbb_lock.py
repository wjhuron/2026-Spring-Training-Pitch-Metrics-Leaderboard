"""xBB%cmd — robust production form, locked coefficients, 2026 deliverable.

A FRAGILITY the validation exposed. The LOSO coefficients drift on 2026
(b0 -0.2398 / b2 0.00535 against ~-0.217 / ~0.0043 elsewhere), and 2026 is
exactly the season whose zone extent comes from a different source: the
pipeline's own SzTop/SzBot put the zone center at 29.0" while statcast's put
2021-2025 at 29.7-30.2". Target distance is measured FROM that center, so a
shifted center shifts the predictor, and a coefficient fit on one convention
applied under another is a silent bias.

The fix is to make the predictor scale-free: standardize target distance
WITHIN each season (z against that season's own pool) before it enters the
model. Any constant shift in the zone-center estimate then cancels. Miss
stays in absolute inches, which is stable across sources and is what makes
the stat interpretable.

This script checks the robust form costs nothing, then locks coefficients on
all six seasons and writes the 2026 values.

Usage: python3 scripts/research/commandplus/commandplus_xbb_lock.py
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from commandplus_xbb_build import PANEL, f3, fit2, mean, pearson
from commandplus_xbb_decide import loso_pred

OUT = os.path.expanduser('~/Downloads/xbbcmd_2026.csv')


def main():
    panel = [r for r in json.load(open(PANEL)) if r['n'] >= 300]
    by_season = defaultdict(dict)
    for r in panel:
        by_season[r['season']][(r['pitcher'], r['throws'])] = r
    seasons = sorted(by_season)
    pairs = list(zip(seasons, seasons[1:]))

    # season-standardized target distance
    zstat = {}
    for y in seasons:
        v = [r['tgt'] for r in by_season[y].values()]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))
        zstat[y] = (mu, sd)
        for r in by_season[y].values():
            r['tgtz'] = (r['tgt'] - mu) / sd

    print('season zone-standardization anchors (mean, SD of target distance)')
    for y in seasons:
        print(f'  {y}  mean {zstat[y][0]:.3f}"   SD {zstat[y][1]:.3f}"')

    print('\n' + '=' * 78)
    print('DOES THE ROBUST FORM COST ANYTHING?  (LOSO, next-season BB%)')
    print('=' * 78)
    ph = ''.join(f'{f"{a % 100}->{b % 100}":>8}' for a, b in pairs)
    print(f'{"model":<32}{ph}{"mean":>8}')

    def mk_xbb(field):
        coef = {}
        for y in seasons:
            rows = [(r['miss'], r[field], r['bb'])
                    for s in seasons if s != y for r in by_season[s].values()]
            coef[y] = fit2(rows)

        def f(r):
            b0, b1, b2 = coef[r['season']]
            return b0 + b1 * r['miss'] + b2 * r[field]
        return f, coef

    res = {}
    for lbl, field in (('raw target inches', 'tgt'),
                       ('season-standardized target', 'tgtz')):
        f, _c = mk_xbb(field)
        vs = loso_pred(by_season, seasons, [lambda r: r['bb'], f])
        res[lbl] = vs
        print(f'{"BB% + xBB%cmd (" + lbl + ")":<32}'[:32]
              + ''.join(f3(v) for v in vs) + f3(mean(vs)))
    a, b = res['season-standardized target'], res['raw target inches']
    print(f'\n  standardized vs raw: {mean(a):.3f} vs {mean(b):.3f} — '
          f'{"no cost, and immune to zone-source drift" if mean(a) >= mean(b) - 0.005 else "COSTS signal"}')

    # ── lock on all six seasons ──
    rows = [(r['miss'], r['tgtz'], r['bb']) for r in panel]
    b0, b1, b2 = fit2(rows)
    print('\n' + '=' * 78)
    print('LOCKED COEFFICIENTS (all six seasons)')
    print('=' * 78)
    print(f'  xBB%cmd = {b0:.5f} + {b1:.5f} * miss_inches '
          f'+ {b2:.5f} * z(target_distance)')
    print(f'  fit on {len(rows)} pitcher-seasons, 2021-2026')
    pred = [b0 + b1 * m + b2 * t for m, t, _y in rows]
    act = [y for _m, _t, y in rows]
    # NOT comparable to the .614 in commandplus_xbb_build: that figure came
    # from per-season LOSO fits on raw target inches, which track each
    # season's own level; this is one pooled fit on standardized targets.
    print(f'  in-sample r vs same-season BB% {pearson(pred, act):.3f}')

    # ── 2026 deliverable ──
    y = seasons[-1]
    out = []
    for (p, th), r in by_season[y].items():
        x = b0 + b1 * r['miss'] + b2 * r['tgtz']
        out.append({'season': y, 'pitcher': p, 'throws': th,
                    'pitches_scored': r['n'],
                    'miss_inches': round(r['miss'], 3),
                    'target_dist_inches': round(r['tgt'], 3),
                    'target_dist_z': round(r['tgtz'], 3),
                    'xbb_cmd': round(x, 5),
                    'bb_actual': round(r['bb'], 5),
                    'bb_less_xbb': round(r['bb'] - x, 5)})
    out.sort(key=lambda d: d['xbb_cmd'])
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'\nwrote {len(out)} rows -> {OUT}')
    print('\n  lowest implied walk rate (best command + aim):')
    for d in out[:5]:
        print(f"    {d['pitcher']:<24} xBB%cmd {100*d['xbb_cmd']:>5.1f}%  "
              f"actual {100*d['bb_actual']:>5.1f}%")
    print('  highest implied walk rate:')
    for d in out[-5:]:
        print(f"    {d['pitcher']:<24} xBB%cmd {100*d['xbb_cmd']:>5.1f}%  "
              f"actual {100*d['bb_actual']:>5.1f}%")


if __name__ == '__main__':
    main()
