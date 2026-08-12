"""xBB%cmd — command-implied walk rate. Panel build + LOSO validation.

THE STAT.  The suppressor result (scripts/commandplus_battery_2026_08.py
section A, replicated 6/6 same-season and 5/5 next-season) says miss distance
and target aggressiveness together explain walks better than miss alone,
because the two partially cancel in the raw correlation. This turns that into
a derived stat: the walk rate a pitcher's EXECUTION implies, independent of
what his walk rate actually was. An expected-stat in the xwOBA mold.

  xBB%cmd = b0 + b1 * miss + b2 * target_distance_from_zone_center

Deliberately NOT inside Command+. Target aggressiveness is partly approach
rather than skill — a pitcher who nibbles is making a choice, not failing to
execute — so folding it into Command+ would blend the two and break its clean
"did he do what he meant to" reading. It belongs beside it, not within it.

THE DECISIVE TEST is not fit quality. Any regression fits its own sample.
The question is whether xBB%cmd forecasts NEXT season's walk rate better than
this season's ACTUAL walk rate does — the xwOBA-beats-wOBA claim. If it does
not, this is a descriptive curiosity and should not get a column.

Coefficients are fit leave-one-season-out (the convention used for the
Pitching+ weight), so every number reported for a season comes from a model
that never saw it.

Stage 1 builds the per-pitcher-season panel and caches it, since scoring six
seasons is the expensive part and the modeling wants iteration.

VERDICT: REJECTED, do not ship (scripts/commandplus_xbb_worth.py). The
headline claim failed — xBB%cmd does not out-forecast a pitcher's own walk
rate at any sample gate. The surviving incremental claim then failed the
baseline test: a SECOND YEAR of BB%, which costs nothing and is already on
the site, beats the whole command model (.575 vs .567, command winning 2/4).
On top of two years of walk data it adds .593 vs .575, winning 3/4 — about
+0.018 correlation on a secondary metric, well under the bar for a
leaderboard column. Kept as the record of a tested-and-rejected idea; the
measurement work is what produced the K=1 and cascade changes that DID ship.

Usage: python3 scripts/commandplus_xbb_build.py [--rebuild]
"""
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commandplus_battery_2026_08 import _cells_fallback, _score
from commandplus_ladder_multiseason import SEASONS, load_season

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, 'data', '_cmd_xbb_panel.json')
MIN_SCORED = 300
MIN_PA = 100


def job(arg):
    key, pitches, zc = arg
    r = _score(_cells_fallback(pitches), zc)
    if not r or r[1] < MIN_SCORED or not r[3]:
        return None
    return key, r[0], r[1], r[3][0]


def build_panel():
    panel = []
    for y in SEASONS:
        by_p, bb_rate, zone = load_season(y)
        zc = ((zone[0] + zone[1]) / 2.0, zone[0], zone[1])
        jobs = [(k, v, zc) for k, v in by_p.items() if len(v) >= MIN_SCORED]
        with Pool() as pool:
            res = [r for r in pool.map(job, jobs, chunksize=4) if r]
        n = 0
        for key, miss, ns, tgt in res:
            if key not in bb_rate:
                continue
            panel.append({'season': y, 'pitcher': key[0], 'throws': key[1],
                          'miss': miss, 'tgt': tgt, 'n': ns,
                          'bb': bb_rate[key]})
            n += 1
        print(f'  {y}: {n} pitcher-seasons', flush=True)
        del by_p
    json.dump(panel, open(PANEL, 'w'))
    return panel


# ── plain 2-predictor least squares, pure python (ports to the pipeline) ──
def fit2(rows):
    n = len(rows)
    mx1 = sum(r[0] for r in rows) / n
    mx2 = sum(r[1] for r in rows) / n
    my = sum(r[2] for r in rows) / n
    s11 = s22 = s12 = s1y = s2y = 0.0
    for a, b, y in rows:
        da, db, dy = a - mx1, b - mx2, y - my
        s11 += da * da; s22 += db * db; s12 += da * db
        s1y += da * dy; s2y += db * dy
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return None
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    return (my - b1 * mx1 - b2 * mx2, b1, b2)


def fit1(rows):
    n = len(rows)
    mx = sum(r[0] for r in rows) / n
    my = sum(r[1] for r in rows) / n
    sxx = sum((a - mx) ** 2 for a, _y in rows)
    sxy = sum((a - mx) * (y - my) for a, y in rows)
    if sxx <= 0:
        return None
    b1 = sxy / sxx
    return (my - b1 * mx, b1)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def partial(r_xy, r_xz, r_zy):
    d = (1 - r_xz ** 2) * (1 - r_zy ** 2)
    return (r_xy - r_xz * r_zy) / math.sqrt(d) if d > 0 else None


def f3(v, w=8):
    return f'{v:>{w}.3f}' if v is not None else f'{"--":>{w}}'


def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


def main():
    if '--rebuild' in sys.argv or not os.path.exists(PANEL):
        print('building panel (scoring six seasons)...')
        panel = build_panel()
    else:
        panel = json.load(open(PANEL))
        print(f'loaded cached panel: {len(panel)} pitcher-seasons '
              f'({PANEL}); --rebuild to refresh')

    by_season = defaultdict(dict)
    for r in panel:
        by_season[r['season']][(r['pitcher'], r['throws'])] = r
    seasons = sorted(by_season)

    # ── LOSO coefficients, fit on same-season BB% ──
    print('\n' + '=' * 78)
    print('LOSO COEFFICIENTS  (xBB%cmd = b0 + b1*miss + b2*target_dist)')
    print('=' * 78)
    print(f'{"held out":<10}{"b0":>10}{"b1 (miss)":>12}{"b2 (target)":>13}'
          f'{"train n":>9}')
    coef = {}
    for y in seasons:
        rows = [(r['miss'], r['tgt'], r['bb'])
                for s in seasons if s != y for r in by_season[s].values()]
        coef[y] = fit2(rows)
        b0, b1, b2 = coef[y]
        print(f'{y:<10}{b0:>10.4f}{b1:>12.5f}{b2:>13.5f}{len(rows):>9}')
    allc = fit2([(r['miss'], r['tgt'], r['bb']) for r in panel])
    print(f'{"ALL":<10}{allc[0]:>10.4f}{allc[1]:>12.5f}{allc[2]:>13.5f}'
          f'{len(panel):>9}')
    print('\n  b1 > 0: bigger miss -> more walks.  b2 > 0: aiming farther from')
    print('  the middle -> more walks, once execution is held fixed.')

    def xbb(r, y):
        b0, b1, b2 = coef[y]
        return b0 + b1 * r['miss'] + b2 * r['tgt']

    # ── same-season fit (descriptive only) ──
    print('\n' + '=' * 78)
    print('SAME-SEASON FIT  (out-of-sample: coefficients never saw this season)')
    print('=' * 78)
    hdr = ''.join(f'{y:>8d}' for y in seasons)
    print(f'{"predictor":<26}{hdr}{"mean":>8}')
    for lbl, fn in (('r(miss, BB%)', lambda r, y: r['miss']),
                    ('r(xBB%cmd, BB%)', xbb)):
        vs = []
        for y in seasons:
            rs = list(by_season[y].values())
            vs.append(pearson([fn(r, y) for r in rs], [r['bb'] for r in rs]))
        print(f'{lbl:<26}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))

    # ── THE DECISIVE TEST ──
    print('\n' + '=' * 78)
    print('DECISIVE TEST — forecasting NEXT season\'s walk rate')
    print('=' * 78)
    pairs = [(a, b) for a, b in zip(seasons, seasons[1:])]
    ph = ''.join(f'{f"{a % 100}->{b % 100}":>8}' for a, b in pairs)
    print(f'{"predictor of BB% next":<26}{ph}{"mean":>8}{"wins":>7}')
    series = {}
    for lbl, fn in (('actual BB% this season', lambda r, y: r['bb']),
                    ('miss (Command+ alone)', lambda r, y: r['miss']),
                    ('xBB%cmd', xbb)):
        vs = []
        for a, b in pairs:
            ks = [k for k in by_season[a] if k in by_season[b]]
            vs.append(pearson([fn(by_season[a][k], a) for k in ks],
                              [by_season[b][k]['bb'] for k in ks]))
        series[lbl] = vs
        print(f'{lbl:<26}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))
    base = series['actual BB% this season']
    for lbl in ('miss (Command+ alone)', 'xBB%cmd'):
        w = sum(1 for a, b in zip(series[lbl], base) if a and b and a > b)
        print(f'  {lbl} beats actual BB% in {w}/{len(base)} year-pairs')

    print('\n  INCREMENTAL — does xBB%cmd add to what actual BB% already says?')
    print(f'{"quantity":<26}{ph}{"mean":>8}')
    inc, inc_m = [], []
    for a, b in pairs:
        ks = [k for k in by_season[a] if k in by_season[b]]
        x = [xbb(by_season[a][k], a) for k in ks]
        m = [by_season[a][k]['miss'] for k in ks]
        c = [by_season[a][k]['bb'] for k in ks]
        nxt = [by_season[b][k]['bb'] for k in ks]
        inc.append(partial(pearson(x, nxt), pearson(x, c), pearson(c, nxt)))
        inc_m.append(partial(pearson(m, nxt), pearson(m, c), pearson(c, nxt)))
    print(f'{"r(miss, next) | actual":<26}' + ''.join(f3(v) for v in inc_m)
          + f3(mean(inc_m)))
    print(f'{"r(xBB%cmd, next) | actual":<26}' + ''.join(f3(v) for v in inc)
          + f3(mean(inc)))
    w = sum(1 for a, b in zip(inc, inc_m) if a and b and a > b)
    print(f'  xBB%cmd carries more incremental signal than miss alone in '
          f'{w}/{len(inc)} year-pairs')

    # ── spread, so the display scale can be judged ──
    print('\n' + '=' * 78)
    print('DISTRIBUTION (2026, for display sanity)')
    print('=' * 78)
    y = seasons[-1]
    rs = list(by_season[y].values())
    xs = sorted(xbb(r, y) for r in rs)
    bs = sorted(r['bb'] for r in rs)
    for lbl, v in (('xBB%cmd', xs), ('actual BB%', bs)):
        print(f'  {lbl:<12} p10 {100*v[len(v)//10]:.1f}%  med '
              f'{100*v[len(v)//2]:.1f}%  p90 {100*v[9*len(v)//10]:.1f}%  '
              f'(min {100*v[0]:.1f}%, max {100*v[-1]:.1f}%)')
    over = sorted(((r['bb'] - xbb(r, y)), r['pitcher']) for r in rs)
    print('\n  biggest OVER-performers (walking fewer than execution implies):')
    for d, p in over[:5]:
        print(f'    {p:<24}{100*d:>+7.1f} pts')
    print('  biggest UNDER-performers:')
    for d, p in over[-5:]:
        print(f'    {p:<24}{100*d:>+7.1f} pts')


if __name__ == '__main__':
    main()
