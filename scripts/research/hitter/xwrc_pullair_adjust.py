"""xwrc_pullair_adjust.py — does a pulled-air-ball term improve the xwOBA
input that xwRC+ is built on?

Public claim (FanGraphs, Paredes-class hitters): EV/LA-only xwOBA
underrates hitters who pull the ball in the air, because a pulled fly at
a given EV/LA clears a shorter fence. Test the hitter-level correction:

    xwOBA_adj_num = xwOBA_num + c * (nPullAir - lgPullShare * nAir)

i.e. c wOBA points per pulled air ball above the league-expected count
for that hitter's air-ball volume (centered, so the league mean moves
nothing). Pull = spray beyond 15 degrees to the pull side
(pipeline.utils spray_direction bins 'pull' + 'pull_side'). Two air-ball
definitions run: bb_type == 'fly_ball', and LA >= 20.

Protocol mirrors xwrc_speed_adjust.py: public Statcast 2021-2025, pool
300+ PA events, DESC = same-season r vs wOBA, PRED = next-season wOBA,
LOSO across replicates, adoption bar = held-out wins in most replicates
with an interior optimum. The per-season trend is reported explicitly:
the speed battery's gain decayed 2021 -> 2026, and any adjustment here
must be checked for the same drift before it ships.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/xwrc_pullair_adjust.py
Output: console + data/_xwrc_pullair_results.json
"""
import json
import math
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pipeline.utils import spray_angle, spray_direction

SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PA = 300

WOBA_W = {'walk': .69, 'hit_by_pitch': .72, 'single': .89, 'double': 1.27,
          'triple': 1.61, 'home_run': 2.10}
NON_DENOM = {'intent_walk', 'sac_bunt', 'catcher_interf',
             'sac_bunt_double_play'}
NON_PA_TOKENS = ('stealing', 'pickoff', 'stolen', 'wild_pitch',
                 'passed_ball', 'truncated', 'game_advisory')
BIP_DESC = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
PULL_BINS = {'pull', 'pull_side'}

# first sweep (0..0.08) was monotone to the edge in DESC every season, so
# the grid was too small; extended per the tuning rule until the optimum
# is interior or the curve proves flat.
C_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30]


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def season_table(year, air_def):
    df = pickle.load(open(os.path.join(
        ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    df = df[df['game_type'] == 'R']
    ev = df[df['events'].notna() & (df['events'] != '')]
    out = {}
    for row in ev.itertuples(index=False):
        e = row.events
        if not isinstance(e, str) or any(t in e for t in NON_PA_TOKENS):
            continue
        bid = str(int(row.batter))
        r = out.setdefault(bid, {'den': 0, 'wnum': 0.0, 'xnum': 0.0,
                                 'nair': 0, 'npull': 0, 'pa': 0})
        r['pa'] += 1
        if e in NON_DENOM:
            continue
        r['den'] += 1
        r['wnum'] += WOBA_W.get(e, 0.0)
        is_bip = isinstance(row.description, str) \
            and row.description in BIP_DESC
        if not is_bip:
            r['xnum'] += WOBA_W.get(e, 0.0)
            continue
        xw = _f(row.estimated_woba_using_speedangle)
        r['xnum'] += xw if xw is not None else WOBA_W.get(e, 0.0)
        la = _f(row.launch_angle)
        if air_def == 'fly':
            is_air = row.bb_type == 'fly_ball'
        else:                                  # 'la20'
            is_air = la is not None and la >= 20.0
        if not is_air:
            continue
        r['nair'] += 1
        ang = spray_angle(_f(row.hc_x), _f(row.hc_y))
        if spray_direction(ang, row.stand) in PULL_BINS:
            r['npull'] += 1
    return {b: r for b, r in out.items() if r['pa'] >= MIN_PA}


def pearson(xs, ys):
    n = len(xs)
    if n < 30:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sx / sy


def curves(tabs):
    lgshare = {}
    for y in SEASONS:
        na = sum(r['nair'] for r in tabs[y].values())
        np_ = sum(r['npull'] for r in tabs[y].values())
        lgshare[y] = np_ / na if na else 0.0
    desc = {y: {} for y in SEASONS}
    pred = {}

    def adj(r, c, y):
        return (r['xnum'] + c * (r['npull'] - lgshare[y] * r['nair'])) \
            / r['den'] if r['den'] else None

    for c in C_GRID:
        for y in SEASONS:
            xs, ys = [], []
            for bid, r in tabs[y].items():
                a = adj(r, c, y)
                if a is None:
                    continue
                xs.append(a)
                ys.append(r['wnum'] / r['den'])
            desc[y][c] = pearson(xs, ys)
        for y in SEASONS[:-1]:
            key = f'{y}->{y + 1}'
            xs, ys = [], []
            for bid, r in tabs[y].items():
                nr = tabs[y + 1].get(bid)
                if nr is None or nr['den'] == 0:
                    continue
                a = adj(r, c, y)
                if a is None:
                    continue
                xs.append(a)
                ys.append(nr['wnum'] / nr['den'])
            pred.setdefault(key, {})[c] = pearson(xs, ys)
    return desc, pred, lgshare


def loso_verdict(curve):
    reps = sorted(curve)
    wins, rows = 0, []
    for held in reps:
        others = [r for r in reps if r != held]
        score = {c: sum(curve[r][c] for r in others) for c in C_GRID}
        cstar = max(score, key=score.get)
        d = curve[held][cstar] - curve[held][0.0]
        rows.append((held, cstar, d))
        if d > 0:
            wins += 1
    return wins, rows


def main():
    out = {}
    for air_def in ('fly', 'la20'):
        print(f'\n########## air-ball set: {air_def} ##########')
        tabs = {y: season_table(y, air_def) for y in SEASONS}
        desc, pred, lgshare = curves(tabs)
        print('  league pulled-air share: '
              + ' '.join(f'{y}:{lgshare[y]:.3f}' for y in SEASONS))
        for name, curve in (('DESC', desc), ('PRED', pred)):
            print(f'  === {name} ===')
            for rep in sorted(curve):
                cs = curve[rep]
                base = cs[0.0]
                cmax = max(cs, key=cs.get)
                print(f'    {rep}: r(c=0) {base:+.4f}  argmax c={cmax} '
                      f'({cs[cmax] - base:+.5f})  curve '
                      + ' '.join(f'{cs[c] - base:+.5f}' for c in C_GRID))
            wins, rows = loso_verdict(curve)
            print(f'    LOSO: chosen-c beats c=0 in {wins}/{len(rows)} '
                  'held-out replicates')
            for held, cstar, d in rows:
                print(f'      held {held}: c*={cstar}  delta {d:+.5f}')
            out[f'{air_def}_{name}'] = {
                'curves': {str(k): {str(c): v for c, v in cs.items()}
                           for k, cs in curve.items()},
                'loso_wins': wins, 'n_reps': len(rows),
                'loso': [(h, c, d) for h, c, d in rows],
                'lg_share': lgshare}
    tmp = os.path.join(ROOT, 'data', '_xwrc_pullair_results.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_xwrc_pullair_results.json'))
    print('\nwrote data/_xwrc_pullair_results.json')


if __name__ == '__main__':
    main()
