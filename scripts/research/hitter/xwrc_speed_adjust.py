"""xwrc_speed_adjust.py — does a sprint-speed term on ground balls improve
the xwOBA input that xwRC+ is built on?

Motivation: the hitter xwOBA is the mean of Savant's per-pitch
estimated_woba_using_speedangle, a function of EV and LA only. Savant's
own leaderboard xwOBA adds a seasonal sprint-speed term on topped/weak
contact, because runner speed changes infield-hit probability. Our xwRC+
therefore underrates fast hitters and overrates slow ones.

Test form (per hitter-season, public Statcast 2021-2025):
    xwOBA_adj_num = xwOBA_num + b * nGB * (speed - lgSpeed)
i.e. b wOBA points per ground ball per ft/s above the league mean.
The cache lacks launch_speed_angle, so the adjustable set is
bb_type == 'ground_ball' (the closest available proxy for topped/weak;
an LA < 10 variant is swept as a robustness arm).

Objectives, per replicate season (pool = 300+ PA events, atlas
convention; wRC+ is affine in wOBA within a season, so wOBA is the
target and slopes translate directly):
  DESC  same-season r(xwOBA_adj, wOBA)
  PRED  r(xwOBA_adj year Y, wOBA year Y+1), same hitter, both pools
Adoption bar: LOSO — choose b on the other replicates, score the held-out
one; the chosen b must beat b=0 in most held-out replicates, with an
interior optimum (or a proven-flat curve) in the sweep.

Hitters with no Savant sprint-speed row take the league mean (no
adjustment) and are counted; the miss rate is reported.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/xwrc_speed_adjust.py
Output: console + data/_xwrc_speed_results.json
"""
import json
import math
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PA = 300
SPEED = json.load(open(os.path.join(ROOT, 'data', '_sprint_speed_savant.json')))

# public statcast wOBA event weights (statcast_hitter_adapter convention)
WOBA_W = {'walk': .69, 'hit_by_pitch': .72, 'single': .89, 'double': 1.27,
          'triple': 1.61, 'home_run': 2.10}
NON_DENOM = {'intent_walk', 'sac_bunt', 'catcher_interf',
             'sac_bunt_double_play'}
NON_PA_TOKENS = ('stealing', 'pickoff', 'stolen', 'wild_pitch',
                 'passed_ball', 'truncated', 'game_advisory')
BIP_DESC = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}

# sweep grid: wOBA points per GB per ft/s vs league. Savant's own term is
# on this order (an extra infield hit per ~75 GB per ft/s ~ 0.012).
B_GRID = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020]


def season_table(year, gb_def):
    """{batter_id: dict} aggregates for one season."""
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
                                 'ngb': 0, 'pa': 0})
        r['pa'] += 1
        if e in NON_DENOM:
            continue
        r['den'] += 1
        r['wnum'] += WOBA_W.get(e, 0.0)
        is_bip = isinstance(row.description, str) \
            and row.description in BIP_DESC
        xw = row.estimated_woba_using_speedangle
        try:
            xw = float(xw)
            if xw != xw:
                xw = None
        except (TypeError, ValueError):
            xw = None
        if is_bip:
            r['xnum'] += xw if xw is not None else WOBA_W.get(e, 0.0)
            la = row.launch_angle
            try:
                la = float(la)
                if la != la:
                    la = None
            except (TypeError, ValueError):
                la = None
            if gb_def == 'bb_type':
                if row.bb_type == 'ground_ball':
                    r['ngb'] += 1
            else:                      # 'la10'
                if la is not None and la < 10.0:
                    r['ngb'] += 1
        else:
            r['xnum'] += WOBA_W.get(e, 0.0)
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


def adjusted(r, b, dspeed):
    return (r['xnum'] + b * r['ngb'] * dspeed) / r['den'] if r['den'] else None


def build(gb_def):
    tabs = {y: season_table(y, gb_def) for y in SEASONS}
    # speed deltas per season (league mean over the season's pool)
    dsp = {}
    miss = {}
    for y in SEASONS:
        sp = SPEED.get(str(y), {})
        pool = tabs[y]
        have = [sp[b] for b in pool if b in sp]
        lg = sum(have) / len(have)
        dsp[y] = {b: (sp.get(b, lg) - lg) for b in pool}
        miss[y] = 1.0 - len(have) / len(pool)
    return tabs, dsp, miss


def curves(tabs, dsp):
    """r per (objective, season/pair, b)."""
    desc = {y: {} for y in SEASONS}
    pred = {}
    for b in B_GRID:
        for y in SEASONS:
            xs, ys = [], []
            for bid, r in tabs[y].items():
                a = adjusted(r, b, dsp[y][bid])
                if a is None or r['den'] == 0:
                    continue
                xs.append(a)
                ys.append(r['wnum'] / r['den'])
            desc[y][b] = pearson(xs, ys)
        for y in SEASONS[:-1]:
            key = f'{y}->{y + 1}'
            xs, ys = [], []
            for bid, r in tabs[y].items():
                nr = tabs[y + 1].get(bid)
                if nr is None or nr['den'] == 0 or r['den'] == 0:
                    continue
                a = adjusted(r, b, dsp[y][bid])
                xs.append(a)
                ys.append(nr['wnum'] / nr['den'])
            pred.setdefault(key, {})[b] = pearson(xs, ys)
    return desc, pred


def loso_verdict(curve):
    """curve: {rep: {b: r}}. Choose b on the other reps, score held-out."""
    reps = sorted(curve)
    wins = 0
    rows = []
    for held in reps:
        others = [r for r in reps if r != held]
        score = {b: sum(curve[r][b] for r in others) for b in B_GRID}
        bstar = max(score, key=score.get)
        d = curve[held][bstar] - curve[held][0.0]
        rows.append((held, bstar, d))
        if d > 0:
            wins += 1
    return wins, rows


def main():
    out = {}
    for gb_def in ('bb_type', 'la10'):
        print(f'\n########## adjustable set: {gb_def} ##########')
        tabs, dsp, miss = build(gb_def)
        print('  speed-miss rate per season: '
              + ' '.join(f'{y}:{miss[y]:.1%}' for y in SEASONS))
        desc, pred = curves(tabs, dsp)
        for name, curve in (('DESC', desc), ('PRED', pred)):
            print(f'  === {name} ===')
            for rep in sorted(curve):
                cs = curve[rep]
                base = cs[0.0]
                bmax = max(cs, key=cs.get)
                print(f'    {rep}: r(b=0) {base:+.4f}  argmax b={bmax} '
                      f'({cs[bmax] - base:+.5f})  curve '
                      + ' '.join(f'{cs[b] - base:+.5f}' for b in B_GRID))
            wins, rows = loso_verdict(curve)
            print(f'    LOSO: chosen-b beats b=0 in {wins}/{len(rows)} '
                  'held-out replicates')
            for held, bstar, d in rows:
                print(f'      held {held}: b*={bstar}  delta {d:+.5f}')
            out[f'{gb_def}_{name}'] = {
                'curves': {str(k): {str(b): v for b, v in cs.items()}
                           for k, cs in curve.items()},
                'loso_wins': wins, 'n_reps': len(rows),
                'loso': [(h, b, d) for h, b, d in rows]}
    tmp = os.path.join(ROOT, 'data', '_xwrc_speed_results.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_xwrc_speed_results.json'))
    print('\nwrote data/_xwrc_speed_results.json')


if __name__ == '__main__':
    main()
