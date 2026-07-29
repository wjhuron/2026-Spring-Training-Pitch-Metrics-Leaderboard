"""xwobasp_multiseason_test.py — does building the xwOBAsp SACQ zone table
(spray direction x LA bin x hand -> league wOBA) from multiple seasons help?

This is the one genuine "maybe": a nonparametric lookup whose tail cells could be
under-sampled in one season (helps with volume) but whose per-zone wOBA drifts
with the run environment (hurts, like Loc+). Build the table from 2026-only vs
2021-2026, score 2026 hitters' xwOBAsp, measure reliability (odd/even split-half)
+ predictiveness (first-half xwOBAsp vs second-half actual wOBAcon). Static
Statcast wOBA weights (same every season), so only outcome-RATE drift is tested.

Usage: python3 scripts/xwobasp_multiseason_test.py
"""
import os, sys, math, pickle, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import spray_angle, spray_direction, safe_float as sf

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LA_BINS = [(-999, -10), (-10, 0), (0, 5), (5, 10), (10, 15), (15, 20),
           (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 999)]
SACQ_MIN_BIP = 20
EV_WOBA = {'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0,
           'field_error': 0.9, 'fielders_choice': 0.9, 'fielders_choice_out': 0.9}
BATTED = {'ground_ball', 'line_drive', 'fly_ball', 'popup'}


def cf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def la_bin(la):
    for bi, (lo, hi) in enumerate(LA_BINS):
        if lo <= la < hi:
            return bi
    return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs); sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def wv_wally(event):
    if event in ('Single', 'Field Error', 'Fielders Choice'):
        return 0.9
    if event == 'Double':   return 1.25
    if event == 'Triple':   return 1.6
    if event == 'Home Run': return 2.0
    return 0.0


def hist_bip(year):
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    out = []
    for r in df.itertuples(index=False):
        if r.bb_type not in BATTED:
            continue
        hx, hy, la = cf(r.hc_x), cf(r.hc_y), cf(r.launch_angle)
        bats = r.stand
        if None in (hx, hy, la) or bats not in ('L', 'R'):
            continue
        lb = la_bin(la)
        if lb is None:
            continue
        direction = spray_direction(spray_angle(hx, hy), bats)
        if not direction:
            continue
        out.append((direction, lb, bats, EV_WOBA.get(r.events, 0.0)))
    return out


def eval_bip(p):
    """2026 BIP -> (direction, la_bin, bats, woba_val) or None."""
    if p.get('Description') != 'In Play':
        return None
    hx, hy, la = sf(p.get('HC_X')), sf(p.get('HC_Y')), sf(p.get('LaunchAngle'))
    bats = p.get('Bats')
    if None in (hx, hy, la) or bats not in ('L', 'R'):
        return None
    lb = la_bin(la)
    if lb is None:
        return None
    direction = spray_direction(spray_angle(hx, hy), bats)
    if not direction:
        return None
    return (direction, lb, bats, wv_wally(p.get('Event')))


def build_sacq(bips):
    hand = collections.defaultdict(lambda: [0.0, 0]); pooled = collections.defaultdict(lambda: [0.0, 0])
    for direction, lb, bats, wv in bips:
        hand[(direction, lb, bats)][0] += wv; hand[(direction, lb, bats)][1] += 1
        pooled[(direction, lb)][0] += wv; pooled[(direction, lb)][1] += 1

    def lookup(direction, lb, bats):
        h = hand.get((direction, lb, bats))
        if h and h[1] >= SACQ_MIN_BIP:
            return h[0] / h[1]
        pl = pooled.get((direction, lb))
        if pl and pl[1] >= SACQ_MIN_BIP:
            return pl[0] / pl[1]
        return None
    return lookup


def main():
    print('loading 2026 ...', flush=True)
    P = [p for p in pickle.load(open(PKL, 'rb')) if p.get('_source') == 'MLB']
    ev26 = []
    for p in P:
        b = eval_bip(p)
        if b is not None:
            ev26.append((p, b))
    print(f'  {len(ev26)} scorable 2026 BIP', flush=True)
    bip26 = [b for _p, b in ev26]

    byhit = collections.defaultdict(list)
    for p, b in ev26:
        byhit[p.get('Batter')].append((p.get('Game Date'), b))
    dbp = collections.defaultdict(set)
    for p, b in ev26:
        dbp[p.get('Batter')].add(p.get('Game Date'))
    half = {}
    for h, ds in dbp.items():
        for idx, dd in enumerate(sorted(ds)):
            half[(h, dd)] = idx % 2

    def xwobasp(bip_list, lookup):
        vals = [lookup(d, lb, ba) for d, lb, ba, _wv in bip_list]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def evaluate(lookup):
        h0 = collections.defaultdict(list); h1 = collections.defaultdict(list)
        first = collections.defaultdict(list); sec = collections.defaultdict(list)
        for h, recs in byhit.items():
            for gd, b in recs:
                (h0 if half[(h, gd)] == 0 else h1)[h].append(b)
                if (gd or '') < '2026-05-01':
                    first[h].append(b)
                else:
                    sec[h].append(b[3])   # actual woba_val
        a0 = {h: xwobasp(v, lookup) for h, v in h0.items() if len(v) >= 30}
        a1 = {h: xwobasp(v, lookup) for h, v in h1.items() if len(v) >= 30}
        com = [h for h in a0 if h in a1 and a0[h] is not None and a1[h] is not None]
        rel = pearson([a0[h] for h in com], [a1[h] for h in com])
        fa = {h: xwobasp(v, lookup) for h, v in first.items() if len(v) >= 50}
        sm = {h: sum(v) / len(v) for h, v in sec.items() if len(v) >= 50}
        kp = [h for h in fa if h in sm and fa[h] is not None]
        pred = pearson([fa[h] for h in kp], [sm[h] for h in kp])
        return rel, pred, len(com)

    print(f"\n{'SACQ table baseline':26s} {'BIP':>9s} {'reliab':>7s} {'pred':>7s}  (n)")
    r = evaluate(build_sacq(bip26))
    print(f"{'2026-only (current)':26s} {len(bip26):9d} {r[0]:7.3f} {r[1]:7.3f}  ({r[2]})")
    for yrs, lbl in [((2024, 2025), '+2024-25'), ((2021, 2022, 2023, 2024, 2025), '+2021-25 (all)')]:
        extra = []
        for y in yrs:
            extra += hist_bip(y)
        rr = evaluate(build_sacq(bip26 + extra))
        print(f"{lbl:26s} {len(bip26)+len(extra):9d} {rr[0]:7.3f} {rr[1]:7.3f}  ({rr[2]})")
    print("\n(the genuine maybe: under-sampled tail cells help vs run-environment drift hurts)")


if __name__ == '__main__':
    main()
