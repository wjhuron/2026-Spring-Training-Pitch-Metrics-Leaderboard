"""Catch probability calculator.

Looks up the official Statcast catch probability for an outfield play from
a locally built surface of 95k tracked plays (2024-2026, official catch_rate
from Savant's player-services/range endpoint). No network access needed.

Usage:
  python3 scripts/tools/catch_prob.py --dist 56 --time 3.6
  python3 scripts/tools/catch_prob.py --dist 91 --time 5.2 --wall --angle 105
  python3 scripts/tools/catch_prob.py --dist 42 --time 3.2 --hang 2.8 --plate 0.44
  python3 scripts/tools/catch_prob.py --dist 91 --time 5.2 --park CLE --hitdist 365
  python3 scripts/tools/catch_prob.py                                # interactive prompts

Inputs:
  --dist   distance needed, feet (fielder start to landing spot)
  --time   opportunity time, seconds (pitch release to landing). One number
           per play; the batter and fielder cards share it.
  --hang   hang time (contact to landing). With --time absent it rebuilds
           opportunity = hang + pitch flight. With BOTH --time and --hang
           given, the two independently rounded reads are intersected to
           narrow the timing window (best accuracy: give time, hang, and
           plate). Flight = --plate if given (exact, from the card), else
           37.6/velo if --velo (mph) given, else --flight (default 0.39s).
  --plate  plate time from the card, seconds (pitch release to plate).
  --wall   force the wall flag on (ball landing near the outfield wall).
  --park / --hitdist
           auto wall assessment: park abbrev (STL, NYM, ...) and projected
           hit distance from the hitter card. Balls landing more than
           20 ft short of the park's SHORTEST wall are confidently no-wall
           (zero leaked wall plays in a 2,809-play validation). Any closer
           and the wall status is genuinely ambiguous (even balls AT the
           marker are only ~83% official wall plays), so BOTH scenarios
           are printed: judge from the play.
  --back   set the back flag directly (otherwise inferred from --angle).
  --angle  angle to ball landing, degrees. Going back = |angle| >= 150
           (within 30 degrees of straight behind; boundary verified
           consistent with official flags, bracketed in [27, 42] degrees).
"""

import argparse
import json
import math
import os

SURFACE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                            'catch_prob_surface.json')

# published wall distances [LF line, LF gap, CF, RF gap, RF line];
# approximate where walls are irregular, may lag renovations
PARKS = {
    'ARI': [330, 374, 407, 374, 334], 'ATH': [330, 375, 403, 375, 325],
    'ATL': [335, 385, 400, 375, 325], 'BAL': [332, 373, 400, 373, 318],
    'BOS': [310, 379, 390, 380, 302], 'CHC': [355, 368, 400, 368, 353],
    'CIN': [328, 379, 404, 370, 325], 'CLE': [325, 370, 400, 375, 325],
    'COL': [347, 390, 415, 375, 350], 'CWS': [330, 375, 400, 375, 335],
    'DET': [345, 370, 412, 365, 330], 'HOU': [315, 362, 409, 373, 326],
    'KC':  [330, 387, 410, 387, 330], 'LAA': [330, 387, 396, 370, 348],
    'LAD': [330, 385, 395, 385, 330], 'MIA': [344, 386, 400, 387, 335],
    'MIL': [342, 371, 400, 374, 345], 'MIN': [339, 377, 404, 367, 328],
    'NYM': [335, 358, 405, 375, 330], 'NYY': [318, 399, 408, 385, 314],
    'PHI': [329, 374, 401, 369, 330], 'PIT': [325, 389, 399, 375, 320],
    'SD':  [336, 390, 396, 391, 322], 'SEA': [331, 378, 401, 381, 326],
    'SF':  [339, 364, 391, 415, 309], 'STL': [336, 375, 400, 375, 335],
    'TB':  [315, 370, 404, 370, 322], 'TEX': [329, 372, 407, 374, 326],
    'TOR': [328, 368, 400, 359, 328], 'WSH': [336, 377, 402, 370, 335],
}
# LOO sweep on 4000 informative 2026 plays: MAE flat at 0.0243-0.0252 for
# K in 1-10, degrades above (0.0269 at 40, 0.0305 at 80). Any K in 1-10 is
# equivalent; 5 is a convention from the middle of the flat region.
K_MIN = 5
MAX_RING = 6        # max window: +-0.30s / +-6ft
TIME_STEP = 0.05    # window half-width per ring, seconds
DIST_STEP = 1       # window half-width per ring, feet


def load_surface():
    with open(SURFACE_PATH) as f:
        data = json.load(f)
    cells = {}
    for k, v in data['cells'].items():
        t, d, b, w = k.split('|')
        cells[(float(t), int(d), int(b), int(w))] = (
            v['p'], v['n'], v['l'], v['h'],
            {int(k2): c for k2, c in v['hist'].items()})
    return cells, data['meta']


def lookup(cells, t, dist, back, wall):
    """Expanding-window weighted lookup around (t, dist) at fixed flags."""
    for ring in range(1, MAX_RING + 1):
        tw, dw = TIME_STEP * ring, DIST_STEP * ring
        num = den = n_tot = 0.0
        for (ct, cd, cb, cw), (p, n, l, h, _hist) in cells.items():
            if cb != back or cw != wall:
                continue
            if abs(ct - t) <= tw + 1e-9 and abs(cd - dist) <= dw:
                num += p * n
                den += n
                n_tot += n
        if den and n_tot >= K_MIN:
            return num / den, int(n_tot), ring
    if den:
        return num / den, int(n_tot), MAX_RING
    return None, 0, MAX_RING


def gradient_spread(cells, t, dist, back, wall):
    """Expected within-bucket spread from the local gradient (medians of
    the four neighboring cells); one-sided fallbacks, then defaults."""
    def med(tt, dd):
        c = cells.get((round(tt, 1), dd, back, wall))
        return c[0] if c else None
    t = round(t, 1)
    tp, tm = med(t + 0.1, dist), med(t - 0.1, dist)
    dp, dm = med(t, dist + 1), med(t, dist - 1)
    c0 = med(t, dist)
    if tp is not None and tm is not None: st = abs(tp - tm) / 2
    elif tp is not None and c0 is not None: st = abs(tp - c0)
    elif tm is not None and c0 is not None: st = abs(c0 - tm)
    else: st = 0.05
    if dp is not None and dm is not None: sd = abs(dp - dm) / 2
    elif dp is not None and c0 is not None: sd = abs(dp - c0)
    elif dm is not None and c0 is not None: sd = abs(c0 - dm)
    else: sd = 0.02
    return st, sd


def stars(p):
    # boundaries verified against 95k official star ratings: 5=0-25,
    # 4=26-50, 3=51-75, 2=76-90, 1=91-95, 0=routine
    if p <= 0.25: return 5
    if p <= 0.50: return 4
    if p <= 0.75: return 3
    if p <= 0.90: return 2
    if p <= 0.95: return 1
    return 0


def evaluate(cells, t, t_a, t_b, dist, back, wall):
    """Point estimate + two-tier band for one flag scenario.

    Band validated per input scenario on 6,000 held-out plays each:
    outer range 99.5-99.8% containment (worst miss 0.03-0.08), inner
    'likely' (pooled q10-q90) 96.2%. Sparse-data penalty 0.10/sqrt(n)
    on the outer range only (swept; smallest c reaching 100% sparse-
    window coverage)."""
    if abs(t - round(t, 1)) < 0.005:
        p, n, ring = lookup(cells, round(t, 1), dist, back, wall)
    else:
        t_lo = math.floor(t * 10) / 10
        t_hi = round(t_lo + 0.1, 1)
        p1, n1, r1 = lookup(cells, round(t_lo, 1), dist, back, wall)
        p2, n2, r2 = lookup(cells, round(t_hi, 1), dist, back, wall)
        w = (t - t_lo) / 0.1
        if p1 is not None and p2 is not None:
            p, n, ring = (1 - w) * p1 + w * p2, n1 + n2, max(r1, r2)
        else:
            p, n, ring = (p1 if p1 is not None else p2), (n1 or n2), 6
    if p is None:
        return None

    st, sd = gradient_spread(cells, t, dist, back, wall)
    U = max(t_b - t_a, 0.02)
    S = st * (U / 0.1) + sd
    lo = p - 1.5 * S / 2 - 0.025
    hi = p + 1.5 * S / 2 + 0.025
    tt0 = min(round(t_a * 10), round(t * 10) - ring + 1)
    tt1 = max(round(t_b * 10), round(t * 10) + ring - 1)
    dw = max(1, ring)
    for ti in range(tt0, tt1 + 1):
        for di in range(int(dist) - dw, int(dist) + dw + 1):
            c = cells.get((round(ti / 10, 1), di, back, wall))
            if c:
                lo = min(lo, c[2] - 0.025)
                hi = max(hi, c[3] + 0.025)
    pool = {}
    for ti in range(round(t_a * 10), round(t_b * 10) + 1):
        for di in (int(dist) - 1, int(dist), int(dist) + 1):
            c = cells.get((round(ti / 10, 1), di, back, wall))
            if c:
                for v2, cnt in c[4].items():
                    pool[v2] = pool.get(v2, 0) + cnt
    tot = sum(pool.values())
    pen = 0.10 / math.sqrt(max(tot, 1))
    lo = max(lo - pen, 0.0)
    hi = min(hi + pen, 1.0)
    lo_pct = max(int(math.floor(lo * 100)), 0)
    hi_pct = min(int(math.ceil(hi * 100)), 99)
    likely = None
    if tot >= 8:
        def pq(pr):
            target = pr * (tot - 1); acc = 0
            for v2 in sorted(pool):
                acc += pool[v2]
                if acc - 1 >= target: return v2
            return max(pool)
        likely = (max(pq(0.10), lo_pct), min(pq(0.90), hi_pct))
    return {'p': p, 'n': n, 'ring': ring, 'lo': lo_pct, 'hi': hi_pct,
            'likely': likely}


def report(r, label=''):
    likely_txt = (f'likely {r["likely"][0]}-{r["likely"][1]}, '
                  if r['likely'] else '')
    print(f'{label}Catch probability: {int(r["p"] * 100 + 0.5)}%  '
          f'({likely_txt}range {r["lo"]}-{r["hi"]})  '
          f'({stars(r["p"])}-star range)  [{r["n"]} comparable plays]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dist', type=float)
    ap.add_argument('--time', type=float)
    ap.add_argument('--hang', type=float)
    ap.add_argument('--flight', type=float, default=0.39)
    ap.add_argument('--velo', type=float, help='pitch velocity mph; refines hang fallback')
    ap.add_argument('--plate', type=float, help='plate time s from the card (exact flight)')
    ap.add_argument('--back', action='store_true')
    ap.add_argument('--wall', action='store_true')
    ap.add_argument('--park', type=str, help='team abbrev (STL, NYM, ...) for wall assessment')
    ap.add_argument('--hitdist', type=float, help='projected hit distance ft from the hitter card')
    ap.add_argument('--angle', type=float)
    args = ap.parse_args()

    if args.dist is None:
        args.dist = float(input('Distance needed (ft): '))
        raw = input('Opportunity time (s), or blank to give hang time: ').strip()
        if raw:
            args.time = float(raw)
        else:
            args.hang = float(input('Hang time (s): '))
        args.wall = input('Near the wall? y/N: ').strip().lower().startswith('y')

    if args.time is None and args.hang is None:
        ap.error('need --time or --hang')
    fl = args.plate if args.plate else (37.6 / args.velo if args.velo
                                        else args.flight)
    # Card display conventions (verified per field, 2026-07-31):
    #   opportunity time TRUNCATES (card T -> true in [T, T+0.1); Benge
    #     3.5 and Mesa 4.5 both read one tenth below Savant's rounded
    #     value, never above)
    #   hang time ROUNDS (H -> true in [H-0.05, H+0.05]; hang-truncation
    #     gives an empty interval on the Mesa play)
    #   plate time ROUNDS (Lee pitch: exact 0.4195 displayed as 0.42)
    # Values entered with two or more decimals are treated as exact.
    def exact(v):
        return v is not None and abs(v * 10 - round(v * 10)) > 1e-9
    if args.time is not None and exact(args.time):
        t_a = t_b = args.time  # unrounded time: use as-is
    elif args.time is None:
        if exact(args.hang):
            t_a = t_b = args.hang + fl
        else:
            t_a, t_b = args.hang - 0.05 + fl, args.hang + 0.05 + fl
        args.time = (t_a + t_b) / 2
        print(f'opportunity time = {args.hang} hang + {fl:.3f} flight '
              f'= {args.time:.2f}s')
    elif args.hang is not None:
        # intersect the two display intervals and take the midpoint
        card = args.time
        lo = max(card, args.hang - 0.05 + fl)
        hi = min(card + 0.1, args.hang + 0.05 + fl)
        if lo <= hi:
            t_a, t_b = lo, hi
            args.time = (lo + hi) / 2
            print(f'combined opportunity estimate {args.time:.3f}s '
                  f'(card [{card}, {card + 0.1:.1f}], hang+flight '
                  f'[{args.hang - 0.05 + fl:.2f}, {args.hang + 0.05 + fl:.2f}])')
        else:
            t_a, t_b = card, card + 0.1
            args.time = card + 0.05
    else:
        t_a, t_b = args.time, args.time + 0.1
        args.time = args.time + 0.05

    # back = --back, or angle within 30 degrees of straight behind, ONLY
    back = 1 if args.back else 0
    if not back and args.angle is not None and abs(args.angle) >= 150:
        back = 1
        print('back flag inferred from angle (|angle| >= 150)')

    # Wall scenarios. The flag is NOT reliably determinable from distance
    # alone (even balls AT the marker are only ~83% official wall plays;
    # wall height/shape and the 5-marker approximation interfere). Policy
    # validated on two weeks of catches: gap > 25 ft -> confidently no
    # wall (official wall rate 0.2-4%); gap <= 25 ft -> ambiguous, BOTH
    # scenarios evaluated so the correct answer is always shown.
    scenarios = [0]
    wall_note = ''
    if args.wall:
        scenarios = [1]
    elif args.park and args.hitdist is not None:
        # park's shortest wall, threshold 20: zero leaked wall plays in a
        # 2,809-play validation (sector and position variants leak or
        # classify less)
        wd = min(PARKS[args.park.upper()])
        short = wd - args.hitdist
        if short > 20:
            scenarios = [0]
            wall_note = (f'{short:.0f} ft short of the shortest wall '
                         f'({wd} ft): no wall')
        else:
            scenarios = [0, 1]
            wall_note = (f'{short:.0f} ft short of the shortest wall '
                         f'({wd} ft): wall status ambiguous, both shown; '
                         'judge from the play')

    cells, meta = load_surface()
    results = []
    for w in scenarios:
        r = evaluate(cells, args.time, t_a, t_b, args.dist, back, w)
        if r: results.append((w, r))
    if not results:
        print('No comparable plays in the surface (inputs outside tracked range).')
        return

    flags = ' BACK' if back else ''
    print(f'\nInputs: {args.dist:.0f} ft in {args.time:.2f}s{flags}')
    if wall_note:
        print(f'wall: {wall_note}')
    if len(results) == 2:
        report(results[0][1], label='  no wall: ')
        report(results[1][1], label='  if wall: ')
    else:
        w, r = results[0]
        report(r, label='  ' + ('WALL: ' if w else ''))
    print(f'  [surface: {meta["seasons"]}, {meta["plays"]} plays]')


if __name__ == '__main__':
    main()
