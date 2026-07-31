"""Catch probability calculator.

Looks up the official Statcast catch probability for an outfield play from
a locally built surface of 95k tracked plays (2024-2026, official catch_rate
from Savant's player-services/range endpoint). No network access needed.

Usage:
  python3 scripts/catch_prob.py --dist 56 --time 3.6
  python3 scripts/catch_prob.py --dist 91 --time 5.2 --wall --angle 105
  python3 scripts/catch_prob.py --dist 56 --hang 3.2          # opportunity time from hang + 0.39
  python3 scripts/catch_prob.py                                # interactive prompts

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
  --wall   ball projected to land within 8 ft of the outfield wall.
           The card's Fielding Zone is a positioning label and does not
           carry wall information; judge from hit distance and park.
  --back   set the back flag directly (otherwise inferred from --angle).
  --angle  angle to ball landing, degrees. Going back = |angle| >= 150
           (within 30 degrees of straight behind, MLB's definition).

Output is the official-style bucketed probability (0.05 steps, 0.99 cap)
plus the star rating and the local surface detail.
"""

import argparse
import json
import os

SURFACE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                            'catch_prob_surface.json')
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
        cells[(float(t), int(d), int(b), int(w))] = (v['p'], v['n'],
                                                     v['l'], v['h'])
    return cells, data['meta']


def lookup(cells, t, dist, back, wall):
    """Expanding-window weighted lookup around (t, dist) at fixed flags.

    Returns (point, n, ring, window_min, window_max) where window_min/max
    are the lowest and highest official catch probabilities among the
    comparable plays in the window."""
    for ring in range(1, MAX_RING + 1):
        tw, dw = TIME_STEP * ring, DIST_STEP * ring
        num = den = n_tot = 0.0
        mn, mx = None, None
        for (ct, cd, cb, cw), (p, n, l, h) in cells.items():
            if cb != back or cw != wall:
                continue
            if abs(ct - t) <= tw + 1e-9 and abs(cd - dist) <= dw:
                num += p * n
                den += n
                n_tot += n
                mn = l if mn is None else min(mn, l)
                mx = h if mx is None else max(mx, h)
        if den and n_tot >= K_MIN:
            return num / den, int(n_tot), ring, mn, mx
    if den:
        return num / den, int(n_tot), MAX_RING, mn, mx
    return None, 0, MAX_RING, None, None


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


def bucket(p):
    """Official display quantization: 0.05 steps, capped at 0.99."""
    b = round(p / 0.05) * 0.05
    return min(max(b, 0.0), 0.99) if b < 0.975 else 0.99


def stars(p, made_out=True):
    if p <= 0.25: return 5
    if p <= 0.50: return 4
    if p <= 0.75: return 3
    if p <= 0.90: return 2
    if p <= 0.95: return 1
    return 0


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
    # Research-portal cards TRUNCATE to one decimal (proven: Benge card
    # 3.5 vs true 3.589 vs Savant's rounded 3.6; a rounding assumption
    # gives an empty intersection with the card's own hang + plate time).
    # So card value T means true time in [T, T + 0.1]. Values entered with
    # two or more decimals are treated as exact (unrounded source).
    def exact(v):
        return v is not None and abs(v * 10 - round(v * 10)) > 1e-9
    if args.time is not None and exact(args.time):
        t_a = t_b = args.time  # unrounded time: use as-is
    elif args.time is None:
        if exact(args.hang):
            t_a = t_b = args.hang + fl
        else:
            t_a, t_b = args.hang + fl, args.hang + 0.1 + fl
        args.time = (t_a + t_b) / 2
        print(f'opportunity time = {args.hang} hang + {fl:.3f} flight '
              f'= {args.time:.2f}s')
    elif args.hang is not None:
        # intersect the two truncation intervals and take the midpoint
        card = args.time
        lo = max(card, args.hang + fl)
        hi = min(card + 0.1, args.hang + 0.1 + fl)
        if lo <= hi:
            t_a, t_b = lo, hi
            args.time = (lo + hi) / 2
            print(f'combined opportunity estimate {args.time:.3f}s '
                  f'(card [{card}, {card + 0.1:.1f}], hang+flight '
                  f'[{args.hang + fl:.2f}, {args.hang + 0.1 + fl:.2f}])')
        else:
            t_a, t_b = card, card + 0.1
            args.time = card + 0.05
    else:
        t_a, t_b = args.time, args.time + 0.1
        args.time = args.time + 0.05

    wall = 1 if args.wall else 0
    # back = --back, or angle within 30 degrees of straight behind, ONLY
    back = 1 if args.back else 0
    if not back and args.angle is not None and abs(args.angle) >= 150:
        back = 1
        print('back flag inferred from angle (|angle| >= 150)')

    cells, meta = load_surface()
    t = args.time
    if abs(t - round(t, 1)) < 0.005:
        p, n, ring, mn, mx = lookup(cells, round(t, 1), args.dist, back, wall)
    else:
        # off-grid time (from a combined estimate): interpolate the two
        # adjacent tenth-of-a-second surfaces
        import math
        t_lo = math.floor(t * 10) / 10
        t_hi = t_lo + 0.1
        p1, n1, r1, mn1, mx1 = lookup(cells, round(t_lo, 1), args.dist, back, wall)
        p2, n2, r2, mn2, mx2 = lookup(cells, round(t_hi, 1), args.dist, back, wall)
        w = (t - t_lo) / 0.1
        if p1 is not None and p2 is not None:
            p, n, ring = (1 - w) * p1 + w * p2, n1 + n2, max(r1, r2)
            mn = min(x for x in (mn1, mn2) if x is not None)
            mx = max(x for x in (mx1, mx2) if x is not None)
        else:
            p, n, ring = (p1 if p1 is not None else p2), (n1 or n2), 6
            mn = mn1 if mn1 is not None else mn2
            mx = mx1 if mx1 is not None else mx2
    if p is None:
        print('No comparable plays in the surface (inputs outside tracked range).')
        return

    # Uncertainty band: gradient component scaled by the feasible time
    # window, UNION observed catch rates over that window (bucket-correct)
    # and the lookup ring, 0.025 pad. Validated on 6,000 held-out plays
    # per input scenario: containment 99.5-99.8%, worst miss 0.03-0.08.
    import math as _m
    st, sd = gradient_spread(cells, t, args.dist, back, wall)
    U = max(t_b - t_a, 0.02)
    S = st * (U / 0.1) + sd
    lo = p - 1.5 * S / 2 - 0.025
    hi = p + 1.5 * S / 2 + 0.025
    tt0 = min(round(t_a * 10), round(t * 10) - ring + 1)
    tt1 = max(round(t_b * 10), round(t * 10) + ring - 1)
    dw = max(1, ring)
    e_lo = e_hi = None
    for ti in range(tt0, tt1 + 1):
        for di in range(int(args.dist) - dw, int(args.dist) + dw + 1):
            c = cells.get((round(ti / 10, 1), di, back, wall))
            if c:
                e_lo = c[2] if e_lo is None else min(e_lo, c[2])
                e_hi = c[3] if e_hi is None else max(e_hi, c[3])
    if e_lo is not None:
        lo = min(lo, e_lo - 0.025)
        hi = max(hi, e_hi + 0.025)
    lo_pct = max(int(_m.floor(lo * 100)), 0)
    hi_pct = min(int(_m.ceil(hi * 100)), 99)

    flags = (' BACK' if back else '') + (' WALL' if wall else '')
    print(f'\nInputs: {args.dist:.0f} ft in {args.time:.2f}s{flags or " (standard)"}')
    print(f'Catch probability: {int(p * 100 + 0.5)}%  '
          f'(plausible {lo_pct}-{hi_pct})  ({stars(p)}-star range)')
    print(f'  from {n} plays within '
          f'±{TIME_STEP * ring:.2f}s / ±{DIST_STEP * ring} ft '
          f'[surface: {meta["seasons"]}, {meta["plays"]} plays]')


if __name__ == '__main__':
    main()
