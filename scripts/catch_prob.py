"""Catch probability calculator.

Looks up the official Statcast catch probability for an outfield play from
a locally built surface of 95k tracked plays (2024-2026, official catch_rate
from Savant's player-services/range endpoint). No network access needed.

Usage:
  python3 scripts/catch_prob.py --dist 56 --time 3.6
  python3 scripts/catch_prob.py --dist 56 --time 3.6 --zone "standard"
  python3 scripts/catch_prob.py --dist 91 --time 5.2 --zone "RF Gap" --angle 105
  python3 scripts/catch_prob.py --dist 56 --hang 3.2          # opportunity time from hang + 0.39
  python3 scripts/catch_prob.py                                # interactive prompts

Inputs:
  --dist   distance needed, feet (fielder start to landing spot)
  --time   opportunity time, seconds (pitch release to landing). One number
           per play; the batter and fielder cards share it.
  --hang   hang time fallback (contact to landing). Used only if --time is
           absent: opportunity = hang + pitch flight. Flight is 37.6/velo
           when --velo (mph) is given (fits real pitches within ~3 ms,
           69-100 mph), else --flight (default 0.39s, typical fastball).
  --zone   fielding zone string from the card, e.g. "Standard", "RF Gap",
           "LF Line", "CF". Anything other than Standard is a wall zone
           (ball landing within 8 ft of the outfield wall) and sets the
           wall flag. Zone strings do NOT encode direction.
  --back / --wall  set the flags directly (override zone/angle).
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
        cells[(float(t), int(d), int(b), int(w))] = (v['p'], v['n'])
    return cells, data['meta']


def lookup(cells, t, dist, back, wall):
    """Expanding-window weighted lookup around (t, dist) at fixed flags."""
    for ring in range(1, MAX_RING + 1):
        tw, dw = TIME_STEP * ring, DIST_STEP * ring
        num = den = n_tot = 0.0
        for (ct, cd, cb, cw), (p, n) in cells.items():
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
    ap.add_argument('--zone', type=str, default='')
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
        args.zone = input('Fielding zone [standard]: ').strip() or 'standard'

    if args.time is None:
        if args.hang is None:
            ap.error('need --time or --hang')
        fl = 37.6 / args.velo if args.velo else args.flight
        args.time = args.hang + fl
        print(f'opportunity time = {args.hang} hang + {fl:.3f} flight '
              f'= {args.time:.2f}s')

    zone = args.zone.strip().lower()
    # "Standard", "Standard Left", "Standard Right" = no wall factor;
    # named wall zones (RF Gap, LF Line, CF, ...) = wall flag
    wall = 1 if (args.wall or (zone and not zone.startswith('standard'))) else 0
    back = 1 if args.back else 0
    if not back and args.angle is not None and abs(args.angle) >= 150:
        back = 1
        print('back flag inferred from angle (|angle| >= 150)')

    cells, meta = load_surface()
    p, n, ring = lookup(cells, round(args.time, 2), args.dist, back, wall)
    if p is None:
        print('No comparable plays in the surface (inputs outside tracked range).')
        return

    flags = (' BACK' if back else '') + (' WALL' if wall else '')
    print(f'\nInputs: {args.dist:.0f} ft in {args.time:.2f}s{flags or " (standard)"}')
    print(f'Catch probability: {p * 100:.1f}%  ({stars(p)}-star range)')
    print(f'  Savant display bucket: {bucket(p) * 100:.0f}%')
    print(f'  from {n} plays within '
          f'±{TIME_STEP * ring:.2f}s / ±{DIST_STEP * ring} ft '
          f'[surface: {meta["seasons"]}, {meta["plays"]} plays]')


if __name__ == '__main__':
    main()
