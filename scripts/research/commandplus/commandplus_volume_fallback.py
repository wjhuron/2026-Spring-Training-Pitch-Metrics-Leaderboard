"""Volume-bias measurement on the MIN_CELL FALLBACK candidate, 2021-2026.

The last gate before FALLBACK could ship (see scripts/research/commandplus/commandplus_battery_2026_08.py
section D, which has it winning coverage 6/6, reliability 6/6, persistence
5/5 and same-season walks 6/6).

WHY THIS IS NOT PREDICTABLE.  The shipped scorer flatters low-volume
pitchers by ~2.4 display points at 400 pitches, driven by two mechanisms:
  (1) MIN_CELL truncation — thin cells are dropped, so a low-volume pitcher
      is graded on his best-commanded pitches.  FALLBACK should SHRINK this:
      it rescues those pitches instead of discarding them.
  (2) in-sample optimism — targets are fit on the same pitches they score,
      and the fit hugs the data harder when there is less of it.  FALLBACK
      could DEEPEN this: a rescued 20-pitch pooled bucket is exactly the
      regime where a fitted mean sits closest to its own points.
The two run opposite ways, so the net is a measurement, not a deduction.
The K-caps hypothesis already failed this way once — removing them was
predicted to shrink the bias and moved it not at all.

Each scorer is normalized by its OWN pool sigma, so the numbers are display
points on each scorer's own scale and are directly comparable.

Usage: python3 scripts/research/commandplus/commandplus_volume_fallback.py
"""
import math
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from commandplus_battery_2026_08 import _cells_fallback, _cells_prod, _score
from commandplus_ladder_multiseason import SEASONS, load_season

MIN_FULL = 300
N_GRID = [400, 600, 900, 1300]
SEEDS = 5
VOL_MIN_FULL = 1400
MODES = ('PROD', 'FALLBACK')


def _one(pitches, mode):
    r = _score(_cells_prod(pitches) if mode == 'PROD' else _cells_fallback(pitches))
    return (r[0], r[1]) if r else (None, 0)


def pool_job(arg):
    key, pitches = arg
    return {m: _one(pitches, m)[0] for m in MODES}


def vol_job(arg):
    key, pitches = arg
    out = {}
    for m in MODES:
        full, _n = _one(pitches, m)
        if full is None:
            continue
        d = {}
        for N in N_GRID:
            if N >= len(pitches):
                continue
            vals = []
            for s in range(SEEDS):
                rng = random.Random(hash((key, N, s)) & 0xFFFFFFFF)
                v, _c = _one(rng.sample(pitches, N), m)
                if v is not None:
                    vals.append(v)
            if vals:
                d[N] = sum(vals) / len(vals) - full
        out[m] = d
    return out


def main():
    grand = {m: defaultdict(list) for m in MODES}
    sig_all = {m: [] for m in MODES}
    print('VOLUME BIAS: shipped PROD vs the FALLBACK candidate')
    print('(positive = the thinner sample scores BETTER, i.e. flattery, in that')
    print(" scorer's own display points)\n")
    for y in SEASONS:
        by_p, _bb, _z = load_season(y)
        with Pool() as p:
            pv = p.map(pool_job, [(k, v) for k, v in by_p.items()
                                  if len(v) >= MIN_FULL], chunksize=8)
        sig = {}
        for m in MODES:
            vals = [r[m] for r in pv if r.get(m) is not None]
            mu = sum(vals) / len(vals)
            sig[m] = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
            sig_all[m].append(sig[m])

        vjobs = [(k, v) for k, v in by_p.items() if len(v) >= VOL_MIN_FULL]
        with Pool() as p:
            res = p.map(vol_job, vjobs, chunksize=2)

        print(f'{y}  (n={len(res)})   ' +
              '   '.join(f'sigma[{m}] {sig[m]:.3f}"' for m in MODES))
        print(f'  {"scorer":<10}' + ''.join(f'{f"N={N}":>16}' for N in N_GRID))
        for m in MODES:
            line = f'  {m:<10}'
            for N in N_GRID:
                ds = [r[m][N] for r in res if m in r and N in r[m]]
                if ds:
                    d = sum(ds) / len(ds)
                    pts = -d * 10 / sig[m]
                    grand[m][N].append(pts)
                    line += f'{d:>+8.3f}" {pts:>+6.2f}'
                else:
                    line += f'{"--":>16}'
            print(line, flush=True)
        del by_p

    print('\n' + '=' * 72)
    print('SIX-SEASON MEANS (display points of low-volume flattery)')
    print('=' * 72)
    print(f'{"scorer":<12}' + ''.join(f'{f"N={N}":>10}' for N in N_GRID)
          + f'{"mean sigma":>13}')
    for m in MODES:
        print(f'{m:<12}' + ''.join(
            f'{sum(grand[m][N]) / len(grand[m][N]):>+10.2f}' for N in N_GRID)
            + f'{sum(sig_all[m]) / len(sig_all[m]):>12.3f}"')
    print(f'{"difference":<12}' + ''.join(
        f'{(sum(grand["FALLBACK"][N]) / len(grand["FALLBACK"][N])) - (sum(grand["PROD"][N]) / len(grand["PROD"][N])):>+10.2f}'
        for N in N_GRID))
    print('\nnegative difference = FALLBACK is LESS biased than the shipped scorer')


if __name__ == '__main__':
    main()
