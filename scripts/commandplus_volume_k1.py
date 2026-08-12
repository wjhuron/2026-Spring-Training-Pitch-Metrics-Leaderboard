"""Volume-bias replication on the SHIPPED K=1 scorer, 2021-2026.

The GMM version was flattered at low volume by ~2.0-3.0 Command+ points at
400 pitches (scripts/commandplus_volume_test.py, replicated 6/6 seasons).
Two mechanisms drove it: the sample-dependent K caps, and in-sample optimism
(targets fit on the same pitches they score, with MIN_CELL dropping thin
cells).  K=1 removes the first mechanism entirely and leaves the second, so
the bias should shrink but not vanish.  This measures how much.

Scores through pipeline_commandplus.fit_targets directly, so it measures the
production code rather than a re-implementation of it.

Usage: python3 scripts/commandplus_volume_k1.py
"""
import math
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from commandplus_ladder_multiseason import SEASONS, load_season
from pipeline_commandplus import MIN_CELL, fit_targets

MIN_FULL = 300
N_GRID = [400, 600, 900, 1300]
SEEDS = 5
VOL_MIN_FULL = 1400


def score(pitches):
    cells = defaultdict(list)
    for pt, bats, cg, x, z, _par in pitches:
        cells[(pt, bats, cg)].append((x, z))
    total, n_tot = 0.0, 0
    for pts in cells.values():
        if len(pts) < MIN_CELL:
            continue
        tg = fit_targets(pts)
        for x, z in pts:
            total += min(math.hypot(x - tx, z - tz) for tx, tz in tg)
            n_tot += 1
    return (total / n_tot, n_tot) if n_tot else (None, 0)


def job(arg):
    key, pitches = arg
    full, _ = score(pitches)
    if full is None:
        return None
    out = {}
    for N in N_GRID:
        if N >= len(pitches):
            continue
        vals = []
        for s in range(SEEDS):
            rng = random.Random(hash((key, N, s)) & 0xFFFFFFFF)
            m, _n = score(rng.sample(pitches, N))
            if m is not None:
                vals.append(m)
        if vals:
            out[N] = sum(vals) / len(vals) - full
    return key, full, out


def main():
    print('VOLUME BIAS ON THE SHIPPED K=1 SCORER')
    print(f'{"season":<8}{"n":>5}{"sigma":>8}' +
          ''.join(f'{f"N={N}":>18}' for N in N_GRID))
    print(f'{"":<8}{"":>5}{"":>8}' +
          ''.join(f'{"inches / Cmd+ pts":>18}' for _ in N_GRID))
    grand = defaultdict(list)
    for y in SEASONS:
        by_p, _bb = load_season(y)
        pool_keys = [(k, v) for k, v in by_p.items() if len(v) >= MIN_FULL]
        with Pool() as p:
            pool_vals = [r[1] for r in
                         (p.map(job, [(k, v) for k, v in pool_keys], chunksize=8))
                         if r]
        mu = sum(pool_vals) / len(pool_vals)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in pool_vals) / len(pool_vals))

        vjobs = [(k, v) for k, v in by_p.items() if len(v) >= VOL_MIN_FULL]
        with Pool() as p:
            res = [r for r in p.map(job, vjobs, chunksize=2) if r]

        line = f'{y:<8}{len(res):>5}{sigma:>8.3f}'
        for N in N_GRID:
            ds = [o[N] for _k, _f, o in res if N in o]
            if ds:
                d = sum(ds) / len(ds)
                pts = -d * 10 / sigma
                grand[N].append(pts)
                line += f'{d:>+9.3f}" {pts:>+7.2f}'
            else:
                line += f'{"--":>18}'
        print(line, flush=True)
        del by_p

    print(f'\n{"mean":<8}{"":>5}{"":>8}' +
          ''.join(f'{"":>10}{sum(grand[N]) / len(grand[N]):>+8.2f}'
                  for N in N_GRID))
    print('\nPositive Cmd+ points = the thinner sample scores BETTER, i.e. the')
    print('low-volume pitcher is flattered by that many display points.')


if __name__ == '__main__':
    main()
