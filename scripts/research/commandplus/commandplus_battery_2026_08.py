"""Command+ multi-season battery, 2026-08: the 2026-only claims, replicated.

Four findings from the 2026-only pass (scripts/research/commandplus/commandplus_mix_and_targets.py)
never faced independent replicates.  Per the standing rule that a config
winning only where it was measured is fitted to that sample, each goes
through 2021-2026 per-season replicates, never pooled.

  A. SUPPRESSOR.  Target aggressiveness (how far from the heart of the zone a
     pitcher's inferred targets sit) appeared to suppress the command->walk
     relationship: r(miss, BB%) rose .571 -> .601 controlling for it, and
     r(target, BB%)|miss was .231.  If real, Command+ understates its own
     walk signal and a two-variable read beats it.

  B. TARGET PLAUSIBILITY DIRECTION.  The 10 best Command+ arms aimed 8.25"
     from the zone center, the 10 worst 5.73" — i.e. aiming at the middle is
     NOT how a pitcher earns a high Command+.  This is the argument that the
     self-reference blind spot is not biting, so it defends the metric and
     must not rest on one season.

  C. ARSENAL-MIX REJECTION.  Mix control was rejected on 2026 (both variants
     lost on reliability and walk correlation).  A rejection on one season is
     also a one-season claim.

  D. MIN_CELL FALLBACK (new candidate, not a prior finding).  MIN_CELL=20
     silently drops thin cells, which falls hardest on secondary pitches
     (CU ~34% of pitches excluded for the thinnest arms vs FF ~11%) and is
     one of the two drivers of the documented low-volume flattery.  FALLBACK
     pools a thin (pt, hand, count) cell's pitches into a (pt, hand) residual
     bucket, then into a (pt) bucket, fitting a target on whichever level
     first clears MIN_CELL.  Tested here as a candidate change, on the same
     objectives the ladder used.

All scoring runs through pipeline_commandplus.fit_targets, so this measures
the shipped K=1 scorer.

ZONE CAVEAT: the 2026 zone comes from the pipeline's own SzTop/SzBot, earlier
seasons from statcast sz_top/sz_bot.  The two differ slightly (2026 centers
at 29.1", 2022 at 29.9").  Every target-geometry measure uses its OWN
season's center, so cross-season comparisons are of correlations, never of
raw inches.

Usage: python3 scripts/research/commandplus/commandplus_battery_2026_08.py
"""
import gc
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from commandplus_ladder_multiseason import SEASONS, load_season
from pipeline.commandplus import MIN_CELL, fit_targets

MIN_FULL, MIN_HALF = 300, 150
ZONE_HALF_W = 8.5


# ═══════════════════════════════ scorers ═══════════════════════════════
def _cells_prod(pitches):
    """Production cells: (pitch type, batter hand, count group), thin cells
    DROPPED — MIN_CELL is what FALLBACK exists to challenge, so PROD must
    apply it or the comparison inverts."""
    c = defaultdict(list)
    for pt, bats, cg, x, z, _p in pitches:
        c[(pt, bats, cg)].append((pt, x, z))
    return [(pts, ) for pts in c.values() if len(pts) >= MIN_CELL]


def _cells_fallback(pitches):
    """Thin (pt, hand, count) cells cascade into (pt, hand), then (pt).
    Only pitches from thin cells enter a residual bucket, so targets are
    always fit on exactly the pitches they score."""
    lvl1 = defaultdict(list)
    for pt, bats, cg, x, z, _p in pitches:
        lvl1[(pt, bats, cg)].append((pt, x, z))
    out, res2 = [], defaultdict(list)
    for (pt, bats, _cg), pts in lvl1.items():
        if len(pts) >= MIN_CELL:
            out.append((pts, ))
        else:
            res2[(pt, bats)].extend(pts)
    res3 = defaultdict(list)
    for (pt, _bats), pts in res2.items():
        if len(pts) >= MIN_CELL:
            out.append((pts, ))
        else:
            res3[pt].extend(pts)
    for _pt, pts in res3.items():
        if len(pts) >= MIN_CELL:
            out.append((pts, ))
    return out


def _score(cells, zc=None):
    """-> (mean miss, n scored, per-type {pt: [sum, n]}, target stats)."""
    total, n_tot = 0.0, 0
    pt_acc = defaultdict(lambda: [0.0, 0])
    tg = [0.0, 0.0, 0]        # dist-from-center, share in zone, n
    for (pts, ) in cells:
        targets = fit_targets([(x, z) for _pt, x, z in pts])
        for p, x, z in pts:
            d, t = min(((math.hypot(x - tx, z - tz), (tx, tz))
                        for tx, tz in targets), key=lambda q: q[0])
            total += d
            n_tot += 1
            pt_acc[p][0] += d
            pt_acc[p][1] += 1
            if zc is not None:
                tx, tz = t
                tg[0] += math.hypot(tx, tz - zc[0])
                tg[1] += 1.0 if (abs(tx) <= ZONE_HALF_W
                                 and zc[1] <= tz <= zc[2]) else 0.0
                tg[2] += 1
    if not n_tot:
        return None
    tstat = (tg[0] / tg[2], tg[1] / tg[2]) if tg[2] else None
    return total / n_tot, n_tot, dict(pt_acc), tstat


def job(arg):
    key, pitches, zc = arg
    out = {}
    for half in ('full', 0, 1):
        sub = ([p for p in pitches if p[5] == half] if half != 'full' else pitches)
        if not sub:
            continue
        p = _score(_cells_prod(sub), zc if half == 'full' else None)
        f = _score(_cells_fallback(sub), None)
        if p:
            out[('PROD', half)] = p
        if f:
            out[('FALLBACK', half)] = f
    out['n_elig'] = len(pitches)
    return key, out


# ═══════════════════════════════ stats ═══════════════════════════════
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


def r2_two(r_my, r_ty, r_mt):
    d = 1 - r_mt ** 2
    return ((r_my ** 2 + r_ty ** 2 - 2 * r_my * r_ty * r_mt) / d) if d > 0 else None


def paired(*ds):
    ks = [k for k in ds[0] if all(k in d for d in ds[1:])]
    return [[d[k] for k in ks] for d in ds], len(ks)


def f3(v, w=8):
    return f'{v:>{w}.3f}' if v is not None else f'{"--":>{w}}'


def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


# ═══════════════════════════════ main ═══════════════════════════════
def main():
    S = {y: {} for y in SEASONS}
    bbr = {}
    for y in SEASONS:
        by_p, bb_rate, zone = load_season(y)
        zc = ((zone[0] + zone[1]) / 2.0, zone[0], zone[1])
        bbr[y] = bb_rate
        jobs = [(k, v, zc) for k, v in by_p.items() if len(v) >= MIN_FULL]
        with Pool() as pool:
            res = pool.map(job, jobs, chunksize=4)

        c_full = {k for k, o in res
                  if all((m, 'full') in o and o[(m, 'full')][1] >= MIN_FULL
                         for m in ('PROD', 'FALLBACK'))}
        c_half = {k for k, o in res
                  if all((m, h) in o and o[(m, h)][1] >= MIN_HALF
                         for m in ('PROD', 'FALLBACK') for h in (0, 1))}
        S[y] = {'res': [(k, o) for k, o in res if k in c_full or k in c_half],
                'c_full': c_full, 'c_half': c_half, 'zc': zc}
        print(f'{y}: {len(jobs)} candidates -> {len(c_full)} full / '
              f'{len(c_half)} half; zone center {zc[0]:.1f}"', flush=True)
        del by_p
        gc.collect()

    def get(y, m, half, idx=0):
        gate = MIN_FULL if half == 'full' else MIN_HALF
        keep = S[y]['c_full'] if half == 'full' else S[y]['c_half']
        return {k: o[(m, half)][idx] for k, o in S[y]['res']
                if k in keep and (m, half) in o and o[(m, half)][1] >= gate}

    def targets(y, which):
        i = 0 if which == 'dist' else 1      # _score's tstat = (dist, share in zone)
        return {k: o[('PROD', 'full')][3][i] for k, o in S[y]['res']
                if k in S[y]['c_full'] and ('PROD', 'full') in o
                and o[('PROD', 'full')][3]}

    hdr = ''.join(f'{y:>8d}' for y in SEASONS)
    pairs = list(zip(SEASONS, SEASONS[1:]))

    # ── A. suppressor ──
    print('\n' + '=' * 82)
    print('A. SUPPRESSOR — does target aggressiveness lift the command/walk signal?')
    print('=' * 82)
    print(f'{"quantity":<26}{hdr}{"mean":>8}')
    rows = {k: [] for k in ('r_my', 'r_ty', 'r_mt', 'p_my', 'p_ty', 'R2_1', 'R2_2')}
    for y in SEASONS:
        (m, t, w), n = paired(get(y, 'PROD', 'full'), targets(y, 'dist'), bbr[y])
        r_my, r_ty, r_mt = pearson(m, w), pearson(t, w), pearson(m, t)
        rows['r_my'].append(r_my); rows['r_ty'].append(r_ty); rows['r_mt'].append(r_mt)
        rows['p_my'].append(partial(r_my, r_mt, r_ty))
        rows['p_ty'].append(partial(r_ty, r_mt, r_my))
        rows['R2_1'].append(r_my ** 2)
        rows['R2_2'].append(r2_two(r_my, r_ty, r_mt))
    for lbl, key in (('r(miss, BB%)', 'r_my'), ('r(target dist, BB%)', 'r_ty'),
                     ('r(miss, target dist)', 'r_mt'),
                     ('r(miss, BB%) | target', 'p_my'),
                     ('r(target, BB%) | miss', 'p_ty'),
                     ('R2  miss only', 'R2_1'), ('R2  miss + target', 'R2_2')):
        print(f'{lbl:<26}' + ''.join(f3(v) for v in rows[key]) + f3(mean(rows[key])))
    gain = [b - a for a, b in zip(rows['R2_1'], rows['R2_2']) if a and b]
    won = sum(1 for g in gain if g > 0)
    print(f'\n  two-variable model gains {mean(gain):+.3f} R2 on average, '
          f'wins {won}/{len(gain)} seasons')

    print('\n  NEXT-SEASON (does it survive out to the forecast?)')
    print(f'{"quantity":<26}' + ''.join(f'{f"{a%100}->{b%100}":>8}' for a, b in pairs)
          + f'{"mean":>8}')
    nrows = {k: [] for k in ('r_my', 'p_ty', 'R2_1', 'R2_2')}
    for a, b in pairs:
        (m, t, w), n = paired(get(a, 'PROD', 'full'), targets(a, 'dist'), bbr[b])
        r_my, r_ty, r_mt = pearson(m, w), pearson(t, w), pearson(m, t)
        nrows['r_my'].append(r_my)
        nrows['p_ty'].append(partial(r_ty, r_mt, r_my))
        nrows['R2_1'].append(r_my ** 2)
        nrows['R2_2'].append(r2_two(r_my, r_ty, r_mt))
    for lbl, key in (('r(miss, BB% next)', 'r_my'),
                     ('r(target, BB% next) | miss', 'p_ty'),
                     ('R2  miss only', 'R2_1'), ('R2  miss + target', 'R2_2')):
        print(f'{lbl:<26}' + ''.join(f3(v) for v in nrows[key]) + f3(mean(nrows[key])))

    # ── B. target plausibility direction ──
    print('\n' + '=' * 82)
    print('B. TARGET PLAUSIBILITY — do the good-command arms aim at the middle?')
    print('=' * 82)
    print(f'{"quantity":<26}{hdr}{"mean":>8}')
    for lbl, which in (('r(miss, dist from ctr)', 'dist'),
                       ('r(miss, share in zone)', 'inz')):
        vs = []
        for y in SEASONS:
            (m, t), n = paired(get(y, 'PROD', 'full'), targets(y, which))
            vs.append(pearson(m, t))
        print(f'{lbl:<26}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))
    print('\n  best-vs-worst decile, mean target distance from zone center:')
    print(f'{"":<26}{hdr}')
    bl, wl = [], []
    for y in SEASONS:
        (m, t), n = paired(get(y, 'PROD', 'full'), targets(y, 'dist'))
        order = sorted(zip(m, t))
        k = max(1, len(order) // 10)
        bl.append(sum(t for _m, t in order[:k]) / k)       # smallest miss = best
        wl.append(sum(t for _m, t in order[-k:]) / k)
    print(f'{"  best decile":<26}' + ''.join(f'{v:>8.2f}' for v in bl))
    print(f'{"  worst decile":<26}' + ''.join(f'{v:>8.2f}' for v in wl))
    print(f'{"  gap (best - worst)":<26}' + ''.join(f'{b - w:>+8.2f}'
                                                    for b, w in zip(bl, wl)))
    print('  positive gap = better-command pitchers aim FARTHER from the middle')

    # ── C. arsenal mix ──
    print('\n' + '=' * 82)
    print('C. ARSENAL MIX — do the mix-controlled variants beat production?')
    print('=' * 82)

    def mixvals(y, half):
        gate = MIN_FULL if half == 'full' else MIN_HALF
        keep = S[y]['c_full'] if half == 'full' else S[y]['c_half']
        lg_s, lg_n = defaultdict(float), defaultdict(int)
        for k, o in S[y]['res']:
            if ('PROD', half) not in o:
                continue
            for pt, (s, n) in o[('PROD', half)][2].items():
                lg_s[pt] += s; lg_n[pt] += n
        tot = sum(lg_n.values())
        lg_miss = {pt: lg_s[pt] / lg_n[pt] for pt in lg_n}
        lg_sh = {pt: lg_n[pt] / tot for pt in lg_n}
        prod, resid, fixed = {}, {}, {}
        for k, o in S[y]['res']:
            if k not in keep or ('PROD', half) not in o:
                continue
            miss, n, acc, _t = o[('PROD', half)]
            if n < gate:
                continue
            prod[k] = miss
            sh = {pt: c / n for pt, (s, c) in acc.items()}
            resid[k] = miss - sum(sh[pt] * lg_miss[pt] for pt in sh if pt in lg_miss)
            w = {pt: lg_sh.get(pt, 0.0) for pt in sh}
            ws = sum(w.values())
            if ws > 0:
                fixed[k] = sum(w[pt] * (acc[pt][0] / acc[pt][1]) for pt in sh) / ws
        return prod, resid, fixed

    names = ('PROD', 'MIXRESID', 'MIXFIXED')
    rel = {v: [] for v in names}
    rbb = {v: [] for v in names}
    for y in SEASONS:
        h0, h1 = mixvals(y, 0), mixvals(y, 1)
        fl = mixvals(y, 'full')
        for i, v in enumerate(names):
            (a, b), n = paired(h0[i], h1[i])
            rel[v].append(pearson(a, b))
            (x, w), n2 = paired(fl[i], bbr[y])
            rbb[v].append(pearson(x, w))
    for lbl, d in (('split-half reliability', rel), ('r vs BB% (same season)', rbb)):
        print(f'\n{lbl}')
        print(f'{"variant":<12}{hdr}{"mean":>8}')
        for v in names:
            print(f'{v:<12}' + ''.join(f3(x) for x in d[v]) + f3(mean(d[v])))

    # ── D. MIN_CELL fallback ──
    print('\n' + '=' * 82)
    print('D. MIN_CELL FALLBACK — does cascading thin cells beat dropping them?')
    print('=' * 82)
    print(f'\ncoverage (share of eligible pitches scored)')
    print(f'{"variant":<12}{hdr}{"mean":>8}')
    for m in ('PROD', 'FALLBACK'):
        vs = []
        for y in SEASONS:
            cov = [o[(m, 'full')][1] / o['n_elig'] for k, o in S[y]['res']
                   if k in S[y]['c_full'] and (m, 'full') in o]
            vs.append(sum(cov) / len(cov))
        print(f'{m:<12}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))

    for lbl, fn in (
        ('split-half reliability',
         lambda y, m: pearson(*paired(get(y, m, 0), get(y, m, 1))[0])),
        ('r vs BB% (same season)',
         lambda y, m: pearson(*paired(get(y, m, 'full'), bbr[y])[0])),
    ):
        print(f'\n{lbl}')
        print(f'{"variant":<12}{hdr}{"mean":>8}')
        for m in ('PROD', 'FALLBACK'):
            vs = [fn(y, m) for y in SEASONS]
            print(f'{m:<12}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))

    print(f'\ninter-season persistence')
    print(f'{"variant":<12}' + ''.join(f'{f"{a%100}->{b%100}":>8}' for a, b in pairs)
          + f'{"mean":>8}')
    for m in ('PROD', 'FALLBACK'):
        vs = [pearson(*paired(get(a, m, 'full'), get(b, m, 'full'))[0])
              for a, b in pairs]
        print(f'{m:<12}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))

    print(f'\nr vs BB% (next season)')
    print(f'{"variant":<12}' + ''.join(f'{f"{a%100}->{b%100}":>8}' for a, b in pairs)
          + f'{"mean":>8}')
    for m in ('PROD', 'FALLBACK'):
        vs = [pearson(*paired(get(a, m, 'full'), bbr[b])[0]) for a, b in pairs]
        print(f'{m:<12}' + ''.join(f3(v) for v in vs) + f3(mean(vs)))


if __name__ == '__main__':
    # --seasons 2025,2026 restricts the run (smoke tests); default is all six.
    if '--seasons' in sys.argv:
        want = [int(x) for x in sys.argv[sys.argv.index('--seasons') + 1].split(',')]
        SEASONS[:] = [y for y in SEASONS if y in want]
    main()
