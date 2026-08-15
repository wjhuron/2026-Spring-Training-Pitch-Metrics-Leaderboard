"""comps_validation.py — test the ford_comps similarity definition.

Three claims under test (2026-08-03, per Wally):
  1. RELIABILITY: shape dims are far stickier than outcome rates, and BB%
     is the least sticky fingerprint feature -> split-half r per feature.
  2. LEVEL BIAS: per-type whiff/zone/GB earned vs AAA hitters run hot vs
     MLB -> league per-type rates by level, emitted as an offsets asset
     (data/aaa_outcome_offsets.json) that ford_comps can subtract.
  3. WEIGHTS: mix_w (arsenal share), outcome_w (per-type outcome dims) and
     BB%-in-fingerprint are sweepable against a real objective:
     PREDICTIVE VALIDITY — build distances from half A of each pitcher's
     season, take the k nearest neighbors, predict the target's half-B
     outcome battery (xRv100, K-BB%, GB%, xwOBAcon, SwStr%) as the
     neighbor mean, score = mean Pearson r across the battery. Top-K
     overlap between half-A and half-B neighbor lists is reported as a
     STABILITY diagnostic only (it is gameable: weighting only ultra-
     stable dims maxes overlap while destroying outcome information).

Splits: interleaved (odd/even game dates per pitcher — same-form halves)
and temporal (first/second half of each pitcher's dates — the deployment
direction). 2026 only for the full sweep (the 2021-25 training caches
lack Event/BBType/InZone/ExitVelo); the shape-vs-whiff subset replicates
on 2024/2025 via _pitchesYYYY_training.pkl (RunExp-based rv100 + whiff).
"""
import os, sys, json, math, pickle, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import ford_comps as fc
from pipeline.utils import safe_float as sf

AAA = {'ROC', 'AAA'}
K_NN = 5                     # neighbors for the prediction (sensitivity: 3/10)
BATTERY = ['xRv100', 'kbbPct', 'gbPct', 'xwOBAcon', 'swStrPct']
MIX_W_GRID = [0.0, 0.2, 1/3, 0.5, 0.8, 1.0]
OUT_W_GRID = [0.0, 0.25, 0.5, 1.0]


def pearson(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def load_2026_mlb():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    mlb, aaa = [], []
    for p in D:
        t = p.get('PTeam')
        if not p.get('Pitcher') or not t:
            continue
        (aaa if t in AAA else mlb).append(p)
    return mlb, aaa


def halves(pitches, mode):
    """Per-pitcher split into two pitch lists. interleaved: odd/even game-date
    index; temporal: first/second half of that pitcher's game dates."""
    dates = defaultdict(set)
    for p in pitches:
        dates[p['Pitcher']].add(p.get('Game Date') or '')
    half_of = {}
    for nm, ds in dates.items():
        sd = sorted(ds)
        for i, d in enumerate(sd):
            if mode == 'interleaved':
                half_of[(nm, d)] = i % 2
            else:
                half_of[(nm, d)] = int(i >= len(sd) / 2)
    A, B = [], []
    for p in pitches:
        (A if half_of[(p['Pitcher'], p.get('Game Date') or '')] == 0 else B).append(p)
    return A, B


def fingerprint_half(pitches):
    """(name -> row) via ford_comps' aggregator, teams collapsed."""
    for p in pitches:
        p['PTeam'] = 'MLB'          # collapse stints; we own this copy
    rows = fc.window_pitchers(None, None, pitches=pitches)
    out = {}
    for r in rows:
        r['kbbPct'] = (r['kPct'] - r['bbPct']
                       if r.get('kPct') is not None and r.get('bbPct') is not None
                       else None)
        # xRv100 is not computed by the window path; rv100-style value from
        # the same aggregation is not stored either, so recompute cheaply:
        out[r['name']] = r
    return out


def add_rv(rows_by_name, pitches):
    agg = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        rv = sf(p.get('RunExp'))
        if rv is not None:
            a = agg[p['Pitcher']]
            a[0] += rv
            a[1] += 1
    for nm, (s, n) in agg.items():
        if nm in rows_by_name and n:
            rows_by_name[nm]['xRv100'] = s / n * 100   # actual RV/100 stand-in


# ── Part 1+3: reliability + sweep on one split ─────────────────────────────

def zscore_rows(rows, feats):
    stats = fc.zstats(rows, feats)
    for r in rows:
        r['_z'] = {f: (r[f] - stats[f][0]) / stats[f][1]
                   for f in feats if r.get(f) is not None}
    return stats


def sweep_split(A_rows, B_rows, label):
    names = sorted(set(A_rows) & set(B_rows))
    pool = [A_rows[n] for n in names]
    print(f"\n=== {label}: {len(names)} pitchers pass floors in both halves ===")

    # Part 1: per-feature split-half reliability
    print("\nSplit-half reliability r (fingerprint features):")
    rel = []
    for f in fc.FEATS_P + ['kbbPct']:
        xs, ys = zip(*[(A_rows[n][f], B_rows[n][f]) for n in names
                       if A_rows[n].get(f) is not None and B_rows[n].get(f) is not None]) \
                 if any(A_rows[n].get(f) is not None and B_rows[n].get(f) is not None
                        for n in names) else ([], [])
        r = pearson(list(xs), list(ys))
        rel.append((r if r is not None else float('nan'), f))
    for r, f in sorted(rel, reverse=True):
        print(f"   {f:18s} {r:+.3f}")

    feats_nb = [f for f in fc.FEATS_P if f != 'bbPct']
    stats = fc.zstats(pool, fc.FEATS_P)

    # per-pair fingerprint pieces (bb separated so bb-in/out is free)
    zs = {n: {f: (A_rows[n][f] - stats[f][0]) / stats[f][1]
              for f in fc.FEATS_P if A_rows[n].get(f) is not None}
          for n in names}
    N = len(names)
    fp_base = [[None] * N for _ in range(N)]
    fp_bb = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            za, zb = zs[names[i]], zs[names[j]]
            ds = [abs(za[f] - zb[f]) for f in feats_nb if f in za and f in zb]
            if len(ds) >= fc.MIN_FEATS - 1:
                fp_base[i][j] = fp_base[j][i] = (sum(ds), len(ds))
            if 'bbPct' in za and 'bbPct' in zb:
                fp_bb[i][j] = fp_bb[j][i] = abs(za['bbPct'] - zb['bbPct'])

    # arsenal EMD per outcome weight
    fc.MIX_W = 1.0
    fc.USE_MIX_OUTCOMES = True
    emd = {}
    for w in OUT_W_GRID:
        fc.MIX_OUTCOME = (('whiff', w), ('zone', w), ('gb', w))
        mz = fc.mix_zstats(pool)
        M = [[None] * N for _ in range(N)]
        for i in range(N):
            ai = A_rows[names[i]].get('arsenal')
            for j in range(i + 1, N):
                aj = A_rows[names[j]].get('arsenal')
                if ai and aj:
                    M[i][j] = M[j][i] = fc.arsenal_dist(ai, aj, mz)
        emd[w] = M

    def config_dist(i, j, mix_w, out_w, use_bb):
        f = fp_base[i][j]
        if use_bb and f is not None and fp_bb[i][j] is not None:
            fd = (f[0] + fp_bb[i][j]) / (f[1] + 1)
        elif f is not None:
            fd = f[0] / f[1]
        else:
            fd = None
        md = emd[out_w][i][j]
        if mix_w >= 1:
            return md
        if fd is None:
            return None
        return (1 - mix_w) * fd + mix_w * md if md is not None else fd

    print(f"\nPredictive sweep (k={K_NN} NN, battery mean r over {BATTERY}):")
    print(f"   {'mix_w':>6s} {'out_w':>6s} {'bb':>4s}   " +
          ' '.join(f'{b:>9s}' for b in BATTERY) + '     MEAN')
    results = {}
    for mix_w in MIX_W_GRID:
        for out_w in OUT_W_GRID:
            for use_bb in (True, False):
                if mix_w >= 1 and use_bb:
                    continue          # bb irrelevant at mix-only
                preds = defaultdict(list)
                for i in range(N):
                    dists = [(config_dist(i, j, mix_w, out_w, use_bb), j)
                             for j in range(N) if j != i]
                    dists = [(d, j) for d, j in dists if d is not None]
                    dists.sort()
                    nn = [names[j] for _, j in dists[:K_NN]]
                    if len(nn) < K_NN:
                        continue
                    for b in BATTERY:
                        vs = [B_rows[n][b] for n in nn if B_rows[n].get(b) is not None]
                        tv = B_rows[names[i]].get(b)
                        if vs and tv is not None:
                            preds[b].append((sum(vs) / len(vs), tv))
                rs = {b: pearson(*zip(*preds[b])) for b in BATTERY if preds[b]}
                mean_r = (sum(v for v in rs.values() if v is not None)
                          / len([v for v in rs.values() if v is not None]))
                results[(mix_w, out_w, use_bb)] = (mean_r, rs)
                print(f"   {mix_w:6.2f} {out_w:6.2f} {'in' if use_bb else 'out':>4s}   "
                      + ' '.join(f"{rs.get(b) if rs.get(b) is not None else float('nan'):+9.3f}"
                                 for b in BATTERY)
                      + f"   {mean_r:+7.3f}")
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"   BEST: mix_w={best[0][0]:.2f} out_w={best[0][1]:.2f} "
          f"bb={'in' if best[0][2] else 'out'} mean r={best[1][0]:+.3f}")
    return results


# ── Part 2: AAA vs MLB per-type outcome rates ──────────────────────────────

def level_bias(mlb, aaa):
    def rates(pitches):
        c = defaultdict(lambda: defaultdict(float))
        for p in pitches:
            pt = p.get('Pitch Type')
            if not pt or pt in ('EP', 'PO'):
                continue
            desc = p.get('Description')
            a = c[pt]
            a['n'] += 1
            if p.get('InZone') == 'Yes':
                a['iz'] += 1
            if desc in fc.SWING_DESCRIPTIONS and 'Bunt' not in (desc or ''):
                a['sw'] += 1
                if desc == 'Swinging Strike':
                    a['wh'] += 1
            bb = p.get('BBType')
            if desc == 'In Play' and bb and bb not in fc.BUNT_BB_TYPES:
                a['bip'] += 1
                if bb == 'ground_ball':
                    a['gb'] += 1
        return c

    m, a = rates(mlb), rates(aaa)
    print("\n=== AAA vs MLB league per-type outcome rates ===")
    print(f"   {'PT':3s} {'n_AAA':>6s}  {'whiff MLB/AAA':>15s}  {'zone MLB/AAA':>15s}  {'GB MLB/AAA':>15s}")
    offsets = {}
    for pt in sorted(m):
        if pt not in a or a[pt]['n'] < 400:
            continue
        def rate(c, num, den):
            return c[pt][num] / c[pt][den] if c[pt][den] else None
        wm, wa = rate(m, 'wh', 'sw'), rate(a, 'wh', 'sw')
        zm, za = rate(m, 'iz', 'n'), rate(a, 'iz', 'n')
        gm, ga = rate(m, 'gb', 'bip'), rate(a, 'gb', 'bip')
        print(f"   {pt:3s} {int(a[pt]['n']):6d}  "
              f"{wm:.3f} / {wa:.3f}    {zm:.3f} / {za:.3f}    {gm:.3f} / {ga:.3f}")
        offsets[pt] = {'whiff': round(wa - wm, 4), 'zone': round(za - zm, 4),
                       'gb': round(ga - gm, 4),
                       'nAAA': int(a[pt]['n'])}
    return offsets


# ── Part 4: 2021-25 shape-vs-whiff replicate (reduced schema) ─────────────

def replicate_season(year, mode):
    path = os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl')
    D = pickle.load(open(path, 'rb'))
    A, B = halves([p for p in D if p.get('Pitcher')], mode)

    def agg(pitches):
        c = defaultdict(lambda: defaultdict(float))
        for p in pitches:
            pt, nm = p.get('Pitch Type'), p['Pitcher']
            a = c[nm]
            a['n'] += 1
            rv = sf(p.get('RunExp'))
            if rv is not None:
                a['rv_sum'] += rv
                a['rv_n'] += 1
            desc = p.get('Description')
            if desc in fc.SWING_DESCRIPTIONS and 'Bunt' not in (desc or ''):
                a['sw'] += 1
                if desc == 'Swinging Strike':
                    a['wh'] += 1
                    a[f'pt_{pt}_wh'] += 1
                a[f'pt_{pt}_sw'] += 1
            if pt and pt not in ('EP', 'PO'):
                a[f'pt_{pt}_n'] += 1
                for fld, k in (('Velocity', 'v'), ('IndVertBrk', 'iv'),
                               ('HorzBrk', 'hb'), ('Spin Rate', 'sp')):
                    v = sf(p.get(fld))
                    if v is not None:
                        a[f'pt_{pt}_{k}_s'] += v
                        a[f'pt_{pt}_{k}_n'] += 1
        out = {}
        for nm, a in c.items():
            if a['n'] < 300 or a['sw'] < 80:
                continue
            pts = {k[3:-2] for k in a if k.startswith('pt_') and k.endswith('_n')
                   and not k.endswith(('_v_n', '_iv_n', '_hb_n', '_sp_n'))}
            tot = sum(a[f'pt_{pt}_n'] for pt in pts)
            entries = []
            for pt in pts:
                mv = {k: a[f'pt_{pt}_{k}_s'] / a[f'pt_{pt}_{k}_n']
                      if a[f'pt_{pt}_{k}_n'] else None
                      for k in ('v', 'iv', 'hb', 'sp')}
                wh = (a[f'pt_{pt}_wh'] / a[f'pt_{pt}_sw']
                      if a[f'pt_{pt}_sw'] >= fc.MIX_OUT_MIN['whiff'] else None)
                entries.append((a[f'pt_{pt}_n'] / tot if tot else 0,
                                mv['v'], mv['iv'], mv['hb'], None, mv['sp'],
                                wh, None, None))
            out[nm] = dict(name=nm, arsenal=fc._mk_arsenal(entries),
                           swStrPct=a['wh'] / a['sw'] if a['sw'] else None,
                           xRv100=a['rv_sum'] / a['rv_n'] * 100 if a['rv_n'] else None)
        return out

    A_rows, B_rows = agg(A), agg(B)
    names = sorted(n for n in set(A_rows) & set(B_rows)
                   if A_rows[n]['arsenal'] and B_rows[n]['arsenal'])
    pool = [A_rows[n] for n in names]
    N = len(names)
    print(f"\n=== {year} replicate ({mode}, shape+whiff subset): {N} pitchers ===")
    fc.MIX_W = 1.0
    fc.USE_MIX_OUTCOMES = True
    battery = ['xRv100', 'swStrPct']
    for w in OUT_W_GRID:
        fc.MIX_OUTCOME = (('whiff', w), ('zone', w), ('gb', w))
        mz = fc.mix_zstats(pool)
        preds = defaultdict(list)
        for i in range(N):
            ai = A_rows[names[i]]['arsenal']
            dists = []
            for j in range(N):
                if j == i:
                    continue
                d = fc.arsenal_dist(ai, A_rows[names[j]]['arsenal'], mz)
                if d is not None:
                    dists.append((d, j))
            dists.sort()
            nn = [names[j] for _, j in dists[:K_NN]]
            if len(nn) < K_NN:
                continue
            for b in battery:
                vs = [B_rows[n][b] for n in nn if B_rows[n].get(b) is not None]
                tv = B_rows[names[i]].get(b)
                if vs and tv is not None:
                    preds[b].append((sum(vs) / len(vs), tv))
        rs = {b: pearson(*zip(*preds[b])) for b in battery if preds[b]}
        print(f"   out_w={w:4.2f}   " +
              '  '.join(f"{b} r={rs.get(b):+.3f}" for b in battery if rs.get(b) is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-replicates', action='store_true')
    args = ap.parse_args()

    print("Loading 2026 cache...")
    mlb, aaa = load_2026_mlb()
    print(f"  {len(mlb)} MLB pitches, {len(aaa)} AAA pitches")

    offsets = level_bias(mlb, aaa)
    out_path = os.path.join(ROOT, 'data', 'aaa_outcome_offsets.json')
    with open(out_path, 'w') as f:
        json.dump({'season': 2026, 'note': 'AAA minus MLB league per-type rates '
                   '(subtract from AAA-earned rates to express on the MLB scale)',
                   'offsets': offsets}, f, indent=1)
    print(f"  offsets written: {out_path}")

    for mode in ('interleaved', 'temporal'):
        A, B = halves(mlb, mode)
        A_rows, B_rows = fingerprint_half(A), fingerprint_half(B)
        add_rv(A_rows, A)
        add_rv(B_rows, B)
        sweep_split(A_rows, B_rows, f'2026 {mode}')

    if not args.skip_replicates:
        for year in (2024, 2025):
            replicate_season(year, 'temporal')


if __name__ == '__main__':
    main()
