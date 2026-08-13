"""hitterplus_weight_sweep.py — is Hitter+ 52/17/31 the optimal BB+/SD+/CT+ blend?

The shipped weights came from scripts/derive_weights_multiseason.py, which
tested a handful of CANDIDATE vectors plus a LOPO-derived one. That answers
"is 52/17/31 better than the alternatives we thought of", not "is it the
optimum". This runs the full simplex.

PANEL: data/_hitterplus_pair_rows.pkl — the same rows that produced the
shipped constant. Per pair (21->22 ... 24->25), one row per hitter:
    (con_plus, sp_plus, sdPlus, ctPlus, next-season wOBA)

BB+ IS RECONSTRUCTED, NOT TAKEN FROM THE CACHE. The panel predates nothing,
but the ORIGINAL script composed BB+ as 0.85*con + 0.15*sp, and production
retired the spray term the same day (BB_PLUS_W_CON = 1.0, process_data.py).
So BB+ here is zscore(con_plus), matching what ships. Both definitions are
reported so the difference is visible rather than assumed.

PROTOCOL (identical to the derivation that set the constant):
  - components z-scored WITHIN pair, target = raw next-season wOBA
  - Hitter+ = w_bb*z(BB+) + w_sd*z(SD+) + w_ct*z(CT+)
  - objective: Pearson r vs next-season wOBA, MAXIMIZED, averaged across the
    four pairs via Fisher z

Two questions, kept separate because they are different claims:
  (A) FULL-GRID: where is the argmax, and how flat is it? Evaluated on all
      four pairs. This is optimistic for any vector fit on those pairs
      (including the shipped one), so it answers "is 52/17/31 at the top of
      the surface", not "does re-weighting help out of sample".
  (B) LOPO: derive the optimum on three pairs, score it on the fourth, and
      compare to shipped. This is the honest decision test.

Unshrunk con_plus is used, as in the original derivation. Production applies
Bayesian shrinkage (n0=60) afterward; changing that here would answer a
different question than the one that set the constant.

Usage: python3 scripts/hitterplus_weight_sweep.py
"""
import math
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, 'data', '_hitterplus_pair_rows.pkl')
SHIPPED = (0.52, 0.17, 0.31)
OLD_PRIOR = (0.70, 0.15, 0.15)
STEP = 0.01


def zscore(v):
    n = len(v)
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / n)
    return [(x - m) / s for x in v] if s > 0 else [0.0] * n


def pear(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def fisher(r):
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def unfisher(z):
    e = math.exp(2 * z)
    return (e - 1) / (e + 1)


def build(pair_rows, bb_mode):
    """-> {pair: (bb_z, sd_z, ct_z, y)}"""
    out = {}
    for pk, rows in pair_rows.items():
        if len(rows) < 30:
            continue
        if bb_mode == 'pure_con':
            bb = zscore([r[0] for r in rows])
        else:                                   # legacy 85/15
            bb = zscore([0.85 * r[0] + 0.15 * r[1] for r in rows])
        out[pk] = (bb, zscore([r[2] for r in rows]),
                   zscore([r[3] for r in rows]), [r[4] for r in rows])
    return out


def score(panel, w, pairs=None):
    """mean Fisher-z of r(blend, next wOBA) over pairs."""
    zs = []
    for pk in (pairs or panel):
        bb, sdz, ctz, y = panel[pk]
        comp = [w[0] * bb[i] + w[1] * sdz[i] + w[2] * ctz[i]
                for i in range(len(y))]
        r = pear(comp, y)
        if r is not None:
            zs.append(fisher(r))
    return sum(zs) / len(zs) if zs else None


def grid(step=STEP):
    n = int(round(1.0 / step))
    g = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a = round(i * step, 2); b = round(j * step, 2)
            g.append((a, b, round(1.0 - a - b, 2)))
    return g


def main():
    pair_rows = pickle.load(open(PANEL, 'rb'))
    tot = sum(len(v) for v in pair_rows.values())
    print(f'panel: {len(pair_rows)} pairs, {tot} hitter-pairs '
          f'({", ".join(f"{a%100}->{b%100}:{len(v)}" for (a, b), v in sorted(pair_rows.items()))})')

    for bb_mode, label in (('pure_con', 'BB+ = pure con (SHIPPED)'),
                           ('legacy', 'BB+ = 0.85 con + 0.15 spray (retired)')):
        panel = build(pair_rows, bb_mode)
        pairs = sorted(panel)
        G = grid()

        print('\n' + '=' * 88)
        print(f'{label}')
        print('=' * 88)

        # univariate, for context
        print('  univariate r vs next-season wOBA (mean over pairs):')
        for i, nm in enumerate(('BB+', 'SD+', 'CT+')):
            w = [0.0, 0.0, 0.0]; w[i] = 1.0
            print(f'    {nm:<5} {unfisher(score(panel, tuple(w))):+.4f}')

        # ── (A) full grid ──
        vals = {g: score(panel, g) for g in G}
        best = max(vals, key=lambda g: vals[g])
        tol = abs(vals[best]) * 0.01
        flat = [g for g in G if vals[g] >= vals[best] - tol]
        print(f'\n  (A) FULL GRID over all {len(pairs)} pairs, step {STEP}')
        print(f'    argmax        bb={best[0]:.2f} sd={best[1]:.2f} ct={best[2]:.2f}'
              f'   mean z {vals[best]:.4f}  (r {unfisher(vals[best]):.4f})')
        print(f'    SHIPPED       bb={SHIPPED[0]:.2f} sd={SHIPPED[1]:.2f} '
              f'ct={SHIPPED[2]:.2f}   mean z {vals[SHIPPED]:.4f}  '
              f'(r {unfisher(vals[SHIPPED]):.4f})')
        print(f'    old 70/15/15                              '
              f'   mean z {vals[OLD_PRIOR]:.4f}  (r {unfisher(vals[OLD_PRIOR]):.4f})')
        print(f'    shipped inside the 1% flat region: {SHIPPED in flat}')
        print(f'    flat region: bb [{min(g[0] for g in flat):.2f}, '
              f'{max(g[0] for g in flat):.2f}]  sd [{min(g[1] for g in flat):.2f}, '
              f'{max(g[1] for g in flat):.2f}]  ct [{min(g[2] for g in flat):.2f}, '
              f'{max(g[2] for g in flat):.2f}]  ({len(flat)} of {len(G)} cells)')
        interior = all(0 < best[i] < 1 for i in range(3))
        print(f'    interior optimum (no weight at 0 or 1): '
              f'{"YES" if interior else "NO — sits on a simplex edge"}')

        # ── (B) LOPO ──
        print(f'\n  (B) LEAVE-ONE-PAIR-OUT — derive on 3 pairs, score on the 4th')
        print(f'    {"held out":<12}{"derived weights":<26}{"derived r":>11}'
              f'{"shipped r":>11}{"winner":>9}')
        d_z, s_z = [], []
        for hold in pairs:
            tr = [p for p in pairs if p != hold]
            sub = {g: score(panel, g, tr) for g in G}
            bw = max(sub, key=lambda g: sub[g])
            bb, sdz, ctz, y = panel[hold]
            def r_of(w):
                return pear([w[0]*bb[i] + w[1]*sdz[i] + w[2]*ctz[i]
                             for i in range(len(y))], y)
            rd, rs = r_of(bw), r_of(SHIPPED)
            d_z.append(fisher(rd)); s_z.append(fisher(rs))
            print(f'    {str(hold):<12}'
                  f'{f"{bw[0]:.2f}/{bw[1]:.2f}/{bw[2]:.2f}":<26}'
                  f'{rd:>11.4f}{rs:>11.4f}'
                  f'{("derived" if rd > rs else "shipped"):>9}')
        print(f'    {"MEAN":<12}{"":<26}'
              f'{unfisher(sum(d_z)/len(d_z)):>11.4f}'
              f'{unfisher(sum(s_z)/len(s_z)):>11.4f}')
        wins = sum(1 for a, b in zip(d_z, s_z) if a > b)
        print(f'    re-derived weights beat shipped in {wins}/{len(pairs)} folds')


if __name__ == '__main__':
    main()
