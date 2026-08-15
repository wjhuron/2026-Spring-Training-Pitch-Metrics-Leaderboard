"""locplus_cs_shape_multiseason.py — should the called-strike surface have
per-count SHAPE, not just the per-count level shift it has today?

The CS count transform (kept, convention call) moves the whole surface by a
logit intercept per (hand, count). Umpire zones plausibly change SHAPE by
count too (3-0 widening more horizontally than vertically). Variant
CS_SHAPE: per-(hand, count) CS grids from that count's own takes, shrunk
toward the count-transformed base surface with K_CS_COUNT = 20 pseudo-takes
(mirroring the winning whiff design). Falls back to the base surface where
takes are thin.

Objectives and seasons: identical to the flags harness (2021-2025
replicates, partial|velo + raw + rel). Placebo discipline: this adds
structure, so if it wins on rel alone it does NOT ship; it must win
partial. (A full permuted placebo follows only if it wins.)

Usage: python3 scripts/research/locplus/locplus_cs_shape_multiseason.py
"""
import gc
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.sdplus import make_rv_xrv

K_CS_COUNT = 20


def build_cs_shape(baseline):
    """{(hand, count): grid} — per-count CS-given-take grids shrunk toward
    the production (count-transformed) base surface."""
    S = lp.build_surfaces(baseline, base.LG, base.SCALE)
    csn = defaultdict(lp._zeros)
    csd = defaultdict(lp._zeros)
    for p in baseline:
        d = p.get('Description')
        if d not in lp.TAKE_DESC:
            continue
        h = p.get('Bats')
        if h not in lp.HANDS:
            continue
        c = lp.get_count(p)
        i = lp._xbin(lp.safe_float(p.get('PlateX')))
        j = lp._zbin(lp._znorm(p))
        csd[(h, c)][i][j] += 1
        if d == 'Called Strike':
            csn[(h, c)][i][j] += 1
    out = {}
    for h in lp.HANDS:
        for c in lp.COUNTS:
            prior = S['PCS'][h][c]
            k = (h, c)
            if k in csd:
                out[k] = lp._smooth(csn[k], csd[k], prior, K_CS_COUNT)
            else:
                out[k] = prior
    return S, out


def score_pitch_css(p, S, CSS):
    key = (lp.group_of(p), p.get('Bats'), p.get('Throws'))
    if key not in S['WH']:
        return None
    c = lp.get_count(p)
    px = lp.safe_float(p.get('PlateX'))
    zn = lp._znorm(p)
    if c is None or px is None or zn is None:
        return None
    i = lp._xbin(px)
    j = lp._zbin(zn)
    psw = S['SW'][key][c][i][j]
    pwh = S['WH'][key][c][i][j]
    pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = (S['XW'][key][i][j] + S['XWOFF'].get(c, 0.0)
            + S['BIPOFF'].get(c, 0.0))
    pcs = CSS[(p['Bats'], c)][i][j] if CSS is not None else S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


def score_map(byp, S, CSS, min_n):
    out = {}
    for k, ps in byp.items():
        v = [s for s in (score_pitch_css(p, S, CSS) for p in ps)
             if s is not None]
        if len(v) >= min_n:
            out[k] = sum(v) / len(v)
    return out


def eval_season(pitches, rv_fn):
    b = [p for p in pitches if lp.is_eligible_baseline(p)]
    dates = sorted({p['Game Date'] for p in b if p['Game Date']})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in b:
        d = p['Game Date']
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)
    par = {d: i % 2 for i, d in enumerate(dates)}
    g0 = [p for p in b if par.get(p['Game Date']) == 0]
    g1 = [p for p in b if par.get(p['Game Date']) == 1]

    prep = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        actual = {}
        for k, ps in base.by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= base.MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        velo = {}
        for k, ps in base.by_pitcher(seg[f]).items():
            v = [x for x in (lp.safe_float(p['Velocity']) for p in ps
                             if p['Pitch Type'] == 'FF') if x is not None]
            if len(v) >= base.MIN_VELO:
                velo[k] = sum(v) / len(v)
        prep[half] = {'first': seg[f], 'byp_f': base.by_pitcher(seg[f]),
                      'actual': actual, 'velo': velo}

    SW_ = {}
    CS_ = {}
    for h in ('A', 'B'):
        SW_[h], CS_[h] = build_cs_shape(prep[h]['first'])
    S0, C0 = build_cs_shape(g0)
    S1, C1 = build_cs_shape(g1)

    res = {}
    for name, use in (('shipped', False), ('cs_shape', True)):
        raws, parts = [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = score_map(c['byp_f'], SW_[half],
                            CS_[half] if use else None, base.MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
        a0 = score_map(base.by_pitcher(g0), S0, C0 if use else None, base.MIN_REL)
        a1 = score_map(base.by_pitcher(g1), S1, C1 if use else None, base.MIN_REL)
        ks = [k for k in a0 if k in a1]
        rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[name] = {'partial': sum(parts) / len(parts),
                     'raw': sum(raws) / len(raws), 'rel': rel}
    return res


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        table[yr] = eval_season(pitches, rv_fn)
        print(f"{yr}: " + '  '.join(
            f"{n}: p={table[yr][n]['partial']:.3f} rel={table[yr][n]['rel']:.3f}"
            for n in ('shipped', 'cs_shape')), flush=True)
        del pitches
        gc.collect()

    print("\nVERDICT — cs_shape vs shipped, per season")
    for metric in ('partial', 'rel'):
        wins, cells = 0, []
        for yr in sorted(table):
            d = table[yr]['cs_shape'][metric] - table[yr]['shipped'][metric]
            cells.append(f"{yr}:{d:+.3f}")
            wins += 1 if d > 0 else 0
        print(f"  {metric}: wins {wins}/{len(cells)}  " + '  '.join(cells))
    print("\nShip bar: must win partial in most seasons (rel-only wins do not")
    print("count, per the placebo standard); a partial win triggers a full")
    print("permuted-count placebo before adoption.")


if __name__ == '__main__':
    main()
