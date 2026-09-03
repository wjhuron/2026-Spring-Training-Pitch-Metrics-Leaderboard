"""locplus_countcompose_multiseason.py — do wh_count and xw_clevel compose?
Both winners encode 2-strike defensiveness (whiff shape by count; contact
quality level by count), so their gains may overlap. Variants: shipped,
wh_count, xw_clevel, both. Same harness/objectives as the flags battery.

Usage: python3 scripts/research/locplus/locplus_countcompose_multiseason.py
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
import locplus_countwhiff_multiseason as W
import locplus_countphys_extension as X
from pipeline.sdplus import make_rv_xrv


def score_pitch_c(p, S, WHC=None, XWC=None):
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
    wt = WHC.get(key) if WHC is not None else None
    # 2026-09-02: live S['WH'][key] is per-count; base.wh_at reads either shape.
    pwh = wt[c][i][j] if wt is not None else base.wh_at(S, key, c, i, j)
    pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = S['XW'][key][i][j] + (XWC.get(c, 0.0) if XWC is not None else 0.0)
    pcs = S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


def score_map_c(byp, S, min_n, WHC=None, XWC=None):
    out = {}
    for k, ps in byp.items():
        v = [s for s in (score_pitch_c(p, S, WHC, XWC) for p in ps)
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

    S_first = {h: lp.build_surfaces(prep[h]['first'], base.LG, base.SCALE)
               for h in ('A', 'B')}
    S_rel = {0: lp.build_surfaces(g0, base.LG, base.SCALE),
             1: lp.build_surfaces(g1, base.LG, base.SCALE)}
    WHf = {h: W.build_wh_count(prep[h]['first']) for h in ('A', 'B')}
    WHr = {0: W.build_wh_count(g0), 1: W.build_wh_count(g1)}
    XWf = {h: X.build_xw_clevel(prep[h]['first'], base.LG, base.SCALE)
           for h in ('A', 'B')}
    XWr = {0: X.build_xw_clevel(g0, base.LG, base.SCALE),
           1: X.build_xw_clevel(g1, base.LG, base.SCALE)}

    variants = {'shipped': (False, False), 'wh_count': (True, False),
                'xw_clevel': (False, True), 'both': (True, True)}
    res = {}
    for name, (uw, ux) in variants.items():
        raws, parts, rlvs = [], [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = score_map_c(c['byp_f'], S_first[half], base.MIN_SCORE,
                              WHf[half] if uw else None,
                              XWf[half] if ux else None)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
            rlvs.append(r_ls)
        a0 = score_map_c(base.by_pitcher(g0), S_rel[0], base.MIN_REL,
                         WHr[0] if uw else None, XWr[0] if ux else None)
        a1 = score_map_c(base.by_pitcher(g1), S_rel[1], base.MIN_REL,
                         WHr[1] if uw else None, XWr[1] if ux else None)
        ks = [k for k in a0 if k in a1]
        rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[name] = {'partial': sum(parts) / len(parts),
                     'raw': sum(raws) / len(raws),
                     'rlv': sum(rlvs) / len(rlvs), 'rel': rel}
    return res


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    names = ('shipped', 'wh_count', 'xw_clevel', 'both')
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
            for n in names), flush=True)
        del pitches
        gc.collect()

    print("\nCOMPOSITION TABLE (mean across seasons; delta vs shipped)")
    s = {m: sum(table[y]['shipped'][m] for y in table) / len(table)
         for m in ('partial', 'rel')}
    for n in names:
        pm = sum(table[y][n]['partial'] for y in table) / len(table)
        rm = sum(table[y][n]['rel'] for y in table) / len(table)
        print(f"  {n:>10s}: partial {pm:.4f} ({pm-s['partial']:+.4f})  "
              f"rel {rm:.4f} ({rm-s['rel']:+.4f})")
    print("\nPer-season wins vs shipped on partial:")
    for n in names:
        if n == 'shipped':
            continue
        wins = sum(1 for y in table
                   if table[y][n]['partial'] > table[y]['shipped']['partial'])
        print(f"  {n}: {wins}/{len(table)}")


if __name__ == '__main__':
    main()
