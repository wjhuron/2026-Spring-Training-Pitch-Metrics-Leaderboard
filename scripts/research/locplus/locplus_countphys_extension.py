"""locplus_countphys_extension.py — do the OTHER physical channels want
count structure too? Companion to locplus_countwhiff_multiseason.py (whiff
won 5/5 on partial). Both candidates have a defensive-swing story: with two
strikes, foul rates jump and contact quality drops at the same location.

Variants (single toggles off shipped):
  fl_count   count-specific FOUL surfaces: per-count grids from that
             count's swings, shrunk toward collapsed_FL x count foul
             multiplier, K_FL_COUNT = 20 (same design as the whiff winner).
  xw_clevel  contact-quality LEVEL by count: XW keeps its collapsed shape,
             plus an additive per-count offset = league mean standardized
             xwOBA-value on BIP in count c minus the overall league mean
             (>= 200 BIP per count, else 0). NOT the rejected BIP anchor —
             that was a delta-RE currency correction; this is the
             contact-quality level itself.

Objectives and seasons: identical to the flags harness.

Usage: python3 scripts/research/locplus/locplus_countphys_extension.py
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

K_FL_COUNT = 20
MIN_XW_COUNT_BIP = 200


def build_fl_count(baseline):
    """{(grp,bh,ph): {count: foul grid}} — per-count foul-given-swing."""
    A = defaultdict(lambda: {'fln': lp._zeros(), 'swn': lp._zeros()})
    AC = defaultdict(lambda: {'fln': lp._zeros(), 'swn': lp._zeros()})
    cnt_fl = defaultdict(lambda: [0, 0])
    for p in baseline:
        d = p.get('Description')
        if d not in lp.SWING_DESC:
            continue
        key = (lp.group_of(p), p['Bats'], p['Throws'])
        c = lp.get_count(p)
        i = lp._xbin(lp.safe_float(p.get('PlateX')))
        j = lp._zbin(lp._znorm(p))
        A[key]['swn'][i][j] += 1
        AC[(key, c)]['swn'][i][j] += 1
        cnt_fl[c][1] += 1
        if d == 'Foul':
            A[key]['fln'][i][j] += 1
            AC[(key, c)]['fln'][i][j] += 1
            cnt_fl[c][0] += 1
    tot_fl = sum(v[0] for v in cnt_fl.values())
    tot_sw = sum(v[1] for v in cnt_fl.values())
    overall = tot_fl / tot_sw if tot_sw else 0.0
    mult = {c: ((v[0] / v[1]) / overall if v[1] and overall else 1.0)
            for c, v in cnt_fl.items()}
    out = {}
    for key, a in A.items():
        kx, kz = lp._kernels_for(key[0])
        swn = lp._gsum(a['swn'])
        coll = lp._smooth(a['fln'], a['swn'],
                          lp._gsum(a['fln']) / max(swn, 1), lp.K_FOUL, kx, kz)
        out[key] = {}
        for c in lp.COUNTS:
            m = mult.get(c, 1.0)
            prior_c = [[min(1.0, coll[i][j] * m) for j in range(lp.NZ)]
                       for i in range(lp.NX)]
            ac = AC.get((key, c))
            out[key][c] = (prior_c if ac is None else
                           lp._smooth(ac['fln'], ac['swn'], prior_c,
                                      K_FL_COUNT, kx, kz))
    return out


def build_xw_clevel(baseline, lg_woba, woba_scale):
    """{count: additive offset to the standardized xwOBA-value BIP branch}."""
    acc = defaultdict(lambda: [0.0, 0])
    for p in baseline:
        if p.get('Description') != 'In Play':
            continue
        xw = lp.safe_float(p.get('xwOBA'))
        c = lp.get_count(p)
        if xw is None or c is None:
            continue
        v = (xw - lg_woba) / woba_scale
        acc[c][0] += v
        acc[c][1] += 1
    tot_s = sum(s for s, _ in acc.values())
    tot_n = sum(n for _, n in acc.values())
    overall = tot_s / tot_n if tot_n else 0.0
    return {c: ((s / n - overall) if n >= MIN_XW_COUNT_BIP else 0.0)
            for c, (s, n) in acc.items()}


def score_pitch_v(p, S, FLC=None, XWC=None):
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
    pwh = S['WH'][key][i][j]
    ft = FLC.get(key) if FLC is not None else None
    pfl = ft[c][i][j] if ft is not None else S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = S['XW'][key][i][j] + (XWC.get(c, 0.0) if XWC is not None else 0.0)
    pcs = S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


def score_map_v(byp, S, min_n, FLC=None, XWC=None):
    out = {}
    for k, ps in byp.items():
        if FLC is None and XWC is None:
            v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        else:
            v = [s for s in (score_pitch_v(p, S, FLC, XWC) for p in ps)
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

    variants = {
        'shipped': (None, None),
        'fl_count': ('FL', None),
        'xw_clevel': (None, 'XW'),
    }
    res = {}
    for name, (fl, xw) in variants.items():
        FLf = {h: build_fl_count(prep[h]['first']) for h in ('A', 'B')} if fl else None
        FLr = {0: build_fl_count(g0), 1: build_fl_count(g1)} if fl else None
        XWf = ({h: build_xw_clevel(prep[h]['first'], base.LG, base.SCALE)
                for h in ('A', 'B')} if xw else None)
        XWr = ({0: build_xw_clevel(g0, base.LG, base.SCALE),
                1: build_xw_clevel(g1, base.LG, base.SCALE)} if xw else None)
        raws, parts = [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = score_map_v(c['byp_f'], S_first[half], base.MIN_SCORE,
                              FLf[half] if FLf else None,
                              XWf[half] if XWf else None)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
        a0 = score_map_v(base.by_pitcher(g0), S_rel[0], base.MIN_REL,
                         FLr[0] if FLr else None, XWr[0] if XWr else None)
        a1 = score_map_v(base.by_pitcher(g1), S_rel[1], base.MIN_REL,
                         FLr[1] if FLr else None, XWr[1] if XWr else None)
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
            for n in ('shipped', 'fl_count', 'xw_clevel')), flush=True)
        del pitches
        gc.collect()

    for name in ('fl_count', 'xw_clevel'):
        print(f"\nVERDICT — {name} vs shipped, per season")
        for metric in ('partial', 'rel'):
            wins, cells = 0, []
            for yr in sorted(table):
                d = table[yr][name][metric] - table[yr]['shipped'][metric]
                cells.append(f"{yr}:{d:+.3f}")
                wins += 1 if d > 0 else 0
            print(f"  {metric}: wins {wins}/{len(cells)}  " + '  '.join(cells))


if __name__ == '__main__':
    main()
