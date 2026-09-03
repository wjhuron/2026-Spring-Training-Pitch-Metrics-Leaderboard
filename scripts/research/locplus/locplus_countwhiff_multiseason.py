"""locplus_countwhiff_multiseason.py — should the Loc+ WHIFF surface be
count-specific, the way the swing surface already is?

Today WH (and FL/XW) are count-collapsed location shapes; only the swing
surface learns per-count location grids (shrunk toward collapsed x count
multiplier, SWING_PRIOR_COUNT_LEVEL). Count-specific physical surfaces
were rejected 2026-07-05 on 2026-only evaluation inside the pooled-seasons
harness — never tested to the replicate standard. This closes that gap for
the whiff surface, the channel with the plausible count-location story
(2-strike chase whiffs).

Variant WH_COUNT: per-count whiff grids from that count's own swings,
shrunk toward prior_c = collapsed_WH x (count whiff level / overall),
K_WH_COUNT = 20 pseudo-obs (mirroring K_SWING_COUNT; if the variant wins,
K gets swept before adoption). Scoring swaps only the pwh lookup.

Objectives and seasons: identical to locplus_flags_multiseason.py.

Usage: python3 scripts/research/locplus/locplus_countwhiff_multiseason.py
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

K_WH_COUNT = 20


def build_wh_count(baseline):
    """{(grp,bh,ph): {count: whiff grid}} — per-count whiff-given-swing
    surfaces, shrunk toward the collapsed shape scaled to the count's
    whiff level."""
    A = defaultdict(lambda: {'whn': lp._zeros(), 'swn': lp._zeros()})
    AC = defaultdict(lambda: {'whn': lp._zeros(), 'swn': lp._zeros()})
    cnt_wh = defaultdict(lambda: [0, 0])   # count -> [whiffs, swings]
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
        cnt_wh[c][1] += 1
        if d == 'Swinging Strike':
            A[key]['whn'][i][j] += 1
            AC[(key, c)]['whn'][i][j] += 1
            cnt_wh[c][0] += 1

    tot_wh = sum(v[0] for v in cnt_wh.values())
    tot_sw = sum(v[1] for v in cnt_wh.values())
    overall = tot_wh / tot_sw if tot_sw else 0.0
    mult = {c: ((v[0] / v[1]) / overall if v[1] and overall else 1.0)
            for c, v in cnt_wh.items()}

    out = {}
    for key, a in A.items():
        kx, kz = lp._kernels_for(key[0])
        swn = lp._gsum(a['swn'])
        coll = lp._smooth(a['whn'], a['swn'],
                          lp._gsum(a['whn']) / max(swn, 1), lp.K_WHIFF, kx, kz)
        out[key] = {}
        for c in lp.COUNTS:
            m = mult.get(c, 1.0)
            prior_c = [[min(1.0, coll[i][j] * m) for j in range(lp.NZ)]
                       for i in range(lp.NX)]
            ac = AC.get((key, c))
            if ac is None:
                out[key][c] = prior_c
            else:
                out[key][c] = lp._smooth(ac['whn'], ac['swn'], prior_c,
                                         K_WH_COUNT, kx, kz)
    return out


def score_pitch_whc(p, S, WHC):
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
    wt = WHC.get(key)
    # 2026-09-02: live S['WH'][key] is per-count; base.wh_at reads either shape.
    pwh = wt[c][i][j] if wt is not None else base.wh_at(S, key, c, i, j)
    pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = S['XW'][key][i][j] + S['BIPOFF'].get(c, 0.0)
    pcs = S['PCS'][p['Bats']][c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


def score_map_whc(byp, S, WHC, min_n):
    out = {}
    for k, ps in byp.items():
        if WHC is None:
            v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        else:
            v = [s for s in (score_pitch_whc(p, S, WHC) for p in ps)
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
    W_first = {h: build_wh_count(prep[h]['first']) for h in ('A', 'B')}
    W_rel = {0: build_wh_count(g0), 1: build_wh_count(g1)}

    res = {}
    for name, use in (('shipped', False), ('wh_count', True)):
        raws, parts, rlvs = [], [], []
        for half in ('A', 'B'):
            c = prep[half]
            loc = score_map_whc(c['byp_f'], S_first[half],
                                W_first[half] if use else None, base.MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly)
            parts.append(base.partial(r_ly, r_ls, r_sy))
            rlvs.append(r_ls)
        a0 = score_map_whc(base.by_pitcher(g0), S_rel[0],
                           W_rel[0] if use else None, base.MIN_REL)
        a1 = score_map_whc(base.by_pitcher(g1), S_rel[1],
                           W_rel[1] if use else None, base.MIN_REL)
        ks = [k for k in a0 if k in a1]
        rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[name] = {'raw': sum(raws) / len(raws),
                     'partial': sum(parts) / len(parts),
                     'rlv': sum(rlvs) / len(rlvs), 'rel': rel, 'n': len(ks)}
    return res


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    print(f"{'season':>7s} {'config':>9s} | {'PARTIAL|velo':>13s} | {'raw':>6s} "
          f"| {'r(velo)':>8s} | {'rel':>6s}")
    print('-' * 62)
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr:>7d}   cache missing, skipped", flush=True)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        res = eval_season(pitches, rv_fn)
        table[yr] = res
        for name in ('shipped', 'wh_count'):
            o = res.get(name)
            if o:
                print(f"{yr:>7d} {name:>9s} | {o['partial']:>13.3f} | {o['raw']:>6.3f} "
                      f"| {o['rlv']:>+8.3f} | {o['rel']:>6.3f}", flush=True)
        del pitches
        gc.collect()

    print("\nVERDICT — wh_count vs shipped, per season")
    for metric in ('partial', 'rel'):
        wins, cells = 0, []
        for yr in sorted(table):
            s, v = table[yr].get('shipped'), table[yr].get('wh_count')
            if not s or not v:
                continue
            d = v[metric] - s[metric]
            cells.append(f"{yr}:{d:+.3f}")
            wins += 1 if d > 0 else 0
        print(f"  {metric}: wins {wins}/{len(cells)}  " + '  '.join(cells))
    print("\nAdopt only on a majority of replicate seasons on partial without")
    print("a rel collapse; a win means sweeping K_WH_COUNT before shipping.")


if __name__ == '__main__':
    main()
