"""locplus_count_2026_replicate.py — sixth replicate for the count-surface
package, on 2026 production data (Wally's retags), with sharper controls:

  - partial r(loc, next-quarter xRV | pitcher STUFF+) — the real control,
    unavailable on historical caches — alongside the harness's | FF velo.
  - two scoring floors: 150 (harness standard, starter-heavy) and 75
    (reliever sensitivity).
  - whiff-skill leak: corr(loc, pitcher whiff rate) shipped vs variant —
    the count-whiff surfaces must not absorb pitcher whiff ability.

2026 was never used to fit or select any count-surface choice, so this is
a legitimate replicate, on the tag distribution production actually runs.

Usage: python3 scripts/locplus_count_2026_replicate.py
"""
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_locplus as lp
import locplus_constants_multiseason as base
import locplus_countwhiff_multiseason as W
import locplus_countphys_extension as X
import locplus_countcompose_multiseason as C
from pipeline_sdplus import make_rv_xrv, SWING_DESCRIPTIONS

LG, SCALE = 0.3172, 1.2343


def main():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    b = [p for p in D if p.get('_source', 'MLB') == 'MLB'
         and lp.is_eligible_baseline(p)]
    del D
    rv_fn = make_rv_xrv(LG, SCALE)

    dates = sorted({p['Game Date'] for p in b if p['Game Date']})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in b:
        d = p['Game Date']
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)

    prep = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        actual, stuff, velo, whiff = {}, {}, {}, {}
        for k, ps in base.by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= base.MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        for k, ps in base.by_pitcher(seg[f]).items():
            sv = [lp.safe_float(p.get('Stuff+')) for p in ps]
            sv = [x for x in sv if x is not None]
            if len(sv) >= 100:
                stuff[k] = sum(sv) / len(sv)
            fv = [lp.safe_float(p['Velocity']) for p in ps
                  if p['Pitch Type'] == 'FF']
            fv = [x for x in fv if x is not None]
            if len(fv) >= base.MIN_VELO:
                velo[k] = sum(fv) / len(fv)
            sw = [p for p in ps if p.get('Description') in SWING_DESCRIPTIONS]
            if len(sw) >= 100:
                whiff[k] = (sum(1 for p in sw
                                if p.get('Description') == 'Swinging Strike')
                            / len(sw))
        prep[half] = {'first': seg[f], 'byp_f': base.by_pitcher(seg[f]),
                      'actual': actual, 'stuff': stuff, 'velo': velo,
                      'whiff': whiff}

    S = {h: lp.build_surfaces(prep[h]['first'], LG, SCALE) for h in ('A', 'B')}
    WH = {h: W.build_wh_count(prep[h]['first']) for h in ('A', 'B')}
    XW = {h: X.build_xw_clevel(prep[h]['first'], LG, SCALE) for h in ('A', 'B')}

    variants = {'shipped': (None, None), 'wh_count': (WH, None),
                'xw_clevel': (None, XW), 'both': (WH, XW)}

    for floor in (150, 75):
        print(f"\n=== floor {floor} scored pitches ===")
        print(f"{'variant':>10s} | {'part|Stuff+':>11s} {'part|velo':>10s} "
              f"{'raw':>7s} | {'r(whiff)':>9s} {'n':>4s}")
        for name, (wh, xw) in variants.items():
            pS, pV, raws, wleak, ns = [], [], [], [], []
            for half in ('A', 'B'):
                c = prep[half]
                loc = C.score_map_c(c['byp_f'], S[half], floor,
                                    wh[half] if wh else None,
                                    xw[half] if xw else None)
                kk = [k for k in loc if k in c['actual'] and k in c['stuff']
                      and k in c['velo'] and k in c['whiff']]
                if len(kk) < 30:
                    continue
                L = [loc[k] for k in kk]
                Y = [c['actual'][k] for k in kk]
                St = [c['stuff'][k] for k in kk]
                V = [c['velo'][k] for k in kk]
                Wh = [c['whiff'][k] for k in kk]
                r_ly = base.pearson(L, Y)
                pS.append(base.partial(r_ly, base.pearson(L, St),
                                       base.pearson(St, Y)))
                pV.append(base.partial(r_ly, base.pearson(L, V),
                                       base.pearson(V, Y)))
                raws.append(r_ly)
                wleak.append(base.pearson(L, Wh))
                ns.append(len(kk))
            print(f"{name:>10s} | {sum(pS)/len(pS):>11.3f} "
                  f"{sum(pV)/len(pV):>10.3f} {sum(raws)/len(raws):>7.3f} | "
                  f"{sum(wleak)/len(wleak):>+9.3f} {sum(ns)//len(ns):>4d}",
                  flush=True)

    print("\nPASS = variants beat shipped on part|Stuff+ at both floors, with")
    print("r(whiff) not materially more negative than shipped (loc scores are")
    print("hitter-perspective ExpRV: lower = better pitcher, so a MORE negative")
    print("r(whiff) would mean absorbing whiff skill).")


if __name__ == '__main__':
    main()
