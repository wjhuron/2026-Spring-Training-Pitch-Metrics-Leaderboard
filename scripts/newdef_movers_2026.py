"""newdef_movers_2026.py — biggest 2026 risers/fallers under the 2026-08-15
definition changes, computed old-config vs new-config through the same code
paths (pure definition deltas, no scale artifacts).

SD+:  old = anchored + cat3 cells, n0 200  ->  new = un-anchored, zone x
      count, n0 180. Profile columns: 2-strike decision share, BRK share of
      decisions (the two channels the change touches).
CT+:  old = zone x count cells  ->  new = cat3 cells (anchor unchanged).
      Profile: BRK+OFF share of swings, whiff rate on BRK vs FB.
Loc+: old = count-collapsed whiff/contact  ->  new = count-aware. Raw mean
      ExpRV percentile shift among MLB pitchers >= 800 scored pitches
      (proxy for ranking movement; the displayed atom-canon shifts follow).
      Profile: 2-strike pitch share, share of pitches above the zone.

Floors: hitters >= 300 decisions / 150 swings; pitchers >= 800 pitches.

Usage: python3 scripts/newdef_movers_2026.py
"""
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_sdplus as sd
import pipeline_contact as ct
import pipeline_locplus as lp

LG, SCALE = 0.3172, 1.2343
MIN_DEC, MIN_SW, MIN_PIT = 300, 150, 800


def pctl_map(d, reverse=False):
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)
    n = len(items)
    return {k: 100.0 * i / (n - 1) for i, (k, _) in enumerate(items)}


def main():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    del D

    by_h = defaultdict(list)
    for p in mlb:
        h, t = p.get('Batter'), p.get('BTeam')
        if h and t:
            by_h[(h, t)].append(p)

    # ── SD+ old vs new ──
    new_res, _ = sd.compute_sd_plus(mlb, by_h, LG, SCALE)

    orig_sdcat, orig_sdcats = sd._sd_cat, sd.SD_CATS
    try:
        sd._sd_cat = sd.cat_of         # cat3 back on
        sd.SD_CATS = sd.CATS           # shrink cascade over the real cats
        elig = [p for p in mlb if sd.is_eligible(p)]
        offsets = sd.build_bip_count_offsets(elig, LG, SCALE)
        rv_fn = sd.make_rv_xrv(LG, SCALE, offsets)   # anchor back on
        raw = sd.build_weight_table(elig, rv_fn)
        zm = sd.zone_level_means(elig, rv_fn)
        table = sd.shrink_table(raw, zm)
        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}
        hitter_raw = sd.compute_hitter_sd(by_h, table, lgw)
        old_res = sd.regress_and_normalize(hitter_raw, n_prior=200, min_n=200)
    finally:
        sd._sd_cat, sd.SD_CATS = orig_sdcat, orig_sdcats

    prof = {}
    for key, ps in by_h.items():
        el = [p for p in ps if sd.is_eligible(p)]
        if len(el) < MIN_DEC:
            continue
        two = sum(1 for p in el if (sd.get_count(p) or (0, 0))[1] == 2)
        brk = sum(1 for p in el if sd.cat_of(p) == 'BRK')
        prof[key] = (len(el), two / len(el), brk / len(el))

    print("=== SD+ (old: anchored+cat3, n0 200 -> new: un-anchored zone x count, n0 180) ===")
    movers = []
    for key in prof:
        o = old_res.get(key)
        n = new_res.get(key)
        if o and n:
            movers.append((n['sdPlus'] - o['sdPlus'], key, o['sdPlus'], n['sdPlus']))
    movers.sort(reverse=True)
    print(f"{'hitter':<26s}{'team':<5s}{'old':>7s}{'new':>7s}{'delta':>7s}"
          f"{'dec':>6s}{'2K%':>6s}{'BRK%':>6s}")
    for row in movers[:8]:
        d, key, o, n = row
        pr = prof[key]
        print(f"{key[0]:<26s}{key[1]:<5s}{o:>7.1f}{n:>7.1f}{d:>+7.1f}"
              f"{pr[0]:>6d}{100*pr[1]:>6.1f}{100*pr[2]:>6.1f}")
    print("  ---")
    for row in movers[-8:]:
        d, key, o, n = row
        pr = prof[key]
        print(f"{key[0]:<26s}{key[1]:<5s}{o:>7.1f}{n:>7.1f}{d:>+7.1f}"
              f"{pr[0]:>6d}{100*pr[1]:>6.1f}{100*pr[2]:>6.1f}")

    # ── CT+ old vs new (cat3 only) ──
    new_ct, _ = ct.compute_ct_plus(mlb, by_h, LG, SCALE)
    orig_sd_cat_of, orig_ct_cat_of, orig_ct_cats = sd.cat_of, ct.cat_of, ct.CATS
    try:
        sd.cat_of = ct.cat_of = (lambda p: 'ALL')
        ct.CATS = ('ALL',)
        old_ct, _ = ct.compute_ct_plus(mlb, by_h, LG, SCALE)
    finally:
        sd.cat_of, ct.cat_of, ct.CATS = orig_sd_cat_of, orig_ct_cat_of, orig_ct_cats

    ct_prof = {}
    for key, ps in by_h.items():
        sw = [p for p in ps if ct.is_ct_eligible(p)]
        if len(sw) < MIN_SW:
            continue
        brk = [p for p in sw if sd.cat_of(p) in ('BRK', 'OFF')]
        whb = (sum(1 for p in brk if ct.classify_contact_outcome(p) == 'whiff')
               / len(brk)) if brk else None
        ct_prof[key] = (len(sw), len(brk) / len(sw), whb)

    print("\n=== CT+ (old: zone x count -> new: cat3 cells; anchor/n0 unchanged) ===")
    movers = []
    for key in ct_prof:
        o = old_ct.get(key)
        n = new_ct.get(key)
        if o and n:
            movers.append((n['ctPlus'] - o['ctPlus'], key, o['ctPlus'], n['ctPlus']))
    movers.sort(reverse=True)
    print(f"{'hitter':<26s}{'team':<5s}{'old':>7s}{'new':>7s}{'delta':>7s}"
          f"{'sw':>6s}{'BRK+OFF%':>9s}{'whiffBRK':>9s}")
    for row in movers[:8]:
        d, key, o, n = row
        pr = ct_prof[key]
        wb = f"{100*pr[2]:>9.1f}" if pr[2] is not None else f"{'--':>9s}"
        print(f"{key[0]:<26s}{key[1]:<5s}{o:>7.1f}{n:>7.1f}{d:>+7.1f}"
              f"{pr[0]:>6d}{100*pr[1]:>9.1f}{wb}")
    print("  ---")
    for row in movers[-8:]:
        d, key, o, n = row
        pr = ct_prof[key]
        wb = f"{100*pr[2]:>9.1f}" if pr[2] is not None else f"{'--':>9s}"
        print(f"{key[0]:<26s}{key[1]:<5s}{o:>7.1f}{n:>7.1f}{d:>+7.1f}"
              f"{pr[0]:>6d}{100*pr[1]:>9.1f}{wb}")

    # ── Loc+ old vs new (count-aware surfaces) ──
    base = [p for p in mlb if lp.is_eligible_baseline(p)]
    S_new = lp.build_surfaces(base, LG, SCALE)
    lp.WH_COUNT_LEVEL = False
    lp.XW_COUNT_LEVEL = False
    S_old = lp.build_surfaces(base, LG, SCALE)
    lp.WH_COUNT_LEVEL = True
    lp.XW_COUNT_LEVEL = True

    acc_o, acc_n = defaultdict(list), defaultdict(list)
    lprof = defaultdict(lambda: [0, 0, 0])   # n, 2K, above-zone
    for p in base:
        k = (p.get('Pitcher'), p.get('Throws'))
        vo = lp.score_pitch(p, S_old)
        vn = lp.score_pitch(p, S_new)
        if vo is None or vn is None:
            continue
        acc_o[k].append(vo)
        acc_n[k].append(vn)
        c = lp.get_count(p)
        zn = lp._znorm(p)
        lprof[k][0] += 1
        if c and c[1] == 2:
            lprof[k][1] += 1
        if zn is not None and zn > 1.1:
            lprof[k][2] += 1

    keep = [k for k in acc_o if len(acc_o[k]) >= MIN_PIT]
    old_m = {k: -sum(acc_o[k]) / len(acc_o[k]) for k in keep}
    new_m = {k: -sum(acc_n[k]) / len(acc_n[k]) for k in keep}
    po = pctl_map(old_m)
    pn = pctl_map(new_m)

    print("\n=== Loc+ (old: count-collapsed -> new: count-aware; raw percentile shift, >=800 pitches) ===")
    movers = sorted(((pn[k] - po[k], k) for k in keep), reverse=True)
    print(f"{'pitcher':<26s}{'oldPct':>8s}{'newPct':>8s}{'delta':>7s}"
          f"{'n':>7s}{'2K%':>6s}{'aboveZ%':>8s}")
    for d, k in movers[:8]:
        pr = lprof[k]
        print(f"{k[0]:<26s}{po[k]:>8.1f}{pn[k]:>8.1f}{d:>+7.1f}"
              f"{pr[0]:>7d}{100*pr[1]/pr[0]:>6.1f}{100*pr[2]/pr[0]:>8.1f}")
    print("  ---")
    for d, k in movers[-8:]:
        pr = lprof[k]
        print(f"{k[0]:<26s}{po[k]:>8.1f}{pn[k]:>8.1f}{d:>+7.1f}"
              f"{pr[0]:>7d}{100*pr[1]/pr[0]:>6.1f}{100*pr[2]/pr[0]:>8.1f}")


if __name__ == '__main__':
    main()
