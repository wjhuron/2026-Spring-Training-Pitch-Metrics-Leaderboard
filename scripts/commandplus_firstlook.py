"""commandplus_firstlook.py — Command+ v1 on 2026: does the design breathe?

NOT validation (that is the multi-season battery). This answers the
go/no-go questions cheaply on the current season:
  1. split-half reliability of mean- and median-miss at the rendered unit
  2. independence: corr vs Loc+ raw (expect moderate), vs FF velo (expect ~0)
  3. face validity: best/worst MLB command, ROC leaders
  4. sanity: miss distributions, cells fitted, K usage

Everything here runs at SEP_MIN's placeholder 8in — nothing ships from this
script; the battery sweeps the constant.

Usage: python3 scripts/commandplus_firstlook.py
"""
import os, sys, math, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pipeline_locplus as lp
from commandplus_v1 import score_pitches, aggregate, eligible

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_DISPLAY = 300     # pitches for the first-look pitcher pools
MIN_HALF = 150


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    # EP appearances excluded by identity, matching every pitcher-facing view
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB'
           and (p.get('Pitcher'), p.get('PTeam')) not in ep]
    roc = [p for p in D if p.get('_source') in ('ROC', 'AAA')
           and (p.get('Pitcher'), p.get('PTeam')) not in ep]

    # ── full-season scoring, MLB + ROC scored separately (own targets) ──
    m_mlb, _pt, ncells = score_pitches(mlb)
    m_roc, _ptr, ncells_r = score_pitches(roc)
    agg_mlb = aggregate(m_mlb, MIN_DISPLAY)
    agg_roc = aggregate(m_roc, 150)
    print(f"MLB: {ncells} cells fitted, {len(agg_mlb)} pitchers >= {MIN_DISPLAY}",
          file=sys.stderr)
    print(f"ROC: {ncells_r} cells fitted, {len(agg_roc)} pitchers >= 150",
          file=sys.stderr)

    means = [v[0] for v in agg_mlb.values()]
    print()
    print(f"MLB mean-miss distribution: mean {np.mean(means):.2f}in, "
          f"sd {np.std(means):.2f}, p10 {np.percentile(means, 10):.2f}, "
          f"p90 {np.percentile(means, 90):.2f}")

    # ── objective 1: split-half reliability (odd/even game dates) ──
    dates = sorted({p.get('Game Date') for p in mlb if p.get('Game Date')})
    par = {d: i % 2 for i, d in enumerate(dates)}
    halves = []
    for h in (0, 1):
        sub = [p for p in mlb if par.get(p.get('Game Date')) == h]
        mh, _p2, _n2 = score_pitches(sub)
        halves.append(aggregate(mh, MIN_HALF))
    keys = [k for k in halves[0] if k in halves[1]]
    rel_mean = pearson([halves[0][k][0] for k in keys], [halves[1][k][0] for k in keys])
    rel_med = pearson([halves[0][k][1] for k in keys], [halves[1][k][1] for k in keys])
    print(f"split-half reliability (n={len(keys)}): mean-miss r={rel_mean:.3f}, "
          f"median-miss r={rel_med:.3f}")

    # ── objective 2: independence ──
    base = [p for p in mlb if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(base, LG, SCALE)
    loc_acc = defaultdict(list)
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            loc_acc[(p.get('Pitcher'), p.get('Throws'))].append(v)
    loc = {k: sum(v) / len(v) for k, v in loc_acc.items() if len(v) >= MIN_DISPLAY}
    velo = defaultdict(list)
    for p in mlb:
        if p.get('Pitch Type') == 'FF':
            v = lp.safe_float(p.get('Velocity'))
            if v is not None:
                velo[(p.get('Pitcher'), p.get('Throws'))].append(v)
    velo = {k: sum(v) / len(v) for k, v in velo.items() if len(v) >= 50}

    kk = [k for k in agg_mlb if k in loc]
    r_loc_mean = pearson([agg_mlb[k][0] for k in kk], [loc[k] for k in kk])
    r_loc_med = pearson([agg_mlb[k][1] for k in kk], [loc[k] for k in kk])
    kv = [k for k in agg_mlb if k in velo]
    r_velo = pearson([agg_mlb[k][0] for k in kv], [velo[k] for k in kv])
    print(f"corr(mean-miss, Loc+ raw) = {r_loc_mean:+.3f}  (raw_loc is hitter-"
          f"perspective: positive here means bigger misses = worse locations)")
    print(f"corr(median-miss, Loc+ raw) = {r_loc_med:+.3f}")
    print(f"corr(mean-miss, FF velo) = {r_velo:+.3f}  (expect ~0)")

    # ── objective 3: face validity ──
    print()
    ranked = sorted(agg_mlb.items(), key=lambda kv2: kv2[1][0])
    print("BEST command (smallest mean miss):")
    for k, (m, md, n) in ranked[:10]:
        print(f"  {k[0]:<24s} {m:5.2f}in  (med {md:4.2f}, n={n})")
    print("WORST command:")
    for k, (m, md, n) in ranked[-10:]:
        print(f"  {k[0]:<24s} {m:5.2f}in  (med {md:4.2f}, n={n})")
    print()
    print("ROC best command:")
    for k, (m, md, n) in sorted(agg_roc.items(), key=lambda kv2: kv2[1][0])[:5]:
        print(f"  {k[0]:<24s} {m:5.2f}in  (med {md:4.2f}, n={n})")


if __name__ == '__main__':
    main()
