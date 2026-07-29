"""phase2_locplus_eval.py — 3-objective A/B for the Phase 2 Loc+ upgrades.

Tests the three model options in pipeline_locplus (PCS_BY_HAND,
BIP_COUNT_ANCHOR, SWING_PRIOR_COUNT_LEVEL): baseline, each alone, all on.

Objectives (same framework the v2 lock used):
  1. reliability  — odd/even-date split-half r of per-pitcher raw_loc
                    (surfaces rebuilt per half), pitchers >= 125/half
  2. stuff-indep  — |r| of full-sample raw_loc vs pitcher whiff% and FF velo
                    (>= 250 pitches)
  3. predictive   — chronological first-half raw_loc vs second-half actual
                    per-pitch xRV allowed (>= 250 pitches each half)

Usage: python3 scripts/phase2_locplus_eval.py
"""
import os, sys, pickle, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393


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


def raw_loc(pitches_by_p, S, min_n=1):
    out = {}
    for k, ps in pitches_by_p.items():
        vals = [v for v in (lp.score_pitch(p, S) for p in ps if lp._is_scorable(p))
                if v is not None]
        if len(vals) >= min_n:
            out[k] = (sum(vals) / len(vals), len(vals))
    return out


def main():
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    by_p = defaultdict(list)
    for p in base:
        by_p[(p.get('Pitcher'), p.get('Throws'))].append(p)

    # stuff proxies
    whiff, ffv = {}, {}
    for k, ps in by_p.items():
        sw = [p for p in ps if p.get('Description') in lp.SWING_DESC]
        wh = [p for p in sw if p.get('Description') == 'Swinging Strike']
        if len(sw) >= 100:
            whiff[k] = len(wh) / len(sw)
        v = [f for f in (lp.safe_float(p.get('Velocity')) for p in ps
                         if p.get('Pitch Type') == 'FF') if f is not None]
        if len(v) >= 50:
            ffv[k] = sum(v) / len(v)

    # actual xRV allowed per pitch (luck-neutral outcome value, no anchoring
    # so the target is identical across configs)
    rv_fn = make_rv_xrv(LG, SCALE)

    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    parity = {d: i % 2 for i, d in enumerate(dates)}          # reliability split
    mid = dates[len(dates) // 2]                              # chronological split

    import os as _os
    if _os.environ.get('LOCPLUS_EVAL_PAIR'):
        configs = [('pcs+swPrior', 1, 0, 1)]
    else:
        configs = [('baseline', 0, 0, 0), ('pcsHand', 1, 0, 0), ('bipAnchor', 0, 1, 0),
                   ('swPrior', 0, 0, 1), ('all3', 1, 1, 1)]
    print(f"{'config':>10s} {'rel_r':>7s} {'|r|whiff':>8s} {'|r|velo':>8s} {'pred_r':>7s}")
    for name, a, b, c in configs:
        lp.PCS_BY_HAND, lp.BIP_COUNT_ANCHOR, lp.SWING_PRIOR_COUNT_LEVEL = \
            bool(a), bool(b), bool(c)

        S_full = lp.build_surfaces(base, LG, SCALE)
        full = raw_loc(by_p, S_full, min_n=250)

        # objective 1: odd/even reliability
        halves = []
        for h in (0, 1):
            sub = [p for p in base if parity.get(p.get('Game Date')) == h]
            S_h = lp.build_surfaces(sub, LG, SCALE)
            byp_h = defaultdict(list)
            for p in sub:
                byp_h[(p.get('Pitcher'), p.get('Throws'))].append(p)
            halves.append(raw_loc(byp_h, S_h, min_n=125))
        keys = [k for k in halves[0] if k in halves[1]]
        rel = pearson([halves[0][k][0] for k in keys], [halves[1][k][0] for k in keys])

        # objective 2: stuff independence
        kw = [k for k in full if k in whiff]
        rw = pearson([full[k][0] for k in kw], [whiff[k] for k in kw])
        kv = [k for k in full if k in ffv]
        rv_ = pearson([full[k][0] for k in kv], [ffv[k] for k in kv])

        # objective 3: predictive (first half score -> second half actual xRV)
        early = [p for p in base if p.get('Game Date') and p.get('Game Date') < mid]
        late = [p for p in base if p.get('Game Date') and p.get('Game Date') >= mid]
        S_e = lp.build_surfaces(early, LG, SCALE)
        byp_e = defaultdict(list)
        for p in early:
            byp_e[(p.get('Pitcher'), p.get('Throws'))].append(p)
        score_e = raw_loc(byp_e, S_e, min_n=250)
        actual_l = {}
        byp_l = defaultdict(list)
        for p in late:
            byp_l[(p.get('Pitcher'), p.get('Throws'))].append(p)
        for k, ps in byp_l.items():
            vals = [v for v in (rv_fn(p) for p in ps) if v is not None]
            if len(vals) >= 250:
                actual_l[k] = sum(vals) / len(vals)
        kp = [k for k in score_e if k in actual_l]
        pred = pearson([score_e[k][0] for k in kp], [actual_l[k] for k in kp])

        def f(x):
            return f"{x:.3f}" if x is not None else "  n/a"
        print(f"{name:>10s} {f(rel):>7s} {f(abs(rw) if rw is not None else None):>8s} "
              f"{f(abs(rv_) if rv_ is not None else None):>8s} {f(pred):>7s} "
              f"(n: rel={len(keys)}, pred={len(kp)})")


if __name__ == '__main__':
    main()
