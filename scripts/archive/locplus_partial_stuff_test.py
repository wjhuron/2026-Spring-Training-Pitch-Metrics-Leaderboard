"""locplus_partial_stuff_test.py — is the prediction gain LOCATION signal, or
absorbed STUFF?

The predictive objective used throughout the bandwidth/shrinkage search is
"first-period Loc+ vs next-period actual xRV allowed." That objective is NOT
neutral, which I got wrong earlier: stuff predicts future runs allowed, so a
"location" metric that quietly absorbs stuff scores BETTER on it while being a
WORSE location metric. Any config that raises prediction and stuff-leak
together is therefore ambiguous on its face.

This resolves it. For each config, hold the pitcher's STUFF fixed and ask
whether Loc+ still predicts:

    partial r(loc, future xRV | stuff)
      = (r_ly - r_ls*r_sy) / sqrt((1 - r_ls^2)(1 - r_sy^2))

Stuff is measured in the SAME window the location score comes from (never the
target window), so this asks the decision-relevant question: given what we
already knew about his stuff, does his location score add anything?

Two stuff proxies, because they fail differently:
  Stuff+   the real thing — a fitted model of pitch quality from physical
           characteristics. Preferred.
  FF velo  cruder, but model-independent, so it cannot share fitting artifacts
           with whatever Loc+ is doing.

READ IT THIS WAY:
  raw pred up, partial pred up      -> genuine location signal. Ship.
  raw pred up, partial pred flat/down -> the gain was stuff absorption. Reject,
                                       no matter how good the raw number looks.

Usage: python3 scripts/locplus_partial_stuff_test.py
"""
import os, sys, math, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393
MIN_SCORE, MIN_ACTUAL = 150, 150

KEYS = ['K_WHIFF', 'K_FOUL', 'K_XWCON', 'K_SWING_COLL', 'K_SWING_COUNT', 'K_CS']
ORIG = {k: getattr(lp, k) for k in KEYS}

CONFIGS = [
    ('shipped',      4.5, {k: 1 for k in KEYS}),
    ('uniform x8',   1.5, {k: 8 for k in KEYS}),
    ('per-surface combo', 1.5, {'K_WHIFF': 32, 'K_FOUL': 2, 'K_XWCON': 8,
                                'K_SWING_COLL': 32, 'K_SWING_COUNT': 2, 'K_CS': 2}),
    ('swingcoll x32 only', 1.5, {'K_WHIFF': 8, 'K_FOUL': 8, 'K_XWCON': 8,
                                 'K_SWING_COLL': 32, 'K_SWING_COUNT': 8, 'K_CS': 8}),
]


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


def partial(r_ly, r_ls, r_sy):
    d = (1 - r_ls ** 2) * (1 - r_sy ** 2)
    if d <= 0:
        return None
    return (r_ly - r_ls * r_sy) / math.sqrt(d)


def by_pitcher(pitches):
    d = defaultdict(list)
    for p in pitches:
        d[(p.get('Pitcher'), p.get('Throws'))].append(p)
    return d


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]
    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in base:
        d = p.get('Game Date')
        if d is None:
            continue
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)

    rv_fn = make_rv_xrv(LG, SCALE)
    ctx = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        byp_f = by_pitcher(seg[f])
        actual = {}
        for k, ps in by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        # stuff proxies measured in the SCORING window only
        stuff, velo = {}, {}
        for k, ps in byp_f.items():
            sp = [x for x in (lp.safe_float(p.get('Stuff+')) for p in ps) if x is not None]
            if len(sp) >= 100:
                stuff[k] = sum(sp) / len(sp)
            v = [x for x in (lp.safe_float(p.get('Velocity')) for p in ps
                             if p.get('Pitch Type') == 'FF') if x is not None]
            if len(v) >= 40:
                velo[k] = sum(v) / len(v)
        ctx[half] = {'first': seg[f], 'byp_f': byp_f, 'actual': actual,
                     'stuff': stuff, 'velo': velo}
        print(f"half {half}: {len(actual)} with actual, {len(stuff)} with Stuff+, "
              f"{len(velo)} with FF velo", file=sys.stderr)

    print()
    print(f"{'config':>20s} {'half':>5s} | {'raw':>6s} | {'r(loc,':>7s} "
          f"{'partial':>8s} | {'r(loc,':>7s} {'partial':>8s}")
    print(f"{'':>20s} {'':>5s} | {'pred':>6s} | {'stuff)':>7s} {'|stuff':>8s} "
          f"| {'velo)':>7s} {'|velo':>8s}")
    print('-' * 72)
    summary = defaultdict(dict)
    for name, x, facs in CONFIGS:
        lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = 0.22
        lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(0.22 / lp.BIN_Z)
        for k in KEYS:
            setattr(lp, k, ORIG[k] * facs[k])
        for half in ('A', 'B'):
            c = ctx[half]
            S = lp.build_surfaces(c['first'], LG, SCALE)
            loc = {}
            for k, ps in c['byp_f'].items():
                v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
                if len(v) >= MIN_SCORE:
                    loc[k] = sum(v) / len(v)
            out = [name, half]
            ks = [k for k in loc if k in c['actual']]
            r_ly = pearson([loc[k] for k in ks], [c['actual'][k] for k in ks])
            row = {'raw': r_ly}
            for proxy in ('stuff', 'velo'):
                kk = [k for k in loc if k in c['actual'] and k in c[proxy]]
                r_ly2 = pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
                r_ls = pearson([loc[k] for k in kk], [c[proxy][k] for k in kk])
                r_sy = pearson([c[proxy][k] for k in kk], [c['actual'][k] for k in kk])
                row['r_l' + proxy] = r_ls
                row['p_' + proxy] = partial(r_ly2, r_ls, r_sy)
            summary[name][half] = row
            print(f"{name:>20s} {half:>5s} | {row['raw']:>6.3f} | "
                  f"{row['r_lstuff']:>+7.3f} {row['p_stuff']:>8.3f} | "
                  f"{row['r_lvelo']:>+7.3f} {row['p_velo']:>8.3f}", flush=True)

    print()
    print("VERDICT — mean over halves, vs shipped:")
    sh = summary['shipped']
    base_raw = (sh['A']['raw'] + sh['B']['raw']) / 2
    base_ps = (sh['A']['p_stuff'] + sh['B']['p_stuff']) / 2
    for name in summary:
        m = summary[name]
        raw = (m['A']['raw'] + m['B']['raw']) / 2
        ps = (m['A']['p_stuff'] + m['B']['p_stuff']) / 2
        pv = (m['A']['p_velo'] + m['B']['p_velo']) / 2
        print(f"  {name:>20s}: raw {raw:+.3f} ({raw-base_raw:+.3f})   "
              f"partial|stuff {ps:.3f} ({ps-base_ps:+.3f})   partial|velo {pv:.3f}")
    print()
    print("If a config's RAW gain is large but its PARTIAL|stuff gain is not,")
    print("the metric absorbed stuff rather than measuring location better.")


if __name__ == '__main__':
    main()
