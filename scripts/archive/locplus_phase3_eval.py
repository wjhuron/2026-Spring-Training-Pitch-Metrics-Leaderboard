"""locplus_phase3_eval.py — 3-objective A/B for two untested Loc+ ideas.

Same harness contract as phase2_locplus_eval.py (reliability / stuff-
independence / predictive validity), run against the CURRENT live config
(PCS_BY_HAND=1, SWING_PRIOR_COUNT_LEVEL=1, CS_COUNT_TRANSFORM=1) as control.

IDEA 1 — count-mix post-stratification (aggregation change, no surface change).
  Loc+ currently averages ExpRV over a pitcher's pitches, so his COUNT MIX
  leaks in: living in 0-2 is a stuff/sequencing effect, not location skill.
  Post-strat computes a per-count mean and recombines with LEAGUE count
  weights. This is NOT count-demeaning (already proven a trap) — within-count
  level information is fully preserved; only the mixing weights change.
  If it works it also removes the mechanism that killed BIP_COUNT_ANCHOR
  (which made ExpRV strongly count-mix dependent), so the anchor is re-tested
  underneath it — that would fix the currency mismatch where four of the five
  outcome values are count-specific dRE and the BIP branch is not.

IDEA 2 — per-batter median strike zone.
  _znorm divides by the PER-PITCH SzTop/SzBot, which are re-estimated from the
  batter's stance every pitch and are jittery. That jitter is worst in the
  shadow zone, which is exactly where location value concentrates. Swap in a
  per-batter median zone and see if the z-coordinate gets cleaner.

Sample is held IDENTICAL across configs (pitches scorable under both znorm
definitions), so nothing here is a sample-composition artifact.

Usage: python3 scripts/locplus_phase3_eval.py
"""
import os, sys, pickle, math, statistics, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

MIN_FULL = 250      # pitches to enter the full-sample / predictive pools
MIN_HALF = 125      # pitches per half for split-half reliability

_ORIG_ZNORM = lp._znorm
MEDZ = {}           # Batter -> (median SzTop, median SzBot)


def znorm_median(p):
    pz = lp.safe_float(p.get('PlateZ'))
    tb = MEDZ.get(p.get('Batter'))
    if pz is None or tb is None:
        return None
    top, bot = tb
    if top <= bot:
        return None
    return (pz - bot) / (top - bot)


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


# ── aggregators ─────────────────────────────────────────────────────────
def agg_plain(scored):
    """scored = list of (count, ExpRV). Current live behavior."""
    return sum(v for _c, v in scored) / len(scored)


def make_agg_poststrat(league_w):
    def agg(scored):
        per = defaultdict(lambda: [0.0, 0])
        for c, v in scored:
            per[c][0] += v; per[c][1] += 1
        tot = sum(league_w.get(c, 0.0) for c in per)
        if tot <= 0:
            return sum(v for _c, v in scored) / len(scored)
        return sum(league_w.get(c, 0.0) / tot * (s / n) for c, (s, n) in per.items())
    return agg


def score_by_pitcher(byp, S, aggs, min_n):
    """Score every pitcher once, then aggregate under each named aggregator."""
    out = {name: {} for name in aggs}
    for k, ps in byp.items():
        scored = []
        for p in ps:
            v = lp.score_pitch(p, S)
            if v is not None:
                scored.append((lp.get_count(p), v))
        if len(scored) < min_n:
            continue
        for name, fn in aggs.items():
            out[name][k] = fn(scored)
    return out


# ── one full config evaluation ──────────────────────────────────────────
def evaluate(label, base, ctx, use_median_zone, bip_anchor):
    lp.BIP_COUNT_ANCHOR = bip_anchor
    lp._znorm = znorm_median if use_median_zone else _ORIG_ZNORM

    aggs = {'plain': agg_plain, 'postStrat': make_agg_poststrat(ctx['league_w'])}
    t0 = time.time()

    S_full = lp.build_surfaces(base, LG, SCALE)
    full = score_by_pitcher(ctx['by_p'], S_full, aggs, MIN_FULL)

    # objective 1 — odd/even split-half reliability (surfaces rebuilt per half)
    halves = []
    for h in (0, 1):
        sub = ctx['halves'][h]
        S_h = lp.build_surfaces(sub, LG, SCALE)
        halves.append(score_by_pitcher(ctx['byp_halves'][h], S_h, aggs, MIN_HALF))

    # objective 3 — first-half score vs second-half ACTUAL xRV allowed
    S_e = lp.build_surfaces(ctx['early'], LG, SCALE)
    score_e = score_by_pitcher(ctx['byp_early'], S_e, aggs, MIN_FULL)

    rows = []
    for name in aggs:
        keys = [k for k in halves[0][name] if k in halves[1][name]]
        rel = pearson([halves[0][name][k] for k in keys],
                      [halves[1][name][k] for k in keys])
        kw = [k for k in full[name] if k in ctx['whiff']]
        rw = pearson([full[name][k] for k in kw], [ctx['whiff'][k] for k in kw])
        kv = [k for k in full[name] if k in ctx['ffv']]
        rv_ = pearson([full[name][k] for k in kv], [ctx['ffv'][k] for k in kv])
        kp = [k for k in score_e[name] if k in ctx['actual_l']]
        pred = pearson([score_e[name][k] for k in kp],
                       [ctx['actual_l'][k] for k in kp])
        rows.append((f"{label}/{name}", rel,
                     abs(rw) if rw is not None else None,
                     abs(rv_) if rv_ is not None else None,
                     pred, len(keys), len(kp)))
    print(f"   ({time.time() - t0:.0f}s)", file=sys.stderr)
    return rows


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))

    # per-batter median zone (from ALL rows with a valid zone, before filtering)
    tops, bots = defaultdict(list), defaultdict(list)
    for p in D:
        t = lp.safe_float(p.get('SzTop')); b = lp.safe_float(p.get('SzBot'))
        if t is not None and b is not None and t > b:
            tops[p.get('Batter')].append(t); bots[p.get('Batter')].append(b)
    for bt in tops:
        MEDZ[bt] = (statistics.median(tops[bt]), statistics.median(bots[bt]))
    print(f"median zones for {len(MEDZ)} batters", file=sys.stderr)

    # IDENTICAL sample for every config: scorable under BOTH znorm definitions
    lp._znorm = _ORIG_ZNORM
    cand = [p for p in D if lp.is_eligible_baseline(p)]
    lp._znorm = znorm_median
    base = [p for p in cand if lp.is_eligible_baseline(p)]
    lp._znorm = _ORIG_ZNORM
    print(f"baseline pitches: {len(base)} (dropped {len(cand) - len(base)} "
          f"for zone-key mismatch)", file=sys.stderr)

    by_p = defaultdict(list)
    for p in base:
        by_p[(p.get('Pitcher'), p.get('Throws'))].append(p)

    # league count weights for post-stratification
    cnt_n = defaultdict(int)
    for p in base:
        cnt_n[lp.get_count(p)] += 1
    tot = sum(cnt_n.values())
    league_w = {c: n / tot for c, n in cnt_n.items()}

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

    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    parity = {d: i % 2 for i, d in enumerate(dates)}
    mid = dates[len(dates) // 2]

    halves = [[p for p in base if parity.get(p.get('Game Date')) == h] for h in (0, 1)]
    byp_halves = []
    for h in (0, 1):
        d = defaultdict(list)
        for p in halves[h]:
            d[(p.get('Pitcher'), p.get('Throws'))].append(p)
        byp_halves.append(d)

    early = [p for p in base if p.get('Game Date') and p.get('Game Date') < mid]
    late = [p for p in base if p.get('Game Date') and p.get('Game Date') >= mid]
    byp_early = defaultdict(list)
    for p in early:
        byp_early[(p.get('Pitcher'), p.get('Throws'))].append(p)

    rv_fn = make_rv_xrv(LG, SCALE)
    byp_l = defaultdict(list)
    for p in late:
        byp_l[(p.get('Pitcher'), p.get('Throws'))].append(p)
    actual_l = {}
    for k, ps in byp_l.items():
        vals = [v for v in (rv_fn(p) for p in ps) if v is not None]
        if len(vals) >= MIN_FULL:
            actual_l[k] = sum(vals) / len(vals)

    ctx = {'by_p': by_p, 'halves': halves, 'byp_halves': byp_halves,
           'early': early, 'byp_early': byp_early, 'actual_l': actual_l,
           'whiff': whiff, 'ffv': ffv, 'league_w': league_w}

    configs = [
        ('live',       False, False),   # control: current shipped model
        ('bipAnchor',  False, True),    # re-test the anchor under CS-transform
        ('medZone',    True,  False),   # idea 2
        ('medZone+bip', True, True),    # idea 2 + anchor (only informative if
                                        # post-strat rescues the anchor)
    ]
    rows = []
    for label, mz, bip in configs:
        print(f"config {label}...", file=sys.stderr)
        rows += evaluate(label, base, ctx, mz, bip)

    lp._znorm = _ORIG_ZNORM
    print()
    print(f"{'config/agg':>22s} {'rel_r':>7s} {'|r|whf':>7s} {'|r|velo':>8s} "
          f"{'pred_r':>7s} {'n_rel':>6s} {'n_pred':>7s}")
    print('-' * 70)

    def f(x):
        return f"{x:.3f}" if x is not None else "  n/a"
    for name, rel, rw, rv_, pred, nrel, npred in rows:
        print(f"{name:>22s} {f(rel):>7s} {f(rw):>7s} {f(rv_):>8s} "
              f"{f(pred):>7s} {nrel:>6d} {npred:>7d}")
    print()
    print("rel_r: higher = better.  |r|whf, |r|velo: LOWER = better (it's a "
          "location metric).  pred_r: higher = better (both raw_loc and the "
          "target are hitter-perspective, so the correlation is positive).")


if __name__ == '__main__':
    main()
