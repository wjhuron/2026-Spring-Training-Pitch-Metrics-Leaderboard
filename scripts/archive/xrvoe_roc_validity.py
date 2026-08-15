"""xrvoe_roc_validity.py — should RVOE/xRVOE be unhidden for ROC?

The RVOE family is rocHide'd on every leaderboard. compute_xrvoe already
SCORES ROC (translation framing, 2026-07-25) — the flags are the only thing
between those numbers and the site. Before unhiding, two validity questions:

1. CURRENCY. The trainer reads the pickle directly, so ROC rv_raw/target_xrv
   are in MiLB run currency (~1.27x MLB) while the EXPECTATION side (MLB-
   trained stuff model + MLB Loc surfaces + MLB stacking betas) is in MLB
   currency. The residual subtracts across currencies. process_data corrects
   RunExp in-memory for its own consumers, but the trainer is a separate
   process reading the raw cache — it never sees the fix. Measured here by
   running the production compute_xrvoe twice, with and without applying
   process_data.compute_runexp_scale to the ROC pitches first.

2. LEAGUE-GAP BIAS. Even in the right currency, ROC actuals come against AAA
   hitters while expectations encode MLB hitter behavior. If AAA hitters are
   systematically easier than the MLB expectation at the same stuff/location,
   every ROC pitcher inherits a positive offset that reads as "outperformance"
   but is really the league gap. Diagnosed by the MEAN of raw ROC xRVOE/100 vs
   the MLB mean (~0 by construction of the stacking fit).

Also reported: spread comparison and a small-n split-half reliability for ROC
(honest caveat: ~15-25 ROC pitchers clear the floor, so that estimate is
noisy — the bias check is the decisive, well-powered part).

DECISION RULE (pre-registered):
  - |ROC mean bias| (currency-corrected) <= ~0.15 runs/100 (under half the
    xrvoe100 display SD) AND spread not degenerate -> recommend unhiding,
    with the currency fix applied in the trainer first.
  - Large bias -> keep hidden; the number would be a league-gap artifact.
    (Re-centering on the ROC mean would fix the display but change the
    metric's meaning for ROC — flag as an option, don't assume.)

Usage: python3 scripts/xrvoe_roc_validity.py
"""
import os, sys, math, pickle, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))

import numpy as np
import stuff_plus.train_stuff as T
from pipeline.process_data import compute_runexp_scale, runexp_factor
from pipeline.utils import safe_float

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
MIN_RELIAB = 100      # pitches per half for the ROC reliability estimate


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


def stuff_raw_oof(df, bundle):
    """MLB rows: OOF via the bundle's fold map (production path). Negated to
    pitcher-positive exactly as _oof_predict does."""
    X = T.design(df).reindex(columns=bundle['features'], fill_value=0)
    fold_of = {p: k for k, ps in enumerate(bundle['fold_pitchers']) for p in ps}
    pf = np.array([fold_of.get(p, 0) for p in df['pitcher'].values])
    out = np.full(len(X), np.nan)
    for k, mm in enumerate(bundle['fold_models']):
        mask = pf == k
        if mask.any():
            out[mask] = mm.predict(X[mask])
    return -out


def stuff_raw_roc(df, bundle):
    """ROC rows: full model where arm angle exists, no-arm companion otherwise
    (the trainer's per-pitch _arm mask)."""
    out = np.full(len(df), np.nan)
    arm = df['arm_angle'].notna().values
    if arm.any():
        Xr = T.design(df[arm]).reindex(columns=bundle['features'], fill_value=0)
        out[arm] = -bundle['model'].predict(Xr)
    if (~arm).any():
        Xn = T.design(df[~arm], T.NOARM_FEATS).reindex(
            columns=bundle['features_na'], fill_value=0)
        out[~arm] = -bundle['model_na'].predict(Xn)
    return out


def run_variant(mlb_pitches, roc_pitches, bundle, label):
    """Production compute_xrvoe on prepared frames. Returns (pt, ov) dicts."""
    df = T.build_df(mlb_pitches)
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    df['stuff_raw'] = stuff_raw_oof(df, bundle)
    df = df[df['stuff_raw'].notna()].reset_index(drop=True)

    roc_df = T.build_df(roc_pitches)
    roc_df = roc_df[roc_df['target_xrv'].notna()].reset_index(drop=True)
    roc_df['stuff_raw'] = stuff_raw_roc(roc_df, bundle)
    roc_df = roc_df[roc_df['stuff_raw'].notna()].reset_index(drop=True)

    print(f"  [{label}] mlb frame {len(df)}, roc frame {len(roc_df)}",
          file=sys.stderr)
    pt, ov = T.compute_xrvoe(df, mlb_pitches, roc_df=roc_df,
                             roc_pitches=roc_pitches)
    del df, roc_df
    gc.collect()
    return pt, ov


def summarize(ov, roc_names, tag):
    mlb_raw, roc_raw = [], []
    for key, rec in ov.items():
        n = rec.get('n', 0)
        if n < 150 or 'xrvoe' not in rec:
            continue
        raw100 = 100.0 * rec['xrvoe'] / n
        (roc_raw if key[0] in roc_names else mlb_raw).append(raw100)
    def stats(v):
        if not v:
            return (float('nan'),) * 3 + (0,)
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        med = sorted(v)[len(v) // 2]
        return m, med, sd, len(v)
    mm, mmed, msd, mn = stats(mlb_raw)
    rm, rmed, rsd, rn = stats(roc_raw)
    print(f"  {tag:>12s} | MLB mean {mm:+.3f} med {mmed:+.3f} sd {msd:.3f} "
          f"(n={mn}) | ROC mean {rm:+.3f} med {rmed:+.3f} sd {rsd:.3f} (n={rn})")
    return rm, rsd, rn


def main():
    print("loading...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    bundle = pickle.load(open(os.path.join(ROOT, 'stuff_plus',
                                           'stuff_models.pkl'), 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    roc = [p for p in D if p.get('_source') in ('ROC', 'AAA')]
    roc_names = {p.get('Pitcher') for p in roc}
    print(f"mlb {len(mlb)}, roc/aaa {len(roc)} pitches", file=sys.stderr)

    # currency-corrected copy of the ROC pitches (MLB untouched: factor n/a)
    scale = compute_runexp_scale(D)
    roc_fixed = []
    for p in roc:
        q = dict(p)
        sc = scale.get(p.get('_source'))
        v = safe_float(p.get('RunExp'))
        if sc and v is not None:
            f = runexp_factor(sc, p.get('Description'), p.get('Count'))
            if f:
                q['RunExp'] = v / f
        roc_fixed.append(q)

    print()
    print("RAW per-pitcher xRVOE/100 (unshrunk, >=150 pitches):")
    run_a = run_variant(mlb, roc, bundle, 'as-production (uncorrected ROC currency)')
    bias_a = summarize(run_a[1], roc_names, 'uncorrected')
    run_b = run_variant(mlb, roc_fixed, bundle, 'currency-corrected ROC')
    bias_b = summarize(run_b[1], roc_names, 'corrected')

    # split-half reliability for ROC (corrected variant), odd/even dates
    dates = sorted({p.get('Game Date') for p in roc_fixed if p.get('Game Date')})
    par = {d: i % 2 for i, d in enumerate(dates)}
    halves = []
    for h in (0, 1):
        roc_h = [p for p in roc_fixed if par.get(p.get('Game Date')) == h]
        _, ov_h = run_variant(mlb, roc_h, bundle, f'half{h}')
        raw = {k[0]: 100.0 * r['xrvoe'] / r['n'] for k, r in ov_h.items()
               if k[0] in roc_names and r.get('n', 0) >= MIN_RELIAB and 'xrvoe' in r}
        halves.append(raw)
    keys = [k for k in halves[0] if k in halves[1]]
    rel = pearson([halves[0][k] for k in keys], [halves[1][k] for k in keys])

    print()
    print(f"currency effect on ROC mean: {bias_a[0]:+.3f} -> {bias_b[0]:+.3f} "
          f"runs/100 (shift {bias_b[0]-bias_a[0]:+.3f})")
    print(f"ROC split-half reliability (corrected, n={len(keys)} pitchers, "
          f">= {MIN_RELIAB}/half): r = {rel if rel is None else round(rel,3)}")
    print()
    print("DECISION RULE: |corrected ROC mean| <= ~0.15 runs/100 and sane spread")
    print("-> unhide WITH the trainer currency fix. Large bias -> keep hidden")
    print("(league-gap artifact; re-centering is an option but changes meaning).")


if __name__ == '__main__':
    main()
