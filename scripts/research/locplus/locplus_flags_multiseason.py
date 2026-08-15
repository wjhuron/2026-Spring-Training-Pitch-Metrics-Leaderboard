"""locplus_flags_multiseason.py — do the Loc+ decomposition FLAGS transfer
across seasons?

Companion to locplus_constants_multiseason.py (which settled bandwidth + the
six shrinkage K, shipped won 5/5). The four structural on/off flags were
decided on 2026 only (scripts/archive/phase2_locplus_eval.py,
locplus_cs_transform_test.py, locplus_phase3_eval.py):

    PCS_BY_HAND = True              (rel .568 -> .575 on 2026)
    CS_COUNT_TRANSFORM = True       (rel .591 -> .602 on 2026)
    SWING_PRIOR_COUNT_LEVEL = True  (whiff leak .031 -> .019 on 2026)
    BIP_COUNT_ANCHOR = False        (rejected twice on 2026)

Here every season 2021-2025 builds its OWN surfaces (replicates, never
pooled) and each config toggles exactly one flag off the shipped set. Same
objectives as the constants harness: partial r(loc, next-quarter xRV | FF
velo), raw predictive r, and odd/even-date split-half reliability.

Ship bar: the shipped flag set must win (or hold within noise) in most of
the five replicate seasons it was never fitted on. A flag whose toggle wins
most seasons flags a config change.

Usage: python3 scripts/research/locplus/locplus_flags_multiseason.py
"""
import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from collections import defaultdict

FLAGS = ('PCS_BY_HAND', 'CS_COUNT_TRANSFORM', 'SWING_PRIOR_COUNT_LEVEL',
         'BIP_COUNT_ANCHOR')
SHIPPED = {'PCS_BY_HAND': True, 'CS_COUNT_TRANSFORM': True,
           'SWING_PRIOR_COUNT_LEVEL': True, 'BIP_COUNT_ANCHOR': False}

CONFIGS = [
    ('shipped', dict(SHIPPED)),
    ('pcs_pooled', {**SHIPPED, 'PCS_BY_HAND': False}),
    ('no_cs_trans', {**SHIPPED, 'CS_COUNT_TRANSFORM': False}),
    ('sw_collapsed', {**SHIPPED, 'SWING_PRIOR_COUNT_LEVEL': False}),
    ('bip_anchor', {**SHIPPED, 'BIP_COUNT_ANCHOR': True}),
]

ORIG_FLAGS = {k: getattr(lp, k) for k in FLAGS}


def eval_season(pitches, rv_fn):
    """Same segmentation and objectives as base.eval_season, but the configs
    toggle decomposition flags instead of bandwidth/K (which stay shipped)."""
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

    res = {}
    for name, flags in CONFIGS:
        for k, v in flags.items():
            setattr(lp, k, v)
        try:
            raws, parts, rlvs = [], [], []
            for half in ('A', 'B'):
                c = prep[half]
                S = lp.build_surfaces(c['first'], base.LG, base.SCALE)
                loc = base.score_map(c['byp_f'], S, base.MIN_SCORE)
                kk = [k for k in loc if k in c['actual'] and k in c['velo']]
                if len(kk) < 30:
                    continue
                r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
                r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
                r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
                raws.append(r_ly)
                parts.append(base.partial(r_ly, r_ls, r_sy))
                rlvs.append(r_ls)
            S0 = lp.build_surfaces(g0, base.LG, base.SCALE)
            S1 = lp.build_surfaces(g1, base.LG, base.SCALE)
            a0 = base.score_map(base.by_pitcher(g0), S0, base.MIN_REL)
            a1 = base.score_map(base.by_pitcher(g1), S1, base.MIN_REL)
            ks = [k for k in a0 if k in a1]
            rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
            res[name] = {'raw': sum(raws) / len(raws),
                         'partial': sum(parts) / len(parts),
                         'rlv': sum(rlvs) / len(rlvs), 'rel': rel, 'n': len(ks)}
        finally:
            for k, v in ORIG_FLAGS.items():
                setattr(lp, k, v)
    return res


def main():
    from pipeline.sdplus import make_rv_xrv
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    print(f"{'season':>7s} {'config':>13s} | {'PARTIAL|velo':>13s} | {'raw':>6s} "
          f"| {'r(velo)':>8s} | {'rel':>6s}")
    print('-' * 66)
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr:>7d}   cache missing, skipped", flush=True)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        print(f"  {yr}: {len(pitches)} usable pitches", file=sys.stderr)
        res = eval_season(pitches, rv_fn)
        table[yr] = res
        for name, _f in CONFIGS:
            o = res.get(name)
            if o:
                print(f"{yr:>7d} {name:>13s} | {o['partial']:>13.3f} | {o['raw']:>6.3f} "
                      f"| {o['rlv']:>+8.3f} | {o['rel']:>6.3f}", flush=True)
        del pitches
        gc.collect()

    for metric in ('partial', 'rel'):
        print()
        print(f"TRANSFER VERDICT — toggle vs shipped, on {metric}, per season")
        names = [c[0] for c in CONFIGS if c[0] != 'shipped']
        print(f"{'config':>13s} " + "".join(f"{yr:>9d}" for yr in sorted(table))
              + f"{'wins':>7s}")
        print('-' * (14 + 9 * len(table) + 7))
        for name in names:
            wins, cells = 0, ''
            for yr in sorted(table):
                s, p = table[yr].get('shipped'), table[yr].get(name)
                if not s or not p or s.get(metric) is None or p.get(metric) is None:
                    cells += f"{'-':>9s}"
                    continue
                d = p[metric] - s[metric]
                wins += 1 if d > 0 else 0
                cells += f"{d:>+9.3f}"
            print(f"{name:>13s} {cells}{wins:>4d}/{len(table)}")
    print()
    print("Positive = the TOGGLE beats the shipped flag set that season.")
    print("Shipped flags stand unless a toggle wins most replicate seasons.")


if __name__ == '__main__':
    main()
