"""locplus_groupbw_multiseason.py — per-group horizontal bandwidth for Loc+,
on a per-group objective, across season replicates 2021-2025.

History: a 2026 per-group bandwidth pass was discarded because its objective
(pitcher-overall reliability) is gameable by over-smoothing and dilutes any
single group's effect. This sweep fixes both problems:
  - PHYS_BW_PT = {group: (x_bw, 0.22)} varies ONE group at a time,
    x in {3.5, 5.5} vs shipped 4.5 (z stays 0.22 everywhere).
  - Objectives are GROUP-RESTRICTED: only pitches of the varied group score.
      rel_g   split-half (odd/even dates) r of the pitcher's group Loc+ mean
              (>= 50 group pitches per half)
      pred_g  1st-quarter group Loc+ vs 2nd-quarter group xRV mean
              (>= 60 group pitches per side), partialled on FF velo
Ship bar: a bandwidth override for group g must beat shipped on pred_g in
most of the five seasons without a rel_g collapse. Monotone improvement to
an edge (3.5 or 5.5 winning everywhere) means extend the grid before
concluding.

Usage: python3 scripts/research/locplus/locplus_groupbw_multiseason.py
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

SWEEP_GROUPS = ('FF', 'SI', 'FC', 'SL', 'CU', 'CH')
BWS = (3.5, 5.5)          # vs shipped 4.5
MIN_REL_G = 50
MIN_PRED_G = 60


def group_maps(seg_pitches, S, rv_fn, gset):
    """Per (pitcher, group): mean Loc+ score and mean actual xRV, plus FF
    velo per pitcher — restricted to pitches of groups in gset."""
    loc = defaultdict(list)
    act = defaultdict(list)
    velo = defaultdict(list)
    for p in seg_pitches:
        k = (p.get('Pitcher'), p.get('Throws'))
        g = lp.group_of(p)
        if p['Pitch Type'] == 'FF':
            v = lp.safe_float(p['Velocity'])
            if v is not None:
                velo[k].append(v)
        if g not in gset:
            continue
        s = lp.score_pitch(p, S)
        if s is not None:
            loc[(k, g)].append(s)
        a = rv_fn(p)
        if a is not None:
            act[(k, g)].append(a)
    return loc, act, {k: sum(v) / len(v) for k, v in velo.items() if len(v) >= base.MIN_VELO}


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

    res = {}          # (group, bwname) -> {'rel': r, 'pred': r}
    configs = [('shipped', None, None)] + [
        (f'{g}_{bw}', g, bw) for g in SWEEP_GROUPS for bw in BWS]

    for name, g, bw in configs:
        lp.PHYS_BW_PT = {} if g is None else {g: (bw, lp.PHYS_Z_FRAC)}
        try:
            gset = set(SWEEP_GROUPS) if g is None else {g}
            # predictive: quarters A1->A2 and B1->B2
            preds = defaultdict(list)
            for f, s in (('A1', 'A2'), ('B1', 'B2')):
                S = lp.build_surfaces(seg[f], base.LG, base.SCALE)
                locf, _, velof = group_maps(seg[f], S, rv_fn, gset)
                _, acts, _ = group_maps(seg[s], S, rv_fn, gset)
                by_g = defaultdict(lambda: ([], [], []))
                for (k, gg), vs in locf.items():
                    if len(vs) < MIN_PRED_G:
                        continue
                    a = acts.get((k, gg))
                    if a is None or len(a) < MIN_PRED_G or k not in velof:
                        continue
                    xs, ys, vv = by_g[gg]
                    xs.append(sum(vs) / len(vs))
                    ys.append(sum(a) / len(a))
                    vv.append(velof[k])
                for gg, (xs, ys, vv) in by_g.items():
                    if len(xs) < 30:
                        continue
                    r_ly = base.pearson(xs, ys)
                    r_ls = base.pearson(xs, vv)
                    r_sy = base.pearson(vv, ys)
                    pr = base.partial(r_ly, r_ls, r_sy)
                    if pr is not None:
                        preds[gg].append(pr)
            # reliability: odd/even
            S0 = lp.build_surfaces(g0, base.LG, base.SCALE)
            S1 = lp.build_surfaces(g1, base.LG, base.SCALE)
            l0, _, _ = group_maps(g0, S0, rv_fn, gset)
            l1, _, _ = group_maps(g1, S1, rv_fn, gset)
            rels = {}
            for gg in gset:
                xs, ys = [], []
                for (k, g2), vs in l0.items():
                    if g2 != gg or len(vs) < MIN_REL_G:
                        continue
                    w = l1.get((k, gg))
                    if w is None or len(w) < MIN_REL_G:
                        continue
                    xs.append(sum(vs) / len(vs))
                    ys.append(sum(w) / len(w))
                rels[gg] = base.pearson(xs, ys)
            for gg in gset:
                pv = preds.get(gg)
                res[(name, gg)] = {
                    'pred': (sum(pv) / len(pv)) if pv else None,
                    'rel': rels.get(gg),
                }
        finally:
            lp.PHYS_BW_PT = {}
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
            print(f"{yr}: cache missing, skipped", flush=True)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        print(f"  {yr}: {len(pitches)} usable", file=sys.stderr)
        table[yr] = eval_season(pitches, rv_fn)
        for g in SWEEP_GROUPS:
            s = table[yr].get(('shipped', g), {})
            row = f"{yr} {g}: shipped pred {s.get('pred') if s.get('pred') is None else round(s['pred'],3)} rel {s.get('rel') if s.get('rel') is None else round(s['rel'],3)}"
            for bw in BWS:
                v = table[yr].get((f'{g}_{bw}', g), {})
                dp = (v['pred'] - s['pred']) if (v.get('pred') is not None and s.get('pred') is not None) else None
                dr = (v['rel'] - s['rel']) if (v.get('rel') is not None and s.get('rel') is not None) else None
                row += f" | {bw}: dpred {dp if dp is None else format(dp, '+.3f')} drel {dr if dr is None else format(dr, '+.3f')}"
            print(row, flush=True)
        del pitches
        gc.collect()

    print("\nVERDICT — per group, per bandwidth: seasons won on pred_g (vs shipped)")
    for g in SWEEP_GROUPS:
        line = f"{g:>3s}:"
        for bw in BWS:
            wins = tot = 0
            dsum = 0.0
            for yr in sorted(table):
                s = table[yr].get(('shipped', g), {})
                v = table[yr].get((f'{g}_{bw}', g), {})
                if s.get('pred') is None or v.get('pred') is None:
                    continue
                tot += 1
                d = v['pred'] - s['pred']
                dsum += d
                wins += 1 if d > 0 else 0
            line += (f"   x={bw}: {wins}/{tot} (mean {dsum/tot:+.3f})" if tot
                     else f"   x={bw}: no data")
        print(line, flush=True)
    print("\nShipped 4.5 stands for a group unless an override wins most seasons")
    print("on pred_g without a rel_g collapse. An edge win (3.5 or 5.5 winning")
    print("everywhere) requires extending the grid before adopting.")


if __name__ == '__main__':
    main()
