"""locplus_fullseason_replicate.py — confirm a Loc+ constant set at the
PITCHER level, surfaces built on the FULL season (2026-09-02).

Companion to locplus_cellfit_sweep.py (the decider for the smoothing and
shrinkage constants). That sweep measures the surfaces themselves; this
script checks that a constant set which fits held-out cells better also
produces a better pitcher grade, on six independent seasons (2021-2026).

PROTOCOL. Per season, per config:
  1. Build the league surfaces ONCE on the full-season baseline
     (is_eligible_baseline), exactly as production does, with the config's
     constants monkeypatched into pipeline.locplus (including _KX/_KZ).
  2. Score every scorable pitch (score_pitch).
  3. First-half Loc+ per pitcher (chronological halves of game dates), in
     two units:
       raw      pooled mean ExpRV over the pitcher's first-half pitches
       rendered mean of per-pitch INTEGER atoms 100 - 10 (v - mu_g)/sigma_g,
                the coherent-canon unit the leaderboard shows, with
                per-group anchors from the full-season per-(pitcher, type)
                pool (MIN_POOL_PT), as _normalize_by_group builds them.
  4. Second-half luck-neutral xRV per pitcher (make_rv_xrv), FF velocity
     partial, the existing harness's floors (150 scored / 150 actual /
     40 FF).
  Decider: r(first-half Loc+, second-half xRV), paired across seasons.

SELF-INFLUENCE CAVEAT. Full-season surfaces include the scored pitcher's
own pitches. Each pitcher is well under 0.5% of any surface's pool, and the
same influence is present in production, so the comparison between configs
is fair, but the absolute r is a hair optimistic.

The quarter-season protocol of locplus_constants_multiseason.py is run as
well (`--quarter`) for continuity with earlier decisions. It builds its
surfaces on a quarter of a season, which is not the sample size production
runs at.

Usage:
  python3 scripts/research/locplus/locplus_fullseason_replicate.py \
      --config cellfit=bx:bz:K_WHIFF:K_WH_COUNT:K_FOUL:K_XWCON:K_SWING_COLL:K_SWING_COUNT:K_CS \
      [--zvariant offset|w011] [--quarter] [--seasons 2021,2022,...]
"""
import argparse
import gc
import json
import math
import os
import pickle
import statistics as st
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.sdplus import make_rv_xrv

LG, SCALE = base.LG, base.SCALE
SEASONS = {2021: 'data/_statcast2021_cache.pkl',
           2022: 'data/_statcast2022_cache.pkl',
           2023: 'data/_statcast2023_cache.pkl',
           2024: 'data/_statcast2024_cache.pkl',
           2025: 'data/_statcast2025_full_cache.pkl',
           2026: 'data/all_pitches_rs_cache.pkl'}
KEYS = ['PHYS_X_IN', 'PHYS_Z_FRAC', 'K_WHIFF', 'K_WH_COUNT', 'K_FOUL', 'K_XWCON',
        'K_SWING_COLL', 'K_SWING_COUNT', 'K_CS']
GEOM = ['Z_MIN', 'Z_MAX', 'BIN_Z', 'NZ']
ORIG = {k: getattr(lp, k) for k in KEYS + GEOM}
SHIPPED = {k: ORIG[k] for k in KEYS}
OUT_JSON = os.path.join(ROOT, 'data', '_loc_fullseason_replicate.json')


def parse_config(spec):
    name, vals = spec.split('=', 1)
    parts = [float(x) for x in vals.split(':')]
    if len(parts) != len(KEYS):
        raise SystemExit(f"config {name}: need {len(KEYS)} values {KEYS}")
    return name, dict(zip(KEYS, parts))


def apply(cfg, zvariant=None):
    for k, v in cfg.items():
        setattr(lp, k, v)
    if zvariant == 'offset':
        lp.Z_MIN, lp.Z_MAX = -0.65, 1.55
    elif zvariant == 'w011':
        lp.BIN_Z = 0.11
    lp.NZ = int(round((lp.Z_MAX - lp.Z_MIN) / lp.BIN_Z))
    lp._KX = lp._k1d(lp.PHYS_X_IN / lp.BIN_X_IN)
    lp._KZ = lp._k1d(lp.PHYS_Z_FRAC / lp.BIN_Z)


def restore():
    for k, v in ORIG.items():
        setattr(lp, k, v)
    lp._KX = lp._k1d(lp.PHYS_X_IN / lp.BIN_X_IN)
    lp._KZ = lp._k1d(lp.PHYS_Z_FRAC / lp.BIN_Z)


def load_season(yr):
    path = os.path.join(ROOT, SEASONS[yr])
    if yr == 2026:
        D = pickle.load(open(path, 'rb'))
        out = [p for p in D if p.get('_source') == 'MLB']
        del D
        gc.collect()
        return out
    return base.adapt(path)


def group_anchors(scored, min_pool=lp.MIN_POOL_PT):
    """{group: (mu, sigma)} from per-(pitcher, throws, pitch type) full-season
    means with n >= min_pool, mirroring _normalize_by_group with n_prior 0."""
    cell = defaultdict(list)
    for p, v in scored:
        cell[(p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'))].append(v)
    by_g = defaultdict(list)
    for k, vs in cell.items():
        if len(vs) >= min_pool:
            by_g[lp.group_of_code(k[2])].append(sum(vs) / len(vs))
    anc = {}
    for g, xs in by_g.items():
        mu = sum(xs) / len(xs)
        sig = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
        if sig > 1e-12:
            anc[g] = (mu, sig)
    return anc


def eval_config(base_p, scorable, first_dates, rv_actual, velo, par):
    S = lp.build_surfaces(base_p, LG, SCALE)
    scored = []
    for p in scorable:
        v = lp.score_pitch(p, S)
        if v is not None:
            scored.append((p, v))
    anc = group_anchors(scored)
    raw_f, atom_f, raw_par = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for p, v in scored:
        k = (p.get('Pitcher'), p.get('Throws'))
        d = p['Game Date']
        if d in first_dates:
            raw_f[k].append(v)
            a = anc.get(lp.group_of(p))
            if a:
                atom_f[k].append(int(round(100.0 - lp.LOC_SCALE_K * (v - a[0]) / a[1])))
        raw_par[par.get(d)][k].append(v)
    loc_raw = {k: sum(v) / len(v) for k, v in raw_f.items() if len(v) >= base.MIN_SCORE}
    loc_ren = {k: sum(v) / len(v) for k, v in atom_f.items() if len(v) >= base.MIN_SCORE}
    kk = [k for k in loc_raw if k in loc_ren and k in rv_actual and k in velo]
    Y = [rv_actual[k] for k in kk]; V = [velo[k] for k in kk]
    r_sy = base.pearson(V, Y)
    out = {'n': len(kk)}
    for unit, loc, sign in (('raw', loc_raw, 1.0), ('rendered', loc_ren, -1.0)):
        L = [sign * loc[k] for k in kk]
        r_ly = base.pearson(L, Y); r_ls = base.pearson(L, V)
        out[unit] = r_ly
        out[unit + '_partial'] = base.partial(r_ly, r_ls, r_sy)
        out[unit + '_rvelo'] = r_ls
    a0 = {k: sum(v) / len(v) for k, v in raw_par[0].items() if len(v) >= base.MIN_REL}
    a1 = {k: sum(v) / len(v) for k, v in raw_par[1].items() if len(v) >= base.MIN_REL}
    ks = [k for k in a0 if k in a1]
    out['rel_diag'] = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
    out['anchors'] = {g: (round(a[0], 5), round(a[1], 5)) for g, a in anc.items()}
    return out


def eval_season_full(pitches, rv_fn, configs, zvariant):
    base_p = [p for p in pitches if lp.is_eligible_baseline(p)]
    scorable = [p for p in pitches if lp._is_scorable(p)]
    dates = sorted({p['Game Date'] for p in base_p if p['Game Date']})
    half = len(dates) // 2
    first_dates = set(dates[:half])
    par = {d: i % 2 for i, d in enumerate(dates)}
    rv_actual, velo = {}, {}
    by_second = defaultdict(list); by_first = defaultdict(list)
    for p in base_p:
        k = (p.get('Pitcher'), p.get('Throws'))
        (by_first if p['Game Date'] in first_dates else by_second)[k].append(p)
    for k, ps in by_second.items():
        v = [x for x in (rv_fn(p) for p in ps) if x is not None]
        if len(v) >= base.MIN_ACTUAL:
            rv_actual[k] = sum(v) / len(v)
    for k, ps in by_first.items():
        v = [x for x in (lp.safe_float(p['Velocity']) for p in ps
                         if p['Pitch Type'] == 'FF') if x is not None]
        if len(v) >= base.MIN_VELO:
            velo[k] = sum(v) / len(v)
    res = {}
    for name, cfg in configs.items():
        t0 = time.time()
        apply(cfg, zvariant if name != 'shipped' or zvariant == 'all' else None)
        try:
            res[name] = eval_config(base_p, scorable, first_dates, rv_actual, velo, par)
        finally:
            restore()
        r = res[name]
        print(f"    {name:>12s} raw {r['raw']:+.4f} (part {r['raw_partial']:+.4f}) "
              f"rendered {r['rendered']:+.4f} (part {r['rendered_partial']:+.4f}) "
              f"rel {r['rel_diag']:.4f} n {r['n']} [{time.time() - t0:.0f}s]", flush=True)
    return res, {'dates': len(dates), 'baseline': len(base_p), 'scorable': len(scorable)}


def eval_season_quarter(pitches, rv_fn, configs, zvariant):
    """The locplus_constants_multiseason.eval_season protocol, verbatim in
    structure, with a full config setter instead of its x-only one."""
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
    for name, cfg in configs.items():
        apply(cfg, zvariant if name != 'shipped' or zvariant == 'all' else None)
        try:
            raws, parts, rlvs = [], [], []
            for half in ('A', 'B'):
                c = prep[half]
                S = lp.build_surfaces(c['first'], LG, SCALE)
                loc = base.score_map(c['byp_f'], S, base.MIN_SCORE)
                kk = [k for k in loc if k in c['actual'] and k in c['velo']]
                if len(kk) < 30:
                    continue
                r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
                r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
                r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
                raws.append(r_ly); parts.append(base.partial(r_ly, r_ls, r_sy)); rlvs.append(r_ls)
            S0 = lp.build_surfaces(g0, LG, SCALE); S1 = lp.build_surfaces(g1, LG, SCALE)
            a0 = base.score_map(base.by_pitcher(g0), S0, base.MIN_REL)
            a1 = base.score_map(base.by_pitcher(g1), S1, base.MIN_REL)
            ks = [k for k in a0 if k in a1]
            res[name] = {'raw': sum(raws) / len(raws), 'partial': sum(parts) / len(parts),
                         'rlv': sum(rlvs) / len(rlvs),
                         'rel': base.pearson([a0[k] for k in ks], [a1[k] for k in ks]),
                         'n': len(ks)}
        finally:
            restore()
        r = res[name]
        print(f"    Q {name:>12s} partial {r['partial']:+.4f} raw {r['raw']:+.4f} "
              f"rel {r['rel']:.4f} n {r['n']}", flush=True)
    return res


def summarise(table, configs, metric_keys, label):
    seasons = sorted(table)
    print(f"\n===== {label}: each config vs shipped, paired across {len(seasons)} seasons =====")
    for m in metric_keys:
        print(f"-- {m}")
        print(f"{'config':>12s} " + ''.join(f"{y:>9d}" for y in seasons) + f"{'mean':>9s}{'d':>9s}{'SE':>8s}{'t':>7s}{'wins':>7s}")
        for name in configs:
            vals = [table[y][name][m] for y in seasons if name in table[y]]
            if name == 'shipped':
                print(f"{name:>12s} " + ''.join(f"{v:>+9.4f}" for v in vals) + f"{st.mean(vals):>+9.4f}")
                continue
            d = [table[y][name][m] - table[y]['shipped'][m] for y in seasons]
            se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
            t = st.mean(d) / se if se and se > 0 else float('nan')
            wins = sum(1 for x in d if x > 0)
            print(f"{name:>12s} " + ''.join(f"{v:>+9.4f}" for v in vals)
                  + f"{st.mean(vals):>+9.4f}{st.mean(d):>+9.4f}{se:>8.4f}{t:>7.2f}{wins:>4d}/{len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', action='append', default=[])
    ap.add_argument('--zvariant', default=None, choices=[None, 'offset', 'w011'])
    ap.add_argument('--quarter', action='store_true')
    ap.add_argument('--no-full', action='store_true')
    ap.add_argument('--seasons', default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    configs = {'shipped': dict(SHIPPED)}
    for spec in a.config:
        n, c = parse_config(spec)
        configs[n] = c
    if a.zvariant and len(configs) == 1:
        configs[a.zvariant] = dict(SHIPPED)      # geometry-only variant
    seasons = [int(s) for s in a.seasons.split(',')] if a.seasons else sorted(SEASONS)
    rv_fn = make_rv_xrv(LG, SCALE)
    full, quarter, meta = {}, {}, {}
    for yr in seasons:
        t0 = time.time()
        pitches = load_season(yr)
        print(f"=== {yr}: {len(pitches)} pitches loaded ({time.time() - t0:.0f}s)", flush=True)
        if not a.no_full:
            full[yr], meta[yr] = eval_season_full(pitches, rv_fn, configs, a.zvariant)
            print(f"  meta {meta[yr]}", flush=True)
        if a.quarter:
            quarter[yr] = eval_season_quarter(pitches, rv_fn, configs, a.zvariant)
        del pitches
        gc.collect()
    if full:
        summarise(full, configs, ['raw', 'raw_partial', 'rendered', 'rendered_partial', 'rel_diag'],
                  'FULL-SEASON surfaces, first-half Loc+ vs second-half xRV')
    if quarter:
        summarise(quarter, configs, ['partial', 'raw', 'rel'], 'QUARTER harness (continuity)')
    out = OUT_JSON.replace('.json', f'{a.tag}.json')
    json.dump({'configs': configs, 'zvariant': a.zvariant, 'full': {str(y): v for y, v in full.items()},
               'quarter': {str(y): v for y, v in quarter.items()}, 'meta': {str(y): v for y, v in meta.items()}},
              open(out, 'w'), indent=1)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
