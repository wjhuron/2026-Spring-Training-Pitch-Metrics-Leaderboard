"""locplus_structure_multiseason.py — Loc+ structural candidates never swept,
tested across season replicates 2021-2025 (surfaces per season, never pooled).

Candidates (each changes ONE thing off the shipped config):
  pcs_cat3   called-strike surface per (batter hand x FB/BRK/OFF category)
             instead of per hand only. CS probability plausibly depends on
             pitch shape through the zone (a curveball at the bottom edge is
             not called like a fastball there). Count transform re-calibrated
             per (hand, cat, count), same 50-take floor.
  pcs_ph     called-strike surface per (batter hand x pitcher hand).
  st_group   ST/SW split out of the SL group into their own surface group.
             Sweeper usage exploded 2022->2026; the SL group pools two
             location profiles. (2021-2022 have few/no ST tags — those
             seasons report with small n and count accordingly.)

Objectives: identical to locplus_constants_multiseason.py — partial r(loc,
next-quarter xRV | FF velo), raw predictive r, odd/even-date split-half
reliability. Ship bar: a candidate must win most replicate seasons.

Usage: python3 scripts/research/locplus/locplus_structure_multiseason.py
"""
import gc
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.sdplus import make_rv_xrv, cat_of

ORIG_GROUP = dict(lp.GROUP)
ORIG_GROUPS = list(lp.GROUPS)
ST_GROUP = {**ORIG_GROUP, 'ST': 'ST', 'SW': 'ST'}
ST_GROUPS = ORIG_GROUPS + ['ST']

CONFIGS = ('shipped', 'pcs_cat3', 'pcs_ph', 'st_group')


def cs_keyfn(name):
    if name == 'pcs_cat3':
        return lambda p: (p.get('Bats'), cat_of(p))
    if name == 'pcs_ph':
        return lambda p: (p.get('Bats'), p.get('Throws'))
    return None


def build_cs_variant(baseline, keyfn):
    """Called-strike surface per keyfn(p) key, mirroring build_surfaces'
    per-hand CS build including the count transform. Returns
    {key: {count: grid}}."""
    csn = defaultdict(lp._zeros)
    csd = defaultdict(lp._zeros)
    csd_kc = defaultdict(lp._zeros)
    cs_obs_kc = defaultdict(int)
    for p in baseline:
        d = p.get('Description')
        if d not in lp.TAKE_DESC:
            continue
        k = keyfn(p)
        if k is None or k[0] is None:
            continue
        c = lp.get_count(p)
        i = lp._xbin(lp.safe_float(p.get('PlateX')))
        j = lp._zbin(lp._znorm(p))
        csn[k][i][j] += 0 if d != 'Called Strike' else 1
        csd[k][i][j] += 1
        csd_kc[(k, c)][i][j] += 1
        if d == 'Called Strike':
            cs_obs_kc[(k, c)] += 1

    MIN_CT_TAKES = 50
    out = {}
    for k in csd:
        prior = lp._gsum(csn[k]) / max(lp._gsum(csd[k]), 1)
        b = lp._smooth(csn[k], csd[k], prior, lp.K_CS)
        out[k] = {}
        for c in lp.COUNTS:
            delta = 0.0
            if lp.CS_COUNT_TRANSFORM:
                tk = csd_kc.get((k, c))
                obs = cs_obs_kc.get((k, c), 0)
                if tk is not None:
                    tk_n = lp._gsum(tk)
                    if tk_n >= MIN_CT_TAKES and 0 < obs < tk_n:
                        pred = sum(tk[i][j] * b[i][j]
                                   for i in range(lp.NX) for j in range(lp.NZ))
                        if pred > 0:
                            delta = lp._logit(obs / tk_n) - lp._logit(pred / tk_n)
            if delta == 0.0:
                out[k][c] = b
            else:
                out[k][c] = [[lp._sig(lp._logit(b[i][j]) + delta)
                              for j in range(lp.NZ)] for i in range(lp.NX)]
    return out


def score_pitch_cs(p, S, PCSV, keyfn):
    """lp.score_pitch with the PCS lookup swapped to the variant surface
    (falls back to the shipped per-hand surface when the key is unseen)."""
    key = (lp.group_of(p), p.get('Bats'), p.get('Throws'))
    if key not in S['WH']:
        return None
    c = lp.get_count(p)
    px = lp.safe_float(p.get('PlateX'))
    zn = lp._znorm(p)
    if c is None or px is None or zn is None:
        return None
    i = lp._xbin(px)
    j = lp._zbin(zn)
    psw = S['SW'][key][c][i][j]
    pwh = S['WH'][key][i][j]
    pfl = S['FL'][key][i][j]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = S['XW'][key][i][j] + S['BIPOFF'].get(c, 0.0)
    vk = keyfn(p)
    pcs_tab = PCSV.get(vk)
    if pcs_tab is None:
        pcs_tab = S['PCS'][p['Bats']]
    pcs = pcs_tab[c][i][j]
    RV = S['RV']
    swing_val = (pwh * RV['whiff'].get(c, 0.0) + pfl * RV['foul'].get(c, 0.0)
                 + pbip * vbip)
    take_val = pcs * RV['cs'].get(c, 0.0) + (1 - pcs) * RV['ball'].get(c, 0.0)
    return psw * swing_val + (1 - psw) * take_val


class st_patch:
    def __enter__(self):
        lp.GROUP = ST_GROUP
        lp.GROUPS = ST_GROUPS

    def __exit__(self, *a):
        lp.GROUP = ORIG_GROUP
        lp.GROUPS = ORIG_GROUPS


def score_map_variant(byp, S, min_n, PCSV=None, keyfn=None):
    out = {}
    for k, ps in byp.items():
        if PCSV is None:
            v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        else:
            v = [s for s in (score_pitch_cs(p, S, PCSV, keyfn) for p in ps)
                 if s is not None]
        if len(v) >= min_n:
            out[k] = sum(v) / len(v)
    return out


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

    # shipped surfaces built once per segment; CS variants reuse them.
    S_first = {h: lp.build_surfaces(prep[h]['first'], base.LG, base.SCALE)
               for h in ('A', 'B')}
    S_rel = {0: lp.build_surfaces(g0, base.LG, base.SCALE),
             1: lp.build_surfaces(g1, base.LG, base.SCALE)}

    res = {}
    for name in CONFIGS:
        if name == 'st_group':
            with st_patch():
                Sf = {h: lp.build_surfaces(prep[h]['first'], base.LG, base.SCALE)
                      for h in ('A', 'B')}
                Sr = {0: lp.build_surfaces(g0, base.LG, base.SCALE),
                      1: lp.build_surfaces(g1, base.LG, base.SCALE)}
                res[name] = _objectives(prep, g0, g1, Sf, Sr, None, None)
        elif name == 'shipped':
            res[name] = _objectives(prep, g0, g1, S_first, S_rel, None, None)
        else:
            kf = cs_keyfn(name)
            Pf = {h: build_cs_variant(prep[h]['first'], kf) for h in ('A', 'B')}
            Pr = {0: build_cs_variant(g0, kf), 1: build_cs_variant(g1, kf)}
            res[name] = _objectives(prep, g0, g1, S_first, S_rel, (Pf, Pr), kf)
    return res


def _objectives(prep, g0, g1, S_first, S_rel, PCSVs, keyfn):
    raws, parts, rlvs = [], [], []
    for half in ('A', 'B'):
        c = prep[half]
        PCSV = PCSVs[0][half] if PCSVs else None
        loc = score_map_variant(c['byp_f'], S_first[half], base.MIN_SCORE,
                                PCSV, keyfn)
        kk = [k for k in loc if k in c['actual'] and k in c['velo']]
        if len(kk) < 30:
            continue
        r_ly = base.pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
        r_ls = base.pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
        r_sy = base.pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
        raws.append(r_ly)
        parts.append(base.partial(r_ly, r_ls, r_sy))
        rlvs.append(r_ls)
    a0 = score_map_variant(base.by_pitcher(g0), S_rel[0], base.MIN_REL,
                           PCSVs[1][0] if PCSVs else None, keyfn)
    a1 = score_map_variant(base.by_pitcher(g1), S_rel[1], base.MIN_REL,
                           PCSVs[1][1] if PCSVs else None, keyfn)
    ks = [k for k in a0 if k in a1]
    rel = base.pearson([a0[k] for k in ks], [a1[k] for k in ks])
    return {'raw': sum(raws) / len(raws), 'partial': sum(parts) / len(parts),
            'rlv': sum(rlvs) / len(rlvs), 'rel': rel, 'n': len(ks)}


def main():
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    print(f"{'season':>7s} {'config':>10s} | {'PARTIAL|velo':>13s} | {'raw':>6s} "
          f"| {'r(velo)':>8s} | {'rel':>6s}")
    print('-' * 64)
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr:>7d}   cache missing, skipped", flush=True)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = base.adapt(p)
        n_st = sum(1 for x in pitches if x.get('Pitch Type') in ('ST', 'SW'))
        print(f"  {yr}: {len(pitches)} usable pitches ({n_st} ST/SW)",
              file=sys.stderr)
        res = eval_season(pitches, rv_fn)
        table[yr] = res
        for name in CONFIGS:
            o = res.get(name)
            if o:
                print(f"{yr:>7d} {name:>10s} | {o['partial']:>13.3f} | {o['raw']:>6.3f} "
                      f"| {o['rlv']:>+8.3f} | {o['rel']:>6.3f}", flush=True)
        del pitches
        gc.collect()

    for metric in ('partial', 'rel'):
        print()
        print(f"VERDICT — candidate vs shipped, on {metric}, per season")
        print(f"{'config':>10s} " + "".join(f"{yr:>9d}" for yr in sorted(table))
              + f"{'wins':>7s}")
        print('-' * (11 + 9 * len(table) + 7))
        for name in CONFIGS:
            if name == 'shipped':
                continue
            wins, cells = 0, ''
            for yr in sorted(table):
                s, p = table[yr].get('shipped'), table[yr].get(name)
                if not s or not p or s.get(metric) is None or p.get(metric) is None:
                    cells += f"{'-':>9s}"
                    continue
                d = p[metric] - s[metric]
                wins += 1 if d > 0 else 0
                cells += f"{d:>+9.3f}"
            print(f"{name:>10s} {cells}{wins:>4d}/{len(table)}")
    print()
    print("Positive = candidate beats shipped that season. Adopt only on a")
    print("majority of replicate seasons with no reliability collapse.")


if __name__ == '__main__':
    main()
