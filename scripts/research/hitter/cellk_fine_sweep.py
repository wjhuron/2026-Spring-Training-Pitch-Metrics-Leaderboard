"""cellk_fine_sweep.py — fine grids inside the basins found by the coarse
cell-shrinkage audits, for all three cell tables at once.

Coarse results (train on one FULL season, evaluate on another, 2021-2026):

    CT+       basin ~10-100, production 200 outside      (12/12 pairs)
    SD+       basin ~0-50,   production 200 outside      (30/30 at k=100)
    xwOBA3D   basin ~0-10,   production 20 just outside  (29/30 at k=10)

Those grids were coarse (roughly powers of two), so the argmin locations
are only known to within a factor of ~2. This refines them.

Proper scoring rule per table, matching what each cell actually predicts:
    CT+       p_whiff  -> binary log loss
    SD+       mean rv  -> squared error
    xwOBA3D   mean xwOBA -> squared error

Reported per k: the mean score across ordered season pairs, and the number
of pairs where k beats production. Also the PER-PAIR argmin distribution —
if the argmin jumps around inside the basin, the basin is flat and no
single value is identifiable, which is a result to state rather than
paper over with a decimal.

All seasons are loaded once and all three tables built from that single
pass. Train/eval sets are FULL seasons, never halves, so the tables being
tuned are the size production builds.

Usage: python3 scripts/research/hitter/cellk_fine_sweep.py
"""
import math
import os
import sys
from array import array
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline.sdplus as sd
import pipeline.contact as ct
import pipeline.xwoba3d as x3
from pipeline.utils import safe_float
from handsplit_sdct_test import load_season, guts

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
KS_CT = [5, 8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128]
KS_SD = [0, 3, 5, 8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]
KS_X3 = [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 20]
EPS = 1e-6

SD_KEYS = [(z, c, cat, d) for z in sd.ZONES for c in sd.COUNTS
           for cat in sd.SD_CATS for d in ('swing', 'take')]
SD_ID = {k: i for i, k in enumerate(SD_KEYS)}
CT_KEYS = [(z, c, cat) for z in ct.ZONES for c in ct.COUNTS for cat in ct.CATS]
CT_ID = {k: i for i, k in enumerate(CT_KEYS)}
X3_KEYS = [(e, l, s, b) for e in range(len(x3.EV_BINS))
           for l in range(len(x3.LA_BINS))
           for s in x3.SPRAY_DIRS for b in x3.HANDS]
X3_ID = {k: i for i, k in enumerate(X3_KEYS)}


def build_season(year):
    P = load_season(year)
    lg, sc = guts(year)
    elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)

    # SD+
    sd_raw = sd.build_weight_table(elig, rv_fn)
    sd_zm = sd.zone_level_means(elig, rv_fn)
    sd_ids, sd_rv = array('i'), array('d')
    for p in elig:
        rv = rv_fn(p)
        d = sd.classify_decision(p)
        if rv is None or d is None:
            continue
        i = SD_ID.get((sd.classify_zone(p), sd.get_count(p), sd._sd_cat(p), d))
        if i is not None:
            sd_ids.append(i); sd_rv.append(rv)

    # CT+
    swings = [p for p in elig if ct.is_ct_eligible(p)]
    ct_raw = ct.build_contact_cell_weights(swings, rv_fn)
    ct_zm = ct.zone_level_contact_means(swings, rv_fn)
    ct_ids, ct_y = array('i'), array('b')
    for p in swings:
        i = CT_ID.get((sd.classify_zone(p), sd.get_count(p), ct.cat_of(p)))
        if i is not None:
            ct_ids.append(i)
            ct_y.append(1 if ct.classify_contact_outcome(p) == 'contact' else 0)

    # xwOBA3D
    bip = [p for p in elig if x3.classify_bip(p) is not None
           and safe_float(p.get('xwOBA')) is not None]
    x3_raw = x3.build_xwoba3d_table(bip)
    x3_mg = x3._two_d_marginals(bip)
    x3_ids, x3_v = array('i'), array('d')
    for p in bip:
        i = X3_ID.get(x3.classify_bip(p))
        if i is not None:
            x3_ids.append(i); x3_v.append(safe_float(p.get('xwOBA')))

    del P, elig, swings, bip
    return dict(sd=(sd_raw, sd_zm, sd_ids, sd_rv),
                ct=(ct_raw, ct_zm, ct_ids, ct_y),
                x3=(x3_raw, x3_mg, x3_ids, x3_v))


def vec_sd(raw, zm, k):
    t = sd.shrink_table(raw, zm, k=k)
    v = [None] * len(SD_KEYS)
    for key, val in t.items():
        i = SD_ID.get(key)
        if i is not None:
            v[i] = val[0] if isinstance(val, tuple) else val
    return v


def vec_ct(raw, zm, k):
    t = ct.shrink_contact_cells(raw, zm, k=k)
    v = [None] * len(CT_KEYS)
    for key, cell in t.items():
        i = CT_ID.get(key)
        if i is not None:
            v[i] = cell['p_whiff']
    return v


def vec_x3(raw, mg, k):
    """Local shrink using precomputed marginals (mirrors x3.shrink_xwoba3d)."""
    v = [None] * len(X3_KEYS)
    for (evi, lai, sp, bats), i in X3_ID.items():
        cell_mean, n = raw.get((evi, lai, sp, bats), (None, 0))
        priors = []
        for mk, md in [((evi, lai, bats), 'ev_la'), ((lai, sp, bats), 'la_sp'),
                       ((evi, sp, bats), 'ev_sp')]:
            val = mg[md].get(mk)
            if val is not None:
                priors.append(val)
        prior = (sum(priors) / len(priors)) if priors else \
            mg['by_bats'].get(bats, mg['global'])
        v[i] = prior if cell_mean is None else (n * cell_mean + k * prior) / (n + k)
    return v


def sq_err(vec, ids, vals):
    tot = 0.0; n = 0
    for i, a in zip(ids, vals):
        p = vec[i]
        if p is None:
            continue
        d = p - a
        tot += d * d; n += 1
    return tot / n if n else None


def log_loss(vec, ids, ys):
    tot = 0.0; n = 0
    for i, y in zip(ids, ys):
        pw = vec[i]
        if pw is None:
            continue
        pc = min(1.0 - EPS, max(EPS, 1.0 - pw))
        tot += -(math.log(pc) if y else math.log(1.0 - pc))
        n += 1
    return tot / n if n else None


def report(name, ks, prod, scores, argmins, npairs):
    print(f"\n{name} — {npairs} ordered full-season pairs, production k = {prod}")
    print(f"  {'k':>6s} {'score':>13s} {'beats prod':>11s}")
    best = None
    for k in ks:
        s = scores[k]
        m = sum(s) / len(s)
        w = sum(1 for a, b in zip(s, scores[prod]) if a < b) if prod in scores else 0
        star = ''
        if best is None or m < best[1]:
            best = (k, m)
        print(f"  {k:6d} {m:13.8f} {w:8d}/{npairs}")
    print(f"  argmin per pair: ", end='')
    dist = defaultdict(int)
    for a in argmins:
        dist[a] += 1
    print(", ".join(f"k={k}:{n}" for k, n in sorted(dist.items())))
    print(f"  best mean score at k={best[0]}")
    return best[0]


def main():
    print(f"Loading {SEASONS} (one pass, three tables)...", flush=True)
    S = {}
    for y in SEASONS:
        S[y] = build_season(y)
        print(f"  {y}: sd={len(S[y]['sd'][2])} ct={len(S[y]['ct'][2])} "
              f"x3={len(S[y]['x3'][2])}", flush=True)

    vecs = {}
    for y in SEASONS:
        sd_raw, sd_zm, _, _ = S[y]['sd']
        ct_raw, ct_zm, _, _ = S[y]['ct']
        x3_raw, x3_mg, _, _ = S[y]['x3']
        vecs[y] = dict(
            sd={k: vec_sd(sd_raw, sd_zm, k) for k in set(KS_SD) | {200}},
            ct={k: vec_ct(ct_raw, ct_zm, k) for k in set(KS_CT) | {200}},
            x3={k: vec_x3(x3_raw, x3_mg, k) for k in set(KS_X3) | {20}},
        )
        print(f"  {y} tables shrunk", flush=True)

    sc = {'sd': defaultdict(list), 'ct': defaultdict(list), 'x3': defaultdict(list)}
    am = {'sd': [], 'ct': [], 'x3': []}
    npairs = 0
    for tr in SEASONS:
        for ev in SEASONS:
            if tr == ev:
                continue
            npairs += 1
            _, _, sd_ids, sd_rv = S[ev]['sd']
            _, _, ct_ids, ct_y = S[ev]['ct']
            _, _, x3_ids, x3_v = S[ev]['x3']
            row = {}
            for k in set(KS_SD) | {200}:
                sc['sd'][k].append(sq_err(vecs[tr]['sd'][k], sd_ids, sd_rv))
            for k in set(KS_CT) | {200}:
                sc['ct'][k].append(log_loss(vecs[tr]['ct'][k], ct_ids, ct_y))
            for k in set(KS_X3) | {20}:
                sc['x3'][k].append(sq_err(vecs[tr]['x3'][k], x3_ids, x3_v))
            am['sd'].append(min(KS_SD, key=lambda k: sc['sd'][k][-1]))
            am['ct'].append(min(KS_CT, key=lambda k: sc['ct'][k][-1]))
            am['x3'].append(min(KS_X3, key=lambda k: sc['x3'][k][-1]))
        print(f"  train {tr} scored", flush=True)

    report("CT+  CELL_SHRINK_K (log loss)", KS_CT, 200, sc['ct'], am['ct'], npairs)
    report("SD+  CELL_SHRINK_K (MSE)", KS_SD, 200, sc['sd'], am['sd'], npairs)
    report("xwOBA3D CELL_SHRINK_K (MSE)", KS_X3, 20, sc['x3'], am['x3'], npairs)


if __name__ == '__main__':
    main()
