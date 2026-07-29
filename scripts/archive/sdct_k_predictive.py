"""sdct_k_predictive.py — predictive leg of the cell-k sweep.

Split-half reliability rises monotonically with k (more shrinkage = more
stable scores), so reliability alone can't pick k. This measures what heavier
shrinkage costs in signal: full-season year-N raw SD+/CT+ per k vs year-N+1
wOBA (pairs 21->22 .. 24->25), production floors.

Usage: python3 scripts/sdct_k_predictive.py
"""
import os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import statcast_hitter_adapter as A
import pipeline_sdplus as sd
import pipeline_contact as ct
from handsplit_sdct_test import load_season, guts, pearson, sd_score_baseline, ct_score
from sdct_constant_sweeps import KS

PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
FULL_MIN_DEC = sd.MIN_HITTER_DECISIONS
FULL_MIN_SW = ct.MIN_HITTER_SWINGS

agg = defaultdict(list)
for yn, yn1 in PAIRS:
    P = load_season(yn)
    lg, sc = guts(yn)
    elig = [p for p in P if sd.is_eligible(p)]
    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)
    swings = [p for p in elig if ct.is_ct_eligible(p)]

    raw_sd_table = sd.build_weight_table(elig, rv_fn)
    zm_sd = sd.zone_level_means(elig, rv_fn)
    raw_ct_table = ct.build_contact_cell_weights(swings, rv_fn)
    zm_ct = ct.zone_level_contact_means(swings, rv_fn)

    zc = defaultdict(int)
    for p in elig:
        zc[sd.classify_zone(p)] += 1
    tot = sum(zc.values())
    lgw = {z: n / tot for z, n in zc.items()}

    by_hitter = defaultdict(list)
    for p in elig:
        h = p.get('Batter')
        if h:
            by_hitter[h].append(p)

    y_map = A.target_y(yn1)
    line = [f"{yn}->{yn1}:"]
    for k in KS:
        table = sd.shrink_table(raw_sd_table, zm_sd, k=k)
        tctb = ct.shrink_contact_cells(raw_ct_table, zm_ct, k=k)
        sd_scores = sd_score_baseline(by_hitter, table, lgw, FULL_MIN_DEC)
        ct_scores = ct_score(by_hitter, lambda p: tctb, FULL_MIN_SW)
        for name, scores in (('sd', sd_scores), ('ct', ct_scores)):
            xs, ys = [], []
            for h, v in scores.items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = pearson(xs, ys)
            agg[(name, k)].append(r)
            line.append(f"{name}k{k}={r:+.3f}")
    print('  '.join(line), flush=True)
    del P

print(f"\n{'k':>5s} {'SD+ pred r':>11s} {'CT+ pred r':>11s}")
for k in KS:
    ms = sum(agg[('sd', k)]) / len(agg[('sd', k)])
    mc = sum(agg[('ct', k)]) / len(agg[('ct', k)])
    print(f"{k:5d} {ms:11.4f} {mc:11.4f}")
