"""plus_constants_impact.py — what a SD+/CT+ constant change does to the
SHIPPED number, measured against the real pitch cache.

Replicates compute_sd_plus / the CT+ path exactly, with k and n_prior passed
EXPLICITLY.

(Module-level defaults like `def shrink_table(..., k=CELL_SHRINK_K)` bind at
def time, so monkeypatching the module constant after import would silently
do nothing. Every knob is passed here.)

SD+ specifics copied from compute_sd_plus: rv_fn has NO count offsets (SD+
dropped its anchor 2026-08-15) and the hitter aggregate is mix-neutral via
lg_zone_w. CT+ keeps its anchor.

VALIDATE FIRST. The OLD config must reproduce the shipped leaderboard, and
"reproduce" means AFTER the documented downstream rescale: shipped values are
re-anchored and matched to the live wRC+ spread (metadata plusReanchor /
plusWrcScale — e.g. sdPlus x0.582, ctPlus x1.429 applied to the deviation
from 100). regress_and_normalize output is NOT the shipped number. Skipping
that step on 2026-08-16 made a 0.29-point SD+ shift look like 5.1 points.

Usage: python3 scripts/research/hitter/plus_constants_impact.py
"""
import json
import pickle
import sys
from collections import defaultdict

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import pipeline.sdplus as sd
import pipeline.contact as ct

R = ROOT
P = [p for p in pickle.load(open(f'{R}/data/all_pitches_rs_cache.pkl', 'rb'))
     if p.get('_source', 'MLB') == 'MLB']
md = json.load(open(f'{R}/data/metadata_rs.json'))
g = md.get('gutsConstants') or {}
LG, SCALE = float(g['lgWOBA']), float(g['wOBAScale'])
print(f"{len(P)} MLB pitches | lgWOBA={LG} scale={SCALE}")

by_hitter = defaultdict(list)
for p in P:
    key = (p.get('Batter'), p.get('BTeam'))
    if key[0]:
        by_hitter[key].append(p)

elig = [p for p in P if sd.is_eligible(p)]

# ── SD+ (un-anchored rv, mix-neutral aggregate) ─────────────────────────
rv_sd = sd.make_rv_xrv(LG, SCALE)
sd_raw_tab = sd.build_weight_table(elig, rv_sd)
sd_zm = sd.zone_level_means(elig, rv_sd)
zc = defaultdict(int)
for p in elig:
    zc[sd.classify_zone(p)] += 1
tot = sum(zc.values())
lgw = {z: n / tot for z, n in zc.items()}

def sd_run(k, n0, floor):
    sm = sd.shrink_table(sd_raw_tab, sd_zm, k=k)
    raw = sd.compute_hitter_sd(by_hitter, sm, lgw)
    out = sd.regress_and_normalize(raw, n_prior=n0, min_n=floor)
    return {kk: v['sdPlus'] for kk, v in out.items()}

# ── CT+ (anchored rv) ───────────────────────────────────────────────────
offs = sd.build_bip_count_offsets(elig, LG, SCALE)
rv_ct = sd.make_rv_xrv(LG, SCALE, offs)
swings = [p for p in elig if ct.is_ct_eligible(p)]
ct_raw_tab = ct.build_contact_cell_weights(swings, rv_ct)
ct_zm = ct.zone_level_contact_means(swings, rv_ct)

def ct_run(k, n0, floor):
    sm = ct.shrink_contact_cells(ct_raw_tab, ct_zm, k=k)
    raw = ct.compute_hitter_ct(by_hitter, sm)
    out = ct.regress_and_normalize(raw, n_prior=n0, min_n=floor)
    return {kk: v['ctPlus'] for kk, v in out.items()}

print("OLD config...", flush=True)
sd_old, ct_old = sd_run(200, 180, 180), ct_run(200, 61, 65)
print("NEW config...", flush=True)
sd_new, ct_new = sd_run(50, 190, 190), ct_run(50, 66, 65)

# ── validation against the shipped leaderboard ──────────────────────────
lb = json.load(open(f'{R}/data/hitter_leaderboard_rs.json'))
ship_sd = {(r['hitter'], r['team']): r.get('sdPlus') for r in lb}
ship_ct = {(r['hitter'], r['team']): r.get('ctPlus') for r in lb}

def validate(name, mine, ship):
    common = [k for k in mine if ship.get(k) is not None]
    if not common:
        print(f"  {name}: no overlap to validate"); return
    d = sorted(abs(mine[k] - ship[k]) for k in common)
    print(f"  {name}: n={len(common)} median|Δ vs shipped|={d[len(d)//2]:.3f} "
          f"max={d[-1]:.3f}")

print("\nVALIDATION — OLD config vs shipped leaderboard (should be ~0):")
validate("sdPlus", sd_old, ship_sd)
validate("ctPlus", ct_old, ship_ct)

def rep(name, a, b):
    common = [x for x in a if x in b]
    d = sorted(abs(b[x] - a[x]) for x in common)
    print(f"  {name}: n={len(common)} median|Δ|={d[len(d)//2]:.2f} "
          f"p90={d[int(len(d)*.9)]:.2f} max={d[-1]:.2f} "
          f"| rows lost={len(set(a)-set(b))}")

print("\nIMPACT — old -> new:")
rep("sdPlus", sd_old, sd_new)
rep("ctPlus", ct_old, ct_new)
