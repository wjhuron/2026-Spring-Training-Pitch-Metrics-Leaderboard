"""hitterplus_weights_v2.py — Hitter+ composite re-verification on the atoms
that won the 2026-08 multi-season batteries, plus the pitch-diet 4th-atom
screen. LOPO protocol of derive_weights_lopo2 (4 pairs 21->22..24->25,
z-scored within pair, derive on 3 pairs / score on the 4th).

Questions:
 A. Do the winning atom configs (SD+ no-anchor, CT+ cat3) beat the shipped
    atoms at the COMPOSITE level, at the shipped 52/17/31 weights?
 B. Is 52/17/31 still optimal on the new atoms? (full simplex, step .01,
    argmax + flatness + LOPO re-derive check)
 C. Composite-level check of the CT+ form: does RAWRATE-CT (which won
    univariate prediction by re-adding swing-selection overlap) actually
    help or hurt INSIDE the composite where SD+ already carries decisions?
 D. Pitch-diet 4th atom: does what pitchers THROW a hitter (fear signal)
    add anything beyond the three atoms?
      diet_heart   share of eligible pitches in the heart zone
      diet_edge    share in chase+waste
      diet_swrv    mean league swing-cell RV of pitches faced (hittability)

Usage: python3 scripts/hitterplus_weights_v2.py
"""
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import statcast_hitter_adapter as A
import pipeline_sdplus as sd
import pipeline_contact as ct
import hitter_phase2_multiseason as H
import hitter_phase2b_followup as F

PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
FULL_MIN_DEC, FULL_MIN_SW, FULL_MIN_BIP = 200, 65, 80
SHIPPED_W = (0.52, 0.17, 0.31)


def season_rows(yn, yn1):
    """Rows for one pair: {hitter: (bb, sd_base, sd_noa, ct_base, ct_cat3,
    ct_raw, diet_heart, diet_edge, diet_swrv, y_next)}."""
    P = H.load_season(yn)
    elig = H.precompute(P)
    lg, sc = H.guts(yn)
    comp = F.season_components(elig, lg, sc, FULL_MIN_DEC, FULL_MIN_SW,
                               FULL_MIN_BIP)

    with H.patched('_z16', True):
        # RAWRATE CT on the shipped anchored table
        swings = [p for p in elig if ct.is_ct_eligible(p)]
        by_sw = defaultdict(list)
        for p in swings:
            h = p.get('Batter')
            if h:
                by_sw[h].append(p)
        offsets = ct.build_bip_count_offsets(swings, lg, sc)
        rv_fn = ct.make_rv_xrv(lg, sc, offsets)
        craw = ct.build_contact_cell_weights(swings, rv_fn)
        czm = ct.zone_level_contact_means(swings, rv_fn)
        ctab = ct.shrink_contact_cells(craw, czm)
        ct_rawrate = H.ct_score(by_sw, ctab, FULL_MIN_SW, lift=False)

        # diet metrics off the shipped SD table
        offsets_sd = sd.build_bip_count_offsets(elig, lg, sc)
        rv_sd = sd.make_rv_xrv(lg, sc, offsets_sd)
        raw = sd.build_weight_table(elig, rv_sd)
        zm = sd.zone_level_means(elig, rv_sd)
        table = sd.shrink_table(raw, zm)
        by_h = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                by_h[h].append(p)
        diet = {}
        for h, ps in by_h.items():
            if len(ps) < FULL_MIN_DEC:
                continue
            nh = ne = 0
            swrv = []
            for p in ps:
                z = sd.classify_zone(p)
                if z == 'heart':
                    nh += 1
                elif z in ('chase', 'waste'):
                    ne += 1
                swing_rv, _ = table[(z, sd.get_count(p), sd.cat_of(p), 'swing')]
                swrv.append(swing_rv)
            diet[h] = (nh / len(ps), ne / len(ps), sum(swrv) / len(swrv))

    y_map = A.target_y(yn1)
    rows = {}
    for h in comp['bb_BASE']:
        if (h in comp['sd_BASE'] and h in comp['sd_NOANCHOR']
                and h in comp['ct_BASE'] and h in comp['ct_CAT3']
                and h in ct_rawrate and h in diet):
            yv = y_map.get(h)
            if yv and yv[1] >= 200:
                rows[h] = (comp['bb_BASE'][h], comp['sd_BASE'][h],
                           comp['sd_NOANCHOR'][h], comp['ct_BASE'][h],
                           comp['ct_CAT3'][h], ct_rawrate[h],
                           diet[h][0], diet[h][1], diet[h][2],
                           yv[0] / yv[1])
    del P, elig
    return rows


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def pear(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def composite_r(pairs_z, w, cols):
    """Mean r(composite, y) across pairs at fixed weights."""
    rs = []
    for Z in pairs_z:
        comp = sum(wi * Z[c] for wi, c in zip(w, cols))
        rs.append(pear(comp, Z['y']))
    return rs


def lopo_derive(pairs_z, cols):
    """Derive OLS weights on 3 pairs, score on the 4th; returns per-fold r."""
    out = []
    for i in range(len(pairs_z)):
        train = [Z for j, Z in enumerate(pairs_z) if j != i]
        X = np.vstack([np.column_stack([Z[c] for c in cols]) for Z in train])
        y = np.concatenate([Z['y'] for Z in train])
        Xd = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        w = beta[1:]
        Zt = pairs_z[i]
        comp = sum(wi * Zt[c] for wi, c in zip(w, cols))
        out.append((pear(comp, Zt['y']), tuple(round(float(x), 3) for x in w)))
    return out


def main():
    pairs_z = []
    for yn, yn1 in PAIRS:
        rows = season_rows(yn, yn1)
        print(f"{yn}->{yn1}: {len(rows)} hitters", flush=True)
        cols = list(zip(*rows.values()))
        (bb, sdb, sdn, ctb, ct3, ctr, dh, de, dsw, y) = cols
        pairs_z.append({
            'bb': z(bb), 'sd_base': z(sdb), 'sd_noa': z(sdn),
            'ct_base': z(ctb), 'ct_cat3': z(ct3), 'ct_raw': z(ctr),
            'diet_heart': z(dh), 'diet_edge': z(de), 'diet_swrv': z(dsw),
            'y': np.asarray(y, float),
        })

    def show(name, rs):
        print(f"  {name}: mean {sum(rs)/len(rs):+.4f}  "
              + '  '.join(f"{r:+.4f}" for r in rs))

    print("\nA. SHIPPED WEIGHTS 52/17/31 — old atoms vs new atoms")
    show('old (sd_BASE, ct_BASE)   ',
         composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_base', 'ct_base')))
    show('new (sd_NOA,  ct_CAT3)   ',
         composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_noa', 'ct_cat3')))
    show('sd only (sd_NOA, ct_BASE)',
         composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_noa', 'ct_base')))
    show('ct only (sd_BASE, ct_CAT3)',
         composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_base', 'ct_cat3')))

    print("\nB. SIMPLEX (step .01) on new atoms — argmax + flatness + LOPO")
    best, bw = None, None
    grid = []
    for a in range(0, 101):
        for b in range(0, 101 - a):
            c = 100 - a - b
            w = (a / 100, b / 100, c / 100)
            rs = composite_r(pairs_z, w, ('bb', 'sd_noa', 'ct_cat3'))
            m = sum(rs) / len(rs)
            grid.append((m, w))
            if best is None or m > best:
                best, bw = m, w
    ship_m = sum(composite_r(pairs_z, SHIPPED_W,
                             ('bb', 'sd_noa', 'ct_cat3'))) / 4
    within = [w for m, w in grid if m >= best - 0.001]
    lo = tuple(min(w[i] for w in within) for i in range(3))
    hi = tuple(max(w[i] for w in within) for i in range(3))
    print(f"  argmax {bw} at {best:+.4f}; shipped 52/17/31 at {ship_m:+.4f}")
    print(f"  0.001-flat region: bb [{lo[0]:.2f},{hi[0]:.2f}] "
          f"sd [{lo[1]:.2f},{hi[1]:.2f}] ct [{lo[2]:.2f},{hi[2]:.2f}]")
    folds = lopo_derive(pairs_z, ('bb', 'sd_noa', 'ct_cat3'))
    print("  LOPO re-derived (OLS on 3, score 4th): "
          + '  '.join(f"{r:+.4f} w={w}" for r, w in folds))
    print(f"  LOPO mean {sum(r for r, _ in folds)/4:+.4f} vs shipped-weights "
          f"{ship_m:+.4f}")

    print("\nC. CT FORM AT COMPOSITE LEVEL (shipped weights, sd_NOA held)")
    show('ct_CAT3 ', composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_noa', 'ct_cat3')))
    show('ct_RAW  ', composite_r(pairs_z, SHIPPED_W, ('bb', 'sd_noa', 'ct_raw')))

    print("\nD. DIET 4TH ATOM (vs 3-atom new-config baseline)")
    base3 = lopo_derive(pairs_z, ('bb', 'sd_noa', 'ct_cat3'))
    m3 = sum(r for r, _ in base3) / 4
    print(f"  3-atom LOPO mean {m3:+.4f}")
    for dcol in ('diet_heart', 'diet_edge', 'diet_swrv'):
        f4 = lopo_derive(pairs_z, ('bb', 'sd_noa', 'ct_cat3', dcol))
        m4 = sum(r for r, _ in f4) / 4
        wins = sum(1 for (r4, _), (r3, _) in zip(f4, base3) if r4 > r3)
        print(f"  + {dcol}: LOPO mean {m4:+.4f} (delta {m4-m3:+.4f}, "
              f"folds won {wins}/4)  betas {[w[-1] for _, w in f4]}")
        rs_uni = [pear(Z[dcol], Z['y']) for Z in pairs_z]
        print(f"      univariate r: " + '  '.join(f"{r:+.3f}" for r in rs_uni))


if __name__ == '__main__':
    main()
