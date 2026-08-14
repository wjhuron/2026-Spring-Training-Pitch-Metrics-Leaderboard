"""hitter_battrack_screen.py — does bat tracking add anything BEYOND the
three Hitter+ atoms? 2026-only screening gate for the 2024-2025 pull.

Protocol (pre-registered):
  1H (chronological, by median game date) per hitter:
    - the three shipped atom raws: BB+ raw (mean xwOBA on non-bunt BIP),
      SD+ raw (production tables + mix-neutral), CT+ raw (lift ratio)
    - bat-tracking candidates over tracked swings (>=50 tracked):
        bs_mean       mean BatSpeed
        fast_rate     share of tracked swings >= 75 mph (Savant fast-swing)
        sl_mean       mean SwingLength
        aa_mean       mean AttackAngle
        aa_sd         within-hitter SD of AttackAngle (consistency)
        tilt_mean     mean SwingPathTilt
        dir_mean      mean AttackDirection
  2H target: actual wOBA (>=100 PA events; IBB excluded).

  For each candidate: raw r vs 2H wOBA, PARTIAL r controlling the three 1H
  atoms (residual-on-residual), incremental R2 of OLS(3 atoms + candidate)
  over OLS(3 atoms), and the candidate's correlation with each atom
  (redundancy). A candidate justifies the 2024-2025 bat-tracking pull only
  if the partial holds at meaningful size; the multi-season replicates then
  decide adoption (3-replicate bar: 2024, 2025, 2026).

Usage: python3 scripts/hitter_battrack_screen.py
"""
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_sdplus as sd
import pipeline_contact as ct

LG, SCALE = 0.3172, 1.2343
MIN_DEC, MIN_SW, MIN_BIP, MIN_TRACK = 125, 45, 40, 50
MIN_PA_2H = 100
CANDS = ('bs_mean', 'fast_rate', 'sl_mean', 'aa_mean', 'aa_sd',
         'tilt_mean', 'dir_mean')

BIP_WOBA = {'Single': 0.9, 'Field Error': 0.9, 'Fielders Choice': 0.9,
            'Double': 1.25, 'Triple': 1.6, 'Home Run': 2.0}
PA_WOBA = {'Walk': 0.7, 'Hit By Pitch': 0.72}


def pearson(xs, ys):
    n = len(xs)
    if n < 20:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def ols_r2(X, y):
    """R2 of y on columns of X (with intercept), via normal equations."""
    import numpy as np
    A = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    resid = np.asarray(y) - A @ beta
    ss_res = float(resid @ resid)
    yv = np.asarray(y) - np.mean(y)
    ss_tot = float(yv @ yv)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def residuals(y, X):
    import numpy as np
    A = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    return np.asarray(y) - A @ beta


def atom_raws(pitches_1h):
    """Production-path raw atoms per hitter for the 1H slice."""
    elig = [p for p in pitches_1h if sd.is_eligible(p)]
    offsets = sd.build_bip_count_offsets(elig, LG, SCALE)
    rv_fn = sd.make_rv_xrv(LG, SCALE, offsets)
    raw = sd.build_weight_table(elig, rv_fn)
    zm = sd.zone_level_means(elig, rv_fn)
    table = sd.shrink_table(raw, zm)
    zc = defaultdict(int)
    for p in elig:
        zc[sd.classify_zone(p)] += 1
    tot = sum(zc.values())
    lgw = {z: n / tot for z, n in zc.items()}

    swings = [p for p in elig if ct.is_ct_eligible(p)]
    craw = ct.build_contact_cell_weights(swings, rv_fn)
    czm = ct.zone_level_contact_means(swings, rv_fn)
    ctable = ct.shrink_contact_cells(craw, czm)

    by_h = defaultdict(list)
    for p in elig:
        h = p.get('Batter')
        if h:
            by_h[h].append(p)

    out = {}
    for h, ps in by_h.items():
        rec = {}
        # SD+ raw (mix-neutral)
        if len(ps) >= MIN_DEC:
            zone_dvs = defaultdict(list)
            for p in ps:
                zone_dvs[sd.classify_zone(p)].append(sd.compute_dv(p, table))
            zmeans = {z: sum(v) / len(v) for z, v in zone_dvs.items()}
            wsum = sum(lgw.get(z, 0.0) for z in zmeans)
            if wsum > 0:
                rec['sd'] = sum(m * lgw.get(z, 0.0)
                                for z, m in zmeans.items()) / wsum
        # CT+ raw (lift)
        sws = [p for p in ps if ct.is_ct_eligible(p)]
        if len(sws) >= MIN_SW:
            A_ = E = 0.0
            for p in sws:
                lev, con = ct.compute_ct_swing(p, ctable)
                if lev <= 0:
                    continue
                cell = ctable[(sd.classify_zone(p), sd.get_count(p))]
                A_ += lev * con
                E += lev * (1.0 - cell['p_whiff'])
            if E > 0:
                rec['ct'] = A_ / E
        # BB+ raw (xwOBAcon)
        xs = [sd.safe_float(p.get('xwOBA')) for p in ps
              if p.get('Description') == 'In Play'
              and p.get('BBType') not in sd.BUNT_BB_TYPES]
        xs = [x for x in xs if x is not None]
        if len(xs) >= MIN_BIP:
            rec['bb'] = sum(xs) / len(xs)
        # bat-tracking candidates over tracked swings
        tracked = [(sd.safe_float(p.get('BatSpeed')),
                    sd.safe_float(p.get('SwingLength')),
                    sd.safe_float(p.get('AttackAngle')),
                    sd.safe_float(p.get('SwingPathTilt')),
                    sd.safe_float(p.get('AttackDirection')))
                   for p in sws]
        bs = [t[0] for t in tracked if t[0] is not None]
        if len(bs) >= MIN_TRACK:
            rec['bs_mean'] = sum(bs) / len(bs)
            rec['fast_rate'] = sum(1 for v in bs if v >= 75.0) / len(bs)
            sl = [t[1] for t in tracked if t[1] is not None]
            aa = [t[2] for t in tracked if t[2] is not None]
            tl = [t[3] for t in tracked if t[3] is not None]
            dr = [t[4] for t in tracked if t[4] is not None]
            if sl:
                rec['sl_mean'] = sum(sl) / len(sl)
            if len(aa) >= MIN_TRACK:
                m = sum(aa) / len(aa)
                rec['aa_mean'] = m
                rec['aa_sd'] = math.sqrt(sum((v - m) ** 2 for v in aa) / len(aa))
            if tl:
                rec['tilt_mean'] = sum(tl) / len(tl)
            if dr:
                rec['dir_mean'] = sum(dr) / len(dr)
        out[h] = rec
    return out


def woba_2h(pitches_2h):
    acc = defaultdict(lambda: [0.0, 0])
    for p in pitches_2h:
        ev = p.get('Event')
        if not ev or ev == 'Intent Walk':
            continue
        val = None
        if p.get('Description') == 'In Play':
            if p.get('BBType') in sd.BUNT_BB_TYPES:
                continue
            val = BIP_WOBA.get(ev, 0.0)
        elif ev in PA_WOBA:
            val = PA_WOBA[ev]
        elif 'Strikeout' in ev:
            val = 0.0
        elif ev in ('Groundout', 'Flyout', 'Lineout', 'Pop Out', 'Forceout',
                    'Grounded Into DP', 'Double Play', 'Triple Play',
                    'Field Error', 'Fielders Choice', 'Fielders Choice Out'):
            val = 0.0
        if val is None:
            continue
        h = p.get('Batter')
        if h:
            acc[h][0] += val
            acc[h][1] += 1
    return {h: (s / n, n) for h, (s, n) in acc.items() if n >= MIN_PA_2H}


def main():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    del D
    dates = sorted({p.get('Game Date') for p in mlb if p.get('Game Date')})
    mid = dates[len(dates) // 2]
    H1 = [p for p in mlb if (p.get('Game Date') or '') < mid]
    H2 = [p for p in mlb if (p.get('Game Date') or '') >= mid]
    print(f"1H: {len(H1)} pitches (< {mid}), 2H: {len(H2)} pitches", flush=True)

    atoms = atom_raws(H1)
    target = woba_2h(H2)

    rows = []
    for h, rec in atoms.items():
        if h in target and all(k in rec for k in ('bb', 'sd', 'ct')):
            rows.append((h, rec, target[h][0]))
    print(f"hitters with all 3 atoms + 2H target: {len(rows)}", flush=True)

    bb = [r[1]['bb'] for r in rows]
    sdv = [r[1]['sd'] for r in rows]
    ctv = [r[1]['ct'] for r in rows]
    y = [r[2] for r in rows]
    base_r2 = ols_r2([bb, sdv, ctv], y)
    print(f"\n3-atom OLS R2 on 2H wOBA: {base_r2:.4f}  (n={len(y)})")

    print(f"\n{'candidate':>10s} {'n':>5s} {'r_raw':>7s} {'r_partial':>10s} "
          f"{'dR2':>7s} | {'r_bb':>6s} {'r_sd':>6s} {'r_ct':>6s}")
    print('-' * 66)
    for cand in CANDS:
        sub = [(r[1][cand], r[1]['bb'], r[1]['sd'], r[1]['ct'], r[2])
               for r in rows if cand in r[1]]
        if len(sub) < 50:
            print(f"{cand:>10s} {len(sub):>5d}   too few")
            continue
        cv, b_, s_, c_, y_ = (list(t) for t in zip(*sub))
        r_raw = pearson(cv, y_)
        rc = residuals(cv, [b_, s_, c_])
        ry = residuals(y_, [b_, s_, c_])
        r_part = pearson(list(rc), list(ry))
        r2_with = ols_r2([b_, s_, c_, cv], y_)
        r2_wo = ols_r2([b_, s_, c_], y_)
        dr2 = (r2_with - r2_wo) if (r2_with is not None and r2_wo is not None) else None
        print(f"{cand:>10s} {len(sub):>5d} {r_raw:>+7.3f} {r_part:>+10.3f} "
              f"{dr2:>7.4f} | {pearson(cv, b_):>+6.2f} {pearson(cv, s_):>+6.2f} "
              f"{pearson(cv, c_):>+6.2f}", flush=True)

    print("\nGate: a candidate justifies the 2024-2025 bat-tracking pull only")
    print("on a meaningful partial (|r| >= ~0.15). Adoption then needs the")
    print("3-replicate multi-season pass (2024, 2025, 2026).")


if __name__ == '__main__':
    main()
