#!/usr/bin/env python3
"""hitter_stuff_profile.py -- pitcher-agnostic hitter "stuff profile".

For each hitter, measures how they perform (whiff, chase, etc.) against every
TYPE of pitch, where "type" is defined physically, not by label. Pitch types
are k-means clusters of the league's pitch population over velocity, IVB, HB,
and release slot (arm angle).

A hitter's performance vs a type is a similarity-kernel-weighted average over
all the pitches they have faced -- weights peak at the cluster centroid and
fall off with physical distance. This soft matching borrows from physically
similar pitches, which is what makes a ~600-pitch sample enough to profile
12 types: a "changeup" that moves like a sinker contributes to the sinker
type, and a hitter's slider read informs the nearby sweeper read.

Each hitter is profiled separately vs RHP and vs LHP (LHP pitches are mirrored
into a common frame so the cluster set is shared, but performance is split by
the pitcher's actual hand so the platoon is preserved). Per-type rates are
regressed for sample size toward the hitter's baseline plus the league type
effect.

A specific-pitcher matchup is then just a lookup: locate that pitcher's
pitches among the clusters and read the hitter's profile rows.

Run from anywhere:   python3 scripts/research/hitter/hitter_stuff_profile.py
Reads  data/all_pitches_rs_cache.pkl ; writes a text report + CSV to ~/Downloads.
"""

import csv
import math
import os
import pickle
import random
import statistics
from collections import Counter, defaultdict

# ── Config ───────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PICKLE_PATH = os.path.join(REPO_ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT_DIR = os.path.expanduser('~/Downloads')

N_CLUSTERS = 12
CLUSTER_SAMPLE = 25000
KMEANS_ITERS = 50
KMEANS_SEED = 0
ARM_RANGE = (-10.0, 80.0)     # arm angles outside this are data errors
BANDWIDTHS = [0.30, 0.40, 0.50, 0.60, 0.75, 0.95, 1.20]
TARGET_NEFF = 0.40            # each type's effective sample ~ this share of pool
QUALIFIED_MIN_PITCHES = 400
PROJ_METRICS = ['whiff', 'chase', 'zcontact', 'csw']

TARGET_HITTERS = [
    'Wood, James', 'Lile, Daylen', 'Abrams, CJ', 'Young, Jacob',
    'Nuñez, Nasim', 'García Jr., Luis', 'Vivas, Jorbit', 'Tena, José',
    'Ruiz, Keibert', 'Millas, Drew', 'Crews, Dylan',
]

RAW = {'velo': 'Velocity', 'ivb': 'IndVertBrk', 'hb': 'HorzBrk',
       'arm': 'ArmAngle', 'relx': 'RelPosX', 'relz': 'RelPosZ',
       'ext': 'Extension', 'spin': 'Spin Rate', 'vaa': 'VAA', 'haa': 'HAA'}
# A pitch TYPE is velocity + movement + slot. Extension is a pitcher mechanic
# that fragments clusters without defining a type; spin/VAA/HAA are redundant
# with velo+movement. So clustering and matching use velo/IVB/HB/slot only.
CENTROID_FEATS = ['velo', 'ivb', 'hb', 'arm', 'relx', 'relz']
CLUSTER_DIMS = ['velo', 'ivb', 'hb', 'arm']

SWING = {'Swinging Strike', 'Foul', 'In Play'}
CONTACT = {'Foul', 'In Play'}
CSW_DESC = {'Called Strike', 'Swinging Strike'}

# Movement-shape archetypes (IVB, HB), RHP perspective -- from
# pitch_subtype_classifier.py. Used only to name each data-driven cluster.
SUBTYPES = [
    ('Gyro Fastball', 'FF', 9, 0), ('Inefficient FF', 'FF', 13, 6),
    ('Deadzone FB', 'FF', 14, 13.5), ('Running Fastball', 'FF', 15, 16),
    ('Relative Cut FF', 'FF', 16, 1.5), ('Standard FF', 'FF', 17, 9),
    ('Rider', 'FF', 19.5, 6.5), ("Ride n' Run", 'FF', 19.5, 11),
    ('Gyro Sinker', 'SI', 9, 12), ('Sinker', 'SI', 9, 16),
    ('Running Sinker', 'SI', 9.5, 19.5), ('Heavy Sinker', 'SI', 5, 15.5),
    ('Heavy Runner', 'SI', 5, 20), ('Diver', 'SI', -1, 17),
    ('Gyro Cutter', 'FC', 8, 0), ('Standard Cutter', 'FC', 10, -3),
    ('Sweeping Cutter', 'FC', 9.5, -6.5), ('Backspinner', 'FC', 13.5, 0.5),
    ('Gyro Slider', 'SL', 1, -2), ('Slutter', 'SL', 5, -4.5),
    ('Standard Slider', 'SL', 0.5, -6.5), ('Sweeper', 'ST', -1.5, -15),
    ('Gyro Curve', 'CU', -5, -2), ('IE Downer', 'CU', -9, -2),
    ('Standard Curve', 'CU', -13, -10), ('Downer Curve', 'CU', -15, -5),
    ('Efficient Curve', 'CU', -18, -14), ('IE Slurve', 'SV', -6.5, -7),
    ('Slurve', 'SV', -8, -14), ('Efficient Slurve', 'SV', -12, -19),
]


# ── Helpers ──────────────────────────────────────────────────────────────
def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ── Feature space (LHP mirrored to RHP perspective) ──────────────────────
def mirrored_raw(p):
    """Physical features in RHP perspective. For LHP, HB and release-X are
    sign-flipped. Arm angle is handedness-neutral; out-of-range values are
    data errors and dropped."""
    flip = -1.0 if p.get('Throws') == 'L' else 1.0
    out = {'velo': safe_float(p.get(RAW['velo'])),
           'ivb': safe_float(p.get(RAW['ivb'])),
           'relz': safe_float(p.get(RAW['relz']))}
    hb = safe_float(p.get(RAW['hb']))
    out['hb'] = hb * flip if hb is not None else None
    rx = safe_float(p.get(RAW['relx']))
    out['relx'] = rx * flip if rx is not None else None
    arm = safe_float(p.get(RAW['arm']))
    if arm is not None and not (ARM_RANGE[0] <= arm <= ARM_RANGE[1]):
        arm = None
    out['arm'] = arm
    return out


def compute_norms(pitches):
    norms = {}
    for f in CENTROID_FEATS:
        vals = [mirrored_raw(p)[f] for p in pitches]
        vals = [v for v in vals if v is not None]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        norms[f] = (m, sd)
    return norms


def get_z(p, norms):
    """Standardized features, or None if the pitch cannot be placed."""
    r = mirrored_raw(p)
    z = {}
    for f in ('velo', 'ivb', 'hb'):
        if r[f] is None:
            return None
        m, sd = norms[f]
        z[f] = (r[f] - m) / sd if sd > 0 else 0.0
    for f in ('arm', 'relx', 'relz'):
        v = r[f]
        if v is None:
            z[f] = None
        else:
            m, sd = norms[f]
            z[f] = (v - m) / sd if sd > 0 else 0.0
    if z['arm'] is None and (z['relx'] is None or z['relz'] is None):
        return None
    return z


def dist(cz, pz):
    """Mean per-block squared standardized distance: velo, IVB, HB, geometry.
    Geometry is arm angle, or the release-point pair when arm angle is
    missing (AAA / data-error fallback). Centroids carry both."""
    d = ((cz['velo'] - pz['velo']) ** 2 + (cz['ivb'] - pz['ivb']) ** 2
         + (cz['hb'] - pz['hb']) ** 2)
    if pz['arm'] is not None:
        d += (cz['arm'] - pz['arm']) ** 2
    else:
        d += 0.5 * ((cz['relx'] - pz['relx']) ** 2
                    + (cz['relz'] - pz['relz']) ** 2)
    return d / 4.0


# ── k-means clustering ───────────────────────────────────────────────────
def kmeans(points, k, iters, seed):
    """Pure-Python k-means with k-means++ initialization."""
    rng = random.Random(seed)
    dim = len(points[0])
    centroids = [list(points[rng.randrange(len(points))])]
    for _ in range(k - 1):
        d2s = []
        for p in points:
            best = min(sum((p[i] - c[i]) ** 2 for i in range(dim))
                       for c in centroids)
            d2s.append(best)
        tot = sum(d2s)
        if tot <= 0:
            centroids.append(list(points[rng.randrange(len(points))]))
            continue
        target = rng.random() * tot
        acc = 0.0
        for p, d in zip(points, d2s):
            acc += d
            if acc >= target:
                centroids.append(list(p))
                break
        else:
            centroids.append(list(points[-1]))
    for _ in range(iters):
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for p in points:
            best, bd = 0, float('inf')
            for ci, c in enumerate(centroids):
                d = 0.0
                for i in range(dim):
                    diff = p[i] - c[i]
                    d += diff * diff
                if d < bd:
                    bd, best = d, ci
            counts[best] += 1
            cs = sums[best]
            for i in range(dim):
                cs[i] += p[i]
        moved = 0.0
        newc = []
        for ci in range(k):
            if counts[ci] > 0:
                nc = [s / counts[ci] for s in sums[ci]]
            else:
                nc = list(points[rng.randrange(len(points))])
            moved += sum((nc[i] - centroids[ci][i]) ** 2 for i in range(dim))
            newc.append(nc)
        centroids = newc
        if moved < 1e-9:
            break
    return centroids


def nearest_subtype(ivb, hb):
    best, bd, bbase = None, 1e18, None
    for name, base, rivb, rhb in SUBTYPES:
        d = (ivb - rivb) ** 2 + (hb - rhb) ** 2
        if d < bd:
            bd, best, bbase = d, name, base
    return best, bbase


def slot_word(arm):
    if arm >= 46:
        return 'high'
    if arm >= 33:
        return '3/4'
    if arm >= 18:
        return 'low'
    return 'sidearm'


def cluster_label(velo, ivb, hb, arm):
    name, base = nearest_subtype(ivb, hb)
    if base in ('FF', 'SI') and velo < 88:    # slow + arm-side shape = offspeed
        name = 'Changeup'
    return f"{velo:.0f} {name}, {slot_word(arm)}"


def build_clusters(mlb_pitches, norms):
    pts, refs = [], []
    for p in mlb_pitches:
        z = p.get('_z')
        if z is None or z['arm'] is None or z['relx'] is None \
                or z['relz'] is None:
            continue
        pts.append(tuple(z[d] for d in CLUSTER_DIMS))
        refs.append(p)
    rng = random.Random(KMEANS_SEED)
    sample = pts
    if len(pts) > CLUSTER_SAMPLE:
        sample = [pts[i] for i in rng.sample(range(len(pts)), CLUSTER_SAMPLE)]
    centroids4 = kmeans(sample, N_CLUSTERS, KMEANS_ITERS, KMEANS_SEED)

    sums = [{f: 0.0 for f in CENTROID_FEATS} for _ in range(N_CLUSTERS)]
    counts = [0] * N_CLUSTERS
    for p, pt in zip(refs, pts):
        best, bd = 0, float('inf')
        for ci, c in enumerate(centroids4):
            d = sum((pt[i] - c[i]) ** 2 for i in range(len(CLUSTER_DIMS)))
            if d < bd:
                bd, best = d, ci
        counts[best] += 1
        z = p['_z']
        for f in CENTROID_FEATS:
            sums[best][f] += z[f]

    clusters = []
    for ci in range(N_CLUSTERS):
        if counts[ci] == 0:
            continue
        cz = {f: sums[ci][f] / counts[ci] for f in CENTROID_FEATS}
        raw = {f: cz[f] * norms[f][1] + norms[f][0] for f in CENTROID_FEATS}
        clusters.append({'z': cz, 'raw': raw, 'n': counts[ci]})

    clusters.sort(key=lambda c: -c['raw']['velo'])     # hard to soft
    seen = {}
    for i, c in enumerate(clusters):
        c['id'] = i
        r = c['raw']
        lab = cluster_label(r['velo'], r['ivb'], r['hb'], r['arm'])
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab} #{seen[lab]}"
        else:
            seen[lab] = 1
        c['label'] = lab
    return clusters


# ── Kernel-weighted metrics ──────────────────────────────────────────────
def make_wf(cid, h):
    inv = 1.0 / (2 * h * h)
    return lambda p: math.exp(-p['_d2'][cid] * inv)


def wmetrics(pitches, weight_fn):
    """Kernel-weighted performance (weight_fn==1 -> plain rates)."""
    wsum = w2 = 0.0
    d_tot = d_sw = d_ooz = d_izsw = 0.0
    n_whiff = n_swooz = n_izc = n_csw = 0.0
    for p in pitches:
        w = weight_fn(p)
        if w <= 0:
            continue
        wsum += w
        w2 += w * w
        d_tot += w
        desc = p['Description']
        swung = desc in SWING
        if swung:
            d_sw += w
            if desc == 'Swinging Strike':
                n_whiff += w
        if desc in CSW_DESC:
            n_csw += w
        iz = p.get('InZone')
        if iz == 'No':
            d_ooz += w
            if swung:
                n_swooz += w
        elif iz == 'Yes' and swung:
            d_izsw += w
            if desc in CONTACT:
                n_izc += w
    return {
        'n_eff': (wsum * wsum / w2) if w2 > 0 else 0.0,
        'd_sw': d_sw, 'd_ooz': d_ooz, 'd_izsw': d_izsw, 'd_tot': d_tot,
        'whiff': n_whiff / d_sw if d_sw > 0 else None,
        'chase': n_swooz / d_ooz if d_ooz > 0 else None,
        'zcontact': n_izc / d_izsw if d_izsw > 0 else None,
        'csw': n_csw / d_tot if d_tot > 0 else None,
    }


def RAW_WF(p):
    return 1.0


# ── Diagnostics ──────────────────────────────────────────────────────────
def diag_redundancy(rhp_mlb):
    feats = ['velo', 'ivb', 'hb', 'arm', 'ext', 'spin', 'vaa', 'haa']
    sample = rhp_mlb if len(rhp_mlb) <= 60000 else rhp_mlb[:60000]
    cols = {}
    for f in feats:
        if f == 'hb':
            cols[f] = [mirrored_raw(p)['hb'] for p in sample]
        elif f == 'arm':
            cols[f] = [mirrored_raw(p)['arm'] for p in sample]
        else:
            cols[f] = [safe_float(p.get(RAW[f])) for p in sample]
    print("\n── Feature redundancy (Pearson r) ──")
    print("        " + "".join(f"{f:>7}" for f in feats))
    for fa in feats:
        row = [f"{fa:>6}: "]
        for fb in feats:
            xs, ys = [], []
            for va, vb in zip(cols[fa], cols[fb]):
                if va is not None and vb is not None:
                    xs.append(va)
                    ys.append(vb)
            r = pearson(xs, ys)
            row.append(f"{r:>7.2f}" if r is not None else f"{'--':>7}")
        print("".join(row))
    print("  Clustering uses velo/ivb/hb/arm. ext fragments types without")
    print("  defining one; spin/vaa/haa are redundant with velo+movement.")


def diag_reliability(qualified, cids, h):
    """Split-half reliability at bandwidth h. 'r' = raw kernel-metric
    reliability (blur-inflated); 'dev' = reliability of the pitch-specific
    deviation from the hitter's overall rate (the honest tuning target)."""
    floors = {'whiff': ('d_sw', 8), 'chase': ('d_ooz', 8),
              'zcontact': ('d_izsw', 8), 'csw': ('d_tot', 15)}
    pairs = {m: ([], []) for m in floors}
    dev = {m: ([], []) for m in floors}
    for ps in qualified.values():
        half = len(ps) // 2
        h1, h2 = ps[:half], ps[half:]
        o1, o2 = wmetrics(h1, RAW_WF), wmetrics(h2, RAW_WF)
        for cid in cids:
            m1 = wmetrics(h1, make_wf(cid, h))
            m2 = wmetrics(h2, make_wf(cid, h))
            for m, (dk, fl) in floors.items():
                v1, v2 = m1[m], m2[m]
                if v1 is None or v2 is None or m1[dk] < fl or m2[dk] < fl:
                    continue
                pairs[m][0].append(v1)
                pairs[m][1].append(v2)
                if o1[m] is not None and o2[m] is not None:
                    dev[m][0].append(v1 - o1[m])
                    dev[m][1].append(v2 - o2[m])

    def sb(r):
        return (2 * r / (1 + r)) if (r is not None and r > -1) else None
    out = {}
    for m in floors:
        out[m] = {'r': sb(pearson(*pairs[m])),
                  'dev': sb(pearson(*dev[m])),
                  'n': len(pairs[m][0])}
    return out


def diag_discrimination(qualified, cids, h):
    ratios = []
    for ps in qualified.values():
        n = len(ps)
        if n == 0:
            continue
        for cid in cids:
            ratios.append(wmetrics(ps, make_wf(cid, h))['n_eff'] / n)
    return median(ratios) or 0.0


def derive_k(r, m_neff):
    if r is None or r <= 0.02:
        return 1e5
    if r >= 0.98:
        return m_neff * 0.02
    return m_neff * (1 - r) / r


# ── Formatting ───────────────────────────────────────────────────────────
def pct(x, d=1):
    return f"{x * 100:.{d}f}%" if x is not None else "  --"


def signed(x, d=1):
    return f"{x * 100:.{d}f}" if x is not None else "--"


# ── Profile building ─────────────────────────────────────────────────────
def build_profile(pool, clusters, league_raw, league_cl, h, kvals):
    baseline = wmetrics(pool, RAW_WF)
    per = {}
    for c in clusters:
        cid = c['id']
        obs = wmetrics(pool, make_wf(cid, h))
        reg = {}
        for m in PROJ_METRICS:
            o = obs[m]
            effect = (league_cl[cid][m] or 0) - (league_raw[m] or 0)
            prior = min(1.0, max(0.0, (baseline[m] or 0) + effect))
            if o is None:
                reg[m] = prior
            else:
                ne, k = obs['n_eff'], kvals[m]
                reg[m] = (ne * o + k * prior) / (ne + k)
        per[cid] = {'obs': obs, 'reg': reg}
    return {'baseline': baseline, 'clusters': per}


# ── Report ───────────────────────────────────────────────────────────────
def write_report(hitters, clusters, league, h, reliab, kvals, latest_date):
    txt = os.path.join(OUT_DIR, 'wsh_stuff_profiles.txt')
    csvp = os.path.join(OUT_DIR, 'wsh_stuff_profiles.csv')
    THIN = 75   # below this many pitches a per-type breakdown is not shown

    with open(txt, 'w', encoding='utf-8') as f:
        def emit(s=''):
            f.write(s + '\n')

        emit("WSH HITTERS: PITCHER-AGNOSTIC STUFF PROFILE")
        emit(f"How each hitter fares vs physically-defined pitch types | "
             f"data through {latest_date}")
        emit("=" * 94)
        emit()
        emit(f"METHOD: the league's pitches are k-means clustered into "
             f"{len(clusters)} TYPES by")
        emit("velocity, IVB, HB and release slot (labels ignored). A hitter's")
        emit("rate vs a type is a similarity-kernel-weighted average over every")
        emit("pitch they faced (weights peak at the type, fade with physical")
        emit("distance), then regressed for sample size. RHP and LHP are")
        emit("profiled separately. Whiff is the strongest signal.")
        emit()
        emit(f"THE {len(clusters)} PITCH TYPES (movement in RHP perspective; "
             f"LgN = league size):")
        emit(f"  {'#':>2}  {'Type':<27}{'Velo':>6}{'IVB':>7}{'HB':>7}"
             f"{'Slot':>8}{'LgN':>9}")
        for c in clusters:
            r = c['raw']
            emit(f"  {c['id'] + 1:>2}  {c['label']:<27}{r['velo']:>6.1f}"
                 f"{r['ivb']:>+7.1f}{r['hb']:>+7.1f}{r['arm']:>7.0f}d"
                 f"{c['n']:>9}")
        emit()
        emit(f"TUNING: kernel bandwidth h = {h} (tuned on pitch-specific "
             f"reliability)")
        for m in PROJ_METRICS:
            rr = reliab[m]['dev']
            rs = f"{rr:.2f}" if rr is not None else "--"
            emit(f"  {m:<9} pitch-type reliability {rs:>5}   "
                 f"regression k = {kvals[m]:.0f}")
        emit("  Higher k = thinner signal, regressed harder toward baseline.")
        emit()

        for hand, hname in (('R', 'RHP'), ('L', 'LHP')):
            emit("=" * 94)
            emit(f"{hname} WHIFF PROFILE -- regressed whiff% by type "
                 f"(col # = type above; compare each hitter to LEAGUE)")
            head = f"  {'Hitter':<18}{'N':>6}{'base':>5} "
            head += "".join(f"{c['id'] + 1:>5}" for c in clusters)
            emit(head)
            for hd in hitters:
                prof = hd['profiles'].get(hand)
                if not prof:
                    emit(f"  {hd['name']:<18}  (no {hname} sample)")
                    continue
                b = prof['baseline']
                row = (f"  {hd['name']:<18}{b['n_eff']:>6.0f}"
                       f"{(b['whiff'] or 0) * 100:>5.0f} ")
                for c in clusters:
                    row += f"{prof['clusters'][c['id']]['reg']['whiff'] * 100:>5.0f}"
                emit(row)
            lg = league[hand]
            lrow = (f"  {'LEAGUE avg':<18}{'':>6}"
                    f"{(lg['raw']['whiff'] or 0) * 100:>5.0f} ")
            for c in clusters:
                lrow += f"{(lg['cl'][c['id']]['whiff'] or 0) * 100:>5.0f}"
            emit(lrow)
            emit()

        for hd in hitters:
            emit("=" * 94)
            emit(f"{hd['name'].upper()}")
            for hand, hname in (('R', 'RHP'), ('L', 'LHP')):
                prof = hd['profiles'].get(hand)
                if not prof:
                    emit(f"  vs {hname}: no sample on record")
                    continue
                b = prof['baseline']
                emit(f"  vs {hname} (bats {hd['bats'].get(hand, '?')}, "
                     f"{b['n_eff']:.0f} pitches) -- baseline whiff "
                     f"{pct(b['whiff'])}  chase {pct(b['chase'])}  "
                     f"CSW {pct(b['csw'])}")
                if b['n_eff'] < THIN:
                    emit("    (thin sample -- per-type breakdown omitted; the "
                         "matrix row is regressed near baseline)")
                    continue
                lgcl = league[hand]['cl']
                emit(f"    {'Type':<27}{'Neff':>6}{'Whiff':>9}{'vsLg':>7}"
                     f"{'Chase':>9}")
                for c in clusters:
                    pc = prof['clusters'][c['id']]
                    reg = pc['reg']
                    dw = reg['whiff'] - (lgcl[c['id']]['whiff'] or 0)
                    emit(f"    {c['label']:<27}{pc['obs']['n_eff']:>6.0f}"
                         f"{pct(reg['whiff']):>9}{signed(dw):>7}"
                         f"{pct(reg['chase']):>9}")
            for hand, hname in (('R', 'RHP'), ('L', 'LHP')):
                prof = hd['profiles'].get(hand)
                if not prof or prof['baseline']['n_eff'] < THIN:
                    continue
                lgcl = league[hand]['cl']
                exp = []
                for c in clusters:
                    rw = prof['clusters'][c['id']]['reg']['whiff']
                    exp.append((rw - (lgcl[c['id']]['whiff'] or 0), c, rw))
                exp.sort(key=lambda e: e[0], reverse=True)
                hi, lo = exp[0], exp[-1]
                emit(f"  READ vs {hname}: most exposed to {hi[1]['label']} "
                     f"({pct(hi[2])} whiff, {signed(hi[0])} vs league); "
                     f"best vs {lo[1]['label']} ({signed(lo[0])} vs league).")
        emit("=" * 94)
        emit()
        emit("CAVEATS: a scouting lens, not a precise projection. Per-type")
        emit("rates are kernel-smoothed and regressed -- trust direction and")
        emit("rank over the exact number. 'vsLg' is vs the league-average")
        emit("hitter for that type. LHP samples are ~3x thinner than RHP.")
        emit("Crews is ~93% AAA (lower competition; release-point geometry")
        emit("where arm angle is missing).")

    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        cw = csv.writer(f)
        cw.writerow(['hitter', 'vs_hand', 'pitch_type', 'n_eff',
                     'whiff', 'whiff_vs_league', 'chase', 'chase_vs_league',
                     'zcontact', 'csw'])

        def num(x):
            return round(x * 100, 1) if x is not None else ''

        for hd in hitters:
            for hand in ('R', 'L'):
                prof = hd['profiles'].get(hand)
                if not prof:
                    continue
                b = prof['baseline']
                lg = league[hand]
                cw.writerow([hd['name'], hand, 'BASELINE',
                             round(b['n_eff'], 1),
                             num(b['whiff']),
                             num((b['whiff'] or 0) - (lg['raw']['whiff'] or 0)),
                             num(b['chase']),
                             num((b['chase'] or 0) - (lg['raw']['chase'] or 0)),
                             num(b['zcontact']), num(b['csw'])])
                for c in clusters:
                    pc = prof['clusters'][c['id']]
                    reg = pc['reg']
                    lc = lg['cl'][c['id']]
                    cw.writerow([
                        hd['name'], hand, c['label'],
                        round(pc['obs']['n_eff'], 1),
                        num(reg['whiff']), num(reg['whiff'] - (lc['whiff'] or 0)),
                        num(reg['chase']), num(reg['chase'] - (lc['chase'] or 0)),
                        num(reg['zcontact']), num(reg['csw'])])
    return txt, csvp


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading pitch cache…")
    with open(PICKLE_PATH, 'rb') as fh:
        pitches = pickle.load(fh)
    print(f"  {len(pitches)} pitches")

    norms = compute_norms(pitches)
    usable = 0
    for p in pitches:
        p['_z'] = get_z(p, norms)
        if p['_z'] is not None:
            usable += 1
    print(f"  {usable} pitches usable")

    mlb = [p for p in pitches if p.get('_source') == 'MLB'
           and p.get('_z') is not None]
    clusters = build_clusters(mlb, norms)
    cids = [c['id'] for c in clusters]
    print(f"  built {len(clusters)} pitch-type clusters")

    for p in pitches:
        if p.get('_z') is not None:
            p['_d2'] = {c['id']: dist(c['z'], p['_z']) for c in clusters}

    latest_date = max((p.get('Game Date') for p in pitches
                       if p.get('Game Date')), default='?')

    print("\n" + "=" * 70)
    print("DIAGNOSTIC PHASE")
    print("=" * 70)
    print("\nPITCH-TYPE CLUSTERS:")
    print(f"  {'#':>2}  {'Label':<27}{'Velo':>6}{'IVB':>7}{'HB':>7}"
          f"{'Arm':>6}{'N':>8}")
    for c in clusters:
        r = c['raw']
        print(f"  {c['id'] + 1:>2}  {c['label']:<27}{r['velo']:>6.1f}"
              f"{r['ivb']:>+7.1f}{r['hb']:>+7.1f}{r['arm']:>6.0f}{c['n']:>8}")

    diag_redundancy([p for p in mlb if p.get('Throws') == 'R'])

    by_batter = defaultdict(list)
    for p in mlb:
        if p.get('Throws') == 'R':
            by_batter[p.get('Batter')].append(p)
    rng = random.Random(42)
    qualified = {}
    for batter, ps in by_batter.items():
        if batter and len(ps) >= QUALIFIED_MIN_PITCHES:
            shuffled = ps[:]
            rng.shuffle(shuffled)
            qualified[batter] = shuffled
    print(f"\nReliability pool: {len(qualified)} MLB hitters")

    print("\n── Bandwidth sweep (raw vs pitch-specific-deviation reliability) ──")
    print(f"  {'h':>6}{'whiffR':>9}{'whiffDev':>10}{'chaseDev':>10}"
          f"{'zconDev':>9}{'cswDev':>9}{'Neff%':>8}")
    sweep, disc = {}, {}
    for hh in BANDWIDTHS:
        rel = diag_reliability(qualified, cids, hh)
        sweep[hh] = rel
        disc[hh] = diag_discrimination(qualified, cids, hh)
        vals = []
        for m, key in [('whiff', 'r'), ('whiff', 'dev'), ('chase', 'dev'),
                       ('zcontact', 'dev'), ('csw', 'dev')]:
            v = rel[m][key]
            vals.append(f"{v:.2f}" if v is not None else "--")
        print(f"  {hh:>6.2f}{vals[0]:>9}{vals[1]:>10}{vals[2]:>10}"
              f"{vals[3]:>9}{vals[4]:>9}{disc[hh] * 100:>7.0f}%")

    # The deviation-reliability curve rises monotonically with h (a wide kernel
    # blurs every type toward the hitter's average -- reproducible but flat).
    # So among bandwidths with a real, reliable signal, pick the one whose
    # per-type effective sample is closest to TARGET_NEFF: wide enough to
    # borrow from similar pitches, narrow enough to stay type-specific.
    eligible = [hh for hh in BANDWIDTHS
                if (sweep[hh]['whiff']['r'] or 0) >= 0.65
                and (sweep[hh]['whiff']['dev'] or 0) >= 0.55]
    if not eligible:
        eligible = list(BANDWIDTHS)
    best_h = min(eligible, key=lambda hh: abs(disc[hh] - TARGET_NEFF))
    print(f"  -> h = {best_h}: whiff reliability "
          f"{sweep[best_h]['whiff']['dev']:.2f}, per-type N_eff "
          f"{disc[best_h] * 100:.0f}% of pool")

    reliab = sweep[best_h]
    neffs = [wmetrics(ps, make_wf(cid, best_h))['n_eff']
             for ps in qualified.values() for cid in cids]
    m_neff = median(neffs) or 100.0
    kvals = {m: derive_k(reliab[m]['dev'], m_neff) for m in PROJ_METRICS}
    print(f"\n── Regression constants (median N_eff = {m_neff:.0f}) ──")
    for m in PROJ_METRICS:
        rr = reliab[m]['dev']
        rs = f"{rr:.2f}" if rr is not None else "--"
        print(f"  {m:<9} dev-reliability {rs:>5}   k = {kvals[m]:.0f}")

    print("\n" + "=" * 70)
    print("BUILDING PROFILES")
    print("=" * 70)
    league = {}
    for hand in ('R', 'L'):
        lg = [p for p in mlb if p.get('Throws') == hand]
        league[hand] = {'raw': wmetrics(lg, RAW_WF),
                        'cl': {cid: wmetrics(lg, make_wf(cid, best_h))
                               for cid in cids}}

    def faced(batter, hand):
        crews = batter == 'Crews, Dylan'
        out = []
        for p in pitches:
            if p.get('Batter') != batter or p.get('Throws') != hand:
                continue
            if p.get('_z') is None:
                continue
            if not crews and p.get('_source') != 'MLB':
                continue
            out.append(p)
        return out

    hitters = []
    for name in TARGET_HITTERS:
        profiles, bats = {}, {}
        for hand in ('R', 'L'):
            pool = faced(name, hand)
            if not pool:
                continue
            bats[hand] = Counter(p.get('Bats') for p in pool).most_common(1)[0][0]
            profiles[hand] = build_profile(pool, clusters, league[hand]['raw'],
                                           league[hand]['cl'], best_h, kvals)
        hitters.append({'name': name, 'profiles': profiles, 'bats': bats})
        rn = profiles['R']['baseline']['n_eff'] if 'R' in profiles else 0
        ln = profiles['L']['baseline']['n_eff'] if 'L' in profiles else 0
        print(f"  {name}: {rn:.0f} vs RHP, {ln:.0f} vs LHP")

    txt, csvp = write_report(hitters, clusters, league, best_h, reliab,
                             kvals, latest_date)
    print(f"\n  wrote {txt}")
    print(f"  wrote {csvp}")
    print("\nDone.")


if __name__ == '__main__':
    main()
