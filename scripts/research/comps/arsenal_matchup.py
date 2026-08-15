#!/usr/bin/env python3
"""arsenal_matchup.py -- hitter-vs-pitcher arsenal scouting via pitch similarity.

Builds a (potentially predictive) scouting report for a set of hitters against
one pitcher's arsenal. Instead of trusting pitch-type labels, every pitch is a
point in a physical feature space (velocity, IVB, HB, release geometry,
extension). For each of the target pitcher's pitches we find, via a Gaussian
similarity kernel, the pitches each hitter has actually faced that physically
resemble it, and measure how the hitter performed against that soft-matched
set. A 90 mph "changeup" that moves like the pitcher's sinker still counts.

Run from anywhere:   python3 scripts/research/comps/arsenal_matchup.py

Reads  data/all_pitches_rs_cache.pkl  (the retagged pitch cache).
Writes a CSV + a text scouting report to ~/Downloads.

The script runs a DIAGNOSTIC phase first (feature redundancy, sample coverage,
split-half reliability, count confound), prints it, and uses the measured
reliability to auto-tune the kernel bandwidth and the per-metric regression
constants. The report is then built with those data-justified parameters.
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

TARGET_PITCHER = 'Elder, Bryce'
ELDER_PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'CH']

# Exact Batter strings ("Last, First") as stored in the pickle.
TARGET_HITTERS = [
    'Wood, James', 'Lile, Daylen', 'Abrams, CJ', 'Young, Jacob',
    'Nuñez, Nasim', 'García Jr., Luis', 'Vivas, Jorbit', 'Tena, José',
    'Ruiz, Keibert', 'Millas, Drew', 'Crews, Dylan',
]

# Raw pickle field name for each feature key.
RAW = {'velo': 'Velocity', 'ivb': 'IndVertBrk', 'hb': 'HorzBrk',
       'ext': 'Extension', 'arm': 'ArmAngle', 'relx': 'RelPosX',
       'relz': 'RelPosZ', 'spin': 'Spin Rate', 'vaa': 'VAA', 'haa': 'HAA'}

# Features every usable pitch must have. Arm angle is handled separately
# (missing for all AAA data -> release-point fallback).
MATCH_FEATS = ['velo', 'ivb', 'hb', 'ext', 'relx', 'relz']

SWING = {'Swinging Strike', 'Foul', 'In Play'}
CONTACT = {'Foul', 'In Play'}
CSW_DESC = {'Called Strike', 'Swinging Strike'}

BANDWIDTHS = [0.30, 0.40, 0.50, 0.60, 0.75, 0.95, 1.20]
QUALIFIED_MIN_PITCHES = 400   # vs-RHP pitches to enter the reliability pool
PROJ_METRICS = ['whiff', 'chase', 'zcontact', 'csw']
ARM_RANGE = (-10.0, 80.0)     # arm angles outside this are data errors


# ── Small helpers ────────────────────────────────────────────────────────
def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def arm_angle(p):
    """Arm angle, or None if missing or an out-of-range data error."""
    a = safe_float(p.get('ArmAngle'))
    if a is None or not (ARM_RANGE[0] <= a <= ARM_RANGE[1]):
        return None
    return a


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


def is_two_strike(p):
    c = p.get('Count')
    return bool(c) and '-' in c and c.split('-')[1] == '2'


# ── Feature space ────────────────────────────────────────────────────────
def compute_norms(rhp_pitches):
    """Mean and SD of each feature over the RHP pitch universe (for z-scoring)."""
    norms = {}
    for f in ['velo', 'ivb', 'hb', 'ext', 'relx', 'relz', 'arm']:
        if f == 'arm':
            vals = [arm_angle(p) for p in rhp_pitches]
        else:
            vals = [safe_float(p.get(RAW[f])) for p in rhp_pitches]
        vals = [v for v in vals if v is not None]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        norms[f] = (m, sd)
    return norms


def get_z(p, norms):
    """Standardized feature dict for a pitch, or None if unusable for matching."""
    z = {}
    for f in MATCH_FEATS:
        v = safe_float(p.get(RAW[f]))
        if v is None:
            return None
        m, sd = norms[f]
        z[f] = (v - m) / sd if sd > 0 else 0.0
    a = arm_angle(p)
    if a is None:
        z['arm'] = None
    else:
        m, sd = norms['arm']
        z['arm'] = (a - m) / sd if sd > 0 else 0.0
    return z


def elder_centroids(elder_pitches, norms):
    """Mean physical vector of each of Elder's retagged pitch types, z-scored."""
    cents = {}
    for pt in ELDER_PITCH_ORDER:
        grp = [p for p in elder_pitches if p.get('Pitch Type') == pt]
        raw = {}
        for f in ['velo', 'ivb', 'hb', 'ext', 'relx', 'relz', 'arm', 'spin']:
            if f == 'arm':
                vals = [arm_angle(p) for p in grp]
            else:
                vals = [safe_float(p.get(RAW[f])) for p in grp]
            vals = [v for v in vals if v is not None]
            raw[f] = (sum(vals) / len(vals)) if vals else None
        z = {}
        for f in MATCH_FEATS + ['arm']:
            m, sd = norms[f]
            z[f] = (raw[f] - m) / sd if sd > 0 else 0.0
        cents[pt] = {'z': z, 'raw': raw, 'n': len(grp)}
    return cents


def dist2(cz, pz):
    """Mean per-block squared standardized distance, centroid -> pitch.

    Five equal blocks: velo, IVB, HB, extension, geometry. Geometry is arm
    angle when the pitch has it, else the release-point pair (AAA fallback),
    collapsed to one block so MLB and AAA distances stay on the same scale.
    """
    d = ((cz['velo'] - pz['velo']) ** 2 + (cz['ivb'] - pz['ivb']) ** 2
         + (cz['hb'] - pz['hb']) ** 2 + (cz['ext'] - pz['ext']) ** 2)
    if pz['arm'] is not None:
        d += (cz['arm'] - pz['arm']) ** 2
    else:
        d += 0.5 * ((cz['relx'] - pz['relx']) ** 2
                    + (cz['relz'] - pz['relz']) ** 2)
    return d / 5.0


# ── Weighted metrics ─────────────────────────────────────────────────────
def make_wf(pt, h):
    """Single-centroid Gaussian kernel weight function."""
    inv = 1.0 / (2 * h * h)
    return lambda p: math.exp(-p['_d2'][pt] * inv)


def make_blend_wf(usage, h):
    """Usage-blended kernel: how much a pitch resembles the whole arsenal."""
    inv = 1.0 / (2 * h * h)
    items = [(pt, u) for pt, u in usage.items() if u > 0]
    return lambda p: sum(u * math.exp(-p['_d2'][pt] * inv) for pt, u in items)


def wmetrics(pitches, weight_fn):
    """Kernel-weighted performance over a pitch list. weight_fn -> raw if ==1."""
    wsum = w2 = 0.0
    d_tot = d_sw = d_ooz = d_izsw = 0.0
    n_whiff = n_swooz = n_izc = n_csw = 0.0
    bipw = bipx = 0.0
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
        if desc == 'In Play':
            xw = safe_float(p.get('xwOBA'))
            if xw is not None:
                bipw += w
                bipx += w * xw
    return {
        'n_eff': (wsum * wsum / w2) if w2 > 0 else 0.0,
        'w': wsum,
        'd_sw': d_sw, 'd_ooz': d_ooz, 'd_izsw': d_izsw,
        'd_tot': d_tot, 'bipw': bipw,
        'swing': d_sw / d_tot if d_tot > 0 else None,
        'whiff': n_whiff / d_sw if d_sw > 0 else None,
        'chase': n_swooz / d_ooz if d_ooz > 0 else None,
        'zcontact': n_izc / d_izsw if d_izsw > 0 else None,
        'csw': n_csw / d_tot if d_tot > 0 else None,
        'xwobacon': bipx / bipw if bipw > 0 else None,
    }


RAW_WF = lambda p: 1.0   # unweighted; wmetrics(pool, RAW_WF) -> plain rates


# ── Diagnostics ──────────────────────────────────────────────────────────
def diag_redundancy(rhp_mlb):
    """Pearson correlation among all candidate features over MLB RHP pitches."""
    feats = ['velo', 'ivb', 'hb', 'arm', 'ext', 'spin', 'vaa', 'haa']
    sample = rhp_mlb if len(rhp_mlb) <= 60000 else rhp_mlb[:60000]
    cols = {f: [safe_float(p.get(RAW[f])) for p in sample] for f in feats}
    print("\n── Feature redundancy (Pearson r, MLB RHP pitches) ──")
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
    print("  Core matching block uses velo/ivb/hb/geometry/ext. spin, vaa, haa")
    print("  are reported here to show they carry little independent signal.")


def diag_coverage(target_pools, cents, h):
    """Effective sample size per hitter per Elder pitch at bandwidth h."""
    print(f"\n── Coverage: effective sample size per matchup (h={h}) ──")
    print(f"  {'Hitter':<20}" + "".join(f"{pt:>8}" for pt in ELDER_PITCH_ORDER)
          + f"{'pool':>8}")
    for hitter, pool in target_pools.items():
        cells = []
        for pt in ELDER_PITCH_ORDER:
            ne = wmetrics(pool, make_wf(pt, h))['n_eff']
            cells.append(f"{ne:>8.0f}")
        print(f"  {hitter:<20}" + "".join(cells) + f"{len(pool):>8}")


def diag_reliability(qualified, h):
    """Split-half reliability at bandwidth h, Spearman-Brown corrected.

    Two flavours per metric:
      'r'   raw kernel-weighted metric reliability. Inflated by blur -- a wide
            kernel makes every Elder pitch collapse to the hitter's global
            average, which is trivially reproducible but has no specificity.
      'dev' reliability of the pitch-SPECIFIC deviation (the centroid metric
            minus the hitter's overall rate on that half). This is the honest
            tuning target: it falls off for a too-wide kernel (no signal left)
            and for a too-narrow one (pure noise).

    Denominator floors keep unstable thin cells out of the correlation.
    """
    floors = {'whiff': ('d_sw', 8), 'chase': ('d_ooz', 8),
              'zcontact': ('d_izsw', 8), 'csw': ('d_tot', 15),
              'xwobacon': ('bipw', 8)}
    pairs = {m: ([], []) for m in floors}
    dev = {m: ([], []) for m in floors}
    for ps in qualified.values():
        half = len(ps) // 2
        h1, h2 = ps[:half], ps[half:]
        o1, o2 = wmetrics(h1, RAW_WF), wmetrics(h2, RAW_WF)
        for pt in ELDER_PITCH_ORDER:
            m1 = wmetrics(h1, make_wf(pt, h))
            m2 = wmetrics(h2, make_wf(pt, h))
            for m, (dkey, floor) in floors.items():
                v1, v2 = m1[m], m2[m]
                if (v1 is None or v2 is None
                        or m1[dkey] < floor or m2[dkey] < floor):
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


def diag_discrimination(qualified, h):
    """Median effective-sample share (N_eff / pool size).

    A high share means the kernel is so wide that every Elder pitch pools
    nearly the hitter's whole history -- high reliability but no pitch
    specificity. A low share means the 5 pitches draw genuinely distinct
    sub-samples. Used to keep bandwidth selection from washing out the signal.
    """
    ratios = []
    for ps in qualified.values():
        n = len(ps)
        if n == 0:
            continue
        for pt in ELDER_PITCH_ORDER:
            ratios.append(wmetrics(ps, make_wf(pt, h))['n_eff'] / n)
    return median(ratios) or 0.0


def derive_k(r, m_neff):
    """Regression constant from measured reliability: r = N/(N+k) -> k.

    A metric with reliability r at typical effective sample m_neff needs
    ballast k = m_neff*(1-r)/r. Low reliability -> heavy regression to prior.
    """
    if r is None or r <= 0.02:
        return 1e5
    if r >= 0.98:
        return m_neff * 0.02
    return m_neff * (1 - r) / r


def diag_count_confound(cents, league, h):
    """Compare Elder's 2-strike usage to the 2-strike share of the matched pool."""
    print(f"\n── Count confound: 2-strike share, Elder vs matched pool (h={h}) ──")
    print(f"  {'Pitch':<8}{'Elder 2K%':>12}{'MatchedPool 2K%':>18}")
    elder = cents['_elder_pitches']
    for pt in ELDER_PITCH_ORDER:
        grp = [p for p in elder if p.get('Pitch Type') == pt]
        e2k = 100 * sum(1 for p in grp if is_two_strike(p)) / len(grp) if grp else 0
        wf = make_wf(pt, h)
        num = den = 0.0
        for p in league:
            w = wf(p)
            den += w
            if is_two_strike(p):
                num += w
        m2k = 100 * num / den if den > 0 else 0
        print(f"  {pt:<8}{e2k:>11.1f}%{m2k:>17.1f}%")


# ── Formatting ───────────────────────────────────────────────────────────
def pct(x, d=1):
    return f"{x * 100:.{d}f}%" if x is not None else "  --"


def woba(x):
    if x is None:
        return " --"
    s = f"{x:.3f}"
    return s[1:] if s.startswith('0.') else s   # .XXX, no leading zero


def signed(x, d=1):
    """Delta in points; no leading + on positives (house style)."""
    if x is None:
        return "  --"
    return f"{x * 100:.{d}f}"


# ── Report ───────────────────────────────────────────────────────────────
def elder_usage(elder_pitches):
    """Elder's pitch mix vs LHH and vs RHH (the aggregation weights)."""
    out = {}
    for hand in ('L', 'R'):
        grp = [p for p in elder_pitches if p.get('Bats') == hand]
        c = Counter(p.get('Pitch Type') for p in grp)
        tot = sum(c.get(pt, 0) for pt in ELDER_PITCH_ORDER)
        out[hand] = ({pt: c.get(pt, 0) / tot for pt in ELDER_PITCH_ORDER}
                     if tot else {pt: 0.0 for pt in ELDER_PITCH_ORDER})
    return out


def build_hitter_row(hitter, pool, actual, cents, league_pt, league_raw,
                     usage, h, kvals):
    """All numbers for one hitter: per-pitch, regressed, aggregate, actual."""
    baseline = wmetrics(pool, RAW_WF)
    hand = Counter(p.get('Bats') for p in pool).most_common(1)[0][0]
    use = usage.get(hand, usage['L'])

    per_pitch = {}
    for pt in ELDER_PITCH_ORDER:
        obs = wmetrics(pool, make_wf(pt, h))
        reg = {}
        for m in PROJ_METRICS:
            o = obs[m]
            pitch_effect = (league_pt[pt][m] or 0) - (league_raw[m] or 0)
            prior = min(1.0, max(0.0, (baseline[m] or 0) + pitch_effect))
            if o is None:
                reg[m] = prior
            else:
                ne, k = obs['n_eff'], kvals[m]
                reg[m] = (ne * o + k * prior) / (ne + k)
        per_pitch[pt] = {'obs': obs, 'reg': reg}

    agg = {}
    for m in PROJ_METRICS:
        agg[m] = sum(use[pt] * per_pitch[pt]['reg'][m] for pt in ELDER_PITCH_ORDER)

    # xwOBAcon: aggregate only (per-pitch BIP samples are too thin).
    blend = wmetrics(pool, make_blend_wf(use, h))
    lg_blend = wmetrics(league_pt['_league'], make_blend_wf(use, h))
    xc_obs = blend['xwobacon']
    xc_prior = (baseline['xwobacon'] or 0) + \
               ((lg_blend['xwobacon'] or 0) - (league_raw['xwobacon'] or 0))
    xc_prior = max(0.0, xc_prior)
    if xc_obs is None:
        xc_reg = xc_prior
    else:
        ne, k = blend['bipw'], kvals['xwobacon']
        xc_reg = (ne * xc_obs + k * xc_prior) / (ne + k)

    actual_m = wmetrics(actual, RAW_WF) if actual else None
    return {
        'hitter': hitter, 'hand': hand, 'usage': use,
        'baseline': baseline, 'per_pitch': per_pitch, 'agg': agg,
        'xwobacon_obs': xc_obs, 'xwobacon_reg': xc_reg,
        'xwobacon_base': baseline['xwobacon'],
        'proj_neff': blend['n_eff'],
        'actual': actual_m, 'actual_n': len(actual) if actual else 0,
    }


def weapon(row, metric='whiff'):
    """Elder's most exploitable pitch vs this hitter for a metric: the biggest
    regressed value above the hitter's own RHP baseline, among pitches Elder
    throws at least 8% of the time. Returns (pitch, value, gap, usage)."""
    base = row['baseline'][metric] or 0
    cand = [pt for pt in ELDER_PITCH_ORDER if row['usage'][pt] >= 0.08]
    pt = max(cand, key=lambda p: row['per_pitch'][p]['reg'][metric] - base)
    val = row['per_pitch'][pt]['reg'][metric]
    return pt, val, val - base, row['usage'][pt]


def matchup_read(row):
    """Specific scouting takeaway: name the pitch Elder can exploit."""
    b = row['baseline']
    last = row['hitter'].split(',')[0]
    wp, wval, wgap, wuse = weapon(row, 'whiff')
    if wgap >= 0.05:
        out = (f"Elder's {wp} is the swing-and-miss lever vs {last}: "
               f"{pct(wval)} projected whiff, {signed(wgap)} pts over his RHP "
               f"norm, and he throws it {wuse * 100:.0f}% to {row['hand']}HB.")
    else:
        out = (f"No pronounced swing-and-miss hole for Elder to target vs "
               f"{last}; the matchup tracks his overall profile.")
    cp, cval, cgap, cuse = weapon(row, 'chase')
    if (b['chase'] or 0) >= 0.34:
        out += (f" Already an aggressive chaser ({pct(b['chase'])} vs RHP), "
                f"so Elder can work off the plate.")
    elif cgap >= 0.05:
        out += f" Also expands the zone on the {cp} ({signed(cgap)} pts chase)."
    return out


def write_report(rows, cents, usage, h, kvals, reliab, latest_date):
    txt = os.path.join(OUT_DIR, 'bryce_elder_matchup_report.txt')
    csvp = os.path.join(OUT_DIR, 'bryce_elder_matchup.csv')

    # Order: most exploitable first, by the gap on Elder's best whiff weapon.
    ordered = sorted(rows, reverse=True, key=lambda r: weapon(r, 'whiff')[2])
    top_pitch, top_count = Counter(
        weapon(r, 'whiff')[0] for r in rows).most_common(1)[0]

    with open(txt, 'w', encoding='utf-8') as f:
        def emit(s=''):
            f.write(s + '\n')

        emit("WSH HITTERS vs BRYCE ELDER (ATL, RHP): ARSENAL MATCHUP REPORT")
        emit(f"Pitch-similarity scouting | data through {latest_date}")
        emit("=" * 78)
        emit()
        emit("METHOD: every pitch a hitter has faced vs RHP is soft-matched to")
        emit("each of Elder's pitches by physical similarity (velocity, IVB, HB,")
        emit("release geometry, extension). Pitch-type labels are ignored; a")
        emit("changeup that moves like Elder's sinker still counts as a match.")
        emit("Per-pitch rates are regressed toward the hitter's RHP baseline.")
        emit("The (Δ) columns are vs that hitter's own RHP norm.")
        emit()
        emit("ELDER'S ARSENAL (your retagged tags, season to date):")
        for pt in ELDER_PITCH_ORDER:
            c = cents[pt]
            r = c['raw']
            emit(f"  {pt}: n={c['n']:>3}  {r['velo']:.1f} mph  "
                 f"IVB {r['ivb']:+.1f}  HB {r['hb']:+.1f}  "
                 f"spin {r['spin']:.0f}  arm {r['arm']:.1f}deg  ext {r['ext']:.2f}")
        emit("  Usage vs LHH:  " + "  ".join(
            f"{pt} {usage['L'][pt] * 100:.0f}%" for pt in ELDER_PITCH_ORDER))
        emit("  Usage vs RHH:  " + "  ".join(
            f"{pt} {usage['R'][pt] * 100:.0f}%" for pt in ELDER_PITCH_ORDER))
        emit()
        emit("TUNING (bandwidth tuned on pitch-specific-deviation reliability):")
        emit(f"  kernel bandwidth h = {h}")
        for m in PROJ_METRICS + ['xwobacon']:
            rr = reliab[m]['dev']
            rs = f"{rr:.2f}" if rr is not None else "--"
            emit(f"  {m:<10} pitch-specific reliability {rs:>5}   "
                 f"regression k = {kvals[m]:.0f}")
        emit("  Whiff carries the strongest pitch-specific signal. Zone-contact,")
        emit("  CSW and xwOBAcon are weak signals: heavily regressed, so they")
        emit("  read near-flat across pitches by design.")
        emit()
        emit("=" * 78)
        emit(f"HEADLINE: Elder's {top_pitch} is the top swing-and-miss weapon for "
             f"{top_count} of {len(rows)} hitters")
        emit("here. Aggregate matchups sit near-neutral because his balanced mix")
        emit("averages out, so the real edge is pitch-specific (per-pitch rows).")
        emit("=" * 78)
        emit()
        emit("SUMMARY (sorted by Elder's best whiff weapon vs each hitter):")
        emit(f"  {'Hitter':<18}{'B':>2}  {'Proj whiff':>12}  "
             f"{'Top whiff weapon':<24}{'xwOBAcon':>9}")
        for r in ordered:
            wp, wval, wgap, wuse = weapon(r, 'whiff')
            aw = r['agg']['whiff']
            ab = r['baseline']['whiff'] or 0
            projw = f"{pct(aw)} ({signed(aw - ab)})"
            wpn = f"{wp} {pct(wval)} Δ{signed(wgap)} u{wuse * 100:.0f}%"
            emit(f"  {r['hitter']:<18}{r['hand']:>2}  {projw:>12}  "
                 f"{wpn:<24}{woba(r['xwobacon_reg']):>9}")
        emit()

        for r in ordered:
            b = r['baseline']
            a = r['agg']
            emit("=" * 78)
            emit(f"{r['hitter'].upper()}  ({r['hand']}HB vs RHP)")
            emit(f"  Baseline vs RHP  : whiff {pct(b['whiff'])}  "
                 f"chase {pct(b['chase'])}  Zcon {pct(b['zcontact'])}  "
                 f"CSW {pct(b['csw'])}  xwOBAcon {woba(b['xwobacon'])}   "
                 f"[{b['n_eff']:.0f} pitches]")

            def ad(m):
                return signed(a[m] - (b[m] or 0))
            emit(f"  Projected v Elder: whiff {pct(a['whiff'])}({ad('whiff')})  "
                 f"chase {pct(a['chase'])}({ad('chase')})  "
                 f"Zcon {pct(a['zcontact'])}({ad('zcontact')})  "
                 f"CSW {pct(a['csw'])}({ad('csw')})")
            emit(f"  Per pitch (regressed; Δ vs this hitter's RHP baseline; "
                 f"usage vs {r['hand']}HH):")
            emit(f"    {'Pitch':<6}{'Use':>5}{'Neff':>6}  {'Whiff':>13}"
                 f"{'Chase':>13}{'Zcon':>13}{'CSW':>13}")
            for pt in ELDER_PITCH_ORDER:
                pp = r['per_pitch'][pt]
                reg = pp['reg']

                def mc(m):
                    return f"{pct(reg[m])}(Δ{signed(reg[m] - (b[m] or 0))})"
                emit(f"    {pt:<6}{r['usage'][pt] * 100:>4.0f}%"
                     f"{pp['obs']['n_eff']:>6.0f}  "
                     f"{mc('whiff'):>13}{mc('chase'):>13}"
                     f"{mc('zcontact'):>13}{mc('csw'):>13}")
            if r['actual'] and r['actual_n'] > 0:
                ac = r['actual']
                emit(f"  Actual vs Elder ({r['actual_n']} pitches, tiny sample): "
                     f"whiff {pct(ac['whiff'])}  chase {pct(ac['chase'])}  "
                     f"CSW {pct(ac['csw'])}")
            else:
                emit("  Actual vs Elder  : no MLB plate appearances on record")
            emit(f"  READ: {matchup_read(r)}")
        emit("=" * 78)
        emit()
        emit("CAVEATS: a scouting lens, not a precise projection. Per-pitch rates")
        emit("are regressed, so magnitudes are conservative; trust direction and")
        emit("rank over the exact number. A positive whiff delta partly reflects")
        emit("that the pitch type misses bats league-wide, not only a hitter-")
        emit("specific hole. Crews is ~93% AAA (lower competition, release-point")
        emit("geometry fallback). Ruiz and Millas have thin vs-RHP samples.")

    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        cw = csv.writer(f)
        cw.writerow(['hitter', 'bats', 'row', 'usage_pct', 'n_eff',
                     'whiff', 'whiff_vs_base', 'chase', 'chase_vs_base',
                     'zcontact', 'csw', 'xwobacon'])

        def num(x, scale=100, d=1):
            return round(x * scale, d) if x is not None else ''

        for r in ordered:
            b = r['baseline']
            for pt in ELDER_PITCH_ORDER:
                reg = r['per_pitch'][pt]['reg']
                cw.writerow([r['hitter'], r['hand'], pt,
                             round(r['usage'][pt] * 100, 1),
                             round(r['per_pitch'][pt]['obs']['n_eff'], 1),
                             num(reg['whiff']),
                             num(reg['whiff'] - (b['whiff'] or 0)),
                             num(reg['chase']),
                             num(reg['chase'] - (b['chase'] or 0)),
                             num(reg['zcontact']), num(reg['csw']), ''])
            a = r['agg']
            cw.writerow([r['hitter'], r['hand'], 'PROJ_vs_Elder', '',
                         round(r['proj_neff'], 1),
                         num(a['whiff']), num(a['whiff'] - (b['whiff'] or 0)),
                         num(a['chase']), num(a['chase'] - (b['chase'] or 0)),
                         num(a['zcontact']), num(a['csw']),
                         num(r['xwobacon_reg'], 1, 3)])
            cw.writerow([r['hitter'], r['hand'], 'Baseline_vs_RHP', '',
                         round(b['n_eff'], 1),
                         num(b['whiff']), 0, num(b['chase']), 0,
                         num(b['zcontact']), num(b['csw']),
                         num(b['xwobacon'], 1, 3)])
            ac = r['actual']
            cw.writerow([r['hitter'], r['hand'], 'Actual_vs_Elder', '',
                         r['actual_n'],
                         num(ac['whiff']) if ac else '', '',
                         num(ac['chase']) if ac else '', '',
                         num(ac['zcontact']) if ac else '',
                         num(ac['csw']) if ac else '', ''])
    return txt, csvp


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading pitch cache…")
    with open(PICKLE_PATH, 'rb') as fh:
        pitches = pickle.load(fh)
    print(f"  {len(pitches)} pitches")

    rhp = [p for p in pitches if p.get('Throws') == 'R']
    norms = compute_norms(rhp)
    elder = [p for p in pitches if p.get('Pitcher') == TARGET_PITCHER]
    if not elder:
        raise SystemExit(f"No pitches found for {TARGET_PITCHER}")
    cents = elder_centroids(elder, norms)
    cents['_elder_pitches'] = elder

    # Score every usable RHP pitch against the 5 Elder centroids once.
    usable = 0
    for p in rhp:
        z = get_z(p, norms)
        if z is None:
            p['_z'] = None
            continue
        p['_z'] = z
        p['_d2'] = {pt: dist2(cents[pt]['z'], z) for pt in ELDER_PITCH_ORDER}
        usable += 1
    print(f"  {usable} RHP pitches usable for matching")

    latest_date = max((p.get('Game Date') for p in elder if p.get('Game Date')),
                      default='?')

    # Comparison pools: pitches faced vs RHP, excluding Elder's own pitches.
    # Crews combines MLB + AAA; the other 10 are MLB only (per instruction).
    def comp_pool(batter):
        crews = batter == 'Crews, Dylan'
        out = []
        for p in rhp:
            if p.get('Batter') != batter:
                continue
            if p.get('Pitcher') == TARGET_PITCHER or p.get('_z') is None:
                continue
            if not crews and p.get('_source') != 'MLB':
                continue
            out.append(p)
        return out

    target_pools = {h: comp_pool(h) for h in TARGET_HITTERS}
    actuals = {h: [p for p in elder if p.get('Batter') == h]
               for h in TARGET_HITTERS}

    # League pool (MLB RHP, Elder's own pitches excluded) for baselines.
    league = [p for p in rhp if p.get('_source') == 'MLB'
              and p.get('_z') is not None
              and p.get('Pitcher') != TARGET_PITCHER]

    # Qualified reliability pool: MLB hitters with enough vs-RHP pitches.
    by_batter = defaultdict(list)
    for p in league:
        by_batter[p['Batter']].append(p)
    rng = random.Random(42)
    qualified = {}
    for batter, ps in by_batter.items():
        if len(ps) >= QUALIFIED_MIN_PITCHES:
            shuffled = ps[:]
            rng.shuffle(shuffled)
            qualified[batter] = shuffled

    print("\n" + "=" * 70)
    print("DIAGNOSTIC PHASE")
    print("=" * 70)
    print(f"Elder: {len(elder)} pitches, arsenal {[cents[pt]['n'] for pt in ELDER_PITCH_ORDER]}")
    print(f"Reliability pool: {len(qualified)} MLB hitters "
          f"(>={QUALIFIED_MIN_PITCHES} vs-RHP pitches)")

    diag_redundancy([p for p in rhp if p.get('_source') == 'MLB'])

    # Bandwidth sweep. Raw reliability is the WRONG tuning target: a wide kernel
    # makes every Elder pitch collapse to the hitter's global average (Neff%
    # near 100), trivially reproducible but with no pitch specificity. We tune
    # on whiff_dev -- the reliability of the pitch-SPECIFIC deviation -- which
    # peaks at an interior bandwidth (blur kills signal, noise kills the rest).
    print("\n── Bandwidth sweep: raw vs pitch-specific-deviation reliability ──")
    print(f"  {'h':>6}{'whiffR':>9}{'whiffDev':>10}{'chaseDev':>10}"
          f"{'zconDev':>9}{'cswDev':>9}{'Neff%':>8}")
    sweep, disc = {}, {}
    for h in BANDWIDTHS:
        rel = diag_reliability(qualified, h)
        sweep[h] = rel
        disc[h] = diag_discrimination(qualified, h)
        vals = []
        for m, key in [('whiff', 'r'), ('whiff', 'dev'), ('chase', 'dev'),
                       ('zcontact', 'dev'), ('csw', 'dev')]:
            v = rel[m][key]
            vals.append(f"{v:.2f}" if v is not None else "--")
        print(f"  {h:>6.2f}{vals[0]:>9}{vals[1]:>10}{vals[2]:>10}"
              f"{vals[3]:>9}{vals[4]:>9}{disc[h] * 100:>7.0f}%")

    # whiff_dev plateaus across a wide h range, so its argmax is noise. Take
    # the NARROWEST (most discriminating) bandwidth that still retains >=95% of
    # the peak deviation-reliability -- the elbow, where the kernel is as sharp
    # as possible before the pitch-specific signal starts degrading into noise.
    peak_dev = max((sweep[h]['whiff']['dev'] or 0) for h in BANDWIDTHS)
    eligible = [h for h in BANDWIDTHS
                if (sweep[h]['whiff']['r'] or 0) >= 0.55
                and (sweep[h]['whiff']['dev'] or 0) >= 0.95 * peak_dev]
    if not eligible:
        eligible = list(BANDWIDTHS)
    best_h = min(eligible)
    print(f"  -> h = {best_h}: narrowest bandwidth within 95% of peak whiff "
          f"deviation-reliability ({sweep[best_h]['whiff']['dev']:.2f} vs peak "
          f"{peak_dev:.2f}); per-pitch N_eff {disc[best_h] * 100:.0f}% of pool")

    diag_coverage(target_pools, cents, best_h)
    diag_count_confound(cents, league, best_h)

    # Typical effective sample at chosen h -> per-metric regression constant k.
    neffs = []
    for ps in qualified.values():
        for pt in ELDER_PITCH_ORDER:
            neffs.append(wmetrics(ps, make_wf(pt, best_h))['n_eff'])
    m_neff = median(neffs) or 100.0
    reliab = sweep[best_h]
    # k from pitch-specific (deviation) reliability: the per-pitch estimate is
    # what gets regressed, so weak-signal metrics (csw, zcontact) regress hard.
    kvals = {m: derive_k(reliab[m]['dev'], m_neff)
             for m in PROJ_METRICS + ['xwobacon']}
    print(f"\n── Regression constants (from pitch-specific reliability; "
          f"median N_eff = {m_neff:.0f}) ──")
    for m in PROJ_METRICS + ['xwobacon']:
        rr = reliab[m]['dev']
        rs = f"{rr:.2f}" if rr is not None else "--"
        print(f"  {m:<10} dev-reliability {rs:>6}   k = {kvals[m]:.0f}")

    # ── Report phase ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILDING REPORT")
    print("=" * 70)
    league_raw = wmetrics(league, RAW_WF)
    league_pt = {'_league': league}
    for pt in ELDER_PITCH_ORDER:
        league_pt[pt] = wmetrics(league, make_wf(pt, best_h))
    usage = elder_usage(elder)

    rows = []
    for h in TARGET_HITTERS:
        pool = target_pools[h]
        if not pool:
            print(f"  WARNING: no comparison pitches for {h}")
            continue
        rows.append(build_hitter_row(h, pool, actuals[h], cents, league_pt,
                                     league_raw, usage, best_h, kvals))

    txt, csvp = write_report(rows, cents, usage, best_h, kvals, reliab,
                             latest_date)
    print(f"  wrote {txt}")
    print(f"  wrote {csvp}")
    print("\nDone.")


if __name__ == '__main__':
    main()
