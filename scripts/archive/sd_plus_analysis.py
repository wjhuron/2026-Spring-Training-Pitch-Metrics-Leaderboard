#!/usr/bin/env python3
"""SD+ (Swing Decisions+) validation script.

Builds the full SD+ pipeline off the cached pitch-level dataset
(data/all_pitches_rs_cache.pkl) and prints it for eyeball review.

Loads raw pitches → filters eligible decisions → classifies zones →
builds per-(zone, count, decision) RV weight table with continuous
Bayesian shrinkage → computes per-hitter dv → regresses and normalizes
→ prints weights, distributions, top/bottom hitters, and comparison
vs legacy pdPlus.

No production pipeline is touched. After eyeball approval, SD+ moves
into pipeline_compute.py / process_data.py.
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

# Make pipeline_utils importable when running from scripts/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pipeline_utils import (
    safe_float, SWING_DESCRIPTIONS, BB_EVENTS, BUNT_BB_TYPES,
    MLB_TEAMS, AAA_TEAMS, ZONE_HALF_WIDTH,
)

DATA_DIR = os.path.join(_ROOT, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'all_pitches_rs_cache.pkl')
HITTER_LB_PATH = os.path.join(DATA_DIR, 'hitter_leaderboard_rs.json')

# FanGraphs Guts constants for xRV computation (fetched fresh at runtime).
_GUTS = None
def _get_guts():
    global _GUTS
    if _GUTS is None:
        from pipeline_fetch import fetch_guts_constants
        _, _, extras = fetch_guts_constants()
        _GUTS = extras
    return _GUTS

# ── Zone classification thresholds (Baseball Savant attack zones) ────────
# All in feet. Rulebook plate: 17" wide → ±8.5"/12 ft horizontal edges.
# Vertical edges are hitter-specific (SzTop/SzBot from the pipeline, which
# already include the ABS adjustment per the user's note).
HEART_X   = 6.7 / 12     # |PlateX| ≤ 6.7" = inner 79% of plate width
SHADOW_X  = 13.3 / 12    # |PlateX| ≤ 13.3" = edge of shadow ring
CHASE_X   = 20.0 / 12    # |PlateX| ≤ 20" = edge of chase ring
# Vertical thresholds as fractions of the hitter's strike zone height.
# Per the Savant attack-zone diagram: Heart is the inner ±33% of zone
# half-height (middle 33% of zone, 8" tall in a 24" zone), Shadow outer
# is ±133% of zone half-height, Chase outer is ±200%.
HEART_VERT_FRAC  = 1.0 / 3.0   # Heart covers middle 33% of zone height
SHADOW_VERT_FRAC = 1.0 / 6.0   # Shadow extends 16.7% of zone_ht above/below = 33% of half-ht
CHASE_VERT_FRAC  = 0.5         # Chase extends 50% of zone_ht above/below = 100% of half-ht

# ── Decision sets ────────────────────────────────────────────────────────
TAKE_DESCRIPTIONS = {'Called Strike', 'Ball'}
# (SWING_DESCRIPTIONS is imported from pipeline_utils: {'Swinging Strike', 'Foul', 'In Play'}
#  — already covers Foul Tip, which the scraper folds into 'Swinging Strike'.)

# ── Hyperparameters ──────────────────────────────────────────────────────
CELL_SHRINK_K  = 50     # pseudo-count for cell → zone shrinkage
HITTER_PRIOR_N = 400    # pseudo-count for hitter → league regression
MIN_HITTER_DECISIONS = 100  # floor for appearing in the distribution
PA_PER_TEAM_GAME = 3.1      # qualification standard

# ═══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def classify_zone(p):
    """Return one of {'heart','shadow_in','shadow_out','chase','waste'} or None.

    Uses the pitch's hitter-specific SzTop/SzBot (ABS-adjusted in the pipeline).
    Shadow is split into in-zone vs out-of-zone using the pipeline's InZone,
    which is ball-radius-adjusted per compute_in_zone().
    """
    px = safe_float(p.get('PlateX'))
    pz = safe_float(p.get('PlateZ'))
    top = safe_float(p.get('SzTop'))
    bot = safe_float(p.get('SzBot'))
    if any(v is None for v in (px, pz, top, bot)):
        return None
    if top <= bot:
        return None

    sz_ht = top - bot
    ax = abs(px)

    z_heart_low  = bot + HEART_VERT_FRAC * sz_ht
    z_heart_high = top - HEART_VERT_FRAC * sz_ht
    z_shadow_low  = bot - SHADOW_VERT_FRAC * sz_ht
    z_shadow_high = top + SHADOW_VERT_FRAC * sz_ht
    z_chase_low   = bot - CHASE_VERT_FRAC * sz_ht
    z_chase_high  = top + CHASE_VERT_FRAC * sz_ht

    # Heart: inside zone, ≥ threshold from each edge.
    if ax <= HEART_X and z_heart_low <= pz <= z_heart_high:
        return 'heart'
    # Shadow ring: in the 6.7" band straddling each edge.
    if ax <= SHADOW_X and z_shadow_low <= pz <= z_shadow_high:
        return 'shadow_in' if p.get('InZone') == 'Yes' else 'shadow_out'
    # Chase ring.
    if ax <= CHASE_X and z_chase_low <= pz <= z_chase_high:
        return 'chase'
    return 'waste'


def classify_decision(p):
    desc = p.get('Description')
    if desc in SWING_DESCRIPTIONS:
        return 'swing'
    if desc in TAKE_DESCRIPTIONS:
        return 'take'
    return None


def get_count(p):
    """Parse the 'Count' column (e.g. '2-2') into (balls, strikes) ints.
    Returns None if missing or malformed."""
    c = p.get('Count')
    if not isinstance(c, str) or '-' not in c:
        return None
    try:
        b_str, s_str = c.split('-', 1)
        b, s = int(b_str), int(s_str)
    except (TypeError, ValueError):
        return None
    if not (0 <= b <= 3 and 0 <= s <= 2):
        return None
    return (b, s)


def is_eligible(p):
    """Filter to pitches where a genuine swing/take decision occurred."""
    if p.get('_source') != 'MLB':
        return False
    if p.get('Event') == 'Intent Walk':
        return False
    desc = p.get('Description') or ''
    if 'bunt' in desc.lower():
        return False
    if 'pitchout' in desc.lower():
        return False
    if desc == 'Hit By Pitch':
        return False
    if p.get('BBType') in BUNT_BB_TYPES:
        return False
    if p.get('RunExp') is None or safe_float(p.get('RunExp')) is None:
        return False
    if classify_decision(p) is None:
        return False
    if classify_zone(p) is None:
        return False
    if get_count(p) is None:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
#  WEIGHT TABLE
# ═══════════════════════════════════════════════════════════════════════

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
COUNTS = [(b, s) for b in range(4) for s in range(3)]  # 12 counts


def rv_hitter_runexp(p):
    """Raw hitter-perspective RV: -RunExp. Includes BIP outcomes (luck + execution)."""
    return -safe_float(p['RunExp'])


def rv_hitter_xrv(p):
    """Luck-neutral hitter-perspective RV.
    For BIP with xwOBA: (xwOBA - lgWOBA) / wOBAScale (expected run value given EV/LA).
    For all other pitches: -RunExp (same as raw).
    """
    if p.get('Description') == 'In Play':
        xw = safe_float(p.get('xwOBA'))
        if xw is not None:
            g = _get_guts()
            return (xw - g['lgWOBA']) / g['wOBAScale']
    return -safe_float(p['RunExp'])


def build_weight_table(pitches, rv_fn=rv_hitter_runexp):
    """Return dict[(zone, count, decision)] -> (mean_rv, n). rv_fn controls
    whether BIP outcomes enter raw (rv_hitter_runexp) or luck-neutral
    (rv_hitter_xrv)."""
    cells = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    for p in pitches:
        zone = classify_zone(p)
        decision = classify_decision(p)
        count = get_count(p)
        rv = rv_fn(p)
        if rv is None:
            continue
        cells[(zone, count, decision)]['sum'] += rv
        cells[(zone, count, decision)]['n'] += 1

    out = {}
    for key, agg in cells.items():
        out[key] = (agg['sum'] / agg['n'], agg['n'])
    return out


def zone_level_means(pitches, rv_fn=rv_hitter_runexp):
    """Decision-specific means per zone (collapses across counts)."""
    zone_sum  = defaultdict(float)
    zone_n    = defaultdict(int)
    for p in pitches:
        zone = classify_zone(p)
        decision = classify_decision(p)
        rv = rv_fn(p)
        if rv is None:
            continue
        zone_sum[(zone, decision)] += rv
        zone_n[(zone, decision)] += 1
    return {k: (zone_sum[k] / zone_n[k], zone_n[k]) for k in zone_sum}


def shrink_table(raw_table, zone_means, k=CELL_SHRINK_K):
    """Continuous Bayesian shrinkage of each cell toward its
    decision-specific zone mean.

        smoothed = (n × cell_mean + k × zone_mean) / (n + k)
    """
    smoothed = {}
    for zone in ZONES:
        for count in COUNTS:
            for decision in ('swing', 'take'):
                cell_key = (zone, count, decision)
                if cell_key in raw_table:
                    cell_mean, n = raw_table[cell_key]
                else:
                    cell_mean, n = 0.0, 0
                zone_mean, _ = zone_means.get((zone, decision), (0.0, 0))
                rv = (n * cell_mean + k * zone_mean) / (n + k)
                smoothed[cell_key] = (rv, n)
    return smoothed


# ═══════════════════════════════════════════════════════════════════════
#  PER-HITTER SD+
# ═══════════════════════════════════════════════════════════════════════

def compute_dv_A(p, table):
    """dv_A: RV(chosen) - RV(opposite). Absolute opportunity cost."""
    zone = classify_zone(p)
    decision = classify_decision(p)
    count = get_count(p)
    swing_rv, _ = table[(zone, count, 'swing')]
    take_rv,  _ = table[(zone, count, 'take')]
    if decision == 'swing':
        return swing_rv - take_rv
    else:
        return take_rv - swing_rv


def compute_dv_C(p, table):
    """dv_C: RV(chosen) - volume-weighted league average in the cell.
    A league-average-behavior hitter has mean dv_C = 0 exactly."""
    zone = classify_zone(p)
    decision = classify_decision(p)
    count = get_count(p)
    swing_rv, n_sw = table[(zone, count, 'swing')]
    take_rv,  n_tk = table[(zone, count, 'take')]
    total = n_sw + n_tk
    if total == 0:
        return 0.0
    league_avg_rv = (n_sw * swing_rv + n_tk * take_rv) / total
    if decision == 'swing':
        return swing_rv - league_avg_rv
    else:
        return take_rv - league_avg_rv


compute_dv = compute_dv_A  # backwards-compat for older code paths


def compute_hitter_sd(pitches_by_hitter, table, dv_fn=compute_dv_A):
    """Return dict[(hitter, team)] → {'raw_sd', 'n_decisions', 'zone_dv'}."""
    results = {}
    for key, pitches in pitches_by_hitter.items():
        dvs = [dv_fn(p, table) for p in pitches]
        if not dvs:
            continue
        zone_dvs = defaultdict(list)
        for p, dv in zip(pitches, dvs):
            zone_dvs[classify_zone(p)].append(dv)
        results[key] = {
            'raw_sd': sum(dvs) / len(dvs),
            'n_decisions': len(dvs),
            'zone_dv': {z: (sum(vs)/len(vs) if vs else None) for z, vs in zone_dvs.items()},
        }
    return results


def regress_and_normalize(hitter_raw, n_prior=HITTER_PRIOR_N,
                          min_n=MIN_HITTER_DECISIONS, scale_k=100):
    """Apply Bayesian regression toward league mean, then SD-scale with
    sdPlus = 100 + scale_k × z. scale_k controls the spread: 100 gives
    SD=100 across hitters (IQ-style); 30 matches wRC+ visual convention."""
    eligible = {k: v for k, v in hitter_raw.items() if v['n_decisions'] >= min_n}
    if not eligible:
        return {}

    lg_raw = sum(v['raw_sd'] for v in eligible.values()) / len(eligible)
    for k, v in eligible.items():
        n = v['n_decisions']
        v['raw_sd_adj'] = (n * v['raw_sd'] + n_prior * lg_raw) / (n + n_prior)

    adj_vals = [v['raw_sd_adj'] for v in eligible.values()]
    lg_mean = sum(adj_vals) / len(adj_vals)
    lg_sd = math.sqrt(sum((x - lg_mean) ** 2 for x in adj_vals) / len(adj_vals))

    for v in eligible.values():
        if lg_sd > 0:
            z = (v['raw_sd_adj'] - lg_mean) / lg_sd
            v['z']      = z
            v['sdPlus'] = round(100 + scale_k * z, 1)
        else:
            v['z']      = 0.0
            v['sdPlus'] = 100.0
        v['lg_mean'] = lg_mean
        v['lg_sd'] = lg_sd
    return eligible


# ═══════════════════════════════════════════════════════════════════════
#  PRINTING
# ═══════════════════════════════════════════════════════════════════════

def hr(title=None, char='═'):
    print()
    if title:
        print(char * 78)
        print(f"  {title}")
        print(char * 78)
    else:
        print(char * 78)


def print_filter_summary(all_pitches, eligible_pitches, zone_counts, decision_counts):
    hr("SECTION 1 — FILTER SUMMARY")
    print("""
What you're looking at: how many pitches survived each filter on the way to
being an "eligible decision" for the SD+ weight table. Key things to check:
  • Eligible-rate should be ~70-85% of all pitches. Too low → a filter is
    eating into real decisions. Too high → a filter isn't firing.
  • Zone distribution: Heart is usually 20-25%, Shadow 30-40%, Chase 25-30%,
    Waste 5-10%. Major deviations suggest a zone-classifier bug.
  • Swing rate should be 45-50% overall across MLB.
""".rstrip())
    n_all = len(all_pitches)
    n_elig = len(eligible_pitches)
    print(f"  Total cached pitches:   {n_all:>9,}")
    print(f"  Eligible decisions:     {n_elig:>9,} ({100*n_elig/n_all:.1f}%)")
    print(f"  Dropped:                {n_all - n_elig:>9,}")
    print()
    print("  Zone distribution of eligible pitches:")
    total = sum(zone_counts.values())
    for zone in ZONES:
        n = zone_counts.get(zone, 0)
        print(f"    {zone:<12} {n:>8,}  ({100*n/total:.1f}%)")
    print()
    print("  Swing vs take in eligible pitches:")
    for decision in ('swing', 'take'):
        n = decision_counts.get(decision, 0)
        print(f"    {decision:<12} {n:>8,}  ({100*n/total:.1f}%)")


def print_weight_table(raw_table, smoothed, zone_means):
    hr("SECTION 2 — RV WEIGHT TABLE (60 cells)")
    print("""
What you're looking at: the heart of SD+. For every (zone × count) cell, we
show the mean hitter-perspective RV for swings and takes separately, plus
the gap between them (swing_rv − take_rv). Each dv for a hitter's pitch is
this gap, signed by what they did.

Three columns per swing/take: RAW mean (unsmoothed), SMOOTHED mean (after
continuous shrinkage toward the zone-level mean with k=50 pseudo-obs), and
n (cell count). The closer SMOOTHED is to RAW, the more data we had; a big
gap means the cell was thin and got pulled toward the zone average.

Red flags to scan for:
  • Heart swing−take should be strongly POSITIVE (swing beats take) at every
    count. Especially big at 0-0 and 2-0 (early-count meatballs). If a heart
    count shows negative, something is broken.
  • Waste swing−take should be strongly NEGATIVE at every count, most
    extreme at 3-0/3-1/3-2 (you're throwing away a walk).
  • Shadow_out take should be POSITIVE (takes gain RV when pitch is a ball).
  • Shadow_in take should be NEGATIVE and growing with strike count (called
    strike → deeper count → K risk).
  • Chase swing should be NEGATIVE but less bad in 2-strike counts (since
    swinging on 0-2 at a chase pitch is closer to neutral than at 0-0).
  • Tiny n (< 20) on many cells → smoothing is doing real work; inspect what
    got pulled.
""".rstrip())

    print("\n  Zone-level means (the shrinkage priors):")
    print(f"    {'zone':<12} {'swing_rv':>10} {'take_rv':>10} {'gap':>8} {'n_swing':>10} {'n_take':>10}")
    for zone in ZONES:
        sw, n_sw = zone_means.get((zone, 'swing'), (None, 0))
        tk, n_tk = zone_means.get((zone, 'take'), (None, 0))
        gap = (sw - tk) if (sw is not None and tk is not None) else None
        print(f"    {zone:<12} {_fmt(sw):>10} {_fmt(tk):>10} {_fmt(gap):>8} {n_sw:>10,} {n_tk:>10,}")

    for zone in ZONES:
        print(f"\n  ── Cells for zone = {zone} ──")
        print(f"    {'count':<6}  {'raw_sw':>8} {'sm_sw':>8} {'n_sw':>7}  "
              f"{'raw_tk':>8} {'sm_tk':>8} {'n_tk':>7}  {'sm_gap':>8}")
        for count in COUNTS:
            rs = raw_table.get((zone, count, 'swing'))
            rt = raw_table.get((zone, count, 'take'))
            ss = smoothed[(zone, count, 'swing')]
            st = smoothed[(zone, count, 'take')]
            raw_sw = rs[0] if rs else None
            n_sw = rs[1] if rs else 0
            raw_tk = rt[0] if rt else None
            n_tk = rt[1] if rt else 0
            sm_sw = ss[0]
            sm_tk = st[0]
            gap = sm_sw - sm_tk
            print(f"    {count[0]}-{count[1]}   "
                  f"{_fmt(raw_sw):>8} {_fmt(sm_sw):>8} {n_sw:>7,}  "
                  f"{_fmt(raw_tk):>8} {_fmt(sm_tk):>8} {n_tk:>7,}  "
                  f"{_fmt(gap):>8}")


def _fmt(x):
    if x is None:
        return '   —   '
    return f'{x:+.4f}'


def print_distribution(hitter_raw, scales=(100, 40, 30, 25, 20)):
    hr("SECTION 3 — SD+ DISTRIBUTION ACROSS SCALING OPTIONS")
    print("""
What you're looking at: the same underlying z-scores, rescaled to sdPlus =
100 + k × z for several choices of k. Z-scores and rankings are invariant
under k — only the numeric spread changes. Pick the k whose spread feels
right compared to wRC+ (typical qualified range 50-180, SD ~25-30).

Rows: min / p05 / p25 / p50 / p75 / p95 / max.

Legend to calibrate:
  • If your target is wRC+ visual similarity, pick the k where p05 lands
    around 55-65 and p95 around 140-150.
  • If you want extremes more visible (fewer hitters clipped near 100),
    pick a larger k but accept outliers going past 200 or below 0.
  • k = 100 is the "1 SD = 100 points" convention (IQ-style); mathematically
    clean but visually too wide for season-long leaderboards.
""".rstrip())

    # Compute all scales in parallel from the same z-scores.
    per_scale = {}
    for k in scales:
        norm_k = regress_and_normalize(
            {kk: dict(vv) for kk, vv in hitter_raw.items()},
            scale_k=k,
        )
        vals = sorted(v['sdPlus'] for v in norm_k.values())
        per_scale[k] = (norm_k, vals)

    def pct(vals, p):
        n = len(vals)
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return vals[i]

    print(f"\n  n qualifying: {len(list(per_scale.values())[0][1])}")
    print()
    header = f"  {'scale k':>8}  {'min':>7} {'p05':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'max':>7}  {'SD':>6}"
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for k in scales:
        vals = per_scale[k][1]
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        print(f"  {k:>8}  "
              f"{vals[0]:>7.1f} {pct(vals,0.05):>7.1f} {pct(vals,0.25):>7.1f} "
              f"{pct(vals,0.50):>7.1f} {pct(vals,0.75):>7.1f} {pct(vals,0.95):>7.1f} "
              f"{vals[-1]:>7.1f}  {sd:>6.2f}")

    return per_scale


def print_top_bottom(normalized, legacy_lookup):
    hr("SECTION 4 — TOP AND BOTTOM HITTERS")
    print("""
What you're looking at: top/bottom 20 hitters by SD+, alongside their
legacy pdPlus for side-by-side comparison. Use this to face-check the metric:

  • Top 20 SHOULD include elite plate-discipline hitters: Soto, Tucker,
    contact-patient types (Arraez), disciplined power (Harper, Freeman).
  • Bottom 20 SHOULD include free-swingers with elevated K% or chase:
    historical examples are Gallo, Sánchez, Adames (chasey). Heavy
    free-swingers who are still productive (Vlad Jr) might show low SD+
    despite good offensive output — that's correct, SD+ is decision-only.
  • Where SD+ and pdPlus DISAGREE most is the most interesting signal.
    A hitter who drops hard in SD+ vs legacy is probably someone whose
    IZSw/Chase/Contact rates looked OK but whose count-aware decisions
    were worse than averaged rates suggested.

The "Δ vs pdPlus" column highlights disagreements. Big positive = SD+
rewards them more than legacy did; big negative = vice versa.
""".rstrip())

    def _row(key, v):
        hitter, team = key
        legacy = legacy_lookup.get((hitter, team))
        legacy_str = f"{legacy:.1f}" if legacy is not None else "  —  "
        delta = (v['sdPlus'] - legacy) if legacy is not None else None
        delta_str = f"{delta:+.1f}" if delta is not None else "  —  "
        print(f"  {hitter:<26} {team:<4}  "
              f"n={v['n_decisions']:>5}  "
              f"SD+={v['sdPlus']:>6.1f}  "
              f"pdPlus={legacy_str:>6}  "
              f"Δ={delta_str:>6}")

    sorted_hitters = sorted(normalized.items(), key=lambda kv: -kv[1]['sdPlus'])
    print("\n  TOP 20 by SD+:")
    for key, v in sorted_hitters[:20]:
        _row(key, v)

    print("\n  BOTTOM 20 by SD+:")
    for key, v in sorted_hitters[-20:]:
        _row(key, v)

    print("\n  BIGGEST DISAGREEMENTS (|SD+ − pdPlus|):")
    disagreements = []
    for key, v in normalized.items():
        legacy = legacy_lookup.get(key)
        if legacy is not None:
            disagreements.append((key, v, abs(v['sdPlus'] - legacy)))
    disagreements.sort(key=lambda x: -x[2])
    for key, v, _ in disagreements[:15]:
        _row(key, v)


def print_variant_comparison(variant_results, ref_lookup=None):
    """Compare sdPlus across formula variants: dv_A vs dv_C × RunExp vs xRV.

    Shows rank correlations, top/bottom-20 overlap, and the largest
    disagreements so the user can see which changes actually move the
    metric vs just relabeling it."""
    hr_title = "SECTION 6 — VARIANT COMPARISON (RunExp vs xRV, dv_A vs dv_C)"
    hr(hr_title)
    print("""
What you're looking at: four versions of SD+ computed from the same 60-cell
structure but with two independent toggles:
  • RV source: raw RunExp (includes BIP outcome/luck/execution) OR xRV
    (xwOBA-based for BIP, luck-neutral on the BIP outcome axis).
  • dv formula: dv_A = RV(chosen) - RV(opposite) OR dv_C = RV(chosen) -
    league_avg_RV_in_cell.

Reference: 'A_runexp (CURRENT)' is the v1 metric we've been validating.

For each variant vs the current, I show: pairwise Pearson r between
sdPlus values, number of top-20 hitters preserved (overlap), and the
number of hitters whose sdPlus shifts by more than 10 points.
""".rstrip())

    def _pearson(xs, ys):
        valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(valid) < 5:
            return None
        n = len(valid)
        mx = sum(x for x, _ in valid) / n
        my = sum(y for _, y in valid) / n
        num = sum((x - mx) * (y - my) for x, y in valid)
        denx = math.sqrt(sum((x - mx) ** 2 for x, _ in valid))
        deny = math.sqrt(sum((y - my) ** 2 for _, y in valid))
        if denx == 0 or deny == 0:
            return None
        return num / (denx * deny)

    names = list(variant_results.keys())
    ref_name = names[0]  # A_runexp is current baseline
    ref = variant_results[ref_name]
    keys = sorted(ref.keys())
    ref_vals = [ref[k]['sdPlus'] for k in keys]
    ref_top20 = set(k for k, v in
                    sorted(ref.items(), key=lambda kv: -kv[1]['sdPlus'])[:20])
    ref_bot20 = set(k for k, v in
                    sorted(ref.items(), key=lambda kv: kv[1]['sdPlus'])[:20])

    print(f"\n  Reference variant: {ref_name}")
    print(f"  Distribution across variants (n = {len(keys)} hitters, k=30):\n")
    print(f"  {'variant':<32}  {'min':>6} {'p05':>6} {'p50':>6} {'p95':>6} "
          f"{'max':>6}  {'r_vs_ref':>9}  {'top20 ∩':>8}  {'bot20 ∩':>8}  {'|Δ|>10':>7}")

    for name in names:
        norm = variant_results[name]
        vals_this = [norm[k]['sdPlus'] if k in norm else None for k in keys]
        valid_this = [v for v in vals_this if v is not None]
        if not valid_this:
            continue
        valid_this_sorted = sorted(valid_this)
        def _p(pct):
            n = len(valid_this_sorted)
            i = max(0, min(n - 1, int(round(pct * (n - 1)))))
            return valid_this_sorted[i]
        r = _pearson(ref_vals, vals_this)
        r_str = f"{r:+.3f}" if r is not None else "   —  "
        top20 = set(k for k, v in
                    sorted(norm.items(), key=lambda kv: -kv[1]['sdPlus'])[:20])
        bot20 = set(k for k, v in
                    sorted(norm.items(), key=lambda kv: kv[1]['sdPlus'])[:20])
        top_overlap = len(ref_top20 & top20)
        bot_overlap = len(ref_bot20 & bot20)
        diff10 = sum(1 for a, b in zip(ref_vals, vals_this)
                     if a is not None and b is not None and abs(a - b) > 10)
        print(f"  {name:<32}  "
              f"{valid_this_sorted[0]:>6.1f} {_p(0.05):>6.1f} {_p(0.50):>6.1f} "
              f"{_p(0.95):>6.1f} {valid_this_sorted[-1]:>6.1f}  "
              f"{r_str:>9}  {top_overlap:>3}/20   {bot_overlap:>3}/20   {diff10:>5}")

    # Show biggest movers vs reference for each alternative variant.
    for name in names[1:]:
        norm = variant_results[name]
        print(f"\n  ── Biggest movers: {name} vs {ref_name} ──")
        moves = []
        for k in keys:
            if k in norm and k in ref:
                d = norm[k]['sdPlus'] - ref[k]['sdPlus']
                moves.append((k, ref[k]['sdPlus'], norm[k]['sdPlus'], d))
        moves.sort(key=lambda t: -abs(t[3]))
        print(f"    {'hitter':<26} {'team':<4}  {'ref':>6}  {'alt':>6}  {'Δ':>6}")
        for k, ref_v, alt_v, d in moves[:15]:
            print(f"    {k[0]:<26} {k[1]:<4}  {ref_v:>6.1f}  {alt_v:>6.1f}  {d:>+6.1f}")


def print_correlation(normalized, legacy_lookup, hitter_ref_lookup):
    hr("SECTION 5 — CORRELATIONS")
    print("""
What you're looking at: Pearson r between SD+ and other metrics, computed
across qualified hitters. Expectations:

  • SD+ vs legacy pdPlus: 0.50–0.75. If r > 0.85, the new metric is
    essentially a re-expression of the old one — gains are cosmetic.
    If r < 0.35, they're measuring fundamentally different things, which
    deserves investigation (bug? or real?).
  • SD+ vs wRC+ / xwOBA: 0.25–0.45. Discipline is one real but modest
    component of offense; this is a sanity upper bound.
  • SD+ vs kPct: should be NEGATIVE (good decisions → fewer Ks). Expect
    r ≈ −0.4 to −0.6.
  • SD+ vs bbPct: should be POSITIVE. Expect r ≈ +0.3 to +0.5.
  • SD+ vs whiffPct: should be weakly negative — SD+ is decision quality,
    not contact ability. Expect r ≈ −0.1 to −0.3. If strongly negative,
    SD+ is leaking contact signal (a sign the decision-vs-execution
    separation isn't clean).
""".rstrip())
    pairs = []
    for key, v in normalized.items():
        ref = hitter_ref_lookup.get(key)
        if ref is None:
            continue
        pairs.append((v['sdPlus'], legacy_lookup.get(key), ref))

    def _corr(xs, ys):
        valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(valid) < 5:
            return None, 0
        n = len(valid)
        mx = sum(x for x, _ in valid) / n
        my = sum(y for _, y in valid) / n
        num = sum((x - mx) * (y - my) for x, y in valid)
        denx = math.sqrt(sum((x - mx) ** 2 for x, _ in valid))
        deny = math.sqrt(sum((y - my) ** 2 for _, y in valid))
        if denx == 0 or deny == 0:
            return None, n
        return num / (denx * deny), n

    sds = [t[0] for t in pairs]
    legacies = [t[1] for t in pairs]
    refs = [t[2] for t in pairs]

    def _report(label, key):
        vals = [r.get(key) if r else None for r in refs]
        r, n = _corr(sds, vals)
        r_str = f"{r:+.3f}" if r is not None else "  —  "
        print(f"    SD+ vs {label:<14}  r = {r_str:<7}  (n = {n})")

    r, n = _corr(sds, legacies)
    print(f"\n  Pearson correlations across {n} hitters:")
    print(f"    SD+ vs pdPlus (legacy)   r = {r:+.3f}" if r is not None else "    SD+ vs pdPlus: insufficient data")
    _report('wRCplus', 'wRCplus')
    _report('xwOBA', 'xwOBA')
    _report('kPct', 'kPct')
    _report('bbPct', 'bbPct')
    _report('whiffPct', 'whiffPct')
    _report('izSwChase', 'izSwChase')
    _report('contactPct', 'contactPct')


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: pitch cache not found at {CACHE_PATH}")
        print("Run process_data.py once to populate it, then re-run this script.")
        sys.exit(1)

    print(f"Loading pitch cache from {CACHE_PATH} ...")
    with open(CACHE_PATH, 'rb') as f:
        all_pitches = pickle.load(f)
    print(f"  Loaded {len(all_pitches):,} pitches.")

    print("Filtering to eligible decisions ...")
    eligible = [p for p in all_pitches if is_eligible(p)]
    print(f"  {len(eligible):,} eligible.")

    zone_counts = defaultdict(int)
    decision_counts = defaultdict(int)
    for p in eligible:
        zone_counts[classify_zone(p)] += 1
        decision_counts[classify_decision(p)] += 1

    print_filter_summary(all_pitches, eligible, zone_counts, decision_counts)

    print("\nBuilding raw weight table ...")
    raw_table = build_weight_table(eligible)
    zone_means = zone_level_means(eligible)
    print(f"  {len(raw_table)} non-empty (zone × count × decision) cells out of 120 possible.")

    print("Applying continuous Bayesian shrinkage (k = 50, toward decision-specific zone means) ...")
    smoothed = shrink_table(raw_table, zone_means)

    print_weight_table(raw_table, smoothed, zone_means)

    # Group pitches by hitter.
    print("\nGrouping by hitter ...")
    pitches_by_hitter = defaultdict(list)
    for p in eligible:
        batter = p.get('Batter')
        b_team = p.get('BTeam')
        if not batter or not b_team or b_team not in MLB_TEAMS:
            continue
        pitches_by_hitter[(batter, b_team)].append(p)
    print(f"  {len(pitches_by_hitter)} (hitter, team) entries.")

    print("Computing per-hitter dv and SD+ ...")
    hitter_raw = compute_hitter_sd(pitches_by_hitter, smoothed)
    # Section 3 compares multiple scaling options on the same raw data.
    per_scale = print_distribution(hitter_raw)
    # Sections 4 and 5 use k=30 as the recommended default for face checks.
    RECOMMENDED_K = 30
    normalized = per_scale[RECOMMENDED_K][0]
    print(f"\n  Sections 4 and 5 use k = {RECOMMENDED_K}.")
    print(f"  {len(normalized)} hitters pass the MIN_HITTER_DECISIONS = {MIN_HITTER_DECISIONS} floor.")

    # ── Variant comparison (RunExp vs xRV weights, dv_A vs dv_C) ──────
    print("\nBuilding variant comparison tables ...")
    raw_xrv   = build_weight_table(eligible, rv_fn=rv_hitter_xrv)
    zm_xrv    = zone_level_means(eligible, rv_fn=rv_hitter_xrv)
    smooth_xrv = shrink_table(raw_xrv, zm_xrv)
    variants = {
        'A_runexp (CURRENT)': (smoothed,  compute_dv_A),
        'A_xrv (BIP luck-neutral)':  (smooth_xrv, compute_dv_A),
        'C_runexp (league-relative)': (smoothed,  compute_dv_C),
        'C_xrv (both changes)':   (smooth_xrv, compute_dv_C),
    }
    variant_results = {}
    for name, (tab, dvf) in variants.items():
        hr_var = compute_hitter_sd(pitches_by_hitter, tab, dv_fn=dvf)
        norm_var = regress_and_normalize(hr_var, scale_k=RECOMMENDED_K)
        variant_results[name] = norm_var
    print_variant_comparison(variant_results, ref_lookup=None)


    # Load legacy pdPlus and reference stats.
    with open(HITTER_LB_PATH) as f:
        hitter_lb = json.load(f)
    legacy_lookup = {}
    ref_lookup = {}
    for row in hitter_lb:
        if not row.get('hitter') or not row.get('team'):
            continue
        key = (row['hitter'], row['team'])
        legacy_lookup[key] = row.get('pdPlus')
        ref_lookup[key] = row

    print_top_bottom(normalized, legacy_lookup)
    print_correlation(normalized, legacy_lookup, ref_lookup)

    hr("DONE")


if __name__ == '__main__':
    main()
