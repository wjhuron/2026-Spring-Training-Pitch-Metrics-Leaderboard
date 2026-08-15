"""SD+ (Swing Decisions+) — per-pitch decision-quality metric.

Builds a 120-cell (5 zones × 12 counts × 2 decisions; category-collapsed
since 2026-08-15, see the CATS block) run-value weight table from
league-wide MLB pitch data, then scores each
hitter on the decision-value of their own decisions (swing/take) using the
league cell weights, reweighted to the league zone mix. 100 = league-average
decision-maker.

Design highlights (config validated 2026-07-02, scripts/research/hitter/phase2_sdct_harness.py
+ scripts/research/hitter/phase2_sdplus_extensions.py):
- Zone classification: Baseball Savant attack zones, applied to the hitter-
  specific SzTop/SzBot (which already incorporate the ABS adjustment in
  this pipeline). Five buckets: heart / shadow_in / shadow_out / chase /
  waste. Shadow is split on whether the pitch is a strike (via compute_in_zone).
- Counts: all 12 as-is. Pitch categories: NONE since 2026-08-15 — the
  cat3 dimension failed its permuted-label placebo (see the CATS block).
- RV for cell weights: luck-neutral (xwOBA-based for BIP, -RunExp for
  non-BIP), BIP branch NOT count-anchored (changed 2026-08-15). The
  anchor was adopted 2026-07-02 on a single 2026 split-half; the
  multi-season battery (scripts/research/hitter/hitter_phase2_multiseason.py +
  hitter_phase2b_followup.py) showed removing it wins split-half
  reliability 6/6 seasons 2021-2026 (+0.014-0.026) and next-season
  prediction 3/4 pairs (+0.010), and the edge survives count-neutral
  aggregation — the estimated offsets add noise beyond any count-mix
  effect. CT+ KEEPS its anchor (flat/helpful there); Loc+ rejected the
  same anchor 0/5, so all three models now agree.
- Cell smoothing: Bayesian shrinkage cell → zone, k=CELL_SHRINK_K.
- Aggregation: MIX-NEUTRAL — per-zone mean dv reweighted to the league
  zone distribution, so opportunity (the pitch diet faced) doesn't leak
  into the decision score.
- Per-hitter regression: Bayesian regression toward the league mean with
  n_prior=HITTER_PRIOR_N pseudo-obs (= measured n0, MMSE-optimal).
- Normalization: ratio-to-league ×100 (see regress_and_normalize), NOT a
  z-score scale. Floor: MIN_HITTER_DECISIONS (split-half r=.50 point);
  the MLB 3.1 PA × team_games_played qualification is applied separately
  by the leaderboard consumer.

The decision-value formula is `dv = RV(chosen) - RV(opposite)` — absolute
opportunity cost, not league-relative.
"""
import math
from collections import defaultdict

from pipeline.utils import (
    safe_float, SWING_DESCRIPTIONS, BUNT_BB_TYPES, MLB_TEAMS,
    ZONE_HALF_WIDTH,
)

# ── Zone thresholds ─────────────────────────────────────────────────────
# Baseball Savant attack-zone diagram, all measured relative to zone center.
# Horizontal: fractions of zone half-width (≈10"). Vertical: fractions of
# the hitter-specific strike zone.
HEART_X   = 6.7 / 12      # ±6.7" from plate center = inner 67% of plate
SHADOW_X  = 13.3 / 12     # ±13.3" = outer 133% of plate
CHASE_X   = 20.0 / 12     # ±20"  = outer 200% of plate
HEART_VERT_FRAC  = 1.0 / 6.0    # trim 1/6 per side → heart = middle 67% of
                                # zone height, the true Savant heart. (Was
                                # 1/3 = middle 33% through 2026-07-02, which
                                # made shadow_in a mega-zone mixing meatballs
                                # with edge pitches; heart held 12% of
                                # decisions vs Savant's ~26%.)
SHADOW_VERT_FRAC = 1.0 / 6.0    # Shadow extends 17% of zone_ht above/below
CHASE_VERT_FRAC  = 0.5          # Chase extends 50% of zone_ht above/below

TAKE_DESCRIPTIONS = {'Called Strike', 'Ball'}

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
COUNTS = [(b, s) for b in range(4) for s in range(3)]

# Pitch categories. cat_of/CATS are the REAL three-way split, still used
# by CT+ (whose cat3 cells survived a permuted-label placebo on predictive
# evidence, 4/4 pairs) and other consumers.
#
# SD+'s OWN cells are CATEGORY-COLLAPSED (reverted 2026-08-15): the cat3
# dimension added 2026-07-02 (+0.016 split-half on 2026) turned out to be
# a flexibility artifact — a PERMUTED-label placebo beat the real cat3 on
# split-half reliability (0.7218 vs 0.7190) while real cat3 was WORST on
# next-season prediction (nocat +0.1723, placebo +0.1709, real +0.1684;
# scripts/research/hitter/sd_cat3_placebo.py). Given zone + count, pitch category carries
# no hitter-separating decision signal; split-half reliability is gameable
# by added partition structure and cannot justify structure by itself.
CATS = ('FB', 'BRK', 'OFF')
FB_CAT_TYPES = {'FF', 'SI', 'FC', 'FA'}
OFF_CAT_TYPES = {'CH', 'FS', 'SC', 'KN'}

SD_CATS = ('ALL',)


def _sd_cat(p):
    """SD+ cell category: collapsed (see revert note above)."""
    return 'ALL'


def cat_of(p):
    pt = p.get('Pitch Type')
    if pt in FB_CAT_TYPES:
        return 'FB'
    if pt in OFF_CAT_TYPES:
        return 'OFF'
    return 'BRK'

# ── Hyperparameters ─────────────────────────────────────────────────────
CELL_SHRINK_K  = 200      # cell → zone shrinkage pseudo-obs. Raised from 50
                          # on 2026-07-13: the k-sweep (scripts/
                          # sdct_constant_sweeps.py + sdct_k_predictive.py,
                          # 2021-2026) showed split-half reliability rising
                          # monotonically with k (SD+ .655→.682 at 200) while
                          # next-season prediction stayed dead flat — the
                          # count-level cell detail beyond k=200 was noise
                          # hitters were being scored against.
HITTER_PRIOR_N = 180      # hitter → league regression pseudo-obs. Set to
                          # the measured stabilization constant n0, i.e. the
                          # MMSE-optimal pseudo-count K=n0, matching CT+'s
                          # convention. Re-measured 2026-08-15 for the final
                          # definition (un-anchored, category-collapsed;
                          # scripts/research/hitter/n0_remeasure_sd_revert.py, 2024-2026:
                          # implied 170-186 across N, consensus 179). The
                          # anchored cat3 form measured 198/200; un-anchored
                          # cat3 measured 166.
MIN_HITTER_DECISIONS = 180  # floor = split-half r=.50 point (signal=
                            # noise), = n0 by construction; moves with the
                            # re-measure above. Leaderboard qualification
                            # (3.1 × TGP) is a separate stricter gate.

# MLB standard qualification: PA ≥ 3.1 × team games played.
PA_PER_TEAM_GAME = 3.1


# ═════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════

def classify_zone(p):
    """Return one of {'heart','shadow_in','shadow_out','chase','waste'} or None.

    Uses hitter-specific SzTop/SzBot (ABS-adjusted in the pipeline).
    Shadow is split in/out of zone via pipeline's InZone field.
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

    if ax <= HEART_X and z_heart_low <= pz <= z_heart_high:
        return 'heart'
    if ax <= SHADOW_X and z_shadow_low <= pz <= z_shadow_high:
        return 'shadow_in' if p.get('InZone') == 'Yes' else 'shadow_out'
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
    """Parse 'Count' column (e.g., '2-2') into (balls, strikes). None if invalid."""
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
    """Filter to pitches where a genuine swing/take decision occurred.

    Note: _source is intentionally NOT filtered here. The cell weight
    tables get an explicit MLB-only filter at the table-build step in
    compute_sd_plus / compute_ct_plus (keeping the baseline MLB-only),
    while per-hitter aggregation uses this class-based filter so ROC
    hitters can be measured against the MLB tables (translation
    framing — same convention as xwOBAsp / percentile pool / wRC+).
    """
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
    # RunExp not required here: per-hitter aggregation doesn't use it (the
    # cell table provides the RV via compute_dv / compute_ct_swing). The
    # table-build step in compute_sd_plus / compute_ct_plus already self-
    # filters pitches without RV via `if rv is None: continue`. Keeping
    # this filter would re-block ROC (RunExp 0% populated for AAA, same as
    # xwOBA/wOBAval — Savant model fields aren't published for AAA).
    if classify_decision(p) is None:
        return False
    if classify_zone(p) is None:
        return False
    if get_count(p) is None:
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════
#  RUN-VALUE STRATEGIES
# ═════════════════════════════════════════════════════════════════════════

def rv_hitter_runexp(p):
    """Raw hitter-perspective RV: -RunExp."""
    rv = safe_float(p.get('RunExp'))
    return -rv if rv is not None else None


# Multi-season pooled count-anchor offsets (2021-2026 means, measured
# 2026-07-13 via scripts/research/hitter/sdct_constant_sweeps.py Part C). The per-count
# offsets are near-constants of baseball (cross-season spread ≤0.016 runs
# for most counts, 0.045 worst case at 3-0), so when the in-season sample
# fails min_n — early season only — this is strictly better information
# than the old 0.0 fallback.
FALLBACK_COUNT_OFFSETS = {
    (0, 0): -0.002, (0, 1): +0.033, (0, 2): +0.095,
    (1, 0): -0.038, (1, 1): +0.010, (1, 2): +0.073,
    (2, 0): -0.102, (2, 1): -0.041, (2, 2): +0.035,
    (3, 0): -0.157, (3, 1): -0.148, (3, 2): -0.056,
}


def build_bip_count_offsets(pitches, lg_woba, woba_scale, min_n=50):
    """Per-count additive offset that puts the BIP xwOBA-value branch in the
    same count-conditional delta-RE currency as takes/whiffs/fouls.

        offset(c) = mean(-RunExp | BIP in count c)
                  - mean((xwOBA - lg_woba)/woba_scale | BIP in count c)

    -RunExp on a BIP is the actual count-conditional value of ending the PA
    from count c; the xwOBA branch is anchored to a neutral PA state. Their
    per-count means differ by exactly the count-state correction (outcome
    luck averages out within a count at league scale). Measured span is
    ~0.24 runs (0-2 BIPs undervalued ~0.10, 3-0/3-1 overvalued ~0.14) — see
    scripts/builders/count_anchor_offsets.py. Because the offset is a count-level
    constant, within-count variation stays 100% xwOBA-driven (luck-neutral).

    Counts with < min_n BIPs on either side fall back to the multi-season
    pooled offset (FALLBACK_COUNT_OFFSETS) — the offsets are near-constants
    across 2021-2026, so a pooled value beats the old 0.0 (neutral)
    fallback whenever the in-season sample is thin (early season)."""
    if lg_woba is None or woba_scale in (None, 0):
        return {}
    acc = {}
    for p in pitches:
        if p.get('Description') != 'In Play':
            continue
        c = get_count(p)
        if c is None:
            continue
        a = acc.setdefault(c, [0.0, 0, 0.0, 0])  # re_sum, re_n, xw_sum, xw_n
        re = safe_float(p.get('RunExp'))
        xw = safe_float(p.get('xwOBA'))
        if re is not None:
            a[0] += -re; a[1] += 1
        if xw is not None:
            a[2] += (xw - lg_woba) / woba_scale; a[3] += 1
    offsets = dict(FALLBACK_COUNT_OFFSETS)
    for c, (rs, rn, xs, xn) in acc.items():
        if rn >= min_n and xn >= min_n:
            offsets[c] = rs / rn - xs / xn
    return offsets


def make_rv_xrv(lg_woba, woba_scale, count_offsets=None):
    """Return an rv_fn that produces luck-neutral hitter-perspective RV:
    xwOBA-based for BIP pitches with xwOBA, -RunExp for everything else
    (including BIP without xwOBA). Falls back gracefully if Guts constants
    are missing.

    count_offsets (from build_bip_count_offsets) count-anchors the BIP
    branch so it shares the delta-RE currency of the non-BIP outcomes."""
    has_guts = (lg_woba is not None and woba_scale is not None
                and woba_scale != 0)

    def _fn(p):
        if has_guts and p.get('Description') == 'In Play':
            xw = safe_float(p.get('xwOBA'))
            if xw is not None:
                v = (xw - lg_woba) / woba_scale
                if count_offsets:
                    c = get_count(p)
                    if c is not None:
                        v += count_offsets.get(c, 0.0)
                return v
        rv = safe_float(p.get('RunExp'))
        return -rv if rv is not None else None
    return _fn


# ═════════════════════════════════════════════════════════════════════════
#  WEIGHT TABLE
# ═════════════════════════════════════════════════════════════════════════

def build_weight_table(pitches, rv_fn):
    """dict[(zone, count, cat, decision)] -> (mean_rv, n)."""
    cells = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    for p in pitches:
        zone = classify_zone(p)
        decision = classify_decision(p)
        count = get_count(p)
        rv = rv_fn(p)
        if rv is None:
            continue
        key = (zone, count, _sd_cat(p), decision)
        cells[key]['sum'] += rv
        cells[key]['n'] += 1
    return {k: (v['sum'] / v['n'], v['n']) for k, v in cells.items()}


def zone_level_means(pitches, rv_fn):
    """(zone × cat × decision) and (zone × decision) means — the two levels
    of the shrinkage cascade."""
    zc_sum = defaultdict(float); zc_n = defaultdict(int)
    z_sum = defaultdict(float); z_n = defaultdict(int)
    for p in pitches:
        zone = classify_zone(p)
        decision = classify_decision(p)
        rv = rv_fn(p)
        if rv is None:
            continue
        zc_sum[(zone, _sd_cat(p), decision)] += rv
        zc_n[(zone, _sd_cat(p), decision)] += 1
        z_sum[(zone, decision)] += rv
        z_n[(zone, decision)] += 1
    return ({k: (zc_sum[k] / zc_n[k], zc_n[k]) for k in zc_sum},
            {k: (z_sum[k] / z_n[k], z_n[k]) for k in z_sum})


def shrink_table(raw_table, zone_means, k=CELL_SHRINK_K):
    """Cascade Bayesian shrinkage: cell → (zone × cat) → zone, k pseudo-obs
    per level. Returns dict keyed by (zone, count, cat, decision) with ALL
    360 combinations populated."""
    zc_means, z_means = zone_means
    smoothed = {}
    for zone in ZONES:
        for count in COUNTS:
            for cat in SD_CATS:
                for decision in ('swing', 'take'):
                    z_mean, _zn = z_means.get((zone, decision), (0.0, 0))
                    zc_mean, zc_n = zc_means.get((zone, cat, decision), (0.0, 0))
                    zc_shrunk = ((zc_n * zc_mean + k * z_mean) / (zc_n + k)
                                 if (zc_n + k) else z_mean)
                    key = (zone, count, cat, decision)
                    cell_mean, n = raw_table.get(key, (0.0, 0))
                    rv = (n * cell_mean + k * zc_shrunk) / (n + k)
                    smoothed[key] = (rv, n)
    return smoothed


# ═════════════════════════════════════════════════════════════════════════
#  PER-HITTER SCORING
# ═════════════════════════════════════════════════════════════════════════

def compute_dv(p, table):
    """dv = RV(chosen) - RV(opposite). Symmetric opportunity cost."""
    zone = classify_zone(p)
    decision = classify_decision(p)
    count = get_count(p)
    cat = _sd_cat(p)
    swing_rv, _ = table[(zone, count, cat, 'swing')]
    take_rv,  _ = table[(zone, count, cat, 'take')]
    if decision == 'swing':
        return swing_rv - take_rv
    else:
        return take_rv - swing_rv


def compute_hitter_sd(pitches_by_hitter, table, lg_zone_w=None):
    """dict[(hitter, team)] -> {'raw_sd', 'n_decisions', 'zone_dv'}.

    raw_sd is MIX-NEUTRAL (2026-07-02): the hitter's per-zone mean dv is
    reweighted to the LEAGUE zone distribution (lg_zone_w), so seeing a more
    separable pitch diet (more heart+waste, fewer coin-flip shadow pitches)
    no longer inflates the score — that is opportunity, not decision skill
    (SEAGER controls the same confound). Weights renormalize over the zones
    the hitter actually has. Validated: split-half r +0.02-0.03 and the
    stabilization n0 drops ~10-15% vs the plain per-decision mean
    (scripts/research/hitter/phase2_sdplus_extensions.py). Falls back to the plain mean when
    lg_zone_w is None."""
    results = {}
    for key, pitches in pitches_by_hitter.items():
        elig = [p for p in pitches if is_eligible(p)]
        if not elig:
            continue
        dvs = [compute_dv(p, table) for p in elig]
        zone_dvs = defaultdict(list)
        for p, dv in zip(elig, dvs):
            zone_dvs[classify_zone(p)].append(dv)
        zone_means = {z: sum(vs) / len(vs) for z, vs in zone_dvs.items() if vs}
        if lg_zone_w:
            wsum = sum(lg_zone_w.get(z, 0.0) for z in zone_means)
            raw_sd = (sum(m * lg_zone_w.get(z, 0.0) for z, m in zone_means.items()) / wsum
                      if wsum > 0 else sum(dvs) / len(dvs))
        else:
            raw_sd = sum(dvs) / len(dvs)
        results[key] = {
            'raw_sd': raw_sd,
            'n_decisions': len(dvs),
            'zone_dv': {z: zone_means.get(z) for z in zone_dvs},
        }
    return results


def regress_and_normalize(hitter_raw, n_prior=HITTER_PRIOR_N,
                          min_n=MIN_HITTER_DECISIONS):
    """Ratio-to-league scaling, matching BB+ convention:
        sdPlus = 100 × hitter_raw_adj / league_mean_raw_adj
    where raw_sd_adj is the Bayesian-regressed per-hitter mean decision
    value, and league mean is computed across eligible hitters.

    Because the raw metric is signed and centered near a small positive
    league mean (~0.015), the ratio spread is wider than BB+'s. Hitters
    below league mean produce values below 100; hitters with negative
    raw_sd_adj produce negative sdPlus.
    """
    eligible = {k: v for k, v in hitter_raw.items() if v['n_decisions'] >= min_n}
    if not eligible:
        return {}

    # League anchors (lg_raw, lg_mean) use a de-duplicated POOL: for a multi-team
    # hitter, only the combined 2TM/3TM row represents them — their per-team stint
    # rows are excluded so a traded hitter isn't counted 2-3x. sdPlus is still
    # computed for every eligible row (combined AND stints).
    def _is_combined(t):
        return isinstance(t, str) and t.endswith('TM') and t[:-2].isdigit()
    combined_ids = {k[:1] for k in eligible if _is_combined(k[1])}
    pool = {k: v for k, v in eligible.items()
            if _is_combined(k[1]) or k[:1] not in combined_ids}

    lg_raw = sum(v['raw_sd'] for v in pool.values()) / len(pool)
    for v in eligible.values():
        n = v['n_decisions']
        v['raw_sd_adj'] = (n * v['raw_sd'] + n_prior * lg_raw) / (n + n_prior)

    adj_vals = [pool[k]['raw_sd_adj'] for k in pool]
    lg_mean = sum(adj_vals) / len(adj_vals)

    for v in eligible.values():
        if abs(lg_mean) > 1e-6:
            v['sdPlus'] = round(100.0 * v['raw_sd_adj'] / lg_mean, 1)
        else:
            v['sdPlus'] = 100.0
    return eligible


# ═════════════════════════════════════════════════════════════════════════
#  PACKAGING
# ═════════════════════════════════════════════════════════════════════════

def serialize_weight_table(smoothed):
    """Turn the smoothed cell table into a JSON-friendly dict keyed by
    `{zone}|{cat}|{balls}-{strikes}|{decision}` → {'rv': float, 'n': int}."""
    out = {}
    for (zone, count, cat, decision), (rv, n) in smoothed.items():
        key = f"{zone}|{cat}|{count[0]}-{count[1]}|{decision}"
        out[key] = {'rv': round(rv, 5), 'n': n}
    return out


def compute_team_games_played(all_pitches):
    """Distinct (Game Date) count per MLB team, using both pitcher and
    batter team columns. Close enough for MLB-standard qualification;
    double-headers (rare) would be undercounted by at most a handful
    per team per season."""
    team_dates = defaultdict(set)
    for p in all_pitches:
        if p.get('_source') != 'MLB':
            continue
        date = p.get('Game Date')
        if not date:
            continue
        for team_col in ('PTeam', 'BTeam'):
            team = p.get(team_col)
            if team and team in MLB_TEAMS:
                team_dates[team].add(date)
    return {t: len(d) for t, d in team_dates.items()}


def compute_sd_plus(all_pitches, pitches_by_hitter, lg_woba, woba_scale):
    """Main entry point.

    Args:
        all_pitches: flat list of pitch dicts (MLB + AAA/ROC filtered
            inside via is_eligible)
        pitches_by_hitter: dict[(hitter, team)] -> list of pitch dicts
        lg_woba, woba_scale: FanGraphs Guts constants for xRV

    Returns:
        normalized: dict[(hitter, team)] -> {sdPlus, raw_sd, raw_sd_adj,
            n_decisions, zone_dv, z}
        weight_table_json: dict for metadata output (audit/frontend)
    """
    # Cell weight tables stay MLB-baselined (translation framing); ROC
    # hitters are looked up against this MLB table by compute_hitter_sd.
    eligible = [p for p in all_pitches if p.get('_source','MLB')=='MLB' and is_eligible(p)]

    # BIP branch un-anchored (2026-08-15, see module docstring). The
    # count-anchor machinery stays exported for CT+ (which keeps it).
    rv_fn = make_rv_xrv(lg_woba, woba_scale)

    raw_table = build_weight_table(eligible, rv_fn)
    zone_means = zone_level_means(eligible, rv_fn)
    smoothed = shrink_table(raw_table, zone_means)

    # League zone distribution for the mix-neutral aggregation.
    zone_counts = defaultdict(int)
    for p in eligible:
        zone_counts[classify_zone(p)] += 1
    tot = sum(zone_counts.values())
    lg_zone_w = {z: n / tot for z, n in zone_counts.items()} if tot else None

    hitter_raw = compute_hitter_sd(pitches_by_hitter, smoothed, lg_zone_w)
    normalized = regress_and_normalize(hitter_raw)

    return normalized, serialize_weight_table(smoothed)
