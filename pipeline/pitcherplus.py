"""pipeline_pitcherplus.py — Pitcher+ , the all-encompassing pitcher metric.

Pitcher+ answers "how good is this pitcher, really" on a single 100 +/- 10
scale. It is a weighted composite of six pitcher-level components, each
z-scored against the current-season MLB pool and each SHRUNK toward league
average by its own measured stabilization rate, so one formula serves a
40-inning reliever and a 190-inning starter without a role term.

    Pitcher+ = 100 + 10 * z( sum_f  w_f * z_f * n/(n + k_f) )

    component      w      k (pitches)   what it contributes
    stuffScore    0.20        42        raw weapons, stabilizes almost instantly
    locPlus       0.06       215        command, the only early-season location signal
    kPct          0.21       398        bat-missing translated to outcomes
    izWhiffPct    0.19       421        in-zone swing-and-miss, chase-independent
    xRv100        0.23      1046        luck-neutral results, slowest but most informative
    gbPct         0.12       333        contact identity (near-orthogonal to the rest)

DERIVATION (2026-07-24, scripts/research/stuff/pitcherplus_search.py + _combo.py). Weights
were found by exhaustive best-subset search over ~60 candidates and 20
survivors, validated on two panels against FUTURE xRV/100: split-half
(odd/even games within season, both directions) and year-pair (2021->22 ...
2024->25), 2021-2025. Reported weights are the mean of the two panels'
normalized OLS fits, rounded to 2 decimals (rounding costs r < 0.001).
Out-of-fold combined r = 0.552, vs benchmarks: shipped Pitching+ 0.461,
regressed xRV/100 0.447, kwERA-core (K-BB%) 0.437, FIP-core 0.387.

NO ROLE TERM, by design (Wally, 2026-07-24). pitchesPerG predicts future
xRV/100 negatively (relievers produce better per-pitch value: max effort,
never a third time through the order) and adding it scores marginally
better (0.559 vs 0.549). It is excluded because it measures the JOB, not
the pitcher: two identical arms would grade differently by bullpen
assignment, and it demotes elite starters (Skubal -5.1, Skenes -6.1 in the
2025 backtest) for the context they pitch in. Weights were re-fit WITH the
role term partialled out to confirm the other six are not secretly carrying
role signal: max coefficient shift 0.05, transfer r identical (0.5456 vs
0.5444), so the plain fit ships. SP/RP comparison is a filter, not a term.

NOT AN ATOM METRIC, AND SEASON-LEVEL. Unlike Stuff+/Loc+/Pitching+,
Pitcher+ does not decompose into per-pitch grades: K%, GB% and xRV/100 are
pitcher-level aggregates with PA/BIP denominators. It therefore does NOT
obey the coherent-canon ledger property (mean of per-pitch atoms). It is
also NOT re-derived under date/hand filters: its heaviest component
(xRv100, weight 0.23) is boxscore-merged on the client and already ignores
those filters, so a composite rebuilt from half-filtered parts would be
less honest than carrying the season value. Pitcher+ therefore behaves
exactly like SIERA/FIP/xRV100 on the leaderboard — a season-level number
merged through js/aggregator.js boxFields. The (mu, sigma) baseline is
still published to metadata as pitcherPlusBaseline for auditability and to
leave the door open for a fully-filtered recompute later.
"""
import os

from pipeline.utils import safe_float

# component -> (weight, stabilization constant in pitches, invert)
# invert=True means lower raw value is better for the pitcher (none today;
# every component is already oriented so higher = better).
COMPONENTS = (
    ('stuffScore', 0.20, 42.0),
    ('locPlus',    0.06, 215.0),
    ('kPct',       0.21, 398.0),
    ('izWhiffPct', 0.19, 421.0),
    ('xRv100',     0.23, 1046.0),
    ('gbPct',      0.12, 333.0),
)

QUAL_N = 300          # min pitches to enter the (mu, sigma) baseline pool
SCALE_K = 10          # points per SD of the composite
MIN_POOL = 50         # below this many qualified pitchers, don't score at all

# pitcherRuns100 = PRED_SLOPE * (Pitcher+ - 100): the run-denominated
# companion for the hover tooltip, and — like pitchingRuns100 — deliberately
# a MONOTONIC transform of the displayed number so it can never invert
# against the column it annotates.
#
# The slope is FROZEN and PREDICTIVE (0.039 runs/100 per Pitcher+ point, so
# +10 = +0.39 runs/100), fit as future-season xRV/100 on current-season
# Pitcher+ over the 2021-25 year-pair panel. Two reasons it is not the
# same-season OLS that pitchingRuns100 uses:
#   1. Circularity. xRV/100 is a COMPONENT of Pitcher+ (weight 0.23), so a
#      same-season fit partly regresses a variable on itself and inflates
#      the slope to 0.061 — a ~55% overstatement of the runs claim.
#      Pitching+ has no results term, so its descriptive slope is clean.
#   2. It answers the question the metric is for: "a pitcher at this
#      Pitcher+ prevents about this many runs/100 GOING FORWARD."
# Frozen rather than re-fit per run because the predictive slope needs a
# following season that doesn't exist mid-year. Stability across the four
# year-pairs is 0.0370-0.0411 (2021->22 .0385, 22->23 .0391, 23->24 .0370,
# 24->25 .0411), pooled 0.03897.
PRED_SLOPE = 0.039

# ── Rejected: Results+ and a talent-minus-results Gap column ────────────
# Both were built on 2026-07-24 and removed the same day. Recorded here so
# they don't get reinvented:
#   Results+ (xRV/100 rescaled to 100 +/- 10) was a strictly LOSSY transform.
#   Indexing earns its place when the raw unit is meaningless (Stuff+, Loc+,
#   Pitcher+) or when the index adds an adjustment the raw lacks (ERA- adds
#   park/league). A plain rescale of xRV/100 does neither: "0.8 runs per 100
#   pitches" is already concrete, and 112 is less informative. It also runs
#   backwards against pitchingRuns100/pitcherRuns100, which exist to convert
#   indexes INTO runs precisely because runs mean more.
#   Gap (Pitcher+ - Results+) was worse: unit-incoherent. One Pitcher+ point
#   is 0.039 runs/100 (heavily shrunk); one Results+ point is 0.079
#   (unshrunk) — 2.04x apart — so subtracting them compares different
#   currencies. Misiorowski showed +26 points = +0.06 runs while Hader showed
#   -16 points = -1.92 runs, i.e. the display ranked them backwards.
#   The coherent runs version (pitcherRuns100 - xRv100) then correlates
#   -0.784 with the xRVOE/100 column that already ships, so it was a
#   near-duplicate of existing work. Use xRVOE/100 for results-vs-process.

# ── Pitcher+ Proj ───────────────────────────────────────────────────────
# Pitcher+ Proj = 70% current / 30% prior-season Pitcher+, re-standardized.
# Validated on 2021-25 year-pairs predicting NEXT-SEASON xRV/100: OOF r
# .6104 -> .6271. A free 2-variable fit independently lands on 32% prior.
# Settled by research (scripts/pitcherplus_v2_architecture.py):
#   - blend the COMPOSITE, not the components (free component-level fitting
#     overfits: .6113; component-70/30-then-weight is algebraically the
#     same as composite 70/30 anyway)
#   - do NOT sample-weight the prior (n-weighting it is strictly worse:
#     .6253/.6237/.6213 at k=500/1000/2000; dual weighting buys +.002)
#   - two years only; a third adds +.002 (fitted shares 59/33/8)
#   - pitchers with no prior season keep their current-only Pitcher+, the
#     standard Marcel/Steamer pattern
# NOT aging-adjusted (no birthdates in the pipeline) — a known limitation.
PROJ_CUR_W, PROJ_PRIOR_W = 0.70, 0.30
PRIOR_ASSET = 'pitcher_plus_prior.json'


def _count_of(row):
    return safe_float(row.get('count')) or 0.0


def _is_baseline(row, aaa_teams):
    """MLB only, qualified, combined 2TM/3TM rows excluded. ROC/AAA pitchers
    are SCORED but never define the baseline — same convention as Stuff+,
    Loc+ and Pitching+."""
    team = row.get('team')
    if row.get('_isROC') or team in aaa_teams:
        return False
    if row.get('_isCombined') or _is_combined_team(team):
        return False
    return _count_of(row) >= QUAL_N


def _is_combined_team(t):
    """'2TM'/'3TM' style multi-team aggregate rows."""
    return isinstance(t, str) and t.endswith('TM') and t[:-2].isdigit()


def build_baseline(rows, aaa_teams=('ROC', 'AAA')):
    """Per-component (mu, sigma) from the qualified MLB pool, plus the
    composite's own (mu, sigma) for the final 100 +/- 10 rescale.

    mu is COUNT-WEIGHTED (matches the site's league-average convention:
    a league mean is the pitch-weighted mean, not the mean of pitcher
    means); sigma is the unweighted between-pitcher SD, which is what the
    z-score is expressed in.
    """
    aaa_teams = set(aaa_teams)
    pool = [r for r in rows if _is_baseline(r, aaa_teams)]
    if len(pool) < MIN_POOL:
        return None
    base = {}
    for key, _w, _k in COMPONENTS:
        vals, wts = [], []
        for r in pool:
            v = safe_float(r.get(key))
            if v is None:
                continue
            vals.append(v)
            wts.append(_count_of(r))
        if len(vals) < MIN_POOL:
            return None
        tw = sum(wts)
        mu = sum(v * w for v, w in zip(vals, wts)) / tw if tw else \
            sum(vals) / len(vals)
        mean_u = sum(vals) / len(vals)
        var = sum((v - mean_u) ** 2 for v in vals) / (len(vals) - 1)
        sd = var ** 0.5
        if not sd:
            return None
        base[key] = (mu, sd)

    raws = [_raw_score(r, base) for r in pool]
    raws = [x for x in raws if x is not None]
    if len(raws) < MIN_POOL:
        return None
    rmu = sum(raws) / len(raws)
    rvar = sum((x - rmu) ** 2 for x in raws) / (len(raws) - 1)
    rsd = rvar ** 0.5
    if not rsd:
        return None
    base['_composite'] = (rmu, rsd)
    return base


def _raw_score(row, base):
    """Weighted sum of shrunk z-scores. A missing component contributes 0,
    which under shrinkage logic is exactly right: no evidence = league
    average. Returns None only if EVERY component is missing."""
    n = _count_of(row)
    total, seen = 0.0, 0
    for key, w, k in COMPONENTS:
        v = safe_float(row.get(key))
        if v is None:
            continue
        mu, sd = base[key]
        z = (v - mu) / sd
        total += w * z * (n / (n + k))
        seen += 1
    return total if seen else None


def score_row(row, base):
    """Pitcher+ for one row (100 = league average, +10 = 1 SD better).
    None when the row has no scorable components."""
    if not base:
        return None
    raw = _raw_score(row, base)
    if raw is None:
        return None
    rmu, rsd = base['_composite']
    return round(100.0 + SCALE_K * (raw - rmu) / rsd, 1)


def load_prior(data_dir, current_season=None):
    """{mlbId(str): prior-season Pitcher+}, plus the season it covers.

    LOUD on failure by design. Without the asset the projection silently
    collapses to an exact copy of Pitcher+ — a column that still looks like
    a projection but no longer is, with no error and no visual tell. That is
    the worst failure mode available here, so a missing, unreadable or STALE
    asset prints a warning; the caller surfaces it in the inject log. The
    asset covers season Y and is only valid for season Y+1, so it must be
    regenerated every year:
        python3 scripts/builders/pitcherplus_build_prior.py --season <last season>
    """
    import json
    path = os.path.join(data_dir, PRIOR_ASSET)
    try:
        with open(path) as f:
            blob = json.load(f)
    except OSError:
        print(f'  [WARN] Pitcher+ prior missing ({path}) — Pitcher+ Proj '
              f'will equal Pitcher+. Run scripts/builders/pitcherplus_build_prior.py')
        return {}, None
    except ValueError as e:
        print(f'  [WARN] Pitcher+ prior unreadable ({e}) — Pitcher+ Proj '
              f'will equal Pitcher+')
        return {}, None
    values = blob.get('values') or {}
    season = blob.get('season')
    if not values:
        print('  [WARN] Pitcher+ prior has no values — Pitcher+ Proj will '
              'equal Pitcher+')
    elif current_season is not None and season is not None:
        try:
            stale = int(current_season) - int(season)
        except (TypeError, ValueError):
            stale = None
        if stale is not None and stale != 1:
            print(f'  [WARN] Pitcher+ prior covers {season} but the current '
                  f'season is {current_season} — expected exactly one season '
                  f'back. Regenerate: python3 '
                  f'scripts/builders/pitcherplus_build_prior.py --season '
                  f'{int(current_season) - 1}')
    return values, season


from pipeline.utils import _pctl  # single-homed percentile convention


def apply_pitcher_plus(rows, aaa_teams=('ROC', 'AAA'), data_dir=None,
                       current_season=None):
    """Set row['pitcherPlus'] + row['pitcherPlus_pctl'] in place. Returns the
    baseline bundle (for metadata / the client-side aggregator), or None if
    the pool was too thin to calibrate.

    Every row with scorable components gets BOTH a score and a rank; the
    percentile pool is the qualified MLB pool (matching the stuffScore /
    pitchingScore convention), and qualification stays a render-time
    coloring gate on the leaderboard.
    """
    aaa = set(aaa_teams)
    base = build_baseline(rows, aaa_teams)
    for r in rows:
        v = score_row(r, base) if base else None
        r['pitcherPlus'] = v
        r['pitcherRuns100'] = (round(PRED_SLOPE * (v - 100.0), 2)
                               if v is not None else None)

    # ── Pitcher+ Proj: 70/30 with the frozen prior season ──
    prior, prior_season = (load_prior(data_dir, current_season)
                           if data_dir else ({}, None))
    blended, n_prior = [], 0
    for r in rows:
        v = r.get('pitcherPlus')
        if v is None:
            r['_projRaw'] = None
            continue
        pv = prior.get(str(r.get('mlbId'))) if r.get('mlbId') else None
        if pv is not None:
            r['_projRaw'] = PROJ_CUR_W * v + PROJ_PRIOR_W * float(pv)
            n_prior += 1
        else:
            r['_projRaw'] = v          # no prior -> current-only (Marcel)
        if _is_baseline(r, aaa):
            blended.append(r['_projRaw'])
    # Re-standardize: blending two imperfectly-correlated 100+/-10 values
    # compresses the SD (~9.4), so rescale to keep "+10 = 1 SD" true.
    if len(blended) >= MIN_POOL:
        bmu = sum(blended) / len(blended)
        bsd = (sum((x - bmu) ** 2 for x in blended)
               / (len(blended) - 1)) ** 0.5
    else:
        bmu = bsd = None
    for r in rows:
        raw = r.pop('_projRaw', None)
        r['pitcherPlusProj'] = (round(100.0 + SCALE_K * (raw - bmu) / bsd, 1)
                                if (raw is not None and bsd) else None)

    for key in ('pitcherPlus', 'pitcherPlusProj'):
        pool = [r[key] for r in rows
                if r.get(key) is not None and _is_baseline(r, aaa)]
        for r in rows:
            r[key + '_pctl'] = _pctl(r.get(key), pool)
    if base is not None:
        base['_projPriorSeason'] = prior_season
        base['_projNPrior'] = n_prior
    return base


def serialize_baseline(base):
    """JSON-friendly snapshot: the client recomputes Pitcher+ under filters
    against this frozen season-long baseline."""
    if not base:
        return None
    return {
        'components': [{'key': k, 'weight': w, 'k': kk,
                        'mu': round(base[k][0], 6), 'sd': round(base[k][1], 6)}
                       for k, w, kk in COMPONENTS],
        'composite': {'mu': round(base['_composite'][0], 6),
                      'sd': round(base['_composite'][1], 6)},
        'qualN': QUAL_N, 'scaleK': SCALE_K, 'predSlope': PRED_SLOPE,
        'proj': {'curW': PROJ_CUR_W, 'priorW': PROJ_PRIOR_W,
                 'priorSeason': base.get('_projPriorSeason'),
                 'nWithPrior': base.get('_projNPrior')},
    }
