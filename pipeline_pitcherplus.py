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

DERIVATION (2026-07-24, scripts/pitcherplus_search.py + _combo.py). Weights
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
from pipeline_utils import safe_float

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


def _pctl(v, pool):
    """Percentile rank of v within pool (ties averaged), 0-100."""
    if v is None or not pool:
        return None
    below = sum(1 for x in pool if x < v)
    equal = sum(1 for x in pool if x == v)
    return round(100.0 * (below + 0.5 * equal) / len(pool), 1)


def apply_pitcher_plus(rows, aaa_teams=('ROC', 'AAA')):
    """Set row['pitcherPlus'] + row['pitcherPlus_pctl'] in place. Returns the
    baseline bundle (for metadata / the client-side aggregator), or None if
    the pool was too thin to calibrate.

    Every row with scorable components gets BOTH a score and a rank; the
    percentile pool is the qualified MLB pool (matching the stuffScore /
    pitchingScore convention), and qualification stays a render-time
    coloring gate on the leaderboard.
    """
    base = build_baseline(rows, aaa_teams)
    for r in rows:
        r['pitcherPlus'] = score_row(r, base) if base else None
    pool = [r['pitcherPlus'] for r in rows
            if r.get('pitcherPlus') is not None and _is_baseline(r, set(aaa_teams))]
    for r in rows:
        r['pitcherPlus_pctl'] = _pctl(r.get('pitcherPlus'), pool)
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
        'qualN': QUAL_N, 'scaleK': SCALE_K,
    }
