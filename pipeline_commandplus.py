"""Command+ — per-pitch execution metric: miss distance from inferred intent.

Did he put the pitch where he meant to? Loc+ grades the VALUE of a location;
Command+ grades EXECUTION: the distance (inches) between where the pitch went
and the nearest of the pitcher's own inferred targets. The two are related
(r ~ 0.5-0.7) but distinct: target quality vs target attainment.

MODEL (research: scripts/commandplus_v1.py + _battery + _bb_followup,
commit 5518ec4 — validated BEFORE this port existed, per the multi-season
standard):
  - cell = (pitcher, pitch type, batter hand, count group), count groups
      EVEN     0-0, 0-1, 1-1          (establish intent)
      PUTAWAY  any two-strike count   (bury / put-away intent)
      BEHIND   1-0, 2-0, 3-0, 2-1, 3-1 (must-strike intent)
  - intended targets = component means of a Gaussian mixture fit to the
    cell's plate locations in PHYSICAL INCHES; K = 1..3 chosen by BIC,
    K capped by cell size (1 under 30 pitches, 2 under 60, 3 above)
  - per-pitch miss = Euclidean inches to the NEAREST target
  - pitcher raw = plain mean of his pitch misses (mean beat median on both
    reliability and persistence in all six seasons)
  - NO minimum-separation merge guard: swept 0-16in and proven flat to the
    third decimal on both objectives in every season — BIC plus the sample
    caps already prevent a wild pitcher's cloud from splitting itself into
    flattering targets. (Misses to component MEANS in fixed inches — never
    Mahalanobis with the pitcher's own covariance — remains load-bearing.)

VALIDATION HEADLINES (2021-2025 per-season replicates + 2026, never pooled):
  split-half reliability 0.795; inter-season persistence 0.793 (four clean
  year-pairs; Loc+ ~0.4); +0.85 agreement with release-angle repeatability
  (independent kinematics route — external validation); predicts NEXT-season
  BB% beyond current BB% (+0.08..+0.26) and beyond Loc+ (+0.14..+0.33) in
  4/4 pairs. It does NOT predict future xRV beyond Loc+ and velocity — the
  site copy must frame it as execution repeatability that forecasts walk
  rate, not a run-prevention claim. A real velocity tradeoff exists
  (r ~ +0.3 with FF velo, every season).

ROC/AAA: command is SELF-REFERENTIAL — a pitcher's misses vs his own
targets — so ROC needs no MLB baseline translation (unlike Stuff+/Loc+).
ROC pitchers are scored from their own pitches and excluded from the
normalization pool, per the site convention.

Normalization: commandPlus = 100 + 10 * (lg_miss - miss) / sigma over the
qualified MLB pool (higher = better command). Display gating: the
stabilization constant must be measured at the RENDERED unit before any
percentile coloring ships (see STABILIZE_TODO below).

Pure Python by design (no numpy/sklearn in the pipeline): 2x2 EM is
hand-rolled below and parity-tested against the research engine
(scripts/commandplus_port_parity.py).
"""
import math
from collections import defaultdict

from pipeline_utils import safe_float

# ── Model constants (all measured or proven-flat; see module docstring) ──
MIN_CELL = 20                          # min pitches to fit a cell
K_CAPS = ((60, 3), (30, 2), (0, 1))    # (min pitches, max K), first match wins
REG_COVAR = 1e-3                       # variance floor (matches research)
EM_TOL = 1e-4                          # loglik convergence per pitch
EM_MAX_ITER = 200

MIN_POOL = 300                         # pitches to enter the (mu, sigma) pool
CMD_SCALE_K = 10                       # display points per pool SD

# STABILIZE_TODO: the render-time coloring gate must come from a measured
# split-half r=0.5 crossing at the RENDERED unit (pitcher overall; per-type
# if an Arsenal column ships) — measured on the production scorer, not the
# research engine. Do NOT ship coloring on a guessed value.
STABILIZE_N = None

EXCLUDE_DESC = {'Hit By Pitch', 'Foul Bunt', 'Missed Bunt', 'Bunt Foul Tip',
                'Pitchout', 'Swinging Pitchout', 'Foul Pitchout', 'Intent Ball'}
EXCLUDE_PT = {'EP', 'PO'}
HANDS = ('L', 'R')


def count_group(count_str):
    """EVEN / PUTAWAY / BEHIND. None if unparseable."""
    if not isinstance(count_str, str) or '-' not in count_str:
        return None
    try:
        b, s = count_str.split('-', 1)
        b, s = int(b), int(s)
    except (TypeError, ValueError):
        return None
    if not (0 <= b <= 3 and 0 <= s <= 2):
        return None
    if s == 2:
        return 'PUTAWAY'
    if b > s:
        return 'BEHIND'
    return 'EVEN'


def is_eligible(p):
    """Competitive pitch with a location and a complete cell key."""
    if p.get('Description') in EXCLUDE_DESC:
        return False
    if p.get('Event') == 'Intent Walk':
        return False
    pt = p.get('Pitch Type')
    if not pt or pt in EXCLUDE_PT:
        return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS:
        return False
    if safe_float(p.get('PlateX')) is None or safe_float(p.get('PlateZ')) is None:
        return False
    return count_group(p.get('Count')) is not None


# ═════════════════════════════════════════════════════════════════════════
#  2-D GAUSSIAN MIXTURE, PURE PYTHON
# ═════════════════════════════════════════════════════════════════════════
# Deterministic by construction (no RNG): component means initialize via
# farthest-point seeding from the data mean, so a given cell always yields
# the same targets. sklearn's random restarts can land in different local
# optima per run; determinism is worth more to the pipeline than the last
# drop of likelihood, and the parity test measures the actual gap.

def _mean2(pts):
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    return (sx / n, sy / n)


def _cov2(pts, mx, my):
    n = len(pts)
    sxx = syy = sxy = 0.0
    for x, y in pts:
        dx = x - mx; dy = y - my
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy
    return (sxx / n + REG_COVAR, syy / n + REG_COVAR, sxy / n)


def _logpdf2(x, y, mx, my, cxx, cyy, cxy):
    det = cxx * cyy - cxy * cxy
    if det <= 1e-12:
        det = 1e-12
    dx = x - mx; dy = y - my
    # inverse of [[cxx, cxy], [cxy, cyy]] applied to (dx, dy)
    q = (cyy * dx * dx - 2.0 * cxy * dx * dy + cxx * dy * dy) / det
    return -0.918938533204673 * 2 - 0.5 * math.log(det) - 0.5 * q
    # -0.918... = -0.5*log(2*pi); doubled for two dimensions


def _farthest_point_init(pts, k):
    """Deterministic k-seed: data mean first, then repeatedly the point
    farthest from its nearest existing seed."""
    seeds = [_mean2(pts)]
    while len(seeds) < k:
        best_d, best_p = -1.0, None
        for x, y in pts:
            d = min((x - sx) ** 2 + (y - sy) ** 2 for sx, sy in seeds)
            if d > best_d:
                best_d, best_p = d, (x, y)
        seeds.append(best_p)
    return seeds


def _em_fit(pts, k):
    """EM for a k-component full-covariance 2-D GMM.
    Returns (means, weights, loglik). means = [(mx, my)] * k."""
    n = len(pts)
    means = _farthest_point_init(pts, k)
    covs = [_cov2(pts, mx, my) for mx, my in means]
    weights = [1.0 / k] * k
    prev_ll = None
    resp = [[0.0] * k for _ in range(n)]
    for _it in range(EM_MAX_ITER):
        # E-step
        ll = 0.0
        for i, (x, y) in enumerate(pts):
            row = resp[i]
            mx_l = None
            for j in range(k):
                m = means[j]; c = covs[j]
                lp = math.log(weights[j] + 1e-300) + _logpdf2(x, y, m[0], m[1], c[0], c[1], c[2])
                row[j] = lp
                if mx_l is None or lp > mx_l:
                    mx_l = lp
            s = 0.0
            for j in range(k):
                row[j] = math.exp(row[j] - mx_l)
                s += row[j]
            for j in range(k):
                row[j] /= s
            ll += mx_l + math.log(s)
        if prev_ll is not None and abs(ll - prev_ll) < EM_TOL * n:
            prev_ll = ll
            break
        prev_ll = ll
        # M-step
        for j in range(k):
            nj = sum(resp[i][j] for i in range(n))
            if nj < 1e-8:
                # dead component: reseed at the worst-fit point
                worst_i = max(range(n), key=lambda i: -max(resp[i]))
                means[j] = pts[worst_i]
                covs[j] = _cov2(pts, means[j][0], means[j][1])
                weights[j] = 1.0 / n
                continue
            mx = sum(resp[i][j] * pts[i][0] for i in range(n)) / nj
            my = sum(resp[i][j] * pts[i][1] for i in range(n)) / nj
            sxx = syy = sxy = 0.0
            for i in range(n):
                r = resp[i][j]
                dx = pts[i][0] - mx; dy = pts[i][1] - my
                sxx += r * dx * dx; syy += r * dy * dy; sxy += r * dx * dy
            means[j] = (mx, my)
            covs[j] = (sxx / nj + REG_COVAR, syy / nj + REG_COVAR, sxy / nj)
            weights[j] = nj / n
    return means, weights, prev_ll


def _bic(loglik, k, n):
    # full-covariance 2-D mixture: 2k means + 3k covariances + (k-1) weights
    p = 6 * k - 1
    return -2.0 * loglik + p * math.log(n)


def fit_targets(pts):
    """BIC-selected targets for one cell. pts = [(x_in, z_in), ...].
    Returns list of (mx, my) target means."""
    n = len(pts)
    kmax = next(k for thr, k in K_CAPS if n >= thr)
    best_means, best_bic = None, None
    for k in range(1, kmax + 1):
        means, _w, ll = _em_fit(pts, k)
        b = _bic(ll, k, n)
        if best_bic is None or b < best_bic:
            best_means, best_bic = means, b
    return best_means


# ═════════════════════════════════════════════════════════════════════════
#  SCORING + AGGREGATION
# ═════════════════════════════════════════════════════════════════════════
def score_misses(pitches_by_key):
    """pitches_by_key: dict[key] -> list of pitch dicts (key is the caller's
    pitcher grouping, e.g. (Pitcher, PTeam, Throws)). Cells are built WITHIN
    each key, so combined 2TM rows and per-team stints each get their own
    targets from their own pitches.

    Returns dict[key] -> {'raw_miss': mean inches, 'n_pitches': int,
                          'pt_miss': {pitch type: (mean, n)}}."""
    out = {}
    for key, plist in pitches_by_key.items():
        cells = defaultdict(list)
        for p in plist:
            if not is_eligible(p):
                continue
            ck = (p.get('Pitch Type'), p.get('Bats'), count_group(p.get('Count')))
            x = safe_float(p.get('PlateX')) * 12.0
            z = safe_float(p.get('PlateZ')) * 12.0
            cells[ck].append((x, z))
        total = 0.0
        n_tot = 0
        pt_acc = defaultdict(lambda: [0.0, 0])
        for ck, pts in cells.items():
            if len(pts) < MIN_CELL:
                continue
            targets = fit_targets(pts)
            for x, z in pts:
                d = min(math.hypot(x - tx, z - tz) for tx, tz in targets)
                total += d
                n_tot += 1
                pt_acc[ck[0]][0] += d
                pt_acc[ck[0]][1] += 1
        if n_tot == 0:
            continue
        out[key] = {
            'raw_miss': total / n_tot,
            'n_pitches': n_tot,
            'pt_miss': {pt: (s / n, n) for pt, (s, n) in pt_acc.items()},
        }
    return out


def normalize(results, pool_filter):
    """Adds 'commandPlus' to each result: 100 + 10*(lg - miss)/sigma, with
    (lg, sigma) from qualified MLB pool rows (pool_filter(key) True and
    n_pitches >= MIN_POOL). ROC keys are scored, never pooled. Mutates and
    returns results plus the (lg, sigma) anchors."""
    pool = [v['raw_miss'] for k, v in results.items()
            if pool_filter(k) and v['n_pitches'] >= MIN_POOL]
    if len(pool) < 10:
        for v in results.values():
            v['commandPlus'] = None
        return results, (None, None)
    mu = sum(pool) / len(pool)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in pool) / len(pool))
    for v in results.values():
        if sigma > 1e-9:
            v['commandPlus'] = round(100.0 + CMD_SCALE_K * (mu - v['raw_miss']) / sigma, 1)
        else:
            v['commandPlus'] = None
    return results, (mu, sigma)
