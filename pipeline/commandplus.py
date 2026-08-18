"""Command+ — per-pitch execution metric: miss distance from inferred intent.

Did he put the pitch where he meant to? Loc+ grades the VALUE of a location;
Command+ grades EXECUTION: the distance (inches) between where the pitch went
and the nearest of the pitcher's own inferred targets. The two are related
(r ~ 0.5-0.7) but distinct: target quality vs target attainment.

MODEL (research: scripts/research/commandplus/commandplus_v1.py + _battery + _bb_followup,
commit 5518ec4 — validated BEFORE this port existed, per the multi-season
standard):
  - cell = (pitcher, pitch type, batter hand, count group), count groups
      EVEN     0-0, 0-1, 1-1          (establish intent)
      PUTAWAY  any two-strike count   (bury / put-away intent)
      BEHIND   1-0, 2-0, 3-0, 2-1, 3-1 (must-strike intent)
  - a cell under MIN_CELL is NOT dropped: its pitches cascade into a
    (pitch type, batter hand) residual bucket, then a (pitch type) bucket,
    and a target is fit at whichever level first clears MIN_CELL. Only
    pitches from thin cells enter a bucket, so a target is always fit on
    exactly the pitches it scores. See the CASCADE finding below.
  - intended target = the MEAN of the cell's plate locations in PHYSICAL
    INCHES (one target per cell; see the K=1 finding below)
  - per-pitch miss = Euclidean inches to that target
  - pitcher raw = plain mean of his pitch misses (mean beat median on both
    reliability and persistence in all six seasons)
  - misses measured in fixed inches — never Mahalanobis with the pitcher's
    own covariance, which would divide out the very scatter being graded.
    That remains load-bearing.

K=1 — WHY THE GAUSSIAN MIXTURE WAS REMOVED (2026-08-12,
scripts/research/commandplus/commandplus_ladder_multiseason.py). v1 fit a 1-3 component GMM per
cell, K by BIC with sample caps. A four-rung ladder run over 2021-2026
(GLOBAL: one centroid, no cells / CELLMEAN: cells, K=1 / PROD: the v1 GMM /
KFREE: K up to 6, no caps), every rung scored on the same three objectives
and restricted to a COMMON pitcher pool, settled it:

                    reliability  persistence  BB% same  BB% next
    GLOBAL             0.789        0.703       0.511     0.401
    CELLMEAN (K=1)     0.812        0.791       0.565     0.440
    PROD (GMM)         0.799        0.786       0.555     0.432
    KFREE              0.774        0.778       0.547     0.430

The CELLS are load-bearing (GLOBAL -> CELLMEAN gains .088 persistence).
The MIXTURE is not: K=1 beat the GMM on 22 of 22 comparisons (6 seasons
reliability, 5 year-pairs persistence, 6 same-season BB%, 5 next-season
BB%) with no exceptions, and KFREE is worse still — a monotone gradient
that says BIC was selecting overfit components. The mixture captured
something real for a minority of genuinely two-target pitchers (Mahle, who
works two distinct fastball spots vs LHH, drops 124.5 -> 115.2) and
manufactured noise for the majority. r(GMM, K=1) = 0.990 on the displayed
scale, median move 0.7 points.

Removing it also retires the whole circularity apparatus — the separation
merge guard, the BIC selection, the K caps — since one target per cell
cannot split a wild pitcher's cloud into flattering sub-targets at all.

CASCADE — WHY THIN CELLS ARE POOLED RATHER THAN DROPPED (2026-08-12,
scripts/research/commandplus/commandplus_battery_2026_08.py section D and
scripts/research/commandplus/commandplus_volume_fallback.py). Dropping sub-MIN_CELL cells fell
hardest on secondary pitches (CU ~34% of pitches excluded for the thinnest
arms vs FF ~11%), so a low-volume pitcher was graded largely on his
best-commanded pitch. Cascading those pitches into coarser buckets instead:

                     coverage  reliability  persistence  BB% same  BB% next
    drop thin cells     .914       .812         .796       .565      .440
    CASCADE             .972       .840         .809       .573      .439

Wins coverage 6/6, reliability 6/6, persistence 5/5, same-season walks 6/6,
ties next-season walks. r(drop, cascade) = 0.987 on the displayed scale;
the movers are thin-arsenal relievers whose secondary pitches had been
excluded, which is the intended effect.

KNOWN BIAS — LOW VOLUME IS STILL SOMEWHAT FLATTERED (measured, 6/6 seasons,
scripts/research/commandplus/commandplus_volume_fallback.py). Targets are fit on the same pitches
they score, and that fit hugs the data harder when there is less of it.
Downsampling high-volume pitchers to reliever scale makes their miss
SMALLER, by these display points:

    N=400  +0.78    N=600  +0.84    N=900  +0.60    N=1300  +0.35
    (six-season means; the pre-cascade scorer measured 2.35 / 1.67 / 1.00 /
     0.56, so the cascade cut it by two thirds and won 24/24 comparisons)

Two mechanisms drove the old bias, MIN_CELL truncation and in-sample
optimism, and they were NOT separable by argument — the cascade could have
deepened the optimism half, since a rescued 20-pitch bucket is where a
fitted mean hugs its own points hardest. Measurement settled it: truncation
dominated. What remains is nearly flat across sample size (0.78 / 0.84 /
0.60 / 0.35 rather than a steep decay), which is the signature of in-sample
optimism with the volume-dependent component removed.

The sample-dependent K caps were also suspected and were NOT a driver: the
v1 GMM measured 2.35 at N=400 against K=1's 2.41.

It does not invert anything — real high-volume arms command ~4 points better,
so the artifact shrinks a true gap rather than inventing one.

VALIDATION HEADLINES (2021-2025 per-season replicates + 2026, never pooled).
The v1 GMM scorer measured 0.795 reliability / 0.793 persistence; the K=1
scorer above measures 0.812 / 0.791 on the ladder harness (common pool,
game-date halves) and wins every objective in every season. Because
r(GMM, K=1) = 0.990, everything below carries over unchanged:
  inter-season persistence ~0.79 (four clean year-pairs; Loc+ ~0.4);
  +0.85 agreement with release-angle repeatability
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

Normalization: commandPlus = 200 - 100 * miss / lg_miss over the qualified
MLB pool (higher = better command). One point is one percent: a 115 misses by
15% less distance than league average. The anchor is recomputed from the pool
on every run — there is no stored scale constant to re-fit when the scorer
changes.

Pure Python by design (no numpy/sklearn in the pipeline).
"""
import math
from collections import defaultdict

from pipeline.utils import safe_float

# ── Model constants (all measured or proven-flat; see module docstring) ──
MIN_CELL = 20                          # min pitches to fit a cell

MIN_POOL = 300                         # pitches to enter the (mu, sigma) pool
CMD_SCALE_K = 10                       # RETIRED 2026-08-18. Command+ is now
                                       # 200 - 100*miss/lg (one point = one
                                       # percent), not a z-ruler. Kept only so
                                       # a stale import cannot break; delete
                                       # once nothing references it.

# RE-MEASURED 2026-08-12 on the cascade scorer (scripts/research/commandplus/commandplus_gate_measure.py):
# split-half r=0.5 crossing at the rendered unit (pitcher-season), random
# within-pitcher splits, targets fit per half on THIS scorer. Median of 5
# seeds = 143 (range 125-147). The lineage: v1 GMM 328, K=1 325 (the mixture
# never touched the gate), cascade 143 — pooling thin cells more than halves
# it, because a given pitch budget now yields far more SCORED pitches (.972
# coverage vs .914) at higher reliability. Comfortably below the reliever
# IP-qualification's ~450+ pitches, so Command+ rides the ordinary IP qual
# gate exactly as Loc+ does. Re-measure at season end with the Loc+ gates.
STABILIZE_N = 143

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
#  TARGET INFERENCE
# ═════════════════════════════════════════════════════════════════════════
def fit_targets(pts):
    """Inferred target for one cell. pts = [(x_in, z_in), ...].
    Returns a list of (mx, my) — always length 1; see the K=1 note above.
    The list shape is kept so score_misses stays a nearest-target lookup and
    a future multi-target model can drop straight back in."""
    n = len(pts)
    return [(sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)]


# ═════════════════════════════════════════════════════════════════════════
#  SCORING + AGGREGATION
# ═════════════════════════════════════════════════════════════════════════
def build_cells(plist):
    """Scoring cells for one pitcher, with the thin-cell CASCADE.

    Primary cell = (pitch type, batter hand, count group). A cell that misses
    MIN_CELL is not dropped: its pitches fall into a (pitch type, batter hand)
    residual bucket, and if that still misses, into a (pitch type) bucket.
    Only pitches from thin cells enter a residual bucket, so a target is
    always fit on exactly the pitches it scores.

    Returns [[(pitch_type, x_in, z_in), ...], ...] — one list per cell that
    cleared MIN_CELL at some level. Pitches that clear it at no level are
    dropped, which is now rare (mean coverage .972, was .914).
    """
    lvl1 = defaultdict(list)
    for p in plist:
        if not is_eligible(p):
            continue
        pt = p.get('Pitch Type')
        lvl1[(pt, p.get('Bats'), count_group(p.get('Count')))].append(
            (pt, safe_float(p.get('PlateX')) * 12.0,
             safe_float(p.get('PlateZ')) * 12.0))
    cells, res2 = [], defaultdict(list)
    for (pt, bats, _cg), pts in lvl1.items():
        if len(pts) >= MIN_CELL:
            cells.append(pts)
        else:
            res2[(pt, bats)].extend(pts)
    res3 = defaultdict(list)
    for (pt, _bats), pts in res2.items():
        if len(pts) >= MIN_CELL:
            cells.append(pts)
        else:
            res3[pt].extend(pts)
    for _pt, pts in res3.items():
        if len(pts) >= MIN_CELL:
            cells.append(pts)
    return cells


def score_misses(pitches_by_key):
    """pitches_by_key: dict[key] -> list of pitch dicts (key is the caller's
    pitcher grouping, e.g. (Pitcher, PTeam, Throws)). Cells are built WITHIN
    each key, so combined 2TM rows and per-team stints each get their own
    targets from their own pitches.

    Returns dict[key] -> {'raw_miss': mean inches, 'n_pitches': int,
                          'pt_miss': {pitch type: (mean, n)}}."""
    out = {}
    for key, plist in pitches_by_key.items():
        total = 0.0
        n_tot = 0
        pt_acc = defaultdict(lambda: [0.0, 0])
        for pts in build_cells(plist):
            targets = fit_targets([(x, z) for _pt, x, z in pts])
            for pt, x, z in pts:
                d = min(math.hypot(x - tx, z - tz) for tx, tz in targets)
                total += d
                n_tot += 1
                pt_acc[pt][0] += d
                pt_acc[pt][1] += 1
        if n_tot == 0:
            continue
        out[key] = {
            'raw_miss': total / n_tot,
            'n_pitches': n_tot,
            'pt_miss': {pt: (s / n, n) for pt, (s, n) in pt_acc.items()},
        }
    return out


def normalize(results, pool_filter):
    """Adds 'commandPlus' on the "+" contract: 200 - 100*miss/lg_miss.

    100 is league average and one point is one percent, the way wRC+ reads:
    a 115 misses by 15% less distance than the league does. Before 2026-08-18
    this was 100 + 10*(lg - miss)/sigma, a standard-deviation ruler, so 115
    meant "1.5 SD better" and a reader applying the wRC+ reading got a wrong
    answer.

    Why a self-ratio rather than a run-denominated scale, which is what
    Stuff+/Loc+/Pitching+ would need: command has almost no same-season run
    signal. Measured over 2021-2026 in
    scripts/research/misc/pitcher_plus_scale_atlas.py, r against runs
    prevented ran +0.019 to +0.229 (mean +0.103), and forcing slope 1 in 2023
    would have produced a 99-to-101 dead column. But raw_miss is average miss
    distance in inches — strictly positive with a real zero at perfect
    command — so the ratio is meaningful on its own terms. That scale is also
    the most stable thing in the study: SD 5.4-6.0 and a range near 83-118 in
    every one of the six seasons. Same resolution CT+ reached on the hitter
    side.

    (lg, sigma) still come from qualified MLB pool rows (pool_filter(key) True
    and n_pitches >= MIN_POOL). sigma is no longer used to scale, and is
    returned only because callers log it. ROC keys are scored against the MLB
    lg, never pooled into it. Mutates and returns results plus the anchors.
    """
    pool = [v['raw_miss'] for k, v in results.items()
            if pool_filter(k) and v['n_pitches'] >= MIN_POOL]
    if len(pool) < 10:
        for v in results.values():
            v['commandPlus'] = None
        return results, (None, None)
    mu = sum(pool) / len(pool)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in pool) / len(pool))
    for v in results.values():
        # Guard on the DENOMINATOR now, not the spread: a zero-spread pool is
        # fine for a ratio, a zero league miss is not (and cannot happen with
        # real data, since a miss distance is strictly positive).
        if mu > 1e-9:
            v['commandPlus'] = round(200.0 - 100.0 * v['raw_miss'] / mu, 1)
        else:
            v['commandPlus'] = None
    return results, (mu, sigma)
