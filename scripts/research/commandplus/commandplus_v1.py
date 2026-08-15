"""commandplus_v1.py — Command+ research engine (APPROVED design, 2026-07-28).

Did he put the pitch where he meant to? Loc+ grades the VALUE of locations;
Command+ grades EXECUTION: distance from inferred intent. The two are known
to be distinct axes (r ~ -0.46, command stickier year-over-year).

DESIGN (per Wally's sign-off):
  cell        (pitcher, pitch type, batter hand, count group)
  count groups  EVEN  b<=s, s<2   (0-0, 0-1, 1-1)        — establish intent
                PUTAWAY  s==2     (0-2, 1-2, 2-2, 3-2)   — bury/put-away intent
                BEHIND  b>s, s<2  (1-0, 2-0, 3-0, 2-1, 3-1) — must-strike intent
  targets     GMM over plate locations in PHYSICAL INCHES, K=1..3 by BIC,
              K capped by cell size; component means = inferred targets
  misses      Euclidean inches to the NEAREST target
  aggregates  mean miss (pooled over pitches) AND median miss — both reported
              per Wally; the battery decides which ships
  scope       MLB + ROC/AAA. Command is self-referential (a pitcher's misses
              vs his own targets), so ROC needs no MLB baseline — unlike
              Loc+/Stuff+. The NORMALIZATION pool stays MLB-only per the site
              convention; ROC is scored and excluded from (mu, sigma).

CIRCULARITY GUARDS (the design's load-bearing wall — a wild pitcher's own
scatter must not "discover" targets that flatter him):
  1. misses in fixed physical inches to component MEANS — never Mahalanobis
     with the pitcher's own covariance, which would absorb the spray
  2. minimum target separation SEP_MIN: components whose means sit closer
     than this merge (weighted) — a diffuse cloud cannot split itself into
     two "targets" to halve its misses. DEFAULT 8in IS A PLACEHOLDER: the
     battery sweeps it (never-estimate rule); nothing ships on the default.
  3. K capped by sample: 1 under 30 pitches, 2 under 60, 3 above.

Research code: numpy + scikit-learn allowed (CI already carries both). The
pipeline port will be pure-Python EM per house rules.
"""
import math
from collections import defaultdict

import numpy as np
from sklearn.mixture import GaussianMixture

SEP_MIN_DEFAULT = 8.0        # inches — PLACEHOLDER, swept in the battery
MIN_CELL = 20                # min pitches to fit a cell's targets
K_CAPS = [(60, 3), (30, 2), (0, 1)]   # (min pitches, max K)

EXCLUDE_DESC = {'Hit By Pitch', 'Foul Bunt', 'Missed Bunt', 'Bunt Foul Tip',
                'Pitchout', 'Swinging Pitchout', 'Foul Pitchout', 'Intent Ball'}
EXCLUDE_PT = {'EP', 'PO'}
HANDS = ('L', 'R')


def count_group(count_str):
    """EVEN / PUTAWAY / BEHIND per the approved grouping. None if unparseable."""
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


def _sf(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def eligible(p):
    """Competitive pitch with a location and a full cell key."""
    if p.get('Description') in EXCLUDE_DESC:
        return False
    if p.get('Event') == 'Intent Walk':
        return False
    pt = p.get('Pitch Type')
    if not pt or pt in EXCLUDE_PT:
        return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS:
        return False
    if _sf(p.get('PlateX')) is None or _sf(p.get('PlateZ')) is None:
        return False
    return count_group(p.get('Count')) is not None


def fit_targets(xy, sep_min=SEP_MIN_DEFAULT, seed=0):
    """Infer intended targets for one cell.

    xy: (n, 2) array of plate locations in INCHES.
    Returns (targets, k_used): targets is (k, 2). BIC selects K within the
    sample cap; components closer than sep_min merge (weight-averaged), and
    merging cascades until all pairs clear the separation.
    """
    n = len(xy)
    kmax = next(k for thr, k in K_CAPS if n >= thr)
    best, best_bic = None, None
    for k in range(1, kmax + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, n_init=2, reg_covar=1e-3)
        gm.fit(xy)
        bic = gm.bic(xy)
        if best_bic is None or bic < best_bic:
            best, best_bic = gm, bic
    means = best.means_.copy()
    weights = best.weights_.copy()
    # cascade-merge components violating the separation guard
    changed = True
    while changed and len(means) > 1:
        changed = False
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                if np.linalg.norm(means[i] - means[j]) < sep_min:
                    w = weights[i] + weights[j]
                    means[i] = (weights[i] * means[i] + weights[j] * means[j]) / w
                    weights[i] = w
                    means = np.delete(means, j, axis=0)
                    weights = np.delete(weights, j)
                    changed = True
                    break
            if changed:
                break
    return means, len(means)


def _fit_gmm(xy, kmax, seed=0):
    """BIC-selected GMM, separation-agnostic (the merge guard is applied
    afterward, per sep value — fitting is the expensive step and does not
    depend on SEP_MIN, which is what makes the battery's sweep cheap)."""
    best, best_bic = None, None
    for k in range(1, kmax + 1):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, n_init=2, reg_covar=1e-3)
        gm.fit(xy)
        bic = gm.bic(xy)
        if best_bic is None or bic < best_bic:
            best, best_bic = gm, bic
    return best.means_.copy(), best.weights_.copy()


def _merge(means, weights, sep_min):
    means = means.copy(); weights = weights.copy()
    changed = True
    while changed and len(means) > 1:
        changed = False
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                if np.linalg.norm(means[i] - means[j]) < sep_min:
                    w = weights[i] + weights[j]
                    means[i] = (weights[i] * means[i] + weights[j] * means[j]) / w
                    weights[i] = w
                    means = np.delete(means, j, axis=0)
                    weights = np.delete(weights, j)
                    changed = True
                    break
            if changed:
                break
    return means, weights


def score_pitches_multi(pitches, sep_list, min_cell=MIN_CELL, seed=0):
    """Fit every cell ONCE, then score misses at each separation value.
    Returns {sep: (misses, pt_misses)} with the same shapes as score_pitches.
    """
    cells = defaultdict(list)
    for p in pitches:
        if not eligible(p):
            continue
        key = (p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'),
               p.get('Bats'), count_group(p.get('Count')))
        cells[key].append((_sf(p.get('PlateX')) * 12.0, _sf(p.get('PlateZ')) * 12.0))
    out = {s: (defaultdict(list), defaultdict(list)) for s in sep_list}
    for key, pts in cells.items():
        if len(pts) < min_cell:
            continue
        xy = np.asarray(pts)
        kmax = next(k for thr, k in K_CAPS if len(pts) >= thr)
        means, weights = _fit_gmm(xy, kmax, seed)
        pitcher, throws, pt = key[0], key[1], key[2]
        for s in sep_list:
            tg, _w = _merge(means, weights, s) if s > 0 else (means, weights)
            d = np.linalg.norm(xy[:, None, :] - tg[None, :, :], axis=2).min(axis=1)
            out[s][0][(pitcher, throws)].extend(d.tolist())
            out[s][1][(pitcher, throws, pt)].extend(d.tolist())
    return out


def score_pitches(pitches, sep_min=SEP_MIN_DEFAULT, min_cell=MIN_CELL, seed=0):
    """Fit targets per cell and score every pitch's miss.

    Returns:
      misses:    dict[(pitcher, throws)] -> list of per-pitch miss inches
      pt_misses: dict[(pitcher, throws, pitch type)] -> list of miss inches
      cells_fit: number of cells fitted
    """
    cells = defaultdict(list)
    for p in pitches:
        if not eligible(p):
            continue
        key = (p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'),
               p.get('Bats'), count_group(p.get('Count')))
        x = _sf(p.get('PlateX')) * 12.0
        z = _sf(p.get('PlateZ')) * 12.0
        cells[key].append((x, z))

    misses = defaultdict(list)
    pt_misses = defaultdict(list)
    n_fit = 0
    for key, pts in cells.items():
        if len(pts) < min_cell:
            continue
        xy = np.asarray(pts)
        targets, _k = fit_targets(xy, sep_min=sep_min, seed=seed)
        d = np.linalg.norm(xy[:, None, :] - targets[None, :, :], axis=2).min(axis=1)
        pitcher, throws, pt = key[0], key[1], key[2]
        misses[(pitcher, throws)].extend(d.tolist())
        pt_misses[(pitcher, throws, pt)].extend(d.tolist())
        n_fit += 1
    return misses, pt_misses, n_fit


def aggregate(miss_dict, min_n=1):
    """(mean, median, n) per key — both variants carried until the battery
    picks one."""
    out = {}
    for k, v in miss_dict.items():
        if len(v) >= min_n:
            a = np.asarray(v)
            out[k] = (float(a.mean()), float(np.median(a)), len(a))
    return out
