"""xwOBA3D — EV × LA × spray × bats lookup table of SAVANT MODEL xwOBA.

Shipped role (the only role): imputing per-pitch xwOBA for ROC/AAA balls in
play, where Savant's estimated_woba_using_speedangle is absent. Validated
held-out at r=0.915 per BIP for that purpose (see process_data.py wiring).

IMPORTANT SCOPE LIMIT: the cell value is the mean of Savant's
estimated_woba_using_speedangle — a model output that is a function of
EV × LA only — NOT actual outcome wOBA. Savant xwOBA is constant in spray
given EV/LA, so the spray dimension here only captures within-bin EV/LA
composition differences; the table CANNOT express real spray interactions
(a pulled vs oppo 95 mph / 20° liner having different league value). Do
NOT promote this into BB+ or use it as an xwOBAsp replacement without
first rebuilding the cell values on outcome wOBA (_bip_woba_value, the way
the SACQ table in process_data.py does).

Builds the table from league BIP data with hierarchical Bayesian shrinkage
toward 2D marginals for thin cells.
"""
import math
from collections import defaultdict

from pipeline.utils import (
    safe_float, BUNT_BB_TYPES, spray_angle, spray_direction,
)

# ── Bucket boundaries ───────────────────────────────────────────────────
# Coarse bins for stability. With ~18k BIP early in season, these give
# ~100 obs/cell on average across the ~288 cells.
EV_BINS = [(0, 70), (70, 80), (80, 88), (88, 95), (95, 102), (102, 200)]
LA_BINS = [(-90, -10), (-10, 0), (0, 10), (10, 20), (20, 30), (30, 40), (40, 91)]  # 91: _bucket is half-open, so LA=90.0 exactly must not fall out the top
SPRAY_DIRS = ['pull', 'pull_side', 'center_pull', 'center_oppo', 'oppo_side', 'oppo']
# Per-pitch Bats is always the side batted from ('L'/'R'; the scraper writes
# per-PA batSide). 'S' is deliberately NOT accepted: spray_direction() would
# treat it as LHB and mirror pull/oppo wrongly for a switch hitter batting R.
HANDS = ['R', 'L']

# Hierarchical shrinkage pseudo-count toward 2D marginals.
CELL_SHRINK_K = 20


# ── Classification ──────────────────────────────────────────────────────

def _bucket(v, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= v < hi:
            return i
    return None


def classify_bip(p):
    """Return (ev_bin, la_bin, spray_direction, bats) or None."""
    if p.get('Description') != 'In Play':
        return None
    if p.get('BBType') in BUNT_BB_TYPES:
        return None
    ev = safe_float(p.get('ExitVelo'))
    la = safe_float(p.get('LaunchAngle'))
    hcx = safe_float(p.get('HC_X'))
    hcy = safe_float(p.get('HC_Y'))
    bats = p.get('Bats')
    if any(v is None for v in (ev, la, hcx, hcy, bats)):
        return None
    if bats not in HANDS:
        return None
    angle = spray_angle(hcx, hcy)
    direction = spray_direction(angle, bats)
    if not direction:
        return None
    evi = _bucket(ev, EV_BINS)
    lai = _bucket(la, LA_BINS)
    if evi is None or lai is None:
        return None
    return (evi, lai, direction, bats)


# ── Weight table ────────────────────────────────────────────────────────

def build_xwoba3d_table(bip_pitches):
    """Raw cell means: dict[(evi, lai, spray, bats)] → (mean_xwoba, n)."""
    cells = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    for p in bip_pitches:
        key = classify_bip(p)
        if key is None:
            continue
        xw = safe_float(p.get('xwOBA'))
        if xw is None:
            continue
        cells[key]['sum'] += xw
        cells[key]['n'] += 1
    return {k: (v['sum'] / v['n'], v['n']) for k, v in cells.items() if v['n'] > 0}


def _two_d_marginals(bip_pitches):
    """Compute the three 2D marginals used as shrinkage priors:
      ev_la_bats   = mean xwOBA by (evi, lai, bats)
      la_sp_bats   = mean xwOBA by (lai, spray, bats)
      ev_sp_bats   = mean xwOBA by (evi, spray, bats)
    plus the bats-level grand mean as an ultimate fallback.
    """
    ev_la = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    la_sp = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    ev_sp = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    by_bats = defaultdict(lambda: {'sum': 0.0, 'n': 0})
    overall = {'sum': 0.0, 'n': 0}
    for p in bip_pitches:
        key = classify_bip(p)
        if key is None:
            continue
        xw = safe_float(p.get('xwOBA'))
        if xw is None:
            continue
        evi, lai, sp, bats = key
        for k, d in [((evi, lai, bats), ev_la),
                     ((lai, sp, bats),   la_sp),
                     ((evi, sp, bats),   ev_sp),
                     (bats,              by_bats)]:
            d[k]['sum'] += xw
            d[k]['n']   += 1
        overall['sum'] += xw
        overall['n']   += 1

    def flat(d):
        return {k: v['sum'] / v['n'] for k, v in d.items() if v['n'] > 0}

    return {
        'ev_la':   flat(ev_la),
        'la_sp':   flat(la_sp),
        'ev_sp':   flat(ev_sp),
        'by_bats': flat(by_bats),
        'global':  overall['sum'] / overall['n'] if overall['n'] else 0.33,
    }


def shrink_xwoba3d(raw_table, bip_pitches, k=CELL_SHRINK_K):
    """Shrink each 3D cell toward the mean of its 2D marginals.
    Missing cells fall back through marginals → bats avg → global avg.
    Returns every possible cell (fully-populated table)."""
    mg = _two_d_marginals(bip_pitches)
    smoothed = {}
    for evi in range(len(EV_BINS)):
        for lai in range(len(LA_BINS)):
            for sp in SPRAY_DIRS:
                for bats in HANDS:
                    key = (evi, lai, sp, bats)
                    cell_mean, n = raw_table.get(key, (None, 0))
                    # Build prior from 2D marginals (average of three if available).
                    priors = []
                    for m_key, m_dict in [((evi, lai, bats), 'ev_la'),
                                          ((lai, sp,  bats), 'la_sp'),
                                          ((evi, sp,  bats), 'ev_sp')]:
                        v = mg[m_dict].get(m_key)
                        if v is not None:
                            priors.append(v)
                    if priors:
                        prior = sum(priors) / len(priors)
                    else:
                        prior = mg['by_bats'].get(bats, mg['global'])
                    if cell_mean is None:
                        smoothed[key] = (prior, 0)
                    else:
                        rv = (n * cell_mean + k * prior) / (n + k)
                        smoothed[key] = (rv, n)
    return smoothed


# ── Packaging ───────────────────────────────────────────────────────────
# (Per-hitter aggregation helpers were removed 2026-07-02: never wired into
# the pipeline, and a hitter-level xwOBA3D built on model-xwOBA cells would
# be nearly degenerate with xwOBAcon — see the module docstring.)

def serialize_table(smoothed):
    """For metadata output."""
    out = {}
    for key, (rv, n) in smoothed.items():
        evi, lai, sp, bats = key
        out[f"ev{evi}|la{lai}|{sp}|{bats}"] = {'rv': round(rv, 5), 'n': n}
    return out
