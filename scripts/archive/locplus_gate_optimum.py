"""locplus_gate_optimum.py — find the OPTIMAL pitch-type Loc+ coloring gate.

The r>=0.5 floor shipped in b34ad89 was a convention, not a measured optimum.
This finds the optimum by defining what a colored cell PROMISES the reader.

WHAT A COLOR MEANS. A colored Loc+ cell tells the reader "this pitch's location
quality ranks HERE among its peers." The cell is honest if the percentile the
reader sees is close to the pitcher's true percentile, and misleading if it
lands in a different quartile band than the truth. So:

    materially wrong  <=>  |displayed_pctl - true_pctl| > 25 points

MEASUREMENT (no distributional assumptions in step 1). Split each cell's
pitches by odd/even game date and rebuild surfaces per half, so the two halves
are INDEPENDENT estimates. Truncate both halves to exactly n pitches. Then

    corr(halfA_n, halfB_n)  ==  rho(n), the reliability of an n-pitch estimate

directly — no Spearman-Brown needed, because both sides are n-pitch estimates
of the same quantity. Sweeping n traces the reliability curve empirically and
lets us check the n/(n+k) true-score model against measured k.

FROM RELIABILITY TO HONESTY. With reliability rho, a displayed estimate D and
the latent truth T satisfy corr(D,T) = sqrt(rho). Under a bivariate normal the
percentile error distribution follows, and P(|pctl_D - pctl_T| <= 25) can be
computed exactly. Step 2 VALIDATES that normal model against the observed
A-vs-B percentile error (whose predicted correlation is rho, not sqrt(rho)),
so the extrapolation to truth is checked rather than assumed.

THE OPTIMUM. Raising the gate buys honesty and costs coverage. The exchange
rate is not free — but it does not have to be assumed, because the MARGINAL
cell settles it: admit a cell only while it is more likely to be honestly
ranked than materially misranked, i.e.

    n* = the n where P(|pctl_D - pctl_T| <= 25) crosses 0.50

Below n*, adding cells adds more misinformation than information, so any
sensible loss function rejects them; above n*, each admitted cell is net
informative. That is a genuine interior optimum, not a taste call.

Usage: python3 scripts/locplus_gate_optimum.py
"""
import os, sys, math, pickle, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp

PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3169, 1.2393

N_GRID = [20, 25, 30, 40, 50, 60, 71, 85, 100, 117, 135, 150, 175, 200, 250]
QUARTILE_BAND = 25.0        # percentile points; "materially wrong" beyond this
MIN_CELLS = 40              # don't report a correlation from a thin cell pool


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


# ── bivariate-normal percentile agreement ───────────────────────────────
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_phi(p):
    # Acklam's inverse normal CDF (plenty accurate here)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def p_within_band(rho_dt, band=QUARTILE_BAND, grid=400):
    """P(|pctl(D) - pctl(T)| <= band) for standard bivariate normal (D,T) with
    correlation rho_dt. Integrates over the marginal of D."""
    if rho_dt <= 0:
        return 2 * (band / 100.0) - (band / 100.0) ** 2   # independent case
    if rho_dt >= 0.9999:
        return 1.0
    s = math.sqrt(max(1e-12, 1 - rho_dt * rho_dt))
    tot = 0.0
    for i in range(grid):
        u = (i + 0.5) / grid                 # percentile of D, uniform
        zd = _inv_phi(u)
        lo_u = max(0.0, u - band / 100.0); hi_u = min(1.0, u + band / 100.0)
        # T | D=zd  ~  N(rho*zd, s^2); convert percentile bounds on T to z
        lo_z = _inv_phi(lo_u) if lo_u > 0 else -40.0
        hi_z = _inv_phi(hi_u) if hi_u < 1 else 40.0
        tot += _phi((hi_z - rho_dt * zd) / s) - _phi((lo_z - rho_dt * zd) / s)
    return tot / grid


def pctl_ranks(vals):
    """Percentile rank (0-100) of each value within the list."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    n = len(vals)
    for rank, i in enumerate(order):
        out[i] = 100.0 * rank / (n - 1) if n > 1 else 50.0
    return out


def main():
    print("loading cache...", file=sys.stderr)
    D = pickle.load(open(PKL, 'rb'))
    base = [p for p in D if lp.is_eligible_baseline(p)]

    dates = sorted({p.get('Game Date') for p in base if p.get('Game Date')})
    parity = {d: i % 2 for i, d in enumerate(dates)}

    print("building per-half surfaces...", file=sys.stderr)
    halves_pitches = [[p for p in base if parity.get(p.get('Game Date')) == h]
                      for h in (0, 1)]
    S = [lp.build_surfaces(hp, LG, SCALE) for hp in halves_pitches]

    # Per-half, per-CELL score lists. Cell = (pitcher, throws, pitch type) —
    # the leaderboard's pitch-type row. Order preserved so truncating to the
    # first n pitches is a clean fixed-n estimate.
    cells = [defaultdict(list), defaultdict(list)]
    for h in (0, 1):
        for p in halves_pitches[h]:
            v = lp.score_pitch(p, S[h])
            if v is None:
                continue
            cells[h][(p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'))].append(v)
    print(f"cells: half0={len(cells[0])} half1={len(cells[1])}", file=sys.stderr)

    groups = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH']
    k_measured = lp.STABILIZE_N_PT

    # ── STEP 1: empirical reliability rho(n) = corr of two independent n-pitch estimates
    print()
    print("STEP 1 — empirical reliability of an n-pitch cell estimate")
    print("(corr of two INDEPENDENT n-pitch estimates; no Spearman-Brown needed)")
    print()
    hdr = f"{'n':>5s} " + ''.join(f"{g:>13s}" for g in groups) + f"{'ALL':>13s}"
    print(hdr)
    print('-' * len(hdr))

    rho_obs = defaultdict(dict)     # group -> n -> rho
    pctl_err_obs = defaultdict(dict)  # group -> n -> P(|pctlA-pctlB| <= band)
    for n in N_GRID:
        line = f"{n:>5d} "
        for g in groups + ['ALL']:
            pool_a, pool_b = [], []
            for key in cells[0]:
                if key not in cells[1]:
                    continue
                pt = key[2]
                grp = lp.GROUP.get(pt)
                if g != 'ALL' and grp != g:
                    continue
                if g == 'ALL' and grp is None:
                    continue
                a, b = cells[0][key], cells[1][key]
                if len(a) < n or len(b) < n:
                    continue
                pool_a.append(sum(a[:n]) / n)
                pool_b.append(sum(b[:n]) / n)
            if len(pool_a) < MIN_CELLS:
                line += f"{'  -':>13s}"
                continue
            r = pearson(pool_a, pool_b)
            rho_obs[g][n] = (r, len(pool_a))
            # observed percentile agreement between the two halves
            pa, pb = pctl_ranks(pool_a), pctl_ranks(pool_b)
            within = sum(1 for x, y in zip(pa, pb) if abs(x - y) <= QUARTILE_BAND)
            pctl_err_obs[g][n] = within / len(pa)
            line += f"{r:>8.3f}({len(pool_a):>3d})"
        print(line)

    # ── STEP 2: validate the n/(n+k) model and the normal percentile model
    print()
    print("STEP 2 — model validation")
    print()
    print(f"{'group':>6s} {'k_shipped':>10s} {'k_refit':>8s} {'rms_rho':>8s} "
          f"{'rms_pctl':>9s}   (pctl model = bivariate normal at corr=rho)")
    print('-' * 78)
    k_refit = {}
    for g in groups + ['ALL']:
        if not rho_obs[g]:
            continue
        # refit k by least squares on rho(n) = n/(n+k)  ->  k = n(1-rho)/rho
        ks = [n * (1 - r) / r for n, (r, _c) in rho_obs[g].items() if r and r > 0]
        if not ks:
            continue
        k_fit = statistics.median(ks)
        k_refit[g] = k_fit
        rms_r = math.sqrt(statistics.mean(
            [(r - n / (n + k_fit)) ** 2 for n, (r, _c) in rho_obs[g].items() if r]))
        # normal model predicts A-vs-B percentile agreement at corr = rho itself
        errs = []
        for n, obs in pctl_err_obs[g].items():
            r = rho_obs[g][n][0]
            if r is None:
                continue
            errs.append((obs - p_within_band(r)) ** 2)
        rms_p = math.sqrt(statistics.mean(errs)) if errs else float('nan')
        shipped = k_measured.get(g, lp.STABILIZE_N_OVERALL if g == 'ALL' else None)
        print(f"{g:>6s} {str(shipped):>10s} {k_fit:>8.0f} {rms_r:>8.3f} {rms_p:>9.3f}")

    # ── STEP 3: the optimum — P(honest) vs n, crossing 0.50
    print()
    print("STEP 3 — honesty of a COLORED cell vs the latent truth")
    print("P_honest(n) = P(|displayed pctl - true pctl| <= 25 pts), corr(D,T)=sqrt(rho)")
    print("n* = where P_honest crosses 0.50: below it a cell is more likely")
    print("materially misranked than honestly ranked.")
    print()
    print(f"{'group':>6s} {'k_refit':>8s} {'r>=0.5 gate':>12s} {'P_honest@gate':>14s} "
          f"{'n*(P=0.50)':>11s} {'n*(P=0.60)':>11s} {'n*(P=0.667)':>12s}")
    print('-' * 82)

    def n_for_p(k, target):
        lo, hi = 1.0, 5000.0
        for _ in range(80):
            mid = (lo + hi) / 2
            rho = mid / (mid + k)
            if p_within_band(math.sqrt(rho)) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    rows_out = {}
    for g in groups + ['ALL']:
        if g not in k_refit:
            continue
        k = k_refit[g]
        gate = k_measured.get(g, lp.STABILIZE_N_OVERALL if g == 'ALL' else k)
        p_at_gate = p_within_band(math.sqrt(gate / (gate + k)))
        n50 = n_for_p(k, 0.50); n60 = n_for_p(k, 0.60); n67 = n_for_p(k, 2.0 / 3)
        rows_out[g] = (k, gate, p_at_gate, n50, n60, n67)
        print(f"{g:>6s} {k:>8.0f} {gate:>12.0f} {p_at_gate:>14.3f} "
              f"{n50:>11.0f} {n60:>11.0f} {n67:>12.0f}")

    # ── STEP 4: coverage cost of each candidate floor
    print()
    print("STEP 4 — coverage (full-season cells, both halves pooled)")
    full = defaultdict(int)
    for p in base:
        if lp._is_scorable(p):
            full[(p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'))] += 1
    tot_25 = sum(1 for k_, v in full.items() if v >= 25)
    print(f"cells with >=25 pitches (colored under the OLD flat gate): {tot_25}")
    print()
    print(f"{'policy':>26s} {'cells colored':>14s} {'vs old flat-25':>15s}")
    print('-' * 58)

    def coverage(fn):
        return sum(1 for key, v in full.items()
                   if v >= fn(lp.GROUP.get(key[2])))
    pol = {
        'flat 25 (old)': lambda g: 25,
        'r>=0.5 (shipped)': lambda g: k_measured.get(g, lp.STABILIZE_N_OVERALL),
        'P_honest>=0.50 (n*)': lambda g: rows_out.get(g, rows_out['ALL'])[3],
        'P_honest>=0.60': lambda g: rows_out.get(g, rows_out['ALL'])[4],
        'P_honest>=0.667': lambda g: rows_out.get(g, rows_out['ALL'])[5],
    }
    for name, fn in pol.items():
        c = coverage(fn)
        print(f"{name:>26s} {c:>14d} {100.0 * c / tot_25:>14.1f}%")

    print()
    print("Per-group gate under each policy:")
    print(f"{'group':>6s} {'r>=0.5':>8s} {'P>=0.50':>9s} {'P>=0.60':>9s} {'P>=0.667':>10s}")
    for g in groups:
        if g not in rows_out:
            continue
        k, gate, _p, n50, n60, n67 = rows_out[g]
        print(f"{g:>6s} {gate:>8.0f} {n50:>9.0f} {n60:>9.0f} {n67:>10.0f}")


if __name__ == '__main__':
    main()
