#!/usr/bin/env python3
"""pitcherplus_scale_check.py — can Pitcher+ honour the "+" contract?

Companion to pitcher_plus_scale_atlas.py, which could not cover Pitcher+
because it is a six-component composite rather than a single cached raw.

THE CIRCULARITY THAT DECIDES THIS. xRv100 is a COMPONENT of Pitcher+ at
weight 0.23, so a same-season correlation against any run measure is partly
Pitcher+ correlating with itself. pitcherplus.py already says so, and it is
why pitcherRuns100 uses a FROZEN PREDICTIVE slope (0.039 runs/100 per point,
fit on future-season xRV/100) instead of a same-season OLS.

So the contract question for Pitcher+ is NOT "what does a point buy this
season" — that number is inflated. It is "what does a point buy NEXT season",
which is the claim the metric actually makes.

This measures both and reports them side by side, so the inflation is visible
rather than assumed.

Reconstruction (2021-2025), from cached artifacts only:
    stuffScore  <- _era_internal_stuff  stuff_full   (z, higher better)
    locPlus     <- _era_internal_cmdloc loc_full     (z, LOWER better)
    kPct        <- _era_battery         k_pct
    izWhiffPct  <- _era_battery         1 - zcon_pct
    xRv100      <- _era_xrv100          full
    gbPct       <- _era_battery         gb_pct
with the shipped weights and stabilization constants read from
pipeline.pitcherplus.COMPONENTS, and Pitcher+ = 100 + SCALE_K * z(composite).

CAVEAT, stated up front: there is no shipped multi-season pitcher leaderboard
to validate this reconstruction against, so unlike the Command+ check in the
atlas (r = 0.976 vs shipped) this one is UNVALIDATED. Treat the slopes as
indicative, not settled.

Usage: python3 scripts/research/misc/pitcherplus_scale_check.py [--min-ip 60]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

from pipeline.pitcherplus import COMPONENTS, SCALE_K, QUAL_N

MIN_IP = 60.0
for i, a in enumerate(sys.argv):
    if a == '--min-ip' and i + 1 < len(sys.argv):
        MIN_IP = float(sys.argv[i + 1])


def load(n):
    with open(os.path.join(DATA, n)) as f:
        return json.load(f)


def psd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def lin(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def main():
    targets = load('_era_targets.json')
    battery = load('_era_battery.json')
    stuff = load('_era_internal_stuff.json')
    cmdloc = load('_era_internal_cmdloc.json')
    xrv = load('_era_xrv100.json')

    print(f"Pitcher+ scale check — min {MIN_IP:.0f} IP")
    print("Components (weight, stabilization) read from pipeline.pitcherplus:")
    for name, w, k in COMPONENTS:
        print(f"    {name:12} w={w:.2f}  k={k:.0f}")
    print()

    # ── FIP+ per season (same currency as the atlas) ──
    fipplus = {}
    for y in sorted(targets):
        rows = {}
        pit = targets[y]['pitchers']
        tot = [0.0, 0.0, 0.0, 0.0, 0.0]  # ip, er, hr, bbhbp, so
        keep = {}
        for pid, t in pit.items():
            ip = (t.get('outs') or 0) / 3.0
            if ip < MIN_IP:
                continue
            bb = (t.get('bb') or 0) - (t.get('ibb') or 0)
            keep[pid] = (ip, t.get('er') or 0, t.get('hr') or 0,
                         bb + (t.get('hbp') or 0), t.get('so') or 0)
            for i, v in enumerate(keep[pid]):
                tot[i] += v
        if len(keep) < 40:
            continue
        lg_era = tot[1] * 9.0 / tot[0]
        core = (13.0 * tot[2] + 3.0 * tot[3] - 2.0 * tot[4]) / tot[0]
        cfip = lg_era - core
        f = {}
        for pid, (ip, er, hr, bbh, so) in keep.items():
            f[pid] = ((13.0 * hr + 3.0 * bbh - 2.0 * so) / ip) + cfip
        lgf = sum(f[p] * keep[p][0] for p in f) / tot[0]
        fipplus[y] = {p: 200.0 - 100.0 * f[p] / lgf for p in f}

    # ── Reconstruct Pitcher+ per season ──
    pplus = {}
    for y in sorted(battery):
        bat, st, cl, xr = battery[y], stuff.get(y, {}), cmdloc.get(y, {}), xrv.get(y, {})
        if not st or not cl or not xr:
            print(f"{y}: components missing from cache — skipped")
            continue
        raw = {}
        for pid, b in bat.items():
            full = b.get('full') or {}
            s, c, x = st.get(pid), cl.get(pid), xr.get(pid)
            if not (s and c and x):
                continue
            if full.get('zcon_pct') is None or full.get('k_pct') is None:
                continue
            raw[pid] = {
                'stuffScore': s.get('stuff_full'),
                'locPlus': (-c['loc_full'] if c.get('loc_full') is not None else None),
                'kPct': full.get('k_pct'),
                'izWhiffPct': 1.0 - full['zcon_pct'],
                'xRv100': x.get('full'),
                'gbPct': full.get('gb_pct'),
                'pitches': full.get('pitches') or 0,
            }
        pool = {p: v for p, v in raw.items()
                if v['pitches'] >= QUAL_N and all(v[c] is not None for c, _, _ in COMPONENTS)}
        if len(pool) < 50:
            print(f"{y}: pool {len(pool)} too small — skipped")
            continue
        anch = {}
        for name, _w, _k in COMPONENTS:
            vals = [pool[p][name] for p in pool]
            anch[name] = (sum(vals) / len(vals), psd(vals))
        comp = {}
        for p, v in raw.items():
            if any(v[c] is None for c, _, _ in COMPONENTS):
                continue
            tot = 0.0
            for name, w, k in COMPONENTS:
                mu, sd = anch[name]
                if sd <= 0:
                    continue
                z = (v[name] - mu) / sd
                n = v['pitches']
                tot += w * z * (n / (n + k))
            comp[p] = tot
        cv = [comp[p] for p in comp if p in pool]
        cmu, csd = sum(cv) / len(cv), psd(cv)
        pplus[y] = {p: 100.0 + SCALE_K * (c - cmu) / csd for p, c in comp.items()}

    yrs = sorted(pplus)
    print("SAME-SEASON vs NEXT-SEASON, against runs prevented (FIP+)")
    print("The same-season column is INFLATED: xRv100 is 23% of Pitcher+.\n")
    print(f"{'yr':6} {'n':>4} {'SD':>6} {'min':>6} {'max':>6} | "
          f"{'r same':>7} {'slope':>6} | {'r next':>7} {'slope':>6} {'n next':>6}")
    same_r, next_r, next_sl = [], [], []
    for y in yrs:
        pp, fp = pplus[y], fipplus.get(y, {})
        ids = [p for p in pp if p in fp]
        if len(ids) < 40:
            continue
        v = [pp[p] for p in ids]; f = [fp[p] for p in ids]
        s1, r1 = lin(v, f)
        ny = str(int(y) + 1)
        fp2 = fipplus.get(ny, {})
        ids2 = [p for p in pp if p in fp2]
        if len(ids2) >= 40:
            v2 = [pp[p] for p in ids2]; f2 = [fp2[p] for p in ids2]
            s2, r2 = lin(v2, f2)
            next_r.append(r2); next_sl.append((s2, psd(v2), r2, psd(f2)))
        else:
            s2 = r2 = None
        same_r.append(r1)
        print(f"{y:6} {len(ids):>4} {psd(v):>6.2f} {min(v):>6.1f} {max(v):>6.1f} | "
              f"{r1:>+7.3f} {s1:>6.2f} | "
              f"{(('%+.3f' % r2) if r2 is not None else '-'):>7} "
              f"{(('%.2f' % s2) if s2 is not None else '-'):>6} "
              f"{(len(ids2) if r2 is not None else 0):>6}")

    if same_r:
        print(f"\n  same-season r: mean {sum(same_r)/len(same_r):+.3f} "
              f"(inflated by the xRv100 component)")
    if next_r:
        print(f"  NEXT-season r: mean {sum(next_r)/len(next_r):+.3f}, "
              f"range {min(next_r):+.3f} to {max(next_r):+.3f}, "
              f"spread {max(next_r)-min(next_r):.3f}   <- the honest number")
        print("\nRANGE if Pitcher+ were calibrated to slope 1 on the HONEST")
        print("(next-season) relationship:")
        print(f"  {'yr':6} {'SD now':>7} {'SD needed':>10} {'implied range':>16}")
        for y, (s2, sdv, r2, sdf) in zip(yrs, next_sl):
            need = abs(r2) * sdf
            fac = need / sdv
            lo, hi = 100 - 3.0 * need, 100 + 3.0 * need
            print(f"  {y:6} {sdv:>7.2f} {need:>10.2f} "
                  f"{('%.0f to %.0f (+/-3SD)' % (lo, hi)):>16}")

    print("\nNOTE: unvalidated. No shipped multi-season pitcher leaderboard")
    print("exists to check this reconstruction against.")


if __name__ == '__main__':
    main()
