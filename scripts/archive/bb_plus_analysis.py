"""BB+ composite weighting exploration using 2026 data.

Tests:
1. Correlation between xwOBAcon and xwOBAsp (orthogonality check)
2. 60/40 vs 50/50 weighting: distribution shape, extreme movers, rank correlation
"""
import json
import math
from collections import Counter


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs, ys):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    return pearson(rx, ry)


def summarize(name, vals):
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    def pct(p):
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return vals[i]
    print(f"  {name}: n={n}, mean={mean:.2f}, sd={sd:.2f}, "
          f"p05={pct(0.05):.1f}, p25={pct(0.25):.1f}, p50={pct(0.50):.1f}, "
          f"p75={pct(0.75):.1f}, p95={pct(0.95):.1f}, min={vals[0]:.1f}, max={vals[-1]:.1f}")


def main():
    with open('data/hitter_leaderboard_rs.json') as f:
        hitters = json.load(f)
    with open('data/metadata_rs.json') as f:
        md = json.load(f)

    lg_xwobacon = md['hitterLeagueAverages']['xwOBAcon']
    lg_xwobasp = md['hitterLeagueAverages']['xwOBAsp']
    print(f"League avg xwOBAcon = {lg_xwobacon:.4f}")
    print(f"League avg xwOBAsp  = {lg_xwobasp:.4f}")
    print(f"Total hitters in file: {len(hitters)}")

    # Multi-team rows exclusion check
    teams = Counter(h.get('team') for h in hitters)
    multi = {k: v for k, v in teams.items() if isinstance(k, str) and k.endswith('TM')}
    print(f"Multi-team rows present: {multi}")

    # Qualification: exclude multi-team aggregate rows (avoid including same player twice),
    # require non-null xwOBAcon, xwOBAsp, and a BIP floor.
    BIP_FLOOR = 40
    qualified = []
    for h in hitters:
        team = h.get('team') or ''
        if isinstance(team, str) and team.endswith('TM') and team != 'TM':
            continue
        if h.get('xwOBAcon') is None or h.get('xwOBAsp') is None:
            continue
        if (h.get('nBip') or 0) < BIP_FLOOR:
            continue
        qualified.append(h)
    print(f"Qualified hitters (>= {BIP_FLOOR} BIPs, single-team): {len(qualified)}")

    # ─── Test 1: correlation ──────────────────────────────────────────────
    print("\n=== TEST 1: Orthogonality of xwOBAcon and xwOBAsp ===")
    x_con = [h['xwOBAcon'] for h in qualified]
    x_sp = [h['xwOBAsp'] for h in qualified]
    r = pearson(x_con, x_sp)
    rho = spearman(x_con, x_sp)
    print(f"Pearson  r = {r:.3f}")
    print(f"Spearman ρ = {rho:.3f}")
    # Guidance from earlier msg: r<0.5 great, 0.5-0.7 OK, >0.8 redundant
    if r < 0.5:
        verdict = "LOW correlation — composite adds real signal"
    elif r < 0.7:
        verdict = "MODERATE correlation — composite likely adds signal but weights matter less"
    elif r < 0.85:
        verdict = "HIGH correlation — diminishing returns from blending"
    else:
        verdict = "VERY HIGH correlation — redundant, rethink the composite"
    print(f"Verdict: {verdict}")

    # ─── Test 2: 60/40 vs 50/50 ─────────────────────────────────────────────
    print("\n=== TEST 2: 60/40 vs 50/50 weighting ===")

    def bb_plus(h, w_con):
        con_plus = 100.0 * h['xwOBAcon'] / lg_xwobacon
        sp_plus = 100.0 * h['xwOBAsp'] / lg_xwobasp
        return w_con * con_plus + (1.0 - w_con) * sp_plus

    for h in qualified:
        h['_bb_plus_6040'] = bb_plus(h, 0.60)
        h['_bb_plus_5050'] = bb_plus(h, 0.50)
        h['_bb_plus_con_only'] = 100.0 * h['xwOBAcon'] / lg_xwobacon
        h['_bb_plus_sp_only'] = 100.0 * h['xwOBAsp'] / lg_xwobasp

    v6040 = [h['_bb_plus_6040'] for h in qualified]
    v5050 = [h['_bb_plus_5050'] for h in qualified]
    vcon = [h['_bb_plus_con_only'] for h in qualified]
    vsp = [h['_bb_plus_sp_only'] for h in qualified]

    print("\nDistribution (index, 100 = league avg):")
    summarize("BB+ 60/40  ", v6040)
    summarize("BB+ 50/50  ", v5050)
    summarize("xwOBAcon+  ", vcon)
    summarize("xwOBAsp+   ", vsp)

    # Rank correlation between 60/40 and 50/50 (if high, weighting barely matters for ordering)
    rho_weights = spearman(v6040, v5050)
    pearson_weights = pearson(v6040, v5050)
    print(f"\n60/40 vs 50/50: Pearson r = {pearson_weights:.4f}, Spearman ρ = {rho_weights:.4f}")

    # How many hitters move by more than X ranks between the two weightings?
    def rank_map(vals):
        # higher value = better, rank 1 = best
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        ranks = [0] * len(vals)
        for r_idx, orig in enumerate(order):
            ranks[orig] = r_idx + 1
        return ranks

    ranks_60 = rank_map(v6040)
    ranks_50 = rank_map(v5050)
    diffs = [abs(ranks_60[i] - ranks_50[i]) for i in range(len(qualified))]
    moves = Counter()
    for d in diffs:
        if d <= 2: moves['<=2'] += 1
        elif d <= 5: moves['3-5'] += 1
        elif d <= 10: moves['6-10'] += 1
        elif d <= 20: moves['11-20'] += 1
        else: moves['>20'] += 1
    print(f"Rank-shift distribution (60/40 → 50/50): {dict(moves)}")

    # Biggest movers: hitters whose BB+ changes most between the two weightings
    by_delta = sorted(
        qualified,
        key=lambda h: abs(h['_bb_plus_6040'] - h['_bb_plus_5050']),
        reverse=True
    )
    print("\nTop 10 biggest 60/40 vs 50/50 movers (|ΔBB+|):")
    print(f"  {'hitter':<24} {'xwOBAcon':>9} {'xwOBAsp':>9} {'60/40':>7} {'50/50':>7} {'Δ':>6}")
    for h in by_delta[:10]:
        d = h['_bb_plus_6040'] - h['_bb_plus_5050']
        print(f"  {h['hitter']:<24} {h['xwOBAcon']:>9.3f} {h['xwOBAsp']:>9.3f} "
              f"{h['_bb_plus_6040']:>7.1f} {h['_bb_plus_5050']:>7.1f} {d:>+6.2f}")

    # Top 10 leaders under each weighting
    print("\nTop 10 BB+ under 60/40:")
    top_60 = sorted(qualified, key=lambda h: -h['_bb_plus_6040'])[:10]
    for i, h in enumerate(top_60, 1):
        print(f"  {i:>2}. {h['hitter']:<24} BB+={h['_bb_plus_6040']:.1f} "
              f"(xwOBAcon={h['xwOBAcon']:.3f}, xwOBAsp={h['xwOBAsp']:.3f})")

    print("\nTop 10 BB+ under 50/50:")
    top_50 = sorted(qualified, key=lambda h: -h['_bb_plus_5050'])[:10]
    for i, h in enumerate(top_50, 1):
        print(f"  {i:>2}. {h['hitter']:<24} BB+={h['_bb_plus_5050']:.1f} "
              f"(xwOBAcon={h['xwOBAcon']:.3f}, xwOBAsp={h['xwOBAsp']:.3f})")


if __name__ == '__main__':
    main()
