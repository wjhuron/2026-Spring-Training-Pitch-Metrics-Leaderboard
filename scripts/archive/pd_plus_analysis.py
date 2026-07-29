"""PD+ composite weighting exploration using 2026 data.

Tests:
1. Correlation between Disc+ (IZSw-Ch%) and Exec+ (Contact%) (orthogonality)
2. 50/50 vs 55/45 vs 45/55: distribution, rank correlation, biggest movers
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
    return pearson(ranks(xs), ranks(ys))


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


def rank_map(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    ranks = [0] * len(vals)
    for r_idx, orig in enumerate(order):
        ranks[orig] = r_idx + 1
    return ranks


def main():
    with open('data/hitter_leaderboard_rs.json') as f:
        hitters = json.load(f)
    with open('data/metadata_rs.json') as f:
        md = json.load(f)

    hla = md['hitterLeagueAverages']
    lg_izsw = hla.get('izSwingPct')
    lg_chase = hla.get('chasePct')
    lg_contact = hla.get('contactPct')
    lg_izswchase = lg_izsw - lg_chase if (lg_izsw and lg_chase) else None

    print(f"League avg IZSw%     = {lg_izsw:.4f}")
    print(f"League avg Chase%    = {lg_chase:.4f}")
    print(f"League avg IZSw-Ch%  = {lg_izswchase:.4f}")
    print(f"League avg Contact%  = {lg_contact:.4f}")
    print(f"Total hitters: {len(hitters)}")

    # Qualification: exclude multi-team aggregates, require non-null inputs, min PA
    PA_FLOOR = 50
    qualified = []
    for h in hitters:
        team = h.get('team') or ''
        if isinstance(team, str) and team.endswith('TM') and team != 'TM':
            continue
        if h.get('izSwChase') is None or h.get('contactPct') is None:
            continue
        if (h.get('pa') or 0) < PA_FLOOR:
            continue
        qualified.append(h)
    print(f"Qualified hitters (>= {PA_FLOOR} PA, single-team): {len(qualified)}")

    # ─── Test 1: correlation ──────────────────────────────────────────────
    print("\n=== TEST 1: Orthogonality of Disc+ (IZSw-Ch%) and Exec+ (Contact%) ===")
    disc_raw = [h['izSwChase'] for h in qualified]
    exec_raw = [h['contactPct'] for h in qualified]
    r = pearson(disc_raw, exec_raw)
    rho = spearman(disc_raw, exec_raw)
    print(f"Pearson  r = {r:.3f}")
    print(f"Spearman ρ = {rho:.3f}")
    if abs(r) < 0.5:
        verdict = "LOW correlation — composite adds real signal"
    elif abs(r) < 0.7:
        verdict = "MODERATE correlation — composite likely adds signal, weights matter less"
    elif abs(r) < 0.85:
        verdict = "HIGH correlation — diminishing returns"
    else:
        verdict = "VERY HIGH correlation — redundant, rethink"
    print(f"Verdict: {verdict}")

    # ─── Test 2: weighting comparison ────────────────────────────────────
    print("\n=== TEST 2: Weighting comparison ===")

    def pd_plus(h, w_disc):
        disc_plus = 100.0 * h['izSwChase'] / lg_izswchase
        exec_plus = 100.0 * h['contactPct'] / lg_contact
        return w_disc * disc_plus + (1.0 - w_disc) * exec_plus

    for h in qualified:
        h['_pd_5050'] = pd_plus(h, 0.50)
        h['_pd_5545'] = pd_plus(h, 0.55)
        h['_pd_4555'] = pd_plus(h, 0.45)
        h['_pd_6040'] = pd_plus(h, 0.60)
        h['_disc_only'] = 100.0 * h['izSwChase'] / lg_izswchase
        h['_exec_only'] = 100.0 * h['contactPct'] / lg_contact

    v5050 = [h['_pd_5050'] for h in qualified]
    v5545 = [h['_pd_5545'] for h in qualified]
    v4555 = [h['_pd_4555'] for h in qualified]
    v6040 = [h['_pd_6040'] for h in qualified]
    vdisc = [h['_disc_only'] for h in qualified]
    vexec = [h['_exec_only'] for h in qualified]

    print("\nDistribution (index, 100 = league avg):")
    summarize("PD+ 50/50 ", v5050)
    summarize("PD+ 55/45 ", v5545)
    summarize("PD+ 45/55 ", v4555)
    summarize("PD+ 60/40 ", v6040)
    summarize("Disc+ only", vdisc)
    summarize("Exec+ only", vexec)

    # Rank correlations
    print(f"\n50/50 vs 55/45: Pearson r = {pearson(v5050, v5545):.4f}, Spearman ρ = {spearman(v5050, v5545):.4f}")
    print(f"50/50 vs 45/55: Pearson r = {pearson(v5050, v4555):.4f}, Spearman ρ = {spearman(v5050, v4555):.4f}")
    print(f"50/50 vs 60/40: Pearson r = {pearson(v5050, v6040):.4f}, Spearman ρ = {spearman(v5050, v6040):.4f}")

    # Rank-shift distribution 50/50 → 60/40
    ranks_50 = rank_map(v5050)
    ranks_60 = rank_map(v6040)
    diffs_6040 = [abs(ranks_50[i] - ranks_60[i]) for i in range(len(qualified))]
    moves = Counter()
    for d in diffs_6040:
        if d <= 2: moves['<=2'] += 1
        elif d <= 5: moves['3-5'] += 1
        elif d <= 10: moves['6-10'] += 1
        elif d <= 20: moves['11-20'] += 1
        else: moves['>20'] += 1
    print(f"Rank-shift 50/50 → 60/40: {dict(moves)}")

    # Biggest movers 50/50 vs 60/40
    by_delta = sorted(
        qualified,
        key=lambda h: abs(h['_pd_6040'] - h['_pd_5050']),
        reverse=True
    )
    print("\nTop 10 biggest 50/50 vs 60/40 movers (|ΔPD+|):")
    print(f"  {'hitter':<24} {'IZSw-Ch%':>8} {'Contact%':>8} {'50/50':>6} {'60/40':>6} {'Δ':>6}")
    for h in by_delta[:10]:
        d = h['_pd_6040'] - h['_pd_5050']
        print(f"  {h['hitter']:<24} {h['izSwChase']*100:>7.1f}% {h['contactPct']*100:>7.1f}% "
              f"{h['_pd_5050']:>6.1f} {h['_pd_6040']:>6.1f} {d:>+6.2f}")

    # Top 10 at 50/50
    print("\nTop 10 PD+ under 50/50:")
    for i, h in enumerate(sorted(qualified, key=lambda h: -h['_pd_5050'])[:10], 1):
        print(f"  {i:>2}. {h['hitter']:<24} PD+={h['_pd_5050']:.1f} "
              f"(IZSw-Ch%={h['izSwChase']*100:.1f}%, Contact%={h['contactPct']*100:.1f}%)")

    # Bottom 10 at 50/50
    print("\nBottom 10 PD+ under 50/50:")
    for i, h in enumerate(sorted(qualified, key=lambda h: h['_pd_5050'])[:10], 1):
        print(f"  {i:>2}. {h['hitter']:<24} PD+={h['_pd_5050']:.1f} "
              f"(IZSw-Ch%={h['izSwChase']*100:.1f}%, Contact%={h['contactPct']*100:.1f}%)")


if __name__ == '__main__':
    main()
