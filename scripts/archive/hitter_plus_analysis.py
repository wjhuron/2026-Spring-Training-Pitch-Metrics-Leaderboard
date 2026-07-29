"""Hitter+ composite exploration: BB+ vs PD+ orthogonality and composite shapes."""
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


def main():
    with open('data/hitter_leaderboard_rs.json') as f:
        hitters = json.load(f)
    with open('data/metadata_rs.json') as f:
        md = json.load(f)

    hla = md['hitterLeagueAverages']
    lg_xc = hla['xwOBAcon']
    lg_xs = hla['xwOBAsp']
    lg_disc = hla['izSwChase']
    lg_exec = hla['contactPct']

    # Compute BB+ and PD+ for each hitter that qualifies
    PA_FLOOR = 50
    BIP_FLOOR = 30
    qualified = []
    for h in hitters:
        team = h.get('team') or ''
        if isinstance(team, str) and team.endswith('TM') and team != 'TM':
            continue
        xc = h.get('xwOBAcon'); xs = h.get('xwOBAsp')
        disc = h.get('izSwChase'); ex = h.get('contactPct')
        if xc is None or xs is None or disc is None or ex is None:
            continue
        if (h.get('pa') or 0) < PA_FLOOR: continue
        if (h.get('nBip') or 0) < BIP_FLOOR: continue

        h['_bb'] = 0.6 * (100 * xc / lg_xc) + 0.4 * (100 * xs / lg_xs)
        h['_pd'] = 0.5 * (100 * disc / lg_disc) + 0.5 * (100 * ex / lg_exec)
        h['_hitter_add'] = 0.5 * h['_bb'] + 0.5 * h['_pd']
        h['_hitter_mul'] = h['_bb'] * h['_pd'] / 100.0
        qualified.append(h)

    print(f"Qualified hitters (PA >= {PA_FLOOR}, BIP >= {BIP_FLOOR}, single-team): {len(qualified)}")

    # ─── Orthogonality check ─────────────────────────────────────────────
    print("\n=== BB+ vs PD+ orthogonality ===")
    bb = [h['_bb'] for h in qualified]
    pd = [h['_pd'] for h in qualified]
    r = pearson(bb, pd)
    rho = spearman(bb, pd)
    print(f"Pearson  r = {r:.3f}")
    print(f"Spearman ρ = {rho:.3f}")
    if abs(r) < 0.5:
        verdict = "LOW correlation — multiplicative form is sound, no overstating extremes"
    elif abs(r) < 0.7:
        verdict = "MODERATE correlation — multiplicative slightly overstates, but workable"
    elif abs(r) < 0.85:
        verdict = "HIGH correlation — multiplicative overstates extremes meaningfully"
    else:
        verdict = "VERY HIGH — redundant, additive might be better"
    print(f"Verdict: {verdict}")

    # ─── Distributions ─────────────────────────────────────────────────
    print("\n=== Distributions ===")
    summarize("BB+          ", bb)
    summarize("PD+          ", pd)
    summarize("Hitter+ add  ", [h['_hitter_add'] for h in qualified])
    summarize("Hitter+ mult ", [h['_hitter_mul'] for h in qualified])

    # ─── Rank correlation: additive vs multiplicative ──────────────────
    add_vs_mult_r = pearson([h['_hitter_add'] for h in qualified],
                             [h['_hitter_mul'] for h in qualified])
    add_vs_mult_rho = spearman([h['_hitter_add'] for h in qualified],
                                [h['_hitter_mul'] for h in qualified])
    print(f"\nAdditive vs Multiplicative: Pearson r = {add_vs_mult_r:.4f}, Spearman ρ = {add_vs_mult_rho:.4f}")

    # ─── Biggest movers: additive → multiplicative ─────────────────────
    print("\nTop 10 biggest movers (additive → multiplicative):")
    by_delta = sorted(qualified, key=lambda h: abs(h['_hitter_mul'] - h['_hitter_add']), reverse=True)
    print(f"  {'hitter':<24} {'BB+':>6} {'PD+':>6} {'Add':>6} {'Mult':>6} {'Δ':>6}")
    for h in by_delta[:10]:
        d = h['_hitter_mul'] - h['_hitter_add']
        print(f"  {h['hitter']:<24} {h['_bb']:>6.1f} {h['_pd']:>6.1f} "
              f"{h['_hitter_add']:>6.1f} {h['_hitter_mul']:>6.1f} {d:>+6.2f}")

    # ─── Top 10 leaders under each form ────────────────────────────────
    print("\nTop 15 Hitter+ (multiplicative):")
    top_mult = sorted(qualified, key=lambda h: -h['_hitter_mul'])[:15]
    print(f"  {'rk':>2}  {'hitter':<24} {'BB+':>6} {'PD+':>6} {'H+mult':>7} {'xWRC+':>6}")
    for i, h in enumerate(top_mult, 1):
        xwrc = h.get('xWRCplus')
        xwrc_s = str(xwrc) if xwrc is not None else '—'
        print(f"  {i:>2}.  {h['hitter']:<24} {h['_bb']:>6.1f} {h['_pd']:>6.1f} "
              f"{h['_hitter_mul']:>7.1f} {xwrc_s:>6}")

    # ─── Correlation of Hitter+ forms vs xWRC+ (sanity check) ─────────
    xwrc_pairs = [(h['_hitter_mul'], h['xWRCplus']) for h in qualified if h.get('xWRCplus') is not None]
    if xwrc_pairs:
        hmul = [a for a, _ in xwrc_pairs]
        xwrc = [b for _, b in xwrc_pairs]
        r_m = pearson(hmul, xwrc)
        rho_m = spearman(hmul, xwrc)
        print(f"\nHitter+ (mult) vs xWRC+: Pearson r = {r_m:.3f}, Spearman ρ = {rho_m:.3f}, n = {len(hmul)}")
        hadd = [h['_hitter_add'] for h in qualified if h.get('xWRCplus') is not None]
        r_a = pearson(hadd, xwrc)
        rho_a = spearman(hadd, xwrc)
        print(f"Hitter+ (add)  vs xWRC+: Pearson r = {r_a:.3f}, Spearman ρ = {rho_a:.3f}, n = {len(hadd)}")


if __name__ == '__main__':
    main()
