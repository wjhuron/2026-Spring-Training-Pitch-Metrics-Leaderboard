"""handsplit_sdct_test.py — does a batter-hand dimension help SD+/CT+?

Production SD+/CT+ cell tables are handedness-blind (classify_zone uses
abs(PlateX); no hand key anywhere). This tests HANDSPLIT variants: cell
tables, shrinkage priors, and league zone mix all built per batter hand
(switch hitters contribute per-PA side, exactly as they should).

Count-anchor offsets stay GLOBAL in both variants — they are a currency
conversion (count-state correction), not a hand effect.

Metrics, seasons 2021-2025 (public Statcast via adapter) + 2026 (cache):
  1. Split-half reliability of the raw metric (3 random game-date partitions;
     hitters >=125 decisions / >=45 swings per half — half the production floors).
  2. Predictive: full-season year-N raw component vs year-N+1 wOBA
     (pairs 21->22 .. 24->25; hitters >= production floors in year N).

Adopt handsplit only if reliability holds AND prediction doesn't drop.

Usage: python3 scripts/handsplit_sdct_test.py
"""
import os, sys, math, pickle, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statcast_hitter_adapter as A
import pipeline_sdplus as sd
import pipeline_contact as ct

SEEDS = (0, 1, 2)
HALF_MIN_DEC = 125
HALF_MIN_SW = 45
FULL_MIN_DEC = sd.MIN_HITTER_DECISIONS   # 250
FULL_MIN_SW = ct.MIN_HITTER_SWINGS       # 85
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
GUTS_2026 = (0.3172, 1.2343)   # train_stuff fallback constants


def pearson(xs, ys):
    n = len(xs)
    if n < 20:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def load_season(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB']
    return A.season_dicts(year)


def guts(year):
    return GUTS_2026 if year == 2026 else A.GUTS[year]


# ── SD+ scoring paths ────────────────────────────────────────────────────

def sd_tables_baseline(elig, rv_fn):
    raw = sd.build_weight_table(elig, rv_fn)
    zm = sd.zone_level_means(elig, rv_fn)
    table = sd.shrink_table(raw, zm)
    zc = defaultdict(int)
    for p in elig:
        zc[sd.classify_zone(p)] += 1
    tot = sum(zc.values())
    lgw = {z: n / tot for z, n in zc.items()}
    return table, lgw


def sd_score_baseline(by_hitter, table, lgw, min_n):
    out = {}
    for h, pitches in by_hitter.items():
        elig = [p for p in pitches if sd.is_eligible(p)]
        if len(elig) < min_n:
            continue
        zone_dvs = defaultdict(list)
        for p in elig:
            zone_dvs[sd.classify_zone(p)].append(sd.compute_dv(p, table))
        zmeans = {z: sum(v) / len(v) for z, v in zone_dvs.items()}
        wsum = sum(lgw.get(z, 0.0) for z in zmeans)
        if wsum <= 0:
            continue
        out[h] = sum(m * lgw.get(z, 0.0) for z, m in zmeans.items()) / wsum
    return out


def sd_tables_handsplit(elig, rv_fn):
    tables, lgw = {}, {}
    zc = defaultdict(int)
    by_hand = {'L': [], 'R': []}
    for p in elig:
        b = p.get('Bats')
        if b in by_hand:
            by_hand[b].append(p)
            zc[(b, sd.classify_zone(p))] += 1
    tot = sum(zc.values())
    for hand, sub in by_hand.items():
        raw = sd.build_weight_table(sub, rv_fn)
        zm = sd.zone_level_means(sub, rv_fn)
        tables[hand] = sd.shrink_table(raw, zm)
    lgw = {k: n / tot for k, n in zc.items()}
    return tables, lgw


def sd_score_handsplit(by_hitter, tables, lgw, min_n):
    out = {}
    for h, pitches in by_hitter.items():
        elig = [p for p in pitches if sd.is_eligible(p) and p.get('Bats') in tables]
        if len(elig) < min_n:
            continue
        zone_dvs = defaultdict(list)
        for p in elig:
            key = (p['Bats'], sd.classify_zone(p))
            zone_dvs[key].append(sd.compute_dv(p, tables[p['Bats']]))
        zmeans = {k: sum(v) / len(v) for k, v in zone_dvs.items()}
        wsum = sum(lgw.get(k, 0.0) for k in zmeans)
        if wsum <= 0:
            continue
        out[h] = sum(m * lgw.get(k, 0.0) for k, m in zmeans.items()) / wsum
    return out


# ── CT+ scoring paths ────────────────────────────────────────────────────

def ct_tables_baseline(swings, rv_fn):
    raw = ct.build_contact_cell_weights(swings, rv_fn)
    zm = ct.zone_level_contact_means(swings, rv_fn)
    return ct.shrink_contact_cells(raw, zm)


def ct_score(by_hitter, table_for, min_n):
    """table_for(p) -> cell table to use for pitch p (baseline: constant)."""
    out = {}
    for h, pitches in by_hitter.items():
        swings = [p for p in pitches if ct.is_ct_eligible(p)]
        swings = [p for p in swings if table_for(p) is not None]
        if len(swings) < min_n:
            continue
        actual = expected = 0.0
        for p in swings:
            cell = table_for(p)[(sd.classify_zone(p), sd.get_count(p))]
            lev = cell['rv_contact'] - cell['rv_whiff']
            if lev <= 0:
                continue
            con = 1 if ct.classify_contact_outcome(p) == 'contact' else 0
            actual += lev * con
            expected += lev * (1.0 - cell['p_whiff'])
        if expected <= 0:
            continue
        out[h] = actual / expected
    return out


def ct_tables_handsplit(swings, rv_fn):
    tables = {}
    for hand in ('L', 'R'):
        sub = [p for p in swings if p.get('Bats') == hand]
        tables[hand] = ct_tables_baseline(sub, rv_fn)
    return tables


# ── Harness ──────────────────────────────────────────────────────────────

def season_components(P, lg, sc, min_dec, min_sw):
    """Return per-variant {'sd': {...}, 'ct': {...}} raw components."""
    elig = [p for p in P if p.get('_source', 'MLB') == 'MLB' and sd.is_eligible(p)]
    offsets = sd.build_bip_count_offsets(elig, lg, sc)
    rv_fn = sd.make_rv_xrv(lg, sc, offsets)
    swings = [p for p in elig if ct.is_ct_eligible(p)]

    by_hitter = defaultdict(list)
    for p in elig:
        h = p.get('Batter')
        if h:
            by_hitter[h].append(p)

    res = {}
    table, lgw = sd_tables_baseline(elig, rv_fn)
    res['base_sd'] = sd_score_baseline(by_hitter, table, lgw, min_dec)
    tables_h, lgw_h = sd_tables_handsplit(elig, rv_fn)
    res['hand_sd'] = sd_score_handsplit(by_hitter, tables_h, lgw_h, min_dec)

    tb = ct_tables_baseline(swings, rv_fn)
    res['base_ct'] = ct_score(by_hitter, lambda p: tb, min_sw)
    th = ct_tables_handsplit(swings, rv_fn)
    res['hand_ct'] = ct_score(by_hitter, lambda p: th.get(p.get('Bats')), min_sw)
    return res


def main():
    # ── 1. split-half reliability ──
    print("SPLIT-HALF RELIABILITY (raw metric, per-half floors "
          f"{HALF_MIN_DEC} dec / {HALF_MIN_SW} sw)", flush=True)
    agg = defaultdict(list)
    for year in SEASONS:
        P = load_season(year)
        lg, sc = guts(year)
        dates = sorted({p.get('Game Date') for p in P if p.get('Game Date')})
        for seed in SEEDS:
            rnd = random.Random(seed)
            sh = dates[:]
            rnd.shuffle(sh)
            ha = set(sh[:len(sh) // 2])
            Pa = [p for p in P if p.get('Game Date') in ha]
            Pb = [p for p in P if p.get('Game Date') and p.get('Game Date') not in ha]
            ra = season_components(Pa, lg, sc, HALF_MIN_DEC, HALF_MIN_SW)
            rb = season_components(Pb, lg, sc, HALF_MIN_DEC, HALF_MIN_SW)
            row = {}
            for k in ('base_sd', 'hand_sd', 'base_ct', 'hand_ct'):
                common = [h for h in ra[k] if h in rb[k]]
                r = pearson([ra[k][h] for h in common], [rb[k][h] for h in common])
                row[k] = (r, len(common))
                if r is not None:
                    agg[k].append(r)
            print(f"  {year} seed{seed}: " + '  '.join(
                f"{k} r={row[k][0]:.3f}(n={row[k][1]})" if row[k][0] is not None
                else f"{k} r=NA" for k in row), flush=True)
        del P
    print("\n  MEAN split-half r:")
    for k in ('base_sd', 'hand_sd', 'base_ct', 'hand_ct'):
        rs = agg[k]
        print(f"    {k}: {sum(rs)/len(rs):.4f}  (n={len(rs)} season-seeds)")

    # ── 2. predictive: year N component -> year N+1 wOBA ──
    print("\nPREDICTIVE (full-season raw, floors "
          f"{FULL_MIN_DEC} dec / {FULL_MIN_SW} sw, vs next-season wOBA)", flush=True)
    pagg = defaultdict(list)
    for yn, yn1 in PAIRS:
        P = load_season(yn)
        lg, sc = guts(yn)
        comp = season_components(P, lg, sc, FULL_MIN_DEC, FULL_MIN_SW)
        y_map = A.target_y(yn1)
        line = [f"  {yn}->{yn1}:"]
        for k in ('base_sd', 'hand_sd', 'base_ct', 'hand_ct'):
            xs, ys = [], []
            for h, v in comp[k].items():
                yv = y_map.get(h)
                if yv and yv[1] >= 200:
                    xs.append(v)
                    ys.append(yv[0] / yv[1])
            r = pearson(xs, ys)
            line.append(f"{k} r={r:+.3f}(n={len(xs)})" if r is not None else f"{k} NA")
            if r is not None:
                pagg[k].append(r)
        print('  '.join(line), flush=True)
        del P
    print("\n  MEAN predictive r:")
    for k in ('base_sd', 'hand_sd', 'base_ct', 'hand_ct'):
        rs = pagg[k]
        print(f"    {k}: {sum(rs)/len(rs):+.4f}")


if __name__ == '__main__':
    main()
