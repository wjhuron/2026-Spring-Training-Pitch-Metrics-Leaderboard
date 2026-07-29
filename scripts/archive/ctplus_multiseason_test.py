"""ctplus_multiseason_test.py — does building the CT+ (zone x count) league cell
table from multiple seasons help or hurt? Confirmation of the "same as Loc+"
prediction (CT+ is a zone x count value table, structurally like Loc+).

Builds the league contact-cell table (p_whiff, rv_contact, rv_whiff per zone x
count) from 2026-only vs 2021-2026, then scores 2026 hitters against each table
and measures reliability (odd/even split-half of raw_ct) + predictiveness
(first-half raw_ct vs second-half actual contact rate). Only the TABLE baseline
changes; rv currency + all evaluation stay on 2026. Reuses the production CT+
functions so the test is faithful.

Usage: python3 scripts/ctplus_multiseason_test.py
"""
import os, sys, math, pickle, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pipeline_contact as CT
import pipeline_sdplus as SD
from pipeline_utils import safe_float as sf

LG, SCALE = 0.3172, 1.2343
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
DMAP_SWING = {'swinging_strike': 'Swinging Strike', 'swinging_strike_blocked': 'Swinging Strike',
              'foul_tip': 'Swinging Strike', 'foul': 'Foul', 'hit_into_play': 'In Play',
              'hit_into_play_no_out': 'In Play', 'hit_into_play_score': 'In Play'}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs); sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def cf(x):
    """NA-safe float for pandas itertuples values (pandas NA breaks safe_float)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def hist_swings(year):
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    out = []
    for r in df.itertuples(index=False):
        d = DMAP_SWING.get(r.description)
        if d is None:
            continue
        px, pz = cf(r.plate_x), cf(r.plate_z); top, bot = cf(r.sz_top), cf(r.sz_bot)
        b, s = cf(r.balls), cf(r.strikes)
        if None in (px, pz, top, bot, b, s) or top <= bot:
            continue
        re = cf(getattr(r, 'delta_pitcher_run_exp', None))
        if re is None:
            dre = cf(r.delta_run_exp); re = -dre if dre is not None else None
        xw = cf(r.estimated_woba_using_speedangle) if d == 'In Play' else None
        out.append({'PlateX': px, 'PlateZ': pz, 'SzTop': top, 'SzBot': bot,
                    'InZone': 'Yes' if (abs(px) <= 0.83 and bot <= pz <= top) else 'No',
                    'Count': f'{int(b)}-{int(s)}', 'Description': d, 'xwOBA': xw, 'RunExp': re})
    return out


def build_table(swings, rv_fn):
    raw = CT.build_contact_cell_weights(swings, rv_fn)
    zmeans = CT.zone_level_contact_means(swings, rv_fn)
    return CT.shrink_contact_cells(raw, zmeans)


def main():
    print('loading 2026 ...', flush=True)
    P = [p for p in pickle.load(open(PKL, 'rb')) if p.get('_source') == 'MLB']
    rv_fn = SD.make_rv_xrv(LG, SCALE, SD.build_bip_count_offsets(P, LG, SCALE))
    sw26 = [p for p in P if CT.is_ct_eligible(p)]
    print(f'  {len(sw26)} CT-eligible 2026 swings', flush=True)

    # eval infra: per-hitter pitches, odd/even game split, first/second half
    byhit = collections.defaultdict(list)
    for p in sw26:
        byhit[p.get('Batter')].append(p)
    dbp = collections.defaultdict(set)
    for p in sw26:
        dbp[p.get('Batter')].add(p.get('Game Date'))
    half = {}
    for h, ds in dbp.items():
        for idx, dd in enumerate(sorted(ds)):
            half[(h, dd)] = idx % 2

    def evaluate(table):
        # reliability: odd/even split-half of raw_ct
        h0 = collections.defaultdict(list); h1 = collections.defaultdict(list)
        for h, ps in byhit.items():
            for p in ps:
                (h0 if half[(h, p.get('Game Date'))] == 0 else h1)[h].append(p)
        r0 = CT.compute_hitter_ct(h0, table); r1 = CT.compute_hitter_ct(h1, table)
        com = [k for k in r0 if k in r1 and r0[k]['n_swings'] >= 40 and r1[k]['n_swings'] >= 40]
        rel = pearson([r0[k]['raw_ct'] for k in com], [r1[k]['raw_ct'] for k in com])
        # predictiveness: first-half raw_ct vs second-half actual contact rate
        first = collections.defaultdict(list); sec_ct = collections.defaultdict(lambda: [0, 0])
        for h, ps in byhit.items():
            for p in ps:
                if (p.get('Game Date') or '') < '2026-05-01':
                    first[h].append(p)
                else:
                    sec_ct[h][1] += 1
                    if CT.classify_contact_outcome(p) == 'contact':
                        sec_ct[h][0] += 1
        rf = CT.compute_hitter_ct(first, table)
        kp = [k for k in rf if k in sec_ct and rf[k]['n_swings'] >= 85 and sec_ct[k][1] >= 85]
        pred = pearson([rf[k]['raw_ct'] for k in kp], [sec_ct[k][0] / sec_ct[k][1] for k in kp])
        return rel, pred, len(com)

    print(f"\n{'CT+ table baseline':26s} {'swings':>9s} {'reliab':>7s} {'pred':>7s}  (n)")
    base_tbl = build_table(sw26, rv_fn)
    r = evaluate(base_tbl)
    print(f"{'2026-only (current)':26s} {len(sw26):9d} {r[0]:7.3f} {r[1]:7.3f}  ({r[2]})")
    for yrs, lbl in [((2024, 2025), '+2024-25'), ((2021, 2022, 2023, 2024, 2025), '+2021-25 (all)')]:
        extra = []
        for y in yrs:
            extra += hist_swings(y)
        tbl = build_table(sw26 + extra, rv_fn)
        rr = evaluate(tbl)
        print(f"{lbl:26s} {len(sw26)+len(extra):9d} {rr[0]:7.3f} {rr[1]:7.3f}  ({rr[2]})")
    print("\n(prediction: multi-season HURTS, like Loc+ — CT+ is a zone x count value table)")


if __name__ == '__main__':
    main()
