"""hitter_spread_atlas.py — measured truthful spreads for the hitter plus
metrics, 2021-2026 replicates.

Question (2026-08-15): the hitter plus family displays at wRC+-matched
spread by convention. What spread does each metric EARN under run-truth,
i.e. how many points of wRC+ does one SD of the metric actually carry,
descriptively (same season) and predictively (next season)?

Per season 2021-2025 (public Statcast via statcast_hitter_adapter) and
2026 (sheet cache): rebuild the shipped-config components —
  BB+ raw  = xwOBAcon shrunk toward league at n0 = 60 BIP (shipped)
  SD+ raw  = shipped BASE config (heart 1/6, cat3, count-anchor, mix),
             via the hitter_phase2_multiseason machinery
  CT+ raw  = shipped BASE (count-anchor, lift ratio)
  Hitter+  = 0.52 z(BB+) + 0.17 z(SD+) + 0.31 z(CT+)   (shipped weights)
  xwOBApa  = PA-level xwOBA (the xwRC+ analog)
Target: wRC+ approximation from the same pitch data,
  wRC+ ~ 100 * (1 + ((wOBA - lgwOBA)/scale) / lgRPA)
with per-season Guts and league R/PA from the Stats API (park-less; fine
for slopes). Qualified pool = 300+ PA events.

Outputs per metric: descriptive slope (wRC+ pts per 1 SD) + r per season
(6 replicates), predictive slope/r per year pair (5 replicates).
Results: console + data/_hitter_spread_atlas.json

Usage: python3 scripts/hitter_spread_atlas.py   (long: rebuilds SD/CT
tables per season; run in background)
"""
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import hitter_phase2_multiseason as H
import pipeline_sdplus as sd
import pipeline_contact as ct

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
MIN_PA = 300
N0_BB = 60.0
W_BB, W_SD, W_CT = 0.52, 0.17, 0.31
# dual vocabulary: the 2026 sheet cache uses feed tokens, the 2021-2025
# adapter passes RAW statcast tokens (only Intent Walk is normalized)
WOBA_W = {'Walk': .69, 'Hit By Pitch': .72, 'Single': .89, 'Double': 1.27,
          'Triple': 1.61, 'Home Run': 2.10,
          'walk': .69, 'hit_by_pitch': .72, 'single': .89, 'double': 1.27,
          'triple': 1.61, 'home_run': 2.10}
BB_HBP = {'Walk', 'Hit By Pitch', 'walk', 'hit_by_pitch'}
K_EVENTS = {'Strikeout', 'Strikeout Double Play',
            'strikeout', 'strikeout_double_play'}
IBB = {'Intent Walk', 'intent_walk'}
NON_PA_TOKENS = ('stealing', 'pickoff', 'stolen', 'wild_pitch',
                 'passed_ball', 'truncated', 'Wild Pitch', 'Passed Ball',
                 'Stolen Base', 'Caught Stealing', 'Pickoff')
RPA_CACHE = os.path.join(ROOT, 'data', '_lg_rpa.json')


def lg_rpa(season):
    try:
        cache = json.load(open(RPA_CACHE))
    except (OSError, json.JSONDecodeError):
        cache = {}
    if str(season) in cache:
        return cache[str(season)]
    url = ('https://statsapi.mlb.com/api/v1/teams/stats?stats=season'
           f'&group=hitting&season={season}&sportId=1')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    runs = pa = 0
    for sp in d['stats'][0]['splits']:
        st = sp['stat']
        runs += int(st.get('runs') or 0)
        pa += int(st.get('plateAppearances') or 0)
    cache[str(season)] = runs / pa
    json.dump(cache, open(RPA_CACHE, 'w'))
    return cache[str(season)]


def pa_outcomes(P):
    """Per-hitter PA-level tallies from pipeline-shaped pitches."""
    acc = defaultdict(lambda: defaultdict(float))
    for p in P:
        ev = p.get('Event')
        if not ev:
            continue
        h = p.get('Batter')
        if not h:
            continue
        c = acc[h]
        from pipeline_utils import NON_PA_EVENTS, BUNT_BB_TYPES
        if (ev in NON_PA_EVENTS or ev in IBB
                or any(t in ev for t in NON_PA_TOKENS)):
            continue
        c['pa'] += 1
        w = WOBA_W.get(ev, 0.0)
        c['w_num'] += w
        if ev in K_EVENTS:
            c['xw_num'] += 0.0
        elif ev in BB_HBP:
            c['xw_num'] += w
        else:
            xw = p.get('xwOBA')
            try:
                xw = float(xw)
                if xw != xw:
                    xw = None
            except (TypeError, ValueError):
                xw = None
            c['xw_num'] += xw if xw is not None else 0.0
        bbt = p.get('BBType')
        if bbt and bbt not in BUNT_BB_TYPES:
            xw = p.get('xwOBA')
            try:
                xw = float(xw)
                if xw == xw:
                    c['xwc_sum'] += xw
                    c['xwc_n'] += 1
            except (TypeError, ValueError):
                pass
    return acc


def zmap(vals):
    m = sum(vals.values()) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals.values()) / len(vals))
    return {k: (v - m) / s for k, v in vals.items()} if s else {}


def pearson_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None, None
    sdx = math.sqrt(sxx / n)
    return sxy / math.sqrt(sxx * syy), (sxy / sxx) * sdx


CACHE = os.path.join(ROOT, 'data', '_hitter_spread_cache.json')


def build_season_cached(year):
    try:
        c = json.load(open(CACHE))
    except (OSError, json.JSONDecodeError):
        c = {}
    if str(year) in c:
        f, t = c[str(year)]
        print(f'{year}: cache hit ({len(t)} hitters)', flush=True)
        return f, t
    feats, targ = build_season(year)
    c[str(year)] = [feats, targ]
    json.dump(c, open(CACHE, 'w'))
    return feats, targ


def build_season(year):
    print(f'{year}: loading', flush=True)
    P = H.load_season(year)
    lg, sc = H.guts(year)
    rpa = lg_rpa(year)
    elig = H.precompute(P)
    print(f'  {len(P)} pitches, {len(elig)} SD-eligible; lgRPA {rpa:.4f}',
          flush=True)

    # SD+/CT+ BASE only (trimmed from season_components)
    with H.patched('_z16', True):
        offsets = sd.build_bip_count_offsets(elig, lg, sc)
        rv_fn = sd.make_rv_xrv(lg, sc, offsets)
        raw = sd.build_weight_table(elig, rv_fn)
        table = sd.shrink_table(raw, sd.zone_level_means(elig, rv_fn))
        zc = defaultdict(int)
        for p in elig:
            zc[sd.classify_zone(p)] += 1
        tot = sum(zc.values())
        lgw = {z: n / tot for z, n in zc.items()}
        by_h = defaultdict(list)
        for p in elig:
            if p.get('Batter'):
                by_h[p['Batter']].append(p)
        sd_raw = H.sd_score(by_h, table, lgw, 200)

        swings = [p for p in elig if ct.is_ct_eligible(p)]
        by_h_sw = defaultdict(list)
        for p in swings:
            if p.get('Batter'):
                by_h_sw[p['Batter']].append(p)
        c_off = ct.build_bip_count_offsets(swings, lg, sc)
        c_rv = ct.make_rv_xrv(lg, sc, c_off)
        c_tab = ct.shrink_contact_cells(
            ct.build_contact_cell_weights(swings, c_rv),
            ct.zone_level_contact_means(swings, c_rv))
        ct_raw = H.ct_score(by_h_sw, c_tab, 65, True)

    out = pa_outcomes(P)
    pool = {h: c for h, c in out.items() if c['pa'] >= MIN_PA}
    lg_xwc = (sum(c['xwc_sum'] for c in pool.values())
              / sum(c['xwc_n'] for c in pool.values()))
    lg_woba = (sum(c['w_num'] for c in pool.values())
               / sum(c['pa'] for c in pool.values()))

    feats, targ = {}, {}
    for h, c in pool.items():
        woba = c['w_num'] / c['pa']
        targ[h] = 100.0 * (1 + ((woba - lg_woba) / sc) / rpa)
        f = {'xwoba_pa': c['xw_num'] / c['pa']}
        if c['xwc_n'] > 0:
            f['bb'] = ((c['xwc_sum'] + N0_BB * lg_xwc)
                       / (c['xwc_n'] + N0_BB))
        if h in sd_raw:
            f['sd'] = sd_raw[h]
        if h in ct_raw:
            f['ct'] = ct_raw[h]
        feats[h] = f

    # Hitter+ from component z's (pool-wide)
    zs = {}
    for k in ('bb', 'sd', 'ct'):
        zs[k] = zmap({h: f[k] for h, f in feats.items() if k in f})
    for h, f in feats.items():
        if all(h in zs[k] for k in ('bb', 'sd', 'ct')):
            f['hitterplus'] = (W_BB * zs['bb'][h] + W_SD * zs['sd'][h]
                               + W_CT * zs['ct'][h])
    print(f'  pool {len(pool)} hitters; components bb {len(zs["bb"])} '
          f'sd {len(zs["sd"])} ct {len(zs["ct"])}', flush=True)
    return feats, targ


def main():
    data = {}
    for y in SEASONS:
        data[y] = build_season_cached(y)

    # ── 2026 key normalization (2026-08-15 fix) ──
    # 2021-2025 come from the statcast adapter keyed by MLB id; 2026 comes
    # from the sheet cache keyed by 'Last, First'. The 2025->2026
    # predictive pair silently joined on ZERO common keys. Bridge via the
    # hitter leaderboard (name + mlbId); ambiguous duplicate names dropped.
    lb = json.load(open(os.path.join(ROOT, 'data',
                                     'hitter_leaderboard_rs.json')))
    nm, ambig = {}, set()
    for r in lb:
        n, mid = r.get('hitter'), r.get('mlbId')
        if not n or not mid:
            continue
        if n in nm and nm[n] != str(mid):
            ambig.add(n)
        nm[n] = str(mid)
    for n in ambig:
        del nm[n]
    f26, t26 = data[2026]
    f_id = {nm[h]: v for h, v in f26.items() if h in nm}
    t_id = {nm[h]: v for h, v in t26.items() if h in nm}
    print(f'2026 keys remapped name->id: feats {len(f_id)}/{len(f26)}, '
          f'targ {len(t_id)}/{len(t26)}'
          + (f' ({len(ambig)} ambiguous names dropped)' if ambig else ''))
    data[2026] = (f_id, t_id)

    METRICS = ['hitterplus', 'xwoba_pa', 'bb', 'sd', 'ct']
    res = {'desc': {}, 'pred': {}}
    print('\n=== DESCRIPTIVE: wRC+ points per 1 SD, per season ===')
    for m in METRICS:
        rows = []
        for y in SEASONS:
            feats, targ = data[y]
            xs = [f[m] for h, f in feats.items() if m in f and h in targ]
            ys = [targ[h] for h, f in feats.items() if m in f and h in targ]
            r, slope = pearson_slope(xs, ys)
            if r is not None:
                rows.append((y, r, slope, len(xs)))
        res['desc'][m] = rows
        mean_s = sum(s for _, _, s, _ in rows) / len(rows)
        mean_r = sum(r for _, r, _, _ in rows) / len(rows)
        print(f'  {m:<10} mean slope {mean_s:6.1f}  mean r {mean_r:+.3f}  '
              + ' '.join(f'{y}:{s:.1f}' for y, _, s, _ in rows))

    print('\n=== PREDICTIVE: next-season wRC+ points per 1 SD ===')
    for m in METRICS:
        rows = []
        for y in SEASONS[:-1]:
            f1, _ = data[y]
            _, t2 = data[y + 1]
            xs = [f[m] for h, f in f1.items() if m in f and h in t2]
            ys = [t2[h] for h, f in f1.items() if m in f and h in t2]
            if len(xs) < 50:
                continue
            r, slope = pearson_slope(xs, ys)
            if r is None:
                continue
            rows.append((f'{y}->{y + 1}', r, slope, len(xs)))
        res['pred'][m] = rows
        if rows:
            mean_s = sum(s for _, _, s, _ in rows) / len(rows)
            mean_r = sum(r for _, r, _, _ in rows) / len(rows)
            print(f'  {m:<10} mean slope {mean_s:6.1f}  mean r {mean_r:+.3f}  '
                  + ' '.join(f'{lab}:{s:.1f}' for lab, _, s, _ in rows))

    with open(os.path.join(ROOT, 'data', '_hitter_spread_atlas.json'),
              'w') as f:
        json.dump(res, f)
    print('\nwrote data/_hitter_spread_atlas.json')


if __name__ == '__main__':
    main()
