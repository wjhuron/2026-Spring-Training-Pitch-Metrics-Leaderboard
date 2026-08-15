"""era_estimator_screen.py — the descriptive + predictive ERA-estimator
screen, 2021-2026.

Three tests, three grading currencies each:
  DESC  same-season metric vs same-season target (6 replicate seasons)
  NEXT  season-N metric vs season-N+1 target (5 replicate pairs)
  ROS   first-half metric vs second-half target (6 replicate seasons;
        2026 second half is partial through the sheet cutoff)

Targets: ERA, park-adjusted ERA (pERA = ERA / park exposure), RA9.
Park exposure = mean over the pitcher's teams of (PF/100 + 1) / 2, PF =
Savant rolling-3 runs index for the team's home park; missing park -> 1.0.

Inputs (built by companion scripts):
  data/_era_targets.json          official season/half lines + ASG dates
  data/_park_factors.json         per-team runs park factors
  data/_era_battery.json          pitch-derived candidate battery
  data/_era_internal_cmdloc.json  Loc+ raw / Command+ raw miss
  data/_era_internal_stuff.json   LOSO Stuff+ raw (2021-2025)

IP gates are swept, not chosen: every correlation is reported at outs
gates equivalent to 20/40/60/80/100/120 IP (both sides of predictive
pairs use the matching gate). The main tables quote the 60 IP column;
that choice is a reporting convention, not a measured optimum, and the
full gate curves ship in the JSON.

Output: data/_era_screen_results.json + console tables.
"""
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))

TARGETS = D('_era_targets.json')
PF = D('_park_factors.json')
BATTERY = D('_era_battery.json')
CMDLOC = D('_era_internal_cmdloc.json')
try:
    STUFF = D('_era_internal_stuff.json')
except FileNotFoundError:
    STUFF = {}
    print('NOTE: stuff LOSO file absent — stuff_raw only for 2026')

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
GATES_IP = [20, 40, 60, 80, 100, 120]
MAIN_GATE = 60

# linear-weight constants for constructs; scale-free for correlation
LG_HRFB = {}   # filled per season from battery totals


def pearson(xs, ys):
    n = len(xs)
    if n < 10:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = (i + j) / 2.0
            for k2 in range(i, j + 1):
                rk[order[k2]] = r
            i = j + 1
        return rk
    return pearson(ranks(xs), ranks(ys))


def park_exposure(season, teams):
    pfs = PF.get(str(season), {})
    if not teams:
        return 1.0
    vals = [(pfs.get(str(t), 100.0) / 100.0 + 1.0) / 2.0 for t in teams]
    return sum(vals) / len(vals)


def targets_for(season, scope):
    """scope: 'full' | 'h1' | 'h2' -> {pid: (era, pera, ra9, outs)}"""
    out = {}
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        src = rec if scope == 'full' else rec[scope]
        outs = src['outs']
        if outs <= 0:
            continue
        era = src['er'] * 27.0 / outs
        ra9 = src['r'] * 27.0 / outs
        pex = park_exposure(season, rec['teams'])
        out[pid] = {'era': era, 'pera': era / pex, 'ra9': ra9, 'outs': outs}
    return out


def official_line(season, scope, pid):
    rec = TARGETS[str(season)]['pitchers'].get(pid)
    if rec is None:
        return None
    return rec if scope == 'full' else rec[scope]


def lg_hrfb(season):
    if season in LG_HRFB:
        return LG_HRFB[season]
    hr = fb = 0
    for pid, rec in BATTERY[str(season)].items():
        kc = rec['full']['k_counts']
        hr += kc['hr']
        fb += kc['fb']
    LG_HRFB[season] = hr / fb if fb else 0.1
    return LG_HRFB[season]


def siera(kc, pa, gb, fb, pu, ip_outs):
    """Swartz SIERA, per-season constants omitted (scale-free screen)."""
    if pa <= 0:
        return None
    so_pa = kc['k'] / pa
    bb_pa = kc['bb'] / pa
    net_gb = (gb - fb - pu) / pa if pa else 0.0
    x = (6.145 - 16.986 * so_pa + 11.434 * bb_pa - 1.858 * net_gb
         + 7.653 * so_pa ** 2
         + (6.664 if net_gb < 0 else -6.664) * net_gb ** 2
         + 10.130 * so_pa * net_gb - 5.195 * bb_pa * net_gb)
    return x


def feature_rows(season, scope):
    """{pid: {feature: value}} for one season and scope ('full'|'h1')."""
    sk = str(season)
    rows = {}
    bat = BATTERY.get(sk, {})
    cl = CMDLOC.get(sk, {})
    stf = STUFF.get(sk, {})
    hrfb = lg_hrfb(season)
    for pid, rec in bat.items():
        m = dict(rec[scope])          # copy battery metrics
        kc = m.pop('k_counts')
        line = official_line(season, scope, pid)
        r = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        r['pa'] = m['pa']
        # official-count rates + constructs (need IP/BF from the line)
        if line and line['bf'] > 0 and line['outs'] > 0:
            bf, outs = line['bf'], line['outs']
            ip = outs / 3.0
            r['k_pct_off'] = line['so'] / bf
            r['bb_pct_off'] = line['bb'] / bf
            r['kbb_off'] = (line['so'] - line['bb']) / bf
            r['hr9_off'] = line['hr'] * 27.0 / outs
            r['fip_core'] = (13 * line['hr']
                             + 3 * (line['bb'] + line['hbp'])
                             - 2 * line['so']) / ip
            r['xfip_core'] = (13 * (kc['fb'] * hrfb)
                              + 3 * (line['bb'] + line['hbp'])
                              - 2 * line['so']) / ip
            r['kwera_core'] = -(line['so'] - line['bb']) / bf
            r['babip_off'] = ((line['h'] - line['hr'])
                              / (line['ab'] - line['so'] - line['hr']
                                 + line['sf'])
                              if (line['ab'] - line['so'] - line['hr']
                                  + line['sf']) > 0 else None)
            wnum = (0.69 * (line['bb'] - line['ibb']) + 0.72 * line['hbp']
                    + 0.89 * (line['h'] - line['d2'] - line['d3']
                              - line['hr'])
                    + 1.27 * line['d2'] + 1.61 * line['d3']
                    + 2.10 * line['hr'])
            wden = (line['ab'] + line['bb'] - line['ibb'] + line['sf']
                    + line['hbp'])
            r['woba_off'] = wnum / wden if wden > 0 else None
        r['siera_core'] = siera(kc, m['pa'], kc['gb'], kc['fb'], kc['pu'],
                                None)
        # internal metrics
        irec = cl.get(pid, {})
        suf = 'full' if scope == 'full' else 'h1'
        if f'loc_{suf}' in irec:
            # score_pitch currency is batter-positive run value; negate so
            # higher = better, matching the site's Loc+ (the 2026 fallback)
            r['loc_raw'] = -irec[f'loc_{suf}']
        if f'cmd_{suf}' in irec:
            r['cmd_miss'] = irec[f'cmd_{suf}']
        srec = stf.get(pid, {})
        if f'stuff_{suf}' in srec:
            r['stuff_raw'] = srec[f'stuff_{suf}']
        elif r.get('stuff_plus') is not None:     # 2026 sheet values
            r['stuff_raw'] = r['stuff_plus']
        if r.get('loc_raw') is None and r.get('loc_plus') is not None:
            r['loc_raw'] = r['loc_plus']
        rows[pid] = {k: v for k, v in r.items() if v is not None}
    # composite pitching+ = 0.8 z(stuff) + 0.2 z(loc), within season+scope
    for key_out, k1, k2, w1, w2 in (
            ('pitchingplus_z', 'stuff_raw', 'loc_raw', 0.8, 0.2),):
        vals1 = [r[k1] for r in rows.values() if k1 in r]
        vals2 = [r[k2] for r in rows.values() if k2 in r]
        if len(vals1) < 30 or len(vals2) < 30:
            continue
        m1 = sum(vals1) / len(vals1)
        s1 = math.sqrt(sum((v - m1) ** 2 for v in vals1) / len(vals1))
        m2 = sum(vals2) / len(vals2)
        s2 = math.sqrt(sum((v - m2) ** 2 for v in vals2) / len(vals2))
        for r in rows.values():
            if k1 in r and k2 in r and s1 > 0 and s2 > 0:
                r[key_out] = (w1 * (r[k1] - m1) / s1
                              + w2 * (r[k2] - m2) / s2)
    return rows


def run_test(feat_rows, targ, gate_outs, tkey):
    """-> {feature: (r, n)} at one gate for one target currency."""
    by_feat = defaultdict(lambda: ([], []))
    for pid, feats in feat_rows.items():
        t = targ.get(pid)
        if t is None or t['outs'] < gate_outs:
            continue
        for f, v in feats.items():
            xs, ys = by_feat[f]
            xs.append(v)
            ys.append(t[tkey])
    out = {}
    for f, (xs, ys) in by_feat.items():
        r = pearson(xs, ys)
        if r is not None:
            out[f] = (r, len(xs))
    return out


def main():
    results = {'desc': {}, 'next': {}, 'ros': {}}

    # DESC: full-season metric vs same-season target
    for season in SEASONS:
        fr = feature_rows(season, 'full')
        targ = targets_for(season, 'full')
        results['desc'][season] = {}
        for tkey in ('era', 'pera', 'ra9'):
            results['desc'][season][tkey] = {}
            for g in GATES_IP:
                results['desc'][season][tkey][g] = run_test(
                    fr, targ, g * 3, tkey)

    # NEXT: season-N full metric vs season-N+1 full target
    for season in SEASONS[:-1]:
        nxt = season + 1
        fr = feature_rows(season, 'full')
        targ_n = targets_for(nxt, 'full')
        targ_c = targets_for(season, 'full')
        results['next'][season] = {}
        for tkey in ('era', 'pera', 'ra9'):
            results['next'][season][tkey] = {}
            for g in GATES_IP:
                # gate BOTH sides: current-season outs and next-season outs
                eligible = {pid: f for pid, f in fr.items()
                            if targ_c.get(pid, {}).get('outs', 0) >= g * 3}
                results['next'][season][tkey][g] = run_test(
                    eligible, targ_n, g * 3, tkey)

    # ROS: h1 metric vs h2 target, both halves gated at half the IP gate
    for season in SEASONS:
        fr = feature_rows(season, 'h1')
        targ_h2 = targets_for(season, 'h2')
        targ_h1 = targets_for(season, 'h1')
        results['ros'][season] = {}
        for tkey in ('era', 'pera', 'ra9'):
            results['ros'][season][tkey] = {}
            for g in GATES_IP:
                half_gate = g * 3 // 2
                eligible = {pid: f for pid, f in fr.items()
                            if targ_h1.get(pid, {}).get('outs', 0)
                            >= half_gate}
                results['ros'][season][tkey][g] = run_test(
                    eligible, targ_h2, half_gate, tkey)

    out_path = os.path.join(ROOT, 'data', '_era_screen_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f'wrote {out_path}')

    # ── console summary: mean |r| across replicate seasons at MAIN_GATE ──
    for test, label in (('desc', 'DESCRIPTIVE (same season)'),
                        ('next', 'PREDICTIVE (next season)'),
                        ('ros', 'PREDICTIVE (rest of season)')):
        print(f'\n=== {label} — ERA target, {MAIN_GATE}+ IP gate ===')
        acc = defaultdict(list)
        for season, tt in results[test].items():
            for f, (r, n) in tt['era'][MAIN_GATE].items():
                acc[f].append(r)
        rank = sorted(acc.items(), key=lambda kv:
                      -abs(sum(kv[1]) / len(kv[1])))
        for f, rs in rank[:25]:
            mean_r = sum(rs) / len(rs)
            spread = (f'[{min(rs):+.3f},{max(rs):+.3f}]'
                      if len(rs) > 1 else '')
            print(f'  {f:<16} mean r {mean_r:+.3f}  ({len(rs)} seasons) '
                  f'{spread}')


if __name__ == '__main__':
    main()
