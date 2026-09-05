"""war_error_bar.py — how wide is hWAR's error bar, really (2026-09-05).

The shipped hWAR_se carries per-PA xwOBA sampling noise only:
    se = shrink x WAR_XW_PA_SD x sqrt(PA) / wOBA scale / RPW.
Three questions:
  1. SCALE. hdERA converts xwOBA to runs at DH_B / sd(pool) per z, the shipped se at
     PA9 / wOBA scale per xwOBA point. Are those the same runs per xwOBA point?
  2. CALIBRATION. Split-half check: the observed variance of (h1 rate - h2 rate) across
     pitchers against the variance the formula predicts, per season and by PA bin.
     A ratio above 1 means the bar understates even the sampling part.
  3. THE OTHER TERMS on the live 2026 rows: the published park factor's own noise
     (from the year-to-year change of the rolling factor), the pass-through SE (jackknife
     over the LOSO folds), the runs-scale slope SE (jackknife over LOSO DH_B fits), the
     role-gap SE (.09 runs/9, relievers only), and the recentering shift SE. Reported as
     independent (moves one pitcher against another) and shared (moves everyone together).
Usage: python3 scripts/research/era/war_error_bar.py
Output: console + data/_war_error_bar.json
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import war_rate_validation as W
from pipeline.eraplus import (DH_B, N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS, WAR_XW_PA_SD, WAR_PYTH_EXP, XRV_SCALE, _ip_outs)
from pipeline.utils import TEAM_ABBREV_TO_ID, AAA_TEAMS

SEASONS = W.SEASONS; T = W.T; PF = W.PF
SCALE = {2021: 1.209, 2022: 1.259, 2023: 1.204, 2024: 1.242, 2025: 1.2317, 2026: 1.2385}
ROLE_GAP_SE = 0.09          # runs/9, within-season swingman measurement (war_improve_battery2)
SIG = WAR_XW_PA_SD
MIN_PA_HALF = 100


def season_bits(y):
    t = W.season_table(y); lg = W.league(t); mu, sd, anchor = W.pool_stats(t, lg)
    ph = T[str(y)]['pitchers']
    pa9 = sum(ph[p]['bf'] for p in t) / (sum(t[p]['outs'] for p in t) / 27)
    return t, lg, mu, sd, anchor, pa9


def main():
    out = {}
    S = {y: season_bits(y) for y in SEASONS}
    print("1. RUNS PER xwOBA POINT (per 9 IP): hdERA's DH_B / sd(pool) vs the shipped se's PA9 / wOBA scale")
    for y in SEASONS:
        t, lg, mu, sd, anchor, pa9 = S[y]
        a_hd = DH_B / sd; a_se = pa9 / SCALE[y]
        out[f'scale_{y}'] = dict(pool_sd=sd, pa9=pa9, runs_per_xw_hd=a_hd, runs_per_xw_se=a_se)
        print(f"  {y}: pool sd {sd:.4f}  PA/9 {pa9:.1f}  hdERA {a_hd:.1f}  shipped se {a_se:.1f}  ratio hd/se {a_hd / a_se:.3f}")

    print("\n2. SPLIT-HALF CALIBRATION: var(observed h1 - h2 rate) / var(predicted), both halves >= 100 PA")
    print("   pred uses hdERA's scale (DH_B/sd) [and the shipped scale in brackets]; ratio > 1 = bar too narrow")
    ratios, ratios_ship = [], []
    bins = [(100, 200), (200, 320), (320, 10000)]; binned = {b: ([], []) for b in bins}
    for y in SEASONS:
        t, lg, mu, sd, anchor, pa9 = S[y]
        d_obs, v_hd, v_sh = [], [], []
        for pid, r in t.items():
            x1, n1, x2, n2 = r['xw_h1'], r['xw_den_h1'], r['xw_h2'], r['xw_den_h2']
            if x1 is None or x2 is None or n1 < MIN_PA_HALF or n2 < MIN_PA_HALF:
                continue
            s1, s2 = n1 / (n1 + N0_XW), n2 / (n2 + N0_XW)
            sh1 = (x1 * n1 + N0_XW * lg['xw']) / (n1 + N0_XW); sh2 = (x2 * n2 + N0_XW * lg['xw']) / (n2 + N0_XW)
            d = DH_B * (sh1 - sh2) / sd
            noise = SIG ** 2 * (s1 ** 2 / n1 + s2 ** 2 / n2)
            d_obs.append(d); v_hd.append((DH_B / sd) ** 2 * noise); v_sh.append((pa9 / SCALE[y]) ** 2 * noise)
            for b in bins:
                if b[0] <= min(n1, n2) < b[1]:
                    binned[b][0].append(d); binned[b][1].append((DH_B / sd) ** 2 * noise)
        d_obs = np.array(d_obs); r_hd = d_obs.var() / np.mean(v_hd); r_sh = d_obs.var() / np.mean(v_sh)
        ratios.append(r_hd); ratios_ship.append(r_sh)
        print(f"  {y}: n {len(d_obs):4d}  sd(obs diff) {d_obs.std():.3f} runs/9  ratio {r_hd:.3f} [{r_sh:.3f}]")
    print(f"  mean ratio {np.mean(ratios):.3f} [{np.mean(ratios_ship):.3f}]  -> sampling bar x {math.sqrt(np.mean(ratios)):.3f} [{math.sqrt(np.mean(ratios_ship)):.3f}]")
    for b in bins:
        d, v = np.array(binned[b][0]), np.array(binned[b][1])
        print(f"  min-half PA {b[0]}-{b[1] if b[1] < 10000 else '+':>4}: n {len(d):4d}  ratio {d.var() / v.mean():.3f}")
    out['split_half'] = dict(ratio_hd=ratios, ratio_ship=ratios_ship, by_bin={f'{b[0]}-{b[1]}': float(np.array(binned[b][0]).var() / np.mean(binned[b][1])) for b in bins})
    CAL = math.sqrt(np.mean(ratios))

    print("\n3a. PARK FACTOR NOISE: year-to-year change of the published rolling factor (3-year windows only)")
    diffs = []
    for y in range(2021, 2026):
        a, b = PF[str(y)], PF[str(y + 1)]; wa, wb = PF['_window'][str(y)], PF['_window'][str(y + 1)]
        for club in a:
            if club in b and wa.get(club) == 3 and wb.get(club) == 3:
                diffs.append(b[club] - a[club])
    diffs = np.array(diffs); sd_f = math.sqrt(1.5) * diffs.std()
    print(f"  {len(diffs)} club-year pairs, sd of the change {diffs.std():.2f} points -> sd of one published factor {sd_f:.2f} points "
          f"(F_t and F_t+1 share two of three seasons: var(F) = 1.5 var(diff); real change is folded in, so this is an upper bound)")
    out['park_sd_points'] = sd_f

    v = json.load(open(os.path.join(ROOT, 'data', '_war_rate_validation.json')))['park_pass']['hdR9']
    n = len(v); se_pass = math.sqrt((n - 1) / n * sum((x - np.mean(v)) ** 2 for x in v))
    print(f"3b. PASS-THROUGH SE (jackknife over {n} LOSO folds {np.round(v, 3).tolist()}): {se_pass:.3f} around {np.mean(v):.3f}")
    out['pass_se'] = se_pass

    print("3c. DH_B SE: LOSO refits of the ERA-on-z slope over the 30-IP pool")
    folds = []
    for hold in SEASONS:
        X, Y = [], []
        for y in SEASONS:
            if y == hold:
                continue
            t, lg, mu, sd, anchor, pa9 = S[y]
            for pid, r in t.items():
                if r['outs'] >= POOL_MIN_OUTS and r['xw'] is not None:
                    sh = (r['xw'] * r['xw_den'] + N0_XW * lg['xw']) / (r['xw_den'] + N0_XW)
                    X.append((sh - mu) / sd); Y.append(r['er'] * 27 / r['outs'] - anchor)
        folds.append(float(np.polyfit(X, Y, 1)[0]))
    se_dhb = math.sqrt((len(folds) - 1) / len(folds) * sum((f - np.mean(folds)) ** 2 for f in folds))
    print(f"  folds {np.round(folds, 3).tolist()}  mean {np.mean(folds):.3f} (shipped {DH_B})  jackknife SE {se_dhb:.3f} = {se_dhb / DH_B * 100:.1f}% of the scale")
    out['dhb_folds'] = folds; out['dhb_se'] = se_dhb

    print("\n4. LIVE 2026 ROWS: the components in WAR")
    rows = json.load(open(os.path.join(ROOT, 'data', 'pitcher_leaderboard_rs.json')))
    meta = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
    lg_ra9 = meta['pitcherLeagueAverages']['lgRA9']; rpw = 4.0 * lg_ra9 / (2.0 * lg_ra9) ** WAR_PYTH_EXP
    anchor26 = meta['eraPlusConstants']['anchor']
    t26, lg26, mu26, sd26, anc26, pa926 = S[2026]
    live = []
    for r in rows:
        if r.get('team') in AAA_TEAMS or r.get('hWAR') is None or r.get('hdERA') is None:
            continue
        pa = r.get('pa') or r.get('tbf') or 0; o = _ip_outs(r.get('ip')); g = r.get('g') or 0; gs = r.get('gs') or 0
        if pa <= 0 or o <= 0:
            continue
        ip9 = o / 27.0; s = pa / (pa + N0_XW)
        cid = TEAM_ABBREV_TO_ID.get(r.get('team')); pf = PF['2026'].get(str(cid), 100.0) if cid else 100.0
        exp = (pf / 100.0 + 1.0) / 2.0
        z = (r['hdERA'] - anchor26) / DH_B
        c = dict(name=r.get('pitcher'), team=r.get('team'), ip=o / 3.0, pa=pa, gs=gs, g=g, hWAR=r['hWAR'])
        c['samp_ship'] = s * SIG * math.sqrt(pa) / XRV_SCALE / rpw
        c['samp_hd'] = (DH_B / sd26) * s * SIG / math.sqrt(pa) * ip9 / rpw
        c['samp_cal'] = c['samp_hd'] * CAL
        c['park'] = WAR_PARK_PASS * 0.5 * sd_f / 100.0 * lg_ra9 * ip9 / rpw
        c['role'] = ROLE_GAP_SE * (1.0 - (gs / g if g else 0.0)) * ip9 / rpw
        c['pass'] = se_pass * abs(exp - 1.0) * lg_ra9 * ip9 / rpw
        c['dhb'] = se_dhb * abs(z) * ip9 / rpw
        c['_rate_se'] = (DH_B / sd26) * s * SIG / math.sqrt(pa) * CAL; c['_o'] = o; c['_club'] = not str(r.get('team', '')).endswith('TM')
        live.append(c)
    club = [c for c in live if c['_club']]; O = sum(c['_o'] for c in club)
    sd_shift = math.sqrt(sum((c['_o'] / O) ** 2 * c['_rate_se'] ** 2 for c in club))
    print(f"  shift SE {sd_shift:.4f} runs/9 (shared by every row)")
    for c in live:
        c['shift'] = sd_shift * c['ip'] / 9 / rpw
        c['indep'] = math.sqrt(c['samp_cal'] ** 2 + c['park'] ** 2 + c['role'] ** 2)
        c['all'] = math.sqrt(c['indep'] ** 2 + c['pass'] ** 2 + c['dhb'] ** 2 + c['shift'] ** 2)
    cols = ('samp_ship', 'samp_hd', 'samp_cal', 'park', 'role', 'pass', 'dhb', 'shift', 'indep', 'all')
    print("  " + f"{'pitcher':22} {'IP':>5} {'hWAR':>5} " + " ".join(f"{k:>9}" for k in cols))
    sp = sorted([c for c in live if c['g'] and c['gs'] / c['g'] >= 0.8], key=lambda c: -c['ip'])
    rp = sorted([c for c in live if c['gs'] == 0], key=lambda c: -c['ip'])
    picks = [c for c in live if any(k in (c['name'] or '') for k in ('Misiorowski', 'Cease, Dylan', 'Miller, Mason', 'Stewart, Brock'))]
    picks.append(min(sp, key=lambda c: abs(c['ip'] - 120))); picks.append(min(rp, key=lambda c: abs(c['ip'] - 30)))
    for c in picks:
        print("  " + f"{c['name'][:22]:22} {c['ip']:5.1f} {c['hWAR']:5.1f} " + " ".join(f"{c[k]:9.2f}" for k in cols))
    for lab, grp in (('SP (gs/g >= .8)', sp), ('RP (gs = 0)', rp)):
        med = {k: float(np.median([c[k] for c in grp])) for k in cols}
        print(f"  median {lab:16} n {len(grp):3d} IP {np.median([c['ip'] for c in grp]):5.1f}    " + " ".join(f"{med[k]:9.2f}" for k in cols)
              + f"   all/samp_ship {np.median([c['all'] / c['samp_ship'] for c in grp]):.2f}  indep/samp_ship {np.median([c['indep'] / c['samp_ship'] for c in grp]):.2f}")
    out['live'] = [{k: v for k, v in c.items() if not k.startswith('_')} for c in picks]
    out['medians'] = {lab: {k: float(np.median([c[k] for c in grp])) for k in cols} for lab, grp in (('SP', sp), ('RP', rp))}
    out['cal_factor'] = CAL; out['shift_se_runs9'] = sd_shift
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_error_bar.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_error_bar.json")


if __name__ == '__main__':
    main()
