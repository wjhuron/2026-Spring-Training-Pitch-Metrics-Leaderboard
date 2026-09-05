"""war_rate_validation.py — which rate should a pitcher WAR run on? (2026-09-05)

Three WARs under IDENTICAL conventions, differing only in the rate:
  RA9    actual runs per 9 (the bWAR idea, no defense adjustment)
  FIPR9  FIP on the runs scale (the fWAR idea)
  hdR9   hdERA on the runs scale (deserved: shrunk xwOBA against, DH_B 0.917)
Conventions (see the 2026-09-05 formula sheet):
  rate_dp   = rate - PASS * (exposure - 1) * lgRA9      exposure = mean over stint clubs of (PF/100 + 1)/2
  RAA       = (lgRA9 - rate_dp) * IP/9
  RPW       = 4 r / (2 r)^0.287, r = lgRA9                PythagenPat, checked against team records
  WAR       = RAA / RPW + REPL(role) * IP/9,  REPL = REPL_RP + (REPL_SP - REPL_RP) * GS/G
  REPL_SP   = 0.12 wins/9 (fWAR, convention that sets the league total)
  REPL_RP   = REPL_SP - GAP/RPW, GAP = 0.64 runs/9 measured within pitcher (role changes 2021-2026);
              fWAR's published RP level 0.03 (gap ~0.85) reported alongside
PASS for hdR9 is MEASURED here (LOSO innings-weighted regression of hdR9 on park exposure);
RA9 and FIPR9 take PASS = 1 as their parents do, and their measured pass-through is printed as a check.
Tests (rebuilt replicates data/_era_targets.json + _era_battery.json, 2021-2026):
  reliability   corr of the rate on chronological halves (>= 30 IP each), six seasons
  next          rate(Y) -> RA9(Y+1), gates 60 and 30 IP both sides, five pairs   <- the decider
  ros           rate(h1) -> RA9(h2), six seasons
  disagreement  top-50 |WAR_FIP - WAR_RA9| pitchers per season: which rate next season confirms
  sums          league WAR totals per season under the measured and the published replacement gap
Usage: python3 scripts/research/era/war_rate_validation.py
Output: console + data/_war_rate_validation.json
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from pipeline.eraplus import DH_B, N0_XW, POOL_MIN_OUTS

T = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))
B = json.load(open(os.path.join(ROOT, 'data', '_era_battery.json')))
PF = json.load(open(os.path.join(ROOT, 'data', 'park_factors.json')))
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
REPL_SP, GAP_MEASURED, GAP_PUBLISHED = 0.12, 0.64, 0.85
PYTH_EXP = 0.287

def rpw_of(lg_ra9):
    r = lg_ra9
    return 4.0 * r / (2.0 * r) ** PYTH_EXP

def exposure(y, teams):
    pfs = PF.get(str(y), {})
    vals = [(pfs.get(str(t), 100.0) / 100.0 + 1.0) / 2.0 for t in (teams or [])]
    return float(np.mean(vals)) if vals else 1.0

def season_table(y):
    """Per pitcher: outs/r/er/g/gs, xwOBA (full,h1,h2), FIP components (full,h1,h2), exposure."""
    rows = {}
    ph = T[str(y)]['pitchers']; bb = B.get(str(y), {})
    for pid, t in ph.items():
        b = bb.get(pid)
        if not b or t['outs'] <= 0: continue
        f, h = b['full'], b.get('h1') or {}
        kc, kh = f['k_counts'], (h.get('k_counts') or {})
        den_f = f['pa'] - kc['ibb'] - kc['sh'] - kc['ci']
        den_h = (h.get('pa') or 0) - kh.get('ibb', 0) - kh.get('sh', 0) - kh.get('ci', 0) if h else 0
        xw_f = f.get('xwoba'); xw_h = h.get('xwoba') if h else None
        r = dict(outs=t['outs'], r=t['r'], er=t['er'], g=t['g'], gs=t['gs'], hand=t.get('hand'), name=t.get('name'),
                 h1=t.get('h1') or {}, h2=t.get('h2') or {},
                 xw=xw_f, xw_den=den_f, xw_h1=xw_h, xw_den_h1=den_h,
                 fip_num=13 * kc['hr'] + 3 * (kc['bb'] + kc['hbp']) - 2 * kc['k'],
                 fip_num_h1=(13 * kh.get('hr', 0) + 3 * (kh.get('bb', 0) + kh.get('hbp', 0)) - 2 * kh.get('k', 0)) if h else None,
                 exp=exposure(y, t.get('teams')))
        if xw_f is not None and xw_h is not None and den_f > den_h > 0:
            r['xw_h2'] = (xw_f * den_f - xw_h * den_h) / (den_f - den_h); r['xw_den_h2'] = den_f - den_h
        else:
            r['xw_h2'] = None; r['xw_den_h2'] = 0
        r['fip_num_h2'] = (r['fip_num'] - r['fip_num_h1']) if r['fip_num_h1'] is not None else None
        rows[pid] = r
    return rows

def league(rows):
    outs = sum(r['outs'] for r in rows.values()); R = sum(r['r'] for r in rows.values()); ER = sum(r['er'] for r in rows.values())
    lg_ra9, lg_era = R * 27 / outs, ER * 27 / outs
    fipn = sum(r['fip_num'] for r in rows.values()); cfip = lg_era - fipn * 3 / outs
    den = sum(r['xw_den'] for r in rows.values() if r['xw'] is not None); lg_xw = sum(r['xw'] * r['xw_den'] for r in rows.values() if r['xw'] is not None) / den
    return dict(ra9=lg_ra9, era=lg_era, cfip=cfip, xw=lg_xw, rpw=rpw_of(lg_ra9))

def hd_rate(xw, den, lg, pool_z):
    """hdERA on the runs scale from one xwOBA-against value: shrink at N0_XW, z against the 30-IP pool, anchor + DH_B * z, + ER->R gap."""
    if xw is None or den <= 0: return None
    sh = (xw * den + N0_XW * lg['xw']) / (den + N0_XW)
    mu, sd, anchor = pool_z
    return anchor + DH_B * (sh - mu) / sd + (lg['ra9'] - lg['era'])

def pool_stats(rows, lg, scope='full'):
    """(mu, sd) of shrunk xw over the 30-IP pool, and the pool's mean ERA (production apply_era_plus)."""
    pool = [r for r in rows.values() if r['outs'] >= POOL_MIN_OUTS and r['xw'] is not None]
    sh = [(r['xw'] * r['xw_den'] + N0_XW * lg['xw']) / (r['xw_den'] + N0_XW) for r in pool]
    return float(np.mean(sh)), float(np.std(sh)), float(np.mean([r['er'] * 27 / r['outs'] for r in pool]))

def rates_full(rows, lg, pz):
    out = {}
    for pid, r in rows.items():
        o = r['outs']
        out[pid] = dict(RA9=r['r'] * 27 / o, FIPR9=r['fip_num'] * 3 / o + lg['cfip'] + (lg['ra9'] - lg['era']),
                        hdR9=hd_rate(r['xw'], r['xw_den'], lg, pz), ip=o / 3.0, exp=r['exp'], g=r['g'], gs=r['gs'])
    return out

def rates_half(rows, lg, pz, half):
    out = {}
    for pid, r in rows.items():
        h = r[half]; o = h.get('outs') or 0
        if o <= 0: continue
        fn = r['fip_num_h1'] if half == 'h1' else r['fip_num_h2']
        xw, den = (r['xw_h1'], r['xw_den_h1']) if half == 'h1' else (r['xw_h2'], r['xw_den_h2'])
        out[pid] = dict(RA9=h['r'] * 27 / o, FIPR9=(fn * 3 / o + lg['cfip'] + (lg['ra9'] - lg['era'])) if fn is not None else None,
                        hdR9=hd_rate(xw, den, lg, pz), ip=o / 3.0)
    return out

def wls_slope(x, y, w):
    x, y, w = map(np.asarray, (x, y, w)); mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w * (x - mx) ** 2))

def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float('nan')

def main():
    out = {}
    tabs = {y: season_table(y) for y in SEASONS}
    lgs = {y: league(tabs[y]) for y in SEASONS}
    pzs = {y: pool_stats(tabs[y], lgs[y]) for y in SEASONS}
    full = {y: rates_full(tabs[y], lgs[y], pzs[y]) for y in SEASONS}
    print("league rates: " + "  ".join(f"{y}: RA9 {lgs[y]['ra9']:.2f} ERA {lgs[y]['era']:.2f} RPW {lgs[y]['rpw']:.2f}" for y in SEASONS))
    # RPW check against team records
    try:
        import requests
        chk = []
        for y in SEASONS:
            js = requests.get('https://statsapi.mlb.com/api/v1/standings', params={'leagueId': '103,104', 'season': y, 'standingsTypes': 'regularSeason'}, timeout=30).json()
            W, RD = [], []
            for rec in js.get('records', []):
                for tr in rec.get('teamRecords', []):
                    W.append(tr['wins']); RD.append(tr.get('runsScored', 0) - tr.get('runsAllowed', 0))
            sl = np.polyfit(RD, W, 1)[0]
            chk.append((y, 1 / sl, lgs[y]['rpw'], len(W)))
        print("RPW check, team records (1 / slope of wins on run differential) vs PythagenPat: " + "  ".join(f"{y}: {a:.2f} vs {b:.2f} (n {n})" for y, a, b, n in chk))
        out['rpw_check'] = chk
    except Exception as e:
        print(f"RPW check skipped: {e}")

    # park pass-through, LOSO innings-weighted, per rate
    print("\npark pass-through (LOSO slope of rate on (exposure-1)*lgRA9, innings-weighted, >=30 IP): 1.0 = full park factor")
    pas = {k: [] for k in ('RA9', 'FIPR9', 'hdR9')}
    for hold in SEASONS:
        for k in pas:
            x, yv, w = [], [], []
            for y in SEASONS:
                if y == hold: continue
                for pid, r in full[y].items():
                    if r['ip'] >= 30 and r[k] is not None:
                        x.append((r['exp'] - 1) * lgs[y]['ra9']); yv.append(r[k]); w.append(r['ip'])
            pas[k].append(wls_slope(x, yv, w))
    for k in pas:
        print(f"  {k:6} " + " ".join(f"{v:+.2f}" for v in pas[k]) + f"   mean {np.mean(pas[k]):+.2f}")
    out['park_pass'] = {k: pas[k] for k in pas}
    PASS = {'RA9': 1.0, 'FIPR9': 1.0, 'hdR9': float(np.mean(pas['hdR9']))}
    print(f"  applied: RA9 1.00, FIPR9 1.00 (parents' convention), hdR9 {PASS['hdR9']:.2f} (measured)")

    def adj(r, k, y):
        return r[k] - PASS[k] * (r['exp'] - 1) * lgs[y]['ra9'] if r[k] is not None else None

    # reliability on halves (park cancels within season for the same pitcher)
    print("\nreliability: corr of rate on chronological halves, >= 30 IP each half")
    rel = {k: [] for k in PASS}
    for y in SEASONS:
        h1 = rates_half(tabs[y], lgs[y], pzs[y], 'h1'); h2 = rates_half(tabs[y], lgs[y], pzs[y], 'h2')
        ks = [p for p in h1 if p in h2 and h1[p]['ip'] >= 30 and h2[p]['ip'] >= 30]
        line = []
        for k in PASS:
            r = pear([h1[p][k] for p in ks if h1[p][k] is not None and h2[p][k] is not None], [h2[p][k] for p in ks if h1[p][k] is not None and h2[p][k] is not None])
            rel[k].append(r); line.append(f"{k} {r:.3f}")
        print(f"  {y} (n {len(ks)}): " + "  ".join(line))
    print("  mean: " + "  ".join(f"{k} {np.mean(rel[k]):.3f}" for k in PASS)); out['reliability'] = rel

    # next season
    for gate in (60, 30):
        print(f"\nnext season: park-adjusted rate(Y) -> RA9(Y+1), >= {gate} IP both seasons")
        nxt = {k: [] for k in PASS}; nxt_p = {k: [] for k in PASS}
        for y in SEASONS[:-1]:
            a, b = full[y], full[y + 1]
            ks = [p for p in a if p in b and a[p]['ip'] >= gate and b[p]['ip'] >= gate]
            tgt = [b[p]['RA9'] for p in ks]; tgt_p = [b[p]['RA9'] - (b[p]['exp'] - 1) * lgs[y + 1]['ra9'] for p in ks]
            line = []
            for k in PASS:
                x = [adj(a[p], k, y) for p in ks]
                r = pear(x, tgt); rp = pear(x, tgt_p); nxt[k].append(r); nxt_p[k].append(rp); line.append(f"{k} {r:.3f}/{rp:.3f}")
            print(f"  {y}->{y+1} (n {len(ks)}): " + "  ".join(line) + ("   [partial 2026]" if y + 1 == 2026 else ""))
        for k in PASS:
            print(f"  mean {k:6} raw {np.mean(nxt[k]):.4f}  park-adj target {np.mean(nxt_p[k]):.4f}")
        wins = sum(1 for i in range(len(nxt['hdR9'])) if nxt['hdR9'][i] > nxt['FIPR9'][i]); wins2 = sum(1 for i in range(len(nxt['hdR9'])) if nxt['hdR9'][i] > nxt['RA9'][i])
        d = np.array(nxt['hdR9']) - np.array(nxt['FIPR9']); print(f"  hdR9 vs FIPR9: mean d {d.mean():+.4f} (SE {d.std(ddof=1)/math.sqrt(len(d)):.4f}), wins {wins}/{len(d)}; hdR9 vs RA9: wins {wins2}/{len(d)}")
        out[f'next_{gate}'] = dict(raw=nxt, park_adj=nxt_p)

    # rest of season
    print("\nrest of season: rate(h1) -> RA9(h2), >= 30 IP each half")
    ros = {k: [] for k in PASS}
    for y in SEASONS:
        h1 = rates_half(tabs[y], lgs[y], pzs[y], 'h1'); h2 = rates_half(tabs[y], lgs[y], pzs[y], 'h2')
        ks = [p for p in h1 if p in h2 and h1[p]['ip'] >= 30 and h2[p]['ip'] >= 30]
        line = []
        for k in PASS:
            r = pear([h1[p][k] for p in ks], [h2[p]['RA9'] for p in ks]); ros[k].append(r); line.append(f"{k} {r:.3f}")
        print(f"  {y} (n {len(ks)}): " + "  ".join(line))
    print("  mean: " + "  ".join(f"{k} {np.mean(ros[k]):.4f}" for k in PASS) + f";  hdR9 beats FIPR9 in {sum(1 for i in range(6) if ros['hdR9'][i] > ros['FIPR9'][i])}/6")
    out['ros'] = ros

    # WAR totals and the disagreement audit
    print("\nWAR: league totals per season (pitchers >= 1 IP) under the measured gap (RP = SP - 0.64/RPW) and the published one (RP 0.03)")
    sums = {}
    for y in SEASONS:
        rpw = lgs[y]['rpw']; tot = {}
        for gap_name, gap in (('measured', GAP_MEASURED), ('published', GAP_PUBLISHED)):
            repl_rp = REPL_SP - gap / rpw
            for k in PASS:
                s = 0.0
                for pid, r in full[y].items():
                    ra = adj(r, k, y)
                    if ra is None: continue
                    repl = repl_rp + (REPL_SP - repl_rp) * (r['gs'] / r['g'] if r['g'] else 0.0)
                    s += (lgs[y]['ra9'] - ra) * r['ip'] / 9 / rpw + repl * r['ip'] / 9
                tot[(gap_name, k)] = s
        games = sum(r['outs'] for r in tabs[y].values()) / 27 / 2
        sums[y] = {f'{g}_{k}': v for (g, k), v in tot.items()}
        print(f"  {y} (~{games:.0f} games; 43% of 1000 scaled = {430 * games / 2430:.0f}): " + "  ".join(f"{g[:4]}/{k} {v:.0f}" for (g, k), v in tot.items()))
    out['sums'] = sums
    print("\ndisagreement audit: per season the 50 pitchers (>=60 IP) where FIP-WAR and RA9-WAR differ most; corr of next-season RA9 with each rate on that set")
    dis = {k: [] for k in PASS}
    for y in SEASONS[:-1]:
        rpw = lgs[y]['rpw']; a, b = full[y], full[y + 1]
        cand = []
        for pid, r in a.items():
            if r['ip'] < 60 or pid not in b or b[pid]['ip'] < 60 or r['FIPR9'] is None or r['hdR9'] is None: continue
            wf = (lgs[y]['ra9'] - adj(r, 'FIPR9', y)) * r['ip'] / 9 / rpw; wr = (lgs[y]['ra9'] - adj(r, 'RA9', y)) * r['ip'] / 9 / rpw
            cand.append((abs(wf - wr), pid))
        top = [pid for _, pid in sorted(cand, reverse=True)[:50]]
        tgt = [b[p]['RA9'] for p in top]; line = []
        for k in PASS:
            r = pear([adj(a[p], k, y) for p in top], tgt); dis[k].append(r); line.append(f"{k} {r:.3f}")
        print(f"  {y}->{y+1}: " + "  ".join(line))
    print("  mean: " + "  ".join(f"{k} {np.mean(dis[k]):.3f}" for k in PASS)); out['disagreement'] = dis
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_rate_validation.json'), 'w'), indent=1, default=float)
    print("\nwrote data/_war_rate_validation.json")

if __name__ == '__main__':
    main()
