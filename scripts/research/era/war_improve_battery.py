"""war_improve_battery.py — can the hWAR rate be improved? Part 1 (2026-09-05):
every test that runs on the rebuilt replicates alone.

Objectives for a VALUE metric's rate (all on 2021-2026, LOSO where a fit exists):
  rel      corr of the rate on chronological halves (>= 30 IP each)      signal vs noise
  nxt60    rate(Y) -> RA9(Y+1), >= 60 IP both sides                       talent content
  nxt30    same at 30 IP
  ros      rate(h1) -> RA9(h2), >= 30 IP each half
  calib    same-season slope of actual RA9 on the rate (>= 60 IP), want ~1
A candidate must not lose calibration and must win rel and/or nxt in most replicates.

Variants:
  hd250        shipped: anchor + DH_B(.917) z(xw shrunk at 250 PA) + ER->R gap
  hdN0=k       same form, N0 = k, slope refit LOSO on RA9 (k in 0, 50, 125, 250, 500, 1000)
  xwRAA        no regression: lgRA9 + (xw - lg_xw)/wOBAscale * PA_per_9  (linear weights, unshrunk)
  xwRAA250     the same with the 250-PA shrink
  xrv          per-pitch luck-neutral xRV/100 on the RA9 scale, slope LOSO
  hd+xrv w     blends, w on hd
  hd+K+BB      anchor + b1 z(xw) + b2 z(K%) + b3 z(BB%), LOSO OLS on RA9
Also: group calibration of the shipped rate (role, hand, park, workload), the
TBF-vs-IP volume basis, and the dynamic runs-per-win effect.
Usage: python3 scripts/research/era/war_improve_battery.py
Output: console + data/_war_improve_battery.json
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import war_rate_validation as W
from pipeline.eraplus import DH_B, N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS, WAR_REPL_SP, WAR_ROLE_GAP, WAR_PYTH_EXP

SEASONS = W.SEASONS
XRV = json.load(open(os.path.join(ROOT, 'data', '_era_xrv100.json')))
GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259), 2023: (0.318, 1.204), 2024: (0.310, 1.242), 2025: (0.3131, 1.2317), 2026: (0.3165, 1.2385)}

def tables():
    out = {}
    for y in SEASONS:
        t = W.season_table(y); ph = W.T[str(y)]['pitchers']; bb = W.B.get(str(y), {}); xv = XRV.get(str(y), {})
        for pid, r in t.items():
            tg = ph[pid]; b = bb[pid]; kc = b['full']['k_counts']; kh = (b.get('h1') or {}).get('k_counts') or {}
            r.update(bf=tg['bf'], so=tg['so'], bb=tg['bb'], pitches=b['full'].get('pitches') or 0,
                     bf_h1=(tg.get('h1') or {}).get('bf', 0), so_h1=(tg.get('h1') or {}).get('so', 0), bb_h1=(tg.get('h1') or {}).get('bb', 0),
                     bf_h2=(tg.get('h2') or {}).get('bf', 0), so_h2=(tg.get('h2') or {}).get('so', 0), bb_h2=(tg.get('h2') or {}).get('bb', 0),
                     xrv=(xv.get(pid) or {}).get('full'), xrv_n=(xv.get(pid) or {}).get('n_full') or 0,
                     xrv_h1=(xv.get(pid) or {}).get('h1'), xrv_n_h1=(xv.get(pid) or {}).get('n_h1') or 0)
        out[y] = t
    return out

def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float('nan')

def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); return beta

# ── per-season feature builders: return {pid: (z-features tuple, outs, ra9)} for scope full/h1/h2 ──
def feats(t, lg, scope, n0_xw=250, n0_k=90, n0_bb=180):
    """z-scored shrunk xw, K%, BB% (pool >= 30 IP in that scope), plus raw pieces."""
    def pick(r):
        if scope == 'full':
            return r['outs'], r['r'], r['xw'], r['xw_den'], r['so'], r['bb'], r['bf'], r['xrv'], r['xrv_n']
        h = r[scope]; o = h.get('outs') or 0
        xw, den = (r['xw_h1'], r['xw_den_h1']) if scope == 'h1' else (r['xw_h2'], r['xw_den_h2'])
        so, bb, bf = r['so_' + scope], r['bb_' + scope], r['bf_' + scope]
        xrv, xn = (r['xrv_h1'], r['xrv_n_h1']) if scope == 'h1' else (None, 0)
        return o, h.get('r', 0), xw, den, so, bb, bf, xrv, xn
    raw = {}
    for pid, r in t.items():
        o, rr, xw, den, so, bb, bf, xrv, xn = pick(r)
        if o <= 0 or xw is None or den <= 0 or bf <= 0: continue
        raw[pid] = dict(outs=o, ra9=rr * 27 / o, xw=xw, den=den, k=so / bf, bbp=bb / bf, bf=bf, xrv=xrv, xn=xn)
    pool = [v for v in raw.values() if v['outs'] >= POOL_MIN_OUTS]
    lg_xw = sum(v['xw'] * v['den'] for v in pool) / sum(v['den'] for v in pool)
    lg_k = sum(v['k'] * v['bf'] for v in pool) / sum(v['bf'] for v in pool); lg_bb = sum(v['bbp'] * v['bf'] for v in pool) / sum(v['bf'] for v in pool)
    xp = [v['xrv'] for v in pool if v['xrv'] is not None]; lg_xrv = float(np.mean(xp)) if xp else 0.0
    def sh(v):
        return dict(xw=(v['xw'] * v['den'] + n0_xw * lg_xw) / (v['den'] + n0_xw), k=(v['k'] * v['bf'] + n0_k * lg_k) / (v['bf'] + n0_k),
                    bbp=(v['bbp'] * v['bf'] + n0_bb * lg_bb) / (v['bf'] + n0_bb), xrv=v['xrv'] if v['xrv'] is not None else lg_xrv)
    S = {pid: sh(v) for pid, v in raw.items()}
    mu = {k: float(np.mean([S[p][k] for p in S if raw[p]['outs'] >= POOL_MIN_OUTS])) for k in ('xw', 'k', 'bbp', 'xrv')}
    sd = {k: float(np.std([S[p][k] for p in S if raw[p]['outs'] >= POOL_MIN_OUTS])) for k in ('xw', 'k', 'bbp', 'xrv')}
    anchor = float(np.mean([v['ra9'] for v in pool]))
    out = {}
    for pid, v in raw.items():
        z = {k: (S[pid][k] - mu[k]) / sd[k] if sd[k] > 0 else 0.0 for k in ('xw', 'k', 'bbp', 'xrv')}
        out[pid] = dict(z=z, outs=v['outs'], ra9=v['ra9'], xw_raw=v['xw'], den=v['den'], bf=v['bf'], xrv_raw=v['xrv'])
    return out, dict(anchor=anchor, lg_xw=lg_xw, mu=mu, sd=sd, lg_ra9=lg['ra9'], gap=lg['ra9'] - lg['era'], scale=GUTS_Y['scale'])

def main():
    global GUTS_Y
    T = tables(); LG = {y: W.league(T[y]) for y in SEASONS}
    # PA per 9 innings, league, per season
    pa9 = {y: sum(r['bf'] for r in T[y].values()) / (sum(r['outs'] for r in T[y].values()) / 27) for y in SEASONS}
    pitches9 = {y: sum(r['pitches'] for r in T[y].values()) / (sum(r['outs'] for r in T[y].values()) / 27) for y in SEASONS}
    print("PA per 9:", {y: round(v, 2) for y, v in pa9.items()}, "| pitches per 9:", {y: round(v, 1) for y, v in pitches9.items()})
    out = {}

    def run_variant(name, n0=250, cols=('xw',), fixed_b=None, linear=None, blend=None):
        """cols: z features in the OLS (LOSO on RA9, unweighted, pool >= 30 IP);
        fixed_b: use this slope on z(xw) instead of a fit (shipped DH_B);
        linear: 'xwRAA' -> lgRA9 + (xw - lg)/scale * PA9 (no fit);
        blend: (w, other_variant_fn) not used here."""
        F = {}; META = {}
        for y in SEASONS:
            GUTS_Y = {'scale': GUTS[y][1]}
            for sc in ('full', 'h1', 'h2'):
                F[(y, sc)], META[(y, sc)] = feats(T[y], LG[y], sc, n0_xw=n0)
        # LOSO slopes on full-season pools
        beta = {}
        for hold in SEASONS:
            X, Y = [], []
            for y in SEASONS:
                if y == hold: continue
                for pid, f in F[(y, 'full')].items():
                    if f['outs'] >= 180:
                        X.append([f['z'][c] for c in cols]); Y.append(f['ra9'] - META[(y, 'full')]['anchor'])
            b = ols(np.array(X), np.array(Y)) if X else None
            beta[hold] = b
        def rate(y, sc, pid):
            f = F[(y, sc)].get(pid); m = META[(y, sc)]
            if f is None: return None
            if linear == 'xwRAA':
                return m['lg_ra9'] + (f['z']['xw'] * m['sd']['xw'] + m['mu']['xw'] - m['lg_xw']) / m['scale'] * pa9[y]
            if fixed_b is not None:
                return m['anchor'] + fixed_b * f['z']['xw'] + m['gap']
            b = beta[y]
            return m['anchor'] + b[0] + sum(b[i + 1] * f['z'][c] for i, c in enumerate(cols)) + m['gap']
        res = {}
        # rel
        rel = []
        for y in SEASONS:
            ks = [p for p in F[(y, 'h1')] if p in F[(y, 'h2')] and F[(y, 'h1')][p]['outs'] >= 90 and F[(y, 'h2')][p]['outs'] >= 90]
            rel.append(pear([rate(y, 'h1', p) for p in ks], [rate(y, 'h2', p) for p in ks]))
        res['rel'] = rel
        for gate in (60, 30):
            nx = []
            for y in SEASONS[:-1]:
                a, b = F[(y, 'full')], F[(y + 1, 'full')]
                ks = [p for p in a if p in b and a[p]['outs'] >= gate * 3 and b[p]['outs'] >= gate * 3]
                nx.append(pear([rate(y, 'full', p) for p in ks], [b[p]['ra9'] for p in ks]))
            res[f'nxt{gate}'] = nx
        ros = []
        for y in SEASONS:
            ks = [p for p in F[(y, 'h1')] if p in F[(y, 'h2')] and F[(y, 'h1')][p]['outs'] >= 90 and F[(y, 'h2')][p]['outs'] >= 90]
            ros.append(pear([rate(y, 'h1', p) for p in ks], [F[(y, 'h2')][p]['ra9'] for p in ks]))
        res['ros'] = ros
        cal = []
        for y in SEASONS:
            ks = [p for p in F[(y, 'full')] if F[(y, 'full')][p]['outs'] >= 180]
            x = np.array([rate(y, 'full', p) for p in ks]); yy = np.array([F[(y, 'full')][p]['ra9'] for p in ks])
            cal.append(float(np.polyfit(x, yy, 1)[0]))
        res['calib'] = cal
        res['beta'] = {str(k): (list(map(float, v)) if v is not None else None) for k, v in beta.items()}
        out[name] = res
        print(f"{name:12} rel {np.mean(rel):.3f}  nxt60 {np.mean(res['nxt60']):.3f}  nxt30 {np.mean(res['nxt30']):.3f}  ros {np.mean(ros):.3f}  calib {np.mean(cal):.3f}"
              + (f"  b {np.round(beta[2024][1:], 3).tolist()}" if beta[2024] is not None and fixed_b is None and linear is None else ""))
        return res

    print("\n== rate variants (means over replicates; per-replicate values in the JSON) ==")
    base = run_variant('hd250 ship', n0=250, fixed_b=DH_B)
    for n0 in (0, 50, 125, 250, 500, 1000):
        run_variant(f'hdN0={n0}', n0=n0)
    run_variant('xwRAA', n0=0, linear='xwRAA')
    run_variant('xwRAA250', n0=250, linear='xwRAA')
    run_variant('xrv', cols=('xrv',))
    run_variant('hd+xrv', cols=('xw', 'xrv'))
    run_variant('hd+K+BB', cols=('xw', 'k', 'bbp'))
    run_variant('hd+K+BB+xrv', cols=('xw', 'k', 'bbp', 'xrv'))
    # paired deltas vs shipped on the two deciders
    print("\n== paired vs hd250 ship: mean delta (wins) ==")
    for name, res in out.items():
        if name == 'hd250 ship': continue
        line = []
        for k in ('rel', 'nxt60', 'nxt30', 'ros'):
            d = np.array(res[k]) - np.array(base[k]); line.append(f"{k} {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {name:12} " + "  ".join(line))

    # ── group calibration of the shipped park-adjusted rate ──
    print("\n== group calibration of the shipped rate: actual RA9 minus deserved RA9 (park-adjusted, recentered), >= 30 IP, paired across seasons ==")
    groups = defaultdict(lambda: defaultdict(list))
    for y in SEASONS:
        GUTS_Y = {'scale': GUTS[y][1]}
        F, M = feats(T[y], LG[y], 'full', n0_xw=250)
        rows = []
        for pid, f in F.items():
            if f['outs'] < 90: continue
            r = T[y][pid]; rate = M['anchor'] + DH_B * f['z']['xw'] + M['gap'] - WAR_PARK_PASS * (r['exp'] - 1) * LG[y]['ra9']
            rows.append((pid, rate, f['ra9'], f['outs'], r))
        shift = LG[y]['ra9'] - sum(rt * o for _, rt, _, o, _ in rows) / sum(o for _, _, _, o, _ in rows)
        for pid, rt, ra9, o, r in rows:
            res = ra9 - (rt + shift); share = r['gs'] / r['g'] if r['g'] else 0
            groups['role'][('SP' if share >= 0.8 else 'RP' if share <= 0.2 else 'swing')].append((y, res, o))
            groups['hand'][r.get('hand') or '?'].append((y, res, o))
            groups['park'][('hitter' if r['exp'] > 1.02 else 'pitcher' if r['exp'] < 0.98 else 'neutral')].append((y, res, o))
            groups['ip'][('<60' if o < 180 else '60-120' if o < 360 else '120+')].append((y, res, o))
    out['group_calibration'] = {}
    for g, d in groups.items():
        line = []
        for k, v in sorted(d.items()):
            by = defaultdict(list)
            for y, res, o in v: by[y].append((res, o))
            means = [np.average([x for x, _ in vv], weights=[o for _, o in vv]) for vv in by.values()]
            line.append(f"{k} {np.mean(means):+.3f} (SE {np.std(means, ddof=1) / math.sqrt(len(means)):.3f}, n {len(v)})")
            out['group_calibration'][f'{g}:{k}'] = means
        print(f"  {g:5} " + "  ".join(line))

    # ── volume basis and dynamic RPW on 2026 ──
    y = 2026; GUTS_Y = {'scale': GUTS[y][1]}; F, M = feats(T[y], LG[y], 'full', n0_xw=250); lg = LG[y]
    rpw = 4 * lg['ra9'] / (2 * lg['ra9']) ** WAR_PYTH_EXP; repl_rp = WAR_REPL_SP - WAR_ROLE_GAP / rpw
    rows = []
    for pid, f in F.items():
        r = T[y][pid]; rate = M['anchor'] + DH_B * f['z']['xw'] + M['gap'] - WAR_PARK_PASS * (r['exp'] - 1) * lg['ra9']
        rows.append((pid, rate, f['outs'], f['bf'], r))
    shift = lg['ra9'] - sum(rt * o for _, rt, o, _, _ in rows) / sum(o for _, rt, o, _, _ in rows)
    lg_bf_per_out = sum(bf for _, _, _, bf, _ in rows) / sum(o for _, _, o, _, _ in rows)
    war_ip, war_tbf, war_dyn, names = [], [], [], []
    for pid, rt, o, bf, r in rows:
        rt += shift; ip9 = o / 27; raa = (lg['ra9'] - rt) * ip9
        repl = repl_rp + (WAR_REPL_SP - repl_rp) * (r['gs'] / r['g'] if r['g'] else 0)
        w_ip = raa / rpw + repl * ip9
        # TBF basis: the same deserved rate per PA, over the batters he actually faced
        ip9_tbf = bf / lg_bf_per_out / 27
        w_tbf = (lg['ra9'] - rt) * ip9_tbf / rpw + repl * ip9
        r_env = (rt + lg['ra9']) / 2; rpw_i = 4 * r_env / (2 * r_env) ** WAR_PYTH_EXP
        w_dyn = raa / rpw_i + repl * ip9
        war_ip.append(w_ip); war_tbf.append(w_tbf); war_dyn.append(w_dyn); names.append((r['name'], o / 3))
    war_ip, war_tbf, war_dyn = map(np.array, (war_ip, war_tbf, war_dyn))
    d = war_tbf - war_ip; top = np.argsort(-np.abs(d))[:5]
    print(f"\n== volume basis, 2026: TBF-based vs IP-based hWAR: corr {pear(war_ip, war_tbf):.4f}, mean |d| {np.abs(d).mean():.3f}, max |d| {np.abs(d).max():.2f}; sums {war_ip.sum():.1f} vs {war_tbf.sum():.1f}")
    print("   biggest movers: " + "; ".join(f"{names[i][0]} {war_ip[i]:.2f}->{war_tbf[i]:.2f}" for i in top))
    d2 = war_dyn - war_ip; top2 = np.argsort(-np.abs(d2))[:5]
    print(f"== dynamic RPW, 2026: corr {pear(war_ip, war_dyn):.4f}, mean |d| {np.abs(d2).mean():.3f}, max |d| {np.abs(d2).max():.2f}; sums {war_ip.sum():.1f} vs {war_dyn.sum():.1f}")
    print("   biggest movers: " + "; ".join(f"{names[i][0]} {war_ip[i]:.2f}->{war_dyn[i]:.2f}" for i in top2))
    out['volume_2026'] = dict(corr=pear(war_ip, war_tbf), mean_abs=float(np.abs(d).mean()), max_abs=float(np.abs(d).max()))
    out['dynrpw_2026'] = dict(corr=pear(war_ip, war_dyn), mean_abs=float(np.abs(d2).mean()), max_abs=float(np.abs(d2).max()))
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_improve_battery.json'), 'w'), indent=1, default=float)
    print("\nwrote data/_war_improve_battery.json")

if __name__ == '__main__':
    GUTS_Y = {'scale': 1.24}
    main()
