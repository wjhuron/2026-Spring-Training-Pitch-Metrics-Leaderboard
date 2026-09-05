"""war_velo_bias_followup.py — the velocity bias in xwOBA against (2026-09-05 audit follow-up).

war_xwoba_bias_audit.py found actual wOBA - xwOBA on balls in play runs +.009 off pitches
under 85 mph and -.004/-.005 off 90-97 mph, replicated 5/5, and that crediting it (LOSO band
deltas per BIP before the shrink) wins every harness objective (rel 6/6, nxt60 5/5, nxt30 5/5,
ros 4/6). Before it can be a decision:
  1. mechanism: is it velocity inside a pitch type (FF 92 vs 97) or the pitch-type mix, and
     does it survive spray (a late swing goes oppo, and oppo air runs -.01)?
  2. paired SEs on the harness deltas (a win inside one SE is not a result)
  3. the curve: k x the band table for k in 0.5 .. 2, a velo x pitch-type table, a linear form
  4. live 2026 rows: who moves, and by how much hWAR
Usage: python3 scripts/research/era/war_velo_bias_followup.py
Output: console + data/_war_velo_bias_followup.json
"""
import gc, json, math, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from pipeline.utils import K_EVENTS, BB_EVENTS, HBP_EVENTS, NON_PA_EVENTS
from pipeline.eraplus import N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS, DH_B
import war_rate_validation as W
import era_battery_build as EB
import war_improve_battery2 as B2
import war_pullair_fixed as PX
import war_xwoba_bias_audit as BA

SEASONS = W.SEASONS; T = W.T
BANDS = ['v<85', 'v85-90', 'v90-94', 'v94-97', 'v97+']
PTS = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']
K_GRID = [0.5, 1.0, 1.5, 2.0]
VELO_REF = 89.0


def _table(y, pid, date, ev, xw, rs, ptype, la, bbt, hcx, hcy, stand):
    ev = pd.Series(ev)
    is_bb = ev.isin(BB_EVENTS).values; is_hbp = ev.isin(HBP_EVENTS).values; is_k = ev.isin(K_EVENTS).values
    bip = ~(is_bb | is_hbp | is_k)
    x = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, np.where(is_k, 0.0, np.nan_to_num(xw, nan=0.0))))
    w = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, ev.map(PX.W_ACT).fillna(0.0).values))
    P = pd.DataFrame(dict(pid=pid, h1=(date <= T[str(y)]['asg']), x=x, w=w, bip=bip))
    P['rs'] = np.where(bip, rs, np.nan)
    P['velo'] = np.where(bip, np.array([BA._velo(v) for v in rs], dtype=object), None)
    P['ptype'] = np.where(bip, np.array([BA._ptype(v) for v in ptype], dtype=object), None)
    P['spray'] = PX._cells(bip, la, bbt, hcx, hcy, stand)
    P['vp'] = np.where(P['velo'].notna() & P['ptype'].notna(), P['ptype'].astype(str) + '|' + P['velo'].astype(str), None)
    return P


def pa_savant(y):
    df = B2.df_year(y)
    df = df[df['game_type'] == 'R']
    d = df[df['events'].notna()][['pitcher', 'game_date', 'events', 'bb_type', 'pitch_type', 'release_speed', 'launch_angle', 'hc_x', 'hc_y',
                                  'stand', 'estimated_woba_using_speedangle']].copy()
    del df; gc.collect()
    d['ev'] = d['events'].map(EB.EVENT_MAP)
    d = d[d['ev'].notna() & ~d['ev'].isin(NON_PA_EVENTS) & ~d['ev'].isin(PX.EXCL)]
    f = lambda c: pd.to_numeric(d[c], errors='coerce').values.astype(float)
    return _table(y, d['pitcher'].astype(int).astype(str).values, d['game_date'].astype(str).str[:10].values, d['ev'].values,
                  f('estimated_woba_using_speedangle'), f('release_speed'), d['pitch_type'].values, f('launch_angle'), d['bb_type'].values,
                  f('hc_x'), f('hc_y'), d['stand'].values)


def pa_sheet(y):
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    nm = EB.build_2026_name_map()
    rows = [p for p in D if p.get('_source', 'MLB') == 'MLB' and p.get('Event') and p.get('Event') not in NON_PA_EVENTS
            and p.get('Event') not in PX.EXCL]
    del D; gc.collect()
    pid = [nm.get(p.get('Pitcher')) for p in rows]
    keep = [i for i, v in enumerate(pid) if v is not None]
    rows = [rows[i] for i in keep]; pid = np.array([str(pid[i]) for i in keep])
    f = lambda k: np.array([PX._num(p.get(k)) for p in rows], float)
    return _table(y, pid, np.array([str(p.get('Game Date'))[:10] for p in rows]), np.array([p.get('Event') for p in rows], dtype=object),
                  f('xwOBA'), f('Velocity'), np.array([p.get('Pitch Type') for p in rows], dtype=object), f('LaunchAngle'),
                  np.array([p.get('BBType') for p in rows], dtype=object), f('HC_X'), f('HC_Y'), np.array([p.get('Bats') for p in rows], dtype=object))


def cell_table(P, col, min_n=1):
    d = P[P['bip'] & P[col].notna()]; r = (d['w'] - d['x']).values; base = float(r.mean()); out = {}
    for c, g in d.groupby(col):
        if len(g) >= min_n:
            rr = (g['w'] - g['x']).values; out[c] = (float(rr.mean() - base), int(len(rr)), float(rr.std() / math.sqrt(len(rr))))
    return out


def main():
    out = {}
    TAB = {y: W.season_table(y) for y in SEASONS}; LG = {y: W.league(TAB[y]) for y in SEASONS}
    P = {y: (pa_savant(y) if y < 2026 else pa_sheet(y)) for y in SEASONS}
    for y in SEASONS:
        gc.collect()
    ALL = pd.concat([P[y] for y in SEASONS if y < 2026])
    print("1a. VELOCITY BANDS WITHIN PITCH TYPE, actual - xwOBA per BIP, recentered within the type (pooled 2021-2025, cells >= 1500 BIP)")
    print("    type   " + "  ".join(f"{b:>14}" for b in BANDS))
    out['within_type'] = {}
    for pt in PTS:
        sub = ALL[ALL['ptype'] == pt]; tab = cell_table(sub, 'velo', 1500)
        out['within_type'][pt] = tab
        print(f"    {pt:5}  " + "  ".join((f"{tab[b][0]:+.4f}/{tab[b][1]:6d}" if b in tab else f"{'':>14}") for b in BANDS))
    print("1b. SPRAY MEDIATION: velocity band deltas after removing the spray-cell mean (two-way), and spray deltas after removing velocity")
    d = ALL[ALL['bip'] & ALL['velo'].notna() & ALL['spray'].notna()].copy(); d['r'] = d['w'] - d['x']
    raw_v = d.groupby('velo')['r'].mean() - d['r'].mean()
    d['r_s'] = d['r'] - d.groupby('spray')['r'].transform('mean')
    adj_v = d.groupby('velo')['r_s'].mean() - d['r_s'].mean()
    raw_s = d.groupby('spray')['r'].mean() - d['r'].mean()
    d['r_v'] = d['r'] - d.groupby('velo')['r'].transform('mean')
    adj_s = d.groupby('spray')['r_v'].mean() - d['r_v'].mean()
    print("    velo band   raw     after spray removed")
    for b in BANDS:
        print(f"    {b:9} {raw_v[b]:+.4f}   {adj_v[b]:+.4f}")
    print("    spray cell  raw     after velocity removed")
    for c in PX.CELLS:
        print(f"    {c:10} {raw_s[c]:+.4f}   {adj_s[c]:+.4f}")
    sh = d.groupby('velo')['spray'].apply(lambda s: (s == 'air_pull').mean() / max((s.astype(str).str.startswith('air_')).mean(), 1e-9))
    print("    air-pull share of air BIP by band: " + "  ".join(f"{b} {sh[b]:.3f}" for b in BANDS))
    out['mediation'] = dict(raw_v=raw_v.to_dict(), adj_v=adj_v.to_dict(), raw_s=raw_s.to_dict(), adj_s=adj_s.to_dict(), pull_share_by_band=sh.to_dict())

    # LOSO tables
    LOSO = {}
    for y in SEASONS:
        others = pd.concat([P[s] for s in SEASONS if s != y and s < 2026])
        band = {c: v[0] for c, v in cell_table(others, 'velo').items()}
        vp = {c: v[0] for c, v in cell_table(others, 'vp', 500).items()}
        o = others[others['bip'] & others['rs'].notna()]; r = (o['w'] - o['x']).values
        slope = float(np.polyfit(o['rs'].values - VELO_REF, r - r.mean(), 1)[0])
        LOSO[y] = dict(band=band, vp=vp, slope=slope)
    print("\n    LOSO linear slope of (actual - xwOBA) on velocity: " + " ".join(f"{y} {LOSO[y]['slope'] * 10:+.4f}/10mph" for y in SEASONS))
    out['loso'] = LOSO

    def adjusted(Py, y, form, k=1.0):
        x = Py['x'].values.copy(); L = LOSO[y]
        if form == 'band':
            add = np.array([L['band'].get(c, 0.0) if c is not None else 0.0 for c in Py['velo'].values])
        elif form == 'vp':
            add = np.array([L['vp'].get(c, L['band'].get(b, 0.0)) if c is not None else (L['band'].get(b, 0.0) if b is not None else 0.0)
                            for c, b in zip(Py['vp'].values, Py['velo'].values)])
        else:
            rs = Py['rs'].values; add = np.where(np.isnan(rs), 0.0, L['slope'] * (np.nan_to_num(rs, nan=VELO_REF) - VELO_REF))
            add = add - np.nanmean(add[Py['bip'].values]) if Py['bip'].any() else add
            add = np.where(Py['bip'].values, add, 0.0)
        return x + k * add

    def frame(y, sc, A):
        t = TAB[y]; F = {}
        for pid, r in t.items():
            if sc == 'full':
                o, rr = r['outs'], r['r']
            else:
                h = r[sc]; o = h.get('outs') or 0; rr = h.get('r', 0)
            a = A[sc].get(pid)
            if o <= 0 or a is None or a[1] <= 0:
                continue
            F[pid] = dict(outs=o, ra9=rr * 27 / o, xw=a[0], den=a[1], exp=r['exp'])
        pool = [f for f in F.values() if f['outs'] >= POOL_MIN_OUTS]
        lg_xw = sum(f['xw'] * f['den'] for f in pool) / sum(f['den'] for f in pool)
        for f in F.values():
            f['xw_sh'] = (f['xw'] * f['den'] + N0_XW * lg_xw) / (f['den'] + N0_XW)
        mu = float(np.mean([f['xw_sh'] for f in pool])); sd = float(np.std([f['xw_sh'] for f in pool]))
        for f in F.values():
            f['z'] = (f['xw_sh'] - mu) / sd
        return F, dict(anchor=float(np.mean([f['ra9'] for f in pool])), gap=LG[y]['ra9'] - LG[y]['era'], lg_ra9=LG[y]['ra9'], sd=sd)

    def evaluate(name, FS):
        beta = {}
        for hold in SEASONS:
            X, Y = [], []
            for y in SEASONS:
                if y == hold:
                    continue
                F, M = FS[(y, 'full')]
                for f in F.values():
                    if f['outs'] >= 180:
                        X.append(f['z']); Y.append(f['ra9'] - M['anchor'])
            beta[hold] = np.polyfit(X, Y, 1)[::-1]

        def rate(y, sc, pid, park=False):
            F, M = FS[(y, sc)]; f = F.get(pid)
            if f is None:
                return None
            b = beta[y]; v = M['anchor'] + b[0] + b[1] * f['z'] + M['gap']
            if park:
                v -= WAR_PARK_PASS * (f['exp'] - 1) * M['lg_ra9']
            return v
        res = {'rel': [], 'ros': [], 'nxt60': [], 'nxt30': [], 'calib': []}
        for y in SEASONS:
            A, _ = FS[(y, 'h1')]; Bh, _ = FS[(y, 'h2')]
            ks = [p for p in A if p in Bh and A[p]['outs'] >= 90 and Bh[p]['outs'] >= 90]
            res['rel'].append(W.pear([rate(y, 'h1', p) for p in ks], [rate(y, 'h2', p) for p in ks]))
            res['ros'].append(W.pear([rate(y, 'h1', p) for p in ks], [Bh[p]['ra9'] for p in ks]))
            F, _ = FS[(y, 'full')]; ks = [p for p in F if F[p]['outs'] >= 180]
            res['calib'].append(float(np.polyfit([rate(y, 'full', p, park=True) for p in ks], [F[p]['ra9'] for p in ks], 1)[0]))
        for gate in (60, 30):
            for y in SEASONS[:-1]:
                A, _ = FS[(y, 'full')]; Bh, _ = FS[(y + 1, 'full')]
                ks = [p for p in A if p in Bh and A[p]['outs'] >= gate * 3 and Bh[p]['outs'] >= gate * 3]
                res[f'nxt{gate}'].append(W.pear([rate(y, 'full', p, park=True) for p in ks], [Bh[p]['ra9'] for p in ks]))
        out[name] = res
        print(f"  {name:14} rel {np.mean(res['rel']):.4f}  nxt60 {np.mean(res['nxt60']):.4f}  nxt30 {np.mean(res['nxt30']):.4f}  "
              f"ros {np.mean(res['ros']):.4f}  calib {np.mean(res['calib']):.3f}", flush=True)
        return res

    def run(name, form, k=1.0):
        FS = {}
        for y in SEASONS:
            A = PX.agg(P[y], adjusted(P[y], y, form, k) if form else P[y]['x'].values)
            for sc in ('full', 'h1', 'h2'):
                FS[(y, sc)] = frame(y, sc, A)
        return evaluate(name, FS)

    print("\n2/3. HARNESS: forms and the strength curve (k x the LOSO band table)")
    ship = run('ship', None)
    names = []
    for k in K_GRID:
        names.append(f'band k={k:.1f}'); run(names[-1], 'band', k)
    names.append('velo x type'); run('velo x type', 'vp')
    names.append('linear'); run('linear', 'linear')
    print("\nPAIRED vs ship: mean delta, paired SE over seasons, wins/n")
    for nm in names:
        res = out[nm]; line = []
        for key in ('rel', 'nxt60', 'nxt30', 'ros', 'calib'):
            dd = np.array(res[key]) - np.array(ship[key]); se = dd.std(ddof=1) / math.sqrt(len(dd))
            line.append(f"{key} {dd.mean():+.4f}±{se:.4f} ({int((dd > 0).sum())}/{len(dd)})")
        print(f"  {nm:14} " + "  ".join(line))

    print("\n4. LIVE 2026 ROWS: hWAR change under the band table (LOSO 2021-2025 deltas), at the season constant RPW")
    meta = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json'))); rpw = meta['eraPlusConstants']['war']['rpw']
    lb = {r['pitcher']: r for r in json.load(open(os.path.join(ROOT, 'data', 'pitcher_leaderboard_rs.json'))) if r.get('hWAR') is not None}
    sd26 = W.pool_stats(TAB[2026], LG[2026])[1]
    Py = P[2026]; dx = adjusted(Py, 2026, 'band') - Py['x'].values
    g = pd.DataFrame(dict(pid=Py['pid'].values, dx=dx)).groupby('pid')['dx'].agg(['mean', 'size'])
    live = []
    for pid, r in TAB[2026].items():
        if pid not in g.index or r['outs'] < 30:
            continue
        den = g.loc[pid, 'size']; s = den / (den + N0_XW); ip9 = r['outs'] / 27
        dwar = -(DH_B / sd26) * s * g.loc[pid, 'mean'] * ip9 / rpw
        row = lb.get(r.get('name')); live.append(dict(name=r.get('name'), ip=r['outs'] / 3, dwar=float(dwar), hwar=row['hWAR'] if row else None))
    live.sort(key=lambda c: c['dwar'])
    print("  pool sd of the change " + f"{np.std([c['dwar'] for c in live if c['ip'] >= 30]):.3f} WAR (>= 30 IP); largest moves:")
    for c in live[:6] + live[-8:]:
        print(f"    {str(c['name'])[:24]:24} IP {c['ip']:6.1f}  hWAR {c['hwar'] if c['hwar'] is not None else float('nan'):5.1f}  change {c['dwar']:+.2f}")
    out['live'] = live
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_velo_bias_followup.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_velo_bias_followup.json")


if __name__ == '__main__':
    main()
