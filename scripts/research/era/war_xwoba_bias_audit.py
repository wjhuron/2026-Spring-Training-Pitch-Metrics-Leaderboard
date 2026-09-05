"""war_xwoba_bias_audit.py — is Savant xwOBA calibrated where the PITCHER has control (2026-09-05)?

hWAR's deserved rate is shrunk xwOBA against. xwOBA is an EV x LA model; it can be biased in
cells the pitcher controls (pitch type, velocity, hand matchup, batted-ball type, count) and
in cells he does not (park). A bias that lines up with a pitcher trait credits that trait at
the wrong rate. For every dimension:
  1. cell deltas: mean(actual wOBA - xwOBA) per BIP cell, recentered within the dimension,
     pooled 2021-2025, with SE and the number of the five seasons that agree in sign
  2. the pitcher-level adjustment implied by his cell mix (LOSO deltas): split-half
     reliability and pool SD in runs/9
  3. the harness verdict: the deserved rate rebuilt from xwOBA + LOSO delta[cell] against
     ship on rel / nxt60 / nxt30 / ros / calib (war_improve_battery2 evaluate, one channel)
Actual wOBA uses the battery's fixed linear weights; 2026 comes from the sheet cache. Park is
the game_pk -> home club map; the published park factor is compared to the park cell delta
as context (hWAR already removes park at WAR_PARK_PASS).
Usage: python3 scripts/research/era/war_xwoba_bias_audit.py
Output: console + data/_war_xwoba_bias_audit.json
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

SEASONS = W.SEASONS; T = W.T; PF = W.PF; GPK = B2.GPK
DIMS = ['ptype', 'velo', 'hands', 'phand', 'bbtype', 'count', 'park']
PT_GROUP = {'KC': 'CU', 'CS': 'CU', 'SV': 'SL', 'FA': 'FF'}
PT_KEEP = {'FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS'}
BB_KEEP = {'ground_ball', 'line_drive', 'fly_ball', 'popup'}


def _ptype(v):
    if not isinstance(v, str):
        return None
    v = PT_GROUP.get(v, v)
    return v if v in PT_KEEP else 'OT'


def _velo(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return 'v<85' if v < 85 else 'v85-90' if v < 90 else 'v90-94' if v < 94 else 'v94-97' if v < 97 else 'v97+'


def _table(y, pid, date, ev, xw, dims):
    ev = pd.Series(ev)
    is_bb = ev.isin(BB_EVENTS).values; is_hbp = ev.isin(HBP_EVENTS).values; is_k = ev.isin(K_EVENTS).values
    bip = ~(is_bb | is_hbp | is_k)
    x = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, np.where(is_k, 0.0, np.nan_to_num(xw, nan=0.0))))
    w = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, ev.map(PX.W_ACT).fillna(0.0).values))
    P = pd.DataFrame(dict(pid=pid, h1=(date <= T[str(y)]['asg']), x=x, w=w, bip=bip))
    for d, col in dims.items():
        P[d] = np.where(bip, col, None)
    return P


def pa_savant(y):
    df = B2.df_year(y)
    df = df[df['game_type'] == 'R']
    d = df[df['events'].notna()][['pitcher', 'game_date', 'events', 'bb_type', 'pitch_type', 'release_speed', 'p_throws', 'stand',
                                  'balls', 'strikes', 'game_pk', 'estimated_woba_using_speedangle']].copy()
    del df; gc.collect()
    d['ev'] = d['events'].map(EB.EVENT_MAP)
    d = d[d['ev'].notna() & ~d['ev'].isin(NON_PA_EVENTS) & ~d['ev'].isin(PX.EXCL)]
    rs = pd.to_numeric(d['release_speed'], errors='coerce').values.astype(float)
    b = pd.to_numeric(d['balls'], errors='coerce'); s = pd.to_numeric(d['strikes'], errors='coerce')
    dims = dict(ptype=np.array([_ptype(v) for v in d['pitch_type'].values], dtype=object),
                velo=np.array([_velo(v) for v in rs], dtype=object),
                hands=np.array([f"{p}{q}" if isinstance(p, str) and isinstance(q, str) else None for p, q in zip(d['p_throws'], d['stand'])], dtype=object),
                phand=np.array([p if isinstance(p, str) else None for p in d['p_throws']], dtype=object),
                bbtype=np.array([v if v in BB_KEEP else None for v in d['bb_type'].values], dtype=object),
                count=np.array([f"{int(bb)}-{int(ss)}" if not (pd.isna(bb) or pd.isna(ss)) else None for bb, ss in zip(b, s)], dtype=object),
                park=np.array([str(GPK[int(g)]) if int(g) in GPK else None for g in d['game_pk'].values], dtype=object))
    return _table(y, d['pitcher'].astype(int).astype(str).values, d['game_date'].astype(str).str[:10].values, d['ev'].values,
                  pd.to_numeric(d['estimated_woba_using_speedangle'], errors='coerce').values.astype(float), dims)


def pa_sheet(y):
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    nm = EB.build_2026_name_map()
    rows = [p for p in D if p.get('_source', 'MLB') == 'MLB' and p.get('Event') and p.get('Event') not in NON_PA_EVENTS
            and p.get('Event') not in PX.EXCL]
    del D; gc.collect()
    pid = [nm.get(p.get('Pitcher')) for p in rows]
    keep = [i for i, v in enumerate(pid) if v is not None]
    rows = [rows[i] for i in keep]; pid = np.array([str(pid[i]) for i in keep])

    def gpk(p):
        try:
            g = int(str(p.get('PitchID')).split('_')[0])
        except ValueError:
            return None
        return str(GPK[g]) if g in GPK else None
    dims = dict(ptype=np.array([_ptype(p.get('Pitch Type')) for p in rows], dtype=object),
                velo=np.array([_velo(PX._num(p.get('Velocity'))) for p in rows], dtype=object),
                hands=np.array([f"{p.get('Throws')}{p.get('Bats')}" if p.get('Throws') in ('L', 'R') and p.get('Bats') in ('L', 'R') else None for p in rows], dtype=object),
                phand=np.array([p.get('Throws') if p.get('Throws') in ('L', 'R') else None for p in rows], dtype=object),
                bbtype=np.array([p.get('BBType') if p.get('BBType') in BB_KEEP else None for p in rows], dtype=object),
                count=np.array([p.get('Count') if isinstance(p.get('Count'), str) and '-' in p.get('Count') else None for p in rows], dtype=object),
                park=np.array([gpk(p) for p in rows], dtype=object))
    return _table(y, pid, np.array([str(p.get('Game Date'))[:10] for p in rows]), np.array([p.get('Event') for p in rows], dtype=object),
                  np.array([PX._num(p.get('xwOBA')) for p in rows], float), dims)


def deltas(P, dim):
    """{cell: (delta, n, se)} recentered within the dimension, BIP rows in a cell."""
    d = P[P['bip'] & P[dim].notna()]
    r = (d['w'] - d['x']).values; base = float(r.mean()); out = {}
    for c, g in d.groupby(dim):
        rr = (g['w'] - g['x']).values
        out[c] = (float(rr.mean() - base), int(len(rr)), float(rr.std() / math.sqrt(len(rr))))
    return out


def main():
    out = {}
    TAB = {y: W.season_table(y) for y in SEASONS}; LG = {y: W.league(TAB[y]) for y in SEASONS}
    P = {}
    print("PER-PA TABLES")
    for y in SEASONS:
        P[y] = pa_savant(y) if y < 2026 else pa_sheet(y)
        cov = {d: int(P[y][d].notna().sum()) for d in DIMS}
        print(f"  {y}: {len(P[y])} PA, {int(P[y]['bip'].sum())} BIP, cell coverage " + " ".join(f"{d} {cov[d]}" for d in DIMS), flush=True)
        gc.collect()
    ALL = pd.concat([P[y] for y in SEASONS if y < 2026])

    print("\n1. CELL DELTAS, actual wOBA - xwOBA on BIP, recentered within the dimension, pooled 2021-2025")
    print("   flag: |delta| > 2 SE and the five seasons agree in sign (a 'likely' needs 4 of 5)")
    out['cells'] = {}
    for dim in DIMS:
        pooled = deltas(ALL, dim); per = {y: deltas(P[y], dim) for y in SEASONS if y < 2026}
        rows = []
        for c, (dl, n, se) in sorted(pooled.items(), key=lambda kv: -abs(kv[1][0])):
            agree = sum(1 for y in per if c in per[y] and np.sign(per[y][c][0]) == np.sign(dl))
            flag = 'REPLICATED' if abs(dl) > 2 * se and agree == 5 else 'likely' if abs(dl) > 2 * se and agree == 4 else ''
            rows.append(dict(cell=c, delta=dl, n=n, se=se, agree=agree, flag=flag, by_season={y: per[y][c][0] for y in per if c in per[y]}))
        out['cells'][dim] = rows
        shown = rows if dim != 'park' else rows[:8]
        print(f"  {dim}:")
        for r in shown:
            print(f"    {str(r['cell']):10} delta {r['delta']:+.4f}  n {r['n']:6d}  se {r['se']:.4f}  seasons {r['agree']}/5  {r['flag']}   "
                  + " ".join(f"{r['by_season'][y]:+.3f}" for y in sorted(r['by_season'])))
        if dim == 'park':
            pfm = {c: np.mean([PF[str(y)].get(c, 100.0) for y in range(2021, 2026)]) for c in pooled}
            xs = [pfm[c] for c in pooled]; ys = [pooled[c][0] for c in pooled]
            b = np.polyfit(xs, ys, 1)[0]; r = np.corrcoef(xs, ys)[0, 1]
            print(f"    park cell delta vs published factor (2021-25 mean): r {r:.2f}, slope {b * 100:+.4f} xwOBA per 100 points of factor "
                  f"(a 110 park runs {b * 10 * 53:.2f} runs/9 hot in xwOBA terms; hWAR removes {WAR_PARK_PASS * 0.05 * 4.4:.2f} at exposure .55)")
            out['park_vs_factor'] = dict(r=float(r), slope=float(b))

    # LOSO delta tables per dimension
    LOSO = {}
    for y in SEASONS:
        others = pd.concat([P[s] for s in SEASONS if s != y and s < 2026])
        LOSO[y] = {dim: {c: v[0] for c, v in deltas(others, dim).items()} for dim in DIMS}

    def adjusted(Py, y, dim):
        x = Py['x'].values.copy(); tab = LOSO[y][dim]
        cells = Py[dim].values
        add = np.array([tab.get(c, 0.0) if c is not None else 0.0 for c in cells])
        return x + add

    print("\n2. PITCHER-LEVEL IMPLIED ADJUSTMENT (LOSO deltas x his cell mix): split-half reliability, pool SD in runs/9 (at DH_B / pool sd)")
    out['pitcher_adj'] = {}
    for dim in DIMS:
        rels, sds = [], []
        for y in SEASONS:
            Py = P[y]; adj = adjusted(Py, y, dim) - Py['x'].values
            df = pd.DataFrame(dict(pid=Py['pid'].values, adj=adj, h1=Py['h1'].values))
            full = df.groupby('pid')['adj'].agg(['mean', 'size'])
            h1 = df[df['h1']].groupby('pid')['adj'].agg(['mean', 'size']); h2 = df[~df['h1']].groupby('pid')['adj'].agg(['mean', 'size'])
            ks = [p for p in h1.index if p in h2.index and h1.loc[p, 'size'] >= 100 and h2.loc[p, 'size'] >= 100]
            rels.append(W.pear([h1.loc[p, 'mean'] for p in ks], [h2.loc[p, 'mean'] for p in ks]))
            pool = [p for p, r in TAB[y].items() if r['outs'] >= POOL_MIN_OUTS and p in full.index]
            sd_pool = W.pool_stats(TAB[y], LG[y])[1]
            sds.append(float(np.std([full.loc[p, 'mean'] for p in pool])) * DH_B / sd_pool)
        out['pitcher_adj'][dim] = dict(rel=rels, sd_runs9=sds)
        print(f"  {dim:7} rel " + " ".join(f"{r:.3f}" for r in rels) + f"  mean {np.nanmean(rels):.3f}   pool sd {np.mean(sds):.3f} runs/9")

    # harness
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
        return F, dict(anchor=float(np.mean([f['ra9'] for f in pool])), gap=LG[y]['ra9'] - LG[y]['era'], lg_ra9=LG[y]['ra9'])

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
        print(f"  {name:12} rel {np.mean(res['rel']):.4f}  nxt60 {np.mean(res['nxt60']):.4f}  nxt30 {np.mean(res['nxt30']):.4f}  "
              f"ros {np.mean(res['ros']):.4f}  calib {np.mean(res['calib']):.3f}", flush=True)
        return res

    def run(name, dim):
        FS = {}
        for y in SEASONS:
            A = PX.agg(P[y], adjusted(P[y], y, dim) if dim else P[y]['x'].values)
            for sc in ('full', 'h1', 'h2'):
                FS[(y, sc)] = frame(y, sc, A)
        return evaluate(name, FS)

    print("\n3. HARNESS: deserved rate on xwOBA + LOSO delta[cell], one dimension at a time")
    ship = run('ship', None)
    for dim in DIMS:
        run(dim, dim)
    print("\nPAIRED vs ship (mean delta, wins/n):")
    for dim in DIMS:
        res = out[dim]; line = []
        for k in ('rel', 'nxt60', 'nxt30', 'ros', 'calib'):
            d = np.array(res[k]) - np.array(ship[k]); line.append(f"{k} {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {dim:8} " + "  ".join(line))
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_xwoba_bias_audit.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_xwoba_bias_audit.json")


if __name__ == '__main__':
    main()
