"""war_pullair_fixed.py — pulled air as a FIXED per-BIP adjustment to the deserved rate (2026-09-05).

Battery 2 tested pulled-air excess as a FITTED channel next to the shrunk xwOBA against
(LOSO OLS weight, unshrunk): it won next-season RA9 at 60 IP 5/5 and lost half-season
reliability 0/6, because the channel's own reliability is .20 and its weight is fitted.
This script tests the other form: adjust each BIP's xwOBA by a fixed spray term BEFORE
the pitcher aggregate and the 250-PA shrink, so the term is shrunk with everything else
and carries no fitted weight.

Forms:
  hb C        xwOBA' = xwOBA + C x (is_pull - league pull share) on air BIP (LA >= 20),
              the shipped HITTER basis (pipeline/utils XWOBA_PULLAIR_C = .20), swept over C
  cells       xwOBA' = xwOBA + delta[cell], delta = mean(actual - x | cell) - mean(actual - x),
              cells = air/ground x pull/center/oppo, actual on the battery's fixed linear
              weights, table from the OTHER seasons (LOSO; 2026 from 2021-2025)
  cells-air   the same with the three air cells only (recentered over air BIP)
Objectives (war_improve_battery2 evaluate, one channel z(xw shrunk 250), LOSO OLS on RA9):
rel (h1/h2 >= 30 IP), nxt60, nxt30, ros, calib; paired against ship. The ship arm is
REBUILT from the same per-PA table and checked against the harness season table first.
Usage: python3 scripts/research/era/war_pullair_fixed.py
Output: console + data/_war_pullair_fixed.json
"""
import gc, json, math, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from pipeline.utils import (spray_angle, spray_direction, BUNT_BB_TYPES, XWOBA_PULLAIR_C, XWOBA_PULLAIR_LA,
                            K_EVENTS, BB_EVENTS, HBP_EVENTS, SH_EVENTS, CI_EVENTS, NON_PA_EVENTS)
from pipeline.eraplus import N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS
import war_rate_validation as W
import era_battery_build as EB
import war_improve_battery2 as B2

SEASONS = W.SEASONS; T = W.T
W_ACT = {'Single': EB.W_1B, 'Double': EB.W_2B, 'Triple': EB.W_3B, 'Home Run': EB.W_HR}
EXCL = set(SH_EVENTS) | set(CI_EVENTS) | {'Intent Walk'}
C_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
CELLS = ['air_pull', 'air_center', 'air_oppo', 'gb_pull', 'gb_center', 'gb_oppo']


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return f


def _cells(bip, la, bbt, hcx, hcy, stand):
    """cell per row (None outside a cell): air/ground x pull/center/oppo; air = LA >= XWOBA_PULLAIR_LA; bunts out."""
    out = np.array([None] * len(la), dtype=object)
    bunt = pd.Series(bbt).isin(BUNT_BB_TYPES).values
    idx = np.where(bip & ~np.isnan(la) & ~bunt & ~np.isnan(hcx) & ~np.isnan(hcy))[0]
    for i in idx:
        d = spray_direction(spray_angle(float(hcx[i]), float(hcy[i])), stand[i])
        if d is None:
            continue
        side = 'pull' if d in ('pull', 'pull_side') else 'oppo' if d in ('oppo', 'oppo_side') else 'center'
        out[i] = ('air_' if la[i] >= XWOBA_PULLAIR_LA else 'gb_') + side
    return out


def _finish(y, pid, date, ev, xw, la, bbt, hcx, hcy, stand):
    ev = pd.Series(ev)
    is_bb = ev.isin(BB_EVENTS).values; is_hbp = ev.isin(HBP_EVENTS).values; is_k = ev.isin(K_EVENTS).values
    bip = ~(is_bb | is_hbp | is_k)
    x = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, np.where(is_k, 0.0, np.nan_to_num(xw, nan=0.0))))
    w = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, ev.map(W_ACT).fillna(0.0).values))
    return pd.DataFrame(dict(pid=pid, h1=(date <= T[str(y)]['asg']), x=x, w=w, bip=bip, cell=_cells(bip, la, bbt, hcx, hcy, stand)))


def pa_savant(y):
    df = B2.df_year(y)
    df = df[df['game_type'] == 'R']
    d = df[df['events'].notna()][['pitcher', 'game_date', 'events', 'bb_type', 'launch_angle', 'hc_x', 'hc_y', 'stand',
                                  'estimated_woba_using_speedangle']].copy()
    del df; gc.collect()
    d['ev'] = d['events'].map(EB.EVENT_MAP)
    d = d[d['ev'].notna() & ~d['ev'].isin(NON_PA_EVENTS) & ~d['ev'].isin(EXCL)]
    return _finish(y, d['pitcher'].astype(int).astype(str).values, d['game_date'].astype(str).str[:10].values, d['ev'].values,
                   pd.to_numeric(d['estimated_woba_using_speedangle'], errors='coerce').values.astype(float),
                   pd.to_numeric(d['launch_angle'], errors='coerce').values.astype(float), d['bb_type'].values,
                   pd.to_numeric(d['hc_x'], errors='coerce').values.astype(float),
                   pd.to_numeric(d['hc_y'], errors='coerce').values.astype(float), d['stand'].values)


def pa_sheet(y):
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    nm = EB.build_2026_name_map()
    rows = [p for p in D if p.get('_source', 'MLB') == 'MLB' and p.get('Event') and p.get('Event') not in NON_PA_EVENTS
            and p.get('Event') not in EXCL]
    del D; gc.collect()
    pid = [nm.get(p.get('Pitcher')) for p in rows]
    keep = [i for i, v in enumerate(pid) if v is not None]
    rows = [rows[i] for i in keep]; pid = np.array([str(pid[i]) for i in keep])
    f = lambda k: np.array([_num(p.get(k)) for p in rows], float)
    return _finish(y, pid, np.array([str(p.get('Game Date'))[:10] for p in rows]), np.array([p.get('Event') for p in rows], dtype=object),
                   f('xwOBA'), f('LaunchAngle'), np.array([p.get('BBType') for p in rows], dtype=object), f('HC_X'), f('HC_Y'),
                   np.array([p.get('Bats') for p in rows], dtype=object))


def validate(y, P, tab):
    g = P.groupby('pid')['x'].agg(['mean', 'size'])
    d = [(abs(g.loc[pid, 'mean'] - r['xw']), abs(g.loc[pid, 'size'] - r['xw_den'])) for pid, r in tab.items()
         if r['xw'] is not None and pid in g.index]
    a = np.array(d)
    print(f"  {y}: {len(P)} PA, {int(P['bip'].sum())} BIP, {int(P['cell'].notna().sum())} in a cell; rebuilt xw vs harness on {len(a)} arms: "
          f"max |dxw| {a[:, 0].max():.5f}, arms |dxw| > 1e-4: {int((a[:, 0] > 1e-4).sum())}, max |dden| {a[:, 1].max():.0f}", flush=True)
    return dict(n_pa=int(len(P)), n_bip=int(P['bip'].sum()), max_dxw=float(a[:, 0].max()), n_off=int((a[:, 0] > 1e-4).sum()))


def cell_deltas(P, air_only=False):
    """{cell: mean(actual - x | cell) - mean(actual - x | scope)} on one or more seasons' PA tables."""
    d = P[P['cell'].notna()]
    if air_only:
        d = d[d['cell'].str.startswith('air_')]
    res = d['w'] - d['x']; base = float(res.mean())
    return {c: float(res[d['cell'] == c].mean() - base) for c in (CELLS[:3] if air_only else CELLS)}


def adjusted(P, form, param):
    """x' per PA. form 'hb': param = C (season pull share among air cells); form 'cells': param = {cell: delta}."""
    x = P['x'].values.copy()
    if form == 'hb':
        air = P['cell'].notna().values & P['cell'].astype(str).str.startswith('air_').values
        pull = (P['cell'].values == 'air_pull')
        share = pull[air].mean()
        x[air] += param * (pull[air].astype(float) - share)
    else:
        for c, dv in param.items():
            x[P['cell'].values == c] += dv
    return x


def agg(P, xcol):
    out = {}
    for sc, mask in (('full', np.ones(len(P), bool)), ('h1', P['h1'].values), ('h2', ~P['h1'].values)):
        g = pd.DataFrame({'pid': P['pid'].values[mask], 'x': xcol[mask]}).groupby('pid')['x'].agg(['mean', 'size'])
        out[sc] = {pid: (float(m), int(n)) for pid, m, n in zip(g.index, g['mean'], g['size'])}
    return out


def main():
    out = {}
    TAB = {y: W.season_table(y) for y in SEASONS}; LG = {y: W.league(TAB[y]) for y in SEASONS}
    P = {}
    print("PER-PA TABLES (battery tally convention: BB/HBP weights, K 0, BIP xwOBA or 0, IBB/SH/CI out)")
    for y in SEASONS:
        P[y] = pa_savant(y) if y < 2026 else pa_sheet(y)
        out[f'validate_{y}'] = validate(y, P[y], TAB[y])
        gc.collect()

    print("\nCELL DELTAS BY SEASON, actual - xwOBA (recentered over all six cells), n BIP per cell:")
    print("  season  " + "  ".join(f"{c:>10}" for c in CELLS) + "   air pull share")
    out['cell_deltas'] = {}
    for y in SEASONS:
        dl = cell_deltas(P[y]); cnt = P[y]['cell'].value_counts()
        air = P[y][P[y]['cell'].notna() & P[y]['cell'].astype(str).str.startswith('air_')]
        share = float((air['cell'] == 'air_pull').mean())
        print(f"  {y}    " + "  ".join(f"{dl[c]:+.4f}/{int(cnt.get(c, 0)):5d}" for c in CELLS) + f"   {share:.3f}")
        out['cell_deltas'][y] = dl; out['cell_deltas'][f'{y}_share'] = share

    # LOSO tables
    LOSO = {}
    for y in SEASONS:
        others = pd.concat([P[s] for s in SEASONS if s != y and s < 2026])
        LOSO[y] = dict(cells=cell_deltas(others), air=cell_deltas(others, air_only=True))
    out['loso_tables'] = {y: LOSO[y] for y in SEASONS}
    print("\nLOSO air-cell deltas used (pull / center / oppo): " + "  ".join(
        f"{y}: {LOSO[y]['air']['air_pull']:+.3f}/{LOSO[y]['air']['air_center']:+.3f}/{LOSO[y]['air']['air_oppo']:+.3f}" for y in SEASONS))

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
            beta[hold] = np.polyfit(X, Y, 1)[::-1]   # [intercept, slope]

        def rate(y, sc, pid, park=False):
            F, M = FS[(y, sc)]; f = F.get(pid)
            if f is None:
                return None
            b = beta[y]; v = M['anchor'] + b[0] + b[1] * f['z'] + M['gap']
            if park:
                v -= WAR_PARK_PASS * (f['exp'] - 1) * M['lg_ra9']
            return v
        res = {}
        rel, ros = [], []
        for y in SEASONS:
            A, _ = FS[(y, 'h1')]; Bh, _ = FS[(y, 'h2')]
            ks = [p for p in A if p in Bh and A[p]['outs'] >= 90 and Bh[p]['outs'] >= 90]
            rel.append(W.pear([rate(y, 'h1', p) for p in ks], [rate(y, 'h2', p) for p in ks]))
            ros.append(W.pear([rate(y, 'h1', p) for p in ks], [Bh[p]['ra9'] for p in ks]))
        res['rel'] = rel; res['ros'] = ros
        for gate in (60, 30):
            nx = []
            for y in SEASONS[:-1]:
                A, _ = FS[(y, 'full')]; Bh, _ = FS[(y + 1, 'full')]
                ks = [p for p in A if p in Bh and A[p]['outs'] >= gate * 3 and Bh[p]['outs'] >= gate * 3]
                nx.append(W.pear([rate(y, 'full', p, park=True) for p in ks], [Bh[p]['ra9'] for p in ks]))
            res[f'nxt{gate}'] = nx
        cal = []
        for y in SEASONS:
            F, _ = FS[(y, 'full')]; ks = [p for p in F if F[p]['outs'] >= 180]
            cal.append(float(np.polyfit([rate(y, 'full', p, park=True) for p in ks], [F[p]['ra9'] for p in ks], 1)[0]))
        res['calib'] = cal; res['slope_2024'] = float(beta[2024][1]); res['pool_sd'] = [FS[(y, 'full')][1]['sd'] for y in SEASONS]
        out[name] = res
        print(f"  {name:16} rel {np.mean(rel):.4f}  nxt60 {np.mean(res['nxt60']):.4f}  nxt30 {np.mean(res['nxt30']):.4f}  "
              f"ros {np.mean(ros):.4f}  calib {np.mean(cal):.3f}  slope {beta[2024][1]:.3f}  pool sd {np.mean(res['pool_sd']):.4f}", flush=True)
        return res

    def run(name, form, params):
        FS = {}
        for y in SEASONS:
            A = agg(P[y], adjusted(P[y], form, params[y]) if form else P[y]['x'].values)
            for sc in ('full', 'h1', 'h2'):
                FS[(y, sc)] = frame(y, sc, A)
        return evaluate(name, FS)

    print("\nVARIANTS (one channel, LOSO OLS on RA9; nxt uses the park-adjusted rate):")
    ship = run('ship (rebuilt)', None, None)
    for C in C_GRID:
        run(f'hb C={C:.2f}', 'hb', {y: C for y in SEASONS})
    run('cells LOSO', 'cells', {y: LOSO[y]['cells'] for y in SEASONS})
    run('cells-air LOSO', 'cells', {y: LOSO[y]['air'] for y in SEASONS})

    print("\nPAIRED vs ship (mean delta, wins/n):")
    for name, res in out.items():
        if not isinstance(res, dict) or 'rel' not in res or name.startswith('ship'):
            continue
        line = []
        for k in ('rel', 'nxt60', 'nxt30', 'ros', 'calib'):
            d = np.array(res[k]) - np.array(ship[k]); line.append(f"{k} {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {name:16} " + "  ".join(line))
    h = json.load(open(os.path.join(ROOT, 'data', '_war_improve_battery2.json')))
    print("\nharness ship (battery 2, from _era_battery xw): rel {:.4f} nxt60 {:.4f} nxt30 {:.4f} ros {:.4f}".format(
        np.mean(h['ship']['rel']), np.mean(h['ship']['nxt60']), np.mean(h['ship']['nxt30']), np.mean(h['ship']['ros'])))
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_pullair_fixed.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_pullair_fixed.json")


if __name__ == '__main__':
    main()
