"""hwar_park_pass_within.py — park pass-through WITHIN player (home vs road), hitters and pitchers (2026-09-05).

hwar_hitter_rate_validation.py measured the pass-through ACROSS batters (rate on venue exposure)
and got .30 for actual wOBA and -.18 for xwOBA: team quality rides on the same axis as the park
(30 clubs), so that design is confounded. Here each player is his own control:
    d_rate = rate at his home park - rate on the road (same season)
    d_park = (PF_home - PA-weighted PF of his road venues) / 100 x league runs per PA x wOBA scale
    pass   = PA-weighted slope of d_rate on d_park (weight = harmonic mean of home and road PA),
             >= 100 PA on each side, per season and LOSO; 1.0 = the full published factor
Home park = the venue with the most of his PA, kept only when it holds >= 40% of them (single-club
seasons). Batters on woba / xw / xhb, pitchers on the same three rates against (xw is hdERA's
basis, so this also checks the shipped WAR_PARK_PASS .91, which came from the across-pitcher
club-exposure design).
Also: the hitter calibration slope by PA tercile (selection check), and the N0 grid extended to
1000 / 2000 for xw and xhb.
Usage: python3 scripts/research/hitter/hwar_park_pass_within.py
Output: console + data/_hwar_park_pass_within.json
"""
import gc, json, math, os, pickle, sys
from collections import defaultdict
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'era'))
from pipeline.utils import NON_PA_EVENTS, AAA_TEAMS
import war_rate_validation as W
import era_battery_build as EB
import war_improve_battery2 as B2
import war_pullair_fixed as PX
import hwar_hitter_rate_validation as HR

SEASONS = HR.SEASONS; T = HR.T; PF = HR.PF; GPK = HR.GPK; SCALE = HR.SCALE; CANDS = HR.CANDS
MIN_SIDE, HOME_SHARE = 100, 0.40


def table(y):
    if y < 2026:
        df = B2.df_year(y); df = df[df['game_type'] == 'R']
        d = df[df['events'].notna()][['batter', 'pitcher', 'game_date', 'events', 'bb_type', 'launch_angle', 'hc_x', 'hc_y', 'stand', 'game_pk',
                                      'estimated_woba_using_speedangle']].copy()
        del df; gc.collect()
        d['ev'] = d['events'].map(EB.EVENT_MAP)
        d = d[d['ev'].notna() & ~d['ev'].isin(NON_PA_EVENTS) & ~d['ev'].isin(PX.EXCL)]
        f = lambda c: pd.to_numeric(d[c], errors='coerce').values.astype(float)
        venue = np.array([str(GPK[int(g)]) if int(g) in GPK else None for g in d['game_pk'].values], dtype=object)
        P, _ = HR.finish(y, d['batter'].astype(int).astype(str).values, d['game_date'].astype(str).str[:10].values, d['ev'].values,
                         f('estimated_woba_using_speedangle'), f('launch_angle'), d['bb_type'].values, f('hc_x'), f('hc_y'), d['stand'].values, venue)
        P['pid'] = d['pitcher'].astype(int).astype(str).values; P['venue'] = venue
        return P
    ids = defaultdict(set)
    for r in json.load(open(os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json'))):
        if r.get('team') not in AAA_TEAMS and r.get('mlbId'):
            ids[r['hitter']].add(int(r['mlbId']))
    nm = {n: str(next(iter(s))) for n, s in ids.items() if len(s) == 1}; pm = EB.build_2026_name_map()
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    rows = [p for p in D if p.get('_source', 'MLB') == 'MLB' and p.get('BTeam') not in AAA_TEAMS and p.get('Event')
            and p.get('Event') not in NON_PA_EVENTS and p.get('Event') not in PX.EXCL and p.get('Batter') in nm and pm.get(p.get('Pitcher'))]
    del D; gc.collect()

    def gpk(p):
        try:
            g = int(str(p.get('PitchID')).split('_')[0])
        except ValueError:
            return None
        return str(GPK[g]) if g in GPK else None
    f = lambda k: np.array([PX._num(p.get(k)) for p in rows], float)
    venue = np.array([gpk(p) for p in rows], dtype=object)
    P, _ = HR.finish(y, np.array([nm[p.get('Batter')] for p in rows]), np.array([str(p.get('Game Date'))[:10] for p in rows]),
                     np.array([p.get('Event') for p in rows], dtype=object), f('xwOBA'), f('LaunchAngle'),
                     np.array([p.get('BBType') for p in rows], dtype=object), f('HC_X'), f('HC_Y'), np.array([p.get('Bats') for p in rows], dtype=object), venue)
    P['pid'] = np.array([str(pm[p.get('Pitcher')]) for p in rows]); P['venue'] = venue
    return P


def within(P, y, key):
    """per player: home venue (most PA, >= 40%), d_rate per candidate, d_park, weight."""
    rpa = HR_RPA[y]; rows = []
    for k, g in P.groupby(key):
        vc = g['venue'].value_counts()
        if vc.iloc[0] / len(g) < HOME_SHARE:
            continue
        home = vc.index[0]; hm = g['venue'] == home
        if hm.sum() < MIN_SIDE or (~hm).sum() < MIN_SIDE:
            continue
        pf_home = PF[str(y)].get(home, 100.0) / 100.0; pf_road = float(g.loc[~hm, 'pf'].mean())
        rows.append(dict(k=k, d_park=(pf_home - pf_road) * rpa * SCALE[y], w=2 * hm.sum() * (~hm).sum() / len(g),
                         **{c: float(g.loc[hm, c].mean() - g.loc[~hm, c].mean()) for c in CANDS}))
    return pd.DataFrame(rows)


def main():
    global HR_RPA
    out = {}
    P = {y: table(y) for y in SEASONS}
    HR_RPA = {y: (lambda ph: sum(v['r'] for v in ph.values()) / sum(v['bf'] for v in ph.values()))(T[str(y)]['pitchers']) for y in SEASONS}
    for side, key in (('BATTERS', 'bid'), ('PITCHERS', 'pid')):
        print(f"\nWITHIN-{side} PASS-THROUGH (home - road on the same player, >= 100 PA each side): slope of d_rate on d_park, 1.0 = the published factor")
        D = {y: within(P[y], y, key) for y in SEASONS}
        print("  season   n   sd(d_park in wOBA)  " + "  ".join(f"{c:>7}" for c in CANDS))
        res = {c: [] for c in CANDS}
        for y in SEASONS:
            d = D[y]; line = []
            for c in CANDS:
                s = W.wls_slope(d['d_park'].values, d[c].values, d['w'].values); res[c].append(s); line.append(f"{s:7.3f}")
            print(f"  {y}    {len(d):4d}   {d['d_park'].std():.4f}            " + "  ".join(line))
        loso = {c: [] for c in CANDS}
        for hold in SEASONS:
            d = pd.concat([D[y] for y in SEASONS if y != hold])
            for c in CANDS:
                loso[c].append(W.wls_slope(d['d_park'].values, d[c].values, d['w'].values))
        print("  per-season mean: " + "  ".join(f"{c} {np.mean(res[c]):.3f} ± {np.std(res[c], ddof=1) / math.sqrt(len(res[c])):.3f}" for c in CANDS))
        print("  LOSO folds:      " + "  ".join(f"{c} " + "/".join(f"{v:.2f}" for v in loso[c]) for c in CANDS))
        out[f'within_{key}'] = dict(per_season=res, loso=loso)

    print("\nHITTER CALIBRATION BY PA TERCILE (>= 300 PA, N0 = 0, park-adjusted at the within-batter xw pass): slope of actual wOBA on the rate")
    pass_xw = float(np.mean(out['within_bid']['per_season']['xw']))
    S = {}
    for y in SEASONS:
        g = P[y].groupby('bid').agg(woba=('woba', 'mean'), xw=('xw', 'mean'), xhb=('xhb', 'mean'), n=('woba', 'size'), pf=('pf', 'mean'))
        S[y] = g[g['n'] >= 300]
    allr = []
    for y in SEASONS:
        g = S[y]; lg = {c: float(P[y][c].mean()) for c in CANDS}
        for bid, r in g.iterrows():
            allr.append((r['xw'] - lg['xw'] - pass_xw * (r['pf'] - 1) * HR_RPA[y] * SCALE[y], r['woba'] - lg['woba'], r['n']))
    a = np.array(allr); q1, q2 = np.percentile(a[:, 2], [33, 67])
    for lo, hi, lab in ((0, q1, 'low PA'), (q1, q2, 'mid PA'), (q2, 1e9, 'high PA')):
        m = (a[:, 2] >= lo) & (a[:, 2] < hi)
        print(f"  {lab:8} n {int(m.sum()):4d}  PA {a[m, 2].mean():5.0f}  slope {W.wls_slope(a[m, 0], a[m, 1], a[m, 2]):.3f}")
    print(f"  pooled slope {W.wls_slope(a[:, 0], a[:, 1], a[:, 2]):.3f}")

    print("\nN0 GRID EXTENDED (xw, xhb): nxt300 / nxt150 / ros, N0 in 0 .. 2000")
    lgc = {y: {c: float(P[y][c].mean()) for c in CANDS} for y in SEASONS}
    full = {y: P[y].groupby('bid').agg(woba=('woba', 'mean'), xw=('xw', 'mean'), xhb=('xhb', 'mean'), n=('woba', 'size'), pf=('pf', 'mean')) for y in SEASONS}
    half = {y: {h: P[y][P[y]['h1'].values == (h == 'h1')].groupby('bid').agg(woba=('woba', 'mean'), xw=('xw', 'mean'), xhb=('xhb', 'mean'), n=('woba', 'size')) for h in ('h1', 'h2')} for y in SEASONS}
    out['n0_ext'] = {}
    for c in ('xw', 'xhb'):
        for n0 in (0, 100, 300, 500, 1000, 2000):
            nx3, nx1, ros = [], [], []
            for y in SEASONS[:-1]:
                g, g2 = full[y], full[y + 1]
                for gate, acc in ((300, nx3), (150, nx1)):
                    ks = g.index[g['n'] >= gate].intersection(g2.index[g2['n'] >= gate])
                    r = ((g.loc[ks, c] * g.loc[ks, 'n'] + n0 * lgc[y][c]) / (g.loc[ks, 'n'] + n0)) - pass_xw * (g.loc[ks, 'pf'] - 1) * HR_RPA[y] * SCALE[y]
                    acc.append(W.pear(r.values, g2.loc[ks, 'woba'].values))
            for y in SEASONS:
                a1, a2 = half[y]['h1'], half[y]['h2']; ks = a1.index[a1['n'] >= 150].intersection(a2.index[a2['n'] >= 150])
                lg1 = float(P[y].loc[P[y]['h1'].values, c].mean())
                r = (a1.loc[ks, c] * a1.loc[ks, 'n'] + n0 * lg1) / (a1.loc[ks, 'n'] + n0)
                ros.append(W.pear(r.values, a2.loc[ks, 'woba'].values))
            out['n0_ext'][f'{c}_{n0}'] = dict(nxt300=nx3, nxt150=nx1, ros=ros)
            print(f"  {c:4} N0={n0:5d}  nxt300 {np.mean(nx3):.4f}  nxt150 {np.mean(nx1):.4f}  ros {np.mean(ros):.4f}")
    json.dump(out, open(os.path.join(ROOT, 'data', '_hwar_park_pass_within.json'), 'w'), indent=1, default=float)
    print("wrote data/_hwar_park_pass_within.json")


if __name__ == '__main__':
    main()
