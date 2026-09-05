"""hwar_hitter_rate_validation.py — which batting rate should position-player hWAR stand on (2026-09-05)?

Three candidates per batter-season, each shrunk at N0 PA toward the league mean:
  woba   actual wOBA on fixed linear weights                   the fWAR / bWAR choice
  xw     Savant xwOBA per PA (EV x LA; K 0, BB/HBP weights, IBB out)
  xhb    xw + XWOBA_PULLAIR_C x (is_pull - league share) on air BIP, the shipped hitter basis
Objectives, six seasons 2021-2026 (2026 from the sheet cache, batter ids from the hitter board):
  rel        split-half r, first vs second half at the ASG date, >= 150 PA each
  nxt300/150 park-adjusted rate in y against actual wOBA in y+1, >= gate PA both sides
  ros        first-half rate against second-half actual wOBA, >= 150 PA each
  calib      slope of actual wOBA on the park-adjusted rate, same season, >= 300 PA
  disagree   the 50 batters per season where xw and woba differ most: which predicts y+1
Park: exposure = mean over his PAs of the published runs factor at the VENUE (game_pk -> home
club), and the pass-through is the LOSO PA-weighted slope of the rate on (exposure - 1) x league
runs per PA x wOBA scale, so 1.0 = the full published factor (war_rate_validation convention).
Then the N0 sweep, all three candidates. Reliability inflates under shrinkage: diagnostic only.
Usage: python3 scripts/research/hitter/hwar_hitter_rate_validation.py
Output: console + data/_hwar_hitter_rate_validation.json
"""
import gc, json, math, os, pickle, sys
from collections import defaultdict
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'era'))
from pipeline.utils import K_EVENTS, BB_EVENTS, HBP_EVENTS, NON_PA_EVENTS, XWOBA_PULLAIR_C, AAA_TEAMS
import war_rate_validation as W
import era_battery_build as EB
import war_improve_battery2 as B2
import war_pullair_fixed as PX

SEASONS = W.SEASONS; T = W.T; PF = W.PF; GPK = B2.GPK
SCALE = {2021: 1.209, 2022: 1.259, 2023: 1.204, 2024: 1.242, 2025: 1.2317, 2026: 1.2385}
CANDS = ['woba', 'xw', 'xhb']
N0_GRID = [0, 25, 50, 100, 150, 200, 300, 500]
GATE_HALF, GATE_CAL = 150, 300


def finish(y, bid, date, ev, xw, la, bbt, hcx, hcy, stand, venue):
    ev = pd.Series(ev)
    is_bb = ev.isin(BB_EVENTS).values; is_hbp = ev.isin(HBP_EVENTS).values; is_k = ev.isin(K_EVENTS).values
    bip = ~(is_bb | is_hbp | is_k)
    x = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, np.where(is_k, 0.0, np.nan_to_num(xw, nan=0.0))))
    w = np.where(is_bb, EB.W_BB, np.where(is_hbp, EB.W_HBP, ev.map(PX.W_ACT).fillna(0.0).values))
    cell = PX._cells(bip, la, bbt, hcx, hcy, stand)
    air = np.array([c is not None and c.startswith('air_') for c in cell]); pull = np.array([c == 'air_pull' for c in cell])
    share = float(pull[air].mean()) if air.any() else 0.28
    xhb = x + np.where(air, XWOBA_PULLAIR_C * (pull.astype(float) - share), 0.0)
    pf = np.array([PF[str(y)].get(v, 100.0) / 100.0 if v is not None else np.nan for v in venue])
    return pd.DataFrame(dict(bid=bid, h1=(date <= T[str(y)]['asg']), woba=w, xw=x, xhb=xhb, pf=pf)), share


def pa_savant(y):
    df = B2.df_year(y)
    df = df[df['game_type'] == 'R']
    d = df[df['events'].notna()][['batter', 'game_date', 'events', 'bb_type', 'launch_angle', 'hc_x', 'hc_y', 'stand', 'game_pk',
                                  'estimated_woba_using_speedangle']].copy()
    del df; gc.collect()
    d['ev'] = d['events'].map(EB.EVENT_MAP)
    d = d[d['ev'].notna() & ~d['ev'].isin(NON_PA_EVENTS) & ~d['ev'].isin(PX.EXCL)]
    f = lambda c: pd.to_numeric(d[c], errors='coerce').values.astype(float)
    venue = np.array([str(GPK[int(g)]) if int(g) in GPK else None for g in d['game_pk'].values], dtype=object)
    return finish(y, d['batter'].astype(int).astype(str).values, d['game_date'].astype(str).str[:10].values, d['ev'].values,
                  f('estimated_woba_using_speedangle'), f('launch_angle'), d['bb_type'].values, f('hc_x'), f('hc_y'), d['stand'].values, venue)


def pa_sheet(y):
    ids = defaultdict(set)
    for r in json.load(open(os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json'))):
        if r.get('team') not in AAA_TEAMS and r.get('mlbId'):
            ids[r['hitter']].add(int(r['mlbId']))
    nm = {n: str(next(iter(s))) for n, s in ids.items() if len(s) == 1}
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    rows = [p for p in D if p.get('_source', 'MLB') == 'MLB' and p.get('BTeam') not in AAA_TEAMS and p.get('Event')
            and p.get('Event') not in NON_PA_EVENTS and p.get('Event') not in PX.EXCL and p.get('Batter') in nm]
    del D; gc.collect()

    def gpk(p):
        try:
            g = int(str(p.get('PitchID')).split('_')[0])
        except ValueError:
            return None
        return str(GPK[g]) if g in GPK else None
    f = lambda k: np.array([PX._num(p.get(k)) for p in rows], float)
    return finish(y, np.array([nm[p.get('Batter')] for p in rows]), np.array([str(p.get('Game Date'))[:10] for p in rows]),
                  np.array([p.get('Event') for p in rows], dtype=object), f('xwOBA'), f('LaunchAngle'),
                  np.array([p.get('BBType') for p in rows], dtype=object), f('HC_X'), f('HC_Y'),
                  np.array([p.get('Bats') for p in rows], dtype=object), np.array([gpk(p) for p in rows], dtype=object))


def main():
    out = {}
    P, SHARE, LG, RPA = {}, {}, {}, {}
    print("PER-PA TABLES")
    for y in SEASONS:
        P[y], SHARE[y] = pa_savant(y) if y < 2026 else pa_sheet(y)
        ph = T[str(y)]['pitchers']; RPA[y] = sum(v['r'] for v in ph.values()) / sum(v['bf'] for v in ph.values())
        LG[y] = {sc: {c: float(P[y].loc[m, c].mean()) for c in CANDS} for sc, m in (('full', slice(None)), ('h1', P[y]['h1'].values), ('h2', ~P[y]['h1'].values))}
        nb = P[y]['bid'].nunique(); miss = int(P[y]['pf'].isna().sum())
        print(f"  {y}: {len(P[y])} PA, {nb} batters, air pull share {SHARE[y]:.3f}, league woba {LG[y]['full']['woba']:.4f} xw {LG[y]['full']['xw']:.4f} "
              f"xhb {LG[y]['full']['xhb']:.4f}, runs/PA {RPA[y]:.4f}, PA without venue {miss}", flush=True)
        gc.collect()
    S = {}
    for y in SEASONS:
        S[y] = {}
        for sc, m in (('full', np.ones(len(P[y]), bool)), ('h1', P[y]['h1'].values), ('h2', ~P[y]['h1'].values)):
            g = P[y][m].groupby('bid').agg(woba=('woba', 'mean'), xw=('xw', 'mean'), xhb=('xhb', 'mean'), n=('woba', 'size'), pf=('pf', 'mean'))
            g['pf'] = g['pf'].fillna(1.0); S[y][sc] = g

    def sh(y, sc, c, n0):
        g = S[y][sc]; return (g[c] * g['n'] + n0 * LG[y][sc][c]) / (g['n'] + n0)

    def full_park(y, g):
        return (g['pf'] - 1.0) * RPA[y] * SCALE[y]

    print("\nPARK PASS-THROUGH: LOSO PA-weighted slope of the raw rate on (venue exposure - 1) x runs/PA x scale, >= 300 PA; 1.0 = the published factor")
    PASS = {}
    for c in CANDS:
        folds = []
        for hold in SEASONS:
            x, yv, w = [], [], []
            for y in SEASONS:
                if y == hold:
                    continue
                g = S[y]['full']; g = g[g['n'] >= GATE_CAL]
                x += list(full_park(y, g)); yv += list(g[c] - LG[y]['full'][c]); w += list(g['n'])
            folds.append(W.wls_slope(np.array(x), np.array(yv), np.array(w, float)))
        PASS[c] = float(np.mean(folds)); out[f'pass_{c}'] = folds
        print(f"  {c:5} folds " + " ".join(f"{f:.3f}" for f in folds) + f"  mean {PASS[c]:.3f}")

    def evaluate(c, n0, tag=None):
        res = {'rel': [], 'ros': [], 'nxt300': [], 'nxt150': [], 'calib': []}
        for y in SEASONS:
            a, b = S[y]['h1'], S[y]['h2']; ks = a.index[a['n'] >= GATE_HALF].intersection(b.index[b['n'] >= GATE_HALF])
            r1 = sh(y, 'h1', c, n0).loc[ks]; res['rel'].append(W.pear(r1.values, sh(y, 'h2', c, n0).loc[ks].values))
            res['ros'].append(W.pear(r1.values, b.loc[ks, 'woba'].values))
            g = S[y]['full']; ks = g.index[g['n'] >= GATE_CAL]
            adj = (sh(y, 'full', c, n0) - PASS[c] * full_park(y, g)).loc[ks]
            res['calib'].append(float(np.polyfit(adj.values, g.loc[ks, 'woba'].values, 1)[0]))
        for gate in (300, 150):
            for y in SEASONS[:-1]:
                g, g2 = S[y]['full'], S[y + 1]['full']; ks = g.index[g['n'] >= gate].intersection(g2.index[g2['n'] >= gate])
                adj = (sh(y, 'full', c, n0) - PASS[c] * full_park(y, g)).loc[ks]
                res[f'nxt{gate}'].append(W.pear(adj.values, g2.loc[ks, 'woba'].values))
        name = tag or f'{c} N0={n0}'; out[name] = res
        print(f"  {name:14} rel {np.mean(res['rel']):.3f}  nxt300 {np.mean(res['nxt300']):.3f}  nxt150 {np.mean(res['nxt150']):.3f}  "
              f"ros {np.mean(res['ros']):.3f}  calib {np.mean(res['calib']):.3f}", flush=True)
        return res

    print("\nCANDIDATES, unshrunk (N0 = 0):")
    base = {c: evaluate(c, 0) for c in CANDS}
    print("\nDISAGREEMENT SET: the 50 batters per season (>= 300 PA both years) where xw and woba differ most; r with next-season wOBA")
    dis = {c: [] for c in CANDS}
    for y in SEASONS[:-1]:
        g, g2 = S[y]['full'], S[y + 1]['full']; ks = g.index[g['n'] >= 300].intersection(g2.index[g2['n'] >= 300])
        top = (g.loc[ks, 'xw'] - g.loc[ks, 'woba']).abs().sort_values(ascending=False).index[:50]
        for c in CANDS:
            dis[c].append(W.pear((g.loc[top, c] - PASS[c] * full_park(y, g.loc[top])).values, g2.loc[top, 'woba'].values))
    for c in CANDS:
        print(f"  {c:5} " + " ".join(f"{v:.3f}" for v in dis[c]) + f"  mean {np.mean(dis[c]):.3f}")
    out['disagree'] = dis

    print("\nPAIRED vs woba (unshrunk): mean delta, wins/n")
    for c in ('xw', 'xhb'):
        line = []
        for k in ('rel', 'nxt300', 'nxt150', 'ros', 'calib'):
            d = np.array(base[c][k]) - np.array(base['woba'][k]); line.append(f"{k} {d.mean():+.3f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {c:5} " + "  ".join(line))
    print("  xhb vs xw: " + "  ".join(f"{k} {(np.array(base['xhb'][k]) - np.array(base['xw'][k])).mean():+.4f} ({int(((np.array(base['xhb'][k]) - np.array(base['xw'][k])) > 0).sum())}/{len(base[k if False else 'xw'][k])})" for k in ('rel', 'nxt300', 'nxt150', 'ros')))

    print("\nN0 SWEEP (reliability inflates under shrinkage: diagnostic; nxt and ros decide)")
    for c in CANDS:
        print(f"  {c}:")
        for n0 in N0_GRID:
            evaluate(c, n0)
    json.dump(out, open(os.path.join(ROOT, 'data', '_hwar_hitter_rate_validation.json'), 'w'), indent=1, default=float)
    print("wrote data/_hwar_hitter_rate_validation.json")


if __name__ == '__main__':
    main()
