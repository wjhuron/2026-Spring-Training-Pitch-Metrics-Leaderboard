"""war_improve_battery2.py — hWAR improvement battery, Part 2 (2026-09-05):
the tests that need per-pitch data.

Candidates, each as an extra channel next to the shipped deserved rate
(shrunk xwOBA against), LOSO OLS on next-season-free objectives as in Part 1:
  fr9      framing runs RECEIVED per 9, from the Loc+ called-strike surface:
           sum over takes of (called strike - P(called strike | location, hand,
           count)) x (RV strike - RV ball), pitcher-positive. A deserved rate
           should not credit the catcher; if fr9 earns a positive next-season
           weight, the shipped rate is borrowing the catcher's runs.
  pullx    pulled-air excess per PA: (n pulled air balls - league share x n air
           balls) / PA, the xwRC+ pulled-air idea on the pitcher side
  rg9      running game: (0.45 x CS - 0.20 x SB) per 9 above league, from the
           official lines (targets rebuilt with sb/cs)
  home     park exposure from the pitcher's ACTUAL home pitch share instead of
           the 50/50 assumption (game_pk -> home club map)
  role gap within season: the same pitcher as starter and as reliever in the
           SAME season (side and starter reconstructed from pitch order), the
           selection-free version of the 0.64 runs/9 role gap
Objectives: rel (h1/h2 >= 30 IP), nxt60, nxt30, ros, calib; paired vs ship.
Usage: python3 scripts/research/era/war_improve_battery2.py
Output: console + data/_war_improve_battery2.json
"""
import gc, json, math, os, pickle, sys
from collections import defaultdict
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'locplus'))
import pipeline.locplus as lp
import locplus_constants_multiseason as base
from pipeline.utils import get_count, safe_float, spray_angle, spray_direction, _fullname_to_lastfirst, TEAM_ABBREV_TO_ID, AAA_TEAMS
from pipeline.eraplus import DH_B, N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS
import war_rate_validation as W

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl', 2023: 'data/_statcast2023_cache.pkl',
         2024: 'data/_statcast2024_cache.pkl', 2025: 'data/_statcast2025_full_cache.pkl'}
GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259), 2023: (0.318, 1.204), 2024: (0.310, 1.242), 2025: (0.3131, 1.2317), 2026: (0.3165, 1.2385)}
T = W.T; B = W.B; PF = W.PF
GPK = {int(k): int(v) for k, v in json.load(open(os.path.join(ROOT, 'data', '_mlb_gamepk_home.json'))).items()}
SB_RV, CS_RV = 0.20, 0.45      # run values of a steal and a caught stealing (standard linear weights)
WP_RV = 0.25                   # a wild pitch or balk advances runners; standard linear weight
PULL_BINS = {'pull', 'pull_side'}

def name_map(y):
    m, amb = {}, set()
    for pid, rec in T[str(y)]['pitchers'].items():
        full = (rec['name'] or '').strip(); v = {_fullname_to_lastfirst(full).lower()}
        parts = full.split()
        if len(parts) >= 3: v.add((' '.join(parts[-2:]) + ', ' + ' '.join(parts[:-2])).lower())
        for lf in v:
            if lf in m and m[lf] != pid: amb.add(lf)
            m[lf] = pid
    for lf in amb: m.pop(lf, None)
    return m

def framing(y, pitches, nm):
    """{pid: {'full': runs, 'h1': runs, 'takes': n}} pitcher-positive framing runs received."""
    lg, sc = GUTS[y]; asg = T[str(y)]['asg']
    basep = [p for p in pitches if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(basep, lg, sc)
    out = defaultdict(lambda: {'full': 0.0, 'h1': 0.0, 'takes': 0})
    for p in pitches:
        d = p.get('Description')
        if d not in ('Called Strike', 'Ball') or not lp._is_scorable(p): continue
        pid = nm.get(str(p.get('Pitcher') or '').lower())
        if pid is None: continue
        c = get_count(p); i = lp._xbin(safe_float(p.get('PlateX'))); j = lp._zbin(lp._znorm(p))
        pcs = S['PCS'][p['Bats']][c][i][j]; drv = S['RV']['cs'].get(c, 0.0) - S['RV']['ball'].get(c, 0.0)   # hitter-perspective, negative
        gain = -((1.0 if d == 'Called Strike' else 0.0) - pcs) * drv
        o = out[pid]; o['full'] += gain; o['takes'] += 1
        if str(p.get('Game Date'))[:10] <= asg: o['h1'] += gain
    return out

def df_year(y):
    df = pickle.load(open(os.path.join(ROOT, CACHE[y]), 'rb'))
    if y == 2025 and 'hc_x' not in df.columns:
        # the full 2025 cache dropped the hit coordinates; the earlier pull is row-aligned
        # (verified 2026-09-05 on game_pk/batter/pitcher/count/plate_x), so borrow them by position
        old = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2025_cache.pkl'), 'rb'))
        assert len(old) == len(df) and (old['game_pk'].values == df['game_pk'].values).all()
        df['hc_x'] = old['hc_x'].values; df['hc_y'] = old['hc_y'].values
    return df

def home_share(y, df=None, sheet=None):
    """{pid: share of pitches thrown at home}; single-club pitcher-seasons only."""
    share = {}
    if df is not None:
        g = df.groupby(['pitcher', 'game_pk']).size().reset_index(name='n')
        g['home_id'] = g['game_pk'].map(GPK)
        for pid, sub in g.groupby('pitcher'):
            teams = T[str(y)]['pitchers'].get(str(int(pid)), {}).get('teams') or []
            if len(teams) != 1: continue
            tid = int(teams[0]); hs = sub[sub['home_id'] == tid]['n'].sum(); tot = sub['n'].sum()
            if tot > 0: share[str(int(pid))] = hs / tot
    else:
        nm = name_map(y); acc = defaultdict(lambda: [0, 0])
        for p in sheet:
            pid = nm.get(str(p.get('Pitcher') or '').lower())
            if pid is None: continue
            try: gpk = int(str(p.get('PitchID')).split('_')[0])
            except ValueError: continue
            tid = TEAM_ABBREV_TO_ID.get(p.get('PTeam')); hid = GPK.get(gpk)
            if tid is None or hid is None: continue
            a = acc[pid]; a[1] += 1; a[0] += 1 if hid == tid else 0
        for pid, (h, n) in acc.items():
            teams = T[str(y)]['pitchers'].get(pid, {}).get('teams') or []
            if len(teams) == 1 and n > 0: share[pid] = h / n
    return share

def pull_excess(y, df):
    """{pid: {'full': (excess, pa), 'h1': (...)}} pulled-air excess counts; PA from PA-ending rows."""
    asg = T[str(y)]['asg']
    if 'hc_x' not in df.columns or 'hc_y' not in df.columns:
        print(f'  {y}: no hit coordinates in this cache; pull-air excess set to zero for the season', flush=True)
        return {}, float('nan')
    bip = df[df['events'].notna() & df['bb_type'].notna() & df['launch_angle'].notna()].copy()
    bip = bip[~bip['bb_type'].isin(['bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive'])]
    for c in ('hc_x', 'hc_y', 'launch_angle'):
        bip[c] = pd.to_numeric(bip[c], errors='coerce')
    bip = bip[bip['hc_x'].notna() & bip['hc_y'].notna()]
    air = bip['launch_angle'] >= 20
    ang = [spray_angle(x, yv) for x, yv in zip(bip['hc_x'], bip['hc_y'])]
    pulled = np.array([spray_direction(a, s) in PULL_BINS if a is not None else False for a, s in zip(ang, bip['stand'])])
    bip['air'] = air.values; bip['pullair'] = pulled & air.values; bip['h1'] = bip['game_date'].astype(str).str[:10] <= asg
    lg_share = bip['pullair'].sum() / max(bip['air'].sum(), 1)
    out = {}
    for pid, sub in bip.groupby('pitcher'):
        full = float(sub['pullair'].sum() - lg_share * sub['air'].sum()); h1s = sub[sub['h1']]
        out[str(int(pid))] = {'full': full, 'h1': float(h1s['pullair'].sum() - lg_share * h1s['air'].sum())}
    return out, float(lg_share)

def role_gap(y, df):
    """Within-season, within-pitcher: xwOBA against as starter vs reliever (>=50 PA each)."""
    d = df[df['events'].notna()].copy()
    d['order'] = np.arange(len(d))[::-1]           # Savant descending within game -> reverse
    d = d.sort_values(['game_pk', 'order'])
    WOBA = {'walk': 0.69, 'hit_by_pitch': 0.72}
    def pa_val(r):
        ev = r['events']
        if ev in WOBA: return WOBA[ev]
        x = r['estimated_woba_using_speedangle']
        return 0.0 if pd.isna(x) else float(x)
    skip = {'intent_walk', 'sac_bunt', 'catcher_interf', 'sac_bunt_double_play'}
    rows = []
    for gpk, g in d.groupby('game_pk', sort=False):
        pit = g['pitcher'].values; bat = g['batter'].values
        first = pit[0]                                    # top of the 1st: the HOME starter
        sets = defaultdict(set)
        for p_, b_ in zip(pit, bat): sets[p_].add(b_)
        home_side = {p_ for p_ in sets if sets[p_] & sets[first]}
        away_first = next((p_ for p_ in pit if p_ not in home_side), None)
        starters = {first, away_first}
        for r_ in g.itertuples(index=False):
            rr = r_._asdict()
            if rr['events'] in skip: continue
            rows.append((rr['pitcher'], rr['pitcher'] in starters, pa_val(rr)))
    acc = defaultdict(lambda: {'sp': [], 'rp': []})
    for pid, is_sp, v in rows: acc[pid]['sp' if is_sp else 'rp'].append(v)
    gaps = []
    for pid, a in acc.items():
        if len(a['sp']) >= 50 and len(a['rp']) >= 50:
            gaps.append((np.mean(a['rp']) - np.mean(a['sp']), min(len(a['sp']), len(a['rp']))))
    return gaps

def main():
    out = {}
    FR, HS, PX, RG = {}, {}, {}, {}
    gaps_all = []
    for y in SEASONS:
        print(f"season {y} ...", flush=True)
        nm = name_map(y)
        if y == 2026:
            D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb')); pitches = [p for p in D if p.get('_source', 'MLB') == 'MLB']; del D
            FR[y] = framing(y, pitches, nm); HS[y] = home_share(y, sheet=pitches); PX[y] = {}
            del pitches
        else:
            pitches = base.adapt(os.path.join(ROOT, CACHE[y])); FR[y] = framing(y, pitches, nm); del pitches; gc.collect()
            df = df_year(y); HS[y] = home_share(y, df=df); PX[y], lgs = pull_excess(y, df)
            g = role_gap(y, df); gaps_all += [(y, gp, w) for gp, w in g]
            wmean = np.average([gp for gp, _ in g], weights=[w for _, w in g]) if g else float('nan')
            print(f"  {y}: framing {len(FR[y])} arms, home share {len(HS[y])}, pull-air lg share {lgs:.3f}, swingmen {len(g)} gap RP-SP {wmean:+.4f} xwOBA", flush=True)
            del df; gc.collect()
        ph = T[str(y)]['pitchers']
        RG[y] = {pid: {sc: tuple((rec if sc == 'full' else rec.get(sc) or {}).get(k, 0) for k in ('sb', 'cs', 'outs', 'wp', 'bk')) for sc in ('full', 'h1', 'h2')} for pid, rec in ph.items()}
    # swingman gap summary
    if gaps_all:
        w = np.array([x[2] for x in gaps_all], float); gp = np.array([x[1] for x in gaps_all])
        mu = np.average(gp, weights=w); se = math.sqrt(np.average((gp - mu) ** 2, weights=w) / len(gp))
        pa9 = 38.3; sc = 1.23
        print(f"\nWITHIN-SEASON ROLE GAP (same pitcher, same season, >=50 PA each role): n {len(gp)}, RP - SP xwOBA {mu:+.4f} (SE {se:.4f}) => {mu / sc * pa9:+.2f} runs/9 ; by season " + " ".join(f"{y}:{np.average([g for yy, g, _ in gaps_all if yy == y], weights=[ww for yy, _, ww in gaps_all if yy == y]) / sc * pa9:+.2f}" for y in SEASONS[:-1]))
        out['role_gap_within'] = dict(n=len(gp), xw=float(mu), se=float(se), runs9=float(mu / sc * pa9))

    # ── feature frames per (season, scope) and the evaluator ──
    tabs = {y: W.season_table(y) for y in SEASONS}; lgs = {y: W.league(tabs[y]) for y in SEASONS}
    pa9 = {y: sum(r['bf'] if 'bf' in r else T[str(y)]['pitchers'][p]['bf'] for p, r in tabs[y].items()) / (sum(r['outs'] for r in tabs[y].values()) / 27) for y in SEASONS}
    def frame(y, sc):
        t = tabs[y]; F = {}
        for pid, r in t.items():
            if sc == 'full': o, rr, xw, den = r['outs'], r['r'], r['xw'], r['xw_den']
            else:
                h = r[sc]; o = h.get('outs') or 0; rr = h.get('r', 0); xw, den = (r['xw_h1'], r['xw_den_h1']) if sc == 'h1' else (r['xw_h2'], r['xw_den_h2'])
            if o <= 0 or xw is None or den <= 0: continue
            fr = FR[y].get(pid); fr_runs = (fr['full'] if sc == 'full' else fr['h1'] if sc == 'h1' else fr['full'] - fr['h1']) if fr else 0.0
            px = PX[y].get(pid); pxv = (px['full'] if sc == 'full' else px['h1'] if sc == 'h1' else px['full'] - px['h1']) if px else 0.0
            sb, cs, o2, wp, bk = RG[y][pid][sc]
            F[pid] = dict(outs=o, ra9=rr * 27 / o, xw=xw, den=den, fr9=fr_runs * 27 / o, pullx=pxv / den, rg9=(CS_RV * cs - SB_RV * sb) * 27 / o,
                          wpbk9=-WP_RV * (wp + bk) * 27 / o, exp=r['exp'], hs=HS[y].get(pid))
        pool = [f for f in F.values() if f['outs'] >= POOL_MIN_OUTS]
        lg_xw = sum(f['xw'] * f['den'] for f in pool) / sum(f['den'] for f in pool)
        for f in F.values(): f['xw_sh'] = (f['xw'] * f['den'] + N0_XW * lg_xw) / (f['den'] + N0_XW)
        mu = {k: float(np.mean([f[k] for f in pool])) for k in ('xw_sh', 'fr9', 'pullx', 'rg9', 'wpbk9')}; sd = {k: float(np.std([f[k] for f in pool])) for k in ('xw_sh', 'fr9', 'pullx', 'rg9', 'wpbk9')}
        for f in F.values(): f['z'] = {k: (f[k] - mu[k]) / sd[k] if sd[k] > 0 else 0.0 for k in mu}
        return F, dict(anchor=float(np.mean([f['ra9'] for f in pool])), gap=lgs[y]['ra9'] - lgs[y]['era'], lg_ra9=lgs[y]['ra9'], mu=mu, sd=sd)
    FS = {(y, sc): frame(y, sc) for y in SEASONS for sc in ('full', 'h1', 'h2')}
    # channel reliability
    print("\nCHANNEL RELIABILITY (h1 vs h2, >= 30 IP each), raw per-9 values:")
    for k in ('fr9', 'pullx', 'rg9', 'wpbk9'):
        rs = []
        for y in SEASONS:
            A, _ = FS[(y, 'h1')]; Bh, _ = FS[(y, 'h2')]
            ks = [p for p in A if p in Bh and A[p]['outs'] >= 90 and Bh[p]['outs'] >= 90]
            rs.append(W.pear([A[p][k] for p in ks], [Bh[p][k] for p in ks]))
        sds = [FS[(y, 'full')][1]['sd'][k] for y in SEASONS]
        print(f"  {k:6} rel " + " ".join(f"{r:.3f}" for r in rs) + f"  mean {np.mean(rs):.3f}   pool sd {np.mean(sds):.3f}")
        out[f'rel_{k}'] = rs

    def evaluate(name, cols, exp_key='exp'):
        beta = {}
        for hold in SEASONS:
            X, Y = [], []
            for y in SEASONS:
                if y == hold: continue
                F, M = FS[(y, 'full')]
                for pid, f in F.items():
                    if f['outs'] >= 180: X.append([f['z'][c] for c in cols]); Y.append(f['ra9'] - M['anchor'])
            beta[hold] = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), np.array(X)]), np.array(Y), rcond=None)[0]
        def rate(y, sc, pid, park=False):
            F, M = FS[(y, sc)]; f = F.get(pid)
            if f is None: return None
            b = beta[y]; v = M['anchor'] + b[0] + sum(b[i + 1] * f['z'][c] for i, c in enumerate(cols)) + M['gap']
            if park:
                e = f['exp'] if (exp_key == 'exp' or f['hs'] is None) else (f['hs'] * (2 * f['exp'] - 1) + (1 - f['hs']) * 1.0)
                v -= WAR_PARK_PASS * (e - 1) * M['lg_ra9']
            return v
        res = {}
        rel = []
        for y in SEASONS:
            A, _ = FS[(y, 'h1')]; Bh, _ = FS[(y, 'h2')]; ks = [p for p in A if p in Bh and A[p]['outs'] >= 90 and Bh[p]['outs'] >= 90]
            rel.append(W.pear([rate(y, 'h1', p) for p in ks], [rate(y, 'h2', p) for p in ks]))
        res['rel'] = rel
        for gate in (60, 30):
            nx = []
            for y in SEASONS[:-1]:
                A, _ = FS[(y, 'full')]; Bh, _ = FS[(y + 1, 'full')]; ks = [p for p in A if p in Bh and A[p]['outs'] >= gate * 3 and Bh[p]['outs'] >= gate * 3]
                nx.append(W.pear([rate(y, 'full', p, park=True) for p in ks], [Bh[p]['ra9'] for p in ks]))
            res[f'nxt{gate}'] = nx
        ros = []
        for y in SEASONS:
            A, _ = FS[(y, 'h1')]; Bh, _ = FS[(y, 'h2')]; ks = [p for p in A if p in Bh and A[p]['outs'] >= 90 and Bh[p]['outs'] >= 90]
            ros.append(W.pear([rate(y, 'h1', p) for p in ks], [Bh[p]['ra9'] for p in ks]))
        res['ros'] = ros
        cal = []
        for y in SEASONS:
            F, _ = FS[(y, 'full')]; ks = [p for p in F if F[p]['outs'] >= 180]
            cal.append(float(np.polyfit([rate(y, 'full', p, park=True) for p in ks], [F[p]['ra9'] for p in ks], 1)[0]))
        res['calib'] = cal; res['beta_2024'] = list(map(float, beta[2024]))
        out[name] = res
        print(f"  {name:14} rel {np.mean(rel):.3f}  nxt60 {np.mean(res['nxt60']):.3f}  nxt30 {np.mean(res['nxt30']):.3f}  ros {np.mean(ros):.3f}  calib {np.mean(cal):.3f}  b {np.round(beta[2024][1:], 3).tolist()}")
        return res
    print("\nVARIANTS (LOSO OLS on RA9; nxt uses the park-adjusted rate):")
    ship = evaluate('ship', ('xw_sh',))
    evaluate('ship+home', ('xw_sh',), exp_key='hs')
    evaluate('framing', ('xw_sh', 'fr9'))
    evaluate('pullair', ('xw_sh', 'pullx'))
    evaluate('rungame', ('xw_sh', 'rg9'))
    evaluate('wp+bk', ('xw_sh', 'wpbk9'))
    evaluate('all four', ('xw_sh', 'fr9', 'pullx', 'rg9', 'wpbk9'))
    print("\nPAIRED vs ship:")
    for name, res in out.items():
        if name in ('ship',) or not isinstance(res, dict) or 'rel' not in res: continue
        line = []
        for k in ('rel', 'nxt60', 'nxt30', 'ros', 'calib'):
            d = np.array(res[k]) - np.array(ship[k]); line.append(f"{k} {d.mean():+.4f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {name:14} " + "  ".join(line))
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_improve_battery2.json'), 'w'), indent=1, default=float)
    print("\nwrote data/_war_improve_battery2.json")

if __name__ == '__main__':
    main()
