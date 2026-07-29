"""changed_teams_holdout.py — does Stuff+ predict NEXT-year results for pitchers
who CHANGED teams? (BP's honest predictive test; report validation item)

Same-team YoY confounds Stuff+->results with persistent defense/park/framing.
Switchers reset all three, so their year-N Stuff+ -> year-(N+1) results is a clean
read on skill that TRAVELS WITH THE ARM. FanGraphs Stuff+ fell .41 (same team) ->
.14 (switchers) predicting ERA. Wally's target is luck-neutral (xwOBA-on-contact,
defense-independent), so his gap should be SMALLER — the hypothesis under test.

Predictor:  overall Stuff+ per (pitcher, year) = mean(-xRV), one pooled 6-season model.
Results (year N+1), two metrics:
  rv100    = mean(RunExp)*100      [RunExp=delta_pitcher_run_exp, HIGHER=better]  -> expect +corr
  xwobacon = mean(xwOBA on BIP)    [HIGHER=worse]                                 -> expect -corr
Both stuff and result z-scored WITHIN season before pooling the 5 transitions, so
season-level shifts don't leak into the pooled correlation.

Usage: python3 scripts/changed_teams_holdout.py   (needs data/_team_maps.pkl)
"""
import os, sys, pickle, warnings
from collections import defaultdict
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
import scripts.build_historical_training_set as H

GUTS = dict(H.GUTS); GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)
GUTS[2026] = (T.LG_WOBA, T.WOBA_SCALE)
MIN_PITCHES = 400
sf = T.sf


def load_pitches(year):
    if year == 2026:
        return [p for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
                if p.get('_source') == 'MLB']
    return pickle.load(open(os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl'), 'rb'))


def zscore(d, key):
    vals = np.array([v[key] for v in d.values()], float)
    m, s = np.nanmean(vals), np.nanstd(vals)
    for v in d.values():
        v[key + '_z'] = (v[key] - m) / s if s else 0.0


def pear(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 8:
        return None, int(ok.sum())
    return float(np.corrcoef(xs[ok], ys[ok])[0, 1]), int(ok.sum())


def main():
    teams = pickle.load(open(os.path.join(ROOT, 'data', '_team_maps.pkl'), 'rb'))
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    raw = {y: load_pitches(y) for y in years}

    # ---- train one pooled model on all seasons; score per pitch ----
    dfs = {}
    for y in years:
        lg, sc = GUTS[y]
        _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
        T.LG_WOBA, T.WOBA_SCALE = lg, sc
        d = T.build_df(raw[y])
        T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
        d = d[d.target_xrv.notna()].reset_index(drop=True)
        d['year'] = y
        dfs[y] = d
        print(f'built {y}: {len(d)}', flush=True)
    allp = pd.concat(dfs.values(), ignore_index=True)
    X = T.design(allp); ymodel = allp.target_xrv.values
    print('training pooled model ...', flush=True)
    m = xgb.XGBRegressor(**T._params_for(X)); m.fit(X, ymodel)
    allp['stuff'] = -m.predict(X)

    # ---- per (pitcher, year): mean stuff, results, pitch count ----
    stuff_py = allp.groupby(['year', 'pitcher']).agg(stuff=('stuff', 'mean'), n=('stuff', 'size'))
    # results from raw dicts (RunExp all pitches; xwOBA on BIP)
    res = defaultdict(lambda: {'rv': 0.0, 'rvn': 0, 'xw': 0.0, 'xwn': 0})
    for y in years:
        for p in raw[y]:
            name = p.get('Pitcher')
            if not name:
                continue
            re = sf(p.get('RunExp'))
            if re is not None:
                res[(y, name)]['rv'] += re; res[(y, name)]['rvn'] += 1
            if p.get('Description') == 'In Play':
                xw = sf(p.get('xwOBA'))
                if xw is not None:
                    res[(y, name)]['xw'] += xw; res[(y, name)]['xwn'] += 1

    # per-season dict: pitcher -> {stuff, rv100, xwobacon} for qualified arms
    season = {y: {} for y in years}
    for (y, name), row in stuff_py.iterrows():
        if row['n'] < MIN_PITCHES:
            continue
        r = res.get((y, name), {})
        rv100 = (r['rv'] / r['rvn'] * 100) if r.get('rvn') else np.nan
        xwc = (r['xw'] / r['xwn']) if r.get('xwn') else np.nan
        season[y][name] = {'stuff': row['stuff'], 'rv100': rv100, 'xwobacon': xwc}
    for y in years:
        for k in ('stuff', 'rv100', 'xwobacon'):
            zscore(season[y], k)
        print(f'{y}: {len(season[y])} qualified (>= {MIN_PITCHES} pitches)', flush=True)

    # ---- assemble transitions: year-N stuff vs year-(N+1) results ----
    same = {'rv': ([], []), 'xw': ([], [])}
    switch = {'rv': ([], []), 'xw': ([], [])}
    n_same = n_switch = n_skip = 0
    for n in years[:-1]:
        m1 = n + 1
        for name, cur in season[n].items():
            nxt = season[m1].get(name)
            if nxt is None:
                continue
            t0, t1 = teams.get(n, {}).get(name), teams.get(m1, {}).get(name)
            if not t0 or not t1:
                n_skip += 1
                continue
            moved = t1['primary'] != t0['primary']
            bucket = switch if moved else same
            n_switch += moved; n_same += (not moved)
            bucket['rv'][0].append(cur['stuff_z']); bucket['rv'][1].append(nxt['rv100_z'])
            bucket['xw'][0].append(cur['stuff_z']); bucket['xw'][1].append(nxt['xwobacon_z'])

    print(f'\ntransitions: same-team={n_same}  switchers={n_switch}  skipped(no team)={n_skip}\n')
    print(f"{'group':10s} {'n':>5s} {'Stuff+ -> next-yr RV100':>26s} {'Stuff+ -> next-yr xwOBAcon':>28s}")
    print('  (RV100: higher r = better prediction, expect +;  xwOBAcon: expect -, we flip sign)')
    for label, grp in [('SAME-TEAM', same), ('SWITCHERS', switch)]:
        rrv, nrv = pear(grp['rv'][0], grp['rv'][1])
        rxw, nxw = pear(grp['xw'][0], grp['xw'][1])
        rxw_flip = -rxw if rxw is not None else None
        print(f"{label:10s} {nrv:5d}   RV100 r = {rrv:+.3f}            xwOBAcon r(pred) = {rxw_flip:+.3f}")
    # gap summary
    rrv_s, _ = pear(same['rv'][0], same['rv'][1]); rrv_w, _ = pear(switch['rv'][0], switch['rv'][1])
    rxw_s, _ = pear(same['xw'][0], same['xw'][1]); rxw_w, _ = pear(switch['xw'][0], switch['xw'][1])
    print(f"\nSAME->SWITCH drop:  RV100 {rrv_s:+.3f}->{rrv_w:+.3f} (Δ{rrv_w-rrv_s:+.3f})   "
          f"xwOBAcon {-rxw_s:+.3f}->{-rxw_w:+.3f} (Δ{-(rxw_w-rxw_s):+.3f})")
    print("FanGraphs Stuff+ -> ERA benchmark: .41 -> .14 (Δ-0.27). Smaller drop = travels better.")


if __name__ == '__main__':
    main()
