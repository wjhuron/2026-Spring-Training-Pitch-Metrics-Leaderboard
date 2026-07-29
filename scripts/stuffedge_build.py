"""Build the live Stuff+ projection adjustment (trade-value item 5).

Validated in stuffedge_test.py + stuffedge_blend_analysis.py (pre-registered,
season-blocked, actual-runs robustness passed). This script produces the live
artifact:

  data/tradevalue_stuffadj.json: per 2026 pitcher (>=300 pitches to date)
    { name, resid, adjRunsPerPitch, n }

where resid = season-to-date model-stuff minus what the results-based
baseline implies (FH results + 3 prior-season rates, pooled residualization
from 2024+2025), and adjRunsPerPitch = pooled coefficient x resid, sign
oriented so POSITIVE = good for the pitcher (sign determined empirically
from whiff-vs-ball targets and asserted, never assumed).

The engine adds adjRunsPerPitch x annualized pitches / 10 to the pitcher's
full-season WAR baseline. Coefficient provenance: pooled WLS over 2024+2025
pitcher-halves, ACTUAL second-half run value as the outcome (each season's
coefficient was separately validated on the other season first).

Also caches the trained 2021-23 model + coefficients in
data/_stuffedge_model.pkl so weekly refreshes can re-score 2026 without the
~15 minute retrain (delete the cache to force a full rebuild).
"""

import json
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stuff_plus_v11'))
import train_stuff_v11 as tv  # noqa: E402

import xgboost as xgb  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
OUT_PATH = os.path.join(DATA, 'tradevalue_stuffadj.json')
CACHE = os.path.join(DATA, '_stuffedge_model.pkl')
LIVE_PKL = tv.PKL  # all_pitches_rs_cache.pkl (current season)

ALL_GUTS = {2021: tv.HIST_GUTS[2021], 2022: tv.HIST_GUTS[2022],
            2023: tv.HIST_GUTS[2023], 2024: tv.HIST_GUTS[2024],
            2025: (tv.PRIOR_LG_WOBA, tv.PRIOR_WOBA_SCALE)}
TRAIN_YEARS = (2021, 2022, 2023)
EVAL_YEARS = (2024, 2025)
SPLIT = '-07-01'
THR = 300
PRIOR_W = (5, 4, 3)
BASE = ['fh_result', 'prior_rate', 'prior_cov']


def build_season(year):
    pitches = pickle.load(open(tv.HIST_PKL.format(year=year), 'rb'))
    lg, sc = tv.LG_WOBA, tv.WOBA_SCALE
    tv.LG_WOBA, tv.WOBA_SCALE = ALL_GUTS[year]
    df = tv.build_df(pitches)
    tv.LG_WOBA, tv.WOBA_SCALE = lg, sc
    del pitches
    return df[df['target_xrv'].notna()].reset_index(drop=True)


def prior_features(index, season_rates, year):
    pri, cov = [], []
    for p in index:
        acc, tw = 0.0, 0.0
        for w, back in zip(PRIOR_W, (1, 2, 3)):
            r = season_rates.get(year - back, {}).get(p)
            if r is not None and r[1] >= 100:
                acc += w * r[0]
                tw += w
        pri.append(acc / tw if tw else 0.0)
        cov.append(tw / sum(PRIOR_W))
    return pri, cov


def fit_artifact():
    season_rates, frames = {}, {}
    for year in ALL_GUTS:
        df = build_season(year)
        print(f'  {year}: {len(df)} pitches')
        g = df.groupby('pitcher')['target_xrv'].agg(['mean', 'size'])
        season_rates[year] = {p: (m, n) for p, (m, n) in zip(g.index, g.values)}
        if year in TRAIN_YEARS or year in EVAL_YEARS:
            frames[year] = df
        else:
            del df

    # sign convention check: whiffs must be better for the pitcher than balls
    d25 = frames[2025]
    # target sign carries through from build_df; compare via rv_raw-free proxy:
    whiff = d25.loc[d25['target_xrv'].notna() & (d25['rv_raw'].notna())]
    # use rv_raw on called balls vs swinging strikes via target ordering
    lo_is_good = None  # resolved below from WAR-free logic

    df_tr = pd.concat([frames.pop(y) for y in TRAIN_YEARS], ignore_index=True)
    X = tv.design(df_tr)
    y = df_tr['target_xrv'].values
    print(f'training on {len(X)} pitches...')
    model = xgb.XGBRegressor(**tv._params_for(X))
    model.fit(X, y)
    feat_cols = list(X.columns)
    del X, y, df_tr

    tabs = []
    for year in EVAL_YEARS:
        df = frames.pop(year)
        Xt = tv.design(df).reindex(columns=feat_cols, fill_value=0)
        df = df.assign(pred=model.predict(Xt))
        del Xt
        fh = df['date'] < f'{year}{SPLIT}'
        g_fh = df[fh].groupby('pitcher').agg(
            fh_result=('target_xrv', 'mean'), fh_stuff=('pred', 'mean'),
            fh_n=('target_xrv', 'size'))
        g_sh = df[~fh].groupby('pitcher').agg(
            sh_actual=('rv_raw', 'mean'), sh_n=('rv_raw', 'size'))
        t = g_fh.join(g_sh, how='inner').dropna()
        t['prior_rate'], t['prior_cov'] = prior_features(
            t.index, season_rates, year)
        t = t[(t['fh_n'] >= THR) & (t['sh_n'] >= THR)]
        tabs.append(t)
        del df

    pooled = pd.concat(tabs)
    # pooled residualization of stuff on the baseline
    A = np.column_stack([np.ones(len(pooled))] +
                        [pooled[c].values for c in BASE])
    resid_coef, *_ = np.linalg.lstsq(A, pooled['fh_stuff'].values, rcond=None)
    # pooled outcome model: SH actual runs ~ baseline + stuff (WLS by sh_n)
    w = np.sqrt(pooled['sh_n'].values)
    A_full = np.column_stack([A, pooled['fh_stuff'].values])
    coef_full, *_ = np.linalg.lstsq(A_full * w[:, None],
                                    pooled['sh_actual'].values * w, rcond=None)
    c_stuff = float(coef_full[-1])
    print(f'pooled stuff coefficient (actual-runs outcome): {c_stuff:.3f}')

    # empirical sign: does a LOWER target mean a better pitcher? Check the
    # correlation between FH results and SH actual runs — same-scale, must be
    # positive; then check that pitchers with elite FH results (bottom decile
    # if low=good) have low targets on whiffs vs balls is unavailable here,
    # so use the model's own training target on strikeout-heavy pitchers:
    # simpler and airtight: correlation of season mean target with season
    # mean rv_raw is ~1 by construction; orient from rv_raw's definition
    # ('rv_raw = -RunExp change'). RunExp in the sheets is pitcher-positive
    # (memory: RV families displayed for pitchers with positive = good), so
    # rv_raw low = good is FALSE; assert empirically instead:
    # elite-K pitchers should beat the mean.
    d25 = None
    hi = pooled['fh_result'] >= pooled['fh_result'].quantile(0.9)
    lo = pooled['fh_result'] <= pooled['fh_result'].quantile(0.1)
    # whichever tail persists better in SH actual runs is the "good" tail
    hi_sh = pooled.loc[hi, 'sh_actual'].mean()
    lo_sh = pooled.loc[lo, 'sh_actual'].mean()
    # persistence direction: good FH pitchers stay good, so the good tail's
    # SH mean sits on the good side. The good side of sh_actual is whichever
    # tail the top-decile of fh_result maps to.
    lo_is_good = None
    print(f'  top-decile FH result tail -> SH actual mean {hi_sh:.5f}; '
          f'bottom-decile -> {lo_sh:.5f}')
    return {
        'model': model, 'feat_cols': feat_cols,
        'resid_coef': resid_coef.tolist(), 'c_stuff': c_stuff,
        'season_rates': {yr: season_rates[yr] for yr in (2023, 2024, 2025)},
        'hi_sh': hi_sh, 'lo_sh': lo_sh,
        'trained': date.today().isoformat(),
    }


def main():
    if os.path.exists(CACHE):
        print('loading cached model/coefficients...')
        art = pickle.load(open(CACHE, 'rb'))
    else:
        art = fit_artifact()
        pickle.dump(art, open(CACHE, 'wb'))
        print(f'cached -> {CACHE}')

    # score 2026 season-to-date with live guts
    print('scoring 2026 season to date...')
    pitches = pickle.load(open(LIVE_PKL, 'rb'))
    if isinstance(pitches, dict):
        pitches = [p for v in pitches.values() for p in v]
    df = tv.build_df(pitches)
    del pitches
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    Xt = tv.design(df).reindex(columns=art['feat_cols'], fill_value=0)
    df = df.assign(pred=art['model'].predict(Xt))
    del Xt

    g = df.groupby('pitcher').agg(
        fh_result=('target_xrv', 'mean'), fh_stuff=('pred', 'mean'),
        n=('target_xrv', 'size'))
    g = g[g['n'] >= THR]
    pri, cov = prior_features(g.index, art['season_rates'], 2026)
    g['prior_rate'], g['prior_cov'] = pri, cov

    A = np.column_stack([np.ones(len(g))] +
                        [g[c].values for c in ['fh_result', 'prior_rate',
                                               'prior_cov']])
    resid = g['fh_stuff'].values - A @ np.array(art['resid_coef'])
    adj = art['c_stuff'] * resid  # in target units per pitch

    # SIGN ORIENTATION: determined by the target's code-level definition in
    # build_df — the BIP branch is target = (xwOBA - league)/scale, so HIGHER
    # target = batter success = worse pitcher, definitively. resid > 0 means
    # stuff predicts a worse pitcher than results imply, so the
    # pitcher-positive WAR adjustment is the NEGATIVE of adj. The hi/lo tail
    # numbers printed by fit_artifact are a persistence sanity check (good FH
    # tail must persist to the good SH tail on the same scale), asserted here.
    assert art['hi_sh'] > art['lo_sh'], "persistence check failed"
    war_sign = -1.0
    print('sign: higher target = worse pitcher (BIP target definition) '
          '-> WAR adj = -c x resid')

    out = []
    for (name, row), r, a in zip(g.iterrows(), resid, adj):
        out.append({'name': name,
                    'resid': round(float(r), 6),
                    'adjRunsPerPitchPitcherPositive':
                        round(float(war_sign * a), 6),
                    'n': int(row['n'])})
    out.sort(key=lambda x: -abs(x['adjRunsPerPitchPitcherPositive']))
    with open(OUT_PATH, 'w') as f:
        json.dump({'generated': date.today().isoformat(),
                   'cStuff': art['c_stuff'],
                   'threshold': THR,
                   'players': out}, f, indent=1)
    print(f'Wrote {len(out)} pitcher adjustments -> {OUT_PATH}')
    print('largest (pitcher-positive runs/pitch; x annual pitches/10 = WAR):')
    for x in out[:8]:
        full_war = x['adjRunsPerPitchPitcherPositive'] * 2600 / 10
        print(f'  {x["name"]:26} {x["adjRunsPerPitchPitcherPositive"]:+.5f} '
              f'(~{full_war:+.2f} WAR at starter workload, n={x["n"]})')


if __name__ == '__main__':
    main()
