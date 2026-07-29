"""Blend analysis (report-only): stuff's increment over a PROJECTION-LIKE baseline.

Follow-up to stuffedge_test.py (which passed but used raw FH results as the
baseline). Here the results-only baseline gets everything results-based a
projection would use: FH mean xRV plus the pitcher's three prior season xRV
rates, combined by regression (weights fitted, not hand-picked). Question:
what does FH stuff still add, out of season, and how big is the implied
WAR adjustment?

Protocol: model trained on 2021-2023 (production v11 features) scores
2024/2025 held-out. Two-way cross-season transfer at the 300/300-pitch
threshold. Nothing here touches the live engine.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stuff_plus_v11'))
import train_stuff_v11 as tv  # noqa: E402

import xgboost as xgb  # noqa: E402

ALL_GUTS = {2021: tv.HIST_GUTS[2021], 2022: tv.HIST_GUTS[2022],
            2023: tv.HIST_GUTS[2023], 2024: tv.HIST_GUTS[2024],
            2025: (tv.PRIOR_LG_WOBA, tv.PRIOR_WOBA_SCALE)}
TRAIN_YEARS = (2021, 2022, 2023)
EVAL_YEARS = (2024, 2025)
SPLIT = '-07-01'
THR = 300
PRIOR_W = (5, 4, 3)          # weights on seasons S-1, S-2, S-3
PITCHES_PER_SEASON = 2600    # full-time pitcher, for WAR translation
RUNS_PER_WIN = 10.0


def build_season(year):
    pkl = tv.HIST_PKL.format(year=year)
    pitches = pickle.load(open(pkl, 'rb'))
    lg, sc = tv.LG_WOBA, tv.WOBA_SCALE
    tv.LG_WOBA, tv.WOBA_SCALE = ALL_GUTS[year]
    df = tv.build_df(pitches)
    tv.LG_WOBA, tv.WOBA_SCALE = lg, sc
    del pitches
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    print(f'  {year}: {len(df)} pitches')
    return df


def main():
    season_rates = {}   # year -> pitcher -> (mean, n): full-season results
    frames = {}
    for year in ALL_GUTS:
        df = build_season(year)
        g = df.groupby('pitcher')['target_xrv'].agg(['mean', 'size'])
        season_rates[year] = {p: (m, n) for p, (m, n)
                              in zip(g.index, g.values)}
        if year in TRAIN_YEARS or year in EVAL_YEARS:
            frames[year] = df
        else:
            del df

    df_tr = pd.concat([frames.pop(y) for y in TRAIN_YEARS], ignore_index=True)
    X = tv.design(df_tr)
    y = df_tr['target_xrv'].values
    print(f'training on {len(X)} pitches...')
    model = xgb.XGBRegressor(**tv._params_for(X))
    model.fit(X, y)
    feat_cols = list(X.columns)
    del X, y, df_tr

    tables = {}
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
            sh_result=('target_xrv', 'mean'), sh_n=('target_xrv', 'size'),
            sh_actual=('rv_raw', 'mean'))
        t = g_fh.join(g_sh, how='inner').dropna()
        # projection-like prior: weighted mean of prior-season full rates
        pri, pri_w = [], []
        for p in t.index:
            acc, tw = 0.0, 0.0
            for w, back in zip(PRIOR_W, (1, 2, 3)):
                r = season_rates.get(year - back, {}).get(p)
                if r is not None and r[1] >= 100:
                    acc += w * r[0]
                    tw += w
            pri.append(acc / tw if tw else 0.0)
            pri_w.append(tw / sum(PRIOR_W))
        t['prior_rate'] = pri
        t['prior_cov'] = pri_w
        t = t[(t['fh_n'] >= THR) & (t['sh_n'] >= THR)]
        tables[year] = t
        del df

    def wls(cols, frame):
        w = np.sqrt(frame['sh_n'].values)
        A = np.column_stack([np.ones(len(frame))] +
                            [frame[c].values for c in cols])
        coef, *_ = np.linalg.lstsq(A * w[:, None],
                                   frame['sh_result'].values * w, rcond=None)
        return coef

    def wmse(cols, coef, frame):
        A = np.column_stack([np.ones(len(frame))] +
                            [frame[c].values for c in cols])
        return float(np.average((frame['sh_result'].values - A @ coef) ** 2,
                                weights=frame['sh_n'].values))

    BASE = ['fh_result', 'prior_rate', 'prior_cov']
    FULL = BASE + ['fh_stuff']

    print('\n=== cross-season transfer, projection-like baseline ===')
    for fit_yr in EVAL_YEARS:
        eval_yr = EVAL_YEARS[1] if fit_yr == EVAL_YEARS[0] else EVAL_YEARS[0]
        f, e = tables[fit_yr], tables[eval_yr]
        m_base = wmse(BASE, wls(BASE, f), e)
        coef_full = wls(FULL, f)
        m_full = wmse(FULL, coef_full, e)
        print(f'fit {fit_yr} -> eval {eval_yr} (n={len(e)}): '
              f'baseline wMSE={m_base:.7f}  +stuff={m_full:.7f}  '
              f'improvement={100 * (1 - m_full / m_base):.1f}%  '
              f'stuff coef={coef_full[-1]:.3f}')

    print('\n=== ROBUSTNESS: same test, SH outcome = ACTUAL run value ===')
    for fit_yr in EVAL_YEARS:
        eval_yr = EVAL_YEARS[1] if fit_yr == EVAL_YEARS[0] else EVAL_YEARS[0]
        f = tables[fit_yr].rename(columns={'sh_result': 'sh_xrv',
                                           'sh_actual': 'sh_result'})
        e = tables[eval_yr].rename(columns={'sh_result': 'sh_xrv',
                                            'sh_actual': 'sh_result'})
        m_base = wmse(BASE, wls(BASE, f), e)
        coef_full = wls(FULL, f)
        m_full = wmse(FULL, coef_full, e)
        print(f'fit {fit_yr} -> eval {eval_yr}: baseline wMSE={m_base:.7f}  '
              f'+stuff={m_full:.7f}  improvement={100 * (1 - m_full / m_base):.1f}%  '
              f'stuff coef={coef_full[-1]:.3f}')
    for year, t in tables.items():
        def resid_a(a, cols):
            A = np.column_stack([np.ones(len(t))] +
                                [t[c].values for c in cols])
            c, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ c
        r = float(np.corrcoef(resid_a(t['fh_stuff'].values, BASE),
                              resid_a(t['sh_actual'].values, BASE))[0, 1])
        print(f'{year}: partial(stuff | baseline) on ACTUAL SH runs = {r:.3f}')

    print('\n=== partial correlation of stuff given the full baseline ===')
    for year, t in tables.items():
        def resid(a, cols):
            A = np.column_stack([np.ones(len(t))] +
                                [t[c].values for c in cols])
            c, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ c
        r = float(np.corrcoef(resid(t['fh_stuff'].values, BASE),
                              resid(t['sh_result'].values, BASE))[0, 1])
        print(f'{year}: partial(stuff | FH results + priors) = {r:.3f}  (n={len(t)})')

    print('\n=== implied WAR adjustment sizes (coef from the other season) ===')
    for year in EVAL_YEARS:
        other = EVAL_YEARS[1] if year == EVAL_YEARS[0] else EVAL_YEARS[0]
        coef = wls(FULL, tables[other])
        t = tables[year]
        A = np.column_stack([np.ones(len(t))] +
                            [t[c].values for c in BASE])
        c_r, *_ = np.linalg.lstsq(A, t['fh_stuff'].values, rcond=None)
        stuff_resid = t['fh_stuff'].values - A @ c_r
        adj_war = coef[-1] * stuff_resid * PITCHES_PER_SEASON / RUNS_PER_WIN
        q = np.percentile(np.abs(adj_war), [50, 90, 99])
        print(f'{year}: |WAR adj/season| median={q[0]:.2f}  p90={q[1]:.2f}  '
              f'p99={q[2]:.2f}  max={np.abs(adj_war).max():.2f}')
        top = np.argsort(-np.abs(adj_war))[:5]
        for i in top:
            print(f'    {t.index[i]:26} adj {adj_war[i]:+.2f} WAR '
                  f'(stuff resid {stuff_resid[i] * 1000:+.2f} xRV/1000)')


if __name__ == '__main__':
    main()
