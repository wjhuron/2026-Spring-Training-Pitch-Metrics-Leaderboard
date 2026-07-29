"""Projection-edge test: does Stuff+ add forecasting signal beyond results?

Question (trade-value item 5): would blending pitch-model metrics into the
WAR projection beat results-only inputs? Reduced to the cleanest testable
form: does FIRST-HALF pitch-level stuff (model-predicted xRV) predict
SECOND-HALF luck-neutral outcomes (target_xrv) beyond first-half outcomes
themselves?

Leakage protocol (season-blocked, matching the v11 validation scheme): an
XGBoost model with the production v11 features/params is trained on
2021-2023 ONLY, then scores 2024 and 2025 as fully held-out replicate
seasons. (The shipped v11 bundle trained on pooled 2021-2025, so it cannot
be used here.)

PRE-REGISTERED CRITERIA (before running): stuff has incremental signal iff
  (a) partial correlation of FH stuff with SH outcome, controlling FH
      outcome, is positive in BOTH held-out seasons at the primary
      threshold (>=300 FH and >=300 SH pitches), AND
  (b) cross-season transfer: a two-feature linear model fit on one held-out
      season lowers pitch-weighted SH MSE on the OTHER season vs the
      baseline-only model, in both directions.
Anything less is a null result and the projection blend is not built.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stuff_plus_v11'))
import train_stuff_v11 as tv  # noqa: E402  (feature pipeline parity)

import xgboost as xgb  # noqa: E402

TRAIN_YEARS = (2021, 2022, 2023)
TEST_GUTS = {2024: tv.HIST_GUTS[2024],
             2025: (tv.PRIOR_LG_WOBA, tv.PRIOR_WOBA_SCALE)}
SPLIT = '-07-01'          # FH = before July 1
THRESHOLDS = (200, 300, 500)
PRIMARY = 300


def build_season(year, guts):
    pkl = tv.HIST_PKL.format(year=year)
    pitches = pickle.load(open(pkl, 'rb'))
    lg, sc = tv.LG_WOBA, tv.WOBA_SCALE
    tv.LG_WOBA, tv.WOBA_SCALE = guts
    df = tv.build_df(pitches)
    tv.LG_WOBA, tv.WOBA_SCALE = lg, sc
    del pitches
    df = df[df['target_xrv'].notna()].reset_index(drop=True)
    print(f'  {year}: {len(df)} pitches with targets')
    return df


def partial_corr(x, y, z):
    """corr(x, y | z) via residualization."""
    def resid(a, b):
        b1 = np.column_stack([np.ones(len(b)), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    return float(np.corrcoef(resid(x, z), resid(y, z))[0, 1])


def season_table(df, pred, year):
    df = df.assign(pred=pred)
    fh = df['date'] < f'{year}{SPLIT}'
    g_fh = df[fh].groupby('pitcher').agg(
        fh_result=('target_xrv', 'mean'), fh_stuff=('pred', 'mean'),
        fh_n=('target_xrv', 'size'))
    g_sh = df[~fh].groupby('pitcher').agg(
        sh_result=('target_xrv', 'mean'), sh_n=('target_xrv', 'size'))
    return g_fh.join(g_sh, how='inner').dropna()


def main():
    print('building training frames (2021-2023)...')
    frames = [build_season(y, tv.HIST_GUTS[y]) for y in TRAIN_YEARS]
    df_tr = pd.concat(frames, ignore_index=True)
    del frames
    X = tv.design(df_tr)
    y = df_tr['target_xrv'].values
    print(f'training on {len(X)} pitches, {X.shape[1]} features...')
    model = xgb.XGBRegressor(**tv._params_for(X))
    model.fit(X, y)
    feat_cols = list(X.columns)
    del X, y, df_tr

    tables = {}
    for year, guts in TEST_GUTS.items():
        df = build_season(year, guts)
        Xt = tv.design(df).reindex(columns=feat_cols, fill_value=0)
        pred = model.predict(Xt)
        tables[year] = season_table(df, pred, year)
        del df, Xt

    print('\n=== per-season results ===')
    prim = {}
    for year, t in tables.items():
        for thr in THRESHOLDS:
            s = t[(t['fh_n'] >= thr) & (t['sh_n'] >= thr)]
            r_base = float(np.corrcoef(s['fh_result'], s['sh_result'])[0, 1])
            r_stuff = float(np.corrcoef(s['fh_stuff'], s['sh_result'])[0, 1])
            r_part = partial_corr(s['fh_stuff'].values, s['sh_result'].values,
                                  s['fh_result'].values)
            tag = ' <- primary' if thr == PRIMARY else ''
            print(f'{year} thr={thr}: n={len(s):4}  '
                  f'r(FHres,SH)={r_base:.3f}  r(FHstuff,SH)={r_stuff:.3f}  '
                  f'partial(stuff|res)={r_part:.3f}{tag}')
            if thr == PRIMARY:
                prim[year] = (s, r_part)

    print('\n=== cross-season transfer (weighted MSE, primary threshold) ===')
    transfer_ok = True
    years = sorted(prim)
    for fit_yr in years:
        eval_yr = years[1] if fit_yr == years[0] else years[0]
        f, e = prim[fit_yr][0], prim[eval_yr][0]
        w_f, w_e = f['sh_n'].values, e['sh_n'].values

        def wls(cols, frame, w):
            A = np.column_stack([np.ones(len(frame))] +
                                [frame[c].values for c in cols])
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A * sw[:, None],
                                       frame['sh_result'].values * sw,
                                       rcond=None)
            return coef

        for label, cols in (('baseline', ['fh_result']),
                            ('with stuff', ['fh_result', 'fh_stuff'])):
            coef = wls(cols, f, w_f)
            A_e = np.column_stack([np.ones(len(e))] +
                                  [e[c].values for c in cols])
            mse = float(np.average((e['sh_result'].values - A_e @ coef) ** 2,
                                   weights=w_e))
            print(f'fit {fit_yr} -> eval {eval_yr}  {label:11}: '
                  f'wMSE={mse:.7f}')
            if label == 'baseline':
                base_mse = mse
        if mse >= base_mse:
            transfer_ok = False

    both_positive = all(r > 0 for _, r in prim.values())
    print('\n=== verdict (pre-registered) ===')
    print(f'(a) partial corr positive in both seasons: {both_positive} '
          f'({ {y: round(r, 3) for y, (_, r) in prim.items()} })')
    print(f'(b) cross-season transfer improves both directions: {transfer_ok}')
    print('BUILD the projection blend' if (both_positive and transfer_ok)
          else 'NULL RESULT - do not build the blend')


if __name__ == '__main__':
    main()
