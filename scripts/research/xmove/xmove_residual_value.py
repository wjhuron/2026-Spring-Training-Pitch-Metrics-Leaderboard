"""Which expected-movement residual explains pitch RESULTS better?

Wally's question (2026-09-03): to explain why a specific pitch does well, is
the useful number the deviation from what the HITTER expects (slot only: what
he can see) or from what the PHYSICS expects (slot, extension, velocity, spin,
release axis)? Raw movement is the control, because the slot-only residual is
85-91% a copy of it.

Unit: pitcher x pitch type x hand x season, 50+ pitches, 2021-2025. Outcomes:
whiff rate (swinging strikes over swings) and mean RunExp per pitch (run value,
negative = good for the pitcher). Each expected-movement form is cross-fit by
game parity within (type, hand, season) so the residual never saw its own game.

Test, per pitch type: fit outcome ~ (ivb_res, hb_res) on four seasons, score
the fifth, rotate, report the out-of-season correlation between predicted and
actual outcome. Then the same with raw IVB/HB ADDED to each model, to show
whether a residual adds anything a hitter-facing reader could not get from
the raw column.

Usage: python3 scripts/research/xmove/xmove_residual_value.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from xmove_agnostic_flight import SCRATCH, SEASONS, PTS  # noqa: E402
from xmove_pertype_ladder import to_arrays, fit_cv, B0, B1  # noqa: E402

# 2025 carries the Sheets strings, 2021-2024 the Savant codes (found 2026-09-03:
# a mapping on the strings alone gave four seasons of zero swings and a fake
# r = -0.92). Bunts stay out of swings on both sides, per the repo rule.
SWINGS = {'Swinging Strike', 'Foul', 'In Play', 'Foul Tip',
          'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'}
WHIFFS = {'Swinging Strike', 'swinging_strike', 'swinging_strike_blocked'}
COLS = ['Pitcher', 'Throws', 'Pitch Type', 'Velocity', 'Spin Rate', 'SpinAxis',
        'xIndVrtBrk', 'xHorzBrk', 'Extension', 'ArmAngle', 'KinEff', 'KinCd',
        'RelPosZ', 'RelPosX', '_game_pk', 'Description', 'RunExp']
FORMS = [('hitter: slot only', ['aa']), ('old: slot+ext+velo', B0), ('physics: +spin+axis', B1)]


def load(year):
    path = f'{SCRATCH}/xmove_value_{year}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    with open(f'{ROOT}/data/_pitches{year}_training.pkl', 'rb') as f:
        rows = pickle.load(f)
    df = pd.DataFrame([{c: r.get(c) for c in COLS} for r in rows])
    for c in ['Velocity', 'Spin Rate', 'SpinAxis', 'xIndVrtBrk', 'xHorzBrk', 'Extension',
              'ArmAngle', 'KinEff', 'KinCd', 'RelPosZ', 'RelPosX', 'RunExp']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['season'] = year
    df.to_parquet(path, index=False)
    return df


def main():
    d = pd.concat([load(y) for y in SEASONS], ignore_index=True)
    d = d[d['Pitch Type'].isin(PTS) & d['Throws'].isin(['L', 'R'])]
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension', 'ArmAngle',
                         'SpinAxis', 'Spin Rate', 'KinEff', 'RelPosZ', 'RelPosX']).reset_index(drop=True)
    A = to_arrays(d)
    assert len(A['ivb']) == len(d)
    res = {n: fit_cv(A, f) for n, f in FORMS}
    ok = np.all([np.isfinite(res[n][0]) for n, _ in FORMS], axis=0)
    d = d[ok].reset_index(drop=True)
    s = np.where(d['Throws'] == 'R', 1.0, -1.0)
    d['ivb'] = d['xIndVrtBrk']
    d['hb'] = d['xHorzBrk'] * s
    for n, _ in FORMS:
        xi, xh = res[n][0][ok], res[n][1][ok]
        d[f'oi_{n}'] = d['ivb'] - xi
        d[f'oh_{n}'] = d['hb'] - xh
    d['swing'] = d['Description'].isin(SWINGS).astype(float)
    d['whiff'] = d['Description'].isin(WHIFFS).astype(float)
    agg = {'n': ('ivb', 'size'), 'ivb': ('ivb', 'mean'), 'hb': ('hb', 'mean'),
           'swings': ('swing', 'sum'), 'whiffs': ('whiff', 'sum'), 'rv': ('RunExp', 'mean')}
    for n, _ in FORMS:
        agg[f'oi_{n}'] = (f'oi_{n}', 'mean')
        agg[f'oh_{n}'] = (f'oh_{n}', 'mean')
    u = d.groupby(['Pitcher', 'Pitch Type', 'Throws', 'season']).agg(**agg).reset_index()
    u = u[(u.n >= 50) & (u.swings >= 20)].copy()
    u['whiff_rate'] = u.whiffs / u.swings
    print(f'{len(u):,} pitcher x type x hand x season units, 50+ pitches, 20+ swings\n')

    def loso(pt, feats, y):
        b = u[u['Pitch Type'] == pt]
        pred, act = [], []
        for yr in SEASONS:
            tr, te = b[b.season != yr], b[b.season == yr]
            if len(tr) < 60 or len(te) < 20:
                continue
            X = np.column_stack([np.ones(len(tr))] + [tr[f].values for f in feats])
            beta = np.linalg.lstsq(X, tr[y].values, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(te))] + [te[f].values for f in feats])
            pred.append(Xt @ beta)
            act.append(te[y].values)
        if not pred:
            return np.nan
        return np.corrcoef(np.concatenate(pred), np.concatenate(act))[0, 1]

    for y, label in (('whiff_rate', 'WHIFF RATE'), ('rv', 'RUN VALUE per pitch')):
        print(f'== {label}: out-of-season r, outcome ~ (ivb residual, hb residual), per type')
        cols = ['raw IVB/HB'] + [n for n, _ in FORMS]
        print(f'{"pt":<5}' + ''.join(f'{c:>22}' for c in cols))
        means = {c: [] for c in cols}
        for pt in PTS:
            line = f'{pt:<5}'
            r = loso(pt, ['ivb', 'hb'], y); means['raw IVB/HB'].append(r); line += f'{r:>22.3f}'
            for n, _ in FORMS:
                r = loso(pt, [f'oi_{n}', f'oh_{n}'], y); means[n].append(r); line += f'{r:>22.3f}'
            print(line)
        print(f'{"mean":<5}' + ''.join(f'{np.nanmean(means[c]):>22.3f}' for c in cols))
        print(f'\n   with raw IVB/HB ADDED to each residual model (what the residual adds beyond raw):')
        print(f'{"pt":<5}' + ''.join(f'{c:>22}' for c in cols[1:]))
        means = {c: [] for c in cols[1:]}
        for pt in PTS:
            line = f'{pt:<5}'
            for n, _ in FORMS:
                r = loso(pt, ['ivb', 'hb', f'oi_{n}', f'oh_{n}'], y); means[n].append(r); line += f'{r:>22.3f}'
            print(line)
        print(f'{"mean":<5}' + ''.join(f'{np.nanmean(means[c]):>22.3f}' for c in cols[1:]))
        print()


if __name__ == '__main__':
    main()
