#!/usr/bin/env python3
"""zwhiff_incremental.py — does Z-Whiff% earn a pitcher-card bubble?

Question (2026-07-30): the card already shows Whiff%, Chase% (SWING & MISS)
and K% (RESULT). Z-Whiff% was pruned 2026-07-20 at r=.83 with Whiff%. Wally
would reconsider IF it carries incremental signal. Test: partial correlation
of season-N Z-Whiff% with season-N+1 outcomes (PA-level xwOBA against, K%)
after residualizing BOTH sides on what the card already shows.

Control sets:
  WC   = Whiff% + Chase%          (the SWING & MISS bubbles)
  WCK  = Whiff% + Chase% + K%     (adds the RESULT-section K%)

Pairs: 2021->22, 22->23, 23->24, 24->25 (independent replicates; never pooled).
Pre-registered bar: recommend the bubble only if the WCK partial is
sign-consistent (helpful direction) in >=3/4 pairs with mean |r| >= 0.08.

Reuses leaderboard_metric_battery.load_season() so every flag (zone geometry,
bunt exclusion, wOBA weights) matches the 2026-07-20 battery exactly.

Usage: python3 scripts/zwhiff_incremental.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from leaderboard_metric_battery import load_season  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PITCHES = 1000          # both seasons of a pair; ~40+ IP
MIN_PITCHES_SP = 2000       # starter-weight robustness cut


def pitcher_table(year):
    df = load_season(year)
    g = df.groupby('pitcher')
    t = pd.DataFrame({
        'pitches': g.size(),
        'whiff': g['whiff'].sum() / g['swing'].sum(),
        'chase': g['chase_sw'].sum() / g['ooz'].sum(),
        'izwhiff': g.apply(lambda x: (x['iz'] & x['whiff']).sum()
                           / max((x['iz'] & x['swing']).sum(), 1)),
        'kpct': g['k'].sum() / g['pa_end'].sum(),
        'paxw': g['paxw_num'].sum() / g['paxw_den'].sum(),
    })
    return t


def partial_r(x, y, controls):
    """Correlate the residuals of x and y after OLS on the control matrix."""
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    x, y, c = x[m], y[m], controls[m]
    c = np.column_stack([np.ones(len(c)), c])
    bx, *_ = np.linalg.lstsq(c, x, rcond=None)
    by, *_ = np.linalg.lstsq(c, y, rcond=None)
    rx, ry = x - c @ bx, y - c @ by
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def main():
    tabs = {y: pitcher_table(y) for y in SEASONS}
    print('season tables built:', {y: len(t) for y, t in tabs.items()})

    for gate, name in ((MIN_PITCHES, 'all >=1000 pitches'),
                       (MIN_PITCHES_SP, 'starters >=2000 pitches')):
        print(f'\n=== gate: {name} ===')
        print('pair    n    r(zW,W)  | paxw+1: raw   |WC     |WCK   '
              '| kpct+1: raw   |WC     |WCK   | sym: W|zC->paxw+1')
        rows = []
        for y in SEASONS[:-1]:
            a, b = tabs[y], tabs[y + 1]
            j = a.join(b, lsuffix='_n', rsuffix='_p', how='inner')
            j = j[(j['pitches_n'] >= gate) & (j['pitches_p'] >= gate)]
            zw = j['izwhiff_n'].values
            w, ch, k = j['whiff_n'].values, j['chase_n'].values, j['kpct_n'].values
            red = float(pd.Series(zw).corr(pd.Series(w)))
            out = {'pair': f'{y}->{y+1}', 'n': len(j), 'redund': red}
            for tgt in ('paxw', 'kpct'):
                yv = j[f'{tgt}_p'].values
                raw = float(pd.Series(zw).corr(pd.Series(yv)))
                pwc, _ = partial_r(zw, yv, np.column_stack([w, ch]))
                pwck, _ = partial_r(zw, yv, np.column_stack([w, ch, k]))
                out[f'{tgt}_raw'], out[f'{tgt}_wc'], out[f'{tgt}_wck'] = raw, pwc, pwck
            # symmetry check: does WHIFF carry unique signal given zWhiff+chase?
            psym, _ = partial_r(w, j['paxw_p'].values, np.column_stack([zw, ch]))
            out['sym_w'] = psym
            rows.append(out)
            print(f"{out['pair']}  {out['n']:4d}  {red:6.3f}   |"
                  f"       {out['paxw_raw']:6.3f} {out['paxw_wc']:6.3f} {out['paxw_wck']:6.3f} |"
                  f"       {out['kpct_raw']:6.3f} {out['kpct_wc']:6.3f} {out['kpct_wck']:6.3f} |"
                  f"   {psym:6.3f}")
        r = pd.DataFrame(rows)
        print('mean          {:6.3f}   |       {:6.3f} {:6.3f} {:6.3f} |'
              '       {:6.3f} {:6.3f} {:6.3f} |   {:6.3f}'.format(
                  r['redund'].mean(), r['paxw_raw'].mean(), r['paxw_wc'].mean(),
                  r['paxw_wck'].mean(), r['kpct_raw'].mean(), r['kpct_wc'].mean(),
                  r['kpct_wck'].mean(), r['sym_w'].mean()))


if __name__ == '__main__':
    main()
