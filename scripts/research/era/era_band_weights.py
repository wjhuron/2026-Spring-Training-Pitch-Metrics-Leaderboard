"""era_band_weights.py — does a sample-size-dependent weight schedule
improve hpERA?

Public prior (FanGraphs Stuff+ primer): Stuff+ stabilizes at 300-400
pitches, before any outcome stat, so at low IP the process channels
(Stuff+/Loc+) should carry more weight and the outcome channels less.
hpERA already encodes part of this through per-channel shrinkage (the
outcome channels pull to league at measured n0 while Stuff+ shrinks at
n0=15), so the question is whether an EXPLICIT band schedule adds
anything beyond the shrinkage.

Test: production 8-channel fit (era_weights_final harness). For each
LOSO fold, fit (a) one global OLS on the train seasons, (b) one OLS per
innings band, fit only on train units in that band. Score held-out units
with their band's weights (fallback to global when a band has under 50
train units). Bands are on the FEATURE-scope outs (h1 for ROS, full-season
prior year for NEXT).

Adoption bar: banded beats global on held-out pooled r in most folds, and
the per-band comparison says WHERE any gain lives.

Usage: PYTHONHASHSEED=0 python3 scripts/research/era/era_band_weights.py
Output: console + data/_era_band_weights.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import era_weights_final as wf
from era_estimator_screen import pearson, targets_for
from era_combo_preview import ols

FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']
# outs bands on the feature scope. ROS features are h1 (45+ outs = 15 IP
# floor from the harness); NEXT features are the full prior season.
BANDS = {
    'ros': [(45, 135), (135, 270), (270, 10 ** 9)],     # 15-45, 45-90, 90+ IP
    'next': [(90, 270), (270, 450), (450, 10 ** 9)],    # 30-90, 90-150, 150+ IP
}


def build_reps(test, gate):
    """Like wf.build_reps but each unit carries its feature-scope outs."""
    reps = []
    if test == 'next':
        for season in wf.SEASONS[:-1]:
            fr = wf.shrunk_features(season, 'full')
            tc = targets_for(season, 'full')
            tn = targets_for(season + 1, 'full')
            units = [(fr[pid], tn[pid]['era'], tc[pid]['outs'])
                     for pid in fr
                     if pid in tn and pid in tc
                     and tc[pid]['outs'] >= gate * 3
                     and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
    else:
        for season in wf.SEASONS:
            fr = wf.shrunk_features(season, 'h1')
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = max(gate * 3 // 2, 45)
            units = [(fr[pid], t2[pid]['era'], t1[pid]['outs'])
                     for pid in fr
                     if pid in t2 and pid in t1
                     and t1[pid]['outs'] >= hg
                     and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
    return reps


def band_of(outs, bands):
    for i, (lo, hi) in enumerate(bands):
        if lo <= outs < hi:
            return i
    return len(bands) - 1


def predict(beta, x):
    return beta[0] + sum(b * x[f] for b, f in zip(beta[1:], FEATS))


def main():
    out = {}
    for test in ('ros', 'next'):
        bands = BANDS[test]
        # the ROS harness floor is 45 outs, so run the wide gate (30) —
        # a 60 IP gate would empty the low band and test nothing.
        gate = 30
        reps = build_reps(test, gate)
        print(f'\n===== {test.upper()} gate {gate} '
              f'bands {[(lo // 3, "+" if hi > 10**8 else hi // 3) for lo, hi in bands]} IP =====')
        per_global, per_band = [], []
        band_detail = {i: {'g': [], 'b': []} for i in range(len(bands))}
        fallbacks = 0
        for i, (label, test_units) in enumerate(reps):
            train = [u for j, (_, us) in enumerate(reps) if j != i
                     for u in us]
            g_beta = ols([(x, y) for x, y, _ in train], FEATS)
            if g_beta is None:
                continue
            b_betas = []
            for bi in range(len(bands)):
                sub = [(x, y) for x, y, o in train
                       if band_of(o, bands) == bi]
                bb = ols(sub, FEATS)
                if bb is None:
                    bb = g_beta
                    fallbacks += 1
                b_betas.append(bb)
            gp, bp, ys = [], [], []
            byband = {bi: ([], [], []) for bi in range(len(bands))}
            for x, y, o in test_units:
                if not all(f in x for f in FEATS):
                    continue
                bi = band_of(o, bands)
                pg = predict(g_beta, x)
                pb = predict(b_betas[bi], x)
                gp.append(pg)
                bp.append(pb)
                ys.append(y)
                byband[bi][0].append(pg)
                byband[bi][1].append(pb)
                byband[bi][2].append(y)
            rg, rb = pearson(gp, ys), pearson(bp, ys)
            per_global.append((label, rg))
            per_band.append((label, rb))
            print(f'  {label:<12} global {rg:+.4f}  banded {rb:+.4f}  '
                  f'delta {rb - rg:+.4f}')
            for bi, (g2, b2, y2) in byband.items():
                r1, r2 = pearson(g2, y2), pearson(b2, y2)
                if r1 is not None and r2 is not None:
                    band_detail[bi]['g'].append(r1)
                    band_detail[bi]['b'].append(r2)
        mg = sum(r for _, r in per_global) / len(per_global)
        mb = sum(r for _, r in per_band) / len(per_band)
        wins = sum(1 for (_, a), (_, b) in zip(per_band, per_global)
                   if a > b)
        print(f'  POOLED: global {mg:+.4f}  banded {mb:+.4f}  '
              f'banded wins {wins}/{len(per_band)} folds '
              f'({fallbacks} thin-band fallbacks)')
        for bi in band_detail:
            g2, b2 = band_detail[bi]['g'], band_detail[bi]['b']
            if g2:
                print(f'  band {bi} ({bands[bi][0] // 3}+ IP): '
                      f'global {sum(g2) / len(g2):+.4f}  '
                      f'banded {sum(b2) / len(b2):+.4f}  '
                      f'({len(g2)} folds)')
        out[test] = {'global': dict(per_global), 'banded': dict(per_band),
                     'wins': wins}
    tmp = os.path.join(ROOT, 'data', '_era_band_weights.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_era_band_weights.json'))
    print('\nwrote data/_era_band_weights.json')


if __name__ == '__main__':
    main()
