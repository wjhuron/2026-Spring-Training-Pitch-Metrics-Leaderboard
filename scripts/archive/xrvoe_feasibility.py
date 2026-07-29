"""xrvoe_feasibility.py — measure whether xRVOE is worth building.

xRVOE = per-pitch actual luck-neutral xRV minus the model-expected xRV from
a stacked combination of the Stuff+ per-pitch prediction (OOF fold models)
and the Loc+ per-pitch ExpRV (count-aware location expectation), aggregated
per (pitcher, team, pitch_type).

Measures, on 2026 MLB (EP excluded):
  0. Stacking: per-pitch R^2 of stuff-only / loc-only / combined vs actual.
  1. Reliability: odd/even-date split-half r of unit xRVOE at several
     min-pitch floors, and the implied n0.
  2. Orthogonality: unit-level corr of xRVOE vs Stuff+ (stuff_pred mean),
     Loc ExpRV mean, velocity, whiff% — residual should be near-orthogonal.
  3. Persistence: chrono 1st-half unit xRVOE vs 2nd-half unit xRVOE.
  4. Incrementality: does 1st-half xRVOE improve prediction of 2nd-half
     actual xRV beyond 1st-half stuff+loc expectations?

Usage: python3 scripts/xrvoe_feasibility.py
"""
import os, sys, json, pickle, math
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── load pitches (MLB, EP appearances excluded — mirrors Stuff+ training) ──
D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
_ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
mlb = [p for p in D if p.get('_source', 'MLB') == 'MLB'
       and (p.get('Pitcher'), p.get('PTeam')) not in _ep]
print(f"MLB non-EP pitches: {len(mlb)}", flush=True)

# ── Stuff per-pitch OOF predictions (exec-patch build_df to carry PitchID) ──
src = open(os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')).read()
NEEDLE = "            'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'spin_rate': spin,"
assert NEEDLE in src
src = src.replace(NEEDLE, NEEDLE + "\n            'pid': p.get('PitchID'),")
T = {'__name__': '_stuff_mod', '__file__': os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')}
exec(compile(src.split("def main()")[0], 'train_stuff_v11.py', 'exec'), T)

df = T['build_df'](mlb)
df = df[df['target_xrv'].notna()].reset_index(drop=True)
print(f"stuff rows with target: {len(df)}", flush=True)

B = pickle.load(open(os.path.join(ROOT, 'stuff_plus_v11', 'stuff_models_v11.pkl'), 'rb'))
X = T['design'](df).reindex(columns=B['features'], fill_value=0)
fold_of = {p: k for k, ps in enumerate(B['fold_pitchers']) for p in ps}
pf = np.array([fold_of.get(p, 0) for p in df['pitcher'].values])
pred = np.full(len(df), np.nan)
for k, m in enumerate(B['fold_models']):
    mask = pf == k
    if mask.any():
        pred[mask] = m.predict(X[mask])
df['stuff_pred'] = pred          # model predicts target_xrv directly (hitter-persp.)
print("stuff preds done", flush=True)

# ── Loc per-pitch ExpRV ──
import pipeline_locplus as L
md = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
guts = md.get('gutsConstants') or {}
baseline = [p for p in mlb if L.is_eligible_baseline(p)]
S = L.build_surfaces(baseline, guts.get('lgWOBA'), guts.get('wOBAScale'))
loc_exprv = {}
for p in baseline:
    v = L.score_pitch(p, S)
    if v is not None and p.get('PitchID'):
        loc_exprv[p['PitchID']] = v
print(f"loc ExpRV scored: {len(loc_exprv)}", flush=True)

# ── join ──
df['loc_exprv'] = df['pid'].map(loc_exprv)
J = df[df['loc_exprv'].notna() & df['stuff_pred'].notna()].reset_index(drop=True)
y = J['target_xrv'].values
s = J['stuff_pred'].values
l = J['loc_exprv'].values
print(f"joined pitches: {len(J)}", flush=True)

def r2(yhat):
    ss = np.sum((y - yhat) ** 2)
    return 1 - ss / np.sum((y - y.mean()) ** 2)

# sign check + stacking OLS
print(f"corr(y, stuff_pred) {np.corrcoef(y, s)[0,1]:+.4f}   corr(y, loc_exprv) {np.corrcoef(y, l)[0,1]:+.4f}")
A = np.column_stack([np.ones(len(J)), s, l])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
yhat = A @ beta
b_s, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(J)), s]), y, rcond=None)
b_l, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(J)), l]), y, rcond=None)
print(f"stacking betas: intercept {beta[0]:+.5f}, stuff {beta[1]:+.3f}, loc {beta[2]:+.3f}")
print(f"per-pitch R^2: stuff-only {r2(np.column_stack([np.ones(len(J)), s]) @ b_s)*100:.3f}%  "
      f"loc-only {r2(np.column_stack([np.ones(len(J)), l]) @ b_l)*100:.3f}%  "
      f"combined {r2(yhat)*100:.3f}%")

J['resid'] = y - yhat            # per-pitch xRVOE (hitter-perspective; negative = pitcher beat expectation)
J['unit'] = list(zip(J['pitcher'], J['team'], J['pitch_type']))

# ── 1. split-half reliability of unit xRVOE ──
date_order = {d: i for i, d in enumerate(sorted(J['date'].dropna().unique()))}
J['half'] = J['date'].map(date_order).fillna(0).astype(int) % 2

def unit_means(sub, min_n):
    g = sub.groupby('unit')['resid'].agg(['mean', 'size'])
    return g[g['size'] >= min_n]['mean']

print("\nSPLIT-HALF RELIABILITY of unit xRVOE (odd/even dates):")
print(f"{'n/half':>7s} {'r':>7s} {'units':>6s} {'implied n0':>10s}")
for N in (25, 50, 75, 100, 150, 200):
    a = unit_means(J[J.half == 0], N)
    b = unit_means(J[J.half == 1], N)
    common = a.index.intersection(b.index)
    if len(common) < 30:
        continue
    r = np.corrcoef(a[common], b[common])[0, 1]
    n0 = N * (1 - r) / r if r > 0 else float('inf')
    print(f"{N:7d} {r:7.3f} {len(common):6d} {n0:10.0f}")

# for context: same-protocol reliability of the RAW xRV (what xRV/100 shows)
print("\n(for context) split-half r of raw unit xRV at same floors:")
for N in (50, 100, 200):
    Ja = J[J.half == 0].groupby('unit')['target_xrv'].agg(['mean', 'size'])
    Jb = J[J.half == 1].groupby('unit')['target_xrv'].agg(['mean', 'size'])
    a = Ja[Ja['size'] >= N]['mean']; b = Jb[Jb['size'] >= N]['mean']
    common = a.index.intersection(b.index)
    if len(common) < 30: continue
    r = np.corrcoef(a[common], b[common])[0, 1]
    print(f"  n/half {N}: r {r:.3f} (units {len(common)})")

# ── 2. orthogonality at unit level (n >= 100 full season) ──
g = J.groupby('unit').agg(resid=('resid', 'mean'), n=('resid', 'size'),
                          stuff=('stuff_pred', 'mean'), loc=('loc_exprv', 'mean'),
                          velo=('velocity', 'mean'))
gq = g[g['n'] >= 100]
whiff = J.groupby('unit').apply(lambda x: np.nan, include_groups=False)  # placeholder
print(f"\nORTHOGONALITY (units n>=100, {len(gq)}):")
for k in ('stuff', 'loc', 'velo'):
    print(f"  corr(xRVOE, {k}): {np.corrcoef(gq['resid'], gq[k])[0,1]:+.3f}")

# ── 3+4. persistence + incrementality (chrono halves) ──
dates = sorted(J['date'].dropna().unique())
mid = dates[len(dates) // 2]
E = J[J['date'] < mid]; Lh = J[J['date'] >= mid]
ge = E.groupby('unit').agg(resid=('resid','mean'), n=('resid','size'),
                           exp=('stuff_pred','mean'))
ge2 = E.groupby('unit').apply(lambda x: (x['stuff_pred'] * beta[1] + x['loc_exprv'] * beta[2] + beta[0]).mean(), include_groups=False)
gl = Lh.groupby('unit').agg(act=('target_xrv','mean'), resid=('resid','mean'), n=('target_xrv','size'))
common = ge[(ge['n'] >= 100)].index.intersection(gl[gl['n'] >= 100].index)
print(f"\nPERSISTENCE / INCREMENTALITY (units with >=100 pitches each half: {len(common)})")
if len(common) >= 30:
    r_persist = np.corrcoef(ge.loc[common, 'resid'], gl.loc[common, 'resid'])[0, 1]
    print(f"  1st-half xRVOE -> 2nd-half xRVOE: r = {r_persist:+.3f}")
    exp1 = ge2[common].values
    res1 = ge.loc[common, 'resid'].values
    act2 = gl.loc[common, 'act'].values
    r_exp = np.corrcoef(exp1, act2)[0, 1]
    Astack = np.column_stack([np.ones(len(common)), exp1, res1])
    bb, *_ = np.linalg.lstsq(Astack, act2, rcond=None)
    pred2 = Astack @ bb
    r_both = np.corrcoef(pred2, act2)[0, 1]
    r_res_partial = np.corrcoef(res1, act2 - np.polyval(np.polyfit(exp1, act2, 1), exp1))[0, 1]
    print(f"  1st-half expectation -> 2nd-half actual xRV: r = {r_exp:+.3f}")
    print(f"  + 1st-half xRVOE (stacked):                  r = {r_both:+.3f}")
    print(f"  partial corr of xRVOE with 2nd-half actual (expectation removed): {r_res_partial:+.3f}")
EOF_MARKER = True
