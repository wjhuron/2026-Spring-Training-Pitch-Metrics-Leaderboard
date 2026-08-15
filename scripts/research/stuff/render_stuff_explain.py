#!/usr/bin/env python3
"""Explain each article arm's per-pitch-type Stuff+ with the model's own math.

Runs every pitch through the shipped Stuff+ v11 bundle with XGBoost
pred_contribs (exact SHAP-style attribution: bias + per-feature contributions
sum to the prediction), then converts contributions to STUFF+ POINTS on the
same per-type anchors the leaderboard uses:

    atom  = 100 + K * (-pred - mu_pt) / sd_pt
    pts_f = -K * contrib_f / sd_pt          (feature f, per pitch)
    base  = 100 + K * (-bias - mu_pt) / sd_pt

so base + sum(pts_f) = the pitch's atom, and the (pitcher, pitch type) page
shows the mean ledger: what a league-neutral pitch of that type starts at,
and how his velocity / shape / release move it. Full-model rows only (pitches
with arm angle, 96%+ everywhere) — identical to how the leaderboard scores
ROC. Bird/Cruz/Dion are sourced from the NEW tab (full MLB+AAA seasons).

Usage: python3 scripts/research/stuff/render_stuff_explain.py
Outputs ~/Downloads/ArticleVisuals/<Last>/StuffExplain_<LastFirst>.png
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stuff_plus'))
from stuff_plus.train_stuff import (build_df, design, K_SCALE, sf, FB_TYPES,
                             FC_ANCHOR_PITCHERS)  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
BUNDLE = os.path.join(ROOT, 'stuff_plus', 'stuff_models.pkl')
OUT_ROOT = os.path.expanduser('~/Downloads/ArticleVisuals')

for f in os.listdir(os.path.join(ROOT, 'assets', 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(ROOT, 'assets', 'fonts', f))
plt.rcParams['font.family'] = 'IBM Plex Sans'
TITLE_FONT = {'fontfamily': 'Bitter', 'fontweight': 700}

PAPER = (240 / 255, 232 / 255, 216 / 255)
CREAM = '#e8dfcb'
INK = (58 / 255, 48 / 255, 38 / 255)
BRICK = (176 / 255, 64 / 255, 47 / 255)
SLATE = (66 / 255, 100 / 255, 138 / 255)

ROC_ARMS = {'Perales, Luis': ['ROC'], 'Kent, Jackson': ['ROC'],
            'Sinclair, Jack': ['ROC'], 'Tolman, Erik': ['ROC']}
NEW_TAB_ARMS = {'Bird, Jake', 'Cruz, Yovanny', 'Dion, Will'}
NEW_TAB_WORKBOOK = '1BypxxlWgQAltETOLqccOYigeo8nXX-FIuVv6rhT4anA'

PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball',
               'KC': 'Knuckle-Curve', 'CH': 'Changeup', 'FS': 'Splitter'}
MIN_TYPE_N = 25
# Override hooks so any arm / any window can be rendered without editing the
# article lists below. EXTRA_ARMS maps "Last, First" -> pitch dicts (already
# in pipeline schema); SUBTITLE replaces the sample description in the header.
EXTRA_ARMS = {}
SUBTITLE = None

FRIENDLY = {
    'velocity': ('Velocity', 'mph', 1),
    'ivb': ('Ride (adj IVB)', 'in', 1),
    'hb': ('Arm-side run (adj HB)', 'in', 1),
    'velo_diff': ('Velo gap off his FB', 'mph', 1),
    'ivb_diff': ('IVB gap off his FB', 'in', 1),
    'hb_diff': ('HB gap off his FB', 'in', 1),
    'spin_rate': ('Spin rate', 'rpm', 0),
    'extension': ('Extension', 'ft', 1),
    'arm_angle': ('Arm angle', 'deg', 0),
    'vaa': ('Approach angle (nVAA)', 'deg', 1),
    'vaa_diff': ('nVAA gap off his FB', 'deg', 1),
    'rel_x': ('Horizontal release', 'ft', 1),
    'axis_dev': ('Axis deviation (SSW)', 'deg', 0),
    'axis_dev_abs': ('|Axis deviation|', 'deg', 0),
    'cross': ('Seam-shifted break (cross-axis)', 'in', 1),
    'cross_abs': ('|Cross-axis break|', 'in', 1),
    'platoon_same': ('Same-hand share', '', 2),
}


def reference_type(name, df):
    """Which pitch type is this arm's fastball ANCHOR — the one the *_diff
    features are measured against. Mirrors build_df: the most-thrown TRUE
    fastball (FF/SI), with FC only when he throws neither, plus the
    FC_ANCHOR_PITCHERS override.

    Do NOT infer this from velo_diff ~= 0. That worked until v12 masked
    velo_diff to None on FF/SI, after which the reference pitch's diff
    features stopped folding into the base and showed up as real bars on
    his own fastball ("HB gap off his FB" on the FB itself).
    """
    n = df.groupby('pitch_type').size().to_dict()
    fb = {pt: c for pt, c in n.items() if pt in FB_TYPES}
    if not fb:
        return None
    if name in FC_ANCHOR_PITCHERS and 'FC' in fb:
        return 'FC'
    true_fb = {pt: c for pt, c in fb.items() if pt in ('FF', 'SI')} or fb
    return max(true_fb, key=lambda pt: true_fb[pt])


def load_new_tab_pitches():
    import gspread
    ws = gspread.service_account().open_by_key(NEW_TAB_WORKBOOK).worksheet('NEW')
    out = defaultdict(list)
    for r in ws.get_all_records():
        if r.get('Pitcher') in NEW_TAB_ARMS:
            out[r['Pitcher']].append(r)
    return out


def main():
    print('Loading bundle + cache ...')
    with open(BUNDLE, 'rb') as f:
        B = pickle.load(f)
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    league = B['league']
    feats = B['features']
    booster = B['model'].get_booster()

    # League per-type physical context (MLB pitches, engineered space:
    # hand-signed hb, x-adjusted ivb) for the value annotations.
    mlb = [p for p in allp if p.get('_source') == 'MLB']
    print('Building league context frame ...')
    lg_df = build_df(mlb)
    lg_df = lg_df[lg_df['arm_angle'].notna()].reset_index(drop=True)
    lg_means = lg_df.groupby('pitch_type')[
        [f for f in feats if f != 'platoon_same']].mean()
    # League mean SHAP per (type, feature): bars display DEVIATION from the
    # average pitch of the type, so a league-average curveball reads ~0
    # everywhere instead of a giant global-baseline velocity bar.
    print('League attribution baseline (sampled) ...')
    rng = np.random.RandomState(11)
    parts = []
    for pt, sub in lg_df.groupby('pitch_type'):
        take = sub.sample(n=min(4000, len(sub)), random_state=rng)
        parts.append(take)
    lg_sample = pd.concat(parts).reset_index(drop=True)
    Xl = design(lg_sample).reindex(columns=feats, fill_value=0)
    cl = booster.predict(xgb.DMatrix(Xl), pred_contribs=True)
    lg_pts = {}
    for pt, subm in lg_sample.groupby('pitch_type'):
        ii = subm.index.values
        lg_pts[pt] = {feats[j]: float(np.mean(-cl[ii, j])) for j in range(len(feats))}

    pitches_by_arm = {}
    for name, teams in ROC_ARMS.items():
        pitches_by_arm[name] = [p for p in allp
                                if p.get('Pitcher') == name and p.get('PTeam') in teams]
    if NEW_TAB_ARMS:
        print('Fetching NEW tab ...')
        for name, plist in load_new_tab_pitches().items():
            pitches_by_arm[name] = plist
    for name, plist in EXTRA_ARMS.items():
        pitches_by_arm[name] = plist

    for name, plist in sorted(pitches_by_arm.items()):
        last, first = [s.strip() for s in name.split(',')]
        df = build_df(plist)
        df = df[df['arm_angle'].notna()].reset_index(drop=True)
        if not len(df):
            print(f'  {name}: no scoreable pitches, skipped')
            continue
        X = design(df).reindex(columns=feats, fill_value=0)
        contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        pred = contrib.sum(axis=1)           # margin = bias + sum(features)
        raw = -pred
        bias = contrib[:, -1]
        fc = contrib[:, :-1]                 # per-feature, order = feats

        # Per-type ledgers in Stuff+ points. For the pitcher's REFERENCE
        # fastball the *_diff features are structurally zero — their SHAP is
        # pooled-model accounting for "this is the reference pitch", not a
        # trait of the pitch — so those contributions fold into the base.
        DIFF_FEATS = {'velo_diff', 'ivb_diff', 'hb_diff', 'vaa_diff'}
        ref_pt = reference_type(name, df)
        panels = []
        for pt, sub in df.groupby('pitch_type'):
            if len(sub) < MIN_TYPE_N or pt not in league:
                continue
            sc = league[pt]
            if not sc.get('sd') or sc['sd'] <= 0:
                continue
            idx = sub.index.values
            k_sd = K_SCALE / sc['sd']
            atoms = 100 + K_SCALE * (raw[idx] - sc['mu']) / sc['sd']
            base = float(np.mean(100 + K_SCALE * (-bias[idx] - sc['mu']) / sc['sd']))
            lgc = lg_pts.get(pt, {})
            pts = {feats[j]: float((np.mean(-fc[idx, j]) - lgc.get(feats[j], 0.0)) * k_sd)
                   for j in range(len(feats))}
            vals = {f: float(sub[f].mean()) for f in feats if f in sub.columns}
            is_ref = (pt == ref_pt)
            if is_ref:
                for f in list(pts):
                    if f in DIFF_FEATS:
                        pts.pop(f)
            grade = float(np.mean(atoms))
            base = grade - sum(pts.values())
            panels.append({'pt': pt, 'n': len(sub), 'grade': grade,
                           'base': base, 'pts': pts, 'vals': vals,
                           'is_ref': is_ref})
        panels.sort(key=lambda d: -d['n'])
        if not panels:
            continue

        ncols = 3
        nrows = math.ceil(len(panels) / ncols)
        fig = plt.figure(figsize=(17.0, 2.6 + 4.6 * nrows), dpi=200)
        fig.patch.set_facecolor(CREAM)
        gs = GridSpec(nrows + 1, ncols, figure=fig,
                      height_ratios=[0.5] + [1.3] * nrows,
                      hspace=0.42, wspace=0.55,
                      left=0.14, right=0.965, top=0.955, bottom=0.045)
        hd = fig.add_subplot(gs[0, :])
        hd.axis('off')
        hd.text(0, 0.93, f'{first} {last}: Why the Model Grades His Stuff '
                'the Way It Does', fontsize=22, color=INK, va='top', **TITLE_FONT)
        hd.text(0, 0.46,
                'Bars: how many Stuff+ points each trait adds or subtracts VS THE LEAGUE-AVERAGE '
                'PITCH OF THAT TYPE, from the v12 model’s own per-pitch attributions (SHAP).\n'
                'Each panel starts at its BASE (what the league-average version of the pitch '
                'grades, plus his share of interactions) and base + bars = his printed grade.\n'
                'Values show his average trait vs the MLB average for the type. Red helps, blue hurts.',
                fontsize=10.5, color=INK, va='top', linespacing=1.6)
        if SUBTITLE:
            hd.text(0, 0.06, SUBTITLE, fontsize=10.5, color=BRICK,
                    va='top', fontweight=700)

        for i, d in enumerate(panels):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[1 + r, c])
            ax.set_facecolor(CREAM)
            order = sorted(d['pts'].items(), key=lambda kv: -abs(kv[1]))[:8]
            other = sum(v for f, v in d['pts'].items()
                        if f not in {f0 for f0, _ in order})
            rows = order + [('__other', other)]
            rows = rows[::-1]
            ys = range(len(rows))
            for y, (f, v) in zip(ys, rows):
                color = BRICK if v > 0 else SLATE
                ax.barh(y, v, height=0.62, color=color, alpha=0.9,
                        edgecolor=(*INK, 0.3), linewidth=0.6)
                ax.text(v + (0.12 if v >= 0 else -0.12), y, f'{v:.1f}',
                        va='center', ha='left' if v >= 0 else 'right',
                        fontsize=8, fontweight=700, color=INK)
            labels = []
            for f, v in rows:
                if f == '__other':
                    labels.append('everything else')
                    continue
                nm, unit, dec = FRIENDLY.get(f, (f, '', 1))
                val = d['vals'].get(f)
                lgv = (lg_means.loc[d['pt'], f]
                       if (d['pt'] in lg_means.index and f in lg_means.columns)
                       else None)
                if val is not None and lgv is not None and not math.isnan(lgv):
                    labels.append(f'{nm}  {val:.{dec}f} (lg {lgv:.{dec}f}{(" " + unit) if unit else ""})')
                else:
                    labels.append(nm)
            ax.set_yticks(list(ys))
            ax.set_yticklabels(labels, fontsize=7.6, color=INK)
            ax.tick_params(axis='y', length=0, pad=4)
            ax.axvline(0, color=(*INK, 0.5), linewidth=1.0)
            lim = max(2.0, max(abs(v) for _, v in rows) * 1.35)
            ax.set_xlim(-lim, lim)
            ax.set_xticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            tot = d['base'] + sum(d['pts'].values())
            ax.set_title(f'{PITCH_NAMES.get(d["pt"], d["pt"])} · Stuff+ '
                         f'{d["grade"]:.0f} · {d["n"]} pitches',
                         fontsize=12, color=INK, loc='left', pad=8, **TITLE_FONT)
            ax.text(0, -0.14, f'ledger: base {d["base"]:.0f} + traits '
                    f'{sum(d["pts"].values()):.1f} = {tot:.0f}',
                    transform=ax.transAxes, fontsize=7.5, color=(*INK, 0.8))

        out = os.path.join(OUT_ROOT, last, f'StuffExplain_{last}{first}.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        grades = ', '.join(f'{d["pt"]} {d["grade"]:.0f}' for d in panels)
        print(f'{name}: {grades} -> {out}')


if __name__ == '__main__':
    main()
