#!/usr/bin/env python3
"""aaa_translation.py — how much of a ROC hitter's contact quality survives MLB?

The existing asset (data/aaa_outcome_offsets.json) is a LEAGUE-MEAN offset:
AAA league rate minus MLB league rate, per pitch type. That construction is
confounded for translating a player. A league mean differs across levels for
two reasons at once — the hitters are worse AND the pitchers are worse — and
those pull the mean in OPPOSITE directions, so the difference is a blend of a
thing you want (weaker pitching) and a thing you must not apply to a hitter
(weaker hitters). It can even carry the wrong sign.

This script measures the piece that IS identifiable, in Wally's own currency:

  1. OPPONENT QUALITY, measured. Stuff+ and Loc+ are already computed on an
     MLB-anchored scale for every AAA-tab pitch, so "how much weaker is the
     pitching a ROC hitter saw" is a measurement, not an assumption. 43% of
     those pitches were thrown by arms who also pitched in MLB in 2026.

  2. SENSITIVITY, estimated. Within MLB, how much does xwOBA on a batted ball
     move per point of the pitch's Stuff+ / Loc+? Estimated WITHIN HITTER
     (each hitter's own mean removed) so it is not contaminated by good
     hitters facing different pitchers, on ~100k batted balls.

  3. The adjustment is (1) x (2), per hitter, on his own faced-quality mix.

What this does NOT do: it prices only the part of the level gap that shows up
in pitch quality. Sequencing, defense, pitcher variety, familiarity, park and
ball are not in it. The script therefore also computes the naive league-mean
offset and the (tiny) same-player bridge, and reports all three side by side,
so the gap between them is visible rather than hidden inside one number.

Usage:  python3 scripts/aaa_translation.py [--out data/aaa_contact_offsets.json]
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pipeline_utils import (                                    # noqa: E402
    safe_float, MLB_TEAMS, BUNT_BB_TYPES, SWING_DESCRIPTIONS,
)

DATA_DIR = os.path.join(REPO, 'data')
MIN_HITTER_BIP = 50        # hitters below this add noise, not identification
SEED = 20260804


def load():
    with open(os.path.join(DATA_DIR, 'all_pitches_rs_cache.pkl'), 'rb') as f:
        return pickle.load(f)


def grade(p, key):
    v = p.get(key)
    if v in (None, '') or str(v).strip() == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_bip(p):
    bb = p.get('BBType')
    return bool(bb) and bb not in BUNT_BB_TYPES


# ── 1. Sensitivity of contact quality to pitch quality, within hitter ────

def fit_sensitivity(pitches, verbose=True):
    """Within-hitter OLS of xwOBA-on-contact against the pitch's Stuff+/Loc+.

    Demeaning by hitter is what makes this a causal-ish slope rather than a
    selection artifact: without it, the slope would partly reflect that better
    hitters get pitched differently."""
    by_hitter = defaultdict(list)
    for p in pitches:
        if p.get('_source', 'MLB') != 'MLB' or not is_bip(p):
            continue
        xw = safe_float(p.get('xwOBA'))
        s, l = grade(p, 'Stuff+'), grade(p, 'Loc+')
        if xw is None or s is None or l is None:
            continue
        by_hitter[p.get('Batter')].append((xw, s, l))
    by_hitter = {k: v for k, v in by_hitter.items() if len(v) >= MIN_HITTER_BIP}

    def solve(keys):
        Y, X = [], []
        for k in keys:
            rows = np.array(by_hitter[k], dtype=float)
            rows -= rows.mean(axis=0)          # within-hitter demeaning
            Y.append(rows[:, 0]); X.append(rows[:, 1:])
        Y = np.concatenate(Y); X = np.concatenate(X)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        dof = max(1, len(Y) - X.shape[1] - len(keys))
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / dof)
        return beta, se, len(Y)

    keys = sorted(by_hitter)
    beta, se, n = solve(keys)
    if verbose:
        print(f"  Sensitivity fit: {len(keys)} MLB hitters, {n} batted balls")
        print(f"    d(xwOBAcon) per Stuff+ point = {beta[0]:+.5f}  (SE {se[0]:.5f})")
        print(f"    d(xwOBAcon) per Loc+   point = {beta[1]:+.5f}  (SE {se[1]:.5f})")

    # Split-half by hitter: a slope that only exists in the half it was fit on
    # is noise. This is the weak form of out-of-sample — same season, same
    # league — so treat agreement as necessary, not sufficient.
    rng = random.Random(SEED)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    b1, _, n1 = solve(shuffled[:half])
    b2, _, n2 = solve(shuffled[half:])
    if verbose:
        print(f"    split-half A (n={n1}): stuff {b1[0]:+.5f}, loc {b1[1]:+.5f}")
        print(f"    split-half B (n={n2}): stuff {b2[0]:+.5f}, loc {b2[1]:+.5f}")
    return beta, se, {'nHitters': len(keys), 'nBip': n,
                      'splitHalf': [list(np.round(b1, 6)), list(np.round(b2, 6))]}


# ── 2. Measured opponent quality ─────────────────────────────────────────

def faced_quality(pitches):
    """Pitch-weighted mean Stuff+/Loc+ faced, by hitter, plus the MLB baseline.

    Weighted over BATTED BALLS, not all pitches: the adjustment multiplies a
    per-BIP sensitivity, so the exposure that matters is the quality of the
    pitches he actually put in play."""
    mlb_s, mlb_l = [], []
    roc = defaultdict(lambda: {'s': [], 'l': []})
    for p in pitches:
        if not is_bip(p):
            continue
        s, l = grade(p, 'Stuff+'), grade(p, 'Loc+')
        if s is None or l is None:
            continue
        src = p.get('_source', 'MLB')
        if src == 'MLB' and p.get('BTeam') in MLB_TEAMS:
            mlb_s.append(s); mlb_l.append(l)
        elif src == 'AAA' and p.get('BTeam') == 'ROC':
            r = roc[p.get('Batter')]
            r['s'].append(s); r['l'].append(l)
    base = (float(np.mean(mlb_s)), float(np.mean(mlb_l)), len(mlb_s))
    out = {k: (float(np.mean(v['s'])), float(np.mean(v['l'])), len(v['s']))
           for k, v in roc.items() if len(v['s']) >= 20}
    return base, out


# ── 3. The two comparison estimates ──────────────────────────────────────

def league_mean_offset(pitches):
    """The naive construction: AAA league xwOBAcon minus MLB league xwOBAcon.
    Reported for contrast, NOT recommended for translating a player."""
    def m(f):
        v = [safe_float(p.get('xwOBA')) for p in pitches
             if is_bip(p) and f(p) and safe_float(p.get('xwOBA')) is not None]
        return float(np.mean(v)), len(v)
    mlb = m(lambda p: p.get('_source', 'MLB') == 'MLB' and p.get('BTeam') in MLB_TEAMS)
    aaa = m(lambda p: p.get('_source') == 'AAA' and p.get('BTeam') == 'ROC')
    return aaa[0] - mlb[0], aaa, mlb


def same_player_bridge(pitches):
    """Hitters with batted balls at BOTH levels in 2026. The cleanest design
    and, in this data, far too small to settle anything — reported so the
    sample size is visible instead of assumed."""
    aaa, mlb = defaultdict(list), defaultdict(list)
    for p in pitches:
        if not is_bip(p):
            continue
        xw = safe_float(p.get('xwOBA'))
        if xw is None:
            continue
        if p.get('_source') == 'AAA' and p.get('BTeam') == 'ROC':
            aaa[p.get('Batter')].append(xw)
        elif p.get('_source', 'MLB') == 'MLB' and p.get('BTeam') in MLB_TEAMS:
            mlb[p.get('Batter')].append(xw)
    rows = []
    for n in sorted(set(aaa) & set(mlb)):
        if len(aaa[n]) >= 30 and len(mlb[n]) >= 30:
            rows.append((n, float(np.mean(aaa[n])), len(aaa[n]),
                         float(np.mean(mlb[n])), len(mlb[n])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DATA_DIR, 'aaa_contact_offsets.json'))
    args = ap.parse_args()

    print("=== Loading pitch data ===")
    pitches = load()
    print(f"  {len(pitches)} pitches")

    print("\n=== 1. Sensitivity of xwOBAcon to pitch quality (within MLB hitter) ===")
    beta, se, meta = fit_sensitivity(pitches)

    print("\n=== 2. Measured opponent quality on batted balls ===")
    (mlb_s, mlb_l, n_mlb), roc_q = faced_quality(pitches)
    print(f"  MLB baseline: Stuff+ {mlb_s:.2f}, Loc+ {mlb_l:.2f} (n={n_mlb} BIP)")

    print("\n=== 3. Per-hitter opponent-quality adjustment ===")
    targets = ['Ortiz, Abimelec', 'Morales, Yohandy', 'King, Seaver',
               'Pinckney, Andrew', 'Glasser, Phillip']
    rows = {}
    for name in sorted(roc_q, key=lambda k: -roc_q[k][2]):
        s, l, n = roc_q[name]
        adj = beta[0] * (mlb_s - s) + beta[1] * (mlb_l - l)
        rows[name] = {'facedStuff': round(s, 2), 'facedLoc': round(l, 2),
                      'nBip': n, 'adjXwobacon': round(adj, 4)}
        if name in targets:
            print(f"  {name:20s} faced Stuff+ {s:6.2f} Loc+ {l:6.2f} "
                  f"({n:4d} BIP) -> xwOBAcon adjustment {adj:+.4f}")

    print("\n=== Comparison: what the other two constructions say ===")
    lm, aaa, mlb = league_mean_offset(pitches)
    print(f"  Naive league mean: AAA {aaa[0]:.4f} (n={aaa[1]}) vs "
          f"MLB {mlb[0]:.4f} (n={mlb[1]}) -> offset {lm:+.4f}")
    print("    (confounded: weaker hitters and weaker pitchers push this in "
          "opposite directions)")
    bridge = same_player_bridge(pitches)
    print(f"  Same-player bridge (30+ BIP at both levels): n = {len(bridge)}")
    for n, a, na, m, nm in bridge:
        print(f"    {n:20s} AAA {a:.3f} ({na}) -> MLB {m:.3f} ({nm})  "
              f"delta {m - a:+.3f}")
    if bridge:
        d = [m - a for _, a, _, m, _ in bridge]
        print(f"    unweighted mean delta {np.mean(d):+.4f}, "
              f"SD {np.std(d, ddof=1):.4f}, SE {np.std(d, ddof=1) / np.sqrt(len(d)):.4f}")

    blob = {
        'season': 2026,
        'method': ('opponent-quality adjustment: per-hitter mean Stuff+/Loc+ '
                   'faced on batted balls, priced by a within-MLB-hitter '
                   'sensitivity of xwOBA-on-contact to those grades'),
        'scope': ('prices ONLY the pitch-quality channel of the level gap. '
                  'Sequencing, defense, pitcher variety, park and ball are '
                  'not included, so this is a floor on the true translation, '
                  'not the translation.'),
        'sensitivity': {'perStuffPoint': round(float(beta[0]), 6),
                        'perLocPoint': round(float(beta[1]), 6),
                        'seStuff': round(float(se[0]), 6),
                        'seLoc': round(float(se[1]), 6), **meta},
        'mlbBaseline': {'stuffPlus': round(mlb_s, 3), 'locPlus': round(mlb_l, 3),
                        'nBip': n_mlb},
        'hitters': rows,
        'comparisons': {
            'naiveLeagueMeanOffset': round(lm, 4),
            'samePlayerBridge': [
                {'hitter': n, 'aaaXwobacon': round(a, 4), 'nAAA': na,
                 'mlbXwobacon': round(m, 4), 'nMLB': nm} for n, a, na, m, nm in bridge],
        },
    }
    with open(args.out, 'w') as f:
        json.dump(blob, f, indent=1, sort_keys=True)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
