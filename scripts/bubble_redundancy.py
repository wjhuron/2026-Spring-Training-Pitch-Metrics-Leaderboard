#!/usr/bin/env python3
"""bubble_redundancy.py — which card bubbles duplicate each other, and which
candidate metrics would actually add a new axis?

A percentile bubble costs a row of scarce card space. It earns that row only if
it tells the reader something the other bubbles do not already imply. Two
questions, one method:

  1. REDUNDANCY: among the bubbles currently on the card, which pairs are so
     correlated that the second one is decoration?
  2. ADDITIVITY: for each candidate metric not on the card, what is its highest
     |r| against anything already there? A low ceiling means a genuinely new
     axis; a high one means it is a restatement.

Pearson r over the qualified pool, using the shipped leaderboards (the same
numbers the cards read), so this measures the displayed quantities rather than
some upstream version of them.

Usage: python3 scripts/bubble_redundancy.py
"""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import QUAL_PA_PER_GAME_MLB

NON_MLB = {'ROC', 'AAA'}
COMBINED = re.compile(r'^\dTM$')

HITTER_ON_CARD = [
    ('Hitter+', 'hitterPlus'), ('Batted Ball+', 'bbPlus'), ('Contact+', 'ctPlus'),
    ('Swing Dec+', 'sdPlus'), ('Bat Speed', 'batSpeed'), ('xwOBA', 'xwOBA'),
    ('xwOBAcon', 'xwOBAcon'), ('BABIP', 'babip'), ('Max EV', 'maxEV'),
    ('Hard-Hit%', 'hardHitPct'), ('Barrel%', 'barrelPct'), ('AirPull%', 'airPullPct'),
    ('GB%', 'gbPct'), ('BB%', 'bbPct'), ('K%', 'kPct'), ('Swing%', 'swingPct'),
    ('Z-Sw-Ch%', 'izSwChase'), ('Z-Contact%', 'izContactPct'), ('Chase%', 'chasePct'),
]
HITTER_CANDIDATES = [
    ('Sprint Speed', 'sprintSpeed'), ('EV50', 'ev50'), ('Avg EV', 'avgEVAll'),
    ('xwOBAsp', 'xwOBAsp'), ('SprayVal', 'sprayVal'), ('ISO', 'iso'),
    ('LD%', 'ldPct'), ('HR/FB%', 'hrFbPct'), ('Avg FB Dist', 'avgFbDist'),
    ('2K Whiff%', 'twoStrikeWhiffPct'), ('Whiff%', 'whiffPct'),
    ('Ideal AA%', 'idealAAPct'), ('Attack Angle', 'attackAngle'),
    ('Squared-Up%', 'squaredUpPct'), ('Blast%', 'blastPct'),
    ('Z-Swing%', 'izSwingPct'), ('BB/K', 'bbToK'),
]

PITCHER_ON_CARD = [
    ('xRV', 'xRunValue'), ('xRV/100', 'xRv100'), ('Pitcher+', 'pitcherPlus'),
    ('xwOBA', 'xwOBA'), ('K%', 'kPct'), ('BB%', 'bbPct'), ('K-BB%', 'kbbPct'),
    ('Whiff%', 'swStrPct'), ('Z-Whiff%', 'izWhiffPct'), ('Chase%', 'chasePct'),
    ('xwOBAcon', 'xwOBAcon'), ('BABIP', 'babip'), ('Hard-Hit%', 'hardHitPct'),
    ('Barrel%', 'barrelPctAgainst'), ('GB%', 'gbPct'), ('Velocity', 'fbVelo'),
    ('Stuff+', 'stuffScore'), ('Loc+', 'locPlus'), ('Pitching+', 'pitchingScore'),
]
PITCHER_CANDIDATES = [
    ('Command+', 'commandPlus'), ('xwOBAsp', 'xwOBAsp'), ('Extension', 'extension'),
    ('Arm Angle', 'armAngle'), ('Zone%', 'izPct'), ('F-Strike%', 'fpsPct'),
    ('CSW%', 'cswPct'), ('2K Whiff%', 'twoStrikeWhiffPct'), ('HR/FB%', 'hrFbPct'),
    ('LD%', 'ldPct'), ('PU%', 'puPct'),
]


def pearson(xs, ys):
    n = len(xs)
    if n < 20:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def paired(rows, ka, kb):
    xs, ys = [], []
    for r in rows:
        a, b = r.get(ka), r.get(kb)
        if a is not None and b is not None:
            xs.append(float(a)); ys.append(float(b))
    return xs, ys


def report(title, rows, on_card, candidates):
    print(f'\n{"=" * 78}\n{title}  (n={len(rows)})\n{"=" * 78}')

    print('\n-- Redundancy among bubbles already on the card (|r| >= .80) --')
    pairs = []
    for i, (la, ka) in enumerate(on_card):
        for lb, kb in on_card[i + 1:]:
            r = pearson(*paired(rows, ka, kb))
            if r is not None and abs(r) >= 0.80:
                pairs.append((abs(r), la, lb, r))
    if not pairs:
        print('   none')
    for a, la, lb, r in sorted(pairs, reverse=True):
        print(f'   {la:<13} vs {lb:<13} r = {r:+.2f}   '
              f'({a**2*100:.0f}% of one explained by the other)')

    print('\n-- Candidates: highest |r| against ANY bubble already on the card --')
    print(f'   {"candidate":<15}{"n":>5}{"max |r|":>9}   closest existing bubble')
    scored = []
    for lc, kc in candidates:
        best_r, best_l, nn = 0.0, '-', 0
        for lo, ko in on_card:
            xs, ys = paired(rows, kc, ko)
            r = pearson(xs, ys)
            if r is not None and abs(r) > abs(best_r):
                best_r, best_l, nn = r, lo, len(xs)
        if best_l != '-':
            scored.append((abs(best_r), lc, nn, best_r, best_l))
    for a, lc, nn, r, lo in sorted(scored):
        flag = 'NEW AXIS' if a < 0.55 else ('partial' if a < 0.75 else 'restates')
        print(f'   {lc:<15}{nn:>5}{r:>+9.2f}   {lo:<13} {flag}')


def main():
    hitters = json.load(open(os.path.join(ROOT, 'data/hitter_leaderboard_rs.json')))
    md = json.load(open(os.path.join(ROOT, 'data/metadata_rs.json')))
    tg = md['teamGamesPlayed']
    hq = []
    for r in hitters:
        t = r.get('team') or ''
        if t in NON_MLB or COMBINED.match(t):
            continue
        g = tg.get(t) or max(tg.values())
        if (r.get('pa') or 0) >= QUAL_PA_PER_GAME_MLB * g:
            hq.append(r)
    report('HITTER CARD', hq, HITTER_ON_CARD, HITTER_CANDIDATES)

    pitchers = json.load(open(os.path.join(ROOT, 'data/pitcher_leaderboard_rs.json')))
    # No PA analogue published per row; require the core bubbles to be present,
    # which selects the same population the card's percentiles rank against.
    pq = [r for r in pitchers
          if (r.get('team') or '') not in NON_MLB
          and not COMBINED.match(r.get('team') or '')
          and r.get('stuffScore') is not None and r.get('xRv100') is not None
          and r.get('swStrPct') is not None]
    report('PITCHER CARD', pq, PITCHER_ON_CARD, PITCHER_CANDIDATES)


if __name__ == '__main__':
    main()
