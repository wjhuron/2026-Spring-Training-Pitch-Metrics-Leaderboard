#!/usr/bin/env python3
"""ibb_injection_test.py — which denominators break if no-pitch IBBs enter the
pitch stream?

Scoping tool for the IBB fix. A no-pitch intentional walk is a PA with no
pitch, so it cannot exist in pitch-level data; the season path papers over it
with the boxscore merge, but every pitch-derived path (date-window cards,
research scripts) runs short. Wood: 531 PA-ending rows against an official
535. League-wide the cache holds 106 Intent Walk rows against roughly 290.

Putting the rows in is easy. The risk is that EVERY per-pitch denominator
counts rows, so a PA-marker row silently inflates them. Rather than grep for
call sites and hope, this injects the rows into the golden harness's frozen
Sheets capture, runs the full pipeline both ways, and diffs. Whatever moves
that should not is the guard list.

Usage (harness must already have captured inputs this session):
    python3 scripts/research/misc/ibb_injection_test.py --harness DIR inject
    cd DIR && PYTHONHASHSEED=0 python3 golden_run.py run IBB_BASE      # original
    # swap golden_input_rs.pkl for the .INJECT.pkl, then:
    cd DIR && PYTHONHASHSEED=0 python3 golden_run.py run IBB_INJECT
    python3 scripts/research/misc/ibb_injection_test.py --harness DIR diff

CRITICAL TEST-CONSTRUCTION TRAP, learned the hard way. The harness captures
Sheets rows AFTER pipeline/fetch.py:293 turns '' into None. An injected row
must therefore use None for blank columns, NOT ''. With '' the row passes the
`BBType is not None` batted-ball filter and shows up as five phantom batted
balls, moving nBip/GB%/FB%/LD%/PU%/AirPull% and inventing a bug that is not
there. The first run of this test did exactly that.

RESULT, 2026-08-18, five rows injected for one hitter:

  Hitter side, contaminated (needs a guard):
      count, swingPct, firstPitchSwingPct, rv100, xRv100
  Hitter side, clean (no guard needed - they key on a None column):
      every batted-ball rate, every zone/whiff/chase/contact rate
  Hitter side, unchanged because the boxscore already overrides them:
      pa, avg, obp, slg, wOBA, bbPct, kPct, wRC+

  Pitcher side, contaminated:
      strikePct, izPct, cswPct, swStrRate, rv100, xRv100,
      plus the count-state family (fpsPct, earlyActionPct, oneOneWinPct)
  Pitcher side, already correct by policy once the row exists:
      pa rises (an IBB is a PA); bbPct's denominator rises while its
      numerator does not, because pitcher BB% uses unintentional walks -
      exactly the intended rule; tbf is boxscore-overridden and unmoved.

  Blast radius on shipped season output: 16 of 773 hitters moved, all through
  percentile-pool boundaries. The season leaderboard is essentially unchanged,
  because the values that would move are boxscore-overridden already.
"""
import argparse
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

# Columns a no-pitch IBB legitimately carries. Everything else is a PITCH
# measurement and must be None.
KEEP = {'Game Date', 'PTeam', 'Pitcher', 'Throws', 'BTeam', 'Batter', 'Bats',
        'Count', 'Runners', 'Outs', 'Event', 'PitchID', '_source',
        '_sheet_tab', '_sheet_row', '_barrelSource'}

PER_PITCH = {'count', 'swingPct', 'izSwingPct', 'chasePct', 'izSwChase',
             'contactPct', 'izContactPct', 'whiffPct', 'izWhiffPct', 'nSwings',
             'twoStrikeWhiffPct', 'firstPitchSwingPct', 'rv100', 'xRv100',
             'runValue', 'xRunValue', 'strikePct', 'izPct', 'cswPct',
             'swStrRate', 'swStrPct', 'fpsPct', 'earlyActionPct',
             'oneOneWinPct'}
BIP_RATES = {'nBip', 'gbPct', 'fbPct', 'ldPct', 'puPct', 'airPullPct',
             'pullPct', 'middlePct', 'oppoPct', 'hardHitPct', 'barrelPct',
             'avgEVAll', 'maxEV', 'ev50'}


def inject(harness, batter, n):
    src = os.path.join(harness, 'golden_input_rs.ORIG.pkl')
    if not os.path.exists(src):
        src = os.path.join(harness, 'golden_input_rs.pkl')
    with open(src, 'rb') as f:
        rows = pickle.load(f)
    cols = list(rows[0].keys())
    donors = [r for r in rows if r.get('Batter') == batter and r.get('Event')][:n]
    if len(donors) < n:
        print(f"ERROR: only {len(donors)} donor PAs for {batter}")
        return 2
    out = []
    for i, d in enumerate(donors):
        r = {c: None for c in cols}          # None, never '' — see the docstring
        for c in KEEP:
            r[c] = d.get(c)
        r['Event'] = 'Intent Walk'
        r['Description'] = None
        gp, ab, _ = str(d['PitchID']).split('_')
        r['PitchID'] = f"{gp}_{int(ab) + 900:03d}_00"   # _00 = no pitch thrown
        r['_sheet_row'] = 999000 + i
        out.append(r)
    dst = os.path.join(harness, 'golden_input_rs.INJECT.pkl')
    with open(dst, 'wb') as f:
        pickle.dump(rows + out, f, protocol=4)
    print(f"injected {len(out)} no-pitch IBB rows for {batter} -> {dst}")
    print(f"  ({len(rows)} -> {len(rows) + len(out)} rows)")
    return 0


def diff(harness, base, change, batter):
    runs = os.path.join(harness, 'runs')

    def load(lbl, name):
        with open(os.path.join(runs, lbl, 'data', name)) as f:
            return json.load(f)

    B = {(r['hitter'], r['team']): r for r in load(base, 'hitter_leaderboard_rs.json')}
    I = {(r['hitter'], r['team']): r for r in load(change, 'hitter_leaderboard_rs.json')}
    key = next((k for k in B if k[0] == batter), None)
    if key is None:
        print(f"ERROR: {batter} not in the base run")
        return 2
    b, i = B[key], I[key]
    moved = sorted(k for k in set(b) | set(i) if b.get(k) != i.get(k))
    print(f"HITTER {key[0]} ({key[1]}): {len(moved)} fields moved\n")
    guards = []
    for k in moved:
        if k in PER_PITCH or k in BIP_RATES:
            cls = 'NEEDS GUARD'
            guards.append(k)
        elif k.endswith('_pctl'):
            cls = 'percentile'
        else:
            cls = 'PA-level (intended)'
        print(f"  {k:22} {str(b.get(k))[:13]:>13} -> {str(i.get(k))[:13]:<13} {cls}")
    print(f"\nGUARD LIST (hitter): {guards or 'none'}")

    PB = {(r['pitcher'], r['team']): r for r in load(base, 'pitcher_leaderboard_rs.json')}
    PI = {(r['pitcher'], r['team']): r for r in load(change, 'pitcher_leaderboard_rs.json')}
    pv = set()
    for k in PB:
        for f in set(PB[k]) | set(PI[k]):
            if PB[k].get(f) != PI[k].get(f) and not f.endswith('_pctl'):
                pv.add(f)
    print(f"\nGUARD LIST (pitcher, values only): "
          f"{sorted(f for f in pv if f in PER_PITCH) or 'none'}")
    print(f"  other pitcher values that moved (expected - PA rose): "
          f"{sorted(f for f in pv if f not in PER_PITCH)}")
    spill = sum(1 for k in B if k != key
                and any(B[k].get(f) != I[k].get(f) for f in set(B[k]) | set(I[k])))
    print(f"\nleague spill: {spill} of {len(B) - 1} other hitters moved")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['inject', 'diff'])
    ap.add_argument('--harness', required=True, help='copied golden_harness dir')
    ap.add_argument('--batter', default='Wood, James')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--base', default='IBB_BASE')
    ap.add_argument('--change', default='IBB_INJECT2')
    a = ap.parse_args()
    sys.exit(inject(a.harness, a.batter, a.n) if a.mode == 'inject'
             else diff(a.harness, a.base, a.change, a.batter))


if __name__ == '__main__':
    main()
