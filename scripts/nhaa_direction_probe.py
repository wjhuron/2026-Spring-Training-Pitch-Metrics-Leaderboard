#!/usr/bin/env python3
"""Does a LARGER |nHAA| actually help, and is the {FF, FC} carve-out right?

The pipeline (process_data.py:4362-4378) abs-ranks nHAA so bigger magnitude =
better, then INVERTS for FF/FC so closer-to-zero = better there. The card now
mirrors that. The comment suggests the carve-out was carried over from the VAA
rule by symmetry rather than measured, so measure it.

DESIGN
  Unit  : pitcher x pitch type, 2026 pitch leaderboard, >=MIN_N pitches. This
          is the rendered unit, so the answer is in the currency displayed.
  Test  : PARTIAL correlation of |nHAA| with each outcome, after projecting out
          velocity, IVB, HB, release point and extension. Raw correlation is
          useless here: |nHAA| is largely a function of horizontal break and
          release side, so it would mostly re-measure "more sweep is better".
          The question is whether approach angle adds anything on top.
  Sign  : outcomes flipped so POSITIVE always means better for the pitcher.
          A positive partial r therefore means "more |nHAA| helps".

POSITIVE CONTROL
  |nVAA| runs through the identical pipeline. The flat-four-seam effect is well
  established, so if this design cannot recover it (FF/FC negative, others
  positive) then it is not sensitive enough for the nHAA answer to mean
  anything either.

LIMIT: 2026 only, and 2026 is partial. Per the multi-season rule this is a
probe, not a verdict. HAA is absent from the 2021-2025 training pickles.
"""
import json, os, sys
import numpy as np

MIN_N = 100          # pitches for a pitcher-pitchtype row to enter
MIN_ROWS = 120       # rows for a pitch type to be reported
CTRL = ['velocity', 'indVertBrk', 'horzBrk', 'relPosX', 'relPosZ', 'extension']
# outcome -> +1 if higher is better for the PITCHER, -1 if lower is better
OUTCOMES = {'swStrPct': 1, 'cswPct': 1, 'rv100': 1, 'xrvoe100': 1, 'xwOBAcon': -1}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')))
meta = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
roc = set(meta.get('rocTeams') or [])
rows = [r for r in rows if (r.get('count') or 0) >= MIN_N and r.get('team') not in roc]
print(f'{len(rows):,} pitcher x pitch-type rows, 2026, >={MIN_N} pitches, MLB only\n')


def partial_r(y, x, C):
    """corr(y, x) with the columns of C projected out of both."""
    C = np.column_stack([np.ones(len(y))] + [C[:, i] for i in range(C.shape[1])])
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    if ry.std() == 0 or rx.std() == 0:
        return None
    return float(np.corrcoef(ry, rx)[0, 1])


for metric in ('nHAA', 'nVAA'):
    tag = 'POSITIVE CONTROL' if metric == 'nVAA' else 'THE QUESTION'
    print(f'=== |{metric}| partial correlation with each outcome  [{tag}] ===')
    print('    positive = MORE magnitude is better;  negative = CLOSER TO ZERO is better')
    hdr = f'{"pt":>4}{"n":>6}' + ''.join(f'{o:>11}' for o in OUTCOMES)
    print(hdr)
    for pt in ('FF', 'FC', 'SI', 'SL', 'ST', 'CH', 'CU', 'FS'):
        sub = [r for r in rows if r.get('pitchType') == pt
               and r.get(metric) is not None
               and all(r.get(c) is not None for c in CTRL)]
        if len(sub) < MIN_ROWS:
            continue
        C = np.array([[float(r[c]) for c in CTRL] for r in sub])
        x = np.abs(np.array([float(r[metric]) for r in sub]))
        line = f'{pt:>4}{len(sub):>6}'
        for o, sign in OUTCOMES.items():
            ok = [i for i, r in enumerate(sub) if r.get(o) is not None]
            if len(ok) < MIN_ROWS:
                line += f'{"-":>11}'
                continue
            y = sign * np.array([float(sub[i][o]) for i in ok])
            pr = partial_r(y, x[ok], C[ok])
            line += f'{pr:>11.3f}' if pr is not None else f'{"-":>11}'
        print(line)
    print()
