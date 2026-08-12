"""plus_scale_ratio_check.py — is the wRC+/component SD ratio stable enough to freeze?

Decision context: BB+/SD+/CT+ are to be rescaled so their spread matches wRC+,
the way Hitter+ already is (process_data.py: wrcScaleMatch). The open question
is whether that rescale factor should be measured live every run (tracks wRC+
as its spread drifts) or frozen to a constant (comparable year over year).

That turns on one empirical question: how much does

    factor = SD(wRC+) / SD(metric)          [qualified pool]

move over a season? Reconstructed per day from git history of the shipped
leaderboard, using the production qualification rule (3.1 PA x team games,
MLB only, all three components present).

metadata_rs.json began logging both sides of the ratio in mid-July
(hitterPlusStandardization.wrcScaleMatch). Where that log exists it is used to
verify the reconstruction rather than replace it, so a single method spans the
whole season.

Reports:
  1. Factor time series by date, per metric.
  2. Whether the drift comes from wRC+'s spread or the metric's.
  3. Display wobble: points a player displaying 100 +/- D at season end would
     have shown earlier under a live-tracked factor.

Usage: python3 scripts/plus_scale_ratio_check.py [--every N] [--from YYYY-MM-DD]
"""
import json
import math
import os
import re
import subprocess
import statistics as st
import sys
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import QUAL_PA_PER_GAME_MLB

LB = 'data/hitter_leaderboard_rs.json'
MD = 'data/metadata_rs.json'
METRICS = ('sdPlus', 'ctPlus', 'bbPlus', 'hitterPlus')
NON_MLB = {'ROC', 'AAA'}
COMBINED = re.compile(r'^\dTM$')
# Yardstick player: displays this far off 100 at season end.
DISPLAY_DEV = 30.0
# Below this the qualified pool is too thin to read anything into.
MIN_POOL = 40


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True,
                          cwd=ROOT).stdout


def dated_shas(path):
    """Latest sha per calendar date, oldest first."""
    by_date = OrderedDict()
    for line in git('log', '--format=%H %ad', '--date=short', '--', path).strip().splitlines():
        sha, date = line.split()
        by_date.setdefault(date, sha)      # git log is newest-first
    return sorted(by_date.items())


def psd(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def qualified(rows):
    """Production rule: MLB only, pa >= 3.1 x team games, all components present."""
    team_g = defaultdict(int)
    for r in rows:
        t = r.get('team')
        if t in NON_MLB or COMBINED.match(t or ''):
            continue
        team_g[t] = max(team_g[t], r.get('g') or 0)
    if not team_g:
        return []
    out = []
    for r in rows:
        t = r.get('team')
        if t in NON_MLB or COMBINED.match(t or ''):
            continue
        tg = team_g.get(t) or max(team_g.values())
        if (r.get('pa') or 0) < QUAL_PA_PER_GAME_MLB * tg:
            continue
        if r.get('wRCplus') is None:
            continue
        if any(r.get(k) is None for k in ('sdPlus', 'ctPlus', 'bbPlus')):
            continue
        out.append(r)
    return out


def logged_wrc_sd(sha):
    """poolWrcSd as the pipeline itself recorded it, if that run logged one."""
    blob = git('show', f'{sha}:{MD}')
    if not blob.strip():
        return None
    try:
        md = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return (((md.get('hitterPlusStandardization') or {}).get('wrcScaleMatch') or {})
            .get('poolWrcSd'))


def main():
    every = 1
    start = None
    if '--every' in sys.argv:
        every = int(sys.argv[sys.argv.index('--every') + 1])
    if '--from' in sys.argv:
        start = sys.argv[sys.argv.index('--from') + 1]

    md_by_date = dict(dated_shas(MD))
    pairs = [p for p in dated_shas(LB) if not start or p[0] >= start][::every]

    snaps = []
    for date, sha in pairs:
        blob = git('show', f'{sha}:{LB}')
        if not blob.strip():
            continue
        try:
            rows = json.loads(blob)
        except json.JSONDecodeError:
            continue
        q = qualified(rows)
        if len(q) < MIN_POOL:
            continue
        rec = {'date': date, 'n': len(q),
               'wrcSd': psd([r['wRCplus'] for r in q])}
        for k in METRICS:
            vals = [r[k] for r in q if r.get(k) is not None]
            rec[k] = psd(vals) if len(vals) == len(q) else None
        rec['loggedWrcSd'] = logged_wrc_sd(md_by_date[date]) if date in md_by_date else None
        snaps.append(rec)
        print(f'  ..{date} n={len(q)}', file=sys.stderr)

    if not snaps:
        print('No usable snapshots.')
        return

    core = ('sdPlus', 'ctPlus', 'bbPlus')
    print(f'\n{len(snaps)} snapshots, {snaps[0]["date"]} to {snaps[-1]["date"]}\n')
    hdr = f'{"date":<12}{"nQual":>6}{"wRC+SD":>8}'
    for k in core:
        hdr += f'{k[:-4] + "SD":>8}{k[:-4] + " f":>8}'
    hdr += f'{"H+ f":>7}{"chk":>7}'
    print(hdr)
    print('-' * len(hdr))
    for r in snaps:
        line = f'{r["date"]:<12}{r["n"]:>6}{r["wrcSd"]:>8.2f}'
        for k in core:
            line += (f'{r[k]:>8.2f}{r["wrcSd"] / r[k]:>8.3f}') if r[k] else f'{"-":>16}'
        line += f'{r["wrcSd"] / r["hitterPlus"]:>7.3f}' if r.get('hitterPlus') else f'{"-":>7}'
        # chk: reconstruction vs the pipeline's own logged wRC+ SD
        line += f'{r["wrcSd"] - r["loggedWrcSd"]:>7.2f}' if r.get('loggedWrcSd') else f'{"-":>7}'
        print(line)

    print('\n=== Drift decomposition (first -> last snapshot) ===')
    a, b = snaps[0], snaps[-1]
    print(f'{"series":<12}{"start":>9}{"end":>9}{"change":>9}')
    print(f'{"wRC+ SD":<12}{a["wrcSd"]:>9.2f}{b["wrcSd"]:>9.2f}'
          f'{(b["wrcSd"] / a["wrcSd"] - 1) * 100:>8.1f}%')
    for k in core:
        if a.get(k) and b.get(k):
            print(f'{k + " SD":<12}{a[k]:>9.2f}{b[k]:>9.2f}'
                  f'{(b[k] / a[k] - 1) * 100:>8.1f}%')

    print('\n=== Factor stability + display cost of freezing ===')
    print(f'{"metric":<10}{"min f":>8}{"max f":>8}{"final f":>9}{"range %":>9}'
          f'{"CV %":>7}{"wobble":>9}')
    for k in core + ('hitterPlus',):
        fs = [r['wrcSd'] / r[k] for r in snaps if r.get(k)]
        if len(fs) < 2:
            continue
        # A player displaying 100 +/- DISPLAY_DEV at season end has raw
        # deviation DISPLAY_DEV / f_final; under a live factor his displayed
        # deviation at snapshot i is DISPLAY_DEV * f_i / f_final.
        wobble = DISPLAY_DEV * max(abs(f / fs[-1] - 1) for f in fs)
        rng = (max(fs) - min(fs)) / st.mean(fs) * 100
        cv = st.pstdev(fs) / st.mean(fs) * 100
        print(f'{k:<10}{min(fs):>8.3f}{max(fs):>8.3f}{fs[-1]:>9.3f}'
              f'{rng:>9.1f}{cv:>7.1f}{wobble:>9.1f}')
    print(f'\nwobble = points a player displaying 100 +/- {DISPLAY_DEV:.0f} at season end '
          f'would have\nshown at the worst earlier snapshot, under a live-tracked factor.')
    print('chk = reconstructed wRC+ SD minus the pipeline\'s own logged value '
          '(0.00 = method verified).')


if __name__ == '__main__':
    main()
