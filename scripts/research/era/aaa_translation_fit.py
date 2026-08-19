"""aaa_translation_fit.py — the AAA-to-MLB channel translation, 2023-2026.

WHAT THIS IS FOR. Pitcher+, hdERA and hpERA all read the same raw outcome
channels, and those channels are NOT level-neutral: the same arm posts a
higher strikeout rate and a lower xwOBA against Triple-A hitters than
against major-league ones. Every ROC row on the site is scored on an MLB
ruler with untranslated channels, so Rochester's staff currently grades
above the average qualified major-league pitcher. Stuff+ and Loc+ need no
translation (measured flat), which is why they are reported here as a
control rather than corrected.

WHY A SLOPE AND NOT AN OFFSET. hpERA is linear in its z-scored channels,
so adding a constant to each channel shifts every pitcher's hpERA by the
SAME amount and changes no ranking whatsoever. An offset table is a
six-number way of writing one number. The translation only earns its keep
if it is

    MLB_c  =  a_c  +  b_c * AAA_c

where b_c < 1 means the channel COMPRESSES: an extreme Triple-A strikeout
rate carries over less than proportionally, so a strikeout arm and a
ground-ball arm translate differently. That is what moves rankings.

    a_c   sets the level. Biased by promote-on-good / demote-on-bad, so it
          is reported as a bound, not a settled constant.
    b_c   is attenuated two ways: regression to the mean inside a selected
          group, and regression dilution because the AAA value is itself a
          noisy estimate. Both push b_c toward 0, so an in-sample slope is
          a FLOOR. This is why the slope is validated out of sample rather
          than trusted where it was fit.

OBJECTIVE. Predictive correlation cannot decide this: r is invariant to a
constant shift, so it cannot identify a_c at all. Calibration and ranking
are therefore reported SEPARATELY and never averaged into one score.

    calibration   mean signed error of the translated AAA channel against
                  the pitcher's actual MLB channel that season
    ranking       Spearman of translated AAA against actual MLB

RANKING IS INVARIANT PER CHANNEL, AND THE OUTPUT PROVES IT. a + b*x is
monotone, so the Spearman columns for 'none' and 'slope' come out
IDENTICAL to three decimals on every channel in every season. They are
kept as a running check that the transform is being applied at all, not
as evidence for it. A translation can only move a RANKING once the
channels are combined, because each one compresses by a different b and
that reweights them against each other. That evaluation belongs at the
composite level - translated-AAA hpERA against the pitcher's actual MLB
ERA - and is NOT in this script yet.

DOUBLE SHRINKAGE IS AN OPEN ISSUE. hpERA already shrinks every channel
toward league at its own measured n0 before z-scoring. The slopes here
are fit on RAW season values, so a fitted b below 1 and the pipeline's
own shrinkage both pull the same value toward the mean. Applying both
would regress a Triple-A line to league twice. The fix is to fit on
pipeline-shrunk values so the two are consistent; until that is done
these slopes are a measurement, not a shipping config.

VALIDATION. Leave-one-season-out over 2023-2026. Each held-out season is
scored against three references, and the winner must be named per season
rather than pooled:

    none        no translation (a=0, b=1) — what the site ships today
    intercept   level only (b=1, a = weighted mean paired delta)
    slope       the full a_c + b_c * AAA_c

Pairs are pitcher-seasons that appear at BOTH levels in the same season,
weighted by the harmonic mean of the two sample sizes, since a translation
is only as good as the thinner side.

    python3 scripts/research/era/aaa_translation_fit.py
    python3 scripts/research/era/aaa_translation_fit.py --floor 60
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
SEASONS = ('2023', '2024', '2025', '2026')

# channel -> (how to read it from a battery record, denominator field)
# izWhiff% is 1 - zcon_pct: the battery stores in-zone CONTACT per in-zone
# swing, and hpERA's channel is the whiff side of the same denominator.
CHANNELS = {
    'k_pct':     ('battery', 'k_pct',     'pa'),
    'bb_pct':    ('battery', 'bb_pct',    'pa'),
    'gb_pct':    ('battery', 'gb_pct',    'bip'),
    'xwoba':     ('battery', 'xwoba',     'pa'),
    'izwhiff':   ('izwhiff', 'zcon_pct',  'pitches'),
    'loc_raw':   ('loc',     None,        None),
    'xrv100':    ('xrv',     None,        None),
    # CONTROLS. Velocity and arm angle are physical properties of the
    # pitcher, measured on thousands of pitches, and cannot depend on the
    # quality of the hitters standing in. They are the diagnostic that
    # separates the two things a slope below 1 can mean:
    #   b(velo) near 1  -> the low slopes on the rate channels are a real
    #                      level effect plus some dilution
    #   b(velo) also low -> the attenuation is REGRESSION DILUTION across
    #                      the board and no channel slope can be trusted
    # Stuff+ would be the ideal control but adapt_statcast leaves StuffPlus
    # None, so it needs the model pass and is not available here.
    'velo':      ('battery', 'velo',      'pitches'),
    'arm':       ('battery', 'arm',       'pitches'),
}
CONTROLS = {'velo', 'arm'}


def _get(rec, kind, field, aaa):
    """One channel value + its sample size from a side's record."""
    if kind == 'battery':
        b = rec.get('battery') if aaa else rec
        if not b:
            return None, 0
        return b.get(field), b.get('pitches') or 0
    if kind == 'izwhiff':
        b = rec.get('battery') if aaa else rec
        if not b or b.get(field) is None:
            return None, 0
        return 1.0 - b[field], b.get('pitches') or 0
    if kind == 'loc':
        if aaa:
            r = rec.get('loc')
            return (r['v'], r['n']) if r else (None, 0)
        return rec.get('loc_full'), rec.get('loc_n_full') or 0
    if kind == 'xrv':
        if aaa:
            r = rec.get('xrv')
            return (r['v'], r['n']) if r else (None, 0)
        return rec.get('full'), rec.get('n_full') or 0
    return None, 0


def build_pairs(floor):
    """[(season, pid, channel, aaa_v, mlb_v, weight)] over every season."""
    aaa_all = D('_aaa_battery.json')
    mlb_bat = D('_era_battery.json')
    mlb_loc = D('_era_internal_cmdloc.json')
    mlb_xrv = D('_era_xrv100.json')
    rows = []
    per_season = defaultdict(set)
    for s in SEASONS:
        A = aaa_all.get(s) or {}
        B = mlb_bat.get(s) or {}
        L = mlb_loc.get(s) or {}
        X = mlb_xrv.get(s) or {}
        for pid, arec in A.items():
            for ch, (kind, field, _den) in CHANNELS.items():
                if kind == 'loc':
                    mrec = L.get(pid)
                elif kind == 'xrv':
                    mrec = X.get(pid)
                else:
                    mrec = (B.get(pid) or {}).get('full')
                if mrec is None:
                    continue
                av, an = _get(arec, kind, field, True)
                mv, mn = _get(mrec, kind, field, False)
                if av is None or mv is None or an < floor or mn < floor:
                    continue
                w = 2.0 * an * mn / (an + mn)      # harmonic mean
                rows.append((s, pid, ch, av, mv, w))
                per_season[s].add(pid)
    return rows, per_season


def wls(pairs):
    """Weighted least squares MLB = a + b * AAA. Returns (a, b, n, sw)."""
    sw = sum(w for _, _, w in pairs)
    if sw <= 0 or len(pairs) < 3:
        return None
    mx = sum(w * x for x, _, w in pairs) / sw
    my = sum(w * y for _, y, w in pairs) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pairs)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pairs)
    if sxx <= 0:
        return None
    b = sxy / sxx
    return my - b * mx, b, len(pairs), sw


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=int, default=60,
                    help='minimum sample on BOTH sides (channel-native units)')
    a = ap.parse_args()

    rows, per_season = build_pairs(a.floor)
    print(f'== paired pitcher-seasons at both levels, floor {a.floor}')
    for s in SEASONS:
        print(f'   {s}: {len(per_season[s])} pitchers')
    print(f'   total distinct pitcher-seasons: '
          f'{sum(len(v) for v in per_season.values())}\n')

    by_ch = defaultdict(list)
    for s, pid, ch, av, mv, w in rows:
        by_ch[ch].append((s, av, mv, w))

    print('== full-corpus fit, MLB = a + b * AAA (weighted)')
    print(f"{'channel':12s} {'n':>5s} {'a':>10s} {'b':>7s} "
          f"{'wtd mean delta':>15s}  note")
    fits = {}
    for ch in CHANNELS:
        pr = [(x, y, w) for _, x, y, w in by_ch.get(ch, [])]
        f = wls(pr)
        if f is None:
            print(f'{ch:12s} {len(pr):5d}   (too few pairs)')
            continue
        a_, b_, n, sw = f
        d = sum(w * (y - x) for x, y, w in pr) / sw
        fits[ch] = (a_, b_)
        note = 'CONTROL' if ch in CONTROLS else ''
        print(f'{ch:12s} {n:5d} {a_:10.5f} {b_:7.4f} {d:15.5f}  {note}')

    print('\n== leave-one-season-out')
    print(f"{'channel':12s} {'held out':>9s} {'n':>5s} | "
          f"{'calib none':>10s} {'calib icpt':>10s} {'calib slope':>11s} | "
          f"{'rank none':>9s} {'rank slope':>10s}  winner")
    tally = defaultdict(lambda: defaultdict(int))
    for ch in CHANNELS:
        for held in SEASONS:
            tr = [(x, y, w) for s, x, y, w in by_ch.get(ch, []) if s != held]
            te = [(x, y, w) for s, x, y, w in by_ch.get(ch, []) if s == held]
            if len(te) < 10 or len(tr) < 30:
                continue
            f = wls(tr)
            if f is None:
                continue
            a_, b_ = f[0], f[1]
            sw_tr = sum(w for _, _, w in tr)
            d_tr = sum(w * (y - x) for x, y, w in tr) / sw_tr
            sw = sum(w for _, _, w in te)
            c_none = sum(w * (y - x) for x, y, w in te) / sw
            c_icpt = sum(w * (y - (x + d_tr)) for x, y, w in te) / sw
            c_slope = sum(w * (y - (a_ + b_ * x)) for x, y, w in te) / sw
            xs = [x for x, _, _ in te]
            ys = [y for _, y, _ in te]
            r_none = spearman(xs, ys)
            r_slope = spearman([a_ + b_ * x for x in xs], ys)
            best = min((abs(c_none), 'none'), (abs(c_icpt), 'intercept'),
                       (abs(c_slope), 'slope'))[1]
            tally[ch][best] += 1
            print(f'{ch:12s} {held:>9s} {len(te):5d} | {c_none:10.5f} '
                  f'{c_icpt:10.5f} {c_slope:11.5f} | {r_none:9.3f} '
                  f'{r_slope:10.3f}  {best}')
    print('\n== calibration winner count by channel (4 held-out seasons)')
    for ch in CHANNELS:
        t = tally.get(ch)
        if t:
            print(f'   {ch:12s} ' + '  '.join(f'{k}={v}' for k, v in
                                              sorted(t.items())))
    print('\nNOTE: a monotone slope b < 1 is expected under BOTH a real '
          'level effect and pure regression dilution. The held-out '
          'calibration column is what separates them.')


if __name__ == '__main__':
    main()
