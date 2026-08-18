#!/usr/bin/env python3
"""pitcher_plus_scale_atlas.py — do the pitcher "+" metrics honour the "+" contract?

Context (2026-08-18, Wally): a stat named X+ promises 100 = league average and
one point = one percent. The hitter side was moved onto that contract in
6b5a6d68. Every pitcher "+" is still built as `100 +/- 10z`, i.e. a point is a
tenth of a standard deviation, so none of them honour it.

For a metric whose raw quantity is a RUN VALUE centred on zero (Stuff+, Loc+,
and the composites) there is no self-ratio, so the contract has to be met the
way Hitter+ meets it: calibrate the spread so the regression slope against a
run-prevention currency is exactly 1. That happens iff

    SD(metric) = r x SD(currency)

For a metric whose raw quantity is strictly positive with a real zero
(Command+, whose raw is average miss distance in inches) a self-ratio exists
and is the honest scale, exactly as CT+ resolved on the hitter side.

This measures both, on 2021-2026 replicates, and answers the two questions
that decide whether any of it can ship:

  1. Is Command+'s run correlation reliably ~0, or was 2026 a one-off?
  2. Are the Stuff+/Loc+/Pitching+ slopes stable enough to quote a range?

Currency: FIP+, in the linear form pipeline/eraplus.py already uses, so one
point above 100 = one percent fewer runs allowed:

    FIP+ = 200 - 100 x FIP / lgFIP

cFIP is derived from each season's OWN pool (the constant that makes league
FIP equal league ERA), so no external per-season constant is guessed.

Inputs, all already cached — this reads only, and pulls nothing:
  data/_era_targets.json        official boxscore counts per pitcher-season
  data/_era_internal_stuff.json raw Stuff run value      (2021-2025)
  data/_era_internal_cmdloc.json raw Loc run value + raw command miss (2021-2026)

Usage: python3 scripts/research/misc/pitcher_plus_scale_atlas.py [--min-ip 60]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

MIN_IP = 60.0
for i, a in enumerate(sys.argv):
    if a == '--min-ip' and i + 1 < len(sys.argv):
        MIN_IP = float(sys.argv[i + 1])

# Pitching+ is a fixed blend of Stuff+ and Loc+; read the weight from its one
# home rather than retyping it.
from pipeline.utils import PITCHING_W_STUFF


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def psd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def lin(xs, ys):
    """(slope, r) of ys on xs."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def zscale(vals, k=10.0, higher_is_better=True):
    """The shipped convention: 100 +/- k x z. Sign flips for a metric whose
    raw is better when SMALLER (Loc+ run value allowed, Command+ miss)."""
    mu, sd = sum(vals) / len(vals), psd(vals)
    if sd <= 0:
        return [100.0] * len(vals)
    s = 1.0 if higher_is_better else -1.0
    return [100.0 + s * k * (v - mu) / sd for v in vals]


def main():
    targets = load('_era_targets.json')
    stuff = load('_era_internal_stuff.json')
    cmdloc = load('_era_internal_cmdloc.json')
    seasons = sorted(targets)

    print(f"Pitcher '+' scale atlas — replicates {seasons[0]}-{seasons[-1]}, "
          f"min {MIN_IP:.0f} IP")
    print(f"Currency: FIP+ = 200 - 100 x FIP/lgFIP  (one point = one percent "
          f"fewer runs allowed)")
    print(f"Pitching+ weight read from pipeline.utils: "
          f"{PITCHING_W_STUFF:.2f} Stuff / {1-PITCHING_W_STUFF:.2f} Loc\n")

    per_season = {}
    for y in seasons:
        pit = targets[y]['pitchers']
        st = stuff.get(y, {})
        cl = cmdloc.get(y, {})
        rows = []
        for pid, t in pit.items():
            ip = (t.get('outs') or 0) / 3.0
            if ip < MIN_IP:
                continue
            s = st.get(pid)
            c = cl.get(pid) or {}
            # Each metric joins independently: the 2026 cmdloc cache carries
            # command only, so requiring all three raws silently dropped the
            # whole live season.
            if (c.get('cmd_full') is None and c.get('loc_full') is None
                    and (s or {}).get('stuff_full') is None):
                continue
            rows.append({
                'id': pid, 'ip': ip,
                'er': t.get('er') or 0, 'so': t.get('so') or 0,
                'bb': (t.get('bb') or 0) - (t.get('ibb') or 0),
                'hbp': t.get('hbp') or 0, 'hr': t.get('hr') or 0,
                'stuff_raw': (s or {}).get('stuff_full'),
                'loc_raw': c.get('loc_full'),
                'cmd_raw': c.get('cmd_full'),
            })
        if len(rows) < 40:
            print(f"{y}: only {len(rows)} joined rows — skipped")
            continue

        # FIP with cFIP derived from THIS pool: the constant that makes the
        # pool's FIP equal its ERA. No external season constant is guessed.
        tip = sum(r['ip'] for r in rows)
        lg_era = sum(r['er'] for r in rows) * 9.0 / tip
        core = (13.0 * sum(r['hr'] for r in rows)
                + 3.0 * sum(r['bb'] + r['hbp'] for r in rows)
                - 2.0 * sum(r['so'] for r in rows)) / tip
        cfip = lg_era - core
        for r in rows:
            r['fip'] = ((13.0 * r['hr'] + 3.0 * (r['bb'] + r['hbp'])
                         - 2.0 * r['so']) / r['ip']) + cfip
        lg_fip = sum(r['fip'] * r['ip'] for r in rows) / tip
        for r in rows:
            r['fipPlus'] = 200.0 - 100.0 * r['fip'] / lg_fip

        # Build each metric on the SHIPPED convention so the measured slope is
        # the slope a reader gets today.
        metrics = {}
        have_loc = [r for r in rows if r['loc_raw'] is not None]
        have_cmd = [r for r in rows if r['cmd_raw'] is not None]
        have_stuff = [r for r in rows if r['stuff_raw'] is not None]
        # Loc raw is a run value ALLOWED: smaller is better.
        if len(have_loc) >= 40:
            for r, v in zip(have_loc, zscale([r['loc_raw'] for r in have_loc],
                                             higher_is_better=False)):
                r['locPlus'] = v
            metrics['Loc+'] = have_loc
        # Command raw is miss distance: smaller is better.
        if len(have_cmd) >= 40:
            for r, v in zip(have_cmd, zscale([r['cmd_raw'] for r in have_cmd],
                                             higher_is_better=False)):
                r['commandPlus'] = v
            metrics['Command+'] = have_cmd
        if len(have_stuff) >= 40:
            for r, v in zip(have_stuff, zscale([r['stuff_raw'] for r in have_stuff],
                                               higher_is_better=True)):
                r['stuffPlus'] = v
            metrics['Stuff+'] = have_stuff
            blend = [r for r in have_stuff if r.get('locPlus') is not None]
            if len(blend) >= 40:
                for r in blend:
                    r['pitchingPlus'] = (PITCHING_W_STUFF * r['stuffPlus']
                                         + (1 - PITCHING_W_STUFF) * r['locPlus'])
                metrics['Pitching+'] = blend

        out = {'n': len(rows), 'lgFIP': lg_fip, 'cFIP': cfip,
               'sdFipPlus': psd([r['fipPlus'] for r in rows]), 'metrics': {}}
        for name, pool in metrics.items():
            key = {'Loc+': 'locPlus', 'Command+': 'commandPlus',
                   'Stuff+': 'stuffPlus', 'Pitching+': 'pitchingPlus'}[name]
            v = [r[key] for r in pool]
            y_ = [r['fipPlus'] for r in pool]
            slope, r_ = lin(v, y_)
            sd_now = psd(v)
            sd_need = abs(r_) * psd(y_) if r_ is not None else None
            scaled = ([100 + (x - 100) * (sd_need / sd_now) for x in v]
                      if sd_need and sd_now > 0 else None)
            out['metrics'][name] = {
                'n': len(pool), 'sd': sd_now, 'r': r_, 'slope': slope,
                'sdNeeded': sd_need,
                'minNow': min(v), 'maxNow': max(v),
                'minNew': min(scaled) if scaled else None,
                'maxNew': max(scaled) if scaled else None,
            }
        per_season[y] = out

    NAMES = ['Stuff+', 'Loc+', 'Command+', 'Pitching+']
    print("=" * 78)
    print("PER SEASON — r and slope against runs prevented, on TODAY's scale")
    print("=" * 78)
    print(f"{'yr':6} {'n':>4} {'SD(FIP+)':>9}  " +
          "  ".join(f"{m:>9}" for m in NAMES))
    for y, o in per_season.items():
        cells = []
        for m in NAMES:
            d = o['metrics'].get(m)
            cells.append(f"{d['r']:>+9.3f}" if d else f"{'-':>9}")
        print(f"{y:6} {o['n']:>4} {o['sdFipPlus']:>9.1f}  " + "  ".join(cells) + "   <- r")
        cells = []
        for m in NAMES:
            d = o['metrics'].get(m)
            cells.append(f"{d['slope']:>9.2f}" if d else f"{'-':>9}")
        print(f"{'':6} {'':>4} {'':>9}  " + "  ".join(cells) + "   <- slope")

    print("\n" + "=" * 78)
    print("STABILITY of r  (the question that decides if a rescale can ship)")
    print("=" * 78)
    for m in NAMES:
        rs = [o['metrics'][m]['r'] for o in per_season.values() if m in o['metrics']]
        if not rs:
            continue
        print(f"  {m:10} n={len(rs)} seasons   mean {sum(rs)/len(rs):+.3f}   "
              f"range {min(rs):+.3f} to {max(rs):+.3f}   spread {max(rs)-min(rs):.3f}")

    print("\n" + "=" * 78)
    print("RANGE under the '+' contract (slope forced to 1.000), per season")
    print("=" * 78)
    for m in NAMES:
        print(f"\n  {m}")
        print(f"    {'yr':6} {'SD now':>7} {'SD needed':>10} {'range now':>15} "
              f"{'range at slope 1':>18}")
        for y, o in per_season.items():
            d = o['metrics'].get(m)
            if not d:
                continue
            print(f"    {y:6} {d['sd']:>7.2f} {d['sdNeeded']:>10.2f} "
                  f"{('%.0f to %.0f' % (d['minNow'], d['maxNow'])):>15} "
                  f"{('%.0f to %.0f' % (d['minNew'], d['maxNew'])):>18}")

    print("\n" + "=" * 78)
    print("COMMAND+ as a SELF-RATIO instead (its raw is miss distance in inches,")
    print("strictly positive with a real zero — the CT+ resolution)")
    print("=" * 78)
    print(f"    {'yr':6} {'lg miss':>8} {'SD':>6} {'min':>7} {'max':>7}")
    for y in seasons:
        cl = cmdloc.get(y, {})
        pit = targets[y]['pitchers']
        miss = [cl[p]['cmd_full'] for p, t in pit.items()
                if (t.get('outs') or 0) / 3.0 >= MIN_IP
                and p in cl and cl[p].get('cmd_full')]
        if len(miss) < 40:
            continue
        lg = sum(miss) / len(miss)
        plus = [200.0 - 100.0 * v / lg for v in miss]
        print(f"    {y:6} {lg:>8.3f} {psd(plus):>6.2f} {min(plus):>7.1f} {max(plus):>7.1f}")

    # ── Validate the replication against shipped values ──
    # CLAUDE.md: never trust a replication until it is checked against a
    # known-shipped number. The harness rebuilds each metric from cached raws
    # with a plain 100+/-10z, while the shipped scorers add per-pitch atoms and
    # cascade shrinkage, so exact equality is NOT expected — a high correlation
    # is what says the harness is measuring the same thing.
    print("\n" + "=" * 78)
    print("VALIDATION vs the shipped 2026 leaderboard")
    print("=" * 78)
    try:
        with open(os.path.join(DATA, 'pitcher_leaderboard_rs.json')) as f:
            ship = json.load(f)
        by_id = {}
        for r in ship:
            mid = r.get('mlbId')
            if mid is not None:
                by_id[str(int(mid))] = r
        o26 = per_season.get('2026')
        if o26 is None:
            print("  no 2026 rows in the atlas — cannot validate")
        else:
            cl26 = cmdloc.get('2026', {})
            pit26 = targets['2026']['pitchers']
            rows26 = [(p, cl26[p]['cmd_full']) for p, t in pit26.items()
                      if (t.get('outs') or 0) / 3.0 >= MIN_IP
                      and p in cl26 and cl26[p].get('cmd_full')]
            mu = sum(v for _, v in rows26) / len(rows26)
            sd = psd([v for _, v in rows26])
            recon = {p: 100.0 - 10.0 * (v - mu) / sd for p, v in rows26}
            pairs = [(recon[p], by_id[p]['commandPlus']) for p in recon
                     if p in by_id and by_id[p].get('commandPlus') is not None]
            if len(pairs) >= 30:
                sl, rr = lin([a for a, _ in pairs], [b for _, b in pairs])
                print(f"  Command+ reconstructed vs shipped: n={len(pairs)}, "
                      f"r={rr:.4f}, slope={sl:.3f}")
                print(f"    (r near 1 means the harness measures the shipped "
                      f"metric; slope off 1 is the shrinkage the harness omits)")
            else:
                print(f"  only {len(pairs)} matched rows — too few to validate")
    except (OSError, ValueError, KeyError) as e:
        print(f"  validation skipped ({type(e).__name__}: {e})")

    out_path = os.path.join(DATA, '_pitcher_plus_scale_atlas.json')
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(per_season, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
