"""team_tag_audit.py -- league-wide mistag audit, compared WITHIN individual games.

READ-ONLY. Runs the validated ROC methodology (scripts/audits/roc_tag_audit.py) across
every team, and reports HIGH-confidence candidates only.

Everything here is inherited from roc_tag_audit: game-level leave-one-out
centroids, within-game noise scales, no d_own gate (measured inert), release as
a diagnostic rather than an axis, the sparse-tracking guard, and orphan handling
tiered off its measured precision cliff. See that file for the evidence behind
each choice; nothing is re-derived here.

Two things ARE specific to going league-wide:

  * NOISE SCALES ARE POOLED PER TRACKING POPULATION, NOT PER TEAM. Measured
    across the 30 MLB clubs, per-team within-game scales sit within roughly
    +/-12% of the pooled MLB value on every axis and within +/-5% for most
    (Velocity med 1.000, RTilt med 1.008, OTilt med 0.993). They share one
    Hawk-Eye deployment, so a per-team estimate would add sampling noise
    without capturing a real difference. AAA is a different measurement
    population and keeps its own scales -- pooled MLB RTilt is 4.580 against
    ROC's 5.483, and OTilt 7.590 against 9.056.

  * HIGH TIER ONLY. Confidence is 100*(0.50*m + 0.30*a + 0.20*c) with m the
    margin term, a the fraction of axes that agree, and c same-game
    reinforcement. High (>=75) therefore needs a large margin AND most axes
    agreeing AND ideally more than one pitch flipping the same way in the same
    game. It is a deliberately narrow gate: ROC's whole 2026 produced exactly
    one High flag. Counts for every tier still print to stdout so the filtering
    is visible.

Usage:
    python3 scripts/audits/team_tag_audit.py             # report + per-team summary
    python3 scripts/audits/team_tag_audit.py --csv       # + ~/Downloads CSV
    python3 scripts/audits/team_tag_audit.py --plots     # + discriminant plots
    python3 scripts/audits/team_tag_audit.py --tier Medium   # widen the tier filter
"""
import os, sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pitch_tag_audit as A
import roc_tag_audit as R

OUT_CSV = os.path.expanduser('~/Downloads/team_tag_audit_2026.csv')
PLOTDIR = os.path.expanduser('~/Downloads/team_tag_plots')
TIERS = {'High'}


def main():
    global TIERS
    if '--tier' in sys.argv:
        want = sys.argv[sys.argv.index('--tier') + 1]
        TIERS = {'High'} if want == 'High' else {'High', 'Medium'}

    subj = A.load()
    allp = R.prep(subj)

    # split by tracking population, not by team
    pops = {
        'MLB': [p for p in allp if p.get('_source') == 'MLB'],
        'AAA': [p for p in allp if p.get('_source') != 'MLB'],
    }

    flags, groups_all, scales_by_pop = [], {}, {}
    for name, rows in pops.items():
        if not rows:
            continue
        scl = R.game_scales(rows)
        scales_by_pop[name] = scl
        rs = R.release_stats(rows)
        res, orph = R.score_game(rows, scl)
        for p in rows:
            groups_all.setdefault(name, defaultdict(lambda: defaultdict(list)))
            groups_all[name][(p['Pitcher'], p['_g'])][p['Pitch Type']].append(p)

        cand = [v for v in res.values()
                if v['margin'] >= R.MARGIN_MIN and v['d_best'] <= R.D_BEST_MAX]
        cand += [o for o in orph if o['d_best'] <= R.ORPHAN_DBEST]
        for f in cand:
            f['rel'] = R.rel_outlier(f['p'], rs)
            f['pop'] = name
        R.confidence(cand, rows)
        flags += cand
        print(f"{name}: {len(rows)} usable pitches, {len(res)} compared in-game, "
              f"{len(orph)} orphans -> {len(cand)} candidates")
        print(f"   within-game scales: " +
              "  ".join(f"{m}={scl[m]:.2f}" for m in ('Velocity', 'RTilt', 'OTilt')))

    print(f"\ntier breakdown across all teams: "
          f"{dict(Counter(f['tier'] for f in flags))}")

    keep = [f for f in flags if f['tier'] in TIERS]
    keep.sort(key=lambda f: (-f['conf'], f['p'].get('PTeam') or '', f['p']['Pitcher']))
    print(f"reporting {len(keep)} candidates at tier(s) {'/'.join(sorted(TIERS))}\n")

    byteam = defaultdict(list)
    for f in keep:
        byteam[f['p'].get('PTeam')].append(f)
    print(f"=== {len(keep)} CANDIDATES ACROSS {len(byteam)} TEAMS ===")
    print("  swaps:", Counter((f['own'], f['tgt']) for f in keep).most_common())
    for team in sorted(byteam, key=lambda t: (-len(byteam[t]), t or '')):
        print(f"\n  --- {team} ({len(byteam[team])}) ---")
        for f in sorted(byteam[team], key=lambda f: -f['conf']):
            p = f['p']
            rel = f['rel']
            rstr = (f"rel z{rel:.1f}" + ("  <-- LIKELY TRACKING GLITCH" if rel >= 3 else "")
                    ) if rel is not None else "rel n/a"
            if f['kind'] == 'orphan':
                print(f"    {f['conf']:3d} {f['tier']:6s} {p['Pitcher']:22s} "
                      f"{p['Game Date']} {f['own']}->{f['tgt']} ORPHAN "
                      f"(only {f['nsame']} this game) d_best{f['d_best']:.2f} {rstr}")
            else:
                print(f"    {f['conf']:3d} {f['tier']:6s} {p['Pitcher']:22s} "
                      f"{p['Game Date']} {f['own']}->{f['tgt']} "
                      f"margin{f['margin']:.2f} d_own{f['d_own']:.2f} "
                      f"d_best{f['d_best']:.2f} agree{f['agree']}/{f['tot']} "
                      f"flip{f['nflip']}/{f['ntype']} {rstr}")
            print(f"        {f['why']}")

    if '--csv' in sys.argv:
        import csv
        with open(OUT_CSV, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['Pitcher', 'Team', 'Game Date', 'Kind', 'Tagged',
                        'Suggested', 'Tier', 'Velocity', 'SpinRate', 'RTilt',
                        'OTilt', 'IVB', 'HB', 'RelPosZ', 'RelPosX', 'ArmAngle',
                        'ReleaseOutlierZ', 'Confidence', 'Margin', 'DistOwn',
                        'DistBest', 'MetricsAgree', 'SameGameFlips',
                        'TypePitchesInGame', 'Why', 'PitchID'])
            for f in sorted(keep, key=lambda f: (f['p'].get('PTeam') or '',
                                                 f['p']['Pitcher'],
                                                 f['p']['Game Date'], -f['conf'])):
                p = f['p']
                w.writerow([p['Pitcher'], p.get('PTeam'), p['Game Date'], f['kind'],
                            f['own'], f['tgt'], f['tier'],
                            p.get('Velocity'), p.get('Spin Rate'), p.get('RTilt'),
                            p.get('OTilt'), p.get('IndVertBrk'), p.get('HorzBrk'),
                            p.get('RelPosZ'), p.get('RelPosX'), p.get('ArmAngle'),
                            '' if f['rel'] is None else round(f['rel'], 2),
                            f['conf'],
                            '' if f['margin'] is None else round(f['margin'], 2),
                            '' if f['d_own'] is None else round(f['d_own'], 2),
                            round(f['d_best'], 2),
                            '' if f['tot'] is None else f"{f['agree']}/{f['tot']}",
                            f.get('nflip'), f.get('ntype'), f['why'],
                            p.get('PitchID')])
        print(f"\nWrote {OUT_CSV}  ({len(keep)} rows, tier {'/'.join(sorted(TIERS))})")

    if '--plots' in sys.argv:
        os.makedirs(PLOTDIR, exist_ok=True)
        pages = defaultdict(list)
        for f in keep:
            pages[(f['p'].get('PTeam'), f['p']['Pitcher'],
                   f['p']['Game Date'], f['pop'])].append(f)
        made = 0
        for (team, pit, dt, pop), fs in sorted(pages.items(),
                                               key=lambda kv: (kv[0][0] or '', kv[0][1])):
            if R.make_plot(f'{pit}  [{team}]', dt, fs, groups_all[pop],
                           scales_by_pop[pop], PLOTDIR):
                made += 1
        print(f"Wrote {made} plots to {PLOTDIR}")


if __name__ == '__main__':
    main()
