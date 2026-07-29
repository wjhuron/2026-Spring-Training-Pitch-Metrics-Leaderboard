"""tracking_diff_report.py — genuine Statcast revisions in the un-backfilled
tracking fields (Velocity, Spin Rate, Extension), plus the deleted-spin analysis.

Uses auto-ball-aware matching (renumber Savant excluding automatic events) so the
numbering offset does NOT masquerade as a revision. Read-only.

Part 1: velocity / spin / extension revisions (your value vs current Savant).
Part 2: pitches where you DELETED the spin (blank) but Savant now has one — flagged
        usable (near the pitcher's pitch-type average) or still implausible.

Usage: python3 scripts/tracking_diff_report.py
"""
import os, pickle
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def build_aligned_lookup(sc):
    """(game_pk, at_bat, feed_num) -> statcast row, with autos removed + renumber."""
    bypa = defaultdict(list)
    for r in sc.itertuples(index=False):
        try:
            bypa[(int(r.game_pk), int(r.at_bat_number))].append((int(r.pitch_number), r))
        except Exception:
            continue
    look = {}
    for (pk, ab), evs in bypa.items():
        evs.sort(key=lambda t: t[0])
        feed = 0
        for pn, r in evs:
            if 'automatic' in str(r.description or '').lower():
                continue
            feed += 1
            look[(pk, ab, feed)] = r
    return look


def main():
    allp = [p for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
            if p.get('_source') == 'MLB' and p.get('PitchID')]
    sc = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2026_diff.pkl'), 'rb'))
    look = build_aligned_lookup(sc)
    print(f"pickle MLB pitches: {len(allp)}   aligned savant keys: {len(look)}\n")

    # pitcher pitch-type average spin (from YOUR tagged data, spins present)
    spin_sum = defaultdict(float); spin_n = defaultdict(int)
    for p in allp:
        s = sf(p.get('Spin Rate'))
        if s is not None:
            spin_sum[(p.get('Pitcher'), p.get('Pitch Type'))] += s
            spin_n[(p.get('Pitcher'), p.get('Pitch Type'))] += 1
    def type_avg(pitcher, ptype):
        k = (pitcher, ptype)
        return spin_sum[k] / spin_n[k] if spin_n[k] >= 5 else None

    # ---- Part 1: genuine revisions ----
    checks = [('Velocity', 'Velocity', 'release_speed', [0.2, 0.5, 1.0]),
              ('Spin Rate', 'Spin Rate', 'release_spin_rate', [25, 75, 150]),
              ('Extension', 'Extension', 'release_extension', [0.1, 0.25, 0.5])]
    rev = {c[0]: {'n': 0, 'buckets': [0, 0, 0], 'ex': []} for c in checks}
    matched = 0
    deleted_now = []   # (pid, pitcher, ptype, savant_spin, type_avg)
    for p in allp:
        parts = p['PitchID'].split('_')
        if len(parts) != 3:
            continue
        try:
            k = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        sr = look.get(k)
        if sr is None:
            continue
        matched += 1
        for label, pcol, sattr, thr in checks:
            pv = sf(p.get(pcol)); sv = sf(getattr(sr, sattr, None))
            if pv is None or sv is None:
                continue
            d = abs(pv - sv)
            if d > thr[0]:
                rev[label]['n'] += 1
                for bi, t in enumerate(thr):
                    if d > t:
                        rev[label]['buckets'][bi] += 1
                if len(rev[label]['ex']) < 8:
                    rev[label]['ex'].append(f"{p['PitchID']} ({p.get('Pitcher')}, {p.get('Pitch Type')}): yours={pv} savant={sv}")
        # ---- Part 2: deleted spin now present in Savant ----
        if sf(p.get('Spin Rate')) is None:
            ss = sf(getattr(sr, 'release_spin_rate', None))
            if ss is not None:
                deleted_now.append((p['PitchID'], p.get('Pitcher'), p.get('Pitch Type'),
                                    ss, type_avg(p.get('Pitcher'), p.get('Pitch Type'))))

    print(f"matched (aligned): {matched}\n")
    print("=== Part 1: genuine tracking revisions (auto-ball offset excluded) ===")
    for label, pcol, sattr, thr in checks:
        r = rev[label]
        print(f"  {label:10s}: {r['buckets'][0]:5d} > {thr[0]}   {r['buckets'][1]:4d} > {thr[1]}   {r['buckets'][2]:4d} > {thr[2]}")
    for label, *_ in [(c[0],) for c in checks]:
        print(f"\n  {label} examples:")
        for e in rev[label]['ex'][:6]:
            print(f"    {e}")

    # ---- Part 2 summary ----
    print(f"\n=== Part 2: deleted spins that Savant now has ({len(deleted_now)} pitches) ===")
    usable = []; still_bad = []; no_ref = []
    for pid, pitcher, ptype, ss, avg in deleted_now:
        if avg is None:
            no_ref.append((pid, pitcher, ptype, ss))
        elif abs(ss - avg) <= max(200, 0.12 * avg):
            usable.append((pid, pitcher, ptype, ss, avg))
        else:
            still_bad.append((pid, pitcher, ptype, ss, avg))
    print(f"  now PLAUSIBLE (within ~12%/200rpm of your {ptype if False else 'pitch-type'} avg) -> re-addable: {len(usable)}")
    print(f"  still IMPLAUSIBLE (outlier vs pitcher avg) -> keep deleted: {len(still_bad)}")
    print(f"  no reference avg (pitcher/type <5 spins): {len(no_ref)}")
    print("\n  usable examples (pid | pitcher | type | savant_spin | your avg):")
    for pid, pitcher, ptype, ss, avg in sorted(usable, key=lambda x: x[1] or '')[:15]:
        print(f"    {pid}  {pitcher:22s} {ptype:3s}  {ss:6.0f}  avg {avg:6.0f}")
    print("\n  still-implausible examples:")
    for pid, pitcher, ptype, ss, avg in sorted(still_bad, key=lambda x: -abs(x[3] - x[4]))[:15]:
        print(f"    {pid}  {pitcher:22s} {ptype:3s}  {ss:6.0f}  avg {avg:6.0f}  (off {ss - avg:+.0f})")


if __name__ == '__main__':
    main()
