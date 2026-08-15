"""diet_prototype_2026.py — standalone prototype of the pitch-diet (fear)
metric on 2026. NOT wired to the site. Writes a CSV to ~/Downloads.

diet_swrv = mean league swing-cell RV of the pitches a hitter is thrown
(SD+ table, shipped config). Lower = pitchers avoid giving him anything
hittable = feared. fearPctl = percentile of -diet_swrv among MLB hitters
with >= 200 eligible pitches (all-comers pool, render gate separate, per
house percentile convention).

Usage: python3 scripts/research/hitter/diet_prototype_2026.py
"""
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline.sdplus as sd
import hitter_phase2_multiseason as H

MIN_ELIG = 200
LG, SCALE = 0.3172, 1.2343


def main():
    P = H.load_season(2026)
    elig = H.precompute(P)
    with H.patched('_z16', True):
        offsets = sd.build_bip_count_offsets(elig, LG, SCALE)
        rv_fn = sd.make_rv_xrv(LG, SCALE, offsets)
        raw = sd.build_weight_table(elig, rv_fn)
        zm = sd.zone_level_means(elig, rv_fn)
        table = sd.shrink_table(raw, zm)
        by_h = defaultdict(list)
        for p in elig:
            h = p.get('Batter')
            if h:
                by_h[h].append(p)
        rows = []
        for h, ps in by_h.items():
            if len(ps) < MIN_ELIG:
                continue
            v = [table[(sd.classify_zone(p), sd.get_count(p), sd.cat_of(p),
                        'swing')][0] for p in ps]
            rows.append((h, len(ps), sum(v) / len(v)))

    rows.sort(key=lambda r: r[2])            # most feared first
    n = len(rows)
    out = os.path.expanduser('~/Downloads/diet_fear_prototype_2026.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Hitter', 'EligPitches', 'diet_swrv', 'fearPctl'])
        for i, (h, np_, v) in enumerate(rows):
            pctl = round(100.0 * (n - 1 - i) / (n - 1), 1) if n > 1 else 50.0
            w.writerow([h, np_, round(v, 5), pctl])
    print(f"wrote {out} ({n} hitters)")
    print("\nMOST FEARED (lowest diet value):")
    for h, np_, v in rows[:12]:
        print(f"  {h:<24s} n={np_:<5d} diet_swrv={v:+.4f}")
    print("\nLEAST FEARED (most hittable diet):")
    for h, np_, v in rows[-12:]:
        print(f"  {h:<24s} n={np_:<5d} diet_swrv={v:+.4f}")


if __name__ == '__main__':
    main()
