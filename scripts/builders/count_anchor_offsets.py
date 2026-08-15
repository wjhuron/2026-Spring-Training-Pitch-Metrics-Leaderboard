"""count_anchor_offsets.py — measure the BIP count-anchoring offset table.

The shared fix for SD+/CT+/Loc+/Stuff+: takes/whiffs/fouls/CS are valued in
count-conditional delta run expectancy (-RunExp), but balls in play are valued
as (xwOBA - lg)/scale anchored to a neutral PA state. This script measures the
per-count offset that aligns the two currencies:

    offset(c) = mean(-RunExp | BIP in count c) - mean((xwOBA - lg)/scale | BIP in count c)

-RunExp on a BIP is the actual count-conditional value of ending the PA from
count c with that outcome; the xwOBA branch is the luck-neutral value on a
neutral anchor. Their per-count means differ by exactly the count-state
correction (outcome luck averages out within a count at league scale).

Usage: python3 scripts/builders/count_anchor_offsets.py
"""
import os, pickle, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_count(p):
    c = p.get('Count') or ''
    if '-' in str(c):
        try:
            b, s = str(c).split('-')
            return (int(b), min(int(s), 2))
        except ValueError:
            return None
    return None


def main():
    D = pickle.load(open(PKL, 'rb'))
    acc = defaultdict(lambda: {'re': [0.0, 0], 'xw': [0.0, 0]})
    for p in D:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        if p.get('Description') != 'In Play':
            continue
        c = get_count(p)
        if c is None:
            continue
        re = sf(p.get('RunExp'))
        xw = sf(p.get('xwOBA'))
        a = acc[c]
        if re is not None:
            a['re'][0] += -re
            a['re'][1] += 1
        if xw is not None:
            a['xw'][0] += (xw - LG_WOBA) / WOBA_SCALE
            a['xw'][1] += 1

    print(f"{'count':>6s} {'nBIP':>7s} {'mean(-RunExp)':>14s} {'mean(xw_rv)':>12s} {'offset':>8s}")
    offsets = {}
    for c in sorted(acc):
        a = acc[c]
        if a['re'][1] < 50 or a['xw'][1] < 50:
            continue
        m_re = a['re'][0] / a['re'][1]
        m_xw = a['xw'][0] / a['xw'][1]
        off = m_re - m_xw
        offsets[c] = off
        print(f"{c[0]}-{c[1]:>4d} {a['re'][1]:7d} {m_re:14.4f} {m_xw:12.4f} {off:8.4f}")

    if offsets:
        span = max(offsets.values()) - min(offsets.values())
        print(f"\noffset span across counts: {span:.4f} runs "
              f"(the currency mismatch a neutral-anchored BIP value ignores)")


if __name__ == '__main__':
    main()
