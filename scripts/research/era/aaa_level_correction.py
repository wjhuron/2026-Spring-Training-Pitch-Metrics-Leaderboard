"""aaa_level_correction.py — the level correction for every MLB-scaled
pitcher metric, measured on paired pitcher-seasons 2023-2025.

THE QUESTION. Pitcher+, hdERA and hpERA are all scored on an MLB ruler and
all populate (or would populate) for Triple-A rows using untranslated
channels. How far off is each one, in its own units?

THE ESTIMATOR is the within-pitcher paired shift, not the group-mean gap.
The group gap between Rochester and MLB mixes the level effect with the
fact that a Triple-A staff is genuinely worse; shifting AAA to match the
MLB mean would assert the two populations are equally good. Only the same
arm measured at both levels in the same season separates them.

STUFF+ USES THE INSTRUMENT-CORRECTED AAA VALUE. The cross-level Stuff+ gap
partitions into altitude 11%, release-point geometry 20%, velocity 45% and
an unexplained 24%. The geometry piece is two Hawk-Eye installations
disagreeing about where release happens (extension off 1.2 inches, t 28.7,
coupled with release height in the geometrically consistent direction), so
it is corrected. Velocity is NOT corrected: 0.31 mph within-pitcher is a
real difference in what the pitcher did, and a Triple-A row should show
what he did. Correct measurement, not performance.

ROLE AND PARK ARE EXCLUDED from the hpERA shift. gs/g and the home park
are the two hpERA channels this corpus cannot supply for Triple-A, and
both are usage/context rather than level: a pitcher who starts at AAA and
relieves in MLB SHOULD score differently, and that is not a translation.
The reported shift is therefore the six-channel shift, and the two
excluded weights (.277 and .168 of 1.384) are named rather than
renormalized away.

    python3 scripts/research/era/aaa_level_correction.py
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from pipeline.eraplus import (N0_XW, N0_K, N0_IZWH, N0_GB, N0_XRV,
                              IZSW_PER_PITCH, DH_B, W_PH, POOL_MIN_OUTS)
from pipeline.pitcherplus import COMPONENTS, SCALE_K

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
Y = ('2023', '2024', '2025')
PP_EXCLUDED = ('gs', 'park')


def shrink(v, n, n0, lg):
    return None if v is None or n is None else (v * n + n0 * lg) / (n + n0)


def zstats(vals):
    m = sum(vals) / len(vals)
    return m, (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=int, default=200)
    a = ap.parse_args()
    aaa = D('_aaa_battery.json')
    mlb = D('_era_battery.json')
    loc_m = D('_era_internal_cmdloc.json')
    xrv_m = D('_era_xrv100.json')
    tg = D('_era_targets.json')
    st_aaa = D('_aaa_stuff_geomfix.json')['AAA']    # instrument-corrected
    st_mlb = D('_aaa_stuff_lo3000.json')['MLB']     # same elevation filter

    rows_by_season = {}
    for s in Y:
        B, T = mlb.get(s) or {}, (tg.get(s) or {}).get('pitchers') or {}
        A = aaa.get(s) or {}
        # MLB pool = the 30+ IP population, the one apply_era_plus uses
        pool = [(pid, B[pid]['full']) for pid in B
                if B[pid].get('full') and (T.get(pid, {}).get('outs') or 0)
                >= POOL_MIN_OUTS]
        if len(pool) < 50:
            continue

        def wmean(vk, nk, d):
            num = sum(f[vk] * (f.get(nk) or 0) for _, f in pool
                      if f.get(vk) is not None)
            den = sum((f.get(nk) or 0) for _, f in pool if f.get(vk) is not None)
            return num / den if den else d
        lg = {'xw': wmean('xwoba', 'pa', .311), 'k': wmean('k_pct', 'pa', .226),
              'gb': wmean('gb_pct', 'bip', .421)}
        nz = sum((f.get('pitches') or 0) for _, f in pool
                 if f.get('zcon_pct') is not None)
        lg['izwh'] = (sum((1 - f['zcon_pct']) * (f.get('pitches') or 0)
                          for _, f in pool if f.get('zcon_pct') is not None)
                      / nz) if nz else .167
        xs = [(xrv_m[s][pid]['full'], xrv_m[s][pid]['n_full'])
              for pid, _ in pool if pid in (xrv_m.get(s) or {})]
        lg['xrv'] = (sum(v * n for v, n in xs) / sum(n for _, n in xs)
                     if xs else 0.0)
        anchor = sum(T[pid]['er'] * 27.0 / T[pid]['outs'] for pid, _ in pool
                     if T.get(pid, {}).get('outs')) / len(pool)

        def channels(pid, src, f, loc_v, loc_n, xrv_v, xrv_n, stf):
            n_p = f.get('pitches') or 0
            c = {}
            c['xw'] = shrink(f.get('xwoba'), f.get('pa'), N0_XW, lg['xw'])
            c['k'] = shrink(f.get('k_pct'), f.get('pa'), N0_K, lg['k'])
            c['gb'] = shrink(f.get('gb_pct'), f.get('bip'), N0_GB, lg['gb'])
            c['izwh'] = (shrink(1 - f['zcon_pct'], n_p * IZSW_PER_PITCH,
                                N0_IZWH, lg['izwh'])
                         if f.get('zcon_pct') is not None else None)
            c['xrv'] = shrink(xrv_v, xrv_n, N0_XRV, lg['xrv'])
            c['loc'] = loc_v
            c['stuff'] = stf
            c['n'] = n_p
            c['loc_n'] = loc_n
            c['xrv_n'] = xrv_n
            return c

        # MLB baseline distribution for every channel, over the same pool
        base = {}
        for pid, f in pool:
            lr = (loc_m.get(s) or {}).get(pid) or {}
            xr = (xrv_m.get(s) or {}).get(pid) or {}
            sf = st_mlb.get(s, {}).get(pid)
            base[pid] = channels(pid, 'MLB', f, lr.get('loc_full'),
                                 lr.get('loc_n_full'), xr.get('full'),
                                 xr.get('n_full'),
                                 sf['v'] if sf else None)
        stats = {}
        for ch in ('xw', 'k', 'gb', 'izwh', 'xrv', 'loc', 'stuff'):
            v = [b[ch] for b in base.values() if b.get(ch) is not None]
            stats[ch] = zstats(v) if len(v) >= 30 else None
        pairs = []
        for pid, arec in A.items():
            af = arec.get('battery')
            mf = (B.get(pid) or {}).get('full')
            if not af or not mf:
                continue
            if (af.get('pitches') or 0) < a.floor or (mf.get('pitches') or 0) < a.floor:
                continue
            sa = st_aaa.get(s, {}).get(pid)
            sm = st_mlb.get(s, {}).get(pid)
            if not sa or not sm:
                continue
            lr = (loc_m.get(s) or {}).get(pid) or {}
            xr = (xrv_m.get(s) or {}).get(pid) or {}
            ac = channels(pid, 'AAA', af, (arec.get('loc') or {}).get('v'),
                          (arec.get('loc') or {}).get('n'),
                          (arec.get('xrv') or {}).get('v'),
                          (arec.get('xrv') or {}).get('n'), sa['v'])
            mc = channels(pid, 'MLB', mf, lr.get('loc_full'),
                          lr.get('loc_n_full'), xr.get('full'),
                          xr.get('n_full'), sm['v'])
            pairs.append((pid, ac, mc))
        rows_by_season[s] = (pairs, stats, anchor)

    def z(ch, v, stats):
        stt = stats.get(ch)
        return None if v is None or not stt or stt[1] <= 0 else (v - stt[0]) / stt[1]

    print(f'== paired pitcher-seasons, floor {a.floor} pitches both sides')
    out = defaultdict(list)
    CONTRIB = defaultdict(list)
    for s in Y:
        if s not in rows_by_season:
            continue
        pairs, stats, anchor = rows_by_season[s]
        dh, dp, dpp = [], [], []
        for pid, ac, mc in pairs:
            # hdERA: single xwOBA channel, ERA units
            za, zm = z('xw', ac['xw'], stats), z('xw', mc['xw'], stats)
            if za is not None and zm is not None:
                dh.append(DH_B * (zm - za))
            # hpERA: six available channels, ERA direction
            tot = 0.0
            ok = True
            for ch, sgn in (('k', -1), ('izwh', -1), ('gb', -1),
                            ('xrv', 1), ('loc', 1), ('stuff', -1)):
                zaa, zmm = z(ch, ac[ch], stats), z(ch, mc[ch], stats)
                if zaa is None or zmm is None:
                    ok = False
                    break
                tot += W_PH[ch] * sgn * (zmm - zaa)
            if ok:
                dp.append(tot)
                for ch, sgn in (('k', -1), ('izwh', -1), ('gb', -1),
                                ('xrv', 1), ('loc', 1), ('stuff', -1)):
                    CONTRIB[ch].append(
                        W_PH[ch] * sgn * (z(ch, mc[ch], stats)
                                          - z(ch, ac[ch], stats)))
            # Pitcher+: its own six components, higher = better
            ca = cm = 0.0
            ok = True
            for name, w, k in COMPONENTS:
                ch = {'stuffScore': 'stuff', 'locPlus': 'loc', 'kPct': 'k',
                      'izWhiffPct': 'izwh', 'xRv100': 'xrv',
                      'gbPct': 'gb'}[name]
                sgn = -1 if ch in ('loc', 'xrv') else 1
                zaa, zmm = z(ch, ac[ch], stats), z(ch, mc[ch], stats)
                if zaa is None or zmm is None:
                    ok = False
                    break
                na, nm = ac['n'], mc['n']
                ca += w * sgn * zaa * na / (na + k)
                cm += w * sgn * zmm * nm / (nm + k)
            if ok:
                dpp.append(SCALE_K * (cm - ca))
        for lab, d in (('hdERA', dh), ('hpERA', dp), ('Pitcher+', dpp)):
            out[lab].append((s, len(d), st.mean(d), st.stdev(d)))
    print()
    print(f"{'metric':10s} {'season':7s} {'n':>5s} {'MLB - AAA':>11s} "
          f"{'sd':>7s} {'t':>7s}")
    for lab in ('hdERA', 'hpERA', 'Pitcher+'):
        allv = []
        for s, n, m, sd in out[lab]:
            se = sd / n ** 0.5
            print(f'{lab:10s} {s:7s} {n:5d} {m:+11.3f} {sd:7.3f} {m/se:+7.2f}')
            allv.append((n, m, sd))
        N = sum(n for n, _, _ in allv)
        M = sum(n * m for n, m, _ in allv) / N
        print(f'{lab:10s} {"POOLED":7s} {N:5d} {M:+11.3f}')
        print()
    print('== why hpERA nearly cancels: each channel\'s contribution to the '
          'hpERA shift, in ERA units')
    tot = 0.0
    for ch in sorted(CONTRIB, key=lambda c: -abs(st.mean(CONTRIB[c]))):
        v = st.mean(CONTRIB[ch])
        tot += v
        print(f'   {ch:6s} {v:+7.3f}   (weight {W_PH[ch]:.3f})')
    print(f'   {"NET":6s} {tot:+7.3f}')
    print()
    print('hdERA and hpERA are in ERA units (add to a Triple-A value to put '
          'it on the MLB scale). Pitcher+ is in Pitcher+ points.')
    print(f'hpERA excludes {PP_EXCLUDED} — usage and context, not level; '
          f'their weights are .277 and .168 of 1.384.')


if __name__ == '__main__':
    main()
