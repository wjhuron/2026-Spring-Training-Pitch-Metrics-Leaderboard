"""aaa_translation_shrunk.py — refit the AAA-to-MLB translation on the
values the pipeline actually feeds hpERA, then test it end to end on hdERA.

WHY REFIT. aaa_translation_fit.py fits MLB_c = a + b * AAA_c on RAW season
values. hpERA does not see raw values: pipeline/eraplus.py shrinks every
channel toward the MLB league rate at its own measured n0 BEFORE z-scoring.
A fitted b below 1 and that shrinkage both pull the same number toward the
mean, so shipping both would regress a Triple-A line to league twice. Here
both sides are shrunk with the SHIPPED constants first, so the fitted slope
is the residual translation on top of the regularization that already runs.

The shrink target is the MLB league rate in both cases. That is deliberate,
and it means the shrinkage is itself already part of the translation: a thin
Triple-A line is pulled toward the MLB average before anything else
happens, and a and b only correct what survives.

END-TO-END TEST ON hdERA. hdERA is a single channel,

    hdERA = anchor + DH_B * z(shrunk xwOBA against)

so it can be evaluated completely with what this corpus holds, and it is
the worst offender: untranslated, Rochester's arms read about 1.5 runs too
good. The test scores a pitcher's Triple-A season into an hdERA and
compares it to the ERA he ACTUALLY posted in the majors that same season,
on held-out seasons, against three references (none / intercept / slope).

hpERA cannot be tested the same way yet. Its heaviest channel is Stuff+
at weight .297 and adapt_statcast leaves StuffPlus None, so AAA Stuff+
needs the model pass. Dropping it and renormalizing would raise the
translated channels from 46% of the weight to 100%, which would OVERSTATE
how much the translation moves the composite. That test waits for AAA
Stuff+ rather than being run on a flattering subset.

League anchors, league rates and the z-pool are rebuilt PER SEASON from
the MLB battery over that season's 30+ IP population, mirroring
apply_era_plus. DH_B stays at the shipped 0.917.

    python3 scripts/research/era/aaa_translation_shrunk.py
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

from pipeline.eraplus import (N0_XW, N0_K, N0_IZWH, N0_GB, N0_XRV,
                              IZSW_PER_PITCH, DH_B, POOL_MIN_OUTS)

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
SEASONS = ('2023', '2024', '2025', '2026')


def shrink(v, n, n0, lg):
    if v is None or n is None:
        return None
    return (v * n + n0 * lg) / (n + n0)


def league(season, mlb_bat, targets):
    """Per-season MLB league rates + the ERA anchor, over the 30+ IP pool —
    the same population apply_era_plus uses."""
    B = mlb_bat.get(season) or {}
    T = (targets.get(season) or {}).get('pitchers') or {}
    pool = []
    eras = []
    for pid, rec in B.items():
        t = T.get(pid)
        if not t or (t.get('outs') or 0) < POOL_MIN_OUTS:
            continue
        f = rec.get('full')
        if not f:
            continue
        pool.append(f)
        eras.append(t['er'] * 27.0 / t['outs'])
    if len(pool) < 50:
        return None
    def wmean(vk, nk, default):
        num = sum((p[vk]) * (p.get(nk) or 0) for p in pool
                  if p.get(vk) is not None)
        den = sum((p.get(nk) or 0) for p in pool if p.get(vk) is not None)
        return num / den if den else default
    lg = {
        'xw': wmean('xwoba', 'pa', 0.311),
        'k': wmean('k_pct', 'pa', 0.226),
        'gb': wmean('gb_pct', 'bip', 0.421),
    }
    # izWhiff = 1 - zcon_pct, weighted by pitches (the denominator eraplus
    # uses via IZSW_PER_PITCH)
    num = sum((1.0 - p['zcon_pct']) * (p.get('pitches') or 0) for p in pool
              if p.get('zcon_pct') is not None)
    den = sum((p.get('pitches') or 0) for p in pool
              if p.get('zcon_pct') is not None)
    lg['izwh'] = num / den if den else 0.167
    lg['anchor'] = sum(eras) / len(eras)
    lg['n_pool'] = len(pool)
    return lg


def shrunk_xw(rec_full, lg_xw):
    return shrink(rec_full.get('xwoba'), rec_full.get('pa'), N0_XW, lg_xw)


def wls(pairs):
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
    return my - b * mx, b


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
    return sum((a - mx) * (b - my)
               for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


CH = ('xw', 'k', 'izwh', 'gb')


def build(floor):
    """Per season: shrunk AAA and shrunk MLB channel values for pitchers at
    both levels, plus the pitcher's actual MLB ERA and outs."""
    aaa = D('_aaa_battery.json')
    mlb = D('_era_battery.json')
    tg = D('_era_targets.json')
    out = {}
    lgs = {}
    for s in SEASONS:
        lg = league(s, mlb, tg)
        if lg is None:
            print(f'  {s}: MLB pool too thin, skipped')
            continue
        lgs[s] = lg
        A = aaa.get(s) or {}
        B = mlb.get(s) or {}
        T = (tg.get(s) or {}).get('pitchers') or {}
        rows = []
        for pid, arec in A.items():
            af = arec.get('battery')
            mrec = (B.get(pid) or {}).get('full')
            t = T.get(pid)
            if not af or not mrec or not t or not t.get('outs'):
                continue
            if (af.get('pitches') or 0) < floor or (mrec.get('pitches') or 0) < floor:
                continue
            r = {'pid': pid, 'era_mlb': t['er'] * 27.0 / t['outs'],
                 'outs': t['outs'], 'w': 2.0 * af['pitches'] * mrec['pitches']
                 / (af['pitches'] + mrec['pitches'])}
            ok = True
            for c, vk, nk, n0 in (('xw', 'xwoba', 'pa', N0_XW),
                                  ('k', 'k_pct', 'pa', N0_K),
                                  ('gb', 'gb_pct', 'bip', N0_GB)):
                a_ = shrink(af.get(vk), af.get(nk), n0, lg[c])
                m_ = shrink(mrec.get(vk), mrec.get(nk), n0, lg[c])
                if a_ is None or m_ is None:
                    ok = False
                    break
                r['a_' + c], r['m_' + c] = a_, m_
            if not ok:
                continue
            for tag, src in (('a', af), ('m', mrec)):
                if src.get('zcon_pct') is None:
                    ok = False
                    break
                niz = (src.get('pitches') or 0) * IZSW_PER_PITCH
                r[tag + '_izwh'] = shrink(1.0 - src['zcon_pct'], niz,
                                          N0_IZWH, lg['izwh'])
            if not ok:
                continue
            rows.append(r)
        out[s] = rows
    return out, lgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=int, default=200)
    a = ap.parse_args()
    data, lgs = build(a.floor)
    print(f'== shrunk pairs, floor {a.floor} pitches on both sides')
    for s in SEASONS:
        if s in data:
            print(f'   {s}: {len(data[s]):4d} pitchers | MLB pool '
                  f"{lgs[s]['n_pool']:3d} | anchor ERA {lgs[s]['anchor']:.3f}"
                  f" | lg xwOBA {lgs[s]['xw']:.4f}")
    print()

    print('== full-corpus fit on SHRUNK values, MLB = a + b * AAA')
    print(f"{'channel':8s} {'n':>5s} {'a':>10s} {'b':>7s} {'wtd delta':>11s}")
    for c in CH:
        pr = [(r['a_' + c], r['m_' + c], r['w'])
              for s in data for r in data[s]]
        f = wls(pr)
        if f is None:
            continue
        sw = sum(w for _, _, w in pr)
        d = sum(w * (y - x) for x, y, w in pr) / sw
        print(f'{c:8s} {len(pr):5d} {f[0]:10.5f} {f[1]:7.4f} {d:11.5f}')

    print('\n== hdERA end to end: Triple-A season -> hdERA vs the ERA he '
          'actually posted in MLB that season')
    print(f"{'held out':>9s} {'n':>5s} | {'bias none':>10s} {'bias icpt':>10s}"
          f" {'bias slope':>11s} | {'rmse none':>10s} {'rmse icpt':>10s}"
          f" {'rmse slope':>11s} | {'rank':>6s}  winner")
    wins = defaultdict(int)
    for held in SEASONS:
        if held not in data or len(data[held]) < 20:
            continue
        tr = [(r['a_xw'], r['m_xw'], r['w'])
              for s in data if s != held for r in data[s]]
        f = wls(tr)
        if f is None:
            continue
        a_, b_ = f
        sw_tr = sum(w for _, _, w in tr)
        d_tr = sum(w * (y - x) for x, y, w in tr) / sw_tr
        lg = lgs[held]
        # z-pool for the held-out season: shrunk MLB xwOBA over its own pool
        pool = [r['m_xw'] for s in data for r in data[s]] if False else None
        mlb_bat = D('_era_battery.json')[held]
        tgs = D('_era_targets.json')[held]['pitchers']
        zs = []
        for pid, rec in mlb_bat.items():
            t = tgs.get(pid)
            if not t or (t.get('outs') or 0) < POOL_MIN_OUTS:
                continue
            v = shrunk_xw(rec.get('full') or {}, lg['xw'])
            if v is not None:
                zs.append(v)
        mu = sum(zs) / len(zs)
        sd = (sum((x - mu) ** 2 for x in zs) / len(zs)) ** 0.5
        rows = data[held]
        def hd(x):
            return lg['anchor'] + DH_B * (x - mu) / sd
        act = [r['era_mlb'] for r in rows]
        wts = [r['outs'] for r in rows]          # weight the ERA side by IP
        variants = {
            'none':      [hd(r['a_xw']) for r in rows],
            'intercept': [hd(r['a_xw'] + d_tr) for r in rows],
            'slope':     [hd(a_ + b_ * r['a_xw']) for r in rows],
        }
        sw = sum(wts)
        res = {}
        for k, pred in variants.items():
            bias = sum(w * (p - y) for p, y, w in zip(pred, act, wts)) / sw
            rmse = (sum(w * (p - y) ** 2
                        for p, y, w in zip(pred, act, wts)) / sw) ** 0.5
            res[k] = (bias, rmse)
        rk = spearman(variants['none'], act)
        best = min(res, key=lambda k: abs(res[k][0]))
        wins[best] += 1
        print(f'{held:>9s} {len(rows):5d} | {res["none"][0]:10.3f} '
              f'{res["intercept"][0]:10.3f} {res["slope"][0]:11.3f} | '
              f'{res["none"][1]:10.3f} {res["intercept"][1]:10.3f} '
              f'{res["slope"][1]:11.3f} | {rk:6.3f}  {best}')
    print(f'\n   bias winner count: '
          + '  '.join(f'{k}={v}' for k, v in sorted(wins.items())))
    print('   rank is identical across variants by construction '
          '(hdERA is one channel and a + b*x is monotone); it is shown as '
          'the ceiling a single-channel translation can reach.')


if __name__ == '__main__':
    main()
