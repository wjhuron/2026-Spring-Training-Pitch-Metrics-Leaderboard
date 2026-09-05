"""war_calibration_slope.py — the runs scale of hWAR (2026-09-05).

Part 1 found the shipped deserved rate (hdERA, DH_B .917 fit on ERA over the
30-IP pool) over-disperses actual RA9 among 60-IP arms: slope of actual RA9 on
the rate = .856. WAR multiplies that spread by innings, so a 14% over-
dispersion is 14% too much WAR at the top. Three questions:
  1. innings-weighted LOSO slope of RA9 on z(xw shrunk 250) over all >= 30 IP
     arms (the population WAR sums over), per held-out season
  2. is the attenuation workload-dependent after the 250-PA shrink?
     slope within IP terciles
  3. the SE of a pitcher's deserved rate from per-PA xwOBA sampling noise,
     for an hWAR error bar: sd of per-PA xwOBA-against values (2024 cache)
Usage: python3 scripts/research/era/war_calibration_slope.py
"""
import json, math, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import war_rate_validation as W
from pipeline.eraplus import DH_B, N0_XW, POOL_MIN_OUTS
SEASONS = W.SEASONS

def zfeat(y):
    t = W.season_table(y); lg = W.league(t); pool = [r for r in t.values() if r['outs'] >= POOL_MIN_OUTS and r['xw'] is not None]
    sh = {p: (r['xw'] * r['xw_den'] + N0_XW * lg['xw']) / (r['xw_den'] + N0_XW) for p, r in t.items() if r['xw'] is not None and r['xw_den'] > 0}
    ps = [sh[p] for p, r in t.items() if r['outs'] >= POOL_MIN_OUTS and p in sh]; mu, sd = float(np.mean(ps)), float(np.std(ps))
    anchor = float(np.mean([r['er'] * 27 / r['outs'] for r in pool])); anchor_r = float(np.mean([r['r'] * 27 / r['outs'] for r in pool]))
    return {p: dict(z=(sh[p] - mu) / sd, outs=r['outs'], ra9=r['r'] * 27 / r['outs'], era=r['er'] * 27 / r['outs']) for p, r in t.items() if p in sh and r['outs'] >= POOL_MIN_OUTS}, dict(anchor=anchor, anchor_r=anchor_r, gap=lg['ra9'] - lg['era'])

def wls(x, y, w):
    x, y, w = map(lambda a: np.asarray(a, float), (x, y, w)); mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w * (x - mx) ** 2))

def main():
    Z = {y: zfeat(y) for y in SEASONS}
    print("1. LOSO slope of RA9 on z(xw shrunk 250), >= 30 IP pool: unweighted vs innings-weighted (fit on the other five seasons)")
    for hold in SEASONS:
        x, yv, w = [], [], []
        for y in SEASONS:
            if y == hold: continue
            F, M = Z[y]
            for f in F.values(): x.append(f['z']); yv.append(f['ra9'] - M['anchor_r']); w.append(f['outs'])
        b_u = np.polyfit(x, yv, 1)[0]; b_w = wls(x, yv, w)
        # apply to the held-out season: calibration slope of actual RA9 on the rate
        F, M = Z[hold]; xs = [f['z'] for f in F.values()]; ys = [f['ra9'] for f in F.values()]; ws = [f['outs'] for f in F.values()]
        cal_ship = wls([M['anchor'] + DH_B * z + M['gap'] for z in xs], ys, ws); cal_w = wls([M['anchor_r'] + b_w * z for z in xs], ys, ws)
        print(f"  hold {hold}: b unweighted {b_u:.3f}  b innings-weighted {b_w:.3f}   held-out calibration (IP-weighted slope of RA9 on rate): shipped DH_B {cal_ship:.3f}  refit {cal_w:.3f}")
    print("\n2. workload dependence of the shipped rate's calibration (pooled seasons, IP-weighted slope of RA9 on the shipped rate within tercile):")
    allr = [(M['anchor'] + DH_B * f['z'] + M['gap'], f['ra9'], f['outs']) for y in SEASONS for f, M in ((f, Z[y][1]) for f in Z[y][0].values())]
    o = np.array([r[2] for r in allr]); q1, q2 = np.percentile(o, [33, 67])
    for lo, hi, lab in ((0, q1, 'low IP'), (q1, q2, 'mid IP'), (q2, 1e9, 'high IP')):
        sub = [r for r in allr if lo <= r[2] < hi]
        print(f"  {lab:8} n {len(sub):4d}  IP {np.mean([r[2] for r in sub]) / 3:5.0f}  slope {wls([r[0] for r in sub], [r[1] for r in sub], [r[2] for r in sub]):.3f}")
    print("\n3. per-PA xwOBA-against noise (2024 cache), for the hWAR error bar")
    df = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2024_cache.pkl'), 'rb'))
    d = df[df['events'].notna() & ~df['events'].isin(['intent_walk', 'sac_bunt', 'catcher_interf'])]
    vals = np.where(d['events'].isin(['walk']), 0.69, np.where(d['events'].isin(['hit_by_pitch']), 0.72, d['estimated_woba_using_speedangle'].fillna(0.0)))
    sd_pa = float(np.std(vals)); scale = 1.242; pa9 = 38.1; rpw = 9.5
    print(f"  sd of per-PA xwOBA value {sd_pa:.3f} (n {len(vals)}); SE of the deserved rate per 9 = {sd_pa:.3f}/sqrt(PA) / scale * {pa9} => 150 PA {sd_pa / math.sqrt(150) / scale * pa9:.2f}, 400 PA {sd_pa / math.sqrt(400) / scale * pa9:.2f}, 800 PA {sd_pa / math.sqrt(800) / scale * pa9:.2f} runs/9")
    for pa in (150, 400, 800):
        se_r9 = sd_pa / math.sqrt(pa) / scale * pa9; ip = pa / pa9 * 9; se_war = se_r9 * ip / 9 / rpw
        print(f"  PA {pa:4d} (~{ip:4.0f} IP): SE hWAR +/- {se_war:.2f} (unshrunk); the 250-PA shrink scales it by {pa / (pa + 250):.2f} -> +/- {se_war * pa / (pa + 250):.2f}")

if __name__ == '__main__':
    main()
