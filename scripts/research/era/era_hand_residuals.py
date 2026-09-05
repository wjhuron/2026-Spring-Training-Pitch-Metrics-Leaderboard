"""era_hand_residuals.py — does hpERA carry a pitcher-hand bias? (2026-09-05)

On the rebuilt ERA replicates (era_targets_build / era_battery_build /
era_cmd_loc_scores / era_xrv100_pass / era_stuff_loso_scores), refit the
production hpERA channel set LOSO exactly as era_weights_final does, then:
  1. held-out residual (actual ERA - forecast) by pitcher hand, per replicate
     and paired across replicates, for NEXT (Y -> Y+1) and ROS (h1 -> h2);
  2. the same for the xwOBA-only (hdERA-style) model, for contrast;
  3. a LHP indicator as a ninth channel: its fitted weight (ERA units) per
     replicate and the paired held-out r change;
  4. the per-season LHP-minus-RHP mean z of the stuff and loc channels (the
     cancellation seen on 2026).
Usage: python3 scripts/research/era/era_hand_residuals.py
"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, HERE)
import era_weights_final as W
from era_estimator_screen import pearson, targets_for, SEASONS
from era_combo_preview import ols

FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']   # production hpERA set
HAND = {sk: {pid: rec.get('hand') for pid, rec in v['pitchers'].items()} for sk, v in W.TARGETS.items()}

def reps_with_pid(test, gate):
    reps = []
    if test == 'next':
        for season in SEASONS[:-1]:
            fr = W.shrunk_features(season, 'full'); tc = targets_for(season, 'full'); tn = targets_for(season + 1, 'full')
            units = [(str(season), pid, fr[pid], tn[pid]['era']) for pid in fr
                     if pid in tn and tc[pid]['outs'] >= gate * 3 and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
    else:
        for season in SEASONS:
            fr = W.shrunk_features(season, 'h1'); t1 = targets_for(season, 'h1'); t2 = targets_for(season, 'h2')
            hg = max(gate * 3 // 2, 45)
            units = [(str(season), pid, fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and t1[pid]['outs'] >= hg and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
    return reps

def mean_se(v):
    n = len(v); m = sum(v) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n) if n > 1 else float('nan'))

def loso_hand(reps, feats, tag):
    """Per replicate: held-out r, residual mean by hand, L-R diff + SE."""
    print(f"\n--- {tag}: feats {feats}")
    print(f"{'rep':12} {'n R/L':>9} {'r':>7} {'resid RHP':>10} {'resid LHP':>10} {'L-R':>7} {'SE':>6}" + ("  beta_lhp" if 'lhp' in feats else ''))
    diffs, rs, betas = [], [], []
    for i, (label, test) in enumerate(reps):
        train = [(x, y) for j, (_, us) in enumerate(reps) if j != i for _, _, x, y in us]
        beta = ols(train, feats)
        if beta is None:
            print(f"{label:12} insufficient"); continue
        res = {'L': [], 'R': []}; preds, ys = [], []
        for sk, pid, x, y in test:
            if not all(f in x for f in feats): continue
            p = beta[0] + sum(b * x[f] for b, f in zip(beta[1:], feats))
            preds.append(p); ys.append(y)
            h = HAND[sk].get(pid)
            if h in res: res[h].append(y - p)
        r = pearson(preds, ys); rs.append(r)
        mr, ser = mean_se(res['R']); ml, sel = mean_se(res['L'])
        d = ml - mr; se = math.sqrt(ser ** 2 + sel ** 2); diffs.append(d)
        extra = f"  {beta[1 + feats.index('lhp')]:+.3f}" if 'lhp' in feats else ''
        if 'lhp' in feats: betas.append(beta[1 + feats.index('lhp')])
        print(f"{label:12} {len(res['R']):4d}/{len(res['L']):<4d} {r:7.4f} {mr:+10.3f} {ml:+10.3f} {d:+7.3f} {se:6.3f}{extra}")
    md, sd_ = mean_se(diffs); mr_, _ = mean_se(rs)
    line = f"  paired: L-R residual mean {md:+.3f} (SE {sd_:.3f}, t {md / sd_ if sd_ > 0 else float('nan'):+.1f}, LHP better in {sum(1 for d in diffs if d < 0)}/{len(diffs)}); mean held-out r {mr_:.4f}"
    if betas:
        mb, sb = mean_se(betas); line += f"; beta_lhp mean {mb:+.3f} (SE {sb:.3f})"
    print(line)
    return diffs, rs

def main():
    # 4. channel levels by hand, per season (full scope, 30 IP pool)
    print("LHP - RHP mean z by season (full scope, 30+ IP pool): stuff channel, loc channel, their W_PH-weighted ERA sum (0.297 / 0.136)")
    for season in SEASONS:
        z = W.shrunk_features(season, 'full'); sk = str(season)
        acc = {c: {'L': [], 'R': []} for c in ('stuff', 'loc')}
        for pid, f in z.items():
            h = HAND[sk].get(pid)
            if h not in ('L', 'R'): continue
            for c in acc:
                if c in f: acc[c][h].append(f[c])
        g = {c: (sum(v['L']) / len(v['L']) - sum(v['R']) / len(v['R'])) if v['L'] and v['R'] else float('nan') for c, v in acc.items()}
        print(f"  {season}: stuff {g['stuff']:+.3f}  loc {g['loc']:+.3f}  -> ERA {0.297 * g['stuff'] + 0.136 * g['loc']:+.3f}   (n L {len(acc['stuff']['L'])}, R {len(acc['stuff']['R'])})")
    for test in ('next', 'ros'):
        for gate in (60, 30):
            reps = reps_with_pid(test, gate)
            # add the LHP indicator to every unit's feature dict (copy)
            reps_h = [(lab, [(sk, pid, dict(x, lhp=1.0 if HAND[sk].get(pid) == 'L' else 0.0), y) for sk, pid, x, y in us]) for lab, us in reps]
            tag = f"{test.upper()} gate {gate} IP"
            d0, r0 = loso_hand(reps, FEATS, tag + " | production hpERA set")
            loso_hand(reps, ['xw'], tag + " | xwOBA only (hdERA-style)")
            d1, r1 = loso_hand(reps_h, FEATS + ['lhp'], tag + " | + LHP indicator")
            if r0 and r1 and len(r0) == len(r1):
                dr = [b - a for a, b in zip(r0, r1)]; m, se = mean_se(dr)
                print(f"  => LHP indicator: held-out r change {m:+.4f} (SE {se:.4f}), wins {sum(1 for x in dr if x > 0)}/{len(dr)}")

if __name__ == '__main__':
    main()
