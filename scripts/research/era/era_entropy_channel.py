"""era_entropy_channel.py — does arsenal-mix entropy improve hpERA?

Public prior (MDPI 2015, Brennan 2021): Shannon entropy of pitch
selection proxies batter expectation, with lower sequence complexity
linked to worse run prevention after stuff is controlled. Stuff+ and
Loc+ grade pitches one at a time, so mix diversity is a candidate
orthogonal channel.

Channel: H = -sum p_t * log2(p_t) over pitch-type shares, per
pitcher-scope, from the public Statcast caches (Savant tags; the tags
are noisier than the sheet retags, which biases the channel TOWARD
finding entropy where there is none, so a pass here would still need a
retagged confirmation). Floor 100 pitches per scope; below it the
channel imputes z = 0.

Comparison: production 8-channel fit with and without the entropy z,
LOSO. The caches cover 2021-2025 only, so BOTH arms run on those
replicates (ROS 5 folds, NEXT 4 pairs).

Sidecar cache: data/_pitcher_mix_entropy.json (delete to rebuild).

Usage: PYTHONHASHSEED=0 python3 scripts/research/era/era_entropy_channel.py
Output: console + data/_era_entropy_results.json
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import era_weights_final as wf
from era_estimator_screen import pearson, targets_for
from era_combo_preview import ols

BASE_FEATS = ['stuff', 'loc', 'k', 'izwh', 'xrv', 'gb', 'gs_share', 'park']
SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PITCHES = 100
CACHE = os.path.join(ROOT, 'data', '_pitcher_mix_entropy.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))


def build_entropy():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for y in SEASONS:
        asg = TARGETS[str(y)]['asg']
        df = pickle.load(open(os.path.join(
            ROOT, 'data', f'_statcast{y}_cache.pkl'), 'rb'))
        df = df[df['game_type'] == 'R']
        counts = {'full': defaultdict(lambda: defaultdict(int)),
                  'h1': defaultdict(lambda: defaultdict(int))}
        for row in df[['pitcher', 'pitch_type', 'game_date']].itertuples(
                index=False):
            pt = row.pitch_type
            if not isinstance(pt, str) or not pt:
                continue
            pid = str(int(row.pitcher))
            counts['full'][pid][pt] += 1
            if str(row.game_date)[:10] <= asg:
                counts['h1'][pid][pt] += 1
        srec = {}
        for scope in ('full', 'h1'):
            for pid, mix in counts[scope].items():
                n = sum(mix.values())
                if n < MIN_PITCHES:
                    continue
                h = -sum((c / n) * math.log2(c / n)
                         for c in mix.values() if c > 0)
                srec.setdefault(pid, {})[scope] = h
        out[str(y)] = srec
        print(f'  entropy {y}: {len(srec)} pitchers', flush=True)
    tmp = CACHE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, CACHE)
    return out


def features(ent, season, scope):
    z = wf.shrunk_features(season, scope)
    es = ent.get(str(season), {})
    real = [es[p][scope] for p in z if p in es and scope in es[p]]
    if len(real) >= 30:
        m = sum(real) / len(real)
        s = math.sqrt(sum((x - m) ** 2 for x in real) / len(real))
    else:
        m, s = 0.0, 0.0
    cov = 0
    for pid, f in z.items():
        h = es.get(pid, {}).get(scope)
        if h is not None and s > 0:
            f['entropy'] = (h - m) / s
            cov += 1
        else:
            f['entropy'] = 0.0
    return z, (cov / len(z) if z else 0.0)


def build_reps(ent, test, gate):
    reps, cov = [], []
    if test == 'next':
        for season in SEASONS[:-1]:
            fr, c = features(ent, season, 'full')
            tc = targets_for(season, 'full')
            tn = targets_for(season + 1, 'full')
            units = [(fr[pid], tn[pid]['era']) for pid in fr
                     if pid in tn and pid in tc
                     and tc[pid]['outs'] >= gate * 3
                     and tn[pid]['outs'] >= gate * 3]
            reps.append((f'{season}->{season + 1}', units))
            cov.append(c)
    else:
        for season in SEASONS:
            fr, c = features(ent, season, 'h1')
            t1 = targets_for(season, 'h1')
            t2 = targets_for(season, 'h2')
            hg = max(gate * 3 // 2, 45)
            units = [(fr[pid], t2[pid]['era']) for pid in fr
                     if pid in t2 and pid in t1
                     and t1[pid]['outs'] >= hg
                     and t2[pid]['outs'] >= hg]
            reps.append((f'{season}h', units))
            cov.append(c)
    return reps, cov


def loso(reps, feats):
    per = []
    for i, (label, test_units) in enumerate(reps):
        train = [u for j, (_, us) in enumerate(reps) if j != i for u in us]
        beta = ols(train, feats)
        if beta is None:
            return None
        preds, ys = [], []
        for x, y in test_units:
            if all(f in x for f in feats):
                preds.append(beta[0] + sum(b * x[f]
                                           for b, f in zip(beta[1:], feats)))
                ys.append(y)
        r = pearson(preds, ys)
        if r is None:
            return None
        per.append((label, r))
    return per, sum(r for _, r in per) / len(per)


def main():
    ent = build_entropy()
    out = {}
    for gate in (60, 30):
        for test in ('ros', 'next'):
            reps, cov = build_reps(ent, test, gate)
            print(f'\n===== {test.upper()} gate {gate} '
                  f'(entropy coverage {sum(cov) / len(cov):.1%}) =====')
            res = {}
            for tag, feats in (('control8', BASE_FEATS),
                               ('with_entropy', BASE_FEATS + ['entropy'])):
                ev = loso(reps, feats)
                if ev is None:
                    print(f'  {tag}: insufficient coverage')
                    continue
                per, mean = ev
                res[tag] = {'per': dict(per), 'mean': mean}
                print(f'  {tag:<13} mean r {mean:+.4f}   '
                      + ' '.join(f'{r:+.3f}' for _, r in per))
            if len(res) == 2:
                c, t = res['control8'], res['with_entropy']
                wins = sum(1 for k in t['per']
                           if t['per'][k] > c['per'].get(k, -9))
                print(f'  entropy wins {wins}/{len(t["per"])} folds, '
                      f'delta mean {t["mean"] - c["mean"]:+.4f}')
            out[f'{test}_{gate}'] = res
    tmp = os.path.join(ROOT, 'data', '_era_entropy_results.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, os.path.join(ROOT, 'data', '_era_entropy_results.json'))
    print('\nwrote data/_era_entropy_results.json')


if __name__ == '__main__':
    main()
