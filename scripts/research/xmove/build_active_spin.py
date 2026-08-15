"""Build data/active_spin_prior.json — prior-season spin efficiency per
(pitcher, hand, OUR pitch-type name).

Why prior season and not current: Savant's active spin may be partly derived
from movement (scripts/research/xmove/xmove_activespin_probe.py rules out reconstruction from
season-aggregate movement, but cannot rule out a per-pitch formula over the
full trajectory). Last season's value cannot leak this season's residual no
matter how it is computed, and efficiency persists at r = 0.79-0.92 by pitch
type, so almost nothing is lost. Using the CURRENT season would be circular.

Why a bridge is needed: active_spin_slider is computed over the pitches SAVANT
calls sliders. Wally re-tags manually, so his FC may be Savant's SL, and
joining on the displayed name mis-attributes those arsenals. The bridge is
built WITHIN the current season on the SAME pitches -- majority Savant tag per
class -- because comparing this year's names to last year's Savant arsenal
cannot tell a relabel apart from a pitcher changing his pitch mix. (Baz dropped
a slider and added a sinker between 2025 and 2026; a set-difference rule pairs
them and attaches a 23.5% efficiency to a sinker.)

Coverage: about 80% of pitches join directly. The rest are pitchers who did not
throw that pitch last season. For those, a sibling pitch is used where the
relationship is strong enough to beat the league mean out of sample -- in
practice the fastball group, where FF and SI efficiencies correlate at 0.88.
Breaking balls correlate NEGATIVELY with fastballs, so fallbacks never cross
groups.

Usage:
    python3 scripts/research/xmove/build_active_spin.py --season 2026 --prior 2025 [--apply]
"""
import os, sys, json, argparse, io
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT = os.path.join(ROOT, 'data', 'active_spin_prior.json')
LEADERBOARD = ('https://baseballsavant.mlb.com/leaderboard/active-spin'
               '?year={year}&team=&min=1&hand=&csv=true')

COL = {'fourseam': 'FF', 'sinker': 'SI', 'cutter': 'FC', 'changeup': 'CH',
       'splitter': 'FS', 'curve': 'CU', 'slider': 'SL', 'sweeper': 'ST',
       'slurve': 'SV'}
MIN_CLASS = 50          # pitches before a class gets its own efficiency value
MIN_PURITY = 0.80       # a class whose Savant tags are this mixed is not one class


def fetch_active(year):
    r = requests.get(LEADERBOARD.format(year=year), timeout=90,
                     headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text), encoding='utf-8-sig')
    rows = []
    for _, x in d.iterrows():
        for c, pt in COL.items():
            v = x.get(f'active_spin_{c}')
            if pd.notna(v):
                rows.append((x.entity_name, x.pitch_hand, pt, float(v)))
    return pd.DataFrame(rows, columns=['pitcher', 'thr', 'pt', 'active'])


def savant_tags(season):
    """Per-pitch Savant classification for the CURRENT season, keyed by
    PitchID (game_pk_atbat_pitchno) so it joins to the cache."""
    path = os.path.join(ROOT, 'data', f'_statcast{season}_full.pkl')
    if not os.path.exists(path):
        raise SystemExit(f'missing {path}; run the statcast pull for {season} first')
    sc = pd.read_pickle(path)
    sc = sc[['game_pk', 'at_bat_number', 'pitch_number', 'pitch_type']].dropna()
    sc['pitch_type'] = sc.pitch_type.replace({'KC': 'CU', 'FO': 'FS'})
    pid = (sc.game_pk.astype(int).astype(str) + '_' +
           sc.at_bat_number.astype(int).astype(str).str.zfill(3) + '_' +
           sc.pitch_number.astype(int).astype(str).str.zfill(2))
    return dict(zip(pid, sc.pitch_type))


def our_classes():
    import pickle
    with open(CACHE, 'rb') as f:
        raw = pickle.load(f)
    rows = [(p.get('Pitcher'), p.get('Throws'), p.get('Pitch Type'), p.get('PitchID'))
            for p in raw
            if p.get('_source') == 'MLB' and p.get('PitchID') and p.get('Pitch Type')]
    return pd.DataFrame(rows, columns=['pitcher', 'thr', 'pt', 'pid'])


def fit_fallbacks(act, folds=5):
    """Which sibling pitch types can stand in for a missing one?

    Not a hand-picked correlation threshold: for every ordered pair, check
    out-of-fold whether a fitted linear map from the source beats simply using
    the target's league mean. Pairs that fail are not used at all.
    """
    w = act.pivot_table(index=['pitcher', 'thr'], columns='pt', values='active')
    keep = {}
    rng = np.random.default_rng(3)
    for tgt in w.columns:
        for src in w.columns:
            if src == tgt:
                continue
            m = w[[src, tgt]].dropna()
            if len(m) < 60:
                continue
            x, y = m[src].values, m[tgt].values
            fold = rng.integers(0, folds, len(y))
            pred = np.empty(len(y))
            for f in range(folds):
                tr, te = fold != f, fold == f
                if tr.sum() < 20:
                    pred[te] = np.nan
                    continue
                b = np.polyfit(x[tr], y[tr], 1)
                pred[te] = np.polyval(b, x[te])
            ok = np.isfinite(pred)
            sse = ((y[ok] - pred[ok]) ** 2).sum()
            sst = ((y[ok] - y[ok].mean()) ** 2).sum()
            r2 = 1 - sse / sst
            if r2 > 0:                      # beats the league mean out of fold
                b = np.polyfit(x, y, 1)
                keep.setdefault(tgt, []).append(
                    dict(src=src, slope=float(b[0]), intercept=float(b[1]),
                         r2=float(r2), n=int(len(m))))
    for tgt in keep:
        keep[tgt].sort(key=lambda d: -d['r2'])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--prior', type=int, default=None)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    prior = a.prior or a.season - 1

    print(f'building the {prior} active-spin prior for {a.season} classes\n')
    act = fetch_active(prior)
    print(f'  {len(act)} pitcher x pitch-type efficiency values for {prior}')
    tags = savant_tags(a.season)
    cls = our_classes()
    cls['savant'] = cls.pid.map(tags)
    print(f'  {cls.savant.notna().sum():,}/{len(cls):,} {a.season} pitches carry a '
          f'Savant tag ({cls.savant.notna().mean()*100:.1f}%)')

    lut = {(r.pitcher, r.thr, r.pt): r.active for r in act.itertuples()}
    lg = act.groupby('pt').active.mean().to_dict()
    fb = fit_fallbacks(act)
    print(f'\n  sibling fallbacks that beat the league mean out of fold:')
    for tgt, opts in sorted(fb.items()):
        best = opts[0]
        print(f'    {tgt:>3} <- {best["src"]:<3} r2 {best["r2"]:.2f} '
              f'(n={best["n"]})' + (f'   [+{len(opts)-1} more]' if len(opts) > 1 else ''))

    out, stats = {}, Counter()
    renames = []
    for (pitcher, thr, pt), g in cls.groupby(['pitcher', 'thr', 'pt']):
        if len(g) < MIN_CLASS:
            continue
        vc = g.savant.value_counts()
        if vc.empty:
            stats['no_savant_tag'] += len(g)
            continue
        major, purity = vc.index[0], vc.iloc[0] / vc.sum()
        if purity < MIN_PURITY:
            stats['impure_class'] += len(g)
            continue
        rec = None
        v = lut.get((pitcher, thr, major))
        if v is not None:
            rec = dict(active=v, via='direct', savant=major, purity=round(purity, 3))
            stats['direct' if major == pt else 'renamed'] += len(g)
            if major != pt:
                renames.append((pitcher, thr, major, pt, v, purity, len(g)))
        else:
            for opt in fb.get(major, []):
                sv = lut.get((pitcher, thr, opt['src']))
                if sv is None:
                    continue
                est = opt['slope'] * sv + opt['intercept']
                rec = dict(active=round(float(np.clip(est, 0, 100)), 1),
                           via=f'from_{opt["src"]}', savant=major,
                           purity=round(purity, 3), r2=round(opt['r2'], 3))
                stats['sibling'] += len(g)
                break
        if rec is None:
            stats['unmapped'] += len(g)
            continue
        out[f'{pitcher}|{thr}|{pt}'] = rec

    tot = sum(stats.values())
    print(f'\n  attribution across {tot:,} pitches in classes >= {MIN_CLASS}:')
    for k in ('direct', 'renamed', 'sibling', 'unmapped', 'impure_class',
              'no_savant_tag'):
        if stats[k]:
            print(f'    {k:>14}: {stats[k]:>8,}  {stats[k]/tot*100:>5.1f}%')
    cov = (stats['direct'] + stats['renamed'] + stats['sibling']) / tot
    print(f'  -> {cov*100:.1f}% of pitches carry a prior-season efficiency value')

    print(f'\n  {len(renames)} classes bridged to a different Savant tag '
          f'(largest first):')
    for r in sorted(renames, key=lambda x: -x[6])[:8]:
        print(f'    {r[0]:<24} {r[1]}  Savant {r[2]} -> ours {r[3]:<3} '
              f'n={r[6]:>4} purity {r[5]*100:>5.1f}% active {r[4]:.1f}%')

    if a.apply:
        payload = dict(season=a.season, priorSeason=prior,
                       leagueMeanByPitchType={k: round(v, 2) for k, v in lg.items()},
                       entries=out)
        with open(OUT, 'w') as f:
            json.dump(payload, f, indent=0, sort_keys=True)
        print(f'\nwrote {OUT}  ({len(out)} classes)')
    else:
        print(f'\n(dry run -- {len(out)} classes would be written; pass --apply)')


if __name__ == '__main__':
    main()
