"""pitcher_comps.py — closest pitcher comparisons (attributes + arsenal).

Distance = attribute z-distance + arsenal-mix distance, same-handed pool:
  - attributes (13, z-scored over the pool): fbVelo, armAngle, extension,
    kPct, bbPct, gbPct, swStrPct, chasePct, strikePct, cswPct, xwOBAcon,
    stuffScore, locPlus
  - arsenal: usage shares by Loc+ pitch group (FF/SI/FC/SL/CU/CH/OTHER),
    half-L1 distance in [0,1], weighted x1.5 so a typical mix gap counts
    about as much as the attribute gap
  - pool: same THROWS hand, MLB + ROC, >= 400 pitches (ROC scored vs MLB
    baselines site-wide, so cross-level comps are legitimate)

Shape comps, not talent equivalences. Role (SP/RP) is not forced — a
starter can comp to a reliever if he throws like one.

Usage:
  python3 scripts/research/comps/pitcher_comps.py --pitcher "Lara, Andry"
  python3 scripts/research/comps/pitcher_comps.py --team ROC [--min-pitches 250]
"""
import os, sys, json, math, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pipeline.locplus import group_of_code

ATTRS = ['fbVelo', 'armAngle', 'extension', 'kPct', 'bbPct', 'gbPct',
         'swStrPct', 'chasePct', 'strikePct', 'cswPct', 'xwOBAcon',
         'stuffScore', 'locPlus']
NICE = {'fbVelo': 'FBVelo', 'armAngle': 'ArmAngle', 'extension': 'Ext',
        'kPct': 'K%', 'bbPct': 'BB%', 'gbPct': 'GB%', 'swStrPct': 'SwStr%',
        'chasePct': 'Chase%', 'strikePct': 'Strike%', 'cswPct': 'CSW%',
        'xwOBAcon': 'xwOBAcon', 'stuffScore': 'Stuff+', 'locPlus': 'Loc+'}
GROUPS = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'OTHER']
ARSENAL_W = 1.5
MIN_ATTRS = 9


def load():
    pp = json.load(open(os.path.join(ROOT, 'data', 'pitcher_leaderboard_rs.json')))
    pl = json.load(open(os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')))
    usage = defaultdict(lambda: defaultdict(float))
    for r in pl:
        key = (r['pitcher'], r['team'])
        g = group_of_code(r.get('pitchType'))
        if g:
            usage[key][g] += r.get('count') or 0
    rows = []
    for r in pp:
        if (r.get('team') or '').endswith('TM'):
            continue
        key = (r['pitcher'], r['team'])
        tot = sum(usage[key].values())
        rec = {f: r.get(f) for f in ATTRS}
        rec.update(name=r['pitcher'], team=r['team'], throws=r.get('throws'),
                   n=r.get('count') or 0, ip=r.get('ip'),
                   mix={g: usage[key].get(g, 0.0) / tot for g in GROUPS} if tot else None)
        rows.append(rec)
    return rows


def zstats(pool):
    out = {}
    for f in ATTRS:
        vals = [r[f] for r in pool if r.get(f) is not None]
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        out[f] = (m, s if s > 0 else 1.0)
    return out


def arsenal_dist(a, b):
    if not a.get('mix') or not b.get('mix'):
        return None
    return sum(abs(a['mix'][g] - b['mix'][g]) for g in GROUPS) / 2.0


def dist(a, b, stats):
    ds = []
    for f in ATTRS:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        m, s = stats[f]
        ds.append((abs((va - m) / s - (vb - m) / s), f))
    ad = arsenal_dist(a, b)
    if len(ds) < MIN_ATTRS or ad is None:
        return None, None, None
    attr = sum(d for d, _ in ds) / len(ds)
    return attr + ARSENAL_W * ad, attr, ad


def mix_str(r):
    return '/'.join(f'{g}{r["mix"][g]*100:.0f}' for g in GROUPS if r['mix'][g] >= 0.05)


def why(t, b, stats):
    def z(r, f):
        v = r.get(f)
        if v is None:
            return None
        m, s = stats[f]
        return (v - m) / s
    shared, gaps = [], []
    for f in ATTRS:
        za, zb = z(t, f), z(b, f)
        if za is None or zb is None:
            continue
        if abs(za) >= 0.75 and abs(zb) >= 0.75 and za * zb > 0:
            shared.append((min(abs(za), abs(zb)), f, za))
        gaps.append((abs(za - zb), f))
    shared.sort(reverse=True)
    gap_d, gap_f = max(gaps)
    tr = ', '.join(f"{NICE[f]} {'high' if zz > 0 else 'low'}"
                   for _, f, zz in shared[:4]) or 'league-average attributes'
    return (f"   arsenal: {mix_str(t)}  vs  {mix_str(b)}\n"
            f"   shared: {tr}\n"
            f"   biggest attr gap: {NICE[gap_f]} (Δz {gap_d:.1f}: "
            f"{t.get(gap_f)} vs {b.get(gap_f)})")


def report(t, pool, stats, topn=3):
    scored = []
    for r in pool:
        d, attr, ad = dist(t, r, stats)
        if d is not None:
            scored.append((d, attr, ad, r))
    scored.sort(key=lambda x: x[0])
    flag = ' [small sample]' if t['n'] < 400 else ''
    print(f"\n{t['name']} ({t['team']}, {t['throws']}HP, {t['n']} pitches){flag}")
    for d, attr, ad, r in scored[:topn]:
        print(f"   {d:.3f} (attr {attr:.2f} + mix {ad:.2f})  "
              f"{r['name']:24s} {r['team']:3s}  {r['n']}p")
    if scored[0][3]['team'] == t['team'] == 'ROC':
        mlb = next(((d, r) for d, a, ad, r in scored if r['team'] != 'ROC'), None)
        if mlb:
            print(f"   best MLB: {mlb[0]:.3f}  {mlb[1]['name']} {mlb[1]['team']}")
    print(why(t, scored[0][3], stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pitcher', default=None)
    ap.add_argument('--pitcher-team', default='ROC')
    ap.add_argument('--team', default=None)
    ap.add_argument('--min-pitches', type=int, default=250)
    ap.add_argument('--min-ip', type=float, default=None,
                    help='batch mode: qualify targets by IP instead of pitches')
    ap.add_argument('--mlb-pool', action='store_true',
                    help='draw comps from MLB pitchers only (no ROC)')
    args = ap.parse_args()

    from pipeline.utils import ip_str_to_float
    rows = load()
    targets = []
    if args.team:
        cand = [r for r in rows if r['team'] == args.team]
        if args.min_ip is not None:
            cand = [r for r in cand if (ip_str_to_float(r.get('ip') or '0') or 0) >= args.min_ip]
        else:
            cand = [r for r in cand if r['n'] >= args.min_pitches]
        targets = sorted(cand, key=lambda r: -r['n'])
    else:
        name = args.pitcher or 'Lara, Andry'
        targets = [next(r for r in rows
                        if r['name'] == name and r['team'] == args.pitcher_team)]
    for t in targets:
        pool = [r for r in rows if r['n'] >= 400 and r['throws'] == t['throws']
                and (r['name'], r['team']) != (t['name'], t['team'])]
        if args.mlb_pool:
            pool = [r for r in pool if r['team'] != 'ROC']
        stats = zstats(pool + [t])
        report(t, pool, stats)


if __name__ == '__main__':
    main()
