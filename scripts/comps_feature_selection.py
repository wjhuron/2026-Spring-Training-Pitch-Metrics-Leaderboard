"""comps_feature_selection.py — which leaderboard metrics make the best comps?

Systematic follow-up to comps_validation.py (2026-08-03). Every candidate
feature the site's leaderboards surface that can be recomputed from the pitch
cache is scored for:

  1. split-half RELIABILITY (interleaved + temporal halves), and
  2. marginal value to the comp OBJECTIVE — greedy forward selection of the
     fingerprint that maximizes k-NN predictive validity: neighbors chosen on
     half A must predict the target's half-B outcome battery (mean Pearson r).
     Selection runs independently per split; each split's selected set is then
     TESTED on the other split (its own number is in-sample by construction).

Pitchers: fingerprint selected at mix_w=0 (arsenal held out) so the feature
ranking is not confounded by the separately-validated mix component.
Hitters: fingerprint-only (no arsenal analogue).

The 2021-25 training caches lack Event/BBType/InZone/EV, so cross-season
replication is impossible for most candidates — the temporal split (predicting
the season's second half from its first) is the deployment-shaped check.
Greedy stop: marginal gain < 0.002 (convention). Full gain curves printed.
"""
import os, sys, math, pickle, argparse
from array import array
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import (safe_float as sf, NON_PA_EVENTS, BB_EVENTS,
                            K_EVENTS, BUNT_BB_TYPES, SWING_DESCRIPTIONS,
                            spray_angle, spray_direction)

AAA = {'ROC', 'AAA'}
K_NN = 5
MIN_GAIN = 0.002
MAX_FEATS = 14
BATTERY_P = ['rv100', 'kbbPct', 'gbPct', 'xwOBAcon', 'whiffPct']
BATTERY_H = ['rv100', 'kPct', 'bbPct', 'xwOBAcon', 'barrelPct']
BALL_DESCS = ('Ball', 'Ball In Dirt', 'Intent Ball', 'Hit By Pitch', 'Pitchout')


def pearson(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def halves(pitches, mode, who):
    dates = defaultdict(set)
    for p in pitches:
        dates[p[who]].add(p.get('Game Date') or '')
    half_of = {}
    for nm, ds in dates.items():
        sd = sorted(ds)
        for i, d in enumerate(sd):
            half_of[(nm, d)] = (i % 2) if mode == 'interleaved' else int(i >= len(sd) / 2)
    A, B = [], []
    for p in pitches:
        (A if half_of[(p[who], p.get('Game Date') or '')] == 0 else B).append(p)
    return A, B


def _strikes(count):
    try:
        return int(str(count).split('-')[1])
    except (ValueError, IndexError, AttributeError):
        return None


# ── candidate aggregators ──────────────────────────────────────────────────

def agg_pitchers(pitches):
    """name -> {feature: value}. Every candidate the leaderboards surface."""
    c = defaultdict(lambda: defaultdict(float))
    for p in pitches:
        nm = p.get('Pitcher')
        if not nm:
            continue
        a = c[nm]
        desc, pt = p.get('Description'), p.get('Pitch Type')
        a['n'] += 1
        for fld, key in (('Extension', 'ext'), ('ArmAngle', 'ang'),
                         ('Stuff+', 'stuff'), ('Loc+', 'loc'), ('RunExp', 'rv')):
            v = sf(p.get(fld))
            if v is not None:
                a[f'{key}_s'] += v
                a[f'{key}_n'] += 1
        rx = sf(p.get('RelPosX'))
        if rx is not None:
            a['relx_s'] += abs(rx)
            a['relx_n'] += 1
        rz = sf(p.get('RelPosZ'))
        if rz is not None:
            a['relz_s'] += rz
            a['relz_n'] += 1
        if pt in ('FF', 'SI'):
            a[f'{pt}_n'] += 1
            for fld, key in (('Velocity', 'v'), ('VAA', 'vaa'), ('HAA', 'haa')):
                v = sf(p.get(fld))
                if v is not None:
                    a[f'{pt}_{key}_s'] += v
                    a[f'{pt}_{key}_n'] += 1
        in_z = p.get('InZone') == 'Yes'
        strikes = _strikes(p.get('Count'))
        if in_z:
            a['iz'] += 1
        else:
            a['ooz'] += 1
        if str(p.get('Count')) == '0-0':
            a['fp'] += 1
            if desc not in BALL_DESCS:
                a['fps'] += 1
        is_swing = desc in SWING_DESCRIPTIONS and 'Bunt' not in (desc or '')
        if desc in ('Called Strike', 'Swinging Strike'):
            a['csw'] += 1
        if is_swing:
            a['sw'] += 1
            if in_z:
                a['izsw'] += 1
            else:
                a['oozsw'] += 1
            if strikes == 2:
                a['sw2k'] += 1
            if desc == 'Swinging Strike':
                a['wh'] += 1
                if in_z:
                    a['izwh'] += 1
                if strikes == 2:
                    a['wh2k'] += 1
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS and ev != 'Intent Walk':
            a['tbf'] += 1
            if ev in K_EVENTS:
                a['k'] += 1
            elif ev in BB_EVENTS:
                a['bb'] += 1
        bb = p.get('BBType')
        if desc == 'In Play' and bb and bb not in BUNT_BB_TYPES:
            a['bip'] += 1
            if bb == 'ground_ball':
                a['gb'] += 1
            elif bb == 'popup':
                a['pu'] += 1
            elif bb == 'line_drive':
                a['ld'] += 1
            evf = sf(p.get('ExitVelo'))
            if evf is not None:
                a['ev_s'] += evf
                a['ev_n'] += 1
                if evf >= 95:
                    a['hh'] += 1
            try:
                if int(sf(p.get('Barrel')) or 0) == 6:
                    a['brl'] += 1
            except (TypeError, ValueError):
                pass
            xw = sf(p.get('xwOBA'))
            if xw is not None:
                a['xw_s'] += xw
                a['xw_n'] += 1
    out = {}
    for nm, a in c.items():
        if a['n'] < 200 or a['tbf'] < 50 or a['bip'] < 30 or a['sw'] < 50:
            continue
        fb = 'FF' if a['FF_n'] >= a['SI_n'] else 'SI'
        r = lambda num, den: a[num] / a[den] if a[den] else None
        haa = r(f'{fb}_haa_s', f'{fb}_haa_n')
        out[nm] = dict(
            name=nm,
            fbVelo=r(f'{fb}_v_s', f'{fb}_v_n'), vaa=r(f'{fb}_vaa_s', f'{fb}_vaa_n'),
            haa=abs(haa) if haa is not None else None,
            extension=r('ext_s', 'ext_n'), armAngle=r('ang_s', 'ang_n'),
            relPosZ=r('relz_s', 'relz_n'), absRelPosX=r('relx_s', 'relx_n'),
            stuffScore=r('stuff_s', 'stuff_n'), locPlus=r('loc_s', 'loc_n'),
            whiffPct=a['wh'] / a['sw'], izWhiffPct=r('izwh', 'izsw'),
            twoStrikeWhiffPct=r('wh2k', 'sw2k'),
            chasePct=r('oozsw', 'ooz'), izPct=a['iz'] / a['n'],
            cswPct=a['csw'] / a['n'], fpsPct=r('fps', 'fp'),
            swStrRate=a['wh'] / a['n'],
            kPct=a['k'] / a['tbf'], bbPct=a['bb'] / a['tbf'],
            kbbPct=(a['k'] - a['bb']) / a['tbf'],
            gbPct=a['gb'] / a['bip'], puPct=a['pu'] / a['bip'],
            ldPct=a['ld'] / a['bip'],
            avgEVAgainst=r('ev_s', 'ev_n'),
            hardHitPct=r('hh', 'ev_n'), barrelPctAgainst=a['brl'] / a['bip'],
            xwOBAcon=r('xw_s', 'xw_n'),
            rv100=(a['rv_s'] / a['rv_n'] * 100) if a['rv_n'] else None,
        )
    return out


def agg_hitters(pitches):
    c = defaultdict(lambda: defaultdict(float))
    evlists = defaultdict(list)
    for p in pitches:
        nm = p.get('Batter')
        if not nm:
            continue
        a = c[nm]
        desc = p.get('Description')
        bats = p.get('Bats')
        a['n'] += 1
        in_z = p.get('InZone') == 'Yes'
        strikes = _strikes(p.get('Count'))
        if in_z:
            a['iz'] += 1
        else:
            a['ooz'] += 1
        if desc in ('Called Strike', 'Swinging Strike'):
            a['csw'] += 1
        rv = sf(p.get('RunExp'))
        if rv is not None:
            a['rv_s'] -= rv                      # batter perspective
            a['rv_n'] += 1
        is_swing = desc in SWING_DESCRIPTIONS and 'Bunt' not in (desc or '')
        if is_swing:
            a['sw'] += 1
            if in_z:
                a['izsw'] += 1
            else:
                a['oozsw'] += 1
            if strikes == 2:
                a['sw2k'] += 1
            if desc == 'Swinging Strike':
                a['wh'] += 1
                if in_z:
                    a['izwh'] += 1
                if strikes == 2:
                    a['wh2k'] += 1
            for fld, key, gate in (('BatSpeed', 'bs', 50.0), ('SwingLength', 'sl', None),
                                   ('AttackAngle', 'aa', None), ('SwingPathTilt', 'spt', None)):
                v = sf(p.get(fld))
                if v is not None and (gate is None or v > gate):
                    a[f'{key}_s'] += v
                    a[f'{key}_n'] += 1
            ad = sf(p.get('AttackDirection'))
            if ad is not None:
                a['ad_s'] += (-ad if bats == 'L' else ad)   # mirror LHB
                a['ad_n'] += 1
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS and ev != 'Intent Walk':
            a['pa'] += 1
            if ev in K_EVENTS:
                a['k'] += 1
            elif ev in BB_EVENTS:
                a['bb'] += 1
        bb = p.get('BBType')
        if desc == 'In Play' and bb and bb not in BUNT_BB_TYPES:
            a['bip'] += 1
            if bb == 'ground_ball':
                a['gb'] += 1
            elif bb == 'popup':
                a['pu'] += 1
            elif bb == 'line_drive':
                a['ld'] += 1
            evf = sf(p.get('ExitVelo'))
            if evf is not None:
                a['ev_s'] += evf
                a['ev_n'] += 1
                evlists[nm].append(evf)
                if evf >= 95:
                    a['hh'] += 1
            try:
                if int(sf(p.get('Barrel')) or 0) == 6:
                    a['brl'] += 1
            except (TypeError, ValueError):
                pass
            xw = sf(p.get('xwOBA'))
            if xw is not None:
                a['xw_s'] += xw
                a['xw_n'] += 1
            d = spray_direction(spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y'))), bats)
            if d in ('pull', 'pull_side'):
                a['pull'] += 1
                if bb in ('line_drive', 'fly_ball'):
                    a['airpull'] += 1
            elif d in ('oppo', 'oppo_side'):
                a['oppo'] += 1
    out = {}
    for nm, a in c.items():
        if a['bip'] < 25 or a['sw'] < 50 or a['pa'] < 50:
            continue
        r = lambda num, den: a[num] / a[den] if a[den] else None
        evs = sorted(evlists.get(nm) or [])
        out[nm] = dict(
            name=nm,
            avgEVAll=r('ev_s', 'ev_n'),
            p90EV=evs[int(0.9 * (len(evs) - 1))] if len(evs) >= 10 else None,
            hardHitPct=r('hh', 'ev_n'), barrelPct=a['brl'] / a['bip'],
            xwOBAcon=r('xw_s', 'xw_n'),
            gbPct=a['gb'] / a['bip'], puPct=a['pu'] / a['bip'],
            ldPct=a['ld'] / a['bip'],
            pullPct=a['pull'] / a['bip'], oppoPct=a['oppo'] / a['bip'],
            airPullPct=a['airpull'] / a['bip'],
            swingPct=a['sw'] / a['n'], izSwingPct=r('izsw', 'iz'),
            chasePct=r('oozsw', 'ooz'), whiffPct=a['wh'] / a['sw'],
            izWhiffPct=r('izwh', 'izsw'), twoStrikeWhiffPct=r('wh2k', 'sw2k'),
            cswPct=a['csw'] / a['n'],
            kPct=a['k'] / a['pa'], bbPct=a['bb'] / a['pa'],
            batSpeed=r('bs_s', 'bs_n') if a['bs_n'] >= 20 else None,
            swingLength=r('sl_s', 'sl_n') if a['sl_n'] >= 20 else None,
            attackAngle=r('aa_s', 'aa_n') if a['aa_n'] >= 20 else None,
            attackDirection=r('ad_s', 'ad_n') if a['ad_n'] >= 20 else None,
            swingPathTilt=r('spt_s', 'spt_n') if a['spt_n'] >= 20 else None,
            rv100=(a['rv_s'] / a['rv_n'] * 100) if a['rv_n'] else None,
        )
    return out


# ── selection machinery ────────────────────────────────────────────────────

def zscale(pool, feats):
    out = {}
    for f in feats:
        vals = [r[f] for r in pool if r.get(f) is not None]
        if len(vals) < 30:
            out[f] = None
            continue
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        out[f] = (m, s if s > 0 else 1.0)
    return out


def greedy(names, A_rows, B_rows, cands, battery, label):
    pool = [A_rows[n] for n in names]
    N = len(names)
    zs = zscale(pool, cands)
    cands = [f for f in cands if zs[f] is not None]

    # reliability
    print(f"\n{label}: {N} players. Split-half reliability r:")
    rel = {}
    for f in cands:
        pairs = [(A_rows[n][f], B_rows[n][f]) for n in names
                 if A_rows[n].get(f) is not None and B_rows[n].get(f) is not None]
        rel[f] = pearson(*zip(*pairs)) if len(pairs) >= 30 else None
    for f in sorted(cands, key=lambda f: -(rel[f] or -9)):
        print(f"   {f:20s} {rel[f] if rel[f] is not None else float('nan'):+.3f}")

    # per-feature pair |z-diff| tables (nan = missing)
    npairs = N * (N - 1) // 2
    pd = {}
    for f in cands:
        m, s = zs[f]
        col = array('d', [float('nan')] * npairs)
        vals = [(A_rows[n].get(f) - m) / s if A_rows[n].get(f) is not None else None
                for n in names]
        k = 0
        for i in range(N):
            vi = vals[i]
            for j in range(i + 1, N):
                if vi is not None and vals[j] is not None:
                    col[k] = abs(vi - vals[j])
                k += 1
        pd[f] = col

    def score(sel):
        sums = array('d', [0.0] * npairs)
        cnts = array('d', [0.0] * npairs)
        for f in sel:
            col = pd[f]
            for k in range(npairs):
                v = col[k]
                if v == v:
                    sums[k] += v
                    cnts[k] += 1
        need = max(1, int(0.7 * len(sel)))
        idx = lambda i, j: (i * (2 * N - i - 1)) // 2 + (j - i - 1) if i < j \
            else (j * (2 * N - j - 1)) // 2 + (i - j - 1)
        preds = defaultdict(list)
        for i in range(N):
            ds = []
            for j in range(N):
                if j == i:
                    continue
                k = idx(i, j)
                if cnts[k] >= need:
                    ds.append((sums[k] / cnts[k], j))
            ds.sort()
            nn = [names[j] for _, j in ds[:K_NN]]
            if len(nn) < K_NN:
                continue
            for b in battery:
                vs = [B_rows[n][b] for n in nn if B_rows[n].get(b) is not None]
                tv = B_rows[names[i]].get(b)
                if vs and tv is not None:
                    preds[b].append((sum(vs) / len(vs), tv))
        rs = [pearson(*zip(*preds[b])) for b in battery if preds[b]]
        rs = [r for r in rs if r is not None]
        return sum(rs) / len(rs) if rs else -9

    sel, cur = [], -9
    print(f"{label}: greedy forward selection (objective: kNN battery mean r):")
    while len(sel) < MAX_FEATS:
        best_f, best_r = None, cur
        for f in cands:
            if f in sel:
                continue
            r = score(sel + [f])
            if r > best_r:
                best_f, best_r = f, r
        if best_f is None or best_r - cur < MIN_GAIN:
            break
        sel.append(best_f)
        cur = best_r
        print(f"   + {best_f:20s} -> {cur:+.3f}")
    print(f"   SELECTED ({len(sel)}): {', '.join(sel)}")
    return sel, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=['pitcher', 'hitter', 'both'], default='both')
    args = ap.parse_args()

    print("Loading 2026 cache (MLB only)...")
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    for role in (['pitcher', 'hitter'] if args.role == 'both' else [args.role]):
        who = 'Pitcher' if role == 'pitcher' else 'Batter'
        team_key = 'PTeam' if role == 'pitcher' else 'BTeam'
        mlb = [p for p in D if p.get(who) and p.get(team_key) not in AAA]
        aggfn = agg_pitchers if role == 'pitcher' else agg_hitters
        battery = BATTERY_P if role == 'pitcher' else BATTERY_H
        sel_by_mode, score_by_mode, names_by_mode = {}, {}, {}
        for mode in ('interleaved', 'temporal'):
            A, B = halves(mlb, mode, who)
            A_rows, B_rows = aggfn(A), aggfn(B)
            names = sorted(set(A_rows) & set(B_rows))
            cands = [f for f in next(iter(A_rows.values())) if f != 'name']
            sel, scorer = greedy(names, A_rows, B_rows, cands, battery,
                                 f'{role.upper()} {mode}')
            sel_by_mode[mode] = sel
            score_by_mode[mode] = scorer
        # cross-split test: each split's selected set scored on the OTHER split
        for m1, m2 in (('interleaved', 'temporal'), ('temporal', 'interleaved')):
            r = score_by_mode[m2](sel_by_mode[m1])
            print(f"{role}: {m1}-selected set on {m2} split -> {r:+.3f} "
                  f"(that split's own selection reached "
                  f"{score_by_mode[m2](sel_by_mode[m2]):+.3f})")


if __name__ == '__main__':
    main()
