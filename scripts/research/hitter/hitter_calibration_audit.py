"""hitter_calibration_audit.py — is a displayed hitter point a true point,
decile by decile and hand by hand? (2026-09-05)

Year-pair calibration (Y -> Y+1, 2021..2026, self-contained seasons) for the
four shipped hitter plus metrics, in the percent units each one claims:
  BB+      displayed = the process_data recipe (xwOBAcon ratio shrunk at n0
           130, EV95 ratio through the live beta, 60/40 blend, slope match
           1.2352; the 2024+ bat-tracking prior is OMITTED here)
           realized = 100 * actual wOBAcon(Y+1) / league
  CT+      displayed = pipeline.contact.compute_ct_plus (prior-free)
           realized = 100 * contact-per-swing(Y+1) / league
  SD+      displayed = pipeline.sdplus.compute_sd_plus
           realized = 100 * raw decision value(Y+1) / league (unshrunk)
  Hitter+  displayed = 52/17/31 z-composite, scale = r x SD(realized) as
           production matches wRC+ live (here the realized wOBA percent)
           realized = 100 + 100 * (wOBA(Y+1) - lg) / (scale * lgRPA)
Reports per metric: slope of realized on displayed per pair; decile table
pooled across pairs; residual (realized - displayed) by batter hand with a
paired SE across pairs; residual against the trait each channel is blind
to (BB+ vs GB% and mean launch angle; CT+ vs swing% and chase%; SD+ vs
swing%; Hitter+ vs GB%).
LGRPA for 2021-2025 is the published league runs per PA to three decimals
(a unit conversion, not a tuning constant); 2026 from metadata.
Usage: python3 scripts/research/hitter/hitter_calibration_audit.py
Output: console + data/_hitter_calibration_audit.json
"""
import gc, json, math, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import statcast_hitter_adapter as A
import hitter_phase2_multiseason as H
import pipeline.sdplus as sd
import pipeline.contact as ct
from pipeline.utils import safe_float

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
PAIRS = [(y, y + 1) for y in (2021, 2022, 2023, 2024, 2025)]
LGRPA = {2021: 0.118, 2022: 0.113, 2023: 0.120, 2024: 0.116, 2025: 0.117}
# BB+ recipe constants, single home pipeline/process_data.py (read 2026-09-05)
BB_W_CON, BB_W_EV, BB_N0_CON, BB_N0_EV = 0.60, 0.40, 130, 0
BB_SLOPE_MATCH, BB_BETA_FROZEN, BB_BETA_BAND = 1.2352, 4.205, (3.0, 6.0)
BB_BETA_GATE_BIP, BB_BETA_MIN_POOL, BB_MIN_BIP = 80, 40, 30
HP_W = (0.52, 0.17, 0.31)
SWING = {'Swinging Strike', 'Foul', 'In Play'}
BUNT_DESC = {'Foul Bunt', 'Missed Bunt'}
WOBA_EV = {'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0,
           'Single': 0.9, 'Double': 1.25, 'Triple': 1.6, 'Home Run': 2.0,
           'field_error': 0.9, 'fielders_choice': 0.9, 'fielders_choice_out': 0.9,
           'Field Error': 0.9, 'Fielders Choice': 0.9, 'Fielders Choice Out': 0.9}

def is_bip(p):
    return p.get('Description') == 'In Play' and (p.get('BBType') or '') not in ('bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive')

def is_swing(p):
    return p.get('Description') in SWING and p.get('Description') not in BUNT_DESC and (p.get('BBType') or '') not in ('bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive')

def pct(vals, q):
    return float(np.percentile(vals, q)) if vals else None

def season_frame(Y, name2id):
    P = H.load_season(Y)
    lg, sc = H.guts(Y)
    if Y == 2026:
        # sheet dicts: Batter is a name; key on the leaderboard mlbId
        for p in P:
            b = p.get('Batter'); p['_bid'] = name2id.get(b)
    else:
        for p in P:
            p['_bid'] = p.get('Batter')
    P = [p for p in P if p.get('_bid')]
    by_h = defaultdict(list)
    for p in P:
        by_h[(p['_bid'], '')].append(p)
    # ---- displayed SD+ and CT+ via the production modules ----
    sdn, _ = sd.compute_sd_plus(P, by_h, lg, sc)
    ctn, _ = ct.compute_ct_plus(P, by_h, lg, sc)
    # ---- per-hitter raw ingredients ----
    rows = {}
    for (b, _t), ps in by_h.items():
        bip = [p for p in ps if is_bip(p)]
        xw = [safe_float(p.get('xwOBA')) for p in bip]; xw = [v for v in xw if v is not None]
        ev = [safe_float(p.get('ExitVelo')) for p in bip]; ev = [v for v in ev if v is not None]
        la = [safe_float(p.get('LaunchAngle')) for p in bip]; la = [v for v in la if v is not None]
        sw = [p for p in ps if is_swing(p)]
        contact = sum(1 for p in sw if p.get('Description') != 'Swinging Strike')
        # InZone is a bool on adapted 2021-2025 dicts and a 'Yes'/'No' string on the 2026 sheet dicts
        ooz = [p for p in ps if p.get('InZone') in (False, 0, 'No', 'no', 'N')]
        ooz_sw = sum(1 for p in ooz if is_swing(p))
        gb = sum(1 for p in bip if (p.get('BBType') or '') in ('ground_ball', 'Ground Ball', 'groundball'))
        won = sum(WOBA_EV.get(p.get('Event') or p.get('event_raw') or '', 0.0) for p in bip)
        hands = defaultdict(int)
        for p in ps:
            if p.get('Bats') in ('L', 'R'):
                hands[p['Bats']] += 1
        nh = sum(hands.values()); fl = hands['L'] / nh if nh else None
        bats = None if fl is None else ('S' if 0.2 <= fl <= 0.8 else ('L' if fl > 0.8 else 'R'))
        rows[b] = dict(nbip=len(bip), xwcon=(sum(xw) / len(xw)) if xw else None, ev95=pct(ev, 95), mean_la=(sum(la) / len(la)) if la else None,
                       gb_pct=(gb / len(bip)) if bip else None, wobacon=(won / len(bip)) if bip else None,
                       n_sw=len(sw), contact=(contact / len(sw)) if sw else None, swing_pct=(len(sw) / len(ps)) if ps else None,
                       chase=(ooz_sw / len(ooz)) if ooz else None, bats=bats, n_pitch=len(ps))
        s = sdn.get((b, '')) or {}
        rows[b]['sdPlus'] = s.get('sdPlus'); rows[b]['raw_sd'] = s.get('raw_sd'); rows[b]['n_dec'] = s.get('n_decisions') or 0
        c = ctn.get((b, '')) or {}
        rows[b]['ctPlus'] = c.get('ctPlus'); rows[b]['raw_ct_adj'] = c.get('raw_ct_adj')
    # ---- realized wOBA / PA ----
    if Y == 2026:
        lb = json.load(open(os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')))
        for r in lb:
            if r.get('mlbId') is not None and str(int(r['mlbId'])) in rows and r.get('team') not in ('ROC', 'AAA') and not str(r.get('team', '')).endswith('TM'):
                rows[str(int(r['mlbId']))]['woba'] = r.get('wOBA'); rows[str(int(r['mlbId']))]['pa'] = r.get('pa') or 0
        md = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
        lgrpa = (md.get('gutsConstants') or {}).get('lgRPA', 0.118)
    else:
        ty = A.target_y(Y)
        for b, (num, den) in ty.items():
            if b in rows:
                rows[b]['woba'] = (num / den) if den else None; rows[b]['pa'] = den
        lgrpa = LGRPA[Y]
    for r in rows.values():
        r.setdefault('woba', None); r.setdefault('pa', 0)
    # ---- displayed BB+ (process_data recipe, prior-free) ----
    pool = [r for r in rows.values() if r['nbip'] >= BB_MIN_BIP and r['xwcon'] is not None and r['ev95'] is not None]
    lg_xc = sum(r['xwcon'] * r['nbip'] for r in pool) / sum(r['nbip'] for r in pool)
    lg_ev = sum(r['ev95'] * r['nbip'] for r in pool) / sum(r['nbip'] for r in pool)
    bx = [100 * r['ev95'] / lg_ev for r in pool if r['nbip'] >= BB_BETA_GATE_BIP]
    by = [100 * r['xwcon'] / lg_xc for r in pool if r['nbip'] >= BB_BETA_GATE_BIP]
    beta = BB_BETA_FROZEN
    if len(bx) >= BB_BETA_MIN_POOL:
        mx, my = np.mean(bx), np.mean(by); cand = float(np.sum((np.array(bx) - mx) * (np.array(by) - my)) / np.sum((np.array(bx) - mx) ** 2))
        if BB_BETA_BAND[0] <= cand <= BB_BETA_BAND[1]:
            beta = cand
    for r in pool:
        con = 100 * r['xwcon'] / lg_xc; evp = 100 + (100 * r['ev95'] / lg_ev - 100) * beta
        con_adj = (r['nbip'] * con + BB_N0_CON * 100) / (r['nbip'] + BB_N0_CON)
        ev_adj = (r['nbip'] * evp + BB_N0_EV * 100) / (r['nbip'] + BB_N0_EV)
        r['bbPlus'] = 100 + (BB_W_CON * con_adj + BB_W_EV * ev_adj - 100) * BB_SLOPE_MATCH
    # realized channel percents
    lg_wc = sum(r['wobacon'] * r['nbip'] for r in rows.values() if r['wobacon'] is not None) / sum(r['nbip'] for r in rows.values() if r['wobacon'] is not None)
    lg_ct = sum(r['contact'] * r['n_sw'] for r in rows.values() if r['contact'] is not None) / sum(r['n_sw'] for r in rows.values() if r['contact'] is not None)
    sdpool = [r['raw_sd'] for r in rows.values() if r['raw_sd'] is not None and r['n_dec'] >= sd.MIN_HITTER_DECISIONS]
    lg_sd = float(np.mean(sdpool))
    ctpool = [r['raw_ct_adj'] for r in rows.values() if r.get('raw_ct_adj') is not None and r['n_sw'] >= ct.MIN_HITTER_SWINGS]
    lg_cta = float(np.mean(ctpool)) if ctpool else None
    for r in rows.values():
        r['real_ct_adj'] = 100 * r['raw_ct_adj'] / lg_cta if (r.get('raw_ct_adj') is not None and lg_cta) else None
        r['real_bb'] = 100 * r['wobacon'] / lg_wc if r['wobacon'] is not None else None
        r['real_bbx'] = 100 * r['xwcon'] / lg_xc if r['xwcon'] is not None else None
        r['real_ct'] = 100 * r['contact'] / lg_ct if r['contact'] is not None else None
        r['real_sd'] = 100 * r['raw_sd'] / lg_sd if (r['raw_sd'] is not None and abs(lg_sd) > 1e-9) else None
        r['real_hp'] = 100 + 100 * (r['woba'] - lg) / (sc * lgrpa) if r['woba'] is not None else None
    # ---- displayed Hitter+ (production construction, realized-wOBA scale match) ----
    pa_q = 502 if Y != 2026 else 430
    q = [r for r in rows.values() if r['pa'] >= pa_q and all(r.get(k) is not None for k in ('bbPlus', 'sdPlus', 'ctPlus', 'real_hp'))]
    if len(q) >= 30:
        m = {k: float(np.mean([r[k] for r in q])) for k in ('bbPlus', 'sdPlus', 'ctPlus')}
        s = {k: float(np.std([r[k] for r in q])) for k in ('bbPlus', 'sdPlus', 'ctPlus')}
        def zc(r):
            return sum(w * (r[k] - m[k]) / s[k] for w, k in zip(HP_W, ('bbPlus', 'sdPlus', 'ctPlus')))
        zq = np.array([zc(r) for r in q]); yq = np.array([r['real_hp'] for r in q])
        rr = float(np.corrcoef(zq, yq)[0, 1]); scale = rr * float(np.std(yq)) / float(np.std(zq))
        for r in rows.values():
            r['hitterPlus'] = 100 + scale * zc(r) if all(r.get(k) is not None for k in ('bbPlus', 'sdPlus', 'ctPlus')) else None
        print(f"  {Y}: hitters {len(rows)}, BB+ pool {len(pool)} (beta {beta:.3f}), Hitter+ pool {len(q)} (live r {rr:.3f}, scale {scale:.2f}), lgRPA {lgrpa:.4f}", flush=True)
    else:
        for r in rows.values(): r['hitterPlus'] = None
        print(f"  {Y}: hitters {len(rows)}, Hitter+ pool too thin ({len(q)})", flush=True)
    del P; gc.collect()
    return rows

METRICS = {  # displayed key, realized key, floor key in Y, floor, floor key in Y+1, floor, trait keys
    'bbPlus': ('real_bb', 'nbip', 100, 'nbip', 100, ['gb_pct', 'mean_la']),
    # the channel's own quantity next season (xwOBAcon percent, unshrunk): drift only, no BABIP luck
    'bbPlus_x': ('real_bbx', 'nbip', 100, 'nbip', 100, ['gb_pct', 'mean_la']),
    'ctPlus': ('real_ct', 'n_sw', 200, 'n_sw', 200, ['swing_pct', 'chase']),
    # same metric next season, location-adjusted like the display (the forward-factor view)
    'ctPlus_adj': ('real_ct_adj', 'n_sw', 200, 'n_sw', 200, ['swing_pct', 'chase']),
    'sdPlus': ('real_sd', 'n_dec', 200, 'n_dec', 200, ['swing_pct', 'chase']),
    'hitterPlus': ('real_hp', 'pa', 300, 'pa', 200, ['gb_pct']),
}

def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); sl, ic = np.polyfit(x, y, 1); return float(sl), float(ic), float(np.corrcoef(x, y)[0, 1])

def main():
    lb = json.load(open(os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')))
    name2id = {r['hitter']: str(int(r['mlbId'])) for r in lb if r.get('mlbId') is not None and r.get('hitter')}
    F = {}
    for Y in SEASONS:
        print(f"season {Y} ...", flush=True); F[Y] = season_frame(Y, name2id)
    out = {}
    for met, (rk, fk, fmin, fk1, fmin1, traits) in METRICS.items():
        print(f"\n===== {met}: realized(Y+1) on displayed(Y) =====")
        pooled = []   # (pair, disp, real, bats, traits...)
        res = {'pairs': {}, 'deciles': {}, 'hand': {}, 'traits': {}}
        for Y, Y1 in PAIRS:
            a, b = F[Y], F[Y1]
            dk = {'ctPlus_adj': 'ctPlus', 'bbPlus_x': 'bbPlus'}.get(met, met)
            ks = [k for k in a if k in b and a[k].get(dk) is not None and b[k].get(rk) is not None and a[k][fk] >= fmin and b[k][fk1] >= fmin1]
            if len(ks) < 40:
                print(f"  {Y}->{Y1}: n {len(ks)} too thin"); continue
            x = [a[k][dk] for k in ks]; y = [b[k][rk] for k in ks]
            sl, ic, r = ols(x, y)
            res['pairs'][f'{Y}-{Y1}'] = dict(slope=sl, intercept=ic, r=r, n=len(ks), mean_disp=float(np.mean(x)), mean_real=float(np.mean(y)))
            print(f"  {Y}->{Y1}: n {len(ks):4d}  slope {sl:.3f}  r {r:.3f}  mean disp {np.mean(x):6.1f}  mean real {np.mean(y):6.1f}")
            dec = np.floor(np.argsort(np.argsort(x)) * 10 / len(x)).astype(int)
            for k, xi, yi, d in zip(ks, x, y, dec):
                pooled.append((f'{Y}-{Y1}', xi, yi, a[k].get('bats'), int(d), {t: a[k].get(t) for t in traits}))
        same = []
        for Y in SEASONS:
            a = F[Y]; ks = [k for k in a if a[k].get(dk) is not None and a[k].get(rk) is not None and a[k][fk] >= fmin]
            if len(ks) >= 40:
                same.append(ols([a[k][dk] for k in ks], [a[k][rk] for k in ks])[0])
        res['same_season_slopes'] = same
        print(f"  same-season (descriptive) slope of realized on displayed: {' '.join(f'{v:.3f}' for v in same)}  mean {np.mean(same):.3f}")
        sls = [v['slope'] for v in res['pairs'].values()]
        print(f"  mean slope {np.mean(sls):.3f} (range {min(sls):.3f}-{max(sls):.3f})")
        # deciles pooled
        print("  decile   n   mean displayed   mean realized   realized - displayed")
        for d in range(10):
            sub = [p for p in pooled if p[4] == d]
            md, mr = np.mean([p[1] for p in sub]), np.mean([p[2] for p in sub])
            res['deciles'][d] = dict(n=len(sub), disp=float(md), real=float(mr))
            print(f"    {d + 1:2d}   {len(sub):4d}   {md:8.1f}        {mr:8.1f}        {mr - md:+7.1f}")
        # hand residuals, per pair then paired
        print("  hand: residual (realized - displayed) by bats, per pair; L-R with paired SE")
        diffs = []
        for pr in res['pairs']:
            g = defaultdict(list)
            for p in pooled:
                if p[0] == pr and p[3] in ('L', 'R', 'S'):
                    g[p[3]].append(p[2] - p[1])
            mL, mR, mS = (np.mean(g['L']) if g['L'] else np.nan), (np.mean(g['R']) if g['R'] else np.nan), (np.mean(g['S']) if g['S'] else np.nan)
            se = math.sqrt(np.var(g['L'], ddof=1) / len(g['L']) + np.var(g['R'], ddof=1) / len(g['R'])) if len(g['L']) > 2 and len(g['R']) > 2 else np.nan
            diffs.append(mL - mR); res['hand'][pr] = dict(L=float(mL), R=float(mR), S=float(mS), nL=len(g['L']), nR=len(g['R']), nS=len(g['S']), LmR=float(mL - mR), se=float(se))
            print(f"    {pr}: L {mL:+6.2f} (n {len(g['L'])})  R {mR:+6.2f} (n {len(g['R'])})  S {mS:+6.2f} (n {len(g['S'])})  L-R {mL - mR:+6.2f}  SE {se:.2f}")
        dd = [d for d in diffs if np.isfinite(d)]; mu = np.mean(dd); se_p = np.std(dd, ddof=1) / math.sqrt(len(dd))
        res['hand']['paired'] = dict(mean_LmR=float(mu), se=float(se_p), n=len(dd), wins=int(sum(1 for d in dd if d > 0)))
        print(f"    paired L-R: mean {mu:+.2f}  SE {se_p:.2f}  t {mu / se_p:+.1f}  (L above R in {sum(1 for d in dd if d > 0)}/{len(dd)})")
        # residual vs trait
        for t in traits:
            rs = []
            for pr in res['pairs']:
                sub = [p for p in pooled if p[0] == pr and p[5].get(t) is not None]
                if len(sub) < 40: continue
                rs.append(float(np.corrcoef([p[2] - p[1] for p in sub], [p[5][t] for p in sub])[0, 1]))
            res['traits'][t] = rs
            print(f"  residual vs {t}: r per pair {' '.join(f'{v:+.3f}' for v in rs)}  mean {np.mean(rs):+.3f}")
        out[met] = res
    json.dump(out, open(os.path.join(ROOT, 'data', '_hitter_calibration_audit.json'), 'w'), indent=1, default=float)
    print('\nwrote data/_hitter_calibration_audit.json')

if __name__ == '__main__':
    main()
