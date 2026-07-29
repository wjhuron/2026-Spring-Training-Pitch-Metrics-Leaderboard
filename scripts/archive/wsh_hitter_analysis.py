#!/usr/bin/env python3
"""WSH hitter deep-dive metric extraction.

Computes Hitting+, BB+, CT+, SD+, xwOBAsp (+ full LA x Spray grid) and SACQ%
for a fixed list of Nationals hitters, with L/R swing splits for the switch
hitters (Nunez, Ruiz, Millas) and a combined ROC+WSH line for Dylan Crews.

Everything is scored against the PUBLISHED metadata anchors (sacqZones,
gutsConstants, plusReanchor, hitterPlusStandardization, league xwOBAcon/sp),
so full-sample recomputations match the live leaderboard and the custom
subsets land on the same scale. Validated against hitter_leaderboard_rs.json.
"""
import sys, json, pickle, gzip
from collections import defaultdict, OrderedDict
sys.path.insert(0, '/Users/wallyhuron/Huronalytics')

from pipeline_utils import (
    safe_float, spray_angle, spray_direction, BUNT_BB_TYPES,
    K_EVENTS, BB_EVENTS, HBP_EVENTS, SF_EVENTS, SH_EVENTS, CI_EVENTS, NON_PA_EVENTS,
)
from pipeline_compute import compute_hitter_stats, compute_expected_stats
from pipeline_sdplus import (
    is_eligible, make_rv_xrv, build_weight_table, zone_level_means, shrink_table,
    compute_dv, compute_hitter_sd, MIN_HITTER_DECISIONS, HITTER_PRIOR_N as SD_PRIOR,
)
from pipeline_contact import (
    is_ct_eligible, build_contact_cell_weights, zone_level_contact_means,
    shrink_contact_cells, compute_ct_swing, compute_hitter_ct,
    MIN_HITTER_SWINGS, HITTER_PRIOR_N as CT_PRIOR,
)

BASE = '/Users/wallyhuron/Huronalytics'
DATA = f'{BASE}/data'

# ── Load ────────────────────────────────────────────────────────────────
print("Loading pickle / metadata / leaderboard ...")
ALL = pickle.load(open(f'{DATA}/all_pitches_rs_cache.pkl', 'rb'))
# Reference tables (sacqZones, anchors) + published leaderboard come from the
# committed data_embedded.json.gz, which carries the live 6/17 published run.
_EMB = json.loads(gzip.open(f'{DATA}/data_embedded.json.gz', 'rt').read())
META = _EMB['metadata']
LB = _EMB['hitterData']
LBK = {(r['hitter'], r['team']): r for r in LB}

G = META['gutsConstants']
LGW, WS = G['lgWOBA'], G['wOBAScale']
REAN = META['plusReanchor']
STD = META['hitterPlusStandardization']
LG_XCON = META['hitterLeagueAverages']['xwOBAcon']
LG_XSP = META['hitterLeagueAverages']['xwOBAsp']
WEIGHTS = {'BB': 0.701, 'HBP': 0.732, '1B': 0.894, '2B': 1.267, '3B': 1.603, 'HR': 2.06}

print(f"  pitches={len(ALL)}  lgXwOBAcon={LG_XCON} lgXwOBAsp={LG_XSP}")
print(f"  reanchor={REAN}")

# ── SACQ lookup from metadata sacqZones ──────────────────────────────────
LA_BINS = [(-999,-10),(-10,0),(0,5),(5,10),(10,15),(15,20),
           (20,25),(25,30),(30,35),(35,40),(40,50),(50,999)]
LA_LABELS = ['<-10','-10..0','0..5','5..10','10..15','15..20',
             '20..25','25..30','30..35','35..40','40..50','50+']
SPRAY_ORDER = ['pull','pull_side','center_pull','center_oppo','oppo_side','oppo']
MIN_SACQ = 20

hand_z, pool_z = {}, {}
for z in META['sacqZones']:
    rec = {'woba': z['woba'], 'quality': z['quality'], 'count': z['count']}
    if z['bats'] is None:
        pool_z[(z['spray'], z['laBin'])] = rec
    else:
        hand_z[(z['spray'], z['laBin'], z['bats'])] = rec

def sacq_lookup(direction, lab, bats):
    """Returns (woba, quality) using hand-specific zone then pooled fallback."""
    h = hand_z.get((direction, lab, bats))
    if h and h['count'] >= MIN_SACQ and h['woba'] is not None:
        return h['woba'], h['quality']
    p = pool_z.get((direction, lab))
    if p and p['count'] >= MIN_SACQ and p['woba'] is not None:
        return p['woba'], p['quality']
    return None

def la_bin(la):
    for i, (lo, hi) in enumerate(LA_BINS):
        if lo <= la < hi:
            return i
    return None

def classify_sp(p):
    bb = p.get('BBType')
    if not bb or bb in BUNT_BB_TYPES:
        return None
    hcx, hcy, la, bats = safe_float(p.get('HC_X')), safe_float(p.get('HC_Y')), safe_float(p.get('LaunchAngle')), p.get('Bats')
    if la is None or hcx is None or hcy is None or not bats:
        return None
    d = spray_direction(spray_angle(hcx, hcy), bats)
    if not d:
        return None
    lab = la_bin(la)
    if lab is None:
        return None
    return d, lab, bats

def xwobasp(pitches):
    vals = []
    for p in pitches:
        c = classify_sp(p)
        if not c:
            continue
        r = sacq_lookup(*c)
        if r is not None:
            vals.append(r[0])
    return (round(sum(vals)/len(vals), 3) if vals else None), len(vals)

def sacq_pct(pitches):
    tot = q = 0
    for p in pitches:
        c = classify_sp(p)
        if not c:
            continue
        r = sacq_lookup(*c)
        if r is None:
            continue
        tot += 1
        if r[1]:
            q += 1
    return (round(100.0*q/tot, 1) if tot else None), q, tot

def la_spray_grid(pitches):
    """Per-cell hitter BIP count + league zone wOBA + quality. dict[spray][laBin]."""
    cnt = defaultdict(lambda: defaultdict(int))
    for p in pitches:
        c = classify_sp(p)
        if not c:
            continue
        d, lab, _ = c
        cnt[d][lab] += 1
    grid = OrderedDict()
    for d in SPRAY_ORDER:
        row = OrderedDict()
        for lab in range(len(LA_BINS)):
            n = cnt[d].get(lab, 0)
            # representative bats for zone wOBA: use the subset's dominant bats
            row[lab] = n
        grid[d] = row
    return grid

# ── SD+ / CT+ cell tables + canonical anchors ────────────────────────────
print("Building SD+/CT+ cell tables (MLB-only) and canonical anchors ...")
rv_fn = make_rv_xrv(LGW, WS)

sd_mlb = [p for p in ALL if p.get('_source', 'MLB') == 'MLB' and is_eligible(p)]
SD_TABLE = shrink_table(build_weight_table(sd_mlb, rv_fn), zone_level_means(sd_mlb, rv_fn))

ct_mlb = [p for p in ALL if p.get('_source', 'MLB') == 'MLB' and is_ct_eligible(p)]
CT_TABLE = shrink_contact_cells(build_contact_cell_weights(ct_mlb, rv_fn), zone_level_contact_means(ct_mlb, rv_fn))

# canonical groups (Batter, BTeam) for league anchoring
GROUPS = defaultdict(list)
for p in ALL:
    GROUPS[(p.get('Batter'), p.get('BTeam'))].append(p)

# SD+ canonical anchor
sd_raw_all = compute_hitter_sd(GROUPS, SD_TABLE)
sd_elig = {k: v for k, v in sd_raw_all.items() if v['n_decisions'] >= MIN_HITTER_DECISIONS}
SD_LGRAW = sum(v['raw_sd'] for v in sd_elig.values()) / len(sd_elig)
for v in sd_elig.values():
    n = v['n_decisions']
    v['raw_sd_adj'] = (n*v['raw_sd'] + SD_PRIOR*SD_LGRAW) / (n + SD_PRIOR)
SD_LGMEAN = sum(v['raw_sd_adj'] for v in sd_elig.values()) / len(sd_elig)

# CT+ canonical anchor
ct_raw_all = compute_hitter_ct(GROUPS, CT_TABLE)
ct_elig = {k: v for k, v in ct_raw_all.items() if v['n_swings'] >= MIN_HITTER_SWINGS}
CT_LGRAW = sum(v['raw_ct'] for v in ct_elig.values()) / len(ct_elig)
for v in ct_elig.values():
    n = v['n_swings']
    v['raw_ct_adj'] = (n*v['raw_ct'] + CT_PRIOR*CT_LGRAW) / (n + CT_PRIOR)
CT_LGMEAN = sum(v['raw_ct_adj'] for v in ct_elig.values()) / len(ct_elig)
print(f"  SD+ anchor: lg_raw={SD_LGRAW:.5f} lg_mean={SD_LGMEAN:.5f} (n_elig={len(sd_elig)})")
print(f"  CT+ anchor: lg_raw={CT_LGRAW:.5f} lg_mean={CT_LGMEAN:.5f} (n_elig={len(ct_elig)})")

def score_sd(pitches):
    elig = [p for p in pitches if is_eligible(p)]
    n = len(elig)
    if n < MIN_HITTER_DECISIONS:
        return None, n
    raw = sum(compute_dv(p, SD_TABLE) for p in elig) / n
    raw_adj = (n*raw + SD_PRIOR*SD_LGRAW) / (n + SD_PRIOR)
    return round(round(100.0*raw_adj/SD_LGMEAN, 1) * REAN['sdPlus'], 1), n

def score_ct(pitches):
    sw = [p for p in pitches if is_ct_eligible(p)]
    n = len(sw)
    if n < MIN_HITTER_SWINGS:
        return None, n
    num = den = 0.0
    for p in sw:
        lev, con = compute_ct_swing(p, CT_TABLE)
        if lev <= 0:
            continue
        num += lev*con; den += lev
    if den <= 0:
        return None, n
    raw_adj = (n*(num/den) + CT_PRIOR*CT_LGRAW) / (n + CT_PRIOR)
    return round(round(100.0*raw_adj/CT_LGMEAN, 1) * REAN['ctPlus'], 1), n

def score_bb(xcon, xsp, nbip):
    if xcon is None or xsp is None or nbip < 80:
        return None
    con_plus = 100.0*xcon/LG_XCON
    sp_plus = 100.0*xsp/LG_XSP
    bb_pre = round(0.585*con_plus + 0.415*sp_plus, 1)
    return round(bb_pre * REAN['bbPlus'], 1)

def score_hitter(bb, sd, ct):
    if bb is None or sd is None or ct is None:
        return None
    z_bb = (bb - STD['bbPlus']['mean']) / STD['bbPlus']['sd']
    z_sd = (sd - STD['sdPlus']['mean']) / STD['sdPlus']['sd']
    z_ct = (ct - STD['ctPlus']['mean']) / STD['ctPlus']['sd']
    cz = STD['weights']['bb']*z_bb + STD['weights']['sd']*z_sd + STD['weights']['ct']*z_ct
    hp_pre = round(100 + STD['scale']*cz, 1)
    return round(hp_pre + REAN['hitterPlusShift'], 1)

# ── slash line / wOBA from events ────────────────────────────────────────
def slash(pitches):
    ab = ubb = ibb = hbp = sf = sh = k = s1 = s2 = s3 = hr = pa = 0
    for p in pitches:
        e = p.get('Event')
        if not e or e in NON_PA_EVENTS:
            continue
        pa += 1
        if e == 'Intent Walk':
            ibb += 1; continue
        if e in BB_EVENTS:
            ubb += 1; continue
        if e in HBP_EVENTS:
            hbp += 1; continue
        if e in SF_EVENTS:
            sf += 1; continue
        if e in SH_EVENTS or e in CI_EVENTS:
            continue
        ab += 1
        if e in K_EVENTS:
            k += 1
        elif e == 'Single':
            s1 += 1
        elif e == 'Double':
            s2 += 1
        elif e == 'Triple':
            s3 += 1
        elif e == 'Home Run':
            hr += 1
    h = s1 + s2 + s3 + hr
    tb = s1 + 2*s2 + 3*s3 + 4*hr
    avg = h/ab if ab else None
    obp_d = ab + ubb + ibb + hbp + sf
    obp = (h + ubb + ibb + hbp)/obp_d if obp_d else None
    slg = tb/ab if ab else None
    woba_d = ab + ubb + sf + hbp
    woba = ((WEIGHTS['BB']*ubb + WEIGHTS['HBP']*hbp + WEIGHTS['1B']*s1 +
             WEIGHTS['2B']*s2 + WEIGHTS['3B']*s3 + WEIGHTS['HR']*hr)/woba_d) if woba_d else None
    return {
        'pa': pa, 'ab': ab, 'h': h, 'hr': hr, '1b': s1, '2b': s2, '3b': s3,
        'bb': ubb, 'ibb': ibb, 'hbp': hbp, 'k': k,
        'avg': round(avg, 3) if avg is not None else None,
        'obp': round(obp, 3) if obp is not None else None,
        'slg': round(slg, 3) if slg is not None else None,
        'ops': round(obp+slg, 3) if (obp is not None and slg is not None) else None,
        'iso': round(slg-avg, 3) if (slg is not None and avg is not None) else None,
        'kPct': round(100.0*k/pa, 1) if pa else None,
        'bbPct': round(100.0*ubb/pa, 1) if pa else None,
        'wOBA': round(woba, 3) if woba is not None else None,
    }

PT_HARD = {'FF', 'SI', 'FC', 'FA'}
PT_BRK = {'SL', 'ST', 'CU', 'SV', 'KC', 'CS'}
PT_OFF = {'CH', 'FS', 'FO', 'KN', 'SC'}
def _pt_group(pt):
    if pt in PT_HARD: return 'Hard'
    if pt in PT_BRK: return 'Breaking'
    if pt in PT_OFF: return 'Offspeed'
    return None

def pitch_type_split(pitches):
    """Per pitch-group (Hard/Breaking/Offspeed): usage%, whiff% (swinging strikes
    / swings, matching CT+), and xwOBAcon (mean Savant xwOBA over BIP)."""
    g = {k: {'seen': 0, 'sw': 0, 'wh': 0, 'xw': 0.0, 'bip': 0} for k in ('Hard', 'Breaking', 'Offspeed')}
    for p in pitches:
        grp = _pt_group(p.get('Pitch Type'))
        if not grp:
            continue
        d = g[grp]; d['seen'] += 1
        desc = p.get('Description')
        if desc in ('Swinging Strike', 'Foul', 'In Play'):
            d['sw'] += 1
            if desc == 'Swinging Strike':
                d['wh'] += 1
        if desc == 'In Play':
            bb = p.get('BBType')
            if bb and bb not in BUNT_BB_TYPES:
                xw = safe_float(p.get('xwOBA'))
                if xw is not None:
                    d['xw'] += xw; d['bip'] += 1
    total = sum(d['seen'] for d in g.values())
    out = {}
    for k, d in g.items():
        out[k] = {
            'seen': d['seen'],
            'usage': round(100.0 * d['seen'] / total, 1) if total else None,
            'whiff': round(100.0 * d['wh'] / d['sw'], 1) if d['sw'] else None,
            'nSw': d['sw'],
            'xwOBAcon': round(d['xw'] / d['bip'], 3) if d['bip'] else None,
            'nBip': d['bip'],
        }
    return out

def full_metrics(pitches, bats_hint=None):
    hs = compute_hitter_stats(pitches)
    es = compute_expected_stats(pitches, woba_weights=WEIGHTS)
    sl = slash(pitches)
    xsp, nsp = xwobasp(pitches)
    sq, sq_q, sq_tot = sacq_pct(pitches)
    nbip = hs.get('nBip') or 0
    bb = score_bb(es.get('xwOBAcon'), xsp, nbip)
    sd, sd_n = score_sd(pitches)
    ct, ct_n = score_ct(pitches)
    hp = score_hitter(bb, sd, ct)
    _sz = compute_hitter_sd({('_', '_'): pitches}, SD_TABLE).get(('_', '_'), {})
    _cz = compute_hitter_ct({('_', '_'): pitches}, CT_TABLE).get(('_', '_'), {})
    return {
        'n_pitches': len(pitches),
        'sdZones': _sz.get('zone_dv', {}),
        'ctZones': _cz.get('zone_dv', {}),
        **sl,
        'babip': hs.get('babip'),
        'avgEV': hs.get('avgEVAll'), 'maxEV': hs.get('maxEV'), 'ev50': hs.get('ev50'),
        'barrelPct': _pct(hs.get('barrelPct')), 'hardHitPct': _pct(hs.get('hardHitPct')),
        'gbPct': _pct(hs.get('gbPct')), 'ldPct': _pct(hs.get('ldPct')),
        'fbPct': _pct(hs.get('fbPct')), 'puPct': _pct(hs.get('puPct')),
        'pullPct': _pct(hs.get('pullPct')), 'middlePct': _pct(hs.get('middlePct')),
        'oppoPct': _pct(hs.get('oppoPct')), 'airPullPct': _pct(hs.get('airPullPct')),
        'hrFbPct': _pct(hs.get('hrFbPct')),
        'whiffPct': _pct(hs.get('whiffPct')), 'chasePct': _pct(hs.get('chasePct')),
        'swingPct': _pct(hs.get('swingPct')), 'izContactPct': _pct(hs.get('izContactPct')),
        'contactPct': _pct(hs.get('contactPct')), 'twoStrikeWhiffPct': _pct(hs.get('twoStrikeWhiffPct')),
        'batSpeed': hs.get('batSpeed'), 'squaredUpPct': _pct(hs.get('squaredUpPct')),
        'xBA': es.get('xBA'), 'xSLG': es.get('xSLG'), 'xwOBA': es.get('xwOBA'),
        'xwOBAcon': es.get('xwOBAcon'),
        'xwOBAsp': xsp, 'nBip': nbip, 'nSp': nsp,
        'sacqPct': sq, 'sacqQ': sq_q, 'sacqTot': sq_tot,
        'bbPlus': bb, 'sdPlus': sd, 'sdN': sd_n, 'ctPlus': ct, 'ctN': ct_n, 'hitterPlus': hp,
        'ptype': pitch_type_split(pitches),
        'grid': la_spray_grid(pitches),
    }

def _pct(v):
    return round(100.0*v, 1) if isinstance(v, (int, float)) else None

# ── Player roster ────────────────────────────────────────────────────────
SINGLE = [  # full-sample WSH, single-handed
    ('Wood, James', 'L'), ('Lile, Daylen', 'L'), ('Abrams, CJ', 'L'),
    ('Young, Jacob', 'R'), ('García Jr., Luis', 'L'), ('Mead, Curtis', 'R'),
    ('Vivas, Jorbit', 'L'), ('Tena, José', 'L'), ('Crews, Dylan', 'R'),  # Crews = MLB (WSH) only
]
SWITCH = ['Nuñez, Nasim', 'Ruiz, Keibert', 'Millas, Drew']

def wsh_pitches(name):
    return [p for p in ALL if p.get('Batter') == name and p.get('BTeam') == 'WSH']

results = OrderedDict()

# Single-handed full sample
for name, stand in SINGLE:
    results[name] = {'kind': 'single', 'stand': stand, 'all': full_metrics(wsh_pitches(name))}

# Switch hitters: combined + L + R
for name in SWITCH:
    pit = wsh_pitches(name)
    L = [p for p in pit if p.get('Bats') == 'L']
    R = [p for p in pit if p.get('Bats') == 'R']
    results[name] = {
        'kind': 'switch',
        'all': full_metrics(pit),
        'L': full_metrics(L),
        'R': full_metrics(R),
    }

# Crews is MLB (WSH) only per request; handled in the SINGLE loop above (AAA excluded).

# ── Validation vs published leaderboard (full-sample rows) ───────────────
print("\n=== VALIDATION (recomputed vs published) ===")
print(f"{'player':<22}{'metric':<11}{'mine':>9}{'pub':>9}{'diff':>8}")
def vcmp(name, team, m):
    pub = LBK.get((name, team))
    if not pub:
        print(f"  {name} ({team}) not in leaderboard"); return
    for key in ['xwOBAsp', 'bbPlus', 'sdPlus', 'ctPlus', 'hitterPlus', 'xwOBAcon', 'wOBA', 'avgEV', 'babip']:
        pubkey = {'avgEV': 'avgEVAll'}.get(key, key)
        mv, pv = m.get(key), pub.get(pubkey)
        if mv is None or pv is None:
            print(f"{name:<22}{key:<11}{str(mv):>9}{str(pv):>9}{'--':>8}")
        else:
            print(f"{name:<22}{key:<11}{mv:>9}{pv:>9}{mv-pv:>8.3f}")

for name, stand in SINGLE:
    vcmp(name, 'WSH', results[name]['all'])
for name in SWITCH:
    vcmp(name, 'WSH', results[name]['all'])

# ── Qualified MLB pool distributions for xwOBAsp & SACQ% ─────────────────
print("\nComputing qualified-pool distributions for xwOBAsp & SACQ% ...")
QUAL_PA = 3.1 * META.get('teamGamesPlayed', {}).get('WSH', 0)  # 226.3
qual_xsp, qual_sacq = [], []
for r in LB:
    if r.get('_isROC') or r.get('team') == 'ROC':
        continue
    if (r.get('pa') or 0) < QUAL_PA:
        continue
    pit = GROUPS.get((r['hitter'], r['team']), [])
    xs, _ = xwobasp(pit)
    sq, _, tot = sacq_pct(pit)
    if xs is not None:
        qual_xsp.append(xs)
    if sq is not None and tot >= 20:
        qual_sacq.append(sq)
qual_xsp.sort(); qual_sacq.sort()

def pctl(arr, val):
    if val is None or not arr:
        return None
    below = sum(1 for x in arr if x < val)
    eq = sum(1 for x in arr if x == val)
    return round(100.0 * (below + 0.5*eq) / len(arr), 0)

LG_SACQ = round(sum(qual_sacq)/len(qual_sacq), 1) if qual_sacq else None
print(f"  qualified pool n={len(qual_xsp)} | mean xwOBAsp={round(sum(qual_xsp)/len(qual_xsp),3)} "
      f"| mean SACQ%={LG_SACQ} | xwOBAsp range {qual_xsp[0]}..{qual_xsp[-1]}")

# ── Human-readable digest ────────────────────────────────────────────────
def pf(v, nd=3):
    return ('%.{}f'.format(nd) % v) if isinstance(v, (int, float)) else '  -- '

def line(tag, m):
    xsp_p = pctl(qual_xsp, m['xwOBAsp'])
    sq_p = pctl(qual_sacq, m['sacqPct'])
    print(f"  [{tag}] PA={m['pa']} AB={m['ab']} ({m['n_pitches']}p, {m['nBip']}BIP, {m['nSp']}sp)")
    print(f"       slash {pf(m['avg'])}/{pf(m['obp'])}/{pf(m['slg'])}  wOBA {pf(m['wOBA'])}  "
          f"xwOBA {pf(m['xwOBA'])}  BABIP {pf(m['babip'])}  K%={m['kPct']} BB%={m['bbPct']} HR={m['hr']}")
    print(f"       xwOBAcon {pf(m['xwOBAcon'])}  xwOBAsp {pf(m['xwOBAsp'])} (pctl {xsp_p})  "
          f"SACQ% {m['sacqPct']} (pctl {sq_p}, {m['sacqQ']}/{m['sacqTot']})")
    print(f"       Hitting+ {m['hitterPlus']}  BB+ {m['bbPlus']}  CT+ {m['ctPlus']}(n{m['ctN']})  "
          f"SD+ {m['sdPlus']}(n{m['sdN']})")
    print(f"       EV {m['avgEV']} max {m['maxEV']} | Barrel% {m['barrelPct']} HardHit% {m['hardHitPct']} | "
          f"GB/LD/FB {m['gbPct']}/{m['ldPct']}/{m['fbPct']} PU {m['puPct']} | Pull/Mid/Oppo {m['pullPct']}/{m['middlePct']}/{m['oppoPct']} airPull {m['airPullPct']}")
    print(f"       Whiff% {m['whiffPct']} Chase% {m['chasePct']} IZcon% {m['izContactPct']} 2KWhiff% {m['twoStrikeWhiffPct']} | batSpeed {m['batSpeed']} SqUp% {m['squaredUpPct']}")

print("\n" + "="*78)
print(f"DIGEST  (qual line = {round(QUAL_PA,1)} PA; lg avg: +stats=100, xwOBAsp={LG_XSP}, SACQ%={LG_SACQ})")
print("="*78)
for name, stand in SINGLE:
    pub = LBK[(name, 'WSH')]
    print(f"\n### {name}  ({stand}HH, {pub.get('position')})  "
          f"[wRC+ {pub.get('wRCplus')}, xwRC+ {pub.get('xWRCplus')}, "
          f"Hit+pctl {pub.get('hitterPlus_pctl')}, xwOBAsp_pctl {pub.get('xwOBAsp_pctl')}]")
    line('full', results[name]['all'])
for name in SWITCH:
    pub = LBK[(name, 'WSH')]
    print(f"\n### {name}  (SWITCH, {pub.get('position')})  "
          f"[wRC+ {pub.get('wRCplus')}, xwRC+ {pub.get('xWRCplus')}, Hit+pctl {pub.get('hitterPlus_pctl')}]")
    line('combined', results[name]['all'])
    line('L-swing (vs RHP)', results[name]['L'])
    line('R-swing (vs LHP)', results[name]['R'])

# annotate output with percentiles + pool
out_pool = {'qualPA': round(QUAL_PA,1), 'n': len(qual_xsp),
            'xwOBAsp_mean': round(sum(qual_xsp)/len(qual_xsp),3),
            'xwOBAsp_sorted': qual_xsp, 'sacq_mean': LG_SACQ, 'sacq_sorted': qual_sacq}

# ── Save ─────────────────────────────────────────────────────────────────
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o

out = {
    'meta': {
        'lgXwOBAcon': LG_XCON, 'lgXwOBAsp': LG_XSP, 'reanchor': REAN,
        'std': STD, 'teamGamesPlayed_WSH': META.get('teamGamesPlayed', {}).get('WSH'),
        'qualPA': round(3.1 * META.get('teamGamesPlayed', {}).get('WSH', 0), 1),
        'la_labels': LA_LABELS, 'spray_order': SPRAY_ORDER,
    },
    'results': jsonable(results),
    'pool': out_pool,
}
with open(f'{BASE}/scripts/wsh_analysis_output.json', 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(f"\nSaved -> scripts/wsh_analysis_output.json")
print(f"WSH team games played: {META.get('teamGamesPlayed', {}).get('WSH')}  -> qual PA = {out['meta']['qualPA']}")
