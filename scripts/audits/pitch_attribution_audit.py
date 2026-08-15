"""pitch_attribution_audit.py — perfect per-pitch pitcher + batter (+ handedness)
against the current feed, for every pitch of every MLB game. READ-ONLY.

Reproduces Pitcher2026's own mid-PA attribution (seed with the outgoing player of
the first sub, advance on pitching_substitution / pinch-hitter events) so each
pitch is credited to whoever was actually in at that moment — then compares to the
sheet. Names matched by accent-folded token SETS ("Del Castillo, Adrian" ==
"Adrian Del Castillo"), so only true person mismatches flag. Handedness compared
after resolving switch-hitters to the side vs the pitcher.

Writes ~/Downloads/attribution_mismatches_2026.csv.

Usage: python3 scripts/audits/pitch_attribution_audit.py [--reread]
"""
import os, sys, csv, time, pickle, unicodedata, requests
from collections import defaultdict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
import scrapers.Pitcher2026 as Pitcher2026
from concurrent.futures import ThreadPoolExecutor

dl = Pitcher2026.BaseballSavantFocusedDownloader()
SHEET_CACHE = os.path.join(ROOT, 'data', '_attr_sheet.pkl')


def toks(name):
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    for ch in ',.-':
        s = s.replace(ch, ' ')
    return frozenset(p for p in s.split() if len(p) > 1 and p not in ('jr', 'sr', 'ii', 'iii', 'iv'))


def per_pitch_correct(pk, js):
    """Per-pitch pitcher/batter, robust to the feed omitting replacedPlayer.

    Pitcher: track the running pitcher PER SIDE across the whole game. A mid-PA
    pitching change only names the entering pitcher, so the outgoing pitcher is
    simply whoever was already throwing for that side (carried from the prior PA)
    — this reproduces the hand-attribution Wally does (each pitch to whoever was
    actually in). Batter resets each PA to matchup.batter (the finisher); a mid-PA
    pinch-hitter's earlier pitches keep the original via replacedPlayer/description.
    """
    gd = js.get('gameData', {})
    out = {}
    cur_by_side = {}  # halfInning -> pitcher_id currently throwing for that side
    for play in js.get('liveData', {}).get('plays', {}).get('allPlays', []):
        side = play.get('about', {}).get('halfInning', 'top')
        ab = play.get('atBatIndex', 0) + 1
        m = play.get('matchup', {})
        ev = play.get('playEvents', [])
        # pitcher: carry the running pitcher for this side; fall back to the
        # PA's finisher only for a side's very first PA.
        cp = cur_by_side.get(side, m.get('pitcher', {}).get('id'))
        # batter: seed with finisher; if a mid-PA pinch-hitter exists, back it out
        # to the original batter (replacedPlayer, else "pinch-hits for X" in desc).
        cb = m.get('batter', {}).get('id')
        for e in ev:
            if dl._is_pinch_hitter(e):
                rp = (e.get('replacedPlayer') or {}).get('id')
                if rp:
                    cb = rp
                else:
                    desc = (e.get('details', {}) or {}).get('description', '') or ''
                    who = None
                    if 'replaces' in desc.lower():
                        who = desc.lower().split('replaces', 1)[1].strip().rstrip('.')
                    elif 'pinch-hits for' in desc.lower():
                        who = desc.lower().split('pinch-hits for', 1)[1].strip().rstrip('.')
                    if who:
                        for _pid, _p in gd.get('players', {}).items():
                            if toks(_p.get('fullName', '')) == toks(who):
                                cb = _p.get('id'); break
                break
        for e in ev:
            if e.get('type') == 'action':
                if dl._is_pitching_sub(e):
                    cp = (e.get('player') or {}).get('id') or cp
                elif dl._is_pinch_hitter(e):
                    cb = (e.get('player') or {}).get('id') or cb
            if e.get('isPitch'):
                th = dl._throws_of(cp, gd)  # registered code; 'S' for switch pitchers
                # bats: prefer the feed's ACTUAL side for the PA's finisher (handles
                # switch hitters who batted the "wrong" way); resolve only a mid-PA
                # original batter naively.
                if cb == m.get('batter', {}).get('id'):
                    ba = (m.get('batSide', {}) or {}).get('code', '')
                else:
                    ba = dl._batside_of(cb, gd, th)
                out[f"{pk}_{ab:03d}_{e['pitchNumber']:02d}"] = (
                    dl.get_player_name(cp, '', gd), dl.get_player_name(cb, '', gd), th, ba)
        cur_by_side[side] = cp  # finisher carries to this side's next PA
    return out


def fetch(pk):
    try:
        return pk, requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=30).json()
    except Exception:
        return pk, None


def main():
    # ---- read sheets: PitchID -> (Pitcher, Batter, Throws, Bats) ----
    if os.path.exists(SHEET_CACHE) and '--reread' not in sys.argv:
        sheet = pickle.load(open(SHEET_CACHE, 'rb'))
        print(f"loaded sheet cache ({len(sheet)} pitches)", flush=True)
    else:
        gc = gspread.service_account()
        sheet = {}
        for label, sid in B.SPREADSHEET_IDS.items():
            sh = gc.open_by_key(sid)
            for ws in sh.worksheets():
                t = ws.title.upper()
                if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                    continue
                time.sleep(0.7)
                vals = ws.get_all_values()
                if not vals or len(vals) < 2 or 'PitchID' not in vals[0]:
                    continue
                ci = {n: j for j, n in enumerate(vals[0]) if n}
                for r in vals[1:]:
                    pid = str(r[ci['PitchID']]) if ci['PitchID'] < len(r) else ''
                    if pid.count('_') != 2:
                        continue
                    sheet[pid] = (r[ci['Pitcher']] if 'Pitcher' in ci else '',
                                  r[ci['Batter']] if 'Batter' in ci else '',
                                  r[ci['Throws']] if 'Throws' in ci else '',
                                  r[ci['Bats']] if 'Bats' in ci else '')
                print(f"  [{label}/{ws.title}] read", flush=True)
        pickle.dump(sheet, open(SHEET_CACHE, 'wb'))

    pks = sorted({int(pid.split('_')[0]) for pid in sheet})
    print(f"\nfetching {len(pks)} feeds ...", flush=True)
    feeds = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (pk, js) in enumerate(ex.map(fetch, pks)):
            feeds[pk] = js
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(pks)}", flush=True)

    # ---- compare ----
    rows = []
    cats = Counter()
    for pk in pks:
        js = feeds.get(pk)
        if not js:
            continue
        correct = per_pitch_correct(pk, js)
        for pid, (cp, cb, cth, cba) in correct.items():
            s = sheet.get(pid)
            if s is None:
                continue
            sp, sb, sth, sba = s
            tb_s, tb_c = toks(sb), toks(cb)
            tp_s, tp_c = toks(sp), toks(cp)
            # real identity error = no shared name token (different person). A shared
            # token with a different set = a name/nickname variant (Cauley/Cameron).
            bat_bad = bool(tb_s and tb_c and not (tb_s & tb_c))
            pit_bad = bool(tp_s and tp_c and not (tp_s & tp_c))
            bat_var = (tb_s != tb_c) and not bat_bad
            pit_var = (tp_s != tp_c) and not pit_bad
            # handedness: only when the feed value is a concrete L/R (skip switch 'S')
            hand_bad = ((cba in ('L', 'R') and sba and sba != cba) or
                        (cth in ('L', 'R') and sth and sth != cth))
            if bat_bad or pit_bad or hand_bad:
                issue = '+'.join([x for x, on in
                                  [('BATTER', bat_bad), ('PITCHER', pit_bad), ('HAND', hand_bad)] if on])
                cats[issue] += 1
                rows.append({'PitchID': pid,
                             'sheet_pitcher': sp, 'feed_pitcher': cp,
                             'sheet_batter': sb, 'feed_batter': cb,
                             'sheet_bats': sba, 'feed_bats': cba,
                             'sheet_throws': sth, 'feed_throws': cth,
                             'issue': issue})

    out = os.path.expanduser('~/Downloads/attribution_mismatches_2026.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['PitchID', 'issue', 'sheet_pitcher', 'feed_pitcher',
                                          'sheet_batter', 'feed_batter', 'sheet_bats', 'feed_bats',
                                          'sheet_throws', 'feed_throws'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['PitchID']):
            w.writerow(r)

    # PA-level rollup for readability
    bypa = defaultdict(list)
    for r in rows:
        bypa['_'.join(r['PitchID'].split('_')[:2])].append(r)
    print(f"\nwrote {out}")
    print(f"\n=== ATTRIBUTION MISMATCHES: {len(rows)} pitches across {len(bypa)} PAs ===")
    print(f"by issue: {dict(cats)}")
    # show PAs with name (batter/pitcher) mismatches, most impactful
    name_pas = {pa: rs for pa, rs in bypa.items() if any('BATTER' in r['issue'] or 'PITCHER' in r['issue'] for r in rs)}
    print(f"\nPAs with a BATTER or PITCHER identity error: {len(name_pas)}")
    for pa in sorted(name_pas)[:40]:
        r = name_pas[pa][0]
        who = []
        if toks(r['sheet_batter']) != toks(r['feed_batter']):
            who.append(f"batter '{r['sheet_batter']}'->'{r['feed_batter']}'")
        if toks(r['sheet_pitcher']) != toks(r['feed_pitcher']):
            who.append(f"pitcher '{r['sheet_pitcher']}'->'{r['feed_pitcher']}'")
        print(f"  {pa} ({len(name_pas[pa])}p): {'; '.join(who)}")
    if len(name_pas) > 40:
        print(f"  ... and {len(name_pas)-40} more")


if __name__ == '__main__':
    main()
