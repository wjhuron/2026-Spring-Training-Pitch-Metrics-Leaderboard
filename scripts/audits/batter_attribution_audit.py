"""batter_attribution_audit.py — sheets-wide check that each PA's batter (and bat
side) matches the current feed. READ-ONLY.

Catches whole-PA batter-attribution errors (e.g. 824290_076 labeled Rengifo but the
feed says Joey Ortiz) that the mid-PA handedness audit and the pitch-sequence audit
don't see. For every PA in the MLB tabs, compares the sheet's Batter/Bats against
the feed's matchup.batter/batSide. Writes ~/Downloads/batter_mismatches_2026.csv.

Usage: python3 scripts/audits/batter_attribution_audit.py
"""
import os, sys, csv, time, unicodedata, requests
from collections import defaultdict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
from concurrent.futures import ThreadPoolExecutor


def norm_last(name):
    """Normalize to a comparable last-name token."""
    if not name:
        return ''
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    s = s.replace('.', '').replace(',', ' ')
    for suf in [' jr', ' sr', ' ii', ' iii', ' iv']:
        s = s.replace(suf, '')
    return s.strip()


def last_token_sheet(name):
    # sheet format "Last, First" -> last
    if ',' in str(name):
        return norm_last(str(name).split(',')[0])
    return norm_last(str(name).split()[-1]) if name else ''


def last_token_feed(name):
    # feed "First Last" -> last word
    n = norm_last(name)
    return n.split()[-1] if n else ''


def fetch(pk):
    try:
        return pk, requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=30).json()
    except Exception:
        return pk, None


def main():
    gc = gspread.service_account()
    # sheet: per (pk, ab) -> (batter, bats, pitcher, tab)
    sheet_pa = {}
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
                pa = pid.split('_')
                if len(pa) != 3:
                    continue
                try:
                    pk, ab = int(pa[0]), int(pa[1])
                except ValueError:
                    continue
                if (pk, ab) not in sheet_pa:
                    sheet_pa[(pk, ab)] = (r[ci['Batter']] if 'Batter' in ci else '',
                                          r[ci['Bats']] if 'Bats' in ci else '',
                                          r[ci['Pitcher']] if 'Pitcher' in ci else '', t)
            print(f"  [{label}/{ws.title}] read", flush=True)

    pks = sorted({pk for pk, ab in sheet_pa})
    print(f"\nfetching {len(pks)} feeds ...", flush=True)
    feeds = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for pk, js in ex.map(fetch, pks):
            feeds[pk] = js

    rows = []
    for (pk, ab), (sbat, sbats, spit, tab) in sheet_pa.items():
        js = feeds.get(pk)
        if not js:
            continue
        plays = js.get('liveData', {}).get('plays', {}).get('allPlays', [])
        if ab - 1 >= len(plays):
            continue
        play = plays[ab - 1]
        fbat = play.get('matchup', {}).get('batter', {}).get('fullName', '')
        fside = play.get('matchup', {}).get('batSide', {}).get('code', '')
        name_bad = last_token_sheet(sbat) != last_token_feed(fbat)
        hand_bad = (sbats and fside and sbats != fside)
        if name_bad:
            rows.append({'PA': f"{pk}_{ab:03d}", 'tab': tab, 'pitcher': spit,
                         'sheet_batter': sbat, 'feed_batter': fbat,
                         'sheet_bats': sbats, 'feed_bats': fside,
                         'issue': 'BATTER' + ('+HAND' if hand_bad else '')})

    out = os.path.expanduser('~/Downloads/batter_mismatches_2026.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['PA', 'tab', 'pitcher', 'sheet_batter', 'feed_batter',
                                          'sheet_bats', 'feed_bats', 'issue'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['PA']):
            w.writerow(r)

    print(f"\nwrote {out}")
    print(f"\n=== BATTER-ATTRIBUTION MISMATCHES: {len(rows)} PAs ===")
    # is it concentrated in EP/position-player blowout innings?
    ep_like = sum(1 for r in rows if 'position' in r['pitcher'].lower())  # placeholder
    for r in sorted(rows, key=lambda x: x['PA'])[:60]:
        print(f"  {r['PA']} [{r['tab']}] vs {r['pitcher']:20s}: sheet='{r['sheet_batter']}'({r['sheet_bats']}) feed='{r['feed_batter']}'({r['feed_bats']})")
    if len(rows) > 60:
        print(f"  ... and {len(rows)-60} more (see CSV)")


if __name__ == '__main__':
    main()
