"""show_8pa_details.py — print sheet-vs-feed detail for the 8 feed-revision MLB
PAs before any fix. READ-ONLY.

For each PA: the current feed sequence (pitch #, description, pre-pitch count,
velo, feed pitch type, event) side by side with the sheet's current rows, and
what the restore would do (insert / remove / reassign), including the proposed
Pitch Type for new pitches (EP for position-player pitchers, feed type for
Jansen).

Usage: python3 scripts/show_8pa_details.py
"""
import os, sys, time, requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B

# (game_pk, ab, workbook, tab, pitcher, EP override?)
CASES = [
    ('824295', 80, 'ALC2026', 'DET', 'Jansen (real pitcher)', False),
    ('824295', 81, 'ALC2026', 'DET', 'Jansen (real pitcher)', False),
    ('824290', 78, 'ALC2026', 'DET', 'Rogers (pos player)', True),
    ('824290', 79, 'ALC2026', 'DET', 'Rogers (pos player)', True),
    ('823867', 72, 'NLE2026', 'MIA', 'Mateo (pos player)', True),
    ('825081', 67, 'NLW2026', 'ARI', 'Sullivan (pos player)', True),
    ('823374', 78, 'NLC2026', 'PIT', 'Callihan (pos player)', True),
    ('823535', 75, 'ALE2026', 'NYY', 'Acuña (pos player)', True),
]


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    gc = gspread.service_account()
    tab_cache = {}
    for pk, ab, wb, tab, who, ep in CASES:
        js = requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay", timeout=30).json()
        play = js['allPlays'][ab - 1]
        # feed replay with pre-pitch counts
        feed = []
        b = s = 0
        for e in play.get('playEvents', []):
            d = str(e.get('details', {}).get('description', ''))
            dl = d.lower()
            if not e.get('isPitch'):
                if 'automatic ball' in dl:
                    b += 1
                elif 'automatic strike' in dl:
                    s += 1
                continue
            pd = e.get('pitchData', {})
            feed.append({
                'n': e.get('pitchNumber'), 'desc': d, 'count': f"{b}-{s}",
                'velo': pd.get('startSpeed'),
                'px': (pd.get('coordinates') or {}).get('pX'),
                'pz': (pd.get('coordinates') or {}).get('pZ'),
                'type': (e.get('details', {}).get('type') or {}).get('code'),
            })
            if 'in play' in dl or 'hit by pitch' in dl:
                pass
            elif 'foul tip' in dl or 'foul bunt' in dl:
                s += 1
            elif 'foul' in dl:
                s += 1 if s < 2 else 0
            elif 'strike' in dl or 'missed bunt' in dl:
                s += 1
            elif 'ball' in dl or 'pitchout' in dl:
                b += 1
        result = play.get('result', {}).get('event')

        if (wb, tab) not in tab_cache:
            ws = gc.open_by_key(B.SPREADSHEET_IDS[wb]).worksheet(tab)
            tab_cache[(wb, tab)] = B.read_sheet_with_retry(ws)
            time.sleep(1)
        rows = tab_cache[(wb, tab)]
        ci = {n: j for j, n in enumerate(rows[0]) if n}
        pref = f"{pk}_{ab:03d}_"
        srows = sorted([r for r in rows[1:] if str(r[ci['PitchID']]).startswith(pref)],
                       key=lambda r: r[ci['PitchID']])
        # physics keys of sheet rows for matching
        skeys = {}
        for r in srows:
            k = (round(sf(r[ci['Velocity']]) or -9, 1), round(sf(r[ci['PlateX']]) or -9, 2))
            skeys[k] = r

        print(f"\n{'='*88}")
        print(f"{pk}_{ab:03d}  [{wb}/{tab}]  {who}  result={result}")
        print(f"  FEED (current, {len(feed)} pitches):")
        for f in feed:
            k = (round(sf(f['velo']) or -9, 1), round(sf(f['px']) or -9, 2))
            m = skeys.get(k)
            if m is None:
                newtype = 'EP' if ep else f['type']
                tag = f"<== NEW (would insert, type={newtype})"
            else:
                tag = f"(in sheet as {m[ci['PitchID']]}, your type={m[ci['Pitch Type']]})"
            print(f"    #{f['n']}  {f['desc']:24s} pre={f['count']:4s} velo={f['velo']}  feed_type={f['type']}  {tag}")
        print(f"  SHEET (current, {len(srows)} rows):")
        fkeys = {(round(sf(f['velo']) or -9, 1), round(sf(f['px']) or -9, 2)) for f in feed}
        for r in srows:
            k = (round(sf(r[ci['Velocity']]) or -9, 1), round(sf(r[ci['PlateX']]) or -9, 2))
            gone = '' if k in fkeys else '  <== NOT IN FEED (would remove)'
            ev = r[ci['Event']] if 'Event' in ci else ''
            print(f"    {r[ci['PitchID']]}  {r[ci['Description']]:16s} ({r[ci['Count']]}) type={r[ci['Pitch Type']]:3s} "
                  f"velo={r[ci['Velocity']]} {('Event=' + str(ev)) if ev else ''}{gone}")


if __name__ == '__main__':
    main()
