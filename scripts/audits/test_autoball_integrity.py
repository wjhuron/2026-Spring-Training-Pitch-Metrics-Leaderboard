"""test_autoball_integrity.py — regression guard for Pitcher2026's automatic
ball/strike handling. Re-scrapes games with known automatic events and asserts,
for every plate appearance that contains one, the three invariants that the old
coordinate-lag bug violated:

  1. NO PHANTOM ROW  — an automatic event (no pitch thrown) must not create a
     pitch row; the scraped real-pitch set must equal the feed's real-pitch set.
  2. NO COORDINATE LAG — each pitch's PlateX/PlateZ must equal *its own* feed
     coordinates, never a neighbor's.
  3. COUNT STEPS — the stored count (which is the count BEFORE the pitch) must
     reflect the automatic ball/strike, i.e. equal the feed's count after the
     immediately preceding event.

Ground truth is the live MLB feed, pulled fresh in the test, so it stays valid
across Statcast reprocessing (no hardcoded coordinates). Exit 0 = pass, 1 = fail.

  python3 scripts/audits/test_autoball_integrity.py
"""
import os, sys, warnings
from collections import Counter
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import requests
import scrapers.Pitcher2026 as Pitcher2026

# Games known to contain automatic events (from the 2026 lag-corruption fixes).
TEST_GAMES = [822757, 822998, 823161, 823319, 824297, 823727, 824535, 824781]
COORD_TOL = 0.01   # scrape and feed both store round(coord, 3) -> should match exactly
FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"


def feed_pas(pk):
    """{pa_num -> {'pitches': [(pn, px3, pz3, precount_str)], 'autos': [desc]}}"""
    plays = requests.get(FEED.format(pk), timeout=60).json()['liveData']['plays']['allPlays']
    out = {}
    for play in plays:
        pa = play.get('atBatIndex', 0) + 1
        pitches = []; autos = []
        prev_after = (0, 0)   # count after the previous event; pre-count of next pitch
        for ev in play.get('playEvents', []):
            cnt = ev.get('count', {}) or {}
            after = (cnt.get('balls', prev_after[0]), cnt.get('strikes', prev_after[1]))
            if ev.get('isPitch', False):
                c = ev.get('pitchData', {}).get('coordinates', {})
                px, pz = c.get('pX'), c.get('pZ')
                pitches.append((ev.get('pitchNumber') or 0,
                                round(px, 3) if px is not None else None,
                                round(pz, 3) if pz is not None else None,
                                f"{prev_after[0]}-{prev_after[1]}"))
            else:
                desc = (ev.get('details', {}) or {}).get('description', '') or ''
                if 'automatic' in desc.lower():
                    autos.append(desc)
            prev_after = after
        out[pa] = {'pitches': pitches, 'autos': autos}
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    dl = Pitcher2026.BaseballSavantFocusedDownloader()
    failures = []; checked = 0; auto_types = Counter()

    for pk in TEST_GAMES:
        try:
            feed = feed_pas(pk)
            df = dl.download_game_data(pk)
        except Exception as e:
            failures.append(f"{pk}: scrape/feed error {e!r}"); continue
        pid_col = df['PitchID'].astype(str)
        for pa, info in feed.items():
            if not info['autos']:
                continue
            checked += 1
            for a in info['autos']:
                auto_types['strike' if 'strike' in a.lower() else
                           ('intentional' if 'intentional' in a.lower() else 'ball')] += 1
            prefix = f"{pk}_{pa:03d}_"
            rows = df[pid_col.str.startswith(prefix)]
            scr = {}
            for _, r in rows.iterrows():
                pn = int(str(r['PitchID']).split('_')[2])
                scr[pn] = r
            feed_pns = sorted(pn for pn, *_ in info['pitches'])
            # 1) no phantom / no missing
            if sorted(scr) != feed_pns:
                failures.append(f"{prefix}: pitch set {sorted(scr)} != feed {feed_pns} "
                                f"(phantom auto row or missing pitch)")
                continue
            for pn, px, pz, pre in info['pitches']:
                r = scr[pn]
                # 2) no coordinate lag
                sx, sz = fnum(r.get('PlateX')), fnum(r.get('PlateZ'))
                if px is not None and (sx is None or abs(sx - px) > COORD_TOL):
                    failures.append(f"{prefix}{pn:02d}: PlateX {sx} != feed {px} (LAG)")
                if pz is not None and (sz is None or abs(sz - pz) > COORD_TOL):
                    failures.append(f"{prefix}{pn:02d}: PlateZ {sz} != feed {pz} (LAG)")
                # 3) count steps across the automatic event
                if str(r.get('Count')) != pre:
                    failures.append(f"{prefix}{pn:02d}: Count {r.get('Count')!r} != expected "
                                    f"{pre!r} (automatic event not counted)")

    print(f"auto-event PAs checked: {checked}   types: {dict(auto_types)}")
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for f in failures[:40]:
            print("  ", f)
        sys.exit(1)
    print("PASS — no phantom rows, no coordinate lag, counts step across every automatic event.")
    sys.exit(0)


if __name__ == '__main__':
    main()
