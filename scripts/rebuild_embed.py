"""rebuild_embed.py — swap the Stuff+-injected leaderboards into the already-
built data_core.json.gz, without re-running the full pipeline.
(Embed split 2026-07-29: the leaderboards live in the CORE chunk; the heavy
chunk — micro/details — is untouched by Stuff+ injection.)

process_data.py builds data_embedded from its in-memory result (with the OLD,
preserved Stuff+). The Stuff+ trainer injects fresh scores into the leaderboard
JSON files afterward. This re-reads those JSONs and swaps only pitchData /
pitcherData back into the bundle (keeping micro-data, pitch details, metadata,
etc. that process_data just produced), then re-gzips. Run it right after
train_stuff_v11.py --inject.
"""
import json, gzip, os, sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
GZ = os.path.join(DATA, 'data_core.json.gz')

def main(gz=GZ):
    obj = json.loads(gzip.decompress(open(gz, 'rb').read()))
    obj['pitchData'] = json.load(open(os.path.join(DATA, 'pitch_leaderboard_rs.json')))
    obj['pitcherData'] = json.load(open(os.path.join(DATA, 'pitcher_leaderboard_rs.json')))
    payload = json.dumps(obj, separators=(',', ':')).encode()
    with open(gz, 'wb') as f:
        # mtime=0 for byte-identical output on unchanged data (matches
        # write_embedded_js), so a re-run with no new data yields no spurious commit.
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))
    n_stuff = sum(1 for r in obj['pitcherData'] if r.get('stuffScore') is not None)
    print(f"Rebuilt {os.path.basename(gz)}: {len(obj['pitcherData'])} pitchers "
          f"({n_stuff} with Stuff+), {len(obj['pitchData'])} pitch rows, "
          f"{os.path.getsize(gz)/1e6:.1f} MB")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else GZ)
