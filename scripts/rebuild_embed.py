"""rebuild_embed.py — swap the Stuff+-injected outputs into the already-built
embed chunks, without re-running the full pipeline.

process_data.py builds the embed from its in-memory result (with the OLD,
preserved Stuff+). The Stuff+ trainer injects fresh scores into the leaderboard
JSON files afterward, and refresh_micro_grades.py rebuilds the micro-data with
both grade dumps fresh. This script re-reads those artifacts and swaps them
into the split chunks (2026-07-29):

  data_core.json.gz   <- pitchData / pitcherData from the injected leaderboard
                         JSONs (everything else process_data built is kept)
  data_heavy.json.gz  <- microData from data/micro_data_rs.json, when present
                         (so filtered site views / grade atoms never lag the
                         sheets and cards by one Stuff+ cycle)

Run it right after train_stuff_v11.py --inject + refresh_micro_grades.py.
"""
import json, gzip, os, sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CORE_GZ = os.path.join(DATA, 'data_core.json.gz')
HEAVY_GZ = os.path.join(DATA, 'data_heavy.json.gz')


def _read_gz(path):
    return json.loads(gzip.decompress(open(path, 'rb').read()))


def _write_gz(path, obj):
    payload = json.dumps(obj, separators=(',', ':')).encode()
    with open(path, 'wb') as f:
        # mtime=0 for byte-identical output on unchanged data (matches
        # write_embedded_js), so a re-run with no new data yields no spurious commit.
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))


def main(core_gz=CORE_GZ, heavy_gz=HEAVY_GZ):
    obj = _read_gz(core_gz)
    obj['pitchData'] = json.load(open(os.path.join(DATA, 'pitch_leaderboard_rs.json')))
    obj['pitcherData'] = json.load(open(os.path.join(DATA, 'pitcher_leaderboard_rs.json')))
    _write_gz(core_gz, obj)
    n_stuff = sum(1 for r in obj['pitcherData'] if r.get('stuffScore') is not None)
    print(f"Rebuilt {os.path.basename(core_gz)}: {len(obj['pitcherData'])} pitchers "
          f"({n_stuff} with Stuff+), {len(obj['pitchData'])} pitch rows, "
          f"{os.path.getsize(core_gz)/1e6:.1f} MB")

    # refresh_micro_grades.py writes micro_data_rs.json with CURRENT grade
    # atoms; without this swap the heavy chunk ships the micro data built
    # mid-process_data with the previous run's Stuff+ dump.
    micro_path = os.path.join(DATA, 'micro_data_rs.json')
    if os.path.exists(micro_path):
        heavy = _read_gz(heavy_gz)
        heavy['microData'] = json.load(open(micro_path))
        _write_gz(heavy_gz, heavy)
        print(f"Rebuilt {os.path.basename(heavy_gz)}: "
              f"{len(heavy['microData'].get('pitchMicro', []))} pitch micro rows, "
              f"{os.path.getsize(heavy_gz)/1e6:.1f} MB")
    else:
        print(f"{os.path.basename(heavy_gz)} untouched (no {os.path.basename(micro_path)} "
              f"refresh found — run scripts/refresh_micro_grades.py first)")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else CORE_GZ)
