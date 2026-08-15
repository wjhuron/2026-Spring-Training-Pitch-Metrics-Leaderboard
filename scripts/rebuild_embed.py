"""rebuild_embed.py — swap the Stuff+-injected outputs into the already-built
embed chunks, without re-running the full pipeline.

process_data.py builds the embed from its in-memory result (with the OLD,
preserved Stuff+). The Stuff+ trainer injects fresh scores into the leaderboard
JSON files afterward, and refresh_micro_grades.py rebuilds the micro-data with
both grade dumps fresh. This script re-reads those artifacts and swaps them
into the split chunks (2026-07-29):

  data_core.json.gz   <- pitcherData from the injected leaderboard JSON, plus
                         a recomputed metadata.teamGames (everything else
                         process_data built is kept)
  data_tables.json.gz <- pitchData from the injected leaderboard JSON
  data_heavy.json.gz  <- microData from data/micro_data_rs.json, when present
                         (so filtered site views / grade atoms never lag the
                         sheets and cards by one Stuff+ cycle)

pitchData moved to data_tables in the 2026-08-03 split. Writing it back into
data_core here did two bad things at once: it undid the split (first paint
parsed it again) and it left data_tables holding the PRE-injection pitch rows,
so the Arsenal tab shipped Stuff+ a full cycle stale. validate_output's
check_core_stays_lean catches the first; the second was silent.

Run it right after train_stuff.py --inject + refresh_micro_grades.py.
"""
import json, gzip, os, sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CORE_GZ = os.path.join(DATA, 'data_core.json.gz')
TABLES_GZ = os.path.join(DATA, 'data_tables.json.gz')
HEAVY_GZ = os.path.join(DATA, 'data_heavy.json.gz')


def _team_games_played(micro):
    """Mirrors process_data._team_games_played / Aggregator.getTeamGamesPlayed()
    with no date range. Recomputed whenever microData is swapped below, so the
    qualification denominator in data_core always matches the microData that
    actually ships; a mismatch silently moves the Qualified cutoff."""
    ci = {c: i for i, c in enumerate(micro['pitcherCols'])}
    team_idx, date_idx = ci['teamIdx'], ci['dateIdx']
    teams = micro['lookups']['teams']
    seen = {}
    for row in micro['pitcherMicro']:
        seen.setdefault(row[team_idx], set()).add(row[date_idx])
    return {teams[t]: len(dates) for t, dates in seen.items()}


def _read_gz(path):
    return json.loads(gzip.decompress(open(path, 'rb').read()))


def _write_gz(path, obj):
    payload = json.dumps(obj, separators=(',', ':')).encode()
    with open(path, 'wb') as f:
        # mtime=0 for byte-identical output on unchanged data (matches
        # write_embedded_js), so a re-run with no new data yields no spurious commit.
        f.write(gzip.compress(payload, compresslevel=9, mtime=0))


def main(core_gz=CORE_GZ, heavy_gz=HEAVY_GZ, tables_gz=TABLES_GZ):
    obj = _read_gz(core_gz)
    obj['pitcherData'] = json.load(open(os.path.join(DATA, 'pitcher_leaderboard_rs.json')))
    # The inject step also updates metadata (Pitcher+ baseline, hdERA/hpERA
    # anchors + constants). Merge the inject-owned keys so the embedded
    # metadata never ships a cycle stale.
    try:
        _md = json.load(open(os.path.join(DATA, 'metadata_rs.json')))
        _core_md = obj.setdefault('metadata', {})
        for _k in ('pitcherPlusBaseline', 'eraPlusConstants'):
            if _k in _md:
                _core_md[_k] = _md[_k]
        _pla = _md.get('pitcherLeagueAverages') or {}
        if _pla:
            _core_pla = _core_md.setdefault('pitcherLeagueAverages', {})
            for _k in ('hdera', 'hpera'):
                if _k in _pla:
                    _core_pla[_k] = _pla[_k]
    except (OSError, json.JSONDecodeError):
        pass
    _write_gz(core_gz, obj)
    n_stuff = sum(1 for r in obj['pitcherData'] if r.get('stuffScore') is not None)
    print(f"Rebuilt {os.path.basename(core_gz)}: {len(obj['pitcherData'])} pitchers "
          f"({n_stuff} with Stuff+), {os.path.getsize(core_gz)/1e6:.1f} MB")

    # pitchData lives in data_tables since the 2026-08-03 split. It must be
    # swapped here too or the Arsenal tab ships last cycle's Stuff+.
    tables = _read_gz(tables_gz)
    tables['pitchData'] = json.load(open(os.path.join(DATA, 'pitch_leaderboard_rs.json')))
    _write_gz(tables_gz, tables)
    print(f"Rebuilt {os.path.basename(tables_gz)}: {len(tables['pitchData'])} pitch rows, "
          f"{os.path.getsize(tables_gz)/1e6:.1f} MB")

    # refresh_micro_grades.py writes micro_data_rs.json with CURRENT grade
    # atoms; without this swap the heavy chunk ships the micro data built
    # mid-process_data with the previous run's Stuff+ dump.
    #
    # STALENESS GUARD (2026-08-15): a LOCAL run can have a weeks-old
    # micro_data_rs.json lying around (it is not git-tracked). Swapping it
    # in silently replaces the CI-fresh micro in data_heavy with old rows
    # and old-scale grade atoms — every FILTERED site view then shows a
    # previous model's Stuff+ and long-departed team labels (shipped live
    # 2026-08-15 for ~2h that way: Ribalta 98 vs 107, released ROC players
    # resurrected). Swap ONLY when the micro refresh is newer than the
    # injected leaderboard it must match.
    micro_path = os.path.join(DATA, 'micro_data_rs.json')
    lb_path = os.path.join(DATA, 'pitcher_leaderboard_rs.json')
    micro_fresh = (os.path.exists(micro_path)
                   and os.path.getmtime(micro_path)
                   >= os.path.getmtime(lb_path) - 6 * 3600)
    heavy = _read_gz(heavy_gz)
    if micro_fresh:
        heavy['microData'] = json.load(open(micro_path))
        _write_gz(heavy_gz, heavy)
        print(f"Rebuilt {os.path.basename(heavy_gz)}: "
              f"{len(heavy['microData'].get('pitchMicro', []))} pitch micro rows, "
              f"{os.path.getsize(heavy_gz)/1e6:.1f} MB")
    elif os.path.exists(micro_path):
        print(f"{os.path.basename(heavy_gz)} untouched: {os.path.basename(micro_path)} "
              f"is STALE (older than the injected leaderboard) — keeping the "
              f"embedded micro. Run scripts/refresh_micro_grades.py to refresh.")
    else:
        print(f"{os.path.basename(heavy_gz)} untouched (no {os.path.basename(micro_path)} "
              f"refresh found — run scripts/refresh_micro_grades.py first)")
    # Qualification denominators always follow the micro that actually
    # ships in data_heavy (swapped or kept), never a skipped candidate.
    obj.setdefault('metadata', {})['teamGames'] = _team_games_played(heavy['microData'])
    _write_gz(core_gz, obj)
    print(f"  teamGames synced to shipping microData: "
          f"{len(obj['metadata']['teamGames'])} teams")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else CORE_GZ)
