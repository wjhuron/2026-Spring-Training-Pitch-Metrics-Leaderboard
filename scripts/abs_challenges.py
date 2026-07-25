"""ABS challenge dataset builder (2026 MLB).

Pulls every called ball / called strike near the ABS zone edge from MLB feed
data, with full game state (pre-pitch count, outs, base state, score) plus
challenge events: who challenged (exact player), whether the call was
overturned, and each team's remaining challenges at the moment of the pitch.

Feed semantics (verified against 2026-07-19 LAD@NYY, gamePk 823523):
  - pitchData.strikeZoneTop/Bottom are the fixed height-based ABS zone,
    constant per batter all game (strikeZoneWidth=17, strikeZoneDepth=8.5).
  - Mid-PA challenges (and failed PA-ending ones) carry ev['reviewDetails']
    with isOverturned, challengeTeamId, and the challenging player.
  - PA-ending OVERTURNED calls (e.g. ball four flipped to strike three) are
    rewritten with NO reviewDetails; the only marker is the play-level
    result.description: "X challenged (pitch result), call on the field was
    overturned: ...". Both paths are scanned and deduped by playId.
  - ev['count'] is the post-pitch count with the FINAL (post-challenge) call
    applied; pre-pitch count = post count minus the final call.
  - ev['count']['outs'] is the current outs DURING the PA (pre-result).
  - gameData.absChallenges holds per-team usedSuccessful/usedFailed/remaining
    totals, used here as a per-game audit of challenge detection.

Output: data/abs_challenges_2026.json (incremental; skips gamePks already
present unless --refresh).

Usage:
    python3 scripts/abs_challenges.py                     # season to date
    python3 scripts/abs_challenges.py --start 2026-07-01 --end 2026-07-19
    python3 scripts/abs_challenges.py --limit 5           # smoke test
"""

import argparse
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests

SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule"
                "?sportId=1&gameTypes=R&startDate={start}&endDate={end}")
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

TAKE_CODES = {"B": "ball", "*B": "ball", "C": "strike"}
ZONE_HALF_WIDTH_IN = 8.5          # 17" plate width / 2
KEEP_BAND_IN = 4.5                # keep takes within this |edge distance|
CHALLENGES_PER_TEAM = 2
GHOST_RUNNER = -1        # sentinel for the extra-innings automatic runner
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "abs_challenges_2026.json")

CHAL_TOKEN = "challenged (pitch result)"
# challenger name = trailing token run before "challenged": first token must be
# capitalized, later tokens may be lowercase particles ("d'Arnaud", "de la Cruz")
NAME_RUN_RE = re.compile(r"((?:[A-ZÀ-ÖØ-Þ][\w.'’-]*\s+)(?:[\w.'’-]+\s+)*)$", re.UNICODE)


def parse_desc_challenges(desc):
    """Extract every pitch-result challenge from a play description.

    Observed formats:
      "X challenged (pitch result), call on the field was overturned: ..."
      "X challenged (pitch result), call on the field was confirmed: ..."
      "X challenged (pitch result): Steven Kwan walks."   <- no clause = failed
      "Nationals challenged (tag play), ... overturned: X challenged
       (pitch result), call on the field was ..."         <- concatenated
    Returns a list of (challengerName, overturned) in order of appearance.
    """
    out = []
    for m in re.finditer(re.escape(CHAL_TOKEN), desc or ""):
        # challenger = trailing run of capitalized tokens before "challenged",
        # so "overturned: Keibert Ruiz challenged" -> "Keibert Ruiz" and dotted
        # names ("J.P. Crawford", "Ronald Acuna Jr.") survive intact
        nm = NAME_RUN_RE.search(desc[:m.start()])
        name = nm.group(1).strip() if nm else ""
        cm = re.match(r", call on the field was (\w+)", desc[m.end():])
        out.append((name, bool(cm and cm.group(1) == "overturned")))
    return out


def norm_name(s):
    """Accent-insensitive name normalization for description-text matching."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


ABS_PLANE_Y_FT = 17.0 / 12.0 - 8.5 / 12.0  # plate midpoint: 8.5" behind the front edge


def coords_at_plane(coords, y_plane):
    """Ball (x, z) at a given y plane from the 9-parameter kinematic fit.

    Statcast pX/pZ are evaluated at the plate front; the ABS zone lives at the
    plate midpoint (strikeZoneDepth 8.5), so challenge rulings key off the
    position ~8.5 inches deeper. Solves y(t) = y_plane for t, then evaluates
    x(t), z(t).
    """
    try:
        y0, vy, ay = coords["y0"], coords["vY0"], coords["aY"]
        disc = vy * vy - 2.0 * ay * (y0 - y_plane)
        if disc < 0:
            return None, None
        t = (-vy - disc ** 0.5) / ay
        x = coords["x0"] + coords["vX0"] * t + 0.5 * coords["aX"] * t * t
        z = coords["z0"] + coords["vZ0"] * t + 0.5 * coords["aZ"] * t * t
        return x, z
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None


def zone_distance_in(px, pz, sz_top, sz_bot):
    """Signed distance (inches) from ball CENTER to the ABS zone rectangle.

    Negative = center inside the zone (magnitude = distance to nearest edge).
    Positive = center outside (Euclidean distance to the rectangle).
    Ball-edge conventions are applied downstream by offsetting the radius.
    """
    x_in = abs(px) * 12.0
    z_in = pz * 12.0
    dx = x_in - ZONE_HALF_WIDTH_IN
    dz = max(sz_bot * 12.0 - z_in, z_in - sz_top * 12.0)
    if dx <= 0 and dz <= 0:
        return max(dx, dz)  # inside: negative, nearest-edge
    return (max(dx, 0.0) ** 2 + max(dz, 0.0) ** 2) ** 0.5


def apply_movements(bases, entries):
    """Apply one playIndex batch of runner movements to the base-state dict.

    Removals first (by runner id AND by the base being vacated), then
    placements, so same-index chains like batter->1B + R1->2B can't clobber
    each other. Vacating movement["start"] matters for the extra-innings
    automatic runner: he is pre-placed by the rule and never appears in a
    runners entry until he moves, so he has no id we could have tracked.
    Returns runs scored in this batch.
    """
    runs = 0
    ids = {e["details"]["runner"]["id"] for e in entries}
    for base in list(bases):
        if bases[base] in ids:
            del bases[base]
    for e in entries:
        start = e["movement"].get("start")
        if start in ("1B", "2B", "3B"):
            bases.pop(start, None)
    for e in entries:
        mv = e["movement"]
        end = mv.get("end")
        if mv.get("isOut"):
            continue
        if end == "score":
            runs += 1
        elif end in ("1B", "2B", "3B"):
            bases[end] = e["details"]["runner"]["id"]
    return runs


def starting_catchers(live):
    """Starting catcher id per side: non-substitute whose allPositions has '2'."""
    out = {}
    for side in ("away", "home"):
        cid = None
        for p in live["boxscore"]["teams"][side]["players"].values():
            if (not p.get("gameStatus", {}).get("isSubstitute")
                    and any(ap.get("code") == "2" for ap in p.get("allPositions", []))):
                cid = p["person"]["id"]
                break
        out[side] = cid
    return out


def find_player_team(live, player_id):
    """Which side ('away'/'home') a player is on, via the boxscore rosters."""
    key = "ID" + str(player_id)
    for side in ("away", "home"):
        if key in live["boxscore"]["teams"][side]["players"]:
            return side
    return None


def parse_game(feed):
    """Extract take/challenge records + challenge audit from one game feed."""
    gd = feed["gameData"]
    live = feed["liveData"]
    game_pk = gd["game"]["pk"]
    game_date = gd["datetime"]["officialDate"]
    abs_info = gd.get("absChallenges") or {}
    players_by_name = {norm_name(p["fullName"]): p["id"]
                       for p in gd.get("players", {}).values()}
    players_by_name_nodots = {k.replace(".", ""): v for k, v in players_by_name.items()}
    players_by_id = {p["id"]: p["fullName"] for p in gd.get("players", {}).values()}
    # Position players mopping up in a blowout are not a fair test of anyone's
    # umpiring or challenge decisions; flag them so consumers can drop them.
    pos_pitcher = {p["id"] for p in gd.get("players", {}).values()
                   if (p.get("primaryPosition") or {}).get("code") not in ("1", "Y")}
    catchers = starting_catchers(live)
    fielder_chal = {"n": 0, "catcherMatch": 0}

    records = []
    detected = {"away": {"ok": 0, "fail": 0}, "home": {"ok": 0, "fail": 0}}
    remaining = {"away": CHALLENGES_PER_TEAM, "home": CHALLENGES_PER_TEAM}
    score = {"away": 0, "home": 0}
    prev_half = None
    bases = {}

    for play in live["plays"]["allPlays"]:
        about = play["about"]
        half_key = (about["inning"], about["halfInning"])
        if half_key != prev_half:
            # Extra innings start with the automatic runner on 2nd. The feed
            # pre-places him, so he never shows up in a runners entry unless he
            # moves -- without this the highest-leverage innings of the season
            # are recorded as bases empty.
            bases = {"2B": GHOST_RUNNER} if about["inning"] >= 10 else {}
            prev_half = half_key
        bat_side = "away" if about["halfInning"] == "top" else "home"

        movements = sorted(
            (e for e in play.get("runners", []) if e.get("movement")),
            key=lambda e: e["details"].get("playIndex", 0))
        mv_pos = 0

        # PA-ending challenges often live only in the result description
        desc_chals = parse_desc_challenges(play.get("result", {}).get("description"))
        events = play.get("playEvents", [])
        last_pitch_idx = max((e["index"] for e in events if e.get("isPitch")), default=None)

        for ev in events:
            idx = ev["index"]
            # advance base/score state with movements that precede this event
            while mv_pos < len(movements) and movements[mv_pos]["details"].get("playIndex", 0) < idx:
                j = mv_pos
                while (j < len(movements) and movements[j]["details"].get("playIndex", 0)
                       == movements[mv_pos]["details"].get("playIndex", 0)):
                    j += 1
                score[bat_side] += apply_movements(bases, movements[mv_pos:j])
                mv_pos = j

            if not ev.get("isPitch"):
                det = ev.get("details", {})
                if (det.get("eventType") in ("defensive_substitution", "defensive_switch")
                        and ev.get("position", {}).get("code") == "2"
                        and ev.get("player")):
                    catchers["home" if bat_side == "away" else "away"] = ev["player"]["id"]
                continue
            det = ev["details"]
            code = det.get("code")
            final_call = TAKE_CODES.get(code)
            if final_call is None or "Automatic" in (det.get("description") or ""):
                continue
            pd = ev.get("pitchData", {})
            coords = pd.get("coordinates", {})
            px, pz = coords.get("pX"), coords.get("pZ")
            sz_top, sz_bot = pd.get("strikeZoneTop"), pd.get("strikeZoneBottom")
            if px is None or pz is None or sz_top is None or sz_bot is None:
                continue

            post = ev["count"]
            pre_balls = post["balls"] - (1 if final_call == "ball" else 0)
            pre_strikes = post["strikes"] - (1 if final_call == "strike" else 0)

            challenge = None
            rd = ev.get("reviewDetails")
            # ABS challenges are reviewType MJ and name the challenging player;
            # MA/MI etc. are ordinary replay reviews and are ignored here.
            if rd is not None and (rd.get("reviewType") != "MJ" or "player" not in rd):
                rd = None
            if rd is not None:
                chal_side = "away" if rd.get("challengeTeamId") == gd["teams"]["away"]["id"] else "home"
                challenge = {"playerId": rd["player"]["id"],
                             "playerName": rd["player"]["fullName"],
                             "side": chal_side,
                             "overturned": bool(rd["isOverturned"])}
            elif desc_chals and idx == last_pitch_idx:
                name, overturned = desc_chals[-1]
                pid = (players_by_name.get(norm_name(name))
                       or players_by_name_nodots.get(norm_name(name).replace(".", "")))
                chal_side = find_player_team(live, pid) if pid else None
                challenge = {"playerId": pid,
                             "playerName": name,
                             "side": chal_side,
                             "overturned": overturned}

            original_call = final_call
            if challenge and challenge["overturned"]:
                original_call = "strike" if final_call == "ball" else "ball"

            dist = zone_distance_in(px, pz, sz_top, sz_bot)
            x_mid, z_mid = coords_at_plane(coords, ABS_PLANE_Y_FT)
            dist_mid = (zone_distance_in(x_mid, z_mid, sz_top, sz_bot)
                        if x_mid is not None else None)
            if challenge:
                batter_id = play["matchup"]["batter"]["id"]
                pitcher_id = play["matchup"]["pitcher"]["id"]
                if challenge["playerId"] == batter_id:
                    challenge["role"] = "batter"
                elif challenge["playerId"] == pitcher_id:
                    challenge["role"] = "pitcher"
                else:
                    challenge["role"] = "fielder"
                challenge["remainingBefore"] = remaining.get(challenge["side"])

            band_dist = dist_mid if dist_mid is not None else dist
            fld_side = "home" if bat_side == "away" else "away"
            if challenge and challenge["role"] == "fielder":
                fielder_chal["n"] += 1
                fielder_chal["catcherMatch"] += challenge["playerId"] == catchers[fld_side]

            keep = challenge is not None or abs(band_dist) <= KEEP_BAND_IN
            if keep:
                records.append({
                    "gamePk": game_pk,
                    "date": game_date,
                    "playId": ev.get("playId"),
                    "inning": about["inning"],
                    "half": about["halfInning"],
                    "outs": post["outs"],
                    "balls": pre_balls,
                    "strikes": pre_strikes,
                    "bases": "".join("1" if b in bases else "0" for b in ("1B", "2B", "3B")),
                    "awayScore": score["away"],
                    "homeScore": score["home"],
                    "batSide": bat_side,
                    "batterId": play["matchup"]["batter"]["id"],
                    "batter": play["matchup"]["batter"]["fullName"],
                    "pitcherId": play["matchup"]["pitcher"]["id"],
                    "pitcher": play["matchup"]["pitcher"]["fullName"],
                    "posPitcher": play["matchup"]["pitcher"]["id"] in pos_pitcher,
                    "catcherId": catchers[fld_side],
                    "catcher": players_by_id.get(catchers[fld_side]),
                    "batHand": play["matchup"]["batSide"]["code"],
                    "pitchHand": play["matchup"]["pitchHand"]["code"],
                    "pitchType": (det.get("type") or {}).get("code"),
                    "finalCall": final_call,
                    "originalCall": original_call,
                    "pX": px,
                    "pZ": pz,
                    "szTop": sz_top,
                    "szBot": sz_bot,
                    "distIn": round(dist, 3),
                    "pXmid": round(x_mid, 4) if x_mid is not None else None,
                    "pZmid": round(z_mid, 4) if z_mid is not None else None,
                    "distMidIn": round(dist_mid, 3) if dist_mid is not None else None,
                    "remAway": remaining["away"],
                    "remHome": remaining["home"],
                    "challenge": challenge,
                })
            if challenge and challenge["side"] in remaining:
                if challenge["overturned"]:
                    detected[challenge["side"]]["ok"] += 1
                else:
                    detected[challenge["side"]]["fail"] += 1
                    remaining[challenge["side"]] -= 1

        # flush movements at/after the last event index
        while mv_pos < len(movements):
            j = mv_pos
            while (j < len(movements) and movements[j]["details"].get("playIndex", 0)
                   == movements[mv_pos]["details"].get("playIndex", 0)):
                j += 1
            score[bat_side] += apply_movements(bases, movements[mv_pos:j])
            mv_pos = j

    audit = {"ok": True, "detected": detected, "official": {},
             "fielderChal": fielder_chal}
    for side in ("away", "home"):
        off = abs_info.get(side) or {}
        audit["official"][side] = off
        if off and (off.get("usedSuccessful") != detected[side]["ok"]
                    or off.get("usedFailed") != detected[side]["fail"]):
            audit["ok"] = False
    return {"gamePk": game_pk, "date": game_date,
            "away": gd["teams"]["away"]["abbreviation"],
            "home": gd["teams"]["home"]["abbreviation"],
            "audit": audit, "records": records}


def fetch_game(session, game_pk):
    r = session.get(FEED_URL.format(pk=game_pk), timeout=30)
    r.raise_for_status()
    return parse_game(r.json())


def get_final_game_pks(session, start, end):
    r = session.get(SCHEDULE_URL.format(start=start, end=end), timeout=30)
    r.raise_for_status()
    pks = []
    for d in r.json().get("dates", []):
        for g in d["games"]:
            # codedGameState 'F' = actually played to completion; postponed
            # games also report abstractGameState 'Final' but must NOT be
            # ingested, or their gamePk would be skipped when the makeup date
            # actually gets played
            if (g["status"]["abstractGameState"] == "Final"
                    and g["status"].get("codedGameState") == "F"
                    and g["gameType"] == "R"):
                pks.append(g["gamePk"])
    return sorted(set(pks))


def main():
    ap = argparse.ArgumentParser(description="Build ABS challenge dataset from MLB feeds")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max games (smoke test)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch games already in output")
    args = ap.parse_args()

    existing = {"games": [], "records": []}
    if os.path.exists(args.out) and not args.refresh:
        with open(args.out) as f:
            existing = json.load(f)
    have = {g["gamePk"] for g in existing.get("games", [])}

    session = requests.Session()
    pks = [pk for pk in get_final_game_pks(session, args.start, args.end) if pk not in have]
    if args.limit:
        pks = pks[:args.limit]
    print(f"{len(have)} games already in {args.out}; fetching {len(pks)} new")

    games, failures = list(existing.get("games", [])), []
    records = list(existing.get("records", []))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_game, session, pk): pk for pk in pks}
        for i, fut in enumerate(as_completed(futures), 1):
            pk = futures[fut]
            try:
                g = fut.result()
            except Exception as e:
                failures.append({"gamePk": pk, "error": str(e)})
                print(f"  FAIL {pk}: {e}")
                continue
            records.extend(g.pop("records"))
            games.append(g)
            if i % 100 == 0 or i == len(pks):
                print(f"  {i}/{len(pks)} games parsed")

    games.sort(key=lambda g: (g["date"], g["gamePk"]))
    records.sort(key=lambda r: (r["date"], r["gamePk"], r["inning"], r["playId"] or ""))
    bad_audits = [g["gamePk"] for g in games if not g["audit"]["ok"]]
    n_chal = sum(1 for r in records if r["challenge"])
    out = {
        "meta": {
            "generated": date.today().isoformat(),
            "start": args.start, "end": args.end,
            "games": len(games), "records": len(records),
            "challenges": n_chal,
            "keepBandIn": KEEP_BAND_IN,
            "auditMismatchGamePks": bad_audits,
            "fetchFailures": failures,
        },
        "games": games,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(games)} games, {len(records)} records, "
          f"{n_chal} challenges, {len(bad_audits)} audit mismatches")
    if bad_audits:
        print("audit mismatches (detected vs official):", bad_audits[:20])


if __name__ == "__main__":
    main()
