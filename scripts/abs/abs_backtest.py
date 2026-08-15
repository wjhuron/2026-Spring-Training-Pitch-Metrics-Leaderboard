"""ABS backtest: how many wins does optimal challenging add?

Replays every 2026 team-game's challenge opportunities chronologically under
the matrix policy and compares value captured with what teams actually
captured. Both sides gain value ONLY from overturned calls (challenges are
retained on success), so gross captured value is the clean comparison; failed
challenges cost nothing except the option they burn, which shows up as
opportunities the depleted team can no longer take.

The simulated decider is a LEAGUE-AVERAGE perceiver: it sees each pitch with
the fitted per-region noise, forms Bayes confidence, and challenges iff
confidence >= C(k, T, score) / (gain + C). No hindsight - this is attainable
skill, not oracle play.

Output: per-team table + ~/Downloads/abs_backtest_2026.csv.

Usage: python3 scripts/abs/abs_backtest.py
"""

import csv
import json
import os
import random
from collections import defaultdict

import abs_value_engine as ve
from abs_option_model import count_class, edge_region

N_SIM = 20
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(REPO_ROOT, "data", "abs_challenges_2026.json")
TABLES = os.path.join(REPO_ROOT, "data", "abs_value_tables_2026.json")
OPTION = os.path.join(REPO_ROOT, "data", "abs_option_model_2026.json")
DOWNLOADS = os.path.expanduser("~/Downloads")


def interp(grid, x):
    if x <= grid[0][0]:
        return grid[0][1]
    if x >= grid[-1][0]:
        return grid[-1][1]
    step = grid[1][0] - grid[0][0]
    i = int((x - grid[0][0]) / step)
    x0, p0 = grid[i]
    x1, p1 = grid[min(i + 1, len(grid) - 1)]
    return p0 + (p1 - p0) * (x - x0) / max(x1 - x0, 1e-9)


def main():
    with open(DATASET) as f:
        data = json.load(f)
    with open(OPTION) as f:
        opt = json.load(f)
    tables = ve.tables_from_json(TABLES)
    thr = opt["meta"]["rulingThrIn"]
    g_avg = tables["gAvg"]
    Cg = {}
    for key, v in opt["Cgrid"].items():
        k, T, d = key.split("|")
        Cg[(int(k), int(T), int(d))] = v
    posts = opt["posterior"]                       # "side|reg" -> [[x, p], ...]
    sigma = {f"{s}|{r}": opt["perception"][s][r]["sigma"]
             for s in opt["perception"] for r in opt["perception"][s]}
    game_teams = {g["gamePk"]: (g["away"], g["home"]) for g in data["games"]}

    # chronological opportunity stream per (gamePk, team)
    streams = defaultdict(list)
    for r in data["records"]:
        if r["distMidIn"] is None or r.get("posPitcher"):
            continue
        if r["originalCall"] == "strike":
            side, m = "bat", r["distMidIn"] - thr
            wronged = r["batSide"]
        else:
            side, m = "fld", thr - r["distMidIn"]
            wronged = "home" if r["batSide"] == "away" else "away"
        team = game_teams[r["gamePk"]][0 if wronged == "away" else 1]
        d_team = (r["awayScore"] - r["homeScore"]) if wronged == "away" \
            else (r["homeScore"] - r["awayScore"])
        v = ve.value_of_flip(r["balls"], r["strikes"], r["bases"], r["outs"],
                             r["inning"], r["half"], r["homeScore"] - r["awayScore"],
                             tables)
        inning = min(r["inning"], 9)
        T = 2 * (9 - inning) + (2 if r["half"] == "top" else 1)
        reg = (edge_region(r["pXmid"], r["pZmid"], r["szTop"], r["szBot"])
               if r.get("pXmid") is not None else "side")
        chal = r["challenge"]
        actual = chal is not None and chal.get("side") == wronged
        streams[(r["gamePk"], team)].append({
            "ord": (r["inning"], 0 if r["half"] == "top" else 1),
            "side": side, "m": m, "g": v["leveragedRuns"],
            "T": T, "d": max(-12, min(12, d_team)), "reg": reg,
            "actual": actual,
            "overturned": bool(chal and chal.get("overturned")) and actual,
        })

    actual_lev = defaultdict(float)
    optimal_lev = defaultdict(float)
    n_games = defaultdict(int)
    att_total = win_total = 0.0
    rng = random.Random(20260720)
    for (pk, team), opps in streams.items():
        opps.sort(key=lambda o: o["ord"])
        n_games[team] += 1
        actual_lev[team] += sum(o["g"] for o in opps if o["overturned"])
        cap = 0.0
        for _ in range(N_SIM):
            k = 2
            for o in opps:
                if k == 0 or o["g"] <= 0:
                    continue
                key = f"{o['side']}|{o['reg']}"
                x = o["m"] + rng.gauss(0.0, sigma[key])
                p = interp(posts[key], x)
                cost = Cg[(k, o["T"], o["d"])]
                if p * o["g"] < (1.0 - p) * cost:
                    continue
                att_total += 1.0 / N_SIM
                if o["m"] > 0:
                    cap += o["g"] / N_SIM
                    win_total += 1.0 / N_SIM
                else:
                    k -= 1
        optimal_lev[team] += cap

    teams = sorted(actual_lev, key=lambda t: -(optimal_lev[t] - actual_lev[t]))
    rows = []
    for t in teams:
        act_w = actual_lev[t] * g_avg
        opt_w = optimal_lev[t] * g_avg
        rows.append((t, n_games[t], actual_lev[t], act_w, optimal_lev[t], opt_w,
                     opt_w - act_w))
    path = os.path.join(DOWNLOADS, "abs_backtest_2026.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Team", "Games", "ActualLevRuns", "ActualWins",
                    "OptimalLevRuns", "OptimalWins", "WinsLeftOnTable"])
        for r in rows:
            w.writerow([r[0], r[1]] + [round(x, 2) for x in r[2:]])
    print(f"wrote {path}\n")
    # committed JSON for the live site (Downloads CSV is not deployable)
    jpath = os.path.join(REPO_ROOT, "data", "abs_backtest_2026.json")
    with open(jpath, "w") as f:
        json.dump({"teams": [{"team": r[0], "games": r[1],
                              "actW": round(r[3], 2), "optW": round(r[5], 2),
                              "gapW": round(r[6], 2)} for r in rows]}, f)
    print(f"wrote {jpath}")

    tg = sum(n_games.values())
    a = sum(actual_lev.values()) * g_avg
    o = sum(optimal_lev.values()) * g_avg
    print(f"league, {tg} team-games:")
    print(f"  actual challenge value captured:  {a:6.1f} wins "
          f"({162 * a / tg:.2f} per team-162)")
    print(f"  optimal-policy value captured:    {o:6.1f} wins "
          f"({162 * o / tg:.2f} per team-162)")
    print(f"  left on the table:                {o - a:6.1f} wins league-wide "
          f"({162 * (o - a) / tg:.2f} per team-162)")
    print(f"  optimal usage: {att_total / tg:.2f} attempts/game at "
          f"{100 * win_total / att_total:.0f}% success\n")
    print("most wins left on the table:")
    for r in rows[:5]:
        print(f"  {r[0]:<4} actual {r[3]:5.2f}W optimal {r[5]:5.2f}W  gap {r[6]:5.2f}W")
    print("least:")
    for r in rows[-3:]:
        print(f"  {r[0]:<4} actual {r[3]:5.2f}W optimal {r[5]:5.2f}W  gap {r[6]:5.2f}W")


if __name__ == "__main__":
    main()
