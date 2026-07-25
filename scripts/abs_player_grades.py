"""ABS player grading: who challenges well, and who leaves value on the table.

Consumes the challenge dataset, value tables, and option model. Grades every
challenge and every unchallenged wrong call (truth-based, per Wally: rulings
are deterministic).

Grading is DECISION-based, not outcome-based (Wally, 2026-07-20: "losing a
challenge that is deemed worth challenging due to proximity and leverage
should not be negative"). Every challenge is scored at its expected value at
the moment of the decision, using the confidence an attentive decider could
have at the pitch's TRUE location:

    decisionEV = p(m) * g  -  (1 - p(m)) * C(k, T)

where p(m) is selection-conditioned for challenges actually made (pSel: the
decider saw enough to go, so their conditional confidence is above a blind
look - self-checked to reproduce observed success rates) and attentive-look
(pLook) for unchallenged pitches.

A matrix-approved challenge (p >= p* = C/(g+C), i.e. EV >= 0) earns that
positive EV whether it wins or loses; only matrix-disapproved challenges
(too far, too little leverage) grade negative. Realized outcome columns
(success rate, realized CVA) are kept for reference but do not drive the
ranking.

Missed opportunity: an unchallenged take with a challenge in hand where the
same decisionEV was positive - declining a gamble the matrix approves. It is
charged at that EV (what the decision was worth when made), not at the full
gain. Wrong calls too close to the edge for anyone to identify are NOT
counted as misses.

Attribution: batting-side to the batter, fielding-side to the tracked catcher
(challenges themselves credit whoever actually challenged, incl. pitchers).

Outputs: data/abs_player_grades_2026.json + CSVs in ~/Downloads.

Usage: python3 scripts/abs_player_grades.py
"""

import csv
import json
import math
import os
from collections import defaultdict
from datetime import date

import abs_value_engine as ve
from abs_option_model import count_class, edge_region, phi

SHRINK_N0 = 40      # pseudo-challenges pulling a player's sigma to league
SIGMA_GRID = [0.4 + 0.2 * i for i in range(24)]
OBS_BIN = 0.25      # inches, per-player observation binning
CONS_G_MIN = 0.05   # leveraged runs: below this a pitch isn't a real decision
CONS_M_MAX = 2.5    # inches: beyond this the call isn't plausibly challengeable
QUAL_REL = 0.40     # reliability floor to rank a player as a talent estimate
SKILL_PREF = 0.434  # reference break-even p* for leverage-blind skill scoring
                    # (median p* over consequential decisions; a fixed constant
                    # so the skill score carries no leverage information)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "data", "abs_challenges_2026.json")
TABLES = os.path.join(REPO_ROOT, "data", "abs_value_tables_2026.json")
OPTION = os.path.join(REPO_ROOT, "data", "abs_option_model_2026.json")
OUT_JSON = os.path.join(REPO_ROOT, "data", "abs_player_grades_2026.json")
EVENTS_JSON = os.path.join(REPO_ROOT, "data", "abs_challenge_events_2026.json")
DOWNLOADS = os.path.expanduser("~/Downloads")
VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"


def posterior_at(grid, x):
    """Interpolate the [x, p] posterior grid at perceived location x."""
    if x <= grid[0][0]:
        return grid[0][1]
    if x >= grid[-1][0]:
        return grid[-1][1]
    step = grid[1][0] - grid[0][0]
    i = int((x - grid[0][0]) / step)
    x0, p0 = grid[i]
    x1, p1 = grid[i + 1]
    return p0 + (p1 - p0) * (x - x0) / (x1 - x0)


def half_innings_left(inning, half):
    inning = min(inning, 9)
    return 2 * (9 - inning) + (2 if half == "top" else 1)


def solve_xstar(bins, n_chal, sigma):
    """x* such that the probit policy reproduces the player's challenge count."""
    lo, hi = -6.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        pred = sum(n * phi((m - mid) / sigma) for m, (_c, n) in bins.items())
        if pred > n_chal:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fit_player_sigmas(obs_by_player, league_sigma):
    """Per-player perception sigma: probit MLE on the player's own pooled
    (margin, challenged) record, shrunk toward league with SHRINK_N0
    pseudo-challenges.

    This is what makes proximity matter in the grades: sharp-eyed deciders get
    a skill scale > 1, which stretches their margins when evaluated on the
    league confidence curves - deep-in-zone challenges grade higher, way-off
    ones grade lower. Scattershot deciders compress toward flat.
    """
    out = {}
    for pid, ob in obs_by_player.items():
        bins, n_chal = ob["bins"], ob["nChal"]
        if n_chal >= 1 and bins:
            best = None
            for sigma in SIGMA_GRID:
                xs = solve_xstar(bins, n_chal, sigma)
                ll = 0.0
                for m, (c, n) in bins.items():
                    p = min(max(phi((m - xs) / sigma), 1e-9), 1 - 1e-9)
                    ll += c * math.log(p) + (n - c) * math.log(1.0 - p)
                if best is None or ll > best[0]:
                    best = (ll, sigma)
            sig_hat = best[1]
        else:
            sig_hat = league_sigma
        out[pid] = (n_chal * sig_hat + SHRINK_N0 * league_sigma) / (n_chal + SHRINK_N0)
    return out


def new_ledger():
    return {"chalN": 0, "chalWon": 0, "cva": 0.0, "procVal": 0.0, "badChalN": 0,
            "chalMarginSum": 0.0, "missN": 0, "missValue": 0.0, "oppN": 0,
            "consN": 0, "consSum": 0.0, "consSq": 0.0,
            "sklSum": 0.0, "sklSq": 0.0, "teams": defaultdict(int)}


def shrinkage(ledgers, sumkey="consSum", sqkey="consSq"):
    """Empirical-Bayes shrinkage of a per-consequential-decision metric.

    A one-way random-effects (ANOVA) variance decomposition: pooled
    within-player variance (MSW) is binomial-style noise, between-player
    variance is talent. Their ratio gives the stabilization point n0 (decisions
    for 0.5 reliability). Each player's rate is shrunk toward the league mean by
    its reliability, and gets a 95% CI from the normal-normal posterior
    variance MSW / (n + n0). Runs on consSum/consSq (leveraged VALUE) or
    sklSum/sklSq (leverage-blind SKILL). Returns ({pid: {...}}, population).
    """
    pts = [(pid, l["consN"], l[sumkey], l[sqkey])
           for pid, l in ledgers.items() if l["consN"] >= 1]
    N = sum(n for _, n, _, _ in pts)
    k = len(pts)
    if N == 0 or k < 2:
        return {}, {"grand": 0.0, "n0": float("inf"), "sigmaW2": 0.0, "players": k}
    grand = sum(s for _, _, s, _ in pts) / N
    wss = sum(sq - s * s / n for _, n, s, sq in pts if n >= 2)
    dfw = sum(n - 1 for _, n, _, _ in pts if n >= 2)
    msw = wss / dfw if dfw > 0 else 0.0
    msb = sum(n * (s / n - grand) ** 2 for _, n, s, _ in pts) / (k - 1)
    n0bar = (N - sum(n * n for _, n, _, _ in pts) / N) / (k - 1)
    sigb2 = max((msb - msw) / n0bar, 1e-12) if n0bar > 0 else 1e-12
    n0 = msw / sigb2 if sigb2 > 0 else float("inf")
    talent_sd = sigb2 ** 0.5
    out = {}
    for pid, n, s, sq in pts:
        xbar = s / n
        rel = n / (n + n0) if n0 != float("inf") else 0.0
        shrunk = grand + (xbar - grand) * rel
        postvar = msw / (n + n0) if n0 != float("inf") else msw / n
        ci = 1.96 * postvar ** 0.5
        # plus-stat: 100 = league average, 15 index points per talent SD
        plus = 100.0 + (shrunk - grand) / talent_sd * 15.0 if talent_sd > 1e-9 else 100.0
        plusci = ci / talent_sd * 15.0 if talent_sd > 1e-9 else 0.0
        out[pid] = {"consN": n, "rawRate": xbar, "shrunkRate": shrunk,
                    "reliability": rel, "ci95": ci, "plus": plus, "plusCI": plusci}
    return out, {"grand": grand, "n0": n0, "sigmaW2": msw, "talentSD": talent_sd,
                 "players": k}


def main():
    with open(DATASET) as f:
        data = json.load(f)
    with open(OPTION) as f:
        opt = json.load(f)
    tables = ve.tables_from_json(TABLES)
    thr = opt["meta"]["rulingThrIn"]
    Cg = {}
    for key, v in opt["Cgrid"].items():
        k, T, d = key.split("|")
        Cg[(int(k), int(T), int(d))] = v
    pooled_sigma = {s: opt["perceptionPooled"][s]["sigma"] for s in ("bat", "fld")}
    p_look_L = {k: v for k, v in opt["pLook"].items()}      # "side|reg"
    p_sel_L = {k: v for k, v in opt["pSel"].items()}        # "side|reg|cls"
    game_teams = {g["gamePk"]: (g["away"], g["home"]) for g in data["games"]}

    def cost_at(k, T, d):
        return Cg[(max(1, min(2, k)), T, max(-12, min(12, d)))]

    # ---- pass 1: parse records, collect per-player perception observations
    parsed = []
    obs = {"fld": defaultdict(lambda: {"bins": defaultdict(lambda: [0, 0]), "nChal": 0}),
           "bat": defaultdict(lambda: {"bins": defaultdict(lambda: [0, 0]), "nChal": 0})}
    for r in data["records"]:
        if r["distMidIn"] is None:
            continue
        if r["originalCall"] == "strike":
            side, m = "bat", r["distMidIn"] - thr
            wronged = r["batSide"]
        else:
            side, m = "fld", thr - r["distMidIn"]
            wronged = "home" if r["batSide"] == "away" else "away"
        rem = r["remAway"] if wronged == "away" else r["remHome"]
        team_abbr = game_teams[r["gamePk"]][0 if wronged == "away" else 1]
        if wronged == "away":
            d_team = r["awayScore"] - r["homeScore"]
        else:
            d_team = r["homeScore"] - r["awayScore"]
        v = ve.value_of_flip(r["balls"], r["strikes"], r["bases"], r["outs"],
                             r["inning"], r["half"], r["homeScore"] - r["awayScore"],
                             tables)
        g = v["leveragedRuns"]
        T = half_innings_left(r["inning"], r["half"])
        chal = r["challenge"]
        reg = (edge_region(r["pXmid"], r["pZmid"], r["szTop"], r["szBot"])
               if r.get("pXmid") is not None else "side")
        cls = count_class(r["balls"], r["strikes"])
        if side == "bat":
            owner_id, owner_name = r["batterId"], r["batter"]
        else:
            owner_id, owner_name = r["catcherId"], r["catcher"]
        extra = (r["playId"], r["date"], r["balls"], r["strikes"],
                 r["inning"], r["half"], r["batter"], r["catcher"], r["pitcher"])
        parsed.append((side, m, wronged, rem, team_abbr, d_team, g, T, chal,
                       owner_id, owner_name, reg, cls, extra))
        if rem > 0 and owner_id is not None:
            o = obs[side][owner_id]
            b = round(max(-6.0, min(6.0, m)) / OBS_BIN) * OBS_BIN
            o["bins"][b][1] += 1
            owner_challenged = (chal is not None and chal.get("side") == wronged
                                and ((side == "bat" and chal["role"] == "batter")
                                     or (side == "fld" and chal["role"] == "fielder")))
            if owner_challenged:
                o["bins"][b][0] += 1
                o["nChal"] += 1

    psig = {s: fit_player_sigmas(obs[s], pooled_sigma[s]) for s in ("bat", "fld")}
    n_fit = sum(len(psig[s]) for s in psig)
    print(f"fit perception sigma for {n_fit} deciders "
          f"(shrunk toward league with n0={SHRINK_N0})")

    def skill_scale(side_key, pid):
        """>1 = sharper than league; margins stretch by this on league curves."""
        sp = psig[side_key].get(pid)
        return pooled_sigma[side_key] / sp if sp else 1.0

    catchers = defaultdict(new_ledger)   # id -> ledger (fielding side)
    hitters = defaultdict(new_ledger)    # id -> ledger (batting side)
    pitchers = defaultdict(new_ledger)   # pitcher-initiated challenges only
    teams = defaultdict(new_ledger)
    names = {}
    sigmas = {}

    # ---- pass 2: grade
    events = []   # every challenge + every counted miss, with Savant video ids
    for (side, m, wronged, rem, team_abbr, d_team, g, T, chal,
         owner_id, owner_name, reg, cls, extra) in parsed:
        play_id, ev_date, balls, strikes, inning, half, ev_batter, ev_catcher, ev_pitcher = extra
        if side == "bat":
            book = hitters
        else:
            book = catchers
        if owner_id is not None:
            led = book[owner_id]
            names[owner_id] = owner_name
            led["teams"][team_abbr] += 1
            led["oppN"] += 1
            sp = psig[side].get(owner_id)
            if sp is not None:
                sigmas[owner_id] = sp
            # consequential-decision talent metric: LEAGUE curves (no player
            # skill scale), so this is a clean input for shrinkage/reliability
            if rem > 0 and g >= CONS_G_MIN and abs(m) <= CONS_M_MAX:
                oc = (chal is not None and chal.get("side") == wronged
                      and ((side == "bat" and chal["role"] == "batter")
                           or (side == "fld" and chal["role"] == "fielder")))
                cc = cost_at(rem, T, d_team)
                if oc:
                    pc = posterior_at(p_sel_L[f"{side}|{reg}|{cls}"], m)
                    contrib = pc * g - (1.0 - pc) * cc
                else:
                    pc = posterior_at(p_look_L[f"{side}|{reg}"], m)
                    evl = pc * g - (1.0 - pc) * cc
                    contrib = -evl if evl > 0 else 0.0
                # leverage-BLIND skill: confidence-weighted decision correctness.
                # Right action (challenge iff pc>=ref) scores +|pc-ref|, wrong
                # scores -|pc-ref|. No g, no C, no leverage -> pure read/judgment.
                correct = (pc >= SKILL_PREF) == oc
                skl = (1.0 if correct else -1.0) * abs(pc - SKILL_PREF)
                led["consN"] += 1
                led["consSum"] += contrib
                led["consSq"] += contrib * contrib
                led["sklSum"] += skl
                led["sklSq"] += skl * skl

        if chal is not None and chal.get("side") == wronged:
            k = chal.get("remainingBefore") or rem or 1
            cost = cost_at(k, T, d_team)
            value = g if chal["overturned"] else -cost      # realized (reference)
            pid = chal.get("playerId")
            pname = chal.get("playerName")
            if chal["role"] == "batter":
                led_c, c_side = hitters[pid], "bat"
            elif chal["role"] == "pitcher":
                led_c, c_side = pitchers[pid], "fld"
            else:
                led_c, c_side = catchers[pid], "fld"
            scale = 1.0 if chal["role"] == "pitcher" else skill_scale(c_side, pid)
            p_conf = posterior_at(p_sel_L[f"{c_side}|{reg}|{cls}"], m * scale)
            ev = p_conf * g - (1.0 - p_conf) * cost          # decision grade
            if pid is not None:
                names[pid] = pname
            led_c["chalN"] += 1
            led_c["chalWon"] += chal["overturned"]
            led_c["cva"] += value
            led_c["procVal"] += ev
            led_c["badChalN"] += ev < 0
            led_c["chalMarginSum"] += m
            led_c["teams"][team_abbr] += 1
            teams[team_abbr]["chalN"] += 1
            teams[team_abbr]["chalWon"] += chal["overturned"]
            teams[team_abbr]["cva"] += value
            teams[team_abbr]["procVal"] += ev
            teams[team_abbr]["badChalN"] += ev < 0
            events.append({"type": "challenge", "player": pname, "team": team_abbr,
                           "batter": ev_batter, "catcher": ev_catcher,
                           "pitcher": ev_pitcher,
                           "date": ev_date, "role": chal["role"],
                           "count": f"{balls}-{strikes}", "inning": inning,
                           "half": half, "marginIn": round(m, 2),
                           "gain": round(g, 3), "ev": round(ev, 3),
                           "result": "won" if chal["overturned"] else "lost",
                           "playId": play_id})
        elif chal is None and rem > 0 and g > 0:
            cost = cost_at(rem, T, d_team)
            p_conf = posterior_at(p_look_L[f"{side}|{reg}"],
                                  m * skill_scale(side, owner_id))
            ev = p_conf * g - (1.0 - p_conf) * cost
            if ev > 0:                                       # matrix-approved gamble declined
                if owner_id is not None:
                    led["missN"] += 1
                    led["missValue"] += ev
                teams[team_abbr]["missN"] += 1
                teams[team_abbr]["missValue"] += ev
                events.append({"type": "miss", "player": owner_name,
                               "batter": ev_batter, "catcher": ev_catcher,
                               "pitcher": ev_pitcher,
                               "team": team_abbr, "date": ev_date,
                               "role": "fielder" if side == "fld" else "batter",
                               "count": f"{balls}-{strikes}", "inning": inning,
                               "half": half, "marginIn": round(m, 2),
                               "gain": round(g, 3), "ev": round(ev, 3),
                               "result": "would-win" if m > 0 else "would-lose",
                               "playId": play_id})

    cat_val, cat_vpop = shrinkage(catchers, "consSum", "consSq")
    hit_val, hit_vpop = shrinkage(hitters, "consSum", "consSq")
    cat_skl, cat_spop = shrinkage(catchers, "sklSum", "sklSq")
    hit_skl, hit_spop = shrinkage(hitters, "sklSum", "sklSq")
    def n0str(p):
        return "inf (no talent spread)" if p["n0"] > 1e5 else f"{p['n0']:.0f}"
    print(f"catcher VALUE n0={n0str(cat_vpop)} decisions (league "
          f"{100*cat_vpop['grand']:.3f} lev.runs/100); "
          f"SKILL n0={n0str(cat_spop)}; hitter VALUE n0={n0str(hit_vpop)}, "
          f"SKILL n0={n0str(hit_spop)}")

    def rows(book, val, skl, min_opp=0):
        out = []
        for pid, led in book.items():
            if led["chalN"] == 0 and led["missN"] == 0:
                continue
            if led["oppN"] < min_opp and led["chalN"] == 0:
                continue
            team = max(led["teams"], key=led["teams"].get) if led["teams"] else ""
            vs = val.get(pid)
            ks = skl.get(pid)
            vrel = vs["reliability"] if vs else None
            krel = ks["reliability"] if ks else None
            vq = bool(vrel is not None and vrel >= QUAL_REL)
            kq = bool(krel is not None and krel >= QUAL_REL)
            out.append({
                "playerId": pid, "player": names.get(pid, str(pid)), "team": team,
                "challenges": led["chalN"], "won": led["chalWon"],
                "successPct": (100.0 * led["chalWon"] / led["chalN"]) if led["chalN"] else None,
                "avgChalMargin": (led["chalMarginSum"] / led["chalN"]) if led["chalN"] else None,
                "readSigma": sigmas.get(pid),
                "consN": vs["consN"] if vs else 0,
                "netValue": led["procVal"] - led["missValue"],
                # leverage-BLIND skill index (100 = league avg, +-15 per talent SD)
                "skill": ks["plus"] if ks else None,
                "skillCI": ks["plusCI"] if ks else None,
                "skillRel": krel, "skillQual": kq,
                # leveraged VALUE added per 100 consequential decisions
                "value": (100.0 * vs["shrunkRate"]) if vs else None,
                "valueCI": (100.0 * vs["ci95"]) if vs else None,
                "valueRel": vrel, "valueQual": vq,
            })
        # rank by leverage-blind skill (qualified first), then value, then net
        out.sort(key=lambda r: (r["skillQual"],
                                r["skill"] if r["skillQual"] else -1e9,
                                r["netValue"]), reverse=True)
        return out

    result = {
        "meta": {"generated": date.today().isoformat(),
                 "games": data["meta"]["games"], "rulingThrIn": thr,
                 "catValN0": round(cat_vpop["n0"], 1), "catSklN0": round(cat_spop["n0"], 1),
                 "qualRel": QUAL_REL, "consGmin": CONS_G_MIN, "consMmax": CONS_M_MAX,
                 "note": "TWO orthogonal talent metrics, each empirical-Bayes shrunk "
                         "with a 95% CI and qualified only at reliability>=0.40. "
                         "skill = leverage-blind decision quality (confidence-weighted "
                         "correctness x100), independent of stakes. value = leveraged "
                         "runs added per 100 consequential decisions (skill x leverage "
                         "faced). netValue/successPct are descriptive, not talent."},
        "catchers": rows(catchers, cat_val, cat_skl),
        "hitters": rows(hitters, hit_val, hit_skl),
        "pitchers": rows(pitchers, {}, {}), "teams": rows(teams, {}, {}),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {OUT_JSON}")

    events.sort(key=lambda e: (e["date"], e["team"], e["inning"]))
    with open(EVENTS_JSON, "w") as f:
        json.dump({"meta": result["meta"], "events": events}, f, separators=(",", ":"))
    print(f"wrote {EVENTS_JSON} ({len(events)} events)")
    for etype, fname in (("challenge", "abs_challenge_log_2026.csv"),
                         ("miss", "abs_missed_opps_2026.csv")):
        path = os.path.join(DOWNLOADS, fname)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Player", "Tm", "Role", "Inning", "Half", "Count",
                        "MarginIn", "Gain", "DecisionEV", "Result", "VideoURL"])
            for e in events:
                if e["type"] != etype:
                    continue
                w.writerow([e["date"], e["player"], e["team"], e["role"],
                            e["inning"], e["half"], e["count"], e["marginIn"],
                            round(e["gain"], 2), round(e["ev"], 2), e["result"],
                            VIDEO_URL.format(pid=e["playId"])])
        print(f"wrote {path}")

    for key, fname in (("catchers", "abs_catcher_grades_2026.csv"),
                       ("hitters", "abs_hitter_grades_2026.csv"),
                       ("teams", "abs_team_grades_2026.csv")):
        path = os.path.join(DOWNLOADS, fname)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Player" if key != "teams" else "Team", "Tm",
                        "Skill", "SkillCI", "SkillRel", "SkillQual",
                        "Value100", "ValueCI", "ValueRel", "ValueQual",
                        "ConsDecisions", "Challenges", "Success%",
                        "AvgChalMargin", "ReadSigma", "NetValue"])
            for r in result[key]:
                w.writerow([
                    r["player"] if key != "teams" else r["team"], r["team"],
                    "" if r["skill"] is None else round(r["skill"], 2),
                    "" if r["skillCI"] is None else round(r["skillCI"], 2),
                    "" if r["skillRel"] is None else round(r["skillRel"], 2),
                    "Y" if r["skillQual"] else "",
                    "" if r["value"] is None else round(r["value"], 2),
                    "" if r["valueCI"] is None else round(r["valueCI"], 2),
                    "" if r["valueRel"] is None else round(r["valueRel"], 2),
                    "Y" if r["valueQual"] else "",
                    r["consN"], r["challenges"],
                    "" if r["successPct"] is None else round(r["successPct"]),
                    "" if r["avgChalMargin"] is None else round(r["avgChalMargin"], 2),
                    "" if r.get("readSigma") is None else round(r["readSigma"], 2),
                    round(r["netValue"], 2)])
        print(f"wrote {path}")

    def show(title, rs, key, n=8):
        print(f"\n{title}")
        for r in rs[:n]:
            sk = " n/a" if r["skill"] is None else f"{r['skill']:+6.2f}"
            sci = "" if r["skillCI"] is None else f"+-{r['skillCI']:.2f}"
            vl = " n/a" if r["value"] is None else f"{r['value']:+5.2f}"
            vci = "" if r["valueCI"] is None else f"+-{r['valueCI']:.2f}"
            print(f"  {r['player']:<22} {r['team']:<4} SKILL {sk} {sci:>7} | "
                  f"VALUE {vl} {vci:>7} | dec {r['consN']:>3} chal {r['challenges']:>3}")

    qs = [r for r in result["catchers"] if r["skillQual"]]
    qv = [r for r in result["catchers"] if r["valueQual"]]
    print(f"\nqualified catchers: SKILL {len(qs)}, VALUE {len(qv)} of {len(result['catchers'])}")
    show("TOP CATCHERS by leverage-blind SKILL:", qs, "skill")
    show("TOP CATCHERS by leveraged VALUE:",
         sorted(qv, key=lambda r: r["value"], reverse=True), "value")
    show("TOP HITTERS (DESCRIPTIVE ONLY):", result["hitters"], "skill", 5)
    n_miss = sum(led["missN"] for led in teams.values())
    v_miss = sum(led["missValue"] for led in teams.values())
    print(f"\nleague missed opportunities: {n_miss} worth {v_miss:.1f} leveraged runs of decision EV")
    print(f"catcher SKILL talent SD = {cat_spop['talentSD']*100:.3f}/100 (small = catchers "
          f"uniform in leverage-blind skill); VALUE talent SD = {cat_vpop['talentSD']*100:.3f}/100")


if __name__ == "__main__":
    main()
