"""Build 2026 FanGraphs stat sheets for the MiLB free-agent list.

Reads the released-player list exported from Numbers, joins it to FanGraphs
2026 minor-league (per level + season total) and major-league leaderboards,
applies the playing-time filters, and writes Hitters/Pitchers deliverables.

FanGraphs JSON payloads are pulled separately (Cloudflare blocks curl) and
cached in SCRATCH as milb_{bat,pit}_{TOTAL,AAA,AA,Aplus,A,CPX}.json and
mlb_{bat,pit}.json.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from numbers_parser import Document

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/"
    "3d1e3d65-a140-4663-afec-7bd02ee9d72c/scratchpad"
)
FA_FILE = Path("/Users/wallyhuron/Downloads/milb_free_agents_2026-07-11.numbers")
OUT_DIR = Path("/Users/wallyhuron/Downloads")

LEVEL_ORDER = ["TOTAL", "MLB", "AAA", "AA", "A+", "A", "CPX"]
MILB_SPLITS = {"AAA": "AAA", "AA": "AA", "Aplus": "A+", "A": "A", "CPX": "CPX"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# full club name -> FanGraphs org abbreviation, used to break same-name ties
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "OAK", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def norm(name: str) -> str:
    """Accent/punctuation-insensitive key, suffixes dropped."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z ]", " ", s).lower()
    parts = [p for p in s.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def load_free_agents() -> pd.DataFrame:
    doc = Document(str(FA_FILE))
    rows = doc.sheets[0].tables[0].rows(values_only=True)
    fa = pd.DataFrame(rows[1:], columns=rows[0])
    # source export swaps Player/Released By on the last ~25 rows; the player
    # cell is always "Last, First" while the org cell never has a comma
    swap = ~fa["Player"].astype(str).str.contains(",") & fa[
        "Released By"
    ].astype(str).str.contains(",")
    fa.loc[swap, ["Player", "Released By"]] = fa.loc[
        swap, ["Released By", "Player"]
    ].values

    # "Hatch, Thomas" -> "Thomas Hatch"
    def flip(n):
        n = str(n).strip()
        if "," in n:
            last, first = n.split(",", 1)
            return f"{first.strip()} {last.strip()}"
        return n
    fa["FullName"] = fa["Player"].map(flip)
    fa["key"] = fa["FullName"].map(norm)
    fa = fa.rename(columns={"Date": "ReleaseDate", "Level": "ReleaseLevel"})
    return fa[
        ["key", "FullName", "Position", "Released By", "ReleaseDate",
         "Transaction Type", "ReleaseLevel"]
    ]


def load_fg(fname: str) -> pd.DataFrame:
    df = pd.DataFrame(json.load(open(SCRATCH / f"{fname}.json")))
    df["key"] = df["PlayerName"].map(norm)
    return df


HIT_COLS = ["PA", "AVG", "OBP", "SLG", "OPS", "BABIP", "K%", "BB%", "wRC+"]
PIT_COLS = ["G", "GS", "IP", "K%", "BB", "K-BB%", "GB%", "ERA", "FIP", "xFIP"]


def slice_stats(df: pd.DataFrame, cols, level, team_col, extra_level_col=None):
    out = df.copy()
    out["Level"] = level
    out["Team"] = out[team_col]
    out["LevelsPlayed"] = out[extra_level_col] if extra_level_col else level
    id_col = "playerids" if "playerids" in out.columns else "playerid"
    out["fgid"] = out[id_col].astype(str)
    keep = [
        "key", "fgid", "xMLBAMID", "PlayerName", "Level", "Team",
        "LevelsPlayed", "Age",
    ] + cols
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    return out[keep]


def build(stat: str):
    cols = HIT_COLS if stat == "bat" else PIT_COLS
    frames = []

    total = load_fg(f"milb_{stat}_TOTAL")
    frames.append(slice_stats(total, cols, "TOTAL", "AffAbbName", "aLevel"))

    for tag, label in MILB_SPLITS.items():
        d = load_fg(f"milb_{stat}_{tag}")
        frames.append(slice_stats(d, cols, label, "AffAbbName"))

    mlb = load_fg(f"mlb_{stat}")
    frames.append(slice_stats(mlb, cols, "MLB", "TeamNameAbb"))

    return pd.concat(frames, ignore_index=True)


def resolve_name_collisions(df: pd.DataFrame, fa: pd.DataFrame, label: str):
    """Two FanGraphs players can share a name; keep the one the FA row means.

    Preference order: the org that released him, then a level match against the
    level he was released from, then the larger sample.
    """
    fa_by_key = fa.set_index("key")
    keep = []
    for key, grp in df.groupby("key", sort=False):
        ids = list(grp["fgid"].unique())
        if len(ids) == 1:
            keep.append(grp)
            continue
        row = fa_by_key.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        abbr = TEAM_ABBR.get(str(row["Released By"]))
        rel_lv = str(row["ReleaseLevel"])
        scored = []
        for i in ids:
            sub = grp[grp["fgid"] == i]
            levels = set()
            for s in sub["LevelsPlayed"].astype(str):
                levels.update(x.strip() for x in s.split(","))
            score = 0
            if abbr and abbr in set(sub["Team"].astype(str)):
                score += 10
            if rel_lv in levels:
                score += 5
            size = len(sub)
            scored.append((score, size, i))
        scored.sort(reverse=True)
        best = scored[0]
        if best[0] == 0:
            print(f"   WARNING unresolved name tie in {label}: {key} -> kept fgid {best[2]}")
        else:
            others = [s[2] for s in scored[1:]]
            print(f"   note: {label} name tie {key} -> fgid {best[2]} (dropped {others})")
        keep.append(grp[grp["fgid"] == best[2]])
    return pd.concat(keep, ignore_index=True) if keep else df


def main():
    fa = load_free_agents()
    bat = build("bat")
    pit = build("pit")

    is_pitcher = fa["Position"].astype(str).str.upper().str.contains("HP|^P$", regex=True)

    bat = resolve_name_collisions(bat[bat["key"].isin(set(fa["key"]))], fa, "hitters")
    pit = resolve_name_collisions(pit[pit["key"].isin(set(fa["key"]))], fa, "pitchers")

    bat_j = fa.merge(bat, on="key", how="inner")
    pit_j = fa.merge(pit, on="key", how="inner")

    # --- playing-time filters -------------------------------------------------
    def hitter_pa(k):
        rows = bat_j[(bat_j["key"] == k) & bat_j["Level"].isin(["TOTAL", "MLB"])]
        return pd.to_numeric(rows["PA"], errors="coerce").fillna(0).sum()

    def pitcher_ok(k):
        rows = pit_j[pit_j["key"] == k]
        levels = set(rows["Level"])
        lp = " ".join(rows["LevelsPlayed"].astype(str))
        if "MLB" in levels or "AAA" in levels or "AAA" in lp:
            return True
        ip = pd.to_numeric(
            rows[rows["Level"].isin(["TOTAL", "MLB"])]["IP"], errors="coerce"
        ).fillna(0).sum()
        return ip >= 10

    hit_keys = {k for k in bat_j["key"].unique() if hitter_pa(k) >= 100}
    pit_keys = {k for k in pit_j["key"].unique() if pitcher_ok(k)}

    # the FA list only ever lists RHP/LHP or a fielding position, so the listed
    # position decides the sheet (keeps position players who mopped up an
    # inning out of the pitcher sheet)
    pitcher_keys_listed = set(fa.loc[is_pitcher, "key"])
    hitters = bat_j[
        bat_j["key"].isin(hit_keys) & ~bat_j["key"].isin(pitcher_keys_listed)
    ].copy()
    pitchers = pit_j[
        pit_j["key"].isin(pit_keys) & pit_j["key"].isin(pitcher_keys_listed)
    ].copy()

    for df in (hitters, pitchers):
        df["Level"] = pd.Categorical(df["Level"], LEVEL_ORDER, ordered=True)
        df.sort_values(["FullName", "Level"], inplace=True)

    def finalize(df, cols, mlb_stat):
        """One row per player: the MiLB season total, or the MLB line for the
        handful who never appeared in the minors in 2026."""
        out = df.rename(columns={"FullName": "Player", "Released By": "ReleasedBy"})

        # how much big-league time sits outside the MiLB total
        mlb = out[out["Level"] == "MLB"].set_index("key")[mlb_stat]
        mlb = mlb[~mlb.index.duplicated()]

        total = out[out["Level"] == "TOTAL"]
        mlb_only = out[out["Level"] == "MLB"] [
            ~out[out["Level"] == "MLB"]["key"].isin(set(total["key"]))
        ]
        out = pd.concat([total, mlb_only], ignore_index=True)

        out["Source"] = out["Level"].map(
            lambda lv: "MiLB total" if lv == "TOTAL" else "MLB only"
        )
        out[f"MLB_{mlb_stat}"] = out["key"].map(mlb).fillna(0)

        # side file: MLBAM ids keyed by display name, for the signing check
        ids = out[["Player", "xMLBAMID", "ReleaseDate", "ReleasedBy"]].dropna(
            subset=["xMLBAMID"]
        )
        ids.to_csv(SCRATCH / f"player_ids_{mlb_stat}.csv", index=False)

        ordered = [
            "Player", "Position", "ReleasedBy", "ReleaseDate", "ReleaseLevel",
            "Age", "Team", "Source", "LevelsPlayed",
        ] + cols + [f"MLB_{mlb_stat}"]
        return out[ordered].sort_values("Player").reset_index(drop=True)

    hitters = finalize(hitters, HIT_COLS, "PA")
    pitchers = finalize(pitchers, PIT_COLS, "IP")

    # rounding for readability; rate stats keep FanGraphs precision
    for df, rate3, rate1 in (
        (hitters, ["AVG", "OBP", "SLG", "OPS", "BABIP"], []),
        (pitchers, [], ["ERA", "FIP", "xFIP", "IP", "MLB_IP"]),
    ):
        for c in rate3:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
        for c in rate1:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    for c in ["K%", "BB%"]:
        hitters[c] = (pd.to_numeric(hitters[c], errors="coerce") * 100).round(1)
    for c in ["K%", "K-BB%", "GB%"]:
        pitchers[c] = (pd.to_numeric(pitchers[c], errors="coerce") * 100).round(1)
    hitters["wRC+"] = pd.to_numeric(hitters["wRC+"], errors="coerce").round(0)

    hitters = hitters.sort_values("wRC+", ascending=False).reset_index(drop=True)
    pitchers = pitchers.sort_values("K-BB%", ascending=False).reset_index(drop=True)

    matched = set(hitters["Player"]) | set(pitchers["Player"])
    unmatched = sorted(set(fa["FullName"]) - set(bat_j["FullName"]) - set(pit_j["FullName"]))
    filtered_out = sorted(set(fa["FullName"]) - matched - set(unmatched))

    nodata = fa[fa["FullName"].isin(unmatched)][
        ["FullName", "Position", "Released By", "ReleaseDate", "ReleaseLevel"]
    ].rename(columns={"FullName": "Player", "Released By": "ReleasedBy"})
    nodata["Note"] = "no 2026 FanGraphs line (did not play / injured)"

    OUT_DIR.mkdir(exist_ok=True)
    hitters.to_csv(OUT_DIR / "milb_fa_2026_hitters.csv", index=False)
    pitchers.to_csv(OUT_DIR / "milb_fa_2026_pitchers.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "milb_fa_2026_fangraphs.xlsx") as xw:
        hitters.to_excel(xw, sheet_name="Hitters", index=False)
        pitchers.to_excel(xw, sheet_name="Pitchers", index=False)
        nodata.to_excel(xw, sheet_name="No 2026 Data", index=False)

    # same-name collisions would duplicate a TOTAL row for one FA
    for label, df in (("hitters", hitters), ("pitchers", pitchers)):
        dup = df["Player"].value_counts()
        for name, n in dup[dup > 1].items():
            print(f"   WARNING duplicate name in {label}: {name} ({n} FanGraphs rows)")

    print(f"FA list: {len(fa)} players")
    print(f"Hitters kept: {hitters['Player'].nunique()} ({len(hitters)} rows)")
    print(f"Pitchers kept: {pitchers['Player'].nunique()} ({len(pitchers)} rows)")
    print(f"No FanGraphs 2026 line: {len(unmatched)}")
    for n in unmatched:
        print("   NOMATCH", n)
    print(f"Dropped by playing-time filter: {len(filtered_out)}")
    for n in filtered_out:
        print("   FILTERED", n)


if __name__ == "__main__":
    main()
