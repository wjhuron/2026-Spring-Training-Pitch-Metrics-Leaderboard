"""Drop players who have since signed, and format the FA sheets for Numbers.

Input is the hand-curated Numbers file (a trimmed copy of the report from
milb_fa_fangraphs_report.py). Signing status comes from the MLB Stats API
transactions feed, which covers affiliated and Mexican League deals.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd
from numbers_parser import Document

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/"
    "3d1e3d65-a140-4663-afec-7bd02ee9d72c/scratchpad"
)
SRC = Path("/Users/wallyhuron/Downloads/milb_fa_2026_fangraphs.numbers")
OUT_DIR = Path("/Users/wallyhuron/Downloads")
OUT_XLSX = OUT_DIR / "milb_fa_2026_unsigned.xlsx"

TX_URL = (
    "https://statsapi.mlb.com/api/v1/transactions"
    "?startDate=2026-04-01&endDate=2026-07-24"
)
# transaction types that mean a club now controls him
SIGNED_TYPES = {
    "Signed as Free Agent", "Signed", "Claimed Off Waivers",
    "Selected", "Trade", "Acquired", "Obtained",
}

PCT_COLS = ["K%", "BB%", "K-BB%", "GB%"]
RATE3_COLS = ["AVG", "OBP", "SLG", "OPS", "BABIP"]
RATE2_COLS = ["ERA", "FIP", "xFIP"]
IP_COLS = ["IP", "MLB_IP"]


def load_sheets():
    doc = Document(str(SRC))
    out = {}
    for sheet in doc.sheets:
        rows = sheet.tables[0].rows(values_only=True)
        out[sheet.name] = pd.DataFrame(rows[1:], columns=rows[0])
    return out


def load_transactions():
    cache = SCRATCH / "tx.json"
    if not cache.exists():
        with urllib.request.urlopen(TX_URL, timeout=120) as r:
            cache.write_bytes(r.read())
    return json.load(open(cache))["transactions"]


def signing_index(tx):
    """mlbam id -> list of (date, description) for control-establishing moves."""
    idx = {}
    for t in tx:
        if t.get("typeDesc") not in SIGNED_TYPES:
            continue
        pid = (t.get("person") or {}).get("id")
        if pid is None:
            continue
        idx.setdefault(pid, []).append(
            (t.get("effectiveDate") or t.get("date"), t.get("description", ""))
        )
    return idx


def main():
    sheets = load_sheets()
    ids = pd.concat(
        [
            pd.read_csv(SCRATCH / "player_ids_PA.csv"),
            pd.read_csv(SCRATCH / "player_ids_IP.csv"),
        ]
    ).drop_duplicates(subset="Player")
    id_map = dict(zip(ids["Player"], ids["xMLBAMID"].astype(int)))
    rel_map = dict(zip(ids["Player"], ids["ReleaseDate"].astype(str)))

    idx = signing_index(load_transactions())

    kept, dropped, unknown = {}, [], []
    for name, df in sheets.items():
        flags = []
        for player in df["Player"]:
            pid = id_map.get(player)
            if pid is None:
                unknown.append(player)
                flags.append(False)
                continue
            rel = rel_map.get(player, "1900-01-01")
            after = [(d, desc) for d, desc in idx.get(pid, []) if d and d > rel]
            if after:
                after.sort()
                dropped.append((name, player, rel, after[0][0], after[0][1]))
                flags.append(True)
            else:
                flags.append(False)
        kept[name] = df[~pd.Series(flags, index=df.index)].reset_index(drop=True)

    print(f"signed since release, removed: {len(dropped)}")
    for sheet, player, rel, when, desc in sorted(dropped, key=lambda r: (r[0], r[3])):
        print(f"   {sheet[:3]} {player:22s} released {rel} -> {when}  {desc[:80]}")
    if unknown:
        print(f"no MLBAM id (left in): {unknown}")

    write(kept)
    for name, df in kept.items():
        print(f"{name}: {len(sheets[name])} -> {len(df)}")


def write(sheets):
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        for name, df in sheets.items():
            out = df.copy()
            # percent cells are stored as fractions so the 0.0% format reads right
            for c in PCT_COLS:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce") / 100.0
            out.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            fmt = {}
            for c in PCT_COLS:
                fmt[c] = "0.0%"
            for c in RATE3_COLS:
                fmt[c] = ".000"
            for c in RATE2_COLS:
                fmt[c] = "0.00"
            for c in IP_COLS:
                fmt[c] = "0.0"
            for i, col in enumerate(out.columns, start=1):
                if col not in fmt:
                    continue
                letter = ws.cell(row=1, column=i).column_letter
                for cell in ws[letter][1:]:
                    cell.number_format = fmt[col]
                ws.column_dimensions[letter].width = max(8, len(col) + 3)


if __name__ == "__main__":
    main()
