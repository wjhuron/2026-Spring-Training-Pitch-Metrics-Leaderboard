"""Generate the ABS grades + backtest leaderboard artifact page."""
import base64
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "artifacts", "abs_tables.html")

with open(os.path.join(REPO, "data", "abs_player_grades_2026.json")) as f:
    GR = json.load(f)
with open(os.path.join(REPO, "data", "abs_challenge_events_2026.json")) as f:
    EV = json.load(f)["events"]
backtest = []
with open(os.path.expanduser("~/Downloads/abs_backtest_2026.csv")) as f:
    for row in csv.DictReader(f):
        backtest.append({"team": row["Team"], "games": int(row["Games"]),
                         "actW": float(row["ActualWins"]),
                         "optW": float(row["OptimalWins"]),
                         "gapW": float(row["WinsLeftOnTable"])})


def font_b64(name):
    with open(os.path.join(REPO, "assets", "fonts", name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def slim(rows):
    out = []
    for r in rows:
        # qualified (reliability>=0.40) players show the shrunk talent estimate;
        # everyone else reads "provisional" so the board never fakes precision
        sq, vq = r.get("skillQual"), r.get("valueQual")
        out.append({
            "player": r["player"], "team": r["team"],
            "skill": None if (r.get("skill") is None or not sq) else round(r["skill"]),
            "skci": None if (r.get("skillCI") is None or not sq) else round(r["skillCI"]),
            "value": None if (r.get("value") is None or not vq) else round(r["value"], 2),
            "vci": None if (r.get("valueCI") is None or not vq) else round(r["valueCI"], 2),
            "cons": r.get("consN", 0), "chal": r["challenges"],
            "succ": None if r["successPct"] is None else round(r["successPct"]),
            "net": round(r["netValue"], 2),
        })
    return out


data = {
    "generated": GR["meta"]["generated"],
    "catchers": slim(GR["catchers"]),
    "hitters": slim(GR["hitters"]),
    "teams": slim(GR["teams"]),
    "backtest": backtest,
    "film": [{"player": e["player"], "team": e["team"], "date": e["date"],
              "type": e["type"], "result": e["result"], "count": e["count"],
              "inning": e["inning"], "half": e["half"][:3],
              "marginIn": e["marginIn"], "ev": round(e["ev"], 2),
              "playId": e["playId"]} for e in EV],
}

page = """<title>ABS Challenge Grades &mdash; Huronalytics</title>
<style>
@font-face{font-family:'Bitter';font-weight:700;src:url(data:font/ttf;base64,__BITTER700__) format('truetype');font-display:swap}
@font-face{font-family:'IBM Plex Sans';font-weight:400;src:url(data:font/ttf;base64,__PLEX400__) format('truetype');font-display:swap}
@font-face{font-family:'IBM Plex Sans';font-weight:600;src:url(data:font/ttf;base64,__PLEX600__) format('truetype');font-display:swap}
:root{--paper:#F7F2E9;--card:#FDFAF4;--ink:#2B2320;--ink2:#71645C;--line:rgba(43,35,32,.16);
  --accent:#C4593A;--accent-ink:#8F3A22;--pos:#5F7A54;--neg:#A8452A}
@media (prefers-color-scheme: dark){:root{--paper:#211B16;--card:#2A231D;--ink:#F0E7DA;--ink2:#A99A8E;
  --line:rgba(240,231,218,.16);--accent:#E0714A;--accent-ink:#EE9070;--pos:#93AC85;--neg:#E58767}}
:root[data-theme="light"]{--paper:#F7F2E9;--card:#FDFAF4;--ink:#2B2320;--ink2:#71645C;--line:rgba(43,35,32,.16);
  --accent:#C4593A;--accent-ink:#8F3A22;--pos:#5F7A54;--neg:#A8452A}
:root[data-theme="dark"]{--paper:#211B16;--card:#2A231D;--ink:#F0E7DA;--ink2:#A99A8E;
  --line:rgba(240,231,218,.16);--accent:#E0714A;--accent-ink:#EE9070;--pos:#93AC85;--neg:#E58767}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:15px/1.5 'IBM Plex Sans',system-ui,sans-serif;margin:0;padding:32px 20px 64px}
.wrap{max-width:1060px;margin:0 auto}
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:clamp(28px,5vw,40px);margin:0 0 12px}
.sub{color:var(--ink2);max-width:70ch;margin:0 0 22px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.tabs button{border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink2);
  font:600 14px 'IBM Plex Sans';padding:8px 16px;cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=search]{border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);
  font:14px 'IBM Plex Sans';padding:8px 12px;width:220px;margin-left:auto}
.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.count{color:var(--ink2);font-size:13px}
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;min-width:900px}
th{position:sticky;top:0;background:var(--card);font-size:11px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink2);text-align:right;padding:10px 10px;border-bottom:1px solid var(--line);
  cursor:pointer;white-space:nowrap;user-select:none}
th.l{text-align:left}
th[aria-sort]:not([aria-sort="none"]){color:var(--accent-ink)}
td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);font-size:13.5px;white-space:nowrap}
td.l{text-align:left;font-weight:600}
td.dim{color:var(--ink2);font-weight:400}
tr:last-child td{border-bottom:0}
.pos{color:var(--pos);font-weight:600}
.neg{color:var(--neg);font-weight:600}
.foot{color:var(--ink2);font-size:13px;max-width:76ch;margin-top:18px}
</style>
<div class="wrap">
<p class="eyebrow">Huronalytics &middot; 2026 ABS Challenge System</p>
<h1>Challenge Grades &amp; Backtest</h1>
<p class="sub">Two orthogonal talent metrics, decision-based (a justified challenge grades positive even when it loses).
<b>Skill+</b> is leverage-blind decision quality indexed to 100, independent of the stakes a catcher faced.
<b>Value/100</b> is leveraged runs added per 100 real decisions (skill times leverage times volume). Both are
empirical-Bayes shrunk with 95% CIs and shown only where reliable. Sort by either; they rank differently on
purpose. Click any column.</p>
<div class="bar">
  <div class="tabs" id="tabs">
    <button data-t="catchers" aria-pressed="true">Catchers</button>
    <button data-t="hitters" aria-pressed="false">Hitters</button>
    <button data-t="teams" aria-pressed="false">Teams</button>
    <button data-t="backtest" aria-pressed="false">Backtest</button>
    <button data-t="film" aria-pressed="false">Film Room</button>
  </div>
  <input type="search" id="q" placeholder="Search player or team" aria-label="Search">
  <span class="count" id="count"></span>
</div>
<div class="tblwrap"><table id="tbl"></table></div>
<p class="foot" id="footNote"></p>
</div>
<script>
const D=__DATA__;
const COLS={
 player:[["player","Player","l"],["team","Tm","l"],["skill","Skill+"],["skci","95% CI"],
  ["value","Value/100"],["vci","95% CI"],["cons","Dec"],["chal","Chal"],["succ","Succ%"],["net","NetVal"]],
 backtest:[["team","Team","l"],["games","G"],["actW","ActualWins"],["optW","OptimalWins"],["gapW","LeftOnTable"]],
 film:[["date","Date","l"],["player","Player","l"],["team","Tm","l"],["type","Type","l"],["result","Result","l"],
  ["count","Cnt"],["inning","Inn"],["half","Half","l"],["marginIn","Margin"],["ev","EV"],["playId","Video","l"]]
};
const NOTES={
 catchers:"TWO orthogonal metrics. SKILL+ = leverage-blind decision quality indexed to 100 (league average), 15 points per talent SD, independent of the stakes a catcher happened to face. VALUE/100 = leveraged runs added per 100 consequential decisions (skill x leverage x volume). They are nearly uncorrelated by design, and revealingly, the highest-VALUE catchers are NOT the highest-SKILL ones: raw value is mostly driven by leverage and volume, not per-decision judgment. Both shown only for QUALIFIED catchers (reliability >= 0.40); the wide Skill+ CIs (about +-20) mean most catchers are statistically indistinguishable in pure skill with half a season. Sort by Skill+ or Value/100 for the two leaderboards.",
 hitters:"DESCRIPTIVE ONLY. Half a season of hitter challenges shows no detectable talent spread (reliability ~0), so no hitter is a reliable talent estimate yet. These are records of what happened, ranked by NetVal, not a skill ranking. Expect this to firm up over 2-3 seasons.",
 teams:"Team totals across all deciders, including pitcher-initiated challenges. Ranked by NetVal (descriptive).",
 backtest:"Season replay: value captured by actual challenge usage vs the matrix policy executed with league-average perception (no hindsight). Wins = leveraged runs times the league run-to-win factor. Negative LeftOnTable means the team already beats the league-perceiver benchmark.",
 film:"Every challenge and every counted miss, with the Savant clip. Margin = inches in the challenger's favor (negative = the call was right). Sorted by decision EV by default; search a player to pull their reel. Showing at most 500 rows - refine the search to narrow."};
const DEFSORT={catchers:"skill",hitters:"net",teams:"net",backtest:"gapW",film:"ev"};
let cur={t:"catchers", sort:"skill", dir:-1, q:""};
function fmt(v,k){
 if(v===null||v===undefined)return k==="skill"?"provisional":"";
 if(k==="skci"||k==="vci")return "&plusmn;"+v;
 if(k==="value"||k==="net"||k==="gapW")return v.toFixed(2);
 if(typeof v==="number"&&!Number.isInteger(v))return v.toFixed(2);
 return v;
}
function cls(v,k){
 if(("net"===k||"value"===k||"gapW"===k)&&typeof v==="number") return v>0.005?"pos":(v<-0.005?"neg":"");
 if(k==="skill"&&typeof v==="number") return v>100.5?"pos":(v<99.5?"neg":"");
 if(k==="skill"&&(v===null||v===undefined))return "dim";
 return ["player","team"].includes(k)?"l":( ["skci","vci","cons","succ","chal"].includes(k)?"dim":"");
}
function render(){
 const cols=COLS[cur.t]||COLS.player;
 let rows=D[cur.t].slice();
 if(cur.q){const q=cur.q.toLowerCase();
  rows=rows.filter(r=>(r.player||r.team).toLowerCase().includes(q)||String(r.team).toLowerCase().includes(q));}
 const total=rows.length;
 rows.sort((a,b)=>{
  const x=a[cur.sort],y=b[cur.sort];
  if(x===null||x===undefined)return 1; if(y===null||y===undefined)return -1;
  return (x<y?-1:x>y?1:0)*cur.dir*( typeof x==="string"?-1:1 );
 });
 if(cur.t==="film") rows=rows.slice(0,500);
 let h="<thead><tr>"+cols.map(c=>`<th class="${c[2]||""}" data-k="${c[0]}" aria-sort="${cur.sort===c[0]?(cur.dir<0?"descending":"ascending"):"none"}">${c[1]}</th>`).join("")+"</tr></thead><tbody>";
 for(const r of rows){
  h+="<tr>"+cols.map(c=>{
   const v=r[c[0]];
   if(c[0]==="playId")return `<td class="l"><a href="https://baseballsavant.mlb.com/sporty-videos?playId=${v}" target="_blank" rel="noopener" style="color:var(--accent-ink);font-weight:600;text-decoration:none">&#9654; watch</a></td>`;
   return `<td class="${cls(v,c[0])}">${fmt(v,c[0])}</td>`;}).join("")+"</tr>";
 }
 document.getElementById("tbl").innerHTML=h+"</tbody>";
 document.getElementById("count").textContent=(cur.t==="film"&&total>500?`${total} rows (showing 500)`:total+" rows");
 document.getElementById("footNote").textContent=NOTES[cur.t]+" Generated "+D.generated+".";
 document.querySelectorAll("#tbl th").forEach(th=>th.addEventListener("click",()=>{
  const k=th.dataset.k;
  if(cur.sort===k)cur.dir*=-1; else {cur.sort=k;cur.dir=-1;}
  render();
 }));
}
document.querySelectorAll("#tabs button").forEach(b=>b.addEventListener("click",()=>{
 document.querySelectorAll("#tabs button").forEach(x=>x.setAttribute("aria-pressed","false"));
 b.setAttribute("aria-pressed","true");
 cur.t=b.dataset.t; cur.sort=DEFSORT[cur.t]; cur.dir=-1;
 render();
}));
document.getElementById("q").addEventListener("input",e=>{cur.q=e.target.value;render();});
render();
</script>
"""

for ch, ent in {"—": "&mdash;", "·": "&middot;", "−": "&minus;", "≥": "&ge;",
                "–": "&ndash;", "×": "&times;"}.items():
    page = page.replace(ch, ent)
bad = sorted({c for c in page if ord(c) > 127})
assert not bad, f"non-ASCII left: {bad}"

page = (page
        .replace("__BITTER700__", font_b64("Bitter-700.ttf"))
        .replace("__PLEX400__", font_b64("IBMPlexSans-400.ttf"))
        .replace("__PLEX600__", font_b64("IBMPlexSans-600.ttf"))
        .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
with open(OUT, "w") as f:
    f.write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024}KB)")
