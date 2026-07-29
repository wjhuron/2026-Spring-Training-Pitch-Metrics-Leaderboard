"""Generate the ABS challenge matrix artifact page with embedded tables + fonts."""
import base64
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts", "abs_matrix.html")

with open(os.path.join(REPO, "data", "abs_value_tables_2026.json")) as f:
    VT = json.load(f)
with open(os.path.join(REPO, "data", "abs_option_model_2026.json")) as f:
    OM = json.load(f)
with open(os.path.join(REPO, "data", "abs_hitter_zones_2026.json")) as f:
    HZ = json.load(f)["zones"]
zones = sorted(({"n": z["name"], "t": z["szTop"], "b": z["szBot"]}
                for z in HZ.values()), key=lambda z: z["n"])


def font_b64(name):
    with open(os.path.join(REPO, "assets", "fonts", name), "rb") as f:
        return base64.b64encode(f.read()).decode()


fonts = {
    "bitter700": font_b64("Bitter-700.ttf"),
    "plex400": font_b64("IBMPlexSans-400.ttf"),
    "plex600": font_b64("IBMPlexSans-600.ttf"),
}

# attentive-look expected confidence vs TRUE margin, per side and zone edge
# ("fld|side", "fld|top", ... ) — same curves the player grading uses
posts = OM["pLook"]

data = {
    "countRV288": VT["countRV288"],
    "runDist": VT["runDist"],
    "W": VT["W"],
    "wExtraTie": VT["wExtraTie"],
    "G": VT["G"],
    "gAvg": VT["meta"]["gAvg"],
    "Cgrid": OM["Cgrid"],
    "post": posts,
    "zones": zones,
    "generated": OM["meta"]["generated"],
    "games": OM["meta"]["games"],
    "opps": OM["meta"]["opportunities"],
}

page = """<title>ABS Challenge Matrix — Huronalytics</title>
<style>
@font-face{font-family:'Bitter';font-weight:700;src:url(data:font/ttf;base64,__BITTER700__) format('truetype');font-display:swap}
@font-face{font-family:'IBM Plex Sans';font-weight:400;src:url(data:font/ttf;base64,__PLEX400__) format('truetype');font-display:swap}
@font-face{font-family:'IBM Plex Sans';font-weight:600;src:url(data:font/ttf;base64,__PLEX600__) format('truetype');font-display:swap}
:root{
  --paper:#F7F2E9;--card:#FDFAF4;--ink:#2B2320;--ink2:#71645C;--line:rgba(43,35,32,.16);
  --accent:#C4593A;--accent-ink:#8F3A22;--good:#5F7A54;--good-bg:#E7EDE2;
  --hold-bg:#F3E3DC;--sel:#2B2320;
  --heat-lo:#F6EEDF;--heat-txt-flip:.52;
}
@media (prefers-color-scheme: dark){:root{
  --paper:#211B16;--card:#2A231D;--ink:#F0E7DA;--ink2:#A99A8E;--line:rgba(240,231,218,.16);
  --accent:#E0714A;--accent-ink:#EE9070;--good:#93AC85;--good-bg:#33402D;
  --hold-bg:#463028;--sel:#F0E7DA;--heat-lo:#332B23;
}}
:root[data-theme="light"]{
  --paper:#F7F2E9;--card:#FDFAF4;--ink:#2B2320;--ink2:#71645C;--line:rgba(43,35,32,.16);
  --accent:#C4593A;--accent-ink:#8F3A22;--good:#5F7A54;--good-bg:#E7EDE2;
  --hold-bg:#F3E3DC;--sel:#2B2320;--heat-lo:#F6EEDF;
}
:root[data-theme="dark"]{
  --paper:#211B16;--card:#2A231D;--ink:#F0E7DA;--ink2:#A99A8E;--line:rgba(240,231,218,.16);
  --accent:#E0714A;--accent-ink:#EE9070;--good:#93AC85;--good-bg:#33402D;
  --hold-bg:#463028;--sel:#F0E7DA;--heat-lo:#332B23;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:16px/1.55 'IBM Plex Sans',system-ui,sans-serif;margin:0;padding:32px 20px 64px}
.wrap{max-width:980px;margin:0 auto}
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:clamp(30px,5vw,44px);line-height:1.08;margin:0 0 14px;text-wrap:balance}
.sub{color:var(--ink2);max-width:62ch;margin:0 0 6px}
.sub b{color:var(--ink);font-weight:600}
.rule{border:0;border-top:1px solid var(--line);margin:26px 0}
.controls{display:flex;flex-wrap:wrap;gap:18px 26px;align-items:flex-end;margin-bottom:20px}
.ctl label{display:block;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--ink2);margin-bottom:6px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.seg button{border:0;background:transparent;color:var(--ink2);font:600 14px 'IBM Plex Sans',sans-serif;padding:7px 14px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.seg button:focus-visible,.chip:focus-visible,.cell:focus-visible,.step button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chips{display:inline-flex;gap:6px}
.chip{border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink2);font:600 14px 'IBM Plex Sans',sans-serif;padding:7px 12px;cursor:pointer}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#FDFAF4}
.step{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.step button{border:0;background:transparent;color:var(--ink);font:600 16px/1 'IBM Plex Sans';padding:7px 12px;cursor:pointer}
.step .val{min-width:118px;text-align:center;font-weight:600;font-size:14px}
.matrix-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);padding:14px}
table{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums;width:100%}
th{font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--ink2);padding:4px 6px}
th.rowh{text-align:right;font-size:13px;color:var(--ink);font-weight:600;padding-right:10px;white-space:nowrap}
td{padding:0}
.cell{display:block;width:100%;border:0;border-radius:5px;font:600 12.5px/1 'IBM Plex Sans',sans-serif;padding:9px 2px;text-align:center;cursor:pointer;min-width:44px}
.cell.sel{box-shadow:0 0 0 2px var(--sel)}
.legend{display:flex;align-items:center;gap:12px;margin-top:12px;font-size:12.5px;color:var(--ink2)}
.grad{height:10px;border-radius:5px;flex:1;max-width:340px}
h2{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:24px;margin:0 0 4px}
.panel{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:22px;margin-top:8px}
.stateline{color:var(--ink2);margin:0 0 18px;font-size:15px}
.stateline b{color:var(--ink)}
.facts{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}
.fact{border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:118px}
.fact .k{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}
.fact .v{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:22px;margin-top:2px;font-variant-numeric:tabular-nums}
.fact .v small{font-family:'IBM Plex Sans';font-size:12px;color:var(--ink2);font-weight:400}
.slider-row{display:flex;flex-wrap:wrap;align-items:center;gap:16px;margin-bottom:14px}
input[type=range]{accent-color:var(--accent);width:min(320px,100%)}
.verdict{display:flex;align-items:center;gap:14px;border-radius:10px;padding:14px 18px}
.verdict.go{background:var(--good-bg)}
.verdict.hold{background:var(--hold-bg)}
.verdict .word{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:24px;letter-spacing:.02em}
.verdict.go .word{color:var(--good)}
.verdict.hold .word{color:var(--accent-ink)}
.verdict .why{font-size:14px;color:var(--ink2)}
.foot{color:var(--ink2);font-size:13px;max-width:72ch}
@media (prefers-reduced-motion: no-preference){.cell{transition:box-shadow .12s}}
</style>
<div class="wrap">
<p class="eyebrow">Huronalytics · 2026 ABS Challenge System</p>
<h1>The Challenge Matrix</h1>
<p class="sub">Each cell is the <b>break-even confidence</b> — how sure you must be that the call was wrong
before challenging is worth it. It weighs what the flipped call is worth in this exact spot against the
option value of the challenge you might lose. Built from __GAMES__ games and __OPPS__ near-zone calls.</p>
<hr class="rule">
<div class="controls" role="group" aria-label="Game state">
  <div class="ctl"><label>Challenges left</label>
    <span class="seg" id="segK"><button data-v="2" aria-pressed="true">2</button><button data-v="1" aria-pressed="false">1</button></span></div>
  <div class="ctl"><label>Half</label>
    <span class="seg" id="segHalf"><button data-v="top" aria-pressed="true">Top</button><button data-v="bottom" aria-pressed="false">Bottom</button></span></div>
  <div class="ctl"><label>Outs</label>
    <span class="seg" id="segOuts"><button data-v="0" aria-pressed="true">0</button><button data-v="1" aria-pressed="false">1</button><button data-v="2" aria-pressed="false">2</button></span></div>
  <div class="ctl"><label>Runners on</label>
    <span class="chips" id="chipsBases"><button class="chip" data-b="0" aria-pressed="false">1B</button><button class="chip" data-b="1" aria-pressed="false">2B</button><button class="chip" data-b="2" aria-pressed="false">3B</button></span></div>
  <div class="ctl"><label>Score</label>
    <span class="step"><button id="scDown" aria-label="Batting team one run worse">−</button><span class="val" id="scoreVal">Tie game</span><button id="scUp" aria-label="Batting team one run better">+</button></span></div>
  <div class="ctl"><label>Deciding side</label>
    <span class="seg" id="segSide"><button data-v="fld" aria-pressed="true">Catcher</button><button data-v="bat" aria-pressed="false">Hitter</button></span></div>
</div>
<div class="matrix-scroll"><table id="mx" aria-label="Break-even challenge confidence by count and inning"></table></div>
<div class="legend"><span>Challenge freely</span><div class="grad" id="grad"></div><span>Hold your challenge</span><span style="margin-left:auto">cells are break-even confidence %</span></div>
<hr class="rule">
<h2>Price a specific pitch</h2>
<p class="sub" style="margin-bottom:14px">Click any cell above to load that situation, then set where the ball actually was.</p>
<div class="panel">
  <p class="stateline" id="stateline"></p>
  <div class="facts">
    <div class="fact"><div class="k">Flip is worth</div><div class="v" id="fGain">&ndash;<small> lev. runs</small></div></div>
    <div class="fact"><div class="k">Challenge value</div><div class="v" id="fCost">&ndash;<small> lev. runs</small></div></div>
    <div class="fact"><div class="k">Break-even</div><div class="v" id="fPstar">&ndash;</div></div>
    <div class="fact"><div class="k">Identifiable as</div><div class="v" id="fConf">&ndash;</div></div>
  </div>
  <div class="slider-row" style="align-items:flex-start">
    <div class="ctl"><label>Hitter (sets the exact ABS zone)</label>
      <input list="hitterList" id="hitterPick" placeholder="League average zone"
             style="border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:14px 'IBM Plex Sans';padding:8px 12px;width:220px">
      <datalist id="hitterList"></datalist>
      <div style="font-size:12.5px;color:var(--ink2);margin-top:6px" id="zoneInfo"></div>
      <label style="margin-top:14px">Exact location (Gameday plate_x / plate_z, ft)</label>
      <div style="display:flex;gap:6px">
        <input id="exX" type="number" step="0.01" placeholder="plate_x"
               style="border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:14px 'IBM Plex Sans';padding:8px 10px;width:105px">
        <input id="exZ" type="number" step="0.01" placeholder="plate_z"
               style="border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:14px 'IBM Plex Sans';padding:8px 10px;width:105px">
      </div>
    </div>
    <div class="ctl"><label>Or click where the ball crossed &mdash; <span id="lookVal" style="color:var(--ink);text-transform:none;letter-spacing:0"></span></label>
      <svg id="zonePlot" viewBox="0 0 224 288" width="224" height="288" role="img"
           aria-label="Strike zone plot; click to place the pitch"
           style="cursor:crosshair;border:1px solid var(--line);border-radius:10px;background:var(--paper);touch-action:manipulation">
        <rect id="zEdge" fill="none" stroke="var(--accent)" stroke-dasharray="4 4" stroke-width="1"></rect>
        <rect id="zRect" fill="none" stroke="var(--ink)" stroke-width="1.5"></rect>
        <circle id="zBall" fill="none" stroke="var(--accent-ink)" stroke-width="2"></circle>
      </svg></div>
  </div>
  <div class="verdict" id="verdict"><span class="word">&ndash;</span><span class="why"></span></div>
</div>
<hr class="rule">
<p class="foot">Method: flip value = base-out-conditioned count-transition run values × a win-leverage factor
from a 2026 win-probability model; ball-four / strike-three flips are priced as exact win-probability
transitions (a walk-off walk is worth exactly the win it delivers, no more). Challenge value C(k, T, score)
comes from a dynamic program over the season's challenge-opportunity stream, conditioned on the deciding
team's score margin — challenges are near-free in blowouts and most precious in late tie games. Successful
challenges are retained, so the only cost of challenging is the chance of losing the option. "Identifiable as" is the
confidence an attentive decider ends up with, on average, when the ball is truly at that spot — via each
role's fitted perception noise. Catchers judge the zone better
(σ 2.2 in) than hitters (σ 3.2 in), which is why some early-game thresholds are unreachable for hitters.
Generated __GEN__ · huronalytics</p>
</div>
<script>
const D = __DATA__;
const COUNTS = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,0],[3,1],[3,2]];
const INNINGS = [1,2,3,4,5,6,7,8,9,10];
const state = {k:2, half:"top", outs:0, bases:[false,false,false], lead:0,
               sel:{b:1,s:1,inning:7}, side:"fld", px:8.95, pz:29.4};
// zone geometry, inches; plot is 8 px per inch. Zone defaults to league
// average; picking a hitter loads their exact ABS zone from the feed data.
const ZHW=8.5, THR=1.4495, PXIN=8, PLOT_Z0=12;
state.ztop=39.6; state.zbot=19.2;
function pitchGeom(){
  const ax=Math.abs(state.px);
  const dx=ZHW-ax, dt=state.ztop-state.pz, db=state.pz-state.zbot;
  const dzOut=Math.max(state.pz-state.ztop, state.zbot-state.pz);
  const inside = dx>=0 && dzOut<=0;
  const d = inside ? -Math.min(dx,dt,db)
                   : Math.hypot(Math.max(-dx,0), Math.max(dzOut,0));
  const mEdge=Math.min(dx,dt,db);
  const reg = mEdge===dx ? "side" : (mEdge===dt ? "top" : "bottom");
  return {loc: THR-d, reg};
}
const clamp=(d)=>Math.max(-12,Math.min(12,d));
function contBat(inning, half, d){
  d=clamp(d);
  if(half==="top"){
    const home = inning>=9 ? (d>0 ? 1 : D.W[Math.min(inning,10)+"|bottom|"+d])
                           : D.W[inning+"|bottom|"+d];
    return 1-home;
  }
  if(inning>=9) return d>0 ? 1 : (d<0 ? 0 : D.wExtraTie);
  return D.W[(inning+1)+"|top|"+d];
}
function wpMid(inning, half, diff, bases, outs){
  if(outs>=3) return contBat(inning, half, diff);
  const dist=D.runDist[bases+"|"+outs];
  let tot=0;
  for(const k in dist){
    const r=+k, d2 = half==="top" ? diff-r : diff+r;
    tot += dist[k]*contBat(inning, half, d2);
  }
  return tot;
}
function walkBases(bases){
  const f=bases[0]==="1", s2=bases[1]==="1", t=bases[2]==="1";
  const runs=(f&&s2&&t)?1:0;
  const nt=(f&&s2)?true:t, ns=f?true:s2;
  return ["1"+(ns?"1":"0")+(nt?"1":"0"), runs];
}
function gainFor(b,s,inning){
  // continuation branches: base-out-conditioned count values x run leverage;
  // terminal branches (ball four / strike three): exact win-prob transitions
  const bases=state.bases.map(x=>x?"1":"0").join("");
  const outs=state.outs;
  const diff = state.half==="top" ? -state.lead : state.lead;
  const g = D.G[Math.min(inning,10)+"|"+state.half+"|"+clamp(diff)];
  let wpBase=null, ballWP, strikeWP;
  if(b===3){
    wpBase=wpMid(inning,state.half,diff,bases,outs);
    const [nb,runs]=walkBases(bases);
    const d2 = state.half==="top" ? diff-runs : diff+runs;
    ballWP = wpMid(inning,state.half,d2,nb,outs)-wpBase;
  }else{
    ballWP = D.countRV288[(b+1)+"-"+s+"|"+bases+"|"+outs]*g;
  }
  if(s===2){
    if(wpBase===null) wpBase=wpMid(inning,state.half,diff,bases,outs);
    strikeWP = wpMid(inning,state.half,diff,bases,outs+1)-wpBase;
  }else{
    strikeWP = D.countRV288[b+"-"+(s+1)+"|"+bases+"|"+outs]*g;
  }
  return (ballWP-strikeWP)/D.gAvg;
}
const Tfor=(inning)=> 2*(9-Math.min(inning,9)) + (state.half==="top"?2:1);
function costFor(inning){
  // deciding team's score margin: hitter's team is batting, catcher's fielding
  const dTeam = state.side==="bat" ? state.lead : -state.lead;
  return D.Cgrid[state.k+"|"+Tfor(inning)+"|"+clamp(dTeam)];
}
function pstarFor(b,s,inning){
  const g=Math.max(gainFor(b,s,inning),0.0001);
  const c=costFor(inning);
  return c/(g+c);
}
function isDark(){const a=document.documentElement.getAttribute("data-theme");
  if(a==="dark")return true; if(a==="light")return false;
  return matchMedia("(prefers-color-scheme: dark)").matches;}
const RAMP_L=[[246,238,223],[240,206,168],[221,143,96],[166,59,34],[92,26,13]];
const RAMP_D=[[51,43,35],[92,52,36],[151,74,45],[204,102,66],[238,144,112]];
function heat(p){ // p in 0..0.8+
  const t=Math.max(0,Math.min(1,p/0.8)), R=isDark()?RAMP_D:RAMP_L;
  const x=t*(R.length-1), i=Math.min(Math.floor(x),R.length-2), f=x-i;
  const c=R[i].map((v,j)=>Math.round(v+(R[i+1][j]-v)*f));
  return {bg:`rgb(${c.join(",")})`, dark:(0.299*c[0]+0.587*c[1]+0.114*c[2])<140};
}
function renderMatrix(){
  const mx=document.getElementById("mx");
  let h="<thead><tr><th></th>"+INNINGS.map(i=>`<th scope="col">${i===10?"10+":i}</th>`).join("")+"</tr></thead><tbody>";
  for(const [b,s] of COUNTS){
    h+=`<tr><th scope="row" class="rowh">${b}\\u2013${s}</th>`;
    for(const inn of INNINGS){
      const p=pstarFor(b,s,inn), col=heat(p);
      const sel=(state.sel.b===b&&state.sel.s===s&&state.sel.inning===inn)?" sel":"";
      const txt = isDark() ? (col.dark?"var(--ink)":"#211B16") : (col.dark?"#FDFAF4":"var(--ink)");
      h+=`<td><button class="cell${sel}" style="background:${col.bg};color:${txt}" data-b="${b}" data-s="${s}" data-i="${inn}" aria-label="Count ${b}-${s}, inning ${inn}: break-even ${Math.round(p*100)} percent">${Math.round(p*100)}</button></td>`;
    }
    h+="</tr>";
  }
  mx.innerHTML=h+"</tbody>";
  mx.querySelectorAll(".cell").forEach(el=>el.addEventListener("click",()=>{
    state.sel={b:+el.dataset.b,s:+el.dataset.s,inning:+el.dataset.i};
    renderMatrix(); renderPanel();
    document.getElementById("stateline").scrollIntoView({behavior:"smooth",block:"center"});
  }));
  const stops=[]; for(let i=0;i<=10;i++){stops.push(heat(0.8*i/10).bg+" "+(i*10)+"%");}
  document.getElementById("grad").style.background="linear-gradient(90deg,"+stops.join(",")+")";
}
function posterior(key,x){
  const g=D.post[key];
  if(x<=g[0][0])return g[0][1];
  if(x>=g[g.length-1][0])return g[g.length-1][1];
  const step=g[1][0]-g[0][0], i=Math.floor((x-g[0][0])/step);
  const [x0,p0]=g[i],[x1,p1]=g[Math.min(i+1,g.length-1)];
  return p0+(p1-p0)*(x-x0)/Math.max(x1-x0,1e-9);
}
function neededInches(key,pstar){
  const g=D.post[key];
  if(g[g.length-1][1]<pstar)return null;
  for(const [x,p] of g){if(p>=pstar)return Math.max(x,0);}
  return null;
}
function baseLabel(){
  const on=["1B","2B","3B"].filter((_,i)=>state.bases[i]);
  return on.length? on.join(" & ")+" occupied" : "bases empty";
}
function locText(loc){
  if(loc>0)return loc.toFixed(2).replace(/\\.?0+$/,"")+" in INTO the zone";
  if(loc<0)return (-loc).toFixed(2).replace(/\\.?0+$/,"")+" in OFF the zone";
  return "ball grazing the zone edge";
}
function renderPanel(){
  const {b,s,inning}=state.sel;
  const g=gainFor(b,s,inning), c=costFor(inning), p=c/(Math.max(g,0.0001)+c);
  // margin in the challenger's favor: catcher wants the ball IN the zone,
  // hitter wants it OFF. loc > 0 = ball edge inside the zone boundary.
  const {loc, reg}=pitchGeom();
  const m = state.side==="fld" ? loc : -loc;
  const conf=posterior(state.side+"|"+reg, m);
  const wouldWin = state.side==="fld" ? loc>=0 : loc<0;
  const ruling = loc>=0 ? "strike" : "ball";
  const lead=state.lead===0?"tie game":(state.lead>0?`batting team up ${state.lead}`:`batting team down ${-state.lead}`);
  document.getElementById("stateline").innerHTML=
    `<b>${b}\\u2013${s}</b> count \\u00B7 <b>${inning===10?"extras":ordinal(inning)} inning, ${state.half}</b> \\u00B7 ${state.outs} out \\u00B7 ${baseLabel()} \\u00B7 ${lead} \\u00B7 <b>${state.k}</b> challenge${state.k>1?"s":""} left`;
  document.getElementById("fGain").innerHTML=g.toFixed(2)+"<small> lev. runs</small>";
  document.getElementById("fCost").innerHTML=c.toFixed(2)+"<small> lev. runs</small>";
  document.getElementById("fPstar").textContent=Math.round(p*100)+"%";
  document.getElementById("fConf").textContent=Math.round(conf*100)+"%";
  document.getElementById("lookVal").textContent=locText(loc)+" \\u00B7 "+reg+" edge";
  const ball=document.getElementById("zBall");
  ball.setAttribute("cx",112+state.px*PXIN);
  ball.setAttribute("cy",288-(state.pz-PLOT_Z0)*PXIN);
  ball.setAttribute("r",THR*PXIN);
  const v=document.getElementById("verdict"), need=neededInches(state.side+"|"+reg,p);
  const roleName=state.side==="fld"?"catcher":"hitter";
  const needTxt = need===null ? null :
    (need<0.05 ? (state.side==="fld" ? "that so much as grazes the zone" : "that so much as slips off the edge")
               : (state.side==="fld" ? `${need.toFixed(1)} in or more into the zone` : `${need.toFixed(1)} in or more off the zone`));
  const truth = wouldWin ? `ABS would overturn this (true ${ruling})` : `ABS would uphold the call (true ${ruling})`;
  if(conf>=p){
    v.className="verdict go";
    v.innerHTML=`<span class="word">Challenge</span><span class="why">${truth}, and it's identifiable: ${Math.round(conf*100)}% clears the ${Math.round(p*100)}% bar. Rule of thumb here: challenge any pitch ${needTxt}.</span>`;
  }else if(need===null){
    v.className="verdict hold";
    v.innerHTML=`<span class="word">Hold</span><span class="why">${truth} \\u2014 but no ${roleName} can reach ${Math.round(p*100)}% certainty by eye, so this spot is never worth a challenge in the moment. Save it.</span>`;
  }else{
    v.className="verdict hold";
    v.innerHTML=`<span class="word">Hold</span><span class="why">${truth} \\u2014 in the moment a ${roleName} could only be ${Math.round(conf*100)}% sure, under the ${Math.round(p*100)}% bar. It becomes a good challenge for a pitch ${needTxt}.</span>`;
  }
}
const ordinal=n=>n+(["th","st","nd","rd"][(n%100>10&&n%100<14)?0:Math.min(n%10,4)]||"th");
function seg(id,key,parse){
  document.querySelectorAll("#"+id+" button").forEach(btn=>btn.addEventListener("click",()=>{
    document.querySelectorAll("#"+id+" button").forEach(x=>x.setAttribute("aria-pressed","false"));
    btn.setAttribute("aria-pressed","true"); state[key]=parse?parse(btn.dataset.v):btn.dataset.v;
    renderMatrix(); renderPanel();
  }));
}
seg("segK","k",v=>+v); seg("segHalf","half"); seg("segOuts","outs",v=>+v); seg("segSide","side");
document.querySelectorAll("#chipsBases .chip").forEach(ch=>ch.addEventListener("click",()=>{
  const i=+ch.dataset.b; state.bases[i]=!state.bases[i];
  ch.setAttribute("aria-pressed",String(state.bases[i]));
  renderMatrix(); renderPanel();
}));
function scoreText(){return state.lead===0?"Tie game":(state.lead>0?`Batting up ${state.lead}`:`Batting down ${-state.lead}`);}
document.getElementById("scUp").addEventListener("click",()=>{state.lead=Math.min(6,state.lead+1);document.getElementById("scoreVal").textContent=scoreText();renderMatrix();renderPanel();});
document.getElementById("scDown").addEventListener("click",()=>{state.lead=Math.max(-6,state.lead-1);document.getElementById("scoreVal").textContent=scoreText();renderMatrix();renderPanel();});
function drawZone(){
  const zr=document.getElementById("zRect"), ze=document.getElementById("zEdge");
  const x0=112-ZHW*PXIN, y0=288-(state.ztop-PLOT_Z0)*PXIN;
  zr.setAttribute("x",x0); zr.setAttribute("y",y0);
  zr.setAttribute("width",2*ZHW*PXIN); zr.setAttribute("height",(state.ztop-state.zbot)*PXIN);
  const off=THR*PXIN;
  ze.setAttribute("x",x0-off); ze.setAttribute("y",y0-off);
  ze.setAttribute("width",2*ZHW*PXIN+2*off); ze.setAttribute("height",(state.ztop-state.zbot)*PXIN+2*off);
}
(function initZone(){
  drawZone();
  const svg=document.getElementById("zonePlot");
  svg.addEventListener("click",e=>{
    const r=svg.getBoundingClientRect();
    const sx=(e.clientX-r.left)*224/r.width, sy=(e.clientY-r.top)*288/r.height;
    state.px=(sx-112)/PXIN;
    state.pz=(288-sy)/PXIN+PLOT_Z0;
    document.getElementById("exX").value=(state.px/12).toFixed(2);
    document.getElementById("exZ").value=(state.pz/12).toFixed(2);
    renderPanel();
  });
  const dl=document.getElementById("hitterList");
  dl.innerHTML=D.zones.map(z=>`<option value="${z.n.replace(/"/g,"&quot;")}"></option>`).join("");
  const info=document.getElementById("zoneInfo");
  info.textContent="zone 3.30 / 1.60 ft (league average)";
  document.getElementById("hitterPick").addEventListener("change",e=>{
    const z=D.zones.find(x=>x.n===e.target.value);
    if(z){ state.ztop=z.t*12; state.zbot=z.b*12;
      info.textContent=`zone ${z.t.toFixed(2)} / ${z.b.toFixed(2)} ft (${z.n})`; }
    else { state.ztop=39.6; state.zbot=19.2;
      info.textContent="zone 3.30 / 1.60 ft (league average)"; }
    drawZone(); renderPanel();
  });
  function exact(){
    const x=parseFloat(document.getElementById("exX").value);
    const z=parseFloat(document.getElementById("exZ").value);
    if(!isNaN(x)) state.px=x*12;
    if(!isNaN(z)) state.pz=z*12;
    renderPanel();
  }
  document.getElementById("exX").addEventListener("input",exact);
  document.getElementById("exZ").addEventListener("input",exact);
})();
new MutationObserver(()=>{renderMatrix();renderPanel();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{renderMatrix();renderPanel();});
renderMatrix(); renderPanel();
</script>
"""

# entity-encode remaining non-ASCII (all in HTML zones; JS uses \\u escapes)
for ch, ent in {"—": "&mdash;", "·": "&middot;", "−": "&minus;",
                "σ": "&sigma;", "≥": "&ge;", "–": "&ndash;",
                "×": "&times;"}.items():
    page = page.replace(ch, ent)
bad = sorted({c for c in page if ord(c) > 127})
assert not bad, f"non-ASCII left in page: {bad}"

page = (page
        .replace("__BITTER700__", fonts["bitter700"])
        .replace("__PLEX400__", fonts["plex400"])
        .replace("__PLEX600__", fonts["plex600"])
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__GAMES__", f"{data['games']:,}")
        .replace("__OPPS__", f"{data['opps']:,}")
        .replace("__GEN__", data["generated"]))
with open(OUT, "w") as f:
    f.write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024}KB)")
