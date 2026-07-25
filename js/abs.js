/* ABS challenge section: self-contained. Renders three sub-views (Leaderboards,
   Matrix, Film Room) into #abs-page from committed data/abs_*.json at runtime.
   Isolated from the pitch-data leaderboard engine. Exposed as window.ABS. */
window.ABS = (function () {
  let styled = false, core = null, events = null, zoneMiss = null;
  // Film Room filters live at module scope so they survive view switches and
  // can be driven by leaderboard click-through and by the URL.
  const film = { date: null, bat: '', cat: '', team: '', type: '', wrong: 'wrong', count: '', inning: '' };
  let filmSort = 'value', filmDir = -1;
  let pendingPitch = null;   // a Film Room row waiting to be loaded into the matrix
  const state = {
    view: 'leaders', tab: 'catchers', sort: 'skill', dir: -1, q: '',
    // matrix state
    k: 2, half: 'top', outs: 0, bases: [false, false, false], away: 0, home: 0,
    sel: { b: 1, s: 1, inning: 7 }, side: 'fld', px: 8.95, pz: 29.4,
    ztop: 39.6, zbot: 19.2
  };
  const ZHW = 8.5, THR = 1.4495, PXIN = 8, PLOT_Z0 = 12;
  // starting situation for the matrix tool; Reset restores exactly this
  const MX_DEFAULTS = {
    k: 2, half: 'top', outs: 0, bases: [false, false, false], away: 0, home: 0,
    sel: { b: 1, s: 1, inning: 7 }, side: 'fld', px: 8.95, pz: 29.4,
    ztop: 39.6, zbot: 19.2
  };

  function injectStyles() {
    if (styled) return; styled = true;
    const css = `
#abs-page{max-width:1080px;margin:0 auto;padding:8px 4px 60px;color:var(--text-primary);font-family:'IBM Plex Sans',system-ui,sans-serif}
#abs-page h2{font-family:'Bitter',Georgia,serif;font-size:22px;margin:0 0 4px}
#abs-page .abs-sub{color:var(--text-muted);max-width:78ch;margin:0 0 16px;font-size:14px;line-height:1.5}
#abs-page .abs-sub b{color:var(--text-primary)}
#abs-page .abs-vtabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
#abs-page .abs-vtabs button{border:1px solid var(--border);border-radius:8px;background:var(--bg-card);color:var(--text-secondary);font:600 13px 'IBM Plex Sans';padding:7px 14px;cursor:pointer}
#abs-page .abs-vtabs button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#f7f2e9}
#abs-page .abs-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
#abs-page input[type=search],#abs-page .abs-inp{border:1px solid var(--border);border-radius:8px;background:var(--bg-card);color:var(--text-primary);font:14px 'IBM Plex Sans';padding:7px 11px}
#abs-page .abs-count{color:var(--text-muted);font-size:12px;margin-left:auto}
#abs-page .abs-tblwrap{overflow:auto;max-height:72vh;border:1px solid var(--border);border-radius:10px;background:var(--bg-card);-webkit-overflow-scrolling:touch}
#abs-page table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;min-width:760px}
#abs-page th{position:sticky;top:0;background:var(--bg-th);font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--text-th);text-align:right;padding:9px 10px;border-bottom:1px solid var(--border);cursor:pointer;white-space:nowrap}
#abs-page th.l{text-align:left}
#abs-page th[aria-sort]:not([aria-sort="none"]){color:var(--accent)}
#abs-page td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border-light);font-size:13px;white-space:nowrap}
#abs-page td.l{text-align:left;font-weight:600}
#abs-page td.dim{color:var(--text-muted)}
#abs-page tr:last-child td{border-bottom:0}
#abs-page .pos{color:#3f6b34;font-weight:600}
#abs-page .neg{color:var(--accent);font-weight:600}
#abs-page a.abs-vid{color:var(--accent);font-weight:600;text-decoration:none}
#abs-page .abs-foot{color:var(--text-muted);font-size:12.5px;max-width:80ch;margin-top:14px;line-height:1.5}
#abs-page .mx-ctrls{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-end;margin-bottom:16px}
#abs-page .mx-ctl label{display:block;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:5px}
#abs-page .seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--bg-card)}
#abs-page .seg button{border:0;background:transparent;color:var(--text-secondary);font:600 13px 'IBM Plex Sans';padding:6px 12px;cursor:pointer}
#abs-page .seg button[aria-pressed="true"]{background:var(--text-primary);color:var(--bg-primary)}
#abs-page .chip{border:1px solid var(--border);border-radius:8px;background:var(--bg-card);color:var(--text-secondary);font:600 13px 'IBM Plex Sans';padding:6px 11px;cursor:pointer}
#abs-page .chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#f7f2e9}
#abs-page .step{display:inline-flex;align-items:center;border:1px solid var(--border);border-radius:8px;background:var(--bg-card)}
#abs-page .step button{border:0;background:transparent;color:var(--text-primary);font:600 15px 'IBM Plex Sans';padding:6px 11px;cursor:pointer}
#abs-page .step .val{min-width:110px;text-align:center;font-weight:600;font-size:13px}
#abs-page .mx-scroll{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:var(--bg-card);padding:12px}
#abs-page .mx-grid{border-collapse:separate;border-spacing:2px;min-width:auto}
#abs-page .mx-grid th{position:static;background:transparent;border:0;cursor:default;color:var(--text-muted);padding:3px 6px}
#abs-page .mx-grid th.rowh{text-align:right;color:var(--text-primary);font-size:12px;font-weight:600;padding-right:9px}
#abs-page .cell{display:block;width:100%;border:0;border-radius:5px;font:600 12px 'IBM Plex Sans';padding:8px 2px;text-align:center;cursor:pointer;min-width:42px}
#abs-page .cell.sel{box-shadow:0 0 0 2px var(--text-primary)}
#abs-page .mx-legend{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:12px;color:var(--text-muted)}
#abs-page .mx-grad{height:9px;border-radius:5px;flex:1;max-width:320px}
#abs-page .panel{border:1px solid var(--border);border-radius:10px;background:var(--bg-card);padding:18px;margin-top:8px}
#abs-page .facts{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0}
#abs-page .fact{border:1px solid var(--border);border-radius:9px;padding:9px 13px;min-width:110px}
#abs-page .fact .k{font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted)}
#abs-page .fact .v{font-family:'Bitter',serif;font-weight:700;font-size:20px;margin-top:2px}
#abs-page .verdict{display:flex;align-items:center;gap:12px;border-radius:9px;padding:12px 16px;margin-top:6px}
#abs-page .verdict.go{background:rgba(63,107,52,.14)}
#abs-page .verdict.hold{background:var(--accent-light)}
#abs-page .verdict .word{font-family:'Bitter',serif;font-weight:700;font-size:21px}
#abs-page .verdict.go .word{color:#3f6b34}
#abs-page .verdict.hold .word{color:var(--accent)}
#abs-page .verdict .why{font-size:13px;color:var(--text-secondary)}
#abs-page .zplot{border:1px solid var(--border);border-radius:9px;background:var(--bg-primary);cursor:crosshair;touch-action:manipulation}
#abs-page th[title]{text-decoration:underline dotted rgba(120,110,95,.5);text-underline-offset:3px}
#abs-page .lead-link{color:inherit;text-decoration:underline dotted rgba(120,110,95,.6);text-underline-offset:2px;cursor:pointer}
#abs-page .lead-link:hover{color:var(--accent)}
#abs-page tr.row-click{cursor:pointer}
#abs-page tr.row-click:hover td{background:var(--row-hover)}
#abs-page .empty{padding:26px 18px;text-align:center;color:var(--text-muted);font-size:14px}
#abs-page .empty b{color:var(--text-primary)}
#abs-page details.howto{border:1px solid var(--border);border-radius:10px;background:var(--bg-card);padding:10px 14px;margin-bottom:14px}
#abs-page details.howto summary{cursor:pointer;font-weight:600;font-size:13px;color:var(--accent);list-style:none}
#abs-page details.howto summary::-webkit-details-marker{display:none}
#abs-page details.howto summary::before{content:"\\25B8 ";display:inline-block;transition:transform .15s}
#abs-page details.howto[open] summary::before{content:"\\25BE "}
#abs-page details.howto dl{margin:10px 0 2px;font-size:13px;line-height:1.55;color:var(--text-secondary)}
#abs-page details.howto dt{font-weight:600;color:var(--text-primary);margin-top:7px}
#abs-page details.howto dd{margin:0 0 2px}
#abs-page .strip{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
#abs-page .strip .s{border:1px solid var(--border);border-radius:9px;background:var(--bg-card);padding:8px 13px;min-width:104px}
#abs-page .strip .s .k{font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted)}
#abs-page .strip .s .v{font-family:'Bitter',serif;font-weight:700;font-size:19px;margin-top:1px}
#abs-page .heatwrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:var(--bg-card);padding:14px}
#abs-page .heatlegend{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:12px;color:var(--text-muted)}
@media (max-width:720px){
  #abs-page{padding:6px 2px 48px}
  #abs-page h2{font-size:19px}
  #abs-page .abs-sub{font-size:13px}
  #abs-page .abs-bar{gap:6px}
  #abs-page .abs-bar .abs-inp,#abs-page .abs-bar input[type=search]{flex:1 1 128px;min-width:0;font-size:16px}
  #abs-page .abs-count{width:100%;margin-left:0}
  #abs-page .mx-ctrls{gap:10px 14px}
  #abs-page .seg button,#abs-page .chip{padding:8px 11px}
  #abs-page .cell{min-width:34px;font-size:11px;padding:7px 1px}
  #abs-page .mx-grid th.rowh{font-size:11px;padding-right:5px}
  #abs-page .strip .s{flex:1 1 44%;min-width:0}
  #abs-page .abs-tblwrap{max-height:64vh}
  #abs-page td,#abs-page th{padding:6px 7px}
  #abs-page .zplot{width:100%;height:auto;max-width:224px}
}
`;
    const s = document.createElement('style'); s.id = 'abs-styles'; s.textContent = css;
    document.head.appendChild(s);
  }

  async function ensureCore() {
    if (core) return;
    const [gr, bt, om, vt, hz] = await Promise.all([
      fetch('data/abs_player_grades_2026.json').then(r => r.json()),
      fetch('data/abs_backtest_2026.json').then(r => r.json()),
      fetch('data/abs_option_model_2026.json').then(r => r.json()),
      fetch('data/abs_value_tables_2026.json').then(r => r.json()),
      fetch('data/abs_hitter_zones_2026.json').then(r => r.json())
    ]);
    const zones = Object.values(hz.zones).map(z => ({ n: z.name, t: z.szTop, b: z.szBot }))
      .sort((a, b) => a.n < b.n ? -1 : 1);
    core = {
      grades: gr, backtest: bt.teams,
      countRV288: vt.countRV288, runDist: vt.runDist, W: vt.W, wExtraTie: vt.wExtraTie,
      G: vt.G, gAvg: vt.meta.gAvg, Cgrid: om.Cgrid, post: om.pLook, zones: zones
    };
  }

  // ---------- shared ----------
  const el = () => document.getElementById('abs-page');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // One-line definitions for every column, mirroring the `desc:` convention the
  // main leaderboard uses. Rendered as native title tooltips.
  const COL_DESC = {
    player: 'Player name. Click to pull their reel in the Film Room.',
    team: 'Team the player was with for most of these decisions.',
    skill: 'Skill+ : leverage-blind decision quality, indexed so 100 = league average and 15 points = one talent standard deviation. Independent of the stakes faced, and balanced so that simply challenging rarely cannot inflate it. Catchers only; no hitter sample is large enough yet.',
    skci: '95% confidence interval on Skill+. If it spans 100, this player is not distinguishable from average yet.',
    value: 'Leveraged runs added per 100 consequential decisions. Combines judgment, the leverage faced, and volume.',
    vci: '95% confidence interval on Value/100.',
    cons: 'Consequential decisions: near-zone calls with real stakes that this player could actually have challenged.',
    chal: 'Challenges this player personally initiated.',
    succ: 'Share of their challenges that were overturned.',
    net: 'Descriptive season total in leveraged runs: value earned from challenges minus value left on the table by declining good ones.',
    games: 'Team games in the sample.',
    actW: 'Wins the team actually captured from the challenge system.',
    optW: 'Wins the matrix policy would have captured, played with league-average eyesight and no hindsight.',
    gapW: 'Wins left on the table: optimal minus actual. Negative means the team already beats the benchmark.',
    date: 'Game date.',
    batter: 'Hitter at the plate. Click to filter to them.',
    catcher: 'Catcher behind the plate. Click to filter to them.',
    side: 'Which side owned the decision: the hitter (on a called strike) or the catcher (on a called ball).',
    type: 'Challenged = someone used a challenge. Missed = nobody did, though the matrix says it was worth it.',
    result: 'won / would-win = the call was genuinely wrong. lost / would-lose = the call was right.',
    count: 'Count the pitch was thrown on.',
    inning: 'Inning.',
    marginIn: 'Inches in the challenger\'s favor. Positive means ABS would have flipped the call.',
    ev: 'Expected value of the decision at the moment it was made, in leveraged runs.',
    playId: 'Opens the pitch on Baseball Savant.'
  };
  const th = (k, label, cls, sortKey, dir) =>
    `<th class="${cls || ''}" data-k="${k}"${COL_DESC[k] ? ` title="${esc(COL_DESC[k])}"` : ''}` +
    ` aria-sort="${sortKey === k ? (dir < 0 ? 'descending' : 'ascending') : 'none'}">${label}</th>`;

  function emptyState(msg, hint) {
    return `<div class="empty"><b>${esc(msg)}</b>${hint ? `<br>${hint}` : ''}</div>`;
  }

  function toCSV(cols, rows) {
    const q = v => {
      const s = v == null ? '' : String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [cols.map(c => q(c[1])).join(',')]
      .concat(rows.map(r => cols.map(c => q(r[c[0]])).join(',')))
      .join('\n');
  }
  function downloadCSV(name, cols, rows) {
    const blob = new Blob([toCSV(cols, rows)], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  // ---- URL <-> state (replaceState so we never re-trigger the router) ----
  function writeUrl() {
    const q = [];
    if (state.view === 'film') {
      if (film.date) q.push('date=' + film.date);
      if (film.bat) q.push('bat=' + encodeURIComponent(film.bat));
      if (film.cat) q.push('cat=' + encodeURIComponent(film.cat));
      if (film.team) q.push('team=' + film.team);
      if (film.type) q.push('type=' + film.type);
      if (film.count) q.push('count=' + film.count);
      if (film.inning) q.push('inning=' + film.inning);
      if (film.wrong !== 'wrong') q.push('all=1');
      if (filmSort !== 'value') q.push('sort=' + filmSort);
    } else if (state.view === 'leaders') {
      if (state.tab !== 'catchers') q.push('tab=' + state.tab);
      if (state.q) q.push('q=' + encodeURIComponent(state.q));
    }
    const hash = '#abs/' + state.view + (q.length ? '?' + q.join('&') : '');
    if (location.hash !== hash) history.replaceState(null, '', hash);
  }
  function applyUrl(query) {
    if (!query) return;
    const p = {};
    query.split('&').forEach(kv => {
      const i = kv.indexOf('='); if (i < 0) return;
      p[kv.slice(0, i)] = decodeURIComponent(kv.slice(i + 1));
    });
    if (p.tab) state.tab = p.tab;
    if (p.q) state.q = p.q;
    if (p.date) film.date = p.date;
    if (p.bat) film.bat = p.bat;
    if (p.cat) film.cat = p.cat;
    if (p.team) film.team = p.team;
    if (p.type) film.type = p.type;
    if (p.count) film.count = p.count;
    if (p.inning) film.inning = p.inning;
    if (p.all) film.wrong = '';
    if (p.sort) filmSort = p.sort;
  }

  // view switching is handled by the nav sub-tabs (#abs-subtabs) in app.js,
  // so the in-page switcher is intentionally empty
  function viewTabs() { return ''; }
  function bindViewTabs() { }

  // ================= LEADERBOARDS =================
  const COLS = {
    player: [['player', 'Player', 'l'], ['team', 'Tm', 'l'], ['skill', 'Skill+'], ['skci', '95% CI'],
    ['cons', 'Dec'], ['chal', 'Chal'], ['succ', 'Succ%'], ['net', 'NetVal']],
    backtest: [['team', 'Team', 'l'], ['games', 'G'], ['actW', 'ActualWins'], ['optW', 'OptimalWins'], ['gapW', 'LeftOnTable']]
  };
  const NOTES = {
    catchers: '<b>Skill+</b> is the talent estimate: leverage-blind decision quality indexed to 100, balanced so that simply challenging rarely cannot inflate it. It is shown only for catchers whose sample clears a reliability bar and reads "provisional" otherwise, and the wide CIs (~&plusmn;20) mean most catchers are still statistically indistinguishable from one another. <b>NetVal</b> is separate and purely descriptive &mdash; the leveraged runs a catcher actually gained and left on the table this season. It is <b>not</b> a talent estimate: tested season-half against season-half it has no forward signal at all, because it is mostly the leverage a catcher happened to be handed. <b>Succ%</b> is shown for transparency but is the wrong lens on purpose &mdash; it runs slightly <i>negative</i> against value, since the safest challengers are usually the ones being too conservative.',
    hitters: 'DESCRIPTIVE ONLY &mdash; no Skill+ is shown for hitters on purpose. Out-of-sample testing found no hitter skill metric that predicts forward without mostly measuring how <i>rarely</i> a hitter challenges (the naive version correlates -0.91 with challenge rate). The median hitter has about 32 real decisions, which is not a season. These rows are a record of what happened, ranked by NetVal. Expect this to firm up over 2-3 seasons.',
    teams: 'Team totals across all deciders, including pitcher-initiated challenges. Ranked by NetVal (descriptive).',
    backtest: 'Season replay: value captured by actual challenge usage vs the matrix policy run with league-average perception (no hindsight). Wins = leveraged runs times the league run-to-win factor. Negative LeftOnTable = the team already beats the league-perceiver benchmark.'
  };
  const DEFSORT = { catchers: 'skill', hitters: 'net', teams: 'net', backtest: 'gapW' };

  function slimRow(r) {
    // Skill+ is the TALENT claim, so it is gated on reliability and reads
    // "provisional" below the bar. Value/100 is descriptive by nature -- it
    // deliberately keeps the leverage a player was handed, so it never
    // stabilizes as a talent estimate and is always shown, with its CI
    // carrying the uncertainty.
    const sq = r.skillQual;
    return {
      player: r.player, team: r.team,
      skill: (r.skill == null || !sq) ? null : Math.round(r.skill),
      skci: (r.skillCI == null || !sq) ? null : Math.round(r.skillCI),
      value: r.value == null ? null : +r.value.toFixed(2),
      vci: r.valueCI == null ? null : +r.valueCI.toFixed(2),
      cons: r.consN || 0, chal: r.challenges,
      succ: r.successPct == null ? null : Math.round(r.successPct), net: +r.netValue.toFixed(2)
    };
  }
  function fmt(v, k) {
    if (v == null) return k === 'skill' ? 'provisional' : '';
    if (k === 'skci' || k === 'vci') return '&plusmn;' + v;
    if (k === 'value' || k === 'net' || k === 'gapW') return v.toFixed(2);
    if (typeof v === 'number' && !Number.isInteger(v)) return v.toFixed(2);
    return v;
  }
  function cellCls(v, k) {
    if ((k === 'net' || k === 'value' || k === 'gapW') && typeof v === 'number') return v > 0.005 ? 'pos' : (v < -0.005 ? 'neg' : '');
    if (k === 'skill' && typeof v === 'number') return v > 100.5 ? 'pos' : (v < 99.5 ? 'neg' : '');
    if (k === 'skill' && v == null) return 'dim';
    if (['player', 'team'].includes(k)) return 'l';
    return ['skci', 'vci', 'cons', 'succ', 'chal'].includes(k) ? 'dim' : '';
  }
  function leagueStrip() {
    const L = (core.grades.meta && core.grades.meta.league) || null;
    if (!L) return '';
    const gap = core.backtest.reduce((s, t) => s + (t.gapW || 0), 0);
    const pct = L.challenges ? Math.round(100 * L.won / L.challenges) : 0;
    const s = (k, v) => `<div class="s"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    return `<div class="strip">
      ${s('Challenges', L.challenges.toLocaleString())}
      ${s('Overturned', pct + '%')}
      ${s('Blown calls', L.blownCalls.toLocaleString())}
      ${s('Declined &amp; worth it', L.missN.toLocaleString())}
      ${s('Value left on table', L.missValue + ' runs')}
      ${s('Wins left on table', gap.toFixed(1))}
    </div>`;
  }

  const HOWTO_LEADERS = `<details class="howto"><summary>How to read this</summary><dl>
    <dt>Leveraged runs</dt><dd>The currency here. A run's worth scaled by how much the moment matters, so a flipped call in a tie game in the 9th counts for far more than the same call in a blowout.</dd>
    <dt>A lost challenge can still be a good decision</dt><dd>Everything is graded on the expected value at the moment of the call, not the outcome. A justified challenge that loses still grades positive; a reckless one that happens to win does not.</dd>
    <dt>Skill+ vs Value/100</dt><dd>Skill+ strips leverage out to isolate judgment (100 = average). Value/100 keeps it in, so it rewards judgment plus the stakes faced plus volume. They rank differently on purpose.</dd>
    <dt>Why some cells say "provisional"</dt><dd>Half a season is not enough to call most players good or bad. Only players whose sample clears a reliability bar get a number; everyone else stays descriptive.</dd>
  </dl></details>`;

  function renderLeaders() {
    const g = core.grades;
    const rowsFor = t => t === 'backtest' ? core.backtest : (g[t] || []).map(slimRow);
    const p = el();
    p.innerHTML = viewTabs() +
      `<h2>Challenge grades</h2>` + leagueStrip() + HOWTO_LEADERS +
      `<p class="abs-sub" id="abs-note"></p>
       <div class="abs-bar">
         <div class="abs-vtabs" id="abs-ltabs">${['catchers', 'hitters', 'teams', 'backtest'].map(t =>
        `<button data-ltab="${t}" aria-pressed="${state.tab === t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join('')}</div>
         <input type="search" id="abs-q" placeholder="Search player or team" value="${esc(state.q)}">
         <button id="abs-csv" class="chip" title="Download the rows currently shown">CSV</button>
         <span class="abs-count" id="abs-cnt"></span>
       </div>
       <div class="abs-tblwrap"><table id="abs-tbl"></table></div>
       <div id="abs-empty"></div>`;
    bindViewTabs();
    p.querySelectorAll('#abs-ltabs button').forEach(b => b.addEventListener('click', () => {
      state.tab = b.dataset.ltab; state.sort = DEFSORT[state.tab]; state.dir = -1;
      p.querySelectorAll('#abs-ltabs button').forEach(x =>
        x.setAttribute('aria-pressed', String(x.dataset.ltab === state.tab)));
      drawTable();
    }));
    p.querySelector('#abs-q').addEventListener('input', e => { state.q = e.target.value; drawTable(); });
    if (!COLS[state.tab] && state.tab !== 'backtest') state.tab = 'catchers';
    drawTable();

    function drawTable() {
      const cols = state.tab === 'backtest' ? COLS.backtest : COLS.player;
      let rows = rowsFor(state.tab).slice();
      if (state.q) { const q = state.q.toLowerCase(); rows = rows.filter(r => (r.player || r.team || '').toLowerCase().includes(q) || String(r.team).toLowerCase().includes(q)); }
      rows.sort((a, b) => {
        const x = a[state.sort], y = b[state.sort];
        if (x == null) return 1; if (y == null) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * state.dir * (typeof x === 'string' ? -1 : 1);
      });
      let h = '<thead><tr>' + cols.map(c => th(c[0], c[1], c[2], state.sort, state.dir)).join('') + '</tr></thead><tbody>';
      for (const r of rows) h += '<tr>' + cols.map(c => {
        // player names click through to that player's reel in the Film Room
        if (c[0] === 'player' && r.player) {
          const who = state.tab === 'catchers' ? 'cat' : 'bat';
          return `<td class="l"><span class="lead-link" data-who="${who}" data-name="${esc(r.player)}" role="link" tabindex="0">${esc(r.player)}</span></td>`;
        }
        return `<td class="${cellCls(r[c[0]], c[0])}">${fmt(r[c[0]], c[0])}</td>`;
      }).join('') + '</tr>';
      p.querySelector('#abs-tbl').innerHTML = h + '</tbody>';
      p.querySelector('#abs-cnt').textContent = rows.length + ' rows';
      p.querySelector('#abs-note').innerHTML = NOTES[state.tab];
      const emptyBox = p.querySelector('#abs-empty');
      if (!rows.length) {
        emptyBox.innerHTML = emptyState(
          state.q ? `No ${state.tab} match "${state.q}".` : `Nothing to show here yet.`,
          state.q ? 'Check the spelling, or clear the search box.' : '');
        p.querySelector('.abs-tblwrap').style.display = 'none';
      } else { emptyBox.innerHTML = ''; p.querySelector('.abs-tblwrap').style.display = ''; }
      p.querySelectorAll('#abs-tbl th').forEach(t => t.addEventListener('click', () => {
        const k = t.dataset.k; if (state.sort === k) state.dir *= -1; else { state.sort = k; state.dir = -1; } drawTable();
      }));
      const jump = elm => {
        film.date = ''; film.bat = ''; film.cat = ''; film.team = ''; film.type = '';
        film.count = ''; film.inning = ''; film.wrong = 'wrong';
        film[elm.dataset.who] = elm.dataset.name;
        state.view = 'film';
        document.querySelectorAll('.abs-tab').forEach(t2 => {
          const on = t2.getAttribute('data-absview') === 'film';
          t2.classList.toggle('active', on); t2.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        render();
      };
      p.querySelectorAll('.lead-link').forEach(elm => {
        elm.addEventListener('click', () => jump(elm));
        elm.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); jump(elm); } });
      });
      p.querySelector('#abs-csv').onclick = () => downloadCSV(`abs_${state.tab}_2026.csv`, cols, rows);
      writeUrl();
    }
  }

  // ================= MATRIX =================
  const clamp = d => Math.max(-12, Math.min(12, d));
  function contBat(inning, half, d) {
    d = clamp(d);
    if (half === 'top') {
      const home = inning >= 9 ? (d > 0 ? 1 : core.W[Math.min(inning, 10) + '|bottom|' + d]) : core.W[inning + '|bottom|' + d];
      return 1 - home;
    }
    if (inning >= 9) return d > 0 ? 1 : (d < 0 ? 0 : core.wExtraTie);
    return core.W[(inning + 1) + '|top|' + d];
  }
  function wpMid(inning, half, diff, bases, outs) {
    if (outs >= 3) return contBat(inning, half, diff);
    const dist = core.runDist[bases + '|' + outs]; let tot = 0;
    for (const k in dist) { const r = +k, d2 = half === 'top' ? diff - r : diff + r; tot += dist[k] * contBat(inning, half, d2); }
    return tot;
  }
  function walkBases(bases) {
    const f = bases[0] === '1', s2 = bases[1] === '1', t = bases[2] === '1';
    const runs = (f && s2 && t) ? 1 : 0, nt = (f && s2) ? true : t, ns = f ? true : s2;
    return ['1' + (ns ? '1' : '0') + (nt ? '1' : '0'), runs];
  }
  function gainFor(b, s, inning) {
    const bases = state.bases.map(x => x ? '1' : '0').join(''), outs = state.outs;
    const diff = state.home - state.away;
    const g = core.G[Math.min(inning, 10) + '|' + state.half + '|' + clamp(diff)];
    let wpBase = null, ballWP, strikeWP;
    if (b === 3) { wpBase = wpMid(inning, state.half, diff, bases, outs); const [nb, runs] = walkBases(bases); const d2 = state.half === 'top' ? diff - runs : diff + runs; ballWP = wpMid(inning, state.half, d2, nb, outs) - wpBase; }
    else ballWP = core.countRV288[(b + 1) + '-' + s + '|' + bases + '|' + outs] * g;
    if (s === 2) { if (wpBase === null) wpBase = wpMid(inning, state.half, diff, bases, outs); strikeWP = wpMid(inning, state.half, diff, bases, outs + 1) - wpBase; }
    else strikeWP = core.countRV288[b + '-' + (s + 1) + '|' + bases + '|' + outs] * g;
    return (ballWP - strikeWP) / core.gAvg;
  }
  const Tfor = inning => 2 * (9 - Math.min(inning, 9)) + (state.half === 'top' ? 2 : 1);
  function battingLead() { return state.half === 'top' ? state.away - state.home : state.home - state.away; }
  function costFor(inning) { const dTeam = state.side === 'bat' ? battingLead() : -battingLead(); return core.Cgrid[state.k + '|' + Tfor(inning) + '|' + clamp(dTeam)]; }
  function pstarFor(b, s, inning) { const g = Math.max(gainFor(b, s, inning), 1e-4), c = costFor(inning); return c / (g + c); }
  const RAMP = [[240, 228, 216], [236, 206, 168], [214, 143, 96], [166, 59, 34], [120, 34, 22]];
  function heat(p) {
    const t = Math.max(0, Math.min(1, p / 0.8)), x = t * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), f = x - i;
    const c = RAMP[i].map((v, j) => Math.round(v + (RAMP[i + 1][j] - v) * f));
    return { bg: `rgb(${c.join(',')})`, dark: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) < 140 };
  }
  function posterior(key, x) {
    const g = core.post[key]; if (x <= g[0][0]) return g[0][1]; if (x >= g[g.length - 1][0]) return g[g.length - 1][1];
    const st = g[1][0] - g[0][0], i = Math.floor((x - g[0][0]) / st), a = g[i], b = g[Math.min(i + 1, g.length - 1)];
    return a[1] + (b[1] - a[1]) * (x - a[0]) / Math.max(b[0] - a[0], 1e-9);
  }
  function neededInches(key, pstar) {
    const g = core.post[key]; if (g[g.length - 1][1] < pstar) return null;
    for (const [x, p] of g) if (p >= pstar) return Math.max(x, 0); return null;
  }
  function pitchGeom() {
    const ax = Math.abs(state.px), dx = ZHW - ax, dt = state.ztop - state.pz, db = state.pz - state.zbot;
    const dzOut = Math.max(state.pz - state.ztop, state.zbot - state.pz), inside = dx >= 0 && dzOut <= 0;
    const d = inside ? -Math.min(dx, dt, db) : Math.hypot(Math.max(-dx, 0), Math.max(dzOut, 0));
    const mEdge = Math.min(dx, dt, db), reg = mEdge === dx ? 'side' : (mEdge === dt ? 'top' : 'bottom');
    return { loc: THR - d, reg };
  }
  const ordinal = n => n + (['th', 'st', 'nd', 'rd'][(n % 100 > 10 && n % 100 < 14) ? 0 : Math.min(n % 10, 4)] || 'th');
  const COUNTS = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]];
  const INN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  // Parse a Gameday feed into per-pitch situations (JS port of abs_pull_pitch.py):
  // pre-pitch count, outs, base state, running score, and plate_x/plate_z.
  function parseFeedPitches(feed) {
    const gd = feed.gameData, live = feed.liveData;
    const away = gd.teams.away.abbreviation || gd.teams.away.name;
    const home = gd.teams.home.abbreviation || gd.teams.home.name;
    const score = { away: 0, home: 0 };
    let prevHalf = null, bases = {};
    const out = [];
    for (const play of live.plays.allPlays) {
      const ab = play.about, half = ab.halfInning === 'top' ? 'top' : 'bottom';
      const hk = ab.inning + '|' + half;
      if (hk !== prevHalf) { bases = {}; prevHalf = hk; }
      const batSide = half === 'top' ? 'away' : 'home';
      let b = 0, s = 0;
      for (const ev of (play.playEvents || [])) {
        if (!ev.isPitch) continue;
        const c = (ev.pitchData || {}).coordinates || {};
        if (c.pX != null && c.pZ != null) {
          const det = ev.details;
          out.push({
            batterId: play.matchup.batter.id, batter: play.matchup.batter.fullName,
            pitcher: play.matchup.pitcher.fullName, inning: ab.inning, half: half,
            balls: b, strikes: s, outs: (ev.count || {}).outs || 0,
            bases: ['1B', '2B', '3B'].map(x => bases[x] ? '1' : '0').join(''),
            away: score.away, home: score.home,
            px: c.pX, pz: c.pZ, ptype: (det.type || {}).description || 'pitch',
            call: det.call ? det.call.description : '', playId: ev.playId
          });
        }
        const cnt = ev.count || {};
        if (cnt.balls != null) b = cnt.balls;
        if (cnt.strikes != null) s = cnt.strikes;
      }
      const runners = (play.runners || []).filter(e => e.movement);
      score[batSide] += new Set(runners.filter(e => e.movement.end === 'score')
        .map(e => e.details.runner.id)).size;
      runners.sort((a, b) => (a.details.playIndex || 0) - (b.details.playIndex || 0));
      for (const e of runners) {
        const rid = e.details.runner.id;
        for (const bb in bases) if (bases[bb] === rid) delete bases[bb];
        const end = e.movement.end;
        if (['1B', '2B', '3B'].includes(end) && !e.movement.isOut) bases[end] = rid;
      }
    }
    return out;
  }

  function renderMatrix() {
    const p = el();
    p.innerHTML = viewTabs() +
      `<h2>The challenge matrix</h2>
       <p class="abs-sub">Each cell is the break-even confidence: how sure you must be the call was wrong before challenging is worth it, weighing the flip value in this exact spot against the option value of the challenge you might lose.</p>
       <div class="mx-ctrls">
        <div class="mx-ctl"><label>Challenges left</label><span class="seg" id="mK"><button data-v="2" aria-pressed="true">2</button><button data-v="1">1</button></span></div>
        <div class="mx-ctl"><label>Half</label><span class="seg" id="mHalf"><button data-v="top" aria-pressed="true">Top</button><button data-v="bottom">Bottom</button></span></div>
        <div class="mx-ctl"><label>Outs</label><span class="seg" id="mOuts"><button data-v="0" aria-pressed="true">0</button><button data-v="1">1</button><button data-v="2">2</button></span></div>
        <div class="mx-ctl"><label>Runners</label><span id="mBases">${[0, 1, 2].map(i => `<button class="chip" data-b="${i}">${['1B', '2B', '3B'][i]}</button>`).join(' ')}</span></div>
        <div class="mx-ctl"><label>Score (Away &ndash; Home)</label>
          <span style="display:inline-flex;gap:5px;align-items:center">
            <input id="mAway" class="abs-inp" type="number" min="0" max="30" value="0" style="width:52px" aria-label="Away score">
            <span style="color:var(--text-muted)">&ndash;</span>
            <input id="mHome" class="abs-inp" type="number" min="0" max="30" value="0" style="width:52px" aria-label="Home score"></span>
          <div style="font-size:11px;color:var(--text-muted);margin-top:4px" id="mScoreNote">tie game</div></div>
        <div class="mx-ctl"><label>Deciding side</label><span class="seg" id="mSide"><button data-v="fld" aria-pressed="true">Catcher</button><button data-v="bat">Hitter</button></span></div>
        <div class="mx-ctl"><label>&nbsp;</label><button id="mReset" class="chip" title="Reset every input to the default situation">Reset</button></div>
       </div>
       <div class="mx-scroll"><table class="mx-grid" id="mGrid"></table></div>
       <div class="mx-legend"><span>Challenge freely</span><div class="mx-grad" id="mGrad"></div><span>Hold</span><span style="margin-left:auto">cells = break-even confidence %</span></div>
       <h2 style="margin-top:26px">Price a specific pitch</h2>
       <p class="abs-sub">Click a cell to load that situation, then set where the ball was. Pick a hitter for their exact zone, or type Gameday plate_x / plate_z.</p>
       <div class="panel">
        <div id="mFrom" style="display:none;border:1px solid var(--border);border-radius:9px;background:var(--accent-light);padding:9px 13px;margin-bottom:12px;font-size:13px"></div>
        <div class="mx-live" style="border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:14px;background:var(--bg-primary)">
          <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px">Load from a live game</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
            <input id="mvDate" type="date" class="abs-inp">
            <select id="mvGame" class="abs-inp" style="min-width:220px"><option value="">Pick a date first</option></select>
            <select id="mvBatter" class="abs-inp" style="min-width:170px" disabled><option>Pick a game first</option></select>
            <span id="mvStatus" style="font-size:12px;color:var(--text-muted)"></span>
          </div>
          <div id="mvPitches" style="margin-top:10px;max-height:200px;overflow:auto"></div>
        </div>
        <p id="mState" class="abs-sub" style="margin-bottom:10px"></p>
        <div class="facts">
          <div class="fact"><div class="k">Flip worth</div><div class="v" id="fG">-</div></div>
          <div class="fact"><div class="k">Challenge value</div><div class="v" id="fC">-</div></div>
          <div class="fact"><div class="k">Break-even</div><div class="v" id="fP">-</div></div>
          <div class="fact"><div class="k">Identifiable as</div><div class="v" id="fConf">-</div></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:12px">
          <div class="mx-ctl"><label>Count</label>
            <select id="mCount" class="abs-inp">${COUNTS.map(([b, s]) => `<option value="${b}-${s}">${b}-${s}</option>`).join('')}</select></div>
          <div class="mx-ctl"><label>Inning</label>
            <select id="mInning" class="abs-inp">${INN.map(i => `<option value="${i}">${i === 10 ? '10+ (extras)' : i}</option>`).join('')}</select></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;margin-bottom:10px">
          <div class="mx-ctl"><label>Hitter (exact zone)</label>
            <input list="mHitters" id="mHit" class="abs-inp" placeholder="League average" style="width:200px">
            <datalist id="mHitters"></datalist>
            <div style="font-size:12px;color:var(--text-muted);margin-top:5px" id="mZone"></div>
            <label style="margin-top:12px">Gameday plate_x / plate_z (ft)</label>
            <div style="display:flex;gap:6px"><input id="mX" class="abs-inp" type="number" step="0.01" placeholder="plate_x" style="width:100px"><input id="mZ" class="abs-inp" type="number" step="0.01" placeholder="plate_z" style="width:100px"></div>
          </div>
          <div class="mx-ctl"><label>Or click where the ball crossed - <span id="mLoc" style="color:var(--text-primary)"></span></label>
            <svg id="mPlot" class="zplot" viewBox="0 0 224 288" width="224" height="288">
              <rect id="zEdge" fill="none" stroke="var(--accent)" stroke-dasharray="4 4" stroke-width="1"></rect>
              <rect id="zRect" fill="none" stroke="var(--text-primary)" stroke-width="1.5"></rect>
              <circle id="zBall" fill="none" stroke="var(--accent)" stroke-width="2"></circle>
            </svg></div>
        </div>
        <div class="verdict" id="mVerd"><span class="word">-</span><span class="why"></span></div>
       </div>`;
    bindViewTabs();
    // controls
    const seg = (id, key, parse) => p.querySelectorAll('#' + id + ' button').forEach(btn => btn.addEventListener('click', () => {
      p.querySelectorAll('#' + id + ' button').forEach(x => x.setAttribute('aria-pressed', 'false'));
      btn.setAttribute('aria-pressed', 'true'); state[key] = parse ? parse(btn.dataset.v) : btn.dataset.v; drawGrid(); drawPanel();
    }));
    seg('mK', 'k', v => +v); seg('mHalf', 'half'); seg('mOuts', 'outs', v => +v); seg('mSide', 'side');
    p.querySelectorAll('#mBases .chip').forEach(ch => ch.addEventListener('click', () => {
      const i = +ch.dataset.b; state.bases[i] = !state.bases[i]; ch.setAttribute('aria-pressed', String(state.bases[i])); drawGrid(); drawPanel();
    }));
    const mAway = p.querySelector('#mAway'), mHome = p.querySelector('#mHome');
    function onScore() {
      state.away = Math.max(0, Math.min(30, parseInt(mAway.value) || 0));
      state.home = Math.max(0, Math.min(30, parseInt(mHome.value) || 0));
      drawGrid(); drawPanel();
    }
    mAway.addEventListener('input', onScore); mHome.addEventListener('input', onScore);
    p.querySelector('#mCount').addEventListener('change', e => {
      const [b, s] = e.target.value.split('-').map(Number); state.sel.b = b; state.sel.s = s; drawGrid(); drawPanel();
    });
    p.querySelector('#mInning').addEventListener('change', e => { state.sel.inning = +e.target.value; drawGrid(); drawPanel(); });
    // zone plot + hitter + coords
    const zr = p.querySelector('#zRect'), ze = p.querySelector('#zEdge');
    function drawZone() {
      const x0 = 112 - ZHW * PXIN, y0 = 288 - (state.ztop - PLOT_Z0) * PXIN;
      zr.setAttribute('x', x0); zr.setAttribute('y', y0); zr.setAttribute('width', 2 * ZHW * PXIN); zr.setAttribute('height', (state.ztop - state.zbot) * PXIN);
      const off = THR * PXIN; ze.setAttribute('x', x0 - off); ze.setAttribute('y', y0 - off); ze.setAttribute('width', 2 * ZHW * PXIN + 2 * off); ze.setAttribute('height', (state.ztop - state.zbot) * PXIN + 2 * off);
    }
    drawZone();
    p.querySelector('#mPlot').addEventListener('click', e => {
      const r = e.currentTarget.getBoundingClientRect(), sx = (e.clientX - r.left) * 224 / r.width, sy = (e.clientY - r.top) * 288 / r.height;
      state.px = (sx - 112) / PXIN; state.pz = (288 - sy) / PXIN + PLOT_Z0;
      p.querySelector('#mX').value = (state.px / 12).toFixed(2); p.querySelector('#mZ').value = (state.pz / 12).toFixed(2); drawPanel();
    });
    p.querySelector('#mHitters').innerHTML = core.zones.map(z => `<option value="${esc(z.n)}"></option>`).join('');
    p.querySelector('#mZone').textContent = 'zone 3.30 / 1.60 ft (league average)';
    p.querySelector('#mHit').addEventListener('change', e => {
      const z = core.zones.find(x => x.n === e.target.value);
      if (z) { state.ztop = z.t * 12; state.zbot = z.b * 12; p.querySelector('#mZone').textContent = `zone ${z.t.toFixed(2)} / ${z.b.toFixed(2)} ft (${z.n})`; }
      else { state.ztop = 39.6; state.zbot = 19.2; p.querySelector('#mZone').textContent = 'zone 3.30 / 1.60 ft (league average)'; }
      drawZone(); drawPanel();
    });
    const ex = () => { const x = parseFloat(p.querySelector('#mX').value), z = parseFloat(p.querySelector('#mZ').value); if (!isNaN(x)) state.px = x * 12; if (!isNaN(z)) state.pz = z * 12; drawPanel(); };
    p.querySelector('#mX').addEventListener('input', ex); p.querySelector('#mZ').addEventListener('input', ex);

    // ---- load from a live game ----
    let livePitches = [];
    const mvDate = p.querySelector('#mvDate'), mvGame = p.querySelector('#mvGame');
    const mvBatter = p.querySelector('#mvBatter'), mvPitches = p.querySelector('#mvPitches');
    const mvStatus = p.querySelector('#mvStatus');
    const now = new Date();
    mvDate.value = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    function syncControls() {
      const setSeg = (id, val) => p.querySelectorAll('#' + id + ' button')
        .forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === String(val))));
      setSeg('mK', state.k); setSeg('mHalf', state.half); setSeg('mOuts', state.outs); setSeg('mSide', state.side);
      p.querySelectorAll('#mBases .chip').forEach(ch => ch.setAttribute('aria-pressed', String(state.bases[+ch.dataset.b])));
      p.querySelector('#mAway').value = state.away; p.querySelector('#mHome').value = state.home;
      p.querySelector('#mX').value = (state.px / 12).toFixed(2); p.querySelector('#mZ').value = (state.pz / 12).toFixed(2);
    }
    p.querySelector('#mReset').addEventListener('click', () => {
      Object.assign(state, JSON.parse(JSON.stringify(MX_DEFAULTS)));
      p.querySelector('#mHit').value = '';
      p.querySelector('#mZone').textContent = 'zone 3.30 / 1.60 ft (league average)';
      mvPitches.innerHTML = '';
      syncControls(); drawZone(); drawGrid(); drawPanel();
    });
    // Accepts a live-feed pitch (balls/strikes) or a Film Room event (count
    // string, explicit zone, and the side that owned the decision).
    function loadPitch(pt) {
      const b = pt.balls != null ? pt.balls : +String(pt.count).split('-')[0];
      const s = pt.strikes != null ? pt.strikes : +String(pt.count).split('-')[1];
      state.sel = { b: b, s: s, inning: Math.min(pt.inning, 10) };
      state.half = pt.half; state.outs = Math.min(pt.outs, 2);
      state.bases = [pt.bases[0] === '1', pt.bases[1] === '1', pt.bases[2] === '1'];
      state.away = pt.away; state.home = pt.home;
      state.side = pt.role ? (pt.role === 'batter' ? 'bat' : 'fld') : 'bat';
      state.px = pt.px * 12; state.pz = pt.pz * 12;
      if (pt.szTop != null && pt.szBot != null) {
        state.ztop = pt.szTop * 12; state.zbot = pt.szBot * 12;
        p.querySelector('#mZone').textContent = `zone ${pt.szTop.toFixed(2)} / ${pt.szBot.toFixed(2)} ft (${pt.batter})`;
        p.querySelector('#mHit').value = pt.batter || '';
      } else {
        const z = core.zones.find(x => x.n === pt.batter);
        if (z) { state.ztop = z.t * 12; state.zbot = z.b * 12; p.querySelector('#mZone').textContent = `zone ${z.t.toFixed(2)} / ${z.b.toFixed(2)} ft (${pt.batter})`; }
        else { state.ztop = 39.6; state.zbot = 19.2; p.querySelector('#mZone').textContent = 'zone 3.30 / 1.60 ft (league average)'; }
        p.querySelector('#mHit').value = z ? pt.batter : '';
      }
      syncControls(); drawZone(); drawGrid(); drawPanel();
    }
    function showPitches(bid) {
      const list = livePitches.filter(x => String(x.batterId) === String(bid));
      mvPitches.innerHTML = list.length ? list.map((x, i) =>
        `<button class="mv-pitch" data-i="${i}" style="display:block;width:100%;text-align:left;border:1px solid var(--border);border-radius:6px;background:var(--bg-card);color:var(--text-primary);font:13px 'IBM Plex Sans';padding:6px 10px;margin-bottom:4px;cursor:pointer">${x.half === 'top' ? 'Top' : 'Bot'} ${x.inning}, <b>${x.balls}-${x.strikes}</b> &middot; ${x.ptype} &middot; ${x.call}</button>`).join('')
        : `<span style="font-size:12px;color:var(--text-muted)">no tracked pitches for this batter</span>`;
      mvPitches.querySelectorAll('.mv-pitch').forEach(btn => btn.addEventListener('click', () => loadPitch(list[+btn.dataset.i])));
    }
    async function loadSchedule(d) {
      mvStatus.textContent = 'loading games...'; mvGame.innerHTML = '<option value="">...</option>';
      mvBatter.innerHTML = '<option>Pick a game first</option>'; mvBatter.disabled = true; mvPitches.innerHTML = '';
      try {
        const j = await fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${d}`).then(r => r.json());
        const games = (j.dates && j.dates[0]) ? j.dates[0].games : [];
        mvGame.innerHTML = '<option value="">Pick a game...</option>' + games.map(g =>
          `<option value="${g.gamePk}">${g.teams.away.team.name} @ ${g.teams.home.team.name}${g.status && g.status.abstractGameState ? ' (' + g.status.abstractGameState + ')' : ''}</option>`).join('');
        mvStatus.textContent = games.length ? `${games.length} games` : 'no games that day';
      } catch (e) { mvStatus.textContent = 'could not load schedule'; }
    }
    async function loadGame(pk) {
      mvStatus.textContent = 'loading pitches...'; mvBatter.disabled = true; mvPitches.innerHTML = '';
      try {
        const feed = await fetch(`https://statsapi.mlb.com/api/v1.1/game/${pk}/feed/live`).then(r => r.json());
        livePitches = parseFeedPitches(feed);
        const seen = new Set(), bats = [];
        for (const x of livePitches) if (!seen.has(x.batterId)) { seen.add(x.batterId); bats.push(x); }
        mvBatter.innerHTML = '<option value="">Pick a batter...</option>' + bats.map(x =>
          `<option value="${x.batterId}">${esc(x.batter)}</option>`).join('');
        mvBatter.disabled = false;
        mvStatus.textContent = `${livePitches.length} tracked pitches`;
      } catch (e) { mvStatus.textContent = 'could not load game'; }
    }
    mvDate.addEventListener('change', () => loadSchedule(mvDate.value));
    mvGame.addEventListener('change', () => { if (mvGame.value) loadGame(mvGame.value); });
    mvBatter.addEventListener('change', () => { if (mvBatter.value) showPitches(mvBatter.value); });
    loadSchedule(mvDate.value);

    drawGrid(); drawPanel();
    if (pendingPitch) {                 // arrived here from a Film Room row
      const pp = pendingPitch; pendingPitch = null;
      loadPitch(pp);
      const banner = p.querySelector('#mFrom');
      if (banner) {
        banner.style.display = '';
        banner.innerHTML = `Loaded from the Film Room: <b>${esc(pp.batter)}</b> vs <b>${esc(pp.pitcher || '')}</b>, ${pp.date}, ${pp.count} count &mdash; ${pp.result === 'won' || pp.result === 'would-win' ? 'a blown call' : 'the call was right'}.`;
      }
      p.querySelector('#mState').scrollIntoView({ block: 'center', behavior: 'smooth' });
    }

    function drawGrid() {
      const mx = p.querySelector('#mGrid');
      let h = '<thead><tr><th></th>' + INN.map(i => `<th>${i === 10 ? '10+' : i}</th>`).join('') + '</tr></thead><tbody>';
      for (const [b, s] of COUNTS) {
        h += `<tr><th class="rowh">${b}–${s}</th>`;
        for (const inn of INN) {
          const pp = pstarFor(b, s, inn), col = heat(pp), sel = (state.sel.b === b && state.sel.s === s && state.sel.inning === inn) ? ' sel' : '';
          const txt = col.dark ? '#f7f2e9' : '#1a1612';
          h += `<td><button class="cell${sel}" style="background:${col.bg};color:${txt}" data-b="${b}" data-s="${s}" data-i="${inn}">${Math.round(pp * 100)}</button></td>`;
        }
        h += '</tr>';
      }
      mx.innerHTML = h + '</tbody>';
      mx.querySelectorAll('.cell').forEach(c => c.addEventListener('click', () => {
        state.sel = { b: +c.dataset.b, s: +c.dataset.s, inning: +c.dataset.i }; drawGrid(); drawPanel();
      }));
      const stops = []; for (let i = 0; i <= 10; i++) stops.push(heat(0.8 * i / 10).bg + ' ' + (i * 10) + '%');
      p.querySelector('#mGrad').style.background = 'linear-gradient(90deg,' + stops.join(',') + ')';
    }
    function locText(loc) { return loc > 0 ? loc.toFixed(2).replace(/\.?0+$/, '') + ' in INTO the zone' : loc < 0 ? (-loc).toFixed(2).replace(/\.?0+$/, '') + ' in OFF the zone' : 'grazing the edge'; }
    function drawPanel() {
      const { b, s, inning } = state.sel, g = gainFor(b, s, inning), c = costFor(inning), pstar = c / (Math.max(g, 1e-4) + c);
      const { loc, reg } = pitchGeom(), m = state.side === 'fld' ? loc : -loc, conf = posterior(state.side + '|' + reg, m);
      const wouldWin = state.side === 'fld' ? loc >= 0 : loc < 0, ruling = loc >= 0 ? 'strike' : 'ball';
      const bl = battingLead();
      const lead = bl === 0 ? 'tie game' : (bl > 0 ? `batting up ${bl}` : `batting down ${-bl}`);
      // keep the situation inputs in sync with the selected cell / state
      const cSel = p.querySelector('#mCount'); if (cSel) cSel.value = b + '-' + s;
      const iSel = p.querySelector('#mInning'); if (iSel) iSel.value = inning;
      const sNote = p.querySelector('#mScoreNote'); if (sNote) sNote.textContent = lead;
      const on = ['1B', '2B', '3B'].filter((_, i) => state.bases[i]);
      p.querySelector('#mState').innerHTML = `<b>${b}–${s}</b> · <b>${inning === 10 ? 'extras' : ordinal(inning)}, ${state.half}</b> · ${state.outs} out · ${on.length ? on.join(' & ') + ' on' : 'bases empty'} · ${lead} · <b>${state.k}</b> challenge${state.k > 1 ? 's' : ''} left`;
      p.querySelector('#fG').textContent = g.toFixed(2); p.querySelector('#fC').textContent = c.toFixed(2);
      p.querySelector('#fP').textContent = Math.round(pstar * 100) + '%'; p.querySelector('#fConf').textContent = Math.round(conf * 100) + '%';
      p.querySelector('#mLoc').textContent = locText(loc) + ' · ' + reg + ' edge';
      const ball = p.querySelector('#zBall'); ball.setAttribute('cx', 112 + state.px * PXIN); ball.setAttribute('cy', 288 - (state.pz - PLOT_Z0) * PXIN); ball.setAttribute('r', THR * PXIN);
      const need = neededInches(state.side + '|' + reg, pstar), roleName = state.side === 'fld' ? 'catcher' : 'hitter';
      const needTxt = need == null ? null : (need < 0.05 ? (state.side === 'fld' ? 'that grazes the zone' : 'that slips off the edge') : (state.side === 'fld' ? `${need.toFixed(1)} in or more into the zone` : `${need.toFixed(1)} in or more off the zone`));
      const cp = Math.round(conf * 100), pp = Math.round(pstar * 100);
      const truth = wouldWin ? `ABS would overturn this (true ${ruling})` : `ABS would uphold the call (true ${ruling})`;
      const v = p.querySelector('#mVerd');
      if (conf >= pstar && wouldWin) {
        v.className = 'verdict go';
        v.innerHTML = `<span class="word">Challenge</span><span class="why">${truth}. At ${cp}% confidence you clear the ${pp}% bar. Rule of thumb: challenge any pitch ${needTxt}.</span>`;
      } else if (conf >= pstar) {
        v.className = 'verdict go';
        v.innerHTML = `<span class="word">Challenge</span><span class="why">Worth the gamble: the bar is only ${pp}% in a spot this leveraged. This exact pitch is a ${ruling} and would lose, but a ${roleName} can't know that live, and challenging pitches that look like this still pays off on average.</span>`;
      } else if (need == null) {
        v.className = 'verdict hold';
        v.innerHTML = `<span class="word">Hold</span><span class="why">${truth}, but no ${roleName} can reach ${pp}% certainty by eye here, so it isn't worth a challenge in the moment. Save it.</span>`;
      } else {
        v.className = 'verdict hold';
        v.innerHTML = `<span class="word">Hold</span><span class="why">${truth}. A ${roleName} could only be ${cp}% sure, under the ${pp}% bar. It becomes a good challenge for a pitch ${needTxt}.</span>`;
      }
    }
  }

  // ================= FILM ROOM =================
  const HOWTO_FILM = `<details class="howto"><summary>How to read this</summary><dl>
    <dt>What counts as an opportunity</dt><dd>A pitch ABS would have called the other way. "Challenged" means someone spent a challenge on it; "Missed" means nobody did even though the matrix says it was worth it.</dd>
    <dt>Value</dt><dd>Leveraged runs riding on that single call &mdash; the run swing scaled by how much the moment mattered. This is what the board ranks by.</dd>
    <dt>Margin</dt><dd>How many inches the call was wrong by, from the challenger's point of view. Bigger margin = easier to spot live.</dd>
    <dt>Blown calls only</dt><dd>On by default. Turn it off to also see challenges spent on calls that were actually correct &mdash; including high-leverage gambles that were right to take and still lost.</dd>
    <dt>Click any row</dt><dd>Loads that exact pitch into the Matrix tool so you can see the full decision math behind it.</dd>
  </dl></details>`;

  async function renderFilm() {
    const p = el();
    p.innerHTML = viewTabs() + `<h2>Top challenge opportunities</h2><p class="abs-sub">Loading clips...</p>`;
    bindViewTabs();
    if (!events) { try { events = (await fetch('data/abs_challenge_events_2026.json').then(r => r.json())).events; } catch (e) { events = []; } }
    if (state.view !== 'film') return;
    const rows = events.map(e => ({
      date: e.date, batter: e.batter || '', catcher: e.catcher || '', team: e.team,
      side: e.role === 'batter' ? 'Hitter' : (e.role === 'pitcher' ? 'Pitcher' : 'Catcher'),
      type: e.type, result: e.result, count: e.count, inning: e.inning,
      marginIn: e.marginIn, value: +(+e.gain).toFixed(2), ev: +(+e.ev).toFixed(2),
      playId: e.playId, _raw: e
    }));
    const dates = [...new Set(rows.map(r => r.date))].sort().reverse();
    const teams = [...new Set(rows.map(r => r.team))].sort();
    const batters = [...new Set(rows.map(r => r.batter).filter(Boolean))].sort();
    const catchers = [...new Set(rows.map(r => r.catcher).filter(Boolean))].sort();
    const latest = dates[0];
    if (film.date === null) film.date = latest || '';   // first visit defaults to the newest day
    const COUNTS_L = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2'];
    const sel = (v, cur) => v === cur ? ' selected' : '';
    p.innerHTML = viewTabs() +
      `<h2>Top challenge opportunities</h2>` + HOWTO_FILM +
      `<p class="abs-sub">Blown calls ranked by <b>Value</b> &mdash; the leveraged runs riding on that pitch &mdash; whether or not anyone challenged. Data runs through <b>${latest}</b>; today's games post after the next morning refresh. Click a row to price it in the Matrix.</p>
       <div class="abs-bar">
         <select id="fDate" class="abs-inp" title="Game date"><option value="">All dates</option>${dates.map(d => `<option value="${d}"${sel(d, film.date)}>${d}</option>`).join('')}</select>
         <input id="fBat" class="abs-inp" list="fBatList" placeholder="Hitter" style="width:140px" value="${esc(film.bat)}"><datalist id="fBatList">${batters.map(b => `<option value="${esc(b)}"></option>`).join('')}</datalist>
         <input id="fCat" class="abs-inp" list="fCatList" placeholder="Catcher" style="width:140px" value="${esc(film.cat)}"><datalist id="fCatList">${catchers.map(c => `<option value="${esc(c)}"></option>`).join('')}</datalist>
         <select id="fTeam" class="abs-inp"><option value="">All teams</option>${teams.map(t => `<option value="${t}"${sel(t, film.team)}>${t}</option>`).join('')}</select>
         <select id="fCount" class="abs-inp"><option value="">Any count</option>${COUNTS_L.map(c => `<option value="${c}"${sel(c, film.count)}>${c}</option>`).join('')}</select>
         <select id="fInning" class="abs-inp"><option value="">Any inning</option>${[1,2,3,4,5,6,7,8,9].map(i => `<option value="${i}"${sel(String(i), film.inning)}>Inn ${i}</option>`).join('')}<option value="10"${sel('10', film.inning)}>Inn 10+</option></select>
         <select id="fType" class="abs-inp"><option value="">Taken &amp; declined</option><option value="challenge"${sel('challenge', film.type)}>Challenged</option><option value="miss"${sel('miss', film.type)}>Declined (missed)</option></select>
         <select id="fWrong" class="abs-inp"><option value="wrong"${sel('wrong', film.wrong)}>Blown calls only</option><option value=""${sel('', film.wrong)}>Every decision</option></select>
         <button id="fClear" class="chip">Clear</button>
         <button id="fCsv" class="chip" title="Download the rows currently shown">CSV</button>
         <span class="abs-count" id="fcnt"></span>
       </div>
       <div class="abs-tblwrap"><table id="ftbl"></table></div>
       <div id="fempty"></div>`;
    bindViewTabs();
    const cols = [['date', 'Date', 'l'], ['batter', 'Hitter', 'l'], ['catcher', 'Catcher', 'l'],
    ['team', 'Tm', 'l'], ['side', 'Decider', 'l'], ['result', 'Result', 'l'],
    ['count', 'Cnt'], ['inning', 'Inn'], ['marginIn', 'Margin'], ['value', 'Value'], ['ev', 'EV'],
    ['playId', 'Video', 'l']];
    // 'wrong' keeps only calls ABS would have flipped (won / would-win), i.e.
    // real opportunities. Without it, high-leverage +EV gambles on correct
    // calls float to the top and the board stops meaning what it says.
    const isWrong = x => x.result === 'won' || x.result === 'would-win';
    function draw() {
      let r = rows.filter(x =>
        (!film.date || x.date === film.date) &&
        (!film.team || x.team === film.team) &&
        (!film.type || x.type === film.type) &&
        (!film.count || x.count === film.count) &&
        (!film.inning || String(film.inning === '10' ? Math.min(x.inning, 10) : x.inning) === film.inning) &&
        (!film.wrong || isWrong(x)) &&
        (!film.bat || x.batter.toLowerCase().includes(film.bat.toLowerCase())) &&
        (!film.cat || x.catcher.toLowerCase().includes(film.cat.toLowerCase())));
      r.sort((a, b) => { const x = a[filmSort], y = b[filmSort]; if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * filmDir * (typeof x === 'string' ? -1 : 1); });
      const tot = r.length; r = r.slice(0, 500);
      let h = '<thead><tr>' + cols.map(c => th(c[0], c[1], c[2], filmSort, filmDir)).join('') + '</tr></thead><tbody>';
      r.forEach((x, i) => {
        h += `<tr class="row-click" data-i="${i}" title="Click to price this pitch in the Matrix">` + cols.map(c => {
          if (c[0] === 'playId') return `<td class="l"><a class="abs-vid" href="https://baseballsavant.mlb.com/sporty-videos?playId=${x.playId}" target="_blank" rel="noopener">&#9654; watch</a></td>`;
          const v = x[c[0]];
          const cl = (c[0] === 'ev' || c[0] === 'value') ? (v > 0.005 ? 'pos' : v < -0.005 ? 'neg' : '')
            : (['batter', 'catcher', 'team', 'date', 'type', 'result', 'side'].includes(c[0]) ? 'l' : 'dim');
          const txt = (c[0] === 'ev' || c[0] === 'value' || c[0] === 'marginIn') ? (+v).toFixed(2) : v;
          return `<td class="${cl}">${txt}</td>`;
        }).join('') + '</tr>';
      });
      p.querySelector('#ftbl').innerHTML = h + '</tbody>';
      p.querySelector('#fcnt').textContent = tot > 500 ? `${tot} rows (showing 500)` : tot + ' rows';
      const box = p.querySelector('#fempty');
      if (!tot) {
        const who = film.bat || film.cat;
        const bits = [];
        if (film.date) bits.push('that date');
        if (film.count) bits.push('that count');
        if (film.inning) bits.push('that inning');
        box.innerHTML = emptyState(
          who ? `No blown calls for ${who} with these filters.` : 'No blown calls match these filters.',
          bits.length ? `Try clearing ${bits.join(' or ')} &mdash; most players only have a handful of opportunities on any single day.`
            : 'Try switching to "Every decision", or use Clear to start over.');
        p.querySelector('.abs-tblwrap').style.display = 'none';
      } else { box.innerHTML = ''; p.querySelector('.abs-tblwrap').style.display = ''; }
      p.querySelectorAll('#ftbl th').forEach(t => t.addEventListener('click', () => {
        const k = t.dataset.k; if (filmSort === k) filmDir *= -1; else { filmSort = k; filmDir = -1; } draw();
      }));
      p.querySelectorAll('#ftbl tr.row-click').forEach(tr => tr.addEventListener('click', e => {
        if (e.target.closest('a')) return;          // let the video link do its job
        const x = r[+tr.dataset.i]; if (!x || !x._raw) return;
        pendingPitch = x._raw;
        state.view = 'matrix';
        document.querySelectorAll('.abs-tab').forEach(t2 => {
          const on = t2.getAttribute('data-absview') === 'matrix';
          t2.classList.toggle('active', on); t2.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        render();
      }));
      p.querySelector('#fCsv').onclick = () => downloadCSV('abs_opportunities_2026.csv', cols, r);
      writeUrl();
    }
    const bind = (id, key) => p.querySelector(id).addEventListener('input', e => { film[key] = e.target.value; draw(); });
    bind('#fDate', 'date'); bind('#fBat', 'bat'); bind('#fCat', 'cat');
    bind('#fTeam', 'team'); bind('#fType', 'type'); bind('#fWrong', 'wrong');
    bind('#fCount', 'count'); bind('#fInning', 'inning');
    p.querySelector('#fClear').addEventListener('click', () => {
      film.date = ''; film.bat = ''; film.cat = ''; film.team = ''; film.type = '';
      film.count = ''; film.inning = ''; film.wrong = '';
      ['#fDate', '#fBat', '#fCat', '#fTeam', '#fType', '#fCount', '#fInning'].forEach(s => p.querySelector(s).value = '');
      p.querySelector('#fWrong').value = ''; draw();
    });
    draw();
  }

  // ================= ZONE MAP =================
  const HOWTO_ZONE = `<details class="howto"><summary>How to read this</summary><dl>
    <dt>What the grid shows</dt><dd>Every near-zone taken pitch of the season, binned by where it crossed. Each cell is the share of those calls the human umpire got wrong compared to the ABS zone.</dd>
    <dt>Why the height is normalized</dt><dd>The ABS zone is set to each batter's height, so a fixed grid would smear tall and short hitters together. Here 0 is every batter's own bottom edge and 1 is their own top edge, which makes the edges line up.</dd>
    <dt>The two error types</dt><dd><b>Rung up</b> = a true ball called a strike (the hitter's grievance). <b>Stolen strike</b> = a true strike called a ball (the catcher's grievance).</dd>
  </dl></details>`;

  async function renderZone() {
    const p = el();
    p.innerHTML = viewTabs() + `<h2>Where umpires miss</h2><p class="abs-sub">Loading zone map...</p>`;
    bindViewTabs();
    if (!zoneMiss) { try { zoneMiss = await fetch('data/abs_zone_misses_2026.json').then(r => r.json()); } catch (e) { zoneMiss = { cells: {}, meta: {} }; } }
    if (state.view !== 'zone') return;
    const M = zoneMiss.meta, C = zoneMiss.cells;
    let mode = 'all';   // all | strike (rung up) | ball (stolen strike)
    const xs = [], zs = [];
    Object.keys(C).forEach(k => { const [a, b] = k.split('|').map(Number); xs.push(a); zs.push(b); });
    const xMin = Math.min(...xs), xMax = Math.max(...xs), zMin = Math.min(...zs), zMax = Math.max(...zs);
    const nx = xMax - xMin + 1, nz = zMax - zMin + 1;
    const CELL = 30, PAD = 46;
    const W = nx * CELL + PAD + 16, H = nz * CELL + PAD + 20;
    p.innerHTML = viewTabs() +
      `<h2>Where umpires miss</h2>` + HOWTO_ZONE +
      `<p class="abs-sub">Every near-zone take this season, binned by location. Darker = the human call was wrong more often. The solid box is the ABS strike zone; height is normalized to each batter's own zone so the edges line up.</p>
       <div class="abs-bar">
         <span class="seg" id="zMode">
           <button data-v="all" aria-pressed="true">All misses</button>
           <button data-v="strike">Rung up</button>
           <button data-v="ball">Stolen strikes</button></span>
         <span class="abs-count" id="zcnt"></span>
       </div>
       <div class="heatwrap"><svg id="zsvg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="max-width:100%;height:auto"></svg></div>
       <div class="heatlegend"><span>0%</span><div class="mx-grad" id="zgrad" style="height:9px;border-radius:5px;flex:1;max-width:300px"></div><span id="zmax">max</span><span style="margin-left:auto">catcher's view &middot; x in inches from plate center</span></div>`;
    bindViewTabs();
    const ramp = t => {
      const R = [[240, 228, 216], [236, 206, 168], [214, 143, 96], [166, 59, 34], [120, 34, 22]];
      const x = Math.max(0, Math.min(1, t)) * (R.length - 1), i = Math.min(Math.floor(x), R.length - 2), f = x - i;
      const c = R[i].map((v, j) => Math.round(v + (R[i + 1][j] - v) * f));
      return `rgb(${c.join(',')})`;
    };
    function draw() {
      const rate = c => {
        const n = c[0]; if (!n) return null;
        const w = mode === 'strike' ? c[1] : mode === 'ball' ? c[2] : (c[1] + c[2]);
        return w / n;
      };
      let peak = 0, shown = 0;
      Object.values(C).forEach(c => { if (c[0] >= 20) { const r = rate(c); if (r != null && r > peak) peak = r; } });
      peak = Math.max(peak, 0.05);
      let h = '';
      for (let zi = zMin; zi <= zMax; zi++) {
        for (let xi = xMin; xi <= xMax; xi++) {
          const c = C[xi + '|' + zi]; if (!c || c[0] < 20) continue;
          const r = rate(c); if (r == null) continue;
          shown += c[0];
          const px = PAD + (xi - xMin) * CELL, py = 8 + (zMax - zi) * CELL;
          h += `<rect x="${px}" y="${py}" width="${CELL - 1}" height="${CELL - 1}" fill="${ramp(r / peak)}" rx="2"><title>${(100 * r).toFixed(0)}% wrong (${c[0]} takes)</title></rect>`;
        }
      }
      // ABS zone outline: x = +-8.5in, z = 0..1 normalized
      const zx0 = PAD + ((-8.5 - M.x0) / M.xStep - xMin) * CELL;
      const zx1 = PAD + ((8.5 - M.x0) / M.xStep - xMin) * CELL;
      const zyTop = 8 + (zMax - ((1 - M.z0) / M.zStep) + 1) * CELL;
      const zyBot = 8 + (zMax - ((0 - M.z0) / M.zStep) + 1) * CELL;
      h += `<rect x="${zx0}" y="${zyTop}" width="${zx1 - zx0}" height="${zyBot - zyTop}" fill="none" stroke="var(--text-primary)" stroke-width="2" rx="2"></rect>`;
      h += `<text x="${PAD - 8}" y="${zyTop + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">top</text>`;
      h += `<text x="${PAD - 8}" y="${zyBot + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">bottom</text>`;
      for (const inch of [-12, -8, -4, 0, 4, 8, 12]) {
        const gx = PAD + ((inch - M.x0) / M.xStep - xMin) * CELL + CELL / 2;
        h += `<text x="${gx}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${inch}</text>`;
      }
      p.querySelector('#zsvg').innerHTML = h;
      p.querySelector('#zcnt').textContent = shown.toLocaleString() + ' takes';
      p.querySelector('#zmax').textContent = Math.round(peak * 100) + '%';
      const stops = []; for (let i = 0; i <= 10; i++) stops.push(ramp(i / 10) + ' ' + (i * 10) + '%');
      p.querySelector('#zgrad').style.background = 'linear-gradient(90deg,' + stops.join(',') + ')';
    }
    p.querySelectorAll('#zMode button').forEach(b => b.addEventListener('click', () => {
      p.querySelectorAll('#zMode button').forEach(x => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true'); mode = b.dataset.v; draw();
    }));
    draw();
  }

  async function render() {
    injectStyles();
    const p = el(); if (!p) return;
    if (!core) { p.innerHTML = `<p class="abs-sub" style="padding:20px">Loading ABS data...</p>`; try { await ensureCore(); } catch (e) { p.innerHTML = `<p class="abs-sub" style="padding:20px">Could not load ABS data.</p>`; return; } }
    if (state.view === 'leaders') renderLeaders();
    else if (state.view === 'matrix') renderMatrix();
    else if (state.view === 'zone') renderZone();
    else renderFilm();
  }

  return {
    render: render,
    setView: function (v, query) { state.view = v; applyUrl(query); }
  };
})();
