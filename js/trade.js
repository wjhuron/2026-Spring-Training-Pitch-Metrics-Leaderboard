// Trade calculator page. Data: window.TRADE_VALUES (js/tradevalue_data.js).
// Verdict bands come from the market-layer fit residuals: within 1.5x =
// balanced by market standards; within e^0.9 (~2.4x, the median real-trade
// miss) = inside historical trade noise; beyond that = lopsided.
(function () {
  'use strict';

  var DATA = (window.TRADE_VALUES && window.TRADE_VALUES.players) || [];
  var BAND_FAIR = Math.log(1.5);
  var BAND_NOISE = 0.9;
  var VALUE_FLOOR_M = 0.5; // avoids log blowups on near-empty sides

  var sides = { a: [], b: [] };

  function fmtM(v) {
    var neg = v < 0;
    var s = '$' + Math.abs(v).toFixed(1) + 'M';
    return neg ? '-' + s : s;
  }

  function meta(p) {
    if (p.e === 'prospect') {
      var m = 'FV ' + p.fv + ' · ' + p.t + ' · ' + p.p;
      if (p.eta) m += ' · ETA ' + p.eta;
      return m;
    }
    return p.t + ' · ' + p.p + ' · ' + p.w + ' WAR · ' +
           p.c + (p.c === 1 ? ' yr' : ' yrs') + ' control';
  }

  function norm(s) {
    return s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  }

  function search(q) {
    q = norm(q.trim());
    if (q.length < 2) return [];
    var out = [];
    for (var i = 0; i < DATA.length; i++) {
      if (norm(DATA[i].n).indexOf(q) !== -1) out.push(DATA[i]);
      if (out.length > 200) break;
    }
    out.sort(function (x, y) { return y.s - x.s; });
    return out.slice(0, 12);
  }

  function totals(list) {
    var s = 0, m = 0;
    list.forEach(function (p) { s += p.s; m += p.m; });
    return { s: s, m: m };
  }

  function render() {
    ['a', 'b'].forEach(function (k) {
      var panel = document.getElementById('panel-' + k);
      var ul = panel.querySelector('.trade-list');
      ul.innerHTML = '';
      sides[k].forEach(function (p, idx) {
        var li = document.createElement('li');
        var val = document.createElement('span');
        val.className = 'pl-val' + (p.m < 0 ? ' neg' : '');
        val.textContent = fmtM(p.m);
        val.title = 'Intrinsic ' + fmtM(p.s);
        var rm = document.createElement('button');
        rm.textContent = '×';
        rm.setAttribute('aria-label', 'Remove ' + p.n);
        rm.addEventListener('click', function () {
          sides[k].splice(idx, 1);
          render();
        });
        var nm = document.createElement('span');
        nm.className = 'pl-name';
        nm.textContent = p.n;
        var mt = document.createElement('span');
        mt.className = 'pl-meta';
        mt.textContent = meta(p);
        li.appendChild(nm); li.appendChild(mt); li.appendChild(val); li.appendChild(rm);
        ul.appendChild(li);
      });
      var t = totals(sides[k]);
      panel.querySelector('.tot-intrinsic').textContent = fmtM(t.s);
      panel.querySelector('.tot-market').textContent = fmtM(t.m);
    });
    verdict();
  }

  function verdict() {
    var el = document.getElementById('verdict');
    if (!sides.a.length || !sides.b.length) { el.style.display = 'none'; return; }
    var ta = totals(sides.a), tb = totals(sides.b);
    var va = Math.max(ta.m, VALUE_FLOOR_M), vb = Math.max(tb.m, VALUE_FLOOR_M);
    var lr = Math.log(va / vb);
    var winner = lr > 0 ? 'Side A' : 'Side B';
    var ratio = Math.exp(Math.abs(lr));
    var cls, head, body;
    if (Math.abs(lr) <= BAND_FAIR) {
      cls = 'verdict-fair';
      head = 'Balanced by market standards';
      body = 'The market values are within 1.5x of each other, the range most ' +
             'real trades clear.';
    } else if (Math.abs(lr) <= BAND_NOISE) {
      cls = 'verdict-noise';
      head = 'Favors ' + winner + ', within historical trade noise';
      body = winner + ' receives about ' + ratio.toFixed(1) + 'x the market value. ' +
             'Real 2017–2026 trades miss balance by this much routinely.';
    } else {
      cls = 'verdict-lopsided';
      head = 'Strongly favors ' + winner;
      body = winner + ' receives about ' + ratio.toFixed(1) + 'x the market value, ' +
             'outside the spread of nearly all real trades.';
    }
    el.className = 'trade-verdict ' + cls;
    el.innerHTML = '';
    var h = document.createElement('h4'); h.textContent = head;
    var p1 = document.createElement('p'); p1.textContent = body;
    var p2 = document.createElement('p');
    p2.textContent = 'Market: ' + fmtM(ta.m) + ' vs ' + fmtM(tb.m) +
                     ' · Intrinsic: ' + fmtM(ta.s) + ' vs ' + fmtM(tb.s);
    el.appendChild(h); el.appendChild(p1); el.appendChild(p2);
    el.style.display = '';
  }

  function wireSearch(k) {
    var panel = document.getElementById('panel-' + k);
    var input = panel.querySelector('input');
    var box = panel.querySelector('.trade-suggest');
    var results = [], sel = -1;

    function close() { box.style.display = 'none'; box.innerHTML = ''; sel = -1; }

    function pick(p) {
      sides[k].push(p);
      input.value = '';
      close();
      render();
      input.focus();
    }

    function draw() {
      box.innerHTML = '';
      if (!results.length) { close(); return; }
      results.forEach(function (p, i) {
        var d = document.createElement('div');
        if (i === sel) d.className = 'sel';
        var nm = document.createElement('span');
        nm.textContent = p.n;
        var mt = document.createElement('span');
        mt.className = 'sug-meta';
        mt.textContent = (p.e === 'prospect' ? 'FV ' + p.fv + ' · ' : '') +
                         p.t + ' · ' + fmtM(p.m);
        d.appendChild(nm); d.appendChild(mt);
        d.addEventListener('mousedown', function (ev) { ev.preventDefault(); pick(p); });
        box.appendChild(d);
      });
      box.style.display = '';
    }

    input.addEventListener('input', function () {
      results = search(input.value);
      sel = -1;
      draw();
    });
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'ArrowDown') { sel = Math.min(sel + 1, results.length - 1); draw(); ev.preventDefault(); }
      else if (ev.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); draw(); ev.preventDefault(); }
      else if (ev.key === 'Enter' && results.length) { pick(results[sel < 0 ? 0 : sel]); ev.preventDefault(); }
      else if (ev.key === 'Escape') { close(); }
    });
    input.addEventListener('blur', function () { setTimeout(close, 150); });
  }

  wireSearch('a');
  wireSearch('b');

  // ---- Player Values browse table ----
  var MAX_ROWS_ALL = 300;
  var valuesSort = 's';
  var teamSel = document.getElementById('values-team');
  var valuesSearch = document.getElementById('values-search');
  var valuesBody = document.getElementById('values-body');
  var valuesNote = document.getElementById('values-note');

  function initTeams() {
    var teams = {};
    DATA.forEach(function (p) { teams[p.t] = true; });
    Object.keys(teams).sort().forEach(function (t) {
      var o = document.createElement('option');
      o.value = t; o.textContent = t;
      teamSel.appendChild(o);
    });
  }

  function renderValues() {
    var team = teamSel.value;
    var q = norm(valuesSearch.value.trim());
    var rows = DATA.filter(function (p) {
      if (team !== 'all' && p.t !== team) return false;
      if (q && norm(p.n).indexOf(q) === -1) return false;
      return true;
    });
    rows.sort(function (x, y) { return y[valuesSort] - x[valuesSort]; });
    var capped = team === 'all' && !q && rows.length > MAX_ROWS_ALL;
    var shown = capped ? rows.slice(0, MAX_ROWS_ALL) : rows;

    valuesBody.innerHTML = '';
    var frag = document.createDocumentFragment();
    shown.forEach(function (p, i) {
      var tr = document.createElement('tr');
      function td(cls, text) {
        var c = document.createElement('td');
        if (cls) c.className = cls;
        c.textContent = text;
        tr.appendChild(c);
        return c;
      }
      td('td-rank', String(i + 1));
      td('', p.n).style.fontWeight = '600';
      td('', p.t);
      td('', p.p);
      td('td-profile', p.e === 'prospect'
        ? 'FV ' + p.fv + (p.eta ? ' · ETA ' + p.eta : '')
        : p.w + ' WAR · ' + p.c + (p.c === 1 ? ' yr' : ' yrs'));
      td('td-num' + (p.s < 0 ? ' neg' : ''), fmtM(p.s));
      td('td-num' + (p.m < 0 ? ' neg' : ''), fmtM(p.m));
      var actions = document.createElement('td');
      ['a', 'b'].forEach(function (k) {
        var b = document.createElement('button');
        b.className = 'add-btn';
        b.textContent = k.toUpperCase();
        b.title = 'Add to Side ' + k.toUpperCase();
        b.addEventListener('click', function () {
          sides[k].push(p);
          render();
        });
        actions.appendChild(b);
      });
      tr.appendChild(actions);
      frag.appendChild(tr);
    });
    valuesBody.appendChild(frag);
    valuesNote.textContent = capped
      ? 'Showing top ' + MAX_ROWS_ALL + ' of ' + rows.length +
        ' players. Pick a team or search to see everyone.'
      : rows.length + ' players.';
  }

  document.querySelectorAll('.th-sort').forEach(function (th) {
    th.addEventListener('click', function () {
      valuesSort = th.getAttribute('data-sort');
      document.querySelectorAll('.th-sort').forEach(function (o) {
        o.classList.toggle('active', o === th);
      });
      renderValues();
    });
  });
  document.querySelector('.th-sort[data-sort="s"]').classList.add('active');
  teamSel.addEventListener('change', renderValues);
  valuesSearch.addEventListener('input', renderValues);
  initTeams();
  renderValues();

  var noteEl = document.getElementById('data-note');
  if (noteEl && window.TRADE_VALUES) {
    noteEl.textContent = window.TRADE_VALUES.note +
      ' Values generated ' + window.TRADE_VALUES.generated + '.';
  }
})();
