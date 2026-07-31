// Outfield catch probability calculator.
// Lookup against data/catch_prob_surface.json: official per-play catch_rate
// (Savant player-services/range) aggregated to cells of
// (opportunity time tenths, distance ft, back, wall), 2024-2026.
// Mirrors scripts/catch_prob.py.

(function () {
  'use strict';

  var K_MIN = 5;
  var MAX_RING = 6;
  var FLIGHT = 0.39;

  // published outfield wall distances [LF line, LF gap, CF, RF gap, RF line];
  // approximate where parks have irregular walls, may lag renovations
  var PARKS = {
    'ARI Chase Field':            [330, 374, 407, 374, 334],
    'ATH Sutter Health Park':     [330, 375, 403, 375, 325],
    'ATL Truist Park':            [335, 385, 400, 375, 325],
    'BAL Camden Yards':           [332, 373, 400, 373, 318],
    'BOS Fenway Park':            [310, 379, 390, 380, 302],
    'CHC Wrigley Field':          [355, 368, 400, 368, 353],
    'CIN Great American':         [328, 379, 404, 370, 325],
    'CLE Progressive Field':      [325, 370, 400, 375, 325],
    'COL Coors Field':            [347, 390, 415, 375, 350],
    'CWS Rate Field':             [330, 375, 400, 375, 335],
    'DET Comerica Park':          [345, 370, 412, 365, 330],
    'HOU Daikin Park':            [315, 362, 409, 373, 326],
    'KC Kauffman Stadium':        [330, 387, 410, 387, 330],
    'LAA Angel Stadium':          [330, 387, 396, 370, 348],
    'LAD Dodger Stadium':         [330, 385, 395, 385, 330],
    'MIA loanDepot park':         [344, 386, 400, 387, 335],
    'MIL American Family Field':  [342, 371, 400, 374, 345],
    'MIN Target Field':           [339, 377, 404, 367, 328],
    'NYM Citi Field':             [335, 358, 405, 375, 330],
    'NYY Yankee Stadium':         [318, 399, 408, 385, 314],
    'PHI Citizens Bank Park':     [329, 374, 401, 369, 330],
    'PIT PNC Park':               [325, 389, 399, 375, 320],
    'SD Petco Park':              [336, 390, 396, 391, 322],
    'SEA T-Mobile Park':          [331, 378, 401, 381, 326],
    'SF Oracle Park':             [339, 364, 391, 415, 309],
    'STL Busch Stadium':          [336, 375, 400, 375, 335],
    'TB Tropicana Field':         [315, 370, 404, 370, 322],
    'TEX Globe Life Field':       [329, 372, 407, 374, 326],
    'TOR Rogers Centre':          [328, 368, 400, 359, 328],
    'WSH Nationals Park':         [336, 377, 402, 370, 335]
  };

  var surface = null;
  var meta = null;

  fetch('data/catch_prob_surface.json?v=20260731b')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      meta = data.meta;
      surface = {};
      Object.keys(data.cells).forEach(function (k) {
        var parts = k.split('|'); // t|d|b|w
        var fk = parts[2] + '|' + parts[3];
        if (!surface[fk]) surface[fk] = {};
        surface[fk][parts[0] + '|' + parts[1]] =
          [data.cells[k].p, data.cells[k].n,
           data.cells[k].l, data.cells[k].h, data.cells[k].hist];
      });
      update();
    })
    .catch(function () {
      showError('Could not load the catch probability surface.');
    });

  (function () {
    var sel = document.getElementById('cp-park');
    Object.keys(PARKS).forEach(function (k) {
      var o = document.createElement('option');
      o.value = k; o.textContent = k;
      sel.appendChild(o);
    });
  })();

  function lookup(t, dist, back, wall) {
    var grid = surface[back + '|' + wall];
    if (!grid) return null;
    var tt = Math.round(t * 10);
    var num = 0, den = 0, ring;
    for (ring = 1; ring <= MAX_RING; ring++) {
      var tw = Math.round(0.05 * ring * 100) / 100;
      num = 0; den = 0;
      for (var ti = tt - 3; ti <= tt + 3; ti++) {
        if (Math.abs(ti / 10 - t) > tw + 1e-9) continue;
        for (var di = dist - ring; di <= dist + ring; di++) {
          var c = grid[(ti / 10).toFixed(1) + '|' + di];
          if (c) { num += c[0] * c[1]; den += c[1]; }
        }
      }
      if (den >= K_MIN) return { p: num / den, n: den, ring: ring };
    }
    return den > 0 ? { p: num / den, n: den, ring: MAX_RING } : null;
  }

  // expected within-bucket spread from the local gradient (cell medians)
  function gradientSpread(t, dist, back, wall) {
    var grid = surface[back + '|' + wall] || {};
    var tr = (Math.round(t * 10) / 10);
    function med(tt, dd) {
      var c = grid[tt.toFixed(1) + '|' + dd];
      return c ? c[0] : null;
    }
    var c0 = med(tr, dist);
    var tp = med(Math.round((tr + 0.1) * 10) / 10, dist);
    var tm = med(Math.round((tr - 0.1) * 10) / 10, dist);
    var dp = med(tr, dist + 1), dm = med(tr, dist - 1);
    var st = (tp !== null && tm !== null) ? Math.abs(tp - tm) / 2
           : (tp !== null && c0 !== null) ? Math.abs(tp - c0)
           : (tm !== null && c0 !== null) ? Math.abs(c0 - tm) : 0.05;
    var sd = (dp !== null && dm !== null) ? Math.abs(dp - dm) / 2
           : (dp !== null && c0 !== null) ? Math.abs(dp - c0)
           : (dm !== null && c0 !== null) ? Math.abs(c0 - dm) : 0.02;
    return { st: st, sd: sd };
  }

  // boundaries verified against 95k official star ratings
  function starRange(p) {
    if (p <= 0.25) return '5-star range';
    if (p <= 0.50) return '4-star range';
    if (p <= 0.75) return '3-star range';
    if (p <= 0.90) return '2-star range';
    if (p <= 0.95) return '1-star range';
    return 'routine';
  }

  function showError(msg) {
    var e = document.getElementById('cp-error');
    e.textContent = msg;
    e.style.display = msg ? 'block' : 'none';
    if (msg) document.getElementById('cp-result').style.display = 'none';
  }

  // Point estimate + two-tier band for one wall scenario. Validated per
  // input scenario on 6,000 held-out plays each: outer range 99.5-99.8%
  // containment (worst miss 0.03-0.08), inner likely (pooled q10-q90)
  // 96.2%. Sparse-data penalty 0.10/sqrt(n) on the outer range (swept).
  function evaluate(time, tA, tB, dist, back, wall) {
    var out;
    var tTenth = Math.round(time * 10) / 10;
    if (Math.abs(time - tTenth) < 0.005) {
      out = lookup(tTenth, dist, back, wall);
    } else {
      var tLo = Math.floor(time * 10) / 10;
      var tHi = Math.round((tLo + 0.1) * 10) / 10;
      var o1 = lookup(tLo, dist, back, wall);
      var o2 = lookup(tHi, dist, back, wall);
      if (o1 && o2) {
        var w = (time - tLo) / 0.1;
        out = { p: (1 - w) * o1.p + w * o2.p, n: o1.n + o2.n,
                ring: Math.max(o1.ring, o2.ring) };
      } else {
        out = o1 || o2;
      }
    }
    if (!out) return null;

    var grid = surface[back + '|' + wall] || {};
    var g = gradientSpread(time, dist, back, wall);
    var U = Math.max(tB - tA, 0.02);
    var S = g.st * (U / 0.1) + g.sd;
    var bLo = out.p - 1.5 * S / 2 - 0.025;
    var bHi = out.p + 1.5 * S / 2 + 0.025;
    var tt0 = Math.min(Math.round(tA * 10), Math.round(time * 10) - out.ring + 1);
    var tt1 = Math.max(Math.round(tB * 10), Math.round(time * 10) + out.ring - 1);
    var dwv = Math.max(1, out.ring);
    for (var ti2 = tt0; ti2 <= tt1; ti2++) {
      for (var di2 = dist - dwv; di2 <= dist + dwv; di2++) {
        var c2 = grid[(ti2 / 10).toFixed(1) + '|' + di2];
        if (c2) {
          bLo = Math.min(bLo, c2[2] - 0.025);
          bHi = Math.max(bHi, c2[3] + 0.025);
        }
      }
    }
    var pool = {}, tot = 0;
    for (var ti3 = Math.round(tA * 10); ti3 <= Math.round(tB * 10); ti3++) {
      for (var di3 = dist - 1; di3 <= dist + 1; di3++) {
        var c3 = grid[(ti3 / 10).toFixed(1) + '|' + di3];
        if (c3 && c3[4]) {
          for (var v3 in c3[4]) { pool[v3] = (pool[v3] || 0) + c3[4][v3]; tot += c3[4][v3]; }
        }
      }
    }
    var pen = 0.10 / Math.sqrt(Math.max(tot, 1));
    bLo = Math.max(bLo - pen, 0);
    bHi = Math.min(bHi + pen, 1);
    var loPct = Math.max(Math.floor(bLo * 100), 0);
    var hiPct = Math.min(Math.ceil(bHi * 100), 99);
    var likely = null;
    if (tot >= 8) {
      var keys = Object.keys(pool).map(Number).sort(function (a, b) { return a - b; });
      var pq = function (pr) {
        var target = pr * (tot - 1), acc = 0;
        for (var i3 = 0; i3 < keys.length; i3++) {
          acc += pool[keys[i3]];
          if (acc - 1 >= target) return keys[i3];
        }
        return keys[keys.length - 1];
      };
      likely = [Math.max(pq(0.10), loPct), Math.min(pq(0.90), hiPct)];
    }
    return { pPct: Math.round(out.p * 100), n: out.n,
             lo: loPct, hi: hiPct, likely: likely, p: out.p };
  }

  function fmt(r) {
    return r.pPct + '%' +
      (r.likely ? ', likely ' + r.likely[0] + '–' + r.likely[1] : '') +
      ', range ' + r.lo + '–' + r.hi;
  }

  function update() {
    if (!surface) return;
    showError('');
    var dist = parseFloat(document.getElementById('cp-dist').value);
    var time = parseFloat(document.getElementById('cp-time').value);
    var hang = parseFloat(document.getElementById('cp-hang').value);
    var plate = parseFloat(document.getElementById('cp-plate').value);
    var angle = parseFloat(document.getElementById('cp-angle').value);
    var res = document.getElementById('cp-result');

    // back = angle within 30 degrees of straight behind, ONLY (boundary
    // verified consistent with official flags, bracketed [27, 42] deg)
    var back = (!isNaN(angle) && Math.abs(angle) >= 150) ? '1' : '0';

    // pitch flight: plate time from the card is exact, else the
    // typical-fastball default
    var flight = !isNaN(plate) && plate > 0 ? plate : FLIGHT;

    // Card conventions (verified per field): opportunity time TRUNCATES
    // (T -> [T, T+0.1)), hang time ROUNDS (H -> [H-0.05, H+0.05]), plate
    // time rounds (negligible). Values with 2+ decimals are exact.
    var decs = function (id) {
      var v = document.getElementById(id).value;
      return (v.split('.')[1] || '').length;
    };
    var combined = null, tA = NaN, tB = NaN;
    if (!isNaN(time) && decs('cp-time') >= 2) {
      tA = tB = time;  // exact time: use as-is
    } else if (isNaN(time) && !isNaN(hang)) {
      if (decs('cp-hang') >= 2) { tA = tB = hang + flight; }
      else { tA = hang - 0.05 + flight; tB = hang + 0.05 + flight; }
      time = (tA + tB) / 2;
    } else if (!isNaN(time) && !isNaN(hang)) {
      var lo = Math.max(time, hang - 0.05 + flight);
      var hi = Math.min(time + 0.1, hang + 0.05 + flight);
      if (lo <= hi) { tA = lo; tB = hi; time = (lo + hi) / 2; combined = time; }
      else { tA = time; tB = time + 0.1; time = time + 0.05; }
    } else if (!isNaN(time)) {
      tA = time; tB = time + 0.1;
      time = time + 0.05;
    }
    if (isNaN(dist) || isNaN(time)) { res.style.display = 'none'; return; }

    // Wall scenarios. The flag is NOT reliably determinable from distance
    // alone (even balls AT the marker are only ~83% official wall plays).
    // Policy validated on two weeks of catches: gap > 25 ft -> confident
    // no wall (official wall rate 0.2-4%); gap <= 25 ft -> ambiguous,
    // BOTH scenarios shown so the correct answer is always on screen.
    var wallSel = document.getElementById('cp-wall').value;
    var scenarios = ['0'], wallNote = '';
    if (wallSel === 'auto') {
      var park = document.getElementById('cp-park').value;
      var hitd = parseFloat(document.getElementById('cp-hitdist').value);
      if (park && !isNaN(hitd)) {
        // park's shortest wall, threshold 20: zero leaked wall plays in
        // 2,809-play validation (sector- and position-based variants
        // either leak or classify less)
        var wd = Math.min.apply(null, PARKS[park]);
        var short = Math.round(wd - hitd);
        if (short > 20) {
          scenarios = ['0'];
          wallNote = short + ' ft short of the shortest wall (' + wd + ' ft): no wall';
        } else {
          scenarios = ['0', '1'];
          wallNote = (short < 0 ? 'ball at/over the shortest wall (' + wd + ' ft)'
                     : short + ' ft short of the shortest wall (' + wd + ' ft)') +
                     ': wall status ambiguous, both shown; judge from the play';
        }
      } else {
        wallNote = 'no wall assumed (fill park inputs or override)';
      }
    } else {
      scenarios = [wallSel];
    }

    var d = Math.round(dist);
    var r0 = evaluate(time, tA, tB, d, back, scenarios[0]);
    var r1 = scenarios.length === 2 ? evaluate(time, tA, tB, d, back, scenarios[1]) : null;
    if (!r0 && !r1) {
      showError('No comparable tracked plays for those inputs.');
      return;
    }
    res.style.display = 'block';
    if (r0 && r1) {
      document.getElementById('cp-big').textContent =
        r0.pPct + '% / ' + r1.pPct + '%';
      document.getElementById('cp-stars').textContent =
        'no wall: ' + fmt(r0) + ' · if wall: ' + fmt(r1);
    } else {
      var r = r0 || r1;
      document.getElementById('cp-big').textContent = r.pPct + '%';
      document.getElementById('cp-stars').textContent =
        (r.likely ? 'likely ' + r.likely[0] + '–' + r.likely[1] + ' · ' : '') +
        'range ' + r.lo + '–' + r.hi + ' · ' + starRange(r.p);
    }
    document.getElementById('cp-detail').textContent =
      (combined ? 'combined opportunity estimate ' + combined.toFixed(3) + 's · ' : '') +
      'flags: ' + (back === '1' ? 'going back' : 'not back') +
      (r1 ? ', wall ambiguous' : (scenarios[0] === '1' ? ', wall factor' : ', no wall')) +
      (wallNote ? ' (' + wallNote + ')' : '') + ' · ' +
      (r0 || r1).n + ' comparable plays · ' +
      meta.seasons + ', ' + meta.plays.toLocaleString() + ' tracked plays';
  }

  ['cp-dist', 'cp-time', 'cp-hang', 'cp-plate', 'cp-wall', 'cp-angle',
   'cp-park', 'cp-hitdist'].forEach(function (id) {
    document.getElementById(id).addEventListener('input', update);
  });

  document.getElementById('cp-reset').addEventListener('click', function () {
    ['cp-dist', 'cp-time', 'cp-hang', 'cp-plate', 'cp-angle', 'cp-hitdist'].forEach(function (id) {
      document.getElementById(id).value = '';
    });
    document.getElementById('cp-park').value = '';
    document.getElementById('cp-wall').value = 'auto';
    document.getElementById('cp-result').style.display = 'none';
    showError('');
  });
})();
