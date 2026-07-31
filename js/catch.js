// Outfield catch probability calculator.
// Lookup against data/catch_prob_surface.json: official per-play catch_rate
// (Savant player-services/range) aggregated to cells of
// (opportunity time tenths, distance ft, back, wall), 2024-2026.
// Mirrors scripts/catch_prob.py: expanding window of +-0.05s/+-1ft per ring,
// stop once >= K_MIN plays (K flat 1-10 in LOO validation; 5 by convention).

(function () {
  'use strict';

  var K_MIN = 5;
  var MAX_RING = 6;
  var FLIGHT = 0.39;

  var surface = null;   // {"b|w": Map("t|d" -> [p, n])}
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

  function lookup(t, dist, back, wall) {
    var grid = surface[back + '|' + wall];
    if (!grid) return null;
    var tt = Math.round(t * 10);
    var num = 0, den = 0, ring, mn = null, mx = null;
    for (ring = 1; ring <= MAX_RING; ring++) {
      var tw = Math.round(0.05 * ring * 100) / 100;
      num = 0; den = 0; mn = null; mx = null;
      for (var ti = tt - 3; ti <= tt + 3; ti++) {
        if (Math.abs(ti / 10 - t) > tw + 1e-9) continue;
        for (var di = dist - ring; di <= dist + ring; di++) {
          var c = grid[(ti / 10).toFixed(1) + '|' + di];
          if (c) {
            num += c[0] * c[1]; den += c[1];
            mn = mn === null ? c[2] : Math.min(mn, c[2]);
            mx = mx === null ? c[3] : Math.max(mx, c[3]);
          }
        }
      }
      if (den >= K_MIN) return { p: num / den, n: den, ring: ring, mn: mn, mx: mx };
    }
    return den > 0 ? { p: num / den, n: den, ring: MAX_RING, mn: mn, mx: mx } : null;
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

  function starRange(p) {
    if (p <= 0.25) return '5-star range';
    if (p <= 0.50) return '4-star range';
    if (p <= 0.75) return '3-star range';
    if (p <= 0.90) return '2-star range';
    if (p <= 0.95) return '1-star range';
    return 'routine';
  }

  function bucket(p) {
    var b = Math.round(p / 0.05) * 0.05;
    return b >= 0.975 ? 0.99 : Math.max(b, 0);
  }

  function showError(msg) {
    var e = document.getElementById('cp-error');
    e.textContent = msg;
    e.style.display = msg ? 'block' : 'none';
    if (msg) document.getElementById('cp-result').style.display = 'none';
  }

  function update() {
    if (!surface) return;
    showError('');
    var dist = parseFloat(document.getElementById('cp-dist').value);
    var time = parseFloat(document.getElementById('cp-time').value);
    var hang = parseFloat(document.getElementById('cp-hang').value);
    var plate = parseFloat(document.getElementById('cp-plate').value);
    // wall is an explicit input: the card's Fielding Zone is a positioning
    // label and carries no wall information
    var wall = document.getElementById('cp-wall').value;
    var angle = parseFloat(document.getElementById('cp-angle').value);
    // back = angle within 30 degrees of straight behind, ONLY
    var back = (!isNaN(angle) && Math.abs(angle) >= 150) ? '1' : '0';
    var res = document.getElementById('cp-result');

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

    var out;
    var tTenth = Math.round(time * 10) / 10;
    if (Math.abs(time - tTenth) < 0.005) {
      out = lookup(tTenth, Math.round(dist), back, wall);
    } else {
      // off-grid time from a combined estimate: interpolate the two
      // adjacent tenth-of-a-second surfaces
      var tLo = Math.floor(time * 10) / 10;
      var tHi = Math.round((tLo + 0.1) * 10) / 10;
      var o1 = lookup(tLo, Math.round(dist), back, wall);
      var o2 = lookup(tHi, Math.round(dist), back, wall);
      if (o1 && o2) {
        var w = (time - tLo) / 0.1;
        out = { p: (1 - w) * o1.p + w * o2.p, n: o1.n + o2.n,
                ring: Math.max(o1.ring, o2.ring),
                mn: Math.min(o1.mn, o2.mn), mx: Math.max(o1.mx, o2.mx) };
      } else {
        out = o1 || o2;
      }
    }
    if (!out) {
      showError('No comparable tracked plays for those inputs.');
      return;
    }
    // Uncertainty band: gradient component scaled by the feasible time
    // window, UNION observed catch rates over that window (bucket-correct)
    // and the lookup ring, 0.025 pad. Validated on 6,000 held-out plays
    // per input scenario: containment 99.5-99.8%, worst miss 0.03-0.08.
    var g = gradientSpread(time, Math.round(dist), back, wall);
    var U = Math.max(tB - tA, 0.02);
    var S = g.st * (U / 0.1) + g.sd;
    var bLo = out.p - 1.5 * S / 2 - 0.025;
    var bHi = out.p + 1.5 * S / 2 + 0.025;
    var grid = surface[back + '|' + wall] || {};
    var tt0 = Math.min(Math.round(tA * 10), Math.round(time * 10) - out.ring + 1);
    var tt1 = Math.max(Math.round(tB * 10), Math.round(time * 10) + out.ring - 1);
    var dwv = Math.max(1, out.ring);
    var eLo = null, eHi = null;
    for (var ti2 = tt0; ti2 <= tt1; ti2++) {
      for (var di2 = Math.round(dist) - dwv; di2 <= Math.round(dist) + dwv; di2++) {
        var c2 = grid[(ti2 / 10).toFixed(1) + '|' + di2];
        if (c2) {
          eLo = eLo === null ? c2[2] : Math.min(eLo, c2[2]);
          eHi = eHi === null ? c2[3] : Math.max(eHi, c2[3]);
        }
      }
    }
    if (eLo !== null) {
      bLo = Math.min(bLo, eLo - 0.025);
      bHi = Math.max(bHi, eHi + 0.025);
    }
    // inner 'likely' pool: official values among comparable plays
    var pool = {}, tot = 0;
    for (var ti3 = Math.round(tA * 10); ti3 <= Math.round(tB * 10); ti3++) {
      for (var di3 = Math.round(dist) - 1; di3 <= Math.round(dist) + 1; di3++) {
        var c3 = grid[(ti3 / 10).toFixed(1) + '|' + di3];
        if (c3 && c3[4]) {
          for (var v3 in c3[4]) { pool[v3] = (pool[v3] || 0) + c3[4][v3]; tot += c3[4][v3]; }
        }
      }
    }
    // sparse-data penalty on the OUTER range only (swept; c=0.1 lifts
    // sparse-window range coverage to 100% in validation)
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

    res.style.display = 'block';
    document.getElementById('cp-big').textContent =
      Math.round(out.p * 100) + '%';
    document.getElementById('cp-stars').textContent =
      (likely ? 'likely ' + likely[0] + '\u2013' + likely[1] + ' · ' : '') +
      'range ' + loPct + '\u2013' + hiPct + ' · ' + starRange(out.p);
    document.getElementById('cp-detail').textContent =
      (combined ? 'combined opportunity estimate ' + combined.toFixed(3) +
                  's · ' : '') +
      'flags: ' + (back === '1' ? 'going back' : 'not back') + ', ' +
      (wall === '1' ? 'wall factor' : 'no wall') + ' · ' +
      out.n + ' comparable plays within ±' +
      (0.05 * out.ring).toFixed(2) + 's / ±' + out.ring + ' ft · ' +
      meta.seasons + ', ' + meta.plays.toLocaleString() + ' tracked plays';
  }

  ['cp-dist', 'cp-time', 'cp-hang', 'cp-plate', 'cp-wall', 'cp-angle'].forEach(function (id) {
    document.getElementById(id).addEventListener('input', update);
  });

  document.getElementById('cp-reset').addEventListener('click', function () {
    ['cp-dist', 'cp-time', 'cp-hang', 'cp-plate', 'cp-angle'].forEach(function (id) {
      document.getElementById(id).value = '';
    });
    document.getElementById('cp-wall').value = '0';
    document.getElementById('cp-result').style.display = 'none';
    showError('');
  });
})();
