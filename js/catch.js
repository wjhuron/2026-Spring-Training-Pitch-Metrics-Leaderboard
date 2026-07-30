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

  fetch('data/catch_prob_surface.json')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      meta = data.meta;
      surface = {};
      Object.keys(data.cells).forEach(function (k) {
        var parts = k.split('|'); // t|d|b|w
        var fk = parts[2] + '|' + parts[3];
        if (!surface[fk]) surface[fk] = {};
        surface[fk][parts[0] + '|' + parts[1]] =
          [data.cells[k].p, data.cells[k].n];
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
    // Research-portal cards TRUNCATE to one decimal (card value T means
    // true time in [T, T + 0.1]), so center every card read at +0.05 and,
    // when both clocks are given, intersect the two truncation intervals.
    var combined = null;
    if (isNaN(time) && !isNaN(hang)) {
      time = hang + 0.05 + flight;
    } else if (!isNaN(time) && !isNaN(hang)) {
      var lo = Math.max(time, hang + flight);
      var hi = Math.min(time + 0.1, hang + 0.1 + flight);
      time = lo <= hi ? (lo + hi) / 2 : time + 0.05;
      if (lo <= hi) combined = time;
    } else if (!isNaN(time)) {
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
                ring: Math.max(o1.ring, o2.ring) };
      } else {
        out = o1 || o2;
      }
    }
    if (!out) {
      showError('No comparable tracked plays for those inputs.');
      return;
    }
    res.style.display = 'block';
    document.getElementById('cp-big').textContent =
      Math.round(out.p * 100) + '%';
    document.getElementById('cp-stars').textContent =
      starRange(out.p) + ' · Savant display: ' +
      Math.round(bucket(out.p) * 100) + '%';
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
