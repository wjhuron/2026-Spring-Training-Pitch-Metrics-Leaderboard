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
    var zone = document.getElementById('cp-zone').value.split('|');
    var res = document.getElementById('cp-result');

    if (isNaN(time) && !isNaN(hang)) time = hang + FLIGHT;
    if (isNaN(dist) || isNaN(time)) { res.style.display = 'none'; return; }

    var out = lookup(Math.round(time * 10) / 10, Math.round(dist),
                     zone[0], zone[1]);
    if (!out) {
      showError('No comparable tracked plays for those inputs.');
      return;
    }
    res.style.display = 'block';
    document.getElementById('cp-big').textContent =
      (out.p * 100).toFixed(1) + '%';
    document.getElementById('cp-stars').textContent =
      starRange(out.p) + ' · Savant display: ' +
      Math.round(bucket(out.p) * 100) + '%';
    document.getElementById('cp-detail').textContent =
      out.n + ' comparable plays within ±' +
      (0.05 * out.ring).toFixed(2) + 's / ±' + out.ring + ' ft · ' +
      meta.seasons + ', ' + meta.plays.toLocaleString() + ' tracked plays';
  }

  ['cp-dist', 'cp-time', 'cp-hang', 'cp-zone'].forEach(function (id) {
    document.getElementById(id).addEventListener('input', update);
  });
})();
