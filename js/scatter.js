var ScatterChart = {
  chart: null,
  compareChart: null,
  currentPitcher: null,

  COLORS: {
    FF: { bg: '#0000FF', border: '#0000CC' },
    SI: { bg: '#FFD700', border: '#CCB000' },
    FC: { bg: '#FFA500', border: '#CC8400' },
    SL: { bg: '#006400', border: '#004D00' },
    ST: { bg: '#FF1493', border: '#CC1076' },
    SV: { bg: '#32CD32', border: '#28A428' },
    CU: { bg: '#CD3333', border: '#A42929' },
    CH: { bg: '#800080', border: '#660066' },
    FS: { bg: '#40E0D0', border: '#33B3A6' },
    KN: { bg: '#000000', border: '#333333' },
  },

  MARKER_STYLES: ['circle', 'triangle', 'rect', 'rectRot'],

  getColor: function (pt) {
    return this.COLORS[pt] || { bg: '#999', border: '#777' };
  },

  computeEllipse: function (points) {
    if (points.length < 3) return null;
    var n = points.length;
    var mx = 0, my = 0;
    for (var i = 0; i < n; i++) { mx += points[i].x; my += points[i].y; }
    mx /= n; my /= n;
    var cxx = 0, cxy = 0, cyy = 0;
    for (var i = 0; i < n; i++) {
      var dx = points[i].x - mx, dy = points[i].y - my;
      cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
    }
    cxx /= n; cxy /= n; cyy /= n;
    var trace = cxx + cyy;
    var det = cxx * cyy - cxy * cxy;
    var disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    var l1 = trace / 2 + disc;
    var l2 = trace / 2 - disc;
    var angle = 0;
    if (cxy !== 0) angle = Math.atan2(l1 - cxx, cxy);
    else if (cxx < cyy) angle = Math.PI / 2;
    var rx = 1.5 * Math.sqrt(Math.max(0, l1));
    var ry = 1.5 * Math.sqrt(Math.max(0, l2));
    return { cx: mx, cy: my, rx: rx, ry: ry, angle: angle };
  },

  // Custom Chart.js plugin for ellipses and crosshairs
  ellipsePlugin: {
    id: 'ellipsePlugin',
    afterDatasetsDraw: function (chart) {
      var ctx = chart.ctx;
      var xScale = chart.scales.x;
      var yScale = chart.scales.y;

      ctx.save();

      // Draw dashed crosshairs at (0, 0)
      var zeroX = xScale.getPixelForValue(0);
      var zeroY = yScale.getPixelForValue(0);
      ctx.strokeStyle = '#999';
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(zeroX, yScale.top);
      ctx.lineTo(zeroX, yScale.bottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(xScale.left, zeroY);
      ctx.lineTo(xScale.right, zeroY);
      ctx.stroke();

      // Draw ellipses
      var meta = chart._ellipseMeta;
      if (meta) {
        for (var i = 0; i < meta.length; i++) {
          var e = meta[i];
          if (!e.ellipse) continue;
          var cpx = xScale.getPixelForValue(e.ellipse.cx);
          var cpy = yScale.getPixelForValue(e.ellipse.cy);
          var rpxX = Math.abs(xScale.getPixelForValue(e.ellipse.rx) - xScale.getPixelForValue(0));
          var rpxY = Math.abs(yScale.getPixelForValue(e.ellipse.ry) - yScale.getPixelForValue(0));
          ctx.strokeStyle = e.color;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.save();
          ctx.translate(cpx, cpy);
          ctx.rotate(-e.ellipse.angle);
          ctx.ellipse(0, 0, rpxX, rpxY, 0, 0, 2 * Math.PI);
          ctx.restore();
          ctx.stroke();
        }
      }

      ctx.restore();
    }
  },

  _buildMovementData: function (pitcherName) {
    var details = window.PITCH_DETAILS;
    if (!details) return null;
    var pitches = details[pitcherName];
    if (!pitches || pitches.length === 0) return null;

    var groups = {};
    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (!groups[p.pt]) groups[p.pt] = [];
      groups[p.pt].push({ x: p.hb, y: p.ivb });
    }
    return groups;
  },

  render: function (pitcherName) {
    this.currentPitcher = pitcherName;

    var groups = this._buildMovementData(pitcherName);
    if (!groups) return;

    var datasets = [];
    var ellipseMeta = [];
    var pitchTypes = Object.keys(groups).sort();

    for (var j = 0; j < pitchTypes.length; j++) {
      var pt = pitchTypes[j];
      var pts = groups[pt];
      var color = this.getColor(pt);
      var label = pt + ' - ' + (Utils.pitchTypeLabel(pt) || pt);

      datasets.push({
        label: label,
        data: pts,
        backgroundColor: color.bg,
        borderColor: color.border,
        borderWidth: 1.5,
        pointRadius: 6,
        pointHoverRadius: 8,
      });

      var ellipse = this.computeEllipse(pts);
      ellipseMeta.push({ color: color.border, ellipse: ellipse });
    }

    this.destroyMain();

    var canvas = document.getElementById('pitch-chart');
    var ctx = canvas.getContext('2d');

    this.chart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { usePointStyle: true, pointStyle: 'circle', padding: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': HB ' + ctx.parsed.x + ', IVB ' + ctx.parsed.y;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Horizontal Break (in.)', font: { size: 12, weight: 'bold' } },
            min: -25, max: 25,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: 5 },
          },
          y: {
            title: { display: true, text: 'Induced Vertical Break (in.)', font: { size: 12, weight: 'bold' } },
            min: -25, max: 25,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: 5 },
          },
        },
        animation: { duration: 300 },
      },
      plugins: [this.ellipsePlugin],
    });

    this.chart._ellipseMeta = ellipseMeta;
  },

  // Compare mode: overlay multiple pitchers
  renderCompare: function (pitcherNames) {
    if (!pitcherNames || pitcherNames.length === 0) return;

    var datasets = [];
    var details = window.PITCH_DETAILS;
    if (!details) return;

    for (var pi = 0; pi < pitcherNames.length; pi++) {
      var name = pitcherNames[pi];
      var pitches = details[name];
      if (!pitches) continue;

      var groups = {};
      for (var i = 0; i < pitches.length; i++) {
        var p = pitches[i];
        if (!groups[p.pt]) groups[p.pt] = [];
        groups[p.pt].push({ x: p.hb, y: p.ivb });
      }

      var pitchTypes = Object.keys(groups).sort();
      var markerStyle = this.MARKER_STYLES[pi % this.MARKER_STYLES.length];

      for (var j = 0; j < pitchTypes.length; j++) {
        var pt = pitchTypes[j];
        var color = this.getColor(pt);
        datasets.push({
          label: name + ' - ' + pt,
          data: groups[pt],
          backgroundColor: color.bg,
          borderColor: color.border,
          borderWidth: 1.5,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointStyle: markerStyle,
        });
      }
    }

    this.destroyCompare();

    var canvas = document.getElementById('compare-chart');
    var ctx = canvas.getContext('2d');

    this.compareChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { usePointStyle: true, padding: 12, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': HB ' + ctx.parsed.x + ', IVB ' + ctx.parsed.y;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Horizontal Break (in.)', font: { size: 12, weight: 'bold' } },
            min: -25, max: 25,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: 5 },
          },
          y: {
            title: { display: true, text: 'Induced Vertical Break (in.)', font: { size: 12, weight: 'bold' } },
            min: -25, max: 25,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: 5 },
          },
        },
        animation: { duration: 300 },
      },
      plugins: [this.ellipsePlugin],
    });

    // Add crosshairs meta (no ellipses for compare)
    this.compareChart._ellipseMeta = [];
  },

  destroyMain: function () {
    if (this.chart) { this.chart.destroy(); this.chart = null; }
  },

  destroyCompare: function () {
    if (this.compareChart) { this.compareChart.destroy(); this.compareChart = null; }
  },

  destroy: function () {
    this.destroyMain();
  },
};
