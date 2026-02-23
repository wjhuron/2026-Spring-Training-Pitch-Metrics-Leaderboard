var ScatterChart = {
  chart: null,
  veloChart: null,
  compareChart: null,
  currentView: 'movement', // 'movement' or 'release'
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

  // Custom Chart.js plugin for ellipses, crosshairs, and league average markers
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

      // Draw league average markers
      var leagueMarkers = chart._leagueAvgMarkers;
      if (leagueMarkers) {
        ctx.setLineDash([]);
        for (var j = 0; j < leagueMarkers.length; j++) {
          var m = leagueMarkers[j];
          var px = xScale.getPixelForValue(m.x);
          var py = yScale.getPixelForValue(m.y);
          // Draw diamond marker
          var s = 8;
          ctx.fillStyle = 'rgba(0,0,0,0.15)';
          ctx.strokeStyle = m.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(px, py - s);
          ctx.lineTo(px + s, py);
          ctx.lineTo(px, py + s);
          ctx.lineTo(px - s, py);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          // Label
          ctx.fillStyle = m.color;
          ctx.font = 'bold 9px sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(m.label + ' avg', px + s + 3, py + 3);
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

  _buildReleaseData: function (pitcherName) {
    var details = window.PITCH_DETAILS;
    if (!details) return null;
    var pitches = details[pitcherName];
    if (!pitches || pitches.length === 0) return null;

    var groups = {};
    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (p.rx !== undefined && p.rz !== undefined) {
        if (!groups[p.pt]) groups[p.pt] = [];
        groups[p.pt].push({ x: p.rx, y: p.rz });
      }
    }
    return groups;
  },

  _getLeagueAvgMarkers: function (pitchTypes, view) {
    var meta = DataStore.metadata;
    if (!meta || !meta.leagueAverages) return [];
    var markers = [];
    for (var i = 0; i < pitchTypes.length; i++) {
      var pt = pitchTypes[i];
      var avg = meta.leagueAverages[pt];
      if (!avg) continue;
      var color = this.getColor(pt);
      if (view === 'movement' && avg.horzBrk !== undefined && avg.indVertBrk !== undefined) {
        markers.push({ x: avg.horzBrk, y: avg.indVertBrk, color: color.border, label: pt });
      } else if (view === 'release' && avg.relPosX !== undefined && avg.relPosZ !== undefined) {
        markers.push({ x: avg.relPosX, y: avg.relPosZ, color: color.border, label: pt });
      }
    }
    return markers;
  },

  render: function (pitcherName, view) {
    if (view) this.currentView = view;
    this.currentPitcher = pitcherName;

    var groups;
    if (this.currentView === 'release') {
      groups = this._buildReleaseData(pitcherName);
    } else {
      groups = this._buildMovementData(pitcherName);
    }
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

    var isMovement = this.currentView === 'movement';
    var xLabel = isMovement ? 'Horizontal Break (in.)' : 'Horizontal Release (ft.)';
    var yLabel = isMovement ? 'Induced Vertical Break (in.)' : 'Vertical Release (ft.)';
    var xMin = isMovement ? -25 : -4;
    var xMax = isMovement ? 25 : 4;
    var yMin = isMovement ? -25 : 3;
    var yMax = isMovement ? 25 : 8;
    var step = isMovement ? 5 : 1;

    var leagueMarkers = this._getLeagueAvgMarkers(pitchTypes, this.currentView);

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
                var xL = isMovement ? 'HB' : 'RelX';
                var yL = isMovement ? 'IVB' : 'RelZ';
                return ctx.dataset.label + ': ' + xL + ' ' + ctx.parsed.x + ', ' + yL + ' ' + ctx.parsed.y;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: xLabel, font: { size: 12, weight: 'bold' } },
            min: xMin, max: xMax,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: step },
          },
          y: {
            title: { display: true, text: yLabel, font: { size: 12, weight: 'bold' } },
            min: yMin, max: yMax,
            grid: { display: true, color: 'rgba(0,0,0,0.06)' },
            ticks: { stepSize: step },
          },
        },
        animation: { duration: 300 },
      },
      plugins: [this.ellipsePlugin],
    });

    this.chart._ellipseMeta = ellipseMeta;
    this.chart._leagueAvgMarkers = leagueMarkers;

    // Also render velocity distribution
    this.renderVeloChart(pitcherName);
  },

  renderVeloChart: function (pitcherName) {
    var details = window.PITCH_DETAILS;
    if (!details) return;
    var pitches = details[pitcherName];
    if (!pitches || pitches.length === 0) return;

    // Group velocities by pitch type
    var groups = {};
    for (var i = 0; i < pitches.length; i++) {
      var p = pitches[i];
      if (p.v !== undefined) {
        if (!groups[p.pt]) groups[p.pt] = [];
        groups[p.pt].push(p.v);
      }
    }

    var pitchTypes = Object.keys(groups).sort();
    if (pitchTypes.length === 0) return;

    // Build floating bar data: [min, max] per pitch type, with mean marker
    var labels = [];
    var barData = [];
    var bgColors = [];
    var borderColors = [];
    var meanData = [];

    for (var j = 0; j < pitchTypes.length; j++) {
      var pt = pitchTypes[j];
      var velos = groups[pt].sort(function (a, b) { return a - b; });
      var color = this.getColor(pt);
      labels.push(pt);
      // Use 10th-90th percentile for the bar, not absolute min/max
      var p10 = velos[Math.floor(velos.length * 0.1)];
      var p90 = velos[Math.floor(velos.length * 0.9)];
      barData.push([p10, p90]);
      bgColors.push(color.bg + '80'); // semi-transparent
      borderColors.push(color.border);
      var sum = 0;
      for (var k = 0; k < velos.length; k++) sum += velos[k];
      meanData.push(sum / velos.length);
    }

    this.destroyVelo();

    // Compute smart x-axis range
    var allVelos = [];
    for (var v = 0; v < barData.length; v++) {
      allVelos.push(barData[v][0], barData[v][1]);
    }
    var veloMin = Math.floor(Math.min.apply(null, allVelos) - 3);
    var veloMax = Math.ceil(Math.max.apply(null, allVelos) + 3);

    var canvas = document.getElementById('velo-chart');
    var ctx = canvas.getContext('2d');

    this.veloChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Velo Range (10th-90th)',
            data: barData,
            backgroundColor: bgColors,
            borderColor: borderColors,
            borderWidth: 1,
            borderSkipped: false,
            barPercentage: 0.6,
          },
          {
            label: 'Avg Velo',
            data: meanData,
            type: 'scatter',
            pointBackgroundColor: borderColors,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
            xAxisID: 'x',
            yAxisID: 'y',
          }
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.datasetIndex === 0) {
                  return ctx.label + ': ' + ctx.raw[0].toFixed(1) + ' - ' + ctx.raw[1].toFixed(1) + ' mph';
                }
                return ctx.label + ' avg: ' + ctx.raw.toFixed(1) + ' mph';
              }
            }
          }
        },
        scales: {
          x: {
            min: veloMin,
            max: veloMax,
            title: { display: true, text: 'Velocity (mph)', font: { size: 11 } },
            grid: { color: 'rgba(0,0,0,0.06)' },
          },
          y: {
            grid: { display: false },
          }
        },
        animation: { duration: 200 },
      },
    });
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

  destroyVelo: function () {
    if (this.veloChart) { this.veloChart.destroy(); this.veloChart = null; }
  },

  destroyCompare: function () {
    if (this.compareChart) { this.compareChart.destroy(); this.compareChart = null; }
  },

  destroy: function () {
    this.destroyMain();
    this.destroyVelo();
  },
};
