var Utils = {
  formatPct: function (value) {
    if (value === null || value === undefined) return '--';
    return (value * 100).toFixed(1) + '%';
  },

  formatDecimal: function (places) {
    return function (value) {
      if (value === null || value === undefined) return '--';
      return Number(value).toFixed(places);
    };
  },

  formatInt: function (value) {
    if (value === null || value === undefined) return '--';
    return Math.round(value).toString();
  },

  formatTilt: function (value) {
    if (!value) return '--';
    return value;
  },

  formatStuff: function (value) {
    if (value === null || value === undefined) return '--';
    return Math.round(value).toString();
  },

  PITCH_TYPE_LABELS: {
    FF: 'Four-Seam', SI: 'Sinker', FC: 'Cutter', CH: 'Changeup',
    CU: 'Curveball', SL: 'Slider', ST: 'Sweeper', FS: 'Splitter',
    SV: 'Slurve', KN: 'Knuckleball', SC: 'Screwball', CS: 'Slow Curve',
  },

  pitchTypeLabel: function (code) {
    return this.PITCH_TYPE_LABELS[code] || code;
  },

  // Pitch type colors (matches scatter.js)
  PITCH_COLORS: {
    FF: '#0000FF', SI: '#FFD700', FC: '#FFA500', SL: '#006400',
    ST: '#FF1493', SV: '#32CD32', CU: '#CD3333', CH: '#800080',
    FS: '#40E0D0', KN: '#000000', SC: '#999999', CS: '#666666',
  },

  getPitchColor: function (pt) {
    return this.PITCH_COLORS[pt] || '#999';
  },

  // Percentile color: blue (0) -> white (50) -> red (100)
  percentileColor: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    var r, g, b;
    if (pctl <= 50) {
      var t = pctl / 50;
      r = Math.round(60 + t * 195);
      g = Math.round(100 + t * 155);
      b = Math.round(240 + t * 15);
    } else {
      var t = (pctl - 50) / 50;
      r = 255;
      g = Math.round(255 - t * 180);
      b = Math.round(255 - t * 200);
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  },

  // Text color for percentile backgrounds
  percentileTextColor: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    if (pctl < 15 || pctl > 85) return '#fff';
    return '#1a1a2e';
  },

  // Dark mode: semi-transparent overlays so text stays readable
  percentileColorDark: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    if (pctl <= 50) {
      // Blue tint, stronger at low percentiles
      var opacity = ((50 - pctl) / 50 * 0.45).toFixed(2);
      return 'rgba(60, 120, 255, ' + opacity + ')';
    } else {
      // Red tint, stronger at high percentiles
      var opacity = ((pctl - 50) / 50 * 0.45).toFixed(2);
      return 'rgba(255, 70, 50, ' + opacity + ')';
    }
  },

  percentileTextColorDark: function (pctl) {
    if (pctl === null || pctl === undefined) return null;
    return '#e6edf3';
  },

  TOOLTIPS: {
    '#': 'Rank',
    'Pitcher': 'Pitcher name',
    'Team': 'Team abbreviation',
    'Throws': 'Throwing hand (R/L)',
    'Pitch': 'Pitch type',
    'Count': 'Number of pitches',
    'Usage%': 'Usage rate (% of pitcher\'s total pitches)',
    'Velo': 'Average Velocity (mph)',
    'Spin': 'Average Spin Rate (rpm)',
    'Tilt': 'Average Break Tilt (clock notation)',
    'IVB': 'Induced Vertical Break (inches)',
    'HB': 'Horizontal Break (inches)',
    'RelZ': 'Vertical Release Position (feet)',
    'RelX': 'Horizontal Release Position (feet)',
    'Ext': 'Extension (feet)',
    'VAA': 'Vertical Approach Angle (degrees)',
    'HAA': 'Horizontal Approach Angle (degrees)',
    'VRA': 'Vertical Release Angle (degrees)',
    'HRA': 'Horizontal Release Angle (degrees)',
    'IZ%': 'In-Zone Rate',
    'SwStr%': 'Swinging Strike Rate',
    'CSW%': 'Called Strike + Whiff Rate',
    'Chase%': 'Out-of-Zone Swing Rate',
    'GB%': 'Ground Ball Rate',
  },

  // URL state helpers
  readHash: function () {
    var hash = window.location.hash.replace(/^#/, '');
    if (!hash) return {};
    var params = {};
    hash.split('&').forEach(function (part) {
      var kv = part.split('=');
      if (kv.length === 2) params[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1]);
    });
    return params;
  },

  writeHash: function (params) {
    var parts = [];
    Object.keys(params).forEach(function (k) {
      if (params[k] !== undefined && params[k] !== null && params[k] !== '') {
        parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(params[k]));
      }
    });
    var newHash = parts.join('&');
    if (window.location.hash.replace(/^#/, '') !== newHash) {
      history.replaceState(null, '', '#' + newHash);
    }
  },

  // Export data as CSV string
  toCSV: function (data, columns) {
    var lines = [];
    // Header
    lines.push(columns.map(function (c) { return '"' + c.label + '"'; }).join(','));
    // Rows
    data.forEach(function (row) {
      var cells = columns.map(function (c) {
        var v = row[c.key];
        if (v === null || v === undefined) return '';
        var formatted = c.format(v);
        return '"' + String(formatted).replace(/"/g, '""') + '"';
      });
      lines.push(cells.join(','));
    });
    return lines.join('\n');
  },

  // Download text as file
  downloadFile: function (content, filename, mimeType) {
    var blob = new Blob([content], { type: mimeType || 'text/csv;charset=utf-8;' });
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  },

  // Copy text to clipboard
  copyToClipboard: function (text) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(function () {
        Utils._fallbackCopy(text);
      });
    } else {
      Utils._fallbackCopy(text);
    }
  },

  _fallbackCopy: function (text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  },

  // Tab-separated for clipboard (pastes into Excel/Sheets nicely)
  toTSV: function (data, columns) {
    var lines = [];
    lines.push(columns.map(function (c) { return c.label; }).join('\t'));
    data.forEach(function (row) {
      var cells = columns.map(function (c) {
        var v = row[c.key];
        if (v === null || v === undefined) return '';
        return String(c.format(v));
      });
      lines.push(cells.join('\t'));
    });
    return lines.join('\n');
  },
};
