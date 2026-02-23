var DataStore = {
  pitchData: null,
  pitcherData: null,
  metadata: null,

  load: function () {
    // Use embedded data (works with file:// and http://)
    if (window.PITCH_DATA && window.PITCHER_DATA && window.METADATA) {
      this.pitchData = window.PITCH_DATA;
      this.pitcherData = window.PITCHER_DATA;
      this.metadata = window.METADATA;
      return Promise.resolve();
    }

    // Fallback: try fetch (only works with http server)
    var self = this;
    return Promise.all([
      fetch('data/pitch_leaderboard.json').then(function (r) { return r.json(); }),
      fetch('data/pitcher_leaderboard.json').then(function (r) { return r.json(); }),
      fetch('data/metadata.json').then(function (r) { return r.json(); }),
    ]).then(function (results) {
      self.pitchData = results[0];
      self.pitcherData = results[1];
      self.metadata = results[2];
    }).catch(function (e) {
      console.error('Failed to load data:', e);
    });
  },

  /**
   * Filter data based on current filters.
   * pitchTypes can be an array for multi-select: ['FF', 'SI'] or 'all'
   */
  getFilteredData: function (tab, filters) {
    var source = tab === 'pitch' ? this.pitchData : this.pitcherData;
    if (!source) return [];

    var selectedPitchTypes = filters.pitchTypes; // array or 'all'

    return source.filter(function (row) {
      if (filters.team !== 'all' && row.team !== filters.team) return false;
      if (filters.throws !== 'all' && row.throws !== filters.throws) return false;
      if (tab === 'pitch' && selectedPitchTypes !== 'all') {
        if (selectedPitchTypes.indexOf(row.pitchType) === -1) return false;
      }
      if (row.count < filters.minCount) return false;
      if (filters.search) {
        var name = (row.pitcher || '').toLowerCase();
        if (name.indexOf(filters.search.toLowerCase()) === -1) return false;
      }
      return true;
    });
  },
};
