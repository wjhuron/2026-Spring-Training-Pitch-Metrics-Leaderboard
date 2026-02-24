var DataStore = {
  pitchData: null,
  pitcherData: null,
  hitterData: null,
  metadata: null,

  load: function () {
    // Use embedded data (works with file:// and http://)
    if (window.PITCH_DATA && window.PITCHER_DATA && window.METADATA) {
      this.pitchData = window.PITCH_DATA;
      this.pitcherData = window.PITCHER_DATA;
      this.hitterData = window.HITTER_DATA || [];
      this.metadata = window.METADATA;
      return Promise.resolve();
    }

    // Fallback: try fetch (only works with http server)
    var self = this;
    return Promise.all([
      fetch('data/pitch_leaderboard.json').then(function (r) { return r.json(); }),
      fetch('data/pitcher_leaderboard.json').then(function (r) { return r.json(); }),
      fetch('data/hitter_leaderboard.json').then(function (r) { return r.json(); }).catch(function () { return []; }),
      fetch('data/metadata.json').then(function (r) { return r.json(); }),
    ]).then(function (results) {
      self.pitchData = results[0];
      self.pitcherData = results[1];
      self.hitterData = results[2];
      self.metadata = results[3];
    }).catch(function (e) {
      console.error('Failed to load data:', e);
    });
  },

  /**
   * Filter data based on current filters.
   * pitchTypes can be an array for multi-select: ['FF', 'SI'] or 'all'
   */
  getFilteredData: function (tab, filters) {
    var source;
    if (tab === 'pitch') source = this.pitchData;
    else if (tab === 'pitcher') source = this.pitcherData;
    else if (tab === 'hitter') source = this.hitterData;
    if (!source) return [];

    var selectedPitchTypes = filters.pitchTypes; // array or 'all'

    return source.filter(function (row) {
      if (filters.team !== 'all' && row.team !== filters.team) return false;

      // Throws filter applies to pitchers; stands filter applies to hitters (same dropdown)
      if (filters.throws !== 'all') {
        if (tab === 'hitter') {
          if (row.stands !== filters.throws) return false;
        } else {
          if (row.throws !== filters.throws) return false;
        }
      }

      if (tab === 'pitch' && selectedPitchTypes !== 'all') {
        if (selectedPitchTypes.indexOf(row.pitchType) === -1) return false;
      }
      if (row.count < filters.minCount) return false;
      if (tab === 'hitter' && filters.minSwings && row.nSwings < filters.minSwings) return false;
      if (filters.search) {
        var name = (row.pitcher || row.hitter || '').toLowerCase();
        if (name.indexOf(filters.search.toLowerCase()) === -1) return false;
      }
      return true;
    });
  },
};
