"""Do ballparks leave a systematic fingerprint on measured movement?

Hawkeye is calibrated per installation, so a park-level bias in the tracked
break is plausible, and this repo already has a park-level tracking artifact on
record (the Progressive Field SwingLength glitch). If real, a venue term is a
legitimate regressor: it is a property of the measuring rig, not of the pitch's
own break, so it cannot leak the answer the way OTilt would.

The trap: a venue's mean residual is NOT evidence on its own. Home pitchers
throw ~half a park's pitches, so a park with an odd rotation shows an offset
that is really a pitcher effect, and altitude/air density vary by park too
(already adjusted, but imperfectly). Both of those are real, persistent
confounds -- they would produce a stable venue offset that has nothing to do
with calibration.

So the test has three levels, each stricter:
  1. raw venue offset in the residual
  2. offset after removing each pitcher's own mean (within-pitcher), which
     kills the roster confound
  3. cross-season replication of the within-pitcher offset. A calibration bias
     is a property of the installation and must persist year to year; a roster
     or weather artifact should not. This is the one that decides it.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, run_linear, FORMS, MIN_N

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
MIN_VENUE_PITCHES = 2000


def venue_map():
    with open(f'{DIR}/venue_by_gamepk.json') as f:
        raw = json.load(f)
    return ({int(k): v[0] for k, v in raw.items()},
            {v[0]: v[1] for v in raw.values()})


def demean_by(vals, keys):
    """Subtract each key-group's mean. Kills the roster confound."""
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    tot = np.bincount(inv, weights=vals)
    return vals - (tot / cnt)[inv]


def main():
    A = load_np()
    vmap, vnames = venue_map()
    venue = np.array([vmap.get(int(g), -1) for g in A['game']])
    have = venue > 0
    print(f"{have.sum():,}/{len(venue):,} pitches mapped to a venue")

    xi, xh = run_linear(A, FORMS['S3b +spin x axis'])
    ok = have & np.isfinite(xi)
    io = A['ivb'] - xi
    ho = A['hb_s'] - xh
    # rotate into the release-axis frame: along = Magnus magnitude, cross = seam
    ao = io * A['ct'] + ho * A['st']
    co = -io * A['st'] + ho * A['ct']

    pit = np.array([f'{p}|{t}|{q}' for p, t, q in
                    zip(A['pitcher'], A['thr'], A['pt'])])

    rows = []
    for season in sorted(set(A['season'][ok])):
        m = ok & (A['season'] == season)
        ao_d = demean_by(ao[m], pit[m])       # within-pitcher-arsenal
        co_d = demean_by(co[m], pit[m])
        v = venue[m]
        for vid in np.unique(v):
            k = v == vid
            if k.sum() < MIN_VENUE_PITCHES:
                continue
            rows.append(dict(season=season, venue=vid, n=int(k.sum()),
                             raw_a=ao[m][k].mean(), raw_c=co[m][k].mean(),
                             adj_a=ao_d[k].mean(), adj_c=co_d[k].mean()))
    df = pd.DataFrame(rows)
    print(f'{df.venue.nunique()} venues x {df.season.nunique()} seasons '
          f'({len(df)} venue-seasons, >= {MIN_VENUE_PITCHES} pitches each)\n')

    print('spread of venue offsets, inches:')
    print(f"  {'':<26} {'raw along':>10} {'raw cross':>10} "
          f"{'within-pitcher along':>21} {'within-pitcher cross':>21}")
    print(f"  {'sd across venue-seasons':<26} {df.raw_a.std():>10.3f} "
          f"{df.raw_c.std():>10.3f} {df.adj_a.std():>21.3f} {df.adj_c.std():>21.3f}")

    # cross-season replication -- the deciding test
    print('\ncross-season replication of the WITHIN-PITCHER venue offset')
    print('(a calibration bias must persist; a roster artifact should not)')
    print(f"  {'pair':>12} {'venues':>7} {'r along':>9} {'r cross':>9}")
    seasons = sorted(df.season.unique())
    ra, rc = [], []
    for i in range(len(seasons) - 1):
        a = df[df.season == seasons[i]].set_index('venue')
        b = df[df.season == seasons[i + 1]].set_index('venue')
        common = a.index.intersection(b.index)
        if len(common) < 15:
            continue
        r1 = np.corrcoef(a.loc[common, 'adj_a'], b.loc[common, 'adj_a'])[0, 1]
        r2 = np.corrcoef(a.loc[common, 'adj_c'], b.loc[common, 'adj_c'])[0, 1]
        ra.append(r1); rc.append(r2)
        print(f'  {seasons[i]}->{seasons[i+1]:>5} {len(common):>7} {r1:>9.3f} {r2:>9.3f}')
    if ra:
        print(f"  {'mean':>12} {'':>7} {np.mean(ra):>9.3f} {np.mean(rc):>9.3f}")

    # the largest persistent offenders, averaged over seasons
    agg = df.groupby('venue').agg(n=('n', 'sum'), a=('adj_a', 'mean'),
                                  c=('adj_c', 'mean'), ns=('season', 'nunique'))
    agg = agg[agg.ns >= 4].copy()
    agg['mag'] = np.hypot(agg.a, agg.c)
    print(f'\nlargest mean within-pitcher venue offsets (>=4 seasons):')
    print(f"  {'venue':<34} {'pitches':>9} {'along':>7} {'cross':>7}")
    for vid, r in agg.sort_values('mag', ascending=False).head(8).iterrows():
        print(f'  {str(vnames.get(vid, vid))[:34]:<34} {int(r.n):>9,} '
              f'{r.a:>7.3f} {r.c:>7.3f}')


    # How much does a pitcher's park MIX contaminate his own OE? The R^2 gain
    # from a venue term is negligible (offsets are ~0.2" against a ~2.9" residual
    # RMSE) -- the reason to care is systematic bias on individual pitchers, not
    # fit. A swingman who lives in one park inherits that park's offset.
    off_a = dict(zip(agg.index, agg.a))
    off_c = dict(zip(agg.index, agg.c))
    va = np.array([off_a.get(v, np.nan) for v in venue])
    vc = np.array([off_c.get(v, np.nan) for v in venue])
    g = ok & np.isfinite(va)
    key = np.array([f'{p}|{t}|{q}|{s}' for p, t, q, s in
                    zip(A['pitcher'], A['thr'], A['pt'], A['season'])])[g]
    u, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    ba = np.bincount(inv, weights=va[g]) / cnt
    bc = np.bincount(inv, weights=vc[g]) / cnt
    keep = cnt >= 50
    print(f'\npitcher-season-pitchtype units (>=50 pitches): {keep.sum()}')
    print(f'  park-mix implied bias in alongOE: sd {ba[keep].std():.3f}"  '
          f'p5 {np.percentile(ba[keep],5):+.3f}"  p95 {np.percentile(ba[keep],95):+.3f}"')
    print(f'  park-mix implied bias in crossOE: sd {bc[keep].std():.3f}"  '
          f'p5 {np.percentile(bc[keep],5):+.3f}"  p95 {np.percentile(bc[keep],95):+.3f}"')
    ao_u = np.bincount(inv, weights=ao[g]) / cnt
    co_u = np.bincount(inv, weights=co[g]) / cnt
    print(f'  for scale, observed pitcher-level OE sd: along {ao_u[keep].std():.3f}"  '
          f'cross {co_u[keep].std():.3f}"')
    print(f'  => park mix accounts for {ba[keep].std()/ao_u[keep].std()*100:.1f}% of '
          f'along OE sd, {bc[keep].std()/co_u[keep].std()*100:.1f}% of cross OE sd')


if __name__ == '__main__':
    main()
