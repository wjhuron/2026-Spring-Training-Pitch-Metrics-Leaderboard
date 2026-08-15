import argparse
import requests
import pandas as pd
import numpy as np
import os
import time
from io import StringIO


class BaseballSavantHitterDownloader:
    """
    Hitter-focused downloader. Pulls every pitch a batter saw in the date range
    directly from Statcast Search, keyed by MLBID (so it works across team
    changes and seasons). One CSV per batter, written to ``download_dir``.

    Output schema is hitter-centric and parallel to Pitcher2026 where columns
    overlap (Bats/Throws/PitchType/Velocity, the xStats family, swing tracking,
    Barrel, PitchID format). Adds bat-tracking intercept columns
    (Intercept_X, Intercept_Y) that Pitcher2026 does not include.
    """

    def __init__(self, download_dir=os.path.expanduser("~/Downloads")):
        self.download_dir = download_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://baseballsavant.mlb.com/statcast_search',
        })
        os.makedirs(download_dir, exist_ok=True)

        # Statcast uses slightly different abbreviations than Pitcher2026's internal scheme
        self.statcast_to_team_abbrev = {
            'OAK': 'ATH',
            'KC':  'KCR',
            'SD':  'SDP',
            'SF':  'SFG',
            'TB':  'TBR',
            'CHW': 'CWS',
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def normalize_team(self, t):
        if pd.isna(t):
            return ''
        return self.statcast_to_team_abbrev.get(t, t)

    def is_barrel(self, ev, la):
        """Statcast barrel formula: launch_speed_angle == 6."""
        if ev is None or la is None or pd.isna(ev) or pd.isna(la):
            return False
        try:
            ev = float(ev)
            la = float(la)
        except (ValueError, TypeError):
            return False
        return (la >= 8 and la <= 50 and ev >= 98
                and ev * 1.5 - la >= 117 and ev + la >= 124)

    def simplify_description(self, desc):
        """Map Statcast description values to the same labels Pitcher2026 emits."""
        if pd.isna(desc) or not desc:
            return ''
        DESC_MAP = {
            'hit_into_play':            'In Play',
            'hit_into_play_no_out':     'In Play',
            'hit_into_play_score':      'In Play',
            'swinging_strike':          'Swinging Strike',
            'swinging_strike_blocked':  'Swinging Strike',
            'foul_tip':                 'Swinging Strike',
            'ball':                     'Ball',
            'blocked_ball':             'Ball',
            'called_strike':            'Called Strike',
            'hit_by_pitch':             'Hit By Pitch',
            'foul':                     'Foul',
            'foul_bunt':                'Foul Bunt',
            'bunt_foul_tip':            'Bunt Foul Tip',
            'missed_bunt':              'Missed Bunt',
            'intent_ball':              'Intent Ball',
            'pitchout':                 'Pitchout',
            'swinging_pitchout':        'Swinging Pitchout',
            'foul_pitchout':            'Foul Pitchout',
        }
        return DESC_MAP.get(str(desc).lower(), desc)

    def normalize_event(self, e):
        """Statcast events are snake_case; convert to Title Case for readability."""
        if pd.isna(e) or not e:
            return ''
        return str(e).replace('_', ' ').title()

    def get_player_name(self, mlb_id):
        """Fall back name lookup via MLB Stats API. Returns 'Last, First' or None."""
        url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return None
            data = resp.json()
            people = data.get('people', [])
            if not people:
                return None
            person = people[0]
            last_first = person.get('lastFirstName')
            if last_first:
                return last_first
            first = person.get('firstName', '')
            last = person.get('lastName', '')
            if first and last:
                return f"{last}, {first}"
            return person.get('fullName')
        except Exception:
            return None

    @staticmethod
    def safe_filename(name):
        """'Last, First' / 'Witt Jr., Bobby' → 'Last_First' / 'Witt_Jr_Bobby'."""
        cleaned = name.replace(',', '').replace('.', '').strip()
        return '_'.join(cleaned.split())

    @staticmethod
    def _safe_col(df, name):
        """Return df[name] if it exists, else an all-NaN Series of matching length."""
        if name in df.columns:
            return df[name]
        return pd.Series([np.nan] * len(df), index=df.index)

    # ── Download ──────────────────────────────────────────────────────────────

    def download_hitter_raw(self, mlb_id, start_date, end_date):
        """Fetch the raw Statcast Search CSV for one batter. Returns DataFrame or None."""
        print(f"\n=== MLBID {mlb_id} ({start_date} → {end_date}) ===")

        url = "https://baseballsavant.mlb.com/statcast_search/csv"
        params = {
            'all':              'true',
            'type':             'details',
            'game_date_gt':     start_date,
            'game_date_lt':     end_date,
            'batters_lookup[]': str(mlb_id),
            'player_type':      'batter',
            'min_pitches':      '0',
            'min_results':      '0',
            'sort_col':         'pitches',
            'sort_order':       'desc',
        }

        try:
            response = self.session.get(url, params=params, timeout=120)
            if response.status_code != 200:
                print(f"  Statcast Search returned status {response.status_code}")
                return None

            csv_text = response.text
            if not csv_text or csv_text.strip() == '' or 'No Results' in csv_text[:200]:
                print("  No data returned (no pitches in range or invalid MLBID)")
                return None

            df = pd.read_csv(StringIO(csv_text))
            if df.empty:
                print("  Empty DataFrame")
                return None

            print(f"  Got {len(df)} pitches")
            return df

        except requests.exceptions.Timeout:
            print("  Statcast Search request timed out")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, raw_df):
        """Map raw Statcast columns to the Hitter2026 output schema."""
        out = pd.DataFrame(index=raw_df.index)

        out['Game Date'] = self._safe_col(raw_df, 'game_date').astype(str)

        # Batter's team: home if bottom of inning, else away
        topbot = self._safe_col(raw_df, 'inning_topbot')
        home   = self._safe_col(raw_df, 'home_team')
        away   = self._safe_col(raw_df, 'away_team')
        bteam  = pd.Series(np.where(topbot == 'Bot', home, away), index=raw_df.index)
        out['BTeam'] = bteam.map(self.normalize_team)

        out['Batter'] = self._safe_col(raw_df, 'player_name').fillna('').str.strip()
        out['Bats']   = self._safe_col(raw_df, 'stand').fillna('')
        out['Throws'] = self._safe_col(raw_df, 'p_throws').fillna('')

        # Pitch Type with KC→CU and FO→FS remap (matches Pitcher2026)
        pt = self._safe_col(raw_df, 'pitch_type').fillna('')
        pitch_remap = {'KC': 'CU', 'FO': 'FS'}
        out['Pitch Type'] = pt.map(lambda x: pitch_remap.get(x, x))

        out['Velocity'] = pd.to_numeric(self._safe_col(raw_df, 'release_speed'), errors='coerce').apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ''
        )

        for col, src in [('PlateZ', 'plate_z'), ('PlateX', 'plate_x'),
                         ('SzTop',  'sz_top'),  ('SzBot',  'sz_bot')]:
            out[col] = pd.to_numeric(self._safe_col(raw_df, src), errors='coerce').apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else ''
            )

        out['Zone'] = pd.to_numeric(self._safe_col(raw_df, 'zone'), errors='coerce').astype('Int64')

        balls   = pd.to_numeric(self._safe_col(raw_df, 'balls'),   errors='coerce').fillna(0).astype(int)
        strikes = pd.to_numeric(self._safe_col(raw_df, 'strikes'), errors='coerce').fillna(0).astype(int)
        out['Count'] = balls.astype(str) + '-' + strikes.astype(str)

        out['Description'] = self._safe_col(raw_df, 'description').apply(self.simplify_description)
        out['Event']       = self._safe_col(raw_df, 'events').apply(self.normalize_event)

        out['ExitVelo'] = pd.to_numeric(self._safe_col(raw_df, 'launch_speed'), errors='coerce').apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ''
        )
        out['LaunchAngle'] = pd.to_numeric(self._safe_col(raw_df, 'launch_angle'),    errors='coerce').astype('Int64')
        out['Distance']    = pd.to_numeric(self._safe_col(raw_df, 'hit_distance_sc'), errors='coerce').astype('Int64')
        out['BBType']      = self._safe_col(raw_df, 'bb_type').fillna('')

        for col, src in [('HC_X', 'hc_x'), ('HC_Y', 'hc_y')]:
            out[col] = pd.to_numeric(self._safe_col(raw_df, src), errors='coerce').apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ''
            )

        # 3-decimal stats (xStats family + run expectancy)
        # delta_run_exp is batter-perspective (positive = good for batter),
        # opposite sign of Pitcher2026's delta_pitcher_run_exp
        for col, src in [('xBA',     'estimated_ba_using_speedangle'),
                         ('xSLG',    'estimated_slg_using_speedangle'),
                         ('xwOBA',   'estimated_woba_using_speedangle'),
                         ('wOBAval', 'woba_value'),
                         ('wOBAdom', 'woba_denom'),
                         ('RunExp',  'delta_run_exp')]:
            out[col] = pd.to_numeric(self._safe_col(raw_df, src), errors='coerce').apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else ''
            )

        # Swing-cluster validity: BatSpeed missing or <50 means the entire
        # swing-tracking frame is unreliable; null all members together.
        bs_numeric = pd.to_numeric(self._safe_col(raw_df, 'bat_speed'), errors='coerce')
        swing_invalid = bs_numeric.isna() | (bs_numeric < 50)
        bs_valid = bs_numeric.where(~swing_invalid)
        out['BatSpeed'] = bs_valid.apply(lambda x: f"{x:.1f}" if pd.notna(x) else '')

        for col, src in [('SwingLength',     'swing_length'),
                         ('AttackAngle',     'attack_angle'),
                         ('AttackDirection', 'attack_direction'),
                         ('SwingPathTilt',   'swing_path_tilt')]:
            numeric = pd.to_numeric(self._safe_col(raw_df, src), errors='coerce').where(~swing_invalid)
            out[col] = numeric.apply(lambda x: f"{x:.1f}" if pd.notna(x) else '')

        for col, src in [('Intercept_X', 'intercept_ball_minus_batter_pos_x_inches'),
                         ('Intercept_Y', 'intercept_ball_minus_batter_pos_y_inches')]:
            numeric = pd.to_numeric(self._safe_col(raw_df, src), errors='coerce').where(~swing_invalid)
            out[col] = numeric.apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')

        # PitchID: same scheme as Pitcher2026
        gp = self._safe_col(raw_df, 'game_pk').astype(str)
        ab = pd.to_numeric(self._safe_col(raw_df, 'at_bat_number'), errors='coerce').fillna(0).astype(int)
        pn = pd.to_numeric(self._safe_col(raw_df, 'pitch_number'),  errors='coerce').fillna(0).astype(int)
        out['PitchID'] = gp + '_' + ab.apply(lambda x: f"{x:03d}") + '_' + pn.apply(lambda x: f"{x:02d}")

        # Barrel: prefer official launch_speed_angle, fall back to estimate
        lsa     = pd.to_numeric(self._safe_col(raw_df, 'launch_speed_angle'), errors='coerce')
        ev_raw  = pd.to_numeric(self._safe_col(raw_df, 'launch_speed'),       errors='coerce')
        la_raw  = pd.to_numeric(self._safe_col(raw_df, 'launch_angle'),       errors='coerce')
        estimated = pd.Series(
            [6 if self.is_barrel(ev, la) else None for ev, la in zip(ev_raw, la_raw)],
            index=raw_df.index,
        )
        out['Barrel'] = lsa.combine_first(estimated).astype('Int64')

        final_columns = [
            'Game Date', 'BTeam', 'Batter', 'Bats',
            'Throws', 'Pitch Type', 'Velocity',
            'PlateZ', 'PlateX', 'SzTop', 'SzBot', 'Zone',
            'Count', 'Description', 'Event',
            'ExitVelo', 'LaunchAngle', 'Distance', 'BBType',
            'HC_X', 'HC_Y',
            'xBA', 'xSLG', 'xwOBA', 'wOBAval', 'wOBAdom', 'RunExp',
            'BatSpeed', 'SwingLength',
            'AttackAngle', 'AttackDirection', 'SwingPathTilt',
            'Intercept_X', 'Intercept_Y',
            'PitchID', 'Barrel',
        ]
        out = out[final_columns]
        # Statcast returns rows in pitch-count desc order; sort chronologically
        # by Game Date then PitchID (game_pk_AB_pitch with zero-padding sorts correctly)
        return out.sort_values(['Game Date', 'PitchID'], kind='stable').reset_index(drop=True)

    # ── Orchestration ─────────────────────────────────────────────────────────

    def download_hitters(self, mlb_ids, start_date, end_date):
        """Loop over MLBIDs, write one CSV per batter. Returns list of (id, path)."""
        results = []
        for i, mlb_id in enumerate(mlb_ids):
            raw_df = self.download_hitter_raw(mlb_id, start_date, end_date)

            if raw_df is None or raw_df.empty:
                fallback = self.get_player_name(mlb_id) or f"MLBID {mlb_id}"
                print(f"  Skipped {fallback} (no pitches in range)")
                results.append((mlb_id, None))
            else:
                out_df = self.transform(raw_df)

                first_name = out_df['Batter'].iloc[0] if not out_df.empty else ''
                batter_name = first_name or self.get_player_name(mlb_id)
                name_stem = self.safe_filename(batter_name) if batter_name else ''
                # Include mlb_id so two batters whose display names normalize to
                # the same stem don't silently overwrite each other's CSV.
                stem = f"{name_stem}_{mlb_id}" if name_stem else str(mlb_id)

                output_path = os.path.join(self.download_dir, f"{stem}.csv")
                out_df.to_csv(output_path, index=False)
                print(f"  Saved {len(out_df)} pitches → {output_path}")
                results.append((mlb_id, output_path))

            # Polite delay between Savant requests
            if i < len(mlb_ids) - 1:
                time.sleep(1)

        return results


def main():
    # ── Settings (edit directly or override via CLI) ──
    mlb_ids    = [682928, 671277, 691781, 678554, 683083, 677588, 678391, 686452, 660688, 695734, 686894, 695578, 696285]                  # e.g., [665487, 624413]
    start_date = "2025-03-27"
    end_date   = "2025-11-01"

    parser = argparse.ArgumentParser(description='Download pitch-level data for hitters from Baseball Savant')
    parser.add_argument('--mlb-ids', default=None,
                        help='Comma-separated MLB player IDs (e.g., "665487,624413")')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',   default=None, help='End date YYYY-MM-DD')
    args = parser.parse_args()

    if args.mlb_ids is not None:
        mlb_ids = [int(s.strip()) for s in args.mlb_ids.split(',') if s.strip()]
    if args.start is not None:
        start_date = args.start
    if args.end is not None:
        end_date = args.end

    if not mlb_ids:
        print("Error: provide MLB IDs via --mlb-ids or set them in the script")
        return

    downloader = BaseballSavantHitterDownloader()
    downloader.download_hitters(mlb_ids, start_date, end_date)


if __name__ == "__main__":
    main()
