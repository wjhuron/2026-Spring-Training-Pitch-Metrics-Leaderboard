# Runnable as a file from any directory (IDE run buttons included):
# put the repo root on sys.path before the intra-repo package imports.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import requests
import pandas as pd
import numpy as np
import os
from math import atan, atan2, sqrt, pi
import time
import json
from io import StringIO


# ── Weather-adjustment constants and helpers ──────────────────────────────────

STANDARD_RHO = 1.195  # kg/m³ reference: sea level at ~72.4°F (101325 Pa / 287.05 / 295.6 K)

# Density→movement exponent. Calibrated 2026-07-02 from our own data
# (scripts/research/misc/weather_exponent_calibration.py): within-pitcher fixed-effects
# fit of log|movement| on log(rho) over 307-364k pitches / 1,298 games gave
# e = 1.01 (IVB), 1.11 (HB), ~1.15 (total, month-controlled) — matching
# Nathan trajectory theory (~1.0-1.1) and rejecting the original 2/3.
# 1.05 is the adopted middle. Historical sheet x-columns were backfilled to
# this exponent on 2026-07-02 (scripts/builders/backfill_weather_exponent.py);
# per-game density inputs live in data/game_weather_rs.json.
WEATHER_EXPONENT = 1.05

# Fallback temperature when the feed omits weather.temp, and the clamp used
# when the roof is closed / dome (feeds occasionally report the OUTDOOR temp
# for closed-roof games, which would inject a spurious density error).
# Elevation is ~90% of the density signal at altitude parks, so an assumed
# indoor-ish temperature beats silently skipping the adjustment entirely.
DEFAULT_TEMP_F = 70.0
ROOF_CLOSED_TEMP_F = 72.0
ROOF_CLOSED_CONDITIONS = {'dome', 'roof closed'}

# MLB's live feed omits venue.location.elevation for most international
# parks, so without an override Mexico City games look like sea level and
# IVB/HB go un-adjusted despite ~7,350 ft of thin air.
VENUE_ELEVATION_FT_OVERRIDE = {
    5340: 7349,  # Estadio Alfredo Harp Helú, Mexico City
    2493: 1765,  # Estadio de Béisbol Monterrey, Monterrey
    5150: 125,   # Gocheok Sky Dome, Seoul
    5160: 144,   # Tokyo Dome, Tokyo
    4895: 16,    # London Stadium, London
    5355: 3010,  # Las Vegas Ballpark, Summerlin (Athletics temp. home) — high on
                 # the valley's western bench; MLB feed omits the elevation
    5445: 994,   # Field of Dreams, Dyersville IA (2026 special event) — feed
                 # returns an empty venue.location; DEM at the park's own
                 # coordinates (USGS 994 ft, open-elevation cross-check 997)
    # ── Triple-A: ALL 30 parks ────────────────────────────────────────────
    # The MiLB feed omits elevation for every one of them, so an unlisted park
    # silently yields rho=None -> factor 1.0 -> completely un-adjusted xIVB/xHB.
    # Listing the full league (not just the IL parks ROC visits) is what keeps
    # the NEW tab covered: arms new to the org arrive with PCL and IL road
    # data from anywhere, and Albuquerque at 5,118 ft is a Coors-sized miss.
    #
    # Values below are digital-elevation-model readings at each park's own
    # coordinates. The ten that were hand-entered first are kept as entered;
    # they agreed with the DEM to within 51 ft (median ~10), which is what
    # validates the rest — 51 ft is 0.18% density, ~0.02% of a break.
    #
    # NOTE 2529 (Sutter Health Park) is also the Athletics' MLB home in 2026.
    # No conflict: the override is consulted only when the feed's own
    # elevation is absent, and MLB games there report it (24 ft).
    #
    # Pacific Coast League
    2683: 5118,  # Isotopes Park, Albuquerque NM      <- thin air, Coors-like
    6135: 4892,  # The Ballpark at America First Square, South Jordan UT
    3789: 4505,  # Greater Nevada Field, Reno NV
    4669: 3780,  # Southwest University Park, El Paso TX
    5355: 3010,  # Las Vegas Ballpark, Summerlin NV (also Athletics temp. home)
    2843: 1191,  # Chickasaw Bricktown Ballpark, Oklahoma City OK
    4271: 1150,  # Werner Park, Papillion NE (Omaha)
    2528: 673,   # Dell Diamond, Round Rock TX
    2745: 338,   # Cheney Stadium, Tacoma WA
    5421: 85,    # Constellation Field, Sugar Land TX
    2529: 39,    # Sutter Health Park, Sacramento CA
    # International League
    2797: 920,   # PNC Field, Moosic PA (Scranton/WB)
    3810: 1076,  # Gwinnett Field, Lawrenceville GA
    2844: 807,   # Principal Park, Des Moines IA (Iowa)
    5472: 741,   # CHS Field, St. Paul MN
    3970: 738,   # Huntington Park, Columbus OH
    4670: 730,   # Truist Field, Charlotte NC
    2858: 719,   # Victory Field, Indianapolis IN
    2756: 600,   # Sahlen Field, Buffalo NY
    2767: 581,   # Fifth Third Field, Toledo OH
    2773: 505,   # ESL Ballpark, Rochester NY (ROC home)
    5410: 475,   # Polar Park, Worcester MA
    2724: 469,   # Louisville Slugger Field, Louisville KY
    4715: 420,   # First Horizon Park, Nashville TN
    2823: 400,   # NBT Bank Stadium, Syracuse NY
    2541: 364,   # Durham Bulls Athletic Park, Durham NC
    3230: 350,   # Coca-Cola Park, Allentown PA
    2542: 253,   # AutoZone Park, Memphis TN
    2852: 20,    # VyStar Ballpark, Jacksonville FL
    2781: 16,    # Harbor Park, Norfolk VA
    # ── Non-AAA sites that turn up in the NEW tab ─────────────────────────
    # Spring-training complexes and college parks: rehab outings, spring
    # games, and pre-draft looks all land here and the feed omits elevation
    # for these too.
    2603: 1217,  # Surprise Stadium, Surprise AZ
    4629: 1184,  # Sloan Park, Mesa AZ
    2912: 1158,  # Medlar Field at Lubrano Park, State College PA
    4965: 1070,  # Kendrick Family Ballpark, Granville WV (WVU)
    2738: 889,   # Eastwood Field, Niles OH
    2735: 518,   # Journey Bank Ballpark, Williamsport PA
    2511: 164,   # Joker Marchant Stadium, Lakeland FL
    2842: 52,    # Trenton Thunder Ballpark, Trenton NJ
    2508: 39,    # Ed Smith Stadium, Sarasota FL
    2700: 33,    # BayCare Ballpark, Clearwater FL
}


def compute_air_density(elevation_ft, temp_f):
    """Air density (kg/m³) from elevation and temperature using barometric formula."""
    elevation_m = elevation_ft * 0.3048
    P = 101325 * (1 - 2.2558e-5 * elevation_m) ** 5.2556
    T = (temp_f - 32) * 5 / 9 + 273.15
    return P / (287.05 * T)


def compute_weather_adj_factor(rho_game):
    """Density-only adjustment factor for pitch movement.

    xIndVrtBrk = rawIVB × factor,  xHorzBrk = rawHB × factor.

    Uses ρ^WEATHER_EXPONENT scaling (1.05, empirically calibrated — see the
    constant's comment). Magnus force ∝ ρ, and thinner air also means less
    drag so the ball arrives SOONER: both effects compound, which is why the
    exponent is ~1, not the old 2/3's "partial compensation" story.

    Returns 1.0 when inputs are missing or invalid.
    """
    if not rho_game or rho_game <= 0:
        return 1.0

    return (STANDARD_RHO / rho_game) ** WEATHER_EXPONENT


class BaseballSavantFocusedDownloader:
    def __init__(self, download_dir=os.path.expanduser("~/Downloads")):
        """
        Initialize the Baseball Savant downloader with a focus on individual game data
        """
        self.download_dir = download_dir
        self.session = requests.Session()

        # Set headers to mimic browser request - important for not getting blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://baseballsavant.mlb.com/gamefeed'
        })

        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Initialize team mappings
        self._initialize_team_mappings()

    def _record_game_weather(self, game_pk, info):
        """Append/refresh one game's density inputs in data/game_weather_rs.json.

        Keyed by game_pk (str, JSON keys are strings). Small file (~2.5k games
        a season), so read-modify-write per game is fine."""
        sidecar = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'game_weather_rs.json')
        try:
            with open(sidecar) as f:
                store = json.load(f)
        except (FileNotFoundError, ValueError):
            store = {}
        store[str(game_pk)] = info
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        with open(sidecar, 'w') as f:
            json.dump(store, f, indent=0, sort_keys=True)

    def _initialize_team_mappings(self):
        """Initialize dictionaries for team name/abbreviation/ID mappings"""
        # Team name to abbreviation mapping
        self.team_name_to_abbrev = {
            "Arizona Diamondbacks": "ARI",
            "Athletics": "ATH",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KCR",
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SDP",
            "San Francisco Giants": "SFG",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TBR",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH",
        }

        # Statcast Search abbreviations that differ from ours (all others pass through)
        self.team_abbrev_to_statcast = {
            "ATH": "OAK",
            "KCR": "KC",
            "SDP": "SD",
            "SFG": "SF",
            "TBR": "TB",
        }

        # Team abbreviation to team ID mapping for MLB Stats API
        self.team_abbrev_to_id = {
            "ARI": 109,  # Arizona Diamondbacks
            "ATH": 133,  # Athletics
            "ATL": 144,  # Atlanta Braves
            "BAL": 110,  # Baltimore Orioles
            "BOS": 111,  # Boston Red Sox
            "CHC": 112,  # Chicago Cubs
            "CWS": 145,  # Chicago White Sox
            "CIN": 113,  # Cincinnati Reds
            "CLE": 114,  # Cleveland Guardians
            "COL": 115,  # Colorado Rockies
            "DET": 116,  # Detroit Tigers
            "HOU": 117,  # Houston Astros
            "KCR": 118,  # Kansas City Royals
            "LAA": 108,  # Los Angeles Angels
            "LAD": 119,  # Los Angeles Dodgers
            "MIA": 146,  # Miami Marlins
            "MIL": 158,  # Milwaukee Brewers
            "MIN": 142,  # Minnesota Twins
            "NYM": 121,  # New York Mets
            "NYY": 147,  # New York Yankees
            "PHI": 143,  # Philadelphia Phillies
            "PIT": 134,  # Pittsburgh Pirates
            "SDP": 135,  # San Diego Padres
            "SFG": 137,  # San Francisco Giants
            "SEA": 136,  # Seattle Mariners
            "STL": 138,  # St. Louis Cardinals
            "TBR": 139,  # Tampa Bay Rays
            "TEX": 140,  # Texas Rangers
            "TOR": 141,  # Toronto Blue Jays
            "WSH": 120,  # Washington Nationals
        }

    def get_player_name(self, player_id, player_full_name, game_data):
        """
        Get a player name in 'Last, First' format using the API's own
        firstName/lastName fields from gameData.players.

        Works for both pitchers and batters.

        The API already knows the correct split for multi-word last names
        (e.g., Woods Richardson), suffixes (Jr., III), and prefixes (De La Cruz).

        Falls back to fullName string parsing only if the player data is missing.
        """
        # Look up the player in gameData.players (keyed as "ID{player_id}")
        player_key = f"ID{player_id}"
        player_info = game_data.get('players', {}).get(player_key, {})

        # Best option: the API provides lastFirstName already formatted
        last_first = player_info.get('lastFirstName', '')
        if last_first:
            return last_first.strip()  # API's lastFirstName can carry a trailing space

        # Second option: build it from separate firstName and lastName fields
        first_name = player_info.get('firstName', '')
        last_name = player_info.get('lastName', '')
        if first_name and last_name:
            return f"{last_name}, {first_name}"

        # Fallback: parse the fullName string (least reliable)
        return self._parse_player_name(player_full_name)

    # ── Mid-PA substitution helpers ─────────────────────────────────────
    # A pitcher or batter replaced mid-plate-appearance (injury/ejection) means
    # the pitches thrown/faced before the change belong to the OUTGOING player,
    # not the of-record finisher the matchup reports. These identify the two
    # substitution kinds that move pitches (pitching change; pinch-hitter). A
    # pinch-RUNNER changes a baserunner, not the batter, so it is excluded.
    @staticmethod
    def _is_pitching_sub(e):
        return (e.get('type') == 'action'
                and (e.get('details', {}) or {}).get('eventType') == 'pitching_substitution')

    @staticmethod
    def _is_pinch_hitter(e):
        d = (e.get('details', {}) or {})
        return (e.get('type') == 'action' and d.get('eventType') == 'offensive_substitution'
                and 'pinch-hitter' in (d.get('description', '') or '').lower())

    @staticmethod
    def _throws_of(player_id, game_data):
        pi = game_data.get('players', {}).get(f"ID{player_id}", {})
        return (pi.get('pitchHand', {}) or {}).get('code', 'R')

    @staticmethod
    def _batside_of(player_id, game_data, p_throws):
        """Registered bat side; switch hitters bat opposite the pitcher's hand."""
        code = (game_data.get('players', {}).get(f"ID{player_id}", {})
                .get('batSide', {}) or {}).get('code', '')
        if code == 'S':
            return 'L' if p_throws == 'R' else 'R'
        return code

    @staticmethod
    def _replaced_id_from_desc(e, game_data):
        """Outgoing player id parsed from a substitution description
        ("... X replaces Y."). The feed sometimes omits the replacedPlayer field on
        mid-PA changes (e.g. an injury-driven pitching change), which would otherwise
        leave the pre-substitution pitches credited to the finisher."""
        import unicodedata
        low = ((e.get('details', {}) or {}).get('description', '') or '').lower()
        if 'replaces' not in low:
            return None
        name = low.split('replaces', 1)[1].strip().rstrip('.')

        def _toks(n):
            s = unicodedata.normalize('NFKD', str(n)).encode('ascii', 'ignore').decode().lower()
            for ch in ',.-':
                s = s.replace(ch, ' ')
            return frozenset(p for p in s.split() if len(p) > 1 and p not in ('jr', 'sr', 'ii', 'iii', 'iv'))

        want = _toks(name)
        if not want:
            return None
        for _p in game_data.get('players', {}).values():
            if _toks(_p.get('fullName', '')) == want:
                return _p.get('id')
        return None

    def _parse_player_name(self, full_name):
        """
        Fallback parser for player names when API player data is unavailable.
        Handles suffixes and common multi-word last name prefixes.
        """
        if not full_name or ' ' not in full_name:
            return full_name

        # Suffixes to strip and reattach to the last name
        suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}

        parts = full_name.split()

        # Check for a trailing suffix
        suffix = ''
        if len(parts) > 2 and parts[-1].lower().rstrip('.') in {s.rstrip('.') for s in suffixes}:
            suffix = ' ' + parts.pop()

        # Common last-name prefixes (case-insensitive check)
        last_name_prefixes = {
            'de', 'la', 'da', 'del', 'della', 'di', 'du', 'dos', 'das',
            'van', 'von', 'le', 'el', 'al', 'st.', 'san', 'santa',
            'los', 'las', 'des', "o'",
        }

        if len(parts) <= 2:
            # Simple "First Last" case
            first_name = parts[0]
            last_name = parts[1] + suffix
            return f"{last_name}, {first_name}"

        # For 3+ parts, scan from the second token forward to find where
        # the last name begins. A token that is a known prefix or is followed
        # only by one more token (the final surname word) starts the last name.
        # We assume the first token is always part of the first name.
        last_name_start = None
        for i in range(1, len(parts)):
            if parts[i].lower().rstrip('.') in last_name_prefixes:
                last_name_start = i
                break

        # If no prefix found, assume everything except the first token is the last name.
        # This handles cases like "Simeon Woods Richardson" where "Woods Richardson"
        # is the last name but has no standard prefix.
        if last_name_start is None:
            last_name_start = 1

        first_name = ' '.join(parts[:last_name_start])
        last_name = ' '.join(parts[last_name_start:]) + suffix

        # Safety check: don't produce an empty first or last name
        if not first_name or not last_name:
            return full_name

        return f"{last_name}, {first_name}"

    def get_team_abbreviation(self, team_name):
        """
        Get the team abbreviation from the full team name
        """
        return self.team_name_to_abbrev.get(team_name, team_name)

    @staticmethod
    def normalize_aaa_labels(df):
        """If the download involves Rochester Red Wings, collapse all team
        labels into the {'ROC', 'AAA'} pair.

        The ROC tab in NL 2026 expects PTeam='ROC' for Rochester pitchers and
        BTeam='AAA' for any opponent batter (regardless of which AAA team they
        actually play for). Same logic for the AAA tab in mirror.

        Detection: triggered when 'Rochester Red Wings' appears in PTeam or
        BTeam. AAA games never involve MLB teams, so it's safe to map every
        non-Rochester team in the download to 'AAA'.

        No-op for MLB downloads (Rochester won't appear in the data).
        """
        if df is None or len(df) == 0:
            return df
        if 'PTeam' not in df.columns or 'BTeam' not in df.columns:
            return df

        ROCHESTER = 'Rochester Red Wings'
        has_rochester = (
            (df['PTeam'] == ROCHESTER).any()
            or (df['BTeam'] == ROCHESTER).any()
        )
        if not has_rochester:
            return df

        def remap(team):
            if team is None:
                return team
            return 'ROC' if team == ROCHESTER else 'AAA'

        df['PTeam'] = df['PTeam'].map(remap)
        df['BTeam'] = df['BTeam'].map(remap)
        return df

    def get_team_id(self, team_abbrev):
        """
        Get the MLB team ID from the team abbreviation
        """
        return self.team_abbrev_to_id.get(team_abbrev.upper())

    def get_statcast_team_abbrev(self, team_abbrev):
        """
        Convert internal team abbreviation to Statcast Search abbreviation
        """
        return self.team_abbrev_to_statcast.get(team_abbrev.upper(), team_abbrev.upper())

    def is_barrel(self, exit_velo, launch_angle):
        """
        Estimate whether a batted ball is a barrel using the Statcast formula.

        A barrel (launch_speed_angle = 6) requires:
        - launch_angle in [8, 50]
        - exit_velo >= 98
        - exit_velo * 1.5 - launch_angle >= 117
        - exit_velo + launch_angle >= 124

        Returns: True if barrel, False otherwise
        """
        if exit_velo is None or launch_angle is None:
            return False

        try:
            ev = float(exit_velo)
            la = float(launch_angle)
        except (ValueError, TypeError):
            return False

        return (la >= 8 and la <= 50 and ev >= 98 and ev * 1.5 - la >= 117 and ev + la >= 124)

    def spin_axis_to_tilt(self, spin_axis):
        """
        Convert spin axis (degrees) to tilt (clock format as string).

        From Driveline:
        - 180 degrees = 12:00 (pure backspin)
        - Each hour = 30 degrees
        - Each minute = 0.5 degrees

        Examples:
        - 180 -> 12:00 (pure backspin, 4-seam fastball)
        - 215 -> 1:10 (typical RHP fastball)
        - 0 or 360 -> 6:00 (pure topspin, 12-6 curveball)
        - 270 -> 3:00 (pure sidespin)
        - 90 -> 9:00 (gyro spin)

        Returns: Tilt as string (e.g., "1:10") or None if data is missing
        """
        if spin_axis is None:
            return None

        # Convert spin axis to decimal clock hours
        # 180 degrees = 12:00, so divide by 30 and subtract 6
        decimal_hours = spin_axis / 30 - 6

        # Handle wrap-around for clock (0-12 range)
        if decimal_hours < 0:
            decimal_hours += 12
        elif decimal_hours >= 12:
            decimal_hours -= 12

        # Extract hours and minutes
        hours = int(decimal_hours)
        minutes = int((decimal_hours - hours) * 60)

        # Handle 0 -> 12 for display
        if hours == 0:
            hours = 12

        return f"{hours}:{minutes:02d}"

    def calculate_break_tilt(self, ivb, hb):
        """
        Calculate break tilt from IndVertBrk and HorzBrk.

        Break tilt is the direction of the actual pitch movement vector,
        as opposed to release tilt which is the spin axis direction.
        The difference between the two reveals seam-shifted wake (SSW).

        Uses atan2(HorzBrk, IndVertBrk) to get the angle of the movement
        vector measured clockwise from 12:00 (from the pitcher's perspective).

        In MLB API / Trackman data, positive HorzBrk = arm-side movement,
        which maps to clockwise rotation on the tilt clock.

        Returns: Break tilt as clock string (e.g., "1:30") or None if data is missing
        """
        if ivb is None or hb is None:
            return None

        try:
            ivb = float(ivb)
            hb = float(hb)
        except (ValueError, TypeError):
            return None

        # Handle zero movement edge case
        if ivb == 0 and hb == 0:
            return None

        # atan2(HorzBrk, IndVertBrk) gives angle from 12:00 (pure rise),
        # with positive values going clockwise (arm-side)
        angle_rad = atan2(hb, ivb)
        angle_deg = angle_rad * (180 / pi)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        # Convert degrees to clock notation
        # 360 degrees = 12 hours = 720 minutes
        total_minutes = angle_deg / 360 * 720
        hours = int(total_minutes // 60) % 12
        minutes = int(total_minutes % 60)

        if hours == 0:
            hours = 12

        return f"{hours}:{minutes:02d}"

    def calculate_approach_angles(self, vy0, vz0, vx0, ay, az, ax):
        """
        Calculate VAA (Vertical Approach Angle) and HAA (Horizontal Approach Angle).

        Formulas from Fangraphs:
        https://blogs.fangraphs.com/a-visualized-primer-on-vertical-approach-angle-vaa/

        vy_f = -sqrt(vy0^2 - (2 * ay * (y0 - yf)))
        t = (vy_f - vy0) / ay
        vz_f = vz0 + (az * t)
        VAA = -arctan(vz_f/vy_f) * (180 / pi)

        vx_f = vx0 + (ax * t)
        HAA = -arctan(vx_f/vy_f) * (180 / pi)

        Where:
        - y0 = 50 (release point at y=50 feet)
        - yf = 17/12 (front of home plate, 17 inches in feet)

        Returns: (VAA, HAA) or (None, None) if data is missing
        """
        if any(v is None for v in [vy0, vz0, vx0, ay, az, ax]):
            return None, None

        y0 = 50
        yf = 17 / 12

        try:
            discriminant = vy0 ** 2 - (2 * ay * (y0 - yf))
            if discriminant < 0:
                return None, None
            vy_f = -sqrt(discriminant)
            t = (vy_f - vy0) / ay
            vz_f = vz0 + (az * t)
            vx_f = vx0 + (ax * t)

            VAA = -atan(vz_f / vy_f) * (180 / pi)
            HAA = -atan(vx_f / vy_f) * (180 / pi)
            return VAA, HAA

        except (ValueError, ZeroDivisionError):
            return None, None

    def simplify_description(self, description):
        """
        Simplify pitch descriptions into human-readable labels.

        Grouped:
        - In Play: covers all hit-into-play variants
        - Swinging Strike: swinging strike blocked, foul tips, AND swinging
          pitchout (a swing-and-miss at a pitchout is a whiff)
        - Foul: includes foul pitchout (a foul off a pitchout is a foul/strike)
        - Ball: includes blocked balls

        Standalone (kept granular for analysis):
        - Called Strike, Hit By Pitch, Foul Bunt, Bunt Foul Tip, Missed Bunt,
          Intent Ball, Pitchout. NOTE: 'Pitchout' (a taken pitchout) stays its
          own label but is treated as a ball everywhere in the metrics (it's in
          the not-a-strike set and not a swing).

        Returns simplified description or original if no mapping found
        """
        if not description:
            return description

        # Map from live API human-readable descriptions to simplified labels
        DESCRIPTION_MAP = {
            # Grouped: In Play
            'in play, out(s)': 'In Play',
            'in play, no out': 'In Play',
            'in play, run(s)': 'In Play',
            # Grouped: Swinging Strike (a swinging pitchout is a whiff)
            'swinging strike': 'Swinging Strike',
            'swinging strike (blocked)': 'Swinging Strike',
            'foul tip': 'Swinging Strike',
            'swinging pitchout': 'Swinging Strike',
            # Grouped: Foul (a foul off a pitchout is a foul)
            'foul': 'Foul',
            'foul pitchout': 'Foul',
            # Grouped: Ball
            'ball': 'Ball',
            'ball in dirt': 'Ball',
            # Standalone
            'called strike': 'Called Strike',
            'hit by pitch': 'Hit By Pitch',
            'foul bunt': 'Foul Bunt',
            'bunt foul tip': 'Bunt Foul Tip',
            'missed bunt': 'Missed Bunt',
            'intent ball': 'Intent Ball',
            'pitchout': 'Pitchout',   # taken pitchout: kept as its own label, counts as a ball
        }

        return DESCRIPTION_MAP.get(description.lower(), description)


    def get_team_games(self, team_abbrev, start_date, end_date):
        """
        Get all games for a specific team within a date range
        """
        print(f"Finding games for {team_abbrev} from {start_date} to {end_date}...")

        # Convert team abbreviation to team ID
        team_id = self.get_team_id(team_abbrev)
        if not team_id:
            print(f"Error: Invalid team abbreviation '{team_abbrev}'")
            return []

        # MLB Stats API endpoint for schedule
        url = "https://statsapi.mlb.com/api/v1/schedule"

        params = {
            "teamId": team_id,
            "startDate": start_date,
            "endDate": end_date,
            "sportId": 1,  # MLB
            "gameType": "E,S,R,F,D,L,W"  # Spring Training and Regular Season
        }

        try:
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                return []

            data = response.json()

            # Extract game PKs — FINAL games only. Nothing previously
            # enforced the "only pull finished games" workflow: the feed
            # happily serves a partial allPlays for an in-progress game, and
            # a truncated pull would never be revisited once its rows are in
            # the Sheets. (The 2026-07-13 completeness audit found NO such
            # truncations in practice — the sheets were complete — but the
            # guard makes the intended workflow enforced rather than
            # habitual.) Explicit --game-id mode is NOT filtered (live pulls
            # there are a feature).
            game_pks = []
            skipped = []

            if "dates" in data:
                for date_data in data["dates"]:
                    if "games" in date_data:
                        for game in date_data["games"]:
                            state = (game.get("status", {}) or {}).get("abstractGameState", "")
                            if state == "Final":
                                game_pks.append(game["gamePk"])
                            else:
                                skipped.append((game["gamePk"], state or "Unknown"))

            print(f"Found {len(game_pks)} FINAL games for {team_abbrev} in the specified date range.")
            if skipped:
                print(f"  SKIPPED {len(skipped)} non-final game(s) — re-run after they finish:")
                for pk, st in skipped:
                    print(f"    gamePk {pk} ({st})")
            return game_pks

        except Exception as e:
            print(f"Error getting team games: {str(e)}")
            return []

    def download_game_data(self, game_pk, filter_team=None, filter_player_id=None):
        """
        Download pitch-by-pitch data using the MLB Stats API
        Returns the DataFrame directly rather than saving to a file

        filter_player_id: optional MLB player ID — keep only plays where that
        player was the pitcher or the batter.
        """
        print(f"Downloading data for game {game_pk}...")

        # Using the MLB Stats API endpoint
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

        try:
            response = self.session.get(url, timeout=60)

            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return None

            # Parse the JSON response
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("Error: Failed to parse JSON response")
                print(f"Response: {response.text[:500]}")
                return None

            # Check if we received valid data
            if not data or 'liveData' not in data or 'plays' not in data['liveData']:
                print("Error: No play data in response")
                print(f"Response keys: {list(data.keys()) if data else 'None'}")
                return None

            # Get game information
            game_data = data.get('gameData', {})
            game_date = game_data.get('datetime', {}).get('officialDate', '')
            # Get team information and convert to abbreviations
            teams = game_data.get('teams', {})
            home_team_name = teams.get('home', {}).get('name', 'Unknown')
            away_team_name = teams.get('away', {}).get('name', 'Unknown')
            home_team = self.get_team_abbreviation(home_team_name)
            away_team = self.get_team_abbreviation(away_team_name)

            # ── Weather & venue data for movement adjustment ──
            weather = game_data.get('weather', {})
            venue = game_data.get('venue', {})
            venue_location = venue.get('location', {})
            venue_field_info = venue.get('fieldInfo', {})

            game_temp = None
            game_elevation = None
            game_rho = None

            try:
                temp_str = weather.get('temp')
                elev_val = venue_location.get('elevation')
                if elev_val is None:
                    elev_val = VENUE_ELEVATION_FT_OVERRIDE.get(venue.get('id'))

                if temp_str is not None:
                    game_temp = float(temp_str)
                if elev_val is not None:
                    game_elevation = float(elev_val)

                # Roof closed / dome: clamp to indoor temp. Feeds sometimes
                # report the outdoor reading (e.g. 104°F Phoenix with the
                # roof shut), which would inject a ~4% spurious density error.
                condition = (weather.get('condition') or '').strip().lower()
                if condition in ROOF_CLOSED_CONDITIONS:
                    game_temp = ROOF_CLOSED_TEMP_F

                # Air density. Missing temp falls back to DEFAULT_TEMP_F
                # rather than skipping the adjustment: elevation is ~90% of
                # the signal at altitude parks, and a silent factor of 1.0 at
                # Coors is a far bigger error than an assumed 70°F (this
                # failure mode bit the Las Vegas homestand).
                if game_elevation is not None:
                    game_rho = compute_air_density(
                        game_elevation,
                        game_temp if game_temp is not None else DEFAULT_TEMP_F)
            except (ValueError, TypeError):
                pass  # any parse failure → defaults (no adjustment)

            # Persist per-game density inputs to a sidecar so adjustments are
            # auditable and re-derivable if the exponent ever changes (the
            # sheet x-columns bake the factor in; without this the LV-style
            # forensic rebuild from feeds is the only recovery path).
            try:
                self._record_game_weather(game_pk, {
                    'venueId': venue.get('id'),
                    'condition': weather.get('condition'),
                    'tempF': game_temp,
                    'elevationFt': game_elevation,
                    'rho': round(game_rho, 5) if game_rho else None,
                    'factor': round(compute_weather_adj_factor(game_rho), 5),
                })
            except Exception as e:
                print(f"  WARNING: could not record game weather sidecar: {e}")

            # Extract all plays
            all_plays = data['liveData']['plays'].get('allPlays', [])

            # Get boxscore teams for pitcher team detection
            boxscore_teams = data.get('liveData', {}).get('boxscore', {}).get('teams', {})
            home_players = boxscore_teams.get('home', {}).get('players', {})
            away_players = boxscore_teams.get('away', {}).get('players', {})

            # Extract pitch-by-pitch data
            pitches = []
            for play in all_plays:
                # Get batter stance
                bats = play.get('matchup', {}).get('batSide', {}).get('code', '')

                # Get pitcher information
                pitcher_id = play.get('matchup', {}).get('pitcher', {}).get('id', 0)
                pitcher_full_name = play.get('matchup', {}).get('pitcher', {}).get('fullName', '')

                # Use API player data for correct name formatting
                pitcher_name = self.get_player_name(pitcher_id, pitcher_full_name, game_data)

                # Get batter information
                batter_id = play.get('matchup', {}).get('batter', {}).get('id', 0)
                batter_full_name = play.get('matchup', {}).get('batter', {}).get('fullName', '')
                batter_name = self.get_player_name(batter_id, batter_full_name, game_data)

                # Determine which team the pitcher belongs to
                pitcher_team = None

                # Check if pitcher is on home team
                for player_info in home_players.values():
                    if player_info.get('person', {}).get('id') == pitcher_id:
                        pitcher_team = home_team
                        break

                # If not found, check away team
                if pitcher_team is None:
                    for player_info in away_players.values():
                        if player_info.get('person', {}).get('id') == pitcher_id:
                            pitcher_team = away_team
                            break

                # Determine which team the batter belongs to
                b_team = None

                for player_info in home_players.values():
                    if player_info.get('person', {}).get('id') == batter_id:
                        b_team = home_team
                        break

                if b_team is None:
                    for player_info in away_players.values():
                        if player_info.get('person', {}).get('id') == batter_id:
                            b_team = away_team
                            break

                # If filtering by team and this pitcher is not from that team, skip this play
                if filter_team and pitcher_team != filter_team:
                    continue

                # If filtering by player ID, keep only plays where that player pitched or batted
                if filter_player_id and pitcher_id != filter_player_id and batter_id != filter_player_id:
                    continue

                # Get pitch hand from the matchup data
                p_throws = play.get('matchup', {}).get('pitchHand', {}).get('code', '')

                # Fall back to player data only if matchup data is missing
                if not p_throws:
                    player_key = f"ID{pitcher_id}"
                    player_info = game_data.get('players', {}).get(player_key, {})
                    p_throws = player_info.get('pitchHand', {}).get('code', 'R')

                # Get the PA result event (single, home_run, strikeout, etc.)
                pa_event = play.get('result', {}).get('event', '')

                # Get the at-bat number (1-indexed to match Statcast Search)
                at_bat_number = play.get('atBatIndex', 0) + 1

                # Process each pitch in the at-bat
                # Track the count before each pitch
                play_events = play.get('playEvents', [])
                pre_pitch_balls = 0
                pre_pitch_strikes = 0

                # Pre-compute which pitch index is the last actual pitch in this PA
                last_pitch_idx = None
                for i in range(len(play_events) - 1, -1, -1):
                    if play_events[i].get('isPitch', False):
                        last_pitch_idx = i
                        break

                # Mid-PA substitution reattribution: pitcher_id/batter_id above are
                # the finishers. Seed the "current" ids with whoever STARTED the PA
                # (the outgoing player of the first mid-PA sub, if any), then advance
                # them as substitution events are hit inside the pitch loop so each
                # pitch is credited to the player actually in at the time.
                start_pitcher_id = pitcher_id
                for e in play_events:
                    if self._is_pitching_sub(e):
                        rp = (e.get('replacedPlayer') or {}).get('id') \
                            or self._replaced_id_from_desc(e, game_data)
                        if rp:
                            start_pitcher_id = rp
                        break
                start_batter_id = batter_id
                for e in play_events:
                    if self._is_pinch_hitter(e):
                        rp = (e.get('replacedPlayer') or {}).get('id') \
                            or self._replaced_id_from_desc(e, game_data)
                        if rp:
                            start_batter_id = rp
                        break
                cur_pitcher_id = start_pitcher_id
                cur_batter_id = start_batter_id

                for pitch_idx, pitch in enumerate(play_events):
                    # Advance current pitcher/batter on in-sequence substitutions.
                    if pitch.get('type') == 'action':
                        if self._is_pitching_sub(pitch):
                            _np = (pitch.get('player') or {}).get('id')
                            if _np:
                                cur_pitcher_id = _np
                        elif self._is_pinch_hitter(pitch):
                            _nb = (pitch.get('player') or {}).get('id')
                            if _nb:
                                cur_batter_id = _nb
                    # Automatic ball/strike (pitcher/catcher/batter pitch-timer
                    # violation, or intentional walk) arrives as a no_pitch event
                    # with no pitch thrown. Advance the running count so the NEXT
                    # pitch's stored count reflects it (the feed re-syncs the count
                    # after each real pitch, so this never double-counts).
                    if not pitch.get('isPitch', False):
                        _ad = ((pitch.get('details', {}) or {}).get('description', '') or '').lower()
                        if 'automatic ball' in _ad:
                            pre_pitch_balls += 1
                        elif 'automatic strike' in _ad:
                            pre_pitch_strikes += 1
                    if pitch.get('isPitch', False):
                        # Get velocity and acceleration data for angle calculations
                        coords = pitch.get('pitchData', {}).get('coordinates', {})
                        vx0 = coords.get('vX0')
                        vy0 = coords.get('vY0')
                        vz0 = coords.get('vZ0')
                        ax = coords.get('aX')
                        ay = coords.get('aY')
                        az = coords.get('aZ')
                        release_pos_y = coords.get('y0')

                        # Get break values for break tilt calculation
                        ivb = pitch.get('pitchData', {}).get('breaks', {}).get('breakVerticalInduced')
                        hb = pitch.get('pitchData', {}).get('breaks', {}).get('breakHorizontal')

                        # Calculate release tilt from spin axis
                        spin_direction = pitch.get('pitchData', {}).get('breaks', {}).get('spinDirection')
                        release_tilt = self.spin_axis_to_tilt(spin_direction)

                        # Calculate observed tilt from movement vector (IVB/HB)
                        observed_tilt = self.calculate_break_tilt(ivb, hb)

                        # Calculate approach angles (at plate)
                        VAA, HAA = self.calculate_approach_angles(vy0, vz0, vx0, ay, az, ax)

                        # Count BEFORE this pitch is thrown
                        count_str = f"{pre_pitch_balls}-{pre_pitch_strikes}"

                        # Get raw pitch type code and remap KC -> CU, FO -> FS
                        raw_pitch_type = pitch.get('details', {}).get('type', {}).get('code', '')
                        pitch_type_remap = {'KC': 'CU', 'FO': 'FS'}
                        pitch_type = pitch_type_remap.get(raw_pitch_type, raw_pitch_type)

                        # Extension from the live API (pitchData.extension)
                        extension_raw = pitch.get('pitchData', {}).get('extension')
                        extension = round(extension_raw, 2) if extension_raw is not None else None

                        # Strike zone boundaries
                        sz_top = pitch.get('pitchData', {}).get('strikeZoneTop')
                        sz_bot = pitch.get('pitchData', {}).get('strikeZoneBottom')
                        plate_z = pitch.get('pitchData', {}).get('coordinates', {}).get('pZ')
                        plate_x = pitch.get('pitchData', {}).get('coordinates', {}).get('pX')

                        # Determine if this is the final pitch of the PA
                        is_final_pitch = (pitch_idx == last_pitch_idx)

                        # Get hit data (only present on batted balls)
                        hit_data = pitch.get('hitData', {}) if 'hitData' in pitch else {}

                        # Estimate barrel from exit velo / launch angle (6 = barrel on 1-6 scale)
                        ev_raw = hit_data.get('launchSpeed')
                        la_raw = hit_data.get('launchAngle')
                        estimated_barrel = 6 if self.is_barrel(ev_raw, la_raw) else None

                        # Compute weather-adjusted IVB and HB
                        pitch_velo = pitch.get('pitchData', {}).get('startSpeed')
                        adj_ivb = None
                        adj_hb = None
                        try:
                            adj_factor = compute_weather_adj_factor(game_rho)
                            if ivb is not None:
                                adj_ivb = round(ivb * adj_factor, 1)
                            if hb is not None:
                                adj_hb = round(hb * adj_factor, 1)
                        except (ValueError, TypeError):
                            adj_ivb = ivb  # fall back to raw
                            adj_hb = hb

                        # Per-pitch pitcher/batter identity + handedness. Differs
                        # from the of-record finisher ONLY on mid-PA substitutions;
                        # the team never changes (a sub is always same-team), so
                        # PTeam/BTeam stay as computed above.
                        if cur_pitcher_id == pitcher_id:
                            cp_name, cp_throws = pitcher_name, p_throws
                        else:
                            cp_throws = self._throws_of(cur_pitcher_id, game_data)
                            cp_name = self.get_player_name(cur_pitcher_id, '', game_data)
                        if cur_batter_id == batter_id:
                            # matchup.batSide is the actual side used (already resolves
                            # switch hitters); fall back to registered side only if the
                            # feed left it 'S'/blank.
                            cb_name = batter_name
                            cb_bats = bats if bats in ('L', 'R') else self._batside_of(batter_id, game_data, cp_throws)
                        else:
                            cb_name = self.get_player_name(cur_batter_id, '', game_data)
                            cb_bats = self._batside_of(cur_batter_id, game_data, cp_throws)

                        # Extract pitch data
                        pitch_data = {
                            'Game Date': game_date,
                            'PTeam': pitcher_team,
                            'Pitcher': cp_name,
                            'Throws': cp_throws,
                            'Pitch Type': pitch_type,
                            'Velocity': pitch_velo,
                            'Spin Rate': pitch.get('pitchData', {}).get('breaks', {}).get('spinRate'),
                            'RTilt': release_tilt,
                            'OTilt': observed_tilt,
                            'IndVertBrk': ivb,
                            'HorzBrk': hb,
                            'xIndVrtBrk': adj_ivb,
                            'xHorzBrk': adj_hb,
                            'RelPosZ': pitch.get('pitchData', {}).get('coordinates', {}).get('z0'),
                            'RelPosX': pitch.get('pitchData', {}).get('coordinates', {}).get('x0'),
                            'Extension': extension,
                            'ArmAngle': None,  # filled by Statcast supplement
                            'PlateZ': plate_z,
                            'PlateX': plate_x,
                            'SzTop': sz_top,
                            'SzBot': sz_bot,
                            'VAA': VAA,
                            'HAA': HAA,
                            'BTeam': b_team,
                            'Batter': cb_name,
                            'Bats': cb_bats,
                            'Count': count_str,
                            'Runners': None,  # filled by Statcast supplement (on_1b/on_2b/on_3b)
                            # about.outs is absent in the Triple-A feed (and now the
                            # archived MLB feed), so fall back to this pitch's
                            # count.outs — verified identical to about.outs (822757: 157/157).
                            'Outs': (play.get('about', {}).get('outs')
                                     if play.get('about', {}).get('outs') is not None
                                     else (pitch.get('count', {}) or {}).get('outs')),
                            'Description': self.simplify_description(pitch.get('details', {}).get('description', '')),
                            'Event': pa_event if is_final_pitch else '',
                            'ExitVelo': hit_data.get('launchSpeed'),
                            'LaunchAngle': hit_data.get('launchAngle'),
                            'Distance': hit_data.get('totalDistance'),
                            'BBType': 'bunt' if (hit_data.get('trajectory') or '').startswith('bunt_') else hit_data.get('trajectory'),
                            'HC_X': hit_data.get('coordinates', {}).get('coordX') if hit_data.get('coordinates') else None,
                            'HC_Y': hit_data.get('coordinates', {}).get('coordY') if hit_data.get('coordinates') else None,
                            'xBA': None,    # filled by Statcast supplement (estimated_ba_using_speedangle)
                            'xSLG': None,   # filled by Statcast supplement (estimated_slg_using_speedangle)
                            'xwOBA': None,  # filled by Statcast supplement (estimated_woba_using_speedangle)
                            'RunExp': None,  # filled by Statcast supplement (delta_pitcher_run_exp)
                            'Barrel': estimated_barrel,  # estimate from code_barrel formula; overwritten by Statcast launch_speed_angle if available
                            # Merge keys for Statcast Search supplement (dropped before final CSV)
                            '_game_pk': game_pk,
                            '_at_bat_number': at_bat_number,
                            '_pitch_number': pitch.get('pitchNumber'),
                            # Stable pitch identifier: game_pk + zero-padded at-bat + zero-padded pitch
                            'PitchID': f"{game_pk}_{at_bat_number:03d}_{(pitch.get('pitchNumber') or 0):02d}",
                        }

                        pitches.append(pitch_data)

                        # Update count for next pitch based on this pitch's result
                        # Get the count AFTER this pitch from the API
                        post_count = pitch.get('count', {})
                        pre_pitch_balls = post_count.get('balls', pre_pitch_balls)
                        pre_pitch_strikes = post_count.get('strikes', pre_pitch_strikes)
                        # Cap strikes at 2 for foul balls with 2 strikes
                        if pre_pitch_strikes > 2:
                            pre_pitch_strikes = 2

            # Create DataFrame from pitch data
            if not pitches:
                print("No pitch data found for this game")
                return None

            df = pd.DataFrame(pitches)

            # Format numeric columns with appropriate decimal places
            # Use pd.to_numeric to handle any non-numeric values (empty strings, etc.)
            # that can appear in early Spring Training games or incomplete data
            numeric_round_1 = ['Velocity', 'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk',
                               'ExitVelo', 'BatSpeed', 'SwingLength',
                               'AttackAngle', 'AttackDirection', 'SwingPathTilt',
                               'ArmAngle']
            numeric_round_2 = ['VAA', 'HAA', 'HC_X', 'HC_Y',
                               'RelPosZ', 'RelPosX', 'Extension']
            numeric_round_3 = ['PlateZ', 'PlateX', 'SzTop', 'SzBot',
                               'RunExp', 'xBA', 'xSLG', 'xwOBA']
            numeric_int = ['LaunchAngle', 'Distance']

            for col in numeric_round_1:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.1f}" if pd.notna(x) else '')
            for col in numeric_round_2:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
            for col in numeric_round_3:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
            for col in numeric_int:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

            # Spin Rate: round to 0 decimals, then Int64
            if 'Spin Rate' in df.columns:
                df['Spin Rate'] = pd.to_numeric(df['Spin Rate'], errors='coerce').round(0).astype('Int64')

            print(f"Successfully processed {len(df)} pitches from game {game_pk}")
            return df

        except Exception as e:
            print(f"Error downloading game data: {str(e)}")
            return None

    def download_statcast_supplement(self, team_abbrev, start_date, end_date,
                                     player_id=None, player_type='pitcher',
                                     minors=False):
        """
        Download supplemental data from Statcast Search CSV endpoint.
        Provides columns not available in the live feed API:
        - arm_angle (ArmAngle)
        - bat_speed (BatSpeed)
        - swing_length (SwingLength)
        - attack_angle (AttackAngle)
        - attack_direction (AttackDirection)
        - swing_path_tilt (SwingPathTilt)
        - delta_pitcher_run_exp (RunExp)
        - on_1b, on_2b, on_3b (Runners)

        Queries by team (team_abbrev) by default. If player_id is given, the
        team filter is replaced with a single-player lookup: player_type
        'pitcher' → pitches thrown, 'batter' → pitches faced.

        minors=True queries the minor league Statcast Search instead (AAA and
        below with tracking); those rows lack bat tracking columns but carry
        arm_angle, run exp, expected stats, runners, and launch_speed_angle.

        Returns a DataFrame keyed on game_pk, at_bat_number, pitch_number
        for merging with live feed data, or None if the request fails.
        """
        if minors:
            url = "https://baseballsavant.mlb.com/statcast-search-minors/csv"
        else:
            url = "https://baseballsavant.mlb.com/statcast_search/csv"

        params = {
            'all': 'true',
            'type': 'details',
            'game_date_gt': start_date,
            'game_date_lt': end_date,
            'player_type': player_type,
            'min_pitches': '0',
            'min_results': '0',
            'sort_col': 'pitches',
            'sort_order': 'desc',
        }
        if minors:
            params['minors'] = 'true'

        label = 'minor league Statcast Search' if minors else 'Statcast Search'
        if player_id:
            lookup_key = 'batters_lookup[]' if player_type == 'batter' else 'pitchers_lookup[]'
            params[lookup_key] = str(player_id)
            print(f"Downloading {label} supplement for player {player_id} ({player_type})...")
        else:
            statcast_team = self.get_statcast_team_abbrev(team_abbrev)
            params['team'] = statcast_team
            print(f"Downloading Statcast Search supplement for {team_abbrev} ({statcast_team})...")

        try:
            response = self.session.get(url, params=params, timeout=60)

            if response.status_code != 200:
                print(f"Statcast Search returned status {response.status_code}")
                return None

            csv_text = response.text

            if not csv_text or csv_text.strip() == '' or 'No Results' in csv_text[:100]:
                print("Statcast Search returned no data (data may not be processed yet)")
                return None

            df = pd.read_csv(StringIO(csv_text))

            if df.empty:
                print("Statcast Search returned empty DataFrame")
                return None

            # Columns we need for merging
            merge_cols = ['game_pk', 'at_bat_number', 'pitch_number']
            supplement_cols = [
                'arm_angle',
                'release_pos_y',
                'bat_speed',
                'swing_length',
                'attack_angle',
                'attack_direction',
                'swing_path_tilt',
                'delta_pitcher_run_exp',
                'estimated_ba_using_speedangle',
                'estimated_slg_using_speedangle',
                'estimated_woba_using_speedangle',
                'on_1b',
                'on_2b',
                'on_3b',
                'outs_when_up',
                'launch_speed_angle',
            ]

            available_supplement = [c for c in supplement_cols if c in df.columns]
            missing_supplement = [c for c in supplement_cols if c not in df.columns]

            if missing_supplement:
                print(f"Warning: Statcast Search missing columns: {missing_supplement}")

            if not available_supplement:
                print("No supplement columns found in Statcast Search response")
                return None

            for col in merge_cols:
                if col not in df.columns:
                    print(f"Error: Merge key column '{col}' not found in Statcast Search data")
                    return None

            keep_cols = merge_cols + available_supplement
            df = df[keep_cols].copy()

            print(f"Statcast Search returned {len(df)} pitches with columns: {available_supplement}")
            return df

        except requests.exceptions.Timeout:
            print("Statcast Search request timed out (server may be slow)")
            return None
        except Exception as e:
            print(f"Error downloading Statcast Search data: {str(e)}")
            return None

    def merge_statcast_supplement(self, live_df, supplement_df):
        """
        Merge Statcast Search supplement data into the live feed DataFrame.
        """
        # Column name mapping from Statcast CSV names to our display names
        rename_map = {
            'arm_angle': 'ArmAngle',
            'bat_speed': 'BatSpeed',
            'swing_length': 'SwingLength',
            'attack_angle': 'AttackAngle',
            'attack_direction': 'AttackDirection',
            'swing_path_tilt': 'SwingPathTilt',
            'delta_pitcher_run_exp': 'RunExp',
            'estimated_ba_using_speedangle': 'xBA',
            'estimated_slg_using_speedangle': 'xSLG',
            'estimated_woba_using_speedangle': 'xwOBA',
            'outs_when_up': 'Outs',
            'launch_speed_angle': 'Barrel',
        }
        # on_1b/on_2b/on_3b are not renamed — they're consumed to build the Runners column
        runner_cols = ['on_1b', 'on_2b', 'on_3b']

        supplement_display_cols = list(rename_map.values())
        # release_pos_y is used to compute Extension (60.5 - release_pos_y) but is not a display column
        has_release_pos_y = supplement_df is not None and 'release_pos_y' in supplement_df.columns

        if supplement_df is None or supplement_df.empty:
            # Add empty supplement columns so the CSV schema is consistent
            for col in supplement_display_cols:
                if col not in live_df.columns:
                    live_df[col] = np.nan
            # Drop merge keys
            live_df = live_df.drop(columns=['_game_pk', '_at_bat_number', '_pitch_number'])
            return live_df

        # Rename supplement columns to match our naming convention
        supplement_df = supplement_df.rename(columns=rename_map)

        # Rename Statcast merge keys to match the underscore-prefixed live feed keys
        supplement_df = supplement_df.rename(columns={
            'game_pk': '_game_pk',
            'at_bat_number': '_at_bat_number',
            'pitch_number': '_pitch_number',
        })

        merge_keys = ['_game_pk', '_at_bat_number', '_pitch_number']

        merged = live_df.merge(
            supplement_df,
            on=merge_keys,
            how='left',
            suffixes=('', '_statcast')
        )

        # ArmAngle: prefer official Statcast value; fall back to estimate from x0/z0
        if 'ArmAngle_statcast' in merged.columns:
            statcast_aa = pd.to_numeric(merged['ArmAngle_statcast'], errors='coerce')
            estimated_aa = pd.to_numeric(merged['ArmAngle'], errors='coerce')
            merged['ArmAngle'] = statcast_aa.combine_first(estimated_aa).apply(lambda x: f"{x:.1f}" if pd.notna(x) else '')
            merged = merged.drop(columns=['ArmAngle_statcast'])

        # Compute Extension from release_pos_y where available (60.5 - release_pos_y)
        if 'release_pos_y' in merged.columns:
            statcast_ext = 60.5 - pd.to_numeric(merged['release_pos_y'], errors='coerce')
            # Fill in Extension where the live API value was missing, or overwrite with Statcast value
            merged_ext = pd.to_numeric(merged['Extension'], errors='coerce')
            merged['Extension'] = statcast_ext.combine_first(merged_ext).apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
            merged = merged.drop(columns=['release_pos_y'])

        # Overwrite placeholder columns with Statcast supplement values
        for col in ['RunExp', 'xBA', 'xSLG', 'xwOBA']:
            if f'{col}_statcast' in merged.columns:
                merged[col] = pd.to_numeric(merged[f'{col}_statcast'], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
                merged = merged.drop(columns=[f'{col}_statcast'])

        # Outs: the live feed has no pre-pitch outs field, so this comes from
        # the supplement's outs_when_up (matches backfill_supplement.py)
        if 'Outs_statcast' in merged.columns:
            statcast_outs = pd.to_numeric(merged['Outs_statcast'], errors='coerce')
            merged['Outs'] = statcast_outs.combine_first(
                pd.to_numeric(merged['Outs'], errors='coerce'))
            merged = merged.drop(columns=['Outs_statcast'])
        if 'Outs' in merged.columns:
            merged['Outs'] = pd.to_numeric(merged['Outs'], errors='coerce').astype('Int64')

        # Build Runners column from on_1b/on_2b/on_3b
        # on_1b/on_2b/on_3b contain player IDs when occupied, NaN when empty
        if all(c in merged.columns for c in runner_cols):
            def _runners_str(row):
                bases = []
                if pd.notna(row.get('on_1b')):
                    bases.append('1')
                if pd.notna(row.get('on_2b')):
                    bases.append('2')
                if pd.notna(row.get('on_3b')):
                    bases.append('3')
                return '+'.join(bases) if bases else '0'
            merged['Runners'] = merged.apply(_runners_str, axis=1)
            merged = merged.drop(columns=runner_cols)

        # Barrel column: keep raw launch_speed_angle value (1-6 scale)
        # If Statcast supplement provided a value, it overwrites the estimate;
        # if only estimate exists, it stays (6 for barrel, empty for non-barrel)
        if 'Barrel_statcast' in merged.columns:
            # Statcast official value takes priority over estimate
            statcast_barrel = pd.to_numeric(merged['Barrel_statcast'], errors='coerce')
            merged['Barrel'] = statcast_barrel.combine_first(
                pd.to_numeric(merged['Barrel'], errors='coerce')
            )
            merged = merged.drop(columns=['Barrel_statcast'])
        # Convert to nullable integer (blanks stay blank, values stay as ints)
        if 'Barrel' in merged.columns:
            merged['Barrel'] = pd.to_numeric(merged['Barrel'], errors='coerce').astype('Int64')

        # BatSpeed coalesce: the near-real-time game feed can miss bat speed for
        # swings the official Statcast export does include. Fold the official
        # value in (game-feed first, official fallback) BEFORE the validity check
        # below, mirroring the other supplement columns — otherwise a swing that
        # merely lacked a fast-feed reading is seen as NaN and the whole swing
        # frame gets nulled even though a valid official bat speed existed.
        if 'BatSpeed_statcast' in merged.columns:
            merged['BatSpeed'] = pd.to_numeric(merged['BatSpeed'], errors='coerce').combine_first(
                pd.to_numeric(merged['BatSpeed_statcast'], errors='coerce'))
            merged = merged.drop(columns=['BatSpeed_statcast'])

        # Swing-cluster integrity: BatSpeed is the validity anchor. If it's
        # missing or sub-50 (check swings / artifacts), the rest of the swing
        # frame is unreliable too — null the whole cluster together.
        swing_cluster = ['BatSpeed', 'SwingLength', 'AttackAngle', 'AttackDirection',
                         'SwingPathTilt']
        if 'BatSpeed' in merged.columns:
            merged['BatSpeed'] = pd.to_numeric(merged['BatSpeed'], errors='coerce')
            invalid = merged['BatSpeed'].isna() | (merged['BatSpeed'] < 50)
            for col in swing_cluster:
                if col in merged.columns:
                    merged.loc[invalid, col] = np.nan

        # Count successful merges
        supplement_value_cols = [c for c in supplement_display_cols if c in merged.columns]
        if supplement_value_cols:
            matched = merged[supplement_value_cols[0]].notna().sum()
            print(f"Statcast supplement merged: {matched}/{len(merged)} pitches matched")

        # Format supplement-only columns with trailing zeros
        supplement_round_1 = ['BatSpeed', 'SwingLength', 'AttackAngle',
                              'AttackDirection', 'SwingPathTilt']
        for col in supplement_round_1:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors='coerce').apply(lambda x: f"{x:.1f}" if pd.notna(x) else '')

        # Drop merge keys so they don't appear in the final CSV
        merged = merged.drop(columns=merge_keys)

        return merged

    def download_savant_bat_speed(self, game_pk):
        """
        Download bat speed from the Baseball Savant game feed API.
        Available in near real-time (same day, after game completes).
        Returns a dict keyed by (game_pk, at_bat_number, pitch_number) -> bat_speed value.
        """
        print(f"  Fetching bat speed from Savant game feed for {game_pk}...")
        url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                print(f"  Savant game feed returned status {response.status_code}")
                return {}

            data = response.json()
            lookup = {}

            # Bat speed is in team_away and team_home pitch arrays
            for side in ['team_away', 'team_home']:
                pitches = data.get(side, [])
                for p in pitches:
                    bs = p.get('batSpeed')
                    if bs is not None:
                        bs_val = round(float(bs), 1)
                        if bs_val >= 50:  # Filter out check swings / artifacts
                            key = (str(game_pk), str(p.get('ab_number', '')), str(p.get('pitch_number', '')))
                            lookup[key] = bs_val

            print(f"  Got bat speed for {len(lookup)} swings")
            return lookup

        except Exception as e:
            print(f"  Error fetching Savant game feed: {e}")
            return {}

    def merge_bat_speed(self, df, game_pk):
        """
        Merge bat speed from Savant game feed into the DataFrame.
        Only fills BatSpeed where it's currently empty/NaN.
        """
        lookup = self.download_savant_bat_speed(game_pk)
        if not lookup:
            return df

        filled = 0
        for idx, row in df.iterrows():
            if pd.notna(row.get('BatSpeed')):
                continue  # Already has bat speed (shouldn't happen on fresh download)
            pid = row.get('PitchID', '')
            parts = pid.split('_')
            if len(parts) == 3:
                key = (parts[0], parts[1].lstrip('0') or '0', parts[2].lstrip('0') or '0')
                if key in lookup:
                    bs = lookup[key]
                    if bs >= 50:
                        df.at[idx, 'BatSpeed'] = bs
                    filled += 1

        if filled:
            print(f"  Merged bat speed for {filled} pitches")
        return df

    def download_team_games(self, team_abbrev, start_date, end_date, pitchers_only=False, save_csv=True):
        """
        Download all games for a specific team within a date range and save as a single CSV.
        Combines live feed data with Statcast Search supplement.

        If save_csv=False, the dataframe is built and stashed in
        ``self._last_combined_df`` for a downstream sheets push, but no local
        CSV is written.
        """
        # Get all games for the team within the date range
        game_pks = self.get_team_games(team_abbrev, start_date, end_date)

        if not game_pks:
            print(f"No games found for {team_abbrev} from {start_date} to {end_date}")
            return None

        # Download and process each game into DataFrames
        all_dfs = []
        for game_pk in game_pks:
            # Set filter_team parameter if pitchers_only is True
            filter_team = team_abbrev if pitchers_only else None

            # Download game data as DataFrame
            df = self.download_game_data(game_pk, filter_team)

            if df is not None:
                # Fetch real-time bat speed from Savant game feed
                df = self.merge_bat_speed(df, game_pk)
                all_dfs.append(df)
                print(f"Successfully downloaded data for game {game_pk}")
            else:
                print(f"Failed to download data for game {game_pk}")

            # Add a small delay to avoid rate limiting
            time.sleep(1)

        print(f"Downloaded {len(all_dfs)} out of {len(game_pks)} games for {team_abbrev}")

        # If no games were downloaded, return None
        if not all_dfs:
            print(f"Failed to download any games for {team_abbrev}")
            return None

        # Combine all downloaded games into a single DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Attempt to merge Statcast Search supplement data
        print("\nAttempting Statcast Search supplement download...")
        supplement_df = self.download_statcast_supplement(team_abbrev, start_date, end_date)
        combined_df = self.merge_statcast_supplement(combined_df, supplement_df)

        # Enforce final column order
        final_columns = [
            'Game Date', 'PTeam', 'Pitcher', 'Throws', 'Pitch Type',
            'Velocity', 'Spin Rate', 'RTilt', 'OTilt', 'IndVertBrk', 'HorzBrk',
            'xIndVrtBrk', 'xHorzBrk',
            'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle',
            'PlateZ', 'PlateX', 'SzTop', 'SzBot',
            'VAA', 'HAA',
            'Stuff+', 'Loc+', 'Pitching+',
            'BTeam', 'Batter', 'Bats', 'Count', 'Runners', 'Outs',
            'Description', 'Event',
            'ExitVelo', 'LaunchAngle', 'Distance', 'BBType',
            'HC_X', 'HC_Y', 'xBA', 'xSLG', 'xwOBA', 'RunExp',
            'BatSpeed', 'SwingLength',
            'AttackAngle', 'AttackDirection', 'SwingPathTilt',
            'PitchID', 'Barrel',
        ]

        # Keep the full column schema regardless of data availability so every
        # CSV matches the MLB/ROC/AAA layout; missing columns come out empty
        for col in final_columns:
            if col not in combined_df.columns:
                combined_df[col] = ''
        combined_df = combined_df[final_columns]

        # ROC/AAA normalization (no-op for MLB downloads)
        combined_df = self.normalize_aaa_labels(combined_df)

        # Save the combined data directly to the team file (skip local CSV
        # write when caller is only doing a Sheets push).
        output_filename = os.path.join(self.download_dir, f"{team_abbrev}.csv")
        if save_csv:
            combined_df.to_csv(output_filename, index=False)

        print(f"\nCombined {len(all_dfs)} games with {len(combined_df)} total pitches")
        if save_csv:
            print(f"Combined data saved to: {output_filename}")

        # Stash for downstream Sheets push (caller passes flag to main)
        self._last_combined_df = combined_df

        return output_filename

    def download_games_by_id(self, game_pks, filter_team=None, output_name=None, save_csv=True):
        """
        Download one or more games by their game PK IDs and save as a single CSV.

        Useful for games involving national teams or other non-MLB teams that
        don't have entries in the team abbreviation/ID mappings.

        Args:
            game_pks: A single game PK (int) or a list of game PKs
            filter_team: Optional team name/abbreviation to filter pitchers by.
                         Must match exactly what the API returns as the team
                         abbreviation (e.g., "TOR", "CAN"). If unsure, run
                         once without filter_team to see what team names appear
                         in the PTeam column, then re-run with the correct value.
            output_name: Optional filename stem for the output CSV.
                         Defaults to the game PK (single game) or "custom" (multiple).
        """
        # Accept a single game PK or a list
        if isinstance(game_pks, (int, str)):
            game_pks = [int(game_pks)]
        else:
            game_pks = [int(pk) for pk in game_pks]

        print(f"Downloading {len(game_pks)} game(s) by ID: {game_pks}")
        if filter_team:
            print(f"Filtering to pitchers from: {filter_team}")

        all_dfs = []
        for game_pk in game_pks:
            df = self.download_game_data(game_pk, filter_team)

            if df is not None:
                all_dfs.append(df)
                print(f"Successfully downloaded data for game {game_pk}")
            else:
                print(f"Failed to download data for game {game_pk}")

            # Add a small delay between games to avoid rate limiting
            if len(game_pks) > 1:
                time.sleep(1)

        if not all_dfs:
            print("Failed to download any games")
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # No Statcast supplement for game-ID mode (no team for query),
        # but still need to drop merge keys and ensure schema consistency
        combined_df = self.merge_statcast_supplement(combined_df, None)

        # Enforce final column order
        final_columns = [
            'Game Date', 'PTeam', 'Pitcher', 'Throws', 'Pitch Type',
            'Velocity', 'Spin Rate', 'RTilt', 'OTilt', 'IndVertBrk', 'HorzBrk',
            'xIndVrtBrk', 'xHorzBrk',
            'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle',
            'PlateZ', 'PlateX', 'SzTop', 'SzBot',
            'VAA', 'HAA',
            'Stuff+', 'Loc+', 'Pitching+',
            'BTeam', 'Batter', 'Bats', 'Count', 'Runners', 'Outs',
            'Description', 'Event',
            'ExitVelo', 'LaunchAngle', 'Distance', 'BBType',
            'HC_X', 'HC_Y', 'xBA', 'xSLG', 'xwOBA', 'RunExp',
            'BatSpeed', 'SwingLength',
            'AttackAngle', 'AttackDirection', 'SwingPathTilt',
            'PitchID', 'Barrel',
        ]
        # Keep the full column schema regardless of data availability so every
        # CSV matches the MLB/ROC/AAA layout; missing columns come out empty
        for col in final_columns:
            if col not in combined_df.columns:
                combined_df[col] = ''
        combined_df = combined_df[final_columns]

        # ROC/AAA normalization (no-op for MLB downloads)
        combined_df = self.normalize_aaa_labels(combined_df)

        # Determine output filename
        if output_name:
            stem = output_name
        elif len(game_pks) == 1:
            stem = str(game_pks[0])
        else:
            stem = "custom"

        output_filename = os.path.join(self.download_dir, f"{stem}.csv")
        if save_csv:
            combined_df.to_csv(output_filename, index=False)

        print(f"\nCombined {len(all_dfs)} games with {len(combined_df)} total pitches")
        if save_csv:
            print(f"Data saved to: {output_filename}")

        # Stash for downstream Sheets push (caller passes flag to main)
        self._last_combined_df = combined_df

        return output_filename

    def get_player_info(self, player_id):
        """
        Look up a player's full name and primary position code from the MLB Stats API.

        Returns (full_name, position_code) — e.g., ("Paul Skenes", "1").
        Position codes: "1" = pitcher, "Y" = two-way player, anything else = position player.
        Returns (None, None) if the lookup fails.
        """
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"

        try:
            response = self.session.get(url, timeout=30)

            if response.status_code != 200:
                print(f"Error: player lookup returned status code {response.status_code}")
                return None, None

            people = response.json().get('people', [])
            if not people:
                print(f"Error: no player found for ID {player_id}")
                return None, None

            person = people[0]
            full_name = person.get('fullName', str(player_id))
            position_code = person.get('primaryPosition', {}).get('code', '')
            return full_name, position_code

        except Exception as e:
            print(f"Error looking up player {player_id}: {str(e)}")
            return None, None

    # Sport IDs for every level with a game log; the gameLog endpoint
    # defaults to sportId=1 (MLB + spring training), so minor league games
    # must be requested level by level.
    # 22 covers College Baseball feeds (Cape Cod League, NCAA event games,
    # draft combine); 51 covers international (USA Collegiate National Team).
    # Only some college parks have Statcast tracking — untracked games are
    # skipped by the empty-Velocity check like any untracked minors game.
    PLAYER_GAME_SPORT_IDS = {
        1: 'MLB',
        11: 'AAA',
        12: 'AA',
        13: 'High-A',
        14: 'Single-A',
        16: 'Rookie',
        22: 'College',
        51: 'International',
    }

    # Levels covered by the minor league Statcast Search (college and
    # international games have live feed tracking but no Savant search rows,
    # so they never trigger the minors supplement)
    MINORS_SUPPLEMENT_SPORT_IDS = {11, 12, 13, 14, 16}

    def get_player_games(self, player_id, season, group):
        """
        Get all games a player appeared in for a season from their game logs,
        across every level (MLB through Rookie ball, plus college and
        international feeds where MLB tracks them).

        group: "pitching" or "hitting" — which game log to pull.
        Uses the same game types as team mode (spring training through World Series).

        Returns a list of (date, game_pk, sport_id) tuples.
        """
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"

        games = []
        for sport_id, level_name in self.PLAYER_GAME_SPORT_IDS.items():
            params = {
                "stats": "gameLog",
                "season": season,
                "group": group,
                "gameType": "E,S,R,F,D,L,W",
                "sportId": sport_id,
            }

            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    print(f"Error: {group} game log ({level_name}) returned status code {response.status_code}")
                    continue

                level_games = []
                for stat in response.json().get('stats', []):
                    for split in stat.get('splits', []):
                        game_pk = split.get('game', {}).get('gamePk')
                        if game_pk:
                            level_games.append((split.get('date', ''), game_pk, sport_id))

                if level_games:
                    print(f"  {level_name}: {len(level_games)} {group} games")
                games.extend(level_games)

            except Exception as e:
                print(f"Error getting {group} game log ({level_name}): {str(e)}")

        return games

    def download_player_games(self, player_id, season, output_name=None, save_csv=True):
        """
        Download every game a player appeared in for a season and save as a single CSV.

        Looks up the player's game log via the MLB Stats API, downloads each game's
        live feed data filtered to that player's plays, then merges the Statcast
        Search supplement via single-player lookup.

        Pitchers get every pitch they threw; hitters get every pitch they faced.
        Two-way players get both.

        Args:
            player_id: MLB player ID (e.g., 694973)
            season: Season year (e.g., "2026")
            output_name: Optional filename stem. Defaults to the player's name.
        """
        try:
            player_id = int(str(player_id).strip())
        except (ValueError, TypeError):
            print(f"Error: invalid player ID '{player_id}'")
            return None

        player_name, position_code = self.get_player_info(player_id)
        if player_name is None:
            return None

        is_pitcher = position_code in ('1', 'Y')  # pitcher or two-way player
        is_hitter = position_code != '1'          # everyone except pure pitchers

        role = {'1': 'pitcher', 'Y': 'two-way player'}.get(position_code, 'hitter')
        print(f"Player: {player_name} (ID {player_id}, {role})")

        # Collect game PKs from the applicable game logs across every level
        # (a two-way player appears in both); dedupe and keep chronological order
        entries = []
        if is_pitcher:
            entries.extend(self.get_player_games(player_id, season, 'pitching'))
        if is_hitter:
            entries.extend(self.get_player_games(player_id, season, 'hitting'))

        game_pks = []
        seen = set()
        has_minors = False
        for date, game_pk, sport_id in sorted(set(entries)):
            if sport_id in self.MINORS_SUPPLEMENT_SPORT_IDS:
                has_minors = True
            if game_pk not in seen:
                seen.add(game_pk)
                game_pks.append(game_pk)

        if not game_pks:
            print(f"No {season} games found for {player_name}")
            print("Note: NCAA regular-season games are not in the MLB feed; college "
                  "coverage is limited to Cape Cod League, Team USA, and special-event games.")
            return None

        print(f"Found {len(game_pks)} games for {player_name} in {season}")

        # Download and process each game, keeping only this player's plays.
        # Games without Statcast tracking (untracked minor league parks) are
        # skipped so the CSV only carries pitches with real tracking data.
        all_dfs = []
        skipped_untracked = 0
        for game_pk in game_pks:
            df = self.download_game_data(game_pk, filter_player_id=player_id)

            if df is None or len(df) == 0:
                print(f"Failed to download data for game {game_pk}")
            elif 'Velocity' in df.columns and not df['Velocity'].astype(str).str.strip().ne('').any():
                skipped_untracked += 1
                print(f"No Statcast tracking data for game {game_pk} — skipping")
            else:
                # Fetch real-time bat speed from Savant game feed
                df = self.merge_bat_speed(df, game_pk)
                all_dfs.append(df)
                print(f"Successfully downloaded data for game {game_pk}")

            # Add a small delay to avoid rate limiting
            time.sleep(1)

        if skipped_untracked:
            print(f"Skipped {skipped_untracked} game(s) with no Statcast tracking data")

        print(f"Downloaded {len(all_dfs)} out of {len(game_pks)} games for {player_name}")

        if not all_dfs:
            print(f"Failed to download any games for {player_name}")
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Statcast Search supplement via single-player lookup, bounded by the
        # dates actually downloaded (two-way players need both perspectives)
        print("\nAttempting Statcast Search supplement download...")
        sup_start = combined_df['Game Date'].min()
        sup_end = combined_df['Game Date'].max()

        # MLB search covers MLB + spring training; minor league games need the
        # separate minors search (fetched only when the game log found any)
        minors_flags = [False] + ([True] if has_minors else [])

        supplement_frames = []
        for minors in minors_flags:
            if is_pitcher:
                sup = self.download_statcast_supplement(None, sup_start, sup_end,
                                                        player_id=player_id, player_type='pitcher',
                                                        minors=minors)
                if sup is not None:
                    supplement_frames.append(sup)
            if is_hitter:
                sup = self.download_statcast_supplement(None, sup_start, sup_end,
                                                        player_id=player_id, player_type='batter',
                                                        minors=minors)
                if sup is not None:
                    supplement_frames.append(sup)

        if supplement_frames:
            supplement_df = pd.concat(supplement_frames, ignore_index=True)
            supplement_df = supplement_df.drop_duplicates(
                subset=['game_pk', 'at_bat_number', 'pitch_number'])
        else:
            supplement_df = None

        combined_df = self.merge_statcast_supplement(combined_df, supplement_df)

        # Enforce final column order
        final_columns = [
            'Game Date', 'PTeam', 'Pitcher', 'Throws', 'Pitch Type',
            'Velocity', 'Spin Rate', 'RTilt', 'OTilt', 'IndVertBrk', 'HorzBrk',
            'xIndVrtBrk', 'xHorzBrk',
            'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle',
            'PlateZ', 'PlateX', 'SzTop', 'SzBot',
            'VAA', 'HAA',
            'Stuff+', 'Loc+', 'Pitching+',
            'BTeam', 'Batter', 'Bats', 'Count', 'Runners', 'Outs',
            'Description', 'Event',
            'ExitVelo', 'LaunchAngle', 'Distance', 'BBType',
            'HC_X', 'HC_Y', 'xBA', 'xSLG', 'xwOBA', 'RunExp',
            'BatSpeed', 'SwingLength',
            'AttackAngle', 'AttackDirection', 'SwingPathTilt',
            'PitchID', 'Barrel',
        ]
        # Keep the full column schema regardless of data availability so every
        # CSV matches the MLB/ROC/AAA layout; missing columns come out empty
        for col in final_columns:
            if col not in combined_df.columns:
                combined_df[col] = ''
        combined_df = combined_df[final_columns]

        # ROC/AAA normalization (no-op for MLB downloads)
        combined_df = self.normalize_aaa_labels(combined_df)

        stem = output_name if output_name else str(player_name).replace('/', '_')
        output_filename = os.path.join(self.download_dir, f"{stem}.csv")
        if save_csv:
            combined_df.to_csv(output_filename, index=False)

        print(f"\nCombined {len(all_dfs)} games with {len(combined_df)} total pitches")
        if save_csv:
            print(f"Data saved to: {output_filename}")

        # Stash for downstream Sheets push (caller passes flag to main)
        self._last_combined_df = combined_df

        return output_filename


def main():
    """Main function to download pitcher data"""

    # ── Settings (edit these directly or override via command line) ──
    team_abbrev     = ""
    start_date      = "2026-07-16"
    end_date        = "2026-07-16"
    pitchers_only   = True

    game_id         = ""          # Game PK (e.g., "831437") — leave blank for team/date lookup
    filter_team     = None        # Optional team filter for game ID mode (e.g., "CAN")
    output_name     = ""          # Optional custom filename (without .csv)

    player_id       = ""          # MLB player ID (e.g., "694973") — pulls ALL of the player's games
                                  # for the season (year of start_date), with Statcast supplement.
                                  # Pitchers → every pitch thrown; hitters → every pitch faced.
                                  # Saves a CSV named after the player (or output_name); never
                                  # pushes to Sheets (would duplicate rows from daily team pulls).

    push_to_sheets  = True       # True → also append to AL/NL 2026 Google Sheets after CSV save

    # ── CLI overrides (optional — values above are used if no args passed) ──
    parser = argparse.ArgumentParser(description='Download pitch data from Baseball Savant')
    parser.add_argument('--team', default=None, help='Team abbreviation')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--pitchers-only', dest='pitchers_only', action='store_true', default=None)
    parser.add_argument('--no-pitchers-only', dest='pitchers_only', action='store_false')
    parser.add_argument('--game-id', default=None, help='Game PK (e.g., 831437)')
    parser.add_argument('--filter-team', default=None, help='Team filter for game ID mode')
    parser.add_argument('--player-id', default=None, help='MLB player ID — pull all of their games for the season')
    parser.add_argument('--output-name', default=None, help='Custom filename (without .csv)')
    parser.add_argument('--push-to-sheets', dest='push_to_sheets', action='store_true', default=None,
                        help='After saving the CSV, append the data into the matching tab '
                             'in the AL 2026 / NL 2026 Google Sheets workbook (auth one-time '
                             'via ~/.config/gspread/credentials.json).')
    parser.add_argument('--no-push-to-sheets', dest='push_to_sheets', action='store_false')
    args = parser.parse_args()

    if args.team is not None: team_abbrev = args.team
    if args.start is not None: start_date = args.start
    if args.end is not None: end_date = args.end
    if args.pitchers_only is not None: pitchers_only = args.pitchers_only
    if args.game_id is not None: game_id = args.game_id
    if args.filter_team is not None: filter_team = args.filter_team
    if args.player_id is not None: player_id = args.player_id
    if args.push_to_sheets is not None: push_to_sheets = args.push_to_sheets
    if args.output_name is not None: output_name = args.output_name

    downloader = BaseballSavantFocusedDownloader()

    # Player ID mode always writes a local CSV and never pushes to the remote
    # stores — a season-long single-player pull would duplicate rows the daily
    # team pulls already appended to the Sheets.
    if player_id:
        if push_to_sheets:
            print("Player ID mode: skipping Sheets push — writing local CSV instead.")
            push_to_sheets = False

    # When pushing to Sheets, the local CSV is redundant — skip writing it.
    save_csv = not push_to_sheets
    dest_label = "Sheets"

    # ── Logic: player_id first, then game_id, then team/date lookup ──
    if player_id:
        result = downloader.download_player_games(
            player_id=player_id,
            season=start_date[:4],
            output_name=output_name if output_name else None,
        )

        if not result:
            print("\nFailed to download player data")

    elif game_id:
        result = downloader.download_games_by_id(
            game_pks=game_id,
            filter_team=filter_team,
            output_name=output_name if output_name else None,
            save_csv=save_csv,
        )

        if result:
            if save_csv:
                print(f"\nData saved to: {result}")
            else:
                print(f"\nDownloaded successfully (CSV skipped — pushing to {dest_label})")
        else:
            print("\nFailed to download game data")

    elif team_abbrev:
        result = downloader.download_team_games(team_abbrev, start_date, end_date, pitchers_only,
                                                save_csv=save_csv)

        if result:
            print(f"\nSuccessfully downloaded and combined games for {team_abbrev}")
            if save_csv:
                print(f"Data saved to: {result}")
            else:
                print(f"CSV skipped — pushing to {dest_label}")
        else:
            print(f"\nFailed to download any games for {team_abbrev}")

    else:
        print("Error: Please set player_id, game_id, or team_abbrev")
        return

    # Push to Google Sheets if requested. The dataframe is grouped by PTeam,
    # so a ROC download (PTeam ∈ {ROC, AAA}) splits across the ROC and AAA
    # tabs of NL 2026; an MLB download goes to its single matching tab.
    if push_to_sheets and getattr(downloader, '_last_combined_df', None) is not None:
        combined = downloader._last_combined_df
        try:
            from scrapers.sheets_append import push_csv_to_sheets
            print("\nPushing to Google Sheets...")
            unmapped_teams = push_csv_to_sheets(combined)
        except Exception as e:
            # Loud, not a one-liner: a push can fail AFTER the rows have
            # landed (e.g. the formatting pass dies), in which case the sheet
            # holds real data that renders as raw serials — Game Date as
            # 46227, RTilt/OTilt as decimals. That looks like a bad download
            # but is purely a formatting miss, and it's repairable in place.
            print("\n" + "!" * 72)
            print(f"SHEETS PUSH FAILED: {e}")
            print("Rows may already be in the sheet but unformatted "
                  "(Game Date showing as e.g. 46227).")
            print("Check and repair with:")
            print("  python scripts/ops/fix_unformatted_blocks.py --apply")
            print("!" * 72)
            # Fallback: write the CSV locally so the data isn't lost.
            # We skipped the local write above because push_to_sheets was True,
            # so this is the only on-disk copy if Sheets is down.
            if not save_csv and result:
                try:
                    combined.to_csv(result, index=False)
                    print(f"Wrote fallback CSV to: {result}")
                except Exception as e2:
                    print(f"Could not write fallback CSV either: {e2}")
        else:
            # Teams with no NL/AL/ROC mapping (FCL / complex-league or
            # international games) can't be pushed to a workbook. Write just
            # those rows to a local CSV named after the teams so the data
            # isn't silently lost. output_name overrides the auto-name.
            if unmapped_teams:
                unmapped_rows = combined[combined['PTeam'].astype(str).isin(unmapped_teams)]
                stem = output_name if output_name else " vs ".join(sorted(unmapped_teams))
                csv_path = os.path.join(downloader.download_dir, f"{stem}.csv")
                try:
                    unmapped_rows.to_csv(csv_path, index=False)
                    print(f"\nSkipped Sheets for {len(unmapped_teams)} unmapped "
                          f"team(s); wrote {len(unmapped_rows)} rows to: {csv_path}")
                except Exception as e:
                    print(f"\nCould not write unmapped-team CSV: {e}")


if __name__ == "__main__":
    main()