# Runnable as a file from any directory (IDE run buttons included):
# put the repo root on sys.path before the intra-repo package imports.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re


def scrape_milb_transactions(start_date, end_date):
    """
    Scrape MiLB transactions for signings, releases, and elected free agency between specified dates.
    """

    # MLB Team Affiliates mapping
    team_mapping = {
        'Norfolk Tides': 'Baltimore Orioles', 'Chesapeake Baysox': 'Baltimore Orioles',
        'Frederick Keys': 'Baltimore Orioles', 'Delmarva Shorebirds': 'Baltimore Orioles',
        'FCL Orioles': 'Baltimore Orioles',
        'Worcester Red Sox': 'Boston Red Sox', 'Portland Sea Dogs': 'Boston Red Sox',
        'Greenville Drive': 'Boston Red Sox', 'Salem RidgeYaks': 'Boston Red Sox',
        'FCL Red Sox': 'Boston Red Sox',
        'Scranton/Wilkes-Barre RailRiders': 'New York Yankees', 'Somerset Patriots': 'New York Yankees',
        'Hudson Valley Renegades': 'New York Yankees', 'Tampa Tarpons': 'New York Yankees',
        'FCL Yankees': 'New York Yankees',
        'Durham Bulls': 'Tampa Bay Rays', 'Montgomery Biscuits': 'Tampa Bay Rays',
        'Bowling Green Hot Rods': 'Tampa Bay Rays', 'Charleston RiverDogs': 'Tampa Bay Rays',
        'FCL Rays': 'Tampa Bay Rays',
        'Buffalo Bisons': 'Toronto Blue Jays', 'New Hampshire Fisher Cats': 'Toronto Blue Jays',
        'Vancouver Canadians': 'Toronto Blue Jays', 'Dunedin Blue Jays': 'Toronto Blue Jays',
        'FCL Blue Jays': 'Toronto Blue Jays',
        'Charlotte Knights': 'Chicago White Sox', 'Birmingham Barons': 'Chicago White Sox',
        'Winston-Salem Dash': 'Chicago White Sox', 'Kannapolis Cannon Ballers': 'Chicago White Sox',
        'ACL White Sox': 'Chicago White Sox',
        'Columbus Clippers': 'Cleveland Guardians', 'Akron RubberDucks': 'Cleveland Guardians',
        'Lake County Captains': 'Cleveland Guardians', 'Hill City Howlers': 'Cleveland Guardians',
        'ACL Guardians': 'Cleveland Guardians',
        'Toledo Mud Hens': 'Detroit Tigers', 'Erie SeaWolves': 'Detroit Tigers',
        'West Michigan Whitecaps': 'Detroit Tigers', 'Lakeland Flying Tigers': 'Detroit Tigers',
        'FCL Tigers': 'Detroit Tigers',
        'Omaha Storm Chasers': 'Kansas City Royals', 'Northwest Arkansas Naturals': 'Kansas City Royals',
        'Quad Cities River Bandits': 'Kansas City Royals', 'Columbia Fireflies': 'Kansas City Royals',
        'ACL Royals': 'Kansas City Royals',
        'St. Paul Saints': 'Minnesota Twins', 'Wichita Wind Surge': 'Minnesota Twins',
        'Cedar Rapids Kernels': 'Minnesota Twins', 'Fort Myers Mighty Mussels': 'Minnesota Twins',
        'FCL Twins': 'Minnesota Twins',
        'Sugar Land Space Cowboys': 'Houston Astros', 'Corpus Christi Hooks': 'Houston Astros',
        'Asheville Tourists': 'Houston Astros', 'Fayetteville Woodpeckers': 'Houston Astros',
        'FCL Astros': 'Houston Astros',
        'Salt Lake Bees': 'Los Angeles Angels', 'Rocket City Trash Pandas': 'Los Angeles Angels',
        'Tri-City Dust Devils': 'Los Angeles Angels', 'Inland Empire 66ers': 'Seattle Mariners',
        'ACL Angels': 'Los Angeles Angels',
        'Las Vegas Aviators': 'Athletics', 'Midland RockHounds': 'Athletics',
        'Lansing Lugnuts': 'Athletics', 'Stockton Ports': 'Athletics',
        'ACL Athletics': 'Athletics',
        'Tacoma Rainiers': 'Seattle Mariners', 'Arkansas Travelers': 'Seattle Mariners',
        'Everett AquaSox': 'Seattle Mariners',
        'ACL Mariners': 'Seattle Mariners',
        'Round Rock Express': 'Texas Rangers', 'Frisco RoughRiders': 'Texas Rangers',
        'Hub City Spartanburgers': 'Texas Rangers', 'Hickory Crawdads': 'Texas Rangers',
        'ACL Rangers': 'Texas Rangers',
        'Gwinnett Stripers': 'Atlanta Braves', 'Columbus Clingstones': 'Atlanta Braves',
        'Rome Emperors': 'Atlanta Braves', 'Augusta GreenJackets': 'Atlanta Braves',
        'FCL Braves': 'Atlanta Braves',
        'Jacksonville Jumbo Shrimp': 'Miami Marlins', 'Pensacola Blue Wahoos': 'Miami Marlins',
        'Beloit Sky Carp': 'Miami Marlins', 'Jupiter Hammerheads': 'Miami Marlins',
        'FCL Marlins': 'Miami Marlins',
        'Syracuse Mets': 'New York Mets', 'Binghamton Rumble Ponies': 'New York Mets',
        'Brooklyn Cyclones': 'New York Mets', 'St. Lucie Mets': 'New York Mets',
        'FCL Mets': 'New York Mets',
        'Lehigh Valley IronPigs': 'Philadelphia Phillies', 'Reading Fightin Phils': 'Philadelphia Phillies',
        'Jersey Shore BlueClaws': 'Philadelphia Phillies', 'Clearwater Threshers': 'Philadelphia Phillies',
        'FCL Phillies': 'Philadelphia Phillies',
        'Rochester Red Wings': 'Washington Nationals', 'Harrisburg Senators': 'Washington Nationals',
        'Wilmington Blue Rocks': 'Washington Nationals', 'Fredericksburg Nationals': 'Washington Nationals',
        'FCL Nationals': 'Washington Nationals',
        'Iowa Cubs': 'Chicago Cubs', 'Knoxville Smokies': 'Chicago Cubs',
        'South Bend Cubs': 'Chicago Cubs', 'Myrtle Beach Pelicans': 'Chicago Cubs',
        'ACL Cubs': 'Chicago Cubs',
        'Louisville Bats': 'Cincinnati Reds', 'Chattanooga Lookouts': 'Cincinnati Reds',
        'Dayton Dragons': 'Cincinnati Reds', 'Daytona Tortugas': 'Cincinnati Reds',
        'ACL Reds': 'Cincinnati Reds',
        'Nashville Sounds': 'Milwaukee Brewers', 'Biloxi Shuckers': 'Milwaukee Brewers',
        'Wisconsin Timber Rattlers': 'Milwaukee Brewers', 'Wilson Warbirds': 'Milwaukee Brewers',
        'ACL Brewers': 'Milwaukee Brewers',
        'Indianapolis Indians': 'Pittsburgh Pirates', 'Altoona Curve': 'Pittsburgh Pirates',
        'Greensboro Grasshoppers': 'Pittsburgh Pirates', 'Bradenton Marauders': 'Pittsburgh Pirates',
        'FCL Pirates': 'Pittsburgh Pirates',
        'Memphis Redbirds': 'St. Louis Cardinals', 'Springfield Cardinals': 'St. Louis Cardinals',
        'Peoria Chiefs': 'St. Louis Cardinals', 'Palm Beach Cardinals': 'St. Louis Cardinals',
        'FCL Cardinals': 'St. Louis Cardinals',
        'Reno Aces': 'Arizona Diamondbacks', 'Amarillo Sod Poodles': 'Arizona Diamondbacks',
        'Hillsboro Hops': 'Arizona Diamondbacks', 'Visalia Rawhide': 'Arizona Diamondbacks',
        'ACL D-backs': 'Arizona Diamondbacks',
        'Albuquerque Isotopes': 'Colorado Rockies', 'Hartford Yard Goats': 'Colorado Rockies',
        'Spokane Indians': 'Colorado Rockies', 'Fresno Grizzlies': 'Colorado Rockies',
        'ACL Rockies': 'Colorado Rockies',
        'Oklahoma City Comets': 'Los Angeles Dodgers', 'Tulsa Drillers': 'Los Angeles Dodgers',
        'Great Lakes Loons': 'Los Angeles Dodgers', 'Rancho Cucamonga Quakes': 'Los Angeles Angels',
        'Ontario Tower Buzzers': 'Los Angeles Dodgers', 'ACL Dodgers': 'Los Angeles Dodgers',
        'El Paso Chihuahuas': 'San Diego Padres', 'San Antonio Missions': 'San Diego Padres',
        'Fort Wayne TinCaps': 'San Diego Padres', 'Lake Elsinore Storm': 'San Diego Padres',
        'ACL Padres': 'San Diego Padres',
        'Sacramento River Cats': 'San Francisco Giants', 'Richmond Flying Squirrels': 'San Francisco Giants',
        'Eugene Emeralds': 'San Francisco Giants', 'San Jose Giants': 'San Francisco Giants',
        'ACL Giants': 'San Francisco Giants',
        # DSL Teams
        'DSL Orioles Black': 'Baltimore Orioles', 'DSL Orioles Orange': 'Baltimore Orioles',
        'DSL Red Sox Red': 'Boston Red Sox', 'DSL Red Sox Blue': 'Boston Red Sox',
        'DSL NYY Yankees': 'New York Yankees', 'DSL NYY Bombers': 'New York Yankees', 'DSL Rays': 'Tampa Bay Rays',
        'DSL Tampa Bay': 'Tampa Bay Rays',
        'DSL Blue Jays Blue': 'Toronto Blue Jays', 'DSL Blue Jays Red': 'Toronto Blue Jays',
        'DSL White Sox': 'Chicago White Sox',
        'DSL CLE Goryl': 'Cleveland Guardians', 'DSL CLE Mendoza': 'Cleveland Guardians',
        'DSL Tigers 1': 'Detroit Tigers', 'DSL Tigers 2': 'Detroit Tigers', 'DSL Royals Ventura': 'Kansas City Royals',
        'DSL Royals Fortuna': 'Kansas City Royals',
        'DSL Twins': 'Minnesota Twins', 'DSL Astros Blue': 'Houston Astros', 'DSL Astros Orange': 'Houston Astros',
        'DSL Angels': 'Los Angeles Angels', 'DSL Athletics': 'Athletics', 'DSL Mariners': 'Seattle Mariners',
        'DSL Rangers Red': 'Texas Rangers', 'DSL Rangers Blue': 'Texas Rangers',
        'DSL Braves': 'Atlanta Braves', 'DSL Marlins': 'Miami Marlins', 'DSL Miami': 'Miami Marlins', 'DSL Mets Blue': 'New York Mets',
        'DSL Mets Orange': 'New York Mets',
        'DSL Phillies Red': 'Philadelphia Phillies', 'DSL Phillies White': 'Philadelphia Phillies',
        'DSL Nationals': 'Washington Nationals', 'DSL Cubs Red': 'Chicago Cubs', 'DSL Cubs Blue': 'Chicago Cubs',
        'DSL Reds': 'Cincinnati Reds', 'DSL Rojos': 'Cincinnati Reds', 'DSL Brewers Gold': 'Milwaukee Brewers',
        'DSL Brewers Blue': 'Milwaukee Brewers',
        'DSL Pirates Black': 'Pittsburgh Pirates', 'DSL Pirates Gold': 'Pittsburgh Pirates',
        'DSL Cardinals': 'St. Louis Cardinals',
        'DSL Arizona Black': 'Arizona Diamondbacks', 'DSL Arizona Red': 'Arizona Diamondbacks',
        'DSL Rockies': 'Colorado Rockies', 'DSL Colorado': 'Colorado Rockies',
        'DSL LAD Bautista': 'Los Angeles Dodgers', 'DSL LAD Mega': 'Los Angeles Dodgers',
        'DSL Padres Brown': 'San Diego Padres', 'DSL Padres Gold': 'San Diego Padres',
        'DSL Giants Orange': 'San Francisco Giants', 'DSL Giants Black': 'San Francisco Giants',
    }

    # Level mapping for MiLB teams
    level_mapping = {
        # AAA Teams
        'Norfolk Tides': 'AAA', 'Worcester Red Sox': 'AAA', 'Scranton/Wilkes-Barre RailRiders': 'AAA',
        'Durham Bulls': 'AAA', 'Buffalo Bisons': 'AAA', 'Charlotte Knights': 'AAA',
        'Columbus Clippers': 'AAA', 'Toledo Mud Hens': 'AAA', 'Omaha Storm Chasers': 'AAA',
        'St. Paul Saints': 'AAA', 'Sugar Land Space Cowboys': 'AAA', 'Salt Lake Bees': 'AAA',
        'Las Vegas Aviators': 'AAA', 'Tacoma Rainiers': 'AAA', 'Round Rock Express': 'AAA',
        'Gwinnett Stripers': 'AAA', 'Jacksonville Jumbo Shrimp': 'AAA', 'Syracuse Mets': 'AAA',
        'Lehigh Valley IronPigs': 'AAA', 'Rochester Red Wings': 'AAA', 'Iowa Cubs': 'AAA',
        'Louisville Bats': 'AAA', 'Nashville Sounds': 'AAA', 'Indianapolis Indians': 'AAA',
        'Memphis Redbirds': 'AAA', 'Reno Aces': 'AAA', 'Albuquerque Isotopes': 'AAA',
        'Oklahoma City Comets': 'AAA', 'El Paso Chihuahuas': 'AAA', 'Sacramento River Cats': 'AAA',

        # AA Teams
        'Chesapeake Baysox': 'AA', 'Portland Sea Dogs': 'AA', 'Somerset Patriots': 'AA',
        'Montgomery Biscuits': 'AA', 'New Hampshire Fisher Cats': 'AA', 'Birmingham Barons': 'AA',
        'Akron RubberDucks': 'AA', 'Erie SeaWolves': 'AA', 'Northwest Arkansas Naturals': 'AA',
        'Wichita Wind Surge': 'AA', 'Corpus Christi Hooks': 'AA', 'Rocket City Trash Pandas': 'AA',
        'Midland RockHounds': 'AA', 'Arkansas Travelers': 'AA', 'Frisco RoughRiders': 'AA',
        'Columbus Clingstones': 'AA', 'Pensacola Blue Wahoos': 'AA', 'Binghamton Rumble Ponies': 'AA',
        'Reading Fightin Phils': 'AA', 'Harrisburg Senators': 'AA', 'Knoxville Smokies': 'AA',
        'Chattanooga Lookouts': 'AA', 'Biloxi Shuckers': 'AA', 'Altoona Curve': 'AA',
        'Springfield Cardinals': 'AA', 'Amarillo Sod Poodles': 'AA', 'Hartford Yard Goats': 'AA',
        'Tulsa Drillers': 'AA', 'San Antonio Missions': 'AA', 'Richmond Flying Squirrels': 'AA',

        # A+ Teams
        'Frederick Keys': 'A+', 'Greenville Drive': 'A+', 'Hudson Valley Renegades': 'A+',
        'Bowling Green Hot Rods': 'A+', 'Vancouver Canadians': 'A+', 'Winston-Salem Dash': 'A+',
        'Lake County Captains': 'A+', 'West Michigan Whitecaps': 'A+', 'Quad Cities River Bandits': 'A+',
        'Cedar Rapids Kernels': 'A+', 'Asheville Tourists': 'A+', 'Tri-City Dust Devils': 'A+',
        'Lansing Lugnuts': 'A+', 'Everett AquaSox': 'A+', 'Hub City Spartanburgers': 'A+',
        'Rome Emperors': 'A+', 'Beloit Sky Carp': 'A+', 'Brooklyn Cyclones': 'A+',
        'Jersey Shore BlueClaws': 'A+', 'Wilmington Blue Rocks': 'A+', 'South Bend Cubs': 'A+',
        'Dayton Dragons': 'A+', 'Wisconsin Timber Rattlers': 'A+', 'Greensboro Grasshoppers': 'A+',
        'Peoria Chiefs': 'A+', 'Hillsboro Hops': 'A+', 'Spokane Indians': 'A+',
        'Great Lakes Loons': 'A+', 'Fort Wayne TinCaps': 'A+', 'Eugene Emeralds': 'A+',

        # A Teams
        'Delmarva Shorebirds': 'A', 'Salem RidgeYaks': 'A', 'Tampa Tarpons': 'A',
        'Charleston RiverDogs': 'A', 'Dunedin Blue Jays': 'A', 'Kannapolis Cannon Ballers': 'A',
        'Hill City Howlers': 'A', 'Lakeland Flying Tigers': 'A', 'Columbia Fireflies': 'A',
        'Fort Myers Mighty Mussels': 'A', 'Fayetteville Woodpeckers': 'A', 'Inland Empire 66ers': 'A',
        'Stockton Ports': 'A', 'Ontario Tower Buzzers': 'A', 'Hickory Crawdads': 'A',
        'Augusta GreenJackets': 'A', 'Jupiter Hammerheads': 'A', 'St. Lucie Mets': 'A',
        'Clearwater Threshers': 'A', 'Fredericksburg Nationals': 'A', 'Myrtle Beach Pelicans': 'A',
        'Daytona Tortugas': 'A', 'Wilson Warbirds': 'A', 'Bradenton Marauders': 'A',
        'Palm Beach Cardinals': 'A', 'Visalia Rawhide': 'A', 'Fresno Grizzlies': 'A',
        'Rancho Cucamonga Quakes': 'A', 'Lake Elsinore Storm': 'A', 'San Jose Giants': 'A',
    }

    def get_team_and_level(milb_team_name):
        """
        Return (mlb_team, level) tuple from a MiLB team name.
        Example: 'Norfolk Tides' -> ('Baltimore Orioles', 'AAA')
                 'FCL Orioles' -> ('Baltimore Orioles', 'CPX')
                 'DSL Twins' -> ('Minnesota Twins', 'DSL')
                 Unknown team -> (original_name, '')
        """
        if not milb_team_name:
            return (None, '')

        # Check for DSL teams
        if milb_team_name.startswith('DSL '):
            mlb_team = team_mapping.get(milb_team_name)
            if mlb_team:
                return (mlb_team, 'DSL')
            return (milb_team_name, '')

        # Check for ACL/FCL teams
        if milb_team_name.startswith('ACL ') or milb_team_name.startswith('FCL '):
            mlb_team = team_mapping.get(milb_team_name)
            if mlb_team:
                return (mlb_team, 'CPX')
            return (milb_team_name, '')

        # Regular MiLB teams
        mlb_team = team_mapping.get(milb_team_name)
        level = level_mapping.get(milb_team_name, '')

        if mlb_team:
            return (mlb_team, level)
        else:
            return (milb_team_name, '')

    def format_name_last_first(name):
        """
        Convert 'First Last' to 'Last, First'
        Handles suffixes (Jr., Sr., III), surname particles (de, van, el, o', etc.),
        compound surnames (Montes de Oca, Woods Richardson), and initials (A.J., T.J.)
        """
        if not name or ' ' not in name.strip():
            return name

        name = name.strip()

        # Suffixes to strip and reattach to the last name
        suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}

        parts = name.split()

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
        # the last name begins. A token that is a known prefix starts the last name.
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
            return name

        return f"{last_name}, {first_name}"

    def parse_transaction(transaction_text, team_name):
        """Parse transaction text to extract player, position, and transaction type."""

        # Skip header rows
        if transaction_text == 'Transaction' or not transaction_text:
            return None

        transaction_text = transaction_text.strip()

        # Define position pattern once for reuse
        POS = r'(RHP|LHP|CF|LF|RF|INF|1B|2B|3B|SS|OF|DH|TWP|IF|C)'

        # Pattern for outrighted: "Team outrighted POSITION Player to MiLB Team"
        pattern_outright = rf'outright(?:ed)?\s+{POS}\s+(.+?)\s+(?:to|off)\b'
        match = re.search(pattern_outright, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(2).strip(),
                'position': match.group(1),
                'team': team_name,
                'transaction_type': 'outrighted'
            }

        # Pattern 1: "Team released/signed POSITION Player ."
        # The name is captured token-by-token (word chars, accents, periods,
        # apostrophes, hyphens) rather than as [^.]+ — the old class stopped
        # at the first period, so "P.J. Labriola" came out as "P".
        NAME = (r'([\wÀ-ſ][\wÀ-ſ.\'\-]*'
                r'(?:\s+[\wÀ-ſ][\wÀ-ſ.\'\-]+)*)')
        pattern1 = rf'(released|signed)\s+{POS}\s+{NAME}\s*\.'
        match = re.search(pattern1, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(3).strip(),
                'position': match.group(2),
                'team': team_name,
                'transaction_type': match.group(1).lower()
            }

        # Pattern for retirements: "POSITION Player retired."
        pattern_retired = rf'^{POS}\s+(.+?)\s+retired\s*\.'
        match = re.search(pattern_retired, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(2).strip(),
                'position': match.group(1),
                'team': team_name,
                'transaction_type': 'retired'
            }

        # Pattern 2: "POSITION Player elected free agency"
        pattern2 = rf'^{POS}\s+(.+?)\s+elected free agency'
        match = re.search(pattern2, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(2).strip(),
                'position': match.group(1),
                'team': None,  # Free agency means no current team
                'transaction_type': 'elected free agency'
            }

        # Pattern 3: Just "released" without position (common in DSL/ACL)
        if 'released' in transaction_text.lower():
            # Require whitespace after the optional position so a name whose
            # first letter is a position token (e.g. 'C'arlos) isn't truncated.
            pattern3 = rf'released\s+(?:{POS}\s+)?([\w\u00C0-\u017F][\w\u00C0-\u017F.\']+(?:\s+[\w\u00C0-\u017F][\w\u00C0-\u017F.\']+)*)\s*\.'
            match = re.search(pattern3, transaction_text)
            if match:
                return {
                    'player': match.group(2).strip() if match.group(2) else None,
                    'position': match.group(1) if match.group(1) else None,
                    'team': team_name,
                    'transaction_type': 'released'
                }

        # Pattern 4: Signing - "Team signed free agent [POSITION] Player to a minor league contract"
        # Position is optional to handle both cases
        # Require whitespace after the optional position so a name whose first
        # letter is a position token (e.g. 'C'arlos) isn't truncated.
        pattern4 = rf'signed free agent\s+(?:{POS}\s+)?(.+?)\s+to\b'
        match = re.search(pattern4, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(2).strip(),
                'position': match.group(1),  # Will be None if no position
                'team': team_name,
                'transaction_type': 'signing'
            }

        # Pattern 5: Selected contract - "Team selected the contract of POSITION Player from MiLB Team"
        pattern5 = rf'selected the contract of\s+{POS}\s+(.+?)\s+from'
        match = re.search(pattern5, transaction_text, re.IGNORECASE)
        if match:
            return {
                'player': match.group(2).strip(),
                'position': match.group(1),
                'team': team_name,
                'transaction_type': 'selected contract'
            }

        # If no patterns match, return None
        return None

    def check_for_more_pages(soup, current_page):
        """
        More robust pagination detection.
        Returns the highest page number found, or current_page if no more pages detected.
        """
        max_page = current_page

        # Method 1: Look for pagination div/nav
        pagination = soup.find('div', class_='pagination') or soup.find('nav', class_='pagination')

        if pagination:
            # Look for page links
            page_links = pagination.find_all('a')
            for link in page_links:
                link_text = link.get_text(strip=True)
                # Try to extract page numbers
                if link_text.isdigit():
                    page_num = int(link_text)
                    if page_num > max_page:
                        max_page = page_num

                # Also check href for page numbers (e.g., /p-2, /p-3)
                href = link.get('href', '')
                page_match = re.search(r'/p-(\d+)', href)
                if page_match:
                    page_num = int(page_match.group(1))
                    if page_num > max_page:
                        max_page = page_num

        # Method 2: Look for any links containing /p-X pattern in the entire page
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            page_match = re.search(r'/p-(\d+)', href)
            if page_match:
                page_num = int(page_match.group(1))
                if page_num > max_page:
                    max_page = page_num

        # Method 3: Try to find "Page X" text
        page_text = soup.get_text()
        page_matches = re.findall(r'Page\s+(\d+)', page_text, re.IGNORECASE)
        for match in page_matches:
            page_num = int(match)
            if page_num > max_page:
                max_page = page_num

        return max_page

    # Convert dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    all_transactions = []

    # Iterate through each date
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nScraping transactions for {date_str}...")

        page = 1
        date_total_transactions = 0

        while True:  # Continue until we determine there are no more pages
            if page == 1:
                url = f"https://www.milb.com/transactions/{date_str}"
            else:
                url = f"https://www.milb.com/transactions/{date_str}/p-{page}"

            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers)

                # Check if page exists (404 means no more pages)
                if response.status_code == 404:
                    print(f"  Page {page} returned 404 - no more pages for this date")
                    break

                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find the transaction table
                table = soup.find('table')
                if not table:
                    print(f"  No table found on page {page} - ending pagination for this date")
                    break

                rows = table.find_all('tr')
                transactions_on_page = 0

                for row in rows:
                    cells = row.find_all('td')

                    # Should have 3 cells: Team (logo), Date, Transaction
                    if len(cells) >= 3:
                        # Extract team from first cell
                        team_cell = cells[0]
                        milb_team_name = None

                        img = team_cell.find('img')
                        if img:
                            milb_team_name = img.get('alt') or img.get('title')

                        if not milb_team_name:
                            team_text = team_cell.get_text(strip=True)
                            if team_text and team_text != 'Team':
                                milb_team_name = team_text

                        # Get transaction text
                        transaction_text = cells[2].get_text(separator=' ', strip=True)
                        transaction_text = re.sub(r'^\d{2}/\d{2}/\d{2}\s*', '', transaction_text)
                        transaction_text = re.sub(r'\s+', ' ', transaction_text)

                        parsed = parse_transaction(transaction_text, milb_team_name)

                        if parsed and parsed['transaction_type']:
                            # Determine which team name to use
                            if parsed['team']:
                                milb_team_for_formatting = parsed['team']
                            elif milb_team_name:
                                milb_team_for_formatting = milb_team_name
                            else:
                                milb_team_for_formatting = None

                            # Get team and level separately
                            team, level = get_team_and_level(milb_team_for_formatting)

                            # Format player name as Last, First
                            formatted_player = format_name_last_first(parsed['player'])

                            all_transactions.append({
                                'date': date_str,
                                'transaction_type': parsed['transaction_type'],
                                'player': formatted_player,
                                'position': parsed['position'],
                                'team': team,
                                'level': level
                            })
                            transactions_on_page += 1

                date_total_transactions += transactions_on_page
                print(f"  Page {page}: Found {transactions_on_page} transactions")

                # Check if there are more pages
                max_page = check_for_more_pages(soup, page)

                if max_page > page:
                    print(f"  Detected pages up to page {max_page}")
                    page += 1
                    time.sleep(1)  # Be nice to the server
                else:
                    print(f"  No more pages detected after page {page}")
                    break

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  Page {page} not found (404) - no more pages for this date")
                else:
                    print(f"  HTTP Error on {url}: {e}")
                break
            except Exception as e:
                print(f"  Error on {url}: {e}")
                break

        print(f"  Total transactions for {date_str}: {date_total_transactions}")
        current_date += timedelta(days=1)

    # Create DataFrame
    df = pd.DataFrame(all_transactions)

    if not df.empty:
        df = df.drop_duplicates(subset=['player', 'date', 'transaction_type'], keep='first')
        df.columns = ['Date', 'Transaction Type', 'Player', 'Position', 'Team', 'Level']

    return df


def main():
    # ── Settings (edit these directly or override via command line) ──
    start_date  = '2026-08-02'
    end_date    = datetime.now().strftime('%Y-%m-%d')

    # ── CLI overrides (optional) ──
    parser = argparse.ArgumentParser(description='Scrape MiLB transactions')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    args = parser.parse_args()

    if args.start is not None: start_date = args.start
    if args.end is not None: end_date = args.end

    master_file = os.path.expanduser("~/Downloads/milb_transactions_master.csv")

    print(f"Starting scrape from {start_date} to {end_date}")
    print("=" * 70)

    df_new = scrape_milb_transactions(start_date, end_date)

    if df_new.empty:
        print("\nNo transactions found.")
        return

    print(f"\n{'=' * 70}")
    print(f"SCRAPING COMPLETE: Found {len(df_new)} transactions total")
    print("=" * 70)

    # Load existing master file if it exists
    try:
        df_master = pd.read_csv(master_file)
        print(f"\nLoaded {len(df_master)} existing transactions from master file")

        # Create unique identifiers
        df_master['unique_id'] = (
                df_master['Player'].astype(str) + '|' +
                df_master['Date'].astype(str) + '|' +
                df_master['Transaction Type'].astype(str)
        )
        df_new['unique_id'] = (
                df_new['Player'].astype(str) + '|' +
                df_new['Date'].astype(str) + '|' +
                df_new['Transaction Type'].astype(str)
        )

        # Find new transactions
        new_unique_ids = set(df_new['unique_id']) - set(df_master['unique_id'])
        df_new_only = df_new[df_new['unique_id'].isin(new_unique_ids)].copy()

        df_new_only = df_new_only.drop(columns=['unique_id'])
        df_master = df_master.drop(columns=['unique_id'])

        if not df_new_only.empty:
            print(f"\nFound {len(df_new_only)} NEW transactions not in master file:")
            print(df_new_only.to_string(index=False))

            df_combined = pd.concat([df_master, df_new_only], ignore_index=True)
            df_combined = df_combined.sort_values(['Date', 'Team', 'Player'], ascending=[False, True, True])
            df_combined.to_csv(master_file, index=False)
            print(f"\nAdded {len(df_new_only)} new transactions to master file")

            new_only_file = os.path.expanduser(f"~/Downloads/milb_transactions_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_new_only.to_csv(new_only_file, index=False)
            print(f"New transactions also saved to: {new_only_file}")
        else:
            print(f"\nNo new transactions found - all {len(df_new)} transactions already exist in master file")

    except FileNotFoundError:
        print(f"\nMaster file not found - creating new master file")
        df_new.to_csv(master_file, index=False)
        print(f"Created master file with {len(df_new)} transactions: {master_file}")

        print("\nFirst 10 transactions:")
        print(df_new.head(10).to_string(index=False))

    # Print summary statistics
    try:
        df_master = pd.read_csv(master_file)
        print(f"\n{'=' * 70}")
        print(f"MASTER FILE SUMMARY")
        print("=" * 70)
        print(f"Total transactions: {len(df_master)}")
        print(f"\nTransactions by type:")
        print(df_master['Transaction Type'].value_counts())
    except Exception:
        pass  # summary print is cosmetic; a malformed master file shouldn't kill the run


if __name__ == "__main__":
    main()