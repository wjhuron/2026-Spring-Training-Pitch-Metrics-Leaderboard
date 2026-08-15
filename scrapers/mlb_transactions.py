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


def scrape_mlb_transactions(start_date, end_date):
    """Scrape MLB transactions between specified dates using requests + BeautifulSoup."""

    mlb_teams = [
        'Arizona Diamondbacks', 'Athletics', 'Atlanta Braves', 'Baltimore Orioles', 'Boston Red Sox',
        'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds', 'Cleveland Guardians',
        'Colorado Rockies', 'Detroit Tigers', 'Houston Astros', 'Kansas City Royals',
        'Los Angeles Angels', 'Los Angeles Dodgers', 'Miami Marlins', 'Milwaukee Brewers',
        'Minnesota Twins', 'New York Mets', 'New York Yankees',
        'Philadelphia Phillies', 'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants',
        'Seattle Mariners', 'St. Louis Cardinals', 'Tampa Bay Rays', 'Texas Rangers',
        'Toronto Blue Jays', 'Washington Nationals'
    ]

    # Articles, prepositions, and transaction verbs that can never be part of a
    # player name — used to truncate the position-anchored name and to reject
    # garbage runs in the no-position fallback.
    _NAME_STOP = frozenset({
        'a', 'an', 'the', 'and', 'or', 'to', 'from', 'for', 'with', 'on', 'off', 'of', 'as',
        'was', 'were', 'is', 'are', 'has', 'have', 'had', 'been', 'his', 'their', 'right', 'left',
        'released', 'signed', 'claimed', 'selected', 'purchased', 'traded', 'acquired',
        'returned', 'retired', 'elected', 'free', 'agency', 'waivers', 'waiver', 'assignment',
        'outright', 'outrighted', 'designated', 'activated', 'reinstated', 'optioned',
        'recalled', 'placed', 'transferred', 'sent', 'contract', 'minor', 'league', 'major',
        'disabled', 'injured', 'list', 'day',
    })

    def parse_transaction(transaction_text, team_name):
        """Parse transaction text to extract player, position, and transaction type."""
        if not transaction_text or transaction_text == 'Transaction':
            return None

        transaction_text = transaction_text.strip()
        transaction_lower = transaction_text.lower()

        # Determine transaction type
        transaction_type = None
        if 'released' in transaction_lower:
            transaction_type = 'released'
        elif 'signed' in transaction_lower and 'assigned' not in transaction_lower:
            # Skip minor league signings
            if 'minor league' in transaction_lower:
                return None
            transaction_type = 'signed'
        elif 'elected free agency' in transaction_lower:
            transaction_type = 'elected free agency'
        elif 'claimed' in transaction_lower:
            transaction_type = 'claimed'
        elif 'outright' in transaction_lower:
            transaction_type = 'outrighted'
        elif 'selected the contract' in transaction_lower:
            transaction_type = 'selected'
        elif 'traded' in transaction_lower or 'acquired' in transaction_lower:
            transaction_type = 'traded'
        elif 'purchased' in transaction_lower:
            transaction_type = 'purchased'
        elif 'returned' in transaction_lower:
            transaction_type = 'returned'
        elif 'retired' in transaction_lower:
            transaction_type = 'retired'

        if not transaction_type:
            return None

        # Extract player name and position
        positions = ['RHP', 'LHP', 'C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF',
                     'DH', 'IF', 'INF', 'P', 'UTIL', 'TWP']

        player = None
        position = None

        for pos in positions:
            pattern = rf'\b{pos}\b\s+([\w\u00C0-\u017F.\'-]+(?:\s+[\w\u00C0-\u017F.\'-]+){{0,4}})'
            match = re.search(pattern, transaction_text)
            if match:
                position = pos
                player_name = match.group(1).strip()
                # Truncate at the first transaction-related word (the name ends
                # where the verb/preposition tail begins).
                player_name = re.sub(
                    r'\s+(' + '|'.join(_NAME_STOP) + r')\b.*$', '',
                    player_name, flags=re.IGNORECASE)
                # Strip trailing separators but PRESERVE a "Jr."/"Sr." period so
                # the name dedupes cleanly against other sources that keep it.
                player_name = re.sub(r'[,;:]+$', '', player_name.strip())
                if not re.search(r'\b(?:Jr|Sr)\.$', player_name, flags=re.IGNORECASE):
                    player_name = re.sub(r'\.+$', '', player_name)
                player = player_name.strip() or None
                if player:
                    break

        if not player:
            # No position token found \u2014 take the first maximal run of 2-4
            # consecutive name-like tokens: capitalized (accents allowed), not a
            # transaction/stopword, and not part of a team name. This isolates a
            # leading name ("Juan Soto elected free agency") that a greedy regex
            # would swallow together with the following verb.
            team_tokens = set()
            for tm in mlb_teams:
                team_tokens.update(tm.split())
            run = []
            for tok in transaction_text.split() + ['\x00']:  # sentinel flushes tail
                core = tok.strip(".,;:'\"")
                is_name = bool(re.match(r"[A-Z\u00C0-\u017F]", tok)) and core.lower() not in _NAME_STOP and core not in team_tokens
                if is_name:
                    run.append(tok.strip(",;:"))
                else:
                    if 2 <= len(run) <= 4:
                        cand = ' '.join(run)
                        if not re.search(r'\b(?:Jr|Sr)\.$', cand, flags=re.IGNORECASE):
                            cand = re.sub(r'\.+$', '', cand)
                        player = cand
                        break
                    run = []

        if player:
            return {
                'player': player,
                'position': position,
                'team': team_name,
                'transaction_type': transaction_type
            }

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

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    })

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    all_transactions = []

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\nScraping transactions for {date_str}...")

        page = 1
        date_total_transactions = 0

        while True:
            url_date = current_date.strftime('%Y/%m/%d')
            if page == 1:
                url = f"https://www.mlb.com/transactions/{url_date}"
            else:
                url = f"https://www.mlb.com/transactions/{url_date}/p-{page}"

            try:
                response = session.get(url, allow_redirects=True, timeout=30)

                if response.status_code == 404:
                    print(f"  Page {page} returned 404 - no more pages")
                    break

                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                table = soup.find('table')
                if not table:
                    print(f"  No table found on page {page}")
                    break

                rows = table.find_all('tr')
                transactions_on_page = 0

                for row in rows:
                    cells = row.find_all('td')

                    if len(cells) >= 3:
                        # Extract team from first cell (image alt text)
                        team_cell = cells[0]
                        team_name = None

                        img = team_cell.find('img')
                        if img:
                            team_name = img.get('alt') or img.get('title')

                        if not team_name:
                            team_text = team_cell.get_text(strip=True)
                            if team_text and team_text != 'Team':
                                team_name = team_text

                        # Get transaction text
                        transaction_text = cells[2].get_text(separator=' ', strip=True)
                        # Strip leading date prefix if present
                        transaction_text = re.sub(r'^\d{2}/\d{2}/\d{2}\s*', '', transaction_text)
                        transaction_text = re.sub(r'\s+', ' ', transaction_text)

                        parsed = parse_transaction(transaction_text, team_name)

                        if parsed and parsed['transaction_type'] and parsed['player']:
                            all_transactions.append({
                                'date': date_str,
                                'transaction_type': parsed['transaction_type'],
                                'player': format_name_last_first(parsed['player']),
                                'position': parsed['position'],
                                'team': parsed['team'],
                                'level': '',
                            })
                            transactions_on_page += 1

                date_total_transactions += transactions_on_page
                print(f"  Page {page}: Found {transactions_on_page} transactions")

                # Check for more pages
                max_page = check_for_more_pages(soup, page)
                if max_page > page:
                    print(f"  Detected pages up to page {max_page}")
                    page += 1
                    time.sleep(1)
                else:
                    print(f"  No more pages detected after page {page}")
                    break

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  Page {page} not found (404)")
                else:
                    print(f"  HTTP Error on {url}: {e}")
                break
            except Exception as e:
                print(f"  Error on {url}: {e}")
                break

        print(f"  Total transactions for {date_str}: {date_total_transactions}")
        current_date += timedelta(days=1)

    df = pd.DataFrame(all_transactions)

    if not df.empty:
        df = df.drop_duplicates(subset=['player', 'date', 'transaction_type'], keep='first')
        df.columns = ['Date', 'Transaction Type', 'Player', 'Position', 'Team', 'Level']

    return df


def main():
    # ── Settings (edit these directly or override via command line) ──
    start_date  = '2026-02-19'
    end_date    = datetime.now().strftime('%Y-%m-%d')

    # ── CLI overrides (optional) ──
    parser = argparse.ArgumentParser(description='Scrape MLB transactions')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    args = parser.parse_args()

    if args.start is not None: start_date = args.start
    if args.end is not None: end_date = args.end

    master_file = os.path.expanduser("~/Downloads/mlb_transactions_master.csv")

    print(f"Starting scrape from {start_date} to {end_date}")
    print("-" * 50)

    df_new = scrape_mlb_transactions(start_date, end_date)

    if df_new.empty:
        print("No transactions found.")
        return

    print(f"\nFound {len(df_new)} transactions from scrape")

    try:
        df_master = pd.read_csv(master_file)
        print(f"Loaded {len(df_master)} existing transactions from master file")

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

            new_only_file = os.path.expanduser(f"~/Downloads/mlb_transactions_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_new_only.to_csv(new_only_file, index=False)
            print(f"New transactions also saved to: {new_only_file}")
        else:
            print(f"\nNo new transactions found - all {len(df_new)} transactions already exist in master file")

    except FileNotFoundError:
        print(f"Master file not found - creating new master file")
        df_new.to_csv(master_file, index=False)
        print(f"Created master file with {len(df_new)} transactions: {master_file}")
        print("\nFirst 10 transactions:")
        print(df_new.head(10).to_string(index=False))

    try:
        df_master = pd.read_csv(master_file)
        print(f"\nMaster file now contains {len(df_master)} total transactions")
        print(f"\nTransactions by type:")
        print(df_master['Transaction Type'].value_counts())
    except Exception:
        pass


if __name__ == "__main__":
    main()
