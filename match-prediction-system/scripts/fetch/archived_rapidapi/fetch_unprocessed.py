#!/usr/bin/env python3
"""
Fetch raw tennis matches from RapidAPI with ALL available fields.
Saves to data/raw/unprocessed_matches.csv with complete match information.

Usage:
  python scripts/fetch/fetch_unprocessed.py --start 2025-10-29 --end 2025-10-31
  python scripts/fetch/fetch_unprocessed.py --last-week
  python scripts/fetch/fetch_unprocessed.py --date 2025-10-30
"""

import argparse
import sys
import json
import requests
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

class RawMatchFetcher:
    """Fetch complete raw match data from RapidAPI."""
    
    BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('RAPIDAPI_TENNIS_KEY')
        if not self.api_key:
            raise ValueError("RapidAPI Tennis API key not found. Set RAPIDAPI_TENNIS_KEY environment variable.")
        
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com"
        }
        
        self.tournament_cache = {}
    
    def _make_request(self, endpoint, params=None):
        """Make API request with error handling."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Request failed for {url}: {e}")
            raise
    
    def _get_tournament_details(self, tournament_id, tour="atp"):
        """Get full tournament details."""
        if tournament_id in self.tournament_cache:
            return self.tournament_cache[tournament_id]
        
        try:
            endpoint = f"/tennis/v2/{tour.lower()}/tournament/seasons/{tournament_id}"
            data = self._make_request(endpoint)
            seasons = data.get('data', [])
            if seasons:
                details = {
                    'tournament_name': seasons[0].get('name', ''),
                    'tournament_city': seasons[0].get('city', ''),
                    'tournament_country': seasons[0].get('country', ''),
                }
                self.tournament_cache[tournament_id] = details
                return details
        except Exception as e:
            print(f"Warning: Failed to get tournament details for {tournament_id}: {e}")
        
        return {'tournament_name': '', 'tournament_city': '', 'tournament_country': ''}
    
    def fetch_all_matches(self, start_date, end_date, tour="atp"):
        """Fetch ALL match data with complete fields."""
        print(f"Fetching {tour.upper()} matches from {start_date} to {end_date}...")
        
        # Get fixtures in date range
        endpoint = f"/tennis/v2/{tour.lower()}/fixtures/{start_date}/{end_date}"
        fixtures_data = self._make_request(endpoint, params={"pageSize": "100"})
        fixtures = fixtures_data.get('data', [])
        
        print(f"Found {len(fixtures)} fixtures")
        
        # Get unique tournament IDs
        tournament_ids = set()
        for fixture in fixtures:
            tid = fixture.get('tournamentId')
            if tid:
                tournament_ids.add(tid)
        
        print(f"Found {len(tournament_ids)} tournaments")
        
        # Fetch all matches from these tournaments
        all_raw_matches = []
        
        for tournament_id in tournament_ids:
            try:
                # Get match results (skip detailed tournament API call to save time/requests)
                results_endpoint = f"/tennis/v2/{tour.lower()}/tournament/results/{tournament_id}"
                results_data = self._make_request(results_endpoint)
                
                singles_matches = results_data.get('data', {}).get('singles', [])
                
                # Tournament details can be extracted from first match
                tournament_details = {'tournament_name': '', 'tournament_city': '', 'tournament_country': ''}
                if singles_matches:
                    first_match = singles_matches[0]
                    tournament_details = {
                        'tournament_name': first_match.get('tournament', {}).get('name', ''),
                        'tournament_city': first_match.get('tournament', {}).get('city', ''),
                        'tournament_country': first_match.get('tournament', {}).get('country', ''),
                    }
                
                for match in singles_matches:
                    match_date = match.get('date', '')[:10]
                    if start_date <= match_date <= end_date:
                        # Extract ALL available fields
                        raw_match = self._extract_all_fields(match, tournament_id, tournament_details, tour)
                        all_raw_matches.append(raw_match)
                
                print(f"  Processed tournament {tournament_id}: {tournament_details.get('tournament_name', 'Unknown')}")
                
            except Exception as e:
                print(f"  Warning: Failed to fetch tournament {tournament_id}: {e}")
                continue
        
        print(f"Total matches fetched: {len(all_raw_matches)}")
        return all_raw_matches
    
    def _extract_all_fields(self, match_data, tournament_id, tournament_details, tour):
        """Extract every available field from the match data."""
        
        # Parse date
        date_str = match_data.get('date', '')
        match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d') if date_str else None
        match_time = datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime('%H:%M:%S') if date_str else None
        
        # Player 1 data
        player1 = match_data.get('player1', {})
        player1_id = match_data.get('player1Id')
        player1_name = player1.get('name', '')
        player1_country = player1.get('countryAcr', '')
        player1_seed = player1.get('seed', '')
        
        # Player 2 data
        player2 = match_data.get('player2', {})
        player2_id = match_data.get('player2Id')
        player2_name = player2.get('name', '')
        player2_country = player2.get('countryAcr', '')
        player2_seed = player2.get('seed', '')
        
        # Match outcome
        winner_id = match_data.get('match_winner')
        winner_name = player1_name if winner_id == player1_id else player2_name
        
        # Score information
        score = match_data.get('score', '')
        sets_player1 = match_data.get('setsPlayer1', '')
        sets_player2 = match_data.get('setsPlayer2', '')
        
        # Match details
        round_name = match_data.get('round', '')
        match_status = match_data.get('status', '')
        
        # Tournament info
        tournament_name = tournament_details.get('tournament_name', '')
        tournament_city = tournament_details.get('tournament_city', '')
        tournament_country = tournament_details.get('tournament_country', '')
        
        # Build comprehensive record
        return {
            # Match identifiers
            'match_id': match_data.get('id', ''),
            'tournament_id': tournament_id,
            'tour': tour.upper(),
            
            # Date/Time
            'date': match_date,
            'time': match_time,
            'datetime_full': date_str,
            
            # Player 1
            'player1_id': player1_id,
            'player1_name': player1_name,
            'player1_country': player1_country,
            'player1_seed': player1_seed,
            
            # Player 2
            'player2_id': player2_id,
            'player2_name': player2_name,
            'player2_country': player2_country,
            'player2_seed': player2_seed,
            
            # Match outcome
            'winner_id': winner_id,
            'winner_name': winner_name,
            'score': score,
            'sets_player1': sets_player1,
            'sets_player2': sets_player2,
            
            # Match details
            'round': round_name,
            'status': match_status,
            
            # Tournament
            'tournament_name': tournament_name,
            'tournament_city': tournament_city,
            'tournament_country': tournament_country,
            
            # Raw JSON for any additional fields
            'raw_data': json.dumps(match_data),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Fetch raw tennis matches with ALL available fields',
        epilog='Saves to data/raw/unprocessed_matches.csv'
    )
    
    # Date options
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--date', help='Single date YYYY-MM-DD')
    g.add_argument('--start', help='Start date YYYY-MM-DD (requires --end)')
    g.add_argument('--last-week', action='store_true', help='Fetch last 7 days')
    
    parser.add_argument('--end', help='End date YYYY-MM-DD (required with --start)')
    parser.add_argument('--tour', action='append', choices=['atp', 'wta'], 
                       default=None, help='Tour(s) to fetch (default: atp)')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.last_week:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
    elif args.date:
        start = end = args.date
    else:
        if not args.end:
            parser.error('--end is required when using --start')
        start, end = args.start, args.end
    
    # Default to ATP
    tours = args.tour if args.tour else ['atp']
    
    # Fetch data
    fetcher = RawMatchFetcher()
    all_matches = []
    
    for tour in tours:
        try:
            matches = fetcher.fetch_all_matches(start, end, tour)
            all_matches.extend(matches)
        except Exception as e:
            print(f"Error fetching {tour.upper()} matches: {e}")
            sys.exit(1)
    
    # Save to CSV
    df = pd.DataFrame(all_matches)
    
    root = Path(__file__).resolve().parents[2]
    output_dir = root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'unprocessed_matches.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved {len(df)} matches to {output_path}")
    print(f"\nColumns saved: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(3).to_string())


if __name__ == '__main__':
    main()
