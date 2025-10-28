"""
RapidAPI Tennis API client for fetching match data.
API: https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf
"""

import os
import json
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RapidAPITennisClient:
    """Client for interacting with RapidAPI Tennis API."""
    
    BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('RAPIDAPI_TENNIS_KEY')
        if not self.api_key:
            raise ValueError("RapidAPI Tennis API key not found.")
        
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com"
        }
        
        self.tournament_surface_map = self._load_surface_mapping()
        self.tournament_names_cache = {}
    
    def _load_surface_mapping(self):
        try:
            mapping_file = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'tournament_surface_mapping.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Surface mapping file not found")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load surface mapping: {e}")
            return {}
    
    def _make_request(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def _get_tournament_name(self, tournament_id, tour="atp"):
        if tournament_id in self.tournament_names_cache:
            return self.tournament_names_cache[tournament_id]
        
        try:
            endpoint = f"/tennis/v2/{tour.lower()}/tournament/seasons/{tournament_id}"
            data = self._make_request(endpoint)
            seasons = data.get('data', [])
            if seasons:
                name = seasons[0].get('name', f'Tournament {tournament_id}')
                self.tournament_names_cache[tournament_id] = name
                return name
        except Exception as e:
            logger.warning(f"Failed to get tournament name for {tournament_id}: {e}")
        
        return f'Tournament {tournament_id}'
    
    def fetch_matches_by_date_range(self, start_date, end_date, tour="atp"):
        logger.info(f"Fetching {tour.upper()} matches from {start_date} to {end_date}")
        
        endpoint = f"/tennis/v2/{tour.lower()}/fixtures/{start_date}/{end_date}"
        
        try:
            fixtures_data = self._make_request(endpoint, params={"pageSize": "100"})
            fixtures = fixtures_data.get('data', [])
            
            logger.info(f"Found {len(fixtures)} fixtures")
            
            tournament_ids = set()
            for fixture in fixtures:
                tid = fixture.get('tournamentId')
                if tid:
                    tournament_ids.add(tid)
            
            logger.info(f"Found {len(tournament_ids)} tournaments")
            
            all_matches = []
            
            for tournament_id in tournament_ids:
                try:
                    results_endpoint = f"/tennis/v2/{tour.lower()}/tournament/results/{tournament_id}"
                    results_data = self._make_request(results_endpoint)
                    
                    singles_matches = results_data.get('data', {}).get('singles', [])
                    
                    for match in singles_matches:
                        match_date = match.get('date', '')[:10]
                        if start_date <= match_date <= end_date:
                            all_matches.append(match)
                except Exception as e:
                    logger.warning(f"Failed to fetch results for tournament {tournament_id}: {e}")
                    continue
            
            logger.info(f"Total completed matches in date range: {len(all_matches)}")
            return self._parse_matches(all_matches, tour)
            
        except Exception as e:
            logger.error(f"Failed to fetch matches: {e}")
            raise
    
    def _parse_matches(self, api_matches, tour="atp"):
        parsed_matches = []
        
        for match_data in api_matches:
            try:
                date_str = match_data.get('date', '')
                match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d') if date_str else None
                
                player1 = match_data.get('player1', {})
                player2 = match_data.get('player2', {})
                
                player1_name = player1.get('name', '')
                player2_name = player2.get('name', '')
                
                winner_id = match_data.get('match_winner')
                player1_id = match_data.get('player1Id')
                
                if winner_id == player1_id:
                    winner = player1_name
                else:
                    winner = player2_name
                
                tournament_id = match_data.get('tournamentId')
                tournament_name = self._get_tournament_name(tournament_id, tour)
                
                surface = self.tournament_surface_map.get(tournament_name, 'Hard')
                
                match_record = {
                    'date': match_date,
                    'player1': player1_name,
                    'player2': player2_name,
                    'winner': winner,
                    'stadium': '',
                    'surface': surface,
                    'description': tournament_name,
                    'nation': player1.get('countryAcr', ''),
                }
                
                parsed_matches.append(match_record)
                
            except Exception as e:
                logger.warning(f"Failed to parse match: {e}")
                continue
        
        return parsed_matches


def fetch_tennis_matches(start_date, end_date, tours=None):
    if tours is None:
        tours = ['atp']
    
    client = RapidAPITennisClient()
    all_matches = []
    
    for tour in tours:
        try:
            matches = client.fetch_matches_by_date_range(start_date, end_date, tour)
            all_matches.extend(matches)
            logger.info(f"Fetched {len(matches)} {tour.upper()} matches")
        except Exception as e:
            logger.error(f"Failed to fetch {tour.upper()} matches: {e}")
    
    return all_matches
