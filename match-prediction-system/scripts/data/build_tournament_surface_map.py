#!/usr/bin/env python3
"""
Build a tournament-to-surface mapping by analyzing past matches from well-known players.
We'll fetch past matches from top players and extract tournament ID -> surface mappings.
"""

import requests
import json
import time
from collections import defaultdict

headers = {
    "x-rapidapi-key": "97c34a0787msh8568dbfa4c6b20fp1cdcabjsn91ce02fe6134",
    "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com"
}

# Famous players with lots of match history
# Djokovic: 5992, Federer: 5136, Nadal: 5829, Murray: 6324, Alcaraz: 68074
SAMPLE_PLAYERS = [5992, 5136, 5829, 6324, 68074, 22807, 47275]

tournament_surface_map = {}
tournament_names = {}

print("="*80)
print("BUILDING TOURNAMENT -> SURFACE MAPPING")
print("="*80)

for player_id in SAMPLE_PLAYERS:
    print(f"\nFetching matches for player {player_id}...")
    
    # Get player filter data (contains court/surface info)
    filter_url = f'https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/atp/player/filter/{player_id}'
    
    try:
        response = requests.get(filter_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract court types mapping
            courts = data.get('data', {}).get('courts', [])
            court_mapping = {c['courtId']: c['court'] for c in courts}
            print(f"  Court types: {court_mapping}")
            
            # Get tournaments with their court IDs
            tournaments = data.get('data', {}).get('tournaments', [])
            print(f"  Found {len(tournaments)} tournaments")
            
            # Now get past matches to link tournaments to courts
            matches_url = f'https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/atp/player/past-matches/{player_id}'
            matches_response = requests.get(matches_url, headers=headers, params={'pageSize': '100'})
            
            if matches_response.status_code == 200:
                matches_data = matches_response.json()
                matches = matches_data.get('data', [])
                print(f"  Got {len(matches)} past matches")
                
                # For each match, we need to find the tournament and its surface
                # We'll need to cross-reference with tournament data
                for match in matches:
                    tournament_id = match.get('tournamentId')
                    
                    # Try to get match details that might have surface info
                    # For now, store tournament IDs we've seen
                    if tournament_id and tournament_id not in tournament_surface_map:
                        tournament_names[tournament_id] = f"Tournament {tournament_id}"
                
                print(f"  Tracked {len(tournament_names)} unique tournaments so far")
            
        else:
            print(f"  ❌ Error {response.status_code}")
        
        # Rate limiting - don't hammer the API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")

print(f"\n{'='*80}")
print(f"Discovered {len(tournament_names)} unique tournaments")
print(f"{'='*80}")

# Now try to get surface info by checking match stats or other endpoints
# Let's try a different approach - get tournament seasons which might have surface data
print("\nAttempting to get surface info for tournaments...")

sample_tournaments = list(tournament_names.keys())[:20]  # Test with first 20

for tid in sample_tournaments:
    try:
        # Try getting tournament results which might have embedded surface data
        results_url = f'https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/atp/tournament/results/{tid}'
        response = requests.get(results_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            # Check if there's surface info in the response
            print(f"Tournament {tid}: Checking response structure...")
            print(f"  Keys: {list(data.keys())}")
            
            # Check the first match for any surface indicators
            if data.get('data', {}).get('singles'):
                first_match = data['data']['singles'][0]
                print(f"  Match keys: {list(first_match.keys())}")
                
        time.sleep(0.3)
        
    except Exception as e:
        print(f"  Error for {tid}: {str(e)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nNOTE: The API doesn't seem to expose surface data directly in match responses.")
print("We'll need to:")
print("1. Manually create a tournament -> surface mapping from known tournaments")
print("2. Or use external data sources like Tennis Abstract")
print("3. Or check if there's a getMatchStats endpoint that has surface info")
