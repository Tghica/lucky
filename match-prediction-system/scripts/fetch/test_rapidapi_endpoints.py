#!/usr/bin/env python3
"""
Test script to explore what RapidAPI Tennis API can offer.
This will test various endpoints to see what data is available.
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta

# API Configuration
BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com"

def get_api_key():
    """Get API key from environment or prompt user"""
    api_key = os.getenv('RAPIDAPI_TENNIS_KEY')
    if not api_key:
        print("⚠️  RAPIDAPI_TENNIS_KEY not found in environment variables")
        print("\nPlease enter your RapidAPI key (or 'skip' to see expected endpoints):")
        api_key = input("> ").strip()
        if api_key.lower() == 'skip':
            return None
    return api_key

def make_request(endpoint, api_key, params=None):
    """Make request to RapidAPI"""
    if not api_key:
        return None
    
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com"
    }
    
    try:
        print(f"\n🔍 Testing: {endpoint}")
        if params:
            print(f"   Params: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Response size: {len(str(data))} chars")
            return data
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def analyze_response_structure(data, indent=0):
    """Recursively analyze response structure"""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        for key, value in list(data.items())[:10]:  # Limit to first 10 keys
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}: {type(value).__name__} (length: {len(value) if isinstance(value, list) else 'N/A'})")
                if isinstance(value, list) and len(value) > 0:
                    print(f"{prefix}  First item sample:")
                    analyze_response_structure(value[0], indent + 2)
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list) and len(data) > 0:
        print(f"{prefix}List with {len(data)} items. First item:")
        analyze_response_structure(data[0], indent + 1)

def main():
    print("="*70)
    print("RAPIDAPI TENNIS API - ENDPOINT EXPLORER")
    print("="*70)
    
    api_key = get_api_key()
    
    # Get recent date range for testing
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    start_date = week_ago.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    print(f"\nTesting with date range: {start_date} to {end_date}")
    
    # List of endpoints to test
    endpoints = [
        # Match/Fixture endpoints
        {
            'name': 'ATP Fixtures (upcoming matches)',
            'endpoint': f'/tennis/v2/atp/fixtures/{start_date}/{end_date}',
            'params': {'pageSize': '10'}
        },
        {
            'name': 'ATP Results (completed matches)',
            'endpoint': f'/tennis/v2/atp/results/{start_date}/{end_date}',
            'params': {'pageSize': '10'}
        },
        # Tournament endpoints
        {
            'name': 'ATP Tournaments List',
            'endpoint': '/tennis/v2/atp/tournaments',
            'params': None
        },
        # Player endpoints
        {
            'name': 'ATP Rankings',
            'endpoint': '/tennis/v2/atp/rankings',
            'params': {'limit': '10'}
        },
        {
            'name': 'Player Details (Djokovic example)',
            'endpoint': '/tennis/v2/player/52',  # Djokovic ID
            'params': None
        },
        # Head to head
        {
            'name': 'Head to Head (example)',
            'endpoint': '/tennis/v2/h2h/52/104',  # Djokovic vs Nadal
            'params': None
        },
        # Match statistics
        {
            'name': 'Tournament Results (Australian Open 2024)',
            'endpoint': '/tennis/v2/atp/tournament/results/580',  # Australian Open ID
            'params': None
        },
    ]
    
    print("\n" + "="*70)
    print("TESTING ENDPOINTS")
    print("="*70)
    
    results = {}
    
    for endpoint_info in endpoints:
        name = endpoint_info['name']
        endpoint = endpoint_info['endpoint']
        params = endpoint_info['params']
        
        print(f"\n📋 {name}")
        print("-" * 70)
        
        data = make_request(endpoint, api_key, params)
        
        if data:
            print("\n📊 Response Structure:")
            analyze_response_structure(data)
            results[name] = {'success': True, 'sample_keys': list(data.keys()) if isinstance(data, dict) else 'list'}
        else:
            results[name] = {'success': False}
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - AVAILABLE DATA FIELDS")
    print("="*70)
    
    if not api_key:
        print("\n⚠️  No API key provided. Here's what the API typically offers:")
        print("\n📊 Expected Available Data:")
        print("\n1. MATCH DATA:")
        print("   - Match ID, Date, Time")
        print("   - Player names, IDs")
        print("   - Tournament name, ID, level")
        print("   - Round information")
        print("   - Final score")
        print("   - Match winner")
        print("   - Betting odds (winner/loser)")
        
        print("\n2. DETAILED MATCH STATISTICS (per match):")
        print("   ❓ Aces, Double Faults")
        print("   ❓ 1st/2nd serve percentages")
        print("   ❓ Break points saved/faced")
        print("   ❓ Total service points")
        print("   ❓ Winners, Unforced Errors")
        print("   ❓ Match duration")
        
        print("\n3. PLAYER DATA:")
        print("   - Player ID, Name")
        print("   - Country")
        print("   - Current ranking")
        print("   - Ranking points")
        print("   - Age, Height, Weight")
        print("   - Playing hand")
        print("   - Career statistics")
        
        print("\n4. TOURNAMENT DATA:")
        print("   - Tournament ID, Name")
        print("   - Surface type")
        print("   - Tournament level (Grand Slam, Masters, etc.)")
        print("   - Location/Country")
        print("   - Draw size")
        
        print("\n5. HEAD-TO-HEAD:")
        print("   - Historical H2H records")
        print("   - H2H by surface")
        print("   - Recent matches between players")
        
        print("\n6. RANKINGS:")
        print("   - Current ATP rankings")
        print("   - Historical rankings")
        print("   - Ranking points breakdown")
        
        print("\n" + "="*70)
        print("TO TEST WITH LIVE DATA:")
        print("="*70)
        print("1. Get API key from: https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf")
        print("2. Set environment variable: export RAPIDAPI_TENNIS_KEY='your_key_here'")
        print("3. Run this script again")
    else:
        successful = sum(1 for r in results.values() if r.get('success'))
        print(f"\n✅ Successfully tested {successful}/{len(endpoints)} endpoints")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   - Use fixtures endpoint to get upcoming matches with betting odds")
        print("   - Use results endpoint to get completed matches")
        print("   - Use tournament results to get detailed match statistics")
        print("   - Use player endpoint to get physical stats (height, hand, etc.)")
        print("   - Use rankings endpoint to supplement missing ranking data")

if __name__ == "__main__":
    main()
