#!/usr/bin/env python3
"""
Add country/location information to stadiums.csv

Uses tournament name patterns and known tournament locations
to add country and city information.
"""

import pandas as pd
import re


def create_tournament_location_map():
    """
    Comprehensive mapping of tournaments to locations.
    Based on historical ATP tournament locations.
    """
    return {
        # Grand Slams
        'Australian Open': {'city': 'Melbourne', 'country': 'Australia', 'indoor': False},
        'Roland Garros': {'city': 'Paris', 'country': 'France', 'indoor': False},
        'French Open': {'city': 'Paris', 'country': 'France', 'indoor': False},
        'Wimbledon': {'city': 'London', 'country': 'United Kingdom', 'indoor': False},
        'US Open': {'city': 'New York', 'country': 'USA', 'indoor': False},
        
        # Masters 1000
        'Indian Wells Masters': {'city': 'Indian Wells', 'country': 'USA', 'indoor': False},
        'Miami Masters': {'city': 'Miami', 'country': 'USA', 'indoor': False},
        'Monte Carlo Masters': {'city': 'Monte Carlo', 'country': 'Monaco', 'indoor': False},
        'Monte Carlo': {'city': 'Monte Carlo', 'country': 'Monaco', 'indoor': False},
        'Rome Masters': {'city': 'Rome', 'country': 'Italy', 'indoor': False},
        'Rome': {'city': 'Rome', 'country': 'Italy', 'indoor': False},
        'Madrid Masters': {'city': 'Madrid', 'country': 'Spain', 'indoor': False},
        'Madrid': {'city': 'Madrid', 'country': 'Spain', 'indoor': False},
        'Canada Masters': {'city': 'Montreal/Toronto', 'country': 'Canada', 'indoor': False},
        'Montreal': {'city': 'Montreal', 'country': 'Canada', 'indoor': False},
        'Toronto': {'city': 'Toronto', 'country': 'Canada', 'indoor': False},
        'Cincinnati Masters': {'city': 'Cincinnati', 'country': 'USA', 'indoor': False},
        'Cincinnati': {'city': 'Cincinnati', 'country': 'USA', 'indoor': False},
        'Shanghai Masters': {'city': 'Shanghai', 'country': 'China', 'indoor': False},
        'Shanghai': {'city': 'Shanghai', 'country': 'China', 'indoor': False},
        'Paris Masters': {'city': 'Paris', 'country': 'France', 'indoor': True},
        'Paris': {'city': 'Paris', 'country': 'France', 'indoor': True},
        
        # ATP 500
        'Dubai': {'city': 'Dubai', 'country': 'UAE', 'indoor': False},
        'Dubai Duty Free Tennis Championships': {'city': 'Dubai', 'country': 'UAE', 'indoor': False},
        'Rotterdam': {'city': 'Rotterdam', 'country': 'Netherlands', 'indoor': True},
        'ABN AMRO World Tennis Tournament': {'city': 'Rotterdam', 'country': 'Netherlands', 'indoor': True},
        'Acapulco': {'city': 'Acapulco', 'country': 'Mexico', 'indoor': False},
        'Abierto Mexicano': {'city': 'Acapulco', 'country': 'Mexico', 'indoor': False},
        'Barcelona': {'city': 'Barcelona', 'country': 'Spain', 'indoor': False},
        'Barcelona Open': {'city': 'Barcelona', 'country': 'Spain', 'indoor': False},
        'Hamburg': {'city': 'Hamburg', 'country': 'Germany', 'indoor': False},
        'Hamburg European Open': {'city': 'Hamburg', 'country': 'Germany', 'indoor': False},
        'Washington': {'city': 'Washington DC', 'country': 'USA', 'indoor': False},
        'Beijing': {'city': 'Beijing', 'country': 'China', 'indoor': False},
        'Tokyo': {'city': 'Tokyo', 'country': 'Japan', 'indoor': False},
        'Tokyo Olympics': {'city': 'Tokyo', 'country': 'Japan', 'indoor': False},
        'Vienna': {'city': 'Vienna', 'country': 'Austria', 'indoor': True},
        'Basel': {'city': 'Basel', 'country': 'Switzerland', 'indoor': True},
        
        # ATP 250 & Other Major Cities
        'Adelaide': {'city': 'Adelaide', 'country': 'Australia', 'indoor': False},
        'Auckland': {'city': 'Auckland', 'country': 'New Zealand', 'indoor': False},
        'ASB Classic': {'city': 'Auckland', 'country': 'New Zealand', 'indoor': False},
        'Sydney': {'city': 'Sydney', 'country': 'Australia', 'indoor': False},
        'Brisbane': {'city': 'Brisbane', 'country': 'Australia', 'indoor': False},
        'Doha': {'city': 'Doha', 'country': 'Qatar', 'indoor': False},
        'Montpellier': {'city': 'Montpellier', 'country': 'France', 'indoor': True},
        'Sofia': {'city': 'Sofia', 'country': 'Bulgaria', 'indoor': True},
        'Buenos Aires': {'city': 'Buenos Aires', 'country': 'Argentina', 'indoor': False},
        'Argentina Open': {'city': 'Buenos Aires', 'country': 'Argentina', 'indoor': False},
        'Rio de Janeiro': {'city': 'Rio de Janeiro', 'country': 'Brazil', 'indoor': False},
        'ATP Rio de Janeiro': {'city': 'Rio de Janeiro', 'country': 'Brazil', 'indoor': False},
        'Sao Paulo': {'city': 'Sao Paulo', 'country': 'Brazil', 'indoor': False},
        'Marseille': {'city': 'Marseille', 'country': 'France', 'indoor': True},
        'Delray Beach': {'city': 'Delray Beach', 'country': 'USA', 'indoor': False},
        'Memphis': {'city': 'Memphis', 'country': 'USA', 'indoor': True},
        'Santiago': {'city': 'Santiago', 'country': 'Chile', 'indoor': False},
        'Estoril': {'city': 'Estoril', 'country': 'Portugal', 'indoor': False},
        'Munich': {'city': 'Munich', 'country': 'Germany', 'indoor': False},
        'Geneva': {'city': 'Geneva', 'country': 'Switzerland', 'indoor': False},
        'Lyon': {'city': 'Lyon', 'country': 'France', 'indoor': False},
        'Stuttgart': {'city': 'Stuttgart', 'country': 'Germany', 'indoor': False},
        'Halle': {'city': 'Halle', 'country': 'Germany', 'indoor': False},
        'London': {'city': 'London', 'country': 'United Kingdom', 'indoor': True},
        'Eastbourne': {'city': 'Eastbourne', 'country': 'United Kingdom', 'indoor': False},
        'Newport': {'city': 'Newport', 'country': 'USA', 'indoor': False},
        'Bastad': {'city': 'Bastad', 'country': 'Sweden', 'indoor': False},
        'Umag': {'city': 'Umag', 'country': 'Croatia', 'indoor': False},
        'Gstaad': {'city': 'Gstaad', 'country': 'Switzerland', 'indoor': False},
        'Atlanta': {'city': 'Atlanta', 'country': 'USA', 'indoor': False},
        'Los Cabos': {'city': 'Los Cabos', 'country': 'Mexico', 'indoor': False},
        'Kitzbuhel': {'city': 'Kitzbuhel', 'country': 'Austria', 'indoor': False},
        'Winston-Salem': {'city': 'Winston-Salem', 'country': 'USA', 'indoor': False},
        'Metz': {'city': 'Metz', 'country': 'France', 'indoor': True},
        'St. Petersburg': {'city': 'St. Petersburg', 'country': 'Russia', 'indoor': True},
        'Chengdu': {'city': 'Chengdu', 'country': 'China', 'indoor': False},
        'Shenzhen': {'city': 'Shenzhen', 'country': 'China', 'indoor': False},
        'Zhuhai': {'city': 'Zhuhai', 'country': 'China', 'indoor': False},
        'Stockholm': {'city': 'Stockholm', 'country': 'Sweden', 'indoor': True},
        'Moscow': {'city': 'Moscow', 'country': 'Russia', 'indoor': True},
        'Antwerp': {'city': 'Antwerp', 'country': 'Belgium', 'indoor': True},
        'Athens': {'city': 'Athens', 'country': 'Greece', 'indoor': False},
        'Athens Olympics': {'city': 'Athens', 'country': 'Greece', 'indoor': False},
        'Beijing Olympics': {'city': 'Beijing', 'country': 'China', 'indoor': False},
        'London Olympics': {'city': 'London', 'country': 'United Kingdom', 'indoor': False},
        'Rio Olympics': {'city': 'Rio de Janeiro', 'country': 'Brazil', 'indoor': False},
        
        # Additional tournaments
        'Antalya': {'city': 'Antalya', 'country': 'Turkey', 'indoor': False},
        'Amersfoort': {'city': 'Amersfoort', 'country': 'Netherlands', 'indoor': False},
        'Amsterdam': {'city': 'Amsterdam', 'country': 'Netherlands', 'indoor': False},
        'Almaty': {'city': 'Almaty', 'country': 'Kazakhstan', 'indoor': False},
        'Almaty Open': {'city': 'Almaty', 'country': 'Kazakhstan', 'indoor': False},
        'Astana': {'city': 'Astana', 'country': 'Kazakhstan', 'indoor': True},
        'Bucharest': {'city': 'Bucharest', 'country': 'Romania', 'indoor': False},
        'Casablanca': {'city': 'Casablanca', 'country': 'Morocco', 'indoor': False},
        'Chennai': {'city': 'Chennai', 'country': 'India', 'indoor': False},
        'Cologne': {'city': 'Cologne', 'country': 'Germany', 'indoor': True},
        'Copenhagen': {'city': 'Copenhagen', 'country': 'Denmark', 'indoor': True},
        'Dallas': {'city': 'Dallas', 'country': 'USA', 'indoor': True},
        'Florence': {'city': 'Florence', 'country': 'Italy', 'indoor': False},
        'Guangzhou': {'city': 'Guangzhou', 'country': 'China', 'indoor': False},
        'Houston': {'city': 'Houston', 'country': 'USA', 'indoor': False},
        'Indianapolis': {'city': 'Indianapolis', 'country': 'USA', 'indoor': False},
        'Las Vegas': {'city': 'Las Vegas', 'country': 'USA', 'indoor': False},
        'Long Island': {'city': 'Long Island', 'country': 'USA', 'indoor': False},
        'Mallorca': {'city': 'Mallorca', 'country': 'Spain', 'indoor': False},
        'Marrakech': {'city': 'Marrakech', 'country': 'Morocco', 'indoor': False},
        'Milan': {'city': 'Milan', 'country': 'Italy', 'indoor': True},
        'New Haven': {'city': 'New Haven', 'country': 'USA', 'indoor': False},
        'Nice': {'city': 'Nice', 'country': 'France', 'indoor': False},
        'Nottingham': {'city': 'Nottingham', 'country': 'United Kingdom', 'indoor': False},
        'Oeiras': {'city': 'Oeiras', 'country': 'Portugal', 'indoor': False},
        'Portoroz': {'city': 'Portoroz', 'country': 'Slovenia', 'indoor': False},
        'Pune': {'city': 'Pune', 'country': 'India', 'indoor': False},
        'Quito': {'city': 'Quito', 'country': 'Ecuador', 'indoor': False},
        'San Diego': {'city': 'San Diego', 'country': 'USA', 'indoor': False},
        'San Jose': {'city': 'San Jose', 'country': 'USA', 'indoor': True},
        'Seoul': {'city': 'Seoul', 'country': 'South Korea', 'indoor': False},
        'Singapore': {'city': 'Singapore', 'country': 'Singapore', 'indoor': False},
        'Sopot': {'city': 'Sopot', 'country': 'Poland', 'indoor': False},
        'Tel Aviv': {'city': 'Tel Aviv', 'country': 'Israel', 'indoor': False},
        'Tashkent': {'city': 'Tashkent', 'country': 'Uzbekistan', 'indoor': False},
        'Valencia': {'city': 'Valencia', 'country': 'Spain', 'indoor': True},
        'Vina del Mar': {'city': 'Vina del Mar', 'country': 'Chile', 'indoor': False},
        'Warsaw': {'city': 'Warsaw', 'country': 'Poland', 'indoor': False},
        'Zagreb': {'city': 'Zagreb', 'country': 'Croatia', 'indoor': True},
        
        # Additional mappings from unmatched
        'Bangkok': {'city': 'Bangkok', 'country': 'Thailand', 'indoor': False},
        'Belgrade': {'city': 'Belgrade', 'country': 'Serbia', 'indoor': False},
        'Bogota': {'city': 'Bogota', 'country': 'Colombia', 'indoor': False},
        'Brighton': {'city': 'Brighton', 'country': 'United Kingdom', 'indoor': False},
        'Budapest': {'city': 'Budapest', 'country': 'Hungary', 'indoor': False},
        'Cagliari': {'city': 'Cagliari', 'country': 'Italy', 'indoor': False},
        'Cordoba': {'city': 'Cordoba', 'country': 'Argentina', 'indoor': False},
        'Costa Do Sauipe': {'city': 'Costa Do Sauipe', 'country': 'Brazil', 'indoor': False},
        'Banja Luka': {'city': 'Banja Luka', 'country': 'Bosnia and Herzegovina', 'indoor': False},
        
        # Branded tournament names
        'BNP Paribas Open': {'city': 'Indian Wells', 'country': 'USA', 'indoor': False},
        'BMW Open': {'city': 'Munich', 'country': 'Germany', 'indoor': False},
        'China Open': {'city': 'Beijing', 'country': 'China', 'indoor': False},
        'Citi Open': {'city': 'Washington DC', 'country': 'USA', 'indoor': False},
        'Canadian Open': {'city': 'Montreal/Toronto', 'country': 'Canada', 'indoor': False},
        'Croatia Open': {'city': 'Umag', 'country': 'Croatia', 'indoor': False},
        'Chile Open': {'city': 'Santiago', 'country': 'Chile', 'indoor': False},
        
        # Cup/Team events (keep these at the end as catch-all)
        'Atp Cup': {'city': 'Various', 'country': 'Various', 'indoor': False},
        'Davis Cup': {'city': 'Various', 'country': 'Various', 'indoor': False},
        'Laver Cup': {'city': 'Various', 'country': 'Various', 'indoor': True},
        
        # Additional unmatched tournaments
        'Dusseldorf': {'city': 'Dusseldorf', 'country': 'Germany', 'indoor': False},
        'European Open': {'city': 'Antwerp', 'country': 'Belgium', 'indoor': True},
        'Generali Open': {'city': 'Kitzbuhel', 'country': 'Austria', 'indoor': False},
        'Gijon': {'city': 'Gijon', 'country': 'Spain', 'indoor': False},
        'Grand Prix Hassan II': {'city': 'Marrakech', 'country': 'Morocco', 'indoor': False},
        'Great Ocean Road Open': {'city': 'Melbourne', 'country': 'Australia', 'indoor': False},
        'Hangzhou Open': {'city': 'Hangzhou', 'country': 'China', 'indoor': False},
        'Ho Chi Minh City': {'city': 'Ho Chi Minh City', 'country': 'Vietnam', 'indoor': False},
        'Hong Kong': {'city': 'Hong Kong', 'country': 'Hong Kong', 'indoor': False},
        'Hong Kong Tennis Open': {'city': 'Hong Kong', 'country': 'Hong Kong', 'indoor': False},
        'Internazionali BNL d\'Italia': {'city': 'Rome', 'country': 'Italy', 'indoor': False},
        'Istanbul': {'city': 'Istanbul', 'country': 'Turkey', 'indoor': False},
        'Japan Open Tennis Championships': {'city': 'Tokyo', 'country': 'Japan', 'indoor': False},
        'Johannesburg': {'city': 'Johannesburg', 'country': 'South Africa', 'indoor': False},
        'Kuala Lumpur': {'city': 'Kuala Lumpur', 'country': 'Malaysia', 'indoor': False},
        'Los Angeles': {'city': 'Los Angeles', 'country': 'USA', 'indoor': False},
        'Marbella': {'city': 'Marbella', 'country': 'Spain', 'indoor': False},
        'Masters Cup': {'city': 'Various', 'country': 'Various', 'indoor': True},
        'Mexico City': {'city': 'Mexico City', 'country': 'Mexico', 'indoor': False},
        'Mumbai': {'city': 'Mumbai', 'country': 'India', 'indoor': False},
    }


def fuzzy_match_tournament(tournament_name, location_map):
    """Try to match tournament name to known locations"""
    
    # Special case: All Davis Cup variations should be treated as one tournament
    if 'davis cup' in tournament_name.lower():
        return location_map.get('Davis Cup')
    
    # Exact match first
    if tournament_name in location_map:
        return location_map[tournament_name]
    
    # Try partial matching (city name in tournament name)
    tournament_lower = tournament_name.lower()
    
    for known_tournament, location in location_map.items():
        known_lower = known_tournament.lower()
        
        # Check if known tournament is in the name
        if known_lower in tournament_lower:
            return location
        
        # Check if city name is in tournament name
        if location['city'].lower() in tournament_lower:
            return location
    
    return None


def add_location_info():
    """Add country and city information to stadiums.csv"""
    
    print("="*70)
    print("ADDING LOCATION INFORMATION TO STADIUMS")
    print("="*70)
    
    # Load stadiums
    stadiums = pd.read_csv('data/processed/stadiums.csv')
    print(f"\n  Loaded {len(stadiums)} tournament/surface combinations")
    
    # Get location map
    location_map = create_tournament_location_map()
    print(f"  Location map contains {len(location_map)} tournaments")
    
    # Apply location matching
    matched = 0
    unmatched = []
    
    for idx, row in stadiums.iterrows():
        tournament = row['tournament']
        location = fuzzy_match_tournament(tournament, location_map)
        
        if location:
            stadiums.at[idx, 'city'] = location['city']
            stadiums.at[idx, 'country'] = location['country']
            stadiums.at[idx, 'indoor'] = location['indoor']
            matched += 1
        else:
            unmatched.append(tournament)
    
    print(f"\n  ✓ Matched {matched}/{len(stadiums)} tournaments ({matched/len(stadiums)*100:.1f}%)")
    
    if unmatched:
        print(f"\n  ⚠️  {len(unmatched)} unmatched tournaments (first 20):")
        for t in sorted(set(unmatched))[:20]:
            print(f"     - {t}")
    
    # Save updated file
    stadiums.to_csv('data/processed/stadiums.csv', index=False)
    print(f"\n  ✓ Updated stadiums.csv saved")
    
    # Show statistics
    print(f"\n  Country distribution (top 10):")
    country_counts = stadiums['country'].value_counts().head(10)
    for country, count in country_counts.items():
        print(f"     {country:25s}: {count:3d} tournaments")
    
    print(f"\n  Indoor vs Outdoor:")
    indoor_counts = stadiums['indoor'].value_counts()
    for status, count in indoor_counts.items():
        status_str = "Indoor" if status == True else ("Outdoor" if status == False else "Unknown")
        print(f"     {status_str:10s}: {count:3d} tournaments")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    add_location_info()
