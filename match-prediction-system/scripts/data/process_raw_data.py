#!/usr/bin/env python3
"""
Data Processing Pipeline - Step 1: Merge and Clean Raw Data

This script processes raw ATP data from multiple sources and creates 3 clean files:
1. stadiums.csv - Tournament venues and locations
2. players.csv - Player information and characteristics
3. matches.csv - Match results with all details

Data Sources:
- Jeff Sackmann tennis_atp (2000-2024) - Primary source
- ATP Tennis Daily Update (2025) - Current year supplement
"""

import pandas as pd
import os
from datetime import datetime
import json


def load_jeff_sackmann_matches(start_year=2000, end_year=2024):
    """Load all Jeff Sackmann match files from start_year to end_year"""
    print(f"\n{'='*70}")
    print(f"LOADING JEFF SACKMANN DATA ({start_year}-{end_year})")
    print(f"{'='*70}")
    
    tennis_dir = 'data/raw/Huge Tennis Database/tennis_atp'
    all_matches = []
    
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(tennis_dir, f'atp_matches_{year}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_matches.append(df)
            print(f"  ✓ {year}: {len(df):,} matches")
        else:
            print(f"  ✗ {year}: File not found")
    
    combined = pd.concat(all_matches, ignore_index=True)
    print(f"\n  Total matches: {len(combined):,}")
    return combined


def load_daily_update_2025():
    """Load 2025 matches from ATP Daily Update"""
    print(f"\n{'='*70}")
    print(f"LOADING 2025 DAILY UPDATE DATA")
    print(f"{'='*70}")
    
    df = pd.read_csv('data/raw/ATP Tennis 2000 - 2025 Daily update/atp_tennis.csv')
    
    # Filter for 2025 only
    df['Date'] = pd.to_datetime(df['Date'])
    df_2025 = df[df['Date'].dt.year == 2025].copy()
    
    print(f"  Total 2025 matches: {len(df_2025):,}")
    print(f"  Date range: {df_2025['Date'].min()} to {df_2025['Date'].max()}")
    
    return df_2025


def standardize_jeff_sackmann(df):
    """Standardize Jeff Sackmann data to common format"""
    print(f"\n{'='*70}")
    print("STANDARDIZING JEFF SACKMANN DATA")
    print(f"{'='*70}")
    
    # Convert date format from YYYYMMDD to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    
    # Standardize column names
    standardized = pd.DataFrame({
        # Match identification
        'date': df['date'],
        'tournament_id': df['tourney_id'],
        'tournament': df['tourney_name'],
        'surface': df['surface'],
        'round': df['round'],
        'match_num': df.get('match_num', None),
        
        # Tournament details
        'tournament_level': df['tourney_level'],
        'draw_size': df.get('draw_size', None),
        
        # Players
        'winner_id': df['winner_id'],
        'winner_name': df['winner_name'],
        'winner_hand': df.get('winner_hand', None),
        'winner_height': df.get('winner_ht', None),
        'winner_country': df.get('winner_ioc', None),
        'winner_age': df.get('winner_age', None),
        'winner_rank': df.get('winner_rank', None),
        'winner_rank_points': df.get('winner_rank_points', None),
        'winner_seed': df.get('winner_seed', None),
        
        'loser_id': df['loser_id'],
        'loser_name': df['loser_name'],
        'loser_hand': df.get('loser_hand', None),
        'loser_height': df.get('loser_ht', None),
        'loser_country': df.get('loser_ioc', None),
        'loser_age': df.get('loser_age', None),
        'loser_rank': df.get('loser_rank', None),
        'loser_rank_points': df.get('loser_rank_points', None),
        'loser_seed': df.get('loser_seed', None),
        
        # Match details
        'score': df.get('score', None),
        'best_of': df.get('best_of', None),
        'minutes': df.get('minutes', None),
        
        # Winner stats
        'w_ace': df.get('w_ace', None),
        'w_df': df.get('w_df', None),
        'w_svpt': df.get('w_svpt', None),
        'w_1stIn': df.get('w_1stIn', None),
        'w_1stWon': df.get('w_1stWon', None),
        'w_2ndWon': df.get('w_2ndWon', None),
        'w_SvGms': df.get('w_SvGms', None),
        'w_bpSaved': df.get('w_bpSaved', None),
        'w_bpFaced': df.get('w_bpFaced', None),
        
        # Loser stats
        'l_ace': df.get('l_ace', None),
        'l_df': df.get('l_df', None),
        'l_svpt': df.get('l_svpt', None),
        'l_1stIn': df.get('l_1stIn', None),
        'l_1stWon': df.get('l_1stWon', None),
        'l_2ndWon': df.get('l_2ndWon', None),
        'l_SvGms': df.get('l_SvGms', None),
        'l_bpSaved': df.get('l_bpSaved', None),
        'l_bpFaced': df.get('l_bpFaced', None),
        
        # Betting odds (not available in this source)
        'winner_odds': None,
        'loser_odds': None,
        
        # Source
        'data_source': 'jeff_sackmann'
    })
    
    print(f"  ✓ Standardized {len(standardized):,} matches")
    return standardized


def standardize_daily_update(df):
    """Standardize Daily Update data to common format"""
    print(f"\n{'='*70}")
    print("STANDARDIZING DAILY UPDATE DATA (2025)")
    print(f"{'='*70}")
    
    # Determine winner/loser from Winner column
    df['is_player1_winner'] = df['Winner'] == df['Player_1']
    
    standardized = pd.DataFrame({
        # Match identification
        'date': pd.to_datetime(df['Date']),
        'tournament_id': None,  # Not available
        'tournament': df['Tournament'],
        'surface': df['Surface'],
        'round': df['Round'],
        'match_num': None,
        
        # Tournament details
        'tournament_level': df['Series'],  # Different encoding
        'draw_size': None,
        
        # Winner (conditional on who won)
        'winner_id': None,
        'winner_name': df.apply(lambda x: x['Player_1'] if x['is_player1_winner'] else x['Player_2'], axis=1),
        'winner_hand': None,
        'winner_height': None,
        'winner_country': None,
        'winner_age': None,
        'winner_rank': df.apply(lambda x: x['Rank_1'] if x['is_player1_winner'] else x['Rank_2'], axis=1),
        'winner_rank_points': df.apply(lambda x: x['Pts_1'] if x['is_player1_winner'] else x['Pts_2'], axis=1),
        'winner_seed': None,
        
        # Loser (conditional on who lost)
        'loser_id': None,
        'loser_name': df.apply(lambda x: x['Player_2'] if x['is_player1_winner'] else x['Player_1'], axis=1),
        'loser_hand': None,
        'loser_height': None,
        'loser_country': None,
        'loser_age': None,
        'loser_rank': df.apply(lambda x: x['Rank_2'] if x['is_player1_winner'] else x['Rank_1'], axis=1),
        'loser_rank_points': df.apply(lambda x: x['Pts_2'] if x['is_player1_winner'] else x['Pts_1'], axis=1),
        'loser_seed': None,
        
        # Match details
        'score': df['Score'],
        'best_of': df['Best of'],
        'minutes': None,
        
        # Stats (not available in daily update)
        'w_ace': None,
        'w_df': None,
        'w_svpt': None,
        'w_1stIn': None,
        'w_1stWon': None,
        'w_2ndWon': None,
        'w_SvGms': None,
        'w_bpSaved': None,
        'w_bpFaced': None,
        
        'l_ace': None,
        'l_df': None,
        'l_svpt': None,
        'l_1stIn': None,
        'l_1stWon': None,
        'l_2ndWon': None,
        'l_SvGms': None,
        'l_bpSaved': None,
        'l_bpFaced': None,
        
        # Betting odds (AVAILABLE!)
        'winner_odds': df.apply(lambda x: x['Odd_1'] if x['is_player1_winner'] else x['Odd_2'], axis=1),
        'loser_odds': df.apply(lambda x: x['Odd_2'] if x['is_player1_winner'] else x['Odd_1'], axis=1),
        
        # Source
        'data_source': 'daily_update_2025'
    })
    
    print(f"  ✓ Standardized {len(standardized):,} matches")
    return standardized


def remove_duplicates(df):
    """Remove duplicate matches based on date, players, tournament"""
    print(f"\n{'='*70}")
    print("REMOVING DUPLICATES")
    print(f"{'='*70}")
    
    before = len(df)
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['date', 'winner_name', 'loser_name', 'tournament'], keep='first')
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        print(f"  Found {duplicate_count} duplicates")
        df_clean = df[~duplicates].copy()
        print(f"  ✓ Removed {duplicate_count} duplicates")
    else:
        print(f"  ✓ No duplicates found")
        df_clean = df.copy()
    
    print(f"  Before: {before:,} matches")
    print(f"  After: {len(df_clean):,} matches")
    
    return df_clean


def create_stadiums_file(df):
    """Create stadiums.csv with unique tournament venues"""
    print(f"\n{'='*70}")
    print("CREATING STADIUMS FILE")
    print(f"{'='*70}")
    
    # Group by tournament to get unique venues
    stadiums = df.groupby(['tournament', 'surface']).agg({
        'tournament_id': 'first',
        'tournament_level': 'first',
        'date': ['min', 'max', 'count']
    }).reset_index()
    
    stadiums.columns = ['tournament', 'surface', 'tournament_id', 'tournament_level', 
                        'first_date', 'last_date', 'total_matches']
    
    # Add location info (would need geocoding for full implementation)
    stadiums['city'] = None  # TODO: Add city mapping
    stadiums['country'] = None  # TODO: Add country mapping
    stadiums['indoor'] = None  # TODO: Determine from tournament name
    
    # Sort by tournament name
    stadiums = stadiums.sort_values('tournament').reset_index(drop=True)
    
    print(f"  ✓ Created {len(stadiums)} unique tournament/surface combinations")
    return stadiums


def create_players_file(df):
    """Create players.csv with unique player information"""
    print(f"\n{'='*70}")
    print("CREATING PLAYERS FILE")
    print(f"{'='*70}")
    
    # Get winner info
    winners = df[['winner_id', 'winner_name', 'winner_hand', 'winner_height', 'winner_country']].copy()
    winners.columns = ['player_id', 'player_name', 'hand', 'height', 'country']
    
    # Get loser info
    losers = df[['loser_id', 'loser_name', 'loser_hand', 'loser_height', 'loser_country']].copy()
    losers.columns = ['player_id', 'player_name', 'hand', 'height', 'country']
    
    # Combine and get unique players
    all_players = pd.concat([winners, losers], ignore_index=True)
    
    # Group by player_name (since some might not have IDs from 2025 data)
    players = all_players.groupby('player_name').agg({
        'player_id': 'first',
        'hand': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'height': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'country': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    }).reset_index()
    
    # Add career stats
    players['total_matches'] = players['player_name'].apply(
        lambda x: len(df[(df['winner_name'] == x) | (df['loser_name'] == x)])
    )
    
    players['total_wins'] = players['player_name'].apply(
        lambda x: len(df[df['winner_name'] == x])
    )
    
    players['win_percentage'] = (players['total_wins'] / players['total_matches'] * 100).round(2)
    
    # Sort by total matches
    players = players.sort_values('total_matches', ascending=False).reset_index(drop=True)
    
    print(f"  ✓ Created {len(players)} unique players")
    print(f"  Top 5 players by matches:")
    for idx, row in players.head(5).iterrows():
        print(f"    {row['player_name']}: {row['total_matches']} matches, {row['win_percentage']:.1f}% win rate")
    
    return players


def save_processed_data(matches_df, stadiums_df, players_df):
    """Save processed data to CSV files"""
    print(f"\n{'='*70}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*70}")
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matches
    matches_file = os.path.join(output_dir, 'matches.csv')
    matches_df.to_csv(matches_file, index=False)
    print(f"  ✓ Saved {len(matches_df):,} matches to {matches_file}")
    
    # Save stadiums
    stadiums_file = os.path.join(output_dir, 'stadiums.csv')
    stadiums_df.to_csv(stadiums_file, index=False)
    print(f"  ✓ Saved {len(stadiums_df):,} stadiums to {stadiums_file}")
    
    # Save players
    players_file = os.path.join(output_dir, 'players.csv')
    players_df.to_csv(players_file, index=False)
    print(f"  ✓ Saved {len(players_df):,} players to {players_file}")
    
    # Save metadata
    metadata = {
        'processed_date': datetime.now().isoformat(),
        'total_matches': len(matches_df),
        'total_players': len(players_df),
        'total_stadiums': len(stadiums_df),
        'date_range': {
            'start': str(matches_df['date'].min()),
            'end': str(matches_df['date'].max())
        },
        'data_sources': matches_df['data_source'].value_counts().to_dict()
    }
    
    metadata_file = os.path.join(output_dir, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_file}")


def main():
    """Main processing pipeline"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  DATA PROCESSING PIPELINE - STEP 1: MERGE & CLEAN".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Load data
    jeff_sackmann = load_jeff_sackmann_matches(2000, 2024)
    daily_2025 = load_daily_update_2025()
    
    # Standardize formats
    js_standardized = standardize_jeff_sackmann(jeff_sackmann)
    du_standardized = standardize_daily_update(daily_2025)
    
    # Combine all matches
    print(f"\n{'='*70}")
    print("COMBINING ALL MATCHES")
    print(f"{'='*70}")
    all_matches = pd.concat([js_standardized, du_standardized], ignore_index=True)
    print(f"  ✓ Combined total: {len(all_matches):,} matches")
    
    # Remove duplicates
    all_matches_clean = remove_duplicates(all_matches)
    
    # Sort by date
    all_matches_clean = all_matches_clean.sort_values('date').reset_index(drop=True)
    
    # Create derivative files
    stadiums = create_stadiums_file(all_matches_clean)
    players = create_players_file(all_matches_clean)
    
    # Save everything
    save_processed_data(all_matches_clean, stadiums, players)
    
    # Final summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"  ✅ Matches: {len(all_matches_clean):,}")
    print(f"  ✅ Players: {len(players):,}")
    print(f"  ✅ Stadiums: {len(stadiums):,}")
    print(f"  ✅ Date range: {all_matches_clean['date'].min()} to {all_matches_clean['date'].max()}")
    print(f"\n  Files created in data/processed/:")
    print(f"    - matches.csv")
    print(f"    - players.csv")
    print(f"    - stadiums.csv")
    print(f"    - processing_metadata.json")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
