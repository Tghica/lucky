#!/usr/bin/env python3
"""
Calculate dynamic ELO ratings for all players.

This script:
1. Processes matches chronologically from first to last
2. Calculates ELO before each match (for features)
3. Updates ELO after each match (based on result)
4. Tracks overall ELO + surface-specific ELO (Hard, Clay, Grass, Carpet)
5. Stores final ELO ratings in players.csv
6. Adds ELO columns to matches.csv for feature engineering

ELO Formula:
- New Rating = Old Rating + K * (Actual - Expected)
- Expected = 1 / (1 + 10^((Opponent_ELO - Player_ELO) / 400))
- K-factor = 32 (standard for tennis)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class DynamicEloCalculator:
    """Calculate ELO ratings dynamically through match history"""
    
    def __init__(self, initial_elo=1500, k_factor=32):
        """
        Initialize ELO calculator
        
        Args:
            initial_elo: Starting ELO for all players (default: 1500)
            k_factor: Sensitivity to match results (default: 32)
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        
        # Track overall ELO for each player
        self.player_elos = {}
        
        # Track surface-specific ELO
        self.player_surface_elos = {
            'Hard': {},
            'Clay': {},
            'Grass': {},
            'Carpet': {}
        }
        
    def get_player_elo(self, player_name):
        """Get current overall ELO for a player"""
        if player_name not in self.player_elos:
            self.player_elos[player_name] = self.initial_elo
        return self.player_elos[player_name]
    
    def get_surface_elo(self, player_name, surface):
        """Get current surface-specific ELO for a player"""
        # Handle missing or unknown surfaces
        if pd.isna(surface) or surface not in self.player_surface_elos:
            surface = 'Hard'  # Default to hard court
        
        if player_name not in self.player_surface_elos[surface]:
            self.player_surface_elos[surface][player_name] = self.initial_elo
        
        return self.player_surface_elos[surface][player_name]
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score for player A vs player B
        
        Returns probability that player A wins (0 to 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo(self, winner_elo, loser_elo):
        """
        Update ELO ratings after a match
        
        Args:
            winner_elo: Current ELO of winner
            loser_elo: Current ELO of loser
            
        Returns:
            (new_winner_elo, new_loser_elo)
        """
        # Expected scores
        expected_winner = self.expected_score(winner_elo, loser_elo)
        expected_loser = 1 - expected_winner
        
        # Actual scores (winner gets 1, loser gets 0)
        actual_winner = 1
        actual_loser = 0
        
        # Update ratings
        new_winner_elo = winner_elo + self.k_factor * (actual_winner - expected_winner)
        new_loser_elo = loser_elo + self.k_factor * (actual_loser - expected_loser)
        
        return new_winner_elo, new_loser_elo
    
    def process_match(self, winner_name, loser_name, surface):
        """
        Process a single match and update all ELO ratings
        
        Returns:
            dict with before/after ELO values for features
        """
        # Get ELO ratings BEFORE the match
        winner_elo_before = self.get_player_elo(winner_name)
        loser_elo_before = self.get_player_elo(loser_name)
        
        winner_surface_elo_before = self.get_surface_elo(winner_name, surface)
        loser_surface_elo_before = self.get_surface_elo(loser_name, surface)
        
        # Calculate expected scores (for features)
        expected_winner = self.expected_score(winner_elo_before, loser_elo_before)
        expected_winner_surface = self.expected_score(winner_surface_elo_before, loser_surface_elo_before)
        
        # Update overall ELO
        winner_elo_after, loser_elo_after = self.update_elo(winner_elo_before, loser_elo_before)
        
        # Update surface-specific ELO
        if pd.notna(surface) and surface in self.player_surface_elos:
            winner_surface_elo_after, loser_surface_elo_after = self.update_elo(
                winner_surface_elo_before, loser_surface_elo_before
            )
        else:
            winner_surface_elo_after = winner_surface_elo_before
            loser_surface_elo_after = loser_surface_elo_before
        
        # Store updated ELOs
        self.player_elos[winner_name] = winner_elo_after
        self.player_elos[loser_name] = loser_elo_after
        
        if pd.notna(surface) and surface in self.player_surface_elos:
            self.player_surface_elos[surface][winner_name] = winner_surface_elo_after
            self.player_surface_elos[surface][loser_name] = loser_surface_elo_after
        
        # Return all ELO values for this match
        return {
            'winner_elo_before': winner_elo_before,
            'loser_elo_before': loser_elo_before,
            'winner_elo_after': winner_elo_after,
            'loser_elo_after': loser_elo_after,
            
            'winner_surface_elo_before': winner_surface_elo_before,
            'loser_surface_elo_before': loser_surface_elo_before,
            'winner_surface_elo_after': winner_surface_elo_after,
            'loser_surface_elo_after': loser_surface_elo_after,
            
            'elo_diff': winner_elo_before - loser_elo_before,
            'surface_elo_diff': winner_surface_elo_before - loser_surface_elo_before,
            
            'expected_winner_prob': expected_winner,
            'expected_winner_surface_prob': expected_winner_surface,
        }
    
    def get_final_ratings(self):
        """
        Get final ELO ratings for all players
        
        Returns:
            DataFrame with player names and all ELO ratings
        """
        players = []
        
        for player_name in self.player_elos.keys():
            player_data = {
                'player_name': player_name,
                'final_elo': self.player_elos[player_name],
            }
            
            # Add surface-specific ELOs
            for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                surface_elo = self.player_surface_elos[surface].get(player_name, self.initial_elo)
                player_data[f'final_elo_{surface.lower()}'] = surface_elo
            
            players.append(player_data)
        
        return pd.DataFrame(players)


def calculate_all_elos():
    """
    Main function to calculate ELO ratings for all matches
    """
    print("="*70)
    print("DYNAMIC ELO CALCULATION")
    print("="*70)
    
    # Load matches
    matches_path = Path('data/processed/matches.csv')
    print(f"\n📂 Loading matches from {matches_path}...")
    
    matches = pd.read_csv(matches_path, low_memory=False)
    print(f"   Loaded {len(matches):,} matches")
    
    # Sort by date (CRITICAL - must process chronologically!)
    print("\n📅 Sorting matches chronologically...")
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)
    
    print(f"   Date range: {matches['date'].min()} to {matches['date'].max()}")
    
    # Initialize ELO calculator
    print("\n🎯 Initializing ELO calculator...")
    print(f"   Initial ELO: 1500")
    print(f"   K-factor: 32")
    
    elo_calc = DynamicEloCalculator(initial_elo=1500, k_factor=32)
    
    # Process each match chronologically
    print(f"\n⚙️  Processing {len(matches):,} matches chronologically...")
    
    elo_records = []
    batch_size = 10000
    
    for idx, row in matches.iterrows():
        # Get winner and loser names
        winner_name = row['winner_name']
        loser_name = row['loser_name']
        surface = row['surface']
        
        # Process match and get ELO values
        elo_values = elo_calc.process_match(winner_name, loser_name, surface)
        
        # Store for this match
        elo_records.append(elo_values)
        
        # Progress indicator
        if (idx + 1) % batch_size == 0:
            print(f"   Processed {idx + 1:,} / {len(matches):,} matches ({(idx+1)/len(matches)*100:.1f}%)")
    
    print(f"   ✅ Processed all {len(matches):,} matches")
    
    # Convert ELO records to DataFrame
    print("\n📊 Creating ELO features DataFrame...")
    elo_df = pd.DataFrame(elo_records)
    
    # Add ELO columns to matches
    print("   Merging ELO features with matches...")
    for col in elo_df.columns:
        matches[col] = elo_df[col]
    
    # Save updated matches.csv
    print(f"\n💾 Saving updated matches.csv...")
    matches.to_csv(matches_path, index=False)
    print(f"   ✅ Saved {len(matches):,} matches with ELO features")
    
    # Get final ELO ratings for all players
    print("\n🏆 Calculating final ELO ratings for players...")
    final_elos = elo_calc.get_final_ratings()
    print(f"   Calculated final ELO for {len(final_elos):,} players")
    
    # Load existing players.csv
    players_path = Path('data/processed/players.csv')
    print(f"\n📂 Loading players from {players_path}...")
    players = pd.read_csv(players_path)
    print(f"   Loaded {len(players):,} players")
    
    # Merge ELO ratings into players.csv
    print("   Merging ELO ratings...")
    players = players.merge(final_elos, on='player_name', how='left')
    
    # Fill any missing ELOs with initial value
    elo_cols = ['final_elo', 'final_elo_hard', 'final_elo_clay', 'final_elo_grass', 'final_elo_carpet']
    for col in elo_cols:
        players[col] = players[col].fillna(1500)
    
    # Save updated players.csv
    print(f"\n💾 Saving updated players.csv...")
    players.to_csv(players_path, index=False)
    print(f"   ✅ Saved {len(players):,} players with final ELO ratings")
    
    # Show statistics
    print("\n" + "="*70)
    print("ELO STATISTICS")
    print("="*70)
    
    print(f"\n📈 Overall ELO Distribution:")
    print(players['final_elo'].describe())
    
    print(f"\n🏆 Top 10 Players by ELO:")
    top_players = players.nlargest(10, 'final_elo')[['player_name', 'final_elo', 'total_matches', 'win_percentage']]
    print(top_players.to_string(index=False))
    
    print(f"\n🎾 Surface-Specific ELO Leaders:")
    for surface in ['hard', 'clay', 'grass']:
        col = f'final_elo_{surface}'
        top_player = players.nlargest(1, col).iloc[0]
        print(f"   {surface.capitalize():8s}: {top_player['player_name']:30s} ({top_player[col]:.0f})")
    
    # Show sample matches with ELO
    print("\n" + "="*70)
    print("SAMPLE MATCHES WITH ELO (Most Recent)")
    print("="*70)
    
    sample_cols = ['date', 'winner_name', 'loser_name', 'surface',
                   'winner_elo_before', 'loser_elo_before', 'elo_diff',
                   'expected_winner_prob']
    
    sample = matches[sample_cols].tail(10)
    sample['date'] = sample['date'].dt.strftime('%Y-%m-%d')
    sample['expected_winner_prob'] = (sample['expected_winner_prob'] * 100).round(1).astype(str) + '%'
    sample['winner_elo_before'] = sample['winner_elo_before'].round(0).astype(int)
    sample['loser_elo_before'] = sample['loser_elo_before'].round(0).astype(int)
    sample['elo_diff'] = sample['elo_diff'].round(0).astype(int)
    
    print(sample.to_string(index=False))
    
    # Summary
    print("\n" + "="*70)
    print("✅ ELO CALCULATION COMPLETE!")
    print("="*70)
    print(f"""
✅ Processed {len(matches):,} matches chronologically
✅ Calculated ELO ratings for {len(final_elos):,} players
✅ Added {len(elo_df.columns)} ELO features to matches.csv
✅ Stored final ELO ratings in players.csv

ELO Features Added to matches.csv:
  - winner_elo_before, loser_elo_before (overall ELO before match)
  - winner_elo_after, loser_elo_after (overall ELO after match)
  - winner_surface_elo_before, loser_surface_elo_before (surface-specific)
  - winner_surface_elo_after, loser_surface_elo_after (surface-specific)
  - elo_diff (winner ELO - loser ELO)
  - surface_elo_diff (surface-specific difference)
  - expected_winner_prob (probability based on overall ELO)
  - expected_winner_surface_prob (probability based on surface ELO)

ELO Ratings Added to players.csv:
  - final_elo (overall ELO after all matches)
  - final_elo_hard (Hard court specific)
  - final_elo_clay (Clay court specific)
  - final_elo_grass (Grass court specific)
  - final_elo_carpet (Carpet court specific)

🎯 Next Steps:
  1. Use ELO features for model training
  2. Calculate player career statistics
  3. Build prediction model with ELO + rankings + career stats
""")
    
    # Save metadata
    metadata = {
        'calculation_date': datetime.now().isoformat(),
        'total_matches': len(matches),
        'total_players': len(final_elos),
        'date_range': {
            'start': matches['date'].min().isoformat(),
            'end': matches['date'].max().isoformat()
        },
        'elo_config': {
            'initial_elo': 1500,
            'k_factor': 32
        },
        'elo_features': list(elo_df.columns),
        'elo_stats': {
            'mean': float(players['final_elo'].mean()),
            'median': float(players['final_elo'].median()),
            'min': float(players['final_elo'].min()),
            'max': float(players['final_elo'].max()),
            'std': float(players['final_elo'].std())
        }
    }
    
    metadata_path = Path('data/processed/elo_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved to {metadata_path}")
    print("="*70)


if __name__ == '__main__':
    calculate_all_elos()
