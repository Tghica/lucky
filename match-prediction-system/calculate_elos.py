#!/usr/bin/env python3
"""
Calculate ELO ratings for all players from existing match_info.csv
This adds ELO columns needed for predictions.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor.calculator import EloCalculator

def main():
    print("="*80)
    print("CALCULATING ELO RATINGS FROM MATCH HISTORY")
    print("="*80)
    
    # Load existing match_info
    match_path = Path('data/processed/match_info.csv')
    print(f"\nLoading {match_path}...")
    df = pd.read_csv(match_path)
    print(f"Loaded {len(df):,} matches from {df['date'].min()} to {df['date'].max()}")
    
    # Initialize ELO calculator
    print("\nCalculating ELO ratings...")
    elo_calc = EloCalculator(initial_rating=1500, k_factor=32)
    
    # Calculate all ELO types
    df_with_elo = elo_calc.calculate_match_elos(df)
    
    # Calculate form (last 10 matches)
    print("Calculating player form...")
    df_with_elo = elo_calc.calculate_form(df_with_elo, window=10)
    
    # Calculate fatigue
    print("Calculating fatigue features...")
    df_with_elo = elo_calc.calculate_fatigue(df_with_elo)
    
    # Calculate tournament progression
    print("Calculating tournament progression...")
    df_with_elo = elo_calc.calculate_tournament_progression(df_with_elo)
    
    # Calculate win streaks
    print("Calculating win streaks...")
    df_with_elo = elo_calc.calculate_win_streaks(df_with_elo)
    
    # Calculate head-to-head
    print("Calculating head-to-head statistics...")
    df_with_elo = elo_calc.calculate_head_to_head(df_with_elo)
    
    # Save
    print(f"\nSaving updated match_info.csv...")
    df_with_elo.to_csv(match_path, index=False)
    
    print(f"✅ Successfully added ELO ratings to {len(df_with_elo):,} matches")
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE OF UPDATED DATA (most recent matches):")
    print("="*80)
    sample_cols = ['date', 'player1', 'player2', 'winner', 
                   'player1_elo_before', 'player2_elo_before',
                   'player1_form_wins', 'player2_form_wins']
    print(df_with_elo[sample_cols].tail(10).to_string(index=False))
    
    # Show ELO column stats
    print("\n" + "="*80)
    elo_cols = [col for col in df_with_elo.columns if 'elo' in col.lower()]
    print(f"Added {len(elo_cols)} ELO-related columns:")
    for col in elo_cols[:15]:  # Show first 15
        print(f"  - {col}")
    if len(elo_cols) > 15:
        print(f"  ... and {len(elo_cols) - 15} more")
    
    print("\n" + "="*80)
    print("✅ DONE! Your predictions will now use current ELO ratings!")
    print("="*80)

if __name__ == "__main__":
    main()
