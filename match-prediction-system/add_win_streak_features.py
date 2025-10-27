"""
Script to add win streak features to existing match_info.csv without recalculating everything.
This is faster than regenerating all features from scratch.
"""

import pandas as pd
from src.data_processor.calculator import EloCalculator

def main():
    print("Loading existing match_info.csv...")
    df = pd.read_csv('data/processed/match_info.csv')
    print(f"Loaded {len(df)} matches")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date (required for streak calculation)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize calculator
    elo_calc = EloCalculator()
    
    # Calculate win streaks
    print("\nCalculating win streak features...")
    df = elo_calc.calculate_win_streaks(df)
    
    # Save updated data
    df.to_csv('data/processed/match_info.csv', index=False)
    print(f"\n✓ Win streak features added! Saved to data/processed/match_info.csv")
    
    # Show sample
    print("\n=== Sample with Win Streak Features ===")
    print(df[['date', 'player1', 'player2', 'winner',
              'player1_win_streak', 'player2_win_streak',
              'player1_surface_win_streak', 'player2_surface_win_streak',
              'player1_wins_last_5', 'player2_wins_last_5']].tail(10))
    
    # Show some stats
    print("\n=== Win Streak Statistics ===")
    print(f"Max win streak: {df['player1_win_streak'].max()}")
    print(f"Max loss streak: {df['player1_win_streak'].min()}")
    print(f"Avg wins in last 5: {df['player1_wins_last_5'].mean():.2f}")

if __name__ == '__main__':
    main()
