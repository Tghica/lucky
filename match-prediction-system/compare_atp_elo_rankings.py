"""
Compare ATP Official Rankings vs Elo Ratings on December 8, 2020.

This script shows how well the Elo rating system correlates with official ATP rankings.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_atp_rankings():
    """Load ATP rankings from the CSV file."""
    print("Loading ATP rankings from atp_rankings_dec2020.csv...")
    
    # Read the properly formatted file
    atp_rankings = pd.read_csv('data/sheets/archive/atp_rankings_dec2020.csv')
    atp_rankings.rename(columns={'rank': 'atp_rank'}, inplace=True)
    
    print(f"\nFound {len(atp_rankings)} players in ATP rankings")
    print(f"Top 5: {', '.join(atp_rankings.head(5)['player_name'].tolist())}")
    
    return atp_rankings


def load_elo_ratings(target_date):
    """Load Elo ratings as of a specific date."""
    print(f"\nLoading Elo ratings as of {target_date}...")
    
    # Load match info with Elo ratings
    matches = pd.read_csv('data/processed/match_info.csv')
    matches['date'] = pd.to_datetime(matches['date'])
    
    # Filter matches up to target date
    matches_before = matches[matches['date'] <= target_date].copy()
    
    print(f"Found {len(matches_before)} matches before {target_date}")
    
    # Get the last known Elo for each player
    # We'll use player1 and player2 columns
    player_elos = {}
    
    for _, match in matches_before.iterrows():
        # Update player1's Elo
        if pd.notna(match['player1_elo_after']):
            player_elos[match['player1']] = match['player1_elo_after']
        
        # Update player2's Elo
        if pd.notna(match['player2_elo_after']):
            player_elos[match['player2']] = match['player2_elo_after']
    
    # Create DataFrame
    elo_ratings = pd.DataFrame([
        {'player_name': player, 'elo_rating': elo}
        for player, elo in player_elos.items()
    ])
    
    # Sort by Elo rating
    elo_ratings = elo_ratings.sort_values('elo_rating', ascending=False).reset_index(drop=True)
    elo_ratings['elo_rank'] = range(1, len(elo_ratings) + 1)
    
    print(f"Found {len(elo_ratings)} players with Elo ratings")
    
    return elo_ratings


def normalize_player_name(name):
    """Normalize player names for matching."""
    if pd.isna(name):
        return ""
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Convert to lowercase for comparison
    name = name.lower()
    
    # Remove common prefixes/suffixes
    name = name.replace('jr.', '').replace('jr', '')
    name = name.replace('sr.', '').replace('sr', '')
    
    # Remove dots (for initials)
    name = name.replace('.', '')
    
    return name.strip()


def fuzzy_match_names(atp_name, elo_names):
    """Find best match for ATP player name in Elo names."""
    # ATP format: "First Last" (e.g., "Novak Djokovic")
    # Elo format: "Last F." (e.g., "Djokovic N.")
    
    atp_parts = atp_name.strip().split()
    if len(atp_parts) < 2:
        return None
    
    # Extract first and last name from ATP format
    first_name = atp_parts[0]
    last_name = atp_parts[-1]
    first_initial = first_name[0].upper()
    
    # Look for matches in Elo system
    # Format 1: "Last F." (e.g., "Djokovic N.")
    target1 = f"{last_name} {first_initial}."
    target2 = f"{last_name} {first_initial}. "  # With space
    
    # Direct match
    for elo_name in elo_names:
        elo_clean = elo_name.strip()
        if elo_clean == target1 or elo_clean == target2:
            return elo_name
    
    # Case-insensitive match
    target1_lower = target1.lower()
    for elo_name in elo_names:
        if elo_name.strip().lower() == target1_lower:
            return elo_name
    
    # Match without dot
    target_no_dot = f"{last_name} {first_initial}"
    for elo_name in elo_names:
        elo_clean = elo_name.strip().replace('.', '')
        if elo_clean.lower() == target_no_dot.lower():
            return elo_name
    
    return None


def compare_rankings(atp_rankings, elo_ratings):
    """Compare ATP rankings with Elo ratings."""
    print("\n" + "="*80)
    print("COMPARING ATP RANKINGS VS ELO RATINGS")
    print("="*80)
    
    # Merge the dataframes
    elo_names = elo_ratings['player_name'].tolist()
    
    matched_players = []
    unmatched_atp = []
    
    for _, atp_row in atp_rankings.head(100).iterrows():  # Top 100 ATP
        atp_name = atp_row['player_name']
        
        # Try to find match in Elo ratings
        matched_elo_name = fuzzy_match_names(atp_name, elo_names)
        
        if matched_elo_name:
            elo_row = elo_ratings[elo_ratings['player_name'] == matched_elo_name].iloc[0]
            matched_players.append({
                'player_name': atp_name,
                'atp_rank': atp_row['atp_rank'],
                'atp_points': atp_row['atp_points'],
                'elo_rank': elo_row['elo_rank'],
                'elo_rating': elo_row['elo_rating'],
                'rank_diff': elo_row['elo_rank'] - atp_row['atp_rank']
            })
        else:
            unmatched_atp.append(atp_name)
    
    comparison_df = pd.DataFrame(matched_players)
    
    print(f"\nMatched {len(comparison_df)} players")
    print(f"Unmatched ATP players: {len(unmatched_atp)}")
    
    return comparison_df, unmatched_atp


def analyze_comparison(comparison_df):
    """Analyze the comparison between ATP and Elo rankings."""
    print("\n" + "="*80)
    print("TOP 20 PLAYERS - ATP RANK vs ELO RANK (December 8, 2020)")
    print("="*80)
    print(f"\n{'ATP':<4} {'Player':<25} {'ATP Pts':<10} {'Elo Rank':<10} {'Elo Rating':<12} {'Diff':<8}")
    print("-"*80)
    
    for _, row in comparison_df.head(20).iterrows():
        diff_str = f"+{int(row['rank_diff'])}" if row['rank_diff'] > 0 else str(int(row['rank_diff']))
        print(f"{int(row['atp_rank']):<4} {row['player_name']:<25} {int(row['atp_points']):<10} "
              f"{int(row['elo_rank']):<10} {row['elo_rating']:<12.1f} {diff_str:<8}")
    
    # Statistics
    print("\n" + "="*80)
    print("CORRELATION STATISTICS")
    print("="*80)
    
    correlation = comparison_df['atp_rank'].corr(comparison_df['elo_rank'])
    print(f"\nSpearman Rank Correlation: {correlation:.4f}")
    
    avg_diff = comparison_df['rank_diff'].abs().mean()
    median_diff = comparison_df['rank_diff'].abs().median()
    max_diff = comparison_df['rank_diff'].abs().max()
    
    print(f"Average rank difference: {avg_diff:.1f} positions")
    print(f"Median rank difference: {median_diff:.1f} positions")
    print(f"Maximum rank difference: {int(max_diff)} positions")
    
    # Players ranked higher by Elo
    print("\n" + "="*80)
    print("BIGGEST DIFFERENCES - ELO RANKS HIGHER (Underrated by ATP)")
    print("="*80)
    
    underrated = comparison_df.nsmallest(10, 'rank_diff')
    print(f"\n{'Player':<25} {'ATP Rank':<12} {'Elo Rank':<12} {'Difference':<12}")
    print("-"*80)
    for _, row in underrated.iterrows():
        print(f"{row['player_name']:<25} {int(row['atp_rank']):<12} {int(row['elo_rank']):<12} "
              f"{int(row['rank_diff']):<12}")
    
    # Players ranked higher by ATP
    print("\n" + "="*80)
    print("BIGGEST DIFFERENCES - ATP RANKS HIGHER (Overrated by ATP)")
    print("="*80)
    
    overrated = comparison_df.nlargest(10, 'rank_diff')
    print(f"\n{'Player':<25} {'ATP Rank':<12} {'Elo Rank':<12} {'Difference':<12}")
    print("-"*80)
    for _, row in overrated.iterrows():
        print(f"{row['player_name']:<25} {int(row['atp_rank']):<12} {int(row['elo_rank']):<12} "
              f"+{int(row['rank_diff']):<12}")
    
    # Save results
    comparison_df.to_csv('models/saved_models/atp_vs_elo_comparison_2020.csv', index=False)
    print("\n✓ Results saved to: models/saved_models/atp_vs_elo_comparison_2020.csv")


def main():
    """Main comparison pipeline."""
    print("="*80)
    print("🎾 ATP RANKINGS vs ELO RATINGS COMPARISON")
    print("Date: December 8, 2020")
    print("="*80)
    
    # Load ATP rankings
    try:
        atp_rankings = load_atp_rankings()
    except Exception as e:
        print(f"\nError loading ATP rankings: {e}")
        print("\nPlease ensure atp2020.csv has the correct format:")
        print("The file should contain player names and ATP points")
        return
    
    # Load Elo ratings as of December 8, 2020
    target_date = pd.Timestamp('2020-12-08')
    elo_ratings = load_elo_ratings(target_date)
    
    # Compare rankings
    comparison_df, unmatched = compare_rankings(atp_rankings, elo_ratings)
    
    if len(comparison_df) > 0:
        # Analyze comparison
        analyze_comparison(comparison_df)
        
        if len(unmatched) > 0:
            print(f"\n⚠️  Unmatched ATP players (not found in Elo system):")
            print(", ".join(unmatched[:10]))
            if len(unmatched) > 10:
                print(f"... and {len(unmatched) - 10} more")
    else:
        print("\n❌ No players could be matched between ATP and Elo systems")
        print("This might be a data format issue. Please check the ATP rankings file.")
    
    print("\n✓ COMPARISON COMPLETE!")


if __name__ == '__main__':
    main()
