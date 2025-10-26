from data_processor.data_loader import DataLoader
from data_processor.calculator import EloCalculator

def main():
    # Initialize data loader
    loader = DataLoader(sheets_folder="data/sheets")
    
    # Merge all sheets into match_info.csv
    print("Starting data merge process...")
    merged_df = loader.merge_all_sheets(output_path="data/processed/match_info.csv")
    
    if merged_df is not None:
        # Display summary
        summary = loader.get_data_summary(merged_df)
        print("\n=== Match Data Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Calculate Elo ratings (both general and surface-specific)
        print("\n" + "="*50)
        print("Calculating General, Surface-Specific, and Tournament-Specific Elo ratings...")
        elo_calc = EloCalculator(initial_rating=1500, k_factor=32)
        matches_with_elo = elo_calc.calculate_match_elos(merged_df)
        
        # Calculate form (last 10 matches performance)
        print("Calculating player form (last 10 matches)...")
        matches_with_elo = elo_calc.calculate_form(matches_with_elo, window=10)
        
        # Calculate head-to-head statistics
        print("Calculating head-to-head win rates...")
        matches_with_elo = elo_calc.calculate_head_to_head(matches_with_elo)
        
        # Save matches with Elo ratings, form, and h2h
        matches_with_elo.to_csv("data/processed/match_info.csv", index=False)
        print(f"\nMatch info with Elo ratings, form, and H2H saved to data/processed/match_info.csv")
        
        # Show sample with Elo
        print("\n=== Sample Matches with All Features (Elo, Surface, Tournament, Form, H2H) ===")
        print(matches_with_elo[['date', 'player1', 'player2', 'winner', 'surface', 'tournament_level',
                                 'player1_elo_before', 'player2_elo_before',
                                 'player1_form_wins', 'player2_form_wins',
                                 'player1_h2h_win_rate', 'player2_h2h_win_rate']].tail(10).to_string(index=False))
        
        # Create player database
        print("\n" + "="*50)
        print("Creating player database...")
        player_df = loader.create_player_database(merged_df, output_path="data/processed/player_info.csv")
        
        # Add final Elo ratings to player database (general, surface-specific, and tournament-specific)
        all_ratings = elo_calc.get_all_ratings()
        player_df['current_elo'] = player_df['player_name'].map(all_ratings['general'])
        player_df['current_elo_hard'] = player_df['player_name'].map(all_ratings['hard'])
        player_df['current_elo_clay'] = player_df['player_name'].map(all_ratings['clay'])
        player_df['current_elo_grass'] = player_df['player_name'].map(all_ratings['grass'])
        player_df['current_elo_carpet'] = player_df['player_name'].map(all_ratings['carpet'])
        player_df['current_elo_grand_slam'] = player_df['player_name'].map(all_ratings['grand_slam'])
        player_df['current_elo_masters'] = player_df['player_name'].map(all_ratings['masters'])
        player_df['current_elo_atp500'] = player_df['player_name'].map(all_ratings['atp500'])
        player_df['current_elo_atp250'] = player_df['player_name'].map(all_ratings['atp250'])
        
        # Sort by general Elo
        player_df = player_df.sort_values('current_elo', ascending=False).reset_index(drop=True)
        player_df.to_csv("data/processed/player_info.csv", index=False)
        
        print("\n=== Player Database Summary ===")
        print(f"Total players: {len(player_df)}")
        print(f"\nTop 10 Players by Current General Elo:")
        print(player_df[['player_name', 'current_elo', 'current_elo_hard', 'current_elo_clay', 'current_elo_grass', 'total_matches', 'win_rate']].head(10).to_string(index=False))
        
    else:
        print("Failed to merge data")

if __name__ == "__main__":
    main()