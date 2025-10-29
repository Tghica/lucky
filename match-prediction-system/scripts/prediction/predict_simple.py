#!/usr/bin/env python3
"""
Simple match prediction script that uses the same feature engineering as training.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processor.feature_engineering import FeatureEngineering


def predict_match(player1_name, player2_name, surface='Hard', tournament='ATP250'):
    """
    Predict match outcome between two players.
    
    Args:
        player1_name: Name of player 1
        player2_name: Name of player 2
        surface: Surface type (Hard, Clay, Grass, Carpet)
        tournament: Tournament level (ATP250, ATP500, Masters, Grand_Slam)
    """
    print("="*70)
    print(f"MATCH PREDICTION: {player1_name} vs {player2_name}")
    print(f"Surface: {surface} | Tournament: {tournament}")
    print("="*70)
    
    # Load model
    model_path = Path('models/saved_models/xgboost_model.pkl')
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load match data
    print("Loading match database...")
    match_info = pd.read_csv('data/processed/match_info.csv', low_memory=False)
    
    # Get most recent ELO ratings for both players
    def get_latest_elo(player_name, df):
        """Get player's most recent ELO ratings from their match history."""
        # Find all matches where player participated
        p1_matches = df[df['player1'] == player_name]
        p2_matches = df[df['player2'] == player_name]
        
        # Get most recent match
        if len(p1_matches) > 0:
            latest = p1_matches.iloc[-1]
            return {
                'elo': latest.get('player1_elo_after', 1500.0),
                'surface_elo': latest.get('player1_surface_elo_after', 1500.0),
                'tournament_elo': latest.get('player1_tournament_elo_after', 1500.0)
            }
        elif len(p2_matches) > 0:
            latest = p2_matches.iloc[-1]
            return {
                'elo': latest.get('player2_elo_after', 1500.0),
                'surface_elo': latest.get('player2_surface_elo_after', 1500.0),
                'tournament_elo': latest.get('player2_tournament_elo_after', 1500.0)
            }
        else:
            return {'elo': 1500.0, 'surface_elo': 1500.0, 'tournament_elo': 1500.0}
    
    player1_elo = get_latest_elo(player1_name, match_info)
    player2_elo = get_latest_elo(player2_name, match_info)
    
    print(f"\n{player1_name}: ELO={player1_elo['elo']:.0f}, Surface ELO={player1_elo['surface_elo']:.0f}")
    print(f"{player2_name}: ELO={player2_elo['elo']:.0f}, Surface ELO={player2_elo['surface_elo']:.0f}")
    
    # Create a hypothetical match
    # Get current date
    from datetime import datetime
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create match row with ELO values pre-filled
    hypothetical_match = pd.DataFrame([{
        'date': current_date,
        'player1': player1_name,
        'player2': player2_name,
        'winner': player1_name,  # Dummy value (will be ignored)
        'surface': surface,
        'description': tournament,
        'stadium': 'Unknown',
        'nation': 'Unknown',
        # Pre-fill ELO values from most recent matches
        'player1_elo_before': player1_elo['elo'],
        'player1_surface_elo_before': player1_elo['surface_elo'],
        'player1_tournament_elo_before': player1_elo['tournament_elo'],
        'player2_elo_before': player2_elo['elo'],
        'player2_surface_elo_before': player2_elo['surface_elo'],
        'player2_tournament_elo_before': player2_elo['tournament_elo'],
    }])
    
    # Append to match_info (temporarily)
    combined_data = pd.concat([match_info, hypothetical_match], ignore_index=True)
    
    # Save temporarily
    temp_path = Path('data/processed/temp_match.csv')
    combined_data.to_csv(temp_path, index=False)
    
    # Generate features using the same process as training
    print("Generating features...")
    feature_engineer = FeatureEngineering(
        match_data_path=str(temp_path),
        player_info_path='data/processed/player_info.csv'
    )
    
    # Generate features
    X, y = feature_engineer.prepare_features()
    
    # Get the last row (our hypothetical match)
    X_pred = X.iloc[[-1]]
    
    # Clean up temp file
    temp_path.unlink()
    
    # Make prediction
    print("\nMaking prediction...\n")
    prob = model.predict_proba(X_pred)[0]
    
    # Display results
    print("="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\n{player1_name} win probability: {prob[1]*100:.1f}%")
    print(f"{player2_name} win probability: {prob[0]*100:.1f}%")
    
    if prob[1] > prob[0]:
        margin = (prob[1] - prob[0]) * 100
        print(f"\n✅ Prediction: {player1_name} wins ({margin:.1f}% margin)")
    else:
        margin = (prob[0] - prob[1]) * 100
        print(f"\n✅ Prediction: {player2_name} wins ({margin:.1f}% margin)")
    
    # Show confidence
    confidence = max(prob)
    if confidence > 0.7:
        print(f"Confidence: HIGH ({confidence*100:.1f}%)")
    elif confidence > 0.6:
        print(f"Confidence: MEDIUM ({confidence*100:.1f}%)")
    else:
        print(f"Confidence: LOW ({confidence*100:.1f}%)")
    
    print("="*70)


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_simple.py <player1> <player2> [surface] [tournament]")
        print("Example: python predict_simple.py 'Sinner J.' 'Alcaraz C.' Hard ATP250")
        sys.exit(1)
    
    player1 = sys.argv[1]
    player2 = sys.argv[2]
    surface = sys.argv[3] if len(sys.argv) > 3 else 'Hard'
    tournament = sys.argv[4] if len(sys.argv) > 4 else 'ATP250'
    
    predict_match(player1, player2, surface, tournament)


if __name__ == '__main__':
    main()
