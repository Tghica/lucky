#!/usr/bin/env python3
"""
Quick match prediction script
Usage: python3 predict_match.py "Player1 Name" "Player2 Name" [surface] [tournament_level]
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_model():
    """Load the trained model"""
    model_path = Path('models/saved_models/xgboost_model.pkl')
    if not model_path.exists():
        print("❌ Model not found. Please train the model first using train_quick.py")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_player_features(player_name, match_info_df):
    """Extract latest features for a player from match history"""
    
    # Get all matches where this player played
    player_matches = match_info_df[
        (match_info_df['player1'] == player_name) | 
        (match_info_df['player2'] == player_name)
    ].sort_values('date', ascending=False)
    
    if len(player_matches) == 0:
        print(f"⚠️  {player_name} not found in database - using defaults")
        return {
            'elo_before': 1500.0,
            'surface_elo_before': 1500.0,
            'tournament_elo_before': 1500.0,
            'form_wins': 5,
            'recent_form': 0.0,
            'momentum': 0.0,
            'win_streak': 0,
            'surface_win_streak': 0,
            'wins_last_5': 2,
            'days_since_last': 30,
            'matches_in_tournament': 0,
            'h2h_wins': 0,
            'h2h_matches': 0,
            'height': 180,
            'age': 25,
            'hand': 1,  # Right-handed
        }
    
    # Get most recent match features
    latest = player_matches.iloc[0]
    
    # Determine if player was player1 or player2 in their last match
    if latest['player1'] == player_name:
        prefix = 'player1'
    else:
        prefix = 'player2'
    
    return {
        'elo_before': latest.get(f'{prefix}_elo_before', 1500.0),
        'surface_elo_before': latest.get(f'{prefix}_surface_elo_before', 1500.0),
        'tournament_elo_before': latest.get(f'{prefix}_tournament_elo_before', 1500.0),
        'form_wins': latest.get(f'{prefix}_form_wins', 5),
        'recent_form': latest.get(f'{prefix}_recent_form', 0.0),
        'momentum': latest.get(f'{prefix}_momentum', 0.0),
        'win_streak': latest.get(f'{prefix}_win_streak', 0),
        'surface_win_streak': latest.get(f'{prefix}_surface_win_streak', 0),
        'wins_last_5': latest.get(f'{prefix}_wins_last_5', 2),
        'days_since_last': latest.get(f'{prefix}_days_since_last', 30),
        'matches_in_tournament': 0,  # New tournament
        'h2h_wins': latest.get(f'{prefix}_h2h_wins', 0),
        'h2h_matches': latest.get(f'{prefix}_h2h_matches', 0),
        'height': latest.get(f'{prefix}_height', 180),
        'age': latest.get(f'{prefix}_age', 25),
        'hand': latest.get(f'{prefix}_hand_encoded', 1),
    }

def create_match_features(p1_name, p2_name, p1_feat, p2_feat, surface='Hard', tournament_level='atp250'):
    """Create full feature vector for prediction"""
    
    # ELO features
    features = {
        'player1_elo_before': p1_feat['elo_before'],
        'player2_elo_before': p2_feat['elo_before'],
        'elo_diff': p1_feat['elo_before'] - p2_feat['elo_before'],
        'elo_ratio': p1_feat['elo_before'] / max(p2_feat['elo_before'], 1),
        'combined_elo_p1': p1_feat['elo_before'],
        'combined_elo_p2': p2_feat['elo_before'],
        'combined_elo_diff': p1_feat['elo_before'] - p2_feat['elo_before'],
    }
    
    # Win probability from ELO
    elo_diff = features['elo_diff']
    features['win_probability'] = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Surface ELO features
    features.update({
        'player1_surface_elo_before': p1_feat['surface_elo_before'],
        'player2_surface_elo_before': p2_feat['surface_elo_before'],
        'surface_elo_diff': p1_feat['surface_elo_before'] - p2_feat['surface_elo_before'],
        'surface_elo_ratio': p1_feat['surface_elo_before'] / max(p2_feat['surface_elo_before'], 1),
    })
    
    surface_elo_diff = features['surface_elo_diff']
    features['surface_win_probability'] = 1 / (1 + 10 ** (-surface_elo_diff / 400))
    
    # Tournament ELO features
    features.update({
        'player1_tournament_elo_before': p1_feat['tournament_elo_before'],
        'player2_tournament_elo_before': p2_feat['tournament_elo_before'],
        'tournament_elo_diff': p1_feat['tournament_elo_before'] - p2_feat['tournament_elo_before'],
        'tournament_elo_ratio': p1_feat['tournament_elo_before'] / max(p2_feat['tournament_elo_before'], 1),
    })
    
    tournament_elo_diff = features['tournament_elo_diff']
    features['tournament_win_probability'] = 1 / (1 + 10 ** (-tournament_elo_diff / 400))
    
    # H2H features
    features.update({
        'player1_h2h_wins': p1_feat['h2h_wins'],
        'player1_h2h_matches': p1_feat['h2h_matches'],
        'player1_h2h_win_rate': p1_feat['h2h_wins'] / max(p1_feat['h2h_matches'], 1) if p1_feat['h2h_matches'] > 0 else 0.5,
        'player2_h2h_wins': p2_feat['h2h_wins'],
        'player2_h2h_matches': p2_feat['h2h_matches'],
        'player2_h2h_win_rate': p2_feat['h2h_wins'] / max(p2_feat['h2h_matches'], 1) if p2_feat['h2h_matches'] > 0 else 0.5,
        'h2h_win_rate_diff': (p1_feat['h2h_wins'] / max(p1_feat['h2h_matches'], 1)) - (p2_feat['h2h_wins'] / max(p2_feat['h2h_matches'], 1)) if p1_feat['h2h_matches'] > 0 else 0,
        'h2h_experience': p1_feat['h2h_matches'],
        'player1_h2h_elo_advantage': 0,
        'player2_h2h_elo_advantage': 0,
        'h2h_elo_advantage_diff': 0,
    })
    
    # Win streak features
    features.update({
        'player1_win_streak': p1_feat['win_streak'],
        'player2_win_streak': p2_feat['win_streak'],
        'player1_surface_win_streak': p1_feat['surface_win_streak'],
        'player2_surface_win_streak': p2_feat['surface_win_streak'],
        'player1_wins_last_5': p1_feat['wins_last_5'],
        'player2_wins_last_5': p2_feat['wins_last_5'],
        'streak_advantage': p1_feat['win_streak'] - p2_feat['win_streak'],
        'surface_streak_advantage': p1_feat['surface_win_streak'] - p2_feat['surface_win_streak'],
        'player1_on_winning_streak': 1 if p1_feat['win_streak'] >= 3 else 0,
        'player2_on_winning_streak': 1 if p2_feat['win_streak'] >= 3 else 0,
        'player1_on_losing_streak': 1 if p1_feat['win_streak'] <= -3 else 0,
        'player2_on_losing_streak': 1 if p2_feat['win_streak'] <= -3 else 0,
        'momentum_advantage': 1 if p1_feat['win_streak'] > p2_feat['win_streak'] else 0,
        'both_hot': 1 if (p1_feat['win_streak'] >= 3 and p2_feat['win_streak'] >= 3) else 0,
        'both_cold': 1 if (p1_feat['win_streak'] <= -3 and p2_feat['win_streak'] <= -3) else 0,
        'hot_vs_cold': 1 if (p1_feat['win_streak'] >= 3 and p2_feat['win_streak'] <= -3) or (p2_feat['win_streak'] >= 3 and p1_feat['win_streak'] <= -3) else 0,
    })
    
    # Last 10 match results (dummy data - all 0s for simplicity)
    for i in range(1, 11):
        features[f'player1_match_{i}'] = 0
        features[f'player2_match_{i}'] = 0
    
    # Rest/fatigue features
    features.update({
        'player1_days_since_last': p1_feat['days_since_last'],
        'player2_days_since_last': p2_feat['days_since_last'],
        'rest_advantage': p1_feat['days_since_last'] - p2_feat['days_since_last'],
        'player1_fatigued': 1 if p1_feat['days_since_last'] < 2 else 0,
        'player2_fatigued': 1 if p2_feat['days_since_last'] < 2 else 0,
        'both_rested': 1 if (p1_feat['days_since_last'] >= 7 and p2_feat['days_since_last'] >= 7) else 0,
    })
    
    # Tournament progression features
    features.update({
        'player1_matches_in_tournament': p1_feat['matches_in_tournament'],
        'player2_matches_in_tournament': p2_feat['matches_in_tournament'],
        'tournament_experience_diff': p1_feat['matches_in_tournament'] - p2_feat['matches_in_tournament'],
        'player1_deep_run': 1 if p1_feat['matches_in_tournament'] >= 3 else 0,
        'player2_deep_run': 1 if p2_feat['matches_in_tournament'] >= 3 else 0,
    })
    
    # Physical features
    features.update({
        'player1_height': p1_feat['height'],
        'player2_height': p2_feat['height'],
        'player1_age': p1_feat['age'],
        'player2_age': p2_feat['age'],
        'age_diff': p1_feat['age'] - p2_feat['age'],
        'height_diff': p1_feat['height'] - p2_feat['height'],
        'player1_taller': 1 if p1_feat['height'] > p2_feat['height'] else 0,
        'player1_hand_encoded': p1_feat['hand'],
        'player2_hand_encoded': p2_feat['hand'],
        'same_hand': 1 if p1_feat['hand'] == p2_feat['hand'] else 0,
        'player1_lefty': 1 if p1_feat['hand'] == 0 else 0,
        'player2_lefty': 1 if p2_feat['hand'] == 0 else 0,
        'lefty_vs_righty': 1 if p1_feat['hand'] != p2_feat['hand'] else 0,
    })
    
    # Surface (one-hot encoding)
    features.update({
        'surface_Carpet': 1 if surface == 'Carpet' else 0,
        'surface_Clay': 1 if surface == 'Clay' else 0,
        'surface_Grass': 1 if surface == 'Grass' else 0,
        'surface_Hard': 1 if surface == 'Hard' else 0,
    })
    
    # Tournament level (one-hot encoding)
    features.update({
        'tournament_atp250': 1 if tournament_level == 'atp250' else 0,
        'tournament_atp500': 1 if tournament_level == 'atp500' else 0,
        'tournament_grand_slam': 1 if tournament_level == 'grand_slam' else 0,
        'tournament_masters': 1 if tournament_level == 'masters' else 0,
    })
    
    # Form features
    features.update({
        'player1_form_wins': p1_feat['form_wins'],
        'player2_form_wins': p2_feat['form_wins'],
        'form_win_diff': p1_feat['form_wins'] - p2_feat['form_wins'],
        'player1_recent_form': p1_feat['recent_form'],
        'player2_recent_form': p2_feat['recent_form'],
        'recent_form_diff': p1_feat['recent_form'] - p2_feat['recent_form'],
        'player1_momentum': p1_feat['momentum'],
        'player2_momentum': p2_feat['momentum'],
        'momentum_diff': p1_feat['momentum'] - p2_feat['momentum'],
    })
    
    # Additional derived features
    features.update({
        'player1_fatigue_impact': 0,
        'player2_fatigue_impact': 0,
        'fatigue_impact_diff': 0,
        'player1_rest_quality': min(p1_feat['days_since_last'] / 7, 1),
        'player2_rest_quality': min(p2_feat['days_since_last'] / 7, 1),
        'rest_quality_diff': min(p1_feat['days_since_last'] / 7, 1) - min(p2_feat['days_since_last'] / 7, 1),
        'player1_tournament_momentum': 0,
        'player2_tournament_momentum': 0,
        'tournament_momentum_diff': 0,
        'player1_tournament_rounds': p1_feat['matches_in_tournament'],
        'player2_tournament_rounds': p2_feat['matches_in_tournament'],
        'tournament_rounds_diff': p1_feat['matches_in_tournament'] - p2_feat['matches_in_tournament'],
        'player1_surface_advantage': 0,
        'player2_surface_advantage': 0,
        'surface_advantage_diff': 0,
        'player1_surface_specialist': 0,
        'player2_surface_specialist': 0,
        'player1_surface_advantage_pct': 0.5,
        'player2_surface_advantage_pct': 0.5,
        'surface_advantage_pct_diff': 0,
    })
    
    return features

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 predict_match.py 'Player1 Name' 'Player2 Name' [surface] [tournament_level]")
        print("Example: python3 predict_match.py 'Monday J.' 'Karki R.' Hard atp250")
        sys.exit(1)
    
    player1 = sys.argv[1]
    player2 = sys.argv[2]
    surface = sys.argv[3] if len(sys.argv) > 3 else 'Hard'
    tournament_level = sys.argv[4] if len(sys.argv) > 4 else 'atp250'
    
    print("="*80)
    print(f"MATCH PREDICTION: {player1} vs {player2}")
    print(f"Surface: {surface} | Tournament: {tournament_level.upper()}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    match_info = pd.read_csv('data/processed/match_info.csv')
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Get player features
    print("\nExtracting player features...")
    p1_feat = get_player_features(player1, match_info)
    p2_feat = get_player_features(player2, match_info)
    
    # Display player stats
    print(f"\n{player1}:")
    print(f"  ELO: {p1_feat['elo_before']:.0f}")
    print(f"  Surface ELO: {p1_feat['surface_elo_before']:.0f}")
    print(f"  Form (wins in last 10): {p1_feat['form_wins']}/10")
    print(f"  Win Streak: {p1_feat['win_streak']}")
    print(f"  Days since last match: {p1_feat['days_since_last']}")
    
    print(f"\n{player2}:")
    print(f"  ELO: {p2_feat['elo_before']:.0f}")
    print(f"  Surface ELO: {p2_feat['surface_elo_before']:.0f}")
    print(f"  Form (wins in last 10): {p2_feat['form_wins']}/10")
    print(f"  Win Streak: {p2_feat['win_streak']}")
    print(f"  Days since last match: {p2_feat['days_since_last']}")
    
    # Create feature vector
    print("\nGenerating features...")
    features = create_match_features(player1, player2, p1_feat, p2_feat, surface, tournament_level)
    X = pd.DataFrame([features])
    
    # Make prediction
    print("Making prediction...")
    prob = model.predict_proba(X)[0]
    prob_p1_wins = prob[1]
    prob_p2_wins = prob[0]
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"\n{player1} win probability: {prob_p1_wins:.1%}")
    print(f"{player2} win probability: {prob_p2_wins:.1%}")
    
    if prob_p1_wins > prob_p2_wins:
        confidence = prob_p1_wins
        winner = player1
    else:
        confidence = prob_p2_wins
        winner = player2
    
    print(f"\n✅ Predicted Winner: {winner}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Confidence interpretation
    print("\n" + "="*80)
    if confidence > 0.75:
        print("📊 Strong favorite")
    elif confidence > 0.65:
        print("📊 Moderate favorite")
    elif confidence > 0.55:
        print("📊 Slight favorite")
    else:
        print("📊 Very close match (toss-up)")
    print("="*80)

if __name__ == "__main__":
    main()
