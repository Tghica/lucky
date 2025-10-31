#!/usr/bin/env python3
"""
Simple Challenger match predictions using only ELO ratings
"""

import pandas as pd
import numpy as np
import xgboost as xgb

def load_model_and_data():
    """Load the trained model and player data"""
    print("Loading model and data...")
    
    # Load model
    model = xgb.Booster()
    model.load_model('models/saved_models/xgboost_tuned_model.json')
    
    # Load player data
    players_df = pd.read_csv('data/processed/players.csv')
    
    # Load feature importance to get feature order
    feature_importance = pd.read_csv('models/saved_models/xgboost_tuned_feature_importance.csv')
    feature_names = feature_importance['feature'].tolist()
    
    print(f"✓ Model loaded")
    print(f"✓ {len(players_df)} players in database")
    print(f"✓ {len(feature_names)} features required\n")
    
    return model, players_df, feature_names


def find_player(player_name, players_df):
    """Find player in database with fuzzy matching"""
    # Try exact match first
    exact_match = players_df[players_df['player_name'].str.lower() == player_name.lower()]
    if len(exact_match) > 0:
        return exact_match.iloc[0]
    
    # Try partial match on last name
    last_name = player_name.split()[-1]
    partial_match = players_df[players_df['player_name'].str.contains(last_name, case=False, na=False)]
    if len(partial_match) > 0:
        # Return the first match
        return partial_match.iloc[0]
    
    # Not found
    return None


def create_default_features(p1_elo, p2_elo, p1_surface_elo, p2_surface_elo, surface='Hard'):
    """Create features using only ELO ratings and defaults"""
    features = {}
    
    # === FORM & FATIGUE FEATURES (using defaults for Challenger) ===
    features['days_since_diff'] = 0  # Assume equal rest
    features['rest_quality_diff'] = 0  # Assume equal rest quality
    features['player1_rest_quality'] = 0.5  # Neutral rest
    features['player2_rest_quality'] = 0.5
    features['player1_days_since_last'] = 14  # ~2 weeks typical
    features['player2_days_since_last'] = 14
    features['player1_win_streak'] = 0
    features['player2_win_streak'] = 0
    features['win_streak_diff'] = 0
    features['player1_recent_wins'] = 2  # Neutral form
    features['player2_recent_wins'] = 2
    features['recent_wins_diff'] = 0
    features['player1_form_pct'] = 0.5
    features['player2_form_pct'] = 0.5
    features['form_pct_diff'] = 0
    
    # === ELO FEATURES (THE MAIN DIFFERENTIATOR) ===
    features['combined_elo_p1'] = p1_elo
    features['combined_elo_p2'] = p2_elo
    features['combined_elo_diff'] = p1_elo - p2_elo
    features['elo_diff'] = p1_elo - p2_elo
    features['elo_ratio'] = p1_elo / p2_elo if p2_elo > 0 else 1.0
    
    features['player1_elo_before'] = p1_elo
    features['player2_elo_before'] = p2_elo
    features['player1_surface_elo_before'] = p1_surface_elo
    features['player2_surface_elo_before'] = p2_surface_elo
    features['surface_elo_diff'] = p1_surface_elo - p2_surface_elo
    features['surface_elo_ratio'] = p1_surface_elo / p2_surface_elo if p2_surface_elo > 0 else 1.0
    
    # Win probability based on ELO
    features['win_probability'] = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    features['surface_win_probability'] = 1 / (1 + 10 ** ((p2_surface_elo - p1_surface_elo) / 400))
    
    # === RANKING FEATURES ===
    p1_rank = 300  # Typical Challenger ranking
    p2_rank = 300
    features['player1_rank'] = p1_rank
    features['player2_rank'] = p2_rank
    features['rank_diff'] = 0
    features['rank_ratio'] = 1.0
    features['avg_rank'] = 300
    
    # === PHYSICAL FEATURES ===
    features['player1_height'] = 185
    features['player2_height'] = 185
    features['height_diff'] = 0
    features['player1_age'] = 25
    features['player2_age'] = 25
    features['age_diff'] = 0
    features['player1_lefty'] = 0
    features['player2_lefty'] = 0
    features['lefty_vs_righty'] = 0
    
    # === SURFACE FEATURES ===
    features['surface_Hard'] = 1 if surface == 'Hard' else 0
    features['surface_Clay'] = 1 if surface == 'Clay' else 0
    features['surface_Grass'] = 1 if surface == 'Grass' else 0
    features['surface_Carpet'] = 1 if surface == 'Carpet' else 0
    
    # === TOURNAMENT FEATURES ===
    features['tournament_Grand Slam'] = 0
    features['tournament_Masters 1000'] = 0
    features['tournament_ATP500'] = 0
    features['tournament_ATP250'] = 0
    features['tournament_A'] = 0
    features['tournament_D'] = 0
    features['tournament_F'] = 0
    features['tournament_G'] = 0
    features['tournament_M'] = 0
    
    # === BETTING ODDS FEATURES (not used) ===
    features['implied_prob_p1'] = 0
    features['implied_prob_p2'] = 0
    features['odds_ratio'] = 0
    
    return features


def predict_matches():
    """Predict all challenger matches"""
    
    # Load model and data
    model, players_df, feature_names = load_model_and_data()
    
    # Today's matches
    matches = [
        ('Francesco Passaro', 'Thiago Agustin Tirante'),
        ('Dylan Dietrich', 'Jay Clarke'),
        ('Alex Molcan', 'Titouan Droguet'),
        ('Alexander Blockx', 'Billy Harris'),
        ('Daniil Glinka', 'Patrick Zahraj'),
        ('Gonzalo Villanueva', 'Alex Barrena'),
        ('Zdenek Kolar', 'Lautaro Midon'),
        ('Mackenzie McDonald', 'Chris Rodesch'),
        ('Mark Lajal', 'Norbert Gombos'),
        ('Inaki Montes-De La Torre', 'Oliver Tarvet'),
        ('Tristan Boyer', 'Nicolas Kicker'),
        ('Keegan Smith', 'Martin Damm'),
    ]
    
    surface = 'Hard'  # Assuming hard court for challengers
    
    print("=" * 100)
    print("🎾 TENNIS MATCH PREDICTIONS - Challenger (October 29, 2025)")
    print("=" * 100)
    print()
    
    predictions = []
    
    for i, (p1, p2) in enumerate(matches, 1):
        print(f"Match {i}: {p1} vs {p2}")
        print("-" * 100)
        
        try:
            # Find players
            p1_data = find_player(p1, players_df)
            p2_data = find_player(p2, players_df)
            
            # Get ELO ratings
            default_elo = 1500
            if p1_data is not None:
                p1_elo = p1_data['final_elo']
                p1_surface_elo = p1_data.get(f'final_elo_{surface.lower()}', p1_elo)
                p1_found = True
                print(f"  ✓ {p1} found - ELO: {p1_elo:.0f}, {surface} ELO: {p1_surface_elo:.0f}")
            else:
                p1_elo = default_elo
                p1_surface_elo = default_elo
                p1_found = False
                print(f"  ✗ {p1} NOT found - Using default ELO: {default_elo}")
            
            if p2_data is not None:
                p2_elo = p2_data['final_elo']
                p2_surface_elo = p2_data.get(f'final_elo_{surface.lower()}', p2_elo)
                p2_found = True
                print(f"  ✓ {p2} found - ELO: {p2_elo:.0f}, {surface} ELO: {p2_surface_elo:.0f}")
            else:
                p2_elo = default_elo
                p2_surface_elo = default_elo
                p2_found = False
                print(f"  ✗ {p2} NOT found - Using default ELO: {default_elo}")
            
            # Create features
            features = create_default_features(p1_elo, p2_elo, p1_surface_elo, p2_surface_elo, surface)
            
            # Order features correctly
            feature_values = [features.get(fname, 0) for fname in feature_names]
            
            # Create DMatrix
            dmatrix = xgb.DMatrix(np.array(feature_values).reshape(1, -1), feature_names=feature_names)
            
            # Make prediction
            prob_p1_wins = model.predict(dmatrix)[0]
            prob_p2_wins = 1 - prob_p1_wins
            
            # Determine winner
            predicted_winner = p1 if prob_p1_wins > 0.5 else p2
            confidence = max(prob_p1_wins, prob_p2_wins)
            
            # Confidence level
            if confidence > 0.70:
                conf_level = "HIGH"
                conf_emoji = "🔥"
            elif confidence > 0.60:
                conf_level = "MEDIUM"
                conf_emoji = "⚡"
            else:
                conf_level = "LOW"
                conf_emoji = "⚠️"
            
            print(f"\n  🎾 PREDICTION:")
            print(f"     {p1}: {prob_p1_wins*100:.1f}%")
            print(f"     {p2}: {prob_p2_wins*100:.1f}%")
            print(f"\n  {conf_emoji} PREDICTED WINNER: {predicted_winner} ({confidence*100:.1f}% - {conf_level} confidence)")
            
            # Data quality warning
            if not p1_found or not p2_found:
                print(f"\n  ⚠️  WARNING: Limited data - prediction based on defaults")
            
            predictions.append({
                'match': f"{p1} vs {p2}",
                'player1': p1,
                'player2': p2,
                'prob_p1': prob_p1_wins,
                'prob_p2': prob_p2_wins,
                'predicted_winner': predicted_winner,
                'confidence': conf_level,
                'confidence_pct': confidence * 100,
                'p1_in_db': p1_found,
                'p2_in_db': p2_found,
                'elo_diff': features['combined_elo_diff']
            })
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error predicting match: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary Table
    print("=" * 120)
    print("📊 PREDICTION SUMMARY")
    print("=" * 120)
    print()
    print(f"{'Match':<55} {'Winner':<25} {'Win Prob':<12} {'Confidence':<12} {'ELO Diff':<10}")
    print("-" * 120)
    
    for pred in predictions:
        match_short = f"{pred['player1'][:20]} vs {pred['player2'][:20]}"
        winner_prob = pred['prob_p1'] if pred['predicted_winner'] == pred['player1'] else pred['prob_p2']
        
        # Add emoji based on confidence
        if pred['confidence'] == 'HIGH':
            emoji = '🔥'
        elif pred['confidence'] == 'MEDIUM':
            emoji = '⚡'
        else:
            emoji = '⚠️'
        
        print(f"{match_short:<55} {pred['predicted_winner']:<25} {winner_prob*100:>5.1f}%  {emoji:>5}  {pred['confidence']:<12} {pred['elo_diff']:>+6.0f}")
    
    print()
    print(f"✓ Total predictions: {len(predictions)}")
    high_conf = sum(1 for p in predictions if p['confidence'] == 'HIGH')
    medium_conf = sum(1 for p in predictions if p['confidence'] == 'MEDIUM')
    low_conf = sum(1 for p in predictions if p['confidence'] == 'LOW')
    print(f"✓ High confidence: {high_conf} | Medium confidence: {medium_conf} | Low confidence: {low_conf}")
    
    data_available = sum(1 for p in predictions if p['p1_in_db'] and p['p2_in_db'])
    print(f"✓ Matches with complete data: {data_available}/{len(predictions)}")
    print()
    
    # Best bets
    print("=" * 120)
    print("💰 RECOMMENDED BETS (High Confidence)")
    print("=" * 120)
    high_conf_matches = [p for p in predictions if p['confidence'] == 'HIGH']
    if high_conf_matches:
        for pred in high_conf_matches:
            winner_prob = pred['prob_p1'] if pred['predicted_winner'] == pred['player1'] else pred['prob_p2']
            print(f"  🔥 {pred['predicted_winner']} to beat {pred['player1'] if pred['predicted_winner'] == pred['player2'] else pred['player2']}")
            print(f"     Win probability: {winner_prob*100:.1f}%")
            print(f"     ELO advantage: {abs(pred['elo_diff']):.0f} points")
            print()
    else:
        print("  No high confidence predictions available")
        print()

if __name__ == '__main__':
    predict_matches()
