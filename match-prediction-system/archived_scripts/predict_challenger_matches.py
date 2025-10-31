#!/usr/bin/env python3
"""
Predict Challenger matches for October 29, 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import json

def load_model_and_data():
    """Load the trained model and player data"""
    print("Loading model and data...")
    
    # Load model
    model = xgb.Booster()
    model.load_model('models/saved_models/xgboost_tuned_model.json')
    
    # Load player data
    players_df = pd.read_csv('data/processed/players.csv')
    matches_df = pd.read_csv('data/processed/matches.csv')
    
    # Load feature importance to get feature order
    feature_importance = pd.read_csv('models/saved_models/xgboost_tuned_feature_importance.csv')
    feature_names = feature_importance['feature'].tolist()
    
    print(f"✓ Model loaded")
    print(f"✓ {len(players_df)} players in database")
    print(f"✓ {len(matches_df)} historical matches")
    print(f"✓ {len(feature_names)} features required")
    
    return model, players_df, matches_df, feature_names


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
        return partial_match.iloc[0]
    
    # Not found
    return None


def get_recent_matches(player_name, matches_df, before_date, n=5):
    """Get player's recent matches before a given date"""
    # Convert date to datetime
    before_dt = pd.to_datetime(before_date)
    matches_df['tourney_date'] = pd.to_datetime(matches_df['tourney_date'])
    
    # Find matches where player participated
    player_matches = matches_df[
        ((matches_df['winner_name'].str.lower() == player_name.lower()) | 
         (matches_df['loser_name'].str.lower() == player_name.lower())) &
        (matches_df['tourney_date'] < before_dt)
    ].sort_values('tourney_date', ascending=False).head(n)
    
    return player_matches


def calculate_form_features(player_name, matches_df, prediction_date):
    """Calculate form and fatigue features for a player"""
    recent_matches = get_recent_matches(player_name, matches_df, prediction_date, n=10)
    
    if len(recent_matches) == 0:
        # New player - return neutral defaults
        return {
            'win_streak': 0,
            'recent_wins': 0,
            'form_pct': 0.5,
            'days_since_last': 30,
            'rest_quality': 0.5
        }
    
    # Calculate win streak
    win_streak = 0
    for _, match in recent_matches.iterrows():
        if match['winner_name'].lower() == player_name.lower():
            win_streak += 1
        else:
            break
    
    # Recent wins (last 5)
    last_5 = recent_matches.head(5)
    recent_wins = len(last_5[last_5['winner_name'].str.lower() == player_name.lower()])
    form_pct = recent_wins / len(last_5) if len(last_5) > 0 else 0.5
    
    # Days since last match
    last_match_date = pd.to_datetime(recent_matches.iloc[0]['tourney_date'])
    pred_date = pd.to_datetime(prediction_date)
    days_since_last = (pred_date - last_match_date).days
    
    # Rest quality: 0 if <7 days (tired), 1 if 7-21 days (optimal), 0.5 if >21 days (rusty)
    if days_since_last < 7:
        rest_quality = 0
    elif days_since_last <= 21:
        rest_quality = 1
    else:
        rest_quality = 0.5
    
    return {
        'win_streak': win_streak,
        'recent_wins': recent_wins,
        'form_pct': form_pct,
        'days_since_last': days_since_last,
        'rest_quality': rest_quality
    }


def create_match_features(player1_name, player2_name, surface, tournament_level, prediction_date, 
                         players_df, matches_df):
    """Create all 58 features needed for prediction"""
    
    # Find players
    p1 = find_player(player1_name, players_df)
    p2 = find_player(player2_name, players_df)
    
    # Get form features
    p1_form = calculate_form_features(player1_name, matches_df, prediction_date)
    p2_form = calculate_form_features(player2_name, matches_df, prediction_date)
    
    # Initialize features dictionary
    features = {}
    
    # === FORM & FATIGUE FEATURES (MOST IMPORTANT!) ===
    features['days_since_diff'] = p1_form['days_since_last'] - p2_form['days_since_last']
    features['rest_quality_diff'] = p1_form['rest_quality'] - p2_form['rest_quality']
    features['player1_rest_quality'] = p1_form['rest_quality']
    features['player2_rest_quality'] = p2_form['rest_quality']
    features['player1_days_since_last'] = p1_form['days_since_last']
    features['player2_days_since_last'] = p2_form['days_since_last']
    features['player1_win_streak'] = p1_form['win_streak']
    features['player2_win_streak'] = p2_form['win_streak']
    features['win_streak_diff'] = p1_form['win_streak'] - p2_form['win_streak']
    features['player1_recent_wins'] = p1_form['recent_wins']
    features['player2_recent_wins'] = p2_form['recent_wins']
    features['recent_wins_diff'] = p1_form['recent_wins'] - p2_form['recent_wins']
    features['player1_form_pct'] = p1_form['form_pct']
    features['player2_form_pct'] = p2_form['form_pct']
    features['form_pct_diff'] = p1_form['form_pct'] - p2_form['form_pct']
    
    # === ELO FEATURES ===
    if p1 is not None and p2 is not None:
        # Overall ELO
        p1_elo = p1['final_elo']
        p2_elo = p2['final_elo']
        
        # Surface-specific ELO
        surface_col = f'final_elo_{surface.lower()}'
        p1_surface_elo = p1.get(surface_col, p1_elo)
        p2_surface_elo = p2.get(surface_col, p2_elo)
        
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
    else:
        # Default ELO for unknown players
        default_elo = 1500
        features['combined_elo_p1'] = default_elo
        features['combined_elo_p2'] = default_elo
        features['combined_elo_diff'] = 0
        features['elo_diff'] = 0
        features['elo_ratio'] = 1.0
        features['player1_elo_before'] = default_elo
        features['player2_elo_before'] = default_elo
        features['player1_surface_elo_before'] = default_elo
        features['player2_surface_elo_before'] = default_elo
        features['surface_elo_diff'] = 0
        features['surface_elo_ratio'] = 1.0
        features['win_probability'] = 0.5
        features['surface_win_probability'] = 0.5
    
    # === RANKING FEATURES ===
    # Note: We don't have current rankings, so we'll use defaults
    p1_rank = 500  # Default for challenger players
    p2_rank = 500
    features['player1_rank'] = p1_rank
    features['player2_rank'] = p2_rank
    features['rank_diff'] = abs(p1_rank - p2_rank)
    features['rank_ratio'] = p1_rank / p2_rank if p2_rank > 0 else 1.0
    features['avg_rank'] = (p1_rank + p2_rank) / 2
    
    # === PHYSICAL FEATURES ===
    if p1 is not None and p2 is not None:
        p1_height = p1.get('height', 185)
        p2_height = p2.get('height', 185)
        p1_age = 25  # Default age if not available
        p2_age = 25
        
        features['player1_height'] = p1_height
        features['player2_height'] = p2_height
        features['height_diff'] = p1_height - p2_height
        features['player1_age'] = p1_age
        features['player2_age'] = p2_age
        features['age_diff'] = p1_age - p2_age
        
        p1_lefty = 1 if p1.get('hand', 'R') == 'L' else 0
        p2_lefty = 1 if p2.get('hand', 'R') == 'L' else 0
        features['player1_lefty'] = p1_lefty
        features['player2_lefty'] = p2_lefty
        features['lefty_vs_righty'] = 1 if p1_lefty != p2_lefty else 0
    else:
        features['player1_height'] = 185
        features['player2_height'] = 185
        features['height_diff'] = 0
        features['player1_age'] = 25
        features['player2_age'] = 25
        features['age_diff'] = 0
        features['player1_lefty'] = 0
        features['player2_lefty'] = 0
        features['lefty_vs_righty'] = 0
    
    # === SURFACE FEATURES (one-hot encoding) ===
    features['surface_Hard'] = 1 if surface == 'Hard' else 0
    features['surface_Clay'] = 1 if surface == 'Clay' else 0
    features['surface_Grass'] = 1 if surface == 'Grass' else 0
    features['surface_Carpet'] = 1 if surface == 'Carpet' else 0
    
    # === TOURNAMENT FEATURES ===
    features['tournament_Grand Slam'] = 1 if tournament_level == 'Grand Slam' else 0
    features['tournament_Masters 1000'] = 1 if tournament_level == 'Masters 1000' else 0
    features['tournament_ATP500'] = 1 if tournament_level == 'ATP500' else 0
    features['tournament_ATP250'] = 1 if tournament_level == 'ATP250' else 0
    features['tournament_A'] = 0  # Challenger-specific
    features['tournament_D'] = 0
    features['tournament_F'] = 0
    features['tournament_G'] = 0
    features['tournament_M'] = 0
    
    # === BETTING ODDS FEATURES (not used by model but required) ===
    features['implied_prob_p1'] = 0
    features['implied_prob_p2'] = 0
    features['odds_ratio'] = 0
    
    return features


def predict_matches():
    """Predict all challenger matches"""
    
    # Load model and data
    model, players_df, matches_df, feature_names = load_model_and_data()
    
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
    
    prediction_date = '2025-10-29'
    surface = 'Hard'  # Assuming hard court for challengers
    tournament_level = 'Challenger'
    
    print("\n" + "=" * 100)
    print("TENNIS MATCH PREDICTIONS - Challenger (October 29, 2025)")
    print("=" * 100)
    print()
    
    predictions = []
    
    for i, (p1, p2) in enumerate(matches, 1):
        print(f"\n{'=' * 100}")
        print(f"Match {i}: {p1} vs {p2}")
        print('=' * 100)
        
        try:
            # Create features
            features = create_match_features(
                p1, p2, surface, tournament_level, prediction_date,
                players_df, matches_df
            )
            
            # Check if players are in database
            p1_found = find_player(p1, players_df)
            p2_found = find_player(p2, players_df)
            
            if p1_found is not None:
                print(f"✓ {p1} found in database (ELO: {p1_found['final_elo']:.0f})")
            else:
                print(f"✗ {p1} NOT in database (using defaults)")
            
            if p2_found is not None:
                print(f"✓ {p2} found in database (ELO: {p2_found['final_elo']:.0f})")
            else:
                print(f"✗ {p2} NOT in database (using defaults)")
            
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
            elif confidence > 0.60:
                conf_level = "MEDIUM"
            else:
                conf_level = "LOW"
            
            print(f"\n🎾 PREDICTION:")
            print(f"   {p1}: {prob_p1_wins*100:.1f}%")
            print(f"   {p2}: {prob_p2_wins*100:.1f}%")
            print(f"\n🏆 PREDICTED WINNER: {predicted_winner} ({confidence*100:.1f}% confidence - {conf_level})")
            
            # Key factors
            print(f"\n📊 KEY FACTORS:")
            print(f"   • ELO Difference: {features['combined_elo_diff']:.0f}")
            print(f"   • Rest Quality: {p1}: {features['player1_rest_quality']:.1f} vs {p2}: {features['player2_rest_quality']:.1f}")
            print(f"   • Days Since Last Match: {features['days_since_diff']:.0f} days difference")
            print(f"   • Win Streak: {features['player1_win_streak']} vs {features['player2_win_streak']}")
            
            predictions.append({
                'match': f"{p1} vs {p2}",
                'predicted_winner': predicted_winner,
                'prob_p1': prob_p1_wins,
                'prob_p2': prob_p2_wins,
                'confidence': conf_level,
                'p1_in_db': p1_found is not None,
                'p2_in_db': p2_found is not None
            })
            
        except Exception as e:
            print(f"❌ Error predicting match: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 100)
    print("PREDICTION SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Match':<60} {'Winner':<25} {'Probability':<15} {'Confidence':<10}")
    print("-" * 110)
    
    for pred in predictions:
        winner = pred['predicted_winner']
        prob = pred['prob_p1'] if winner == pred['match'].split(' vs ')[0] else pred['prob_p2']
        print(f"{pred['match']:<60} {winner:<25} {prob*100:>6.1f}% {'':<8} {pred['confidence']:<10}")
    
    print()
    print(f"✓ Total predictions: {len(predictions)}")
    high_conf = sum(1 for p in predictions if p['confidence'] == 'HIGH')
    print(f"✓ High confidence predictions: {high_conf}")
    print()

if __name__ == '__main__':
    predict_matches()
