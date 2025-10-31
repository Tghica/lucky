
#!/usr/bin/env python3
"""
Predict Challenger matches using the full XGBoost model (76.55% accuracy)
with simplified feature generation for quick predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

def load_resources():
    """Load model, players, and feature names"""
    print("Loading resources...")
    
    # Load tuned model
    model = xgb.Booster()
    model.load_model('models/saved_models/xgboost_tuned_model.json')
    
    # Load players
    players_df = pd.read_csv('data/processed/players.csv')
    
    # Load matches for form calculation  
    matches_df = pd.read_csv('data/processed/matches.csv', low_memory=False)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Get feature names from importance file
    feature_importance = pd.read_csv('models/saved_models/xgboost_tuned_feature_importance.csv')
    feature_names = feature_importance['feature'].tolist()
    
    print(f"✓ Model loaded (76.55% accuracy)")
    print(f"✓ {len(players_df)} players")
    print(f"✓ {len(matches_df)} matches")
    print(f"✓ {len(feature_names)} features required\n")
    
    return model, players_df, matches_df, feature_names


def find_player(name, players_df):
    """Find player with fuzzy matching"""
    # Exact match
    exact = players_df[players_df['player_name'].str.lower() == name.lower()]
    if len(exact) > 0:
        return exact.iloc[0]
    
    # Last name match
    last_name = name.split()[-1]
    partial = players_df[players_df['player_name'].str.contains(last_name, case=False, na=False)]
    if len(partial) > 0:
        return partial.iloc[0]
    
    return None


def get_player_recent_matches(player_name, matches_df, before_date, n=10):
    """Get player's recent match history"""
    before_dt = pd.to_datetime(before_date)
    
    # Find matches where player participated
    winner_matches = matches_df[
        (matches_df['winner_name'].str.lower() == player_name.lower()) &
        (matches_df['date'] < before_dt)
    ].copy()
    winner_matches['won'] = 1
    
    loser_matches = matches_df[
        (matches_df['loser_name'].str.lower() == player_name.lower()) &
        (matches_df['date'] < before_dt)
    ].copy()
    loser_matches['won'] = 0
    
    # Combine and sort
    all_matches = pd.concat([winner_matches, loser_matches]).sort_values('date', ascending=False)
    
    return all_matches.head(n)


def calculate_form_features(player_name, matches_df, prediction_date):
    """Calculate form and fatigue features"""
    recent = get_player_recent_matches(player_name, matches_df, prediction_date, n=10)
    
    if len(recent) == 0:
        # New player defaults
        return {
            'win_streak': 0,
            'recent_wins': 2,
            'form_pct': 0.5,
            'days_since_last': 14,
            'rest_quality': np.log1p(14)  # ~2.7 for 14 days
        }
    
    # Win streak (consecutive wins from most recent)
    win_streak = 0
    for _, match in recent.iterrows():
        if match['won'] == 1:
            win_streak += 1
        else:
            break
    
    # Recent wins (last 5 matches)
    last_5 = recent.head(5)
    recent_wins = int(last_5['won'].sum())
    form_pct = recent_wins / len(last_5) if len(last_5) > 0 else 0.5
    
    # Days since last match
    last_match_date = recent.iloc[0]['date']
    pred_date = pd.to_datetime(prediction_date)
    days_since = (pred_date - last_match_date).days
    
    # Rest quality - using log scale like feature engineering
    # np.log1p gives logarithmic diminishing returns (2->3 days matters more than 10->11)
    rest_quality = np.log1p(days_since)
    
    return {
        'win_streak': win_streak,
        'recent_wins': recent_wins,
        'form_pct': form_pct,
        'days_since_last': days_since,
        'rest_quality': rest_quality
    }


def create_features_from_data(p1_name, p2_name, p1_data, p2_data, p1_form, p2_form, 
                               surface='Hard', tournament_level='Challenger'):
    """Create all 58 features using available data"""
    
    features = {}
    
    # Get ELO ratings
    default_elo = 1500
    if p1_data is not None:
        p1_elo = p1_data['final_elo']
        p1_surface_elo = p1_data.get(f'final_elo_{surface.lower()}', p1_elo)
        p1_height = p1_data.get('height', 185)
        p1_hand = p1_data.get('hand', 'R')
    else:
        p1_elo = p1_surface_elo = default_elo
        p1_height = 185
        p1_hand = 'R'
    
    if p2_data is not None:
        p2_elo = p2_data['final_elo']
        p2_surface_elo = p2_data.get(f'final_elo_{surface.lower()}', p2_elo)
        p2_height = p2_data.get('height', 185)
        p2_hand = p2_data.get('hand', 'R')
    else:
        p2_elo = p2_surface_elo = default_elo
        p2_height = 185
        p2_hand = 'R'
    
    # === FORM & FATIGUE FEATURES (57% importance) ===
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
    
    # === ELO FEATURES (24% importance) ===
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
    features['win_probability'] = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    features['surface_win_probability'] = 1 / (1 + 10 ** ((p2_surface_elo - p1_surface_elo) / 400))
    
    # === RANKING FEATURES (11% importance) ===
    p1_rank = 300  # Default Challenger ranking
    p2_rank = 300
    features['player1_rank'] = p1_rank
    features['player2_rank'] = p2_rank
    features['rank_diff'] = abs(p1_rank - p2_rank)
    features['rank_ratio'] = p1_rank / p2_rank if p2_rank > 0 else 1.0
    features['avg_rank'] = (p1_rank + p2_rank) / 2
    
    # === PHYSICAL FEATURES (3% importance) ===
    features['player1_height'] = p1_height
    features['player2_height'] = p2_height
    features['height_diff'] = p1_height - p2_height
    features['player1_age'] = 25  # Default age
    features['player2_age'] = 25
    features['age_diff'] = 0
    p1_lefty = 1 if p1_hand == 'L' else 0
    p2_lefty = 1 if p2_hand == 'L' else 0
    features['player1_lefty'] = p1_lefty
    features['player2_lefty'] = p2_lefty
    features['lefty_vs_righty'] = 1 if p1_lefty != p2_lefty else 0
    
    # === SURFACE FEATURES (6% importance) ===
    features['surface_Hard'] = 1 if surface == 'Hard' else 0
    features['surface_Clay'] = 1 if surface == 'Clay' else 0
    features['surface_Grass'] = 1 if surface == 'Grass' else 0
    features['surface_Carpet'] = 1 if surface == 'Carpet' else 0
    
    # === TOURNAMENT FEATURES (4% importance) ===
    features['tournament_Grand Slam'] = 0
    features['tournament_Masters 1000'] = 0
    features['tournament_ATP500'] = 0
    features['tournament_ATP250'] = 0
    features['tournament_A'] = 0
    features['tournament_D'] = 0
    features['tournament_F'] = 0
    features['tournament_G'] = 0
    features['tournament_M'] = 0
    
    # === BETTING ODDS FEATURES (0% importance) ===
    features['implied_prob_p1'] = 0
    features['implied_prob_p2'] = 0
    features['odds_ratio'] = 0
    
    return features


def predict_matches():
    """Predict Paris Masters 1000 matches"""
    
    # Load resources
    model, players_df, matches_df, feature_names = load_resources()
    
    # Matches to predict - Paris Masters 1000 Last 16 (actual results)
    # Format: (winner, loser, actual_score)
    matches = [
        ('Jannik Sinner', 'Francisco Cerundolo', '7-5, 6-1'),
        ('Daniil Medvedev', 'Lorenzo Sonego', '3-6, 7-6 (5), 6-4'),
        ('Alex De Minaur', 'Karen Khachanov', '6-2, 6-2'),
        ('Alexander Bublik', 'Taylor Fritz', '7-6 (5), 6-2'),
        ('Ben Shelton', 'Andrey Rublev', '7-6 (6), 6-3'),
        ('Felix Auger Aliassime', 'Daniel Altmaier', '3-6, 6-3, 6-2'),
        ('Valentin Vacherot', 'Cameron Norrie', '7-6 (4), 6-4'),
    ]
    
    prediction_date = '2025-10-30'
    surface = 'Hard'
    
    print("=" * 100)
    print("🎾 FULL MODEL PREDICTIONS - Paris Masters 1000 (Last 16 Results)")
    print("Using XGBoost model with 58 features (76.55% test accuracy)")
    print("La Défense Arena, Hard Court, EUR 6,128,940")
    print("=" * 100)
    print()
    
    predictions = []
    
    for i, (actual_winner, actual_loser, score) in enumerate(matches, 1):
        # Test both directions to see model's prediction
        p1, p2 = actual_winner, actual_loser
        
        print(f"Match {i}: {p1} vs {p2} (Actual: {p1} won {score})")
        print("-" * 100)
        
        try:
            # Find players
            p1_data = find_player(p1, players_df)
            p2_data = find_player(p2, players_df)
            
            # Calculate form
            p1_form = calculate_form_features(p1, matches_df, prediction_date)
            p2_form = calculate_form_features(p2, matches_df, prediction_date)
            
            # Show data status
            if p1_data is not None:
                print(f"  ✓ {p1} - ELO: {p1_data['final_elo']:.0f}, Win streak: {p1_form['win_streak']}, Days rest: {p1_form['days_since_last']}")
            else:
                print(f"  ✗ {p1} - NOT in database (using defaults)")
            
            if p2_data is not None:
                print(f"  ✓ {p2} - ELO: {p2_data['final_elo']:.0f}, Win streak: {p2_form['win_streak']}, Days rest: {p2_form['days_since_last']}")
            else:
                print(f"  ✗ {p2} - NOT in database (using defaults)")
            
            # Create features
            features = create_features_from_data(p1, p2, p1_data, p2_data, p1_form, p2_form, surface)
            
            # Order features correctly (CRITICAL!)
            # Use the model's expected feature order (alphabetical)
            model_feature_names = model.feature_names
            feature_vector = [features[fname] for fname in model_feature_names]
            
            # Create DMatrix with correct feature names
            dmatrix = xgb.DMatrix(np.array(feature_vector).reshape(1, -1), feature_names=model_feature_names)
            
            # Predict
            prob_p1_wins = model.predict(dmatrix)[0]
            prob_p2_wins = 1 - prob_p1_wins
            
            winner = p1 if prob_p1_wins > 0.5 else p2
            win_prob = max(prob_p1_wins, prob_p2_wins)
            
            # Check if prediction was correct
            predicted_correctly = (winner == actual_winner)
            result_emoji = "✅" if predicted_correctly else "❌"
            
            # Confidence
            if win_prob > 0.70:
                conf = "HIGH 🔥"
            elif win_prob > 0.60:
                conf = "MEDIUM ⚡"
            else:
                conf = "LOW ⚠️"
            
            print(f"\n  PREDICTION: {p1}: {prob_p1_wins*100:.1f}% | {p2}: {prob_p2_wins*100:.1f}%")
            print(f"  PREDICTED WINNER: {winner} ({win_prob*100:.1f}%) - {conf}")
            print(f"  ACTUAL WINNER: {actual_winner} {score} {result_emoji}")
            print(f"  KEY: Rest quality diff: {features['rest_quality_diff']:.1f}, ELO diff: {features['combined_elo_diff']:.0f}")
            print()
            
            predictions.append({
                'p1': p1,
                'p2': p2,
                'actual_winner': actual_winner,
                'predicted_winner': winner,
                'correct': predicted_correctly,
                'prob_p1': prob_p1_wins,
                'prob_p2': prob_p2_wins,
                'prob': win_prob,
                'conf': conf,
                'elo_diff': features['combined_elo_diff'],
                'rest_diff': features['rest_quality_diff'],
                'form_diff': p1_form['recent_wins'] - p2_form['recent_wins'],
                'score': score
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    print("=" * 100)
    print("📊 PREDICTION RESULTS")
    print("=" * 100)
    print()
    
    correct = sum(1 for p in predictions if p['correct'])
    total = len(predictions)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)\n")
    
    for p in predictions:
        result_emoji = '✅' if p['correct'] else '❌'
        conf_emoji = '🔥' if 'HIGH' in p['conf'] else ('⚡' if 'MEDIUM' in p['conf'] else '⚠️')
        prob = p['prob_p1'] if p['predicted_winner'] == p['p1'] else p['prob_p2']
        print(f"{result_emoji} {p['predicted_winner']:30} ({prob*100:>5.1f}%) {conf_emoji} - Actual: {p['actual_winner']:30} {p['score']}")
        print(f"   ELO: {p['elo_diff']:>+4.0f} | Rest: {p['rest_diff']:>+4.1f}")
    
    print()
    high = len([p for p in predictions if 'HIGH' in p['conf']])
    medium = len([p for p in predictions if 'MEDIUM' in p['conf']])
    low = len([p for p in predictions if 'LOW' in p['conf']])
    print(f"Confidence breakdown: HIGH: {high} | MEDIUM: {medium} | LOW: {low}")
    print()


if __name__ == '__main__':
    predict_matches()
