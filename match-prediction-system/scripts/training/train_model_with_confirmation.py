#!/usr/bin/env python3
"""
Training script with parameter confirmation before starting.

This script will:
1. Load and prepare data
2. Engineer all features (ELO, form, H2H, etc.)
3. Show complete parameter list for confirmation
4. Train model with best parameters from old model
5. Evaluate and save results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
from datetime import datetime
from pathlib import Path


def load_data():
    """Load matches and players data"""
    print("="*70)
    print("📂 LOADING DATA")
    print("="*70)
    
    matches = pd.read_csv('data/processed/matches.csv', low_memory=False)
    players = pd.read_csv('data/processed/players.csv')
    
    print(f"✅ Loaded {len(matches):,} matches")
    print(f"✅ Loaded {len(players):,} players")
    print(f"   Date range: {matches['date'].min()} to {matches['date'].max()}")
    
    return matches, players


def engineer_features(matches):
    """Create all features for training"""
    print("\n" + "="*70)
    print("⚙️  ENGINEERING FEATURES")
    print("="*70)
    
    df = matches.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Target variable: 1 if player1 wins, 0 if player2 wins
    # We need to create player1/player2 format
    print("\n1. Creating player1/player2 format...")
    
    # Randomly assign winner as player1 or player2 to avoid bias
    np.random.seed(42)
    swap = np.random.rand(len(df)) > 0.5
    
    df['player1_name'] = np.where(swap, df['loser_name'], df['winner_name'])
    df['player2_name'] = np.where(swap, df['winner_name'], df['loser_name'])
    df['player1_wins'] = np.where(swap, 0, 1)  # Target
    
    # Swap all stats
    for col in df.columns:
        if col.startswith('winner_'):
            p2_col = 'player2_' + col.replace('winner_', '')
            p1_col = 'player1_' + col.replace('winner_', '')
            loser_col = 'loser_' + col.replace('winner_', '')
            
            if loser_col in df.columns:
                df[p1_col] = np.where(swap, df[loser_col], df[col])
                df[p2_col] = np.where(swap, df[col], df[loser_col])
        elif col.startswith('w_'):
            p1_col = 'player1_' + col.replace('w_', '')
            p2_col = 'player2_' + col.replace('w_', '')  # Fixed: was replace('l_', '')
            l_col = 'l_' + col.replace('w_', '')
            
            if l_col in df.columns:
                df[p1_col] = np.where(swap, df[l_col], df[col])
                df[p2_col] = np.where(swap, df[col], df[l_col])
    
    print(f"   ✅ Created player1/player2 format for {len(df)} matches")
    
    # Feature categories
    features = {}
    feature_count = 0
    
    # Sort by date to calculate form and fatigue chronologically
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. FORM & FATIGUE FEATURES (Calculate before other features)
    print("\n2. Calculating form and fatigue features...")
    
    # Initialize form tracking
    player_form = {}  # player_name -> {'win_streak': int, 'last_5': [results], 'last_match_date': date}
    
    form_features = []
    for idx, row in df.iterrows():
        p1_name = row['player1_name']
        p2_name = row['player2_name']
        match_date = row['date']
        
        # Initialize if first time seeing player
        if p1_name not in player_form:
            player_form[p1_name] = {'win_streak': 0, 'last_5': [], 'last_match_date': None, 'total_matches': 0}
        if p2_name not in player_form:
            player_form[p2_name] = {'win_streak': 0, 'last_5': [], 'last_match_date': None, 'total_matches': 0}
        
        # Get current form BEFORE this match
        p1_form = player_form[p1_name]
        p2_form = player_form[p2_name]
        
        # FORM FEATURES
        # Win streak (consecutive wins)
        p1_win_streak = p1_form['win_streak']
        p2_win_streak = p2_form['win_streak']
        
        # Recent form (wins in last 5 matches)
        p1_recent_wins = sum(p1_form['last_5'])
        p2_recent_wins = sum(p2_form['last_5'])
        
        # Form percentage (0-100)
        p1_form_pct = (p1_recent_wins / 5 * 100) if len(p1_form['last_5']) == 5 else 50.0
        p2_form_pct = (p2_recent_wins / 5 * 100) if len(p2_form['last_5']) == 5 else 50.0
        
        # FATIGUE FEATURES
        # Days since last match
        p1_days_since = (match_date - p1_form['last_match_date']).days if p1_form['last_match_date'] else 30
        p2_days_since = (match_date - p2_form['last_match_date']).days if p2_form['last_match_date'] else 30
        
        # Rest quality (0=tired <7 days, 1=optimal 7-21 days, 0.5=rusty >21 days)
        def rest_quality(days):
            if days < 7:
                return 0.0 + (days / 7) * 0.5  # 0.0 to 0.5 (tired)
            elif days <= 21:
                return 1.0  # optimal
            else:
                return 1.0 - min((days - 21) / 30, 0.5)  # 1.0 to 0.5 (rusty)
        
        p1_rest_quality = rest_quality(p1_days_since)
        p2_rest_quality = rest_quality(p2_days_since)
        
        # Store features for this match
        form_features.append({
            'player1_win_streak': p1_win_streak,
            'player2_win_streak': p2_win_streak,
            'win_streak_diff': p1_win_streak - p2_win_streak,
            'player1_recent_wins': p1_recent_wins,
            'player2_recent_wins': p2_recent_wins,
            'recent_wins_diff': p1_recent_wins - p2_recent_wins,
            'player1_form_pct': p1_form_pct,
            'player2_form_pct': p2_form_pct,
            'form_pct_diff': p1_form_pct - p2_form_pct,
            'player1_days_since_last': p1_days_since,
            'player2_days_since_last': p2_days_since,
            'days_since_diff': p1_days_since - p2_days_since,
            'player1_rest_quality': p1_rest_quality,
            'player2_rest_quality': p2_rest_quality,
            'rest_quality_diff': p1_rest_quality - p2_rest_quality,
        })
        
        # UPDATE form after this match (for next matches)
        p1_won = row['player1_wins']
        
        # Update win streaks
        if p1_won:
            player_form[p1_name]['win_streak'] += 1
            player_form[p2_name]['win_streak'] = 0
        else:
            player_form[p1_name]['win_streak'] = 0
            player_form[p2_name]['win_streak'] += 1
        
        # Update last 5 matches
        player_form[p1_name]['last_5'].append(1 if p1_won else 0)
        player_form[p2_name]['last_5'].append(0 if p1_won else 1)
        
        if len(player_form[p1_name]['last_5']) > 5:
            player_form[p1_name]['last_5'].pop(0)
        if len(player_form[p2_name]['last_5']) > 5:
            player_form[p2_name]['last_5'].pop(0)
        
        # Update last match date
        player_form[p1_name]['last_match_date'] = match_date
        player_form[p2_name]['last_match_date'] = match_date
        
        # Update total matches
        player_form[p1_name]['total_matches'] += 1
        player_form[p2_name]['total_matches'] += 1
    
    # Add form features to dataframe
    form_df = pd.DataFrame(form_features)
    df = pd.concat([df, form_df], axis=1)
    
    form_feature_names = list(form_df.columns)
    features['Form & Fatigue'] = form_feature_names
    feature_count += len(form_feature_names)
    print(f"   ✅ Added {len(form_feature_names)} form and fatigue features")
    print(f"      - Win streaks: player1/player2_win_streak, win_streak_diff")
    print(f"      - Recent form: last 5 matches win count and percentage")
    print(f"      - Fatigue: days since last match and rest quality")
    
    # 2. ELO FEATURES
    print("\n3. ELO features...")
    elo_features = [
        'player1_elo_before', 'player2_elo_before',
        'player1_surface_elo_before', 'player2_surface_elo_before',
    ]
    
    # ELO differences and ratios
    df['elo_diff'] = df['player1_elo_before'] - df['player2_elo_before']
    df['elo_ratio'] = df['player1_elo_before'] / (df['player2_elo_before'] + 1)
    df['surface_elo_diff'] = df['player1_surface_elo_before'] - df['player2_surface_elo_before']
    df['surface_elo_ratio'] = df['player1_surface_elo_before'] / (df['player2_surface_elo_before'] + 1)
    
    # Combined ELO (average of overall and surface)
    df['combined_elo_p1'] = (df['player1_elo_before'] + df['player1_surface_elo_before']) / 2
    df['combined_elo_p2'] = (df['player2_elo_before'] + df['player2_surface_elo_before']) / 2
    df['combined_elo_diff'] = df['combined_elo_p1'] - df['combined_elo_p2']
    
    # Win probabilities from ELO - RECALCULATE for player1 vs player2
    # Formula: 1 / (1 + 10^(-(elo_diff)/400))
    df['win_probability'] = 1 / (1 + 10**(-df['elo_diff']/400))
    df['surface_win_probability'] = 1 / (1 + 10**(-df['surface_elo_diff']/400))
    
    elo_features.extend([
        'elo_diff', 'elo_ratio', 'surface_elo_diff', 'surface_elo_ratio',
        'combined_elo_p1', 'combined_elo_p2', 'combined_elo_diff',
        'win_probability', 'surface_win_probability'
    ])
    
    features['ELO'] = elo_features
    feature_count += len(elo_features)
    print(f"   ✅ Added {len(elo_features)} ELO features")
    
    # 3. RANKING FEATURES
    print("\n4. Ranking features...")
    ranking_features = []
    
    if 'player1_rank' in df.columns and 'player2_rank' in df.columns:
        # Fill missing rankings with high value (unranked)
        df['player1_rank'] = df['player1_rank'].fillna(5000)
        df['player2_rank'] = df['player2_rank'].fillna(5000)
        
        df['rank_diff'] = df['player1_rank'] - df['player2_rank']  # Negative = p1 better
        df['rank_ratio'] = df['player2_rank'] / (df['player1_rank'] + 1)  # >1 = p1 better
        df['avg_rank'] = (df['player1_rank'] + df['player2_rank']) / 2
        
        ranking_features = ['player1_rank', 'player2_rank', 'rank_diff', 'rank_ratio', 'avg_rank']
        features['Ranking'] = ranking_features
        feature_count += len(ranking_features)
        print(f"   ✅ Added {len(ranking_features)} ranking features")
    
    # 4. SURFACE FEATURES
    print("\n5. Surface features...")
    surface_dummies = pd.get_dummies(df['surface'], prefix='surface')
    df = pd.concat([df, surface_dummies], axis=1)
    
    surface_features = list(surface_dummies.columns)
    features['Surface'] = surface_features
    feature_count += len(surface_features)
    print(f"   ✅ Added {len(surface_features)} surface features")
    
    # 5. TOURNAMENT FEATURES
    print("\n6. Tournament level features...")
    if 'tournament_level' in df.columns:
        # One-hot encode tournament level
        df['tournament_level'] = df['tournament_level'].fillna('ATP250')
        level_dummies = pd.get_dummies(df['tournament_level'], prefix='tournament')
        df = pd.concat([df, level_dummies], axis=1)
        
        tournament_features = list(level_dummies.columns)
        features['Tournament'] = tournament_features
        feature_count += len(tournament_features)
        print(f"   ✅ Added {len(tournament_features)} tournament features")
    
    # 6. PHYSICAL FEATURES
    print("\n7. Physical features (age, height, hand)...")
    physical_features = []
    
    if 'player1_age' in df.columns:
        df['player1_age'] = df['player1_age'].fillna(df['player1_age'].median())
        df['player2_age'] = df['player2_age'].fillna(df['player2_age'].median())
        df['age_diff'] = df['player1_age'] - df['player2_age']
        physical_features.extend(['player1_age', 'player2_age', 'age_diff'])
    
    if 'player1_height' in df.columns:
        df['player1_height'] = df['player1_height'].fillna(180)
        df['player2_height'] = df['player2_height'].fillna(180)
        df['height_diff'] = df['player1_height'] - df['player2_height']
        physical_features.extend(['player1_height', 'player2_height', 'height_diff'])
    
    if 'player1_hand' in df.columns:
        df['player1_hand'] = df['player1_hand'].fillna('R')
        df['player2_hand'] = df['player2_hand'].fillna('R')
        df['player1_lefty'] = (df['player1_hand'] == 'L').astype(int)
        df['player2_lefty'] = (df['player2_hand'] == 'L').astype(int)
        df['lefty_vs_righty'] = ((df['player1_lefty'] == 1) & (df['player2_lefty'] == 0)).astype(int)
        physical_features.extend(['player1_lefty', 'player2_lefty', 'lefty_vs_righty'])
    
    if physical_features:
        features['Physical'] = physical_features
        feature_count += len(physical_features)
        print(f"   ✅ Added {len(physical_features)} physical features")
    
    # 7. SERVICE STATISTICS - REMOVED (DATA LEAKAGE)
    # These statistics are FROM the match itself, not available before the match
    # Using them would be like predicting the winner after knowing the score!
    print("\n8. ⚠️  Skipping service statistics (data leakage - from actual match)")
    
    # 8. BETTING ODDS (when available)
    print("\n9. Betting odds features...")
    odds_features = []
    
    if 'player1_odds' in df.columns:
        # Only use if available
        has_odds = df['player1_odds'].notna()
        
        if has_odds.sum() > 0:
            # Implied probabilities
            df['implied_prob_p1'] = 1 / df['player1_odds'].where(df['player1_odds'] > 0, np.nan)
            df['implied_prob_p2'] = 1 / df['player2_odds'].where(df['player2_odds'] > 0, np.nan)
            df['odds_ratio'] = df['player1_odds'] / df['player2_odds'].where(df['player2_odds'] > 0, 1)
            
            # Fill missing with neutral values
            df['implied_prob_p1'] = df['implied_prob_p1'].fillna(0.5)
            df['implied_prob_p2'] = df['implied_prob_p2'].fillna(0.5)
            df['odds_ratio'] = df['odds_ratio'].fillna(1.0)
            
            odds_features = ['implied_prob_p1', 'implied_prob_p2', 'odds_ratio']
            features['Betting Odds'] = odds_features
            feature_count += len(odds_features)
            print(f"   ✅ Added {len(odds_features)} betting odds features ({has_odds.sum()} matches with odds)")
    
    print(f"\n✅ Total features engineered: {feature_count}")
    
    return df, features


def prepare_training_data(df, features):
    """Prepare X and y for training"""
    print("\n" + "="*70)
    print("📋 PREPARING TRAINING DATA")
    print("="*70)
    
    # Flatten feature list
    all_features = []
    for category, feat_list in features.items():
        all_features.extend(feat_list)
    
    # Remove duplicates
    all_features = list(set(all_features))
    
    # Filter to only features that exist
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"\n   Requested features: {len(all_features)}")
    print(f"   Available features: {len(available_features)}")
    
    # Remove rows with missing target
    df = df[df['player1_wins'].notna()].copy()
    
    # Create X and y
    X = df[available_features].copy()
    y = df['player1_wins'].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(X.median())
    
    print(f"\n   Training samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features, df


def show_parameters_for_confirmation(X, y, features, df):
    """Display all parameters and wait for confirmation"""
    print("\n" + "="*70)
    print("🎯 TRAINING PARAMETERS - PLEASE REVIEW")
    print("="*70)
    
    print("\n📊 DATASET INFORMATION:")
    print(f"   Total matches: {len(df):,}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Training samples: {len(X):,}")
    print(f"   Total features: {X.shape[1]}")
    print(f"   Target balance: Player1 wins: {y.sum():,} ({y.mean()*100:.1f}%), Player2 wins: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
    
    print("\n📈 FEATURE BREAKDOWN BY CATEGORY:")
    total_features = 0
    for category, feat_list in features.items():
        available = [f for f in feat_list if f in X.columns]
        print(f"   {category:20s}: {len(available):3d} features")
        total_features += len(available)
    
    print(f"\n   {'TOTAL':20s}: {total_features:3d} features")
    
    print("\n🔧 MODEL PARAMETERS (From old best model):")
    params = {
        'n_estimators': 700,
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    for param, value in params.items():
        print(f"   {param:20s}: {value}")
    
    print("\n📂 DATA SPLIT STRATEGY:")
    print(f"   Method: Temporal (chronological)")
    print(f"   Test size: 20% (most recent matches)")
    print(f"   Train period: 2000-2024")
    print(f"   Test period: 2024-2025 (recent matches)")
    
    print("\n🎯 EXPECTED PERFORMANCE (Based on old model):")
    print(f"   Old model accuracy: 64.87%")
    print(f"   Old model AUC: 0.7109")
    print(f"   Expected new accuracy: 67-72% (better data)")
    print(f"   Expected new AUC: 0.73-0.78")
    
    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Review all parameters above carefully!")
    print("="*70)
    
    print("\nComplete feature list (first 30):")
    for i, feat in enumerate(list(X.columns)[:30], 1):
        print(f"   {i:2d}. {feat}")
    if len(X.columns) > 30:
        print(f"   ... and {len(X.columns) - 30} more features")
    
    print("\n" + "="*70)
    response = input("\n✋ Type 'yes' to start training, or 'no' to cancel: ")
    print("="*70)
    
    if response.lower() != 'yes':
        print("\n❌ Training cancelled by user")
        return False, None
    
    return True, params


def train_model(X, y, params):
    """Train the model with temporal split"""
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    # Temporal split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n   Training set: {len(X_train):,} matches")
    print(f"   Test set: {len(X_test):,} matches")
    
    # Train model
    print(f"\n⚙️  Training XGBoost with {params['n_estimators']} estimators...")
    
    model = XGBClassifier(**params)
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ✅ Training completed in {training_time:.1f} seconds")
    
    # Predictions
    print(f"\n📊 Evaluating model...")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n🎯 RESULTS:")
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Test Accuracy:  {test_acc*100:.2f}%")
    print(f"   Train AUC:      {train_auc:.4f}")
    print(f"   Test AUC:       {test_auc:.4f}")
    print(f"   Overfitting:    {(train_acc - test_acc)*100:.2f}%")
    
    # Comparison to old model
    old_acc = 0.6487
    old_auc = 0.7109
    
    print(f"\n📈 COMPARISON TO OLD MODEL:")
    print(f"   Accuracy improvement: {(test_acc - old_acc)*100:+.2f}%")
    print(f"   AUC improvement:      {(test_auc - old_auc):+.4f}")
    
    if test_acc > old_acc:
        print(f"   ✅ NEW MODEL IS BETTER!")
    else:
        print(f"   ⚠️  Performance similar to old model")
    
    return model, {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'training_time': training_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }


def save_model_and_results(model, metrics, features, X):
    """Save model and results"""
    print("\n" + "="*70)
    print("💾 SAVING MODEL AND RESULTS")
    print("="*70)
    
    # Create output directory
    output_dir = Path('models/saved_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'xgboost_model_new.json'
    model.save_model(str(model_path))
    print(f"   ✅ Model saved to {model_path}")
    
    # Save metrics
    metrics['timestamp'] = datetime.now().isoformat()
    metrics_path = output_dir / 'training_metrics_new.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved to {metrics_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = output_dir / 'feature_importance_new.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved to {importance_path}")
    
    print(f"\n🏆 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)


def main():
    """Main training pipeline"""
    print("="*70)
    print("🎾 TENNIS MATCH PREDICTION - TRAINING PIPELINE")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    matches, players = load_data()
    
    # Engineer features
    df, features = engineer_features(matches)
    
    # Prepare data
    X, y, available_features, df = prepare_training_data(df, features)
    
    # Show parameters and get confirmation
    confirmed, params = show_parameters_for_confirmation(X, y, features, df)
    
    if not confirmed:
        return
    
    # Train model
    model, metrics = train_model(X, y, params)
    
    # Save everything
    save_model_and_results(model, metrics, features, X)
    
    print(f"\n🎉 All done! Model is ready for predictions.")


if __name__ == '__main__':
    main()
