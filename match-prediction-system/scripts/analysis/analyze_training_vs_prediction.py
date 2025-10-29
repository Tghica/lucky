#!/usr/bin/env python3
"""
Analyze the difference between training data and prediction data.
Shows how to handle missing features at prediction time.
"""

import pandas as pd
import numpy as np

def main():
    print('='*70)
    print('TRAINING vs PREDICTION DATA - MISSING FEATURE STRATEGY')
    print('='*70)
    
    matches = pd.read_csv('data/processed/matches.csv', low_memory=False)
    
    # Separate by source
    historical = matches[matches['data_source'] == 'jeff_sackmann']
    live_2025 = matches[matches['data_source'] == 'daily_update_2025']
    
    print('\n📊 DATA AVAILABILITY COMPARISON:')
    print('='*70)
    
    critical_fields = [
        ('winner_rank', 'Rankings'),
        ('loser_rank', 'Rankings'),
        ('winner_age', 'Age'),
        ('loser_age', 'Age'),
        ('winner_height', 'Physical'),
        ('loser_height', 'Physical'),
        ('winner_hand', 'Playing style'),
        ('loser_hand', 'Playing style'),
        ('w_ace', 'Service stats'),
        ('w_df', 'Service stats'),
        ('w_1stIn', 'Service stats'),
        ('winner_odds', 'Betting odds'),
        ('loser_odds', 'Betting odds'),
        ('surface', 'Match context'),
        ('round', 'Match context')
    ]
    
    print(f'\n{"Field":<20} {"Category":<15} {"Historical":<15} {"Live 2025":<15}')
    print('-'*70)
    
    for field, category in critical_fields:
        hist_coverage = (1 - historical[field].isnull().sum() / len(historical)) * 100
        live_coverage = (1 - live_2025[field].isnull().sum() / len(live_2025)) * 100
        
        hist_status = '✅' if hist_coverage > 90 else '⚠️' if hist_coverage > 50 else '❌'
        live_status = '✅' if live_coverage > 90 else '⚠️' if live_coverage > 50 else '❌'
        
        print(f'{field:<20} {category:<15} {hist_status} {hist_coverage:5.1f}%      {live_status} {live_coverage:5.1f}%')
    
    print('\n' + '='*70)
    print('PROBLEM STATEMENT:')
    print('='*70)
    print("""
Training Data (Jeff Sackmann 2000-2024):
  ✅ Has: Service stats (88%), Rankings (99%), Physical stats (95%)
  ❌ Missing: Betting odds (0%)

Prediction Data (Live matches 2025):
  ✅ Has: Betting odds (100%), Rankings (varies), Match context (100%)
  ❌ Missing: Service stats (0%), Player metadata (0%)
  
Challenge: How do we train on features that won't be available at prediction time?
          How do we use features at prediction time that weren't in training?
""")
    
    print('='*70)
    print('SOLUTION STRATEGIES:')
    print('='*70)
    
    print("""
Strategy 1: OPTIONAL FEATURES (Recommended)
─────────────────────────────────────────────────────
Make certain features optional and handle them gracefully:

A. Service Statistics (88% in training, 0% at prediction):
   Training: Use actual service stats when available
   Prediction: Substitute with player career averages
   
   Example:
   - Training: Use actual w_ace from match (e.g., 12 aces)
   - Prediction: Use player's career avg_aces (e.g., 8.5 aces)
   
B. Betting Odds (0% in training, 100% at prediction):
   Training: Train WITHOUT odds features initially
   Prediction: Add odds as BONUS features if available
   
   OR: Create synthetic odds for training using:
   - ELO-based win probability
   - Ranking difference
   - H2H records

Strategy 2: IMPUTATION (Fill missing values)
─────────────────────────────────────────────────────
Replace missing values with reasonable defaults:

A. Mean/Median Imputation:
   - Missing height → Use player average (180cm for men)
   - Missing age → Use median age (27 years)
   
B. Player-specific Imputation:
   - Missing w_ace → Use player's career average aces
   - Missing serve% → Use player's career serve%
   
C. Smart Defaults:
   - Missing surface → 'Hard' (most common)
   - Missing round → 'R32' (typical early round)

Strategy 3: FEATURE ENGINEERING (Make features robust)
─────────────────────────────────────────────────────
Create features that work with partial data:

A. ELO Ratings (Always available):
   - Calculate from match results only
   - No need for service stats or odds
   
B. Ranking-based Features (99% available):
   - ranking_diff = p1_rank - p2_rank
   - ranking_ratio = p1_rank / p2_rank
   
C. Historical Aggregates:
   - Player's last 10 matches on surface
   - Career win rate vs top-10 players
   - H2H record (from match history)

Strategy 4: TWO-STAGE MODELING (Advanced)
─────────────────────────────────────────────────────
Train multiple models for different scenarios:

Model A (Base): Uses only ALWAYS available features
  - ELO ratings
  - Rankings
  - Surface
  - Tournament level
  
Model B (Enhanced): Adds service stats when available
  - All Model A features
  - Service stats (actual or imputed)
  
Model C (Premium): Adds betting odds when available
  - All Model B features
  - Betting odds

Prediction logic:
  if odds_available and service_stats_available:
      use Model C
  elif service_stats_available:
      use Model B
  else:
      use Model A
""")
    
    print('='*70)
    print('RECOMMENDED IMPLEMENTATION:')
    print('='*70)
    
    print("""
Phase 1: Build Base Model (Week 1)
──────────────────────────────────
Features that are ALWAYS available:
  ✅ ELO ratings (calculated from all matches)
  ✅ Rankings (99% coverage, impute missing)
  ✅ Surface/tournament context (100%)
  ✅ Age/height (95%+, impute missing)
  ✅ H2H records (calculated from history)

Phase 2: Add Career Averages (Week 2)
──────────────────────────────────────
For each player, calculate career stats:
  - avg_aces_per_match
  - avg_1st_serve_pct
  - avg_break_points_saved
  
At prediction time:
  - Use actual stats if available (training)
  - Use career averages if missing (prediction)

Phase 3: Optional Odds Features (Week 3)
─────────────────────────────────────────
Create "implied probability" features:
  - Train model WITHOUT odds initially
  - Add odds as optional feature later
  - For prediction: use if available, skip if not

Implementation:
  def get_odds_features(match):
      if match.has_odds():
          return {
              'implied_prob_p1': 1 / odds_p1,
              'implied_prob_p2': 1 / odds_p2,
              'odds_ratio': odds_p1 / odds_p2
          }
      else:
          return {
              'implied_prob_p1': None,  # Will be filled by mean
              'implied_prob_p2': None,
              'odds_ratio': None
          }
""")
    
    print('='*70)
    print('PRACTICAL EXAMPLE:')
    print('='*70)
    
    print("""
Scenario: Predicting Djokovic vs Alcaraz (live match)

Available data:
  ✅ Player names: Djokovic, Alcaraz
  ✅ Surface: Hard
  ✅ Tournament: ATP Masters 1000
  ✅ Rankings: Djokovic #1, Alcaraz #2
  ✅ Betting odds: Djokovic 1.80, Alcaraz 2.10
  ❌ Service stats: Not available (match hasn't happened)
  ❌ Player IDs: Not in 2025 data
  ❌ Height: Not in 2025 data

Feature Generation:

1. ELO Features (calculated from history):
   - djokovic_elo = 2350 (from past matches)
   - alcaraz_elo = 2280
   - elo_diff = 70

2. Ranking Features:
   - rank_diff = 1 - 2 = -1
   - avg_rank = 1.5

3. Service Features (use career averages):
   - djokovic_avg_aces = 5.2 (from historical matches)
   - alcaraz_avg_aces = 7.8
   - ace_diff = -2.6

4. Odds Features:
   - implied_prob_djokovic = 1/1.80 = 55.6%
   - implied_prob_alcaraz = 1/2.10 = 47.6%
   - market_confidence = 55.6% (favorite probability)

5. H2H Features:
   - h2h_wins_djokovic = 3 (from match history)
   - h2h_wins_alcaraz = 2
   - h2h_win_pct = 60%

Result: 
  Model can make prediction using:
  - ELO (always available)
  - Rankings (always available)
  - Career avg service stats (imputed)
  - Odds (when available)
  - H2H (from history)
""")
    
    print('\n' + '='*70)
    print('KEY TAKEAWAYS:')
    print('='*70)
    print("""
✅ DO:
  1. Build features that work with incomplete data
  2. Use career averages for missing player stats
  3. Make betting odds OPTIONAL (bonus feature)
  4. Impute missing values intelligently
  5. Focus on ELO + Rankings as core features

❌ DON'T:
  1. Require features that aren't always available
  2. Drop matches with missing data (lose training examples)
  3. Assume all prediction scenarios have same features
  4. Ignore the training/prediction mismatch
  5. Over-rely on features with low coverage

🎯 Bottom Line:
  Train on historical data with service stats (88%)
  Predict on live data using career averages (100%)
  Add betting odds as optional calibration layer
""")

if __name__ == '__main__':
    main()
