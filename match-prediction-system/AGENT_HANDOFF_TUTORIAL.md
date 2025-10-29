# 🎾 Agent Handoff Tutorial - Prediction System Implementation

**Date**: October 29, 2025  
**Current Status**: Model trained and optimized, ready for prediction implementation  
**Your Mission**: Build the prediction system using the tuned model

---

## 📊 Project Overview

This is a **tennis match prediction system** that achieved **76.55% accuracy** (up from 64.87% in V1).

### Current Performance Metrics
- **Test Accuracy**: 76.55%
- **Test AUC**: 0.8668
- **Features Used**: 58 pre-match features
- **Model**: XGBoost (hyperparameter-tuned with Optuna)
- **Training Data**: 75,605 clean ATP matches (2000-2025)

---

## 🗂️ File Structure You Need to Know

```
match-prediction-system/
├── models/saved_models/
│   ├── xgboost_tuned_model.json          # ✅ BEST MODEL (76.55% accuracy)
│   ├── xgboost_tuned_params.json         # Optimal hyperparameters
│   ├── xgboost_tuned_feature_importance.csv  # Feature rankings
│   └── feature_importance_new.csv        # Detailed feature analysis
│
├── data/processed/
│   ├── matches.csv                       # 75,605 matches with all features
│   ├── players.csv                       # 2,894 players with ELO ratings
│   ├── stadiums.csv                      # 2,010 stadiums (not used yet)
│   └── tournament_surface_mapping.json   # Tournament → surface mapping
│
├── scripts/prediction/
│   ├── predict_match.py                  # OLD prediction script (needs update)
│   └── predict_simple.py                 # Simple prediction script
│
├── src/predictor/
│   ├── prediction_service.py             # Prediction service class
│   └── match_analyzer.py                 # Match analysis utilities
│
└── calculate_elos.py                     # ELO calculation script (important!)
```

---

## 🎯 What You Need to Build

### **Primary Goal**: Create a prediction system that can:

1. **Take match inputs** (player names, tournament, surface, date)
2. **Calculate all 58 features** the model needs
3. **Load the tuned model** and make predictions
4. **Return probabilities** and confidence scores
5. **Handle edge cases** (new players, missing data, etc.)

---

## 🧠 Understanding the Model

### **The 58 Features (in order of importance)**

The model uses 6 categories of features:

#### **1. Form & Fatigue (57% importance) - MOST CRITICAL!**
- `days_since_diff` - Days since last match difference (23.5% importance!)
- `rest_quality_diff` - Rest quality difference (0=tired, 1=optimal, 0.5=rusty)
- `win_streak_diff` - Win streak difference
- `recent_form_*` - Last 5 matches performance
- `player1/2_days_since_last_match` - Individual fatigue metrics
- `player1/2_rest_quality` - Individual rest quality
- `player1/2_win_streak` - Individual win streaks

**How to calculate**: See `scripts/training/train_model_with_confirmation.py` lines 50-120

#### **2. ELO Ratings (24% importance)**
- `combined_elo_diff` - Overall ELO difference (12.4% importance)
- `surface_elo_diff` - Surface-specific ELO difference
- `player1/2_elo` - Individual overall ELOs
- `player1/2_surface_elo` - Individual surface ELOs (Hard/Clay/Grass/Carpet)

**How to calculate**: Use `calculate_elos.py` - processes matches chronologically

#### **3. Rankings (11% importance)**
- `rank_ratio` - player1_rank / player2_rank (7.1% importance)
- `rank_diff` - Absolute rank difference
- `player1/2_rank` - Individual ATP rankings
- `player1/2_rank_points` - Individual ranking points

#### **4. Surface Features (6% importance)**
- `surface_*` - One-hot encoded (Hard/Clay/Grass/Carpet)
- Used to select correct surface_elo

#### **5. Tournament Features (4% importance)**
- `level_*` - Tournament level (Grand Slam, Masters, ATP250, etc.)
- `best_of` - 3 or 5 sets

#### **6. Physical Features (3% importance)**
- `age_diff`, `height_diff`, `hand_advantage`
- `player1/2_age`, `player1/2_height`, `player1/2_hand`

**Note**: Betting odds features exist but have 0% importance (only 3.1% data coverage)

---

## 🔧 Key Technical Details

### **Data Processing Pipeline**

1. **Load player data** from `players.csv`
   - Contains: name, hand, height, birthdate, **ELO ratings** (overall + 4 surfaces)
   - ELO ratings are up-to-date as of last match processed

2. **Calculate match-specific features**
   - Form & Fatigue: Need player's **recent match history**
   - Rankings: From ATP rankings on match date
   - Surface: From tournament_surface_mapping.json
   - Physical: Calculate from player data

3. **Feature Engineering Order** (CRITICAL!)
   - Features must be calculated in **chronological order**
   - Form/fatigue depends on **previous matches only** (no future data!)
   - ELO must be **current as of prediction date**, not final ELO

### **Current ELO System**

Located in `calculate_elos.py`:
```python
# Initial ELO: 1500
# K-factor: 32
# Updates after EACH match chronologically
# Tracks: overall_elo, hard_elo, clay_elo, grass_elo, carpet_elo
```

**Important**: For predictions, you need the **current ELO** (before the match), not the final ELO in `players.csv`!

### **Form & Fatigue Calculation**

From `scripts/training/train_model_with_confirmation.py`:

```python
# For each player, track chronologically:
1. win_streak: consecutive wins (resets on loss)
2. recent_form_last_5: wins in last 5 matches
3. recent_form_pct_last_5: win percentage in last 5
4. days_since_last_match: days between matches
5. rest_quality: 0 if <7 days, 1 if 7-21 days, 0.5 if >21 days
```

**Critical**: These MUST be calculated from matches **before** the prediction date!

---

## 🚀 Recommended Implementation Steps

### **Step 1: Update the Prediction Service**

File: `src/predictor/prediction_service.py`

```python
class PredictionService:
    def __init__(self):
        # Load the tuned model
        self.model = self._load_model('models/saved_models/xgboost_tuned_model.json')
        self.players_df = pd.read_csv('data/processed/players.csv')
        self.matches_df = pd.read_csv('data/processed/matches.csv')
        
    def predict_match(self, player1_name, player2_name, tournament, 
                      surface, date, best_of=3):
        """
        Predict match outcome
        
        Returns:
        --------
        {
            'player1_win_probability': float,
            'player2_win_probability': float,
            'confidence': str,  # High/Medium/Low
            'features_used': dict,
            'key_factors': list
        }
        """
        # 1. Get player data
        # 2. Calculate all 58 features
        # 3. Make prediction
        # 4. Return results with explanation
```

### **Step 2: Build Feature Calculator**

Create: `src/predictor/feature_calculator.py`

```python
class FeatureCalculator:
    def calculate_form_fatigue(self, player_name, prediction_date):
        """Calculate form & fatigue features for a player"""
        # Get player's matches before prediction_date
        # Calculate win_streak, recent_form, days_since, rest_quality
        
    def calculate_elo_features(self, player1, player2, surface):
        """Calculate ELO-based features"""
        # Get current ELO (not final ELO!)
        # Calculate diff features
        
    def calculate_all_features(self, player1, player2, tournament, 
                               surface, date, best_of):
        """Calculate all 58 features for prediction"""
        # Returns: pandas DataFrame with 1 row, 58 columns
```

### **Step 3: Handle Edge Cases**

```python
def handle_new_player(self, player_name):
    """
    If player not in database:
    1. Use default ELO = 1500
    2. Use average physical stats (age=25, height=185, etc.)
    3. Set form features to neutral (win_streak=0, recent_form=0.5)
    4. Log warning about reduced prediction confidence
    """

def handle_missing_ranking(self, player_name, date):
    """
    If ranking not available for date:
    1. Use last known ranking
    2. If never ranked, use rank=500, rank_points=0
    3. Reduce confidence score
    """
```

### **Step 4: Add Prediction Explanation**

```python
def explain_prediction(self, features, prediction):
    """
    Generate human-readable explanation:
    
    Example output:
    - "Player 1 favored 65% due to:"
    - "✓ Better rest (3 days vs 1 day) - HIGH IMPACT"
    - "✓ Higher ELO (2100 vs 1950) - MEDIUM IMPACT"
    - "✓ On 5-match win streak - MEDIUM IMPACT"
    - "✗ Lower ranking (#15 vs #8) - LOW IMPACT"
    """
```

---

## 🔍 Testing Your Implementation

### **Test Cases to Implement**

1. **Known Historical Match**
   - Input: Djokovic vs Nadal, French Open 2020, Clay
   - Expected: Should match actual outcome with reasonable probability

2. **New Players**
   - Input: Two unranked players
   - Expected: ~50% probability, low confidence

3. **Extreme Fatigue**
   - Input: Player 1 last match yesterday, Player 2 rested 10 days
   - Expected: Fatigue should significantly impact prediction

4. **Surface Specialist**
   - Input: Clay specialist on clay vs Hard court specialist
   - Expected: Surface ELO should favor clay specialist

---

## ⚠️ Common Pitfalls to Avoid

### **1. Data Leakage (CRITICAL!)**
❌ **DON'T**: Use final ELO from `players.csv`  
✅ **DO**: Calculate ELO as of prediction date (before match)

❌ **DON'T**: Include future matches in form calculation  
✅ **DO**: Only use matches before prediction date

❌ **DON'T**: Use match outcome statistics (service stats, etc.)  
✅ **DO**: Only use pre-match features

### **2. Feature Naming**
The model expects **exact** feature names. Check:
```python
# Load feature importance to see exact names
feature_importance = pd.read_csv('models/saved_models/xgboost_tuned_feature_importance.csv')
print(feature_importance['feature'].tolist())
```

### **3. Missing Data**
- **Rankings**: ~15% of matches missing (use last known or default)
- **Physical stats**: Some players missing height (use average 185cm)
- **Betting odds**: 96.9% missing (model ignores them anyway)

### **4. Feature Order**
Features must be in the **same order** as training data. Use:
```python
FEATURE_ORDER = [
    'days_since_diff', 'combined_elo_diff', 'rest_quality_diff',
    # ... (see feature_importance_new.csv for full order)
]
```

---

## 📚 Key Files to Reference

### **For Understanding the Model**
1. `scripts/training/train_model_with_confirmation.py` - Feature engineering logic
2. `models/saved_models/xgboost_tuned_feature_importance.csv` - Feature importance
3. `models/saved_models/xgboost_tuned_params.json` - Model parameters

### **For Data Access**
1. `data/processed/matches.csv` - Historical matches with features
2. `data/processed/players.csv` - Player info + final ELO ratings
3. `data/processed/tournament_surface_mapping.json` - Tournament surfaces

### **For Feature Calculation Examples**
1. `calculate_elos.py` - ELO calculation (lines 100-250)
2. `scripts/training/train_model_with_confirmation.py` - Form/fatigue (lines 50-120)

---

## 🎯 Suggested First Steps

1. **Load and test the model**
   ```python
   import xgboost as xgb
   model = xgb.Booster()
   model.load_model('models/saved_models/xgboost_tuned_model.json')
   print("Model loaded successfully!")
   ```

2. **Verify feature names**
   ```python
   import pandas as pd
   df = pd.read_csv('data/processed/matches.csv')
   print("Available features:", df.columns.tolist())
   ```

3. **Test with one historical match**
   - Pick a match from `matches.csv`
   - Extract its 58 features
   - Run prediction
   - Compare with actual outcome

4. **Build incrementally**
   - Start with ELO features only (simpler)
   - Add rankings
   - Add form/fatigue (most complex but most important!)
   - Add physical + surface + tournament

---

## 💡 Pro Tips

1. **Use the training script as reference**: `scripts/training/train_model_with_confirmation.py` has the **exact** feature engineering logic that works.

2. **Form & Fatigue is king**: With 57% importance, getting this right is crucial. Test it thoroughly!

3. **ELO must be dynamic**: Don't use static ELO from `players.csv`. Calculate it up to prediction date.

4. **Test edge cases early**: New players, missing data, etc. These will happen in production!

5. **Validate against training data**: Pick 100 random matches from `matches.csv`, run predictions, check if accuracy is ~76-77%.

---

## 📞 Where to Get Help

- **Feature engineering logic**: `scripts/training/train_model_with_confirmation.py`
- **Model details**: `models/saved_models/xgboost_tuning_summary.json`
- **Data schema**: Check first few rows of CSV files
- **ELO calculation**: `calculate_elos.py` with detailed comments

---

## ✅ Success Criteria

Your prediction system is ready when:

1. ✅ Can predict any match with player names, tournament, surface, date
2. ✅ Returns probabilities + confidence + explanation
3. ✅ Handles new players gracefully
4. ✅ Achieves ~76% accuracy on test set from `matches.csv`
5. ✅ No data leakage (only uses pre-match information)
6. ✅ Feature importance matches model (Form/Fatigue dominant)

---

## 🚀 Next Steps After Predictions Work

1. Build a simple CLI interface
2. Add batch prediction capability
3. Create API endpoint (Flask/FastAPI)
4. Add prediction tracking (compare predictions vs actual outcomes)
5. Build confidence calibration (tune probability thresholds)

---

**Good luck! The model is strong (76.55% accuracy) and ready to make predictions. Your job is to build the interface to use it properly.** 🎾

**Questions to ask yourself**:
- Am I using the **current ELO** or final ELO?
- Are my form/fatigue calculations looking only at **past matches**?
- Do my 58 features match the **exact names and order** from training?
- How do I handle **missing data** gracefully?

**Remember**: The hardest part (model training) is done. Now make it usable! 💪
