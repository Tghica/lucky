# Old Model Parameters & Performance

**Saved from previous training runs**
**Date: October 27-29, 2025**

---

## 1. XGBoost Tuned Parameters (BEST MODEL)

**File:** `models/saved_models/xgboost_tuned_params.json`

### Hyperparameters:
```json
{
  "n_estimators": 700,
  "max_depth": 3,
  "learning_rate": 0.03,
  "subsample": 0.7,
  "colsample_bytree": 0.8,
  "gamma": 0.2,
  "min_child_weight": 5,
  "reg_alpha": 0.01,
  "reg_lambda": 1.5
}
```

### Performance:
- **Test Accuracy:** 64.87%
- **Test AUC:** 0.7109
- **Train Accuracy:** 69.99%
- **Train AUC:** 0.7754
- **CV Score:** 0.7593
- **Tuning Time:** 4.34 minutes

### Improvement Over Baseline:
- Accuracy improvement: +0.13%
- AUC improvement: +0.19%

---

## 2. Calibration Results

**File:** `models/saved_models/calibration_results.json`

### XGBoost (Calibrated):
- **Accuracy:** 64.79%
- **AUC:** 0.7133
- **Brier Score:** 0.2154
- **Log Loss:** 0.6181
- **ECE (Expected Calibration Error):** 0.0271

### LightGBM (Calibrated):
- **Accuracy:** 64.86%
- **AUC:** 0.7149
- **Brier Score:** 0.2150
- **Log Loss:** 0.6172
- **ECE:** 0.0311

---

## 3. Ensemble Models

### Voting Ensemble:
```json
{
  "train_accuracy": 0.7157,
  "train_auc": 0.7989,
  "test_accuracy": 0.6477,
  "test_auc": 0.7126,
  "training_time": 331.57 seconds
}
```

### Stacking Ensemble:
```json
{
  "train_accuracy": 0.7019,
  "train_auc": 0.7788,
  "test_accuracy": 0.6493,
  "test_auc": 0.7115,
  "training_time": 611.89 seconds
}
```

---

## 4. Training Configuration

**File:** `configs/config.yaml`

### Data Split:
- **Test Size:** 20%
- **Split Method:** Temporal (chronological)
- **Split Date:** 2020-02-20
  - Training: Before 2020-02-20
  - Testing: After 2020-02-20

### Model Config (Default):
- **Type:** RandomForest (baseline)
- **n_estimators:** 100
- **max_depth:** 10
- **random_state:** 42

### Paths:
- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Match info: `data/processed/match_info.csv`
- Player info: `data/processed/player_info.csv`

---

## 5. Performance Summary

### Best Individual Model: XGBoost Tuned
- ✅ Test Accuracy: **64.87%**
- ✅ Test AUC: **0.7109**
- ⚠️ Overfitting: Train accuracy 5% higher than test

### Best Ensemble: Stacking
- ✅ Test Accuracy: **64.93%** (slightly better)
- ✅ Test AUC: **0.7115**
- ⏱️ Training time: 10+ minutes

### Performance Ceiling:
- **Maximum accuracy achieved:** ~65%
- **Likely causes:**
  1. Data quality issues (duplicates, inconsistencies)
  2. Missing/incomplete ELO data for some players
  3. Feature engineering limitations
  4. Inherent unpredictability in tennis matches

---

## 6. Key Insights from Old Model

### What Worked:
1. ✅ XGBoost with careful tuning
2. ✅ Temporal split (avoided data leakage)
3. ✅ ELO-based features (general, surface, tournament)
4. ✅ Ensemble methods (marginal improvement)

### What Didn't Work Well:
1. ⚠️ Calibration (minimal improvement)
2. ⚠️ Complex ensembles (high training time, small gains)
3. ⚠️ Couldn't break 65% accuracy ceiling

### Areas for Improvement with New Data:
1. 🎯 Better data quality (clean source)
2. 🎯 More complete ELO coverage
3. 🎯 Additional features (betting odds, head-to-head)
4. 🎯 Better handling of new players
5. 🎯 More training data (2000-2025 vs limited range)

---

## 7. Recommended Parameters for New Training

Based on old results, start with these XGBoost parameters:

```python
xgb_params = {
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
    'n_jobs': -1
}
```

### Then tune further with:
- Increase n_estimators (700 → 1000) if we have more data
- Adjust learning_rate (0.03 → 0.01-0.05) for better convergence
- Consider max_depth (3 → 4-5) with more features

---

## 8. Expected Improvements with New Data

With clean data from Jeff Sackmann + 2025 updates:

### Conservative Estimate:
- Accuracy: 65% → **67-70%**
- AUC: 0.71 → **0.73-0.76**

### Optimistic Estimate (if all goes well):
- Accuracy: 65% → **70-73%**
- AUC: 0.71 → **0.76-0.80**

### Factors That Will Help:
1. ✅ Clean, consistent data (no duplicates)
2. ✅ Complete ELO coverage (all players)
3. ✅ More training samples (~60K vs ~66K before)
4. ✅ Better features (betting odds for calibration)
5. ✅ Proper temporal ordering

---

## 9. Complete List of 110 Features Used

**Legend:** ⭐ High importance (>0.01) | ✓ Used (>0) | ✗ Not used (=0)

### Top 20 Most Important Features:
1. ⭐ **combined_elo_diff** (0.2265) - Combined ELO difference
2. ⭐ **elo_diff** (0.0784) - Standard ELO difference
3. ⭐ **win_probability** (0.0698) - Win probability from ELO
4. ⭐ **player2_match_1** (0.0313) - Player 2's most recent match
5. ⭐ **momentum_diff** (0.0293) - Momentum difference
6. ⭐ **surface_elo_ratio** (0.0251) - Surface-specific ELO ratio
7. ⭐ **player1_rest_quality** (0.0201) - Player 1 rest quality
8. ⭐ **surface_elo_diff** (0.0200) - Surface ELO difference
9. ⭐ **player1_match_1** (0.0188) - Player 1's most recent match
10. ⭐ **player1_days_since_last** (0.0184) - Days since last match
11. ⭐ **player1_deep_run** (0.0183) - Deep tournament run indicator
12. ⭐ **rest_quality_diff** (0.0177) - Rest quality difference
13. ⭐ **rest_advantage** (0.0146) - Rest advantage
14. ⭐ **elo_ratio** (0.0142) - ELO ratio
15. ⭐ **surface_win_probability** (0.0121) - Surface-specific win probability
16. ⭐ **player2_days_since_last** (0.0118) - P2 days since last match
17. ⭐ **tournament_grand_slam** (0.0104) - Grand Slam indicator
18. ⭐ **player2_rest_quality** (0.0102) - Player 2 rest quality

### Feature Categories (110 Total):

#### ELO Features (21 features):
- combined_elo_diff, elo_diff, elo_ratio, surface_elo_diff, surface_elo_ratio
- tournament_elo_diff, tournament_elo_ratio
- player1_elo_before, player2_elo_before
- player1_surface_elo_before, player2_surface_elo_before
- player1_tournament_elo_before, player2_tournament_elo_before
- combined_elo_p1, combined_elo_p2
- h2h_elo_advantage_diff, player1_h2h_elo_advantage, player2_h2h_elo_advantage
- win_probability, surface_win_probability, tournament_win_probability

#### Form & Momentum Features (14 features):
- momentum_diff, player1_momentum, player2_momentum
- form_win_diff, player1_form_wins, player2_form_wins
- recent_form_diff, player1_recent_form, player2_recent_form
- player1_tournament_momentum, player2_tournament_momentum, tournament_momentum_diff
- player1_deep_run, player2_deep_run

#### Rest & Fatigue Features (13 features):
- rest_advantage, rest_quality_diff
- player1_rest_quality, player2_rest_quality
- player1_days_since_last, player2_days_since_last
- fatigue_impact_diff, player1_fatigue_impact, player2_fatigue_impact
- player1_fatigued, player2_fatigued ✗
- both_rested

#### Recent Match History (20 features):
- player1_match_1 through player1_match_10 (10 features)
- player2_match_1 through player2_match_10 (10 features)

#### Surface Features (12 features):
- surface_Hard, surface_Clay, surface_Grass, surface_Carpet
- surface_advantage_diff, surface_advantage_pct_diff
- player1_surface_advantage, player2_surface_advantage
- player1_surface_advantage_pct, player2_surface_advantage_pct
- player1_surface_specialist, player2_surface_specialist ✗

#### Tournament Features (10 features):
- tournament_grand_slam, tournament_masters, tournament_atp500, tournament_atp250
- tournament_experience_diff, tournament_rounds_diff
- player1_tournament_rounds, player2_tournament_rounds
- player1_matches_in_tournament, player2_matches_in_tournament

#### Head-to-Head Features (8 features):
- h2h_win_rate_diff, h2h_experience ✗
- player1_h2h_wins, player2_h2h_wins
- player1_h2h_matches, player2_h2h_matches
- player1_h2h_win_rate, player2_h2h_win_rate

#### Player Characteristics (12 features):
- player1_age, player2_age, age_diff ✗
- player1_height ✗, player2_height ✗, height_diff ✗, player1_taller ✗
- player1_hand_encoded ✗, player2_hand_encoded ✗
- player1_lefty ✗, player2_lefty ✗, lefty_vs_righty ✗, same_hand ✗

### Features NOT Used by Model (14 features with zero importance):
❌ Height-related: player1_height, player2_height, height_diff, player1_taller
❌ Hand-related: player1_hand_encoded, player2_hand_encoded, player1_lefty, player2_lefty, lefty_vs_righty, same_hand
❌ Others: age_diff, h2h_experience, player2_surface_specialist, player2_fatigued

**Key Insight:** ELO-based features account for ~35% of total model importance. Height and handedness features were completely ignored by the model.

---

**Note:** Keep these parameters as a baseline when training with new data!

**Full detailed feature list:** See `OLD_MODEL_FEATURES_110.txt` for complete breakdown with importance scores.
