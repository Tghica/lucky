# Training vs Prediction Data - Strategy Guide

## 🎯 The Challenge

**Training Data** (Jeff Sackmann 2000-2024):
- ✅ Service stats: 91% coverage
- ✅ Rankings: 99% coverage
- ✅ Physical stats: 95-97% coverage
- ❌ **Betting odds: 0% coverage**

**Prediction Data** (Live matches 2025):
- ✅ Betting odds: 100% coverage
- ✅ Rankings: 100% coverage
- ✅ Match context: 100% coverage
- ❌ **Service stats: 0% coverage**
- ❌ **Player metadata: 0% coverage** (height, age, hand, IDs)

**The Problem**: How do we train on features that won't be available at prediction time, and use features at prediction time that weren't in training?

## 💡 Solution: Robust Feature Engineering

### Strategy 1: ALWAYS Available Features (Core Model)

Build a base model using features that are **guaranteed** to be available:

```python
ALWAYS_AVAILABLE = [
    'elo_rating',           # Calculated from match history
    'surface_elo',          # ELO by surface type
    'ranking',              # 99%+ coverage
    'ranking_points',       # 99%+ coverage
    'surface',              # 100% coverage
    'tournament_level',     # 100% coverage
    'h2h_record',           # Calculated from history
]
```

**Accuracy target**: 60-63% (baseline)

### Strategy 2: Career Averages (Imputed Features)

For features missing at prediction time, use **player career averages**:

```python
# Training Phase
if match.has_service_stats():
    features['w_ace'] = match.w_ace              # Actual: 12 aces
else:
    features['w_ace'] = player.career_avg_aces   # Average: 8.5 aces

# Prediction Phase (always use career average)
features['w_ace'] = player.career_avg_aces       # Always available
```

**Player Career Stats to Calculate**:
- `avg_aces_per_match`
- `avg_double_faults_per_match`
- `avg_1st_serve_pct`
- `avg_1st_serve_won_pct`
- `avg_2nd_serve_won_pct`
- `avg_break_points_saved_pct`

**Accuracy boost**: +3-5% (63-68%)

### Strategy 3: Optional Odds Features (Bonus Calibration)

Make betting odds **optional** - use when available, skip when not:

```python
# Feature Engineering
def add_odds_features(match, features):
    if match.has_odds():
        features['implied_prob_p1'] = 1 / match.odds_p1
        features['implied_prob_p2'] = 1 / match.odds_p2
        features['odds_ratio'] = match.odds_p1 / match.odds_p2
        features['has_odds'] = 1
    else:
        # Fill with neutral values or mean
        features['implied_prob_p1'] = 0.5  # Neutral
        features['implied_prob_p2'] = 0.5
        features['odds_ratio'] = 1.0       # Even odds
        features['has_odds'] = 0           # Flag for model
```

**OR**: Create synthetic odds for training:

```python
# Calculate implied odds from ELO
def synthetic_odds(elo_p1, elo_p2):
    prob_p1 = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
    odds_p1 = 1 / prob_p1
    odds_p2 = 1 / (1 - prob_p1)
    return odds_p1, odds_p2
```

**Accuracy boost**: +2-4% (65-72%)

## 🔧 Implementation Phases

### Phase 1: Core Model (Week 1) ⭐ PRIORITY

**Features** (Always available):
1. ELO ratings (overall, surface, tournament)
2. Rankings & ranking points
3. Surface type & tournament level
4. H2H records
5. Recent form (last 10 matches)

**Expected Accuracy**: 60-63%

### Phase 2: Career Stats (Week 2)

**Add Player Career Averages**:
1. Service statistics averages
2. Return statistics averages
3. Surface-specific win rates
4. Performance vs top-10/50/100

**Calculate Once**: 
- Run through all 75K matches
- Build player career profile database
- Save to `data/processed/player_career_stats.csv`

**At Prediction Time**:
- Look up player in career stats database
- Use averages instead of match-specific stats

**Expected Accuracy**: 63-68%

### Phase 3: Odds Integration (Week 3)

**Two Approaches**:

A. **Train Without, Add at Prediction**:
   - Train model on historical data (no odds)
   - At prediction: add odds as extra features
   - Model learns to ignore them when missing

B. **Synthetic Odds for Training**:
   - Generate odds from ELO/rankings for training
   - Use real odds for prediction
   - Model learns odds patterns

**Expected Accuracy**: 65-72%

## 📊 Feature Availability Matrix

| Feature Category | Training (Historical) | Prediction (Live) | Solution |
|-----------------|----------------------|-------------------|----------|
| **ELO Ratings** | ✅ 100% | ✅ 100% | Calculate from history |
| **Rankings** | ✅ 99% | ✅ 100% | Direct use |
| **Surface/Tournament** | ✅ 100% | ✅ 100% | Direct use |
| **Service Stats** | ✅ 91% | ❌ 0% | Career averages |
| **Physical Stats** | ✅ 95-97% | ❌ 0% | Fetch from RapidAPI |
| **Betting Odds** | ❌ 0% | ✅ 100% | Optional/Synthetic |
| **H2H Records** | ✅ 100% | ✅ 100% | Calculate from history |

## 🎓 Machine Learning Best Practices

### 1. Feature Consistency
```python
# ❌ BAD: Different features for training vs prediction
train_features = ['elo', 'ranking', 'actual_aces']      # 91% have actual_aces
pred_features = ['elo', 'ranking']                      # Missing aces feature!

# ✅ GOOD: Same features, different values
train_features = ['elo', 'ranking', 'aces']
pred_features = ['elo', 'ranking', 'aces']              # Use career avg for aces
```

### 2. Imputation Strategy
```python
# ❌ BAD: Drop rows with missing data
df = df.dropna(subset=['w_ace'])  # Lose 9% of training data

# ✅ GOOD: Impute with player-specific average
df['w_ace'] = df.groupby('winner')['w_ace'].transform(
    lambda x: x.fillna(x.mean())
)
```

### 3. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Use imputer as part of pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('model', RandomForestClassifier())
])

# Training: Some matches have actual stats, some use career avg
pipeline.fit(X_train, y_train)

# Prediction: All matches use career avg (consistent)
predictions = pipeline.predict(X_pred)
```

## 🔍 Practical Example

### Scenario: Predict Djokovic vs Alcaraz

**Available at Prediction Time**:
- ✅ Player names
- ✅ Surface: Hard
- ✅ Tournament: ATP Masters 1000
- ✅ Betting odds: Djokovic 1.80, Alcaraz 2.10
- ❌ Service stats (match hasn't happened)

**Feature Generation**:

```python
features = {
    # Always available (from history)
    'djokovic_elo': 2350,
    'alcaraz_elo': 2280,
    'elo_diff': 70,
    'djokovic_rank': 1,
    'alcaraz_rank': 2,
    'rank_diff': -1,
    'surface': 'Hard',
    'tournament_level': 'Masters',
    
    # Career averages (imputed)
    'djokovic_avg_aces': 5.2,      # From career data
    'alcaraz_avg_aces': 7.8,
    'ace_diff': -2.6,
    'djokovic_1st_serve_pct': 64.5,
    'alcaraz_1st_serve_pct': 68.2,
    
    # Odds (when available)
    'implied_prob_djokovic': 0.556,  # 1/1.80
    'implied_prob_alcaraz': 0.476,   # 1/2.10
    
    # H2H (from history)
    'h2h_wins_djokovic': 3,
    'h2h_wins_alcaraz': 2,
    'h2h_hard_wins_djokovic': 2,
}

# Make prediction
win_prob = model.predict_proba(features)[0][1]  # 0.62 = 62% Djokovic
```

## ✅ Key Takeaways

1. **Build Robust Features**: Use features that work with incomplete data
2. **Career Averages**: Best solution for missing match-specific stats
3. **Optional Odds**: Add as bonus, don't require them
4. **Smart Imputation**: Player-specific > Mean > Zero
5. **ELO is King**: Always available, always reliable

## 🚀 Next Steps

1. **Calculate ELO ratings** for all 75K matches ⭐ URGENT
2. **Build player career stats database** (one-time calculation)
3. **Train base model** with always-available features
4. **Add career average features** for service stats
5. **Test with/without odds** to see impact

---

**Bottom Line**: Train with what you have (service stats), predict with what you can get (career averages + odds). The model learns patterns that work in both scenarios! 🎾
