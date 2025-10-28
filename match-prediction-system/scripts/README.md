# Scripts Directory

Organized scripts for the tennis match prediction system.

## 📁 Directory Structure

### `data/`
Data processing and preparation scripts:
- `build_tournament_surface_map.py` - Extract tournament → surface mapping from match history

### `fetch/`
Scripts for fetching match data from external sources:
- **`fetch_and_merge.py`** - Fetch matches from RapidAPI and merge into match_info.csv
- `build_surface_lookup.py` - Build tournament surface lookup table

### `features/`
Feature engineering scripts:
- **`regenerate_features.py`** - Regenerate all features after adding new matches
- `add_new_features.py` - Add new feature columns to existing data
- `add_win_streak_features.py` - Add win streak and momentum features

### `training/`
Model training scripts:
- **`train_quick.py`** - Quick model training with current hyperparameters
- `tune_hyperparameters.py` - Hyperparameter tuning (takes longer)
- `ensemble_model.py` - Train ensemble models (stacking/voting)

### `prediction/`
Match prediction utilities:
- **`predict_match.py`** - Predict a single match outcome

### `analysis/`
Analysis and visualization scripts:
- `analyze_close_matches.py` - Analyze close match predictions
- `compare_atp_elo_rankings.py` - Compare ELO rankings with ATP rankings
- `visualize_atp_elo_comparison.py` - Visualize ELO vs ATP rankings

---

## 🚀 Common Workflows

### Weekly Data Update
```bash
# 1. Fetch new matches (last 7 days)
python3 scripts/fetch/fetch_and_merge.py --last-week

# 2. Regenerate features
python3 scripts/features/regenerate_features.py

# 3. Retrain model (optional, if many new matches)
python3 scripts/training/train_quick.py
```

### Make a Prediction
```bash
python3 scripts/prediction/predict_match.py "Player1 Name" "Player2 Name" Hard atp250
```

### Initial Setup
```bash
# 1. Build tournament surface lookup
python3 scripts/data/build_tournament_surface_map.py

# 2. Generate features
python3 scripts/features/regenerate_features.py

# 3. Train model
python3 scripts/training/train_quick.py
```

---

## �� Notes

- **Always run scripts from project root** (`match-prediction-system/` folder)
- RapidAPI key configured in `src/data_fetcher/fetch_rapidapi.py`
- You have **500 requests/month** - enough for ~20 weekly updates
- Models saved to `models/saved_models/`
- Processed data in `data/processed/`

---

## 🔧 RapidAPI Setup

API key is already configured in the code. If you need to update it:
1. Edit `src/data_fetcher/fetch_rapidapi.py`
2. Find the line with `self.api_key = ...`
3. Replace with your new key

Current limit: **500 requests/month**
