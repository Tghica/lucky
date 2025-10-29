# Data Reset & Rebuild Guide

## Current Issues Identified

### ✅ Fixed Issues
- Missing ELO data
- Feature mismatch between training/prediction

### ⚠️ Remaining Issues
- **Duplicate matches**: 2 matches from Masters Cup 2000 (same date/players, different outcomes)
- **Data quality**: Potential inconsistencies from multiple data sources
- **Model accuracy ceiling**: ~65% (possibly due to data quality issues)
- **New players**: Lack of historical data for emerging players

## Fresh Start Strategy

### Phase 1: Data Collection (YOU DO THIS)

#### Required Raw Data Files
Collect the following CSV files and place them in `data/raw/`:

1. **ATP Match Data** (`atp_matches_YYYY.csv` for each year)
   - Columns needed: date, tournament, surface, player1, player2, winner, score, round
   - Recommended source: Jeff Sackmann's tennis_atp repository
   - Years: 2000-2025 (or your preferred range)

2. **ATP Rankings** (`atp_rankings_YYYY.csv` for each year)
   - Columns needed: date, ranking, player, points
   - For player career trajectory tracking

3. **Tournament Information** (optional but helpful)
   - Tournament level (ATP 250, 500, 1000, Grand Slam)
   - Location, surface confirmation

#### Data Quality Checklist
- [ ] All files use consistent date format (YYYY-MM-DD preferred)
- [ ] Player names are standardized (same format across all files)
- [ ] No duplicate match records
- [ ] Surface information is complete
- [ ] Tournament names are consistent

### Phase 2: Data Directory Structure

```
data/
├── raw/                          # YOUR RAW DATA GOES HERE
│   ├── atp_matches_2000.csv
│   ├── atp_matches_2001.csv
│   ├── ... (one file per year)
│   ├── atp_matches_2025.csv
│   ├── atp_rankings_2000.csv
│   ├── ... (rankings per year)
│   └── README.md                 # Document your data sources
├── processed/                    # AUTO-GENERATED (will be cleared)
│   └── (cleaned, feature-engineered data)
└── sheets/                       # ARCHIVE (keep for reference)
    └── archive/
        └── (old data files)
```

### Phase 3: Clean Slate Preparation

#### Before You Start:
1. **Backup Current Data** (optional, but recommended)
   ```bash
   cd /Users/tudorghica/Desktop/lucky/match-prediction-system
   mkdir -p data/backup_$(date +%Y%m%d)
   cp -r data/processed/* data/backup_$(date +%Y%m%d)/
   ```

2. **Clear Processed Data**
   ```bash
   # We'll do this together after you have raw data
   rm data/processed/*.csv
   rm data/processed/*.json
   ```

3. **Create Raw Data Directory**
   ```bash
   mkdir -p data/raw
   ```

### Phase 4: Data Processing Pipeline (WE'LL BUILD THIS)

Once you have raw data, we'll create:

1. **Data Validator** (`scripts/data/validate_raw_data.py`)
   - Check file formats
   - Identify duplicates
   - Validate required columns
   - Check data completeness

2. **Data Cleaner** (`scripts/data/clean_raw_data.py`)
   - Standardize player names
   - Fix date formats
   - Remove duplicates
   - Handle missing values

3. **Data Merger** (`scripts/data/merge_data.py`)
   - Combine yearly files
   - Sort chronologically
   - Add tournament levels
   - Add surface information

4. **ELO Calculator** (improve existing `calculate_elos.py`)
   - Calculate overall ELO
   - Calculate surface-specific ELO
   - Calculate tournament-specific ELO
   - Track ELO history

5. **Feature Engineer** (rebuild `src/data_processor/feature_engineering.py`)
   - Generate all features
   - Handle new players gracefully
   - Create proper train/test splits
   - Validate feature consistency

### Phase 5: Quality Assurance

After rebuilding, we'll verify:
- [ ] No duplicate matches
- [ ] All players have ELO ratings
- [ ] Feature names match between train/test
- [ ] No data leakage
- [ ] Proper temporal ordering
- [ ] Consistent data types

### Phase 6: Model Retraining

With clean data:
- [ ] Retrain all models
- [ ] Validate accuracy improvements
- [ ] Test on recent matches
- [ ] Deploy updated system

## Recommended Data Source

**Jeff Sackmann's Tennis ATP Data**
- Repository: https://github.com/JeffSackmann/tennis_atp
- Most comprehensive ATP data available
- Well-maintained and regularly updated
- Standardized format

### Files to download:
```
atp_matches_2000.csv through atp_matches_2025.csv
atp_rankings_00s.csv
atp_rankings_10s.csv
atp_rankings_20s.csv
atp_rankings_current.csv
```

## Next Steps

1. **YOU**: Download raw data files to `data/raw/`
2. **WE**: Build validation pipeline
3. **WE**: Build processing pipeline
4. **WE**: Rebuild features and train models
5. **TEST**: Verify improved accuracy

## Notes

- Keep raw data separate from processed data
- Document data sources in `data/raw/README.md`
- Version control: Don't commit large CSV files (use .gitignore)
- Processing should be reproducible (script everything)

---

**Ready to proceed?** 
Once you have the raw data files, let me know and we'll start the validation and processing pipeline!
