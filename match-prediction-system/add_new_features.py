"""
Add new features (fatigue, tournament progression, surface advantage) to existing data.
"""

import pandas as pd
import logging
from src.data_processor.calculator import EloCalculator
from src.data_processor.data_splitter import DataSplitter
from src.data_processor.feature_engineering import FeatureEngineering

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("ADDING NEW FEATURES TO EXISTING DATA")
    logger.info("=" * 80)
    
    # Load existing train and test matches
    logger.info("\n1. Loading existing train and test data...")
    train_df = pd.read_csv('data/processed/train_matches.csv')
    test_df = pd.read_csv('data/processed/test_matches.csv')
    
    logger.info(f"Train: {len(train_df):,} matches")
    logger.info(f"Test: {len(test_df):,} matches")
    
    # Combine them to calculate features chronologically
    logger.info("\n2. Combining and sorting chronologically...")
    df = pd.concat([train_df, test_df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f"Total: {len(df):,} matches")
    
    # Calculate NEW features
    elo_calc = EloCalculator()
    
    logger.info("\n3. Calculating fatigue features...")
    df = elo_calc.calculate_fatigue(df)
    
    logger.info("\n4. Calculating tournament progression features...")
    df = elo_calc.calculate_tournament_progression(df)
    
    # Save as match_info.csv
    logger.info("\n5. Saving updated match_info.csv...")
    df.to_csv('data/processed/match_info.csv', index=False)
    logger.info(f"Saved {len(df):,} matches")
    
    # Show new features
    new_cols = ['player1_days_since_last', 'player2_days_since_last', 'rest_advantage',
                'player1_fatigued', 'player2_fatigued', 'both_rested',
                'player1_matches_in_tournament', 'player2_matches_in_tournament',
                'tournament_experience_diff', 'player1_deep_run', 'player2_deep_run']
    
    logger.info(f"\nNew columns added ({len(new_cols)}):")
    for col in new_cols:
        if col in df.columns:
            logger.info(f"  ✓ {col}")
        else:
            logger.info(f"  ✗ {col} (missing!)")
    
    # Show sample
    logger.info("\nSample of new features:")
    sample_cols = ['date', 'player1', 'player2', 'player1_days_since_last', 
                   'player2_days_since_last', 'player1_matches_in_tournament', 
                   'player2_matches_in_tournament']
    print(df[sample_cols].tail(10).to_string(index=False))
    
    # Re-split into train/test
    logger.info("\n6. Re-splitting data into train/test sets...")
    splitter = DataSplitter('data/processed/match_info.csv')
    train_df, test_df = splitter.temporal_split(
        test_size=0.2,
        output_dir='data/processed'
    )
    logger.info(f"Train: {len(train_df):,} matches, Test: {len(test_df):,} matches")
    
    # Feature engineering
    logger.info("\n7. Generating features for TRAIN set...")
    fe_train = FeatureEngineering('data/processed/train_matches.csv')
    X_train, y_train, feature_names = fe_train.prepare_features(
        scale=True,
        return_feature_names=True
    )
    fe_train.save_preprocessed_data(X_train, y_train)
    
    logger.info("\n8. Generating features for TEST set...")
    fe_test = FeatureEngineering('data/processed/test_matches.csv')
    X_test, y_test = fe_test.prepare_features(scale=True)
    fe_test.save_preprocessed_data(X_test, y_test)
    
    logger.info(f"\nFeature matrix shapes:")
    logger.info(f"  Train: {X_train.shape} ({len(feature_names)} features)")
    logger.info(f"  Test:  {X_test.shape}")
    
    logger.info(f"\nAll features ({len(feature_names)} total):")
    for i, fname in enumerate(feature_names, 1):
        marker = "🆕" if any(x in fname for x in ['fatigue', 'days_since', 'rest_', 
                                                     'tournament', 'matches_in', 'deep_run',
                                                     'surface_advantage', 'surface_specialist']) else "  "
        print(f"  {marker} {i:3d}. {fname}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEW FEATURES ADDED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext step: Retrain models with: python3 train_quick.py")

if __name__ == '__main__':
    main()
