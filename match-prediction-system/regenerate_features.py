"""
Regenerate match_info.csv with new individual match outcome features.
Run this after modifying the form calculation in calculator.py
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
    logger.info("REGENERATING FEATURES WITH NEW FORM CALCULATION")
    logger.info("=" * 80)
    
    # Check if we should reload from original sheets or use existing match_info
    import os
    use_existing = True  # Set to False to reload from original sheets
    
    if use_existing and os.path.exists('data/processed/match_info.csv'):
        logger.info("\n1. Loading existing match_info.csv (without form columns)...")
        df = pd.read_csv('data/processed/match_info.csv')
        
        # Remove old form columns if they exist
        old_form_cols = ['player1_form', 'player2_form', 'player1_form_wins', 
                        'player2_form_wins', 'player1_form_matches', 'player2_form_matches']
        df = df.drop(columns=[col for col in old_form_cols if col in df.columns], errors='ignore')
        
        logger.info(f"Loaded {len(df):,} matches")
    else:
        logger.info("\n1. Loading and merging raw data sheets...")
        from src.data_processor.data_loader import DataLoader
        loader = DataLoader()
        df = loader.merge_all_sheets(output_path=None)  # Don't save yet
        logger.info(f"Merged {len(df):,} matches from all sheets")
    
    # Ensure chronological order
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate NEW form features (individual match outcomes)
    logger.info("\n2. Calculating individual match outcomes (NEW FORM)...")
    elo_calc = EloCalculator()
    df = elo_calc.calculate_form(df, window=10)  # Now creates 20 individual match columns
    
    # Save updated match_info.csv
    logger.info("\n3. Saving updated match_info.csv...")
    df.to_csv('data/processed/match_info.csv', index=False)
    logger.info(f"Saved {len(df):,} matches to data/processed/match_info.csv")
    
    # Check new columns
    match_cols = [col for col in df.columns if 'match_' in col]
    logger.info(f"\nNew match outcome columns ({len(match_cols)}): {match_cols[:5]}...{match_cols[-5:]}")
    
    # Show sample
    logger.info("\nSample of new form features:")
    sample_cols = ['player1', 'player2'] + [f'player1_match_{i}' for i in range(1, 4)] + \
                  [f'player2_match_{i}' for i in range(1, 4)]
    print(df[sample_cols].head(20).to_string())
    
    # Split data into train/test
    logger.info("\n4. Splitting data into train/test sets...")
    splitter = DataSplitter('data/processed/match_info.csv')  # Load from file we just saved
    train_df, test_df = splitter.temporal_split(
        test_size=0.2,
        output_dir='data/processed'
    )
    logger.info(f"Train: {len(train_df):,} matches, Test: {len(test_df):,} matches")
    
    # Feature engineering
    logger.info("\n5. Generating features for TRAIN set...")
    fe_train = FeatureEngineering('data/processed/train_matches.csv')  # Load from saved file
    X_train, y_train, feature_names = fe_train.prepare_features(
        scale=True,
        return_feature_names=True
    )
    
    logger.info("\n6. Generating features for TEST set...")
    fe_test = FeatureEngineering('data/processed/test_matches.csv')  # Load from saved file
    X_test, y_test = fe_test.prepare_features(
        scale=True,
        return_feature_names=False
    )
    
    # Save features
    logger.info("\n7. Saving feature matrices...")
    X_train.to_csv('data/processed/train_features.csv', index=False)
    X_test.to_csv('data/processed/test_features.csv', index=False)
    
    pd.DataFrame({'target': y_train}).to_csv('data/processed/train_target.csv', index=False)
    pd.DataFrame({'target': y_test}).to_csv('data/processed/test_target.csv', index=False)
    
    logger.info(f"\nFeature matrix shapes:")
    logger.info(f"  Train: {X_train.shape} ({len(feature_names)} features)")
    logger.info(f"  Test:  {X_test.shape}")
    
    logger.info(f"\nNew features ({len(feature_names)} total):")
    for i, fname in enumerate(feature_names, 1):
        print(f"  {i:2d}. {fname}")
    
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE REGENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext step: Retrain models with: python train_quick.py")

if __name__ == '__main__':
    main()
