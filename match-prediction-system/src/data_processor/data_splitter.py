import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Split match data into training and test sets with temporal awareness."""
    
    def __init__(self, match_info_path: str = "data/processed/match_info.csv"):
        """
        Initialize DataSplitter.
        
        Args:
            match_info_path: Path to the processed match_info.csv file
        """
        self.match_info_path = Path(match_info_path)
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the match_info data."""
        if not self.match_info_path.exists():
            raise FileNotFoundError(f"Match info file not found at {self.match_info_path}")
        
        self.df = pd.read_csv(self.match_info_path)
        logger.info(f"Loaded {len(self.df)} matches from {self.match_info_path}")
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date to ensure chronological order
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        return self.df
    
    def temporal_split(
        self,
        test_size: float = 0.2,
        output_dir: str = "data/processed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (chronologically).
        
        Training set: Earlier matches
        Test set: Most recent matches
        
        This is the recommended approach for time-series prediction to avoid data leakage.
        
        Args:
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            output_dir: Directory to save the split datasets
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        # Calculate split point
        split_idx = int(len(self.df) * (1 - test_size))
        
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        split_date = train_df['date'].max()
        
        logger.info(f"\n=== Temporal Split ===")
        logger.info(f"Training set: {len(train_df)} matches ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} matches ({test_df['date'].min()} to {test_df['date'].max()})")
        logger.info(f"Split date: {split_date}")
        logger.info(f"Train/Test ratio: {len(train_df)}/{len(test_df)} ({(1-test_size)*100:.1f}%/{test_size*100:.1f}%)")
        
        # Save to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train_matches.csv"
        test_path = output_path / "test_matches.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\nSaved training data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
        
        return train_df, test_df
    
    def date_based_split(
        self,
        split_date: str,
        output_dir: str = "data/processed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on a specific date.
        
        Args:
            split_date: Date string (e.g., '2020-01-01'). Matches before this date go to train.
            output_dir: Directory to save the split datasets
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        split_date = pd.to_datetime(split_date)
        
        train_df = self.df[self.df['date'] < split_date].copy()
        test_df = self.df[self.df['date'] >= split_date].copy()
        
        logger.info(f"\n=== Date-Based Split ===")
        logger.info(f"Split date: {split_date}")
        logger.info(f"Training set: {len(train_df)} matches ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} matches ({test_df['date'].min()} to {test_df['date'].max()})")
        logger.info(f"Train/Test ratio: {len(train_df)}/{len(test_df)} ({len(train_df)/len(self.df)*100:.1f}%/{len(test_df)/len(self.df)*100:.1f}%)")
        
        # Save to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train_matches.csv"
        test_path = output_path / "test_matches.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\nSaved training data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
        
        return train_df, test_df
    
    def year_based_split(
        self,
        test_years: int = 1,
        output_dir: str = "data/processed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by using the last N years as test set.
        
        Args:
            test_years: Number of most recent years to use for testing
            output_dir: Directory to save the split datasets
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        # Get the most recent date and calculate cutoff
        max_date = self.df['date'].max()
        split_date = max_date - pd.DateOffset(years=test_years)
        
        train_df = self.df[self.df['date'] < split_date].copy()
        test_df = self.df[self.df['date'] >= split_date].copy()
        
        logger.info(f"\n=== Year-Based Split ===")
        logger.info(f"Test years: Last {test_years} year(s)")
        logger.info(f"Split date: {split_date}")
        logger.info(f"Training set: {len(train_df)} matches ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} matches ({test_df['date'].min()} to {test_df['date'].max()})")
        logger.info(f"Train/Test ratio: {len(train_df)}/{len(test_df)} ({len(train_df)/len(self.df)*100:.1f}%/{len(test_df)/len(self.df)*100:.1f}%)")
        
        # Save to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train_matches.csv"
        test_path = output_path / "test_matches.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\nSaved training data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
        
        return train_df, test_df
    
    def stratified_temporal_split(
        self,
        test_size: float = 0.2,
        stratify_by: str = 'surface',
        output_dir: str = "data/processed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally while maintaining distribution of a categorical variable.
        
        This creates temporal splits within each stratum (e.g., surface type).
        
        Args:
            test_size: Proportion of data to use for testing (0.0 to 1.0)
            stratify_by: Column to stratify by (e.g., 'surface', 'tournament_level')
            output_dir: Directory to save the split datasets
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        if stratify_by not in self.df.columns:
            raise ValueError(f"Column '{stratify_by}' not found in dataset")
        
        train_dfs = []
        test_dfs = []
        
        # Split each stratum temporally
        for stratum_value in self.df[stratify_by].unique():
            stratum_df = self.df[self.df[stratify_by] == stratum_value].sort_values('date')
            split_idx = int(len(stratum_df) * (1 - test_size))
            
            train_dfs.append(stratum_df.iloc[:split_idx])
            test_dfs.append(stratum_df.iloc[split_idx:])
        
        # Combine all strata
        train_df = pd.concat(train_dfs, ignore_index=True).sort_values('date').reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sort_values('date').reset_index(drop=True)
        
        logger.info(f"\n=== Stratified Temporal Split (by {stratify_by}) ===")
        logger.info(f"Training set: {len(train_df)} matches ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} matches ({test_df['date'].min()} to {test_df['date'].max()})")
        logger.info(f"\nDistribution in training set:")
        logger.info(train_df[stratify_by].value_counts())
        logger.info(f"\nDistribution in test set:")
        logger.info(test_df[stratify_by].value_counts())
        
        # Save to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train_matches.csv"
        test_path = output_path / "test_matches.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\nSaved training data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
        
        return train_df, test_df
    
    def get_split_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """
        Get summary statistics comparing train and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_date_range': (train_df['date'].min(), train_df['date'].max()),
            'test_date_range': (test_df['date'].min(), test_df['date'].max()),
            'train_players': len(set(train_df['player1'].tolist() + train_df['player2'].tolist())),
            'test_players': len(set(test_df['player1'].tolist() + test_df['player2'].tolist())),
        }
        
        # Check for data leakage (players appearing only in test set)
        train_players = set(train_df['player1'].tolist() + train_df['player2'].tolist())
        test_players = set(test_df['player1'].tolist() + test_df['player2'].tolist())
        new_players_in_test = test_players - train_players
        
        summary['new_players_in_test'] = len(new_players_in_test)
        summary['new_players_ratio'] = len(new_players_in_test) / len(test_players) if test_players else 0
        
        # Surface distribution comparison
        if 'surface' in train_df.columns:
            summary['train_surface_dist'] = train_df['surface'].value_counts(normalize=True).to_dict()
            summary['test_surface_dist'] = test_df['surface'].value_counts(normalize=True).to_dict()
        
        return summary


def main():
    """Example usage of DataSplitter."""
    splitter = DataSplitter()
    
    # Option 1: Temporal split (recommended for time-series)
    # 80% training, 20% test
    print("\n" + "="*60)
    print("OPTION 1: Temporal Split (80/20)")
    print("="*60)
    train_df, test_df = splitter.temporal_split(test_size=0.2)
    
    # Get summary
    summary = splitter.get_split_summary(train_df, test_df)
    print(f"\nNew players appearing only in test set: {summary['new_players_in_test']} ({summary['new_players_ratio']*100:.2f}%)")
    
    # Option 2: Year-based split
    # Use last year as test set
    # print("\n" + "="*60)
    # print("OPTION 2: Year-Based Split (Last 1 Year as Test)")
    # print("="*60)
    # train_df, test_df = splitter.year_based_split(test_years=1)
    
    # Option 3: Date-based split
    # Split at specific date
    # print("\n" + "="*60)
    # print("OPTION 3: Date-Based Split (Split at 2020-01-01)")
    # print("="*60)
    # train_df, test_df = splitter.date_based_split('2020-01-01')
    
    # Option 4: Stratified temporal split
    # Maintain surface distribution
    # print("\n" + "="*60)
    # print("OPTION 4: Stratified Temporal Split (by Surface)")
    # print("="*60)
    # train_df, test_df = splitter.stratified_temporal_split(test_size=0.2, stratify_by='surface')


if __name__ == "__main__":
    main()
