import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering for tennis match prediction.
    
    CRITICAL: Only uses static player attributes (hand, height, age) from player_info.
    Does NOT use cumulative stats (total wins, win_rate, etc.) to avoid data leakage.
    Temporal features (Elo, form) come from match_info.csv which are computed chronologically.
    """
    
    def __init__(
        self,
        match_data_path: str,
        player_info_path: str = "data/processed/player_info.csv"
    ):
        """
        Initialize FeatureEngineering.
        
        Args:
            match_data_path: Path to match data CSV (train or test)
            player_info_path: Path to player info CSV (for static attributes only)
        """
        self.match_data_path = Path(match_data_path)
        self.player_info_path = Path(player_info_path)
        self.matches = None
        self.player_info = None
        self.label_encoders = {}
        self.scaler = None
        
    def load_data(self) -> pd.DataFrame:
        """Load match and player data."""
        # Load matches
        self.matches = pd.read_csv(self.match_data_path)
        self.matches['date'] = pd.to_datetime(self.matches['date'])
        logger.info(f"Loaded {len(self.matches)} matches from {self.match_data_path}")
        
        # Load player info (ONLY for static attributes)
        self.player_info = pd.read_csv(self.player_info_path)
        
        # Extract ONLY static attributes to prevent data leakage
        static_columns = ['player_id', 'player_name', 'hand', 'dob', 'country', 'height']
        self.player_info = self.player_info[
            [col for col in static_columns if col in self.player_info.columns]
        ]
        
        logger.info(f"Loaded {len(self.player_info)} players (static attributes only)")
        logger.info(f"Player attributes: {self.player_info.columns.tolist()}")
        
        return self.matches
    
    def merge_player_attributes(self) -> pd.DataFrame:
        """
        Merge static player attributes (hand, height, age) into match data.
        
        WARNING: Only uses STATIC attributes. Does NOT use cumulative stats
        (total_wins, win_rate, etc.) to prevent data leakage.
        """
        if self.matches is None or self.player_info is None:
            self.load_data()
        
        df = self.matches.copy()
        
        # Merge player1 attributes
        df = df.merge(
            self.player_info,
            left_on='player1',
            right_on='player_name',
            how='left',
            suffixes=('', '_p1')
        )
        df = df.rename(columns={
            'hand': 'player1_hand',
            'dob': 'player1_dob',
            'country': 'player1_country',
            'height': 'player1_height',
            'player_id': 'player1_id'
        })
        df = df.drop(columns=['player_name'], errors='ignore')
        
        # Merge player2 attributes
        df = df.merge(
            self.player_info,
            left_on='player2',
            right_on='player_name',
            how='left',
            suffixes=('', '_p2')
        )
        df = df.rename(columns={
            'hand': 'player2_hand',
            'dob': 'player2_dob',
            'country': 'player2_country',
            'height': 'player2_height',
            'player_id': 'player2_id'
        })
        df = df.drop(columns=['player_name'], errors='ignore')
        
        logger.info("Merged static player attributes (hand, height, DOB)")
        
        return df
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age features at match time (safe - no data leakage).
        
        Age is calculated as of the match date, so it's valid historical data.
        """
        # Convert DOB to datetime
        df['player1_dob'] = pd.to_datetime(df['player1_dob'], format='%Y%m%d', errors='coerce')
        df['player2_dob'] = pd.to_datetime(df['player2_dob'], format='%Y%m%d', errors='coerce')
        
        # Calculate age at match time
        df['player1_age'] = (df['date'] - df['player1_dob']).dt.days / 365.25
        df['player2_age'] = (df['date'] - df['player2_dob']).dt.days / 365.25
        
        # Age difference
        df['age_diff'] = df['player1_age'] - df['player2_age']
        
        # Drop DOB columns (not needed as features)
        df = df.drop(columns=['player1_dob', 'player2_dob'], errors='ignore')
        
        logger.info("Created age features")
        
        return df
    
    def create_height_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create height-based features (safe - static attribute)."""
        # Height difference
        df['height_diff'] = df['player1_height'] - df['player2_height']
        
        # Height advantage flag
        df['player1_taller'] = (df['height_diff'] > 0).astype(int)
        
        logger.info("Created height features")
        
        return df
    
    def create_hand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create hand-based features (safe - static attribute)."""
        # Encode hands
        df['player1_hand_encoded'] = df['player1_hand'].map({'R': 1, 'L': 0, 'U': 2})
        df['player2_hand_encoded'] = df['player2_hand'].map({'R': 1, 'L': 0, 'U': 2})
        
        # Same hand flag
        df['same_hand'] = (df['player1_hand'] == df['player2_hand']).astype(int)
        
        # Left-handed advantage (lefties often have advantage)
        df['player1_lefty'] = (df['player1_hand'] == 'L').astype(int)
        df['player2_lefty'] = (df['player2_hand'] == 'L').astype(int)
        df['lefty_vs_righty'] = ((df['player1_lefty'] == 1) & (df['player2_lefty'] == 0)).astype(int)
        
        logger.info("Created hand features")
        
        return df
    
    def create_surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode surface type."""
        # One-hot encode surface
        surface_dummies = pd.get_dummies(df['surface'], prefix='surface', dtype=int)
        df = pd.concat([df, surface_dummies], axis=1)
        
        logger.info("Created surface features")
        
        return df
    
    def create_tournament_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tournament level features."""
        if 'tournament_level' in df.columns:
            # One-hot encode tournament level
            tournament_dummies = pd.get_dummies(df['tournament_level'], prefix='tournament', dtype=int)
            df = pd.concat([df, tournament_dummies], axis=1)
            logger.info("Created tournament features")
        
        return df
    
    def create_elo_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from Elo ratings.
        
        Elo ratings are SAFE - they're computed chronologically before each match.
        """
        # Elo ratio (general)
        df['elo_ratio'] = df['player1_elo_before'] / (df['player2_elo_before'] + 1e-6)
        
        # Surface Elo ratio
        df['surface_elo_ratio'] = df['player1_surface_elo_before'] / (df['player2_surface_elo_before'] + 1e-6)
        
        # Tournament Elo ratio
        if 'player1_tournament_elo_before' in df.columns:
            df['tournament_elo_ratio'] = df['player1_tournament_elo_before'] / (df['player2_tournament_elo_before'] + 1e-6)
        
        # Combined Elo score (weighted average)
        df['combined_elo_p1'] = (
            0.4 * df['player1_elo_before'] + 
            0.4 * df['player1_surface_elo_before'] +
            0.2 * df.get('player1_tournament_elo_before', df['player1_elo_before'])
        )
        df['combined_elo_p2'] = (
            0.4 * df['player2_elo_before'] + 
            0.4 * df['player2_surface_elo_before'] +
            0.2 * df.get('player2_tournament_elo_before', df['player2_elo_before'])
        )
        df['combined_elo_diff'] = df['combined_elo_p1'] - df['combined_elo_p2']
        
        logger.info("Created Elo-based features")
        
        return df
    
    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create form-based features.
        
        Form is SAFE - computed from last 10 matches before each match.
        Now using individual match outcomes (match_1 to match_10) instead of aggregated percentage.
        
        Creates:
        - Individual match outcome features for both players (20 features: player1_match_1..10, player2_match_1..10)
        - Aggregated form metrics for derived features:
          - player1_form_wins, player2_form_wins: count of wins in last 10
          - form_win_diff: difference in win counts
          - player1_recent_form, player2_recent_form: wins in last 3 matches
          - recent_form_diff: difference in recent form
        """
        # The individual match outcome columns are already in the dataframe from calculator.py
        # They are: player1_match_1 through player1_match_10, player2_match_1 through player2_match_10
        # Values: 1=Won, 0=Lost, NaN=No match data
        
        # Create aggregated features from individual matches for derived metrics
        # Count total wins (handling NaN values)
        player1_match_cols = [f'player1_match_{i}' for i in range(1, 11)]
        player2_match_cols = [f'player2_match_{i}' for i in range(1, 11)]
        
        df['player1_form_wins'] = df[player1_match_cols].sum(axis=1, skipna=True)
        df['player2_form_wins'] = df[player2_match_cols].sum(axis=1, skipna=True)
        df['form_win_diff'] = df['player1_form_wins'] - df['player2_form_wins']
        
        # Recent form (last 3 matches) - more weight on very recent performance
        player1_recent_cols = [f'player1_match_{i}' for i in range(1, 4)]
        player2_recent_cols = [f'player2_match_{i}' for i in range(1, 4)]
        
        df['player1_recent_form'] = df[player1_recent_cols].sum(axis=1, skipna=True)
        df['player2_recent_form'] = df[player2_recent_cols].sum(axis=1, skipna=True)
        df['recent_form_diff'] = df['player1_recent_form'] - df['player2_recent_form']
        
        # Form momentum (is player improving?) - use win count instead of percentage
        # High form wins with high Elo = in-form strong player
        df['player1_momentum'] = df['player1_form_wins'] * (df['player1_elo_before'] / 1500)
        df['player2_momentum'] = df['player2_form_wins'] * (df['player2_elo_before'] / 1500)
        df['momentum_diff'] = df['player1_momentum'] - df['player2_momentum']
        
        logger.info("Created form features: 20 individual match outcomes + 9 aggregated metrics")
        
        return df
    
    def create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create head-to-head features.
        
        H2H is SAFE - computed from matches before each match.
        
        Creates:
        - player1_h2h_win_rate: Player 1's win rate against Player 2
        - player2_h2h_win_rate: Player 2's win rate against Player 1
        - h2h_win_rate_diff: Difference in h2h win rates
        - player1_h2h_wins: Number of wins Player 1 has against Player 2
        - player2_h2h_wins: Number of wins Player 2 has against Player 1
        - h2h_matches: Total matches between the two players
        - h2h_experience: 1 if players have faced each other before, 0 otherwise
        """
        # The h2h columns are already in the dataframe from calculator.py
        # They are: player1_h2h_wins, player1_h2h_matches, player1_h2h_win_rate,
        #           player2_h2h_wins, player2_h2h_matches, player2_h2h_win_rate
        
        # Create derived features
        df['h2h_win_rate_diff'] = df['player1_h2h_win_rate'] - df['player2_h2h_win_rate']
        df['h2h_experience'] = (df['player1_h2h_matches'] > 0).astype(int)
        
        # Interaction features: h2h advantage combined with Elo
        # If player has both Elo and h2h advantage, it's a strong signal
        df['player1_h2h_elo_advantage'] = df['player1_h2h_win_rate'] * (df['player1_elo_before'] / 1500)
        df['player2_h2h_elo_advantage'] = df['player2_h2h_win_rate'] * (df['player2_elo_before'] / 1500)
        df['h2h_elo_advantage_diff'] = df['player1_h2h_elo_advantage'] - df['player2_h2h_elo_advantage']
        
        logger.info("Created head-to-head features: 6 base + 4 derived = 10 total")
        
        return df
    
    def create_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fatigue-based features (already calculated by calculator.py).
        
        The calculator already provides:
        - player1_days_since_last, player2_days_since_last
        - rest_advantage
        - player1_fatigued, player2_fatigued
        - both_rested
        
        This method adds derived features for better model learning.
        """
        # Fatigue interaction with Elo (fatigue hurts high-ranked players more)
        df['player1_fatigue_impact'] = df['player1_fatigued'] * (df['player1_elo_before'] / 1500)
        df['player2_fatigue_impact'] = df['player2_fatigued'] * (df['player2_elo_before'] / 1500)
        df['fatigue_impact_diff'] = df['player1_fatigue_impact'] - df['player2_fatigue_impact']
        
        # Rest quality (log scale for diminishing returns: 2->3 days matters more than 10->11)
        df['player1_rest_quality'] = np.log1p(df['player1_days_since_last'])
        df['player2_rest_quality'] = np.log1p(df['player2_days_since_last'])
        df['rest_quality_diff'] = df['player1_rest_quality'] - df['player2_rest_quality']
        
        logger.info("Created fatigue features: 6 base + 6 derived = 12 total")
        
        return df
    
    def create_tournament_progression_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tournament progression features (already calculated by calculator.py).
        
        The calculator already provides:
        - player1_matches_in_tournament, player2_matches_in_tournament
        - tournament_experience_diff
        - player1_deep_run, player2_deep_run
        
        This method adds derived features.
        """
        # Tournament fatigue vs momentum (deep run = tired OR hot streak)
        # Interaction with form: deep run + good form = momentum, deep run + bad form = fatigue
        df['player1_tournament_momentum'] = df['player1_matches_in_tournament'] * df['player1_form_wins']
        df['player2_tournament_momentum'] = df['player2_matches_in_tournament'] * df['player2_form_wins']
        df['tournament_momentum_diff'] = df['player1_tournament_momentum'] - df['player2_tournament_momentum']
        
        # Experience in tournament rounds (squared to emphasize deep runs: 3 matches >> 1 match)
        df['player1_tournament_rounds'] = df['player1_matches_in_tournament'] ** 2
        df['player2_tournament_rounds'] = df['player2_matches_in_tournament'] ** 2
        df['tournament_rounds_diff'] = df['player1_tournament_rounds'] - df['player2_tournament_rounds']
        
        logger.info("Created tournament progression features: 5 base + 6 derived = 11 total")
        
        return df
    
    def create_surface_advantage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create surface specialization advantage features.
        
        Measures how much better/worse each player is on the current surface
        compared to their overall ability. This is different from just comparing
        surface Elos directly.
        
        Example: Nadal clay Elo 2200, general 2000 = +200 advantage
                 Federer clay Elo 1950, general 2000 = -50 disadvantage
        """
        # Surface advantage (how much better on this surface vs overall)
        df['player1_surface_advantage'] = (
            df['player1_surface_elo_before'] - df['player1_elo_before']
        )
        df['player2_surface_advantage'] = (
            df['player2_surface_elo_before'] - df['player2_elo_before']
        )
        df['surface_advantage_diff'] = (
            df['player1_surface_advantage'] - df['player2_surface_advantage']
        )
        
        # Surface specialist flag (significantly better on this surface)
        df['player1_surface_specialist'] = (df['player1_surface_advantage'] > 50).astype(int)
        df['player2_surface_specialist'] = (df['player2_surface_advantage'] > 50).astype(int)
        
        # Surface advantage percentage (relative to general Elo)
        df['player1_surface_advantage_pct'] = (
            df['player1_surface_advantage'] / (df['player1_elo_before'] + 1e-6)
        )
        df['player2_surface_advantage_pct'] = (
            df['player2_surface_advantage'] / (df['player2_elo_before'] + 1e-6)
        )
        df['surface_advantage_pct_diff'] = (
            df['player1_surface_advantage_pct'] - df['player2_surface_advantage_pct']
        )
        
        logger.info("Created surface advantage features: 9 total")
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable: 1 if player1 wins, 0 if player2 wins.
        """
        df['target'] = (df['winner'] == df['player1']).astype(int)
        
        logger.info(f"Created target variable. Class distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Special handling for match outcome columns (player1_match_N, player2_match_N):
        - NaN means "no match data available" (player hasn't played N matches yet)
        - Fill with 0 to indicate neutral/no information
        
        Other numeric features filled with median.
        """
        # First, handle match outcome columns specially
        match_outcome_cols = []
        for i in range(1, 11):
            match_outcome_cols.extend([f'player1_match_{i}', f'player2_match_{i}'])
        
        for col in match_outcome_cols:
            if col in df.columns and df[col].isna().sum() > 0:
                na_count = df[col].isna().sum()
                df[col] = df[col].fillna(0)  # 0 = no data (neutral)
                logger.info(f"Filled {na_count} missing values in {col} with 0 (no match data)")
        
        # For other numeric features, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in match_outcome_cols and df[col].isna().sum() > 0:
                median_val = df[col].median()
                na_count = df[col].isna().sum()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {na_count} missing values in {col} with median: {median_val}")
        
        # For categorical features, fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns to use for modeling.
        
        Excludes metadata and target variable. Only includes numeric columns.
        """
        exclude_cols = [
            # Metadata
            'date', 'player1', 'player2', 'winner', 'stadium', 'description', 'nation',
            'player1_id', 'player2_id', 'player1_country', 'player2_country',
            'player1_hand', 'player2_hand',  # Use encoded versions instead
            'surface', 'tournament_level',  # Use one-hot encoded versions instead
            
            # Target variable
            'target',
            
            # Post-match data (not available at prediction time)
            'player1_elo_after', 'player2_elo_after',
            'player1_surface_elo_after', 'player2_surface_elo_after',
            'player1_tournament_elo_after', 'player2_tournament_elo_after',
        ]
        
        # Get all columns that are numeric
        feature_cols = []
        for col in df.columns:
            # Skip excluded columns
            if col in exclude_cols:
                continue
            
            # Only include if numeric (int or float)
            if df[col].dtype in [np.int64, np.int32, np.float64, np.float32, 'int64', 'int32', 'float64', 'float32', int, float]:
                feature_cols.append(col)
        
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        
        return feature_cols
    
    def prepare_features(
        self,
        scale: bool = True,
        return_feature_names: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[List[str]]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            scale: Whether to scale features
            return_feature_names: Whether to return feature names
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: (optional) List of feature names
        """
        # Load and merge data
        df = self.load_data()
        df = self.merge_player_attributes()
        
                # Create all features
        df = self.create_age_features(df)
        df = self.create_height_features(df)
        df = self.create_hand_features(df)
        df = self.create_surface_features(df)
        df = self.create_tournament_features(df)
        df = self.create_elo_based_features(df)
        df = self.create_form_features(df)
        df = self.create_fatigue_features(df)
        df = self.create_tournament_progression_features(df)
        df = self.create_surface_advantage_features(df)
        df = self.create_h2h_features(df)
        df = self.create_target(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        
        # Extract X and y
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
            logger.info("Scaled features using StandardScaler")
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        if return_feature_names:
            return X, y, feature_cols
        return X, y
    
    def save_preprocessed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str = "data/processed"
    ):
        """Save preprocessed features and target."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine if train or test based on input filename
        is_train = 'train' in str(self.match_data_path)
        prefix = 'train' if is_train else 'test'
        
        X_path = output_path / f"{prefix}_features.csv"
        y_path = output_path / f"{prefix}_target.csv"
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False, header=['target'])
        
        logger.info(f"Saved features to: {X_path}")
        logger.info(f"Saved target to: {y_path}")


def main():
    """Example usage."""
    # Process training data
    print("\n" + "="*60)
    print("PROCESSING TRAINING DATA")
    print("="*60)
    train_fe = FeatureEngineering(
        match_data_path="data/processed/train_matches.csv",
        player_info_path="data/processed/player_info.csv"
    )
    X_train, y_train, feature_names = train_fe.prepare_features(
        scale=True,
        return_feature_names=True
    )
    train_fe.save_preprocessed_data(X_train, y_train)
    
    print(f"\nFeature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    # Process test data
    print("\n" + "="*60)
    print("PROCESSING TEST DATA")
    print("="*60)
    test_fe = FeatureEngineering(
        match_data_path="data/processed/test_matches.csv",
        player_info_path="data/processed/player_info.csv"
    )
    X_test, y_test = test_fe.prepare_features(scale=True)
    test_fe.save_preprocessed_data(X_test, y_test)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Target balance (train): {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Target balance (test): {y_test.value_counts(normalize=True).to_dict()}")


if __name__ == "__main__":
    main()