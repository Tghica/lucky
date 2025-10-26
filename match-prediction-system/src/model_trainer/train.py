import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import joblib
import json
import logging
from typing import Dict, Tuple, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisMatchPredictor:
    """
    Tennis match prediction model trainer.
    Supports Gradient Boosting (baseline) and XGBoost (advanced).
    """
    
    def __init__(
        self,
        model_type: str = "gradientboosting",
        random_state: int = 42
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: "gradientboosting" or "xgboost"
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.training_history = {}
        self.feature_names = None
        
    def create_model(self, params: Optional[Dict] = None) -> object:
        """
        Create model with given parameters.
        
        Args:
            params: Model hyperparameters
        
        Returns:
            Model instance
        """
        if self.model_type == "gradientboosting":
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8,
                'random_state': self.random_state,
                'verbose': 0  # Reduced verbosity
            }
            if params:
                default_params.update(params)
            
            logger.info(f"Creating Gradient Boosting model with params: {default_params}")
            return GradientBoostingClassifier(**default_params)
        
        elif self.model_type == "xgboost":
            default_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 20,
                'verbosity': 1
            }
            if params:
                default_params.update(params)
            
            logger.info(f"Creating XGBoost model with params: {default_params}")
            return xgb.XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> object:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for XGBoost early stopping)
            y_val: Validation targets (optional, for XGBoost early stopping)
            params: Model hyperparameters
        
        Returns:
            Trained model
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.model_type.upper()} model")
        logger.info(f"{'='*60}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Handle both Series and ndarray for class distribution
        if hasattr(y_train, 'value_counts'):
            class_dist = y_train.value_counts().to_dict()
        else:
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create model
        self.model = self.create_model(params)
        
        # Train
        start_time = time.time()
        
        if self.model_type == "xgboost" and X_val is not None and y_val is not None:
            # XGBoost with early stopping
            logger.info(f"Using validation set for early stopping (size: {len(X_val)})")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            # Get best iteration
            best_iteration = self.model.best_iteration
            logger.info(f"Best iteration: {best_iteration}")
            
        else:
            # Standard training
            self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Store training info
        self.training_history = {
            'model_type': self.model_type,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'training_time': training_time,
            'params': self.model.get_params()
        }
        
        return self.model
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            params: Model hyperparameters
        
        Returns:
            Dictionary with CV results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validation: {cv}-fold")
        logger.info(f"{'='*60}")
        
        # Create model
        model = self.create_model(params)
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Cross-validation
        start_time = time.time()
        cv_scores = cross_val_score(
            model, X, y,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        cv_time = time.time() - start_time
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_time': cv_time
        }
        
        logger.info(f"\nCV Results:")
        logger.info(f"  Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        logger.info(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
        logger.info(f"  Time: {cv_time:.2f} seconds")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == "gradientboosting":
            importance = self.model.feature_importances_
        elif self.model_type == "xgboost":
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not available for {self.model_type}")
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def save_model(self, output_dir: str = "models/saved_models"):
        """
        Save trained model and metadata.
        
        Args:
            output_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Model filename
        model_file = output_path / f"{self.model_type}_model.pkl"
        
        # Save model
        joblib.dump(self.model, model_file)
        logger.info(f"Model saved to: {model_file}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'params': self.model.get_params()
        }
        
        metadata_file = output_path / f"{self.model_type}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to: {metadata_file}")
        
        # Save feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.get_feature_importance(top_n=len(self.feature_names))
            importance_file = output_path / f"{self.model_type}_feature_importance.csv"
            feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Feature importance saved to: {importance_file}")
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        # Try to load metadata
        metadata_path = str(model_path).replace('_model.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.training_history = metadata.get('training_history', {})
                logger.info(f"Metadata loaded from: {metadata_path}")


def main():
    """Train baseline and advanced models."""
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = pd.read_csv('data/processed/train_features.csv')
    y_train = pd.read_csv('data/processed/train_target.csv').values.ravel()
    
    X_test = pd.read_csv('data/processed/test_features.csv')
    y_test = pd.read_csv('data/processed/test_target.csv').values.ravel()
    
    # Align features (test may be missing some one-hot columns)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]  # Ensure same order
    
    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Test data: {X_test.shape}")
    
    # Split training data for validation (for XGBoost early stopping)
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # ========================================
    # BASELINE: Gradient Boosting
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: GRADIENT BOOSTING BASELINE")
    print("="*70)
    
    gb_predictor = TennisMatchPredictor(model_type="gradientboosting")
    
    # Cross-validation first
    gb_cv_results = gb_predictor.cross_validate(X_train, y_train, cv=5)
    
    # Train on full training set
    gb_predictor.train(X_train, y_train)
    
    # Evaluate on test set
    from src.model_trainer.evaluation import ModelEvaluator
    evaluator = ModelEvaluator(gb_predictor.model, X_test, y_test)
    gb_results = evaluator.evaluate_all()
    
    # Save model
    gb_predictor.save_model()
    
    # Feature importance
    print("\nTop 20 Most Important Features (Gradient Boosting):")
    print(gb_predictor.get_feature_importance(top_n=20).to_string(index=False))
    
    # ========================================
    # ADVANCED: XGBoost
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: XGBOOST ADVANCED MODEL")
    print("="*70)
    
    xgb_predictor = TennisMatchPredictor(model_type="xgboost")
    
    # Cross-validation first
    xgb_cv_results = xgb_predictor.cross_validate(X_train, y_train, cv=5)
    
    # Train with early stopping
    xgb_predictor.train(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate on test set
    evaluator = ModelEvaluator(xgb_predictor.model, X_test, y_test)
    xgb_results = evaluator.evaluate_all()
    
    # Save model
    xgb_predictor.save_model()
    
    # Feature importance
    print("\nTop 20 Most Important Features (XGBoost):")
    print(xgb_predictor.get_feature_importance(top_n=20).to_string(index=False))
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nGradient Boosting:")
    print(f"  CV Accuracy: {gb_cv_results['mean_accuracy']:.4f} (+/- {gb_cv_results['std_accuracy']:.4f})")
    print(f"  Test Accuracy: {gb_results['accuracy']:.4f}")
    print(f"  Test AUC-ROC: {gb_results['auc_roc']:.4f}")
    
    print(f"\nXGBoost:")
    print(f"  CV Accuracy: {xgb_cv_results['mean_accuracy']:.4f} (+/- {xgb_cv_results['std_accuracy']:.4f})")
    print(f"  Test Accuracy: {xgb_results['accuracy']:.4f}")
    print(f"  Test AUC-ROC: {xgb_results['auc_roc']:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()