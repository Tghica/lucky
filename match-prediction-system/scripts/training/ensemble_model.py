"""
Ensemble model combining multiple classifiers for improved prediction accuracy.
Uses stacking with Gradient Boosting, XGBoost, and Random Forest as base models,
and Logistic Regression as the meta-learner.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier
)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load preprocessed training and test data."""
    logger.info("Loading data...")
    X_train = pd.read_csv('data/processed/train_features.csv')
    y_train = pd.read_csv('data/processed/train_target.csv').values.ravel()
    X_test = pd.read_csv('data/processed/test_features.csv')
    y_test = pd.read_csv('data/processed/test_target.csv').values.ravel()
    
    # Align features
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_tuned_params():
    """Load tuned hyperparameters from JSON file."""
    try:
        with open('models/saved_models/xgboost_tuned_params.json', 'r') as f:
            tuned_params = json.load(f)
        logger.info("Loaded tuned XGBoost parameters")
        return tuned_params
    except FileNotFoundError:
        logger.warning("Tuned params not found, using defaults")
        return None


def create_base_models(tuned_xgb_params=None):
    """
    Create base models for ensemble.
    
    Returns:
        Dictionary of base models
    """
    logger.info("\nCreating base models...")
    
    # Gradient Boosting (optimized params)
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    logger.info("  ✓ Gradient Boosting created")
    
    # XGBoost (use tuned params if available)
    if tuned_xgb_params:
        xgb_model = xgb.XGBClassifier(
            **tuned_xgb_params,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        logger.info("  ✓ XGBoost created (tuned parameters)")
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        logger.info("  ✓ XGBoost created (default parameters)")
    
    # Random Forest (new model)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    logger.info("  ✓ Random Forest created")
    
    return {
        'gradient_boosting': gb_model,
        'xgboost': xgb_model,
        'random_forest': rf_model
    }


def create_voting_ensemble(base_models):
    """
    Create voting ensemble (soft voting).
    
    Args:
        base_models: Dictionary of base models
    
    Returns:
        Voting ensemble model
    """
    logger.info("\nCreating Voting Ensemble...")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('gb', base_models['gradient_boosting']),
            ('xgb', base_models['xgboost']),
            ('rf', base_models['random_forest'])
        ],
        voting='soft',  # Use probability averages
        n_jobs=-1
    )
    
    logger.info("  ✓ Voting Ensemble created (soft voting)")
    return voting_clf


def create_stacking_ensemble(base_models):
    """
    Create stacking ensemble with Logistic Regression meta-learner.
    
    Args:
        base_models: Dictionary of base models
    
    Returns:
        Stacking ensemble model
    """
    logger.info("\nCreating Stacking Ensemble...")
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('gb', base_models['gradient_boosting']),
            ('xgb', base_models['xgboost']),
            ('rf', base_models['random_forest'])
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        cv=5,  # 5-fold cross-validation for meta-features
        n_jobs=-1
    )
    
    logger.info("  ✓ Stacking Ensemble created (LogisticRegression meta-learner)")
    return stacking_clf


def evaluate_model(name, model, X_train, y_train, X_test, y_test, train_time):
    """Evaluate model performance."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{name.upper()} EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Training time: {train_time:.2f}s")
    
    # Training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    logger.info(f"\nTraining Set:")
    logger.info(f"  Accuracy: {train_acc:.4f}")
    logger.info(f"  AUC-ROC:  {train_auc:.4f}")
    
    # Test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    logger.info(f"\nTest Set:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  AUC-ROC:  {test_auc:.4f}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(
        y_test, y_test_pred,
        target_names=['Player2 Win', 'Player1 Win']
    )
    logger.info("\n" + report)
    
    return {
        'train_accuracy': train_acc,
        'train_auc': train_auc,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'training_time': train_time
    }


def save_ensemble_model(model, name, results):
    """Save ensemble model and results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f'models/saved_models/{name}_ensemble.pkl'
    joblib.dump(model, model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")
    
    # Save results
    results_path = f'models/saved_models/{name}_ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to: {results_path}")


def main():
    """Main ensemble training pipeline."""
    logger.info("=" * 80)
    logger.info("🚀 ENSEMBLE MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Load tuned parameters
    tuned_params = load_tuned_params()
    
    # Create base models
    base_models = create_base_models(tuned_params)
    
    # Store all results
    all_results = {}
    baseline_acc = 0.6487  # From tuned XGBoost
    
    # ========================================
    # 1. TRAIN INDIVIDUAL BASE MODELS
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: TRAINING INDIVIDUAL BASE MODELS")
    logger.info("=" * 80)
    
    for name, model in base_models.items():
        logger.info(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        results = evaluate_model(name, model, X_train, y_train, X_test, y_test, train_time)
        all_results[name] = results
    
    # ========================================
    # 2. TRAIN VOTING ENSEMBLE
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING VOTING ENSEMBLE")
    logger.info("=" * 80)
    
    # Recreate base models (to avoid refitting)
    base_models_voting = create_base_models(tuned_params)
    voting_ensemble = create_voting_ensemble(base_models_voting)
    
    logger.info("\nTraining Voting Ensemble...")
    start_time = time.time()
    voting_ensemble.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    voting_results = evaluate_model(
        'Voting Ensemble',
        voting_ensemble,
        X_train, y_train, X_test, y_test,
        train_time
    )
    all_results['voting_ensemble'] = voting_results
    save_ensemble_model(voting_ensemble, 'voting', voting_results)
    
    # ========================================
    # 3. TRAIN STACKING ENSEMBLE
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING STACKING ENSEMBLE")
    logger.info("=" * 80)
    
    # Recreate base models
    base_models_stacking = create_base_models(tuned_params)
    stacking_ensemble = create_stacking_ensemble(base_models_stacking)
    
    logger.info("\nTraining Stacking Ensemble (this may take a few minutes)...")
    start_time = time.time()
    stacking_ensemble.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    stacking_results = evaluate_model(
        'Stacking Ensemble',
        stacking_ensemble,
        X_train, y_train, X_test, y_test,
        train_time
    )
    all_results['stacking_ensemble'] = stacking_results
    save_ensemble_model(stacking_ensemble, 'stacking', stacking_results)
    
    # ========================================
    # 4. COMPARISON SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 80)
    
    logger.info(f"\nBaseline (Tuned XGBoost):")
    logger.info(f"  Accuracy: {baseline_acc:.4f}")
    logger.info(f"  AUC-ROC:  0.7109")
    
    logger.info(f"\n{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'Time (s)':>10} {'vs Baseline':>12}")
    logger.info("-" * 80)
    
    for name, results in all_results.items():
        acc_diff = (results['test_accuracy'] - baseline_acc) * 100
        display_name = name.replace('_', ' ').title()
        logger.info(
            f"{display_name:<25} "
            f"{results['test_accuracy']:>10.4f} "
            f"{results['test_auc']:>10.4f} "
            f"{results['training_time']:>10.1f} "
            f"{acc_diff:>+11.2f}%"
        )
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['test_accuracy'])
    best_name = best_model[0].replace('_', ' ').title()
    best_acc = best_model[1]['test_accuracy']
    improvement = (best_acc - baseline_acc) * 100
    
    logger.info("\n" + "=" * 80)
    logger.info(f"🏆 BEST MODEL: {best_name}")
    logger.info(f"   Accuracy: {best_acc:.4f} (Improvement: {improvement:+.2f}%)")
    logger.info(f"   AUC-ROC: {best_model[1]['test_auc']:.4f}")
    logger.info("=" * 80)
    
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n✓ ENSEMBLE TRAINING COMPLETE!")


if __name__ == '__main__':
    main()
