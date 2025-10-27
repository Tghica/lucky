"""
Hyperparameter tuning for XGBoost model using RandomizedSearchCV.
This will search for optimal parameters to maximize prediction accuracy.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
    
    # Align features (test may be missing some one-hot columns)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def create_param_grid():
    """
    Create parameter grid for RandomizedSearchCV.
    
    Parameter ranges based on best practices for XGBoost:
    - n_estimators: Number of boosting rounds (more = better but slower)
    - learning_rate: Step size shrinkage (lower = more conservative)
    - max_depth: Maximum tree depth (higher = more complex)
    - min_child_weight: Minimum sum of instance weight in a child (higher = more conservative)
    - subsample: Fraction of samples for each tree (prevents overfitting)
    - colsample_bytree: Fraction of features for each tree (prevents overfitting)
    - gamma: Minimum loss reduction for split (higher = more conservative)
    - reg_alpha: L1 regularization (lasso)
    - reg_lambda: L2 regularization (ridge)
    """
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 700],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 2, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
    }
    
    return param_grid


def run_random_search(X_train, y_train, n_iter=50, cv=5):
    """
    Run RandomizedSearchCV to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds
    
    Returns:
        best_model: Best model from search
        results: Dictionary with search results
    """
    logger.info("=" * 80)
    logger.info("STARTING HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Search iterations: {n_iter}")
    logger.info(f"Cross-validation folds: {cv}")
    
    # Create base model
    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Create parameter grid
    param_grid = create_param_grid()
    
    logger.info("\nParameter ranges:")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")
    
    # Create stratified k-fold for better class balance
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create RandomizedSearchCV
    logger.info(f"\nStarting random search (this may take 10-30 minutes)...")
    logger.info(f"Progress: Training {n_iter} parameter combinations x {cv} folds = {n_iter * cv} models total")
    logger.info(f"Each dot (.) represents 10 completed model fits:\n")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=skf,
        scoring='roc_auc',  # Optimize for AUC-ROC
        n_jobs=-1,  # Parallel processing
        verbose=3,  # Maximum verbosity
        random_state=42,
        return_train_score=True
    )
    
    # Run search with progress tracking
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    logger.info(f"\n✓ Search completed in {search_time/60:.1f} minutes")
    
    # Get results
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info("\n" + "=" * 80)
    logger.info("BEST PARAMETERS FOUND")
    logger.info("=" * 80)
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"\nBest CV AUC-ROC: {best_score:.4f}")
    
    # Get all results
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'search_time': search_time,
        'cv_results': pd.DataFrame(random_search.cv_results_)
    }
    
    return best_model, results


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test sets."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    logger.info(f"\nTraining Set:")
    logger.info(f"  Accuracy: {train_acc:.4f}")
    logger.info(f"  AUC-ROC:  {train_auc:.4f}")
    
    # Test set performance
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    logger.info(f"\nTest Set:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  AUC-ROC:  {test_auc:.4f}")
    
    # Classification report
    logger.info("\nClassification Report (Test Set):")
    report = classification_report(
        y_test, y_test_pred,
        target_names=['Player2 Win', 'Player1 Win']
    )
    logger.info("\n" + report)
    
    # Feature importance
    logger.info("\nTop 20 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']:30s}  {row['importance']:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'train_auc': train_auc,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'classification_report': report,
        'feature_importance': feature_importance
    }


def save_results(model, search_results, eval_results, baseline_accuracy=0.6474):
    """Save tuned model and results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f'models/saved_models/xgboost_tuned_model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"\n✓ Tuned model saved to: {model_path}")
    
    # Save parameters
    params_path = f'models/saved_models/xgboost_tuned_params.json'
    with open(params_path, 'w') as f:
        json.dump(search_results['best_params'], f, indent=2)
    logger.info(f"✓ Best parameters saved to: {params_path}")
    
    # Save feature importance
    importance_path = f'models/saved_models/xgboost_tuned_feature_importance.csv'
    eval_results['feature_importance'].to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to: {importance_path}")
    
    # Save detailed results
    results_summary = {
        'timestamp': timestamp,
        'baseline_accuracy': baseline_accuracy,
        'tuned_accuracy': eval_results['test_accuracy'],
        'accuracy_improvement': eval_results['test_accuracy'] - baseline_accuracy,
        'baseline_auc': 0.7090,
        'tuned_auc': eval_results['test_auc'],
        'auc_improvement': eval_results['test_auc'] - 0.7090,
        'best_params': search_results['best_params'],
        'best_cv_score': search_results['best_score'],
        'search_time_minutes': search_results['search_time'] / 60,
        'train_accuracy': eval_results['train_accuracy'],
        'train_auc': eval_results['train_auc']
    }
    
    summary_path = f'models/saved_models/xgboost_tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"✓ Tuning summary saved to: {summary_path}")
    
    # Print improvement summary
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nBaseline (default params):")
    logger.info(f"  Accuracy: {baseline_accuracy:.4f}")
    logger.info(f"  AUC-ROC:  0.7090")
    
    logger.info(f"\nTuned (optimized params):")
    logger.info(f"  Accuracy: {eval_results['test_accuracy']:.4f}")
    logger.info(f"  AUC-ROC:  {eval_results['test_auc']:.4f}")
    
    improvement = (eval_results['test_accuracy'] - baseline_accuracy) * 100
    auc_improvement = (eval_results['test_auc'] - 0.7090) * 100
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Accuracy: {improvement:+.2f}%")
    logger.info(f"  AUC-ROC:  {auc_improvement:+.2f}%")
    
    if improvement > 0:
        logger.info(f"\n🎉 SUCCESS! Model improved by {improvement:.2f}%")
    elif improvement > -0.1:
        logger.info(f"\n✓ Performance maintained (within {abs(improvement):.2f}%)")
    else:
        logger.info(f"\n⚠️  Performance decreased by {abs(improvement):.2f}%")


def main():
    """Main hyperparameter tuning pipeline."""
    logger.info("🚀 XGBOOST HYPERPARAMETER TUNING")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Run hyperparameter search
    best_model, search_results = run_random_search(
        X_train, y_train,
        n_iter=50,  # Try 50 different parameter combinations
        cv=5  # 5-fold cross-validation
    )
    
    # Evaluate best model
    eval_results = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    
    # Save everything
    save_results(best_model, search_results, eval_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ HYPERPARAMETER TUNING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nTuned model is ready for production use!")
    logger.info("You can load it with: joblib.load('models/saved_models/xgboost_tuned_model.pkl')")


if __name__ == '__main__':
    main()
