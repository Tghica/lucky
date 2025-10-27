"""
Test LightGBM as an alternative to XGBoost.
LightGBM often trains faster and can achieve similar or better accuracy.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import time
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with tuned parameters."""
    logger.info("\n" + "="*80)
    logger.info("XGBOOST (TUNED PARAMETERS)")
    logger.info("="*80)
    
    # Load tuned parameters
    try:
        with open('models/saved_models/xgboost_tuned_params.json', 'r') as f:
            params = json.load(f)
        logger.info("Using tuned parameters")
    except FileNotFoundError:
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        logger.info("Using default parameters")
    
    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    logger.info("Training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    logger.info(f"Training time: {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"\nTest Accuracy: {acc:.4f}")
    logger.info(f"Test AUC-ROC:  {auc:.4f}")
    
    return {
        'model': 'XGBoost',
        'accuracy': acc,
        'auc': auc,
        'train_time': train_time
    }


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with optimized parameters."""
    logger.info("\n" + "="*80)
    logger.info("LIGHTGBM")
    logger.info("="*80)
    
    # LightGBM parameters (translated from XGBoost tuned params)
    params = {
        'n_estimators': 700,
        'learning_rate': 0.03,
        'max_depth': 3,
        'num_leaves': 7,  # 2^max_depth - 1
        'min_child_samples': 20,  # similar to min_child_weight
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 1.5,
        'min_split_gain': 0.2,  # similar to gamma
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True  # Faster for wide datasets
    }
    
    model = lgb.LGBMClassifier(**params)
    
    logger.info("Training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    logger.info(f"Training time: {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"\nTest Accuracy: {acc:.4f}")
    logger.info(f"Test AUC-ROC:  {auc:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=['Player2 Win', 'Player1 Win']
    )
    logger.info("\n" + report)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Features:")
    logger.info("\n" + feature_importance.head(15).to_string(index=False))
    
    return {
        'model': 'LightGBM',
        'accuracy': acc,
        'auc': auc,
        'train_time': train_time
    }


def train_lightgbm_fast(X_train, y_train, X_test, y_test):
    """Train LightGBM with speed-optimized parameters."""
    logger.info("\n" + "="*80)
    logger.info("LIGHTGBM (FAST MODE)")
    logger.info("="*80)
    
    # Speed-optimized parameters
    params = {
        'n_estimators': 300,  # Fewer trees
        'learning_rate': 0.05,  # Higher learning rate
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True,
        'boosting_type': 'gbdt'  # Gradient Boosting Decision Tree
    }
    
    model = lgb.LGBMClassifier(**params)
    
    logger.info("Training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    logger.info(f"Training time: {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    logger.info(f"\nTest Accuracy: {acc:.4f}")
    logger.info(f"Test AUC-ROC:  {auc:.4f}")
    
    return {
        'model': 'LightGBM (Fast)',
        'accuracy': acc,
        'auc': auc,
        'train_time': train_time
    }


def main():
    """Compare XGBoost and LightGBM."""
    logger.info("="*80)
    logger.info("🚀 XGBOOST vs LIGHTGBM COMPARISON")
    logger.info("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Train models
    results = []
    
    # XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_results)
    
    # LightGBM (default)
    lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
    results.append(lgb_results)
    
    # LightGBM (fast)
    lgb_fast_results = train_lightgbm_fast(X_train, y_train, X_test, y_test)
    results.append(lgb_fast_results)
    
    # Comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Model':<20} {'Accuracy':>12} {'AUC-ROC':>12} {'Time (s)':>12} {'Speed':>10}")
    logger.info("-"*80)
    
    baseline_time = xgb_results['train_time']
    for result in results:
        speedup = baseline_time / result['train_time']
        logger.info(
            f"{result['model']:<20} "
            f"{result['accuracy']:>12.4f} "
            f"{result['auc']:>12.4f} "
            f"{result['train_time']:>12.2f} "
            f"{speedup:>9.1f}x"
        )
    
    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    logger.info(f"\n🏆 Best Accuracy: {best['model']} ({best['accuracy']:.4f})")
    
    best_auc = max(results, key=lambda x: x['auc'])
    logger.info(f"🏆 Best AUC-ROC: {best_auc['model']} ({best_auc['auc']:.4f})")
    
    fastest = min(results, key=lambda x: x['train_time'])
    logger.info(f"⚡ Fastest: {fastest['model']} ({fastest['train_time']:.2f}s)")
    
    logger.info("\n✓ COMPARISON COMPLETE!")


if __name__ == '__main__':
    main()
