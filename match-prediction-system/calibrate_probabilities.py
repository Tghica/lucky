"""
Probability Calibration for Tennis Match Prediction.

Well-calibrated probabilities are crucial for betting decisions:
- If model says 70% win probability, player should win ~70% of such matches
- Uncalibrated models often produce overconfident or underconfident predictions
- CalibratedClassifierCV uses isotonic regression or Platt scaling to fix this

This script:
1. Trains base models (XGBoost, LightGBM, Stacking Ensemble)
2. Calibrates their probabilities using cross-validation
3. Evaluates calibration quality with reliability diagrams and Brier score
4. Compares calibrated vs uncalibrated predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import logging
from pathlib import Path

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


def get_base_model(model_type='xgboost'):
    """Get base model (uncalibrated)."""
    if model_type == 'xgboost':
        try:
            with open('models/saved_models/xgboost_tuned_params.json', 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6}
        
        return xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)
    
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': 700,
            'learning_rate': 0.03,
            'max_depth': 3,
            'num_leaves': 7,
            'min_child_samples': 20,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.5,
            'random_state': 42,
            'verbosity': -1
        }
        return lgb.LGBMClassifier(**params)
    
    elif model_type == 'stacking':
        try:
            return joblib.load('models/saved_models/stacking_ensemble.pkl')
        except FileNotFoundError:
            logger.warning("Stacking ensemble not found, using XGBoost")
            return get_base_model('xgboost')
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_calibration(y_true, y_proba, model_name):
    """
    Evaluate calibration quality.
    
    Metrics:
    - Brier Score: Mean squared error of probabilities (lower is better)
    - Log Loss: Negative log-likelihood (lower is better)
    - ECE (Expected Calibration Error): Average gap between confidence and accuracy
    """
    brier = brier_score_loss(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    
    # Calculate Expected Calibration Error (ECE)
    n_bins = 10
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # ECE: weighted average of |accuracy - confidence| in each bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    logger.info(f"\n{model_name} Calibration Metrics:")
    logger.info(f"  Brier Score: {brier:.4f} (lower is better)")
    logger.info(f"  Log Loss:    {logloss:.4f} (lower is better)")
    logger.info(f"  ECE:         {ece:.4f} (lower is better, <0.05 is good)")
    
    return {
        'brier_score': brier,
        'log_loss': logloss,
        'ece': ece,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }


def plot_calibration_curve(calibration_data, output_path='models/saved_models/calibration_plot.png'):
    """Plot reliability diagram comparing calibrated vs uncalibrated."""
    plt.figure(figsize=(10, 10))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    
    # Plot each model
    colors = ['blue', 'red', 'green', 'orange']
    for idx, (name, data) in enumerate(calibration_data.items()):
        plt.plot(
            data['mean_predicted_value'],
            data['fraction_of_positives'],
            marker='o',
            label=f"{name} (ECE={data['ece']:.3f})",
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=8
        )
    
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.title('Calibration Plot (Reliability Diagram)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info(f"Calibration plot saved to: {output_path}")
    plt.close()


def main():
    """Main calibration pipeline."""
    logger.info("="*80)
    logger.info("🎯 PROBABILITY CALIBRATION")
    logger.info("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    results = {}
    calibration_data = {}
    
    # Test multiple models and calibration methods
    models_to_test = [
        ('XGBoost', 'xgboost'),
        ('LightGBM', 'lightgbm'),
    ]
    
    for model_name, model_type in models_to_test:
        logger.info("\n" + "="*80)
        logger.info(f"{model_name.upper()}")
        logger.info("="*80)
        
        # ========================================
        # 1. Uncalibrated Model
        # ========================================
        logger.info("\n[1/2] Training uncalibrated model...")
        base_model = get_base_model(model_type)
        base_model.fit(X_train, y_train)
        
        y_proba_uncalib = base_model.predict_proba(X_test)[:, 1]
        y_pred_uncalib = base_model.predict(X_test)
        
        acc_uncalib = accuracy_score(y_test, y_pred_uncalib)
        auc_uncalib = roc_auc_score(y_test, y_proba_uncalib)
        
        logger.info(f"Uncalibrated - Accuracy: {acc_uncalib:.4f}, AUC: {auc_uncalib:.4f}")
        
        uncalib_metrics = evaluate_calibration(y_test, y_proba_uncalib, f"{model_name} (Uncalibrated)")
        calibration_data[f"{model_name} Uncalibrated"] = uncalib_metrics
        
        # ========================================
        # 2. Calibrated Model (Isotonic)
        # ========================================
        logger.info("\n[2/2] Training calibrated model (isotonic regression)...")
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method='isotonic',  # More flexible than 'sigmoid', works better with tree models
            cv=5,  # 5-fold cross-validation for calibration
            ensemble=True  # Use all CV folds (better but slower)
        )
        calibrated_model.fit(X_train, y_train)
        
        y_proba_calib = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred_calib = calibrated_model.predict(X_test)
        
        acc_calib = accuracy_score(y_test, y_pred_calib)
        auc_calib = roc_auc_score(y_test, y_proba_calib)
        
        logger.info(f"Calibrated   - Accuracy: {acc_calib:.4f}, AUC: {auc_calib:.4f}")
        
        calib_metrics = evaluate_calibration(y_test, y_proba_calib, f"{model_name} (Calibrated)")
        calibration_data[f"{model_name} Calibrated"] = calib_metrics
        
        # Store results
        results[model_name] = {
            'uncalibrated': {
                'accuracy': acc_uncalib,
                'auc': auc_uncalib,
                'brier': uncalib_metrics['brier_score'],
                'log_loss': uncalib_metrics['log_loss'],
                'ece': uncalib_metrics['ece']
            },
            'calibrated': {
                'accuracy': acc_calib,
                'auc': auc_calib,
                'brier': calib_metrics['brier_score'],
                'log_loss': calib_metrics['log_loss'],
                'ece': calib_metrics['ece']
            }
        }
        
        # Save calibrated model
        model_path = f'models/saved_models/{model_type}_calibrated.pkl'
        joblib.dump(calibrated_model, model_path)
        logger.info(f"✓ Calibrated model saved to: {model_path}")
    
    # ========================================
    # Final Comparison
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n{'Model':<25} {'Type':<15} {'Accuracy':>10} {'AUC':>10} {'Brier':>10} {'ECE':>10}")
    logger.info("-"*80)
    
    for model_name, data in results.items():
        for calib_type in ['uncalibrated', 'calibrated']:
            metrics = data[calib_type]
            display_name = f"{model_name} {calib_type.title()}"
            logger.info(
                f"{model_name:<25} {calib_type:<15} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['auc']:>10.4f} "
                f"{metrics['brier']:>10.4f} "
                f"{metrics['ece']:>10.4f}"
            )
    
    # Plot calibration curves
    plot_calibration_curve(calibration_data)
    
    # Save results
    results_path = 'models/saved_models/calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {results_path}")
    
    logger.info("\n✓ CALIBRATION COMPLETE!")
    logger.info("\nKey Insight:")
    logger.info("  Calibration improves probability reliability for betting decisions")
    logger.info("  Lower Brier score = better probability estimates")
    logger.info("  Lower ECE (<0.05) = well-calibrated probabilities")


if __name__ == '__main__':
    main()
