#!/usr/bin/env python3
"""
Hyperparameter tuning for XGBoost model using Optuna.

This script will:
1. Load the engineered features
2. Use Optuna to search for best hyperparameters
3. Train with different parameter combinations
4. Find the optimal configuration
5. Save the best model and parameters
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime
from pathlib import Path
import sys

# Import the feature engineering from training script
sys.path.insert(0, '.')
from scripts.training.train_model_with_confirmation import engineer_features, load_data


def create_objective(X_train, y_train, X_val, y_val):
    """Create objective function for Optuna"""
    
    def objective(trial):
        """Objective function to minimize"""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        logloss = log_loss(y_val, y_proba)
        
        # We want to maximize AUC (minimize negative AUC)
        # Combined metric: weighted average of -AUC and logloss
        score = -auc + 0.3 * logloss
        
        # Report intermediate values
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('auc', auc)
        trial.set_user_attr('logloss', logloss)
        
        return score
    
    return objective


def main():
    """Main hyperparameter tuning pipeline"""
    
    print("="*80)
    print("🎾 HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load and prepare data
    print("📂 Loading data...")
    matches, players = load_data()
    
    print("\n⚙️  Engineering features...")
    df, features = engineer_features(matches)
    
    # Flatten feature list
    all_features = []
    for category, feat_list in features.items():
        all_features.extend(feat_list)
    all_features = list(set(all_features))
    
    # Filter to only available features
    available_features = [f for f in all_features if f in df.columns]
    
    # Remove rows with missing target
    df = df[df['player1_wins'].notna()].copy()
    
    # Create X and y
    X = df[available_features].copy()
    y = df['player1_wins'].copy()
    X = X.fillna(X.median())
    
    print(f"   Total samples: {len(X):,}")
    print(f"   Total features: {X.shape[1]}")
    
    # Split: 60% train, 20% validation, 20% test (all temporal)
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Baseline with old parameters
    print("\n" + "="*80)
    print("📊 BASELINE (Old Parameters)")
    print("="*80)
    
    baseline_params = {
        'n_estimators': 700,
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    baseline_model = XGBClassifier(**baseline_params)
    baseline_model.fit(X_train, y_train)
    
    baseline_val_pred = baseline_model.predict(X_val)
    baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
    baseline_val_acc = accuracy_score(y_val, baseline_val_pred)
    baseline_val_auc = roc_auc_score(y_val, baseline_val_proba)
    
    print(f"   Validation Accuracy: {baseline_val_acc*100:.2f}%")
    print(f"   Validation AUC: {baseline_val_auc:.4f}")
    
    # Start hyperparameter tuning
    print("\n" + "="*80)
    print("🔍 STARTING HYPERPARAMETER TUNING")
    print("="*80)
    print(f"   Search space: n_estimators, max_depth, learning_rate, subsample,")
    print(f"                 colsample_bytree, gamma, min_child_weight,")
    print(f"                 reg_alpha, reg_lambda")
    print(f"   Optimization: Maximize AUC, minimize log loss")
    print(f"   Method: TPE (Tree-structured Parzen Estimator)")
    print(f"   Trials: 100")
    print("")
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='xgboost_tuning'
    )
    
    # Create objective with train/val split
    objective = create_objective(X_train, y_train, X_val, y_val)
    
    # Run optimization
    print("⚙️  Running optimization (this may take 10-20 minutes)...")
    print("   Progress updates every 10 trials:")
    print("")
    
    def callback(study, trial):
        if trial.number % 10 == 0 and trial.number > 0:
            best_trial = study.best_trial
            print(f"   Trial {trial.number:3d}: Best AUC so far = {best_trial.user_attrs['auc']:.4f} "
                  f"(accuracy={best_trial.user_attrs['accuracy']*100:.2f}%)")
    
    study.optimize(objective, n_trials=100, callbacks=[callback], show_progress_bar=False)
    
    # Get best parameters
    print("\n" + "="*80)
    print("🏆 BEST PARAMETERS FOUND")
    print("="*80)
    
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"\n   Best trial: #{best_trial.number}")
    print(f"   Validation Accuracy: {best_trial.user_attrs['accuracy']*100:.2f}%")
    print(f"   Validation AUC: {best_trial.user_attrs['auc']:.4f}")
    print(f"   Validation Log Loss: {best_trial.user_attrs['logloss']:.4f}")
    
    print(f"\n   Parameters:")
    for param, value in best_params.items():
        if param in baseline_params:
            old_value = baseline_params[param]
            change = "→" if value != old_value else "="
            print(f"      {param:20s}: {old_value:8} {change} {value}")
        else:
            print(f"      {param:20s}: {value}")
    
    # Train final model with best parameters on train+val, test on test set
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Add fixed parameters
    final_params = best_params.copy()
    final_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    })
    
    # Train on train + validation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    final_model = XGBClassifier(**final_params)
    final_model.fit(X_train_val, y_train_val)
    
    # Evaluate on test
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_logloss = log_loss(y_test, y_test_proba)
    
    # Also evaluate baseline on test for comparison
    baseline_test_pred = baseline_model.predict(X_test)
    baseline_test_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_test_acc = accuracy_score(y_test, baseline_test_pred)
    baseline_test_auc = roc_auc_score(y_test, baseline_test_proba)
    
    print(f"\n   Baseline (Old Params):")
    print(f"      Test Accuracy: {baseline_test_acc*100:.2f}%")
    print(f"      Test AUC: {baseline_test_auc:.4f}")
    
    print(f"\n   Tuned Model:")
    print(f"      Test Accuracy: {test_acc*100:.2f}%")
    print(f"      Test AUC: {test_auc:.4f}")
    print(f"      Test Log Loss: {test_logloss:.4f}")
    
    print(f"\n   Improvement:")
    print(f"      Accuracy: {(test_acc - baseline_test_acc)*100:+.2f}%")
    print(f"      AUC: {test_auc - baseline_test_auc:+.4f}")
    
    if test_auc > baseline_test_auc:
        print(f"      ✅ TUNED MODEL IS BETTER!")
    else:
        print(f"      ⚠️  Baseline was already good")
    
    # Save results
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    output_dir = Path('models/saved_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    model_path = output_dir / 'xgboost_tuned_model.json'
    final_model.save_model(str(model_path))
    print(f"   ✅ Model saved to {model_path}")
    
    # Save parameters
    params_path = output_dir / 'xgboost_tuned_params.json'
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': final_params,
            'baseline_params': baseline_params,
            'test_metrics': {
                'accuracy': test_acc,
                'auc': test_auc,
                'log_loss': test_logloss
            },
            'baseline_metrics': {
                'accuracy': baseline_test_acc,
                'auc': baseline_test_auc
            },
            'improvement': {
                'accuracy': test_acc - baseline_test_acc,
                'auc': test_auc - baseline_test_auc
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"   ✅ Parameters saved to {params_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = output_dir / 'xgboost_tuned_feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved to {importance_path}")
    
    # Save tuning summary
    summary_path = output_dir / 'xgboost_tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'n_trials': len(study.trials),
            'best_trial_number': best_trial.number,
            'best_value': best_trial.value,
            'all_trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'accuracy': t.user_attrs.get('accuracy'),
                    'auc': t.user_attrs.get('auc'),
                    'logloss': t.user_attrs.get('logloss')
                }
                for t in study.trials
            ]
        }, f, indent=2)
    print(f"   ✅ Tuning summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"\n🎉 New tuned model ready for predictions!")
    print(f"   Best parameters optimized from 100 trials")
    print(f"   Test accuracy: {test_acc*100:.2f}%")
    print(f"   Test AUC: {test_auc:.4f}")


if __name__ == '__main__':
    main()
