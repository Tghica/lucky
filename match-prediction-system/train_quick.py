"""Quick training script - Gradient Boosting baseline + XGBoost"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import time

print("="*70)
print("QUICK TRAINING: Gradient Boosting + XGBoost")
print("="*70)

# Load data
print("\nLoading data...")
X_train = pd.read_csv('data/processed/train_features.csv')
y_train = pd.read_csv('data/processed/train_target.csv').values.ravel()
X_test = pd.read_csv('data/processed/test_features.csv')
y_test = pd.read_csv('data/processed/test_target.csv').values.ravel()

# Align features
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ========== BASELINE: Gradient Boosting ==========
print("\n" + "="*70)
print("GRADIENT BOOSTING BASELINE")
print("="*70)

start_time = time.time()
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
    verbose=0
)

print("Training...")
gb_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f}s")

# Evaluate
print("\nEvaluating...")
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_auc = roc_auc_score(y_test, y_pred_proba_gb)

print(f"\nResults:")
print(f"  Accuracy: {gb_accuracy:.4f}")
print(f"  AUC-ROC:  {gb_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred_gb, target_names=['Player2 Win', 'Player1 Win'])}")

# Save
joblib.dump(gb_model, 'models/saved_models/gradientboosting_model.pkl')
print("Model saved to: models/saved_models/gradientboosting_model.pkl")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features (Gradient Boosting):")
print(feature_importance.head(15).to_string(index=False))


# ========== ADVANCED: XGBoost ==========
print("\n" + "="*70)
print("XGBOOST ADVANCED MODEL")
print("="*70)

start_time = time.time()
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

print("Training...")
xgb_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f}s")

# Evaluate
print("\nEvaluating...")
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\nResults:")
print(f"  Accuracy: {xgb_accuracy:.4f}")
print(f"  AUC-ROC:  {xgb_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred_xgb, target_names=['Player2 Win', 'Player1 Win'])}")

# Save
joblib.dump(xgb_model, 'models/saved_models/xgboost_model.pkl')
print("Model saved to: models/saved_models/xgboost_model.pkl")

# Feature importance
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features (XGBoost):")
print(feature_importance_xgb.head(15).to_string(index=False))

# ========== COMPARISON ==========
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"\nGradient Boosting:")
print(f"  Accuracy: {gb_accuracy:.4f}")
print(f"  AUC-ROC:  {gb_auc:.4f}")

print(f"\nXGBoost:")
print(f"  Accuracy: {xgb_accuracy:.4f}")
print(f"  AUC-ROC:  {xgb_auc:.4f}")

improvement = (xgb_accuracy - gb_accuracy) * 100
print(f"\nImprovement: {improvement:+.2f}% accuracy")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
