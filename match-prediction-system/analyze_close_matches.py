"""
Analyze prediction accuracy for matches with similar Elo ratings.

This script investigates how well the model predicts winners when players
have close Elo ratings (competitive matches) vs when there's a clear favorite.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_and_model():
    """Load test data and best model."""
    print("Loading data and model...")
    
    # Load test data
    X_train = pd.read_csv('data/processed/train_features.csv')
    X_test = pd.read_csv('data/processed/test_features.csv')
    y_test = pd.read_csv('data/processed/test_target.csv').values.ravel()
    test_matches = pd.read_csv('data/processed/test_matches.csv')
    
    # Align features (add missing columns)
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]  # Ensure same order
    
    print(f"Aligned features: {X_test.shape[1]} columns")
    
    # Load best model (LightGBM Calibrated)
    try:
        model = joblib.load('models/saved_models/lightgbm_calibrated.pkl')
        model_name = "LightGBM Calibrated"
    except FileNotFoundError:
        # Fallback to XGBoost
        model = joblib.load('models/saved_models/xgboost_model.pkl')
        model_name = "XGBoost"
    
    print(f"Loaded model: {model_name}")
    print(f"Test set size: {len(X_test)}")
    
    return X_test, y_test, test_matches, model, model_name


def analyze_by_elo_difference(X_test, y_test, test_matches, model):
    """Analyze accuracy across different Elo difference ranges."""
    
    print("\n" + "="*80)
    print("ANALYZING PREDICTION ACCURACY BY ELO DIFFERENCE")
    print("="*80)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate Elo difference from test_matches (original match data)
    if 'player1_elo_before' in test_matches.columns and 'player2_elo_before' in test_matches.columns:
        elo_diff = (test_matches['player1_elo_before'] - 
                   test_matches['player2_elo_before']).values
    elif 'elo_diff' in X_test.columns:
        elo_diff = X_test['elo_diff'].values
    else:
        print("ERROR: Elo difference not found in data")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'elo_diff': np.abs(elo_diff),  # Absolute difference
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'correct': (y_test == y_pred).astype(int)
    })
    
    # Define Elo difference bins
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    labels = [
        '0-25 (Very Close)',
        '25-50 (Close)',
        '50-75 (Slight Edge)',
        '75-100 (Moderate Edge)',
        '100-150 (Clear Edge)',
        '150-200 (Strong Edge)',
        '200-300 (Dominant)',
        '300-500 (Very Dominant)',
        '500+ (Extreme)'
    ]
    
    df['elo_bin'] = pd.cut(df['elo_diff'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate statistics for each bin
    print("\n" + "-"*80)
    print(f"{'Elo Difference':<25} {'Count':>8} {'Accuracy':>10} {'Avg Prob':>12} {'Confidence':>12}")
    print("-"*80)
    
    results = []
    for label in labels:
        bin_data = df[df['elo_bin'] == label]
        if len(bin_data) == 0:
            continue
        
        count = len(bin_data)
        accuracy = bin_data['correct'].mean()
        avg_prob = bin_data['y_proba'].mean()
        
        # Confidence = how far probabilities are from 0.5 (50-50)
        confidence = np.abs(bin_data['y_proba'] - 0.5).mean()
        
        print(f"{label:<25} {count:>8} {accuracy:>9.1%} {avg_prob:>11.3f} {confidence:>11.3f}")
        
        results.append({
            'elo_range': label,
            'count': count,
            'accuracy': accuracy,
            'avg_probability': avg_prob,
            'avg_confidence': confidence
        })
    
    print("-"*80)
    
    # Focus on very close matches (0-50 Elo difference)
    print("\n" + "="*80)
    print("🔍 DETAILED ANALYSIS: VERY CLOSE MATCHES (Elo Diff 0-50)")
    print("="*80)
    
    close_matches = df[df['elo_diff'] <= 50]
    print(f"\nTotal close matches: {len(close_matches)}")
    print(f"Accuracy: {close_matches['correct'].mean():.2%}")
    print(f"Average predicted probability: {close_matches['y_proba'].mean():.3f}")
    print(f"Median predicted probability: {close_matches['y_proba'].median():.3f}")
    
    # Distribution of probabilities for close matches
    print(f"\nProbability distribution for close matches:")
    prob_bins = [0, 0.45, 0.50, 0.55, 0.60, 1.0]
    prob_labels = ['0-45% (Low)', '45-50% (Very Low)', '50-55% (Very Low)', '55-60% (Low)', '60-100% (Medium+)']
    close_matches['prob_bin'] = pd.cut(close_matches['y_proba'], bins=prob_bins, labels=prob_labels)
    
    print("\nProbability Range            Count    Accuracy")
    print("-" * 50)
    for label in prob_labels:
        bin_data = close_matches[close_matches['prob_bin'] == label]
        if len(bin_data) > 0:
            print(f"{label:<25} {len(bin_data):>6}    {bin_data['correct'].mean():>6.1%}")
    
    # Very close matches (0-25 Elo difference)
    print("\n" + "="*80)
    print("🎯 ULTRA-CLOSE MATCHES (Elo Diff 0-25)")
    print("="*80)
    
    ultra_close = df[df['elo_diff'] <= 25]
    print(f"\nTotal ultra-close matches: {len(ultra_close)}")
    print(f"Accuracy: {ultra_close['correct'].mean():.2%}")
    print(f"Average predicted probability: {ultra_close['y_proba'].mean():.3f}")
    
    # What percentage of predictions are low confidence?
    low_conf = ultra_close[(ultra_close['y_proba'] >= 0.45) & (ultra_close['y_proba'] <= 0.55)]
    print(f"\nMatches with 45-55% probability (coin flip): {len(low_conf)} ({len(low_conf)/len(ultra_close)*100:.1f}%)")
    if len(low_conf) > 0:
        print(f"Accuracy on these coin-flip predictions: {low_conf['correct'].mean():.2%}")
    
    return df, results


def plot_accuracy_by_elo(df, results):
    """Create visualization of accuracy vs Elo difference."""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy by Elo Range
    ax1 = axes[0, 0]
    elo_ranges = [r['elo_range'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    counts = [r['count'] for r in results]
    
    bars = ax1.bar(range(len(elo_ranges)), accuracies, color='steelblue', alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random Guess (50%)', linewidth=2)
    ax1.axhline(y=0.6486, color='green', linestyle='--', label='Overall Accuracy (64.86%)', linewidth=2)
    ax1.set_xlabel('Elo Difference Range', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Accuracy by Elo Difference', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(elo_ranges)))
    ax1.set_xticklabels([r.split(' ')[0] for r in elo_ranges], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Confidence by Elo Range
    ax2 = axes[0, 1]
    confidences = [r['avg_confidence'] for r in results]
    ax2.bar(range(len(elo_ranges)), confidences, color='coral', alpha=0.7)
    ax2.set_xlabel('Elo Difference Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Confidence (|prob - 0.5|)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Confidence by Elo Difference', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(elo_ranges)))
    ax2.set_xticklabels([r.split(' ')[0] for r in elo_ranges], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot - Elo diff vs Probability
    ax3 = axes[1, 0]
    # Sample for visualization (too many points otherwise)
    sample_df = df.sample(min(5000, len(df)))
    scatter = ax3.scatter(sample_df['elo_diff'], sample_df['y_proba'], 
                         c=sample_df['correct'], cmap='RdYlGn', alpha=0.3, s=10)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Absolute Elo Difference', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Elo Difference vs Predicted Probability', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Correct Prediction')
    
    # Plot 4: Distribution of matches by Elo difference
    ax4 = axes[1, 1]
    ax4.hist(df['elo_diff'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=25, color='red', linestyle='--', label='Ultra-close (±25)', linewidth=2)
    ax4.axvline(x=50, color='orange', linestyle='--', label='Close (±50)', linewidth=2)
    ax4.axvline(x=100, color='yellow', linestyle='--', label='Moderate (±100)', linewidth=2)
    ax4.set_xlabel('Absolute Elo Difference', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Elo Differences in Test Set', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'models/saved_models/elo_difference_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("🎾 ELO DIFFERENCE ANALYSIS")
    print("Analyzing prediction accuracy for matches with similar Elo ratings")
    print("="*80)
    
    # Load data and model
    X_test, y_test, test_matches, model, model_name = load_data_and_model()
    
    # Analyze by Elo difference
    df, results = analyze_by_elo_difference(X_test, y_test, test_matches, model)
    
    # Create visualizations
    plot_accuracy_by_elo(df, results)
    
    # Summary
    print("\n" + "="*80)
    print("📊 KEY FINDINGS")
    print("="*80)
    
    ultra_close = df[df['elo_diff'] <= 25]
    close = df[df['elo_diff'] <= 50]
    moderate = df[(df['elo_diff'] > 50) & (df['elo_diff'] <= 100)]
    large = df[df['elo_diff'] > 100]
    
    print(f"\n1. ULTRA-CLOSE MATCHES (Elo ±0-25):")
    print(f"   - Count: {len(ultra_close)} ({len(ultra_close)/len(df)*100:.1f}% of test set)")
    print(f"   - Accuracy: {ultra_close['correct'].mean():.2%}")
    print(f"   - Avg Probability: {ultra_close['y_proba'].mean():.3f}")
    print(f"   - Interpretation: Nearly coin-flip territory, very hard to predict")
    
    print(f"\n2. CLOSE MATCHES (Elo ±25-50):")
    close_2550 = df[(df['elo_diff'] > 25) & (df['elo_diff'] <= 50)]
    if len(close_2550) > 0:
        print(f"   - Count: {len(close_2550)} ({len(close_2550)/len(df)*100:.1f}% of test set)")
        print(f"   - Accuracy: {close_2550['correct'].mean():.2%}")
        print(f"   - Avg Probability: {close_2550['y_proba'].mean():.3f}")
        print(f"   - Interpretation: Slight favorite emerges")
    
    print(f"\n3. MODERATE DIFFERENCE (Elo ±50-100):")
    print(f"   - Count: {len(moderate)} ({len(moderate)/len(df)*100:.1f}% of test set)")
    print(f"   - Accuracy: {moderate['correct'].mean():.2%}")
    print(f"   - Avg Probability: {moderate['y_proba'].mean():.3f}")
    print(f"   - Interpretation: Model gains confidence")
    
    print(f"\n4. LARGE DIFFERENCE (Elo >100):")
    print(f"   - Count: {len(large)} ({len(large)/len(df)*100:.1f}% of test set)")
    print(f"   - Accuracy: {large['correct'].mean():.2%}")
    print(f"   - Avg Probability: {large['y_proba'].mean():.3f}")
    print(f"   - Interpretation: Clear favorite, high accuracy")
    
    print("\n" + "="*80)
    print("💡 INSIGHTS FOR BETTING:")
    print("="*80)
    print("\n- Ultra-close matches (Elo ±0-25): Avoid betting - essentially random")
    print("- Close matches (Elo ±25-50): Low confidence - only bet if odds offer value")
    print("- Moderate edge (Elo ±50-100): Medium confidence - selective betting")
    print("- Large edge (Elo >100): High confidence - focus betting here")
    
    print("\n✓ ANALYSIS COMPLETE!")


if __name__ == '__main__':
    main()
