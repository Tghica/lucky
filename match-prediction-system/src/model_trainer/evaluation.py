import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for tennis match prediction."""
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        
    def predict(self):
        """Make predictions on test set."""
        logger.info("Making predictions on test set...")
        self.y_pred = self.model.predict(self.X_test)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        return self.y_pred, self.y_pred_proba
    
    def evaluate_all(self) -> Dict:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if self.y_pred is None:
            self.predict()
        
        logger.info(f"\n{'='*60}")
        logger.info("MODEL EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='binary')
        recall = recall_score(self.y_test, self.y_pred, average='binary')
        f1 = f1_score(self.y_test, self.y_pred, average='binary')
        
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        # AUC-ROC (if probabilities available)
        auc_roc = None
        if self.y_pred_proba is not None:
            auc_roc = roc_auc_score(self.y_test, self.y_pred_proba)
            logger.info(f"  AUC-ROC:   {auc_roc:.4f}")
        
        # Log loss (if probabilities available)
        logloss = None
        if self.y_pred_proba is not None:
            logloss = log_loss(self.y_test, self.y_pred_proba)
            logger.info(f"  Log Loss:  {logloss:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
        logger.info(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
        
        # Classification report
        logger.info(f"\nClassification Report:")
        report = classification_report(self.y_test, self.y_pred, target_names=['Player 2 Wins', 'Player 1 Wins'])
        logger.info(f"\n{report}")
        
        # Return all metrics
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'log_loss': logloss,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(self.y_test),
            'n_correct': int((self.y_pred == self.y_test).sum()),
            'n_incorrect': int((self.y_pred != self.y_test).sum())
        }
        
        return results
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Player 2 Wins', 'Player 1 Wins'],
            yticklabels=['Player 2 Wins', 'Player 1 Wins']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.y_pred_proba is None:
            logger.warning("Probabilities not available. Cannot plot ROC curve.")
            return
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_prediction_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of predicted probabilities.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.y_pred_proba is None:
            logger.warning("Probabilities not available. Cannot plot distribution.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot for each class
        plt.hist(
            self.y_pred_proba[self.y_test == 0],
            bins=50, alpha=0.5, label='Actual: Player 2 Wins',
            color='red', edgecolor='black'
        )
        plt.hist(
            self.y_pred_proba[self.y_test == 1],
            bins=50, alpha=0.5, label='Actual: Player 1 Wins',
            color='blue', edgecolor='black'
        )
        
        plt.xlabel('Predicted Probability (Player 1 Wins)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_errors(self, feature_names: Optional[list] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze prediction errors.
        
        Args:
            feature_names: List of feature names (optional)
            top_n: Number of worst errors to return
        
        Returns:
            DataFrame with worst predictions
        """
        if self.y_pred_proba is None:
            logger.warning("Probabilities not available. Cannot analyze errors.")
            return None
        
        # Calculate error magnitude
        errors = np.abs(self.y_test - self.y_pred_proba)
        
        # Get worst predictions
        worst_idx = np.argsort(errors)[-top_n:][::-1]
        
        error_analysis = pd.DataFrame({
            'index': worst_idx,
            'true_label': self.y_test[worst_idx],
            'predicted_prob': self.y_pred_proba[worst_idx],
            'error': errors[worst_idx]
        })
        
        logger.info(f"\nTop {top_n} Worst Predictions:")
        logger.info(error_analysis.to_string(index=False))
        
        return error_analysis