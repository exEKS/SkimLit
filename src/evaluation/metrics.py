"""Evaluation metrics for SkimLit."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple
from loguru import logger


def calculate_results(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, precision, recall, f1-score
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    results = {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }
    
    return results


class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray = None
    ) -> Dict:
        """
        Comprehensive evaluation with multiple metrics.
        
        Args:
            y_true: True labels (integer encoded)
            y_pred: Predicted labels (integer encoded)
            y_pred_probs: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        results = {}
        
        # Basic metrics
        basic_metrics = calculate_results(y_true, y_pred)
        results.update(basic_metrics)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i] * 100),
                "recall": float(recall[i] * 100),
                "f1": float(f1[i] * 100),
                "support": int(support[i])
            }
        
        results["per_class_metrics"] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        results["classification_report"] = report
        
        # Confidence statistics (if probabilities provided)
        if y_pred_probs is not None:
            confidence_stats = self._calculate_confidence_stats(y_pred_probs)
            results["confidence_stats"] = confidence_stats
        
        logger.info(f"Evaluation complete. Overall accuracy: {results['accuracy']:.2f}%")
        
        return results
    
    def _calculate_confidence_stats(self, y_pred_probs: np.ndarray) -> Dict:
        """
        Calculate prediction confidence statistics.
        
        Args:
            y_pred_probs: Prediction probabilities
            
        Returns:
            Dictionary with confidence statistics
        """
        max_probs = np.max(y_pred_probs, axis=1)
        
        stats = {
            "mean_confidence": float(np.mean(max_probs)),
            "median_confidence": float(np.median(max_probs)),
            "std_confidence": float(np.std(max_probs)),
            "min_confidence": float(np.min(max_probs)),
            "max_confidence": float(np.max(max_probs))
        }
        
        return stats
    
    def get_classification_report_df(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Get classification report as DataFrame.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with classification report
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        df = pd.DataFrame(report).transpose()
        return df
    
    def get_confusion_matrix_df(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Get confusion matrix as DataFrame.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        df = pd.DataFrame(
            cm,
            index=[f"True {c}" for c in self.class_names],
            columns=[f"Pred {c}" for c in self.class_names]
        )
        
        return df
    
    def find_most_wrong_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray,
        data: pd.DataFrame,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Find samples with highest confidence but wrong predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities
            data: Original data DataFrame
            top_n: Number of top wrong predictions to return
            
        Returns:
            DataFrame with most wrong predictions
        """
        # Get maximum probability for each prediction
        max_probs = np.max(y_pred_probs, axis=1)
        
        # Create results DataFrame
        results_df = data.copy()
        results_df["true_label"] = [self.class_names[i] for i in y_true]
        results_df["pred_label"] = [self.class_names[i] for i in y_pred]
        results_df["pred_prob"] = max_probs
        results_df["correct"] = y_true == y_pred
        
        # Get wrong predictions sorted by confidence
        wrong_preds = results_df[results_df["correct"] == False].sort_values(
            "pred_prob", ascending=False
        )
        
        logger.info(f"Found {len(wrong_preds)} incorrect predictions")
        
        return wrong_preds.head(top_n)
    
    def get_per_class_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate accuracy for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary mapping class names to accuracies
        """
        accuracies = {}
        
        for i, class_name in enumerate(self.class_names):
            # Get samples for this class
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                accuracies[class_name] = class_acc * 100
            else:
                accuracies[class_name] = 0.0
        
        return accuracies