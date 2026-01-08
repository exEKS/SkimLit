"""Visualization utilities for model evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class ResultsVisualizer:
    """Visualize model training and evaluation results."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
    
    def plot_training_history(
        self, 
        history: Dict,
        metrics: List[str] = None,
        save_name: str = "training_history.png"
    ):
        """
        Plot training history curves.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            save_name: Filename to save plot
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Training metric
            if metric in history:
                ax.plot(history[metric], label=f"Training {metric}", linewidth=2)
            
            # Validation metric
            val_metric = f"val_{metric}"
            if val_metric in history:
                ax.plot(history[val_metric], label=f"Validation {metric}", linewidth=2)
            
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f"{metric.capitalize()} over Epochs", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize
            save_name: Filename to save plot
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count" if not normalize else "Proportion"}
        )
        
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict[str, Dict],
        save_name: str = "per_class_metrics.png"
    ):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            per_class_metrics: Dictionary with per-class metrics
            save_name: Filename to save plot
        """
        # Prepare data
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[c]["precision"] for c in classes]
        recall = [per_class_metrics[c]["recall"] for c in classes]
        f1 = [per_class_metrics[c]["f1"] for c in classes]
        
        # Create plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
        
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(
        self,
        y_pred_probs: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "confidence_distribution.png"
    ):
        """
        Plot distribution of prediction confidences.
        
        Args:
            y_pred_probs: Prediction probabilities
            y_true: True labels
            y_pred: Predicted labels
            save_name: Filename to save plot
        """
        # Get maximum probabilities
        max_probs = np.max(y_pred_probs, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = y_true == y_pred
        correct_probs = max_probs[correct_mask]
        incorrect_probs = max_probs[~correct_mask]
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(correct_probs, bins=50, alpha=0.7, label="Correct", color="green")
        axes[0].hist(incorrect_probs, bins=50, alpha=0.7, label="Incorrect", color="red")
        axes[0].set_xlabel("Prediction Confidence", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("Confidence Distribution", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [correct_probs, incorrect_probs]
        axes[1].boxplot(data_to_plot, labels=["Correct", "Incorrect"])
        axes[1].set_ylabel("Prediction Confidence", fontsize=12)
        axes[1].set_title("Confidence Comparison", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confidence distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict],
        save_name: str = "model_comparison.png"
    ):
        """
        Compare multiple models' performance.
        
        Args:
            results_dict: Dictionary mapping model names to their results
            save_name: Filename to save plot
        """
        # Prepare data
        models = list(results_dict.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]
        
        data = {metric: [] for metric in metrics}
        
        for model in models:
            for metric in metrics:
                data[metric].append(results_dict[model].get(metric, 0))
        
        # Create plot
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics) / 2)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize(), alpha=0.8)
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(
        self,
        results: Dict,
        class_names: List[str],
        history: Optional[Dict] = None
    ):
        """
        Create comprehensive evaluation report with all visualizations.
        
        Args:
            results: Evaluation results dictionary
            class_names: List of class names
            history: Training history (optional)
        """
        logger.info("Creating comprehensive evaluation report...")
        
        # Plot training history if available
        if history:
            self.plot_training_history(history)
        
        # Plot confusion matrix
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"])
            self.plot_confusion_matrix(cm, class_names)
            self.plot_confusion_matrix(cm, class_names, normalize=True, 
                                      save_name="confusion_matrix_normalized.png")
        
        # Plot per-class metrics
        if "per_class_metrics" in results:
            self.plot_per_class_metrics(results["per_class_metrics"])
        
        logger.info("Evaluation report created successfully")