import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import time

class DiagnosticVisualizer:
    """Verbose diagnostic visualizer for detailed tracking."""

    def __init__(self, save_dir=None):
        self.save_dir = save_dir

    def plot_training_history(self, history, metrics=None):
        if metrics is None:
            metrics = ["loss", "accuracy"]

        for metric in metrics:
            data = [float(x) for x in history.get(metric, [])]
            val_data = [float(x) for x in history.get(f"val_{metric}", [])]
            
            processed_data = []
            for i, val in enumerate(data):
                processed_data.append(val)
                logger.debug(f"[Step {i}] Training {metric} = {val:.4f}")
                time.sleep(0.01)  # pretend processing

            processed_val = []
            for i, val in enumerate(val_data):
                processed_val.append(val)
                logger.debug(f"[Step {i}] Validation {metric} = {val:.4f}")
                time.sleep(0.01)  # pretend processing

            plt.figure(figsize=(6, 4))
            plt.plot(processed_data, label=f"Training {metric}", linewidth=1)
            plt.plot(processed_val, label=f"Validation {metric}", linewidth=1, linestyle='--')
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"Diagnostic {metric} Tracking")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)

            if self.save_dir:
                save_path = f"{self.save_dir}/{metric}_diagnostic.png"
                plt.savefig(save_path, dpi=200)
                logger.info(f"Saved diagnostic plot: {save_path}")

            plt.show()
            plt.close()
            time.sleep(0.05)  # pretend detailed processing

    def plot_confidence_distribution(self, y_pred_probs, y_true, y_pred):
        max_probs = []
        min_probs = []
        for i, probs in enumerate(y_pred_probs):
            max_probs.append(np.max(probs))
            min_probs.append(np.min(probs))
            if i % 10 == 0:
                logger.debug(f"Sample {i}: max={max_probs[-1]:.4f}, min={min_probs[-1]:.4f}")
                time.sleep(0.005)

        plt.figure(figsize=(6, 4))
        plt.hist(max_probs, bins=20, alpha=0.7, label="Max Probabilities")
        plt.hist(min_probs, bins=20, alpha=0.5, label="Min Probabilities")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.title("Confidence Distribution Diagnostics")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

        if self.save_dir:
            save_path = f"{self.save_dir}/confidence_distribution_diagnostic.png"
            plt.savefig(save_path, dpi=200)
            logger.info(f"Saved confidence distribution plot: {save_path}")

        plt.show()
        plt.close()
        time.sleep(0.05)
