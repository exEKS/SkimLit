"""Model training orchestration."""

import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
import json
from datetime import datetime

from .callbacks import create_callbacks


class ModelTrainer:
    """Orchestrate model training with all necessary components."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: Dict,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Compiled Keras model
            config: Configuration dictionary
            experiment_name: Name for this experiment
        """
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        self.history = None
        
        # Create experiment directory
        self.exp_dir = Path("experiments") / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer for experiment: {self.experiment_name}")
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        use_callbacks: bool = True
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs (uses config if None)
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            use_callbacks: Whether to use callbacks
            
        Returns:
            Training history
        """
        training_config = self.config.get("training", {})
        
        if epochs is None:
            epochs = training_config.get("epochs", 10)
        
        # Create callbacks
        callbacks = None
        if use_callbacks:
            callbacks = create_callbacks(
                config=self.config,
                exp_dir=str(self.exp_dir)
            )
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        
        try:
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            
            # Save training history
            self._save_history()
            
            # Save final model
            self.save_model()
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._save_history()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        return self.history
    
    def _save_history(self):
        """Save training history to JSON."""
        if self.history is None:
            return
        
        history_path = self.exp_dir / "history.json"
        
        # Convert history to serializable format
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def save_model(self, filename: str = "model"):
        """
        Save model to experiment directory.
        
        Args:
            filename: Model filename (without extension)
        """
        model_path = self.exp_dir / filename
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def save_config(self):
        """Save configuration to experiment directory."""
        config_path = self.exp_dir / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Config saved to {config_path}")
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            save_results: Whether to save results
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Evaluating model on test dataset...")
        
        results = self.model.evaluate(test_dataset, return_dict=True)
        
        logger.info(f"Test results: {results}")
        
        if save_results:
            results_path = self.exp_dir / "test_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {results_path}")
        
        return results
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best model checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        checkpoint_dir = self.exp_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            return None
        
        # Find checkpoint with highest validation accuracy
        checkpoints = list(checkpoint_dir.glob("*.h5"))
        
        if not checkpoints:
            return None
        
        # Extract accuracy from filename (assumes format: model_epoch{X}_val_acc{Y}.h5)
        best_checkpoint = max(
            checkpoints,
            key=lambda p: float(p.stem.split("val_acc")[-1]) if "val_acc" in p.stem else 0
        )
        
        return best_checkpoint
    
    def load_best_checkpoint(self):
        """Load best checkpoint weights."""
        best_checkpoint = self.get_best_checkpoint()
        
        if best_checkpoint is None:
            logger.warning("No checkpoint found")
            return
        
        logger.info(f"Loading best checkpoint: {best_checkpoint}")
        self.model.load_weights(best_checkpoint)
        logger.info("Best checkpoint loaded")