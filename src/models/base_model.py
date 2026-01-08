"""Base model class with common functionality."""

from abc import ABC, abstractmethod
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


class BaseModel(ABC):
    """Abstract base class for all SkimLit models."""
    
    def __init__(self, config: Dict):
        """
        Initialize base model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.history = None
        
    @abstractmethod
    def build(self, **kwargs) -> tf.keras.Model:
        """Build the model architecture."""
        pass
    
    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: list = None
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            metrics: List of metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if metrics is None:
            metrics = ["accuracy"]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer={optimizer}, loss={loss}")
    
    def fit(
        self,
        train_data,
        validation_data=None,
        epochs: int = 10,
        callbacks: list = None,
        **kwargs
    ):
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
            **kwargs: Additional arguments for fit
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, test_data, **kwargs):
        """
        Evaluate the model.
        
        Args:
            test_data: Test dataset
            **kwargs: Additional arguments for evaluate
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info("Evaluating model...")
        results = self.model.evaluate(test_data, **kwargs)
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def predict(self, data, **kwargs):
        """
        Make predictions.
        
        Args:
            data: Input data
            **kwargs: Additional arguments for predict
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        return self.model.predict(data, **kwargs)
    
    def save(self, filepath: str, save_format: str = "tf"):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            save_format: Format to save ('tf' or 'h5')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {filepath}")
        self.model.save(filepath, save_format=save_format)
        logger.info("Model saved successfully")
    
    def load_weights(self, filepath: str):
        """
        Load model weights from disk.
        
        Args:
            filepath: Path to weights file
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Loading weights from {filepath}")
        self.model.load_weights(filepath)
        logger.info("Weights loaded successfully")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        return self.model.summary()
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return self.config
    
    def count_params(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        trainable_params = sum(
            [tf.size(w).numpy() for w in self.model.trainable_weights]
        )
        non_trainable_params = sum(
            [tf.size(w).numpy() for w in self.model.non_trainable_weights]
        )
        
        return {
            "total": trainable_params + non_trainable_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params
        }