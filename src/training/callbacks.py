"""Training callbacks for SkimLit."""

import tensorflow as tf
from pathlib import Path
from typing import Dict, List
from loguru import logger


def create_callbacks(config: Dict, exp_dir: str) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks based on configuration.
    
    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory path
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    callback_config = config.get("training", {}).get("callbacks", {})
    
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # ModelCheckpoint
    if "model_checkpoint" in callback_config:
        checkpoint_config = callback_config["model_checkpoint"]
        checkpoint_dir = exp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "model_epoch{epoch:02d}_val_acc{val_accuracy:.4f}.h5"
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_config.get("monitor", "val_accuracy"),
            save_best_only=checkpoint_config.get("save_best_only", True),
            save_weights_only=checkpoint_config.get("save_weights_only", False),
            mode="max",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"Added ModelCheckpoint callback: {checkpoint_path}")
    
    # EarlyStopping
    if "early_stopping" in callback_config:
        early_stopping_config = callback_config["early_stopping"]
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            patience=early_stopping_config.get("patience", 3),
            restore_best_weights=early_stopping_config.get("restore_best_weights", True),
            mode="min",
            verbose=1
        )
        callbacks.append(early_stopping)
        logger.info("Added EarlyStopping callback")
    
    # ReduceLROnPlateau
    if "reduce_lr" in callback_config:
        reduce_lr_config = callback_config["reduce_lr"]
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_config.get("monitor", "val_loss"),
            factor=reduce_lr_config.get("factor", 0.5),
            patience=reduce_lr_config.get("patience", 2),
            min_lr=reduce_lr_config.get("min_lr", 1e-7),
            mode="min",
            verbose=1
        )
        callbacks.append(reduce_lr)
        logger.info("Added ReduceLROnPlateau callback")
    
    # TensorBoard
    tensorboard_dir = exp_path / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch"
    )
    callbacks.append(tensorboard)
    logger.info(f"Added TensorBoard callback: {tensorboard_dir}")
    
    # CSV Logger
    csv_path = exp_path / "training_log.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_path),
        separator=",",
        append=False
    )
    callbacks.append(csv_logger)
    logger.info(f"Added CSVLogger callback: {csv_path}")
    
    # Custom callback for learning rate logging
    lr_logger = LearningRateLogger()
    callbacks.append(lr_logger)
    
    return callbacks


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Custom callback to log learning rate."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate at end of epoch."""
        if logs is None:
            logs = {}
        
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        logs["lr"] = lr
        logger.info(f"Epoch {epoch + 1} - Learning Rate: {lr:.2e}")


class MetricsLogger(tf.keras.callbacks.Callback):
    """Custom callback to log detailed metrics."""
    
    def __init__(self, log_file: str):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Path to log file
        """
        super().__init__()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write("epoch,batch,loss,accuracy,val_loss,val_accuracy\n")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of epoch."""
        if logs is None:
            logs = {}
        
        with open(self.log_file, "a") as f:
            f.write(
                f"{epoch},-,"
                f"{logs.get('loss', 0):.4f},"
                f"{logs.get('accuracy', 0):.4f},"
                f"{logs.get('val_loss', 0):.4f},"
                f"{logs.get('val_accuracy', 0):.4f}\n"
            )


class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for better progress reporting."""
    
    def on_train_begin(self, logs=None):
        """Initialize training progress."""
        logger.info("=" * 50)
        logger.info("Training started")
        logger.info("=" * 50)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Log epoch start."""
        logger.info(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log epoch results."""
        if logs is None:
            logs = {}
        
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger.info(f"Epoch {epoch + 1} completed - {metrics_str}")
    
    def on_train_end(self, logs=None):
        """Log training completion."""
        logger.info("=" * 50)
        logger.info("Training completed")
        logger.info("=" * 50)