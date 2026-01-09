"""Main training script for SkimLit model."""

import argparse
from pathlib import Path
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.data.dataset_builder import DatasetBuilder
from src.models.architectures import TribridModel
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizer import ResultsVisualizer
from src.utils.config import ConfigManager


def setup_logging(log_file: str = "training.log"):
    """Setup logging configuration."""
    logger.add(
        log_file,
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SkimLit model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data files"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Train on full dataset (not just 10%%)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on test set after training"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 50)
    logger.info("Starting SkimLit Training")
    logger.info("=" * 50)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Override config with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    # Load data
    logger.info("Loading datasets...")
    data_loader = DataLoader(args.data_dir)
    datasets = data_loader.load_all_datasets()
    
    train_df = datasets["train"]
    val_df = datasets["val"]
    test_df = datasets["test"]
    
    # Log dataset statistics
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = TextPreprocessor(config.get("data", {}).get("preprocessing", {}))
    
    # Encode labels
    (
        train_labels_encoded, val_labels_encoded, test_labels_encoded,
        train_labels_one_hot, val_labels_one_hot, test_labels_one_hot,
        num_classes, class_names
    ) = preprocessor.prepare_labels(train_df, val_df, test_df)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    
    # Prepare features
    logger.info("Preparing features...")
    
    # Sentences
    train_sentences = preprocessor.prepare_sentences(train_df)
    val_sentences = preprocessor.prepare_sentences(val_df)
    test_sentences = preprocessor.prepare_sentences(test_df)
    
    # Character sequences
    train_chars = preprocessor.prepare_char_sequences(train_sentences)
    val_chars = preprocessor.prepare_char_sequences(val_sentences)
    test_chars = preprocessor.prepare_char_sequences(test_sentences)
    
    # Positional embeddings
    train_line_nums_oh, train_total_lines_oh = preprocessor.prepare_positional_embeddings(train_df)
    val_line_nums_oh, val_total_lines_oh = preprocessor.prepare_positional_embeddings(val_df)
    test_line_nums_oh, test_total_lines_oh = preprocessor.prepare_positional_embeddings(test_df)
    
    # Build datasets
    logger.info("Building TensorFlow datasets...")
    batch_size = config["training"]["batch_size"]
    dataset_builder = DatasetBuilder(batch_size=batch_size)
    
    data_dict = {
        "train_line_numbers_one_hot": train_line_nums_oh,
        "train_total_lines_one_hot": train_total_lines_oh,
        "train_sentences": train_sentences,
        "train_chars": train_chars,
        "train_labels_one_hot": train_labels_one_hot,
        "val_line_numbers_one_hot": val_line_nums_oh,
        "val_total_lines_one_hot": val_total_lines_oh,
        "val_sentences": val_sentences,
        "val_chars": val_chars,
        "val_labels_one_hot": val_labels_one_hot,
        "test_line_numbers_one_hot": test_line_nums_oh,
        "test_total_lines_one_hot": test_total_lines_oh,
        "test_sentences": test_sentences,
        "test_chars": test_chars,
        "test_labels_one_hot": test_labels_one_hot,
    }
    
    tf_datasets = dataset_builder.build_all_datasets(data_dict)
    train_dataset = tf_datasets["train"]
    val_dataset = tf_datasets["val"]
    test_dataset = tf_datasets["test"]
    
    # Build model
    logger.info("Building model...")
    model_builder = TribridModel(config)
    model = model_builder.build(num_classes=num_classes)
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        config=config,
        experiment_name=args.experiment_name
    )
    
    # Save configuration
    trainer.save_config()
    
    # Calculate steps
    if args.full_dataset:
        steps_per_epoch = None
        validation_steps = None
        logger.info("Training on full dataset")
    else:
        steps_per_epoch = int(0.1 * len(train_dataset))
        validation_steps = int(0.1 * len(val_dataset))
        logger.info(f"Training on 10%% of dataset ({steps_per_epoch} steps)")
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Evaluate on test set if requested
    if args.evaluate:
        logger.info("Evaluating on test set...")
        
        # Make predictions
        test_pred_probs = model.predict(test_dataset)
        test_preds = test_pred_probs.argmax(axis=1)
        
        # Calculate metrics
        evaluator = ModelEvaluator(class_names=class_names)
        results = evaluator.evaluate(
            y_true=test_labels_encoded,
            y_pred=test_preds,
            y_pred_probs=test_pred_probs
        )
        
        logger.info("Test Results:")
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"Precision: {results['precision']:.2f}%")
        logger.info(f"Recall: {results['recall']:.2f}%")
        logger.info(f"F1-Score: {results['f1']:.2f}%")
        
        # Create visualizations
        visualizer = ResultsVisualizer(save_dir=trainer.exp_dir / "plots")
        visualizer.create_evaluation_report(
            results=results,
            class_names=class_names,
            history=history.history if history else None
        )
    
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Results saved to: {trainer.exp_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()