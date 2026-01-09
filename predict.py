"""Inference script for SkimLit model."""

import argparse
import tensorflow as tf
from pathlib import Path
from loguru import logger
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor import TextPreprocessor
from src.utils.config import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with SkimLit model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Abstract text to analyze (direct input)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file containing abstract text"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        help="Show prediction probabilities"
    )
    
    return parser.parse_args()


def load_abstract_text(args):
    """Load abstract text from arguments or file."""
    if args.text:
        return args.text
    elif args.file:
        with open(args.file, "r") as f:
            return f.read()
    else:
        raise ValueError("Must provide either --text or --file argument")


def format_output(sentences, labels, confidences, class_probs=None):
    """Format prediction output."""
    results = []
    
    for idx, (sentence, label, confidence) in enumerate(zip(sentences, labels, confidences)):
        result = {
            "line_number": idx,
            "text": sentence,
            "label": label,
            "confidence": float(confidence)
        }
        
        if class_probs is not None:
            result["probabilities"] = {
                label: float(prob)
                for label, prob in class_probs[idx].items()
            }
        
        results.append(result)
    
    return results


def print_formatted_abstract(results):
    """Print formatted abstract with colors."""
    # Color codes
    colors = {
        "BACKGROUND": "\033[94m",  # Blue
        "OBJECTIVE": "\033[93m",   # Yellow
        "METHODS": "\033[95m",     # Magenta
        "RESULTS": "\033[92m",     # Green
        "CONCLUSIONS": "\033[91m", # Red
        "END": "\033[0m"           # Reset
    }
    
    print("\n" + "=" * 80)
    print("STRUCTURED ABSTRACT")
    print("=" * 80 + "\n")
    
    for result in results:
        label = result["label"]
        text = result["text"]
        confidence = result["confidence"]
        
        color = colors.get(label, colors["END"])
        
        print(f"{color}{label}{colors['END']} ({confidence:.1%}):")
        print(f"  {text}\n")
    
    print("=" * 80)


def main():
    """Main prediction function."""
    args = parse_args()
    
    logger.info("Loading model and configuration...")
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    logger.info("Model loaded successfully")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config.get("data", {}).get("preprocessing", {}))
    preprocessor.load_spacy_model()
    
    # Load abstract text
    abstract_text = load_abstract_text(args)
    logger.info(f"Processing abstract ({len(abstract_text)} characters)...")
    
    # Preprocess
    processed_data = preprocessor.prepare_inference_data(abstract_text)
    
    sentences = processed_data["sentences"]
    char_sequences = processed_data["char_sequences"]
    line_numbers_one_hot = processed_data["line_numbers_one_hot"]
    total_lines_one_hot = processed_data["total_lines_one_hot"]
    
    logger.info(f"Abstract split into {len(sentences)} sentences")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict({
        "line_number_input": line_numbers_one_hot,
        "total_lines_input": total_lines_one_hot,
        "token_input": sentences,
        "char_input": char_sequences
    })
    
    # Decode predictions
    pred_labels = preprocessor.decode_predictions(predictions)
    confidences = predictions.max(axis=1)
    
    # Get class probabilities if requested
    class_probs = None
    if args.show_probabilities:
        class_names = config["data"]["labels"]
        class_probs = [
            {name: predictions[i][j] for j, name in enumerate(class_names)}
            for i in range(len(predictions))
        ]
    
    # Format results
    results = format_output(sentences, pred_labels, confidences, class_probs)
    
    # Print formatted output
    print_formatted_abstract(results)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    # Print statistics
    label_counts = {}
    for label in pred_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()