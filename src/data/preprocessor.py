"""Text preprocessing utilities for SkimLit."""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from loguru import logger
import spacy


class TextPreprocessor:
    """Preprocess text data for model training and inference."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.nlp = None
        
    def load_spacy_model(self):
        """Load spaCy model for sentence splitting."""
        if self.nlp is None:
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found. Using English sentencizer.")
                from spacy.lang.en import English
                self.nlp = English()
                self.nlp.add_pipe("sentencizer")
            logger.info("Spacy model loaded")
    
    def split_chars(self, text: str) -> str:
        """
        Split text into space-separated characters.
        
        Args:
            text: Input text string
            
        Returns:
            Space-separated characters
        """
        return " ".join(list(text))
    
    def prepare_labels(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple:
        """
        Encode labels for all dataset splits.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_encoded, val_encoded, test_encoded,
                     train_one_hot, val_one_hot, test_one_hot,
                     num_classes, class_names)
        """
        logger.info("Encoding labels...")
        
        # Fit label encoder on training data
        train_labels_encoded = self.label_encoder.fit_transform(
            train_df["target"].to_numpy()
        )
        val_labels_encoded = self.label_encoder.transform(
            val_df["target"].to_numpy()
        )
        test_labels_encoded = self.label_encoder.transform(
            test_df["target"].to_numpy()
        )
        
        # One-hot encode
        train_labels_one_hot = self.one_hot_encoder.fit_transform(
            train_df["target"].to_numpy().reshape(-1, 1)
        )
        val_labels_one_hot = self.one_hot_encoder.transform(
            val_df["target"].to_numpy().reshape(-1, 1)
        )
        test_labels_one_hot = self.one_hot_encoder.transform(
            test_df["target"].to_numpy().reshape(-1, 1)
        )
        
        num_classes = len(self.label_encoder.classes_)
        class_names = self.label_encoder.classes_
        
        logger.info(f"Labels encoded. Classes: {class_names}")
        
        return (
            train_labels_encoded, val_labels_encoded, test_labels_encoded,
            train_labels_one_hot, val_labels_one_hot, test_labels_one_hot,
            num_classes, class_names
        )
    
    def prepare_sentences(self, df: pd.DataFrame) -> List[str]:
        """
        Extract sentences from DataFrame.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            List of sentences
        """
        return df["text"].tolist()
    
    def prepare_char_sequences(self, sentences: List[str]) -> List[str]:
        """
        Convert sentences to character-level sequences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of character-separated sentences
        """
        logger.info(f"Converting {len(sentences)} sentences to char sequences...")
        return [self.split_chars(sent) for sent in sentences]
    
    def prepare_positional_embeddings(
        self, 
        df: pd.DataFrame
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create one-hot encoded positional embeddings.
        
        Args:
            df: DataFrame with 'line_number' and 'total_lines' columns
            
        Returns:
            Tuple of (line_numbers_one_hot, total_lines_one_hot)
        """
        line_number_depth = self.config.get("line_number_depth", 15)
        total_lines_depth = self.config.get("total_lines_depth", 20)
        
        line_numbers_one_hot = tf.one_hot(
            df["line_number"].to_numpy(), 
            depth=line_number_depth
        )
        
        total_lines_one_hot = tf.one_hot(
            df["total_lines"].to_numpy(), 
            depth=total_lines_depth
        )
        
        logger.info(f"Created positional embeddings: line_numbers={line_numbers_one_hot.shape}, "
                   f"total_lines={total_lines_one_hot.shape}")
        
        return line_numbers_one_hot, total_lines_one_hot
    
    def split_abstract_into_sentences(self, abstract_text: str) -> List[str]:
        """
        Split abstract text into sentences using spaCy.
        
        Args:
            abstract_text: Raw abstract text
            
        Returns:
            List of sentences
        """
        if self.nlp is None:
            self.load_spacy_model()
        
        doc = self.nlp(abstract_text)
        sentences = [str(sent).strip() for sent in doc.sents]
        
        return sentences
    
    def prepare_inference_data(self, abstract_text: str) -> Dict:
        """
        Prepare data for inference from raw abstract text.
        
        Args:
            abstract_text: Raw abstract text
            
        Returns:
            Dictionary with processed features ready for model input
        """
        # Split into sentences
        sentences = self.split_abstract_into_sentences(abstract_text)
        total_lines = len(sentences)
        
        # Create line numbers
        line_numbers = list(range(total_lines))
        
        # One-hot encode positions
        line_number_depth = self.config.get("line_number_depth", 15)
        total_lines_depth = self.config.get("total_lines_depth", 20)
        
        line_numbers_one_hot = tf.one_hot(line_numbers, depth=line_number_depth)
        total_lines_one_hot = tf.one_hot(
            [total_lines - 1] * total_lines, 
            depth=total_lines_depth
        )
        
        # Character sequences
        char_sequences = self.prepare_char_sequences(sentences)
        
        return {
            "sentences": sentences,
            "char_sequences": char_sequences,
            "line_numbers_one_hot": line_numbers_one_hot,
            "total_lines_one_hot": total_lines_one_hot,
            "total_lines": total_lines
        }
    
    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """
        Convert model predictions to class labels.
        
        Args:
            predictions: Model output probabilities
            
        Returns:
            List of predicted class labels
        """
        pred_indices = np.argmax(predictions, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_indices)
        return pred_labels.tolist()