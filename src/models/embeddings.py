"""Embedding layer components for SkimLit models."""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from typing import Dict, Optional
from loguru import logger


class EmbeddingLayers:
    """Factory class for creating various embedding layers."""
    
    @staticmethod
    def create_token_vectorizer(
        max_tokens: int = 68000,
        output_sequence_length: int = 55,
        vocab: Optional[list] = None
    ) -> layers.TextVectorization:
        """
        Create text vectorization layer for tokens.
        
        Args:
            max_tokens: Maximum vocabulary size
            output_sequence_length: Output sequence length
            vocab: Optional pre-computed vocabulary
            
        Returns:
            TextVectorization layer
        """
        text_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            standardize="lower_and_strip_punctuation",
            name="token_vectorizer"
        )
        
        if vocab is not None:
            text_vectorizer.set_vocabulary(vocab)
        
        logger.info(f"Created token vectorizer: max_tokens={max_tokens}, "
                   f"output_length={output_sequence_length}")
        
        return text_vectorizer
    
    @staticmethod
    def create_char_vectorizer(
        max_tokens: int = 70,
        output_sequence_length: int = 290
    ) -> layers.TextVectorization:
        """
        Create text vectorization layer for characters.
        
        Args:
            max_tokens: Maximum vocabulary size (characters)
            output_sequence_length: Output sequence length
            
        Returns:
            TextVectorization layer
        """
        char_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            standardize="lower_and_strip_punctuation",
            split="character",
            name="char_vectorizer"
        )
        
        logger.info(f"Created char vectorizer: max_tokens={max_tokens}, "
                   f"output_length={output_sequence_length}")
        
        return char_vectorizer
    
    @staticmethod
    def create_token_embedding(
        vocab_size: int,
        embedding_dim: int = 128,
        mask_zero: bool = True
    ) -> layers.Embedding:
        """
        Create custom token embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            mask_zero: Whether to mask zero values
            
        Returns:
            Embedding layer
        """
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=mask_zero,
            name="token_embedding"
        )
        
        logger.info(f"Created token embedding: vocab_size={vocab_size}, "
                   f"dim={embedding_dim}")
        
        return embedding
    
    @staticmethod
    def create_char_embedding(
        vocab_size: int = 70,
        embedding_dim: int = 25,
        mask_zero: bool = False
    ) -> layers.Embedding:
        """
        Create character embedding layer.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of embeddings
            mask_zero: Whether to mask zero values
            
        Returns:
            Embedding layer
        """
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=mask_zero,
            name="char_embedding"
        )
        
        logger.info(f"Created char embedding: vocab_size={vocab_size}, "
                   f"dim={embedding_dim}")
        
        return embedding
    
    @staticmethod
    def create_hub_embedding(
        hub_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
        trainable: bool = False
    ) -> hub.KerasLayer:
        """
        Create TensorFlow Hub embedding layer.
        
        Args:
            hub_url: URL to TensorFlow Hub model
            trainable: Whether to fine-tune the embedding
            
        Returns:
            Hub KerasLayer
        """
        logger.info(f"Loading TensorFlow Hub embedding from: {hub_url}")
        
        hub_layer = hub.KerasLayer(
            hub_url,
            trainable=trainable,
            name="universal_sentence_encoder"
        )
        
        logger.info(f"Hub embedding loaded (trainable={trainable})")
        
        return hub_layer


class TokenEmbeddingBlock(tf.keras.Model):
    """Token embedding processing block."""
    
    def __init__(
        self,
        use_pretrained: bool = True,
        hub_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
        dense_units: int = 128,
        **kwargs
    ):
        """
        Initialize token embedding block.
        
        Args:
            use_pretrained: Whether to use pretrained embeddings
            hub_url: TensorFlow Hub URL
            dense_units: Number of dense layer units
        """
        super().__init__(**kwargs)
        
        if use_pretrained:
            self.embedding = EmbeddingLayers.create_hub_embedding(hub_url)
        else:
            # Custom embedding would be created here
            raise NotImplementedError("Custom token embedding not implemented")
        
        self.dense = layers.Dense(dense_units, activation="relu", name="token_dense")
    
    def call(self, inputs):
        """Forward pass."""
        x = self.embedding(inputs)
        x = self.dense(x)
        return x


class CharacterEmbeddingBlock(tf.keras.Model):
    """Character embedding processing block."""
    
    def __init__(
        self,
        vocab_size: int = 70,
        embedding_dim: int = 25,
        lstm_units: int = 32,
        **kwargs
    ):
        """
        Initialize character embedding block.
        
        Args:
            vocab_size: Character vocabulary size
            embedding_dim: Character embedding dimension
            lstm_units: LSTM units
        """
        super().__init__(**kwargs)
        
        self.embedding = EmbeddingLayers.create_char_embedding(
            vocab_size, embedding_dim
        )
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(lstm_units),
            name="char_bi_lstm"
        )
    
    def call(self, inputs):
        """Forward pass."""
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        return x


class PositionalEmbeddingBlock(tf.keras.Model):
    """Positional embedding processing block."""
    
    def __init__(self, dense_units: int = 32, **kwargs):
        """
        Initialize positional embedding block.
        
        Args:
            dense_units: Number of dense layer units
        """
        super().__init__(**kwargs)
        
        self.line_number_dense = layers.Dense(
            dense_units, 
            activation="relu", 
            name="line_number_dense"
        )
        self.total_lines_dense = layers.Dense(
            dense_units, 
            activation="relu", 
            name="total_lines_dense"
        )
    
    def call(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tuple of (line_numbers_one_hot, total_lines_one_hot)
        """
        line_numbers, total_lines = inputs
        
        line_features = self.line_number_dense(line_numbers)
        total_features = self.total_lines_dense(total_lines)
        
        return line_features, total_features