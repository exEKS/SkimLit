"""Model architectures for SkimLit."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Optional
from loguru import logger

from .embeddings import (
    TokenEmbeddingBlock,
    CharacterEmbeddingBlock,
    PositionalEmbeddingBlock,
    EmbeddingLayers
)


class TribridModel:
    """
    Tribrid model combining token, character, and positional embeddings.
    
    Architecture:
        - Token branch: Pretrained USE + Dense
        - Character branch: Char embedding + Bi-LSTM
        - Position branch: Line number + Total lines Dense layers
        - Fusion: Concatenate all + Dense layers + Dropout
        - Output: Softmax classification
    """
    
    def __init__(self, config: Dict):
        """
        Initialize tribrid model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.token_vectorizer = None
        self.char_vectorizer = None
        
    def build(
        self,
        num_classes: int = 5,
        compile_model: bool = True
    ) -> Model:
        """
        Build the complete tribrid model.
        
        Args:
            num_classes: Number of output classes
            compile_model: Whether to compile the model
            
        Returns:
            Compiled Keras Model
        """
        logger.info("Building tribrid model...")
        
        # Configuration
        cfg = self.config.get("model", {})
        token_cfg = cfg.get("token_embedding", {})
        char_cfg = cfg.get("char_embedding", {})
        pos_cfg = cfg.get("positional_embedding", {})
        arch_cfg = cfg.get("architecture", {})
        
        # ========== INPUTS ==========
        # Token input
        token_input = layers.Input(
            shape=[], 
            dtype=tf.string, 
            name="token_input"
        )
        
        # Character input
        char_input = layers.Input(
            shape=(1,), 
            dtype=tf.string, 
            name="char_input"
        )
        
        # Positional inputs
        line_number_input = layers.Input(
            shape=(pos_cfg.get("line_number_depth", 15),),
            dtype=tf.float32,
            name="line_number_input"
        )
        
        total_lines_input = layers.Input(
            shape=(pos_cfg.get("total_lines_depth", 20),),
            dtype=tf.float32,
            name="total_lines_input"
        )
        
        # ========== TOKEN BRANCH ==========
        token_block = TokenEmbeddingBlock(
            use_pretrained=True,
            hub_url=token_cfg.get("url"),
            dense_units=arch_cfg.get("token_dense_units", 128)
        )
        token_output = token_block(token_input)
        
        # ========== CHARACTER BRANCH ==========
        # Create and adapt character vectorizer
        self.char_vectorizer = EmbeddingLayers.create_char_vectorizer(
            max_tokens=char_cfg.get("vocab_size", 70),
            output_sequence_length=char_cfg.get("max_sequence_length", 290)
        )
        
        char_vectors = self.char_vectorizer(char_input)
        
        char_block = CharacterEmbeddingBlock(
            vocab_size=char_cfg.get("vocab_size", 70),
            embedding_dim=char_cfg.get("embedding_dim", 25),
            lstm_units=char_cfg.get("lstm_units", 32)
        )
        char_output = char_block(char_vectors)
        
        # ========== POSITIONAL BRANCH ==========
        pos_block = PositionalEmbeddingBlock(
            dense_units=pos_cfg.get("dense_units", 32)
        )
        line_features, total_features = pos_block(
            (line_number_input, total_lines_input)
        )
        
        # ========== FUSION NETWORK ==========
        # Concatenate token and character embeddings
        token_char_concat = layers.Concatenate(
            name="token_char_concat"
        )([token_output, char_output])
        
        # Dense layer after token-char fusion
        combined = layers.Dense(
            arch_cfg.get("combined_dense_units", 256),
            activation="relu",
            name="combined_dense"
        )(token_char_concat)
        
        combined = layers.Dropout(
            arch_cfg.get("dropout_rate", 0.5),
            name="combined_dropout"
        )(combined)
        
        # Concatenate with positional features
        tribrid_concat = layers.Concatenate(
            name="tribrid_concat"
        )([line_features, total_features, combined])
        
        # ========== OUTPUT ==========
        output = layers.Dense(
            num_classes,
            activation="softmax",
            name="output"
        )(tribrid_concat)
        
        # ========== CREATE MODEL ==========
        self.model = Model(
            inputs=[
                line_number_input,
                total_lines_input,
                token_input,
                char_input
            ],
            outputs=output,
            name="skimlit_tribrid_model"
        )
        
        logger.info("Tribrid model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Compile if requested
        if compile_model:
            self.compile()
        
        return self.model
    
    def compile(self):
        """Compile the model with optimizer and loss."""
        training_cfg = self.config.get("training", {})
        optimizer_cfg = training_cfg.get("optimizer", {})
        loss_cfg = training_cfg.get("loss", {})
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_cfg.get("learning_rate", 0.001),
            beta_1=optimizer_cfg.get("beta_1", 0.9),
            beta_2=optimizer_cfg.get("beta_2", 0.999)
        )
        
        # Create loss
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=loss_cfg.get("label_smoothing", 0.2)
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )
        
        logger.info("Model compiled")
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model.summary()
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Saving model to {filepath}")
        self.model.save(filepath)
        logger.info("Model saved successfully")
    
    @classmethod
    def load(cls, filepath: str, config: Dict) -> "TribridModel":
        """
        Load a saved model.
        
        Args:
            filepath: Path to saved model
            config: Configuration dictionary
            
        Returns:
            TribridModel instance with loaded model
        """
        logger.info(f"Loading model from {filepath}")
        
        instance = cls(config)
        instance.model = tf.keras.models.load_model(filepath)
        
        logger.info("Model loaded successfully")
        return instance


class SimpleTokenModel:
    """Simple baseline model using only token embeddings."""
    
    def __init__(self, config: Dict):
        """Initialize simple token model."""
        self.config = config
        self.model = None
    
    def build(self, num_classes: int = 5) -> Model:
        """Build simple token-only model."""
        logger.info("Building simple token model...")
        
        # Input
        token_input = layers.Input(shape=[], dtype=tf.string, name="token_input")
        
        # Embedding
        token_block = TokenEmbeddingBlock(
            use_pretrained=True,
            dense_units=128
        )
        token_output = token_block(token_input)
        
        # Output
        output = layers.Dense(num_classes, activation="softmax")(token_output)
        
        self.model = Model(inputs=token_input, outputs=output)
        
        # Compile
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        logger.info("Simple token model built")
        return self.model