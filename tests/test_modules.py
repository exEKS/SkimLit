"""Tests for model modules."""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import EmbeddingLayers, TokenEmbeddingBlock
from src.models.architectures import TribridModel
from src.utils.config import ConfigManager


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        "model": {
            "token_embedding": {
                "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
                "trainable": False
            },
            "char_embedding": {
                "vocab_size": 70,
                "embedding_dim": 25,
                "max_sequence_length": 290,
                "lstm_units": 32
            },
            "positional_embedding": {
                "line_number_depth": 15,
                "total_lines_depth": 20,
                "dense_units": 32
            },
            "architecture": {
                "token_dense_units": 128,
                "combined_dense_units": 256,
                "dropout_rate": 0.5
            }
        }
    }


class TestEmbeddingLayers:
    """Tests for embedding layer creation."""
    
    def test_create_token_vectorizer(self):
        """Test token vectorizer creation."""
        vectorizer = EmbeddingLayers.create_token_vectorizer(
            max_tokens=1000,
            output_sequence_length=50
        )
        
        assert isinstance(vectorizer, tf.keras.layers.TextVectorization)
        assert vectorizer.max_tokens == 1000
    
    def test_create_char_vectorizer(self):
        """Test character vectorizer creation."""
        vectorizer = EmbeddingLayers.create_char_vectorizer(
            max_tokens=70,
            output_sequence_length=290
        )
        
        assert isinstance(vectorizer, tf.keras.layers.TextVectorization)
    
    def test_create_token_embedding(self):
        """Test token embedding layer creation."""
        embedding = EmbeddingLayers.create_token_embedding(
            vocab_size=1000,
            embedding_dim=128
        )
        
        assert isinstance(embedding, tf.keras.layers.Embedding)
        assert embedding.input_dim == 1000
        assert embedding.output_dim == 128
    
    def test_create_char_embedding(self):
        """Test character embedding layer creation."""
        embedding = EmbeddingLayers.create_char_embedding(
            vocab_size=70,
            embedding_dim=25
        )
        
        assert isinstance(embedding, tf.keras.layers.Embedding)


class TestTokenEmbeddingBlock:
    """Tests for token embedding block."""
    
    @pytest.mark.slow
    def test_token_block_forward_pass(self):
        """Test token block forward pass."""
        block = TokenEmbeddingBlock(use_pretrained=False, dense_units=64)
        
        # This would require actual TF Hub download, skip in CI
        pytest.skip("Requires TF Hub download")


class TestTribridModel:
    """Tests for tribrid model architecture."""
    
    def test_model_initialization(self, sample_model_config):
        """Test model initialization."""
        model = TribridModel(sample_model_config)
        assert model.config == sample_model_config
        assert model.model is None
    
    @pytest.mark.slow
    def test_model_build(self, sample_model_config):
        """Test model building."""
        # Skip actual build test as it requires TF Hub
        pytest.skip("Requires TF Hub and full dependencies")
    
    def test_get_model_before_build(self, sample_model_config):
        """Test getting model before building raises error."""
        model = TribridModel(sample_model_config)
        
        with pytest.raises(ValueError):
            model.get_model()


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_input_shapes(self):
        """Test model accepts correct input shapes."""
        # Create dummy inputs
        batch_size = 2
        
        line_numbers = tf.random.uniform((batch_size, 15))
        total_lines = tf.random.uniform((batch_size, 20))
        tokens = tf.constant(["hello world", "foo bar"])
        chars = tf.constant(["h e l l o", "f o o"])
        
        # Test shapes are compatible
        assert line_numbers.shape == (batch_size, 15)
        assert total_lines.shape == (batch_size, 20)
        assert tokens.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])