"""Tests for data processing modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.data.dataset_builder import DatasetBuilder


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "line_number_depth": 15,
        "total_lines_depth": 20,
        "max_vocab_size": 1000,
        "output_sequence_length": 55
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for tests."""
    return pd.DataFrame({
        "target": ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
        "text": [
            "this is background",
            "this is objective",
            "this is methods",
            "these are results",
            "these are conclusions"
        ],
        "line_number": [0, 1, 2, 3, 4],
        "total_lines": [4, 4, 4, 4, 4]
    })


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader("data")
        assert loader.data_dir == Path("data")
    
    def test_get_dataset_statistics(self, sample_dataframe):
        """Test dataset statistics calculation."""
        loader = DataLoader()
        stats = loader.get_dataset_statistics(sample_dataframe)
        
        assert stats["total_samples"] == 5
        assert "label_distribution" in stats
        assert "avg_line_length" in stats


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    def test_initialization(self, sample_config):
        """Test TextPreprocessor initialization."""
        preprocessor = TextPreprocessor(sample_config)
        assert preprocessor.config == sample_config
    
    def test_split_chars(self, sample_config):
        """Test character splitting."""
        preprocessor = TextPreprocessor(sample_config)
        text = "hello"
        result = preprocessor.split_chars(text)
        assert result == "h e l l o"
    
    def test_prepare_sentences(self, sample_config, sample_dataframe):
        """Test sentence preparation."""
        preprocessor = TextPreprocessor(sample_config)
        sentences = preprocessor.prepare_sentences(sample_dataframe)
        
        assert len(sentences) == 5
        assert isinstance(sentences, list)
        assert sentences[0] == "this is background"
    
    def test_prepare_char_sequences(self, sample_config):
        """Test character sequence preparation."""
        preprocessor = TextPreprocessor(sample_config)
        sentences = ["hello", "world"]
        char_seqs = preprocessor.prepare_char_sequences(sentences)
        
        assert len(char_seqs) == 2
        assert char_seqs[0] == "h e l l o"
        assert char_seqs[1] == "w o r l d"
    
    def test_split_abstract_into_sentences(self, sample_config):
        """Test abstract sentence splitting."""
        preprocessor = TextPreprocessor(sample_config)
        preprocessor.load_spacy_model()
        
        abstract = "This is sentence one. This is sentence two. This is sentence three."
        sentences = preprocessor.split_abstract_into_sentences(abstract)
        
        assert len(sentences) == 3
        assert isinstance(sentences, list)


class TestDatasetBuilder:
    """Tests for DatasetBuilder class."""
    
    def test_initialization(self):
        """Test DatasetBuilder initialization."""
        builder = DatasetBuilder(batch_size=32)
        assert builder.batch_size == 32
    
    def test_build_dataset(self, sample_config):
        """Test simple dataset building."""
        import tensorflow as tf
        
        builder = DatasetBuilder(batch_size=2)
        sentences = ["hello world", "foo bar", "test sentence"]
        labels = tf.one_hot([0, 1, 2], depth=5)
        
        dataset = builder.build_dataset(sentences, labels, shuffle=False)
        
        # Check dataset structure
        for batch in dataset.take(1):
            texts, labs = batch
            assert texts.shape[0] <= 2  # batch size
            assert labs.shape[1] == 5  # num classes


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for complete data pipeline."""
    
    def test_full_preprocessing_pipeline(self, sample_config, sample_dataframe):
        """Test complete preprocessing pipeline."""
        preprocessor = TextPreprocessor(sample_config)
        
        # Prepare all components
        sentences = preprocessor.prepare_sentences(sample_dataframe)
        char_sequences = preprocessor.prepare_char_sequences(sentences)
        line_numbers_oh, total_lines_oh = preprocessor.prepare_positional_embeddings(
            sample_dataframe
        )
        
        # Verify outputs
        assert len(sentences) == 5
        assert len(char_sequences) == 5
        assert line_numbers_oh.shape[0] == 5
        assert total_lines_oh.shape[0] == 5
    
    def test_inference_data_preparation(self, sample_config):
        """Test inference data preparation."""
        preprocessor = TextPreprocessor(sample_config)
        preprocessor.load_spacy_model()
        
        abstract = "This is sentence one. This is sentence two."
        data = preprocessor.prepare_inference_data(abstract)
        
        assert "sentences" in data
        assert "char_sequences" in data
        assert "line_numbers_one_hot" in data
        assert "total_lines_one_hot" in data
        assert len(data["sentences"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])