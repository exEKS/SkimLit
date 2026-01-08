"""Build TensorFlow datasets for training and evaluation."""

import tensorflow as tf
from typing import Tuple, Dict
from loguru import logger


class DatasetBuilder:
    """Build efficient TensorFlow datasets from processed data."""
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize DatasetBuilder.
        
        Args:
            batch_size: Batch size for datasets
        """
        self.batch_size = batch_size
    
    def build_dataset(
        self,
        sentences: list,
        labels_one_hot: tf.Tensor,
        shuffle: bool = True,
        repeat: bool = False
    ) -> tf.data.Dataset:
        """
        Build simple token-only dataset.
        
        Args:
            sentences: List of sentences
            labels_one_hot: One-hot encoded labels
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset
            
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((sentences, labels_one_hot))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if repeat:
            dataset = dataset.repeat()
        
        return dataset
    
    def build_char_token_dataset(
        self,
        sentences: list,
        char_sequences: list,
        labels_one_hot: tf.Tensor,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Build dataset with both token and character sequences.
        
        Args:
            sentences: List of sentences
            char_sequences: List of character-separated sentences
            labels_one_hot: One-hot encoded labels
            shuffle: Whether to shuffle
            
        Returns:
            tf.data.Dataset with structure ((sentences, chars), labels)
        """
        # Create data tensors
        data = tf.data.Dataset.from_tensor_slices((sentences, char_sequences))
        labels = tf.data.Dataset.from_tensor_slices(labels_one_hot)
        
        # Combine data and labels
        dataset = tf.data.Dataset.zip((data, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_tribrid_dataset(
        self,
        line_numbers_one_hot: tf.Tensor,
        total_lines_one_hot: tf.Tensor,
        sentences: list,
        char_sequences: list,
        labels_one_hot: tf.Tensor,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Build complete tribrid dataset with all features.
        
        Args:
            line_numbers_one_hot: One-hot encoded line numbers
            total_lines_one_hot: One-hot encoded total lines
            sentences: List of sentences
            char_sequences: List of character-separated sentences
            labels_one_hot: One-hot encoded labels
            shuffle: Whether to shuffle
            
        Returns:
            tf.data.Dataset with structure:
            ((line_numbers, total_lines, sentences, chars), labels)
        """
        # Create data tensors
        data = tf.data.Dataset.from_tensor_slices((
            line_numbers_one_hot,
            total_lines_one_hot,
            sentences,
            char_sequences
        ))
        
        labels = tf.data.Dataset.from_tensor_slices(labels_one_hot)
        
        # Combine data and labels
        dataset = tf.data.Dataset.zip((data, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Built tribrid dataset with batch_size={self.batch_size}")
        
        return dataset
    
    def build_all_datasets(self, data_dict: Dict) -> Dict[str, tf.data.Dataset]:
        """
        Build all dataset splits (train, val, test).
        
        Args:
            data_dict: Dictionary containing all preprocessed data
            
        Returns:
            Dictionary with dataset splits
        """
        datasets = {}
        
        for split in ["train", "val", "test"]:
            logger.info(f"Building {split} dataset...")
            
            shuffle = (split == "train")
            
            dataset = self.build_tribrid_dataset(
                line_numbers_one_hot=data_dict[f"{split}_line_numbers_one_hot"],
                total_lines_one_hot=data_dict[f"{split}_total_lines_one_hot"],
                sentences=data_dict[f"{split}_sentences"],
                char_sequences=data_dict[f"{split}_chars"],
                labels_one_hot=data_dict[f"{split}_labels_one_hot"],
                shuffle=shuffle
            )
            
            datasets[split] = dataset
        
        logger.info("All datasets built successfully")
        return datasets
    
    def get_dataset_element_spec(self, dataset: tf.data.Dataset) -> Dict:
        """
        Get the structure specification of a dataset.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Dictionary describing dataset structure
        """
        element_spec = dataset.element_spec
        
        spec_info = {
            "inputs": {},
            "outputs": {}
        }
        
        # Parse input structure
        if isinstance(element_spec[0], tuple):
            for i, spec in enumerate(element_spec[0]):
                spec_info["inputs"][f"input_{i}"] = {
                    "shape": spec.shape,
                    "dtype": spec.dtype.name
                }
        
        # Parse output structure
        spec_info["outputs"] = {
            "shape": element_spec[1].shape,
            "dtype": element_spec[1].dtype.name
        }
        
        return spec_info