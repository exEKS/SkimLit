"""Data loading utilities for SkimLit."""

from pathlib import Path
from typing import List, Dict
import pandas as pd
from loguru import logger


class DataLoader:
    """Load and parse RCT abstract data from text files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        
    def get_lines(self, filename: str) -> List[str]:
        """
        Read lines from a text file.
        
        Args:
            filename: Path to the text file
            
        Returns:
            List of strings, one per line
        """
        filepath = self.data_dir / filename
        logger.info(f"Reading file: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            logger.info(f"Read {len(lines)} lines from {filename}")
            return lines
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise
    
    def preprocess_text_with_line_numbers(self, filename: str) -> List[Dict]:
        """
        Process text file into structured format with line numbers.
        
        Each abstract is parsed into individual lines with metadata:
        - target: label (BACKGROUND, OBJECTIVE, etc.)
        - text: the sentence text
        - line_number: position in the abstract (0-indexed)
        - total_lines: total sentences in the abstract
        
        Args:
            filename: Name of the text file to process
            
        Returns:
            List of dictionaries containing parsed abstract lines
        """
        input_lines = self.get_lines(filename)
        abstract_lines = ""
        abstract_samples = []
        
        logger.info(f"Processing {filename}...")
        
        for line in input_lines:
            # Check for abstract ID (start of new abstract)
            if line.startswith("###"):
                abstract_id = line
                abstract_lines = ""
                
            # Check for new line (end of abstract)
            elif line.isspace():
                abstract_line_split = abstract_lines.splitlines()
                
                # Process each line in the abstract
                for line_num, abstract_line in enumerate(abstract_line_split):
                    line_data = {}
                    
                    # Split label from text (separated by tab)
                    target_text_split = abstract_line.split("\t")
                    
                    if len(target_text_split) >= 2:
                        line_data["target"] = target_text_split[0]
                        line_data["text"] = target_text_split[1].lower()
                        line_data["line_number"] = line_num
                        line_data["total_lines"] = len(abstract_line_split) - 1
                        
                        abstract_samples.append(line_data)
                        
            else:
                # Regular line - add to current abstract
                abstract_lines += line
        
        logger.info(f"Processed {len(abstract_samples)} lines from {filename}")
        return abstract_samples
    
    def load_dataset(self, split: str = "train") -> pd.DataFrame:
        """
        Load a dataset split as a DataFrame.
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            
        Returns:
            DataFrame with columns: target, text, line_number, total_lines
        """
        filename_map = {
            "train": "train.txt",
            "dev": "dev.txt",
            "val": "dev.txt",
            "test": "test.txt"
        }
        
        if split not in filename_map:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(filename_map.keys())}")
        
        filename = filename_map[split]
        samples = self.preprocess_text_with_line_numbers(filename)
        df = pd.DataFrame(samples)
        
        logger.info(f"Loaded {split} dataset with {len(df)} samples")
        logger.info(f"Label distribution:\n{df['target'].value_counts()}")
        
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all dataset splits.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataFrames
        """
        logger.info("Loading all datasets...")
        
        datasets = {
            "train": self.load_dataset("train"),
            "val": self.load_dataset("dev"),
            "test": self.load_dataset("test")
        }
        
        logger.info("All datasets loaded successfully")
        return datasets
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(df),
            "label_distribution": df["target"].value_counts().to_dict(),
            "avg_line_length": df["text"].str.split().str.len().mean(),
            "max_line_length": df["text"].str.split().str.len().max(),
            "avg_abstract_length": df["total_lines"].mean(),
            "max_abstract_length": df["total_lines"].max(),
        }
        
        return stats