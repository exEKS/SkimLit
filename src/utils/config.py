"""Configuration management utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    # Load based on file extension
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    logger.info("Configuration loaded successfully")
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving configuration to {save_path}")
    
    # Save based on file extension
    if save_path.suffix in [".yaml", ".yml"]:
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif save_path.suffix == ".json":
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {save_path.suffix}")
    
    logger.info("Configuration saved successfully")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["model", "training", "data"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model config
    model_config = config["model"]
    required_model_keys = ["token_embedding", "char_embedding", "positional_embedding"]
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model config key: {key}")
    
    # Validate training config
    training_config = config["training"]
    required_training_keys = ["batch_size", "epochs", "learning_rate"]
    
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")
    
    logger.info("Configuration validation passed")
    return True


class ConfigManager:
    """Manage configuration with defaults and overrides."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load()
    
    def load(self) -> Dict:
        """Load configuration."""
        if self.config_path.exists():
            return load_config(str(self.config_path))
        else:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self.get_default_config()
    
    def save(self, save_path: str = None):
        """
        Save current configuration.
        
        Args:
            save_path: Path to save (uses default if None)
        """
        if save_path is None:
            save_path = self.config_path
        
        save_config(self.config, str(save_path))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "model.token_embedding.url")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict):
        """
        Update configuration with dictionary.
        
        Args:
            updates: Dictionary with updates
        """
        self.config = merge_configs(self.config, updates)
    
    @staticmethod
    def get_default_config() -> Dict:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "model": {
                "name": "skimlit_tribrid",
                "version": "2.0.0",
                "token_embedding": {
                    "type": "tensorflow_hub",
                    "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
                    "output_dim": 512,
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
                    "dropout_rate": 0.5,
                    "num_classes": 5
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 15,
                "learning_rate": 0.001,
                "validation_split": 0.1
            },
            "data": {
                "train_path": "data/train.txt",
                "val_path": "data/dev.txt",
                "test_path": "data/test.txt"
            }
        }