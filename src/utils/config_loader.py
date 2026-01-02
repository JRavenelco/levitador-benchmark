"""
Configuration Loader
===================

Utilities for loading and managing YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_optimizer_config(config: Dict[str, Any], 
                        optimizer_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific optimizer.
    
    Args:
        config: Full configuration dictionary
        optimizer_name: Name of the optimizer
        
    Returns:
        Optimizer-specific configuration or None if not found
    """
    optimizers = config.get('optimizers', {})
    return optimizers.get(optimizer_name)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['benchmark', 'optimizers']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    return True
