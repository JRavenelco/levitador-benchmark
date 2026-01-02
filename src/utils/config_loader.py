"""
Configuration Loader Utility
=============================

Utility for loading and parsing YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Examples
    --------
    >>> config = load_config('config/default.yaml')
    >>> print(config['random_seed'])
    42
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate
    
    Returns
    -------
    bool
        True if configuration is valid
    
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    required_keys = ['algorithms', 'benchmark']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    return True
