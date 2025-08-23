# Configuration management for publication-level ablation studies
import yaml
import os
from typing import Dict, Any

def load_publication_ablation_config(config_file=None):
    """
    Load the comprehensive publication ablation configuration.
    
    Args:
        config_file (str): Path to the YAML config file. If None, uses default
    
    Returns:
        dict: Configuration dictionary
    """
    if config_file is None:
        # Default to the single comprehensive config file
        config_file = os.path.join(os.path.dirname(__file__), 'ablation_config.yaml')
    
    return load_ablation_config(config_file)

def create_preset_config(preset='full', output_file=None):
    """
    Create a configuration based on a preset from the publication config.
    
    Args:
        preset (str): Preset name ('quick', 'ntuple_only', 'objectives_only', 'hyperparams_only', 'architecture_only', 'full_study')
        output_file (str): Output file path. If None, returns config dict
    
    Returns:
        dict: Configuration dictionary if output_file is None
    """
    
    # Load the full publication config
    full_config = load_publication_ablation_config()
    
    if preset not in full_config['presets']:
        available_presets = list(full_config['presets'].keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available_presets}")
    
    # Create a config based on the preset
    preset_info = full_config['presets'][preset]
    config = full_config.copy()
    
    # Disable all experiments first
    for exp_name in config['experiments']:
        config['experiments'][exp_name]['enabled'] = False
    
    # Enable only the experiments specified in the preset
    if 'experiments' in preset_info:
        for exp_name in preset_info['experiments']:
            if exp_name in config['experiments']:
                config['experiments'][exp_name]['enabled'] = True
    
    # Apply quick test settings if specified
    if preset_info.get('quick_test', False):
        config['quick_test']['enabled'] = True
    
    if output_file:
        save_ablation_config(config, output_file)
        return output_file
    else:
        return config

def save_ablation_config(config: Dict[str, Any], filepath: str):
    """Save ablation configuration to YAML file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def load_ablation_config(filepath: str) -> Dict[str, Any]:
    """Load ablation configuration from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

# Utility functions for easy access
def get_available_presets():
    """Get list of available presets from the publication config."""
    config = load_publication_ablation_config()
    return list(config['presets'].keys())

def get_preset_description(preset):
    """Get description of a specific preset."""
    config = load_publication_ablation_config()
    if preset in config['presets']:
        return config['presets'][preset]['description']
    else:
        return f"Unknown preset: {preset}"
