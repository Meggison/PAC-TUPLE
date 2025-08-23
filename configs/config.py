import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy


def load_config(config_path="configs/config.yaml"):
    """Load hyperparameters and settings from the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    merged = deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_experiment_config(experiment_name: Optional[str] = None, 
                         config_path: str = "configs/config.yaml",
                         overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load and merge experiment configuration.
    
    Args:
        experiment_name: Name of preset experiment or 'base' for default
        config_path: Path to the YAML config file
        overrides: Dictionary of parameter overrides
        
    Returns:
        Merged configuration dictionary
    """
    # Load base configuration
    full_config = load_config(config_path)
    
    # Start with base configuration (main experiment setup)
    config = {k: v for k, v in full_config.items() if k not in ['ablations', 'presets']}
    
    # Apply preset configuration if specified
    if experiment_name and experiment_name in full_config.get('presets', {}):
        preset_config = full_config['presets'][experiment_name]
        config = merge_configs(config, preset_config)
        print(f"✓ Applied preset configuration: {experiment_name}")
    elif experiment_name and experiment_name != 'base':
        print(f"⚠️  Preset '{experiment_name}' not found, using base configuration")
    
    # Apply runtime overrides if provided
    if overrides:
        config = merge_configs(config, overrides)
        override_keys = list(flatten_config(overrides).keys())
        print(f"✓ Applied runtime overrides: {override_keys}")
    
    return config


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply runtime parameter overrides to config using dot notation."""
    config_copy = deepcopy(config)
    
    for key, value in overrides.items():
        keys = key.split('.')
        current = config_copy
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    return config_copy
    
    for value in values:
        # Create a copy of base config
        config = deepcopy(base_config)
        
        # Navigate to the parameter using dot notation
        keys = parameter_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the parameter value
        current[keys[-1]] = value
        
        # Update experiment name to include parameter value
        if 'experiment' in config and 'name' in config['experiment']:
            original_name = config['experiment']['name']
            config['experiment']['name'] = f"{original_name}_{parameter_path.replace('.', '_')}-{value}"
        
        configs.append(config)
    
    return configs


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary for wandb logging.
    
    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion
        sep: Separator for flattened keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def extract_runexp_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters for the runexp function from configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary with runexp function parameters
    """
    params = {}
    
    # Map config sections to runexp parameters
    if 'data' in config:
        params.update({
            'name_data': config['data'].get('name', 'cifar10'),
            'N': config['data'].get('N', 5),
            'perc_train': config['data'].get('perc_train', 1.0),
            'perc_prior': config['data'].get('perc_prior', 0.2),
            'batch_size': config['data'].get('batch_size', 128),
        })
    
    if 'model' in config:
        params.update({
            'model': config['model'].get('type', 'cnn'),
            'layers': config['model'].get('layers', 4),
            'embedding_dim': config['model'].get('embedding_dim', 256),
            'dropout_prob': config['model'].get('dropout_prob', 0.2),
        })
    
    if 'pac_bayes' in config:
        params.update({
            'objective': config['pac_bayes'].get('objective', 'fquad'),
            'sigma_prior': config['pac_bayes'].get('sigma_prior', 0.03),
            'pmin': config['pac_bayes'].get('pmin', 1e-5),
            'delta': config['pac_bayes'].get('delta', 0.025),
            'delta_test': config['pac_bayes'].get('delta_test', 0.01),
            'kl_penalty': config['pac_bayes'].get('kl_penalty', 1.0),
            'prior_dist': config['pac_bayes'].get('prior_dist', 'gaussian'),
            'initial_lamb': config['pac_bayes'].get('initial_lamb', 6.0),
        })
    
    if 'training' in config:
        params.update({
            'train_epochs': config['training'].get('train_epochs', 100),
            'prior_epochs': config['training'].get('prior_epochs', 20),
            'learning_rate': config['training'].get('learning_rate', 0.0005),
            'momentum': config['training'].get('momentum', 0.95),
            'learning_rate_prior': config['training'].get('learning_rate_prior', 0.01),
            'momentum_prior': config['training'].get('momentum_prior', 0.9),
            'run_baseline': config['training'].get('run_baseline', True),
        })
    
    if 'evaluation' in config:
        params.update({
            'mc_samples': config['evaluation'].get('mc_samples', 1000),
            'samples_ensemble': config['evaluation'].get('samples_ensemble', 1000),
        })
    
    if 'experiment' in config:
        params.update({
            'random_seed': config['experiment'].get('random_seed', 42),
            'device': config['experiment'].get('device', 'cuda'),
            'debug_mode': config['experiment'].get('debug_mode', True),
            'verbose': config['experiment'].get('verbose', False),
            'verbose_test': config['experiment'].get('verbose_test', False),
        })
    
    if 'wandb' in config:
        params.update({
            'use_wandb': config['wandb'].get('enabled', True),
            'wandb_project': config['wandb'].get('project', None),
        })
    
    return params


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Validate data configuration
    if 'data' in config:
        data_config = config['data']
        if data_config.get('name') not in ['cifar10', 'mnist']:
            errors.append("data.name must be 'cifar10' or 'mnist'")
        if data_config.get('N', 0) < 2:
            errors.append("data.N must be >= 2")
        if not 0 < data_config.get('perc_train', 1) <= 1:
            errors.append("data.perc_train must be between 0 and 1")
        if not 0 <= data_config.get('perc_prior', 0.2) <= 1:
            errors.append("data.perc_prior must be between 0 and 1")
        if data_config.get('batch_size', 1) < 1:
            errors.append("data.batch_size must be >= 1")
    
    # Validate model configuration
    if 'model' in config:
        model_config = config['model']
        if model_config.get('type') not in ['cnn', 'fcn']:
            errors.append("model.type must be 'cnn' or 'fcn'")
        if model_config.get('layers') not in [4, 9, 13, 15]:
            errors.append("model.layers must be 4, 9, 13, or 15")
        if model_config.get('embedding_dim', 1) < 1:
            errors.append("model.embedding_dim must be >= 1")
        if not 0 <= model_config.get('dropout_prob', 0.2) <= 1:
            errors.append("model.dropout_prob must be between 0 and 1")
    
    # Validate PAC-Bayes configuration
    if 'pac_bayes' in config:
        pb_config = config['pac_bayes']
        valid_objectives = ['fquad', 'fclassic', 'flamb', 'nested_ntuple', 'theory_ntuple']
        if pb_config.get('objective') not in valid_objectives:
            errors.append(f"pac_bayes.objective must be one of: {', '.join(valid_objectives)}")
        if pb_config.get('sigma_prior', 0) <= 0:
            errors.append("pac_bayes.sigma_prior must be > 0")
        if pb_config.get('pmin', 0) <= 0:
            errors.append("pac_bayes.pmin must be > 0")
        if not 0 < pb_config.get('delta', 0.025) < 1:
            errors.append("pac_bayes.delta must be between 0 and 1")
        if pb_config.get('kl_penalty', 0) <= 0:
            errors.append("pac_bayes.kl_penalty must be > 0")
    
    # Validate training configuration
    if 'training' in config:
        train_config = config['training']
        if train_config.get('train_epochs', 1) < 1:
            errors.append("training.train_epochs must be >= 1")
        if train_config.get('learning_rate', 0) <= 0:
            errors.append("training.learning_rate must be > 0")
        if not 0 <= train_config.get('momentum', 0.9) <= 1:
            errors.append("training.momentum must be between 0 and 1")
    
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the current configuration."""
    print("\n=== Experiment Configuration Summary ===")
    
    if 'experiment' in config:
        print(f"Experiment: {config['experiment'].get('name', 'unnamed')}")
        print(f"Random Seed: {config['experiment'].get('random_seed', 42)}")
        print(f"Device: {config['experiment'].get('device', 'cuda')}")
    
    if 'data' in config:
        print(f"Dataset: {config['data'].get('name', 'cifar10')}")
        print(f"N-tuple Size: {config['data'].get('N', 5)}")
        print(f"Batch Size: {config['data'].get('batch_size', 128)}")
    
    if 'model' in config:
        print(f"Model: {config['model'].get('type', 'cnn')}")
        print(f"Layers: {config['model'].get('layers', 4)}")
        print(f"Embedding Dim: {config['model'].get('embedding_dim', 256)}")
    
    if 'pac_bayes' in config:
        print(f"Objective: {config['pac_bayes'].get('objective', 'fquad')}")
        print(f"Sigma Prior: {config['pac_bayes'].get('sigma_prior', 0.03)}")
        print(f"KL Penalty: {config['pac_bayes'].get('kl_penalty', 1.0)}")
    
    if 'training' in config:
        print(f"Epochs: {config['training'].get('train_epochs', 100)}")
        print(f"Learning Rate: {config['training'].get('learning_rate', 0.0005)}")
    
    if 'wandb' in config:
        print(f"Wandb Enabled: {config['wandb'].get('enabled', True)}")
        print(f"Project: {config['wandb'].get('project', 'pac-bayes-reid-experiments')}")
    
    print("=" * 45)