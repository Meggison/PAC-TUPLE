# Train Utility Module for Publication-Level Ablation Studies
import torch
import numpy as np
import pandas as pd
import os
import warnings
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# Import from your existing experiment module
from experiment import runexp

def runexp_with_structured_output(**config) -> Dict[str, Any]:
    """
    Wrapper function for runexp that provides structured output for ablation studies.
    This function standardizes the output format and adds error handling for batch experiments.
    
    Args:
        **config: Configuration parameters for runexp
        
    Returns:
        Dict containing structured results with status information
    """
    try:
        # Extract experiment metadata
        experiment_name = config.get('experiment_name', 'unknown')
        
        # Run the main experiment
        results = runexp(**config)
        
        if results is None:
            return {
                'status': 'failed',
                'error': 'runexp returned None',
                'experiment_name': experiment_name,
                'config': config
            }
        
        # Structure the output with all key metrics
        structured_results = {
            'status': 'success',
            'experiment_name': experiment_name,
            'config': config,
            
            # Core PAC-Bayes metrics
            'train_obj': results.get('train_obj'),
            'risk_ntuple': results.get('risk_ntuple'),
            'empirical_risk_ntuple': results.get('empirical_risk_ntuple'),
            'kl_per_n': results.get('kl_per_n'),
            'pseudo_accuracy': results.get('pseudo_accuracy'),
            
            # Stochastic network evaluation
            'stch_risk': results.get('stch_risk'),
            'stch_accuracy': results.get('stch_accuracy'),
            'stch_map': results.get('stch_map'),
            'stch_rank1': results.get('stch_rank1'),
            
            # Posterior mean evaluation
            'post_risk': results.get('post_risk'),
            'post_accuracy': results.get('post_accuracy'),
            'post_map': results.get('post_map'),
            'post_rank1': results.get('post_rank1'),
            
            # Ensemble evaluation
            'ens_risk': results.get('ens_risk'),
            'ens_accuracy': results.get('ens_accuracy'),
            'ens_map': results.get('ens_map'),
            'ens_rank1': results.get('ens_rank1'),
            
            # Baseline metrics
            'baseline_error': results.get('baseline_error'),
            'prior_map': results.get('prior_map'),
            'prior_rank1': results.get('prior_rank1'),
            
            # Quality indicators for filtering
            'is_vacuous': results.get('risk_ntuple', 1.0) >= 1.0,
            'kl_reasonable': results.get('kl_per_n', 0.0) > 1e-6,
            'good_accuracy': results.get('stch_accuracy', 0.0) > 0.6,
        }
        
        return structured_results
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'experiment_name': config.get('experiment_name', 'unknown'),
            'config': config
        }

def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """
    Validate that the configuration contains all required parameters for runexp.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_params = {
        'name_data', 'objective', 'model', 'N', 'sigma_prior', 'pmin', 
        'learning_rate', 'momentum'
    }
    
    missing_params = required_params - set(config.keys())
    if missing_params:
        print(f"Missing required parameters: {missing_params}")
        return False
    
    # Validate ranges
    if config.get('N', 0) < 2:
        print(f"N must be >= 2, got {config.get('N')}")
        return False
        
    if config.get('sigma_prior', 0) <= 0:
        print(f"sigma_prior must be positive, got {config.get('sigma_prior')}")
        return False
        
    return True

def setup_ablation_environment(results_dir: str = "ablation_results") -> str:
    """
    Set up the environment for running ablation studies.
    
    Args:
        results_dir: Directory to store results
        
    Returns:
        Path to the results directory
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(f"{results_dir}/logs", exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    
    return results_dir
