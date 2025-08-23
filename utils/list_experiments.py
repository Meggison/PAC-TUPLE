#!/usr/bin/env python3
"""
Show available experiment configurations and ablation studies.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import load_config

def list_experiments():
    """List all available configurations."""
    
    print("=== PAC-Bayes ReID Experiment System ===\n")
    
    try:
        config = load_config("configs/config.yaml")
        
        # Show base configuration
        print("🏠 **Base Configuration** (your main setup)")
        exp_info = config.get('experiment', {})
        data_info = config.get('data', {})
        model_info = config.get('model', {})
        pac_bayes_info = config.get('pac_bayes', {})
        training_info = config.get('training', {})
        
        print(f"   Name: {exp_info.get('name', 'N/A')}")
        print(f"   Description: {exp_info.get('description', 'No description')}")
        print(f"   Dataset: {data_info.get('name', 'N/A')}, N-tuple: {data_info.get('N', 'N/A')}")
        print(f"   Model: {model_info.get('type', 'N/A')}, Layers: {model_info.get('layers', 'N/A')}")
        print(f"   Objective: {pac_bayes_info.get('objective', 'N/A')}, σ_prior: {pac_bayes_info.get('sigma_prior', 'N/A')}")
        print(f"   LR: {training_info.get('learning_rate', 'N/A')}, Epochs: {training_info.get('train_epochs', 'N/A')}")
        print()
        
        # Show preset configurations
        presets = config.get('presets', {})
        if presets:
            print("⚙️  **Preset Configurations**")
            for preset_name, preset_config in presets.items():
                preset_exp = preset_config.get('experiment', {})
                print(f"   📋 {preset_name}: {preset_exp.get('name', 'N/A')}")
                
                # Show key differences from base
                differences = []
                if 'training' in preset_config and 'train_epochs' in preset_config['training']:
                    differences.append(f"epochs={preset_config['training']['train_epochs']}")
                if 'data' in preset_config and 'name' in preset_config['data']:
                    differences.append(f"dataset={preset_config['data']['name']}")
                if 'wandb' in preset_config and 'enabled' in preset_config['wandb']:
                    differences.append(f"wandb={preset_config['wandb']['enabled']}")
                
                if differences:
                    print(f"      Modifications: {', '.join(differences)}")
            print()
        
        print("=== Usage Examples ===")
        print("# Run base configuration:")
        print("  python experiment.py")
        print("  python run_config_experiment.py")
        print()
        print("# Run preset configurations:")
        if presets:
            for preset_name in presets.keys():
                print(f"  python run_config_experiment.py --experiment {preset_name}")
        print()
        print("# Override specific parameters:")
        print("  python run_config_experiment.py --override training.train_epochs=5")
        print("  python run_config_experiment.py --override wandb.enabled=true")
        
    except Exception as e:
        print(f"Error loading configurations: {e}")
        print("Make sure configs/config.yaml exists and is valid.")


if __name__ == "__main__":
    list_experiments()
