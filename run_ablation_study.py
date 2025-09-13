#!/usr/bin/env python3
"""
Runner script for PAC-Bayes ablation studies.

This script provides a convenient interface to run different types of ablation studies
with proper configuration management and WandB integration.

Usage:
    python run_ablation_study.py --preset quick          # Quick test
    python run_ablation_study.py --preset ntuple_only    # N-tuple analysis only  
    python run_ablation_study.py --preset full_study     # Full ablation study
    python run_ablation_study.py --config path/to/config.yaml  # Custom config
    python run_ablation_study.py --list-presets          # Show available presets
"""

import argparse
import sys
import os
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print a nice banner for the ablation study"""
    print("="*80)
    print("🚀 PAC-Bayes Publication-Level Ablation Study Runner")
    print("="*80)
    print()

def print_config_summary(config, enabled_experiments):
    """Print a summary of the configuration"""
    print("📋 Configuration Summary:")
    print(f"   Random seeds: {config.get('random_seeds', [42])}")
    print(f"   WandB project: {config.get('wandb_settings', {}).get('project', 'Not specified')}")
    print()
    
    print("🧪 Enabled experiments:")
    total_time = 0
    for exp_name in enabled_experiments:
        if exp_name in config['experiments']:
            exp_config = config['experiments'][exp_name]
            time_hours = exp_config.get('estimated_time_hours', 1)
            total_time += time_hours
            print(f"   ✓ {exp_name}: {exp_config.get('description', 'No description')} ({time_hours}h)")
    
    print(f"\n   ⏱️ Total estimated time: {total_time} hours")
    print()

def run_ablation_study(config, enabled_experiments):
    """Run the ablation study with the given configuration"""
    try:
        # Import here to avoid circular imports
        from scripts.publication_level_ablation import PACBayesAblation
        
        print("🔬 Starting Publication-Level PAC-Bayes Ablation Study")
        print("="*60)
        
        # Initialize the ablation study
        base_config = config.get('base_config', {})
        wandb_settings = config.get('wandb_settings', {})
        use_wandb = wandb_settings.get('use_wandb', True)
        wandb_project = wandb_settings.get('project', 'pac-bayes-ablation')
        
        ablation = PACBayesAblation(
            base_config=base_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project
        )
        
        # Pass the full config to the ablation study for experiment selection
        ablation.config = config
        ablation.enabled_experiments = enabled_experiments
        ablation.random_seeds = config.get('random_seeds', [42, 123, 456])
        
        # Run the study
        results = ablation.run_publication_ablation()
        
        print("\n✅ Ablation study completed successfully!")
        print(f"Results saved to: {results.get('output_directory', 'publication_ablation_results')}")
        
        # Print summary
        if 'summary' in results:
            print("\nResults Summary:")
            for key, value in results['summary'].items():
                print(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during ablation study: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run publication-level PAC-Bayes ablation studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ablation_study.py --preset ntuple_only
  python run_ablation_study.py --preset objectives_only
  python run_ablation_study.py --preset full_study
  python run_ablation_study.py --config custom_config.yaml
  python run_ablation_study.py --list-presets
        """
    )
    
    # Add argument for config file or preset
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom YAML configuration file')
    parser.add_argument('--preset', type=str, default='ntuple_only',
                       choices=['ntuple_only', 'objectives_only', 'hyperparams_only', 'architecture_only', 'full_study'],
                       help='Preset configuration to use (default: ntuple_only)')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available presets and exit')
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='Override WandB project name')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    print_banner()
    
    # List presets if requested
    if args.list_presets:
        from configs.ablation_config import get_available_presets, get_preset_description
        print("📋 Available presets:")
        for preset in get_available_presets():
            print(f"   {preset}: {get_preset_description(preset)}")
        return
    
    # Load configuration
    if args.config:
        # Load custom config file
        from configs.ablation_config import load_ablation_config
        config = load_ablation_config(args.config)
        print(f"📁 Loaded custom configuration from: {args.config}")
    else:
        # Use preset
        from configs.ablation_config import create_preset_config
        config = create_preset_config(args.preset)
        print(f"⚙️  Using preset: {args.preset}")
    
    # Override WandB project if specified
    if args.wandb_project:
        config['wandb_settings']['project'] = args.wandb_project
        print(f"📊 WandB project overridden to: {args.wandb_project}")
    
    # Determine enabled experiments
    enabled_experiments = [name for name, exp in config['experiments'].items() if exp.get('enabled', False)]
    
    # Print configuration summary
    print_config_summary(config, enabled_experiments)
    
    if args.dry_run:
        print("📋 Dry run mode - showing what would be executed:")
        print("   (No experiments will actually run)")
        print("\nRun without --dry-run to execute the ablation study.")
        return
    
    # Confirm before running 
    estimated_time = sum(config['experiments'][exp]['estimated_time_hours'] 
                        for exp in enabled_experiments 
                        if exp in config['experiments'])
    
    print(f"⚠️  This will take approximately {estimated_time} hours to complete.")
    confirm = input("Continue with ablation study? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Ablation study cancelled.")
        return
    
    # Record start time
    start_time = datetime.now()
    print(f"🚀 Starting ablation study at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the study
    results = run_ablation_study(config, enabled_experiments)
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏰ Study completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {duration}")
    
    if results:
        print("\n🎉 Ablation study completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Check the results directory for output files")
        print("   2. Review the WandB dashboard for detailed metrics")
        print("   3. Use the generated CSV/LaTeX files for publication")
    else:
        print("\n💥 Ablation study failed - check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
