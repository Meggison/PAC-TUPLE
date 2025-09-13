# Enhanced Publication-Level PAC-Bayes Ablation Study with WandB Integration
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from tqdm import tqdm
import json
import os
import sys
from datetime import datetime
import traceback
import time
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your project modules
from utils.train_util import runexp_with_structured_output, validate_experiment_config, setup_ablation_environment
from utils.wandb_logger import WandbLogger

class PACBayesAblation:
    def __init__(self, base_config, results_dir="publication_ablation_results", use_wandb=True, wandb_project="pac-bayes-ablation"):
        self.base_config = base_config
        self.results_dir = setup_ablation_environment(results_dir)
        self.results = []
        self.all_results = {}
        
        # WandB Integration
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_logger = WandbLogger(use_wandb=use_wandb) if use_wandb else None
        
        # Enhanced tracking integration (using unified wandb_logger)
        self.enhanced_tracker = WandbLogger(use_wandb=use_wandb)
        
        # Define supported parameters (from your experiment.py)
        self.supported_params = {
            'name_data', 'objective', 'model', 'N', 'sigma_prior', 'pmin', 
            'learning_rate', 'momentum', 'learning_rate_prior', 'momentum_prior',
            'delta', 'layers', 'delta_test', 'mc_samples', 'samples_ensemble',
            'kl_penalty', 'initial_lamb', 'train_epochs', 'prior_dist', 
            'verbose', 'device', 'prior_epochs', 'dropout_prob', 'perc_train', 
            'verbose_test', 'perc_prior', 'batch_size', 'embedding_dim', 
            'run_baseline', 'debug_mode', 'random_seed', 'use_wandb', 'wandb_project'
        }
        
        # Enhanced logging and tracking
        self.log_file = f"{results_dir}/logs/publication_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.successful_experiments = 0
        self.failed_experiments = 0
        self.start_time = None
        
        # Statistical tracking
        self.random_seeds = [42, 123, 456]  # Multiple seeds for statistical reliability
        self.experiment_results = defaultdict(list)  # For statistical analysis
        
        # WandB experiment tracking
        self.wandb_experiment_group = f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log(self, message):
        """Enhanced logging with timestamps and wandb integration"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
            f.flush()
    
    def init_wandb_for_experiment(self, experiment_type: str, config: dict):
        """Initialize WandB for a specific experiment within the ablation study"""
        if not self.use_wandb or self.wandb_logger is None:
            return False
            
        # Create experiment-specific config
        wandb_config = {
            **config,
            'ablation_study': True,
            'experiment_type': experiment_type,
            'experiment_group': self.wandb_experiment_group,
            'ablation_timestamp': datetime.now().isoformat()
        }
        
        # Generate experiment name
        experiment_name = f"{experiment_type}_{config.get('experiment_name', 'unknown')}"
        
        # Initialize wandb
        return self.wandb_logger.init_experiment(
            config=wandb_config,
            experiment_name=experiment_name,
            project_override=self.wandb_project,
            tags=[
                'ablation_study', 
                experiment_type, 
                config.get('objective', 'unknown'),
                f"N_{config.get('N', 'unknown')}",
                self.wandb_experiment_group
            ]
        )
    
    def log_ablation_metrics(self, metrics: dict, step: int = None):
        """Log metrics to WandB with ablation-specific prefixes"""
        if self.wandb_logger:
            # Add ablation-specific prefixes
            ablation_metrics = {
                f"ablation/{k}": v for k, v in metrics.items() if v is not None
            }
            self.wandb_logger.log_metrics(ablation_metrics, step=step)
    
    def run_publication_ablation(self):
        """Run comprehensive publication-level ablation study with WandB tracking"""
        self.start_time = time.time()
        self.log("=== Starting Publication-Level PAC-Bayes Ablation Study ===")
        
        # Get enabled experiments from the configuration
        enabled_experiments = getattr(self, 'enabled_experiments', ['ntuple_sizes'])

        # create file for all results
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_json_file = f"{self.results_dir}/all_ablation_results.json_{self.timestamp}"
        self.results_csv_file = f"{self.results_dir}/all_ablation_results.csv_{self.timestamp}"

        with open(self.results_json_file, 'w') as f:
            json.dump({}, f)  # Initialize empty JSON
        
        pd.DataFrame().to_csv(self.results_csv_file, index=False)  # Initialize empty CSV
        
        self.log(f"Enabled experiments: {enabled_experiments}")
        self.log(f"Using random seeds: {getattr(self, 'random_seeds', [42])}")
        self.log(f"WandB integration: {'Enabled' if self.use_wandb else 'Disabled'}")
        self.log(f"Results directory: {self.results_dir}")
        
        all_results = {}
        
        # Run only enabled experiments
        if 'ntuple_sizes' in enabled_experiments:
            try:
                self.log("=== Running N-tuple Size Analysis ===")
                ntuple_results = self.ntuple_size_ablation()
                all_results['ntuple_sizes'] = ntuple_results
                self.log(f"N-tuple experiment completed: {len([r for r in ntuple_results if r.get('status') == 'success'])}/{len(ntuple_results)} successful")
            except Exception as e:
                self.log(f"ERROR in N-tuple ablation: {e}")
                all_results['ntuple_sizes'] = []
        
        if 'training_objectives' in enabled_experiments:
            try:
                self.log("=== Running Enhanced Training Objectives Comparison ===")
                objectives_results = self.enhanced_training_objectives_ablation()
                all_results['objectives'] = objectives_results
                self.log(f"Objectives experiment completed: {len([r for r in objectives_results if r.get('status') == 'success'])}/{len(objectives_results)} successful")
            except Exception as e:
                self.log(f"ERROR in objectives ablation: {e}")
                all_results['objectives'] = []
        
        if 'hyperparameter_refinement' in enabled_experiments:
            try:
                self.log("=== Running Winning Hyperparameter Refinement ===")
                winning_results = self.winning_hyperparameter_refinement()
                all_results['winning_hyperparams'] = winning_results
                self.log(f"Winning hyperparams experiment completed: {len([r for r in winning_results if r.get('status') == 'success'])}/{len(winning_results)} successful")
            except Exception as e:
                self.log(f"ERROR in winning hyperparams ablation: {e}")
                all_results['winning_hyperparams'] = []
        
        if 'architecture_scaling' in enabled_experiments:
            try:
                self.log("=== Running Architecture Scaling Analysis ===")
                architecture_results = self.architecture_ablation()
                all_results['architectures'] = architecture_results
                self.log(f"Architecture experiment completed: {len([r for r in architecture_results if r.get('status') == 'success'])}/{len(architecture_results)} successful")
            except Exception as e:
                self.log(f"ERROR in architecture ablation: {e}")
                all_results['architectures'] = []
        
        if 'prior_analysis' in enabled_experiments:
            try:
                self.log("=== Running Enhanced Prior Analysis ===")
                prior_results = self.enhanced_prior_analysis()
                all_results['priors'] = prior_results
                self.log(f"Prior experiment completed: {len([r for r in prior_results if r.get('status') == 'success'])}/{len(prior_results)} successful")
            except Exception as e:
                self.log(f"ERROR in prior ablation: {e}")
                all_results['priors'] = []
        
        # Save results and generate comprehensive analysis
        df = self.save_publication_results(all_results)
        self.generate_statistical_analysis(all_results, df)
        self.generate_publication_summary(all_results, df)
        self.generate_wandb_summary(all_results, df)
        
        # 🆕 Enhanced WandB Tracking - Log analysis charts and tables matching your notebook
        if df is not None and self.use_wandb:
            self.log("=== Uploading Enhanced Analysis to WandB ===")
            
            # Initialize enhanced tracking for final analysis
            self.enhanced_tracker.init_experiment_tracking("final_analysis", {
                "study_type": "publication_ablation",
                "total_experiments": len(df),
                "enabled_experiments": enabled_experiments
            })
            
            # Log analysis by experiment type
            if 'ntuple_sizes' in enabled_experiments and 'experiment_type' in df.columns:
                df_ntuple = df[df['experiment_type'] == 'ntuple_sizes']
                if len(df_ntuple) > 0:
                    self.enhanced_tracker.log_ntuple_analysis(df_ntuple)
            
            if 'training_objectives' in enabled_experiments:
                df_objectives = df[df['experiment_type'] == 'objectives'] if 'experiment_type' in df.columns else df
                if len(df_objectives) > 0:
                    self.enhanced_tracker.log_objectives_analysis(df_objectives)
            
            if 'architecture_scaling' in enabled_experiments:
                df_arch = df[df['experiment_type'] == 'architectures'] if 'experiment_type' in df.columns else df
                if len(df_arch) > 0:
                    self.enhanced_tracker.log_architecture_analysis(df_arch)
            
            if 'hyperparameter_refinement' in enabled_experiments:
                df_hyperparams = df[df['experiment_type'] == 'winning_hyperparams'] if 'experiment_type' in df.columns else df
                if len(df_hyperparams) > 0:
                    self.enhanced_tracker.log_hyperparameter_analysis(df_hyperparams)
            
            if 'prior_analysis' in enabled_experiments:
                df_priors = df[df['experiment_type'] == 'priors'] if 'experiment_type' in df.columns else df
                if len(df_priors) > 0:
                    self.enhanced_tracker.log_prior_analysis(df_priors)
            
            # Log overall analysis 
            self.enhanced_tracker.log_overall_analysis(df)
            
            # Finalize enhanced tracking
            self.enhanced_tracker.finalize_run()
            
            self.log("✅ Enhanced WandB analysis charts and tables uploaded successfully!")
        
        total_time = time.time() - self.start_time
        self.log(f"=== Publication Ablation Study Complete ===")
        self.log(f"Total time: {total_time/3600:.2f} hours")
        self.log(f"Successful experiments: {self.successful_experiments}")
        self.log(f"Failed experiments: {self.failed_experiments}")
        
        return all_results
    
    def ntuple_size_ablation(self):
        """Test different N-tuple sizes (3-6) with WandB tracking"""
        ntuple_sizes = [3, 4, 5, 6]  # Your requested range
        results = []
        
        self.log(f"Testing N-tuple sizes: {ntuple_sizes}")
        
        for N in ntuple_sizes:
            self.log(f"Running N-tuple size analysis for N={N}")
            
            # Test with multiple seeds for statistical reliability
            for seed_idx, seed in enumerate(self.random_seeds):
                config = self.base_config.copy()
                config.update({
                    'N': N,
                    'experiment_name': f"ntuple_N{N}_seed{seed}",
                    'random_seed': seed,
                    # Use optimized hyperparameters based on your successful results
                    'sigma_prior': 0.01,  # From your winning experiments
                    'kl_penalty': 1e-6,   # From your winning experiments
                    'learning_rate': 0.005,
                    'train_epochs': 30,    # Increased for final results
                    'mc_samples': 1000,     # Increased for better estimates
                    'use_wandb': True,    # Disable individual experiment wandb logging
                })
                
                # Initialize WandB for this specific experiment
                self.init_wandb_for_experiment("ntuple_size", config)
                self.log(f"  N={N}, seed={seed} ({seed_idx+1}/{len(self.random_seeds)})")
                result = self.run_single_experiment(config)
                
                if result.get('status') == 'success':
                    result.update({'N': N, 'seed': seed, 'experiment_type': 'ntuple_size'})
                    self.log(f"✓ N={N} seed={seed}: Risk={result.get('risk_ntuple', 'N/A'):.4f}, Acc={result.get('stch_accuracy', 'N/A'):.4f}")
                    
                    # Log to WandB
                    self.log_ablation_metrics({
                        'N': N,
                        'seed': seed,
                        'risk_certificate': result.get('risk_ntuple'),
                        'stochastic_accuracy': result.get('stch_accuracy'),
                        'kl_per_n': result.get('kl_per_n'),
                        'training_objective': result.get('train_obj')
                    })
                else:
                    self.log(f"    ✗ N={N} seed={seed} failed: {result.get('error', 'Unknown')}")
                
                results.append(result)
                
                # Finish this WandB run
                if self.wandb_logger:
                    self.wandb_logger.finish()
                
                # save intermediate results
                self.save_publication_results({'ntuple_sizes': results})
        
        return results
    
    def enhanced_training_objectives_ablation(self):
        """Enhanced training objectives with statistical analysis and WandB tracking"""
        objectives = ['fquad', 'fclassic', 'ntuple', 'nested_ntuple', 'theory_ntuple']
        results = []
        
        for objective in objectives:
            # Run with multiple seeds for each objective
            for seed_idx, seed in enumerate(self.random_seeds):
                config = self.base_config.copy()
                config.update({
                    'objective': objective,
                    'experiment_name': f"obj_{objective}_seed{seed}",
                    'random_seed': seed,
                    # Use optimized settings
                    'sigma_prior': 0.01,
                    'kl_penalty': 1e-6,
                    'train_epochs': 30,
                    'mc_samples': 1000,
                    'use_wandb': False,
                })
                
                # Initialize WandB for this specific experiment
                self.init_wandb_for_experiment("training_objective", config)
                
                self.log(f"Running {objective} with seed {seed} ({seed_idx+1}/{len(self.random_seeds)})")
                result = self.run_single_experiment(config)
                
                if result.get('status') == 'success':
                    result.update({'objective': objective, 'seed': seed, 'experiment_type': 'training_objective'})
                    self.log(f"  ✓ {objective} seed={seed}: Risk={result.get('risk_ntuple', 'N/A'):.4f}, Acc={result.get('stch_accuracy', 'N/A'):.4f}")
                    
                    # Log to WandB
                    self.log_ablation_metrics({
                        'objective': objective,
                        'seed': seed,
                        'risk_certificate': result.get('risk_ntuple'),
                        'stochastic_accuracy': result.get('stch_accuracy'),
                        'kl_per_n': result.get('kl_per_n'),
                        'training_objective': result.get('train_obj')
                    })
                
                results.append(result)
                
                # Finish this WandB run
                if self.wandb_logger:
                    self.wandb_logger.finish()

                # save intermediate results
                self.save_publication_results({'objectives': results})
        
        return results
    
    def winning_hyperparameter_refinement(self):
        """Refine around your winning hyperparameters with WandB tracking"""
        # Based on your successful results, we'll explore around the winning combinations
        hyperparam_grid = {
            'sigma_prior': [0.005, 0.01, 0.02, 0.05],      # Around 0.01 (your winner)
            'learning_rate': [0.001, 0.005, 0.01],    # Around 0.005
            'kl_penalty': [5e-7, 1e-6, 5e-6],        # Around 1e-6
            'embedding_dim': [128, 256],                     # Test both
            'train_epochs': [30, 50],                       # Longer training
        }
        
        results = []
        # Generate grid combinations (sample systematically)
        n_combinations = 16  # Manageable number for publication
        
        self.log(f"Testing {n_combinations} refined hyperparameter combinations")
        
        # Create systematic grid sampling
        np.random.seed(42)  # Reproducible sampling
        combinations = []
        
        for i in range(n_combinations):
            combination = {
                key: np.random.choice(values) 
                for key, values in hyperparam_grid.items()
            }
            combinations.append(combination)
        
        for i, combo in enumerate(combinations):
            # Run each combination with multiple seeds
            for seed_idx, seed in enumerate(self.random_seeds[:2]):  # Use 2 seeds for grid search
                config = self.base_config.copy()
                config.update(combo)
                config.update({
                    'experiment_name': f"refined_{i}_seed{seed}",
                    'random_seed': seed,
                    'mc_samples': 1000,
                    'samples_ensemble': 100,
                    'use_wandb': False,
                })
                
                # Initialize WandB for this specific experiment
                self.init_wandb_for_experiment("hyperparameter_refinement", config)
                
                self.log(f"Refined combo {i+1}/{n_combinations}, seed {seed}: {combo}")
                result = self.run_single_experiment(config)
                
                if result.get('status') == 'success':
                    result.update(combo)
                    result.update({'combo_id': i, 'seed': seed, 'experiment_type': 'hyperparameter_refinement'})
                    self.log(f"  ✓ Combo {i} seed={seed}: Risk={result.get('risk_ntuple', 'N/A'):.4f}, Acc={result.get('stch_accuracy', 'N/A'):.4f}")
                    
                    # Log to WandB
                    self.log_ablation_metrics({
                        'combo_id': i,
                        'seed': seed,
                        'risk_certificate': result.get('risk_ntuple'),
                        'stochastic_accuracy': result.get('stch_accuracy'),
                        'kl_per_n': result.get('kl_per_n'),
                        **combo
                    })
                
                results.append(result)
                
                # Finish this WandB run
                if self.wandb_logger:
                    self.wandb_logger.finish()
                
                # save intermediate results
                self.save_publication_results({'winning_hyperparams': results})
        
        return results
    
    def architecture_ablation(self):
        """Architecture analysis with statistical significance and WandB tracking"""
        architectures = [
            {'layers': 4, 'embedding_dim': 128},
            {'layers': 9, 'embedding_dim': 256},
            {'layers': 13, 'embedding_dim': 256},
        ]
        results = []
        
        for arch in architectures:
            for seed_idx, seed in enumerate(self.random_seeds[:2]):  # 2 seeds per architecture
                config = self.base_config.copy()
                config.update(arch)
                config.update({
                    'experiment_name': f"arch_{arch['layers']}l_{arch['embedding_dim']}d_seed{seed}",
                    'random_seed': seed,
                    'sigma_prior': 0.01,
                    'kl_penalty': 1e-6,
                    'train_epochs': 30,
                    'mc_samples': 1000,
                    'use_wandb': False,
                })
                
                # Initialize WandB for this specific experiment
                self.init_wandb_for_experiment("architecture", config)
                
                self.log(f"Architecture {arch['layers']}L-{arch['embedding_dim']}D, seed {seed}")
                result = self.run_single_experiment(config)
                
                if result.get('status') == 'success':
                    result.update(arch)
                    result.update({'seed': seed, 'experiment_type': 'architecture'})
                    
                    # Log to WandB
                    self.log_ablation_metrics({
                        'layers': arch['layers'],
                        'embedding_dim': arch['embedding_dim'],
                        'seed': seed,
                        'risk_certificate': result.get('risk_ntuple'),
                        'stochastic_accuracy': result.get('stch_accuracy'),
                        'kl_per_n': result.get('kl_per_n'),
                    })
                
                results.append(result)
                
                # Finish this WandB run
                if self.wandb_logger:
                    self.wandb_logger.finish()

                # save intermediate results
                self.save_publication_results({'architectures': results})
        
        return results
    
    def enhanced_prior_analysis(self):
        """Enhanced prior analysis with WandB tracking"""
        prior_configs = [
            {'perc_prior': 0.0, 'prior_type': 'random'},
            {'perc_prior': 0.1, 'prior_type': 'learned'},  
            {'perc_prior': 0.2, 'prior_type': 'learned'},
            {'perc_prior': 0.5, 'prior_type': 'learned'},
            {'perc_prior': 0.7, 'prior_type': 'learned'},
        ]
        results = []
        
        for prior_config in prior_configs:
            for seed_idx, seed in enumerate(self.random_seeds[:2]):
                config = self.base_config.copy()
                config.update(prior_config)
                config.update({
                    'experiment_name': f"prior_{prior_config['prior_type']}_{prior_config['perc_prior']}_seed{seed}",
                    'random_seed': seed,
                    'sigma_prior': 0.01,
                    'kl_penalty': 1e-6,
                    'train_epochs': 40,
                    'use_wandb': False,
                })
                
                # Initialize WandB for this specific experiment
                self.init_wandb_for_experiment("prior_analysis", config)
                
                self.log(f"Prior {prior_config['prior_type']} {prior_config['perc_prior']}, seed {seed}")
                result = self.run_single_experiment(config)
                
                if result.get('status') == 'success':
                    result.update(prior_config)
                    result.update({'seed': seed, 'experiment_type': 'prior_analysis'})
                    
                    # Log to WandB
                    self.log_ablation_metrics({
                        'perc_prior': prior_config['perc_prior'],
                        'prior_type': prior_config['prior_type'],
                        'seed': seed,
                        'risk_certificate': result.get('risk_ntuple'),
                        'stochastic_accuracy': result.get('stch_accuracy'),
                        'kl_per_n': result.get('kl_per_n'),
                    })
                
                results.append(result)
                
                # Finish this WandB run
                if self.wandb_logger:
                    self.wandb_logger.finish()

                # save intermediate results
                self.save_publication_results({'priors': results})
        
        return results
    
    def run_single_experiment(self, config):
        """Enhanced single experiment with better error handling"""
        experiment_start_time = time.time()
        
        try:
            # Validate configuration
            if not validate_experiment_config(config):
                return {
                    'status': 'failed',
                    'error': 'Configuration validation failed',
                    'experiment_name': config.get('experiment_name', 'unknown'),
                    'experiment_time': 0,
                    'full_config': config
                }
            
            # Filter config to only supported parameters
            filtered_config = {
                k: v for k, v in config.items() 
                if k in self.supported_params
            }
            
            # Ensure consistent settings for ablation
            filtered_config.update({
                'verbose': False,
                'verbose_test': False,
                'run_baseline': False,
                'debug_mode': False,
            })
            
            # Run experiment
            result = runexp_with_structured_output(**filtered_config)
            
            # Add metadata
            result['experiment_time'] = time.time() - experiment_start_time
            result['full_config'] = config
            
            if result.get('status') == 'success':
                self.successful_experiments += 1
            else:
                self.failed_experiments += 1
            
            return result
            
        except Exception as e:
            self.failed_experiments += 1
            self.log(f"Experiment failed: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'experiment_name': config.get('experiment_name', 'unknown'),
                'experiment_time': time.time() - experiment_start_time,
                'full_config': config
            }
    
    def save_publication_results(self, results):
        """Save results in publication-ready format"""        
        # Save comprehensive JSON
        results_file = getattr(self, 'results_json_file', None)
        csv_file = getattr(self, 'results_csv_file', None)

        if not results_file or not csv_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{self.results_dir}/all_ablation_results.json_{timestamp}"
            csv_file = f"{self.results_dir}/all_ablation_results.csv_{timestamp}"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create comprehensive CSV with all metrics
        df_list = []
        for experiment_type, exp_results in results.items():
            for result in exp_results:
                if result.get('status') == 'success':
                    row = {
                        'experiment_type': experiment_type,
                        'experiment_name': result.get('experiment_name', 'unknown'),
                        'seed': result.get('seed', 'unknown'),
                        
                        # Core metrics
                        'risk_ntuple': result.get('risk_ntuple', None),
                        'stch_accuracy': result.get('stch_accuracy', None),
                        'post_accuracy': result.get('post_accuracy', None),
                        'ens_accuracy': result.get('ens_accuracy', None),
                        'kl_per_n': result.get('kl_per_n', None),
                        'train_obj': result.get('train_obj', None),
                        'experiment_time': result.get('experiment_time', None),
                        
                        # Configuration
                        'N': result.get('N', result.get('config', {}).get('N', None)),
                        'objective': result.get('objective', result.get('config', {}).get('objective', None)),
                        'sigma_prior': result.get('sigma_prior', result.get('config', {}).get('sigma_prior', None)),
                        'learning_rate': result.get('learning_rate', result.get('config', {}).get('learning_rate', None)),
                        'kl_penalty': result.get('kl_penalty', result.get('config', {}).get('kl_penalty', None)),
                        'layers': result.get('layers', result.get('config', {}).get('layers', None)),
                        'embedding_dim': result.get('embedding_dim', result.get('config', {}).get('embedding_dim', None)),
                        'perc_prior': result.get('perc_prior', result.get('config', {}).get('perc_prior', None)),
                        
                        # Quality indicators
                        'is_vacuous': result.get('is_vacuous', None),
                        'kl_reasonable': result.get('kl_reasonable', None),
                        'good_accuracy': result.get('good_accuracy', None),
                    }
                    df_list.append(row)
        
        if df_list:
            df = pd.DataFrame(df_list)
            csv_file = f"{self.results_dir}/data/publication_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            self.log(f"Publication results saved to {csv_file}")
            
            return df
        else:
            self.log("No successful experiments to save")
            return None
    
    def generate_statistical_analysis(self, all_results, df):
        """Generate statistical analysis for publication"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"{self.results_dir}/data/statistical_analysis_{timestamp}.txt"
        
        if df is None:
            return
        
        with open(analysis_file, 'w') as f:
            f.write("=== PUBLICATION-LEVEL STATISTICAL ANALYSIS ===\n\n")
            
            # Overall statistics
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Experiment types: {df['experiment_type'].unique()}\n\n")
            
            # N-tuple size analysis
            if 'ntuple_sizes' in all_results:
                f.write("=== N-TUPLE SIZE ANALYSIS ===\n")
                ntuple_df = df[df['experiment_type'] == 'ntuple_size']
                if len(ntuple_df) > 0:
                    # Group by N and compute statistics
                    ntuple_stats = ntuple_df.groupby('N').agg({
                        'stch_accuracy': ['mean', 'std', 'count'],
                        'risk_ntuple': ['mean', 'std'],
                        'kl_per_n': ['mean', 'std']
                    }).round(4)
                    
                    f.write("N-tuple Size Performance:\n")
                    f.write(str(ntuple_stats))
                    f.write("\n\n")
                    
                    # Statistical significance test
                    unique_N = ntuple_df['N'].unique()
                    if len(unique_N) > 1:
                        f.write("Statistical Significance Tests (ANOVA):\n")
                        try:
                            # ANOVA for accuracy
                            groups_acc = [ntuple_df[ntuple_df['N'] == n]['stch_accuracy'].values for n in unique_N]
                            f_stat_acc, p_val_acc = stats.f_oneway(*groups_acc)
                            f.write(f"Accuracy ANOVA: F={f_stat_acc:.4f}, p={p_val_acc:.4f}\n")
                            
                            # ANOVA for risk certificates
                            groups_risk = [ntuple_df[ntuple_df['N'] == n]['risk_ntuple'].values for n in unique_N]
                            f_stat_risk, p_val_risk = stats.f_oneway(*groups_risk)
                            f.write(f"Risk Certificate ANOVA: F={f_stat_risk:.4f}, p={p_val_risk:.4f}\n")
                        except:
                            f.write("Could not compute ANOVA (insufficient data)\n")
                    f.write("\n")
            
            # Best configurations
            f.write("=== BEST CONFIGURATIONS ===\n")
            
            # Best accuracy
            best_acc_idx = df['stch_accuracy'].idxmax()
            if not pd.isna(best_acc_idx):
                best_acc_row = df.loc[best_acc_idx]
                f.write(f"Best Accuracy: {best_acc_row['stch_accuracy']:.4f}\n")
                f.write(f"  Experiment: {best_acc_row['experiment_name']}\n")
                f.write(f"  Risk Certificate: {best_acc_row['risk_ntuple']:.4f}\n")
                f.write(f"  Configuration: N={best_acc_row['N']}, σ={best_acc_row['sigma_prior']}, lr={best_acc_row['learning_rate']}\n\n")
        
        self.log(f"Statistical analysis saved to {analysis_file}")
    
    def generate_publication_summary(self, all_results, df):
        """Generate publication-ready summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.results_dir}/data/publication_summary_{timestamp}.tex"
        
        if df is None:
            return
        
        with open(summary_file, 'w') as f:
            f.write("% Publication-Ready Summary - LaTeX Format\n")
            f.write("% Generated by PAC-Bayes N-tuple Ablation Study\n\n")
            
            # Key findings
            f.write("\\section{Key Experimental Findings}\n\n")
            
            # Best results
            if not df.empty:
                best_acc = df['stch_accuracy'].max()
                best_risk = df['risk_ntuple'].min()
                mean_acc = df['stch_accuracy'].mean()
                mean_risk = df['risk_ntuple'].mean()
                
                f.write(f"\\subsection{{Performance Summary}}\n")
                f.write(f"\\begin{{itemize}}\n")
                f.write(f"\\item Best stochastic accuracy achieved: {best_acc:.3f}\n")
                f.write(f"\\item Tightest risk certificate: {best_risk:.3f}\n")
                f.write(f"\\item Mean accuracy across experiments: {mean_acc:.3f} $\\pm$ {df['stch_accuracy'].std():.3f}\n")
                f.write(f"\\item Non-vacuous bounds achieved in {(df['risk_ntuple'] < 1.0).sum()}/{len(df)} experiments\n")
                f.write(f"\\end{{itemize}}\n\n")
        
        self.log(f"Publication summary saved to {summary_file}")
    
    def generate_wandb_summary(self, all_results, df):
        """Generate comprehensive WandB summary for the entire ablation study"""
        if not self.use_wandb or self.wandb_logger is None or df is None:
            return
        
        # Initialize a final summary run
        summary_config = {
            'ablation_study_summary': True,
            'experiment_group': self.wandb_experiment_group,
            'total_experiments': len(df),
            'successful_experiments': self.successful_experiments,
            'failed_experiments': self.failed_experiments,
            'study_duration_hours': (time.time() - self.start_time) / 3600,
        }
        
        self.wandb_logger.init_experiment(
            config=summary_config,
            experiment_name=f"ablation_summary_{self.wandb_experiment_group}",
            project_override=self.wandb_project,
            tags=['ablation_summary', self.wandb_experiment_group]
        )
        
        # Log summary statistics
        if not df.empty:
            summary_metrics = {
                'summary/best_accuracy': df['stch_accuracy'].max(),
                'summary/best_risk_certificate': df['risk_ntuple'].min(),
                'summary/mean_accuracy': df['stch_accuracy'].mean(),
                'summary/std_accuracy': df['stch_accuracy'].std(),
                'summary/mean_risk_certificate': df['risk_ntuple'].mean(),
                'summary/std_risk_certificate': df['risk_ntuple'].std(),
                'summary/non_vacuous_bounds_pct': (df['risk_ntuple'] < 1.0).mean() * 100,
                'summary/good_accuracy_pct': (df['stch_accuracy'] > 0.6).mean() * 100,
                'summary/total_experiments': len(df),
                'summary/successful_experiments': self.successful_experiments,
                'summary/failed_experiments': self.failed_experiments,
            }
            
            # Log per experiment type statistics
            for exp_type in df['experiment_type'].unique():
                exp_df = df[df['experiment_type'] == exp_type]
                summary_metrics.update({
                    f'summary/{exp_type}/count': len(exp_df),
                    f'summary/{exp_type}/best_accuracy': exp_df['stch_accuracy'].max(),
                    f'summary/{exp_type}/mean_accuracy': exp_df['stch_accuracy'].mean(),
                    f'summary/{exp_type}/best_risk': exp_df['risk_ntuple'].min(),
                    f'summary/{exp_type}/mean_risk': exp_df['risk_ntuple'].mean(),
                })
            
            self.log_ablation_metrics(summary_metrics)
        
        # Create and upload summary plots
        self.create_and_upload_summary_plots(df)
        
        self.wandb_logger.finish()
        self.log("WandB summary completed and uploaded")
    
    def create_and_upload_summary_plots(self, df):
        """Create and upload summary plots to WandB"""
        if df is None or df.empty:
            return
        
        try:
            # Plot 1: N-tuple size comparison
            if 'N' in df.columns and df['N'].notna().any():
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x='N', y='stch_accuracy')
                plt.title('Stochastic Accuracy by N-tuple Size')
                plt.ylabel('Stochastic Accuracy')
                plt.xlabel('N-tuple Size')
                plot_path = f"{self.results_dir}/plots/ntuple_accuracy_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({"plots/ntuple_accuracy_comparison": plot_path})
                plt.close()
            
            # Plot 2: Risk certificate vs accuracy scatter
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(df['risk_ntuple'], df['stch_accuracy'], 
                                 c=df['experiment_type'].astype('category').cat.codes, 
                                 alpha=0.6, cmap='tab10')
            plt.xlabel('Risk Certificate')
            plt.ylabel('Stochastic Accuracy')
            plt.title('Risk Certificate vs Stochastic Accuracy')
            plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Vacuous Bound')
            plt.legend()
            plot_path = f"{self.results_dir}/plots/risk_vs_accuracy.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if self.wandb_logger:
                self.wandb_logger.log_metrics({"plots/risk_vs_accuracy": plot_path})
            plt.close()
            
        except Exception as e:
            self.log(f"Error creating plots: {e}")

# Enhanced configuration based on your successful experiments
def get_publication_base_config():
    """Get the base configuration for publication-level experiments"""
    return {
        'name_data': 'cifar10',
        'model': 'cnn',
        'N': 3,  # Will be varied in N-tuple experiments
        'sigma_prior': 0.01,        # From your winning experiments  
        'pmin': 1e-4,
        'learning_rate': 0.005,     # From your winning experiments
        'momentum': 0.9,
        'learning_rate_prior': 0.01,
        'momentum_prior': 0.9,
        'delta': 0.025,
        'layers': 4,
        'delta_test': 0.01,
        'mc_samples': 1000,          # Increased for publication quality
        'samples_ensemble': 100,     # Increased for publication quality
        'kl_penalty': 1e-6,         # From your winning experiments
        'train_epochs': 30,         # Increased for final results
        'prior_dist': 'gaussian',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'prior_epochs': 30,
        'dropout_prob': 0.2,
        'perc_train': 1.0,
        'perc_prior': 0.3,
        'batch_size': 32,
        'embedding_dim': 128,       # Your successful configuration
        'objective': 'theory_ntuple',      # Your best objective
    }

# Run the publication-level study
if __name__ == "__main__":
    print("Starting Enhanced Publication-Level PAC-Bayes Ablation Study...")
    print("This includes N-tuple size experiments (N=3,4,5,6) and comprehensive WandB tracking")
    
    # Load configuration
    publication_base_config = get_publication_base_config()
    
    # Initialize ablation study with WandB
    ablation = PACBayesAblation(
        publication_base_config, 
        results_dir="publication_ablation_results",
        use_wandb=True,
        wandb_project="pac-bayes-ablation-study"
    )
    
    # Run the comprehensive study
    results = ablation.run_publication_ablation()
    
    print(f"\nPublication-level ablation study completed!")
    print(f"Results directory: {ablation.results_dir}")
    print(f"Successful experiments: {ablation.successful_experiments}")
    print(f"Failed experiments: {ablation.failed_experiments}")
    print(f"Key files generated:")
    print(f"  - data/publication_results_*.csv (main results)")
    print(f"  - data/statistical_analysis_*.txt (statistical tests)")
    print(f"  - data/publication_summary_*.tex (LaTeX summary)")
    print(f"  - logs/publication_log_*.txt (detailed logs)")
    print(f"  - plots/*.png (summary plots)")
    if ablation.use_wandb:
        print(f"  - WandB project: {ablation.wandb_project} (online dashboard)")
