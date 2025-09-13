# Unified Weights & Biases Integration for PAC-Bayes ReID Experiments
# Combines basic logging with enhanced analysis visualizations
import os
from typing import Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. Install with: pip install wandb")

def load_env_config() -> Dict[str, Optional[str]]:
    """Load configuration from .env file or environment variables."""
    config = {
        'api_key': os.getenv('WANDB_API_KEY'),
        'project': os.getenv('WANDB_PROJECT', 'pac-bayes-reid-experiments'),
        'entity': os.getenv('WANDB_ENTITY')
    }
    
    # Try to load from .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key == 'WANDB_API_KEY':
                            config['api_key'] = value
                        elif key == 'WANDB_PROJECT':
                            config['project'] = value
                        elif key == 'WANDB_ENTITY':
                            config['entity'] = value
        except Exception as e:
            warnings.warn(f"Could not read .env file: {e}")
    
    return config

class WandbLogger:
    """Weights & Biases logger for PAC-Bayes experiments."""
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.initialized = False
        self.run = None
        
        if use_wandb and not WANDB_AVAILABLE:
            warnings.warn("wandb requested but not available. Continuing without logging.")
            self.use_wandb = False
    
    def init_experiment(self, config: Dict[str, Any], experiment_name: Optional[str] = None, 
                       project_override: Optional[str] = None, tags: Optional[list] = None) -> bool:
        """Initialize a new wandb experiment."""
        if not self.use_wandb:
            return False
        
        try:
            # Load environment configuration
            env_config = load_env_config()
            
            # Set API key if available
            if env_config['api_key']:
                os.environ['WANDB_API_KEY'] = env_config['api_key']
            elif not os.getenv('WANDB_API_KEY'):
                warnings.warn("No WANDB_API_KEY found. Please set it in .env file or environment.")
                self.use_wandb = False
                return False
            
            # Generate experiment name if not provided
            if experiment_name is None:
                experiment_name = self._generate_experiment_name(config)
            
            # Initialize wandb
            self.run = wandb.init(
                project=project_override or env_config['project'],
                entity=env_config['entity'],
                name=experiment_name,
                config=config,
                tags=tags or self._generate_tags(config),
                reinit=True
            )
            
            self.initialized = True
            print("✓ Weights & Biases initialized successfully")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
            return False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to wandb."""
        if not self.use_wandb or not self.initialized:
            return
        
        try:
            # Filter out None values and convert numpy types
            clean_metrics = {}
            for key, value in metrics.items():
                if value is not None:
                    # Convert numpy types to Python types
                    if hasattr(value, 'item'):
                        clean_metrics[key] = value.item()
                    else:
                        clean_metrics[key] = value
            
            wandb.log(clean_metrics, step=step, commit=commit)
        except Exception as e:
            warnings.warn(f"Failed to log metrics to wandb: {e}")
    
    def log_baseline_training(self, epoch: int, loss: float, accuracy: float, 
                            test_error: Optional[float] = None, test_acc: Optional[float] = None,
                            map_score: Optional[float] = None, rank1_score: Optional[float] = None):
        """Log deterministic baseline training metrics."""
        metrics = {
            "baseline_train/epoch": epoch,
            "baseline_train/loss": loss,
            "baseline_train/accuracy": accuracy
        }
        
        if test_error is not None:
            metrics["baseline_train/test_error"] = test_error
        if test_acc is not None:
            metrics["baseline_train/test_accuracy"] = test_acc
        if map_score is not None:
            metrics["baseline_train/test_map"] = map_score
        if rank1_score is not None:
            metrics["baseline_train/test_rank1"] = rank1_score
            
        self.log_metrics(metrics)
    
    def log_pacbayes_training(self, epoch: int, bound: float, kl_per_n: float, 
                            empirical_loss: float, accuracy: float):
        """Log PAC-Bayes training metrics."""
        metrics = {
            "pacbayes_train/epoch": epoch,
            "pacbayes_train/bound": bound,
            "pacbayes_train/kl_per_n": kl_per_n,
            "pacbayes_train/empirical_loss": empirical_loss,
            "pacbayes_train/accuracy": accuracy
        }
        self.log_metrics(metrics)
    
    def log_checkpoint_evaluation(self, epoch: int, train_obj: float, risk_cert: float,
                                empirical_risk: float, pseudo_acc: float, kl_per_n: float,
                                stch_risk: float, stch_acc: float, stch_map: Optional[float] = None,
                                stch_rank1: Optional[float] = None):
        """Log checkpoint evaluation metrics."""
        metrics = {
            "checkpoint/epoch": epoch,
            "checkpoint/training_objective": train_obj,
            "checkpoint/risk_certificate": risk_cert,
            "checkpoint/empirical_risk": empirical_risk,
            "checkpoint/pseudo_accuracy": pseudo_acc,
            "checkpoint/kl_per_n": kl_per_n,
            "checkpoint/stochastic_risk": stch_risk,
            "checkpoint/stochastic_accuracy": stch_acc
        }
        
        if stch_map is not None:
            metrics["checkpoint/stochastic_map"] = stch_map
        if stch_rank1 is not None:
            metrics["checkpoint/stochastic_rank1"] = stch_rank1
            
        self.log_metrics(metrics)
    
    def log_final_results(self, results: Dict[str, Any]):
        """Log final experiment results."""
        if not self.use_wandb or not self.initialized:
            return
        
        # Core PAC-Bayes metrics
        final_metrics = {
            "final/training_objective": results.get('train_obj'),
            "final/risk_certificate": results.get('risk_ntuple'),
            "final/empirical_risk": results.get('empirical_risk_ntuple'),
            "final/kl_per_n": results.get('kl_per_n'),
            "final/pseudo_accuracy": results.get('pseudo_accuracy'),
        }
        
        # Evaluation method results
        eval_methods = ['stch', 'post', 'ens']
        eval_names = ['stochastic', 'posterior', 'ensemble']
        
        for method, name in zip(eval_methods, eval_names):
            final_metrics.update({
                f"final/{name}_risk": results.get(f'{method}_risk'),
                f"final/{name}_accuracy": results.get(f'{method}_accuracy'),
                f"final/{name}_map": results.get(f'{method}_map'),
                f"final/{name}_rank1": results.get(f'{method}_rank1')
            })
        
        # Baseline and prior results
        final_metrics.update({
            "final/baseline_error": results.get('baseline_error'),
            "final/prior_map": results.get('prior_map'),
            "final/prior_rank1": results.get('prior_rank1')
        })
        
        # Bound quality assessment
        risk_cert = results.get('risk_ntuple', float('inf'))
        if risk_cert < 0.15:
            final_metrics["final/bound_quality"] = "excellent"
        elif risk_cert < 0.5:
            final_metrics["final/bound_quality"] = "good"
        elif risk_cert < 1.0:
            final_metrics["final/bound_quality"] = "ok"
        else:
            final_metrics["final/bound_quality"] = "vacuous"
        
        self.log_metrics(final_metrics)
    
    def log_hyperparameter_summary(self, config: Dict[str, Any]):
        """Log a summary table of hyperparameters."""
        if not self.use_wandb or not self.initialized:
            return
        
        try:
            # Create a summary table
            summary_data = []
            for key, value in config.items():
                summary_data.append([key, str(value)])
            
            table = wandb.Table(data=summary_data, columns=["Parameter", "Value"])
            wandb.log({"hyperparameters": table})
        except Exception as e:
            warnings.warn(f"Failed to log hyperparameter summary: {e}")
    
    def finish(self):
        """Finish the wandb run."""
        if self.use_wandb and self.initialized:
            try:
                wandb.finish()
                print("✓ Weights & Biases run completed")
            except Exception as e:
                warnings.warn(f"Error finishing wandb run: {e}")
            finally:
                self.initialized = False
                self.run = None
    
    def _generate_experiment_name(self, config: Dict[str, Any]) -> str:
        """Generate a descriptive experiment name."""
        objective = config.get('objective', 'unknown')
        dataset = config.get('dataset', 'unknown')
        N = config.get('N', 0)
        sigma = config.get('sigma_prior', 0)
        lr = config.get('learning_rate', 0)
        
        return f"{objective}_{dataset}_N{N}_σ{sigma:.3f}_lr{lr:.4f}"
    
    def _generate_tags(self, config: Dict[str, Any]) -> list:
        """Generate tags for the experiment."""
        tags = []
        
        if 'objective' in config:
            tags.append(config['objective'])
        if 'dataset' in config:
            tags.append(config['dataset'])
        if 'model' in config:
            tags.append(config['model'])
        if 'layers' in config:
            tags.append(f"layers_{config['layers']}")
        
        return tags
    
    @property
    def is_active(self) -> bool:
        """Check if wandb logging is active."""
        return self.use_wandb and self.initialized
    
    # Enhanced Analysis Methods (from enhanced_wandb_tracker.py)
    
    def init_experiment_tracking(self, experiment_type, config, run_name=None):
        """Initialize enhanced tracking (alias for init_experiment for compatibility)"""
        return self.init_experiment(config, run_name, tags=[experiment_type, "pac-bayes", "ablation"])
    
    def finalize_run(self):
        """Finalize the current run (alias for finish for compatibility)"""
        self.finish()
        
    def log_ntuple_analysis(self, df_ntuple, study_name=None):
        """
        Log N-tuple size analysis with charts matching your notebook
        Replicates: N-tuple Size Analysis (Novel Contribution) section
        """
        if not self.use_wandb or not self.initialized:
            print("📊 N-tuple Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging N-tuple Analysis to WandB...")
        
        # 1. Summary Statistics Table
        summary_stats = df_ntuple.groupby('N').agg({
            'stochastic_accuracy': ['mean', 'std', 'count'],
            'risk_certificate': ['mean', 'std'],
            'kl_per_n': ['mean']
        }).round(4)
        
        # Log summary table
        wandb.log({
            "ntuple_summary_table": wandb.Table(
                columns=["N", "Accuracy_Mean", "Accuracy_Std", "Risk_Mean", "Risk_Std", "KL_per_n", "Count"],
                data=[[int(n), 
                      summary_stats.loc[n, ('stochastic_accuracy', 'mean')],
                      summary_stats.loc[n, ('stochastic_accuracy', 'std')],
                      summary_stats.loc[n, ('risk_certificate', 'mean')],
                      summary_stats.loc[n, ('risk_certificate', 'std')],
                      summary_stats.loc[n, ('kl_per_n', 'mean')],
                      int(summary_stats.loc[n, ('stochastic_accuracy', 'count')])]
                     for n in sorted(df_ntuple['N'].unique())]
            )
        })
        
        # 2. Create 4-panel visualization (matching your notebook)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel 1: Stochastic Accuracy by N-tuple size
        sns.boxplot(data=df_ntuple, x='N', y='stochastic_accuracy', ax=ax1)
        ax1.set_title('Stochastic Accuracy vs N-tuple Size')
        ax1.set_ylabel('Stochastic Accuracy')
        
        # Panel 2: Risk Certificate by N-tuple size  
        sns.boxplot(data=df_ntuple, x='N', y='risk_certificate', ax=ax2)
        ax2.set_title('Risk Certificate vs N-tuple Size')
        ax2.set_ylabel('Risk Certificate')
        
        # Panel 3: KL per N by N-tuple size
        sns.boxplot(data=df_ntuple, x='N', y='kl_per_n', ax=ax3)
        ax3.set_title('KL Divergence per N vs N-tuple Size')
        ax3.set_ylabel('KL per N')
        
        # Panel 4: Mean performance summary
        means = df_ntuple.groupby('N')['stochastic_accuracy'].mean()
        stds = df_ntuple.groupby('N')['stochastic_accuracy'].std()
        ax4.errorbar(means.index, means.values, yerr=stds.values, marker='o', capsize=5)
        ax4.set_title('Mean Accuracy with Error Bars')
        ax4.set_xlabel('N-tuple Size')
        ax4.set_ylabel('Mean Stochastic Accuracy')
        
        plt.tight_layout()
        wandb.log({"ntuple_4panel_analysis": wandb.Image(plt)})
        plt.close()
        
        # 3. Statistical Significance Tests (ANOVA)
        n_values = sorted(df_ntuple['N'].unique())
        accuracy_groups = [df_ntuple[df_ntuple['N'] == n]['stochastic_accuracy'].values 
                          for n in n_values]
        
        f_stat, p_value = stats.f_oneway(*accuracy_groups)
        wandb.log({
            "ntuple_anova_f_statistic": f_stat,
            "ntuple_anova_p_value": p_value,
            "ntuple_significant": p_value < 0.05
        })
        
        # 4. Individual N-tuple metrics
        for n in n_values:
            n_data = df_ntuple[df_ntuple['N'] == n]
            wandb.log({
                f"ntuple_{n}_accuracy_mean": n_data['stochastic_accuracy'].mean(),
                f"ntuple_{n}_accuracy_std": n_data['stochastic_accuracy'].std(),
                f"ntuple_{n}_risk_mean": n_data['risk_certificate'].mean(),
                f"ntuple_{n}_kl_per_n": n_data['kl_per_n'].mean(),
                f"ntuple_{n}_count": len(n_data)
            })
            
    def log_objectives_analysis(self, df_objectives, study_name=None):
        """
        Log objectives comparison analysis with statistical tests
        Replicates: Objectives Comparison section
        """
        if not self.use_wandb or not self.initialized:
            print("📊 Objectives Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging Objectives Analysis to WandB...")
        
        objectives = sorted(df_objectives['objective'].unique())
        
        # 1. Objectives comparison boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_objectives, x='objective', y='stochastic_accuracy')
        plt.title('Stochastic Accuracy by Objective Function')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"objectives_accuracy_comparison": wandb.Image(plt)})
        plt.close()
        
        # 2. Risk certificate comparison
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_objectives, x='objective', y='risk_certificate')
        plt.title('Risk Certificate by Objective Function')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"objectives_risk_comparison": wandb.Image(plt)})
        plt.close()
        
        # 3. Statistical tests between objectives
        for i, obj1 in enumerate(objectives):
            for obj2 in objectives[i+1:]:
                data1 = df_objectives[df_objectives['objective'] == obj1]['stochastic_accuracy']
                data2 = df_objectives[df_objectives['objective'] == obj2]['stochastic_accuracy']
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                wandb.log({
                    f"objectives_{obj1}_vs_{obj2}_t_stat": t_stat,
                    f"objectives_{obj1}_vs_{obj2}_p_value": p_value,
                    f"objectives_{obj1}_vs_{obj2}_significant": p_value < 0.05
                })
        
        # 4. Summary statistics table
        obj_summary = df_objectives.groupby('objective').agg({
            'stochastic_accuracy': ['mean', 'std', 'count'],
            'risk_certificate': ['mean', 'std'],
            'status': lambda x: (x == 'success').mean()
        }).round(4)
        
        wandb.log({
            "objectives_summary_table": wandb.Table(
                columns=["Objective", "Accuracy_Mean", "Accuracy_Std", "Risk_Mean", "Success_Rate", "Count"],
                data=[[obj,
                      obj_summary.loc[obj, ('stochastic_accuracy', 'mean')],
                      obj_summary.loc[obj, ('stochastic_accuracy', 'std')],
                      obj_summary.loc[obj, ('risk_certificate', 'mean')],
                      obj_summary.loc[obj, ('status', '<lambda>')],
                      int(obj_summary.loc[obj, ('stochastic_accuracy', 'count')])]
                     for obj in objectives]
            )
        })
        
        # Individual objective metrics
        for obj in objectives:
            obj_data = df_objectives[df_objectives['objective'] == obj]
            wandb.log({
                f"objective_{obj}_accuracy_mean": obj_data['stochastic_accuracy'].mean(),
                f"objective_{obj}_accuracy_std": obj_data['stochastic_accuracy'].std(),
                f"objective_{obj}_risk_mean": obj_data['risk_certificate'].mean(),
                f"objective_{obj}_success_rate": (obj_data['status'] == 'success').mean()
            })
    
    def log_architecture_analysis(self, df_arch, study_name=None):
        """
        Log architecture scaling analysis
        Replicates: Architecture Scaling section
        """
        if not self.use_wandb or not self.initialized:
            print("📊 Architecture Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging Architecture Analysis to WandB...")
        
        # Extract architecture info from config or create synthetic analysis
        if 'architecture' in df_arch.columns:
            arch_col = 'architecture'
        elif 'layers' in df_arch.columns:
            arch_col = 'layers'
        else:
            # Create architecture labels from available data
            df_arch['architecture'] = df_arch.apply(lambda x: f"{x.get('layers', 'Unknown')}L-{x.get('units', 'Unknown')}D", axis=1)
            arch_col = 'architecture'
        
        architectures = sorted(df_arch[arch_col].unique())
        
        # 1. Performance by architecture
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_arch, x=arch_col, y='stochastic_accuracy')
        plt.title('Performance by Architecture')
        plt.xticks(rotation=45)
        plt.tight_layout()
        wandb.log({"architecture_performance_comparison": wandb.Image(plt)})
        plt.close()
        
        # 2. Architecture scaling trends
        arch_summary = df_arch.groupby(arch_col).agg({
            'stochastic_accuracy': ['mean', 'std'],
            'risk_certificate': ['mean'],
            'status': lambda x: (x == 'success').mean()
        }).round(4)
        
        wandb.log({
            "architecture_summary_table": wandb.Table(
                columns=["Architecture", "Accuracy_Mean", "Accuracy_Std", "Risk_Mean", "Success_Rate"],
                data=[[arch,
                      arch_summary.loc[arch, ('stochastic_accuracy', 'mean')],
                      arch_summary.loc[arch, ('stochastic_accuracy', 'std')],
                      arch_summary.loc[arch, ('risk_certificate', 'mean')],
                      arch_summary.loc[arch, ('status', '<lambda>')]]
                     for arch in architectures]
            )
        })
        
        # Individual architecture metrics
        for arch in architectures:
            arch_data = df_arch[df_arch[arch_col] == arch]
            wandb.log({
                f"arch_{arch}_accuracy_mean": arch_data['stochastic_accuracy'].mean(),
                f"arch_{arch}_accuracy_std": arch_data['stochastic_accuracy'].std(),
                f"arch_{arch}_risk_mean": arch_data['risk_certificate'].mean(),
                f"arch_{arch}_success_rate": (arch_data['status'] == 'success').mean()
            })
    
    def log_hyperparameter_analysis(self, df_hyperparams, study_name=None):
        """
        Log hyperparameter analysis
        Replicates: Hyperparameter Analysis section
        """
        if not self.use_wandb or not self.initialized:
            print("📊 Hyperparameter Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging Hyperparameter Analysis to WandB...")
        
        # Identify hyperparameter columns
        hyperparam_cols = ['learning_rate', 'sigma_prior', 'kl_penalty', 'batch_size']
        available_cols = [col for col in hyperparam_cols if col in df_hyperparams.columns]
        
        for param in available_cols:
            if df_hyperparams[param].nunique() > 1:  # Only analyze if there's variation
                # Parameter vs Performance
                plt.figure(figsize=(10, 6))
                
                if df_hyperparams[param].nunique() <= 10:  # Categorical-like
                    sns.boxplot(data=df_hyperparams, x=param, y='stochastic_accuracy')
                else:  # Continuous
                    plt.scatter(df_hyperparams[param], df_hyperparams['stochastic_accuracy'], alpha=0.6)
                    
                plt.title(f'Performance vs {param}')
                plt.tight_layout()
                wandb.log({f"hyperparam_{param}_analysis": wandb.Image(plt)})
                plt.close()
                
                # Correlation with performance
                correlation = df_hyperparams[param].corr(df_hyperparams['stochastic_accuracy'])
                wandb.log({f"hyperparam_{param}_correlation": correlation})
    
    def log_prior_analysis(self, df_priors, study_name=None):
        """
        Log prior analysis (learned vs random prior comparison)
        Replicates: Prior Analysis section
        """
        if not self.use_wandb or not self.initialized:
            print("📊 Prior Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging Prior Analysis to WandB...")
        
        # If prior type information is available
        if 'prior_type' in df_priors.columns:
            prior_types = sorted(df_priors['prior_type'].unique())
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_priors, x='prior_type', y='stochastic_accuracy')
            plt.title('Performance by Prior Type')
            plt.tight_layout()
            wandb.log({"prior_type_comparison": wandb.Image(plt)})
            plt.close()
            
            # Statistical comparison
            if len(prior_types) == 2:
                data1 = df_priors[df_priors['prior_type'] == prior_types[0]]['stochastic_accuracy']
                data2 = df_priors[df_priors['prior_type'] == prior_types[1]]['stochastic_accuracy']
                t_stat, p_value = stats.ttest_ind(data1, data2)
                wandb.log({
                    "prior_comparison_t_stat": t_stat,
                    "prior_comparison_p_value": p_value,
                    "prior_comparison_significant": p_value < 0.05
                })
    
    def log_overall_analysis(self, df_all, study_name=None):
        """
        Log overall correlation analysis and experiment summaries
        Replicates: Overall Analysis section with correlation heatmaps
        """
        if not self.use_wandb or not self.initialized:
            print("📊 Overall Analysis tracking (WandB disabled)")
            return
            
        print("📊 Logging Overall Analysis to WandB...")
        
        # 1. Correlation heatmap of key metrics
        numeric_cols = ['stochastic_accuracy', 'risk_certificate', 'kl_per_n', 'N']
        available_numeric = [col for col in numeric_cols if col in df_all.columns]
        
        if len(available_numeric) > 1:
            correlation_matrix = df_all[available_numeric].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f')
            plt.title('Correlation Matrix of Key Metrics')
            plt.tight_layout()
            wandb.log({"overall_correlation_heatmap": wandb.Image(plt)})
            plt.close()
        
        # 2. Experiment type summary
        if 'experiment_type' in df_all.columns:
            exp_summary = df_all.groupby('experiment_type').agg({
                'stochastic_accuracy': ['mean', 'std', 'count'],
                'risk_certificate': ['mean'],
                'status': lambda x: (x == 'success').mean()
            }).round(4)
            
            wandb.log({
                "experiment_type_summary": wandb.Table(
                    columns=["Experiment_Type", "Accuracy_Mean", "Accuracy_Std", "Risk_Mean", "Success_Rate", "Count"],
                    data=[[exp_type,
                          exp_summary.loc[exp_type, ('stochastic_accuracy', 'mean')],
                          exp_summary.loc[exp_type, ('stochastic_accuracy', 'std')],
                          exp_summary.loc[exp_type, ('risk_certificate', 'mean')],
                          exp_summary.loc[exp_type, ('status', '<lambda>')],
                          int(exp_summary.loc[exp_type, ('stochastic_accuracy', 'count')])]
                         for exp_type in df_all['experiment_type'].unique()]
                )
            })
        
        # 3. Overall statistics
        wandb.log({
            "study_total_experiments": len(df_all),
            "study_success_rate": (df_all['status'] == 'success').mean(),
            "study_mean_accuracy": df_all['stochastic_accuracy'].mean(),
            "study_std_accuracy": df_all['stochastic_accuracy'].std(),
            "study_mean_risk": df_all['risk_certificate'].mean(),
            "study_best_accuracy": df_all['stochastic_accuracy'].max()
        })
        
        print("✅ Enhanced WandB tracking completed successfully!")

# Global logger instance
_global_logger = None

def get_logger() -> WandbLogger:
    """Get the global wandb logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WandbLogger()
    return _global_logger

def init_wandb_experiment(config: Dict[str, Any], use_wandb: bool = True, 
                         experiment_name: Optional[str] = None,
                         project_override: Optional[str] = None, 
                         tags: Optional[list] = None) -> WandbLogger:
    """Initialize a wandb experiment and return the logger."""
    logger = WandbLogger(use_wandb=use_wandb)
    
    # Extract experiment name from config if not provided
    if experiment_name is None:
        experiment_name = config.get('experiment', {}).get('name')
    
    # Extract project from config if not overridden
    if project_override is None:
        project_override = config.get('wandb', {}).get('project')
    
    # Extract tags from config if not provided
    if tags is None:
        config_tags = config.get('wandb', {}).get('tags', [])
        # Add automatic tags based on configuration
        auto_tags = []
        if 'data' in config and isinstance(config['data'], dict):
            auto_tags.append(config['data'].get('name', 'unknown'))
        if 'pac_bayes' in config and isinstance(config['pac_bayes'], dict):
            auto_tags.append(config['pac_bayes'].get('objective', 'unknown'))
        if 'model' in config and isinstance(config['model'], dict):
            auto_tags.append(config['model'].get('type', 'unknown'))
            auto_tags.append(f"layers_{config['model'].get('layers', 4)}")
        
        tags = config_tags + auto_tags
    
    # Flatten config for wandb
    from configs.config import flatten_config
    flat_config = flatten_config(config)
    
    logger.init_experiment(flat_config, experiment_name, project_override, tags)
    
    # Set as global logger
    global _global_logger
    _global_logger = logger
    
    return logger
