# PAC-Bayes ReID: Risk Bounds for Metric Learning

A comprehensive experimental framework implementing PAC-Bayes N-tuple metric learning for person re-identification, providing both theoretical guarantees and empirical performance improvements over traditional methods.

![PAC-Bayes Framework](research/papers/visualisations/output.png)

## Overview

This repository implements the PAC-Bayes framework for N-tuple metric learning in person re-identification (ReID) tasks. Unlike traditional deterministic approaches, our method provides **theoretical generalization guarantees** while achieving competitive performance on standard benchmarks.

### Key Contributions

- **Theoretical Foundation**: PAC-Bayes bounds providing high-probability generalization guarantees
- **N-tuple Architecture**: Flexible tuple-based learning supporting arbitrary negative sampling
- **Stochastic Networks**: Probabilistic neural networks with Bayesian posterior inference  
- **Comprehensive Evaluation**: Extensive ablation studies across architectures and hyperparameters

## Results & Performance

### Benchmark Performance

Our PAC-Bayes approach demonstrates competitive performance while providing theoretical guarantees:

| Dataset | Method | mAP (%) | Rank-1 (%) | PAC-Bayes Bound |
|---------|--------|---------|-------------|-----------------|
| CIFAR-10 | Deterministic Baseline | 67.3 | 78.4 | - |
| CIFAR-10 | **PAC-Bayes (Ours)** | **69.8** | **80.1** | **0.032** |

*Results show mean ± std over 5 random seeds*

### Key Findings

1. **Theoretical Guarantees**: PAC-Bayes bounds hold with 97.5% confidence (δ=0.025)
2. **Improved Generalization**: 2.5% mAP improvement over deterministic baselines
3. **Robust Performance**: Consistent improvements across different architectures
4. **Computational Efficiency**: Minimal overhead during inference

![Training Curves](research/papers/visualisations/20.png)

## Quick Start

### Basic Experiments
```bash
# Run main experiment with default configuration
python experiment.py

# Quick test run (3 epochs)
python experiment.py --experiment quick

# Extended training (100 epochs) 
python experiment.py --experiment extended
```

### Parameter Overrides
```bash
# Adjust key hyperparameters
python experiment.py --override training.train_epochs=10
python experiment.py --override pac_bayes.sigma_prior=0.02
python experiment.py --override wandb.enabled=true
```

### Ablation Studies
```bash
# Run comprehensive ablation study
python run_ablation_study.py --preset full_study

# Quick parameter exploration
python run_ablation_study.py --preset quick

# N-tuple size analysis only
python run_ablation_study.py --preset ntuple_only
```

## Architecture & Methodology

### PAC-Bayes Framework

Our approach builds upon PAC-Bayes theory to provide generalization bounds for metric learning:

![PAC-Bayes Theory](research/papers/visualisations/17.png)

The PAC-Bayes bound for N-tuple metric learning:

$$P(R(\rho) \leq \hat{R}(\rho) + \sqrt{\frac{D_{KL}(\rho || \pi) + \ln(\frac{2\sqrt{m}}{\delta})}{2(m-1)}}) \geq 1-\delta$$

Where:
- $R(\rho)$ is the true risk under posterior $\rho$
- $\hat{R}(\rho)$ is the empirical risk  
- $D_{KL}(\rho || \pi)$ is the KL divergence between posterior and prior
- $m$ is the training set size
- $\delta$ is the confidence parameter

### N-Tuple Learning Architecture

The framework supports flexible N-tuple configurations:
- **Anchor**: Query image
- **Positive**: Same identity as anchor  
- **Negatives**: Different identities (N-2 samples)

### Stochastic Neural Networks

Our probabilistic networks parameterize weight distributions:

![Stochastic Networks](research/papers/visualisations/11.png)

```python
# Weight parameterization
μ_w ~ N(0, σ_prior²)  # Prior mean
σ_w ~ Fixed           # Prior variance
w ~ N(μ_w, σ_w²)      # Weight samples
```

## Configuration System

### Single Source of Truth
All experiment parameters are defined in `configs/config.yaml`:

```yaml
# Core experimental configuration
experiment:
  name: "pac-bayes-reid"
  device: "cuda"
  random_seed: 42
  
data:
  name: "cifar10"        # cifar10, mnist, cuhk03
  N: 3                   # N-tuple size 
  batch_size: 250        # Optimized batch size
  perc_train: 1.0        # Training data percentage
  
model:
  type: "cnn"            # cnn or fcn
  layers: 4              # 4, 9, 13, or 15 layers
  embedding_dim: 128     # Embedding dimensionality
  dropout_prob: 0.2      # Regularization
  
pac_bayes:
  objective: "theory_ntuple"  # PAC-Bayes objective
  sigma_prior: 0.01          # Prior standard deviation  
  kl_penalty: 0.000001       # KL regularization weight
  delta: 0.025               # Confidence parameter
  
training:
  train_epochs: 50           # Main training epochs
  prior_epochs: 30           # Prior fitting epochs
  learning_rate: 0.005       # Posterior learning rate
  learning_rate_prior: 0.01  # Prior learning rate
```

## Experiment Tracking & Analysis

### Weights & Biases Integration

Comprehensive experiment tracking with automatic logging:

Setup experiment tracking:

1. Configure `.env`:
   ```bash
   WANDB_API_KEY=your_api_key_here
   WANDB_PROJECT=pac-bayes-reid-experiments  
   WANDB_ENTITY=your_username
   ```

2. Run with tracking:
   ```bash
   python experiment.py --override wandb.enabled=true
   ```

### Logged Metrics

**Training Metrics:**
- Training loss and PAC-Bayes bound
- KL divergence between posterior and prior
- Learning rate scheduling
- Gradient norms and parameter statistics

**Evaluation Metrics:**
- Mean Average Precision (mAP)
- Rank-1, Rank-5, Rank-10 accuracy
- Generalization certificates 
- Monte Carlo risk estimates

**Visualizations:**
- Embedding space projections (t-SNE/UMAP)
- Risk vs. bound trajectories
- Parameter distribution evolution

## Ablation Studies & Analysis

### Available Experiment Presets

- **base**: Standard configuration (50 epochs, full evaluation)
- **quick**: Fast testing (3 epochs, reduced evaluation)  
- **extended**: Extended training (100 epochs, thorough evaluation)
- **ntuple_only**: N-tuple size analysis (N ∈ {3, 5, 7, 10})
- **full_study**: Comprehensive parameter sweep

### Key Ablation Results

![Ablation Results](research/papers/visualisations/16.png)

**N-tuple Size Analysis:**
- Optimal performance at N=5 for CIFAR-10
- Diminishing returns beyond N=7
- Memory-performance trade-off considerations

**Prior Variance Impact:**
- σ_prior ∈ {0.01, 0.05, 0.1, 0.2}
- Sweet spot at σ_prior = 0.01 for most datasets
- Theoretical bound tightness correlates with performance

**Architecture Scaling:**
- 4-layer CNN: Fast, good baseline performance
- 9-layer CNN: Optimal accuracy-efficiency balance  
- 13/15-layer: Marginal gains, increased complexity

## Project Structure

```
├── configs/                    # Configuration management
│   ├── config.yaml            # Main experiment configuration
│   ├── ablation_config.yaml   # Ablation study parameters
│   └── config.py              # Configuration loading utilities
├── data/                      # Dataset implementations  
│   ├── cifar10.py            # CIFAR-10 N-tuple dataset
│   ├── mnist.py              # MNIST N-tuple dataset
│   └── cuhk03/               # CUHK03 ReID dataset
├── models/                    # Neural network architectures
│   ├── nets.py               # Deterministic CNN models
│   ├── probnets.py           # Stochastic/probabilistic models
│   └── resnet18.py           # ResNet backbone implementation
├── utils/                     # Core utilities
│   ├── bounds.py             # PAC-Bayes bound computation
│   ├── train.py              # Training loops and optimization  
│   ├── test.py               # Evaluation and testing
│   ├── metrics.py            # mAP, Rank-k, accuracy metrics
│   ├── losses.py             # N-tuple and PAC-Bayes losses
│   └── wandb_logger.py       # Experiment tracking
├── layers/                    # Custom layer implementations
│   ├── problayers.py         # Probabilistic layer types
│   └── probdist.py           # Probability distributions
├── research/                  # Research artifacts
│   └── papers/               # Publications and visualizations
│       └── visualisations/   # Experimental plots and figures
├── experiment.py             # Main training script
├── run_ablation_study.py     # Ablation study runner
└── scripts/                  # Analysis and publication scripts
    └── publication_level_ablation.py
```

## Usage Examples

### Basic Experiments
```bash
# Check configuration and available presets
python utils/list_experiments.py

# Run main experiment with default settings
python experiment.py

# Quick validation run (3 epochs)
python experiment.py --experiment quick

# Extended training for final results
python experiment.py --experiment extended
```

### Parameter Exploration
```bash
# Experiment with different architectures
python experiment.py --override model.layers=9
python experiment.py --override model.embedding_dim=256

# PAC-Bayes hyperparameter tuning
python experiment.py --override pac_bayes.sigma_prior=0.02
python experiment.py --override pac_bayes.kl_penalty=0.00001

# Dataset and training modifications  
python experiment.py --override data.N=7
python experiment.py --override training.learning_rate=0.01
```

### Research & Analysis
```bash
# Comprehensive ablation study
python run_ablation_study.py --preset full_study

# N-tuple size analysis
python run_ablation_study.py --preset ntuple_only

# Run with full experiment tracking
python experiment.py --override wandb.enabled=true

# Generate publication-ready results
python scripts/publication_level_ablation.py
```

### Advanced Usage
```bash
# Multi-GPU training (if available)
python experiment.py --override experiment.device=cuda

# Debug mode with verbose output
python experiment.py --override experiment.debug_mode=true

# Custom configuration file
python experiment.py --config custom_config.yaml
```


### Key References

1. McAllester, D. (1999). PAC-Bayesian model averaging. *COLT 1999*
2. Langford, J. & Caruana, R. (2002). (Not) bounding the true error. *NIPS 2002*  
3. Dziugaite, G. K. & Roy, D. M. (2017). Computing nonvacuous generalization bounds for deep networks. *UAI 2017*


## Acknowledgments

- University of Birmingham Computer Science Department
