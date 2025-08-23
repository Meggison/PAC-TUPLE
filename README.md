# PAC-Bayes ReID Experiments

A streamlined experimental framework for PAC-Bayes N-tuple metric learning on person re-identification datasets.

## Quick Start

### Run Your Main Experiment
```bash
python experiment.py                    # Uses your main configuration
python run_config_experiment.py        # Same as above with more options
```

### Run Preset Variations
```bash
python run_config_experiment.py --experiment quick     # Fast test (3 epochs)
python run_config_experiment.py --experiment extended  # Long training (100 epochs)
```

### Override Parameters
```bash
python run_config_experiment.py --override training.train_epochs=10
python run_config_experiment.py --override wandb.enabled=true
python run_config_experiment.py --override pac_bayes.sigma_prior=0.02
```

## Configuration System

### Single Source of Truth
All experiment parameters are defined in `configs/config.yaml`:

```yaml
# Your main configuration
experiment:
  name: "pac-bayes-reid"
  device: "cuda"
  
data:
  name: "cifar10"
  N: 3
  batch_size: 250
  
model:
  type: "cnn"
  layers: 4
  embedding_dim: 128
  
pac_bayes:
  objective: "theory_ntuple"
  sigma_prior: 0.01
  kl_penalty: 0.000001
  
training:
  train_epochs: 50
  learning_rate: 0.005
```

### Benefits
- ✅ **Single configuration file** - No duplicate parameters
- ✅ **Easy parameter changes** - Modify once, use everywhere  
- ✅ **Runtime overrides** - Quick parameter adjustments
- ✅ **Preset variations** - Common configurations pre-defined
- ✅ **Validation** - Catch configuration errors early

## Wandb Integration

Set up experiment tracking:

1. Configure `.env`:
   ```
   WANDB_API_KEY=your_api_key_here
   WANDB_PROJECT=pac-bayes-reid-experiments
   WANDB_ENTITY=your_username
   ```

2. Run with tracking:
   ```bash
   python run_config_experiment.py --override wandb.enabled=true
   ```

## Available Presets

- **base**: Your main configuration (50 epochs, full evaluation)
- **quick**: Fast testing (3 epochs, reduced evaluation)  
- **extended**: Long training (100 epochs, thorough evaluation)

## Project Structure

```
├── configs/
│   ├── config.yaml          # Single configuration file
│   └── config.py            # Configuration loading utilities
├── experiment.py            # Main training script
├── run_config_experiment.py # Advanced runner with overrides
├── list_experiments.py      # Show available configurations
└── utils/                   # Training, testing, metrics, wandb
```

## Usage Examples

```bash
# Check your configuration
python list_experiments.py

# Run your main experiment
python experiment.py

# Quick test run
python run_config_experiment.py --experiment quick

# Test with different parameters
python run_config_experiment.py --override pac_bayes.sigma_prior=0.02
python run_config_experiment.py --override training.learning_rate=0.01

# Run with wandb tracking
python run_config_experiment.py --override wandb.enabled=true

# Extended training run
python run_config_experiment.py --experiment extended --override wandb.enabled=true
```

Clean, simple, and focused on your standard training workflow! 🎯
