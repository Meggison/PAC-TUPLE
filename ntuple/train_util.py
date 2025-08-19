import torch
import numpy as np
import math  # Add missing import
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.optim as optim  # Add missing import

from data_gen.cifar10 import CIFAR10NTupleDataset
from data_gen.mnist import MNISTTupleDataset
from boundsntuple import PBBobj_NTuple
from losses import GeneralizedTripletLoss
from nets import ReIDCNNet4l, ReIDCNNet9l, ReIDCNNet13l, ReIDCNNet15l
from probnets import ProbReIDCNNet4l, ProbReIDCNNet9l, ProbReIDCNNet13l, ProbReIDCNNet15l, ProbReIDNet4l
from tests.testnets import trainReIDNet, trainProbReIDNet, testReIDNet, testStochasticReID, testPosteriorMeanReID, testEnsembleReID, computeRiskCertificatesReID


def debug_probabilistic_init(net):
    """Debug probabilistic layer initialization"""
    print("\n=== Probabilistic Layer Initialization Debug ===")
    for name, module in net.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'mu'):
            weight_mu_mean = module.weight.mu.mean().item()
            weight_sigma_mean = module.weight.sigma.mean().item()
            prior_mu_mean = module.weight_prior.mu.mean().item()
            prior_sigma_mean = module.weight_prior.sigma.mean().item()
            
            print(f"{name}:")
            print(f"  Weight μ: {weight_mu_mean:.6f}, σ: {weight_sigma_mean:.6f}")
            print(f"  Prior μ: {prior_mu_mean:.6f}, σ: {prior_sigma_mean:.6f}")
            print(f"  Difference: {abs(weight_mu_mean - prior_mu_mean):.6f}")

# Add debugging utilities
def debug_kl_components(net, epoch, verbose=True):
    """Debug KL divergence components for probabilistic networks"""
    if hasattr(net, 'compute_kl'):
        total_kl = net.compute_kl()
        
        if verbose:
            print(f"\n=== KL Debug Info (Epoch {epoch}) ===")
            print(f"Total KL: {total_kl:.6f}")
            
            # Check individual layer KL if available
            for name, module in net.named_modules():
                if hasattr(module, 'kl_div'):
                    print(f"{name} KL: {module.kl_div:.6f}")
            
            # Check if KL is suspiciously zero
            if total_kl < 1e-8:
                print("WARNING: KL divergence is essentially zero!")
                print("This suggests probabilistic layers aren't sampling properly.")
        
        return total_kl
    return 0.0

# In train_util.py, add this after debug_kl_components function
def debug_kl_computation(net):
    """Debug individual layer KL values and compare with compute_kl()"""
    total_kl = 0
    print("\n=== Individual KL Computation Debug ===")
    for name, module in net.named_modules():
        if hasattr(module, 'kl_div'):
            kl_val = module.kl_div
            if hasattr(kl_val, 'item'):
                kl_val = kl_val.item()
            print(f"{name}: {kl_val:.8f}")
            total_kl += kl_val
    
    compute_kl_val = net.compute_kl()
    if hasattr(compute_kl_val, 'item'):
        compute_kl_val = compute_kl_val.item()
    
    print(f"Manual KL sum: {total_kl:.8f}")
    print(f"compute_kl(): {compute_kl_val:.8f}")
    print(f"Difference: {abs(total_kl - compute_kl_val):.8f}")
    
    # Check for inconsistencies
    if abs(total_kl - compute_kl_val) > 1e-6:
        print("⚠️  WARNING: Manual sum doesn't match compute_kl()!")
        print("This suggests an issue with the compute_kl() implementation.")
    
    return total_kl, compute_kl_val


def validate_ntuple_loss(data_loader, device, verbose=True):
    """Validate N-tuple loss function with sample data"""
    if verbose:
        print("\n=== N-tuple Loss Validation ===")
    
    try:
        for batch_idx, data in enumerate(data_loader):
            if batch_idx > 0:  # Only check first batch
                break
                
            if isinstance(data, (list, tuple)) and len(data) == 2:  # (data, labels)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if verbose:
                    print(f"Input shape: {inputs.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Unique labels: {torch.unique(labels)}")
                    
            elif isinstance(data, torch.Tensor):  # Direct tensor input
                inputs = data.to(device)
                
                if verbose:
                    print(f"Input tensor shape: {inputs.shape}")
                    
            else:  # Handle other data formats
                inputs = data
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(device)
                    
                if verbose:
                    print(f"Data type: {type(inputs)}")
                    if hasattr(inputs, 'shape'):
                        print(f"Shape: {inputs.shape}")
            
            break
            
    except Exception as e:
        if verbose:
            print(f"Error in data validation: {e}")
            print("This might indicate issues with data loading or format")

def train_deterministic_baseline(name_data, N, learning_rate, momentum, 
                               layers, train_epochs, batch_size, embedding_dim,
                               dropout_prob, perc_train, device, verbose=True):
    """Train a deterministic baseline to establish performance expectations"""
    
    print("\n=== Training Deterministic Baseline ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(7)
    np.random.seed(0)
    
    # Data loading (simplified for baseline)
    if name_data == 'cifar10':
        train_dataset = CIFAR10NTupleDataset(
            train=True, num_negatives=N-2, samples_per_class=int(600 * perc_train)
        )
        test_dataset = CIFAR10NTupleDataset(
            train=False, num_negatives=N-2, samples_per_class=100
        )
    elif name_data == 'mnist':
        train_dataset = MNISTTupleDataset(
            train=True, samples_per_class=int(600 * perc_train)
        )
        test_dataset = MNISTTupleDataset(
            train=False, samples_per_class=100
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Validate loss function with better error handling
    validate_ntuple_loss(train_loader, device, verbose)
    
    # Initialize deterministic model
    if name_data == 'cifar10':
        if layers == 4:
            net = ReIDCNNet4l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
        elif layers == 9:
            net = ReIDCNNet9l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
        elif layers == 13:
            net = ReIDCNNet13l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
        elif layers == 15:
            net = ReIDCNNet15l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
    else:  # MNIST
        net = ReIDCNNet4l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    if verbose:
        print(f"Model parameters: {count_parameters(net):,}")
    
    # Training loop
    for epoch in trange(train_epochs, desc="Baseline Training"):
        trainReIDNet(net, optimizer, epoch, train_loader, device=device, verbose=False)
        
        if verbose and ((epoch+1) % 10 == 0):
            test_error, test_acc = testReIDNet(net, test_loader, device=device, verbose=False)
            print(f"Epoch {epoch+1}: Test Error = {test_error:.4f}, Test Acc = {test_acc:.4f}")
    
    # Final evaluation
    final_error, final_acc = testReIDNet(net, test_loader, device=device, verbose=False)
    
    print(f"\n=== Baseline Results ===")
    print(f"Final Test Error: {final_error:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"Expected: Error < 0.3, Accuracy > 0.7 for good performance")
    
    return net, final_error, final_acc

def runexp(name_data, objective, model, N, sigma_prior, pmin, learning_rate,
           momentum, learning_rate_prior=0.01, momentum_prior=0.9,
           delta=0.025, layers=4, delta_test=0.01, mc_samples=1000, samples_ensemble=100,
           kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
           verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, 
           perc_prior=0.2, batch_size=64, embedding_dim=256, run_baseline=True, debug_mode=True):
    """Run an N-tuple PAC-Bayes experiment for metric learning with enhanced debugging"""

    # Set seeds for reproducibility
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loss_fn = GeneralizedTripletLoss(margin=0.3, strategy='hardest', reduction='mean')

    print(f"\n=== Starting N-tuple PAC-Bayes Experiment ===")
    print(f"Dataset: {name_data}, Model: {model}, N: {N}, Layers: {layers}")
    print(f"Sigma Prior: {sigma_prior}, Learning Rate: {learning_rate}")
    
    # Run deterministic baseline first
    if run_baseline:
        baseline_net, baseline_error, baseline_acc = train_deterministic_baseline(
            name_data, N, learning_rate * 10,  # Higher LR for baseline
            momentum, layers, min(train_epochs, 20), batch_size, 
            embedding_dim, dropout_prob, perc_train, device, verbose
        )
        
        if baseline_acc < 0.6:  # Poor baseline performance
            print(f"\nWARNING: Baseline performance is poor (acc={baseline_acc:.3f})")
            print("Consider:")
            print("- Increasing learning rate")
            print("- Checking data loading/preprocessing")
            print("- Verifying loss function implementation")

    # Data loading
    if name_data == 'cifar10':
        # Training set
        full_train_dataset = CIFAR10NTupleDataset(
            train=True, num_negatives=N-2, samples_per_class=int(600 * perc_train)
        )
        # Test set
        test_dataset = CIFAR10NTupleDataset(
            train=False, num_negatives=N-2, samples_per_class=100
        )
        
    elif name_data == 'mnist':
        # Training set
        full_train_dataset = MNISTTupleDataset(
            train=True, samples_per_class=int(600 * perc_train)
        )
        # Test set
        test_dataset = MNISTTupleDataset(
            train=False, samples_per_class=100
        )
    else:
        raise ValueError(f"Unsupported dataset: {name_data}")

    # Split training data for prior/posterior if needed
    total_train_size = len(full_train_dataset)
    prior_size = int(total_train_size * perc_prior)
    posterior_size = total_train_size - prior_size

    if perc_prior > 0:
        prior_dataset, posterior_dataset = torch.utils.data.random_split(
            full_train_dataset, [prior_size, posterior_size]
        )
        prior_loader = DataLoader(prior_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(posterior_dataset, batch_size=batch_size, shuffle=True)
        val_bound = DataLoader(prior_dataset, batch_size=batch_size, shuffle=False)
    else:
        prior_loader = None
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
        val_bound = train_loader

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute rho_prior with bounds checking
    if sigma_prior <= 0:
        raise ValueError(f"sigma_prior must be positive, got {sigma_prior}")
    
    rho_prior = math.log(math.exp(sigma_prior) - 1.0)
    
    if debug_mode:
        print(f"Rho prior: {rho_prior:.6f}")

    # Initialize networks with error handling
    try:
        # Standard model for prior
        if model == 'cnn':
            if name_data == 'cifar10':
                net_class = {4: ReIDCNNet4l, 9: ReIDCNNet9l, 13: ReIDCNNet13l, 15: ReIDCNNet15l}
                if layers not in net_class:
                    raise ValueError(f"Unsupported layers: {layers}")
                net0 = net_class[layers](embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
            else:  # MNIST
                net0 = ReIDCNNet4l(embedding_dim=embedding_dim, dropout_prob=dropout_prob).to(device)
        elif model == 'fcn':
            if name_data == 'cifar10':
                raise RuntimeError('CIFAR-10 not supported with FCN architecture')
            net0 = ProbReIDNet4l(embedding_dim=embedding_dim, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device).to(device)
        else:
            raise RuntimeError(f'Architecture {model} not supported')

        # Train or evaluate prior
        if perc_prior > 0 and prior_loader is not None:
            optimizer_prior = optim.SGD(net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
            print("Training prior network...")
            for epoch in trange(prior_epochs, desc="Prior Training"):
                trainReIDNet(net0, optimizer_prior, epoch, prior_loader, device=device, verbose=verbose)
            errornet0, accnet0 = testReIDNet(net0, test_loader, device=device, verbose=False)
        else:
            errornet0, accnet0 = testReIDNet(net0, test_loader, device=device, verbose=False)

        print(f"Prior network test error: {errornet0:.4f}")
        print(f"Prior network test accuracy: {accnet0:.4f}")

        # Initialize probabilistic model
        if model == 'cnn':
            if name_data == 'cifar10':
                prob_net_class = {4: ProbReIDCNNet4l, 9: ProbReIDCNNet9l, 13: ProbReIDCNNet13l, 15: ProbReIDCNNet15l}
                net = prob_net_class[layers](embedding_dim=embedding_dim, rho_prior=rho_prior, 
                                           prior_dist=prior_dist, device=device, init_net=net0).to(device)
            else:  # MNIST
                net = ProbReIDCNNet4l(embedding_dim=embedding_dim, rho_prior=rho_prior,
                                    prior_dist=prior_dist, device=device, init_net=net0).to(device)
        elif model == 'fcn':
            net = ProbReIDNet4l(embedding_dim=embedding_dim, rho_prior=rho_prior,
                              prior_dist=prior_dist, device=device, init_net=net0).to(device)

    except Exception as e:
        print(f"Network initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Initialize PAC-Bayes objective with publication-level parameters
    try:
        bound = PBBobj_NTuple(
            objective=objective, 
            pmin=1e-5,  # Paper's value
            delta=0.025,  # Paper's exact value
            delta_test=0.01,  # Paper's exact value
            mc_samples=mc_samples,  # Publication-level 
            kl_penalty=kl_penalty, 
            device=device, 
            n_posterior=len(train_loader.dataset), 
            n_bound=len(val_bound.dataset)
        )
    except Exception as e:
        print(f"PAC-Bayes objective initialization failed: {e}")
        return None

    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    if debug_mode:
        print(f"Probabilistic model parameters: {count_parameters(net):,}")
        
        # Enhanced initial debugging
        debug_probabilistic_init(net)
        initial_kl = debug_kl_components(net, 0, verbose=True)
        
        # Test forward pass
        net.eval()
        try:
            with torch.no_grad():
                dummy_input = torch.randn(2, 3, 32, 32).to(device)
                dummy_output = net(dummy_input, sample=True)
                test_kl = net.compute_kl()
                print(f"Test forward pass: output_shape={dummy_output.shape}, KL={test_kl:.8f}")
        except Exception as e:
            print(f"⚠️ Test forward pass failed: {e}")
        finally:
            net.train()

        if initial_kl < 1e-8:
            print("\nCRITICAL: Initial KL is essentially zero!")
            print("This will lead to vacuous bounds. Check:")
            print("1. Probabilistic layer initialization")
            print("2. Prior/posterior parameter differences") 
            print("3. Rho prior value")
            return None

    # Training loop with enhanced error handling
    print(f"\nStarting publication-level PAC-Bayes training for {train_epochs} epochs...")
    
    for epoch in trange(train_epochs, desc="PAC-Bayes Training"):
        try:
            trainProbReIDNet(net, optimizer, bound, epoch, train_loader, 
                           lambda_var=None, optimizer_lambda=None, verbose=verbose)
            
            # Debug KL periodically (less frequent for long training)
            if debug_mode and ((epoch+1) % max(10, train_epochs//10) == 0):
                current_kl = debug_kl_components(net, epoch+1, verbose=True)
                
                if current_kl < 1e-8:
                    print(f"\nRITICAL: KL divergence collapsed to {current_kl:.2e} at epoch {epoch+1}")
                    print("Training likely failed - consider:")
                    print("- Reducing learning rate")
                    print("- Increasing sigma_prior")
                    print("- Checking gradient flow")
                    break
            
            # Periodic evaluation (less frequent for publication runs)
            if verbose_test and ((epoch+1) % max(20, train_epochs//5) == 0):
                try:
                    train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n = computeRiskCertificatesReID(
                        net, bound, val_bound, device=device, lambda_var=None)
                    stch_risk, stch_acc = testStochasticReID(net, test_loader, bound, device=device)
                    
                    print(f"\n*** Checkpoint Epoch {epoch+1} ***")
                    print(f"KL/n: {kl_per_n:.6f}, Pseudo acc: {pseudo_accuracy:.3f}")
                    print(f"Stochastic: risk={stch_risk:.3f}, acc={stch_acc:.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Evaluation failed at epoch {epoch+1}: {e}")
                    
        except Exception as e:
            print(f"Training failed at epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Final evaluation with publication-level MC sampling
    print("\n=== Final Publication-Level Evaluation ===")
    try:
        train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n = computeRiskCertificatesReID(
            net, bound, val_bound, device=device, lambda_var=None)
        stch_risk, stch_acc = testStochasticReID(net, test_loader, bound, device=device)
        post_risk, post_acc = testPosteriorMeanReID(net, test_loader, bound, device=device)
        ens_risk, ens_acc = testEnsembleReID(net, test_loader, bound, device=device, samples=samples_ensemble)

        # Results summary
        print(f"\n*** Final Publication Results ***")
        print(f"Training Objective: {train_obj:.5f}")
        print(f"Risk Certificate: {risk_ntuple:.5f}")
        print(f"Empirical Risk: {empirical_risk_ntuple:.5f}")
        print(f"KL/n: {kl_per_n:.6f}")
        print(f"Stochastic: risk={stch_risk:.3f}, acc={stch_acc:.3f}")
        print(f"Posterior mean: risk={post_risk:.3f}, acc={post_acc:.3f}")
        print(f"Ensemble: risk={ens_risk:.3f}, acc={ens_acc:.3f}")

        # Print formatted results for logging
        print(f"\n*** Results Analysis ***")
        print(f"Objective, Dataset, N, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_NTuple, Empirical_Risk_NTuple, KL, Pseudo_accuracy, Stch risk, Stch accuracy, Post mean risk, Post mean accuracy, Ens risk, Ens accuracy, 01 error prior net, perc_train, perc_prior")
        print(f"{objective}, {name_data}, {N}, {sigma_prior:.5f}, {1e-5:.5f}, {learning_rate:.5f}, {momentum:.5f}, {learning_rate_prior:.5f}, {momentum_prior:.5f}, {kl_penalty:.5f}, {dropout_prob:.5f}, {train_obj:.5f}, {risk_ntuple:.5f}, {empirical_risk_ntuple:.5f}, {kl_per_n:.5f}, {pseudo_accuracy:.5f}, {stch_risk:.5f}, {stch_acc:.5f}, {post_risk:.5f}, {post_acc:.5f}, {ens_risk:.5f}, {ens_acc:.5f}, {errornet0:.5f}, {perc_train:.5f}, {perc_prior:.5f}")
        
        # Enhanced diagnostic summary
        if debug_mode:
            print(f"\n=== Publication-Level Diagnostic Summary ===")
            print(f"1. KL Divergence: {kl_per_n:.6f} {'✓ OK' if kl_per_n > 1e-6 else 'PROBLEM - Too low!'}")
            print(f"2. Pseudo Accuracy: {pseudo_accuracy:.3f} {'✓ OK' if pseudo_accuracy > 0.6 else '❌ PROBLEM - Too low!'}")
            print(f"3. Bound Quality: {risk_ntuple:.3f} {'✓ Non-vacuous' if risk_ntuple < 1.0 else '❌ Vacuous'}")
            
            # Publication-level bound quality assessment
            if risk_ntuple < 0.15:
                print("EXCELLENT: Tight bound comparable to paper results!")
            elif risk_ntuple < 0.5:
                print("GOOD: Reasonable bound for metric learning")
            elif risk_ntuple < 1.0:
                print("Non-vacuous but loose bound")
            
            print(f"4. Stochastic vs Deterministic: {abs(stch_acc - post_acc):.3f} {'✓ Good difference' if abs(stch_acc - post_acc) > 0.05 else '❌ No meaningful difference'}")
            
            if run_baseline and baseline_acc > 0:
                print(f"5. vs Baseline: PAC-Bayes={stch_acc:.3f}, Baseline={baseline_acc:.3f} {'✓ Competitive' if stch_acc > baseline_acc * 0.8 else '❌ Much worse than baseline'}")
            
            # Publication comparison
            print(f"\nPublication Comparison:")
            print(f"   Paper MNIST: Risk ~0.08-0.15, Acc ~85-90%")
            print(f"   Paper CIFAR: Risk ~0.15-0.25, Acc ~75-85%")
            print(f"   Your result: Risk {risk_ntuple:.3f}, Acc {stch_acc:.1%}")

        return {
            'train_obj': train_obj,
            'risk_ntuple': risk_ntuple,
            'empirical_risk_ntuple': empirical_risk_ntuple,
            'kl_per_n': kl_per_n,
            'pseudo_accuracy': pseudo_accuracy,
            'stch_risk': stch_risk,
            'stch_accuracy': stch_acc,
            'post_risk': post_risk,
            'post_accuracy': post_acc,
            'ens_risk': ens_risk,
            'ens_accuracy': ens_acc,
            'baseline_error': errornet0
        }

    except Exception as e:
        print(f"Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def runexp_with_structured_output(name_data, objective, model, N, sigma_prior, pmin, 
                                 learning_rate, momentum, **kwargs):
    """Modified version of runexp that returns structured results for ablation studies"""
    
    # Filter out unsupported parameters before calling runexp
    supported_params = {
        'learning_rate_prior', 'momentum_prior', 'delta', 'layers', 'delta_test', 
        'mc_samples', 'samples_ensemble', 'kl_penalty', 'initial_lamb', 'train_epochs', 
        'prior_dist', 'verbose', 'device', 'prior_epochs', 'dropout_prob', 'perc_train', 
        'verbose_test', 'perc_prior', 'batch_size', 'embedding_dim', 'run_baseline', 
        'debug_mode', 'random_seed'
    }
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    
    # Call the main runexp function
    results = runexp(
        name_data=name_data, 
        objective=objective, 
        model=model, 
        N=N, 
        sigma_prior=sigma_prior, 
        pmin=pmin,
        learning_rate=learning_rate, 
        momentum=momentum, 
        **filtered_kwargs
    )
    
    # If runexp failed, return failure structure
    if results is None:
        return {
            'status': 'failed',
            'experiment_name': kwargs.get('experiment_name', 'unknown'),
            'config': {
                'objective': objective,
                'sigma_prior': sigma_prior,
                'learning_rate': learning_rate,
                'layers': kwargs.get('layers', 4),
                'embedding_dim': kwargs.get('embedding_dim', 256),
                'kl_penalty': kwargs.get('kl_penalty', 1e-6),
                'perc_prior': kwargs.get('perc_prior', 0.2),
                'dropout_prob': kwargs.get('dropout_prob', 0.2),
                'train_epochs': kwargs.get('train_epochs', 100),
                'prior_epochs': kwargs.get('prior_epochs', 20),
            }
        }
    
    # Structure the successful results
    structured_results = {
        'status': 'success',
        'experiment_name': kwargs.get('experiment_name', 'unnamed'),
        'train_obj': results.get('train_obj', 0.0),
        'risk_ntuple': results.get('risk_ntuple', 1.0),
        'empirical_risk_ntuple': results.get('empirical_risk_ntuple', 1.0),
        'kl_per_n': results.get('kl_per_n', 0.0),
        'pseudo_accuracy': results.get('pseudo_accuracy', 0.0),
        'stch_risk': results.get('stch_risk', 1.0),
        'stch_accuracy': results.get('stch_accuracy', 0.0),
        'post_risk': results.get('post_risk', 1.0),
        'post_accuracy': results.get('post_accuracy', 0.0),
        'ens_risk': results.get('ens_risk', 1.0),
        'ens_accuracy': results.get('ens_accuracy', 0.0),
        'baseline_error': results.get('baseline_error', 1.0),
        'config': {
            'objective': objective,
            'name_data': name_data,
            'model': model,
            'N': N,
            'sigma_prior': sigma_prior,
            'pmin': pmin,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'layers': kwargs.get('layers', 4),
            'embedding_dim': kwargs.get('embedding_dim', 128),
            'kl_penalty': kwargs.get('kl_penalty', 1e-6),
            'perc_prior': kwargs.get('perc_prior', 0.2),
            'dropout_prob': kwargs.get('dropout_prob', 0.2),
            'train_epochs': kwargs.get('train_epochs', 100),
            'prior_epochs': kwargs.get('prior_epochs', 20),
            'mc_samples': kwargs.get('mc_samples', 1500),
            'batch_size': kwargs.get('batch_size', 128),
            'prior_dist': kwargs.get('prior_dist', 'gaussian'),
            'perc_train': kwargs.get('perc_train', 0.8),
        },
        # Additional metrics for analysis
        'is_vacuous': results.get('risk_ntuple', 1.0) >= 1.0,
        'kl_reasonable': results.get('kl_per_n', 0.0) > 1e-6,
        'good_accuracy': results.get('stch_accuracy', 0.0) > 0.6,
        'stoch_det_difference': abs(results.get('stch_accuracy', 0.0) - results.get('post_accuracy', 0.0)),
    }
    
    return structured_results

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Publication-level experiment
    print("=== Publication-Level PAC-Bayes N-tuple Metric Learning Experiment ===")
    
    try:
        results = runexp(
            name_data='cifar10',
            objective='nested_ntuple',  # Paper's best objective
            model='cnn',
            N=4,  # 4-tuple (anchor + positive + 2 negatives)
            sigma_prior=0.01,  # Based on your winning experiments
            pmin=1e-5,  # Paper's exact value
            learning_rate=0.005,  # Your breakthrough value
            kl_penalty=1e-6,  # Your breakthrough value
            momentum=0.9,
            layers=4,
            train_epochs=100,  # Full paper-level training
            batch_size=250,  # Paper's exact value
            embedding_dim=128,  # Your successful configuration
            mc_samples=3000,  # Paper's exact value
            samples_ensemble=1000,  # Paper-level ensemble
            verbose=True,
            verbose_test=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            run_baseline=True,
            debug_mode=True,
            perc_prior=0.4,  # Paper's standard split
            prior_epochs=20,  # Adequate prior training
        )
        
        if results is not None:
            print("\nExperiment completed successfully!")
            print(f"Final stochastic accuracy: {results['stch_accuracy']:.1%}")
            print(f"Risk certificate: {results['risk_ntuple']:.4f}")
            print(f"KL/n: {results['kl_per_n']:.6f}")
            
            # Publication quality assessment
            if results['risk_ntuple'] < 0.2 and results['stch_accuracy'] > 0.7:
                print("EXCELLENT: Results are publication-ready!")
            elif results['risk_ntuple'] < 1.0 and results['stch_accuracy'] > 0.5:
                print("GOOD: Solid results, approaching publication level")
            else:
                print("NEEDS IMPROVEMENT: Consider hyperparameter tuning")
        else:
            print("\nExperiment failed - check error messages above")
            
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        print("This might be due to:")
        print("- Missing dependencies or imports")
        print("- Data loading issues")
        print("- Model architecture mismatches")
        print("- Device compatibility problems")
        import traceback
        traceback.print_exc()
        
        print(f"\n=== Troubleshooting Suggestions ===")
        print("1. Reduce batch_size from 250 to 128 if memory issues")
        print("2. Reduce mc_samples from 150000 to 50000 for testing")
        print("3. Reduce train_epochs from 100 to 30 for initial testing")
        print("4. Check CUDA memory usage with nvidia-smi")
        print("5. Test with smaller embedding_dim (64 instead of 128)")
