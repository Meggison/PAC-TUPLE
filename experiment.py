import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.optim as optim
import os

# Add error handling for missing modules
try:
    from data.cifar10 import CIFAR10NTupleDataset
    from data.mnist import MNISTTupleDataset
except ImportError as e:
    print(f"Warning: Could not import data modules: {e}")
    print("Make sure data_gen module is in your Python path")

from utils.bounds import PACBayesBound
from models.nets import ReIDCNNet4l, ReIDCNNet9l, ReIDCNNet13l, ReIDCNNet15l
from models.probnets import ProbReIDCNNet4l, ProbReIDCNNet9l, ProbReIDCNNet13l, ProbReIDCNNet15l, ProbReIDNet4l
from utils.train import trainReIDNet, trainProbReIDNet
from utils.test import testReIDNet, testStochasticReID, testPosteriorMeanReID, testEnsembleReID, computeRiskCertificatesReID
from utils.metrics import compute_map, compute_rank1_accuracy
from utils.wandb_logger import init_wandb_experiment, get_logger



def train_deterministic_baseline(name_data, N, learning_rate, momentum,
                               layers, train_epochs, batch_size, embedding_dim,
                               dropout_prob, perc_train, device, verbose=True):
    """Train a deterministic baseline to establish performance expectations"""
    
    print("\n=== Training Deterministic Baseline ===")
    
    # Get the wandb logger
    logger = get_logger()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data loading
    if name_data == 'cifar10':
        train_dataset = CIFAR10NTupleDataset(
            train=True, num_negatives=N-2, samples_per_class=int(500 * perc_train)
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
        print(f"Model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    
    # Training loop
    for epoch in trange(train_epochs, desc="Baseline Training"):
        print(f"epoch {epoch+1}/{train_epochs}")
        trainReIDNet(net, optimizer, epoch, train_loader, device=device, verbose=False)
        
        # Log every 5 epochs for more frequent wandb updates
        if verbose and ((epoch+1) % 5 == 0):
            print("Evaluating on test set...")
            test_error, test_acc, _, _, _, _ = testReIDNet(net, test_loader, device=device, verbose=False)
            print(f"Epoch {epoch+1}: Test Error = {test_error:.4f}, Test Acc = {test_acc:.4f}")
            
            # Log to wandb during training
            logger.log_baseline_training(epoch + 1, test_error, test_acc)

    # Final evaluation after all epochs
    final_error, final_acc, query_embeddings, query_labels, gallery_embeddings, gallery_labels = testReIDNet(net, test_loader, device=device, verbose=False)
    map_score = compute_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
    rank1_score = compute_rank1_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
    print(f"\n=== Baseline Results ===")
    print(f"Final Test Error: {final_error:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"Test mAP: {map_score:.4f}, Rank-1: {rank1_score:.4f}")
    print(f"Expected: Error < 0.3, Accuracy > 0.7 for good performance")
    
    # Log final baseline results
    logger.log_baseline_training(train_epochs, final_error, final_acc, 
                               test_error=final_error, test_acc=final_acc,
                               map_score=map_score, rank1_score=rank1_score)
    
    return net, final_error, final_acc


def runexp(name_data, objective, model, N, sigma_prior, pmin, learning_rate,
           momentum, learning_rate_prior=0.01, momentum_prior=0.9,
           delta=0.025, layers=4, delta_test=0.01, mc_samples=1000, samples_ensemble=1000,
           kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
           verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, 
           verbose_test=False, perc_prior=0.2, batch_size=128, embedding_dim=256, 
           run_baseline=True, debug_mode=True, random_seed=42, use_wandb=True, wandb_project=None):
    """Enhanced runexp with production-ready wandb integration"""

    # Initialize wandb experiment configuration
    experiment_config = {
        # Dataset and model config
        "dataset": name_data,
        "objective": objective,
        "model": model,
        "N": N,
        "layers": layers,
        "embedding_dim": embedding_dim,
        
        # PAC-Bayes hyperparameters
        "sigma_prior": sigma_prior,
        "pmin": pmin,
        "kl_penalty": kl_penalty,
        "delta": delta,
        "delta_test": delta_test,
        "prior_dist": prior_dist,
        
        # Training hyperparameters
        "learning_rate": learning_rate,
        "momentum": momentum,
        "learning_rate_prior": learning_rate_prior,
        "momentum_prior": momentum_prior,
        "train_epochs": train_epochs,
        "prior_epochs": prior_epochs,
        "batch_size": batch_size,
        "dropout_prob": dropout_prob,
        
        # Data split
        "perc_train": perc_train,
        "perc_prior": perc_prior,
        
        # Evaluation parameters
        "mc_samples": mc_samples,
        "samples_ensemble": samples_ensemble,
        
        "random_seed": random_seed,
        "device": device
    }

    # Initialize Weights & Biases logger
    logger = init_wandb_experiment(
        config=experiment_config, 
        use_wandb=use_wandb,
        project_override=wandb_project,
        tags=[objective, name_data, model, f"layers_{layers}"]
    )

    # Set random seed 
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Dataset: {name_data}, Model: {model}, N: {N}, Layers: {layers}")
    print(f"Sigma Prior: {sigma_prior}, Learning Rate: {learning_rate}")
    print(f"Objective: {objective}, KL Penalty: {kl_penalty}")
    print(f"Publication Parameters: batch_size={batch_size}, mc_samples={mc_samples}, epochs={train_epochs}")
    print(f"Random Seed: {random_seed}")

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'

    # Run baseline if requested
    baseline_acc = 0.0
    errornet0 = 1.0
    if run_baseline:
        try:
            baseline_net, baseline_error, baseline_acc = train_deterministic_baseline(
                name_data, N, learning_rate * 3,  # Conservative LR boost for baseline
                momentum, layers, min(train_epochs//5, 20), batch_size, 
                embedding_dim, dropout_prob, perc_train, device, verbose
            )
            
            if baseline_acc < 0.5:
                print(f"\nWARNING: Baseline performance is poor (acc={baseline_acc:.3f})")
                print("This might indicate fundamental issues with:")
                print("- Data loading or preprocessing")
                print("- Loss function implementation") 
                print("- Network architecture")
                print("Consider debugging these before proceeding with PAC-Bayes training")
                            
        except Exception as e:
            print(f"Baseline training failed: {e}")
            baseline_acc = 0.0

    # Data loading with error handling
    try:
        if name_data == 'cifar10':
            full_train_dataset = CIFAR10NTupleDataset(
                train=True, num_negatives=N-2, samples_per_class=int(500 * perc_train)
            )
            test_dataset = CIFAR10NTupleDataset(
                train=False, num_negatives=N-2, samples_per_class=100
            )
        elif name_data == 'mnist':
            full_train_dataset = MNISTTupleDataset(
                train=True, samples_per_class=int(1000 * perc_train)
            )
            test_dataset = MNISTTupleDataset(
                train=False, samples_per_class=1000
            )
        else:
            raise ValueError(f"Unsupported dataset: {name_data}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

    # Split data for prior/posterior
    total_train_size = len(full_train_dataset)
    prior_size = int(total_train_size * perc_prior)
    posterior_size = total_train_size - prior_size

    print(f"Dataset sizes: Total={total_train_size}, Prior={prior_size}, Posterior={posterior_size}")

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
                
                # Add periodic evaluation and logging for prior training
                if verbose and ((epoch+1) % 5 == 0):
                    print("Evaluating prior network...")
                    temp_error, temp_acc, temp_query_emb, temp_query_lbl, temp_gallery_emb, temp_gallery_lbl = testReIDNet(net0, test_loader, device=device, verbose=False)
                    temp_map = compute_map(temp_query_emb, temp_query_lbl, temp_gallery_emb, temp_gallery_lbl)
                    temp_rank1 = compute_rank1_accuracy(temp_query_emb, temp_query_lbl, temp_gallery_emb, temp_gallery_lbl)
                    print(f"Prior Epoch {epoch+1}: Error = {temp_error:.4f}, Acc = {temp_acc:.4f}, mAP = {temp_map:.4f}, Rank-1 = {temp_rank1:.4f}")
                    
                    # Log prior training progress to wandb
                    logger.log_metrics({
                        "prior_train/epoch": epoch + 1,
                        "prior_train/error": temp_error,
                        "prior_train/accuracy": temp_acc,
                        "prior_train/map": temp_map,
                        "prior_train/rank1": temp_rank1
                    })
                    
            errornet0, _, query_emb_p, query_lbl_p, gallery_emb_p, gallery_lbl_p = testReIDNet(net0, test_loader, device=device, verbose=False)
        else:
            errornet0, _, query_emb_p, query_lbl_p, gallery_emb_p, gallery_lbl_p = testReIDNet(net0, test_loader, device=device, verbose=False)

        prior_map = compute_map(query_emb_p, query_lbl_p, gallery_emb_p, gallery_lbl_p)
        prior_rank1 = compute_rank1_accuracy(query_emb_p, query_lbl_p, gallery_emb_p, gallery_lbl_p)
        print(f"Prior network test error: {errornet0:.4f}")
        print(f"Prior Test mAP: {prior_map:.4f}, Rank-1: {prior_rank1:.4f}")
        
        # Log prior network results
        logger.log_metrics({
            "prior/error": errornet0,
            "prior/map": prior_map,
            "prior/rank1": prior_rank1,
            "prior/parameters": sum(p.numel() for p in net0.parameters() if p.requires_grad)
        })

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

    # Initialize PAC-Bayes objective 
    try:
        bound = PACBayesBound(
            objective=objective, 
            pmin=pmin,
            delta=delta, 
            delta_test=delta_test, 
            mc_samples=mc_samples, 
            kl_penalty=kl_penalty, 
            device=device, 
            n_posterior=len(train_loader.dataset), 
            n_bound=len(val_bound.dataset)
        )

    except Exception as e:
        print(f"PAC-Bayes objective initialization failed: {e}")
        return None

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    if debug_mode:
        print(f"Probabilistic model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
        
    # Training loop for PAC-Bayes posterior
    print(f"\nStarting PAC-Bayes training for {train_epochs} epochs...")
    
    for epoch in trange(train_epochs, desc="PAC-Bayes Training"):
        try:
            trainProbReIDNet(net, optimizer, bound, epoch, train_loader, 
                           lambda_var=None, optimizer_lambda=None, verbose=verbose)
            
            # Periodic evaluation during training - more frequent logging
            if verbose_test and ((epoch+1) % 5 == 0): # every 5 epochs for better wandb tracking
                try:
                    train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n = computeRiskCertificatesReID(
                        net, bound, val_bound, device=device, lambda_var=None)
                    # stch_risk, stch_acc = testStochasticReID(net, test_loader, bound, device=device)
                    stch_risk, stch_acc, stch_map, stch_rank1 = testStochasticReID(net, test_loader, bound, device=device)
                    
                    print(f"\n*** Checkpoint Epoch {epoch+1} ***")
                    print(f"KL/n: {kl_per_n:.6f}, Pseudo acc: {pseudo_accuracy:.3f}")
                    print(f"Stochastic: risk={stch_risk:.3f}, acc={stch_acc:.3f}", f"mAP={stch_map:.4f}, Rank-1={stch_rank1:.4f}")
                    
                    # Log checkpoint metrics
                    logger.log_checkpoint_evaluation(
                        epoch + 1, train_obj, risk_ntuple, empirical_risk_ntuple, 
                        pseudo_accuracy, kl_per_n, stch_risk, stch_acc, 
                        stch_map, stch_rank1
                    )
                    
                except Exception as e:
                    print(f"Evaluation failed at epoch {epoch+1}: {e}")
                    
        except Exception as e:
            print(f"Training failed at epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n=== Final Publication-Level Evaluation ===")
    try:
        train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n = computeRiskCertificatesReID(
            net, bound, val_bound, device=device, lambda_var=None)
        stch_risk, stch_acc, stch_map, stch_rank1 = testStochasticReID(net, test_loader, bound, device=device)
        post_risk, post_acc, post_map, post_rank1 = testPosteriorMeanReID(net, test_loader, bound, device=device)
        ens_risk, ens_acc, ens_map, ens_rank1 = testEnsembleReID(net, test_loader, bound, device=device, samples=samples_ensemble)

        # Results summary
        print(f"\n*** Final Publication Results ***")
        print(f"Training Objective: {train_obj:.5f}")
        print(f"Risk Certificate: {risk_ntuple:.5f}")
        print(f"Empirical Risk: {empirical_risk_ntuple:.5f}")
        print(f"KL/n: {kl_per_n:.6f}")
        print(f"Stochastic: risk={stch_risk:.3f}, acc={stch_acc:.3f}, mAP={stch_map:.4f}, Rank-1={stch_rank1:.4f}")
        print(f"Posterior mean: risk={post_risk:.3f}, acc={post_acc:.3f}, mAP={post_map:.4f}, Rank-1={post_rank1:.4f}")
        print(f"Ensemble: risk={ens_risk:.3f}, acc={ens_acc:.3f}, mAP={ens_map:.4f}, Rank-1={ens_rank1:.4f}")

        # Log final results to wandb
        if use_wandb:
            final_metrics = {
                # PAC-Bayes core metrics
                "final/training_objective": train_obj,
                "final/risk_certificate": risk_ntuple,
                "final/empirical_risk": empirical_risk_ntuple,
                "final/kl_per_n": kl_per_n,
                "final/pseudo_accuracy": pseudo_accuracy,
                
                # Stochastic evaluation
                "final/stochastic_risk": stch_risk,
                "final/stochastic_accuracy": stch_acc,
                "final/stochastic_map": stch_map if stch_map is not None else 0.0,
                "final/stochastic_rank1": stch_rank1 if stch_rank1 is not None else 0.0,
                
                # Posterior mean evaluation
                "final/posterior_risk": post_risk,
                "final/posterior_accuracy": post_acc,
                "final/posterior_map": post_map if post_map is not None else 0.0,
                "final/posterior_rank1": post_rank1 if post_rank1 is not None else 0.0,
                
                # Ensemble evaluation
                "final/ensemble_risk": ens_risk,
                "final/ensemble_accuracy": ens_acc,
                "final/ensemble_map": ens_map if ens_map is not None else 0.0,
                "final/ensemble_rank1": ens_rank1 if ens_rank1 is not None else 0.0,
                
                # Model info
                "final/probabilistic_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)
            }
            
            # Add bound quality assessment
            if risk_ntuple < 0.15:
                final_metrics["final/bound_quality"] = "excellent"
            elif risk_ntuple < 0.5:
                final_metrics["final/bound_quality"] = "good"
            elif risk_ntuple < 1.0:
                final_metrics["final/bound_quality"] = "ok"
            else:
                final_metrics["final/bound_quality"] = "vacuous"
            
            # Log final results using modular logger
            final_results = {
                'train_obj': train_obj,
                'risk_ntuple': risk_ntuple,
                'empirical_risk_ntuple': empirical_risk_ntuple,
                'kl_per_n': kl_per_n,
                'pseudo_accuracy': pseudo_accuracy,
                'stch_risk': stch_risk,
                'stch_accuracy': stch_acc,
                'stch_map': stch_map,
                'stch_rank1': stch_rank1,
                'post_risk': post_risk,
                'post_accuracy': post_acc,
                'post_map': post_map,
                'post_rank1': post_rank1,
                'ens_risk': ens_risk,
                'ens_accuracy': ens_acc,
                'ens_map': ens_map,
                'ens_rank1': ens_rank1,
                'baseline_error': errornet0,
                'prior_map': prior_map,
                'prior_rank1': prior_rank1
            }
            
            logger.log_final_results(final_results)
            logger.log_hyperparameter_summary(experiment_config)

        print(f"\n*** Results Analysis ***")
        print(f"Objective, Dataset, N, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_NTuple, Empirical_Risk_NTuple, KL, Pseudo_accuracy, Stch risk, Stch accuracy, Stch mAP, Stch Rank-1, Post mean risk, Post mean accuracy, Post mean mAP, Post mean Rank-1, Ens risk, Ens accuracy, Ens mAP, Ens Rank-1, 01 error prior net, Prior mAP, Prior Rank-1, perc_train, perc_prior")
        print(f"{objective}, {name_data}, {N}, {sigma_prior:.5f}, {1e-5:.5f}, {learning_rate:.5f}, {momentum:.5f}, {learning_rate_prior:.5f}, {momentum_prior:.5f}, {kl_penalty:.5f}, {dropout_prob:.5f}, {train_obj:.5f}, {risk_ntuple:.5f}, {empirical_risk_ntuple:.5f}, {kl_per_n:.5f}, {pseudo_accuracy:.5f}, {stch_risk:.5f}, {stch_acc:.5f}, {stch_map:.4f}, {stch_rank1:.4f}, {post_risk:.5f}, {post_acc:.5f}, {post_map:.4f}, {post_rank1:.4f}, {ens_risk:.5f}, {ens_acc:.5f}, {ens_map:.4f}, {ens_rank1:.4f}, {errornet0:.5f}, {prior_map:.4f}, {prior_rank1:.4f}, {perc_train:.5f}, {perc_prior:.5f}")
        
        # Debug mode detailed assessment
        if debug_mode:
            print(f"\n=== Publication-Level Diagnostic Summary ===")
            print(f"1. KL Divergence: {kl_per_n:.6f} {'✓ OK' if kl_per_n > 1e-6 else 'PROBLEM - Too low!'}")
            print(f"2. Pseudo Accuracy: {pseudo_accuracy:.3f} {'✓ OK' if pseudo_accuracy > 0.6 else 'PROBLEM - Too low!'}")
            print(f"3. Bound Quality: {risk_ntuple:.3f} {'✓ Non-vacuous' if risk_ntuple < 1.0 else 'Vacuous'}")
            
            # bound quality assessment
            if risk_ntuple < 0.15:
                print("EXCELLENT: Tight bound comparable to paper results!")
            elif risk_ntuple < 0.5:
                print("GOOD: Reasonable bound for metric learning")
            elif risk_ntuple < 1.0:
                print("OK: Non-vacuous but loose bound")
            
            print(f"4. Stochastic vs Deterministic: {abs(stch_acc - post_acc):.3f} {'✓ Good difference' if abs(stch_acc - post_acc) > 0.05 else '❌ No meaningful difference'}")
            
            if run_baseline and baseline_acc > 0:
                print(f"5. vs Baseline: PAC-Bayes={stch_acc:.3f}, Baseline={baseline_acc:.3f} {'✓ Competitive' if stch_acc > baseline_acc * 0.8 else '❌ Much worse than baseline'}")
            
       
            print(f"   Your result: Risk {risk_ntuple:.3f}, Acc {stch_acc:.1%}")

        return {
            'train_obj': train_obj,
            'risk_ntuple': risk_ntuple,
            'empirical_risk_ntuple': empirical_risk_ntuple,
            'kl_per_n': kl_per_n,
            'pseudo_accuracy': pseudo_accuracy,
            'stch_risk': stch_risk,
            'stch_accuracy': stch_acc,
            'stch_map': stch_map,
            'stch_rank1': stch_rank1,
            'post_risk': post_risk,
            'post_accuracy': post_acc,
            'post_map': post_map,
            'post_rank1': post_rank1,
            'ens_risk': ens_risk,
            'ens_accuracy': ens_acc,
            'ens_map': ens_map,
            'ens_rank1': ens_rank1,
            'baseline_error': errornet0,
            'prior_map': prior_map,
            'prior_rank1': prior_rank1
        }
        
        # Close wandb run
        if use_wandb:
            wandb.finish()
        # Finish wandb run properly
        logger.finish()
        
        return results

    except Exception as e:
        print(f"Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.finish()
        return None

if __name__ == "__main__":
    # Load configuration from YAML config system
    print("=== PAC-Bayes N-tuple Metric Learning Experiment ===")
    print("Loading configuration from configs/config.yaml...")
    
    # Import config system
    from configs.config import get_experiment_config, extract_runexp_params, validate_config
    
    try:
        # Load the base configuration (your best performing setup)
        config = get_experiment_config("base", "configs/config.yaml")
        
        # Validate configuration
        if not validate_config(config):
            print("Configuration validation failed. Please check configs/config.yaml")
            exit(1)
        
        # Extract parameters for runexp function
        params = extract_runexp_params(config)
        
        # Print configuration summary
        print(f"Running experiment: {config['experiment']['name']}")
        print(f"Dataset: {params['name_data']}, Model: {params['model']}, N: {params['N']}")
        print(f"Objective: {params['objective']}, Sigma Prior: {params['sigma_prior']}")
        print(f"Learning Rate: {params['learning_rate']}, KL Penalty: {params['kl_penalty']}")
        print(f"Training Epochs: {params['train_epochs']}, Batch Size: {params['batch_size']}")
        print("=" * 60)
        
        # Run experiment with loaded configuration
        results = runexp(**params)
        
        if results is not None:
            print("\nExperiment completed successfully!")
            print(f"Final stochastic accuracy: {results['stch_accuracy']:.1%}")
            print(f"Risk certificate: {results['risk_ntuple']:.4f}")
            print(f"KL/n: {results['kl_per_n']:.6f}")
            
            if results['risk_ntuple'] < 0.2 and results['stch_accuracy'] > 0.7:
                print("EXCELLENT: Results are publication-ready!")
            elif results['risk_ntuple'] < 1.0 and results['stch_accuracy'] > 0.5:
                print("GOOD: Solid results, approaching publication level")
            else:
                print("NEEDS IMPROVEMENT: Consider hyperparameter tuning")
        else:
            print("\nExperiment failed - check error messages above")
            
    except Exception as e:
        print(f"\nError loading configuration or running experiment: {e}")
        print("\nThis might be due to:")
        print("- Missing or invalid configs/config.yaml file")
        print("- Configuration validation errors")
        print("- Insufficient GPU memory")
        print("- Missing dependencies or imports")
        print("- Data loading issues")
        
        import traceback
        traceback.print_exc()
        
        print("\nTo fix this:")
        print("1. Check that configs/config.yaml exists")
        print("2. Use the config-based runner: python run_config_experiment.py --experiment current_experiment")

