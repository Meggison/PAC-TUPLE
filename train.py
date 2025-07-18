import math
import numpy as np
import torch
from tqdm import tqdm, trange
from data import reid_data_prepare, ntuple_reid_data, DynamicNTupleDataset, loadbatches
from models import ResNet, ProbResNet_BN, ProbBottleneckBlock
from bounds import PBBobj_Ntuple
from loss import NTupleLoss
import torch.optim as optim
import time


def run_ntuple_experiment(config):
    """
    Experiment runner with proper PAC-Bayes methodology:
    Stage 1: Train prior network on subset of data
    Stage 2: Train posterior network with PAC-Bayes objective
    """
    # --- 1. SETUP ---
    print("--- Starting Experiment ---")
    print(f"Config: {config}")
    device = config['device']
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': config.get('num_workers', 4), 'pin_memory': True} if 'cuda' in device else {}
    rho_prior = math.log(math.exp(config['sigma_prior']) - 1.0)

    # --- 2. PROPER DATA SPLITTING FOR PAC-BAYES ---
    print("\n--- Preparing Data with PAC-Bayes Split ---")
    class_img_labels = reid_data_prepare(config['data_list_path'], config['data_dir_path'])
    all_class_ids = list(class_img_labels.keys())

    # Three-way split for proper PAC-Bayes
    prior_split = int(len(all_class_ids) * config.get('perc_prior', 0.2))
    val_split = int(len(all_class_ids) * config['val_perc'])
    
    prior_ids = all_class_ids[:prior_split]                           # 20% for prior training
    train_ids = all_class_ids[prior_split:-val_split]                 # 60% for posterior training  
    val_ids = all_class_ids[-val_split:]                              # 20% for bounds computation

    print(f"Data split: Prior={len(prior_ids)}, Train={len(train_ids)}, Val={len(val_ids)} classes")

    # --- Create datasets for each split ---
    print("Initializing dynamic datasets...")
    prior_dataset = DynamicNTupleDataset(class_img_labels, prior_ids, N=config['N'], 
                                        samples_per_epoch_multiplier=config['samples_per_class'])
    train_dataset = DynamicNTupleDataset(class_img_labels, train_ids, N=config['N'], 
                                        samples_per_epoch_multiplier=config['samples_per_class'])
    val_dataset = DynamicNTupleDataset(class_img_labels, val_ids, N=config['N'], 
                                      samples_per_epoch_multiplier=config['samples_per_class'])

    # Create data loaders
    from torch.utils.data import DataLoader
    prior_loader = DataLoader(prior_dataset, batch_size=config['batch_size'], shuffle=True, **loader_kargs)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, **loader_kargs)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, **loader_kargs)
    
    print("Data preparation complete.")

    # --- 3. STAGE 1: TRAIN PRIOR NETWORK ---
    print("\n--- Stage 1: Training Prior Network ---")
    net0 = ResNet().to(device)
    # Train prior network on prior data subset
    # Use Adam for prior training as N-tuple loss benefits from adaptive learning rates
    prior_optimizer = optim.Adam(net0.parameters(), 
                                lr=config.get('learning_rate_prior', 3e-4), 
                                weight_decay=5e-4)
    
    ntuple_loss_fn = NTupleLoss(mode=config['ntuple_mode'], embedding_dim=2048).to(device)
    
    print("Training prior network...")
    for epoch in trange(config.get('prior_epochs', 20), desc="Prior Training"):
        net0.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(prior_loader, desc=f"Prior Epoch {epoch+1}", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                prior_optimizer.zero_grad()
                
                # Forward pass for all components
                anchor_embed = net0(anchor)
                positive_embed = net0(positive)
                
                # Handle negatives
                batch_size, n_negatives, channels, height, width = negatives.shape
                negatives_flat = negatives.view(-1, channels, height, width)
                negative_embeds_flat = net0(negatives_flat)
                negative_embeds = negative_embeds_flat.view(batch_size, n_negatives, -1)
                
                # Compute N-tuple loss
                loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(net0.parameters(), max_norm=1.0)
                prior_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Warning: Prior training batch failed: {e}")
                continue
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if epoch % 5 == 0:
            print(f"Prior epoch {epoch}, avg loss: {avg_loss:.4f}")
    
    print("Prior training completed!")

    # --- 4. STAGE 2: INITIALIZE POSTERIOR WITH TRAINED PRIOR ---
    print("\n--- Stage 2: Initializing Posterior Network ---")
    
    # FIXED: Pass the ProbBottleneckBlock class as the first argument
    net = ProbResNet_BN(ProbBottleneckBlock, rho_prior=rho_prior, init_net=net0, device=device).to(device)
    
    # Update priors to use trained network weights
    update_priors_from_trained_network(net, net0)
    
    # Use Adam for posterior training too - better for complex N-tuple loss
    optimizer = optim.Adam(net.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config.get('weight_decay', 5e-4))
    
    # Add learning rate scheduler for better training stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # --- 5. SETUP FOR PAC-BAYES ---
    print("\n--- Setting up PAC-Bayes Objective ---")
    # CRITICAL: For PAC-Bayes theory, we need actual number of samples,
    # not just dataset length which might be dynamic
    posterior_n_size = len(train_dataset) * config['samples_per_class']  # Actual samples used
    bound_n_size = len(val_dataset) * config['samples_per_class']      # Actual samples for bounds
    
    print(f"PAC-Bayes setup: posterior_n={posterior_n_size}, bound_n={bound_n_size}")
    
    pbobj = PBBobj_Ntuple(
        objective=config['objective'],
        delta=config['delta'],
        delta_test=config['delta_test'],
        mc_samples=config['mc_samples'],
        kl_penalty=config['kl_penalty'],
        device=device,
        n_posterior=posterior_n_size,
        n_bound=bound_n_size
    )

    # --- THEORETICAL VALIDATION CHECKS ---
    print("\n--- PAC-Bayes Theoretical Validation ---")
    
    # Check N-tuple size consistency
    sample_batch = next(iter(train_loader))
    actual_tuple_size = pbobj.get_tuple_size(sample_batch)
    expected_tuple_size = config['N']
    
    if actual_tuple_size != expected_tuple_size:
        print(f"⚠️ WARNING: Tuple size mismatch! Expected: {expected_tuple_size}, Got: {actual_tuple_size}")
        print("This will invalidate PAC-Bayes bounds!")
    else:
        print(f"✅ N-tuple size consistent: {actual_tuple_size}")
    
    # Validate data splits
    total_classes = len(all_class_ids)
    prior_ratio = len(prior_ids) / total_classes
    train_ratio = len(train_ids) / total_classes
    val_ratio = len(val_ids) / total_classes
    
    print(f"Data split ratios - Prior: {prior_ratio:.2f}, Train: {train_ratio:.2f}, Val: {val_ratio:.2f}")
    
    # Run theoretical validation
    warnings = pbobj.validate_pac_bayes_theory(actual_tuple_size, posterior_n_size)
    for warning in warnings:
        print(f"  {warning}")
    
    if not warnings:
        print("✅ All PAC-Bayes theoretical checks passed!")

    # --- 6. MAIN PAC-BAYES TRAINING LOOP ---
    print("\n--- Stage 2: PAC-Bayes Training ---")
    results = {}
    for epoch in trange(config['train_epochs'], desc="PAC-Bayes Training Progress"):
        net.train()
        epoch_bounds = []
        epoch_kls = []
        epoch_emp_risks = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            try:
                optimizer.zero_grad()
                bound, kl_scaled, emp_risk = pbobj.train_obj_ntuple(net, batch, ntuple_loss_fn)
                
                # Check for numerical issues
                if torch.isnan(bound) or torch.isinf(bound):
                    print(f"⚠️ NaN/Inf detected at epoch {epoch+1}, skipping batch")
                    continue
                
                bound.backward()
                
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                epoch_bounds.append(bound.item())
                epoch_kls.append(kl_scaled.item() if torch.is_tensor(kl_scaled) else kl_scaled)
                epoch_emp_risks.append(emp_risk.item() if torch.is_tensor(emp_risk) else emp_risk)
                
            except Exception as e:
                print(f"Warning: Training batch failed: {e}")
                continue
        
        # Step the learning rate scheduler
        scheduler.step()

        # --- 7. COMPREHENSIVE EVALUATION ---
        if (epoch + 1) % config['test_interval'] == 0:
            print(f"\n--- Evaluating at Epoch {epoch+1} ---")
            
            # Calculate averages for this epoch
            avg_bound = np.mean(epoch_bounds) if epoch_bounds else float('inf')
            avg_kl = np.mean(epoch_kls) if epoch_kls else 0.0
            avg_emp_risk = np.mean(epoch_emp_risks) if epoch_emp_risks else 0.0
            
            try:
                # Compute risk certificates using validation set
                final_risk, kl_div, emp_risk_val, pseudo_acc = pbobj.compute_final_stats_risk_ntuple(
                    net, val_loader, ntuple_loss_fn)
                
                # Multiple inference modes for comprehensive evaluation
                stch_risk, stch_acc = pbobj.test_stochastic_ntuple(net, val_loader, ntuple_loss_fn)
                post_risk, post_acc = pbobj.test_posterior_mean_ntuple(net, val_loader, ntuple_loss_fn)
                ens_risk, ens_acc = pbobj.test_ensemble_ntuple(net, val_loader, ntuple_loss_fn, num_samples=10)
                
                results[epoch+1] = {
                    'train_bound': avg_bound,
                    'train_kl': avg_kl,
                    'train_emp_risk': avg_emp_risk,
                    'certified_risk': final_risk,
                    'kl_divergence': kl_div,
                    'empirical_risk': emp_risk_val,
                    'pseudo_accuracy': pseudo_acc,
                    'stochastic_risk': stch_risk,
                    'stochastic_accuracy': stch_acc,
                    'posterior_mean_risk': post_risk,
                    'posterior_mean_accuracy': post_acc,
                    'ensemble_risk': ens_risk,
                    'ensemble_accuracy': ens_acc,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                print(f"  Train Bound: {avg_bound:.5f}")
                print(f"  Train KL: {avg_kl:.3f}")
                print(f"  Certified Risk: {final_risk:.5f}")
                print(f"  KL Divergence: {kl_div:.5f}")
                print(f"  Pseudo-Accuracy: {pseudo_acc:.4f}")
                print(f"  Stochastic Accuracy: {stch_acc:.4f}")
                print(f"  Posterior Mean Accuracy: {post_acc:.4f}")
                print(f"  Ensemble Accuracy: {ens_acc:.4f}")
                
                # PAC-Bayes bound tightness analysis
                bound_gap = final_risk - emp_risk_val
                print(f"  Bound Gap: {bound_gap:.5f}")
                
                # Enhanced bound analysis for your custom bounds
                if bound_gap < 0:
                    print("  ⚠️ CRITICAL: Negative bound gap - bounds are invalid!")
                elif bound_gap > 0.8:
                    print("  ⚠️ Very loose bounds - consider increasing MC samples or adjusting δ")
                elif bound_gap > 0.5:
                    print("  ⚠️ Loose bounds - your nested structure may need tuning")
                elif bound_gap < 0.2:
                    print("  ✅ Tight bounds - excellent for custom nested structure!")
                else:
                    print("  ✅ Reasonably tight bounds!")
                
                # Specific recommendations for your custom bounds
                if bound_gap > 0.6 and config['mc_samples'] < 200:
                    print("  💡 Try increasing mc_samples for tighter inner bound")
                if bound_gap > 0.7 and config['delta'] > 0.01:
                    print("  💡 Consider lowering delta for tighter outer bound")
                
                # Performance warnings
                if avg_kl > 50000:
                    print("  ⚠️ Very high KL divergence - consider reducing kl_penalty")
                elif final_risk > 0.99:
                    print("  ⚠️ Very loose bounds - bounds may not be meaningful")
                elif pseudo_acc > 0.4:
                    print("  ✅ Good progress!")
                    
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                continue

    print("\n--- Training Finished ---")
    return results


def update_priors_from_trained_network(prob_net, trained_net):
    """
    Update the prior parameters in the probabilistic network to match 
    the trained deterministic network.
    """
    print("Updating priors from trained network...")
    
    try:
        # Get state dict from trained network
        trained_state = trained_net.state_dict()
        
        # Update priors in probabilistic network
        with torch.no_grad():
            for name, module in prob_net.named_modules():
                if hasattr(module, 'weight_prior') and hasattr(module, 'bias_prior'):
                    # Find corresponding layer in trained network
                    # Remove 'prob_' prefix if it exists
                    base_name = name.replace('prob_', '')
                    
                    # Update weight and bias priors
                    for param_name in ['weight', 'bias']:
                        trained_key = f"{base_name}.{param_name}"
                        if trained_key in trained_state:
                            prior_param = getattr(module, f"{param_name}_prior")
                            if hasattr(prior_param, 'mu'):
                                prior_param.mu.data.copy_(trained_state[trained_key])
                                print(f"Updated {name}.{param_name}_prior from {trained_key}")
        
        print("Prior update completed!")
        
    except Exception as e:
        print(f"Warning: Could not update all priors: {e}")
        print("Continuing with default prior initialization...")


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_list_path': '/Users/misanmeggison/Downloads/cukh03/cuhk03/train.txt',
        'data_dir_path': '/Users/misanmeggison/Downloads/cukh03/cuhk03/images_labeled/',
        'val_perc': 0.2,
        'perc_prior': 0.2,  # Percentage of data for prior training
        'batch_size': 64,
        'learning_rate': 1e-4,  # Good for Adam with PAC-Bayes
        'learning_rate_prior': 3e-4,  # Adam-friendly LR for prior training
        'weight_decay': 5e-4,
        'sigma_prior': 0.05,
        'train_epochs': 100,
        'prior_epochs': 20,  # Epochs for prior training
        'test_interval': 10,
        'objective': 'fclassic',
        'delta': 0.025,
        'delta_test': 0.01,
        'mc_samples': 100,
        'kl_penalty': 1.0,  # Good starting point for N-tuple loss
        'N': 4, # Number of samples in each N-tuple
        'samples_per_class': 4,
        'ntuple_mode': 'regular',  # 'regular' or 'mpn'
        'num_workers': 0  # Set to 0 for notebook compatibility
    }

    # --- Run Experiment with Timer ---
    start_time = time.time()
    print("##================== BEGIN EXPERIMENT ==================##")

    exp = run_ntuple_experiment(config)
    print(exp)

    end_time = time.time()
    duration = end_time - start_time
    print("##==================  END EXPERIMENT  ==================##")
    print(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")