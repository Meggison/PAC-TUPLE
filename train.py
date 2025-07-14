import math
import numpy as np
import torch
from tqdm import tqdm, trange
from data import reid_data_prepare, ntuple_reid_data, DynamicNTupleDataset, loadbatches
from models import ResNet, ProbResNet50_BN, ProbBottleneck
from bounds import PBBobj_Ntuple
from loss import NTupleLoss
import torch.optim as optim


def adjust_learning_rate(optimizer, epoch, config):
    """
    Implements the learning rate schedule from the paper:
    Linear warm-up for `warmup_epochs` followed by step decay.
    """
    warmup_epochs = 20
    lr_start = 8e-6
    lr_max = 8e-4
    
    if epoch < warmup_epochs:
        # Linear warm-up from lr_start to lr_max
        lr = lr_start + (lr_max - lr_start) * epoch / warmup_epochs
    else:
        # Step decay after warmup
        # Paper: "decayed by a factor of 0.5 for every 60 epochs"
        decay_steps = (epoch - warmup_epochs) // 60
        lr = lr_max * (0.5 ** decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run_ntuple_experiment(config):
    """
    Experiment runner adapted to use memory-efficient dynamic sampling.
    """
    # --- 1. SETUP ---
    print("--- Starting Experiment ---")
    print(f"Config: {config}")
    device = config['device']
    torch.manual_seed(7)
    np.random.seed(0)

    loader_kargs = {'num_workers': 4, 'pin_memory': True} if 'cuda' in device else {}
    rho_prior = math.log(math.exp(config['sigma_prior']) - 1.0)

    # --- 2. DATA PREPARATION ---
    print("\n--- Preparing Data ---")
    class_img_labels = reid_data_prepare(config['data_list_path'], config['data_dir_path'])
    all_class_ids = list(class_img_labels.keys())

    val_size = int(len(all_class_ids) * config['val_perc'])
    train_ids = all_class_ids[val_size:]
    val_ids = all_class_ids[:val_size]

    # --- MODIFIED: Use DynamicNTupleDataset instead of pre-computing ---
    print("Initializing dynamic datasets...")
    train_dataset = DynamicNTupleDataset(class_img_labels, train_ids, N=config['N'], samples_per_epoch_multiplier=config['samples_per_class'])
    val_dataset = DynamicNTupleDataset(class_img_labels, val_ids, N=config['N'], samples_per_epoch_multiplier=config['samples_per_class'])
    test_dataset = val_dataset # Using val set for testing as a placeholder
    # --------------------------------------------------------------------

    train_loader, test_loader, prior_loader, _, _, _ = loadbatches(
        train_dataset, val_dataset, test_dataset, loader_kargs, config['batch_size']
    )
    print("Data preparation complete.")

    # --- 3. MODEL INITIALIZATION ---
    print("\n--- Initializing Models ---")
    net0 = ResNet().to(device)
    
    net = ProbResNet50_BN(ProbBottleneck, [3, 4, 6, 3], rho_prior=rho_prior, init_net=net0, device=device).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # --- 4. SETUP FOR PAC-BAYES ---
    print("\n--- Setting up PAC-Bayes Objective ---")
    pbobj = PBBobj_Ntuple(
        objective=config['objective'],
        delta=config['delta'],
        delta_test=config['delta_test'],
        mc_samples=config['mc_samples'],
        kl_penalty=config['kl_penalty'],
        device=device,
        n_posterior=len(train_dataset),
        n_bound=len(val_dataset) if val_dataset else 0
    )
    ntuple_loss_fn = NTupleLoss(mode=config['ntuple_mode'], embedding_dim=config['embedding_dim']).to(device)

    # --- 5. MAIN TRAINING LOOP ---
    print("\n--- Starting Training ---")
    results = {}
    for epoch in trange(config['train_epochs'], desc="Training Progress"):
        net.train()
        adjust_learning_rate(optimizer, epoch, config)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            bound, _, _ = pbobj.train_obj_ntuple(net, batch, ntuple_loss_fn)
            bound.backward()
            optimizer.step()

        # --- 6. PERIODIC EVALUATION ---
        if (epoch + 1) % config['test_interval'] == 0:
            if prior_loader:
                print(f"\n--- Evaluating at Epoch {epoch+1} ---")
                
                # Use the comprehensive evaluation from PBBobj_Ntuple
                eval_results = pbobj.comprehensive_evaluation(net, prior_loader, ntuple_loss_fn)
                
                # Print detailed results
                print(f"  Stochastic Predictor - Risk: {eval_results['stochastic']['risk']:.5f}, Pseudo-Acc: {eval_results['stochastic']['pseudo_accuracy']:.4f}")
                print(f"  Posterior Mean Predictor - Risk: {eval_results['posterior_mean']['risk']:.5f}, Pseudo-Acc: {eval_results['posterior_mean']['pseudo_accuracy']:.4f}")
                print(f"  Ensemble Predictor - Risk: {eval_results['ensemble']['risk']:.5f}, Pseudo-Acc: {eval_results['ensemble']['pseudo_accuracy']:.4f}")
                print(f"  Certified N-Tuple Risk: {eval_results['certificate']['final_risk']:.5f}")
                print(f"  KL Divergence: {eval_results['certificate']['kl_divergence']:.5f}")
                print(f"  Empirical N-Tuple Risk (on val set): {eval_results['certificate']['empirical_risk']:.5f}")
                print(f"  Certificate Pseudo-Accuracy (on val set): {eval_results['certificate']['pseudo_accuracy']:.4f}")
                
                # Store comprehensive results
                results[epoch+1] = eval_results

    print("\n--- Training Finished ---")
    return results


# --- Example Usage in a Notebook Cell ---
# Define your configuration


import time

# ... (keep all your other functions and imports as they are) ...

if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'device': 'cuda'    ,
        'data_list_path': '/Users/misanmeggison/Self-certified-Tuple-wise/cuhk03/train.txt',
        'data_dir_path': '/Users/misanmeggison/Self-certified-Tuple-wise/cuhk031/images_detected/',
        'val_perc': 0.2,
        'batch_size': 64,
        'learning_rate': 8e-4,  # Starting LR, will be adjusted by the scheduler
        'weight_decay': 5e-4,
        'sigma_prior': 0.1,
        'train_epochs': 100, # Increased from 5, paper uses 600
        'test_interval': 10, # Adjusted for more epochs
        'objective': 'fclassic',
        'delta': 0.025,
        'delta_test': 0.01,
        'mc_samples': 100,
        'kl_penalty': 1.0,
        'N': 4,
        'samples_per_class': 4,
        'ntuple_mode': 'mpn',
        'embedding_dim': 2048, # Updated for ResNet-50
        # Set num_workers to 0 if you continue to see multiprocessing issues,
        # otherwise you can try a value > 0 for faster data loading.
        'num_workers': 4
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