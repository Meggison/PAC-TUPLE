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


#ch

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
    
    # FIXED: Pass the ProbBottleneckBlock class as the first argument
    net = ProbResNet_BN(ProbBottleneckBlock, rho_prior=rho_prior, init_net=net0, device=device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Add learning rate scheduler for better training stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

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
    ntuple_loss_fn = NTupleLoss(mode=config['ntuple_mode'], embedding_dim=2048).to(device)

    # --- 5. MAIN TRAINING LOOP ---
    print("\n--- Starting Training ---")
    results = {}
    for epoch in trange(config['train_epochs'], desc="Training Progress"):
        net.train()
        # adjust_learning_rate(optimizer, epoch, config)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            bound, _, _ = pbobj.train_obj_ntuple(net, batch, ntuple_loss_fn)
            bound.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Step the learning rate scheduler
        scheduler.step()

        # --- 6. PERIODIC EVALUATION ---
        if (epoch + 1) % config['test_interval'] == 0:
            if prior_loader:
                print(f"\n--- Evaluating at Epoch {epoch+1} ---")
                final_risk, kl, emp_risk, pseudo_acc = pbobj.compute_final_stats_risk_ntuple(net, prior_loader, ntuple_loss_fn)
                print(f"  Certified N-Tuple Risk: {final_risk:.5f}")
                print(f"  KL Divergence: {kl:.5f}")
                print(f"  Empirical N-Tuple Risk (on val set): {emp_risk:.5f}")
                print(f"  Pseudo-Accuracy (on val set): {pseudo_acc:.4f}")
                results[epoch+1] = {'risk': final_risk, 'kl': kl, 'empirical_risk': emp_risk, 'pseudo_accuracy': pseudo_acc}

    print("\n--- Training Finished ---")
    return results


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_list_path': '/Users/misanmeggison/Self-certified-Tuple-wise/cuhk03/train.txt',
        'data_dir_path': '/Users/misanmeggison/Self-certified-Tuple-wise/cuhk031/images_detected/',
        'val_perc': 0.2,
        'batch_size': 64,
        'learning_rate': 1e-4,  # Reduced for stability with large KL
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'sigma_prior': 0.1,
        'train_epochs': 100,
        'test_interval': 10,
        'objective': 'fclassic',
        'delta': 0.025,
        'delta_test': 0.01,
        'mc_samples': 100,
        'kl_penalty': 1.0,  # Restored to normal value
        'N': 4, # Number of samples in each N-tuple
        'samples_per_class': 4,
        'ntuple_mode': 'regular',  # 'regular' or 'mpn'
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