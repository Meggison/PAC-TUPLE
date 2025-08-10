import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ntuple.probdist import PBBobj_NTuple



# theoretical_validation.py
def validate_bound_tightness(model, pbobj, data_loader, num_experiments=10):
    """
    Core experiment: Are your bounds actually meaningful?
    """
    results = {
        'empirical_risks': [],
        'bounds': [],
        'kl_divs': [],
        'vacuous_bounds': 0
    }
    
    model.eval()
    
    for exp in range(num_experiments):
        total_empirical_risk = 0.0
        total_kl = model.compute_kl()
        
        # Compute empirical risk over dataset
        with torch.no_grad():
            for batch in data_loader:
                anchor, positive, negative = batch
                risk, _ = pbobj.compute_losses_triplet(model, anchor, positive, negative)
                total_empirical_risk += risk.item()
        
        avg_empirical_risk = total_empirical_risk / len(data_loader)
        
        # Compute bound
        bound = pbobj.bound(
            empirical_risk=avg_empirical_risk,
            kl=total_kl,
            train_size=len(data_loader.dataset),
            tuple_size=3
        )
        
        results['empirical_risks'].append(avg_empirical_risk)
        results['bounds'].append(bound.item())
        results['kl_divs'].append(total_kl.item())
        
        # Check if bound is vacuous (> 1.0 for risk)
        if bound.item() > 1.0:
            results['vacuous_bounds'] += 1
    
    # Analysis
    avg_gap = np.mean(np.array(results['bounds']) - np.array(results['empirical_risks']))
    
    print(f"Bound Tightness Analysis:")
    print(f"Average empirical risk: {np.mean(results['empirical_risks']):.4f}")
    print(f"Average bound: {np.mean(results['bounds']):.4f}")
    print(f"Average gap: {avg_gap:.4f}")
    print(f"Vacuous bounds: {results['vacuous_bounds']}/{num_experiments}")
    print(f"Non-vacuous rate: {(num_experiments - results['vacuous_bounds'])/num_experiments:.2%}")
    
    return results

def compare_objectives(model, data_loader):
    """Compare different PAC-Bayes objectives"""
    objectives = ['fquad', 'fclassic', 'ntuple']
    results = {}
    
    for obj in objectives:
        pbobj = PBBobj_NTuple(objective=obj)
        model.eval()
        
        # Single evaluation
        total_risk = 0.0
        kl = model.compute_kl()
        
        with torch.no_grad():
            for batch in data_loader:
                anchor, positive, negative = batch
                risk, _ = pbobj.compute_losses_triplet(model, anchor, positive, negative)
                total_risk += risk.item()
        
        avg_risk = total_risk / len(data_loader)
        bound = pbobj.bound(avg_risk, kl, len(data_loader.dataset), tuple_size=3)
        
        results[obj] = {
            'empirical_risk': avg_risk,
            'bound': bound.item(),
            'gap': bound.item() - avg_risk
        }
    
    return results

def sample_complexity_analysis():
    """Test how bounds behave with different dataset sizes"""
    dataset_sizes = [100, 500, 1000, 5000, 10000]
    results = []
    
    for size in dataset_sizes:
        # Create dataset of specific size
        dataset = SyntheticTripletDataset(
            num_classes=10, 
            samples_per_class=size//10,
            class_separation=2.0,
            intra_class_std=0.5
        )
        
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Simple network
        model = ProbReIDNet4l(embedding_dim=64, rho_prior=1.0)
        pbobj = PBBobj_Triplet(objective='ntuple')
        
        # Quick training (just a few epochs to get reasonable weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            model.train()
            for batch in data_loader:
                optimizer.zero_grad()
                train_obj, _, _ = pbobj.train_obj(model, batch, len(dataset))
                train_obj.backward()
                optimizer.step()
        
        # Evaluate bound tightness
        bound_results = validate_bound_tightness(model, pbobj, data_loader, num_experiments=3)
        
        results.append({
            'dataset_size': size,
            'avg_empirical_risk': np.mean(bound_results['empirical_risks']),
            'avg_bound': np.mean(bound_results['bounds']),
            'avg_gap': np.mean(bound_results['bounds']) - np.mean(bound_results['empirical_risks']),
            'non_vacuous_rate': (3 - bound_results['vacuous_bounds']) / 3
        })
    
    return results
