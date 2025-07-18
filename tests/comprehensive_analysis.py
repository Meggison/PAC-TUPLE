#!/usr/bin/env python3
"""
Comprehensive Model Performance Analysis
========================================
This script identifies potential issues affecting model performance and loss.
"""

import torch
import torch.nn as nn
import numpy as np
from models import ResNet, ProbResNet_BN, ProbBottleneckBlock
from loss import NTupleLoss
from bounds import PBBobj_Ntuple
from data import DynamicNTupleDataset, reid_data_prepare
import math

def analyze_model_architecture():
    """Analyze potential issues with model architecture"""
    issues = []
    
    print("=== MODEL ARCHITECTURE ANALYSIS ===")
    
    # Check device consistency
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test standard ResNet
        net0 = ResNet().to(device)
        rho_prior = math.log(math.exp(0.1) - 1.0)
        net = ProbResNet_BN(ProbBottleneckBlock, rho_prior=rho_prior, init_net=net0, device=device).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = net(dummy_input)
        print(f"✓ Model forward pass successful: {output.shape}")
        
        # Test KL computation
        kl = net.compute_kl()
        print(f"✓ KL divergence computation: {kl.item():.6f}")
        
        # Check if KL is reasonable (not too large or zero)
        if kl.item() < 1e-8:
            issues.append("⚠️  KL divergence is very small - may indicate initialization issues")
        elif kl.item() > 1000:
            issues.append("⚠️  KL divergence is very large - may cause training instability")
            
    except Exception as e:
        issues.append(f"❌ Model architecture error: {e}")
    
    return issues

def analyze_loss_function():
    """Analyze potential issues with loss function"""
    issues = []
    
    print("\n=== LOSS FUNCTION ANALYSIS ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test both modes
    for mode in ['regular', 'mpn']:
        try:
            loss_fn = NTupleLoss(mode=mode, embedding_dim=2048).to(device)
            
            # Create test data
            batch_size = 4
            anchor = torch.randn(batch_size, 2048).to(device)
            positive = torch.randn(batch_size, 2048).to(device)
            negatives = torch.randn(batch_size, 2, 2048).to(device)
            
            # Test loss computation
            loss = loss_fn(anchor, positive, negatives)
            print(f"✓ {mode} mode loss: {loss.item():.4f}")
            
            # Check loss magnitude
            if loss.item() < 0.01:
                issues.append(f"⚠️  {mode} loss is very small ({loss.item():.6f}) - embeddings may be too similar")
            elif loss.item() > 10:
                issues.append(f"⚠️  {mode} loss is very large ({loss.item():.4f}) - may cause training instability")
                
            # Test gradient flow
            loss.backward()
            if mode == 'mpn':
                # Check meta-learner gradients
                meta_grads = [p.grad for p in loss_fn.meta_learner.parameters() if p.grad is not None]
                if not meta_grads:
                    issues.append(f"❌ No gradients in meta-learner for {mode} mode")
                else:
                    avg_grad = torch.stack([g.abs().mean() for g in meta_grads]).mean()
                    print(f"   Meta-learner avg gradient magnitude: {avg_grad:.6f}")
                    
        except Exception as e:
            issues.append(f"❌ Loss function error in {mode} mode: {e}")
    
    return issues

def analyze_pac_bayes_bounds():
    """Analyze PAC-Bayes bound computation"""
    issues = []
    
    print("\n=== PAC-BAYES BOUNDS ANALYSIS ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create test objects
        pbobj = PBBobj_Ntuple(
            objective='fclassic',
            delta=0.025,
            mc_samples=10,  # Reduced for testing
            kl_penalty=1.0,
            device=device,
            n_posterior=1000,
            n_bound=1000
        )
        
        # Test bound computation
        empirical_risk = torch.tensor(0.3)
        kl = torch.tensor(5.0)
        train_size = 1000
        tuple_size = 4
        
        bound = pbobj.bound(empirical_risk, kl, train_size, tuple_size)
        print(f"✓ Bound computation: empirical_risk={empirical_risk:.3f}, bound={bound:.3f}")
        
        # Check bound reasonableness
        if bound < empirical_risk:
            issues.append("❌ Bound is smaller than empirical risk - this violates PAC-Bayes theory")
        elif bound > empirical_risk + 2:
            issues.append("⚠️  Bound is much larger than empirical risk - may be too loose")
            
        # Test different tuple sizes
        for ts in [3, 4, 5, 10]:
            try:
                bound_ts = pbobj.bound(empirical_risk, kl, train_size, ts)
                print(f"   Tuple size {ts}: bound = {bound_ts:.3f}")
            except Exception as e:
                issues.append(f"❌ Bound computation failed for tuple_size={ts}: {e}")
                
    except Exception as e:
        issues.append(f"❌ PAC-Bayes bounds error: {e}")
    
    return issues

def analyze_data_pipeline():
    """Analyze potential issues with data pipeline"""
    issues = []
    
    print("\n=== DATA PIPELINE ANALYSIS ===")
    
    # Test data loading and transformations
    try:
        # Test transform pipeline
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dummy image to test transforms
        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        transformed = transform(dummy_img)
        print(f"✓ Image transform successful: {transformed.shape}")
        
        # Check normalization values
        mean_vals = transformed.mean(dim=(1, 2))
        std_vals = transformed.std(dim=(1, 2))
        print(f"   Transformed image stats - mean: {mean_vals.tolist()}, std: {std_vals.tolist()}")
        
        # Check if values are in reasonable range
        if torch.any(torch.abs(mean_vals) > 3):
            issues.append("⚠️  Image normalization may be incorrect - very high mean values")
        if torch.any(std_vals < 0.1):
            issues.append("⚠️  Image normalization may be incorrect - very low std values")
            
    except Exception as e:
        issues.append(f"❌ Data transform error: {e}")
    
    return issues

def analyze_training_hyperparameters():
    """Analyze training hyperparameters for potential issues"""
    issues = []
    
    print("\n=== TRAINING HYPERPARAMETERS ANALYSIS ===")
    
    # Default config from train.py
    config = {
        'learning_rate': 4e-4,
        'weight_decay': 5e-4,
        'batch_size': 64,
        'sigma_prior': 0.1,
        'kl_penalty': 1.0,
        'delta': 0.025,
        'mc_samples': 100,
        'N': 4,
    }
    
    print(f"Current configuration: {config}")
    
    # Analyze learning rate
    if config['learning_rate'] > 1e-2:
        issues.append("⚠️  Learning rate may be too high for stable training")
    elif config['learning_rate'] < 1e-6:
        issues.append("⚠️  Learning rate may be too low - training might be very slow")
    
    # Analyze weight decay
    if config['weight_decay'] > 1e-2:
        issues.append("⚠️  Weight decay may be too high - could over-regularize")
    elif config['weight_decay'] < 1e-6:
        issues.append("⚠️  Weight decay may be too low - underfitting risk")
    
    # Analyze batch size
    if config['batch_size'] < 16:
        issues.append("⚠️  Small batch size may cause noisy gradients")
    elif config['batch_size'] > 256:
        issues.append("⚠️  Large batch size may require learning rate adjustment")
    
    # Analyze sigma_prior
    rho_prior = math.log(math.exp(config['sigma_prior']) - 1.0)
    if rho_prior < -5:
        issues.append("⚠️  Prior sigma is very small - may cause initialization issues")
    elif rho_prior > 0:
        issues.append("⚠️  Prior sigma is large - may cause high initial KL divergence")
    
    # Analyze KL penalty
    if config['kl_penalty'] > 10:
        issues.append("⚠️  KL penalty is very high - may over-regularize")
    elif config['kl_penalty'] < 0.1:
        issues.append("⚠️  KL penalty is very low - may under-regularize")
    
    print(f"✓ Hyperparameter analysis complete")
    
    return issues

def analyze_numerical_stability():
    """Check for potential numerical stability issues"""
    issues = []
    
    print("\n=== NUMERICAL STABILITY ANALYSIS ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test for potential overflow/underflow in loss computation
    try:
        loss_fn = NTupleLoss(mode='regular', embedding_dim=2048).to(device)
        
        # Test with extreme values
        batch_size = 4
        
        # Test 1: Very similar embeddings (potential underflow)
        anchor = torch.randn(batch_size, 2048).to(device)
        positive = anchor + 1e-6 * torch.randn_like(anchor)  # Very similar
        negatives = anchor.unsqueeze(1).repeat(1, 2, 1) + 1e-6 * torch.randn(batch_size, 2, 2048).to(device)
        
        loss_similar = loss_fn(anchor, positive, negatives)
        print(f"✓ Loss with similar embeddings: {loss_similar.item():.6f}")
        
        if torch.isnan(loss_similar) or torch.isinf(loss_similar):
            issues.append("❌ Loss computation produces NaN/Inf with similar embeddings")
        
        # Test 2: Very different embeddings (potential overflow)
        positive_far = -10 * anchor  # Very different
        negatives_far = 10 * torch.randn(batch_size, 2, 2048).to(device)
        
        loss_different = loss_fn(anchor, positive_far, negatives_far)
        print(f"✓ Loss with different embeddings: {loss_different.item():.6f}")
        
        if torch.isnan(loss_different) or torch.isinf(loss_different):
            issues.append("❌ Loss computation produces NaN/Inf with different embeddings")
        
        # Test 3: Temperature parameter
        temp_value = torch.exp(loss_fn.log_s).item()
        print(f"✓ Temperature parameter: {temp_value:.6f}")
        
        if temp_value > 100:
            issues.append("⚠️  Temperature parameter is very high - may cause numerical issues")
        elif temp_value < 0.01:
            issues.append("⚠️  Temperature parameter is very low - may cause gradient issues")
            
    except Exception as e:
        issues.append(f"❌ Numerical stability test failed: {e}")
    
    return issues

def main():
    """Run comprehensive analysis"""
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    all_issues = []
    
    # Run all analyses
    all_issues.extend(analyze_model_architecture())
    all_issues.extend(analyze_loss_function())
    all_issues.extend(analyze_pac_bayes_bounds())
    all_issues.extend(analyze_data_pipeline())
    all_issues.extend(analyze_training_hyperparameters())
    all_issues.extend(analyze_numerical_stability())
    
    # Summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    if not all_issues:
        print("🎉 No major issues detected! Your implementation looks good.")
    else:
        print(f"Found {len(all_issues)} potential issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    print("1. Monitor training metrics:")
    print("   - Loss should decrease steadily")
    print("   - Pseudo-accuracy should increase")
    print("   - KL divergence should stabilize")
    
    print("\n2. Consider these adjustments if training is unstable:")
    print("   - Reduce learning rate (try 1e-4 or 1e-5)")
    print("   - Increase KL penalty if overfitting")
    print("   - Decrease KL penalty if underfitting")
    print("   - Check data augmentation strength")
    
    print("\n3. For better performance:")
    print("   - Ensure sufficient training data per class")
    print("   - Consider curriculum learning (start with easier negatives)")
    print("   - Monitor gradient norms during training")
    print("   - Use learning rate scheduling")

if __name__ == "__main__":
    main()
