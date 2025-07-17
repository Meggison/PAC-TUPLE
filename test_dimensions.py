#!/usr/bin/env python3
"""
Test script to verify that model dimensions are consistent throughout the pipeline.
"""

import torch
import torch.nn as nn
from models import ResNet, ProbResNet_BN, ProbBottleneckBlock, verify_model_dimensions
from loss import NTupleLoss
import math

def test_resnet_dimensions():
    """Test the standard ResNet model dimensions."""
    print("=== Testing Standard ResNet ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResNet().to(device)
    dim = verify_model_dimensions(model, device)
    return dim

def test_prob_resnet_dimensions():
    """Test the probabilistic ResNet model dimensions."""
    print("\n=== Testing Probabilistic ResNet ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize standard ResNet first
    net0 = ResNet().to(device)
    
    # Initialize probabilistic ResNet
    rho_prior = math.log(math.exp(0.1) - 1.0)  # sigma_prior = 0.1
    net = ProbResNet_BN(ProbBottleneckBlock, rho_prior=rho_prior, init_net=net0, device=device).to(device)
    
    dim = verify_model_dimensions(net, device)
    return dim

def test_loss_function_compatibility():
    """Test that the loss function works with the model output."""
    print("\n=== Testing Loss Function Compatibility ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loss function
    ntuple_loss_fn = NTupleLoss(mode='regular', embedding_dim=2048).to(device)
    
    # Create dummy data matching your data loader format
    batch_size = 4
    n_negatives = 2  # N=4 means 1 anchor + 1 positive + 2 negatives
    
    # Simulate model embeddings
    anchor_embed = torch.randn(batch_size, 2048).to(device)
    positive_embed = torch.randn(batch_size, 2048).to(device)
    negative_embeds = torch.randn(batch_size, n_negatives, 2048).to(device)
    
    # Test loss computation
    try:
        loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
        print(f"✓ Loss computation successful! Loss value: {loss.item():.4f}")
        print(f"  Anchor embeddings shape: {anchor_embed.shape}")
        print(f"  Positive embeddings shape: {positive_embed.shape}")
        print(f"  Negative embeddings shape: {negative_embeds.shape}")
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test the complete pipeline from model to loss."""
    print("\n=== Testing End-to-End Pipeline ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize models
    net0 = ResNet().to(device)
    rho_prior = math.log(math.exp(0.1) - 1.0)
    net = ProbResNet_BN(ProbBottleneckBlock, rho_prior=rho_prior, init_net=net0, device=device).to(device)
    
    # Initialize loss
    ntuple_loss_fn = NTupleLoss(mode='regular', embedding_dim=2048).to(device)
    
    # Create dummy batch (simulating your data loader)
    batch_size = 4
    anchor = torch.randn(batch_size, 3, 224, 224).to(device)
    positive = torch.randn(batch_size, 3, 224, 224).to(device)
    negatives = torch.randn(batch_size, 2, 3, 224, 224).to(device)  # 2 negatives per anchor
    
    try:
        # Simulate the forward pass from bounds.py compute_losses
        all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
        all_embeddings = net(all_images)
        
        # Unpack embeddings
        n_negatives = negatives.shape[1]
        anchor_embed = all_embeddings[0:batch_size]
        positive_embed = all_embeddings[batch_size : batch_size * 2]
        negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1)
        
        # Compute loss
        loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
        
        print(f"✓ End-to-end pipeline successful!")
        print(f"  Input batch shapes - Anchor: {anchor.shape}, Positive: {positive.shape}, Negatives: {negatives.shape}")
        print(f"  All images concatenated shape: {all_images.shape}")
        print(f"  All embeddings shape: {all_embeddings.shape}")
        print(f"  Final embedding shapes - Anchor: {anchor_embed.shape}, Positive: {positive_embed.shape}, Negatives: {negative_embeds.shape}")
        print(f"  Loss value: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Model Dimension Consistency")
    print("="*50)
    
    try:
        # Test standard ResNet
        dim1 = test_resnet_dimensions()
        
        # Test probabilistic ResNet
        dim2 = test_prob_resnet_dimensions()
        
        # Verify dimensions match
        assert dim1 == dim2 == 2048, f"Dimension mismatch: ResNet={dim1}, ProbResNet={dim2}"
        print(f"\n✓ Both models output consistent {dim1}-dimensional embeddings!")
        
        # Test loss function
        loss_ok = test_loss_function_compatibility()
        
        # Test complete pipeline
        pipeline_ok = test_end_to_end_pipeline()
        
        if loss_ok and pipeline_ok:
            print(f"\n🎉 All tests passed! Your model pipeline is dimensionally consistent.")
            print(f"   Model output: {dim1} dimensions")
            print(f"   Loss function expects: 2048 dimensions")
            print(f"   ✓ Everything matches!")
        else:
            print(f"\n❌ Some tests failed. Please check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
