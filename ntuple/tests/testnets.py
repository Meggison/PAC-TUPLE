# reid_training_complete.py - All training/testing functions adapted
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from losses import NTupleLoss, GeneralizedTripletLoss, TripletLoss


# ===== TRAINING & TESTING FUNCTIONS =====

def trainReIDNet(net, optimizer, epoch, train_loader, device='cuda', verbose=False):
    """Train function for standard ReID networks with N-tuple loss"""
    
    net.train()
    total_loss = 0.0
    total_accuracy = 0.0
    
    # Use your N-tuple loss
    loss_fn = GeneralizedTripletLoss()
    
    for batch_id, (anchor_imgs, positive_imgs, negative_imgs) in enumerate(tqdm(train_loader)):
        anchor_imgs = anchor_imgs.to(device)
        positive_imgs = positive_imgs.to(device)
        negative_imgs = negative_imgs.to(device)
        
        net.zero_grad()
        
        # Get embeddings
        anchor_embed = net(anchor_imgs)
        positive_embed = net(positive_imgs)
        
        # Handle negative embeddings
        B, N_neg, C, H, W = negative_imgs.shape
        negative_imgs_flat = negative_imgs.view(B * N_neg, C, H, W)
        negative_embeds_flat = net(negative_imgs_flat)
        negative_embeds = negative_embeds_flat.view(B, N_neg, -1)
        
        # Compute loss
        loss = loss_fn(anchor_embed, positive_embed, negative_embeds)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
            positive_norm = F.normalize(positive_embed, p=2, dim=1)
            negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
            
            sim_positive = F.cosine_similarity(anchor_norm, positive_norm)
            sim_negatives = F.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)
            
            similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)
            predictions = torch.argmax(similarities, dim=1)
            accuracy = (predictions == 0).float().mean().item()
            total_accuracy += accuracy
    
    if verbose:
        print(f"Epoch {epoch}: Loss: {total_loss/len(train_loader):.5f}, "
              f"Accuracy: {total_accuracy/len(train_loader):.5f}")

def testReIDNet(net, test_loader, device='cuda', verbose=True):
    """Test function for standard ReID networks"""
    
    net.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    
    loss_fn = GeneralizedTripletLoss()
    
    with torch.no_grad():
        for batch_id, (anchor_imgs, positive_imgs, negative_imgs) in enumerate(tqdm(test_loader)):
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)
            
            # Get embeddings
            anchor_embed = net(anchor_imgs)
            positive_embed = net(positive_imgs)
            
            B, N_neg, C, H, W = negative_imgs.shape
            negative_imgs_flat = negative_imgs.view(B * N_neg, C, H, W)
            negative_embeds_flat = net(negative_imgs_flat)
            negative_embeds = negative_embeds_flat.view(B, N_neg, -1)
            
            # Compute loss
            loss = loss_fn(anchor_embed, positive_embed, negative_embeds)
            total_loss += loss.item()
            
            # Compute accuracy
            anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
            positive_norm = F.normalize(positive_embed, p=2, dim=1)
            negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
            
            sim_positive = F.cosine_similarity(anchor_norm, positive_norm)
            sim_negatives = F.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)
            
            similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)
            predictions = torch.argmax(similarities, dim=1)
            accuracy = (predictions == 0).float().mean().item()
            total_accuracy += accuracy
    
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    if verbose:
        print(f"Test Loss: {avg_loss:.5f}, Test Accuracy: {avg_accuracy:.5f}")
    
    return avg_loss, avg_accuracy

def trainProbReIDNet(net, optimizer, pbobj, epoch, train_loader, lambda_var=None, 
                     optimizer_lambda=None, verbose=False):
    """Train function for probabilistic ReID networks with N-tuple PAC-Bayes bounds"""
    
    net.train()
    avg_bound, avg_kl, avg_loss, avg_accuracy = 0.0, 0.0, 0.0, 0.0
    
    # flamb not implemented yet
    if pbobj.objective == 'flamb' and lambda_var is not None:
        lambda_var.train()
        avg_bound_l, avg_kl_l, avg_loss_l, avg_accuracy_l = 0.0, 0.0, 0.0, 0.0
    
    for batch_id, batch in enumerate(tqdm(train_loader)):
        anchor_imgs, positive_imgs, negative_imgs = batch
        
        net.zero_grad()
        
        # Compute training objective using your N-tuple bounds
        train_obj, empirical_risk, kl_term = pbobj.train_obj(net, batch, len(train_loader.dataset))
        
        train_obj.backward()
        optimizer.step()
        
        avg_bound += train_obj.item()
        avg_kl += kl_term
        avg_loss += empirical_risk.item()
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            _, accuracy = pbobj.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
            avg_accuracy += accuracy
        
        # flamb not implemented yet
        # Handle flamb objective
        if pbobj.objective == 'flamb' and lambda_var is not None:
            lambda_var.zero_grad()
            train_obj_l, empirical_risk_l, kl_term_l = pbobj.train_obj(net, batch, len(train_loader.dataset))
            train_obj_l.backward()
            optimizer_lambda.step()
            
            avg_bound_l += train_obj_l.item()
            avg_kl_l += kl_term_l
            avg_loss_l += empirical_risk_l.item()
            
            with torch.no_grad():
                _, accuracy_l = pbobj.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
                avg_accuracy_l += accuracy_l
    
    if verbose:
        print(f"Epoch {epoch}: Bound: {avg_bound/len(train_loader):.5f}, "
              f"KL/n: {avg_kl/len(train_loader):.5f}, "
              f"Loss: {avg_loss/len(train_loader):.5f}, "
              f"Accuracy: {avg_accuracy/len(train_loader):.5f}")
        
        if pbobj.objective == 'flamb' and lambda_var is not None:
            print(f"After lambda opt: Bound: {avg_bound_l/len(train_loader):.5f}, "
                  f"Lambda: {lambda_var.lamb_scaled.item():.5f}")


def testStochasticReID(net, test_loader, pbobj, device='cuda'):
    """Test function for stochastic ReID predictor"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader)):
            anchor_imgs, positive_imgs, negative_imgs = batch
            
            # Monte Carlo sampling for each sample in batch
            batch_accuracy = 0.0
            batch_risk = 0.0
            
            for i in range(len(anchor_imgs)):
                anchor_single = anchor_imgs[i:i+1].to(device)
                positive_single = positive_imgs[i:i+1].to(device)
                negative_single = negative_imgs[i:i+1].to(device)
                
                # Sample from network
                anchor_embed = net(anchor_single, sample=True)
                positive_embed = net(positive_single, sample=True)
                
                B_neg, N_neg, C, H, W = negative_single.shape
                negative_flat = negative_single.view(B_neg * N_neg, C, H, W)
                negative_embeds_flat = net(negative_flat, sample=True)
                negative_embeds = negative_embeds_flat.view(B_neg, N_neg, -1)
                
                risk, accuracy = pbobj.compute_losses(net, anchor_single, positive_single, negative_single)
                batch_risk += risk.item()
                batch_accuracy += accuracy
            
            batch_risk /= len(anchor_imgs)
            batch_accuracy /= len(anchor_imgs)
            
            total_risk += batch_risk
            total_accuracy += batch_accuracy
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    return avg_risk, avg_accuracy


def testPosteriorMeanReID(net, test_loader, pbobj, device='cuda'):
    """Test function for deterministic ReID predictor (posterior mean)"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            anchor_imgs, positive_imgs, negative_imgs = batch
            
            risk, accuracy = pbobj.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
            total_risk += risk.item()
            total_accuracy += accuracy
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    return avg_risk, avg_accuracy

def testEnsembleReID(net, test_loader, pbobj, device='cuda', samples=100):
    """Test function for ensemble ReID predictor"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader)):
            anchor_imgs, positive_imgs, negative_imgs = batch
            batch_size = anchor_imgs.size(0)
            
            # Collect multiple samples
            anchor_samples = []
            positive_samples = []
            negative_samples = []
            
            for _ in range(samples):
                # Sample embeddings
                anchor_embed = net(anchor_imgs.to(device), sample=True)
                positive_embed = net(positive_imgs.to(device), sample=True)
                
                B, N_neg, C, H, W = negative_imgs.shape
                negative_flat = negative_imgs.view(B * N_neg, C, H, W)
                negative_embeds_flat = net(negative_flat.to(device), sample=True)
                negative_embeds = negative_embeds_flat.view(B, N_neg, -1)
                
                anchor_samples.append(anchor_embed)
                positive_samples.append(positive_embed)
                negative_samples.append(negative_embeds)
            
            # Average embeddings
            avg_anchor = torch.stack(anchor_samples).mean(0)
            avg_positive = torch.stack(positive_samples).mean(0)
            avg_negative = torch.stack(negative_samples).mean(0)
            
            # Compute metrics on averaged embeddings
            risk = pbobj.compute_empirical_risk(avg_anchor, avg_positive, avg_negative)
            accuracy = pbobj.compute_accuracy(avg_anchor, avg_positive, avg_negative)
            
            total_risk += risk.item()
            total_accuracy += accuracy
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    return avg_risk, avg_accuracy

def computeRiskCertificatesReID(net, pbobj, train_loader, device='cuda', lambda_var=None):
    """Compute risk certificates for ReID networks"""
    
    net.eval()
    with torch.no_grad():
        # Use your existing compute_final_stats_risk method
        train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n = pbobj.compute_final_stats_risk(
            net, train_loader, len(train_loader.dataset)
        )
    
    print(f"Training Objective: {train_obj:.5f}")
    print(f"Risk Certificate: {risk_ntuple:.5f}")
    print(f"Empirical Risk: {empirical_risk_ntuple:.5f}")
    print(f"Pseudo Accuracy: {pseudo_accuracy:.5f}")
    print(f"KL/n: {kl_per_n:.5f}")
    
    return train_obj, risk_ntuple, empirical_risk_ntuple, pseudo_accuracy, kl_per_n
