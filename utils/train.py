# reid_training_complete.py - All training/testing functions adapted
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.losses import GeneralizedTripletLoss
from utils.metrics import compute_accuracy


# ===== TRAINING & TESTING FUNCTIONS =====
def trainReIDNet(net, optimizer, epoch, train_loader, loss_fn=GeneralizedTripletLoss(), device='cuda', verbose=False):
    """Train function for standard ReID networks with N-tuple loss"""
    
    net.train()
    total_loss = 0.0
    total_accuracy = 0.0
        
    for batch_id, batch in enumerate(tqdm(train_loader)):
        anchor_imgs, positive_imgs, negative_imgs = batch[:3]
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
            
            accuracy = compute_accuracy(anchor_norm, positive_norm, negative_norm)
            accuracy = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy

            total_accuracy += accuracy
    
    if verbose:
        print(f"Epoch {epoch}: Loss: {total_loss/len(train_loader):.5f}, "
              f"Accuracy: {total_accuracy/len(train_loader):.5f}")
        


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
        # Handle both 3-tuple and 4-tuple batch formats
        if len(batch) == 4:
            anchor_imgs, positive_imgs, negative_imgs, _ = batch
        else:
            anchor_imgs, positive_imgs, negative_imgs = batch
        
        net.zero_grad()
        
        # Compute training objective using your N-tuple bounds
        train_obj, empirical_risk, kl_term = pbobj.train_obj(net, (anchor_imgs, positive_imgs, negative_imgs), len(train_loader.dataset))
        
        train_obj.backward()
        optimizer.step()
        
        avg_bound += train_obj.item()
        avg_kl += kl_term
        avg_loss += empirical_risk.item()
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            _, accuracy = pbobj.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
            avg_accuracy += accuracy
        
        # flamb objective not implemented yet
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
    
    # Log training metrics using modular logger
    try:
        from utils.wandb_logger import get_logger
        logger = get_logger()
        if logger.is_active:
            logger.log_pacbayes_training(
                epoch, avg_bound/len(train_loader), avg_kl/len(train_loader),
                avg_loss/len(train_loader), avg_accuracy/len(train_loader)
            )
    except ImportError:
        pass  # wandb logger not available
