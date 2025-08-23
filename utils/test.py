import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.losses import GeneralizedTripletLoss

from utils.metrics import compute_accuracy, compute_map, compute_rank1_accuracy

def testReIDNet(net, test_loader, loss_fn=GeneralizedTripletLoss(), device='cuda', verbose=True):
    """Test function for standard ReID networks"""
    
    net.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    all_query_embeddings = []
    all_query_labels = []
    all_gallery_embeddings = []
    all_gallery_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Unpack batch: anchor_imgs, positive_imgs, negative_imgs, anchor_labels
            if len(batch) == 4:
                anchor_imgs, positive_imgs, negative_imgs, anchor_labels = batch
            else:
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_labels = None
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)
            anchor_embed = net(anchor_imgs)
            positive_embed = net(positive_imgs)
            B, N_neg, C, H, W = negative_imgs.shape
            negative_imgs_flat = negative_imgs.view(B * N_neg, C, H, W)
            negative_embeds_flat = net(negative_imgs_flat)
            negative_embeds = negative_embeds_flat.view(B, N_neg, -1)
            # Loss and accuracy
            loss = loss_fn(anchor_embed, positive_embed, negative_embeds)
            total_loss += loss.item()
            anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
            positive_norm = F.normalize(positive_embed, p=2, dim=1)
            negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
            accuracy = compute_accuracy(anchor_norm, positive_norm, negative_norm)
            accuracy = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
            total_accuracy += accuracy
            # Embeddings for mAP/Rank-1
            batch_size = anchor_imgs.shape[0]
            if anchor_labels is not None:
                labels = torch.tensor(anchor_labels)
            else:
                raise ValueError("Anchor labels are required for mAP and Rank-1 computation")
            all_query_embeddings.append(anchor_embed.cpu())
            all_query_labels.append(labels)
            all_gallery_embeddings.append(positive_embed.cpu())
            all_gallery_labels.append(labels)
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    query_embeddings = torch.cat(all_query_embeddings, dim=0)
    query_labels = torch.cat(all_query_labels, dim=0)
    gallery_embeddings = torch.cat(all_gallery_embeddings, dim=0)
    gallery_labels = torch.cat(all_gallery_labels, dim=0)
    return avg_loss, avg_accuracy, query_embeddings, query_labels, gallery_embeddings, gallery_labels


def testStochasticReID(net, test_loader, pbobj, device='cuda'):
    """Test function for stochastic ReID predictor"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    all_query_embeddings = []
    all_query_labels = []
    all_gallery_embeddings = []
    all_gallery_labels = []
    
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            if len(batch) == 4:
                anchor_imgs, positive_imgs, negative_imgs, anchor_labels = batch
            else:
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_labels = None
            
            # Monte Carlo sampling for each sample in batch
            batch_accuracy = 0.0
            batch_risk = 0.0
            batch_anchor_embeds = []
            batch_positive_embeds = []
            batch_labels = []
            
            for i in range(len(anchor_imgs)):
                anchor_single = anchor_imgs[i:i+1].to(device)
                positive_single = positive_imgs[i:i+1].to(device)
                negative_single = negative_imgs[i:i+1].to(device)
                
                # Sample embeddings and compute metrics
                risk, accuracy = pbobj.compute_losses(net, anchor_single, positive_single, negative_single)
                batch_risk += risk.item()
                batch_accuracy += accuracy
                
                # Collect embeddings for mAP/Rank-1
                batch_anchor_embeds.append(net(anchor_single).cpu())
                batch_positive_embeds.append(net(positive_single).cpu())
                if anchor_labels is not None:
                    batch_labels.append(torch.tensor([anchor_labels[i]]))
            
            batch_risk /= len(anchor_imgs)
            batch_accuracy /= len(anchor_imgs)
            total_risk += batch_risk
            total_accuracy += batch_accuracy
            
            # Collect embeddings for metric computation
            if anchor_labels is not None:
                all_query_embeddings.append(torch.cat(batch_anchor_embeds, dim=0))
                all_query_labels.append(torch.cat(batch_labels, dim=0))
                all_gallery_embeddings.append(torch.cat(batch_positive_embeds, dim=0))
                all_gallery_labels.append(torch.cat(batch_labels, dim=0))
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    # Compute mAP and Rank-1 if labels available
    if all_query_embeddings:
        query_embeddings = torch.cat(all_query_embeddings, dim=0)
        query_labels = torch.cat(all_query_labels, dim=0)
        gallery_embeddings = torch.cat(all_gallery_embeddings, dim=0)
        gallery_labels = torch.cat(all_gallery_labels, dim=0)
        map_score = compute_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        rank1_score = compute_rank1_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        return avg_risk, avg_accuracy, map_score, rank1_score
    else:
        return avg_risk, avg_accuracy, None, None


def testPosteriorMeanReID(net, test_loader, pbobj, device='cuda'):
    """Test function for deterministic ReID predictor (posterior mean)"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    all_query_embeddings = []
    all_query_labels = []
    all_gallery_embeddings = []
    all_gallery_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 4:
                anchor_imgs, positive_imgs, negative_imgs, anchor_labels = batch
            else:
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_labels = None
            
            risk, accuracy = pbobj.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
            total_risk += risk.item()
            total_accuracy += accuracy
            
            # Collect embeddings for mAP/Rank-1
            anchor_embed = net(anchor_imgs).cpu()
            positive_embed = net(positive_imgs).cpu()
            
            if anchor_labels is not None:
                labels = torch.tensor(anchor_labels)
                all_query_embeddings.append(anchor_embed)
                all_query_labels.append(labels)
                all_gallery_embeddings.append(positive_embed)
                all_gallery_labels.append(labels)
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    # Compute mAP and Rank-1 if labels available
    if all_query_embeddings:
        query_embeddings = torch.cat(all_query_embeddings, dim=0)
        query_labels = torch.cat(all_query_labels, dim=0)
        gallery_embeddings = torch.cat(all_gallery_embeddings, dim=0)
        gallery_labels = torch.cat(all_gallery_labels, dim=0)
        map_score = compute_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        rank1_score = compute_rank1_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        return avg_risk, avg_accuracy, map_score, rank1_score
    else:
        return avg_risk, avg_accuracy, None, None

def testEnsembleReID(net, test_loader, pbobj, device='cuda', samples=100):
    """Test function for ensemble ReID predictor"""
    
    net.eval()
    total_accuracy = 0.0
    total_risk = 0.0
    all_query_embeddings = []
    all_query_labels = []
    all_gallery_embeddings = []
    all_gallery_labels = []
    
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_loader)):
            if len(batch) == 4:
                anchor_imgs, positive_imgs, negative_imgs, anchor_labels = batch
            else:
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_labels = None
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
            
            # Collect embeddings for mAP/Rank-1
            if anchor_labels is not None:
                labels = torch.tensor(anchor_labels)
                all_query_embeddings.append(avg_anchor.cpu())
                all_query_labels.append(labels)
                all_gallery_embeddings.append(avg_positive.cpu())
                all_gallery_labels.append(labels)
    
    avg_risk = total_risk / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    
    # Compute mAP and Rank-1 if labels available
    if all_query_embeddings:
        query_embeddings = torch.cat(all_query_embeddings, dim=0)
        query_labels = torch.cat(all_query_labels, dim=0)
        gallery_embeddings = torch.cat(all_gallery_embeddings, dim=0)
        gallery_labels = torch.cat(all_gallery_labels, dim=0)
        map_score = compute_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        rank1_score = compute_rank1_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
        return avg_risk, avg_accuracy, map_score, rank1_score
    else:
        return avg_risk, avg_accuracy, None, None


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