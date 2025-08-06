import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm, trange
import time
from torch.utils.data import DataLoader

# Your existing imports
from data import reid_data_prepare, ntuple_reid_data, DynamicNTupleDataset, loadbatches
from models import PretrainedResNetWrapper, ProbResNet_bn, ResNet, ProbResNet_BN, ProbBottleneckBlock
from loss import NTupleLoss


def compute_distance_matrix(anchor_features, gallery_features):
    """
    Compute cosine distance matrix between anchor and gallery features.
    
    Args:
        anchor_features: (N_anchor, D) tensor of anchor embeddings
        gallery_features: (N_gallery, D) tensor of gallery embeddings
        
    Returns:
        distance_matrix: (N_anchor, N_gallery) tensor of cosine distances
    """
    # Normalize features
    anchor_norm = F.normalize(anchor_features, p=2, dim=1)
    gallery_norm = F.normalize(gallery_features, p=2, dim=1)
    
    # Compute cosine similarity and convert to distance
    similarity_matrix = torch.mm(anchor_norm, gallery_norm.t())
    distance_matrix = 1.0 - similarity_matrix  # Convert similarity to distance
    
    return distance_matrix


def compute_map_and_rank1(distance_matrix, anchor_labels, gallery_labels):
    """
    Compute Mean Average Precision (mAP) and Rank-1 accuracy.
    
    Args:
        distance_matrix: (N_anchor, N_gallery) tensor of distances
        anchor_labels: (N_anchor,) tensor of anchor identity labels
        gallery_labels: (N_gallery,) tensor of gallery identity labels
        
    Returns:
        map_score: Mean Average Precision
        rank1_acc: Rank-1 accuracy
    """
    num_anchors = distance_matrix.size(0)
    num_gallery = distance_matrix.size(1)
    
    # Convert to numpy for easier processing
    if torch.is_tensor(distance_matrix):
        distance_matrix = distance_matrix.cpu().numpy()
    if torch.is_tensor(anchor_labels):
        anchor_labels = anchor_labels.cpu().numpy()
    if torch.is_tensor(gallery_labels):
        gallery_labels = gallery_labels.cpu().numpy()
    
    ap_scores = []
    rank1_correct = 0
    
    for i in range(num_anchors):
        # Get distances for this anchor
        anchor_dists = distance_matrix[i]
        anchor_label = anchor_labels[i]
        
        # Sort gallery items by distance (ascending)
        sorted_indices = np.argsort(anchor_dists)
        sorted_labels = gallery_labels[sorted_indices]
        
        # Create binary relevance vector (1 if same identity, 0 otherwise)
        relevance = (sorted_labels == anchor_label).astype(int)
        
        # Skip if no relevant items
        if relevance.sum() == 0:
            continue
        
        # Compute Average Precision for this anchor
        ap = compute_average_precision(relevance)
        ap_scores.append(ap)
        
        # Check Rank-1 accuracy
        if sorted_labels[0] == anchor_label:
            rank1_correct += 1
    
    # Compute final metrics
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    rank1_acc = rank1_correct / num_anchors
    
    return map_score, rank1_acc


def compute_average_precision(relevance):
    """
    Compute Average Precision for a single query.
    
    Args:
        relevance: Binary array indicating relevance (1 for relevant, 0 for irrelevant)
        
    Returns:
        ap: Average Precision score
    """
    if relevance.sum() == 0:
        return 0.0
    
    # Compute precision at each rank
    cumsum_rel = np.cumsum(relevance)
    precision_at_k = cumsum_rel / (np.arange(len(relevance)) + 1)
    
    # Average precision is the mean of precision values at relevant positions
    ap = np.sum(precision_at_k * relevance) / relevance.sum()
    
    return ap


def compute_simple_reid_metrics(anchor_embed, positive_embed, negative_embeds):
    """
    Compute simplified reid metrics using just the N-tuple structure.
    Since we know anchor and positive should be closer than anchor and negatives,
    we can compute a simplified rank-1 accuracy.
    
    Args:
        anchor_embed: (B, D) anchor embeddings
        positive_embed: (B, D) positive embeddings  
        negative_embeds: (B, N-2, D) negative embeddings
        
    Returns:
        rank1_acc: Simplified Rank-1 accuracy (how often positive is closest)
        mean_pos_sim: Mean positive similarity
        mean_neg_sim: Mean negative similarity
    """
    batch_size = anchor_embed.size(0)
    
    # Normalize embeddings
    anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
    positive_norm = F.normalize(positive_embed, p=2, dim=1)
    negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
    
    # Compute similarities
    sim_pos = F.cosine_similarity(anchor_norm, positive_norm)  # (B,)
    sim_neg = F.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)  # (B, N-2)
    
    # For each anchor, check if positive similarity is higher than all negative similarities
    max_neg_sim, _ = torch.max(sim_neg, dim=1)  # (B,)
    rank1_correct = (sim_pos > max_neg_sim).sum().item()
    rank1_acc = rank1_correct / batch_size
    
    # Additional metrics
    mean_pos_sim = sim_pos.mean().item()
    mean_neg_sim = sim_neg.mean().item()
    
    return rank1_acc, mean_pos_sim, mean_neg_sim


def update_priors_from_trained_network(prob_net, trained_net):
    """
    FIXED: Proper weight transfer from trained prior to ProbResNet
    """
    print("Attempting to transfer weights from trained prior to ProbResNet...")
    try:
        # Get state dicts
        prior_state = trained_net.model.state_dict()  # From L2NormalizedModel wrapper
        prob_state = prob_net.state_dict()
        
        # Transfer matching parameters
        transferred_count = 0
        for name, param in prior_state.items():
            # Map to ProbResNet structure
            prob_name = f"net.model.{name}"
            if prob_name in prob_state and prob_state[prob_name].shape == param.shape:
                prob_state[prob_name].copy_(param)
                transferred_count += 1
        
        print(f"✅ Successfully transferred {transferred_count} parameters from prior to ProbResNet")
        
    except Exception as e:
        print(f"⚠️ Weight transfer failed: {e}")
        print("Using wrapper initialization only")


# FIXED: Add embedding regularization class
class EmbeddingRegularizer:
    def __init__(self, min_var=1e-4):
        self.min_var = min_var
    
    def __call__(self, embeddings):
        """Add variance regularization to prevent collapse"""
        # Compute variance along batch dimension
        embedding_var = torch.var(embeddings, dim=0)
        # Penalize low variance (encourages diversity)
        var_loss = F.relu(self.min_var - embedding_var).mean()
        return var_loss


def run_ntuple_experiment(config):
    """
    FIXED: Experiment runner using your custom ResNet and ProbResNet models
    with proper weight transfer and sample=False for stability
    """
    # --- 1. SETUP ---
    print("--- Starting Experiment ---")
    print(f"Config: {config}")
    device = config['device']
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    prior_loader = DataLoader(prior_dataset, batch_size=config['batch_size'], shuffle=True, **loader_kargs)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, **loader_kargs)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, **loader_kargs)
    
    print("Data preparation complete.")

    # --- 3. STAGE 1: TRAIN PRIOR NETWORK (USING YOUR CUSTOM RESNET) ---
    print("\n--- Stage 1: Training Prior Network (Pre-trained ResNet50) ---")
    
    # FIXED: Use your custom ResNet class with pre-trained initialization (same as successful test)
    embedding_dim = config.get('embedding_dim', 256)
    net0 = ResNet(embedding_dim=embedding_dim).to(device)
    print("✅ Using custom ResNet with pre-trained ResNet50 backbone (same as successful test)")

    # Train prior network on prior data subset
    prior_optimizer = optim.Adam(net0.parameters(), 
                                lr=config.get('learning_rate_prior', 3e-4), 
                                weight_decay=5e-4)
    
    ntuple_loss_fn = NTupleLoss(mode=config['ntuple_mode'], embedding_dim=embedding_dim).to(device)
    
    print("Training pre-trained prior network...")
    for epoch in trange(config.get('prior_epochs', 20), desc="Prior Training"):
        net0.train()
        epoch_loss = 0
        num_batches = 0
        epoch_correct = 0
        epoch_rank1_acc = 0
        epoch_pos_sim = 0
        epoch_neg_sim = 0
        
        for batch in tqdm(prior_loader, desc=f"Prior Epoch {epoch+1}", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                prior_optimizer.zero_grad()
                
                # Forward pass for all components (ResNet already includes L2 normalization)
                anchor_embed = net0(anchor)
                positive_embed = net0(positive)
                
                # Handle negatives
                batch_size, n_negatives, channels, height, width = negatives.shape
                negatives_flat = negatives.view(-1, channels, height, width)
                negative_embeds_flat = net0(negatives_flat)
                negative_embeds = negative_embeds_flat.view(batch_size, n_negatives, -1)
                
                # Compute N-tuple loss
                loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)

                # Compute metrics (embeddings are already normalized by ResNet)
                sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
                sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
                max_sim_neg, _ = torch.max(sim_neg, dim=1)
                epoch_correct += (sim_pos > max_sim_neg).sum().item()
                
                # Compute ReID metrics for prior training
                rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                    anchor_embed, positive_embed, negative_embeds)
                epoch_rank1_acc += rank1_acc
                epoch_pos_sim += mean_pos_sim
                epoch_neg_sim += mean_neg_sim

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(net0.parameters(), max_norm=1.0)
                prior_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Warning: Prior training batch failed: {e}")
                continue
        
        # FIXED: Added safety checks for division by zero
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_acc = epoch_correct / (num_batches * config['batch_size']) if num_batches > 0 else 0
        avg_rank1 = epoch_rank1_acc / num_batches if num_batches > 0 else 0
        avg_pos_sim = epoch_pos_sim / num_batches if num_batches > 0 else 0
        avg_neg_sim = epoch_neg_sim / num_batches if num_batches > 0 else 0

        if epoch % 5 == 0:
            print(f"Prior epoch {epoch}: loss={avg_loss:.4f}, pseudo-acc={avg_acc:.4f}, rank1={avg_rank1:.4f}, pos_sim={avg_pos_sim:.4f}, neg_sim={avg_neg_sim:.4f}")
    
    print("Pre-trained prior training completed!")

    # --- 4. STAGE 2: INITIALIZE POSTERIOR WITH TRAINED PRIOR ---
    print("\n--- Stage 2: Initializing Posterior Network from Trained Prior ---")
    
    # FIXED: Use PretrainedResNetWrapper initialized with the trained prior model
    wrapped_prior = PretrainedResNetWrapper(embedding_dim=embedding_dim).to(device)
    
    # FIXED: Proper weight transfer using the trained prior
    try:
        print("Transferring weights from trained prior to wrapper...")
        # Since both use the same ResNet backbone, we can transfer weights directly
        prior_state = net0.state_dict()
        wrapper_state = wrapped_prior.state_dict()
        
        # Map weights appropriately
        transferred_keys = []
        for key in prior_state:
            wrapper_key = f"model.{key}"  # Add 'model.' prefix for wrapper
            if wrapper_key in wrapper_state and prior_state[key].shape == wrapper_state[wrapper_key].shape:
                wrapper_state[wrapper_key].copy_(prior_state[key])
                transferred_keys.append(key)
        
        wrapped_prior.load_state_dict(wrapper_state)
        print(f"✅ Successfully transferred {len(transferred_keys)} parameters from trained prior")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not transfer weights: {e}")
        print("Using default PretrainedResNetWrapper initialization")

    # Create ProbResNet initialized from the trained prior
    net = ProbResNet_bn(
        rho_prior=rho_prior, 
        init_net=wrapped_prior,
        device=device
    ).to(device)

    # Enhanced prior update
    update_priors_from_trained_network(net, net0)
    print("✅ ProbResNet initialized from trained prior")

    # FIXED: Use lower learning rate for stability with embedding regularization
    optimizer = optim.Adam(net.parameters(), 
                          lr=1e-6,  # Much lower learning rate to prevent collapse
                          weight_decay=1e-3)  # Higher weight decay for regularization

    # Add learning rate scheduler for better training stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    # FIXED: Add embedding regularizer to prevent collapse
    embedding_regularizer = EmbeddingRegularizer(min_var=1e-3)

    # --- 5. DIRECT N-TUPLE TRAINING SETUP ---
    print("\n--- Setting up Pure N-tuple Training with Pre-trained Base ---")
    print("Training probabilistic model ONLY on N-tuple loss with sample=False")
    print("Using weight decay, gradient clipping, and variance regularization for stability")

    # --- 6. MAIN DIRECT N-TUPLE TRAINING LOOP ---
    print("\n--- Stage 2: Direct N-tuple Training ---")
    results = {}
    
    for epoch in trange(config['train_epochs'], desc="N-tuple Training Progress"):
        net.train()
        epoch_losses = []
        epoch_ntuple_losses = []
        epoch_var_losses = []
        epoch_acc = []
        epoch_rank1 = []
        epoch_pos_sim = []
        epoch_neg_sim = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                optimizer.zero_grad()
                
                # FIXED: Forward pass using sample=False (deterministic posterior mean)
                all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
                all_embeddings = net(all_images, sample=False)  # Use posterior mean for stability
                
                # Unpack embeddings
                batch_size = anchor.shape[0]
                n_negatives = negatives.shape[1]
                anchor_embed = all_embeddings[0:batch_size]
                positive_embed = all_embeddings[batch_size : batch_size * 2]
                negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1)
                
                # FIXED: Add embedding variance regularization to prevent collapse
                all_embeddings_flat = torch.cat([anchor_embed, positive_embed, negative_embeds.view(-1, negative_embeds.shape[-1])], dim=0)
                var_loss = embedding_regularizer(all_embeddings_flat)
                
                # Compute N-tuple loss
                ntuple_loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                
                # FIXED: Combined loss with variance regularization
                total_loss = ntuple_loss + 0.1 * var_loss
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️ NaN/Inf detected at epoch {epoch+1}, skipping batch")
                    continue
                
                # Monitor for extremely large losses
                if total_loss.item() > 1000:
                    print(f"⚠️ Very large loss: {total_loss.item():.2f}")
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)  # Lower clip norm
                
                optimizer.step()
                
                # Compute metrics for monitoring (no re-normalization needed)
                with torch.no_grad():
                    sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
                    sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
                    max_sim_neg, _ = torch.max(sim_neg, dim=1)
                    correct_predictions = (sim_pos > max_sim_neg).sum().item()
                    pseudo_accuracy = correct_predictions / batch_size
                    
                    # Compute ReID metrics
                    rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                        anchor_embed, positive_embed, negative_embeds)
                
                # Track metrics
                epoch_losses.append(total_loss.item())
                epoch_ntuple_losses.append(ntuple_loss.item())
                epoch_var_losses.append(var_loss.item())
                epoch_acc.append(pseudo_accuracy)
                epoch_rank1.append(rank1_acc)
                epoch_pos_sim.append(mean_pos_sim)
                epoch_neg_sim.append(mean_neg_sim)
                
            except Exception as e:
                print(f"Warning: Training batch failed: {e}")
                continue
        
        # Step the learning rate scheduler
        scheduler.step()

        # --- 7. EVALUATION AND MONITORING ---
        if (epoch + 1) % config['test_interval'] == 0:
            print(f"\n--- Evaluating at Epoch {epoch+1} ---")
            
            # Calculate averages for this epoch
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            avg_ntuple_loss = np.mean(epoch_ntuple_losses) if epoch_ntuple_losses else float('inf')
            avg_var_loss = np.mean(epoch_var_losses) if epoch_var_losses else 0.0
            avg_acc = np.mean(epoch_acc) if epoch_acc else 0.0
            avg_rank1 = np.mean(epoch_rank1) if epoch_rank1 else 0.0
            avg_pos_sim = np.mean(epoch_pos_sim) if epoch_pos_sim else 0.0
            avg_neg_sim = np.mean(epoch_neg_sim) if epoch_neg_sim else 0.0
            
            # Evaluate on validation set
            net.eval()
            val_losses = []
            val_acc = []
            val_rank1 = []
            val_pos_sim = []
            val_neg_sim = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    try:
                        anchor, positive, negatives = batch
                        anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                        
                        # FIXED: Forward pass with sample=False for validation
                        all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
                        all_embeddings = net(all_images, sample=False)  # Deterministic for validation
                        
                        # Unpack embeddings
                        batch_size = anchor.shape[0]
                        n_negatives = negatives.shape[1]
                        anchor_embed = all_embeddings[0:batch_size]
                        positive_embed = all_embeddings[batch_size : batch_size * 2]
                        negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1)
                        
                        # Compute validation loss
                        val_loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                        
                        # Compute validation metrics (no re-normalization needed)
                        sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
                        sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
                        max_sim_neg, _ = torch.max(sim_neg, dim=1)
                        correct_predictions = (sim_pos > max_sim_neg).sum().item()
                        pseudo_accuracy = correct_predictions / batch_size
                        
                        # Compute ReID metrics for validation
                        rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                            anchor_embed, positive_embed, negative_embeds)
                        
                        val_losses.append(val_loss.item())
                        val_acc.append(pseudo_accuracy)
                        val_rank1.append(rank1_acc)
                        val_pos_sim.append(mean_pos_sim)
                        val_neg_sim.append(mean_neg_sim)
                        
                    except Exception as e:
                        print(f"Warning: Validation batch failed: {e}")
                        continue
            
            # Calculate validation metrics
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            avg_val_acc = np.mean(val_acc) if val_acc else 0.0
            avg_val_rank1 = np.mean(val_rank1) if val_rank1 else 0.0
            avg_val_pos_sim = np.mean(val_pos_sim) if val_pos_sim else 0.0
            avg_val_neg_sim = np.mean(val_neg_sim) if val_neg_sim else 0.0
            
            results[epoch+1] = {
                'train_loss': avg_loss,
                'train_ntuple_loss': avg_ntuple_loss,
                'train_var_loss': avg_var_loss,
                'train_accuracy': avg_acc,
                'train_rank1': avg_rank1,
                'train_pos_sim': avg_pos_sim,
                'train_neg_sim': avg_neg_sim,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_acc,
                'val_rank1': avg_val_rank1,
                'val_pos_sim': avg_val_pos_sim,
                'val_neg_sim': avg_val_neg_sim,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            print(f"  N-tuple Loss: {avg_ntuple_loss:.5f}")
            print(f"  Variance Loss: {avg_var_loss:.5f}")
            print(f"  Train Pseudo-Accuracy: {avg_acc:.4f}")
            print(f"  Train Rank-1: {avg_rank1:.4f}")
            print(f"  Train Pos Sim: {avg_pos_sim:.4f}")
            print(f"  Train Neg Sim: {avg_neg_sim:.4f}")
            print(f"  Val Loss: {avg_val_loss:.5f}")
            print(f"  Val Pseudo-Accuracy: {avg_val_acc:.4f}")
            print(f"  Val Rank-1: {avg_val_rank1:.4f}")
            print(f"  Val Pos Sim: {avg_val_pos_sim:.4f}")
            print(f"  Val Neg Sim: {avg_val_neg_sim:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Performance monitoring
            if avg_val_rank1 > 0.8:
                print("  ✅ Excellent ReID performance!")
            elif avg_val_rank1 > 0.6:
                print("  ✅ Good ReID performance!")
            elif avg_val_rank1 < 0.3 and epoch > 20:
                print("  ⚠️ Low Rank-1 accuracy - consider adjusting learning rate or N-tuple loss mode")
            
            # FIXED: Enhanced similarity analysis
            sim_gap = avg_val_pos_sim - avg_val_neg_sim
            if sim_gap > 0.3:
                print(f"  ✅ Good similarity separation: {sim_gap:.3f}")
            elif sim_gap > 0.1:
                print(f"  🔶 Moderate similarity separation: {sim_gap:.3f}")
            elif sim_gap > 0.01:
                print(f"  ⚠️ Poor similarity separation: {sim_gap:.3f}")
            else:
                print(f"  🚨 No similarity separation: {sim_gap:.6f} - embeddings may be collapsing!")
            
            net.train()  # Set back to training mode

    print("\n--- Training Finished ---")
    
    # --- FINAL MODEL EVALUATION ---
    print("\n--- Final Model Evaluation ---")
    
    # Test with different inference modes
    final_results = evaluate_trained_model(net, val_loader, ntuple_loss_fn, device)
    results['final_evaluation'] = final_results
    
    return results


def evaluate_trained_model(net, test_loader, ntuple_loss_fn, device):
    """
    FIXED: Comprehensive evaluation using sample=False and sample=True modes
    """
    print("Evaluating trained model with different inference modes...")
    
    results = {}
    
    # 1. Posterior Mean Evaluation (Deterministic)
    print("  - Posterior Mean (sample=False) Evaluation")
    net.eval()
    post_losses = []
    post_acc = []
    post_rank1 = []
    post_pos_sim = []
    post_neg_sim = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Posterior Mean", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                # Forward pass with posterior mean (sample=False)
                all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
                all_embeddings = net(all_images, sample=False)
                
                # Unpack embeddings
                batch_size = anchor.shape[0]
                n_negatives = negatives.shape[1]
                anchor_embed = all_embeddings[0:batch_size]
                positive_embed = all_embeddings[batch_size : batch_size * 2]
                negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1)
                
                # Compute loss
                loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                
                # Compute metrics (embeddings already normalized)
                sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
                sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
                max_sim_neg, _ = torch.max(sim_neg, dim=1)
                correct_predictions = (sim_pos > max_sim_neg).sum().item()
                accuracy = correct_predictions / batch_size
                
                # Compute ReID metrics
                rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                    anchor_embed, positive_embed, negative_embeds)
                
                post_losses.append(loss.item())
                post_acc.append(accuracy)
                post_rank1.append(rank1_acc)
                post_pos_sim.append(mean_pos_sim)
                post_neg_sim.append(mean_neg_sim)
                
            except Exception as e:
                print(f"Warning: Posterior mean batch failed: {e}")
                continue
    
    results['posterior_mean'] = {
        'loss': np.mean(post_losses) if post_losses else float('inf'),
        'accuracy': np.mean(post_acc) if post_acc else 0.0,
        'rank1': np.mean(post_rank1) if post_rank1 else 0.0,
        'pos_sim': np.mean(post_pos_sim) if post_pos_sim else 0.0,
        'neg_sim': np.mean(post_neg_sim) if post_neg_sim else 0.0
    }
    
    # 2. Stochastic Evaluation (sample=True)
    print("  - Stochastic (sample=True) Evaluation")
    stoch_losses = []
    stoch_acc = []
    stoch_rank1 = []
    stoch_pos_sim = []
    stoch_neg_sim = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Stochastic", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                # Forward pass with sampling (sample=True)
                all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
                all_embeddings = net(all_images, sample=True)
                
                # Unpack embeddings
                batch_size = anchor.shape[0]
                n_negatives = negatives.shape[1]
                anchor_embed = all_embeddings[0:batch_size]
                positive_embed = all_embeddings[batch_size : batch_size * 2]
                negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1)
                
                # Compute loss
                loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                
                # Compute metrics (embeddings already normalized)
                sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
                sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
                max_sim_neg, _ = torch.max(sim_neg, dim=1)
                correct_predictions = (sim_pos > max_sim_neg).sum().item()
                accuracy = correct_predictions / batch_size
                
                # Compute ReID metrics
                rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                    anchor_embed, positive_embed, negative_embeds)
                
                stoch_losses.append(loss.item())
                stoch_acc.append(accuracy)
                stoch_rank1.append(rank1_acc)
                stoch_pos_sim.append(mean_pos_sim)
                stoch_neg_sim.append(mean_neg_sim)
                
            except Exception as e:
                print(f"Warning: Stochastic batch failed: {e}")
                continue
    
    results['stochastic'] = {
        'loss': np.mean(stoch_losses) if stoch_losses else float('inf'),
        'accuracy': np.mean(stoch_acc) if stoch_acc else 0.0,
        'rank1': np.mean(stoch_rank1) if stoch_rank1 else 0.0,
        'pos_sim': np.mean(stoch_pos_sim) if stoch_pos_sim else 0.0,
        'neg_sim': np.mean(stoch_neg_sim) if stoch_neg_sim else 0.0
    }
    
    # Print final results
    print(f"\n--- Final Evaluation Results ---")
    print(f"Posterior Mean (sample=False) - Loss: {results['posterior_mean']['loss']:.5f}, Accuracy: {results['posterior_mean']['accuracy']:.4f}, Rank-1: {results['posterior_mean']['rank1']:.4f}")
    print(f"Stochastic (sample=True)      - Loss: {results['stochastic']['loss']:.5f}, Accuracy: {results['stochastic']['accuracy']:.4f}, Rank-1: {results['stochastic']['rank1']:.4f}")
    
    print(f"\n--- Similarity Analysis ---")
    for mode in ['posterior_mean', 'stochastic']:
        pos_sim = results[mode]['pos_sim']
        neg_sim = results[mode]['neg_sim']
        sim_gap = pos_sim - neg_sim
        print(f"{mode.capitalize():15} - Pos Sim: {pos_sim:.4f}, Neg Sim: {neg_sim:.4f}, Gap: {sim_gap:.4f}")
    
    return results


if __name__ == '__main__':
    # --- FIXED Configuration ---
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_list_path': 'ntuple-contrastive-learning/cuhk03/train.txt',
        'data_dir_path': 'ntuple-contrastive-learning/archive/images_labeled/',
        'val_perc': 0.2,
        'perc_prior': 0.2,
        'batch_size': 64,
        'learning_rate': 1e-6,  # FIXED: Much lower learning rate
        'learning_rate_prior': 3e-4,
        'weight_decay': 1e-3,  # FIXED: Higher weight decay
        'sigma_prior': 0.1,
        'train_epochs': 100,
        'prior_epochs': 15,
        'test_interval': 5,
        'N': 4,
        'samples_per_class': 4,
        'ntuple_mode': 'regular',
        'num_workers': 4,
        'embedding_dim': 256
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