import numpy as np
import torch
from tqdm import tqdm, trange
from data import reid_data_prepare, DynamicNTupleDataset
from models import ResNet
from loss import NTupleLoss
import torch.optim as optim
import time
import torch.nn.functional as F


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
    sim_pos = torch.cosine_similarity(anchor_norm, positive_norm)  # (B,)
    sim_neg = torch.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)  # (B, N-2)
    
    # For each anchor, check if positive similarity is higher than all negative similarities
    max_neg_sim, _ = torch.max(sim_neg, dim=1)  # (B,)
    rank1_correct = (sim_pos > max_neg_sim).sum().item()
    rank1_acc = rank1_correct / batch_size
    
    # Additional metrics
    mean_pos_sim = sim_pos.mean().item()
    mean_neg_sim = sim_neg.mean().item()
    
    return rank1_acc, mean_pos_sim, mean_neg_sim


def run_ntuple_experiment(config):
    """
    Simple experiment runner using standard ResNet with N-tuple loss:
    - Train/Validation/Test split
    - Standard ResNet network training
    - N-tuple loss optimization
    """
    # --- 1. SETUP ---
    print("--- Starting Standard ResNet + N-tuple Experiment ---")
    print(f"Config: {config}")
    device = config['device']

    loader_kargs = {'num_workers': config.get('num_workers', 4), 'pin_memory': True} if 'cuda' in device else {}

    # --- 2. STANDARD DATA SPLITTING ---
    print("\n--- Preparing Data with Train/Val/Test Split ---")
    class_img_labels = reid_data_prepare(config['data_list_path'], config['data_dir_path'])
    all_class_ids = list(class_img_labels.keys())

    # Standard three-way split
    val_split = int(len(all_class_ids) * config['val_perc'])
    test_split = int(len(all_class_ids) * config.get('test_perc', 0.2))
    
    train_ids = all_class_ids[:-val_split-test_split]     # ~60% for training
    val_ids = all_class_ids[-val_split-test_split:-test_split]   # ~20% for validation
    test_ids = all_class_ids[-test_split:]                # ~20% for testing

    print(f"Data split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)} classes")

    # --- Create datasets for each split ---
    print("Initializing dynamic datasets...")
    train_dataset = DynamicNTupleDataset(class_img_labels, train_ids, N=config['N'], 
                                        samples_per_epoch_multiplier=config['samples_per_class'])
    val_dataset = DynamicNTupleDataset(class_img_labels, val_ids, N=config['N'], 
                                      samples_per_epoch_multiplier=config['samples_per_class'])
    test_dataset = DynamicNTupleDataset(class_img_labels, test_ids, N=config['N'], 
                                       samples_per_epoch_multiplier=config['samples_per_class'])

    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, **loader_kargs)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, **loader_kargs)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, **loader_kargs)
    
    print("Data preparation complete.")

    # --- 3. INITIALIZE STANDARD RESNET NETWORK ---
    print("\n--- Initializing Standard ResNet Network ---")
    net = ResNet().to(device)
    
    # Use Adam optimizer with reasonable learning rate
    optimizer = optim.Adam(net.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # Add learning rate scheduler for better training stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # --- 4. N-TUPLE LOSS SETUP ---
    print("\n--- Setting up N-tuple Loss ---")
    ntuple_loss_fn = NTupleLoss(mode=config['ntuple_mode'], embedding_dim=2048).to(device)
    print("N-tuple loss initialized.")

    # --- 5. MAIN TRAINING LOOP ---
    print("\n--- Starting N-tuple Training ---")
    results = {}
    
    for epoch in trange(config['train_epochs'], desc="N-tuple Training Progress"):
        net.train()
        epoch_losses = []
        epoch_acc = []
        epoch_rank1 = []
        epoch_pos_sim = []
        epoch_neg_sim = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass for all components - standard deterministic forward
                anchor_embed = net(anchor)
                positive_embed = net(positive)
                
                # Handle negatives
                batch_size, n_negatives, channels, height, width = negatives.shape
                negatives_flat = negatives.view(-1, channels, height, width)
                negative_embeds_flat = net(negatives_flat)
                negative_embeds = negative_embeds_flat.view(batch_size, n_negatives, -1)
                
                # Compute N-tuple loss
                ntuple_loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                
                # Check for numerical issues
                if torch.isnan(ntuple_loss) or torch.isinf(ntuple_loss):
                    print(f"⚠️ NaN/Inf detected at epoch {epoch+1}, skipping batch")
                    continue
                
                # Monitor for extremely large losses that might indicate instability
                if ntuple_loss.item() > 1000:
                    print(f"⚠️ Very large N-tuple loss: {ntuple_loss.item():.2f} - potential instability")
                
                ntuple_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Compute metrics for monitoring
                with torch.no_grad():
                    # Original pseudo-accuracy
                    anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
                    positive_norm = F.normalize(positive_embed, p=2, dim=1)
                    negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
                    
                    sim_pos = torch.cosine_similarity(anchor_norm, positive_norm)
                    sim_neg = torch.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)
                    max_sim_neg, _ = torch.max(sim_neg, dim=1)
                    correct_predictions = (sim_pos > max_sim_neg).sum().item()
                    pseudo_accuracy = correct_predictions / batch_size
                    
                    # Compute ReID metrics
                    rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                        anchor_embed, positive_embed, negative_embeds)
                
                # Track metrics
                epoch_losses.append(ntuple_loss.item())
                epoch_acc.append(pseudo_accuracy)
                epoch_rank1.append(rank1_acc)
                epoch_pos_sim.append(mean_pos_sim)
                epoch_neg_sim.append(mean_neg_sim)
                
            except Exception as e:
                print(f"Warning: Training batch failed: {e}")
                continue
        
        # Step the learning rate scheduler
        scheduler.step()

        # --- 6. EVALUATION AND MONITORING ---
        if (epoch + 1) % config['test_interval'] == 0:
            print(f"\n--- Evaluating at Epoch {epoch+1} ---")
            
            # Calculate averages for this epoch
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
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
            
            # For proper mAP and Rank-1 computation
            all_val_embeddings = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    try:
                        anchor, positive, negatives = batch
                        anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                        
                        # Forward pass - standard deterministic forward
                        anchor_embed = net(anchor)
                        positive_embed = net(positive)
                        
                        # Handle negatives
                        batch_size, n_negatives, channels, height, width = negatives.shape
                        negatives_flat = negatives.view(-1, channels, height, width)
                        negative_embeds_flat = net(negatives_flat)
                        negative_embeds = negative_embeds_flat.view(batch_size, n_negatives, -1)
                        
                        # Compute validation loss
                        val_loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                        
                        # Collect embeddings and create pseudo-labels for mAP/Rank-1 computation
                        # Since we're using N-tuples, we can create synthetic identity labels
                        for i in range(batch_size):
                            # Each anchor gets a unique identity (batch_idx * batch_size + sample_idx)
                            identity_id = len(all_val_labels) // (n_negatives + 2)  # Unique ID per N-tuple
                            
                            # Add anchor embedding with its identity
                            all_val_embeddings.append(anchor_embed[i].cpu())
                            all_val_labels.append(identity_id)
                            
                            # Add positive embedding with same identity as anchor
                            all_val_embeddings.append(positive_embed[i].cpu())
                            all_val_labels.append(identity_id)
                            
                            # Add negative embeddings with different identities
                            for j in range(n_negatives):
                                all_val_embeddings.append(negative_embeds[i, j].cpu())
                                all_val_labels.append(identity_id + 1000 + j)  # Different IDs for negatives
                        
                        # Debug: Check for degenerate embeddings
                        anchor_std = torch.std(anchor_embed, dim=1).mean().item()
                        pos_std = torch.std(positive_embed, dim=1).mean().item()
                        neg_std = torch.std(negative_embeds.view(-1, negative_embeds.size(-1)), dim=1).mean().item()
                        
                        # Compute validation metrics
                        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
                        positive_norm = F.normalize(positive_embed, p=2, dim=1)
                        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
                        
                        sim_pos = torch.cosine_similarity(anchor_norm, positive_norm)
                        sim_neg = torch.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)
                        max_sim_neg, _ = torch.max(sim_neg, dim=1)
                        correct_predictions = (sim_pos > max_sim_neg).sum().item()
                        pseudo_accuracy = correct_predictions / batch_size
                        
                        # Compute ReID metrics for validation
                        rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                            anchor_embed, positive_embed, negative_embeds)
                        
                        # Debug output for first validation batch
                        if len(val_losses) == 0:  # First batch
                            print(f"  Debug - Embedding stds: anchor={anchor_std:.4f}, pos={pos_std:.4f}, neg={neg_std:.4f}")
                            print(f"  Debug - Embedding norms: anchor={torch.norm(anchor_embed, dim=1).mean():.4f}")
                            print(f"  Debug - Raw similarities: pos_min={sim_pos.min():.4f}, pos_max={sim_pos.max():.4f}")
                            print(f"  Debug - Neg similarities: neg_min={sim_neg.min():.4f}, neg_max={sim_neg.max():.4f}")
                            
                            # Additional debugging: check if embeddings are actually different
                            anchor_sample = anchor_embed[0].cpu()
                            pos_sample = positive_embed[0].cpu()
                            neg_sample = negative_embeds[0, 0].cpu()
                            
                            print("  Debug - Embedding differences:")
                            print(f"    anchor-pos L2 dist: {torch.norm(anchor_sample - pos_sample):.6f}")
                            print(f"    anchor-neg L2 dist: {torch.norm(anchor_sample - neg_sample):.6f}")
                            print(f"    pos-neg L2 dist: {torch.norm(pos_sample - neg_sample):.6f}")
                            
                            # Check embedding ranges
                            print("  Debug - Embedding ranges:")
                            print(f"    anchor range: [{anchor_embed.min():.4f}, {anchor_embed.max():.4f}]")
                            print(f"    pos range: [{positive_embed.min():.4f}, {positive_embed.max():.4f}]")
                            print(f"    neg range: [{negative_embeds.min():.4f}, {negative_embeds.max():.4f}]")
                        
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
            
            # Compute proper mAP and Rank-1 metrics
            val_map = 0.0
            val_rank1_proper = 0.0
            if len(all_val_embeddings) > 10:  # Only compute if we have enough samples
                try:
                    all_embeddings_tensor = torch.stack(all_val_embeddings)
                    all_labels_tensor = torch.tensor(all_val_labels)
                    
                    # Split into query and gallery (use every other sample as query)
                    query_indices = list(range(0, len(all_embeddings_tensor), 2))
                    gallery_indices = list(range(1, len(all_embeddings_tensor), 2))
                    
                    if len(query_indices) > 0 and len(gallery_indices) > 0:
                        query_embeddings = all_embeddings_tensor[query_indices]
                        gallery_embeddings = all_embeddings_tensor[gallery_indices]
                        query_labels = all_labels_tensor[query_indices]
                        gallery_labels = all_labels_tensor[gallery_indices]
                        
                        # Compute distance matrix
                        distance_matrix = compute_distance_matrix(query_embeddings, gallery_embeddings)
                        
                        # Compute mAP and Rank-1
                        val_map, val_rank1_proper = compute_map_and_rank1(distance_matrix, query_labels, gallery_labels)
                        
                        print(f"  Proper mAP: {val_map:.4f}")
                        print(f"  Proper Rank-1: {val_rank1_proper:.4f}")
                except Exception as e:
                    print(f"Warning: Could not compute proper mAP/Rank-1: {e}")
                    val_map = 0.0
                    val_rank1_proper = 0.0
            
            results[epoch+1] = {
                'train_loss': avg_loss,
                'train_accuracy': avg_acc,
                'train_rank1': avg_rank1,
                'train_pos_sim': avg_pos_sim,
                'train_neg_sim': avg_neg_sim,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_acc,
                'val_rank1': avg_val_rank1,
                'val_pos_sim': avg_val_pos_sim,
                'val_neg_sim': avg_val_neg_sim,
                'val_map': val_map,
                'val_rank1_proper': val_rank1_proper,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            print(f"  N-tuple Loss: {avg_loss:.5f}")
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
            
            # Performance monitoring (no KL divergence checks)
            if avg_val_rank1 > 0.8:
                print("  ✅ Excellent ReID performance!")
            elif avg_val_rank1 > 0.6:
                print("  ✅ Good ReID performance!")
            elif avg_val_rank1 < 0.3 and epoch > 20:
                print("  ⚠️ Low Rank-1 accuracy - consider adjusting learning rate or N-tuple loss mode")
            
            # Similarity analysis
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
    
    # --- 7. FINAL TEST EVALUATION ---
    print("\n--- Final Test Evaluation ---")
    
    # Test evaluation
    final_results = evaluate_standard_model(net, test_loader, ntuple_loss_fn, device)
    results['final_test_evaluation'] = final_results
    
    return results


def evaluate_standard_model(net, test_loader, ntuple_loss_fn, device):
    """
    Comprehensive evaluation of the trained standard ResNet model with proper mAP and Rank-1 metrics.
    
    Args:
        net: Trained standard ResNet network
        test_loader: DataLoader for test data
        ntuple_loss_fn: N-tuple loss function
        device: Device to run on
        
    Returns:
        dict: Dictionary containing test results including mAP and Rank-1
    """
    print("Evaluating trained standard ResNet model...")
    
    net.eval()
    test_losses = []
    test_acc = []
    test_rank1 = []
    test_pos_sim = []
    test_neg_sim = []
    
    # For proper mAP and Rank-1 computation
    all_test_embeddings = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation", leave=False):
            try:
                anchor, positive, negatives = batch
                anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
                
                # Forward pass - standard deterministic forward
                anchor_embed = net(anchor)
                positive_embed = net(positive)
                
                # Handle negatives
                batch_size, n_negatives, channels, height, width = negatives.shape
                negatives_flat = negatives.view(-1, channels, height, width)
                negative_embeds_flat = net(negatives_flat)
                negative_embeds = negative_embeds_flat.view(batch_size, n_negatives, -1)
                
                # Collect embeddings and create pseudo-labels for mAP/Rank-1 computation
                for i in range(batch_size):
                    # Each anchor gets a unique identity
                    identity_id = len(all_test_labels) // (n_negatives + 2)  # Unique ID per N-tuple
                    
                    # Add anchor embedding with its identity
                    all_test_embeddings.append(anchor_embed[i].cpu())
                    all_test_labels.append(identity_id)
                    
                    # Add positive embedding with same identity as anchor
                    all_test_embeddings.append(positive_embed[i].cpu())
                    all_test_labels.append(identity_id)
                    
                    # Add negative embeddings with different identities
                    for j in range(n_negatives):
                        all_test_embeddings.append(negative_embeds[i, j].cpu())
                        all_test_labels.append(identity_id + 1000 + j)  # Different IDs for negatives
                
                # Compute loss
                loss = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)
                
                # Compute metrics
                anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
                positive_norm = F.normalize(positive_embed, p=2, dim=1)
                negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
                
                sim_pos = torch.cosine_similarity(anchor_norm, positive_norm)
                sim_neg = torch.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)
                max_sim_neg, _ = torch.max(sim_neg, dim=1)
                correct_predictions = (sim_pos > max_sim_neg).sum().item()
                accuracy = correct_predictions / batch_size
                
                # Compute ReID metrics
                rank1_acc, mean_pos_sim, mean_neg_sim = compute_simple_reid_metrics(
                    anchor_embed, positive_embed, negative_embeds)
                
                test_losses.append(loss.item())
                test_acc.append(accuracy)
                test_rank1.append(rank1_acc)
                test_pos_sim.append(mean_pos_sim)
                test_neg_sim.append(mean_neg_sim)
                
            except Exception as e:
                print(f"Warning: Test batch failed: {e}")
                continue
    
    # Compute proper mAP and Rank-1 metrics
    test_map = 0.0
    test_rank1_proper = 0.0
    if len(all_test_embeddings) > 10:  # Only compute if we have enough samples
        try:
            all_embeddings_tensor = torch.stack(all_test_embeddings)
            all_labels_tensor = torch.tensor(all_test_labels)
            
            # Split into query and gallery (use every other sample as query)
            query_indices = list(range(0, len(all_embeddings_tensor), 2))
            gallery_indices = list(range(1, len(all_embeddings_tensor), 2))
            
            if len(query_indices) > 0 and len(gallery_indices) > 0:
                query_embeddings = all_embeddings_tensor[query_indices]
                gallery_embeddings = all_embeddings_tensor[gallery_indices]
                query_labels = all_labels_tensor[query_indices]
                gallery_labels = all_labels_tensor[gallery_indices]
                
                # Compute distance matrix
                distance_matrix = compute_distance_matrix(query_embeddings, gallery_embeddings)
                
                # Compute mAP and Rank-1
                test_map, test_rank1_proper = compute_map_and_rank1(distance_matrix, query_labels, gallery_labels)
                
                print(f"Computed proper mAP and Rank-1 from {len(query_indices)} queries and {len(gallery_indices)} gallery items")
        except Exception as e:
            print(f"Warning: Could not compute proper mAP/Rank-1: {e}")
            test_map = 0.0
            test_rank1_proper = 0.0
    
    results = {
        'test_loss': np.mean(test_losses) if test_losses else float('inf'),
        'test_accuracy': np.mean(test_acc) if test_acc else 0.0,
        'test_rank1': np.mean(test_rank1) if test_rank1 else 0.0,
        'test_pos_sim': np.mean(test_pos_sim) if test_pos_sim else 0.0,
        'test_neg_sim': np.mean(test_neg_sim) if test_neg_sim else 0.0,
        'test_map': test_map,
        'test_rank1_proper': test_rank1_proper
    }
    
    # Print test results
    print("--- Final Test Results ---")
    print(f"Test Loss: {results['test_loss']:.5f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Rank-1 (Simple): {results['test_rank1']:.4f}")
    print(f"Test Pos Sim: {results['test_pos_sim']:.4f}")
    print(f"Test Neg Sim: {results['test_neg_sim']:.4f}")
    print(f"Test mAP (Proper): {results['test_map']:.4f}")
    print(f"Test Rank-1 (Proper): {results['test_rank1_proper']:.4f}")
    
    # Similarity analysis
    sim_gap = results['test_pos_sim'] - results['test_neg_sim']
    print(f"Similarity Gap: {sim_gap:.4f}")
    
    if results['test_rank1_proper'] > 0.8:
        print("  ✅ Excellent ReID performance!")
    elif results['test_rank1_proper'] > 0.6:
        print("  ✅ Good ReID performance!")
    elif results['test_rank1_proper'] > 0.4:
        print("  🔶 Moderate ReID performance")
    else:
        print("  ⚠️ Poor ReID performance")
    
    return results

if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_list_path': 'ntuple-contrastive-learning/cuhk03/train.txt',
        'data_dir_path': 'ntuple-contrastive-learning/archive/images_labeled/',
        'val_perc': 0.2,     # Percentage for validation 
        'test_perc': 0.2,    # Percentage for testing
        'batch_size': 32,    # Reduced batch size for better gradient updates
        'learning_rate': 3e-4,  # Increased learning rate for better convergence
        'weight_decay': 1e-5,   # Reduced weight decay to prevent over-regularization
        'train_epochs': 100,
        'test_interval': 5,
        'N': 3, # Number of samples in each N-tuple
        'samples_per_class': 4,
        'ntuple_mode': 'regular',  # 'regular' or 'mpn'
        'num_workers': 4  # Set to 0 for notebook compatibility
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