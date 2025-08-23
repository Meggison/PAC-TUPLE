import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


class TripletLoss(nn.Module):
    """Standard triplet loss for person re-ID"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, anchor_embed, positive_embed, negative_embeds):
        # Use first negative for triplet loss
        negative_embed = negative_embeds[:, 0, :]  # Take first negative
        return self.triplet_loss(anchor_embed, positive_embed, negative_embed)


class GeneralizedTripletLoss(nn.Module):
    """
    Generalized Triplet Loss that works with N negatives (triplet, quartet, quintet, etc.)
    Maintains the margin-based principle of TripletLoss while using all negatives.
    """
    def __init__(self, margin=0.3, strategy='all', reduction='mean'):
        super().__init__()
        self.margin = margin
        self.strategy = strategy  # 'hardest', 'all', 'adaptive'
        self.reduction = reduction
        
    def forward(self, anchor_embed, positive_embed, negative_embeds):
        """
        Args:
            anchor_embed: [B, D] - anchor embeddings
            positive_embed: [B, D] - positive embeddings  
            negative_embeds: [B, N, D] - N negative embeddings per anchor
        """
        num_negatives = negative_embeds.size(1)
        
        # Normalize embeddings (crucial for good performance)
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
        
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_norm, positive_norm, p=2)
        
        # Compute distances to all negatives: [B, N]
        neg_dists = torch.norm(
            anchor_norm.unsqueeze(1) - negative_norm, p=2, dim=-1
        )
        
        if self.strategy == 'hardest':
            # Use hardest (closest) negative for each anchor
            neg_dist = neg_dists.min(dim=1)[0]  # [B]
            losses = F.relu(pos_dist - neg_dist + self.margin)
            
        elif self.strategy == 'all':
            # Use all negatives (creates B*N triplet losses)
            pos_dist_expanded = pos_dist.unsqueeze(1).expand(-1, num_negatives)  # [B, N]
            losses = F.relu(pos_dist_expanded - neg_dists + self.margin)  # [B, N]
            losses = losses.mean(dim=1)  # Average over negatives: [B]
            
        elif self.strategy == 'adaptive':
            # Adaptive combination: focus on harder negatives
            neg_weights = F.softmax(-neg_dists / 0.1, dim=1)  # Closer negatives get higher weight
            weighted_neg_dist = (neg_weights * neg_dists).sum(dim=1)  # [B]
            losses = F.relu(pos_dist - weighted_neg_dist + self.margin)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class MultiNegativeTripletLoss(nn.Module):
    """
    Alternative: Multiple negative ranking loss (like your original N-tuple but fixed)
    Uses ranking principle with proper temperature scaling
    """
    def __init__(self, temperature=0.1, margin=0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, anchor_embed, positive_embed, negative_embeds):
        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)
        
        # Compute similarities (cosine similarity via dot product after normalization)
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)  # [B]
        neg_sims = torch.sum(
            anchor_norm.unsqueeze(1) * negative_norm, dim=-1
        )  # [B, N]
        
        # Apply temperature scaling
        pos_logits = pos_sim / self.temperature
        neg_logits = neg_sims / self.temperature
        
        # Combine positive and negative logits
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1+N]
        
        # Target is always index 0 (positive)
        targets = torch.zeros(anchor_embed.size(0), dtype=torch.long, device=anchor_embed.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(all_logits, targets)
        
        return loss


# Bugged , do not use
# class NTupleLoss(nn.Module):
#     """Implementatation of the N-tuple loss"""
#     def __init__(self, embedding_dim=256, temp=1.0):
#         super(NTupleLoss, self).__init__()

#         self.log_s = nn.Parameter(torch.log(torch.tensor(1.0/temp)))

    
#     def forward(self, anchor_embed, positive_embed, negative_embeds):
#         """ Calculates the N-tuple loss."""

#         # Normalize embeddings
#         anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
#         positive_norm = F.normalize(positive_embed, p=2, dim=1)
#         negative_norm = F.normalize(negative_embeds, p=2, dim=-2)
        
#         # Calculate similarities
#         sim_positive = cosine_similarity(anchor_norm, positive_norm)
#         sim_negatives = cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)

#         # concatenate similarities
#         similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)

#         # Scale logit by the learnable temperature
#         logits = similarities * torch.exp(self.log_s)

#         # Get target labels (All positives are at index 0)
#         target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

#         # Calculate loss using cross entropy
#         loss = F.cross_entropy(logits, target)

#         return loss