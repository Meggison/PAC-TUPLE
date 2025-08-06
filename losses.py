import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

class NTupleLoss(nn.Module):
    """Implementatation of the N-tuple loss"""
    def __init__(self, embedding_dim=256, temp=1.0):
        super(NTupleLoss, self).__init__()

        self.log_s = nn.Parameter(torch.log(torch.tensor(1.0/temp)))

    
    def forward(self, anchor_embed, positive_embed, negative_embeds):
        """ Calculates the N-tuple loss."""

        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-2)
        
        # Calculate similarities
        sim_positive = cosine_similarity(anchor_norm, positive_norm)
        sim_negatives = cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)

        # concatenate similarities
        similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)

        # Scale logit by the learnable temperature
        logits = similarities * torch.exp(self.log_s)

        # Get target labels (All positives are at index 0)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Calculate loss using cross entropy
        loss = F.cross_entropy(logits, target)

        return loss
        