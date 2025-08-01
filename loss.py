import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

class MetaLearner(nn.Module):
    """
    Implements the meta-learner subnet phi(·) from Equation (7) of the paper.
    It takes instance features and maps them to refined reference nodes.

    Args:
        embedding_dim (int): The dimension of the input feature embeddings (d).
        reduction_ratio (int): The ratio for dimension reduction in the bottleneck layer.
    """
    def __init__(self, embedding_dim=2048, reduction_ratio=8):
        super(MetaLearner, self).__init__()
        bottleneck_dim = embedding_dim // reduction_ratio
        
        self.mapper = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            # The paper does not specify an activation, but ReLU is a common choice.
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, embedding_dim)
        )

    def forward(self, x):
        return self.mapper(x)


class NTupleLoss(nn.Module):
    """
    Implementation of N-tuple and Meta Prototypical N-tuple (MPN-tuple) loss.

    Args:
        mode (str): The loss mode. Must be one of 'regular' or 'mpn'.
                    - 'regular': Standard N-tuple loss using instance features directly.
                    - 'mpn': Meta Prototypical N-tuple loss using a meta-learner.
        embedding_dim (int): The dimension of the feature embeddings.
                             Required only if mode is 'mpn'.
        initial_temp (float): The initial temperature (tau) for scaling similarities.
    """
    def __init__(self, mode='mpn', embedding_dim=2048, initial_temp=1.0):
        super(NTupleLoss, self).__init__()
        
        if mode not in ['regular', 'mpn']:
            raise ValueError("Mode must be either 'regular' or 'mpn'")
        self.mode = mode

        # The paper makes the temperature a learnable parameter by learning s = 1/tau
        # We will do the same for flexibility.
        # Initialize with tau=initial_temp, so s=1/initial_temp
        self.log_s = nn.Parameter(torch.log(torch.tensor(1.0 / initial_temp)))

        if self.mode == 'mpn':
            self.meta_learner = MetaLearner(embedding_dim=embedding_dim)

    def forward(self, anchor_embed, positive_embed, negative_embeds):
        """
        Calculates the N-tuple loss.

        Args:
            anchor_embed (torch.Tensor): Embeddings of the anchor samples.
                                         Shape: (batch_size, embedding_dim)
            positive_embed (torch.Tensor): Embeddings of the positive samples.
                                          Shape: (batch_size, embedding_dim)
            negative_embeds (torch.Tensor): Embeddings of the negative samples.
                                           Shape: (batch_size, N-2, embedding_dim)

        Returns:
            torch.Tensor: The calculated N-tuple loss for the batch.
        """
        # Get the reference nodes for positive and negative samples
        if self.mode == 'mpn':
            # For MPN loss, pass positives and negatives through the meta-learner
            # to get the reference nodes (prototypes).
            # The paper averages multiple instances for a prototype; here we assume
            # the provided single positive/negative is the basis for its prototype.
            positive_ref = self.meta_learner(positive_embed)
            
            # Reshape negatives to pass through the linear layers of the meta-learner
            batch_size, n_negatives, embed_dim = negative_embeds.shape
            negatives_flat = negative_embeds.view(-1, embed_dim)
            negative_ref_flat = self.meta_learner(negatives_flat)
            negative_ref = negative_ref_flat.view(batch_size, n_negatives, embed_dim)
        
        else: # 'regular' mode
            # For regular N-tuple loss, the instance embeddings are the reference nodes.
            positive_ref = positive_embed
            negative_ref = negative_embeds

            # L2 normalise the embeddings to ensure cosine similarity is used
            # anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
            # positive_norm = F.normalize(positive_ref, p=2, dim=1)
            # negative_norm = F.normalize(negative_ref, p=2, dim=-1)

            anchor_norm = anchor_embed
            positive_norm = positive_ref
            negative_norm = negative_ref
        
        # --- Calculate similarities ---
        sim_positive = cosine_similarity(anchor_norm, positive_norm)
        
        # To calculate similarity between anchor and all negatives, we need to unsqueeze
        # the anchor to enable broadcasting across the N-2 dimension.
        # anchor_embed shape: (B, D) -> (B, 1, D)
        # negative_ref shape: (B, N-2, D)
        sim_negatives = cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)
        
        # --- Formulate as a classification problem ---
        # The goal is to classify the anchor as belonging to the positive reference
        # over all negative references. This can be solved with CrossEntropyLoss.
        
        # The logits are the scaled similarities.
        # Concatenate the positive similarity with all negative similarities.
        # Shape: (B, 1+ (N-2)) -> (B, N-1)
        logits = torch.cat([sim_positive.unsqueeze(1), sim_negatives], dim=1)
        
        # Scale logits by the learned temperature parameter s = 1/tau
        logits *= torch.exp(self.log_s)
        
        # The target label for every sample is 0, because the positive class
        # is always at index 0 of our logits tensor.
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # Calculate the cross-entropy loss, which is equivalent to the
        # log-sum-exp formulation in the paper's N-tuple loss equations.
        loss = F.cross_entropy(logits, targets)
        
        return loss
