import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor, l2_normalise: bool = False) -> torch.Tensor:
    """Compute pairwise cosine distance between two sets of vectors."""
    if l2_normalise:
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
    
    sims = a @ b.T
    dists = 1.0 - sims
    return dists


@ torch.no_grad()
def compute_map(query_embeddings: torch.Tensor, query_labels: torch.Tensor,
                gallery_embeddings: torch.Tensor, gallery_labels: torch.Tensor,
                distance_fn=pairwise_cosine_distance, remove_self_match: bool = True) -> float:
    """Compute mean Average Precision (mAP) for ReID."""
    if query_embeddings.is_cuda:
        query_embeddings = query_embeddings.detach().cpu()
    if query_labels.is_cuda:
        query_labels = query_labels.detach().cpu()
    if gallery_embeddings.is_cuda:
        gallery_embeddings = gallery_embeddings.detach().cpu()
    if gallery_labels.is_cuda:
        gallery_labels = gallery_labels.detach().cpu()

    dists = pairwise_cosine_distance(query_embeddings, gallery_embeddings, l2_normalise=False).numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()

    num_queries = query_embeddings.shape[0]
    average_precisions = []

    for i in range(num_queries):
        label = query_labels[i]
        distances = dists[i]
        order = np.argsort(distances)
        ranked_labels = gallery_labels[order]

        if remove_self_match:
            if distances[order[0]] == 0.0 and ranked_labels[0] == label:
                order = order[1:]
                ranked_labels = gallery_labels[order]
            
        relevant = (ranked_labels == label).astype(np.int32)
        num_relevant = relevant.sum()
        if num_relevant == 0:
            average_precisions.append(0.0)
            continue

        cumulative_relevant = np.cumsum(relevant)
        ranks = np.arange(1, len(relevant) + 1)
        precision_at_k = cumulative_relevant / ranks
        average_precision = (precision_at_k * relevant).sum() / num_relevant
        average_precisions.append(float(average_precision))

    return float(np.mean(average_precisions)) if len(average_precisions) > 0 else 0.0


@ torch.no_grad()
def compute_rank1_accuracy(query_embeddings: torch.Tensor, query_labels: torch.Tensor,
                            gallery_embeddings: torch.Tensor, gallery_labels: torch.Tensor,
                            remove_self_match: bool = True) -> float:
    """ Compute Rank-1 accuracy for ReID."""
    if query_embeddings.is_cuda:
        query_embeddings = query_embeddings.detach().cpu()
    if query_labels.is_cuda:
        query_labels = query_labels.detach().cpu()
    if gallery_embeddings.is_cuda:
        gallery_embeddings = gallery_embeddings.detach().cpu()
    if gallery_labels.is_cuda:
        gallery_labels = gallery_labels.detach().cpu()

    dists = pairwise_cosine_distance(query_embeddings, gallery_embeddings, l2_normalise=False).numpy()
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()

    num_queries = query_embeddings.shape[0]
    correct = 0

    for i in range(num_queries):
        distances = dists[i]
        order = np.argsort(distances)
        top_idx = order[0]

        if remove_self_match:
            if distances[order[0]] == 0.0 and gallery_labels[order[0]] == query_labels[i]:
                if len(order) > 1:
                    top_idx = order[1]
        
        correct += (gallery_labels[top_idx] == query_labels[i]).astype(np.int32)

    return float(correct) / num_queries if num_queries > 0 else 0.0


@torch.no_grad()
def compute_accuracy(anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor,
                     negative_embeddings: torch.Tensor) -> float:
    """Compute accuracy for triplet-based ReID."""

    sim_positive = F.cosine_similarity(anchor_embeddings, positive_embeddings)
    sim_negatives = F.cosine_similarity(anchor_embeddings.unsqueeze(1), negative_embeddings, dim=-1)
    
    similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)
    predictions = torch.argmax(similarities, dim=1)
    accuracy = (predictions == 0).float().mean().item()
    
    return accuracy
