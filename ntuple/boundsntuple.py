import math
import numpy as np
import torch
from tqdm import tqdm
from losses import NTupleLoss, GeneralizedTripletLoss, TripletLoss
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

def inv_kl(qs, ks):
    """Numerically invert the binary KL for PAC-Bayes-kl bound."""
    izq = qs
    dch = 1 - 1e-10
    qd = 0
    while((dch - izq) / dch >= 1e-5):
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p) + 0)
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd

class PBBobj_NTuple():
    """PAC-Bayes bound class for probabilistic networks with N-tuple/metric losses."""
    def __init__(self, objective='fquad', pmin=1e-4, delta=0.025,
                 delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda',
                 n_posterior=30000, n_bound=30000):
        self.objective = objective
        self.pmin = pmin
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.device = device
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        self.loss_fn = GeneralizedTripletLoss()  # Default loss function, can be changed later

    def get_tuple_size(self, batch):
        _, _, negatives = batch
        num_negatives = negatives.shape[1]
        return 2 + num_negatives

    def compute_empirical_risk(self, anchor_embed, positive_embed, negative_embeds):
        empirical_risk = self.loss_fn(anchor_embed, positive_embed, negative_embeds)
        
        # ✅ FIX: Remove pmin scaling for metric learning losses
        # Metric learning losses are already in [0,1] range typically
        # Only apply pmin scaling if using cross-entropy surrogate loss
        if hasattr(self.loss_fn, 'use_pmin_scaling') and self.loss_fn.use_pmin_scaling:
            if self.pmin is not None:
                empirical_risk = empirical_risk / np.log(1. / self.pmin)
        
        # ✅ Ensure empirical risk is properly bounded
        empirical_risk = torch.clamp(empirical_risk, 0.0, 1.0)
        return empirical_risk
    
    def compute_accuracy(self, anchor_embed, positive_embed, negative_embeds):
        """Compute accuracy based on the embeddings."""
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)

        # Calculate similarities
        sim_positive = cosine_similarity(anchor_norm, positive_norm)
        sim_negatives = cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)
        
        # Concatenate similarities
        similarities = torch.cat((sim_positive.unsqueeze(1), sim_negatives), dim=1)
        predictions = torch.argmax(similarities, dim=1)
        
        # Calculate accuracy
        correct_predictions = (predictions == 0).float()
        accuracy = correct_predictions.mean().item()

        return accuracy

    def compute_losses(self, net, anchor_data, positive_data, negative_data, clamping=True):
        """
        Modified compute_losses that handles both images and embeddings
        """
        anchor_data = anchor_data.to(self.device)
        positive_data = positive_data.to(self.device) 
        negative_data = negative_data.to(self.device)
        
        # Check if inputs are images (5D) or embeddings (2D/3D)
        if len(anchor_data.shape) == 4:  # Images: [B, C, H, W]
            # Original image processing path
            anchor_embed = net(anchor_data, sample=True)
            positive_embed = net(positive_data, sample=True)
            
            B, N_neg, C, H, W = negative_data.shape
            negative_data_flat = negative_data.view(B * N_neg, C, H, W)
            negative_embed_flat = net(negative_data_flat, sample=True)
            negative_embed = negative_embed_flat.view(B, N_neg, -1)
            
        elif len(anchor_data.shape) == 2:  # Embeddings: [B, embedding_dim]
            # Direct embedding processing path
            if hasattr(net, 'forward'):
                # Pass through network if it exists (for consistency with sampling)
                anchor_embed = net(anchor_data, sample=True)
                positive_embed = net(positive_data, sample=True) 
                
                # Handle negative embeddings: [B, N_neg, embedding_dim]
                B, N_neg, embedding_dim = negative_data.shape
                negative_data_flat = negative_data.view(B * N_neg, embedding_dim)
                negative_embed_flat = net(negative_data_flat, sample=True)
                negative_embed = negative_embed_flat.view(B, N_neg, -1)
            else:
                # Direct use of embeddings (when using MockEmbeddingNetwork)
                anchor_embed = anchor_data
                positive_embed = positive_data
                negative_embed = negative_data
        else:
            raise ValueError(f"Unsupported input shape. Expected 4D (images) or 2D (embeddings), got {len(anchor_data.shape)}D")

        empirical_risk = self.compute_empirical_risk(anchor_embed, positive_embed, negative_embed)
        if clamping:
            empirical_risk = torch.clamp(empirical_risk, 0.0, 1.0)
        
        accuracy = self.compute_accuracy(anchor_embed, positive_embed, negative_embed)

        return empirical_risk, accuracy

    def bound(self, empirical_risk, kl, train_size, tuple_size=None, lambda_var=None):
        kl = kl * self.kl_penalty
        delta = self.delta
        
        # Ensure empirical_risk is a tensor with gradients
        if not torch.is_tensor(empirical_risk):
            empirical_risk = torch.tensor(empirical_risk, dtype=torch.float32, requires_grad=True)
        
        # ✅ Use torch operations to preserve gradients
        log_term = torch.log(torch.tensor(
            2 * torch.sqrt(torch.tensor(train_size, dtype=torch.float32)) / delta,
            dtype=empirical_risk.dtype, device=empirical_risk.device))
    
        if self.objective == 'fquad':
            term = (kl + log_term) / (2 * train_size)
            part1 = torch.sqrt(empirical_risk + term)
            part2 = torch.sqrt(term)
            return torch.pow(part1 + part2, 2)
        
        elif self.objective == 'fclassic':
            term = (kl + log_term) / (2 * train_size)
            return empirical_risk + torch.sqrt(term)
        
        elif self.objective == 'ntuple':
            if tuple_size is not None and tuple_size <= self.n_bound:
                log_combinations = (torch.lgamma(torch.tensor(self.n_bound + 1, dtype=torch.float32)) - 
                                  torch.lgamma(torch.tensor(tuple_size + 1, dtype=torch.float32)) - 
                                  torch.lgamma(torch.tensor(self.n_bound - tuple_size + 1, dtype=torch.float32)))
                combinations = torch.exp(log_combinations)
            else:
                combinations = torch.tensor(float(self.n_bound ** tuple_size if tuple_size else 1), 
                                          dtype=torch.float32)
            
            effective_size = max(1, train_size // (tuple_size if tuple_size else 1))
            log_delta_term = torch.log(combinations / delta)
            kl_ratio = (kl + log_delta_term) / effective_size
            
            return empirical_risk + torch.sqrt(kl_ratio)
        
        elif self.objective == 'nested_ntuple':
            if tuple_size is None:
                tuple_size = 3
                
            # Use torch operations
            tuple_count = max(1, int(train_size // tuple_size))
            
            if tuple_size <= self.n_bound:
                log_combinations = (torch.lgamma(torch.tensor(self.n_bound + 1, dtype=torch.float32)) - 
                                  torch.lgamma(torch.tensor(tuple_size + 1, dtype=torch.float32)) - 
                                  torch.lgamma(torch.tensor(self.n_bound - tuple_size + 1, dtype=torch.float32)))
                log_combinations_plus_one = torch.log(torch.exp(log_combinations) + 1)
            else:
                log_combinations_plus_one = (tuple_size * torch.log(torch.tensor(self.n_bound, dtype=torch.float32)) + 
                                           torch.log(torch.tensor(2.0)))
            
            delta_prime = delta / 2
            delta_outer = delta / 2
            
            # Create a differentiable approximation of inv_kl or use a simpler bound
            # Since inv_kl is complex, let's use a simpler nested bound:
            inner_confidence_term = torch.log(torch.tensor(2 / delta_prime)) / tuple_size
            inner_bound = empirical_risk + torch.sqrt(inner_confidence_term)
            outer_confidence_term = (kl + log_combinations_plus_one / delta_outer) / (2 * tuple_count)
            final_bound = inner_bound + torch.sqrt(outer_confidence_term)
            
            return final_bound
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


    def mcsampling_ntuple(self, net, data_loader):
        net.eval()
        total_risk = 0.0
        total_accuracy = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MC Sampling", leave=False):
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_imgs = anchor_imgs.to(self.device)
                positive_imgs = positive_imgs.to(self.device)
                negative_imgs = negative_imgs.to(self.device)
                risk_mc_batch = 0.0
                accuracy_mc_batch = 0.0
                for _ in range(self.mc_samples):
                    loss, accuracy = self.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs)
                    risk_mc_batch += loss.item()
                    accuracy_mc_batch += accuracy

                # Average the risk and accuracy over the MC samples
                risk_mc_batch /= self.mc_samples
                accuracy_mc_batch /= self.mc_samples
                total_risk += risk_mc_batch
                total_accuracy += accuracy_mc_batch

        # Average over all batches
        avg_risk = total_risk / num_batches
        avg_pseudo_accuracy = total_accuracy / num_batches
        return avg_risk, avg_pseudo_accuracy

    def train_obj(self, net, batch, train_size):
        tuple_size = self.get_tuple_size(batch)
        kl = net.compute_kl()
        empirical_risk, _ = self.compute_losses(net, *batch)
        train_obj = self.bound(empirical_risk, kl, train_size, tuple_size)
        return train_obj, empirical_risk, kl / train_size
    
    def compute_final_stats_risk(self, net, data_loader, train_size):
        """Compute final statistics for risk and KL divergence."""

        kl = net.compute_kl()
        estimated_risk, pseudo_accuracy = self.mcsampling_ntuple(net, data_loader)

        mc_error_term = np.sqrt(np.log(2 / self.delta_test) / self.mc_samples)
        empirical_risk_nt = inv_kl(estimated_risk, mc_error_term)

        train_obj = self.bound(empirical_risk_nt, kl,self.n_posterior)

        risk_nt = inv_kl(empirical_risk_nt, (kl + np.log((2 *
                                                             np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
    
        return train_obj.item(), risk_nt, empirical_risk_nt, pseudo_accuracy, kl.item() / train_size
    