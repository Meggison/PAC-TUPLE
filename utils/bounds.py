import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import GeneralizedTripletLoss
from utils.helpers import inv_kl


# ===== PAC-BAYES BOUNDS =====
class PACBayesBound():
    """PAC-Bayes bound class for probabilistic networks with N-tuple/metric losses."""
    def __init__(self, objective='ntuple', pmin=1e-4, delta=0.025, loss_fn=GeneralizedTripletLoss(),
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
        self.loss_fn = loss_fn if loss_fn is not None else GeneralizedTripletLoss()

    def get_tuple_size(self, batch):
        _, _, negatives = batch
        num_negatives = negatives.shape[1]
        return 2 + num_negatives
    
    def compute_empirical_risk(self, anchor_embed, positive_embed, negative_embeds):
        empirical_risk = self.loss_fn(anchor_embed, positive_embed, negative_embeds)
        
        # Only apply pmin scaling if using cross-entropy surrogate loss
        if hasattr(self.loss_fn, 'use_pmin_scaling') and self.loss_fn.use_pmin_scaling:
            if self.pmin is not None:
                empirical_risk = empirical_risk / np.log(1. / self.pmin)
        
        # Ensure empirical risk is properly bounded
        empirical_risk = torch.clamp(empirical_risk, 0.0, 1.0)
        return empirical_risk
    
    def compute_accuracy(self, anchor_embed, positive_embed, negative_embeds):
        """Compute accuracy based on the embeddings."""
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)

        # Calculate similarities
        sim_positive = F.cosine_similarity(anchor_norm, positive_norm)
        sim_negatives = F.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=-1)
        
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
        # Better KL handling
        original_kl = kl.clone() if torch.is_tensor(kl) else torch.tensor(kl, dtype=torch.float32)
        
        # Check for zero KL
        if original_kl < 1e-8:
            print(f"WARNING: KL divergence is {original_kl:.2e} - this will make bounds vacuous!")
            # Add small epsilon to prevent completely vacuous bounds
            kl = torch.clamp(original_kl, min=1e-6) 
        else:
            kl = original_kl
        
        # Apply KL penalty
        kl = kl * self.kl_penalty
        delta = self.delta
        
        # Ensure empirical_risk is a tensor with gradients
        if not torch.is_tensor(empirical_risk):
            empirical_risk = torch.tensor(empirical_risk, dtype=torch.float32, 
                                        requires_grad=True, device=self.device)
        
        # Ensure all tensors are on the same device
        kl = kl.to(self.device)
        
        # Numerically stable log computation    
        sqrt_train_size = torch.sqrt(torch.tensor(train_size, dtype=torch.float32, device=self.device))
        log_term = torch.log(2 * sqrt_train_size / delta)

        if self.objective == 'fquad':
            term = (kl + log_term) / (2 * train_size)
            part1 = torch.sqrt(empirical_risk + term)
            part2 = torch.sqrt(term)
            return torch.pow(part1 + part2, 2)
        
        elif self.objective == 'fclassic':
            term = (kl + log_term) / (2 * train_size)
            return empirical_risk + torch.sqrt(term)
        
        elif self.objective == 'ntuple':
            # Standard PAC-Bayes bound but adjusted for tuple structure
            if tuple_size is None:
                tuple_size = 3  # Default for triplet
                
            effective_train_size = torch.tensor(max(1, train_size), dtype=torch.float32, device=self.device)
            tuple_adjustment = torch.log(torch.tensor(tuple_size, dtype=torch.float32, device=self.device))
            adjusted_log_term = log_term + tuple_adjustment
            
            term = (kl + adjusted_log_term) / (2 * effective_train_size)
            return empirical_risk + torch.sqrt(term)
        
        elif self.objective == 'nested_ntuple':
            # Nested N-tuple bound with combinatorial complexity
            if tuple_size is None:
                tuple_size = 3  # Default for triplet
                
            if tuple_size is not None and tuple_size <= self.n_bound:
                # Use binomial coefficient C(n_bound, tuple_size) via log-gamma functions
                log_combinations = (torch.lgamma(torch.tensor(self.n_bound + 1, dtype=torch.float32, device=self.device)) - 
                                torch.lgamma(torch.tensor(tuple_size + 1, dtype=torch.float32, device=self.device)) - 
                                torch.lgamma(torch.tensor(self.n_bound - tuple_size + 1, dtype=torch.float32, device=self.device)))
                combinations = torch.exp(log_combinations)
            else:
                # Fallback to n_bound^tuple_size for large tuple_size
                combinations = torch.tensor(float(self.n_bound ** tuple_size if tuple_size else 1), 
                                        dtype=torch.float32, device=self.device)
            
            # Sample size adjustment for tuple structure
            effective_size = max(1, train_size // (tuple_size if tuple_size else 1))
            effective_size_tensor = torch.tensor(effective_size, dtype=torch.float32, device=self.device)
            
            # Combinatorial complexity penalty
            log_delta_term = torch.log(combinations / delta)
            kl_ratio = (kl + log_delta_term) / effective_size_tensor
            return empirical_risk + torch.sqrt(kl_ratio)
        
        elif self.objective == 'theory_ntuple':
            if tuple_size is None:
                raise ValueError("theory_ntuple requires tuple_size (m).")
            # Ensure tensors on correct device
            if not torch.is_tensor(empirical_risk):
                empirical_risk = torch.tensor(empirical_risk, dtype=torch.float32, requires_grad=True, device=self.device)
            if not torch.is_tensor(kl):
                kl = torch.tensor(float(kl), dtype=torch.float32, device=self.device)
            kl = kl * self.kl_penalty

            # Use the same n in both C(n,m) and floor(n/m). We take the bound set size for n.
            # You can switch to train_size if that is how your derivation defines n.
            n_for_bound = float(self.n_bound)

            # The exact PAC term from your theory:
            pac_term = self._theory_pac_term(kl, n_for_bound=n_for_bound, tuple_size=float(tuple_size), delta=self.delta)

            # f_obj = empirical_risk + pac_term^{1/2}? NO — your printed f_obj is linear in the term,
            # not a sqrt. The training-time objective from your formula is:
            #   R_S^{CE}(q) + ( KL + ln((C(n,m)+1)/δ) ) / (2 floor(n/m))
            # i.e., add the term directly (no sqrt).
            return empirical_risk + pac_term
        
        else:
            raise ValueError(f"Objective {self.objective} not supported")
    
    def mcsampling(self, net, data_loader):
        net.eval()
        total_risk = 0.0
        total_accuracy = 0.0
        num_batches = len(data_loader)
        
        if num_batches == 0:
            return 1.0, 0.0

        # Batch MC samples
        mc_batch_size = min(32, self.mc_samples)  # Adjust based on your memory
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Monte Carlo Sampling", leave=False):
                # Handle both 3-tuple and 4-tuple batch formats
                if len(batch) == 4:
                    anchor_imgs, positive_imgs, negative_imgs, _ = batch
                else:
                    anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_imgs = anchor_imgs.to(self.device)
                positive_imgs = positive_imgs.to(self.device)
                negative_imgs = negative_imgs.to(self.device)

                risk_accumulator = []
                accuracy_accumulator = []
                
                # Process MC samples in batches
                for start_idx in range(0, self.mc_samples, mc_batch_size):
                    end_idx = min(start_idx + mc_batch_size, self.mc_samples)
                    current_batch_size = end_idx - start_idx
                    
                    try:
                        # **VECTORIZED MC SAMPLING**
                        batch_risks, batch_accuracies = self._vectorized_mc_batch(
                            net, anchor_imgs, positive_imgs, negative_imgs, current_batch_size
                        )
                        
                        risk_accumulator.extend(batch_risks)
                        accuracy_accumulator.extend(batch_accuracies)
                        
                        # Clear cache periodically to prevent memory buildup
                        if (start_idx + mc_batch_size) % (mc_batch_size * 4) == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"OOM at batch {start_idx}, reducing mc_batch_size")
                            torch.cuda.empty_cache()
                            mc_batch_size = max(1, mc_batch_size // 2)
                            continue
                        else:
                            raise e

                # Aggregate results
                if risk_accumulator:
                    total_risk += sum(risk_accumulator) / len(risk_accumulator)
                    total_accuracy += sum(accuracy_accumulator) / len(accuracy_accumulator)
                else:
                    total_risk += 1.0
                    total_accuracy += 0.0

        return total_risk / num_batches, total_accuracy / num_batches


    def train_obj(self, net, batch, train_size):
        tuple_size = self.get_tuple_size(batch)
        kl = net.compute_kl()
        empirical_risk, _ = self.compute_losses(net, *batch)
        train_obj = self.bound(empirical_risk, kl, train_size, tuple_size)
        return train_obj, empirical_risk, kl / train_size
    

    def compute_final_stats_risk(self, net, data_loader, train_size):
        """Compute final statistics for risk and KL divergence."""
        
        kl = net.compute_kl()
        
        if torch.is_tensor(kl):
            kl_value = kl.item()
        else:
            kl_value = float(kl)
            
        if kl_value < 1e-8:
            print(f"CRITICAL: KL divergence is {kl_value:.2e} - bounds will be vacuous!")
            print("Check probabilistic layer initialization and sampling!")
        
        estimated_risk, pseudo_accuracy = self.mcsampling(net, data_loader)
        
        estimated_risk = max(1e-6, min(0.999, estimated_risk))  # Avoid edge cases
        mc_error_term = np.sqrt(np.log(2 / self.delta_test) / self.mc_samples)
        
        try:
            empirical_risk_nt = inv_kl(estimated_risk, mc_error_term)
        except:
            print(f"inv_kl failed with estimated_risk={estimated_risk}, mc_error_term={mc_error_term}")
            empirical_risk_nt = estimated_risk + mc_error_term  # Fallback
        
        # Convert to tensor for bound computation
        empirical_risk_tensor = torch.tensor(empirical_risk_nt, dtype=torch.float32, device=self.device)
        train_obj = self.bound(empirical_risk_tensor, kl, self.n_posterior)
        
        # Final risk certificate
        bound_term = (kl_value + np.log((2 * np.sqrt(self.n_bound))/self.delta_test))/self.n_bound
        try:
            risk_nt = inv_kl(empirical_risk_nt, bound_term)
        except:
            print(f"Final inv_kl failed, using fallback")
            risk_nt = empirical_risk_nt + bound_term  # Fallback
        
        return train_obj.item(), risk_nt, empirical_risk_nt, pseudo_accuracy, kl_value / train_size
    
    def _vectorized_mc_batch(self, net, anchor_imgs, positive_imgs, negative_imgs, batch_size):
        """Vectorized MC sampling for a batch of samples."""
        risks = []
        accuracies = []
        
        # **MEMORY OPTIMIZATION**: Process in sub-batches if needed
        sub_batch_size = min(8, batch_size)  # Adjust based on your model size
        
        for i in range(0, batch_size, sub_batch_size):
            current_size = min(sub_batch_size, batch_size - i)
            
            # Replicate inputs for vectorized processing
            anchor_batch = anchor_imgs.repeat(current_size, 1, 1, 1)
            positive_batch = positive_imgs.repeat(current_size, 1, 1, 1)
            negative_batch = negative_imgs.repeat(current_size, 1, 1, 1, 1).view(-1, *negative_imgs.shape[2:])
            
            # Forward pass with sampling enabled
            anchor_embeds = net(anchor_batch, sample=True)
            positive_embeds = net(positive_batch, sample=True)
            
            # Process negatives efficiently
            B, N_neg = negative_imgs.shape[:2]
            neg_flat = negative_batch.view(current_size * B * N_neg, *negative_imgs.shape[2:])
            neg_embeds_flat = net(neg_flat, sample=True)
            neg_embeds = neg_embeds_flat.view(current_size, B, N_neg, -1)
            
            # Compute losses and accuracies for the batch
            for j in range(current_size):
                risk = self.compute_empirical_risk(
                    anchor_embeds[j*B:(j+1)*B], 
                    positive_embeds[j*B:(j+1)*B], 
                    neg_embeds[j]
                )
                accuracy = self.compute_accuracy(
                    anchor_embeds[j*B:(j+1)*B], 
                    positive_embeds[j*B:(j+1)*B], 
                    neg_embeds[j]
                )
                
                risks.append(risk.item() if torch.is_tensor(risk) else risk)
                accuracies.append(accuracy)
        
        return risks, accuracies


    # Helper functions for theory_ntuple bound
    def _floor_div_t(self, a, b):
        # differentiable-friendly floor(n/m); we only need a positive scalar tensor
        t_a = torch.tensor(float(a), dtype=torch.float32, device=self.device)
        t_b = torch.tensor(float(b), dtype=torch.float32, device=self.device)
        # use floor on the Python value; clamp to >=1 to avoid div-by-zero
        val = max(1, int(float(a) // float(b)))
        return torch.tensor(float(val), dtype=torch.float32, device=self.device)

    def _log_C_plus_one(self, n_val, m_val):
        # log( C(n, m) + 1 )
        n_t = torch.tensor(float(n_val), dtype=torch.float32, device=self.device)
        m_t = torch.tensor(float(m_val), dtype=torch.float32, device=self.device)
        # Use lgamma on float32; we assume n_val, m_val are reasonable (>= m_val >= 1)
        log_C = (torch.lgamma(n_t + 1.0) - torch.lgamma(m_t + 1.0) - torch.lgamma(n_t - m_t + 1.0))
        # log(exp(log_C)+1)
        return torch.log(torch.exp(log_C) + 1.0)

    def _theory_pac_term(self, kl, n_for_bound, tuple_size, delta):
        """
        Returns the theoretical PAC term:
        [ KL + ln((C(n,m)+1)/δ) ] / (2 * floor(n/m))
        using the same n in both C(n,m) and floor(n/m).
        """
        # log(C(n,m)+1)
        logC1 = self._log_C_plus_one(n_for_bound, tuple_size)
        # ln(1/δ)
        log_inv_delta = torch.log(torch.tensor(1.0/float(delta), dtype=torch.float32, device=self.device))
        # numerator: KL + log(C(n,m)+1) + ln(1/δ)
        if not torch.is_tensor(kl):
            kl = torch.tensor(float(kl), dtype=torch.float32, device=self.device)
        num = kl + (logC1 + log_inv_delta)
        # denominator: 2 * floor(n/m)
        denom = 2.0 * self._floor_div_t(n_for_bound, tuple_size)
        return num / denom
