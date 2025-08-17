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
        # ✅ CRITICAL FIX: Better KL handling
        original_kl = kl.clone() if torch.is_tensor(kl) else torch.tensor(kl, dtype=torch.float32)
        
        # Check for zero KL (major issue!)
        if original_kl < 1e-8:
            print(f"WARNING: KL divergence is {original_kl:.2e} - this will make bounds vacuous!")
            # Add small epsilon to prevent completely vacuous bounds
            kl = torch.clamp(original_kl, min=1e-6) 
        else:
            kl = original_kl
        
        kl = kl * self.kl_penalty
        delta = self.delta
        
        # Ensure empirical_risk is a tensor with gradients
        if not torch.is_tensor(empirical_risk):
            empirical_risk = torch.tensor(empirical_risk, dtype=torch.float32, 
                                        requires_grad=True, device=self.device)
        
        # ✅ FIX: Ensure all tensors are on the same device
        kl = kl.to(self.device)
        
        # ✅ FIX: More numerically stable log computation    
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
            # ✅ FIX: Simplified and more stable n-tuple bound
            # Use standard PAC-Bayes bound but adjusted for tuple structure
            if tuple_size is None:
                tuple_size = 3  # Default for triplet
                
            # ✅ More stable computation
            effective_train_size = torch.tensor(max(1, train_size), dtype=torch.float32, device=self.device)
            
            # Simple adjustment for tuple structure - don't overcomplicate
            tuple_adjustment = torch.log(torch.tensor(tuple_size, dtype=torch.float32, device=self.device))
            adjusted_log_term = log_term + tuple_adjustment
            
            term = (kl + adjusted_log_term) / (2 * effective_train_size)
            return empirical_risk + torch.sqrt(term)
        
        elif self.objective == 'nested_ntuple':
            # ✅ NEW: Combinatorial n-tuple bound with proper complexity modeling
            if tuple_size is None:
                tuple_size = 3  # Default for triplet
                
            # ✅ Combinatorial complexity calculation
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
            
            # ✅ Effective sample size adjustment for tuple structure
            effective_size = max(1, train_size // (tuple_size if tuple_size else 1))
            effective_size_tensor = torch.tensor(effective_size, dtype=torch.float32, device=self.device)
            
            # ✅ Combinatorial complexity penalty
            log_delta_term = torch.log(combinations / delta)
            
            # ✅ More principled bound computation
            kl_ratio = (kl + log_delta_term) / effective_size_tensor
            
            return empirical_risk + torch.sqrt(kl_ratio)

            
        
        else:
            # ✅ Remove overly complex objectives for now
            raise ValueError(f"Objective {self.objective} not supported. Use 'fquad', 'fclassic', 'ntuple', or 'nested_ntuple'")
        



    def mcsampling_ntuple(self, net, data_loader):
        net.eval()
        total_risk = 0.0
        total_accuracy = 0.0
        num_batches = len(data_loader)
        
        if num_batches == 0:
            return 1.0, 0.0

        # **KEY OPTIMIZATION**: Batch MC samples instead of sequential processing
        mc_batch_size = min(32, self.mc_samples)  # Adjust based on your memory
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MC Sampling", leave=False):
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
    
    def _vectorized_mc_batch(self, net, anchor_imgs, positive_imgs, negative_imgs, batch_size):
        """Vectorized MC sampling for a batch of samples."""
        risks = []
        accuracies = []
        
        # **MEMORY OPTIMIZATION**: Process in sub-batches if needed
        sub_batch_size = min(8, batch_size)  # Adjust based on your model size
        
        for i in range(0, batch_size, sub_batch_size):
            current_size = min(sub_batch_size, batch_size - i)
            
            # **EFFICIENT SAMPLING**: Replicate inputs for vectorized processing
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


    def train_obj(self, net, batch, train_size):
        tuple_size = self.get_tuple_size(batch)
        kl = net.compute_kl()
        empirical_risk, _ = self.compute_losses(net, *batch)
        train_obj = self.bound(empirical_risk, kl, train_size, tuple_size)
        return train_obj, empirical_risk, kl / train_size
    
    def compute_final_stats_risk(self, net, data_loader, train_size):
        """Compute final statistics for risk and KL divergence."""
        
        kl = net.compute_kl()
        
        # ✅ FIX: Add validation
        if torch.is_tensor(kl):
            kl_value = kl.item()
        else:
            kl_value = float(kl)
            
        if kl_value < 1e-8:
            print(f"CRITICAL: KL divergence is {kl_value:.2e} - bounds will be vacuous!")
            print("Check probabilistic layer initialization and sampling!")
        
        estimated_risk, pseudo_accuracy = self.mcsampling_ntuple(net, data_loader)
        
        # ✅ FIX: Add bounds checking
        estimated_risk = max(1e-6, min(0.999, estimated_risk))  # Avoid edge cases
        
        mc_error_term = np.sqrt(np.log(2 / self.delta_test) / self.mc_samples)
        
        # ✅ FIX: Better error handling for inv_kl
        try:
            empirical_risk_nt = inv_kl(estimated_risk, mc_error_term)
        except:
            print(f"inv_kl failed with estimated_risk={estimated_risk}, mc_error_term={mc_error_term}")
            empirical_risk_nt = estimated_risk + mc_error_term  # Fallback
        
        # ✅ Convert to tensor for bound computation
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


    def debug_bound_computation(self, net, batch, train_size):
        """Debug method to check bound computation"""
        print("\n=== Bound Computation Debug ===")
        
        tuple_size = self.get_tuple_size(batch)
        print(f"Tuple size: {tuple_size}")
        
        kl = net.compute_kl()
        kl_val = kl.item() if torch.is_tensor(kl) else kl
        print(f"KL divergence: {kl_val:.8f}")
        
        empirical_risk, accuracy = self.compute_losses(net, *batch)
        emp_risk_val = empirical_risk.item() if torch.is_tensor(empirical_risk) else empirical_risk
        print(f"Empirical risk: {emp_risk_val:.6f}")
        print(f"Accuracy: {accuracy:.6f}")
        
        train_obj = self.bound(empirical_risk, kl, train_size, tuple_size)
        train_obj_val = train_obj.item() if torch.is_tensor(train_obj) else train_obj
        print(f"Training objective: {train_obj_val:.6f}")
        
        # Check if bound is vacuous
        if train_obj_val >= 1.0:
            print("⚠️ WARNING: Bound is vacuous (≥ 1.0)")
        elif train_obj_val > 0.5:
            print("⚠️ WARNING: Bound is loose (> 0.5)")
        else:
            print("✓ Bound appears reasonable")
        
        return train_obj, empirical_risk, kl_val / train_size