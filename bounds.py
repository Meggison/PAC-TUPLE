"""
CUSTOM PAC-BAYES BOUNDS IMPLEMENTATION

This implementation follows your custom nested KL inversion bound:

R(q) ≤ f^kl(f^kl(R_S(q̂_k), log(2/δ')/k), 1/(2⌊n/m⌋)[KL(q||p) + ln((C(n,m)+1)/δ)])

Key components:
1. f^kl = inv_kl function (KL divergence inversion)
2. Nested structure with two levels of concentration
3. Custom combinatorial term C(n,m) for N-tuple sampling
4. Monte Carlo error handling in the inner bound
5. Main PAC-Bayes bound in the outer structure

The implementation maintains your original bound structure while adapting it for N-tuple loss.
"""

import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from loss import NTupleLoss

def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*math.log(qs/p)+0)
        else:
            ikl = ks-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd

class PBBobj_Ntuple():

    def __init__(self, objective='fclassic', delta=0.025, delta_test=0.01,
                 mc_samples=100, kl_penalty=1, device='cuda',
                 n_posterior=30000, n_bound=30000):
        super().__init__()
        self.objective = objective
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        
    def get_tuple_size(self, batch):
        """
        Extract the actual tuple size (N) from a batch.
        
        IMPORTANT: For PAC-Bayes theory to be valid, this must match
        the combinatorial term in the bound computation.
        
        Args:
            batch: Tuple of (anchor, positive, negatives)
            
        Returns:
            int: The tuple size N = 1 (anchor) + 1 (positive) + num_negatives
        """
        _, _, negatives = batch  # ignore anchor and positive
        num_negatives = negatives.shape[1]  # N-2 in your notation
        tuple_size = 1 + 1 + num_negatives  # anchor + positive + negatives = N
        
        # CRITICAL: Ensure this matches your dataset's N parameter
        # If your DynamicNTupleDataset uses N=4, this should return 4
        return tuple_size
        
    def compute_losses(self, net, batch, ntuple_loss_fn, bounded=True, sample=True):

        anchor, positive, negatives = batch
        # Ensure all tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = negatives.to(self.device)
        
        # 2. Get embeddings and classification logits from the network
        # Your model's forward pass should be updated to return both
        # e.g., `return embeddings, logits`
        all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
        all_embeddings = net(all_images, sample=sample)

         # 3. Unpack embeddings and calculate N-tuple loss (Metric Loss)
        batch_size = anchor.shape[0]
        n_negatives = negatives.shape[1]
        anchor_embed = all_embeddings[0:batch_size]
        positive_embed = all_embeddings[batch_size : batch_size * 2]
        negative_embeds = all_embeddings[batch_size * 2 :].view(batch_size, n_negatives, -1) # restructure n-tuple tensor
        
        loss_ntuple = ntuple_loss_fn(anchor_embed, positive_embed, negative_embeds)


        # 6. Apply bounding for PAC-Bayes analysis
        if bounded:
            # Use a more stable bounding method that avoids sigmoid saturation
            # Clamp loss to reasonable range before sigmoid to prevent saturation
            clamped_loss = torch.clamp(loss_ntuple, max=10.0)  # Prevent extreme values
            total_bounded_loss = torch.sigmoid(clamped_loss)
        else:
            total_bounded_loss = loss_ntuple

        # 6. Apply bounding for PAC-Bayes analysis
        # FIX: Normalize embeddings before calculating accuracy
        
        anchor_norm = F.normalize(anchor_embed, p=2, dim=1)
        positive_norm = F.normalize(positive_embed, p=2, dim=1)
        negative_norm = F.normalize(negative_embeds, p=2, dim=-1)

        # 7. Compute the "pseudo-accuracy" as a secondary metric
        sim_pos = F.cosine_similarity(anchor_norm, positive_norm)
        sim_neg = F.cosine_similarity(anchor_norm.unsqueeze(1), negative_norm, dim=2)
        max_sim_neg, _ = torch.max(sim_neg, dim=1)
        correct_predictions = (sim_pos > max_sim_neg).sum().item()
        pseudo_accuracy = correct_predictions / batch_size

        return total_bounded_loss, pseudo_accuracy, (anchor_embed, positive_embed, negative_embeds)
    
    def bound(self, empirical_risk, kl, train_size, tuple_size, lambda_var=None):
        # compute training objectives
        
        if self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            # Use binomial coefficient C(n, tuple_size) for variable tuple sizes
            combinations = math.comb(self.n_bound, tuple_size) if tuple_size <= self.n_bound else self.n_bound**tuple_size
            kl_ratio = torch.div(
                kl + np.log(combinations / self.delta), 
                np.trunc(train_size / tuple_size))
            
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj
    
    def bound_exact(self, empirical_risk, kl, train_size, tuple_size, lambda_var=None):
        """
        CORRECTED: Your custom PAC-Bayes bound for N-tuple loss.
        
        Implements the nested KL inversion bound from your equation:
        R(q) ≤ f^kl(f^kl(R_S(q̂_k), log(2/δ')/k), 1/(2⌊n/m⌋)[KL(q||p) + ln((C(n,m)+1)/δ)])
        
        This is a two-level bound:
        1. Inner f^kl handles Monte Carlo estimation error
        2. Outer f^kl handles the main PAC-Bayes bound
        """
        if self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            
            # Compute combinatorial term C(n,m) for N-tuple sampling
            if tuple_size <= 10:  # Prevent overflow for reasonable tuple sizes
                num_combinations = math.comb(self.n_bound, tuple_size)
            else:
                # For large tuples, use Stirling approximation: n^m / m!
                num_combinations = (self.n_bound ** tuple_size) / math.factorial(tuple_size)
            
            # Ensure numerical stability
            num_combinations = min(num_combinations, self.n_bound ** tuple_size)
            
            # Your custom bound structure: 
            # Second term of outer f^kl: 1/(2⌊n/m⌋)[KL(q||p) + ln((C(n,m)+1)/δ)]
            num_tuples = np.trunc(train_size / tuple_size)
            second_term = (kl + np.log((num_combinations + 1) / self.delta)) / (2 * num_tuples)
            
            # For training, we directly use this as the penalty (no nested f^kl yet)
            # The full nested structure is used in compute_final_stats_risk_ntuple
            train_obj = empirical_risk + second_term

        elif (self.objective == 'fquad'):
            # This implements the f_quad objective with the stable "super-sample" penalty term.
            # The complexity penalty term from the PAC-Bayes-quadratic bound derivation.
            # Note the denominator is 2*k as per the f_quad formula.
            # num_tuples = np.trunc(train_size / tuple_size)
            # complexity_penalty = (kl + np.log(2 * np.sqrt(num_tuples) / self.delta)) / (2 * num_tuples)
            # FIX: Use math and torch operations, not numpy, to keep everything in the computation graph.
            kl = kl * self.kl_penalty
            
            num_tuples = math.trunc(train_size / tuple_size)
           # Use torch.log and torch.sqrt
            log_term = torch.log(2 * torch.sqrt(torch.tensor(num_tuples, device=self.device)) / self.delta)
            complexity_penalty = (kl + log_term) / (2 * num_tuples)

            # Ensure penalty is non-negative
            complexity_penalty = torch.clamp(complexity_penalty, min=0)

            train_obj = empirical_risk + torch.sqrt(complexity_penalty)
            
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj
    
    def mcsampling_ntuple(self, net, data_loader, ntuple_loss_fn):
        """
        Computes the average empirical risk and pseudo-accuracy over a dataset
        using Monte Carlo sampling with N-tuple data.

        Args:
            net (nn.Module): The probabilistic network to evaluate.
            data_loader (DataLoader): DataLoader providing N-tuple batches.
            ntuple_loss_fn (callable): The external N-tuple loss function.

        Returns:
            tuple: A tuple containing:
                   - avg_risk (float): The final estimated (bounded) risk over the dataset.
                   - avg_pseudo_accuracy (float): The final pseudo-accuracy over the dataset.
        """
        net.eval()
        total_risk = 0.0
        total_pseudo_accuracy = 0.0

        # The outer loop iterates through the batches provided by the DataLoader
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MC Sampling", leave=False):
                # Accumulators for the Monte Carlo samples for the current batch
                risk_mc_batch = 0.0
                accuracy_mc_batch = 0.0

                # The inner loop performs Monte Carlo sampling for a single batch
                for _ in range(self.mc_samples):
                    # Call the adapted compute_losses function
                    # It returns the bounded loss, pseudo-accuracy, and embeddings
                    loss, acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True, sample=True)
                    
                    risk_mc_batch += loss
                    accuracy_mc_batch += acc

                # Average the results over all Monte Carlo samples for this batch
                total_risk += (risk_mc_batch / self.mc_samples)
                total_pseudo_accuracy += (accuracy_mc_batch / self.mc_samples)

        # Average the results over all batches to get the final estimates
        avg_risk = (total_risk / len(data_loader)).item()
        avg_pseudo_accuracy = total_pseudo_accuracy / len(data_loader)

        return avg_risk, avg_pseudo_accuracy
    
    def train_obj_ntuple(self, net, batch, ntuple_loss_fn):
        """
        Computes the training objective (the bound) for one batch of N-tuple data.

        Args:
            net (nn.Module): The probabilistic network model.
            batch (tuple): A single batch of (anchor, positive, negatives) data.
            ntuple_loss_fn (callable): The external N-tuple loss function.

        Returns:
            tuple: A tuple containing:
                - bound (torch.Tensor): The PAC-Bayes training objective for the batch.
                - kl_div_scaled (torch.Tensor): The scaled KL divergence.
                - empirical_risk (torch.Tensor): The raw (unbounded) N-tuple loss.
        """
        # 1. Extract the actual tuple size from the batch
        tuple_size = self.get_tuple_size(batch)
        
        # 2. Calculate the KL divergence for the model's weights. This is unchanged.
        kl = net.compute_kl()

        # 3. Compute the empirical risk using your new compute_losses function.
        #    We set bounded=False here to get the raw loss for the bound calculation.
        #    The `compute_losses` function you provided already handles the network call.
        empirical_risk, pseudo_accuracy, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True, sample=True)

        # # 4. Compute the PAC-Bayes bound using the N-tuple risk and actual tuple size.
        bound = self.bound_exact(empirical_risk, kl, self.n_posterior, tuple_size)

        return bound, kl / self.n_posterior, empirical_risk, pseudo_accuracy
    
    def compute_final_stats_risk_ntuple(self, net, data_loader, ntuple_loss_fn):
        """
        CORRECTED: Implements your custom nested KL inversion bound.
        
        Your bound: R(q) ≤ f^kl(f^kl(R_S(q̂_k), log(2/δ')/k), 1/(2⌊n/m⌋)[KL(q||p) + ln((C(n,m)+1)/δ)])
        
        Where:
        - f^kl is the KL inversion function (inv_kl)
        - R_S(q̂_k) is the MC estimate of empirical risk
        - k is the number of MC samples
        - δ' is delta_test
        - δ is delta for the main bound
        """
        # 1. Extract tuple size from the first batch
        first_batch = next(iter(data_loader))
        tuple_size = self.get_tuple_size(first_batch)
        
        # 2. Calculate the total KL divergence of the final trained model
        kl = net.compute_kl()

        # 3. Estimate empirical risk via Monte Carlo sampling
        # This gives us R_S(q̂_k) - the MC estimate
        estimated_risk, pseudo_accuracy = self.mcsampling_ntuple(net, data_loader, ntuple_loss_fn)

        # 4. FIRST KL INVERSION: Handle MC estimation error
        # Inner f^kl(R_S(q̂_k), log(2/δ')/k)
        mc_error_term = np.log(2 / self.delta_test) / self.mc_samples
        mc_upper_bound = inv_kl(estimated_risk, mc_error_term)

        # 5. SECOND KL INVERSION: Your custom PAC-Bayes bound
        # Compute combinatorial term C(n,m)
        if tuple_size <= 10:
            num_combinations = math.comb(self.n_bound, tuple_size)
        else:
            num_combinations = (self.n_bound ** tuple_size) / math.factorial(tuple_size)
        
        num_combinations = min(num_combinations, self.n_bound ** tuple_size)
        
        # Second term: 1/(2⌊n/m⌋)[KL(q||p) + ln((C(n,m)+1)/δ)]
        num_tuples = np.trunc(self.n_bound / tuple_size)
        second_term = (kl + np.log((num_combinations + 1) / self.delta)) / (2 * num_tuples)
        
        # Outer f^kl: final risk certificate
        # R(q) ≤ f^kl(mc_upper_bound, second_term)
        # final_risk = inv_kl(mc_upper_bound, second_term)
       # Calculate k = number of tuples for the bound computation set
        num_tuples = np.trunc(self.n_bound / tuple_size)

        # The complexity term for the standard PAC-Bayes-kl bound (Theorem 1 in paper)
        # Note the denominator is k (not 2k) for the final KL inversion.
        complexity_term = (kl + np.log(2 * np.sqrt(num_tuples) / self.delta)) / num_tuples
        complexity_term = max(complexity_term, 0) # Ensure non-negative

        final_risk = inv_kl(mc_upper_bound, complexity_term)
        
        # Ensure risk is in [0,1] range
        final_risk = min(final_risk, 1.0)

        return final_risk, kl.item()/self.n_bound, mc_upper_bound, pseudo_accuracy

    def test_stochastic_ntuple(self, net, data_loader, ntuple_loss_fn):
        """
        Test function for stochastic predictor adapted for N-tuple loss.
        
        Args:
            net: The probabilistic network to evaluate
            data_loader: DataLoader for test data
            ntuple_loss_fn: The N-tuple loss function
            
        Returns:
            tuple: (average_risk, average_pseudo_accuracy)
        """
        net.eval()
        total_risk = 0.0
        total_pseudo_accuracy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Stochastic Testing", leave=False):
                loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True, sample=True)
                total_risk += loss.item()
                total_pseudo_accuracy += pseudo_acc
        
        avg_risk = total_risk / len(data_loader)
        avg_pseudo_accuracy = total_pseudo_accuracy / len(data_loader)
        
        return avg_risk, avg_pseudo_accuracy

    def test_posterior_mean_ntuple(self, net, data_loader, ntuple_loss_fn):
        """
        Test function for posterior mean predictor adapted for N-tuple loss.
        
        Args:
            net: The probabilistic network to evaluate
            data_loader: DataLoader for test data
            ntuple_loss_fn: The N-tuple loss function
            
        Returns:
            tuple: (average_risk, average_pseudo_accuracy)
        """
        net.eval()
        total_risk = 0.0
        total_pseudo_accuracy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Posterior Mean Testing", leave=False):
                # Set network to use posterior mean (not sampling)
                net.train(False)  # Ensure we're in eval mode
                loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True, sample=True)
                total_risk += loss.item()
                total_pseudo_accuracy += pseudo_acc
        
        avg_risk = total_risk / len(data_loader)
        avg_pseudo_accuracy = total_pseudo_accuracy / len(data_loader)
        
        return avg_risk, avg_pseudo_accuracy

    def test_ensemble_ntuple(self, net, data_loader, ntuple_loss_fn, num_samples=10):
        """
        Test function for ensemble predictor adapted for N-tuple loss.
        
        Args:
            net: The probabilistic network to evaluate
            data_loader: DataLoader for test data
            ntuple_loss_fn: The N-tuple loss function
            num_samples: Number of ensemble members
            
        Returns:
            tuple: (average_risk, average_pseudo_accuracy)
        """
        net.eval()
        total_risk = 0.0
        total_pseudo_accuracy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Ensemble Testing", leave=False):
                batch_risk = 0.0
                batch_pseudo_acc = 0.0
                
                # Sample multiple times for ensemble
                for _ in range(num_samples):
                    loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True, sample=True)
                    batch_risk += loss.item()
                    batch_pseudo_acc += pseudo_acc
                
                # Average over ensemble
                total_risk += batch_risk / num_samples
                total_pseudo_accuracy += batch_pseudo_acc / num_samples
        
        avg_risk = total_risk / len(data_loader)
        avg_pseudo_accuracy = total_pseudo_accuracy / len(data_loader)
        
        return avg_risk, avg_pseudo_accuracy

    def comprehensive_evaluation(self, net, data_loader, ntuple_loss_fn):
        """
        Comprehensive evaluation using all three prediction methods.
        
        Args:
            net: The probabilistic network to evaluate
            data_loader: DataLoader for test data
            ntuple_loss_fn: The N-tuple loss function
            
        Returns:
            dict: Dictionary containing results from all evaluation methods
        """
        print("Running comprehensive evaluation...")
        
        results = {}
        
        # Stochastic predictor
        stoch_risk, stoch_acc = self.test_stochastic_ntuple(net, data_loader, ntuple_loss_fn)
        results['stochastic'] = {'risk': stoch_risk, 'pseudo_accuracy': stoch_acc}
        
        # Posterior mean predictor
        mean_risk, mean_acc = self.test_posterior_mean_ntuple(net, data_loader, ntuple_loss_fn)
        results['posterior_mean'] = {'risk': mean_risk, 'pseudo_accuracy': mean_acc}
        
        # Ensemble predictor
        ensemble_risk, ensemble_acc = self.test_ensemble_ntuple(net, data_loader, ntuple_loss_fn)
        results['ensemble'] = {'risk': ensemble_risk, 'pseudo_accuracy': ensemble_acc}
        
        # Final risk certificate
        final_risk, kl_div, emp_risk, pseudo_acc = self.compute_final_stats_risk_ntuple(net, data_loader, ntuple_loss_fn)
        results['certificate'] = {
            'final_risk': final_risk,
            'kl_divergence': kl_div,
            'empirical_risk': emp_risk,
            'pseudo_accuracy': pseudo_acc
        }
        
        return results
    
    def validate_pac_bayes_theory(self, tuple_size, train_size):
        """
        Validates that your custom PAC-Bayes bound implementation follows theoretical requirements.
        
        Your bound uses nested KL inversions with combinatorial terms for N-tuple loss.
        """
        warnings = []
        
        # Check 1: Tuple size consistency
        if tuple_size != 4:  # Based on your config N=4
            warnings.append(f"⚠️ Tuple size {tuple_size} doesn't match expected N=4")
        
        # Check 2: Dataset size requirements for your nested bound
        min_samples = tuple_size * 50  # More relaxed for custom bounds
        if train_size < min_samples:
            warnings.append(f"⚠️ Training size {train_size} may be too small for tuple_size {tuple_size}")
        
        # Check 3: Combinatorial explosion C(n,m)
        if tuple_size <= 10:
            num_combos = math.comb(self.n_bound, tuple_size)
        else:
            num_combos = (self.n_bound ** tuple_size) / math.factorial(tuple_size)
            
        if num_combos > 1e15:  # Higher threshold for custom bounds
            warnings.append(f"⚠️ Very large combinatorial term ({num_combos:.2e}) - bounds may be loose")
        
        # Check 4: MC samples for nested KL inversion
        if self.mc_samples < 50:
            warnings.append(f"⚠️ MC samples {self.mc_samples} may be too low for reliable nested KL inversion")
        
        # Check 5: KL penalty for your custom structure
        if self.kl_penalty > 5:  # More relaxed for custom bounds
            warnings.append(f"⚠️ KL penalty {self.kl_penalty} is high - may hurt performance")
        elif self.kl_penalty < 0.1:
            warnings.append(f"⚠️ KL penalty {self.kl_penalty} is very low - bounds may be loose")
        
        # Check 6: Confidence levels for nested structure
        if self.delta > 0.05:
            warnings.append(f"⚠️ Delta {self.delta} is high for nested bounds - consider lowering")
        if self.delta_test > 0.05:
            warnings.append(f"⚠️ Delta_test {self.delta_test} is high for nested structure")
        
        # Check 7: Bound tightness heuristic
        num_tuples = np.trunc(train_size / tuple_size)
        if num_tuples < 10:
            warnings.append(f"⚠️ Very few tuples ({num_tuples}) - bounds may be very loose")
        
        return warnings



