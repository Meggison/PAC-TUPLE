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
        
        Args:
            batch: Tuple of (anchor, positive, negatives)
            
        Returns:
            int: The tuple size N = 1 (anchor) + 1 (positive) + num_negatives
        """
        _, _, negatives = batch  # ignore anchor and positive
        num_negatives = negatives.shape[1]  # N-2 in your notation
        return 1 + 1 + num_negatives  # anchor + positive + negatives = N
        
    def compute_losses(self, net, batch, ntuple_loss_fn, bounded=True):

        anchor, positive, negatives = batch
        # Ensure all tensors are on the same device
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negatives = negatives.to(self.device)
        
        # 2. Get embeddings and classification logits from the network
        # Your model's forward pass should be updated to return both
        # e.g., `return embeddings, logits`
        all_images = torch.cat([anchor, positive, negatives.view(-1, *anchor.shape[1:])], dim=0)
        all_embeddings = net(all_images)

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

        # 7. Compute the "pseudo-accuracy" as a secondary metric
        sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
        sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
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
        Exact bound following f_obj = R_S(q) + (1/(2*floor(n/m))) * [KL(q||p) + ln((C(n,m)+1)/delta)]
        """
        if self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            # Use binomial coefficient C(n, tuple_size) for variable tuple sizes
            combinations = math.comb(self.n_bound, tuple_size) if tuple_size <= self.n_bound else self.n_bound**tuple_size
            
            # Follow exact objective function from the paper
            penalty_term = (kl + np.log((combinations + 1) / self.delta)) / (2 * np.trunc(train_size / tuple_size))
            
            train_obj = empirical_risk + penalty_term  # Direct addition, no square root
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj
        if self.objective == 'fclassic':
            combinations = math.comb(self.n_bound, tuple_size) if tuple_size <= self.n_bound else self.n_bound**tuple_size
            
            penalty_term = (kl + np.log((combinations + 1) / self.delta)) / (2 * np.trunc(train_size / tuple_size))
            
            train_obj = empirical_risk + penalty_term  # No square root
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
                    loss, acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True)
                    
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
        empirical_risk, _, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=False)

        # 4. Compute the PAC-Bayes bound using the N-tuple risk and actual tuple size.
        bound = self.bound_exact(empirical_risk, kl, self.n_posterior, tuple_size)

        return bound, kl / self.n_posterior, empirical_risk
    
    def compute_final_stats_risk_ntuple(self, net, data_loader, ntuple_loss_fn):
        """
        Computes the final risk certificate for the N-tuple loss.

        Args:
            net (nn.Module): The probabilistic network to evaluate.
            data_loader (DataLoader): DataLoader for the dataset to certify.
            ntuple_loss_fn (callable): The external N-tuple loss function.

        Returns:
            tuple: A tuple containing:
                - final_risk (float): The final certified upper bound on the N-tuple risk.
                - kl_div (float): The KL divergence of the posterior from the prior.
                - empirical_risk_bound (float): The high-confidence empirical risk.
                - pseudo_accuracy (float): The average pseudo-accuracy over the dataset.
        """
        # 1. Extract tuple size from the first batch to determine the combinatorial term
        first_batch = next(iter(data_loader))
        tuple_size = self.get_tuple_size(first_batch)
        
        # 2. Calculate the total KL divergence of the final trained model.
        kl = net.compute_kl()

        # 3. Estimate the true empirical risk and pseudo-accuracy over the entire dataset
        #    by calling the adapted mcsampling_ntuple function.
        estimated_risk, pseudo_accuracy = self.mcsampling_ntuple(net, data_loader, ntuple_loss_fn)

        # 4. Invert the Chernoff bound to get a high-confidence empirical risk value.
        empirical_risk_bound = inv_kl(
            estimated_risk, np.log(2 / self.delta_test) / self.mc_samples
        )

        # 5. Compute the final PAC-Bayes risk certificate using the new empirical risk and actual tuple size.
        # Use binomial coefficient C(n, tuple_size) for variable tuple sizes
        combinations = math.comb(self.n_bound, tuple_size) if tuple_size <= self.n_bound else self.n_bound**tuple_size
        kl_term = kl + np.log((combinations + 1) / self.delta_test)
        # final_risk = inv_kl(empirical_risk_bound, kl_term / np.trunc(self.n_bound / tuple_size))
        # Updated final risk computation
        # CORRECTED CODE
        ks_final = kl_term / (2 * np.trunc(self.n_bound / tuple_size))
        final_risk = inv_kl(empirical_risk_bound, ks_final)

        # The function now returns a single certified risk for the N-tuple loss.
        return final_risk, kl.item()/self.n_bound, empirical_risk_bound, pseudo_accuracy

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
                loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True)
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
                loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True)
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
                    loss, pseudo_acc, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=True)
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

    

