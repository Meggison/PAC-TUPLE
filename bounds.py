import math
import numpy as np
import torch
from tqdm import tqdm
from loss import NTupleLoss
import torch.nn.functional as F

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
        
    def compute_losses(self, net, batch, ntuple_loss_fn, bounded=True):

        anchor, positive, negatives = batch
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
            # Use a robust bounding method like sigmoid since we're combining losses
            total_bounded_loss = torch.sigmoid(loss_ntuple)
        else:
            total_bounded_loss = loss_ntuple

        # 7. Compute the "pseudo-accuracy" as a secondary metric
        with torch.no_grad():
            sim_pos = F.cosine_similarity(anchor_embed, positive_embed)
            sim_neg = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_embeds, dim=2)
            max_sim_neg, _ = torch.max(sim_neg, dim=1)
            correct_predictions = (sim_pos > max_sim_neg).sum().item()
            pseudo_accuracy = correct_predictions / batch_size

        return total_bounded_loss, pseudo_accuracy, (anchor_embed, positive_embed, negative_embeds)
    
    def bound(self, empirical_risk, kl, train_size, lambda_var=None):
        # compute training objectives
        
        if self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(
                kl + np.log((self.n_bound*(self.n_bound-1)/2)/self.delta), 2*np.trunc(train_size/2))
            
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
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
        # 1. Calculate the KL divergence for the model's weights. This is unchanged.
        kl = net.compute_kl()

        # 2. Compute the empirical risk using your new compute_losses function.
        #    We set bounded=False here to get the raw loss for the bound calculation.
        #    The `compute_losses` function you provided already handles the network call.
        empirical_risk, _, _ = self.compute_losses(net, batch, ntuple_loss_fn, bounded=False)

        # 3. Compute the PAC-Bayes bound using the N-tuple risk.
        bound = self.bound(empirical_risk, kl, self.n_posterior)

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
        # 1. Calculate the total KL divergence of the final trained model.
        kl = net.compute_kl()

        # 2. Estimate the true empirical risk and pseudo-accuracy over the entire dataset
        #    by calling the adapted mcsampling_ntuple function.
        estimated_risk, pseudo_accuracy = self.mcsampling_ntuple(net, data_loader, ntuple_loss_fn)

        # 3. Invert the Chernoff bound to get a high-confidence empirical risk value.
        empirical_risk_bound = inv_kl(
            estimated_risk, np.log(2 / self.delta_test) / self.mc_samples
        )

        # 4. Compute the final PAC-Bayes risk certificate using the new empirical risk.
        kl_term = kl + np.log((self.n_bound * (self.n_bound - 1) / 2 + 1) / self.delta_test)
        final_risk = inv_kl(empirical_risk_bound, kl_term / np.trunc(self.n_bound / 2))

        # The function now returns a single certified risk for the N-tuple loss.
        return final_risk, kl.item()/self.n_bound, empirical_risk_bound, pseudo_accuracy

    

