import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from losses import NTupleLoss

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


class PBBobj_NTuple():
    """Class for computing the PBBobj N-tuple loss."""
    def __init__(self, objective='fquad', pmin=1e-4, delta=0.025,
    delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda',
      n_posterior=30000, n_bound=30000):
        super(PBBobj_NTuple, self).__init__()
        self.objective = objective
        self.pmin = pmin
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.device = device
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        self.loss_fn = NTupleLoss()

    def get_tuple_size(self, batch):
        """Extract the tuple size from the batch."""
        _, _, negatives = batch  # ignore anchor and positive
        num_negatives = negatives.shape[1]  # N-2 in your notation
        tuple_size = 1 + 1 + num_negatives  # anchor + positive + negatives = N
        
        # If your DynamicNTupleDataset uses N=4, this should return 4
        return tuple_size
    
    def compute_empirical_risk(self, anchor_embed, positive_embed, negative_embeds):
        empirical_risk = self.loss_fn(anchor_embed, positive_embed, negative_embeds)
        
        if self.pmin is not None:
            # Scale if necessary (optional for N-tuple loss if bounded [0,1])
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
        return empirical_risk

    
    def compute_losses(self, net, anchor_imgs, positive_imgs,
                        negative_imgs, clamping=True):
        """Compute the PBBobj N-tuple loss."""
        anchor_embed = net(anchor_imgs, sample=True)
        positive_embed = net(positive_imgs, sample=True)

        B, N_neg, C, H, W = negative_imgs.shape
        negative_flat = negative_imgs.view(-1, C, H, W)
        negative_embed_flat = net(negative_flat, sample=True)
        negative_embed = negative_embed_flat.view(B, N_neg, -1)

        loss = self.compute_empirical_risk(anchor_embed, positive_embed, negative_embed)

        if clamping:
            loss = torch.clamp(loss, min=0.0, max=1.0)

        return loss, (anchor_embed, positive_embed, negative_embed)
    
    def bound(self, empirical_risk, kl, train_size, tuple_size=None,
              lambda_var=None):
        
        kl = kl * self.kl_penalty
        delta = self.delta
        log_term = torch.as_tensor(np.log((2 * np.sqrt(train_size)) / delta),
                                    dtype=empirical_risk.dtype, device=empirical_risk.device)
        
        """Compute the training objective bound."""
        if self.objective == 'fquad':
            term = (kl + log_term) / (2 * train_size)
            part1 = torch.sqrt(empirical_risk + term)
            part2 = torch.sqrt(term)

            train_obj = torch.pow(part1 + part2, 2)
        
        elif self.objective  == 'fclassic':
            term = (kl + log_term) / (2 * train_size)
            train_obj =  empirical_risk + torch.sqrt(term)

        elif self.objective == 'ntuple':
            kl = kl * self.kl_penalty
            # Use binomial coefficient C(n, tuple_size) for variable tuple sizes
            combinations = math.comb(self.n_bound, tuple_size) if tuple_size <= self.n_bound else self.n_bound**tuple_size
            kl_ratio = torch.div(
                kl + np.log(combinations / self.delta), 
                np.trunc(train_size / tuple_size))
            
            train_obj = empirical_risk + torch.sqrt(kl_ratio)

        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return train_obj


    def mcsampling_ntuple(self, net, data_loader):
        """
        Computes the average empirical risk and pseudo-accuracy over a dataset
        using Monte Carlo sampling with N-tuple data.

        Args:
            net (nn.Module): The probabilistic network to evaluate.
            data_loader (DataLoader): DataLoader providing batches of tuples
                                    (anchor_imgs, positive_imgs, negative_imgs).

        Returns:
            tuple: (avg_risk, avg_pseudo_accuracy)
                avg_risk (float): Estimated average empirical risk over dataset.
                avg_pseudo_accuracy (float): Estimated average pseudo-accuracy.
        """
        net.eval()  # set model to evaluation mode
        total_risk = 0.0
        total_pseudo_accuracy = 0.0
        total_batches = len(data_loader)
        mc_samples = self.mc_samples  # number of MC iterations

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MC Sampling", leave=False):
                # Assuming batch = (anchor_imgs, positive_imgs, negative_imgs)
                anchor_imgs, positive_imgs, negative_imgs = batch

                # Move to device consistent with net
                device = self.device if torch.cuda.is_available() else 'cpu'
                anchor_imgs = anchor_imgs.to(device)
                positive_imgs = positive_imgs.to(device)
                negative_imgs = negative_imgs.to(device)

                risk_mc_batch = 0.0

                for _ in range(mc_samples):
                    # Forward pass stochastic sampling enabled
                    loss, _ = self.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs, clamping=True)

                    # Accumulate loss and accuracy
                    risk_mc_batch += loss.item()

                # Average over MC samples per batch
                risk_mc_batch /= mc_samples

                total_risk += risk_mc_batch

        avg_risk = total_risk / total_batches

        return avg_risk
    
    def train_obj(self, net, batch, train_size, tuple_size=None):

        tuple_size = self.get_tuple_size(batch)

        kl = net.compute_kl()

        empirical_risk, _ = self.compute_losses(net, *batch)

        train_obj = self.bound(empirical_risk, kl, train_size, tuple_size=tuple_size)

        return train_obj, empirical_risk, kl/ train_size
    
    def compute_final_stats_risk




