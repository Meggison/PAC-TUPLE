import math
import numpy as np
import torch
from tqdm import tqdm
from losses import NTupleLoss

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
        self.loss_fn = NTupleLoss()

    def get_tuple_size(self, batch):
        _, _, negatives = batch
        num_negatives = negatives.shape[1]
        return 2 + num_negatives

    def compute_empirical_risk(self, anchor_embed, positive_embed, negative_embeds):
        empirical_risk = self.loss_fn(anchor_embed, positive_embed, negative_embeds)
        if self.pmin is not None:
            empirical_risk = empirical_risk / np.log(1. / self.pmin)
        return empirical_risk

    def compute_losses(self, net, anchor_imgs, positive_imgs, negative_imgs, clamping=True):
        anchor_imgs = anchor_imgs.to(self.device)
        positive_imgs = positive_imgs.to(self.device)
        negative_imgs = negative_imgs.to(self.device)
        anchor_embed = net(anchor_imgs, sample=True)
        positive_embed = net(positive_imgs, sample=True)

        B, N_neg, C, H, W = negative_imgs.shape
        negative_imgs_flat = negative_imgs.view(B * N_neg, C, H, W)
        negative_embed_flat = net(negative_imgs_flat, sample=True)
        negative_embed = negative_embed_flat.view(B, N_neg, -1)

        empirical_risk = self.compute_empirical_risk(anchor_embed, positive_embed, negative_embed)
        if clamping:
            empirical_risk = torch.clamp(empirical_risk, 0.0, 1.0)
        return empirical_risk, (anchor_embed, positive_embed, negative_embed)

    def bound(self, empirical_risk, kl, train_size, tuple_size=None, lambda_var=None):
        kl = kl * self.kl_penalty
        delta = self.delta
        # First, ensure empirical_risk is a tensor
        if not torch.is_tensor(empirical_risk):
             # Default to float32 and CPU if you can't infer from anything else
            empirical_risk = torch.tensor(empirical_risk, dtype=torch.float32)
        log_term = torch.tensor(
            np.log((2 * np.sqrt(train_size)) / delta),
            dtype=empirical_risk.dtype,
            device=empirical_risk.device)


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
                # Use loggamma for numerical stability with large combinations
                combinations = math.exp(math.lgamma(self.n_bound + 1) - math.lgamma(tuple_size + 1) - math.lgamma(self.n_bound - tuple_size + 1))
            else:
                # Fallback (potentially unsafe for very large exponents)
                combinations = float(self.n_bound ** tuple_size if tuple_size else 1)
            kl_ratio = (kl + np.log(combinations / delta)) / max(1, np.trunc(train_size / (tuple_size if tuple_size else 1)))
            return empirical_risk + torch.sqrt(torch.tensor(kl_ratio, dtype=empirical_risk.dtype, device=empirical_risk.device))
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def mcsampling_ntuple(self, net, data_loader):
        net.eval()
        total_risk = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MC Sampling", leave=False):
                anchor_imgs, positive_imgs, negative_imgs = batch
                anchor_imgs = anchor_imgs.to(self.device)
                positive_imgs = positive_imgs.to(self.device)
                negative_imgs = negative_imgs.to(self.device)
                risk_mc_batch = 0.0
                for _ in range(self.mc_samples):
                    loss, _ = self.compute_losses(net, anchor_imgs, positive_imgs, negative_imgs, clamping=True)
                    risk_mc_batch += loss.item()
                risk_mc_batch /= self.mc_samples
                total_risk += risk_mc_batch
        avg_risk = total_risk / num_batches
        avg_pseudo_accuracy = 0.0  # Placeholder; implement if needed
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
    