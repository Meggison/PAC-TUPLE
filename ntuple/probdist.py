import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import torchvision.models as models
import os

# disable NNPACK for compatibility with probabilistic batch normalization
os.environ['TORCH_USE_NNPACK'] = '1'


def no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    outside :math:`[a, b]` redrawn until they are within the bounds.
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        """Standard normal cumulative distribution function."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    with torch.no_grad():
        # Get upper and lower bounds in terms of standard deviations
        lower_bound = norm_cdf((a-mean) / std)
        upper_bound = norm_cdf((b-mean) / std)

        # fill tensor with truncated normal values
        tensor.uniform_(lower_bound, upper_bound)

        # Apply the inverse CDF to get the truncated normal values in range [-1, 1]
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure the tensor is within the specified bounds
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. + eps), max=(1. - eps))
        tensor.erfinv_()

        # Scale and shift to the desired mean and std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        # Clamp the tensor to the specified bounds for safety
        tensor.clamp_(min=a, max=b)

        return tensor
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return no_grad_trunc_normal_(tensor, mean, std, a, b)


class Gaussian(nn.Module):
    """ Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation for variational inference.
    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device if torch.cuda.is_available() else 'cpu'

    
    @property
    def sigma(self):
        """Compute the standard deviation from rho.
        We use rho instead of sigma so that sigma is always positive during
        the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        """
        return torch.log(1 + torch.exp(self.rho))
    
    def sample(self):
        """Sample from the Gaussian distribution."""
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps
    
    def compute_kl(self, other):
        """Compute the KL divergence between two Gaussian distributions."""

        if not isinstance(other, Gaussian):
            raise TypeError("other must be an instance of Gaussian")
        
        # Compute the variance from sigma
        var_self = torch.pow(self.sigma, 2)
        var_other = torch.pow(other.sigma, 2)

        # Kl divergence formula
        term1 = torch.log(var_other / var_self)
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), var_other)
        term3 = torch.div(var_self, var_other)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).mean() # may switch to mean later

        return kl_div
    

class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div