import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from tqdm import tqdm, trange

from probdist import Gaussian, Laplace, trunc_normal_

class Linear(nn.Module):
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)
    
class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer."""
    
    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda',
                 init_prior='weights', init_layer=None, init_layer_prior=None):
        super(ProbLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if torch.cuda.is_available() else 'cpu'

        # ✅ Add bounds checking for rho_prior
        rho_prior = max(-10, min(rho_prior, 10))  # Prevent extreme values

        # compute and set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        if init_prior == 'zeros':
            weights_mu_prior = torch.zeros_like(weights_mu_init)
            bias_mu_prior = torch.zeros_like(bias_mu_init)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features)
        else:  # 'weights' case
            # ✅ Use slightly different initialization to avoid zero KL
            weights_mu_prior = weights_mu_init + torch.randn_like(weights_mu_init) * 0.05
            bias_mu_prior = bias_mu_init + torch.randn_like(bias_mu_init) * 0.05

        # ✅ Ensure device consistency
        weights_mu_init = weights_mu_init.to(self.device)
        bias_mu_init = bias_mu_init.to(self.device)
        weights_rho_init = weights_rho_init.to(self.device)
        bias_rho_init = bias_rho_init.to(self.device)
        weights_mu_prior = weights_mu_prior.to(self.device)
        bias_mu_prior = bias_mu_prior.to(self.device)

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')
        
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        # ✅ Validate input
        if not torch.is_tensor(input):
            raise ValueError("Input must be a tensor")
        
        input = input.to(self.device)  # Ensure input is on correct device
        
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        # ✅ ALWAYS compute KL (not just during training)
        self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                    self.bias.compute_kl(self.bias_prior)
        
        # ✅ Clamp KL to reasonable bounds
        self.kl_div = torch.clamp(self.kl_div, min=1e-8, max=1e6)
        
        return F.linear(input, weight, bias)

class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 rho_prior, prior_dist='gaussian', device='cuda',
                 stride=1, padding=0, dilation=1, init_prior='weights',
                 init_layer=None, init_layer_prior=False):
        super(ProbConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.device = device if torch.cuda.is_available() else 'cpu'

        # ✅ Add bounds checking for rho_prior
        rho_prior = max(-10, min(rho_prior, 10))  # Prevent extreme values

        # compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(
        out_channels, in_channels, kernel_size, kernel_size), 
        0, sigma_weights, -2*sigma_weights, 2*sigma_weights)

            bias_mu_init = torch.zeros(out_channels)

        # set scale parameters
        weight_rho_init = torch.ones(
            out_channels, in_channels, *self.kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        if init_prior == 'zeros':
            weights_mu_prior = torch.zeros_like(weights_mu_init)
            bias_mu_prior = torch.zeros_like(bias_mu_init)
        
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_channels)
        
        elif init_prior == 'weights':
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
        
            else:
                # initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        
        else:
            raise RuntimeError(f'Wrong init_prior {init_prior}')

        
        if prior_dist == "gaussian":
            dist = Gaussian
        elif prior_dist == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')
        
        # initialise the weights and biases
        self.weight = dist(weights_mu_init.clone(),
                            weight_rho_init.clone(), device=self.device, fixed=False)

        self.bias = dist(bias_mu_init.clone(),
                            bias_rho_init.clone(), device=self.device, fixed=False)
        
        self.weight_prior = dist(
            weights_mu_prior.clone(), weight_rho_init.clone(), device=self.device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=self.device, fixed=True)
        
        self.kl_div = 0
        

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors

            # sample from specified distribution
            weight = self.weight.sample()
            bias = self.bias.sample()

        else:
            # otherwise we use the posterior mean and bias
            weight = self.weight.mu
            bias = self.bias.mu

        # ✅ ALWAYS compute KL (same fix as ProbLinear)
        self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                    self.bias.compute_kl(self.bias_prior)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    

# class Lambda_var(nn.Module):
#     """Class for the lambda variable included in the objective
#     flambda

#     Parameters
#     ----------
#     lamb : float
#         initial value

#     n : int
#         Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)

#     """

#     def __init__(self, lamb, n):
#         super().__init__()
#         self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
#         self.min = 1/np.sqrt(n)

#     @property
#     def lamb_scaled(self):
#         # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
#         m = nn.Sigmoid()
#         return (m(self.lamb) * (1-self.min) + self.min)