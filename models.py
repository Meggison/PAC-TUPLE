import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange
import torchvision.models as models
from probbn import ProbBatchNorm2d

import os
os.environ["TORCH_DISABLE_NNPACK"] = "1"


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device if torch.cuda.is_available() else 'cpu'

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        

    def sample(self):
        # Return a sample from the Gaussian distribution
        # epsilon = torch.randn(self.sigma.size()).to(self.device)
        epsilon = torch.randn(self.sigma.size(), device=self.device) # changed to cuda
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div



class ResProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_prior='weights', init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
            
        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior
        
        if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
        else:
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
     
        if prior_dist == 'gaussian':
            dist = Gaussian
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
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()

            
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) +self.bias.compute_kl(self.bias_prior)
        

        return F.linear(input, weight, bias)

    
class ResProbBN(nn.Module):

    def __init__(self, in_channels, rho_prior, prior_dist='gaussian',
                 device='cuda' , init_prior='weights', init_layer=None, init_layer_prior=None,bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels 
        
        self.m = ProbBatchNorm2d(in_channels,affine = False)
        self.m.running_var = init_layer.running_var
        self.m.running_mean = init_layer.running_mean
        self.m.weight1 = init_layer.weight.data 
        self.m.bias1 = init_layer.bias.data  


        out_channels = in_channels

        in_features = self.in_channels

        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias

        # set scale parameters
        weights_rho_init = torch.ones(out_channels ) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        
        weights_mu_prior = weights_mu_init
                
        bias_mu_prior = bias_mu_init


        if prior_dist == 'gaussian':
            dist = Gaussian
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):

        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()

        else:

            # otherwise we use the posterior mean
            weight = self.weight.mu.data
            bias = self.bias.mu.data
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)
        self.m.weight1  = weight
        self.m.bias1  = bias
        return self.m(input)


class ResProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian',
                 device='cuda', stride=1, padding=1, dilation=1, init_prior='weights', init_layer=None, init_layer_prior=None,bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.bias = bias

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        weights_mu_init = init_layer.weight

        weights_rho_init = torch.ones(
            out_channels, in_channels, *self.kernel_size) * rho_prior

        weights_mu_prior = weights_mu_init

        if prior_dist == 'gaussian':
            dist = Gaussian
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.kl_div = 0

    def forward(self, input, sample=False):
        
        
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            # weight = weight.to(input.dtype).to(input.device)
            input = input.to(weight.device)  # ensure input is on the same device as weight
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            # weight = weight.to(input.dtype).to(input.device)
            input = input.to(weight.device)  # ensure input is on the same device as weight

        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior)
        return F.conv2d(input, weight, bias = None, stride=self.stride, padding =self.padding)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel,affine = False)
        
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel,affine = False)
        
        self.conv3 = nn.Sequential()
        self.bn3 = nn.Sequential()
        
        if stride != 1 or inchannel != outchannel:
            self.conv3 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(outchannel,affine = False)

    def forward(self, x1):
        x = self.conv1(x1)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        out = self.bn2(x)

        out1 = self.conv3(x1)
        out1 = self.bn3(out1)

        out = out + out1
        out = nn.ReLU(inplace=True)(out)
        return out

        
class ProbResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super(ProbResidualBlock, self).__init__()

        self.conv1 = ResProbConv2d(
            inchannel, outchannel, 3, rho_prior, prior_dist=prior_dist, device=device, stride=stride, padding=1,
            init_layer=init_net.conv1  )

        self.bn1 = init_net.bn1

        self.conv2 = ResProbConv2d(
            outchannel, outchannel, 3, rho_prior, prior_dist=prior_dist, device=device, stride=1, padding=1,
            init_layer=init_net.conv2 )

        self.bn2 = init_net.bn2
        self.downsample = False

        self.conv3 = nn.Sequential()
        self.bn3 = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = True
            self.conv3 = ResProbConv2d(
                inchannel, outchannel, 1, rho_prior, prior_dist=prior_dist, device=device, stride=stride, padding=0,
                init_layer=init_net.downsample[0]   )

            self.bn3 = init_net.downsample[1]

    def forward(self, x1, sample=False):

        x = self.conv1(x1,sample)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x,sample)
        out = self.bn2(x)

        if self.downsample: 
            out1 = self.conv3(x1,sample)
            out1 = self.bn3(out1)
       
        else:
            out1 = self.conv3(x1)
            out1 = self.bn3(out1)
        out = out + out1
        
        out = nn.ReLU(inplace=True)(out)
        return out

    def resnet_kl(self):
        if self.downsample: 
            return self.conv1, self.conv2, self.conv3 
        else: 
            return self.conv1, self.conv2,0


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet50_model = models.resnet50(pretrained=True)
        self.resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])

        # --- OPTIONAL: Add a classification head as per the paper ---
        # This allows you to combine N-tuple loss with classification loss
        # self.classifier = nn.Linear(2048, num_classes) # num_classes = number of IDs in train set

        # self.fc1 = nn.Linear(2048, 2)

    def forward_once(self, x):
        out = self.resnet50_model(x)
        out = out.view(out.size(0), -1)

        return out

    def forward(self, x):
      """
      ADAPTED FOR N-TUPLE: This forward pass now takes a single batch of
      images and returns their feature embeddings.
      """
      # 1. Extract feature embeddings
      features = self.resnet50_model(x)
      embeddings = features.view(features.size(0), -1)

      # 2. OPTIONAL: Get classification logits if you are using combined loss
      # If you are only using N-tuple loss, you can comment this out.
      # logits = self.classifier(embeddings)
      # return embeddings, logits
        
      # If using only N-tuple loss, just return the embeddings
      return embeddings


def ResNet50_new():
    return ResNet().to('cuda')

class ProbResidualBlock_bn(nn.Module):
    def __init__(self, inchannel, outchannel, stride, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super(ProbResidualBlock_bn, self).__init__()
        self.inchannel = inchannel

        self.conv1 = ResProbConv2d(
            inchannel, outchannel, 3, rho_prior, prior_dist=prior_dist, device=device, stride=stride, padding=1,
            init_layer=init_net.conv1  )

        self.bn1 = ResProbBN(outchannel, rho_prior,prior_dist=prior_dist, device=device,
            init_layer=init_net.bn1)

        self.conv2 = ResProbConv2d(
            outchannel, outchannel, 3, rho_prior, prior_dist=prior_dist, device=device, stride=1, padding=1,
            init_layer=init_net.conv2 )

        self.bn2 = ResProbBN(outchannel, rho_prior,prior_dist=prior_dist, device=device,
            init_layer=init_net.bn2)

        self.downsample = False

        self.conv3 = nn.Sequential()
        self.bn3 = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.downsample = True

            self.conv3 = ResProbConv2d(
                inchannel, outchannel, 1, rho_prior, prior_dist=prior_dist, device=device, stride=stride, padding=0,
                init_layer=init_net.downsample[0]   )
            self.bn3 = ResProbBN(outchannel, rho_prior,prior_dist=prior_dist, device=device,
            init_layer=init_net.downsample[1])

    def forward(self, x1, sample=False):
        x = self.conv1(x1,sample)

        x = self.bn1(x,sample)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x,sample)
        out = self.bn2(x,sample)

        if self.downsample: 
            out1 = self.conv3(x1,sample)
            out1 = self.bn3(out1,sample)
       
        else:
            out1 = self.conv3(x1)
            out1 = self.bn3(out1)
        out = out + out1
        
        out = nn.ReLU(inplace=True)(out)
        return out

    def resnet_kl(self):
        if self.downsample: 
            return self.conv1, self.conv2, self.conv3,self.bn1, self.bn2,self.bn3    
        else: 
            return self.conv1, self.conv2, 0, self.bn1, self.bn2,0


class ProbBottleneckBlock(nn.Module):
    """
    Probabilistic Bottleneck Block for ResNet50 architecture.
    Implements the 1x1 -> 3x3 -> 1x1 convolution pattern with expansion factor of 4.
    """
    def __init__(self, inplanes, planes, stride, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super(ProbBottleneckBlock, self).__init__()
        self.expansion = 4
        
        # 1x1 conv
        self.conv1 = ResProbConv2d(
            inplanes, planes, 1, rho_prior, prior_dist=prior_dist, device=device, 
            stride=1, padding=0, init_layer=init_net.conv1)
        
        self.bn1 = ResProbBN(planes, rho_prior, prior_dist=prior_dist, device=device,
                            init_layer=init_net.bn1)
        
        # 3x3 conv
        self.conv2 = ResProbConv2d(
            planes, planes, 3, rho_prior, prior_dist=prior_dist, device=device,
            stride=stride, padding=1, init_layer=init_net.conv2)
        
        self.bn2 = ResProbBN(planes, rho_prior, prior_dist=prior_dist, device=device,
                            init_layer=init_net.bn2)
        
        # 1x1 conv (expansion)
        self.conv3 = ResProbConv2d(
            planes, planes * self.expansion, 1, rho_prior, prior_dist=prior_dist, device=device,
            stride=1, padding=0, init_layer=init_net.conv3)
        
        self.bn3 = ResProbBN(planes * self.expansion, rho_prior, prior_dist=prior_dist, device=device,
                            init_layer=init_net.bn3)
        
        # Shortcut connection
        self.downsample = False
        self.shortcut = nn.Sequential()
        self.shortcut_bn = nn.Sequential()
        
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = True
            self.shortcut = ResProbConv2d(
                inplanes, planes * self.expansion, 1, rho_prior, prior_dist=prior_dist, device=device,
                stride=stride, padding=0, init_layer=init_net.downsample[0])
            self.shortcut_bn = ResProbBN(planes * self.expansion, rho_prior, prior_dist=prior_dist, device=device,
                                        init_layer=init_net.downsample[1])
    
    def forward(self, x, sample=False):
        identity = x
        
        # First conv block
        out = self.conv1(x, sample)
        out = self.bn1(out, sample)
        out = nn.ReLU(inplace=True)(out)
        
        # Second conv block
        out = self.conv2(out, sample)
        out = self.bn2(out, sample)
        out = nn.ReLU(inplace=True)(out)
        
        # Third conv block
        out = self.conv3(out, sample)
        out = self.bn3(out, sample)
        
        # Shortcut connection
        if self.downsample:
            identity = self.shortcut(x, sample)
            identity = self.shortcut_bn(identity, sample)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
    
    def resnet_kl(self):
        if self.downsample:
            return self.conv1, self.conv2, self.conv3, self.shortcut, self.bn1, self.bn2, self.bn3, self.shortcut_bn
        else:
            return self.conv1, self.conv2, self.conv3, 0, self.bn1, self.bn2, self.bn3, 0


class ProbResNet_BN(nn.Module):

    def __init__(self, ProbBottleneckBlock, rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
        super(ProbResNet_BN,self).__init__()
        self.inplanes = 64
        self.ResidualBlock_kl = 0
        # init_net1 = init_net.resnet50_model[0]
        init_net1 = list(init_net.resnet50_model.children())  # make it a list of layers


        self.con1 = ResProbConv2d(
            3, 64, 7, rho_prior, stride=2, padding=3, prior_dist=prior_dist, device=device,
            init_layer=init_net1[0], bias=False)

        self.bn1 = ResProbBN(64, rho_prior,prior_dist=prior_dist, device=device,
            init_layer=init_net1[1])

        self.layer1 = self.make_layer(ProbBottleneckBlock, 64, 3, rho_prior, prior_dist='gaussian', device='cuda',
                                      init_net=init_net1[4], stride=1)
        self.layer2 = self.make_layer(ProbBottleneckBlock, 128, 4, rho_prior, prior_dist='gaussian', device='cuda',
                                      init_net=init_net1[5], stride=2)
        self.layer3 = self.make_layer(ProbBottleneckBlock, 256, 6, rho_prior, prior_dist='gaussian', device='cuda',
                                      init_net=init_net1[6], stride=2)
        self.layer4 = self.make_layer(ProbBottleneckBlock, 512, 3, rho_prior, prior_dist='gaussian', device='cuda',
                                      init_net=init_net1[7], stride=2)

        self.cnn1 = nn.Sequential(*self.layer1)
        self.cnn2 = nn.Sequential(*self.layer2)
        self.cnn3 = nn.Sequential(*self.layer3)
        self.cnn4 = nn.Sequential(*self.layer4)

        # self.fc = ResProbLinear(512, 2, rho_prior, prior_dist=prior_dist,
        #                         device=device, init_layer=init_net.fc1 )

    def make_layer(self, block, planes, num_blocks, rho_prior, prior_dist='gaussian', device='cuda', init_net=None,
                   stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i in range(len(strides)):
            block1 = block(self.inplanes, planes, strides[i], rho_prior, prior_dist=prior_dist, device=device, init_net=init_net[i])
            layers.append(block1)
            self.inplanes = planes * 4  # Bottleneck expansion = 4

        return layers
        
    def forward_once(self, x, sample=False, clamping=True, pmin=1e-4):
        x = self.con1(x, sample)
        x = self.bn1(x, sample)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        # Layer 1 (2 blocks)
        for i in range(2):
            output = self.cnn1[i](x if i == 0 else output, sample)

        # Layer 2 (2 blocks)
        for i in range(2):
            output = self.cnn2[i](output, sample)

        # Layer 3 (2 blocks)
        for i in range(2):
            output = self.cnn3[i](output, sample)
        
        # Layer 4 (2 blocks)
        for i in range(2):
            output = self.cnn4[i](output, sample)
        output = nn.AdaptiveAvgPool2d((1, 1))(output)

        output = output.view(output.size(0), -1)
        return output

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):

        """
        ADAPTED FOR N-TUPLE: This is the new forward pass. It takes a single batch of
        images 'x' and produces their embeddings. The old logic from `forward_once`
        is moved here.
        """
        x = self.con1(x, sample)
        x = self.bn1(x, sample)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        # Layer 1 (2 blocks)
        for i in range(2):
            output = self.cnn1[i](x if i == 0 else output, sample)

        # Layer 2 (2 blocks)
        for i in range(2):
            output = self.cnn2[i](output, sample)

        # Layer 3 (2 blocks)
        for i in range(2):
            output = self.cnn3[i](output, sample)
        
        # Layer 4 (2 blocks)
        for i in range(2):
            output = self.cnn4[i](output, sample)
        
        output = nn.AdaptiveAvgPool2d((1, 1))(output)
        embeddings = output.view(output.size(0), -1)
        
        # --- Optional: Get classification logits ---
        # If using combined loss, pass embeddings through the classifier.
        # Note: The output_transform is for classification, so it's applied here.
        # logits = self.classifier(embeddings, sample)
        # transformed_logits = output_transform(logits, clamping, pmin)
        # return embeddings, transformed_logits

        # If using only N-tuple loss, just return the embeddings
        return embeddings

    def compute_kl(self):
        self.cnn1_kl_div = 0
        self.cnn2_kl_div = 0

        self.cnn3_kl_div = 0
        self.cnn4_kl_div = 0

        for i in range(len(self.layer1)):
            conv1, conv2, conv3, shortcut, bn1, bn2, bn3, shortcut_bn = self.layer1[i].resnet_kl()
            temp = conv1.kl_div + conv2.kl_div + conv3.kl_div + bn1.kl_div + bn2.kl_div + bn3.kl_div
            if not isinstance(shortcut, int):
                temp += shortcut.kl_div + shortcut_bn.kl_div
            self.cnn1_kl_div += temp
            
        for i in range(len(self.layer2)):
            conv1, conv2, conv3, shortcut, bn1, bn2, bn3, shortcut_bn = self.layer2[i].resnet_kl()
            temp = conv1.kl_div + conv2.kl_div + conv3.kl_div + bn1.kl_div + bn2.kl_div + bn3.kl_div
            if not isinstance(shortcut, int):
                temp += shortcut.kl_div + shortcut_bn.kl_div
            self.cnn2_kl_div += temp

        for i in range(len(self.layer3)):
            conv1, conv2, conv3, shortcut, bn1, bn2, bn3, shortcut_bn = self.layer3[i].resnet_kl()
            temp = conv1.kl_div + conv2.kl_div + conv3.kl_div + bn1.kl_div + bn2.kl_div + bn3.kl_div
            if not isinstance(shortcut, int):
                temp += shortcut.kl_div + shortcut_bn.kl_div
            self.cnn3_kl_div += temp

        for i in range(len(self.layer4)):
            conv1, conv2, conv3, shortcut, bn1, bn2, bn3, shortcut_bn = self.layer4[i].resnet_kl()
            temp = conv1.kl_div + conv2.kl_div + conv3.kl_div + bn1.kl_div + bn2.kl_div + bn3.kl_div
            if not isinstance(shortcut, int):
                temp += shortcut.kl_div + shortcut_bn.kl_div
            self.cnn4_kl_div += temp

        return self.con1.kl_div + self.cnn1_kl_div + self.cnn2_kl_div + self.cnn3_kl_div + self.cnn4_kl_div #+ self.fc.kl_div   + self.fc2.kl_div + self.fc3.kl_div



def ProbResNet_bn(rho_prior, prior_dist='gaussian', device='cuda', init_net=None):
    return ProbResNet_BN(ProbBottleneckBlock, rho_prior, prior_dist=prior_dist, device=device, init_net=init_net).to(device)


def output_transform(x, clamping=True, pmin=1e-4):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)

    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


def trainNNet(net, optimizer, epoch, train_loader,   errornet0,device='cuda', verbose=False):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, ( data1,data2, target) in enumerate(tqdm(train_loader)):

            try:
                target  = target.squeeze(1)
            except:

                pass
            data1,data2, target = data1.to(device),data2.to(device), target.to(device)
            net.zero_grad()

            output = net(data1,data2)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            avgloss = avgloss + loss.detach()



def testNNet(net, test_loader, device='cuda', verbose=True):
    """Test function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    test_loader: DataLoader object
        Test data loader

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.eval()
    correct, total = 0, 0.0
    with torch.no_grad():
        for data1,data2,target in test_loader:
            try:
                    target  = target.squeeze(1)
            except:
                 
                    pass

            data1,data2, target = data1.to(device),data2.to(device), target.to(device)
            outputs = net(data1,data2)
            loss = F.nll_loss(outputs, target)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

                
    print(
        f"-Prior: Test loss: {loss.item() :.5f}, Test err:  {1-(correct/total):.5f}")
    return 1-(correct/total)


def trainPNNet(net, optimizer, pbobj, epoch, train_loader, lambda_var=None, optimizer_lambda=None, verbose=False):
    """Train function for a probabilistic NN (including CNN)

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    optimizer_lambda : optim object
        Optimizer to use for the learning the lambda_variable

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

    clamping = True

    for batch_id, (data1,data2, target) in enumerate(tqdm(train_loader)):

            try:
                target  = target.squeeze(1)
            except:
                
                pass
            data1, data2, target = data1.to(pbobj.device), data2.to(pbobj.device), target.to(pbobj.device)
            net.zero_grad()
            bound, kl, _, loss, err = pbobj.train_obj(
                net, data1,data2, target, lambda_var=lambda_var, clamping=clamping)
    
            bound.backward()
            optimizer.step()
            avgbound += bound.item()
            avgkl += kl
            avgloss += loss.item()
            avgerr += err


def testStochastic(net, test_loader, pbobj, device='cuda'):
    """Test function for the stochastic predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    # compute mean test accuracy
    
    net.eval()
    correct, cross_entropy, total = 0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data1,data2, target) in enumerate(tqdm(test_loader)):
                try:
                        target  = target.squeeze(1)
                except:
                        pass
                data1,data2, target = data1.to(device),data2.to(device), target.to(device)
             
                outputs = net(data1, data2, sample=True, clamping=True, pmin=pbobj.pmin)

                cross_entropy += pbobj.compute_empirical_risk(
                outputs, target, bounded=True)
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

    return cross_entropy/(batch_id+1), 1-(correct/total)


def testPosteriorMean(net, test_loader, pbobj, device='cuda'):
    """Test function for the deterministic predictor using a PNN
    (uses the posterior mean)

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    net.eval()
    correct, cross_entropy,total = 0, 0.0,0.0
    batch_id = 0
    with torch.no_grad():
        for data1,data2,target in test_loader:
            batch_id +=1
            try:
                        target  = target.squeeze(1)
            except:
                        pass
            data1,data2, target = data1.to(device),data2.to(device), target.to(device)
             
            outputs = net(data1, data2, sample=False, clamping=True, pmin=pbobj.pmin)

            cross_entropy += pbobj.compute_empirical_risk(
            outputs, target, bounded=True)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return cross_entropy/(batch_id+1), 1-(correct/total)


def testEnsemble(net, test_loader, pbobj, device='cuda', samples=10):
    """Test function for the ensemble predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    samples: int
        Number of times to sample weights (i.e. members of the ensembles)

    """
    correct, cross_entropy_all, total = 0, 0.0, 0.0
    net.eval()
    with torch.no_grad():

        for batch_id, (data1,data2, target) in enumerate(tqdm(test_loader)):
                data1,data2, target = data1.to(device),data2.to(device), target.to(device)
                try:
                    target  = target.squeeze(1)
                except:
                    pass
                
                outputs = torch.zeros(samples, len(target),
                                      pbobj.classes).to(device)

                for i in range(samples):
                    outputs_i = net(data1, data2, sample=True, clamping=True, pmin=pbobj.pmin)
                    outputs[i] = outputs_i

                avgoutput = outputs.mean(0)
                cross_entropy = pbobj.compute_empirical_risk(
                    avgoutput, target, bounded=True)
                
                cross_entropy_all += cross_entropy
                pred = avgoutput.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

    return cross_entropy_all/(batch_id+1), 1-(correct/total)


def computeRiskCertificates(net, toolarge, pbobj, device='cuda', lambda_var=None, train_loader=None, whole_train=None):
    """Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net.eval()
    with torch.no_grad():
        if toolarge:
            train_obj, kl, loss_ce_train, err_01_train, risk_ce, risk_01 = pbobj.compute_final_stats_risk(
                net, lambda_var=lambda_var, clamping=True, data_loader=train_loader)

    return train_obj, risk_ce, risk_01, kl, loss_ce_train, err_01_train
