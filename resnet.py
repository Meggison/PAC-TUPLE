import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal as Gaussian, Laplace
from torchvision.models import resnet18

from problayers import ProbConv2d, ProbLinear, Linear

class ResNet18(nn.Module):
    """Implementation of a ResNet-18 model with probabilistic layers."""
    def __init__(self, embedding_dim=256, device='cuda'):
        super(ResNet18, self).__init__()

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.embedding_dim = embedding_dim

        # Load the pre-trained ResNet-18 model
        resnet18 = resnet18(weights='IMAGENET1K_V1')

        # remove the last fully connected layer
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

        self.embedding_layer = nn.Sequential(
                nn.Linear(512, self.embedding_dim, bias=False),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=True),
                nn.BatchNorm1d(self.embedding_dim)
            )

    def forward(self, x, embed=True):
        # Forward pass through the ResNet-18 model

        x = self.resnet18(x)
        x = x.view(x.size(0), -1)

        # If embedding is True, pass through the embedding layer
        if embed:
            x = self.embedding_layer(x)

        return x

class ProbBasicBlock(nn.Module):
    """Implementation of a basic block for ResNet with probabilistic layers."""

    expansion = 1

    # Constructor for the ProbBasicBlock
    def __init__(self, in_planes, planes, rho_prior, prior_dist='gaussian', device='cuda', stride=1):
        super(ProbBasicBlock, self).__init__()

        self.conv1 = ProbConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, rho_prior=rho_prior,
                                 prior_dist=prior_dist, device=device)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ProbConv2d(planes, planes, kernel_size=3, stride=1, padding=1, rho_prior=rho_prior,
                                 prior_dist=prior_dist, device=device)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ProbConv2d(in_planes, planes, kernel_size=1,
                                       rho_prior=rho_prior, prior_dist=prior_dist,
                                       device=device, stride=stride, padding=0),
                                       nn.BatchNorm2d(planes)
                                       )
    
    def forward(self, x, sample=False):
        identity = x

        out = self.conv1(x, sample=sample)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out, sample=sample)
        out = self.bn2(out)

        shortcut = identity
        if isinstance(self.shortcut, nn.Sequential):
            # Apply layers sequentially to shortcut
            for l in self.shortcut:
                # Check if the layer is probabilistic and needs a 'sample' argument
                if isinstance(l, (ProbConv2d, ProbLinear)):
                    shortcut = l(shortcut, sample=sample)
                else:
                    shortcut = l(shortcut)
        else:
            # For nn.Identity, shortcut is just x
            shortcut = self.shortcut(shortcut)

        out += shortcut
        out = nn.ReLU(inplace=True)(out)

        return out

class ProbResNet18(nn.Module):
    """Implementation of a ResNet-18 model with probabilistic layers."""
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda'):
        super(ProbResNet18, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.embedding_dim = embedding_dim

        self.in_planes = 64

        self.conv1 = ProbConv2d(3, 64, kernel_size=7, rho_prior=rho_prior,
                                    prior_dist=prior_dist, device=device, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Backbone layers
        self.layer1 = self._make_layer(ProbBasicBlock, 64, 2, stride=1, rho_prior=rho_prior, prior_dist=prior_dist)
        self.layer2 = self._make_layer(ProbBasicBlock, 128, 2, stride=2, rho_prior=rho_prior, prior_dist=prior_dist)
        self.layer3 = self._make_layer(ProbBasicBlock, 256, 2, stride=2, rho_prior=rho_prior, prior_dist=prior_dist)
        self.layer4 = self._make_layer(ProbBasicBlock, 512, 2, stride=2, rho_prior=rho_prior, prior_dist=prior_dist)

        # Embedding Head (ResNet-18)
        self.embedding_head = nn.Sequential(
            ProbLinear(512, self.embedding_dim, rho_prior=rho_prior,
                       prior_dist=prior_dist, device=device),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            ProbLinear(self.embedding_dim, self.embedding_dim, rho_prior=rho_prior,
                       prior_dist=prior_dist, device=device),
            nn.BatchNorm1d(self.embedding_dim)
        )

    def _make_layer(self, block, planes, num_blocks, rho_prior, prior_dist, stride):
        """Create a layer of blocks."""
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            layers.append(block(self.in_planes, planes, rho_prior, prior_dist, self.device, s))
            self.in_planes = planes * block.expansion

        return nn.ModuleList(layers)
    
    def _forward_layer(self, x, layers, sample):
        for block in layers:
            x = block(x, sample=sample)
        return x

    def forward(self, x, embed=True, sample=False):
        """Forward pass through the ResNet-18 model."""
        x = self.conv1(x, sample=sample)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self._forward_layer(x, self.layer1, sample)
        x = self._forward_layer(x, self.layer2, sample)
        x = self._forward_layer(x, self.layer3, sample)
        x = self._forward_layer(x, self.layer4, sample)

        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)

        if embed:
            x = self.embedding_head(x)

        return x
        
    def compute_kl(self):
        """Compute the KL divergence for all probabilistic layers."""
        kl_div = 0
        for layer in self.modules():
            if hasattr(layer, 'kl_div'):
                kl_div += getattr(layer, 'kl_div', 0)
                
        return kl_div