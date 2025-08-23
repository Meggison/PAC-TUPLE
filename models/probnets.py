from __future__ import annotations
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.nn.init import trunc_normal_
from layers.problayers import ProbConv2d, ProbLinear

class ProbReIDNet4l(nn.Module):
    """Probabilistic 4-layer ReID network"""
    
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.l1 = ProbLinear(2048, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l1 if init_net else None)
        self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l2 if init_net else None)
        self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l3 if init_net else None)
        self.l4 = ProbLinear(600, embedding_dim, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l4 if init_net else None)
        
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, sample=False):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = self.l4(x, sample)
        x = self.bn(x)
        return x
    
    def compute_kl(self):
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div
    
class ProbReIDCNNet4l(nn.Module):
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.conv1 = ProbConv2d(3, 32, 3, rho_prior, prior_dist=prior_dist, device=device,
                               padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(32, 64, 3, rho_prior, prior_dist=prior_dist, device=device,
                               padding=1, init_layer=init_net.conv2 if init_net else None)
        
        # FIX: Change from 64 * 64 * 32 to 64 * 16 * 16
        self.fc1 = ProbLinear(64 * 16 * 16, 128, rho_prior, prior_dist=prior_dist,  # ✅ CORRECT: 16384
                              device=device, init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, embedding_dim, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None)
        
        self.bn = nn.BatchNorm1d(embedding_dim)


    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = self.fc2(x, sample)
        x = self.bn(x)
        return x

    def compute_kl(self):
        # divide by the number of layers to normalize the KL divergence (come back to this!!!!)
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div
    

class ProbReIDCNNet9l(nn.Module):
    """Probabilistic 9-layer CNN for ReID"""
    
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.conv1 = ProbConv2d(3, 32, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(32, 64, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(64, 128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(128, 128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(128, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        
        self.fcl1 = ProbLinear(256 * 8 * 4, 1024, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, embedding_dim, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None)
        
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.adaptive_avg_pool2d(x, (8, 4))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = self.bn(x)
        return x

    def compute_kl(self):
        return (self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + 
                self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div +
                self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div)

class ProbReIDCNNet13l(nn.Module):
    """Probabilistic 13-layer CNN for ReID"""
    
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = ProbConv2d(3, 32, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(32, 64, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(64, 128, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(128, 128, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(128, 256, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(256, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(512, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(512, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        
        # Fully connected layers
        self.fcl1 = ProbLinear(512 * 4 * 2, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, embedding_dim, rho_prior=rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fcl3 if init_net else None)
        
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x, sample))
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.adaptive_avg_pool2d(x, (4, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = self.bn(x)
        return x

    def compute_kl(self):
        return (self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + 
                self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + 
                self.conv9.kl_div + self.conv10.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + 
                self.fcl3.kl_div)

class ProbReIDCNNet15l(nn.Module):
    """Probabilistic 15-layer CNN for ReID"""
    
    def __init__(self, embedding_dim=256, rho_prior=1.0, prior_dist='gaussian', device='cuda', init_net=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # All convolutional layers
        self.conv1 = ProbConv2d(3, 32, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(32, 64, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(64, 128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(128, 128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(128, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(256, 256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(256, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(512, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                                kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.conv11 = ProbConv2d(512, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                                kernel_size=3, padding=1, init_layer=init_net.conv11 if init_net else None)
        self.conv12 = ProbConv2d(512, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                                kernel_size=3, padding=1, init_layer=init_net.conv12 if init_net else None)
        
        self.fcl1 = ProbLinear(512 * 4 * 2, 1024, rho_prior=rho_prior,
                              prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                              prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, embedding_dim, rho_prior=rho_prior,
                              prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None)
        
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, sample=False):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.relu(self.conv8(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.relu(self.conv11(x, sample))
        x = F.relu(self.conv12(x, sample))
        x = F.adaptive_avg_pool2d(x, (4, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = self.bn(x)
        return x

    def compute_kl(self):
        # divide by the number of layers
        return (self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + 
                self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + 
                self.conv9.kl_div + self.conv10.kl_div + self.conv11.kl_div + self.conv12.kl_div + 
                self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div)