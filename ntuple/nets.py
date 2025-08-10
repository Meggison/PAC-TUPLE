from __future__ import annotations
import torch.nn.functional as F
import torch.nn as nn
import torch

from problayers import ProbConv2d, ProbLinear, Linear


class ReIDNet4l(nn.Module):
    """4-layer ReID network adapted from NNet4l"""
    
    def __init__(self, embedding_dim=256, dropout_prob=0.0, device='cuda'):
        super().__init__()
        # Assuming input is flattened image features or pre-extracted features
        self.l1 = Linear(2048, 600, device)  # Adjusted for typical ReID feature size
        self.l2 = Linear(600, 600, device)
        self.l3 = Linear(600, 600, device)
        self.l4 = Linear(600, embedding_dim, device)  # Output embedding instead of classes
        self.d = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn(x)  # Normalize embeddings
        return x
    

class ReIDCNNet4l(nn.Module):
    def __init__(self, embedding_dim=256, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        
        # FIX: Change from 64 * 64 * 32 to 64 * 16 * 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # = 16384
        self.fc2 = nn.Linear(128, embedding_dim)
        
        self.d = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim


    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x
    

class ReIDCNNet9l(nn.Module):
    """9-layer CNN for ReID adapted from CNNet9l"""
    def __init__(self, embedding_dim=256, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Embedding layers instead of classification
        self.fcl1 = nn.Linear(256 * 8 * 4, 1024)  # Adjusted for ReID
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, embedding_dim)
        
        self.d = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.d(F.relu(self.conv1(x)))
        x = self.d(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.d(F.relu(self.conv3(x)))
        x = self.d(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.d(F.relu(self.conv5(x)))
        x = self.d(F.relu(self.conv6(x)))
        x = F.adaptive_avg_pool2d(x, (8, 4))
        x = x.view(x.size(0), -1)
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = self.bn(x)
        return x
    
    
class ReIDCNNet13l(nn.Module):
    """13-layer CNN for ReID adapted from CNNet13l"""
    
    def __init__(self, embedding_dim=256, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.fcl1 = nn.Linear(512 * 4 * 2, 1024)  # Adjusted
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, embedding_dim)
        
        self.d = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv8(x)))
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.adaptive_avg_pool2d(x, (4, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = self.bn(x)
        return x
    

class ReIDCNNet15l(nn.Module):
    """15-layer CNN for ReID adapted from CNNet15l"""
    
    def __init__(self, embedding_dim=256, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.fcl1 = nn.Linear(512 * 4 * 2, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, embedding_dim)
        
        self.d = nn.Dropout(dropout_prob)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.relu(self.d(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.relu(self.d(self.conv11(x)))
        x = F.relu(self.d(self.conv12(x)))
        x = F.adaptive_avg_pool2d(x, (4, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = self.bn(x)
        return x