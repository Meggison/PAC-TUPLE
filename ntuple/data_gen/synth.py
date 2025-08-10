from torch.utils.data import Dataset
import torch
import random


class SyntheticTripletDataset(Dataset):
    """Synthetic dataset where you control the theoretical properties"""
    
    def __init__(self, num_classes=10, samples_per_class=100, embedding_dim=64, 
                 class_separation=2.0, intra_class_std=0.5):
        """
        Create synthetic embeddings with controlled properties
        - class_separation: distance between class centers
        - intra_class_std: standard deviation within classes
        """
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.embedding_dim = embedding_dim
        
        # Create class centers in embedding space
        self.class_centers = torch.randn(num_classes, embedding_dim) * class_separation
        
        # Generate all embeddings
        self.embeddings = []
        self.labels = []
        
        for class_id in range(num_classes):
            center = self.class_centers[class_id]
            for _ in range(samples_per_class):
                # Sample around class center
                embedding = center + torch.randn(embedding_dim) * intra_class_std
                self.embeddings.append(embedding)
                self.labels.append(class_id)
        
        self.embeddings = torch.stack(self.embeddings)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        anchor_embed = self.embeddings[idx]
        anchor_label = self.labels[idx]
        
        # Find positive (same class, different sample)
        same_class_indices = torch.where(self.labels == anchor_label)[0]
        same_class_indices = same_class_indices[same_class_indices != idx]
        pos_idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))]
        positive_embed = self.embeddings[pos_idx]
        
        # Find negative (different class)
        diff_class_indices = torch.where(self.labels != anchor_label)[0]
        neg_idx = diff_class_indices[torch.randint(0, len(diff_class_indices), (1,))]
        negative_embed = self.embeddings[neg_idx]
        
        return anchor_embed.squeeze(), positive_embed.squeeze(), negative_embed.squeeze()
    
    # test function

synth_data = SyntheticTripletDataset(num_classes=5, samples_per_class=10)
print(f"Dataset size: {len(synth_data)}")
for i in range(1):
    anchor, positive, negative = synth_data[i]
    print(f"Anchor: {anchor.shape}, Positive: {positive.shape}, Negative: {negative.shape}")