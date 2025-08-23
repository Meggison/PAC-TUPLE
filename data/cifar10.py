import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import random
from torchvision import transforms


class CIFAR10TupleDataset(Dataset):

    def __init__(self, train=True, samples_per_class=1000, transform=None):
        # Defin default transform if none provided
        if transform is None:
            if train:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 statistics
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

        self.cifar10 = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

        # Group images by class
        self.class_data = {}
        for img, label in self.cifar10:
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(img)

        # Ensure minimum samples per class
        self.valid_classes = [k for k, v in self.class_data.items() if len(v) >= 2]
        self.samples_per_class = samples_per_class

        # CIFAR clas names for reference (for convenience)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def __len__(self):
        return self.samples_per_class * len(self.valid_classes)
    
    def __getitem__(self, idx):
        # Choose anchor class
        anchor_class = self.valid_classes[idx % len(self.valid_classes)]

        # Sample anchor and positive images
        anchor_img = random.choice(self.class_data[anchor_class])
        positive_img = random.choice(self.class_data[anchor_class])

        # Ensure anchor and positive are same class but different images
        attempts = 0
        while torch.equal(anchor_img, positive_img) and attempts < 10:
            positive_img = random.choice(self.class_data[anchor_class])
            attempts += 1
            
        # Sample N-2 negatives from different classes
        negative_classes = [c for c in self.valid_classes if c != anchor_class]
        negative_class = random.choice(negative_classes)
        negative_img = random.choice(self.class_data[negative_class])

        return anchor_img, positive_img, negative_img
    
    def get_class_name(self, class_id):
        """Get class name by ID."""
        return self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "Unknown Class"
        

# Alternative version for multiple negatives (N-tuple)
class CIFAR10NTupleDataset(Dataset):
    """CIFAR-10 N-tuple dataset for metric learning with multiple negatives"""
    
    def __init__(self, train=True, num_negatives=3, samples_per_class=1000, transform=None):
        if transform is None:
            if train:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        
        self.cifar10 = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        self.num_negatives = num_negatives
        
        # Group images by class
        self.class_data = {}
        for i, (img, label) in enumerate(self.cifar10):
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(img)
        
        # Ensure minimum samples per class
        self.valid_classes = [k for k, v in self.class_data.items() if len(v) >= 2]
        self.samples_per_class = samples_per_class
        
        # Check if we have enough classes for negatives
        if len(self.valid_classes) < num_negatives + 1:  # +1 for anchor class
            raise ValueError(f"Not enough classes ({len(self.valid_classes)}) for {num_negatives} negatives")
    
    def __len__(self):
        return self.samples_per_class * len(self.valid_classes)
    
    def __getitem__(self, idx):
        # Choose anchor class
        anchor_class = self.valid_classes[idx % len(self.valid_classes)]
        
        # Sample anchor and positive from same class
        anchor_img = random.choice(self.class_data[anchor_class])
        positive_img = random.choice(self.class_data[anchor_class])
        
        # Ensure different images
        attempts = 0
        while torch.equal(anchor_img, positive_img) and len(self.class_data[anchor_class]) > 1 and attempts < 10:
            positive_img = random.choice(self.class_data[anchor_class])
            attempts += 1
        
        # Sample multiple negatives from different classes
        negative_classes = [c for c in self.valid_classes if c != anchor_class]
        
        if len(negative_classes) >= self.num_negatives:
            # Sample without replacement if we have enough classes
            selected_neg_classes = random.sample(negative_classes, self.num_negatives)
        else:
            # Sample with replacement if not enough classes
            selected_neg_classes = random.choices(negative_classes, k=self.num_negatives)
        
        negative_imgs = []
        for neg_class in selected_neg_classes:
            neg_img = random.choice(self.class_data[neg_class])
            negative_imgs.append(neg_img)
        
        # Stack negatives into tensor [num_negatives, C, H, W]
        negatives_tensor = torch.stack(negative_imgs)
        
        # Return anchor, positive, negatives, and anchor class label
        return anchor_img, positive_img, negatives_tensor, anchor_class
    