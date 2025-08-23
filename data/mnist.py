import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import random
from torchvision import transforms

class MNISTTupleDataset(Dataset):
    
    def __init__(self, train=True, samples_per_class=1000):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())

        self.class_data = {}
        for i, (img, label) in enumerate(self.mnist):
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(img)

            # Enaure minimum samples per class
            self.valid_clases = [k for k, v in self.class_data.items() if len(v) >= 2]

            self.samples_per_clas = samples_per_class


    def __len__(self):
        return self.samples_per_class * len(self.valid_clases)
    
    def __getitem__(self, idx):
        # choose anchor class
        anchor_class = self.valid_clases[idx % len(self.valid_clases)]

        # sample anchor and positive images
        anchor_img, _ = random.choice(self.class_data[anchor_class])
        positive_img, _ = random.choice(self.class_data[anchor_class])

        # ensure anchor and postive are same clas but different images
        while torch.equal(anchor_img, positive_img):
            positive_img, _ = random.choice(self.class_data[anchor_class])

        # sample N-2 negatives from different classes
        negative_classes = [c for c in self.valid_clases if c != anchor_class]
        negative_class = random.choice(negative_classes)
        negative_img, _ = random.choice(self.class_data[negative_class])

        return anchor_img, positive_img, negative_img