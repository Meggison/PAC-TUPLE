import os
import json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from random import choice, sample
from torch.utils.data import  Dataset, ConcatDataset, DataLoader
import random
from random import choice, sample

def reid_data_prepare(data_dir, train_dir):
    """Prepare the data for training and testing."""

    class_img_labels = dict()
    class_count = -1
    last_label = -2

    h, w = 256, 128 # Image size for re-identification tasks

    transform_train_list = [
        transforms.Resize((h, w),
        interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization for ImageNet
    ]

    transform = transforms.Compose(transform_train_list)

    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            img_filename = line.strip()

          # Parse label by dataset conventions
            if "cuhk01" in data_dir:
                lbl = int(img_filename[:4])
            elif "cuhk03" in data_dir:
                lbl = int(img_filename.split('_')[1])
            else:
                lbl = int(img_filename.split('_')[0])

            # Group images by label, increment for new class
            if lbl != last_label:
                class_cnt += 1
                class_img_labels[str(class_cnt)] = []
            last_label = lbl

            try:
                img_path = os.path.join(train_dir, img_filename)
                img = Image.open(img_path).convert('RGB')

                img.transform = transform(img)
                class_img_labels[str(class_cnt)].append((img))

                print(f"Processed image: {img_path}, Label: {lbl}")

            
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
            
    return class_img_labels

class DynamicNTupleDataset(Dataset):
    """
    Dataset that samples N-tuples dynamically to save memory.
    It creates tuples on-the-fly in the __getitem__ method.
    """
    def __init__(self, class_img_labels, class_ids, N=3, samples_per_epoch_muliplier=4):
        self.class_img_labels = class_img_labels
        self.class_ids = class_ids
        self.N = N
        self.samples_per_epoch_muliplier = samples_per_epoch_muliplier

        # have at least 2 images, so a positive pair can always be formed.
        self.anchor_pool = []
        for cid in self.class_ids:
            cid_str = str(cid)
            if cid_str in self.class_img_labels and len(self.class_img_labels[cid_str]) >= 2:
                for img in self.class_img_labels[cid_str]:
                    self.anchor_pool.append({'img': img, 'cid': cid})
        
        if not self.anchor_pool:
            raise ValueError("No classes with enough images to form anchor-positive pairs.")
        
    def __len__(self):
        # Return a large number to allow for dynamic sampling
        return len(self.anchor_pool) * self.samples_per_epoch_muliplier
    
    def __getitem__(self, index):
        # Use modulo to cycle through the anchor pool
        anchor_info = self.anchor_pool[index % len(self.anchor_pool)]
        anchor_img = anchor_info['img']
        anchor_cid = anchor_info['cid']

        # Sample a positive image from the same class
        positive_options = self.class_img_labels[str(anchor_cid)]
        positive_img = random.choice(positive_options)
        # Ensure the positive isn't the exact same tensor instance as the anchor
        if len(positive_options) > 1:
            while torch.equal(anchor_img, positive_img):
                positive_img = random.choice(positive_options)

        # Sample N-2 negatives from different classes
        # Get a list of all possible negative class IDs that have at least one image
        possible_neg_cids = [cid for cid in self.class_ids if cid != anchor_cid and str(cid) in self.class_img_labels and self.class_img_labels[str(cid)]]
        
        # Check if enough unique negative classes are available
        if len(possible_neg_cids) < self.N - 2:
            # Fallback: if not enough unique classes, sample with replacement.
            neg_cids_sample = random.choices(possible_neg_cids, k=self.N - 2)
        else:
            neg_cids_sample = random.sample(possible_neg_cids, self.N - 2)
            
        negative_imgs = [random.choice(self.class_img_labels[str(c)]) for c in neg_cids_sample]
        negatives_tensor = torch.stack(negative_imgs)
        
        return anchor_img, positive_img, negatives_tensor




class MetaPrototypicalNtupleDataset(Dataset):
    """
    Dataset to support Meta Prototypical N-tuple (episodic) training:
    - Each batch (episode) samples P classes (N-way).
    - For each class, K images are sampled for support/query.
      -- Typically, 1 is used as query, remaining for support (standard K-shot).
    - Encodes all needed structure for prototypical or meta-learning loss.
    """

    def __init__(self, class_img_labels, class_ids, P=4, K=4, query_per_class=1, samples_per_epoch=1000):
        """
        Args:
            class_img_labels: dict mapping str(class_id) -> list of image tensors
            class_ids: list of class IDs to sample from (as str or int)
            P: number of classes per episode (N-way)
            K: samples per class per episode (K-shot total; usually K-1 support, 1 query)
            query_per_class: how many queries per class in episode
            samples_per_epoch: number of episodes per training epoch
        """
        self.class_img_labels = class_img_labels
        self.class_ids = class_ids
        self.P = P
        self.K = K
        self.query_per_class = query_per_class
        self.samples_per_epoch = samples_per_epoch

        # Ensure there are enough images per class for sampling
        assert all(len(class_img_labels[str(cid)]) >= self.K for cid in self.class_ids), \
            "Each class must have at least K images."

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. Randomly sample P classes for this episode
        episode_classes = random.sample(self.class_ids, self.P)

        support_imgs = []
        support_labels = []
        query_imgs = []
        query_labels = []

        for cls in episode_classes:
            # 2. Randomly sample K images from the class
            imgs = random.sample(self.class_img_labels[str(cls)], self.K)
            # Split into queries and support images
            query = imgs[:self.query_per_class]
            support = imgs[self.query_per_class:]

            support_imgs.extend(support)
            support_labels.extend([cls] * len(support))
            query_imgs.extend(query)
            query_labels.extend([cls] * len(query))

        support_imgs = torch.stack(support_imgs)   # Shape: [P * (K-query), C, H, W]
        support_labels = torch.tensor(support_labels)
        query_imgs = torch.stack(query_imgs)       # Shape: [P * query, C, H, W]
        query_labels = torch.tensor(query_labels)

        return support_imgs, support_labels, query_imgs, query_labels

