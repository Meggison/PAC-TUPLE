import os
import json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from random import choice, sample
from torch.utils.data import  Dataset,  ConcatDataset
import random
from random import choice, sample


def prepare_data_list(data_list_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(data_list_path, 'r') as f:
        data = json.load(f)
        print(data)
        print(len(data))
    image_list = []

    for item in data[0].get('train', []):
        image_path = item[0]
        image_path = image_path.replace('\\', '/')
        image_path = image_path.split('/')[-1]  # Get the filename only
        image_list.append(image_path)

    output_file = os.path.join(save_dir, 'train.txt')
    with open(output_file, 'w') as f:
        for img in image_list:
            f.write(f"{img}\n")

    print(f"Data list saved to {output_file}")

    # Process 'query' as 'test'
    query_list = []
    for item in data[0].get('query', []):
        image_path = item[0].replace('\\', '/')
        image_path = image_path.split('/')[-1]
        query_list.append(image_path)

    test_file = os.path.join(save_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for img in query_list:
            f.write(f"{img}\n")
    print(f"[✓] Saved test.txt to: {test_file}")

    return os.path.join(save_dir, 'train.txt')


def reid_data_prepare(data_list_path, train_dir_path):
    """
    Prepares Re-ID data by loading images, transforming them, and organizing them by class.

    This updated version skips any image files listed in the data_list_path that
    are not actually present in the train_dir_path.
    """
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2

    h, w = 224, 224

    # Define the image transformations
    transform_train_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_train_list)

    # Open the file containing the list of image paths
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img_filename = line

            # Determine the label based on the dataset name in the path
            if "cuhk01" in data_list_path:
                lbl = int(img_filename[:4])
            elif "cuhk03" in data_list_path:
                lbl = int(img_filename.split('_')[1])
            else:
                lbl = int(img_filename.split('_')[0])

            # Update class counter and dictionary for new labels
            if lbl != last_label:
                class_cnt += 1
                class_img_labels[str(class_cnt)] = []
            last_label = lbl

            # --- Start of updated block ---
            try:
                # Construct the full image path
                full_img_path = os.path.join(train_dir_path, img_filename)

                # Attempt to open the image
                img = Image.open(full_img_path)

                # If successful, transform and append the image
                img = transform(img)
                class_img_labels[str(class_cnt)].append(img)

                print(f"Loaded and transformed image: {full_img_path}")

            except FileNotFoundError:
                # If the file does not exist, print a warning (optional) and skip to the next image
                print(f"Warning: File not found at {full_img_path}. Skipping.")
                continue
            # --- End of updated block ---

    return class_img_labels


def ntuple_reid_data(class_img_labels, class_list, N=5, samples_per_class=5):
    """
    Create data for N-tuple loss:
    Each sample = (anchor, positive, N-2 negatives from different classes)

    Returns:
        anchors:  Tensor (B, C, H, W)
        positives: Tensor (B, C, H, W)
        negatives: Tensor (B, N-2, C, H, W)
    """
    anchors = []
    positives = []
    negatives = []

    if not class_img_labels:
        # Handle case where the entire input is empty
        return torch.empty(0), torch.empty(0), torch.empty(0)

    assert N >= 3, "N must be at least 3 (anchor + positive + 1 neg)"

    # --- Start of Fix ---
    # 1. Pre-filter the class list to get lists of valid classes for anchors and negatives.
    # Anchors/positives require at least 2 images per class.
    valid_anchor_classes = [cls for cls in class_list if class_img_labels.get(str(cls)) and len(class_img_labels[str(cls)]) >= 2]
    # Negatives require at least 1 image per class.
    valid_negative_classes = [cls for cls in class_list if class_img_labels.get(str(cls)) and len(class_img_labels[str(cls)]) >= N-2]

    # 2. Check if it's even possible to form an N-tuple with the available valid classes.
    # We need at least 1 anchor class and N-2 negative classes.
    if len(valid_negative_classes) < N - 1:
        print("Warning: Not enough classes with images to form N-tuples. Returning empty tensors.")
        return torch.empty(0), torch.empty(0), torch.empty(0)
    # --- End of Fix ---

    for cls in valid_anchor_classes:  #<-- Using 'cls' as requested.
        class_imgs = class_img_labels[str(cls)]

        # Get a list of all possible negative classes that are valid for this anchor.
        neg_classes = [c for c in valid_negative_classes if c != cls] #<-- Using 'neg_classes'.

        # Check if there are enough negative classes for this specific anchor.
        if len(neg_classes) < N - 2:
            continue

        for i in range(min(samples_per_class, len(class_imgs))):
            anchor = class_imgs[i]
            # pick a different positive from the same class
            pos_idx = choice([j for j in range(len(class_imgs)) if j != i])
            positive = class_imgs[pos_idx]

            negative_samples = []

            # --- Start of Fix ---
            # Use a while loop structure as in the original, but sample from the
            # pre-validated `neg_classes` list to ensure unique negative classes.
            neg_classes_to_sample_from = neg_classes.copy()
            while len(negative_samples) < (N - 2):
                neg_cls = choice(neg_classes_to_sample_from) #<-- Using 'neg_cls'. Safe because list is pre-validated.
                neg_classes_to_sample_from.remove(neg_cls) # Ensures we don't pick the same class twice.

                neg_imgs = class_img_labels[str(neg_cls)] #<-- Using 'neg_imgs'.
                neg_img = choice(neg_imgs) #<-- Using 'neg_img'.
                negative_samples.append(neg_img)
            # --- End of Fix ---

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(torch.stack(negative_samples))

    if not anchors:
        print("Warning: No valid N-tuples could be generated.")
        return torch.empty(0), torch.empty(0), torch.empty(0)

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives

def pair_pretrain_on_dataset(source, project_path='./', dataset_parent='./',val_perc=0.5):

  if source == 'market':
      train_list = project_path + source+ '/train.txt'
      train_dir = dataset_parent + source+ '/bounding_box_train'
      class_count = 750
      test_list = project_path + source+ '/test.txt'
      test_dir = dataset_parent + source+ '/bounding_box_test'

  elif source == 'cuhk03':
      train_list = os.path.join(data_dir, 'train.txt')
      train_dir = os.path.join(data_dir, 'images_labeled')
      class_count = None

      test_list = os.path.join(data_dir, 'test.txt')
      test_dir = os.path.join(data_dir, 'images_labeled')

  else:
      train_list = 'unknown'
      train_dir = 'unknown'
      class_count = -1

  class_img_labels = reid_data_prepare(train_list, train_dir)
  class_train = class_img_labels
  class_num = len(class_img_labels)

  if val_perc > 0: # set val data percentage
    class_val = sample(list(np.arange(len(class_img_labels))), int(len(class_img_labels)*val_perc))
    class_train = list(set(np.arange(len(class_img_labels))) - set(class_val))

    train =ntuple_reid_data(class_img_labels, class_train)
    print("loaded train data")
    val = ntuple_reid_data(class_img_labels, class_val)
    print("loaded validation data")

    class_test_dict = reid_data_prepare(test_list, test_dir)
    class_test = np.arange(len(class_test_dict))

    test = ntuple_reid_data(class_test_dict, class_test, train=False)

    if val:
        print("len train class:", len(train[1]),"len val class:", len(val[1]), "len test class:", len(test[1]))
    else:
        print("len train class:", len(train[1]),"len val class:", 0, "len test class:", len(test[1]))

    return train, val, test,class_img_labels, class_val,class_num
  
  import torch
from torch.utils.data import Dataset

class NTupleDataset(Dataset):
    """
    A Dataset class for pre-computed N-tuples.
    Assumes that anchors, positives, and negatives have already been sampled.
    """

    def __init__(self, anchors, positives, negatives):
        """
        Args:
            anchors (list or Tensor): A list/tensor of all anchor images.
            positives (list or Tensor): A list/tensor of all positive images.
            negatives (list or Tensor): A list/tensor of all negative image sets.
        """
        # Ensure all lists have the same length
        assert len(anchors) == len(positives) == len(negatives), \
            "All data lists must have the same length."

        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.data_len = len(anchors)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        Returns the pre-computed N-tuple at a given index.
        """
        anchor = self.anchors[index]
        positive = self.positives[index]
        negative_set = self.negatives[index]

        return anchor, positive, negative_set

import torch
from torch.utils.data import DataLoader, ConcatDataset

def loadbatches(train, val, test, loader_kargs, batch_size):
    """
    Function to load the batches for the dataset.
    This version works with any standard PyTorch Dataset object, including PrecomputedNTupleDataset.

    Parameters
    ----------
    train : torch.utils.data.Dataset
        Train dataset object (e.g., an instance of PrecomputedNTupleDataset).
    val : torch.utils.data.Dataset
        Validation dataset object.
    test : torch.utils.data.Dataset
        Test dataset object.
    loader_kargs : dict
        Loader arguments (e.g., num_workers, pin_memory).
    batch_size : int
        The size of the batch.
    """

    # Use the standard len() function which works with PyTorch Datasets
    ntrain = len(train)
    ntest = len(test)
    print(f"Train data length: {ntrain}, Test data length: {ntest}")

    # Initialize all loaders to None
    train_loader, prior_loader, set_bound_1batch, set_val_bound = None, None, None, None

    if val:
        concat_data = ConcatDataset([train, val])

        # Main loader for training on both train and validation sets
        train_loader = DataLoader(concat_data, batch_size=batch_size, shuffle=True, **loader_kargs)
        # Loader for the validation/prior set
        prior_loader = DataLoader(val, batch_size=batch_size, **loader_kargs)
        # Single-batch loader for the train set
        set_bound_1batch = DataLoader(train, batch_size=ntrain, **loader_kargs)
        # Standard-batch loader for the train set (for validation-like calculations)
        set_val_bound = DataLoader(train, batch_size=batch_size)
    else:
        # If no validation set, the train_loader only uses the training data
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_kargs)


    # Standard and single-batch loaders for the test set
    test_1batch = DataLoader(test, batch_size=ntest, **loader_kargs)
    test_loader = DataLoader(test, batch_size=batch_size, **loader_kargs)


    return train_loader, test_loader, prior_loader, set_bound_1batch, test_1batch, set_val_bound

import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaLearner(nn.Module):
    """
    Implements the meta-learner subnet phi(·) from Equation (7) of the paper.
    It takes instance features and maps them to refined reference nodes.

    Args:
        embedding_dim (int): The dimension of the input feature embeddings (d).
        reduction_ratio (int): The ratio for dimension reduction in the bottleneck layer.
    """
    def __init__(self, embedding_dim=1024, reduction_ratio=8):
        super(MetaLearner, self).__init__()
        bottleneck_dim = embedding_dim // reduction_ratio

        self.mapper = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            # The paper does not specify an activation, but ReLU is a common choice.
            # nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, embedding_dim)
        )

    def forward(self, x):
        return self.mapper(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaLearner(nn.Module):
    """
    Implements the meta-learner subnet phi(·) from Equation (7) of the paper.
    It takes instance features and maps them to refined reference nodes.

    Args:
        embedding_dim (int): The dimension of the input feature embeddings (d).
        reduction_ratio (int): The ratio for dimension reduction in the bottleneck layer.
    """
    def __init__(self, embedding_dim=1024, reduction_ratio=8):
        super(MetaLearner, self).__init__()
        bottleneck_dim = embedding_dim // reduction_ratio

        self.mapper = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            # The paper does not specify an activation, but ReLU is a common choice.
            # nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, embedding_dim)
        )

    def forward(self, x):
        return self.mapper(x)


class NTupleLoss(nn.Module):
    """
    Implementation of N-tuple and Meta Prototypical N-tuple (MPN-tuple) loss.

    Args:
        mode (str): The loss mode. Must be one of 'regular' or 'mpn'.
                    - 'regular': Standard N-tuple loss using instance features directly.
                    - 'mpn': Meta Prototypical N-tuple loss using a meta-learner.
        embedding_dim (int): The dimension of the feature embeddings.
                             Required only if mode is 'mpn'.
        initial_temp (float): The initial temperature (tau) for scaling similarities.
    """
    def __init__(self, mode='mpn', embedding_dim=1024, initial_temp=0.05):
        super(NTupleLoss, self).__init__()

        if mode not in ['regular', 'mpn']:
            raise ValueError("Mode must be either 'regular' or 'mpn'")
        self.mode = mode

        # The paper makes the temperature a learnable parameter by learning s = 1/tau
        # We will do the same for flexibility.
        self.log_s = nn.Parameter(torch.log(torch.tensor(1.0 / initial_temp)))

        if self.mode == 'mpn':
            self.meta_learner = MetaLearner(embedding_dim=embedding_dim)

    def forward(self, anchor_embed, positive_embed, negative_embeds):
        """
        Calculates the N-tuple loss.

        Args:
            anchor_embed (torch.Tensor): Embeddings of the anchor samples.
                                         Shape: (batch_size, embedding_dim)
            positive_embed (torch.Tensor): Embeddings of the positive samples.
                                          Shape: (batch_size, embedding_dim)
            negative_embeds (torch.Tensor): Embeddings of the negative samples.
                                           Shape: (batch_size, N-2, embedding_dim)

        Returns:
            torch.Tensor: The calculated N-tuple loss for the batch.
        """
        # Get the reference nodes for positive and negative samples
        if self.mode == 'mpn':
            # For MPN loss, pass positives and negatives through the meta-learner
            # to get the reference nodes (prototypes).
            # The paper averages multiple instances for a prototype; here we assume
            # the provided single positive/negative is the basis for its prototype.
            positive_ref = self.meta_learner(positive_embed)

            # Reshape negatives to pass through the linear layers of the meta-learner
            batch_size, n_negatives, embed_dim = negative_embeds.shape
            negatives_flat = negative_embeds.view(-1, embed_dim)
            negative_ref_flat = self.meta_learner(negatives_flat)
            negative_ref = negative_ref_flat.view(batch_size, n_negatives, embed_dim)

        else: # 'regular' mode
            # For regular N-tuple loss, the instance embeddings are the reference nodes.
            positive_ref = positive_embed
            negative_ref = negative_embeds

        # --- Calculate similarities ---
        # Cosine similarity is used as per the paper
        sim_positive = F.cosine_similarity(anchor_embed, positive_ref)

        # To calculate similarity between anchor and all negatives, we need to unsqueeze
        # the anchor to enable broadcasting across the N-2 dimension.
        # anchor_embed shape: (B, D) -> (B, 1, D)
        # negative_ref shape: (B, N-2, D)
        sim_negatives = F.cosine_similarity(anchor_embed.unsqueeze(1), negative_ref, dim=2)

        # --- Formulate as a classification problem ---
        # The goal is to classify the anchor as belonging to the positive reference
        # over all negative references. This can be solved with CrossEntropyLoss.

        # The logits are the scaled similarities.
        # Concatenate the positive similarity with all negative similarities.
        # Shape: (B, 1+ (N-2)) -> (B, N-1)
        logits = torch.cat([sim_positive.unsqueeze(1), sim_negatives], dim=1)

        # Scale logits by the learned temperature parameter s = 1/tau
        logits *= torch.exp(self.log_s)

        # The target label for every sample is 0, because the positive class
        # is always at index 0 of our logits tensor.
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Calculate the cross-entropy loss, which is equivalent to the
        loss = F.cross_entropy(logits, targets)

        return loss

class DynamicNTupleDataset(Dataset):
    """
    Dataset that samples N-tuples dynamically to save memory.
    It creates tuples on-the-fly in the __getitem__ method.
    """
    def __init__(self, class_img_labels, class_ids, N=4, samples_per_epoch_multiplier=4):
        self.class_img_labels = class_img_labels
        self.class_ids = class_ids  # The list of class IDs for this split (e.g., train_ids)
        self.N = N
        self.samples_per_epoch_multiplier = samples_per_epoch_multiplier

        # Create a flat list of all (image_tensor, class_id) pairs for this dataset split.
        # These are the potential anchors. We only include images from classes that
        # have at least 2 images, so a positive pair can always be formed.
        self.anchor_pool = []
        for cid in self.class_ids:
            cid_str = str(cid)
            if cid_str in self.class_img_labels and len(self.class_img_labels[cid_str]) >= 2:
                for img in self.class_img_labels[cid_str]:
                    self.anchor_pool.append({'img': img, 'cid': cid})
        
        if not self.anchor_pool:
            raise ValueError("No classes with enough images to form anchor-positive pairs.")

        # The length is the number of available anchors multiplied by a factor to control epoch size.
        self.length = len(self.anchor_pool) * self.samples_per_epoch_multiplier

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Use modulo to cycle through the anchor pool
        anchor_info = self.anchor_pool[index % len(self.anchor_pool)]
        anchor_img = anchor_info['img']
        anchor_cid = anchor_info['cid']

        # 1. Sample a positive image from the same class
        positive_options = self.class_img_labels[str(anchor_cid)]
        positive_img = random.choice(positive_options)
        # Ensure the positive isn't the exact same tensor instance as the anchor
        if len(positive_options) > 1:
            while torch.equal(anchor_img, positive_img):
                positive_img = random.choice(positive_options)

        # 2. Sample N-2 negatives from different classes
        # Get a list of all possible negative class IDs that have at least one image
        possible_neg_cids = [cid for cid in self.class_ids if cid != anchor_cid and str(cid) in self.class_img_labels and self.class_img_labels[str(cid)]]
        
        # Check if enough unique negative classes are available
        if len(possible_neg_cids) < self.N - 2:
            # Fallback: if not enough unique classes, sample with replacement.
            # This is unlikely in a large dataset but makes the code robust.
            neg_cids_sample = random.choices(possible_neg_cids, k=self.N - 2)
        else:
            neg_cids_sample = random.sample(possible_neg_cids, self.N - 2)
            
        negative_imgs = [random.choice(self.class_img_labels[str(c)]) for c in neg_cids_sample]
        negatives_tensor = torch.stack(negative_imgs)
        
        return anchor_img, positive_img, negatives_tensor
