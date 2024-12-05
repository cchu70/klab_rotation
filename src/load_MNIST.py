import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np

def load_MNIST(batch_size = 2048, validation_ratio=6, download=False, root='./data', subset_frac=None, transform=ToTensor(), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_data = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )
    
    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

    if subset_frac:
        subset_train_idx = np.random.choice(np.arange(len(train_data)), size=int(len(train_data) * subset_frac))
        subset_test_idx = np.random.choice(np.arange(len(test_data)), size=int(len(test_data) * subset_frac))
        
        train_data = torch.utils.data.Subset(train_data, subset_train_idx)
        test_data = torch.utils.data.Subset(test_data, subset_test_idx)
    
    num_val = int(len(train_data) / validation_ratio)
    val_data, train_data = torch.utils.data.random_split(train_data, [num_val, int(len(train_data) - num_val)])
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

### Claude 3.5 Sonnet
"""
Prompt: I want to generate a dataloader in pytorch where the data is pairs of images from the MNIST dataset, and the label is 0 if its the same label and 1 if they are not the same label. How would I implement this?

Followed up with: This seems to load data fairly slowly. Is there a faster implementation?

Next follow up: I do not want to store the full mnist dataset in the self.images and self.labels attributes. I just want a subset (say, randomly select 10% of the dataset). How would you implement this?
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

class MNISTPairs(Dataset):
    def __init__(self, root='./', train=True, transform=None, download=True, 
                 subset_fraction=0.1, num_pairs_per_image=2, seed=42, selected_labels=None):
        """
        Dataset of MNIST pairs with labels indicating if they're the same digit
        Args:
            root: Root directory for MNIST data
            train: If True, use training set, else test set
            transform: Optional transform to be applied to both images
            download: If True, download MNIST dataset
            subset_fraction: Fraction of dataset to use (between 0 and 1)
            num_pairs_per_image: Number of pairs to generate per image
            seed: Random seed for reproducibility
        """
        self.mnist = datasets.MNIST(root=root, train=train, 
                                  transform=None, download=download)
        self.transform = transform
        
        # Set random seed for reproducibility
        np.random.seed(seed)

        # select labels
        if selected_labels is not None:
            # Create masks for each selected label
            label_masks = [(self.mnist.targets == label).numpy() for label in selected_labels]
            # Combine masks using logical OR
            combined_mask = np.zeros_like(self.mnist.targets, dtype=bool)
            for mask in label_masks:
                combined_mask |= mask
            
            # Apply mask to get indices of selected labels
            subset_indices = np.where(combined_mask)[0]
        else:
            # If no labels specified, use all data
            subset_indices = np.arange(len(self.mnist))
        
        # Randomly select subset of indices
        total_samples = len(subset_indices)
        subset_size = int(total_samples * subset_fraction)
        subset_indices = np.random.choice(
            subset_indices, 
            size=subset_size, 
            replace=False
        )
        
        # Store only the subset of data
        self.images = self.mnist.data[subset_indices].numpy()
        self.labels = self.mnist.targets[subset_indices].numpy()
        
        # Pre-generate pairs for the subset
        self.pairs, self.pair_labels = self._generate_pairs(num_pairs_per_image)
        
        # Reset random seed
        np.random.seed(None)
        
    def _generate_pairs(self, num_pairs_per_image):
        """
        Pre-generates all pairs and their labels for the subset
        """
        n_samples = len(self.images)
        n_pairs = n_samples * num_pairs_per_image
        
        pairs = []
        labels = np.zeros(n_pairs, dtype=np.float32)
        
        # Create label-to-indices mapping for efficient pair generation
        label_to_indices = {label: np.where(self.labels == label)[0] 
                          for label in np.unique(self.labels)}
        
        for i in range(n_samples):
            current_label = self.labels[i]
            
            # Get indices for same and different classes
            same_class = label_to_indices[current_label]
            same_class = same_class[same_class != i]  # Remove current index
            
            # Get all indices for different classes
            diff_class_indices = np.concatenate([
                indices for label, indices in label_to_indices.items()
                if label != current_label
            ])
            
            # Generate pairs for current image
            for j in range(num_pairs_per_image):
                idx = i * num_pairs_per_image + j
                
                # Randomly decide if we want a same (0) or different (1) pair
                if random.random() < 0.5 and len(same_class) > 0:
                    # Same class pair
                    idx2 = np.random.choice(same_class)
                    labels[idx] = 0
                else:
                    # Different class pair
                    idx2 = np.random.choice(diff_class_indices)
                    labels[idx] = 1
                
                pairs.append((i, idx2))
        
        return np.array(pairs), labels

    def __len__(self):
        return len(self.pair_labels)

    def __getitem__(self, idx):
        """
        Returns a pair of images and their similarity label
        """
        idx1, idx2 = self.pairs[idx]
        
        # Get images
        img1 = self.images[idx1]
        img2 = self.images[idx2]
        
        # Apply transform if specified
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            # Convert to tensor if no transform
            img1 = torch.FloatTensor(img1.copy())
            img2 = torch.FloatTensor(img2.copy())
        
        return (img1, img2), torch.tensor(self.pair_labels[idx], dtype=torch.float32)

# Example usage:
def get_mnist_pairs_loader(batch_size=32, train=True, subset_fraction=0.1, validation_ratio=None, seed=None, selected_labels=None):
    """
    Creates a DataLoader for MNIST pairs
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    dataset = MNISTPairs(
        root='./data',
        train=train,
        transform=transform,
        download=True,
        subset_fraction=subset_fraction,
        seed=seed,
        selected_labels=selected_labels,
    )

    if train and validation_ratio:
        if seed is not None:
            torch.manual_seed(seed)
        np.random.seed(seed)

        num_val = int(len(dataset) / validation_ratio)
        val_data, train_data = torch.utils.data.random_split(dataset, [num_val, int(len(dataset) - num_val)])

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        return train_dataloader, val_dataloader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

