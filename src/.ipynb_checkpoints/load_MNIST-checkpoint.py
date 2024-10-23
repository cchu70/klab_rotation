import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np

def load_MNIST(batch_size = 2048, validation_ratio=6, download=False, root='./data', subset_frac=None):
    train_data = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=ToTensor()
    )
    
    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=ToTensor()
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
