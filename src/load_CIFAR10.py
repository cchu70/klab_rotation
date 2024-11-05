"""
Copied from https://www.digitalocean.com/community/tutorials/alexnet-pytorch
"""
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_loader(
    data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True, download=True, subsample_frac=1.0
):

    # Normalize over the mean and std of each channel (red, green, blue) across the whole dataset. Already calculated 
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        # increase the dataset size with more transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=download, transform=valid_transform
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    sub_train_idx = np.random.choice(
        train_idx, 
        size=int(len(train_idx) * subsample_frac),
    )

    sub_valid_idx = np.random.choice(
        valid_idx, 
        size=int(len(valid_idx) * subsample_frac),
    )
    
    train_sampler = SubsetRandomSampler(sub_train_idx)
    valid_sampler = SubsetRandomSampler(sub_valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)

def get_test_loader(
    data_dir, batch_size, shuffle=True, download=True
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

    

    

     