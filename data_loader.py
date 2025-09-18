"""
Data Loading and Preprocessing for MNIST Dataset
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_mnist_data(root="data", batch_size=64):
    """
    Load MNIST training and test datasets.
    
    Args:
        root (str): Root directory for storing the dataset
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_dataloader, test_dataloader, training_data, test_data)
    """
    # Download training data
    training_data = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data
    test_data = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, training_data, test_data


def get_data_info(dataloader):
    """
    Get information about the dataset.
    
    Args:
        dataloader (DataLoader): DataLoader to inspect
        
    Returns:
        dict: Dictionary containing dataset information
    """
    for x, y in dataloader:
        return {
            "batch_size": x.shape[0],
            "image_shape": x.shape[1:],  # (C, H, W)
            "label_shape": y.shape,
            "num_classes": len(torch.unique(y)),
            "total_samples": len(dataloader.dataset)
        }

