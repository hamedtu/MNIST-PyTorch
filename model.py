"""
Neural Network Model Definition for MNIST Classification
"""

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for MNIST digit classification.
    
    Architecture:
    - Input: 28x28 grayscale images (flattened to 784 features)
    - Hidden layers: Two fully connected layers with 512 neurons each
    - Activation: ReLU
    - Output: 10 classes (digits 0-9)
    """
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_model(device=None):
    """
    Create and initialize a NeuralNetwork model.
    
    Args:
        device (str, optional): Device to move the model to. If None, uses get_device()
        
    Returns:
        NeuralNetwork: Initialized model on the specified device
    """
    if device is None:
        device = get_device()
    
    model = NeuralNetwork().to(device)
    return model

