"""
Training Functions for MNIST Neural Network
"""

import torch
from torch import nn
from tqdm import trange
import matplotlib.pyplot as plt


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): Neural network model
        dataloader (DataLoader): Training data loader
        loss_fn (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to run training on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in dataloader:
        # Move data to device
        x, y = x.to(device), y.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred_labels = torch.argmax(pred, axis=1)
        correct += sum(pred_labels == y).item()
        total += pred.shape[0]
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_dataloader, loss_fn, optimizer, device, n_epochs=30, verbose=True):
    """
    Train the model for multiple epochs.
    
    Args:
        model (torch.nn.Module): Neural network model
        train_dataloader (DataLoader): Training data loader
        loss_fn (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to run training on
        n_epochs (int): Number of epochs to train
        verbose (bool): Whether to show progress bar
        
    Returns:
        tuple: (losses, accuracies)
    """
    losses = []
    accuracies = []
    
    if verbose:
        pbar = trange(n_epochs, desc="Training")
    else:
        pbar = range(n_epochs)
    
    for epoch in pbar:
        loss, acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        losses.append(loss)
        accuracies.append(acc)
        
        if verbose:
            pbar.set_description(f'Epoch {epoch+1}/{n_epochs} - Loss: {loss:.3f}, Acc: {acc:.2f}%')
    
    return losses, accuracies


def compare_optimizers(model_class, train_dataloader, loss_fn, device, optimizers_config, n_epochs=10):
    """
    Compare different optimizers on the same model.
    
    Args:
        model_class: Model class to instantiate
        train_dataloader (DataLoader): Training data loader
        loss_fn (torch.nn.Module): Loss function
        device (str): Device to run training on
        optimizers_config (dict): Dictionary mapping optimizer names to their configurations
        n_epochs (int): Number of epochs to train each optimizer
        
    Returns:
        dict: Results for each optimizer
    """
    results = {}
    
    for optimizer_name, optimizer_config in optimizers_config.items():
        print(f"\nTraining with {optimizer_name} optimizer...")
        
        # Create fresh model
        model = model_class().to(device)
        
        # Create optimizer
        optimizer = optimizer_config['class'](model.parameters(), **optimizer_config['params'])
        
        # Train model
        losses, accuracies = train_model(
            model, train_dataloader, loss_fn, optimizer, device, n_epochs, verbose=True
        )
        
        results[optimizer_name] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1]
        }
    
    return results


def plot_optimizer_comparison(results):
    """
    Plot comparison of different optimizers.
    
    Args:
        results (dict): Results from compare_optimizers function
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for optimizer_name, result in results.items():
        losses = result['losses']
        accuracies = result['accuracies']
        
        axes[0].plot(losses, label=f"{optimizer_name} (Final: {result['final_loss']:.3f})")
        axes[1].plot(accuracies, label=f"{optimizer_name} (Final: {result['final_accuracy']:.2f}%)")
    
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

