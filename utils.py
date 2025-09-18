"""
Utility Functions for Visualization and Model Evaluation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_batch(images, labels, num_images=64):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): Corresponding labels
        num_images (int): Number of images to display (default: 64)
    """
    # Determine grid size
    grid_size = int(np.sqrt(num_images))
    if grid_size * grid_size < num_images:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(images, labels, predicted_labels, num_images=64):
    """
    Visualize a batch of images with their true and predicted labels.
    Titles are green if the prediction is correct, red if incorrect.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): True labels
        predicted_labels (torch.Tensor): Predicted labels
        num_images (int): Number of images to display (default: 64)
    """
    # Determine grid size
    grid_size = int(np.sqrt(num_images))
    if grid_size * grid_size < num_images:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if labels[i].item() == predicted_labels[i].item() else 'red'
        axes[i].set_title(f'True: {labels[i].item()}\nPred: {predicted_labels[i].item()}', color=color)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, device):
    """
    Plot the confusion matrix for a given model and dataloader.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Test dataloader
        device (str): Device to run inference on
    """
    # Initialize the confusion matrix
    conf_mat = torch.zeros((10, 10))
    total_correct = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_labels = torch.argmax(pred, axis=1)
            
            total_samples += pred.shape[0]
            total_correct += sum(pred_labels == y).item()
            
            for j in range(pred.shape[0]):
                conf_mat[y[j], pred_labels[j].item()] += 1
    
    # Calculate the normalized confusion matrix
    norm_conf_mat = conf_mat / torch.sum(conf_mat, axis=1, keepdim=True)
    
    # Plot the matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_conf_mat, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels and title
    ax.set_title(f'Confusion Matrix (Accuracy: {100 * total_correct / total_samples:.2f}%)')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            count = conf_mat[i, j].item()
            color = 'white' if count > conf_mat.max() / 2 else 'black'
            ax.text(j, i, f'{int(count)}', ha='center', va='center', color=color)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(losses, accuracies=None, title="Training History"):
    """
    Plot training loss and accuracy curves.
    
    Args:
        losses (list): List of training losses
        accuracies (list, optional): List of training accuracies
        title (str): Plot title
    """
    fig, axes = plt.subplots(1, 2 if accuracies else 1, figsize=(12, 5))
    
    if accuracies:
        axes[0].plot(losses, label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(accuracies, label='Training Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes.plot(losses, label='Training Loss')
        axes.set_title('Training Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()
        axes.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

