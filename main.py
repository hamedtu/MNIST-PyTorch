"""
Main Pipeline for MNIST Neural Network Training and Evaluation

This script provides a complete pipeline for training and evaluating
a neural network on the MNIST handwritten digits dataset.
"""

import torch
from torch import nn
import argparse
import os

from model import NeuralNetwork, get_device
from data_loader import load_mnist_data
from train import train_model, compare_optimizers, plot_optimizer_comparison
from evaluate import evaluate_model, print_evaluation_summary, get_model_predictions
from utils import visualize_batch, visualize_predictions, plot_confusion_matrix, plot_training_history


def main():
    """Main function to run the MNIST training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST Neural Network')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store MNIST data')
    parser.add_argument('--compare_optimizers', action='store_true', help='Compare different optimizers')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_dataloader, test_dataloader, training_data, test_data = load_mnist_data(
        root=args.data_dir, batch_size=args.batch_size
    )
    
    print(f"Training samples: {len(training_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create model
    print("Creating neural network model...")
    model = NeuralNetwork().to(device)
    print(f"Model architecture:\n{model}")
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    # Visualize training data if requested
    if args.visualize:
        print("Visualizing training data...")
        images, labels = next(iter(train_dataloader))
        visualize_batch(images, labels)
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    losses, accuracies = train_model(
        model, train_dataloader, loss_fn, optimizer, device, 
        n_epochs=args.epochs, verbose=True
    )
    
    # Plot training history if requested
    if args.visualize:
        plot_training_history(losses, accuracies, "Training History")
    
    # Evaluate model
    print("\nEvaluating model...")
    print_evaluation_summary(model, test_dataloader, loss_fn, device)
    
    # Visualize predictions if requested
    if args.visualize:
        print("Visualizing predictions on test data...")
        images, true_labels, pred_labels = get_model_predictions(model, test_dataloader, device, 64)
        visualize_predictions(images, true_labels, pred_labels)
        
        print("Plotting confusion matrix...")
        plot_confusion_matrix(model, test_dataloader, device)
    
    # Compare optimizers if requested
    if args.compare_optimizers:
        print("\nComparing different optimizers...")
        optimizers_config = {
            'SGD': {
                'class': torch.optim.SGD,
                'params': {'lr': args.lr}
            },
            'Adam': {
                'class': torch.optim.Adam,
                'params': {'lr': args.lr}
            },
            'RMSprop': {
                'class': torch.optim.RMSprop,
                'params': {'lr': args.lr}
            },
            'Adagrad': {
                'class': torch.optim.Adagrad,
                'params': {'lr': args.lr}
            }
        }
        
        results = compare_optimizers(
            NeuralNetwork, train_dataloader, loss_fn, device, 
            optimizers_config, n_epochs=10
        )
        
        if args.visualize:
            plot_optimizer_comparison(results)
        
        print("\nOptimizer Comparison Results:")
        print("-" * 40)
        for optimizer_name, result in results.items():
            print(f"{optimizer_name}: Loss={result['final_loss']:.3f}, Acc={result['final_accuracy']:.2f}%")
    
    # Save model if requested
    if args.save_model:
        model_path = 'mnist_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'losses': losses,
            'accuracies': accuracies
        }, model_path)
        print(f"Model saved to {model_path}")
    
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()

