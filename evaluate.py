import torch
from torch import nn


def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (torch.nn.Module): Trained neural network model
        dataloader (DataLoader): Data loader for evaluation
        loss_fn (torch.nn.Module): Loss function
        device (str): Device to run evaluation on
        
    Returns:
        tuple: (accuracy, average_loss, predictions, true_labels)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)
            
            # Get predictions
            pred_labels = torch.argmax(pred, axis=1)
            
            # Statistics
            total_loss += loss.item()
            correct += sum(pred_labels == y).item()
            total += pred.shape[0]
            
            # Store predictions and labels
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss, all_predictions, all_labels


def get_model_predictions(model, dataloader, device, num_samples=None):
    """
    Get model predictions for a batch of data.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Data loader
        device (str): Device to run inference on
        num_samples (int, optional): Number of samples to get predictions for
        
    Returns:
        tuple: (images, true_labels, predicted_labels)
    """
    model.eval()
    images, labels = next(iter(dataloader))
    
    if num_samples:
        images = images[:num_samples]
        labels = labels[:num_samples]
    
    with torch.no_grad():
        images_device = images.to(device)
        pred = model(images_device)
        pred_labels = torch.argmax(pred, axis=1)
    
    return images, labels, pred_labels.cpu()


def calculate_class_accuracy(model, dataloader, device):
    """
    Calculate accuracy for each class.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Test data loader
        device (str): Device to run inference on
        
    Returns:
        dict: Dictionary mapping class to accuracy
    """
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_labels = torch.argmax(pred, axis=1)
            
            for i in range(y.shape[0]):
                label = y[i].item()
                class_correct[label] += (pred_labels[i] == y[i]).item()
                class_total[label] += 1
    
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0
    
    return class_accuracies


def print_evaluation_summary(model, test_dataloader, loss_fn, device):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        model (torch.nn.Module): Trained model
        test_dataloader (DataLoader): Test data loader
        loss_fn (torch.nn.Module): Loss function
        device (str): Device to run evaluation on
    """
    print("=" * 50)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 50)
    
    # Overall accuracy
    accuracy, avg_loss, predictions, true_labels = evaluate_model(model, test_dataloader, loss_fn, device)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Average Test Loss: {avg_loss:.4f}")
    
    # Class-wise accuracy
    class_accuracies = calculate_class_accuracy(model, test_dataloader, device)
    print("\nClass-wise Accuracy:")
    print("-" * 20)
    for class_id, acc in class_accuracies.items():
        print(f"Class {class_id}: {acc:.2f}%")
    
    # Confusion matrix info
    print(f"\nTotal Test Samples: {len(test_dataloader.dataset)}")
    print(f"Correct Predictions: {sum(p == t for p, t in zip(predictions, true_labels))}")
    print(f"Incorrect Predictions: {sum(p != t for p, t in zip(predictions, true_labels))}")
    
    print("=" * 50)

