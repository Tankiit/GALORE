import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from features import extract_comprehensive_features, select_top_k_samples


class SelectionWeightNetwork(nn.Module):
    """Simple network to learn optimal weights for combining selection methods."""

    def __init__(self, n_methods=3):
        super().__init__()
        # Learnable weights for each method (uncertainty, diversity, boundary)
        self.weights = nn.Parameter(torch.ones(n_methods) / n_methods)
        self.n_methods = n_methods

    def forward(self, features_dict):
        """
        Combine features using learned weights.
        features_dict: dict with keys 'uncertainty', 'diversity', 'boundary'
        Each has 'scores' and 'indices'
        """
        # Normalize scores for each method
        normalized_scores = []

        for method_name in ['uncertainty', 'diversity', 'boundary']:
            if method_name in features_dict:
                scores = features_dict[method_name]['scores']
                # Min-max normalization to [0, 1]
                if len(scores) > 0:
                    scores_min, scores_max = scores.min(), scores.max()
                    if scores_max > scores_min:
                        norm_scores = (scores - scores_min) / (scores_max - scores_min)
                    else:
                        norm_scores = torch.zeros_like(scores)
                    normalized_scores.append(norm_scores)
                else:
                    normalized_scores.append(torch.zeros(1))
            else:
                normalized_scores.append(torch.zeros(1))

        # Pad sequences to same length if needed
        max_len = max(len(scores) for scores in normalized_scores)
        padded_scores = []
        for scores in normalized_scores:
            if len(scores) < max_len:
                padded = torch.zeros(max_len)
                padded[:len(scores)] = scores
                padded_scores.append(padded)
            else:
                padded_scores.append(scores)

        # Stack and apply learned weights
        stacked_scores = torch.stack(padded_scores)

        # Apply softmax to weights to ensure they sum to 1
        normalized_weights = torch.softmax(self.weights, dim=0)

        # Weighted combination
        combined_scores = torch.sum(normalized_weights.unsqueeze(1) * stacked_scores, dim=0)

        return combined_scores[:len(normalized_scores[0])], normalized_weights


def train_selection_network(model, train_loader, val_loader, device, epochs=50, lr=0.01):
    """Train the weight selection network."""
    selection_net = SelectionWeightNetwork().to(device)
    optimizer = optim.Adam(selection_net.parameters(), lr=lr)

    # Extract features for training and validation
    print("Extracting training features...")
    train_features = extract_comprehensive_features(model, train_loader, device)

    print("Extracting validation features...")
    val_features = extract_comprehensive_features(model, val_loader, device)

    train_history = {'loss': [], 'weights': []}
    val_history = {'loss': [], 'weights': []}

    for epoch in tqdm(range(epochs), desc="Training selection network"):
        # Training phase
        selection_net.train()

        # Forward pass on training data
        combined_scores, weights = selection_net(train_features)

        # Simple loss: encourage diversity in top selections
        # We want the selected samples to be spread across the dataset
        loss = compute_selection_loss(combined_scores, k=100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        selection_net.eval()
        with torch.no_grad():
            val_combined_scores, val_weights = selection_net(val_features)
            val_loss = compute_selection_loss(val_combined_scores, k=100)

        # Store history
        train_history['loss'].append(loss.item())
        train_history['weights'].append(weights.detach().cpu().numpy())
        val_history['loss'].append(val_loss.item())
        val_history['weights'].append(val_weights.cpu().numpy())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")
            print(f"Weights: {weights.detach().cpu().numpy()}")

    return selection_net, train_history, val_history


def compute_selection_loss(scores, k=100, lambda_spread=0.1, lambda_entropy=0.05):
    """
    Compute loss to encourage good sample selection.

    Args:
        scores: Combined selection scores
        k: Number of top samples to select
        lambda_spread: Weight for spread diversity term
        lambda_entropy: Weight for entropy diversity term
    """
    # Get top-k indices
    top_k_indices = torch.topk(scores, k).indices

    # 1. Encourage selection of high-value samples (already inherent in top-k)
    selection_value = torch.mean(scores[top_k_indices])

    # 2. Encourage spread across the dataset
    # Higher variance in selected indices = better spread
    indices_variance = torch.var(top_k_indices.float())

    # 3. Encourage entropy in selection (avoid concentration)
    prob_distribution = torch.softmax(scores, dim=0)
    entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-8))

    # Combined loss (negative because we want to maximize these)
    loss = -selection_value - lambda_spread * indices_variance - lambda_entropy * entropy

    return loss


def evaluate_selection_method(model, test_loader, device, selection_net, features_dict, method_name, k=100):
    """Evaluate a selection method by training on selected samples and testing."""

    if method_name == 'learned':
        # Use learned weights to combine features
        combined_scores, weights = selection_net(features_dict)
        selected_indices = torch.topk(combined_scores, k).indices.cpu().numpy()
    else:
        # Use individual method
        selected_indices, _ = select_top_k_samples(features_dict, k, feature_type=method_name)

    print(f"\nEvaluating {method_name} selection...")
    print(f"Selected {len(selected_indices)} samples")

    # Create subset of training data with selected samples
    selected_dataset = Subset(test_loader.dataset, selected_indices[:len(selected_indices)//2])
    selected_loader = DataLoader(selected_dataset, batch_size=32, shuffle=True)

    # Create small test set for evaluation
    test_indices = np.random.choice(len(test_loader.dataset), 1000, replace=False)
    test_dataset = Subset(test_loader.dataset, test_indices)
    eval_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train a simple classifier on selected samples
    simple_model = torchvision.models.resnet18(pretrained=False)
    simple_model.fc = nn.Linear(simple_model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    simple_model = simple_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)

    # Quick training (5 epochs)
    simple_model.train()
    for epoch in range(5):
        for images, labels in selected_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = simple_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    simple_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = simple_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{method_name} selection accuracy: {accuracy:.2f}%")

    return accuracy


def main():
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Create smaller subsets for faster training
    train_subset = Subset(train_dataset, np.random.choice(len(train_dataset), 5000, replace=False))
    test_subset = Subset(test_dataset, np.random.choice(len(test_dataset), 1000, replace=False))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # Load pretrained ResNet for feature extraction
    print("Loading pretrained model...")
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adapt for CIFAR-10
    model = model.to(device)

    # Train selection network
    print("\n=== Training Selection Network ===")
    selection_net, train_hist, val_hist = train_selection_network(
        model, train_loader, val_loader, device, epochs=30
    )

    # Extract features for evaluation
    print("\nExtracting features for evaluation...")
    eval_features = extract_comprehensive_features(model, eval_loader, device)

    # Evaluate different selection methods
    methods = ['uncertainty', 'diversity', 'boundary', 'learned']
    results = {}

    for method in methods:
        accuracy = evaluate_selection_method(
            model, eval_loader, device, selection_net, eval_features, method, k=200
        )
        results[method] = accuracy

    # Print results
    print("\n=== Final Results ===")
    for method, accuracy in results.items():
        print(f"{method:12}: {accuracy:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_hist['loss'], label='Train Loss')
    plt.plot(val_hist['loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()

    plt.subplot(1, 2, 2)
    weights_array = np.array(train_hist['weights'])
    plt.plot(weights_array[:, 0], label='Uncertainty')
    plt.plot(weights_array[:, 1], label='Diversity')
    plt.plot(weights_array[:, 2], label='Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Learned Weights Over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig('selection_training_results.png')
    plt.show()

    return selection_net, results


if __name__ == "__main__":
    selection_net, results = main()