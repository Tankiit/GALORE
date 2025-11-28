#!/usr/bin/env python3
"""
SIMPLIFIED THOMPSON SAMPLING FOR CIFAR-10
Stripped down to focus on core functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random


class SimpleThompsonSampler:
    """Simple Thompson Sampling for curriculum strategy selection"""

    def __init__(self, num_strategies=3):
        self.num_strategies = num_strategies
        # Beta distribution parameters for each strategy
        self.alpha = torch.ones(num_strategies)  # Successes
        self.beta = torch.ones(num_strategies)   # Failures

    def select_strategy(self):
        """Sample from Beta distributions and select best strategy"""
        samples = torch.distributions.Beta(self.alpha, self.beta).sample()
        return torch.argmax(samples).item()

    def update(self, strategy_idx, reward):
        """Update Beta distribution based on observed reward"""
        if reward > 0.5:  # Success
            self.alpha[strategy_idx] += 1
        else:  # Failure
            self.beta[strategy_idx] += 1

    def get_stats(self):
        """Get current strategy statistics"""
        return {
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist(),
            'means': (self.alpha / (self.alpha + self.beta)).tolist()
        }


class SimpleCurriculumStrategies:
    """Simple curriculum selection strategies"""

    @staticmethod
    def uncertainty(images, labels, model, device):
        """Select samples with highest prediction uncertainty"""
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1)
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return uncertainty.cpu()

    @staticmethod
    def random_strategy(images, labels, model, device):
        """Random selection"""
        return torch.rand(len(images))

    @staticmethod
    def loss_based(images, labels, model, device):
        """Select samples with highest loss"""
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(device))
            loss = F.cross_entropy(outputs, labels.to(device), reduction='none')
        return loss.cpu()


class SimpleResNet(nn.Module):
    """Simple ResNet for CIFAR-10"""

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-18 (trained on ImageNet with 1000 classes)
        self.model = torchvision.models.resnet18(pretrained=True)

        # Replace the final fully connected layer for CIFAR-10 (10 classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


class SimpleCIFAR10ThompsonTrainer:
    """Simple CIFAR-10 trainer with Thompson Sampling"""

    def __init__(self, budget=0.3, epochs=20, device='cpu'):
        self.budget = budget  # Fraction of data to use per epoch
        self.epochs = epochs
        self.device = device

        # Setup data
        self.setup_data()

        # Setup model
        self.model = SimpleResNet().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        # Setup Thompson sampler and strategies
        self.ts = SimpleThompsonSampler(num_strategies=3)
        self.strategies = [
            ('uncertainty', SimpleCurriculumStrategies.uncertainty),
            ('random', SimpleCurriculumStrategies.random_strategy),
            ('loss_based', SimpleCurriculumStrategies.loss_based)
        ]

        # Track metrics
        self.train_acc_history = []
        self.strategy_history = []
        self.reward_history = defaultdict(list)

    def setup_data(self):
        """Setup CIFAR-10 data"""
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data',
                                              train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data',
                                             train=False, download=True, transform=transform)

        # Use subset for faster training
        train_indices = random.sample(range(len(trainset)), 5000)
        test_indices = random.sample(range(len(testset)), 1000)

        self.train_dataset = Subset(trainset, train_indices)
        self.test_dataset = Subset(testset, test_indices)

        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()

        return correct / total

    def get_reward(self, prev_acc, current_acc):
        """Calculate reward based on accuracy improvement"""
        improvement = current_acc - prev_acc
        return 1.0 if improvement > 0.01 else 0.0  # Simple binary reward

    def train_epoch(self, epoch):
        """Train one epoch using Thompson Sampling"""
        self.model.train()

        # Select strategy using Thompson Sampling
        strategy_idx = self.ts.select_strategy()
        strategy_name, strategy_fn = self.strategies[strategy_idx]

        print(f"Epoch {epoch+1}: Using {strategy_name} strategy")

        # Get previous accuracy for reward calculation
        prev_acc = self.evaluate() if epoch > 0 else 0.0

        # Apply curriculum strategy to select samples
        selected_samples = []
        total_samples = len(self.train_dataset)
        target_samples = int(total_samples * self.budget)

        for images, labels in self.train_loader:
            batch_scores = strategy_fn(images, labels, self.model, self.device)
            selected_samples.extend(list(zip(images, labels, batch_scores)))

        # Sort by scores and select top samples
        selected_samples.sort(key=lambda x: x[2], reverse=True)
        selected_samples = selected_samples[:target_samples]

        # Train on selected samples
        if selected_samples:
            selected_images = torch.stack([x[0] for x in selected_samples])
            selected_labels = torch.tensor([x[1] for x in selected_samples])

            # Create simple loader
            selected_dataset = list(zip(selected_images, selected_labels))
            selected_loader = DataLoader(selected_dataset, batch_size=32, shuffle=True)

            for batch_images, batch_labels in selected_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_images.to(self.device))
                loss = self.criterion(outputs, batch_labels.to(self.device))
                loss.backward()
                self.optimizer.step()

        # Calculate current accuracy and reward
        current_acc = self.evaluate()
        reward = self.get_reward(prev_acc, current_acc)

        # Update Thompson Sampling
        self.ts.update(strategy_idx, reward)

        # Track results
        self.train_acc_history.append(current_acc)
        self.strategy_history.append(strategy_idx)
        self.reward_history[strategy_idx].append(reward)

        print(f"  Test Accuracy: {current_acc:.4f}, Reward: {reward:.2f}")
        print(f"  Strategy Stats: {self.ts.get_stats()}")

        return current_acc

    def train(self):
        """Main training loop"""
        print(f"Starting Simple CIFAR-10 Thompson Sampling Training")
        print(f"Budget: {self.budget:.1%}, Epochs: {self.epochs}")
        print("="*60)

        for epoch in range(self.epochs):
            self.train_epoch(epoch)

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Final Test Accuracy: {self.train_acc_history[-1]:.4f}")
        print(f"Strategy Selection History: {self.strategy_history}")

        # Plot results
        self.plot_results()

    def plot_results(self):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Simple Thompson Sampling CIFAR-10 Results', fontsize=16)

        # Plot 1: Accuracy over time
        axes[0, 0].plot(self.train_acc_history, 'b-', linewidth=2)
        axes[0, 0].set_title('Test Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].grid(True)

        # Plot 2: Strategy selection
        strategy_names = [s[0] for s in self.strategies]
        strategy_counts = [self.strategy_history.count(i) for i in range(len(strategy_names))]
        axes[0, 1].bar(strategy_names, strategy_counts)
        axes[0, 1].set_title('Strategy Selection Frequency')
        axes[0, 1].set_ylabel('Count')

        # Plot 3: Thompson parameters
        stats = self.ts.get_stats()
        x = np.arange(len(strategy_names))
        width = 0.25

        axes[1, 0].bar(x - width, stats['alpha'], width, label='Alpha (Successes)')
        axes[1, 0].bar(x, stats['beta'], width, label='Beta (Failures)')
        axes[1, 0].bar(x + width, stats['means'], width, label='Mean')
        axes[1, 0].set_title('Thompson Sampling Parameters')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(strategy_names)
        axes[1, 0].legend()

        # Plot 4: Rewards per strategy
        for i, (name, _) in enumerate(self.strategies):
            if self.reward_history[i]:
                rewards = self.reward_history[i]
                epochs = list(range(len(rewards)))
                axes[1, 1].plot(epochs, rewards, 'o-', label=name, alpha=0.7)

        axes[1, 1].set_title('Rewards per Strategy')
        axes[1, 1].set_xlabel('Strategy Usage')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('simple_thompson_results.png', dpi=150, bbox_inches='tight')
        print("Results saved as: simple_thompson_results.png")


def main():
    """Main function"""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = SimpleCIFAR10ThompsonTrainer(
        budget=0.3,  # Use 30% of data per epoch
        epochs=15,   # Train for 15 epochs
        device=device
    )

    trainer.train()


if __name__ == "__main__":
    main()