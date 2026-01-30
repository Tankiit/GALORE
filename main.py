import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm
import copy
import argparse
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gradient_matching import DirectGradientMatchingConfig, DirectGradientMatcher
from flow_matching import FlowMatchingConfig, FlowMatcher, VelocityField, TrajectoryGenerator
from models import SimpleCNN, create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Distillation using Gradient Matching or Flow Matching')
    
    # Method selection
    parser.add_argument('--method', type=str, default='gradient', choices=['gradient', 'flow'],
                        help='Distillation method: gradient or flow')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/Users/tanmoy/research/data',
                        help='Root directory for dataset')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not found')
    
    # Synthetic dataset size
    parser.add_argument('--num_synthetic', type=int, default=500,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--images_per_class', type=int, default=50,
                        help='Images per class (IPC)')
    
    # Gradient matching specific arguments
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help='Number of distillation iterations (gradient matching)')
    parser.add_argument('--lr_synthetic', type=float, default=0.1,
                        help='Learning rate for synthetic data optimization')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='Momentum for optimizer')
    parser.add_argument('--use_subspace', action='store_true', default=True,
                        help='Use gradient subspace projection')
    parser.add_argument('--subspace_dim', type=int, default=50,
                        help='Dimension of gradient subspace')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'cosine'],
                        help='Gradient matching loss type')
    parser.add_argument('--diversity_weight', type=float, default=0.01,
                        help='Weight for diversity regularization')
    parser.add_argument('--tv_weight', type=float, default=0.001,
                        help='Weight for total variation regularization')
    
    # Flow matching specific arguments
    parser.add_argument('--num_flow_steps', type=int, default=100,
                        help='Number of flow integration steps')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs (flow matching)')
    parser.add_argument('--flow_lr', type=float, default=1e-4,
                        help='Learning rate for flow matching')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for velocity field')
    parser.add_argument('--time_embed_dim', type=int, default=128,
                        help='Time embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of layers in velocity field')
    parser.add_argument('--num_gradient_steps', type=int, default=500,
                        help='Number of gradient steps for expert trajectory')
    parser.add_argument('--gradient_lr', type=float, default=0.01,
                        help='Learning rate for expert trajectory generation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for data loading')
    parser.add_argument('--num_gradient_samples', type=int, default=5000,
                        help='Number of real samples for gradient computation')
    parser.add_argument('--gradient_batch_size', type=int, default=256,
                        help='Batch size for gradient computation')
    
    # Device and output
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Default: auto-detect')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_synthetic', action='store_true',
                        help='Save synthetic dataset to file')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate distilled dataset after generation')
    parser.add_argument('--eval_epochs', type=int, default=10,
                        help='Number of epochs for evaluation training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_cnn',
                        help='Model architecture. Use "simple_cnn" for custom CNN or any timm model name (e.g., resnet18, vit_base_patch16_224, efficientnet_b0)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights (for timm models)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    # Load dataset
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_root, 
            train=True, 
            download=args.download, 
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_root, 
            train=False, 
            download=args.download, 
            transform=transform
        )
        num_classes = 10
        data_shape = (3, 32, 32)
    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=args.data_root, 
            train=True, 
            download=args.download, 
            transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_root, 
            train=False, 
            download=args.download, 
            transform=transform
        )
        num_classes = 100
        data_shape = (3, 32, 32)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = create_model(args.model, num_classes=num_classes, pretrained=args.pretrained, device=device)
    
    if args.method == 'gradient':
        # Gradient Matching
        config = DirectGradientMatchingConfig(
            num_synthetic=args.num_synthetic,
            images_per_class=args.images_per_class,
            num_iterations=args.num_iterations,
            lr_synthetic=args.lr_synthetic,
            momentum=args.momentum,
            num_gradient_samples=args.num_gradient_samples,
            gradient_batch_size=args.gradient_batch_size,
            use_subspace=args.use_subspace,
            subspace_dim=args.subspace_dim,
            loss_type=args.loss_type,
            diversity_weight=args.diversity_weight,
            tv_weight=args.tv_weight,
            device=device
        )
        
        matcher = DirectGradientMatcher(model, config)
        synthetic_data, synthetic_labels = matcher.distill(
            train_loader, num_classes, data_shape
        )
        
    elif args.method == 'flow':
        # Flow Matching
        config = FlowMatchingConfig(
            data_dim=np.prod(data_shape),
            hidden_dim=args.hidden_dim,
            time_embed_dim=args.time_embed_dim,
            num_layers=args.num_layers,
            num_flow_steps=args.num_flow_steps,
            num_epochs=args.num_epochs,
            lr=args.flow_lr,
            batch_size=args.batch_size,
            num_gradient_steps=args.num_gradient_steps,
            gradient_lr=args.gradient_lr,
            device=device
        )
        
        velocity_field = VelocityField(
            data_dim=config.data_dim,
            hidden_dim=config.hidden_dim,
            time_embed_dim=config.time_embed_dim,
            num_layers=config.num_layers
        )
        
        flow_matcher = FlowMatcher(velocity_field, config)
        
        # Generate expert trajectory
        criterion = nn.CrossEntropyLoss()
        trajectory_gen = TrajectoryGenerator(
            model, criterion, 
            num_steps=config.num_gradient_steps,
            lr=config.gradient_lr
        )
        
        # Get sample of real data for trajectory generation
        real_data_sample, real_labels_sample = next(iter(train_loader))
        real_data_sample = real_data_sample.to(device)
        real_labels_sample = real_labels_sample.to(device)
        
        trajectory, _ = trajectory_gen.generate_trajectory(
            real_data_sample, real_labels_sample,
            args.num_synthetic, data_shape
        )
        
        # Train flow matcher
        flow_matcher.train_on_trajectory(trajectory, config.num_epochs)
        
        # Generate synthetic data
        synthetic_data = flow_matcher.generate_from_flow(
            args.num_synthetic, data_shape, config.num_flow_steps
        )
        
        # Create balanced labels
        synthetic_labels = torch.arange(args.num_synthetic, device=device) % num_classes
    
    # Save synthetic dataset
    if args.save_synthetic:
        output_path = os.path.join(args.output_dir, f'synthetic_{args.method}_{args.dataset}.pt')
        torch.save({
            'data': synthetic_data,
            'labels': synthetic_labels,
            'config': args.__dict__
        }, output_path)
    
    # Evaluate if requested
    if args.evaluate:
        from gradient_matching import evaluate_distilled_dataset
        
        # Create a model factory class for evaluation
        class ModelFactory:
            def __init__(self, model_name, num_classes, pretrained, device):
                self.model_name = model_name
                self.num_classes = num_classes
                self.pretrained = pretrained
                self.device = device
            
            def __call__(self):
                return create_model(self.model_name, self.num_classes, self.pretrained, self.device)
        
        model_factory = ModelFactory(args.model, num_classes, False, device)
        accuracy = evaluate_distilled_dataset(
            synthetic_data, synthetic_labels,
            test_loader, model_factory,
            num_epochs=args.eval_epochs,
            device=device
        )


if __name__ == "__main__":
    main()
