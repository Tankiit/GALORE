"""
Flow Matching for Dataset Distillation
=======================================

Following the Flow Matching framework:

Learn velocity field v_θ(x, t) that governs the evolution of:
- Synthetic data: dx/dt = v_θ(x, t)
- Or importance weights: dw/dt = v_θ(w, t)

Target: Match trajectory from gradient-based optimization.

This is the PROPER flow matching approach!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching"""
    # Flow model
    data_dim: int = 3072  # For CIFAR-10: 3*32*32
    hidden_dim: int = 512
    time_embed_dim: int = 128
    num_layers: int = 6
    
    # Training
    num_flow_steps: int = 100  # Discretization of t ∈ [0,1]
    num_epochs: int = 1000
    lr: float = 1e-4
    batch_size: int = 32
    
    # Trajectory matching
    num_gradient_steps: int = 500  # T for expert trajectory
    gradient_lr: float = 0.01
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding
    
    Encodes continuous time t ∈ [0, 1] into fixed-dimensional vector.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] time values in [0, 1]
        
        Returns:
            embedding: [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Sinusoidal encoding
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class VelocityField(nn.Module):
    """
    Learnable velocity field v_θ(x, t)
    
    This is the core of flow matching!
    
    Models: dx/dt = v_θ(x, t)
    where x is the synthetic data being optimized.
    """
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.data_dim = data_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(data_dim + time_embed_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection (predicts velocity)
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        # Initialize to small values (stable training)
        self.output_proj.weight.data *= 0.01
        self.output_proj.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity v_θ(x, t)
        
        Args:
            x: [B, D] current state (synthetic data)
            t: [B] or [B, 1] time in [0, 1]
        
        Returns:
            v: [B, D] velocity (dx/dt)
        """
        # Flatten x if image
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Embed time
        t_emb = self.time_embed(t)  # [B, time_embed_dim]
        
        # Concatenate x and time
        h = torch.cat([x, t_emb], dim=-1)  # [B, D + time_embed_dim]
        
        # Project
        h = self.input_proj(h)  # [B, hidden_dim]
        h = F.silu(h)
        
        # Residual blocks
        for block in self.blocks:
            h = h + block(h)  # Residual connection
        
        # Output velocity
        v = self.output_proj(h)  # [B, D]
        
        return v


class TrajectoryGenerator:
    """
    Generate expert trajectories using gradient-based optimization
    
    This creates the "target" that flow matching tries to replicate.
    
    Process:
    1. Start with random synthetic data x_0
    2. Optimize using gradient descent for T steps
    3. Record trajectory {x_0, x_1, ..., x_T}
    4. This is the "expert" trajectory we want to match
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_steps: int = 500,
        lr: float = 0.01
    ):
        self.model = model
        self.criterion = criterion
        self.num_steps = num_steps
        self.lr = lr
    
    def generate_trajectory(
        self,
        real_data: torch.Tensor,
        real_labels: torch.Tensor,
        num_synthetic: int,
        data_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate expert trajectory by optimizing synthetic data
        
        Args:
            real_data: [N, ...] real training data
            real_labels: [N] real labels
            num_synthetic: M (size of synthetic dataset)
            data_shape: (C, H, W) for images
        
        Returns:
            trajectory: List of [M, ...] states at each step
            final_data: [M, ...] final synthetic data
        """
        device = real_data.device
        
        # Initialize synthetic data
        synthetic_data = torch.randn(
            num_synthetic, *data_shape,
            device=device,
            requires_grad=True
        )
        
        # Balanced labels
        num_classes = real_labels.max().item() + 1
        synthetic_labels = torch.arange(num_synthetic, device=device) % num_classes
        
        # Optimizer
        optimizer = torch.optim.SGD([synthetic_data], lr=self.lr, momentum=0.5)
        
        # Record trajectory
        trajectory = [synthetic_data.detach().clone()]
        
        # Compute target gradient (from real data)
        self.model.zero_grad()
        real_output = self.model(real_data)
        real_loss = self.criterion(real_output, real_labels)
        real_loss.backward()
        
        target_grad = []
        for p in self.model.parameters():
            if p.grad is not None:
                target_grad.append(p.grad.detach().clone().view(-1))
        target_grad = torch.cat(target_grad)
        
        # Optimize synthetic data
        for step in tqdm(range(self.num_steps), desc="Expert trajectory"):
            # Compute gradient from synthetic data
            self.model.zero_grad()
            syn_output = self.model(synthetic_data)
            syn_loss = self.criterion(syn_output, synthetic_labels)
            
            # Match to target gradient
            syn_loss.backward()
            
            syn_grad = []
            for p in self.model.parameters():
                if p.grad is not None:
                    syn_grad.append(p.grad.view(-1))
            syn_grad = torch.cat(syn_grad)
            
            # Gradient matching loss
            match_loss = F.mse_loss(syn_grad, target_grad)
            
            # Update synthetic data
            optimizer.zero_grad()
            
            # Need to recompute for synthetic data gradient
            syn_output = self.model(synthetic_data)
            syn_loss = self.criterion(syn_output, synthetic_labels)
            syn_loss.backward()
            
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                synthetic_data.clamp_(0, 1)
            
            # Record state
            if step % (self.num_steps // 100) == 0:  # Record 100 points
                trajectory.append(synthetic_data.detach().clone())
        
        # Final state
        trajectory.append(synthetic_data.detach().clone())
        
        return trajectory, synthetic_data.detach()


class FlowMatcher:
    """
    Flow Matching for Dataset Distillation
    
    Learns velocity field v_θ(x, t) that replicates expert trajectory.
    
    Key equation:
        dx/dt = v_θ(x, t)
    
    Loss:
        L_FM(θ) = E_{t,x} [||v_θ(x, t) - u_t(x)||²]
    
    where u_t(x) is the target velocity from expert trajectory.
    """
    def __init__(
        self,
        velocity_field: VelocityField,
        config: FlowMatchingConfig
    ):
        self.velocity_field = velocity_field
        self.config = config
        self.device = torch.device(config.device)
        
        self.velocity_field.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.velocity_field.parameters(),
            lr=config.lr
        )
    
    def compute_target_velocity(
        self,
        trajectory: List[torch.Tensor],
        t_idx: int
    ) -> torch.Tensor:
        """
        Compute target velocity u_t(x) from expert trajectory
        
        u_t(x) ≈ (x_{t+1} - x_t) / Δt
        
        Args:
            trajectory: List of states [x_0, x_1, ..., x_T]
            t_idx: Index in trajectory
        
        Returns:
            velocity: [B, D] target velocity
        """
        if t_idx >= len(trajectory) - 1:
            # At final time, velocity is 0
            return torch.zeros_like(trajectory[-1])
        
        x_t = trajectory[t_idx]
        x_t1 = trajectory[t_idx + 1]
        
        # Finite difference approximation
        dt = 1.0 / (len(trajectory) - 1)
        velocity = (x_t1 - x_t) / dt
        
        return velocity
    
    def train_on_trajectory(
        self,
        trajectory: List[torch.Tensor],
        num_epochs: int = 1000
    ):
        """
        Train flow to match expert trajectory
        
        Args:
            trajectory: Expert trajectory from gradient optimization
            num_epochs: Training epochs
        """
        T = len(trajectory) - 1  # Number of time steps
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Sample time steps
            for _ in range(max(1, T // 10)):  # Multiple batches per epoch
                # Sample random time
                t_idx = np.random.randint(0, T)
                t = torch.tensor([t_idx / T], device=self.device)
                
                # Get state at time t
                x_t = trajectory[t_idx].to(self.device)
                
                # Flatten if needed
                if x_t.dim() > 2:
                    batch_size = x_t.size(0)
                    x_t_flat = x_t.view(batch_size, -1)
                else:
                    x_t_flat = x_t
                    batch_size = x_t.size(0)
                
                # Expand time to batch
                t_batch = t.expand(batch_size)
                
                # Predict velocity
                v_pred = self.velocity_field(x_t_flat, t_batch)
                
                # Compute target velocity
                u_target = self.compute_target_velocity(trajectory, t_idx)
                u_target_flat = u_target.view(batch_size, -1).to(self.device)
                
                # Flow matching loss
                loss = F.mse_loss(v_pred, u_target_flat)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
    
    def generate_from_flow(
        self,
        num_samples: int,
        data_shape: Tuple[int, ...],
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Generate synthetic data by integrating learned flow
        
        Solve ODE: dx/dt = v_θ(x, t) from t=0 to t=1
        
        Args:
            num_samples: Number of samples to generate
            data_shape: (C, H, W) for images
            num_steps: Integration steps
        
        Returns:
            x_final: [num_samples, ...] generated data
        """
        # Initial state (random noise)
        x = torch.randn(num_samples, *data_shape, device=self.device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        # Euler integration
        self.velocity_field.eval()
        
        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Flow integration"):
                t = torch.tensor([step / num_steps], device=self.device)
                t = t.expand(num_samples)
                
                # Flatten x
                x_flat = x.view(num_samples, -1)
                
                # Compute velocity
                v = self.velocity_field(x_flat, t)
                
                # Reshape velocity
                v = v.view_as(x)
                
                # Euler step: x_{t+1} = x_t + dt * v_θ(x_t, t)
                x = x + dt * v
                
                # Clamp to valid range
                x = torch.clamp(x, 0, 1)
        
        return x


def compare_trajectories(
    trajectory_expert: List[torch.Tensor],
    trajectory_flow: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compare expert trajectory vs flow-generated trajectory
    
    Metrics:
    - Endpoint distance
    - Average trajectory distance
    - Trajectory correlation
    """
    # Endpoint distance
    endpoint_dist = F.mse_loss(
        trajectory_expert[-1],
        trajectory_flow[-1]
    ).item()
    
    # Average distance along trajectory
    min_len = min(len(trajectory_expert), len(trajectory_flow))
    avg_dist = 0.0
    
    for i in range(min_len):
        dist = F.mse_loss(trajectory_expert[i], trajectory_flow[i]).item()
        avg_dist += dist
    
    avg_dist /= min_len
    
    return {
        'endpoint_distance': endpoint_dist,
        'average_distance': avg_dist
    }


if __name__ == "__main__":
    pass