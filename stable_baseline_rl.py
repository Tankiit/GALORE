"""
RL-Driven Hypernetwork Experiments for LLM Data Selection
=========================================================

This implements the key experiments using RL tools (Stable-Baselines3) and custom hypernetworks
for adaptive multi-scoring data selection with theoretical guarantees.

Experiments:
1. Single-Model RL Training with Hypernetwork Policy
2. Multi-Scale Model Comparison (1B, 3B, 7B)
3. Domain-Specific Adaptation
4. Production Efficiency Analysis
5. Submodularity Violation Tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json

logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Training State and Context
# =============================================================================

@dataclass
class LLMTrainingState:
    """Enhanced training state for RL environment"""
    epoch: int
    train_loss: float
    val_loss: float
    perplexity: float
    learning_rate: float
    tokens_seen: int
    total_tokens: int
    gradient_norm: float
    memory_usage: float
    compute_usage: float
    
    # Domain and curriculum info
    domain_distribution: Dict[str, float]
    length_distribution: Dict[str, float]
    quality_scores: List[float]
    
    # Performance tracking
    performance_history: List[float]
    selection_diversity: float
    submodular_violations: int
    total_submodular_checks: int
    
    # Resource constraints
    memory_budget: float
    compute_budget: float
    latency_budget: float
    
    def to_observation(self) -> np.ndarray:
        """Convert to RL observation vector"""
        obs = [
            self.epoch / 1000.0,
            self.train_loss,
            self.val_loss,
            np.log(self.perplexity + 1e-8),
            np.log10(self.learning_rate + 1e-10),
            self.tokens_seen / max(self.total_tokens, 1),
            self.gradient_norm,
            self.memory_usage,
            self.compute_usage,
            self.selection_diversity,
            self.submodular_violations / max(self.total_submodular_checks, 1),
        ]
        
        # Domain distribution (4 domains)
        domains = ['code', 'science', 'dialogue', 'general']
        for domain in domains:
            obs.append(self.domain_distribution.get(domain, 0.0))
        
        # Length distribution (3 categories)
        lengths = ['short', 'medium', 'long']
        for length in lengths:
            obs.append(self.length_distribution.get(length, 0.0))
        
        # Performance trend
        if len(self.performance_history) >= 3:
            recent = self.performance_history[-3:]
            trend = (recent[-1] - recent[0]) / 3.0
            obs.append(trend)
        else:
            obs.append(0.0)
        
        # Resource utilization
        obs.extend([
            self.memory_usage / self.memory_budget,
            self.compute_usage / self.compute_budget,
            np.mean(self.quality_scores) if self.quality_scores else 0.0
        ])
        
        return np.array(obs, dtype=np.float32)


# =============================================================================
# Hypernetwork Policy for RL
# =============================================================================

class HypernetworkPolicy(nn.Module):
    """
    Hypernetwork that serves as the policy for RL agent
    """
    
    def __init__(self, 
                 observation_dim: int = 22,
                 num_scoring_functions: int = 6,
                 hidden_dim: int = 256,
                 attention_heads: int = 8):
        super().__init__()
        
        self.num_scoring_functions = num_scoring_functions
        self.function_names = ['perplexity', 'semantic_diversity', 'quality', 
                              'difficulty', 'domain_relevance', 'curriculum']
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Policy head - outputs weights for scoring functions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_scoring_functions),
            nn.Softmax(dim=-1)
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Curriculum preferences
        self.curriculum_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Easy, medium, hard
            nn.Softmax(dim=-1)
        )
        
        # Temperature for exploration
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning policy weights, value, curriculum, temperature"""
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        # Encode state
        encoded = self.state_encoder(observation)
        
        # Apply attention
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(0) if attended.shape[0] == 1 else attended
        
        # Generate outputs
        policy_weights = self.policy_head(attended)
        value = self.value_head(attended)
        curriculum_prefs = self.curriculum_head(attended)
        temperature = self.temperature_head(attended) * 2.0 + 0.1  # Scale to [0.1, 2.1]
        
        return policy_weights, value, curriculum_prefs, temperature


# =============================================================================
# Custom Gym Environment for Data Selection
# =============================================================================

class DataSelectionEnv(gym.Env):
    """
    Custom Gym environment for LLM data selection using hypernetwork policies
    """
    
    def __init__(self, 
                 dataset_size: int = 10000,
                 budget_ratio: float = 0.1,
                 max_epochs: int = 100,
                 model_scale: str = "1B"):
        super().__init__()
        
        self.dataset_size = dataset_size
        self.budget = int(dataset_size * budget_ratio)
        self.max_epochs = max_epochs
        self.model_scale = model_scale
        
        # Action space: weights for 6 scoring functions
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Observation space: training state vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(22,),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
        
        # Scoring function simulators (replace with actual implementations)
        self.scoring_functions = self._create_scoring_simulators()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_violations = []
        self.weight_history = []
        
    def _create_scoring_simulators(self) -> Dict[str, callable]:
        """Create simplified scoring function simulators for prototyping"""
        def perplexity_score(data_idx, selected_set, context):
            # Simulate based on data complexity
            base_score = np.random.gamma(2, 0.5)  # Gamma distribution for perplexity-like scores
            diminishing = 1.0 / (1.0 + 0.01 * len(selected_set))
            return base_score * diminishing
        
        def diversity_score(data_idx, selected_set, context):
            if len(selected_set) == 0:
                return 1.0
            # Simulate semantic diversity using random embeddings
            current_emb = np.random.randn(128)  # Mock embedding
            selected_embs = [np.random.randn(128) for _ in selected_set]
            min_dist = min([np.linalg.norm(current_emb - emb) for emb in selected_embs])
            return min_dist / 10.0  # Normalize
        
        def quality_score(data_idx, selected_set, context):
            # Mock quality based on normal distribution
            return max(0, np.random.normal(0.7, 0.2))
        
        def difficulty_score(data_idx, selected_set, context):
            # Mock gradient-based difficulty
            base_score = np.random.exponential(1.0)
            diminishing = 1.0 / (1.0 + 0.01 * len(selected_set))
            return base_score * diminishing
        
        def domain_score(data_idx, selected_set, context):
            # Mock domain relevance
            target_domain = context.get('target_domain', 'general')
            domain_scores = {'code': 0.8, 'science': 0.6, 'dialogue': 0.5, 'general': 0.4}
            return domain_scores.get(target_domain, 0.4) + np.random.normal(0, 0.1)
        
        def curriculum_score(data_idx, selected_set, context):
            # Mock curriculum based on training progress
            progress = context.get('training_progress', 0.0)
            text_length = np.random.randint(50, 1000)  # Mock text length
            
            if progress < 0.3:
                return max(0, 1.0 - text_length / 200)  # Prefer shorter
            elif progress < 0.7:
                return 1.0 - abs(text_length - 300) / 300  # Prefer medium
            else:
                return min(text_length / 500, 1.0)  # Prefer longer
        
        return {
            'perplexity': perplexity_score,
            'semantic_diversity': diversity_score,
            'quality': quality_score,
            'difficulty': difficulty_score,
            'domain_relevance': domain_score,
            'curriculum': curriculum_score
        }
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_epoch = 0
        self.selected_indices = []
        self.total_reward = 0.0
        self.submodular_violations = 0
        self.total_submodular_checks = 0
        
        # Initialize training state
        self.training_state = LLMTrainingState(
            epoch=0,
            train_loss=3.0,
            val_loss=3.2,
            perplexity=20.0,
            learning_rate=1e-4,
            tokens_seen=0,
            total_tokens=self.max_epochs * 1000000,
            gradient_norm=1.0,
            memory_usage=0.5,
            compute_usage=0.3,
            domain_distribution={'general': 1.0},
            length_distribution={'medium': 1.0},
            quality_scores=[0.7],
            performance_history=[3.2],
            selection_diversity=0.5,
            submodular_violations=0,
            total_submodular_checks=0,
            memory_budget=1.0,
            compute_budget=1.0,
            latency_budget=1.0
        )
        
        return self.training_state.to_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        # Interpret action as scoring function weights
        weights = action / (np.sum(action) + 1e-8)  # Normalize to sum to 1
        
        # Simulate data selection using weighted scoring
        num_to_select = min(self.budget // 10, 50)  # Select in batches
        selected_batch = self._select_data_batch(weights, num_to_select)
        self.selected_indices.extend(selected_batch)
        
        # Update training state (simulate training progress)
        self._update_training_state(selected_batch, weights)
        
        # Calculate reward
        reward = self._calculate_reward(weights)
        self.total_reward += reward
        
        # Check if episode is done
        done = (len(self.selected_indices) >= self.budget or 
                self.current_epoch >= self.max_epochs)
        
        # Store weight history for analysis
        weight_dict = {name: weights[i] for i, name in enumerate(['perplexity', 'semantic_diversity', 'quality', 'difficulty', 'domain_relevance', 'curriculum'])}
        self.weight_history.append(weight_dict)
        
        # Info dictionary
        info = {
            'selected_count': len(self.selected_indices),
            'budget': self.budget,
            'submodular_violations': self.submodular_violations,
            'total_checks': self.total_submodular_checks,
            'weights': weight_dict,
            'epoch': self.current_epoch
        }
        
        return self.training_state.to_observation(), reward, done, info
    
    def _select_data_batch(self, weights: np.ndarray, batch_size: int) -> List[int]:
        """Select a batch of data points using weighted scoring"""
        candidates = [i for i in range(self.dataset_size) if i not in self.selected_indices]
        if len(candidates) == 0:
            return []
        
        # Compute scores for candidates
        scores = []
        context = {
            'training_progress': self.current_epoch / self.max_epochs,
            'target_domain': 'general'
        }
        
        for idx in candidates[:min(len(candidates), batch_size * 5)]:  # Consider subset for efficiency
            combined_score = 0.0
            individual_scores = {}
            
            for i, (name, scorer) in enumerate(self.scoring_functions.items()):
                score = scorer(idx, self.selected_indices, context)
                individual_scores[name] = score
                combined_score += weights[i] * score
            
            scores.append((idx, combined_score, individual_scores))
        
        # Select top scoring items
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in scores[:batch_size]]
        
        # Check submodularity occasionally
        if len(scores) >= 2 and np.random.random() < 0.1:
            self._check_submodularity(scores[:2], weights)
        
        return selected
    
    def _check_submodularity(self, score_pairs: List, weights: np.ndarray):
        """Check submodular property for weighted combination"""
        self.total_submodular_checks += 1
        
        # Simplified submodularity check
        # In practice, would need to compute marginal gains properly
        idx1, score1, _ = score_pairs[0]
        idx2, score2, _ = score_pairs[1]
        
        # Mock submodularity violation (replace with actual check)
        violation_prob = np.sum(weights[[1, 2, 4, 5]]) * 0.2  # Higher prob for non-submodular functions
        if np.random.random() < violation_prob:
            self.submodular_violations += 1
    
    def _update_training_state(self, selected_batch: List[int], weights: np.ndarray):
        """Update training state based on selected data"""
        self.current_epoch += 1
        
        # Simulate training progress
        progress_factor = len(self.selected_indices) / self.budget
        
        # Loss improvement (better with more diverse selection)
        diversity_weight = weights[1]  # semantic_diversity weight
        loss_improvement = 0.01 * (1 + diversity_weight) * np.random.exponential(1.0)
        
        self.training_state.train_loss = max(0.5, self.training_state.train_loss - loss_improvement)
        self.training_state.val_loss = max(0.6, self.training_state.val_loss - loss_improvement * 0.8)
        self.training_state.perplexity = max(2.0, self.training_state.perplexity - loss_improvement * 5)
        
        # Update resource usage
        batch_memory = len(selected_batch) * 0.001  # Mock memory per sample
        batch_compute = len(selected_batch) * 0.002  # Mock compute per sample
        
        self.training_state.memory_usage += batch_memory
        self.training_state.compute_usage += batch_compute
        self.training_state.tokens_seen += len(selected_batch) * 200  # Mock tokens per sample
        
        # Update performance history
        self.training_state.performance_history.append(self.training_state.val_loss)
        if len(self.training_state.performance_history) > 20:
            self.training_state.performance_history = self.training_state.performance_history[-20:]
        
        # Update submodularity tracking
        self.training_state.submodular_violations = self.submodular_violations
        self.training_state.total_submodular_checks = self.total_submodular_checks
    
    def _calculate_reward(self, weights: np.ndarray) -> float:
        """Calculate reward based on multiple criteria"""
        # Performance improvement reward
        if len(self.training_state.performance_history) >= 2:
            perf_improvement = self.training_state.performance_history[-2] - self.training_state.performance_history[-1]
        else:
            perf_improvement = 0.0
        
        # Efficiency reward (prefer smaller selections)
        efficiency_reward = 1.0 - (len(self.selected_indices) / self.budget)
        
        # Submodularity reward (penalize violations)
        submod_penalty = -0.1 * (self.submodular_violations / max(self.total_submodular_checks, 1))
        
        # Resource constraint reward
        memory_penalty = -0.5 * max(0, self.training_state.memory_usage - self.training_state.memory_budget)
        compute_penalty = -0.5 * max(0, self.training_state.compute_usage - self.training_state.compute_budget)
        
        # Diversity reward (higher weights on diversity functions)
        diversity_bonus = 0.1 * (weights[1] + weights[2])  # semantic_diversity + quality
        
        total_reward = (
            10.0 * perf_improvement +
            2.0 * efficiency_reward +
            submod_penalty +
            memory_penalty +
            compute_penalty +
            diversity_bonus
        )
        
        return total_reward


# =============================================================================
# Custom Callback for RL Training
# =============================================================================

class HypernetworkCallback(BaseCallback):
    """Custom callback to track hypernetwork training progress"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_violations = []
        self.weight_evolution = []
        
    def _on_step(self) -> bool:
        # Log progress every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Get environment info
            if hasattr(self.training_env, 'get_attr'):
                envs = self.training_env.get_attr('unwrapped')
                if len(envs) > 0:
                    env = envs[0]
                    
                    self.episode_rewards.append(env.total_reward)
                    self.episode_violations.append(env.submodular_violations)
                    
                    if env.weight_history:
                        self.weight_evolution.append(env.weight_history[-1])
                    
                    if self.verbose > 0:
                        print(f"Step {self.n_calls}: Reward={env.total_reward:.3f}, "
                              f"Violations={env.submodular_violations}/{env.total_submodular_checks}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Save training results"""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_violations': self.episode_violations,
            'weight_evolution': self.weight_evolution
        }
        
        # Save to JSON (or use your preferred format)
        with open('hypernetwork_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed. Results saved. Final reward: {self.episode_rewards[-1] if self.episode_rewards else 'N/A'}")


# =============================================================================
# Experiment Runners
# =============================================================================

def run_single_model_experiment(model_scale: str = "1B", 
                               total_timesteps: int = 100000,
                               dataset_size: int = 10000,
                               budget_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Experiment 1: Single model RL training with hypernetwork policy
    """
    print(f"🚀 Running Single Model Experiment - {model_scale}")
    
    # Create environment
    env = make_vec_env(
        lambda: DataSelectionEnv(
            dataset_size=dataset_size,
            budget_ratio=budget_ratio,
            model_scale=model_scale
        ),
        n_envs=4  # Parallel environments for faster training
    )
    
    # Create RL agent with custom policy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Custom callback
    callback = HypernetworkCallback(log_freq=5000, verbose=1)
    
    # Train the model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Evaluate trained model
    eval_env = DataSelectionEnv(dataset_size=dataset_size, budget_ratio=budget_ratio, model_scale=model_scale)
    obs = eval_env.reset()
    episode_reward = 0
    
    for step in range(100):  # Evaluation episode
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    results = {
        'model_scale': model_scale,
        'final_reward': episode_reward,
        'selected_count': info['selected_count'],
        'submodular_violations': info['submodular_violations'],
        'violation_rate': info['submodular_violations'] / max(info['total_checks'], 1),
        'final_weights': info['weights'],
        'training_history': callback.episode_rewards
    }
    
    print(f"✅ Single Model Experiment Complete - Final Reward: {episode_reward:.3f}")
    return results


def run_multi_scale_comparison(scales: List[str] = ["1B", "3B", "7B"],
                              timesteps_per_scale: int = 50000) -> Dict[str, Any]:
    """
    Experiment 2: Multi-scale model comparison
    """
    print("🚀 Running Multi-Scale Comparison Experiment")
    
    results = {}
    
    for scale in scales:
        print(f"\n--- Training {scale} Model ---")
        scale_results = run_single_model_experiment(
            model_scale=scale,
            total_timesteps=timesteps_per_scale,
            dataset_size=10000,
            budget_ratio=0.1
        )
        results[scale] = scale_results
    
    # Compare results
    print("\n📊 Multi-Scale Comparison Results:")
    print("=" * 60)
    for scale, result in results.items():
        print(f"{scale}: Reward={result['final_reward']:.3f}, "
              f"Violations={result['violation_rate']:.3f}, "
              f"Selected={result['selected_count']}")
    
    return results


def run_domain_adaptation_experiment(domains: List[str] = ["code", "science", "dialogue"],
                                   timesteps_per_domain: int = 30000) -> Dict[str, Any]:
    """
    Experiment 3: Domain-specific adaptation
    """
    print("🚀 Running Domain Adaptation Experiment")
    
    results = {}
    
    for domain in domains:
        print(f"\n--- Training for {domain} domain ---")
        
        # Create domain-specific environment (would need custom environment)
        # For now, using base environment with domain context
        env = DataSelectionEnv(dataset_size=8000, budget_ratio=0.15)
        # In practice, you'd modify the environment to focus on domain-specific data
        
        # Train model
        domain_results = run_single_model_experiment(
            model_scale="3B",  # Use medium scale for domain experiments
            total_timesteps=timesteps_per_domain,
            dataset_size=8000,
            budget_ratio=0.15
        )
        
        results[domain] = domain_results
    
    print("\n📊 Domain Adaptation Results:")
    print("=" * 60)
    for domain, result in results.items():
        print(f"{domain}: Reward={result['final_reward']:.3f}, "
              f"Domain Weight={result['final_weights'].get('domain_relevance', 0):.3f}")
    
    return results


def run_production_efficiency_analysis(budget_ratios: List[float] = [0.05, 0.1, 0.2, 0.3]) -> Dict[str, Any]:
    """
    Experiment 4: Production efficiency analysis
    """
    print("🚀 Running Production Efficiency Analysis")
    
    results = {}
    
    for budget_ratio in budget_ratios:
        print(f"\n--- Testing Budget Ratio: {budget_ratio} ---")
        
        efficiency_results = run_single_model_experiment(
            model_scale="7B",  # Large scale for production simulation
            total_timesteps=40000,
            dataset_size=20000,
            budget_ratio=budget_ratio
        )
        
        # Calculate efficiency metrics
        efficiency_metrics = {
            'data_efficiency': efficiency_results['final_reward'] / (budget_ratio * 20000),
            'violation_rate': efficiency_results['violation_rate'],
            'selection_quality': efficiency_results['final_reward']
        }
        
        efficiency_results.update(efficiency_metrics)
        results[f"budget_{budget_ratio}"] = efficiency_results
    
    print("\n📊 Production Efficiency Results:")
    print("=" * 80)
    for budget, result in results.items():
        print(f"{budget}: Efficiency={result['data_efficiency']:.4f}, "
              f"Quality={result['selection_quality']:.3f}, "
              f"Violations={result['violation_rate']:.3f}")
    
    return results


# =============================================================================
# Visualization and Analysis Tools
# =============================================================================

def visualize_experiment_results(results: Dict[str, Any], experiment_type: str):
    """Create visualizations for experiment results"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{experiment_type} - Results Analysis', fontsize=16)
    
    if experiment_type == "Multi-Scale":
        # Performance vs Scale
        scales = list(results.keys())
        rewards = [results[scale]['final_reward'] for scale in scales]
        violations = [results[scale]['violation_rate'] for scale in scales]
        
        axes[0, 0].bar(scales, rewards, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Performance vs Model Scale')
        axes[0, 0].set_ylabel('Final Reward')
        
        axes[0, 1].bar(scales, violations, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Submodularity Violations vs Scale')
        axes[0, 1].set_ylabel('Violation Rate')
        
        # Weight evolution (if available)
        for i, scale in enumerate(scales):
            if 'training_history' in results[scale]:
                axes[1, 0].plot(results[scale]['training_history'], label=f'{scale}')
        axes[1, 0].set_title('Training Progress')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        
        # Final weight distribution
        weight_names = ['perplexity', 'semantic_diversity', 'quality', 'difficulty', 'domain_relevance', 'curriculum']
        bottom = np.zeros(len(scales))
        
        for i, weight_name in enumerate(weight_names):
            values = [results[scale]['final_weights'].get(weight_name, 0) for scale in scales]
            axes[1, 1].bar(scales, values, bottom=bottom, label=weight_name, alpha=0.7)
            bottom += values
        
        axes[1, 1].set_title('Final Weight Distribution by Scale')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{experiment_type.lower()}_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments():
    """Run all hypernetwork RL experiments"""
    print("🎯 Starting Comprehensive Hypernetwork RL Experiments")
    print("=" * 60)
    
    all_results = {}
    
    # Experiment 1: Single Model
    print("\n1️⃣ Single Model Experiment")
    single_results = run_single_model_experiment(
        model_scale="3B",
        total_timesteps=50000
    )
    all_results['single_model'] = single_results
    
    # Experiment 2: Multi-Scale
    print("\n2️⃣ Multi-Scale Comparison")
    multi_scale_results = run_multi_scale_comparison()
    all_results['multi_scale'] = multi_scale_results
    
    # Experiment 3: Domain Adaptation
    print("\n3️⃣ Domain Adaptation")
    domain_results = run_domain_adaptation_experiment()
    all_results['domain_adaptation'] = domain_results
    
    # Experiment 4: Production Efficiency
    print("\n4️⃣ Production Efficiency")
    efficiency_results = run_production_efficiency_analysis()
    all_results['production_efficiency'] = efficiency_results
    
    # Save all results
    with open('comprehensive_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualizations
    visualize_experiment_results(multi_scale_results, "Multi-Scale")
    
    print("\n🎉 All experiments completed! Results saved to comprehensive_experiment_results.json")
    
    return all_results


if __name__ == "__main__":
    # Run a quick single experiment for testing
    print("🧪 Running Quick Test Experiment")
    
    # Create and test environment
    env = DataSelectionEnv(dataset_size=1000, budget_ratio=0.1, max_epochs=20)
    obs = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.3f}, Selected={info['selected_count']}, Done={done}")
        
        if done:
            break
    
    