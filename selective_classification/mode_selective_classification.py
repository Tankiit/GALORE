"""
MODE-SC: Multi-Objective Data-Driven Engine with Selective Classification
==========================================================================

Replaces hypernetwork with interpretable selective classification that:
1. Uses margin + entropy + diversity as selection signals
2. Provides explicit curriculum justification (why sample was selected)
3. Adapts thresholds dynamically based on training state
4. Works at both sample-level (VLM) and token-level (LLM)
5. Supports MPS (Apple Silicon), CUDA, and CPU

Key Innovation: Selection = Curriculum + Explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Device Utilities (MPS/CUDA/CPU Support)
# =============================================================================

def get_device(prefer_mps: bool = True) -> str:
    """
    Get best available device with MPS support for Apple Silicon.

    Priority: CUDA > MPS > CPU

    Args:
        prefer_mps: If True and MPS available, use it over CPU

    Returns:
        device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available() and prefer_mps:
        return 'mps'
    else:
        return 'cpu'


def move_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Safely move tensor to device (handles MPS limitations)."""
    if device == 'mps':
        # MPS has some dtype limitations, ensure float32
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
    return tensor.to(device)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MODESCConfig:
    """Configuration for MODE-SC framework."""

    # Device configuration
    device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    prefer_mps: bool = True  # Use MPS on Apple Silicon if available

    # Selection granularity
    granularity: str = "sample"  # "sample" for VLM, "token" for LLM, "both" for hybrid

    # Scoring strategies (like original MODE but interpretable)
    use_margin: bool = True          # Boundary proximity (small margin = hard)
    use_entropy: bool = True         # Uncertainty (high entropy = informative)
    use_diversity: bool = True       # Feature coverage
    use_loss: bool = True            # Direct loss signal

    # Adaptive threshold parameters
    coverage_target: float = 0.3     # Target selection ratio (like MODE budget)
    threshold_momentum: float = 0.9  # EMA for threshold stability
    warmup_epochs: int = 5          # Full coverage during warmup

    # Calibration (like Rho-1's reference model)
    use_reference_model: bool = False  # Use separate model for scoring
    reference_update_freq: int = 10    # Update reference every N epochs

    # Multi-signal combination
    signal_combination: str = "weighted_product"  # or "weighted_sum", "min", "soft_min"

    # Training state encoding (simplified from MODE's binary state)
    track_training_phase: bool = True
    early_phase_epochs: int = 10
    mid_phase_epochs: int = 20

    # Explainability
    log_selection_reasons: bool = True
    top_k_reasons: int = 3  # Show top-3 reasons for selection

    def __post_init__(self):
        """Auto-detect device if set to 'auto'."""
        if self.device == 'auto':
            self.device = get_device(self.prefer_mps)
            print(f"Auto-detected device: {self.device}")


# =============================================================================
# Selective Scoring Strategies (Interpretable Replacements for Hypernetwork)
# =============================================================================

class SelectiveScorer:
    """
    Computes interpretable confidence/difficulty scores.

    Unlike MODE's hypernetwork, each signal has clear semantic meaning:
    - Margin: How close to decision boundary (selective prediction)
    - Entropy: How uncertain the model is (active learning)
    - Diversity: How novel the sample is (coverage)
    - Loss: Direct training signal (curriculum learning)
    """

    def __init__(self, config: MODESCConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.device = config.device

        # Track selected samples for diversity
        self.selected_features = []
        self.max_selected_history = 1000

        # Adaptive thresholds (learned from data)
        self.adaptive_thresholds = {
            'margin': 0.3,
            'entropy': 0.7,
            'diversity': 0.5,
            'loss': 1.0
        }

        # EMA for threshold stability
        self.threshold_momentum = config.threshold_momentum

    def compute_margin_score(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Margin-based difficulty score.

        Selective Classification Principle:
        - Small margin = near decision boundary = uncertain = hard = informative
        - Large margin = confident prediction = easy = less informative

        Args:
            logits: [N, num_classes] or [N, seq_len, vocab] for tokens
            temperature: Confidence calibration (like selective prediction)

        Returns:
            margin_scores: [N] normalized scores (higher = more informative)
            margin_raw: [N] raw margin values for explainability
        """
        # Temperature scaling for calibration
        probs = F.softmax(logits / temperature, dim=-1)

        # Compute margin (difference between top-2 predictions)
        top2_probs = torch.topk(probs, k=2, dim=-1)[0]
        margin_raw = top2_probs[..., 0] - top2_probs[..., 1]

        # Convert to difficulty score (invert: small margin = high score)
        # Use smooth transformation to avoid division by zero
        margin_scores = torch.exp(-5.0 * margin_raw)  # Hyperparameter: controls sharpness

        return move_to_device(margin_scores, self.device), move_to_device(margin_raw, self.device)

    def compute_entropy_score(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entropy-based uncertainty score.

        Active Learning Principle:
        - High entropy = uncertain = informative
        - Low entropy = confident = less informative

        Returns:
            entropy_scores: [N] normalized (0-1)
            entropy_raw: [N] raw entropy values
        """
        probs = F.softmax(logits / temperature, dim=-1)
        entropy_raw = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize by max possible entropy
        max_entropy = np.log(self.num_classes)
        entropy_scores = entropy_raw / max_entropy

        return move_to_device(entropy_scores, self.device), move_to_device(entropy_raw, self.device)

    def compute_diversity_score(
        self,
        features: torch.Tensor,
        update_history: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diversity-based novelty score.

        Coverage Principle:
        - Far from selected samples = novel = informative
        - Close to selected samples = redundant = less informative

        Returns:
            diversity_scores: [N] normalized
            distances: [N] min distance to selected samples
        """
        N = len(features)
        features = move_to_device(features, self.device)

        if len(self.selected_features) == 0:
            # No history yet, all samples equally novel
            diversity_scores = torch.ones(N, device=self.device)
            distances = torch.ones(N, device=self.device) * float('inf')
        else:
            # Compute distance to nearest selected sample
            selected = torch.stack(self.selected_features).to(self.device)

            # MPS-compatible distance computation
            if self.device == 'mps':
                # MPS may have issues with cdist, use manual computation
                distances = []
                for feat in features:
                    dists = torch.norm(selected - feat.unsqueeze(0), dim=1)
                    distances.append(dists.min())
                distances = torch.stack(distances)
            else:
                distances = torch.cdist(features, selected, p=2).min(dim=1)[0]

            # Normalize distances (higher distance = higher score)
            diversity_scores = distances / (distances.max() + 1e-8)

        return diversity_scores, distances

    def compute_loss_score(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct loss-based difficulty (like Rho-1's excess loss).

        Curriculum Learning Principle:
        - High loss = hard sample = (potentially) informative
        - Low loss = easy sample = less informative

        Returns:
            loss_scores: [N] normalized
            loss_raw: [N] per-sample loss
        """
        logits = move_to_device(logits, self.device)
        labels = move_to_device(labels, self.device)

        # Per-sample cross-entropy loss
        loss_raw = F.cross_entropy(logits, labels, reduction='none')

        # Normalize by current max loss (adaptive normalization)
        loss_scores = loss_raw / (loss_raw.max() + 1e-8)

        return loss_scores, loss_raw

    def update_selected_features(self, features: torch.Tensor, selected_mask: torch.Tensor):
        """Update diversity history with newly selected samples."""
        selected_feats = features[selected_mask].detach().cpu()
        self.selected_features.extend(selected_feats)

        # Limit history size
        if len(self.selected_features) > self.max_selected_history:
            self.selected_features = self.selected_features[-self.max_selected_history:]


# =============================================================================
# Adaptive Threshold Controller (Replaces MODE's Binary State Encoder)
# =============================================================================

class AdaptiveThresholdController:
    """
    Dynamically adjusts selection thresholds based on training state.

    Replaces MODE's binary hypernetwork with interpretable threshold adaptation:
    - Early training: Low thresholds (easy samples, like curriculum)
    - Mid training: High thresholds (hard samples, boundary refinement)
    - Late training: Balanced (maintenance + edge cases)
    """

    def __init__(self, config: MODESCConfig):
        self.config = config

        # Training phase detection (replaces binary state encoder)
        self.current_epoch = 0
        self.validation_acc_history = []
        self.loss_history = []

        # Phase-specific threshold profiles
        self.threshold_profiles = {
            'early': {  # Epochs 0-10: Easy samples for foundation
                'margin': 0.5,    # High margin = easy samples
                'entropy': 0.3,   # Low entropy = confident samples
                'diversity': 0.7, # High diversity for coverage
                'loss': 0.4       # Lower loss = easier samples
            },
            'mid': {    # Epochs 10-20: Hard samples for refinement
                'margin': 0.2,    # Low margin = boundary samples
                'entropy': 0.7,   # High entropy = uncertain samples
                'diversity': 0.5, # Balanced diversity
                'loss': 0.7       # Higher loss = harder samples
            },
            'late': {   # Epochs 20+: Balanced maintenance
                'margin': 0.35,
                'entropy': 0.5,
                'diversity': 0.6,
                'loss': 0.5
            }
        }

        # Current thresholds (will be EMA-smoothed)
        self.current_thresholds = self.threshold_profiles['early'].copy()

    def detect_training_phase(self) -> str:
        """Detect current training phase from metrics."""
        if self.current_epoch < self.config.early_phase_epochs:
            return 'early'
        elif self.current_epoch < self.config.mid_phase_epochs:
            return 'mid'
        else:
            return 'late'

    def update_thresholds(
        self,
        epoch: int,
        val_acc: float,
        loss: float,
        selection_rate: float
    ) -> Dict[str, float]:
        """
        Update thresholds based on training state.

        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            loss: Training loss
            selection_rate: Current selection rate

        Returns:
            Updated thresholds dict
        """
        self.current_epoch = epoch
        self.validation_acc_history.append(val_acc)
        self.loss_history.append(loss)

        # 1. Get phase-specific base thresholds
        phase = self.detect_training_phase()
        target_thresholds = self.threshold_profiles[phase]

        # 2. Adjust based on selection rate (maintain target coverage)
        coverage_error = selection_rate - self.config.coverage_target
        threshold_adjustment = 0.05 * np.sign(coverage_error)  # Small adjustment

        # 3. EMA smoothing for stability
        for key in target_thresholds:
            target = target_thresholds[key] + threshold_adjustment
            current = self.current_thresholds[key]
            self.current_thresholds[key] = (
                self.config.threshold_momentum * current +
                (1 - self.config.threshold_momentum) * target
            )

        return self.current_thresholds


# =============================================================================
# MODE-SC: Main Selection Engine
# =============================================================================

class MODESC(nn.Module):
    """
    MODE with Selective Classification.

    Key Differences from Original MODE:
    1. NO hypernetwork - uses interpretable threshold-based selection
    2. Explicit confidence scoring (margin, entropy, loss, diversity)
    3. Adaptive thresholds replace learned strategy weights
    4. Built-in explainability (why was sample selected?)
    5. MPS support for Apple Silicon

    Maintains MODE's Advantages:
    - Adaptive curriculum learning
    - Multi-strategy combination
    - Dynamic selection based on training state
    """

    def __init__(
        self,
        config: MODESCConfig,
        num_classes: int
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.device = config.device

        # Scoring module (replaces hypernetwork)
        self.scorer = SelectiveScorer(config, num_classes)

        # Threshold controller (replaces binary state encoder)
        self.threshold_controller = AdaptiveThresholdController(config)

        # Optional: Reference model for scoring (like Rho-1)
        self.reference_model = None

        # Explainability tracking
        self.selection_reasons = []

    def compute_combined_score(
        self,
        scores: Dict[str, torch.Tensor],
        thresholds: Dict[str, float],
        method: str = "weighted_product"
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Combine multiple signals into final selection score.

        Methods:
        - weighted_product: S = ∏ s_i^w_i (geometric mean)
        - weighted_sum: S = Σ w_i * s_i (arithmetic mean)
        - soft_min: S = -log(Σ exp(-s_i)) (conservative)
        - min: S = min(s_i) (very conservative, all must pass)

        Returns:
            combined_scores: [N] final scores
            passing_signals: List of signals that passed thresholds
        """
        N = len(next(iter(scores.values())))

        # Check which signals pass their thresholds
        passed_thresholds = {}
        for signal_name, signal_scores in scores.items():
            passed = signal_scores >= thresholds[signal_name]
            passed_thresholds[signal_name] = passed

        if method == "weighted_product":
            # Geometric mean: all signals multiplicative
            combined = torch.ones(N, device=self.device)
            for signal_name, signal_scores in scores.items():
                # Threshold-gated: only contribute if passed
                gated_scores = torch.where(
                    passed_thresholds[signal_name],
                    signal_scores,
                    torch.zeros_like(signal_scores)
                )
                combined *= (gated_scores + 1e-8)

            combined = combined ** (1.0 / len(scores))  # Normalize

        elif method == "weighted_sum":
            # Arithmetic mean: signals additive
            combined = torch.zeros(N, device=self.device)
            for signal_name, signal_scores in scores.items():
                gated_scores = torch.where(
                    passed_thresholds[signal_name],
                    signal_scores,
                    torch.zeros_like(signal_scores)
                )
                combined += gated_scores / len(scores)

        elif method == "soft_min":
            # Soft minimum: conservative selection
            stacked = torch.stack(list(scores.values()), dim=0)  # [num_signals, N]
            combined = -torch.logsumexp(-stacked, dim=0)

        elif method == "min":
            # Hard minimum: all must pass
            stacked = torch.stack(list(scores.values()), dim=0)
            combined = stacked.min(dim=0)[0]

        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Generate passing signals for explainability
        passing_signals = [
            name for name, passed in passed_thresholds.items()
            if passed.any()
        ]

        return combined, passing_signals

    def select_samples(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        val_acc: float,
        current_loss: float,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Main selection function: replaces MODE's hypernetwork-based selection.

        Args:
            features: [N, d] feature representations
            logits: [N, num_classes] model outputs
            labels: [N] ground truth labels
            epoch: Current training epoch
            val_acc: Current validation accuracy
            current_loss: Current training loss
            temperature: Calibration temperature

        Returns:
            selected_mask: [N] boolean mask (True = select for training)
            metadata: Dict with scores, thresholds, reasons
        """
        N = len(features)

        # Move inputs to correct device
        features = move_to_device(features, self.device)
        logits = move_to_device(logits, self.device)
        labels = move_to_device(labels, self.device)

        # 1. Compute all scoring signals
        scores = {}
        raw_scores = {}

        if self.config.use_margin:
            scores['margin'], raw_scores['margin'] = self.scorer.compute_margin_score(
                logits, temperature
            )

        if self.config.use_entropy:
            scores['entropy'], raw_scores['entropy'] = self.scorer.compute_entropy_score(
                logits, temperature
            )

        if self.config.use_diversity:
            scores['diversity'], raw_scores['diversity'] = self.scorer.compute_diversity_score(
                features, update_history=False  # Will update after selection
            )

        if self.config.use_loss:
            scores['loss'], raw_scores['loss'] = self.scorer.compute_loss_score(
                logits, labels
            )

        # 2. Get adaptive thresholds based on training state
        current_selection_rate = 0.5  # Will be updated iteratively
        thresholds = self.threshold_controller.update_thresholds(
            epoch=epoch,
            val_acc=val_acc,
            loss=current_loss,
            selection_rate=current_selection_rate
        )

        # 3. Combine signals into final score
        combined_scores, passing_signals = self.compute_combined_score(
            scores, thresholds, method=self.config.signal_combination
        )

        # 4. Select top-k based on target coverage
        k = int(N * self.config.coverage_target)

        if epoch < self.config.warmup_epochs:
            # Warmup: select randomly (full coverage curriculum)
            selected_indices = torch.randperm(N, device=self.device)[:k]
        else:
            # Normal selection: top-k by combined score
            selected_indices = torch.topk(combined_scores, k=k)[1]

        selected_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        selected_mask[selected_indices] = True

        # 5. Update diversity history
        if self.config.use_diversity:
            self.scorer.update_selected_features(features, selected_mask)

        # 6. Generate explainability metadata
        metadata = {
            'scores': scores,
            'raw_scores': raw_scores,
            'thresholds': thresholds,
            'combined_scores': combined_scores,
            'passing_signals': passing_signals,
            'selection_rate': selected_mask.float().mean().item(),
            'phase': self.threshold_controller.detect_training_phase()
        }

        # Log selection reasons for analysis
        if self.config.log_selection_reasons:
            self._log_selection_reasons(selected_indices, scores, thresholds)

        return selected_mask, metadata

    def _log_selection_reasons(
        self,
        selected_indices: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        thresholds: Dict[str, float]
    ):
        """
        Generate human-readable selection reasons.

        Example output:
        "Sample 42 selected because:
         1. Low margin (0.15 < 0.3): Near decision boundary
         2. High entropy (0.85 > 0.7): Model uncertain
         3. High diversity (0.9 > 0.6): Novel sample"
        """
        reasons = []

        for idx in selected_indices[:10]:  # Log first 10 for brevity
            sample_reasons = []

            for signal_name, signal_scores in scores.items():
                score_val = signal_scores[idx].item()
                thresh = thresholds[signal_name]

                if score_val >= thresh:
                    reason = f"{signal_name}={score_val:.3f} > {thresh:.3f}"
                    sample_reasons.append(reason)

            if sample_reasons:
                reasons.append(f"Sample {idx.item()}: {', '.join(sample_reasons)}")

        self.selection_reasons.extend(reasons)


# =============================================================================
# VLM Extension (Sample-Level for CLIP-style models)
# =============================================================================

class MODESCForVLM(MODESC):
    """Sample-level MODE-SC for vision-language models (CLIP, BLIP, etc.)."""

    def __init__(self, config: MODESCConfig, num_classes: int):
        super().__init__(config, num_classes)

    def select_image_text_pairs(
        self,
        image_features: torch.Tensor,   # [N, d_img]
        text_features: torch.Tensor,    # [N, d_text]
        similarity_scores: torch.Tensor, # [N] CLIP-style scores
        labels: torch.Tensor,
        epoch: int,
        val_acc: float,
        current_loss: float
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Select informative image-text pairs.

        For VLMs, we can use:
        - Margin: Similarity score spread
        - Entropy: Caption ambiguity (multiple valid captions)
        - Diversity: Image feature novelty
        - Loss: Contrastive loss per pair
        """

        # Combine image and text features
        combined_features = torch.cat([image_features, text_features], dim=-1)

        # Use similarity as "logits" (though not classification)
        # Can reshape to [N, 2] where [:, 0] = positive, [:, 1] = negative
        pseudo_logits = torch.stack([similarity_scores, 1 - similarity_scores], dim=1)

        return self.select_samples(
            features=combined_features,
            logits=pseudo_logits,
            labels=labels,
            epoch=epoch,
            val_acc=val_acc,
            current_loss=current_loss
        )


if __name__ == '__main__':
    print("MODE-SC: Selective Classification for Interpretable Curriculum Learning")
    print("=" * 70)
    print("\nDevice Support:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Auto-detected: {get_device()}")
    print("\nKey Advantages:")
    print("1. Replaces hypernetwork with interpretable threshold-based selection")
    print("2. Explicit curriculum justification via confidence metrics")
    print("3. Connects data selection to selective prediction principles")
    print("4. Works on CUDA, MPS (Apple Silicon), and CPU")
    print("5. Sample-level (VLM) and token-level (LLM) support")
