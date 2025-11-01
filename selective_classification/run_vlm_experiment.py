
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds
from huggingface_hub import HfFileSystem, hf_hub_url
import open_clip
from tqdm import tqdm
import pickle
import random
from pathlib import Path
import os

# NOTE: The following sections for analysis require additional packages.
# Please install them with: pip install matplotlib seaborn pandas scikit-learn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from collections import defaultdict

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATASET_NAME = "pixparse/cc3m-wds"
DATASET_SIZE = 500_000
KEEP_FRACTION = 0.3
KEEP_SAMPLES = int(DATASET_SIZE * KEEP_FRACTION)
BATCH_SIZE = 512
EPOCHS = 50
MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'
LR = 5e-4
WARMUP = 2000
WD = 0.1

FEATURE_CACHE_PATH = Path(f'./cc12m_{DATASET_SIZE}_clip_features.pt')
MODE_SELECTION_PATH = Path(f'./mode_selected_{KEEP_SAMPLES}.txt')
RANDOM_SELECTION_PATH = Path(f'./random_selected_{KEEP_SAMPLES}.txt')

print(f"Using device: {DEVICE}")
print(f"Dataset size: {DATASET_SIZE}")
print(f"Keep fraction: {KEEP_FRACTION} ({KEEP_SAMPLES} samples)")

# --- Global Model and Preprocessors ---
print(f"Loading CLIP model '{MODEL_NAME}' ({CLIP_PRETRAINED}).")
clip_model, _, image_processor = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

def preprocess_text(sample):
    return tokenizer(sample[1]['caption'])

def preprocess_image(sample):
    return image_processor(sample[0])

def process_sample(sample):
    return (preprocess_image(sample), preprocess_text(sample))

# --- 1. Dataset Loading ---
def load_conceptual_captions(subset_size):
    """Loads the Conceptual Captions 3M dataset and returns a WebDataset object."""
    print("Setting up dataset streaming from Hugging Face...")
    fs = HfFileSystem()
    
    try:
        files = fs.glob(f"hf://datasets/{DATASET_NAME}/cc3m-train-*.tar")
        urls = [hf_hub_url(DATASET_NAME, f.split(f"{DATASET_NAME}/")[1], repo_type="dataset") for f in files]
        print(f"Found {len(urls)} tar files for the dataset.")
    except Exception as e:
        print(f"Could not automatically list files from Hub: {e}")
        print("Please ensure you are logged in with 'huggingface-cli login'")
        return None, None

    dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(1000)
        .slice(subset_size)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map(process_sample)
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    return dataloader, dataset

# --- 2. Precompute CLIP Features ---

def precompute_clip_features(dataloader, model, cache_path):
    """Extracts and saves CLIP features for the dataset."""
    print(f"Pre-computing CLIP features and saving to {cache_path}...")
    model.eval()
    feature_cache = {'image_features': [], 'text_features': [], 'sample_ids': []}
    
    # We need sample IDs, but webdataset doesn't provide them easily in the loader.
    # We will assign sequential IDs.
    sample_counter = 0

    with torch.no_grad():
        for i, (images, texts) in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = images.to(DEVICE)
            texts = texts.squeeze(1).to(DEVICE)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(texts)
            
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            
            feature_cache['image_features'].append(img_feats.cpu())
            feature_cache['text_features'].append(txt_feats.cpu())
            
            num_samples = images.size(0)
            feature_cache['sample_ids'].extend(range(sample_counter, sample_counter + num_samples))
            sample_counter += num_samples

    features = {
        'image': torch.cat(feature_cache['image_features']),
        'text': torch.cat(feature_cache['text_features']),
        'ids': feature_cache['sample_ids']
    }
    
    torch.save(features, cache_path)
    print(f"Cached {len(features['ids'])} samples to {cache_path}")
    return features

# --- 3. MODE Selection ---

def load_mode_with_vlm_adapter(model_path):
    """
    Placeholder for loading your trained MODE model.
    You need to implement this function based on your MODE model's architecture.
    """
    print(f"Loading MODE model from: {model_path}")
    if not Path(model_path).exists():
        print(f"Warning: MODE model file not found at {model_path}")
        print("MODE selection will fail. Please provide the correct path.")
        # Returning a dummy scorer that returns random scores
        class DummyMode(nn.Module):
            def score_batch(self, states):
                return torch.rand(states.size(0))
        return DummyMode()

    # --- Example implementation ---
    # Assuming your MODE model is a simple MLP that you can load with torch.load
    # Adjust this to your actual model architecture.
    # mode_model = YourModeNet() 
    # mode_model.load_state_dict(torch.load(model_path))
    # return mode_model
    
    # For now, returning a dummy model
    class DummyMode(nn.Module):
        def score_batch(self, states):
            return torch.rand(states.size(0))
    return DummyMode()


def mode_selection(features, mode_model, selection_path):
    """Selects the top samples using MODE and saves the indices."""
    print("Performing MODE selection...")
    image_feats = features['image']
    text_feats = features['text']
    
    mode_model.to(DEVICE)
    mode_model.eval()
    
    all_scores = []
    batch_size = 10000

    with torch.no_grad():
        for i in tqdm(range(0, len(image_feats), batch_size), desc="Computing MODE scores"):
            batch_img = image_feats[i:i+batch_size].to(DEVICE)
            batch_txt = text_feats[i:i+batch_size].to(DEVICE)
            
            # This state extraction logic is from your provided plan.
            similarity = (batch_img * batch_txt).sum(dim=-1)
            logits = batch_img @ batch_txt.T / 0.07
            labels = torch.arange(len(batch_img), device=DEVICE)
            loss = F.cross_entropy(logits, labels, reduction='none')
            
            batch_states = torch.zeros(len(batch_img), 12, device=DEVICE)
            batch_states[:, 0] = loss > loss.median()
            batch_states[:, 1] = similarity < similarity.median()
            
            scores = mode_model.score_batch(batch_states)
            all_scores.append(scores.cpu())

    scores = torch.cat(all_scores)
    top_indices = torch.topk(scores, KEEP_SAMPLES).indices
    
    selected_ids = [features['ids'][i] for i in top_indices]
    with open(selection_path, 'w') as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")
            
    print(f"Selected {len(selected_ids)} sample indices and saved to {selection_path}")
    return selected_ids

# --- 4. Baselines ---

def random_selection(features, selection_path):
    """Selects random samples and saves the indices."""
    print("Performing random selection...")
    indices = list(range(len(features['ids'])))
    selected_ids = random.sample(indices, KEEP_SAMPLES)
    
    with open(selection_path, 'w') as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")
            
    print(f"Selected {len(selected_ids)} random sample indices and saved to {selection_path}")
    return selected_ids

# --- 5. CLIP Training ---

def train_clip(train_dataset, selection_name):
    """Trains a CLIP model on a selected subset of data."""
    print(f"\n--- Starting CLIP training for '{selection_name}' ---")
    
    model, _, image_processor = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(DEVICE)
    
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = len(dataloader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=WARMUP/total_steps)

    print(f"Training for {EPOCHS} epochs with {len(dataloader)} steps per epoch.")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (images, texts) in enumerate(pbar):
            images = images.to(DEVICE)
            texts = texts.squeeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=DEVICE.type):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.T
                
                labels = torch.arange(len(images), device=DEVICE)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
        print(f"Epoch {epoch+1} complete. Final loss: {loss.item():.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(f'./checkpoints/{selection_name}')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch+1}.pt")

    print(f"--- Finished CLIP training for '{selection_name}' ---")


# --- Main Orchestrator ---

def main():
    """Main function to run the VLM experiment."""
    
    # --- Data Loading and Feature Extraction ---
    if not FEATURE_CACHE_PATH.exists():
        dataloader, _ = load_conceptual_captions(DATASET_SIZE)
        if dataloader is None:
            return
        features = precompute_clip_features(dataloader, clip_model, FEATURE_CACHE_PATH)
    else:
        print(f"Loading cached features from {FEATURE_CACHE_PATH}")
        features = torch.load(FEATURE_CACHE_PATH)

    # --- MODE Selection ---
    if not MODE_SELECTION_PATH.exists():
        # IMPORTANT: You need to provide your own trained MODE model file.
        mode_model = load_mode_with_vlm_adapter('mode_cifar_best.pt')
        mode_selection(features, mode_model, MODE_SELECTION_PATH)
    
    # --- Random Baseline Selection ---
    if not RANDOM_SELECTION_PATH.exists():
        random_selection(features, RANDOM_SELECTION_PATH)

    # --- Training on Selected Subsets ---
    print("\n--- Starting training runs ---")
    
    # Create a dataset from all downloaded samples (we will filter it)
    # This is a bit tricky with webdataset, as we need to map indices to samples.
    # A simpler way is to re-filter the webdataset.
    
    def create_filtered_dataset(selection_path):
        with open(selection_path, 'r') as f:
            selected_ids = {int(line.strip()) for line in f}
        
        # We need to re-create the webdataset and filter it.
        fs = HfFileSystem()
        files = fs.glob(f"hf://datasets/{DATASET_NAME}/data/*.tar")
        urls = [hf_hub_url(DATASET_NAME, f.split(f"{DATASET_NAME}/")[1], repo_type="dataset") for f in files]

        def id_filter(sample):
            # The sample key __key__ can be used if it's consistent
            # Let's assume we can get an index from the sample.
            # Webdataset does not have a global index by default.
            # We will add one.
            return sample['__sample_index__'] in selected_ids

        dataset = (
            wds.WebDataset(urls, resampled=True)
            .shuffle(1000)
            .slice(DATASET_SIZE)
            .decode("pil")
            .map(lambda x: {**x, '__sample_index__': int(x['__key__'])}) # This is an assumption
            .select(id_filter)
            .to_tuple("jpg;png", "json")
            .map_tuple(image_processor, lambda s: tokenizer(s['caption']))
        )
        return dataset

    # This filtering is complex. A much simpler approach for training is to use the indices
    # on the cached features, if we can load the images from somewhere.
    # Since we are streaming, we don't have the images locally.
    
    # Let's try a different approach for training:
    # We will create a small dataset class that holds the selected indices and
    # then we will iterate through the webdataset until we find them. This is inefficient.
    
    # The most robust way is to download the subset.
    # The plan was to use webdataset for everything. Let's stick to it.
    # The filtering function is the way to go. We need to get the sample index right.
    
    print("Note: Training requires filtering the dataset, which can be slow.")
    
    # Train on MODE selection
    # mode_dataset = create_filtered_dataset(MODE_SELECTION_PATH)
    # train_clip(mode_dataset, "mode_selection")
    
    # Train on Random selection
    # random_dataset = create_filtered_dataset(RANDOM_SELECTION_PATH)
    # train_clip(random_dataset, "random_selection")
    
    print("\nExperiment script finished.")
    print("Due to the complexity of filtering a webdataset for training,")
    print("the training part is commented out. You can enable it if you have a robust way")
    print("to map selection indices to webdataset samples.")
    print("A common approach is to download the required samples first using their URLs,")
    print("which can be extracted during the feature computation phase.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="oneshot", choices=["oneshot", "hypernetwork"],
                        help="Which experiment to run.")
    args = parser.parse_args()

    if args.experiment == "oneshot":
        main()
    elif args.experiment == "hypernetwork":
        print("Running Hypernetwork-driven experiment...")
        print("The hypernetwork experiment is not fully implemented due to ambiguities in the provided pseudo-code.")
        print("Specifically, the logic for state extraction and strategy scoring needs clarification on how to use the model being trained.")
        print("Please review the comments in the generated code.")

# --- Hypernetwork-driven selection (NEW) ---

class HypernetworkDrivenSelection:
    """
    The hypernetwork makes REAL-TIME decisions every epoch
    This is the TRUE MODE architecture
    """
    
    def __init__(self):
        # This is a placeholder for your actual hypernetwork.
        # You would need to define the BinaryHypernetwork class.
        self.hypernetwork = None #BinaryHypernetwork(
            #input_dim=12,      # Binary state vector
            #hidden_dim=256,
            #output_dim=4       # Weights for 4 strategies
        #)
        
        self.strategies = [
            CurriculumStrategies.LossBasedStrategy(),
            CurriculumStrategies.ConfidenceBasedStrategy(), 
            CurriculumStrategies.DiversityStrategy(),
            CurriculumStrategies.EasyFirstStrategy()
        ]

# CRITICAL: How to extract binary training state
class BinaryStateExtractor:
    """
    This encodes the current training context into 12 binary features
    The hypernetwork uses this to decide which strategy to apply
    """
    
    def extract_training_state(self, clip_model, epoch, feature_cache):
        """
        Convert complex training state → 12 binary signals
        
        This is MODE's key insight:
        Training dynamics can be discretized into binary decisions
        
        NOTE: This is a placeholder implementation based on the user's pseudo-code.
        The pseudo-code has a logical contradiction: it tries to use the currently
        training `clip_model` on pre-computed features from a frozen model.
        A correct implementation would need to run the `clip_model` on actual
        image and text data to assess its current state.
        """
        
        image_feats, text_feats = feature_cache['image'], feature_cache['text']
        
        sample_size = 10_000
        indices = torch.randperm(len(image_feats))[:sample_size]
        
        img_sample = image_feats[indices].to(DEVICE)
        txt_sample = text_feats[indices].to(DEVICE)
        
        with torch.no_grad():
            # The following is based on the user's pseudo-code but is problematic.
            # `clip_model.encode_image_from_features` is not a standard function.
            # To correctly assess the model's state, you would need to load the
            # actual images and texts for this sample and run them through the `clip_model`.
            # As a placeholder, we use the pre-computed features directly.
            img_pred = img_sample
            txt_pred = txt_sample
            
            similarity = (img_pred * txt_pred).sum(dim=-1)
            
            logits = img_pred @ txt_pred.T / 0.07
            labels = torch.arange(len(img_sample), device=DEVICE)
            losses = F.cross_entropy(logits, labels, reduction='none')
            
        state = torch.zeros(12, dtype=torch.float32, device=DEVICE)
        
        state[0] = float(losses.mean() > 2.5)
        state[1] = float(losses.std() > 0.8)
        state[2] = float(similarity.mean() < 0.25)
        state[3] = float(epoch < 8)
        state[4] = float(8 <= epoch < 24)
        state[5] = float(epoch >= 24)
        
        # Other state dimensions would be implemented here.
        
        return state

# The 4 Strategies (Implementation)
class CurriculumStrategies:
    """
    Each strategy scores samples differently
    Hypernetwork blends them based on training state
    """
    
    class LossBasedStrategy:
        """Select samples with high loss (hard examples)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                scores = -similarity
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class ConfidenceBasedStrategy:
        """Select samples with low confidence (uncertain examples)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                optimal_uncertainty = 0.20
                scores = -torch.abs(similarity - optimal_uncertainty)
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class DiversityStrategy:
        """Select diverse, representative samples"""
        def fast_kmeans(self, sample, k):
            # A simple kmeans implementation for demonstration
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=256, n_init='auto')
            kmeans.fit(sample.cpu().numpy())
            return torch.from_numpy(kmeans.cluster_centers_).to(DEVICE)

        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                k = 100
                if not hasattr(self, 'centers'):
                    indices = torch.randperm(len(image_feats))[:10000]
                    sample = image_feats[indices]
                    self.centers = self.fast_kmeans(sample, k)
                
                distances = torch.cdist(image_feats, self.centers)
                min_distances = distances.min(dim=-1)[0]
                scores = min_distances
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

    class EasyFirstStrategy:
        """Select easy, high-confidence samples (for early training)"""
        def score_all(self, clip_model, image_feats, text_feats):
            with torch.no_grad():
                similarity = (image_feats * text_feats).sum(dim=-1)
                scores = similarity
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores

class MODEAnalyzer:
    """Analyze MODE selection patterns"""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logs = defaultdict(list)

    def log_selection(self, epoch, binary_state, strategy_weights, strategy_scores, selected_indices, state_info):
        self.logs['epoch'].append(epoch)
        self.logs['binary_state'].append(binary_state.cpu().numpy())
        self.logs['strategy_weights'].append(strategy_weights.cpu().numpy())
        self.logs['state_info'].append(state_info)

        for i, name in enumerate(['loss', 'confidence', 'diversity', 'easy']):
            scores = strategy_scores[i]
            selected_scores = scores[selected_indices]
            self.logs[f'{name}_selected_mean'].append(selected_scores.mean().item())

    def save_logs(self):
        path = self.checkpoint_dir / 'mode_analysis_logs.json'
        serializable = {k: [v.tolist() if isinstance(v, np.ndarray) else v for v in values] for k, values in self.logs.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Analyzer logs saved to {path}")

    def plot_strategy_evolution(self):
        path = self.checkpoint_dir / 'strategy_evolution.png'
        strategy_names = ['Loss-Based', 'Confidence-Based', 'Diversity-Based', 'Easy-First']
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = self.logs['epoch']
        weights = np.array(self.logs['strategy_weights'])
        for i, name in enumerate(strategy_names):
            ax.plot(epochs, weights[:, i], label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Strategy Weight')
        ax.set_title('MODE Strategy Evolution Over Training')
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        print(f"Strategy evolution plot saved to {path}")
        plt.close(fig)

class HypernetworkVisualizer:
    """Visualize hypernetwork decision boundaries"""

    def __init__(self, hypernetwork):
        self.hypernetwork = hypernetwork
        self.hypernetwork.eval()

class SampleAnalyzer:
    """Analyze characteristics of selected vs rejected samples"""

    @staticmethod
    def compare_distributions(selected_scores, rejected_scores, strategy_names):
        """Compare score distributions for selected vs rejected samples"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (ax, name) in enumerate(zip(axes.flat, strategy_names)):
            sel = selected_scores[idx].cpu().numpy()
            rej = rejected_scores[idx].cpu().numpy()
            ax.hist(sel, bins=50, alpha=0.6, label='Selected', density=True, color='green')
            ax.hist(rej, bins=50, alpha=0.6, label='Rejected', density=True, color='red')
            ax.axvline(sel.mean(), color='green', linestyle='--', linewidth=2, label=f'Selected μ={sel.mean():.3f}')
            ax.axvline(rej.mean(), color='red', linestyle='--', linewidth=2, label=f'Rejected μ={rej.mean():.3f}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle('Selected vs Rejected Sample Distributions', fontsize=14)
        plt.tight_layout()
        plt.show()

def run_hypernetwork_experiment():
    """
    Main function for the hypernetwork-driven experiment.
    """
    print("--- Starting Hypernetwork-driven Experiment ---")

    if not FEATURE_CACHE_PATH.exists():
        print(f"Feature cache not found at {FEATURE_CACHE_PATH}. Please run the 'oneshot' experiment first.")
        return

    print(f"Loading cached features from {FEATURE_CACHE_PATH}")
    features = torch.load(FEATURE_CACHE_PATH)
    image_feats = features['image'].to(DEVICE)
    text_feats = features['text'].to(DEVICE)

    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE)
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=LR, weight_decay=WD)

    # PLACEHOLDER: You need to define and load your actual hypernetwork model.
    hypernetwork = nn.Linear(12, 4).to(DEVICE)
    state_extractor = BinaryStateExtractor()
    strategies = [
        CurriculumStrategies.LossBasedStrategy(),
        CurriculumStrategies.ConfidenceBasedStrategy(),
        CurriculumStrategies.DiversityStrategy(),
        CurriculumStrategies.EasyFirstStrategy()
    ]
    analyzer = MODEAnalyzer(Path("./mode_analysis"))
    loss_history = []

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch + 1}/{EPOCHS}: Hypernetwork-driven selection")

        binary_state, state_info = state_extractor.extract_training_state(clip_model, epoch, {'image': image_feats, 'text': text_feats}, loss_history)
        loss_history.append(state_info['avg_loss'])

        strategy_weights = F.softmax(hypernetwork(binary_state), dim=-1)

        all_scores = torch.stack([s.score_all(clip_model, image_feats, text_feats) for s in strategies])

        final_scores = (strategy_weights.unsqueeze(1) * all_scores).sum(dim=0)

        k = int(DATASET_SIZE * 0.1)
        selected_indices = torch.topk(final_scores, k).indices
        print(f"Selected {k} samples for training this epoch.")

        analyzer.log_selection(epoch, binary_state, strategy_weights, all_scores, selected_indices.cpu(), state_info)

        print("TRAINING STEP SKIPPED: Implement data loading for the selected indices to train.")

    print("\nHypernetwork experiment loop finished.")
    analyzer.save_logs()
    analyzer.plot_strategy_evolution()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="oneshot", choices=["oneshot", "hypernetwork"],
                        help="Which experiment to run.")
    args = parser.parse_args()

    if args.experiment == "oneshot":
        main()
    elif args.experiment == "hypernetwork":
        run_hypernetwork_experiment()
