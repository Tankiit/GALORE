#!/usr/bin/env python3
"""
Build Retrieval Database for Interactive Demo

Pre-computes CLIP embeddings for your dataset to enable fast interactive retrieval.

Usage:
    # From your dataset
    python build_retrieval_database.py \
        --data_dir ./datacomp_data \
        --model_path ./mode_output/final_clip_model.pt \
        --output_path ./retrieval_database.pt \
        --max_samples 1000

    # From selected indices
    python build_retrieval_database.py \
        --data_dir ./datacomp_data \
        --selected_indices ./datacomp_mode_cache/selected_indices.pt \
        --model_path ./mode_output/final_clip_model.pt \
        --output_path ./retrieval_database.pt
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Optional
from PIL import Image
import json

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    print("Error: transformers not installed!")
    print("Install with: pip install transformers")
    HAS_TRANSFORMERS = False


def build_database(
    data_dir: str,
    model_path: Optional[str],
    output_path: str,
    max_samples: int = 1000,
    selected_indices: Optional[str] = None,
    clip_model_name: str = 'openai/clip-vit-base-patch32',
    device: str = 'auto'
):
    """
    Build retrieval database with pre-computed embeddings.

    Args:
        data_dir: Directory containing images and captions
        model_path: Path to trained CLIP model (optional)
        output_path: Where to save database
        max_samples: Maximum number of samples to include
        selected_indices: Path to MODE selected indices (optional)
        clip_model_name: HuggingFace model name
        device: Device to use
    """

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print("Building Retrieval Database")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Output: {output_path}\n")

    # Load CLIP model
    print("Loading CLIP model...")
    if model_path and Path(model_path).exists():
        print(f"  Loading trained model: {model_path}")
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        state_dict = torch.load(model_path, map_location='cpu')
        clip_model.load_state_dict(state_dict)
    else:
        print(f"  Loading pretrained model: {clip_model_name}")
        clip_model = CLIPModel.from_pretrained(clip_model_name)

    clip_model = clip_model.to(device)
    clip_model.eval()

    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Load dataset
    print("\nLoading dataset...")
    data_path = Path(data_dir)

    # Try to load from selected indices first
    if selected_indices and Path(selected_indices).exists():
        print(f"  Loading selected indices from: {selected_indices}")
        indices_data = torch.load(selected_indices)
        if isinstance(indices_data, dict):
            indices = indices_data['indices']
        else:
            indices = indices_data
        print(f"  Found {len(indices)} selected samples")
    else:
        indices = None

    # Collect image paths and captions
    images = []
    captions = []
    image_paths = []

    # Method 1: Try to load from metadata file
    metadata_file = data_path / 'metadata.json'
    if metadata_file.exists():
        print("  Loading from metadata.json...")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if indices is not None:
            # Filter by selected indices
            for idx in indices[:max_samples]:
                if idx < len(metadata):
                    item = metadata[idx]
                    img_path = data_path / item.get('image', item.get('file_name', ''))
                    if img_path.exists():
                        image_paths.append(str(img_path))
                        captions.append(item.get('caption', item.get('text', '')))
        else:
            # Use all samples
            for i, item in enumerate(metadata[:max_samples]):
                img_path = data_path / item.get('image', item.get('file_name', ''))
                if img_path.exists():
                    image_paths.append(str(img_path))
                    captions.append(item.get('caption', item.get('text', '')))

    # Method 2: Scan directory for images
    if not image_paths:
        print("  Scanning directory for images...")
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        for ext in extensions:
            image_files = list(data_path.glob(f'**/*{ext}'))
            image_paths.extend([str(p) for p in image_files[:max_samples]])
            if len(image_paths) >= max_samples:
                break

        # Try to find corresponding captions
        print("  Looking for caption files...")
        for img_path in image_paths:
            caption_path = Path(img_path).with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r') as f:
                    captions.append(f.read().strip())
            else:
                # Use filename as caption
                captions.append(Path(img_path).stem.replace('_', ' '))

    print(f"\n  Found {len(image_paths)} images")
    print(f"  Found {len(captions)} captions")

    if not image_paths:
        print("\nError: No images found!")
        print("Please check your data directory or provide metadata.json")
        return

    # Encode images
    print("\nEncoding images...")
    image_embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Images"):
            batch_paths = image_paths[i:i+batch_size]

            # Load images
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
                    continue

            if not batch_images:
                continue

            # Process
            inputs = processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            features = clip_model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
            image_embeddings.append(features.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    print(f"  Encoded {image_embeddings.shape[0]} images → {image_embeddings.shape[1]}-dim")

    # Encode captions
    print("\nEncoding captions...")
    caption_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="Captions"):
            batch_captions = captions[i:i+batch_size]

            inputs = processor(
                text=batch_captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            features = clip_model.get_text_features(**inputs)
            features = F.normalize(features, dim=-1)
            caption_embeddings.append(features.cpu())

    caption_embeddings = torch.cat(caption_embeddings, dim=0)
    print(f"  Encoded {caption_embeddings.shape[0]} captions → {caption_embeddings.shape[1]}-dim")

    # Save database
    print(f"\nSaving database to: {output_path}")
    database = {
        'image_embeddings': image_embeddings,
        'caption_embeddings': caption_embeddings,
        'image_paths': image_paths,
        'captions': captions,
        'metadata': {
            'num_samples': len(image_paths),
            'embedding_dim': image_embeddings.shape[1],
            'model': clip_model_name,
            'trained_model_path': model_path if model_path else 'pretrained'
        }
    }

    torch.save(database, output_path)
    print(f"✓ Database saved successfully!")

    # Print statistics
    print(f"\n{'='*70}")
    print("Database Statistics")
    print(f"{'='*70}")
    print(f"  Images:        {len(image_paths)}")
    print(f"  Captions:      {len(captions)}")
    print(f"  Embedding dim: {image_embeddings.shape[1]}")
    print(f"  File size:     {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")

    # Compute some statistics
    print("Sample captions:")
    for i, caption in enumerate(captions[:5]):
        print(f"  {i+1}. {caption}")

    # Compute average similarity
    with torch.no_grad():
        avg_sim = (image_embeddings @ caption_embeddings.T).diag().mean().item()
        print(f"\nAverage image-text similarity: {avg_sim:.4f}")

    print(f"\n✓ Database ready for interactive retrieval!")
    print(f"\nTo use:")
    print(f"  python interactive_retrieval_demo.py \\")
    print(f"      --database_path {output_path} \\")
    print(f"      --model_path {model_path if model_path else 'pretrained'}")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval database")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing images and captions')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained CLIP model')
    parser.add_argument('--output_path', type=str, default='./retrieval_database.pt',
                       help='Output path for database')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples')
    parser.add_argument('--selected_indices', type=str, default=None,
                       help='Path to MODE selected indices')
    parser.add_argument('--clip_model_name', type=str,
                       default='openai/clip-vit-base-patch32',
                       help='HuggingFace CLIP model name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: transformers not installed!")
        return

    build_database(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        selected_indices=args.selected_indices,
        clip_model_name=args.clip_model_name,
        device=args.device
    )


if __name__ == '__main__':
    main()
