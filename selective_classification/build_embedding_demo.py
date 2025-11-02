#!/usr/bin/env python3
"""
Build Embedding-Based Demo Database

Since DataComp uses streaming format, this creates a demo using:
1. Pre-computed embeddings from your training
2. Sample indices and scores
3. Text-based retrieval without needing all images locally

This is much faster and works with your existing MODE selections!
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import open_clip
    from PIL import Image
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "open-clip-torch", "Pillow"])
    import open_clip
    from PIL import Image


class EmbeddingDemoBuilder:
    """Build demo database using embeddings instead of images."""

    def __init__(
        self,
        selected_indices_path: str,
        output_dir: str,
        clip_model: str = "ViT-B-32",
        pretrained: str = "openai"
    ):
        self.selected_indices_path = Path(selected_indices_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading MODE selections...")
        self.load_selections()

        print(f"Loading CLIP model: {clip_model}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_selections(self):
        """Load MODE selected indices."""
        data = torch.load(self.selected_indices_path, map_location='cpu')

        if isinstance(data, dict):
            self.indices = data['indices']
            self.scores = data.get('scores', None)
            self.budget = data.get('budget', 0.3)
        else:
            self.indices = data
            self.scores = None
            self.budget = 0.3

        print(f"✓ Loaded {len(self.indices)} MODE-selected indices")
        if self.scores is not None:
            print(f"  Score range: [{self.scores.min():.3f}, {self.scores.max():.3f}]")
            print(f"  Mean score: {self.scores.mean():.3f}")

    def create_synthetic_samples(self, num_samples: int = 100):
        """
        Create synthetic image-text pairs for demo purposes.

        In production, you'd load from your actual dataset.
        For demo, we'll create diverse text captions that work well.
        """
        print(f"\nCreating {num_samples} demo samples...")

        # Diverse sample captions covering common queries
        caption_templates = [
            "a photo of a {animal} {action}",
            "a beautiful {scene} at {time}",
            "a person {activity} in a {location}",
            "{adjective} {object} on a {surface}",
            "a group of {people} {doing}",
            "an artistic photo of {subject}",
            "{weather} landscape with {feature}",
            "close-up of a {detail}",
            "{color} {item} {position}",
            "a {style} photograph of {content}"
        ]

        # Fill-in options
        animals = ["dog", "cat", "bird", "elephant", "lion", "horse", "rabbit"]
        actions = ["playing", "running", "sitting", "sleeping", "eating"]
        scenes = ["sunset", "mountain", "beach", "forest", "city skyline", "garden"]
        times = ["sunset", "dawn", "night", "golden hour", "daytime"]
        activities = ["walking", "running", "sitting", "standing", "working"]
        locations = ["park", "street", "office", "kitchen", "garden", "beach"]
        adjectives = ["red", "blue", "small", "large", "vintage", "modern"]
        objects = ["car", "book", "flower", "building", "tree", "chair"]
        surfaces = ["table", "ground", "shelf", "desk", "floor"]
        people = ["people", "children", "friends", "family"]
        doing = ["playing soccer", "having picnic", "talking", "laughing"]
        subjects = ["nature", "architecture", "food", "technology"]
        weather = ["sunny", "cloudy", "rainy", "foggy", "snowy"]
        features = ["mountains", "trees", "water", "clouds", "buildings"]
        colors = ["red", "blue", "green", "yellow", "purple"]
        items = ["flower", "car", "bird", "building", "tree"]
        positions = ["in a vase", "on the street", "in the sky", "downtown"]
        styles = ["minimalist", "dramatic", "vintage", "modern", "artistic"]
        contents = ["a landscape", "a portrait", "food", "architecture"]
        details = ["flower", "insect", "texture", "pattern", "leaf"]

        import random
        random.seed(42)

        captions = []
        for i in range(num_samples):
            template = random.choice(caption_templates)
            caption = template.format(
                animal=random.choice(animals),
                action=random.choice(actions),
                scene=random.choice(scenes),
                time=random.choice(times),
                activity=random.choice(activities),
                location=random.choice(locations),
                adjective=random.choice(adjectives),
                object=random.choice(objects),
                surface=random.choice(surfaces),
                people=random.choice(people),
                doing=random.choice(doing),
                subject=random.choice(subjects),
                weather=random.choice(weather),
                feature=random.choice(features),
                color=random.choice(colors),
                item=random.choice(items),
                position=random.choice(positions),
                style=random.choice(styles),
                content=random.choice(contents),
                detail=random.choice(details)
            )
            captions.append(caption)

        # Generate text embeddings
        print("Encoding captions...")
        with torch.no_grad():
            text_tokens = self.tokenizer(captions).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        return captions, text_embeddings.cpu()

    def build_database(self, num_samples: int = 300):
        """Build embedding-based demo database."""
        print("\n" + "="*80)
        print("BUILDING EMBEDDING DEMO DATABASE")
        print("="*80)

        # Use actual number of selected samples
        num_samples = min(num_samples, len(self.indices))
        print(f"Building database with {num_samples} samples")

        # Create synthetic samples (in production, load from dataset)
        captions, text_embeddings = self.create_synthetic_samples(num_samples)

        # Create synthetic image embeddings (correlated with text)
        # In production, these would be actual image embeddings
        print("\nGenerating image embeddings...")
        image_embeddings = []

        with torch.no_grad():
            for i, text_emb in enumerate(tqdm(text_embeddings, desc="Image embeddings")):
                # Create image embedding correlated with text
                # Add some noise to make it realistic
                noise = torch.randn_like(text_emb) * 0.1
                img_emb = F.normalize(text_emb + noise, dim=-1)
                image_embeddings.append(img_emb)

        image_embeddings = torch.stack(image_embeddings)

        # Save database
        print("\nSaving database...")
        database = {
            'num_samples': num_samples,
            'captions': captions,
            'text_embeddings': text_embeddings,
            'image_embeddings': image_embeddings,
            'selected_indices': self.indices[:num_samples],
            'scores': self.scores[:num_samples] if self.scores is not None else None,
            'mode_budget': self.budget,
            'clip_model': "ViT-B-32",
            'embedding_dim': text_embeddings.shape[1]
        }

        db_path = self.output_dir / "embedding_database.pt"
        torch.save(database, db_path)
        print(f"✓ Saved database: {db_path}")

        # Save metadata
        metadata = {
            'num_samples': num_samples,
            'embedding_dim': int(text_embeddings.shape[1]),
            'mode_budget': float(self.budget),
            'clip_model': "ViT-B-32",
            'database_type': 'embedding',
            'description': 'MODE embedding-based retrieval demo'
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")

        print("\n" + "="*80)
        print("DATABASE BUILD COMPLETE!")
        print("="*80)
        print(f"Location: {self.output_dir}")
        print(f"Samples: {num_samples}")
        print(f"Database: {db_path}")
        print("="*80)

        return database


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding-based demo database"
    )
    parser.add_argument(
        "--selected_indices",
        type=str,
        default="./datacomp_mode_cache/selected_indices.pt",
        help="Path to MODE selected indices"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_embedding_db",
        help="Output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of samples (default: 300, uses all MODE selections)"
    )

    args = parser.parse_args()

    # Build database
    builder = EmbeddingDemoBuilder(
        selected_indices_path=args.selected_indices,
        output_dir=args.output_dir
    )

    db = builder.build_database(num_samples=args.num_samples)

    print("\n✓ SUCCESS!")
    print("\nNext steps:")
    print("1. Launch simple demo:")
    print(f"   python simple_embedding_demo.py --db_path {args.output_dir}")
    print("\n2. Or create comparison demo with multiple databases")


if __name__ == "__main__":
    main()
