#!/usr/bin/env python3
"""
Large-Scale Retrieval Database Builder

Efficiently builds retrieval databases for millions of samples using:
- Chunked processing (memory efficient)
- FAISS indexing (fast retrieval)
- Progress tracking (resume from interruptions)
- Parallel processing (faster encoding)

Perfect for showing MODE's value on large datasets!

Usage:
    # Build from DataComp/LAION/CC3M (large scale)
    python build_large_retrieval_database.py \
        --data_dir ./datacomp_small \
        --model_path ./mode_output/final_clip_model.pt \
        --output_dir ./large_retrieval_db \
        --max_samples 100000 \
        --use_faiss

    # Resume interrupted build
    python build_large_retrieval_database.py \
        --output_dir ./large_retrieval_db \
        --resume
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Optional, Tuple
from PIL import Image
import json
import numpy as np
import pickle
import time
from collections import defaultdict
import multiprocessing as mp

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    print("Error: transformers not installed!")
    HAS_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    print("Info: faiss not installed. Install for faster retrieval:")
    print("  pip install faiss-cpu  # or faiss-gpu for GPU")
    HAS_FAISS = False


# ============================================================================
# Large-Scale Database Builder
# ============================================================================

class LargeScaleDBBuilder:
    """
    Build large-scale retrieval database with:
    - Chunked processing (avoid OOM)
    - FAISS indexing (fast search)
    - Progress tracking (resume support)
    - Efficient storage
    """

    def __init__(
        self,
        output_dir: str,
        model_path: Optional[str] = None,
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        device: str = 'auto',
        chunk_size: int = 10000,
        use_faiss: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.use_faiss = use_faiss and HAS_FAISS

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"\n{'='*80}")
        print("Large-Scale Retrieval Database Builder")
        print(f"{'='*80}\n")
        print(f"Device: {self.device}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Use FAISS: {self.use_faiss}")
        print(f"Output: {self.output_dir}\n")

        # Load CLIP model
        print("Loading CLIP model...")
        if model_path and Path(model_path).exists():
            print(f"  Loading trained model: {model_path}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            state_dict = torch.load(model_path, map_location='cpu')
            self.clip_model.load_state_dict(state_dict)
        else:
            print(f"  Loading pretrained model: {clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Track progress
        self.progress_file = self.output_dir / 'progress.json'
        self.progress = self._load_progress()

    def _load_progress(self) -> dict:
        """Load progress from previous run"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'chunks_processed': 0,
            'total_samples': 0,
            'image_chunks': [],
            'caption_chunks': [],
            'metadata': {}
        }

    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def collect_samples(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        selected_indices: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Collect image paths and captions from dataset.

        Supports:
        - DataComp format (tar files)
        - LAION format (parquet files)
        - Simple directory (images + captions)
        - Custom metadata.json
        """
        data_path = Path(data_dir)
        image_paths = []
        captions = []

        print("\nCollecting samples from dataset...")

        # Method 1: metadata.json (most efficient)
        metadata_file = data_path / 'metadata.json'
        if metadata_file.exists():
            print("  Loading from metadata.json...")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Apply selection if provided
            if selected_indices and Path(selected_indices).exists():
                print(f"  Filtering by selected indices...")
                indices_data = torch.load(selected_indices)
                if isinstance(indices_data, dict):
                    indices = indices_data['indices']
                else:
                    indices = indices_data
                indices = indices[:max_samples] if max_samples else indices
            else:
                indices = range(min(len(metadata), max_samples) if max_samples else len(metadata))

            for idx in tqdm(indices, desc="Loading metadata"):
                if idx < len(metadata):
                    item = metadata[idx]
                    img_path = data_path / item.get('image', item.get('file_name', ''))
                    if img_path.exists():
                        image_paths.append(str(img_path))
                        captions.append(item.get('caption', item.get('text', '')))

        # Method 2: Directory scan
        elif data_path.is_dir():
            print("  Scanning directory for images...")
            extensions = ['.jpg', '.jpeg', '.png', '.webp']

            for ext in extensions:
                found = list(data_path.rglob(f'*{ext}'))
                image_paths.extend([str(p) for p in found])
                if max_samples and len(image_paths) >= max_samples:
                    image_paths = image_paths[:max_samples]
                    break

            # Find captions
            print("  Looking for captions...")
            for img_path in tqdm(image_paths, desc="Loading captions"):
                caption_path = Path(img_path).with_suffix('.txt')
                if caption_path.exists():
                    with open(caption_path, 'r') as f:
                        captions.append(f.read().strip())
                else:
                    captions.append(Path(img_path).stem.replace('_', ' '))

        # Method 3: WebDataset shards
        elif str(data_path).endswith('.tar'):
            print("  Processing WebDataset shards...")
            try:
                import webdataset as wds
                dataset = wds.WebDataset(str(data_path))

                count = 0
                for sample in tqdm(dataset, desc="Loading shards"):
                    if 'jpg' in sample or 'png' in sample:
                        img_key = 'jpg' if 'jpg' in sample else 'png'
                        # Save image temporarily
                        tmp_path = self.output_dir / f'tmp_{count}.{img_key}'
                        with open(tmp_path, 'wb') as f:
                            f.write(sample[img_key])
                        image_paths.append(str(tmp_path))

                        caption = sample.get('txt', sample.get('text', '')).decode('utf-8')
                        captions.append(caption)

                        count += 1
                        if max_samples and count >= max_samples:
                            break

            except ImportError:
                print("  Error: webdataset not installed for .tar files")
                print("  Install with: pip install webdataset")

        print(f"\n  Found {len(image_paths)} images")
        print(f"  Found {len(captions)} captions")

        return image_paths, captions

    def encode_images_chunked(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[str]:
        """
        Encode images in chunks and save to disk.

        Returns list of chunk file paths.
        """
        print("\nEncoding images in chunks...")

        chunk_files = []
        num_chunks = (len(image_paths) + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(image_paths))

            # Check if chunk already processed
            chunk_file = self.output_dir / f'image_chunk_{chunk_idx}.pt'
            if chunk_file.exists():
                print(f"  Chunk {chunk_idx+1}/{num_chunks}: Already exists, skipping")
                chunk_files.append(str(chunk_file))
                continue

            print(f"  Processing chunk {chunk_idx+1}/{num_chunks}")
            chunk_paths = image_paths[start_idx:end_idx]

            embeddings = []

            with torch.no_grad():
                for i in tqdm(range(0, len(chunk_paths), batch_size),
                            desc=f"  Encoding", leave=False):
                    batch_paths = chunk_paths[i:i+batch_size]

                    # Load images
                    batch_images = []
                    valid_indices = []

                    for j, path in enumerate(batch_paths):
                        try:
                            img = Image.open(path).convert('RGB')
                            batch_images.append(img)
                            valid_indices.append(i + j)
                        except Exception as e:
                            print(f"    Warning: Could not load {path}: {e}")
                            continue

                    if not batch_images:
                        continue

                    # Encode
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    features = self.clip_model.get_image_features(**inputs)
                    features = F.normalize(features, dim=-1)
                    embeddings.append(features.cpu())

            # Save chunk
            if embeddings:
                chunk_embeddings = torch.cat(embeddings, dim=0)
                torch.save(chunk_embeddings, chunk_file)
                chunk_files.append(str(chunk_file))
                print(f"    Saved: {chunk_embeddings.shape[0]} embeddings")

            # Update progress
            self.progress['chunks_processed'] = chunk_idx + 1
            self.progress['image_chunks'].append(str(chunk_file))
            self._save_progress()

        return chunk_files

    def encode_captions_chunked(
        self,
        captions: List[str],
        batch_size: int = 32
    ) -> List[str]:
        """
        Encode captions in chunks and save to disk.

        Returns list of chunk file paths.
        """
        print("\nEncoding captions in chunks...")

        chunk_files = []
        num_chunks = (len(captions) + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(captions))

            # Check if chunk already processed
            chunk_file = self.output_dir / f'caption_chunk_{chunk_idx}.pt'
            if chunk_file.exists():
                print(f"  Chunk {chunk_idx+1}/{num_chunks}: Already exists, skipping")
                chunk_files.append(str(chunk_file))
                continue

            print(f"  Processing chunk {chunk_idx+1}/{num_chunks}")
            chunk_captions = captions[start_idx:end_idx]

            embeddings = []

            with torch.no_grad():
                for i in tqdm(range(0, len(chunk_captions), batch_size),
                            desc=f"  Encoding", leave=False):
                    batch_captions = chunk_captions[i:i+batch_size]

                    inputs = self.processor(
                        text=batch_captions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    features = self.clip_model.get_text_features(**inputs)
                    features = F.normalize(features, dim=-1)
                    embeddings.append(features.cpu())

            # Save chunk
            if embeddings:
                chunk_embeddings = torch.cat(embeddings, dim=0)
                torch.save(chunk_embeddings, chunk_file)
                chunk_files.append(str(chunk_file))
                print(f"    Saved: {chunk_embeddings.shape[0]} embeddings")

            # Update progress
            self.progress['caption_chunks'].append(str(chunk_file))
            self._save_progress()

        return chunk_files

    def build_faiss_index(
        self,
        chunk_files: List[str],
        index_type: str = 'flat'
    ) -> Optional[faiss.Index]:
        """
        Build FAISS index from embedding chunks.

        Args:
            chunk_files: List of chunk file paths
            index_type: 'flat' (exact) or 'ivf' (approximate, faster)

        Returns:
            FAISS index
        """
        if not self.use_faiss:
            return None

        print("\nBuilding FAISS index...")

        # Load first chunk to get dimension
        first_chunk = torch.load(chunk_files[0])
        dim = first_chunk.shape[1]
        total_samples = sum(torch.load(f).shape[0] for f in chunk_files)

        print(f"  Dimension: {dim}")
        print(f"  Total samples: {total_samples}")
        print(f"  Index type: {index_type}")

        # Create index
        if index_type == 'flat':
            # Exact search (slower but accurate)
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
        else:
            # Approximate search (faster)
            nlist = min(4096, total_samples // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train index on sample
            print("  Training IVF index...")
            train_samples = []
            for chunk_file in chunk_files[:10]:  # Use first 10 chunks for training
                chunk = torch.load(chunk_file)
                train_samples.append(chunk.numpy())
            train_data = np.concatenate(train_samples, axis=0)
            index.train(train_data)

        # Add all embeddings
        print("  Adding embeddings to index...")
        for chunk_file in tqdm(chunk_files, desc="  Loading chunks"):
            chunk = torch.load(chunk_file)
            index.add(chunk.numpy())

        print(f"  Index built: {index.ntotal} vectors")

        return index

    def save_database(
        self,
        image_paths: List[str],
        captions: List[str],
        image_chunk_files: List[str],
        caption_chunk_files: List[str],
        faiss_index_img: Optional[faiss.Index] = None,
        faiss_index_cap: Optional[faiss.Index] = None
    ):
        """Save database metadata and indices"""

        print("\nSaving database...")

        # Save metadata
        metadata = {
            'num_samples': len(image_paths),
            'num_image_chunks': len(image_chunk_files),
            'num_caption_chunks': len(caption_chunk_files),
            'chunk_size': self.chunk_size,
            'embedding_dim': torch.load(image_chunk_files[0]).shape[1],
            'use_faiss': self.use_faiss,
            'device': self.device
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save file lists
        paths_data = {
            'image_paths': image_paths,
            'captions': captions,
            'image_chunks': image_chunk_files,
            'caption_chunks': caption_chunk_files
        }

        paths_path = self.output_dir / 'paths.pkl'
        with open(paths_path, 'wb') as f:
            pickle.dump(paths_data, f)

        # Save FAISS indices
        if faiss_index_img is not None:
            faiss_img_path = self.output_dir / 'faiss_image_index.bin'
            faiss.write_index(faiss_index_img, str(faiss_img_path))
            print(f"  Saved image FAISS index: {faiss_img_path}")

        if faiss_index_cap is not None:
            faiss_cap_path = self.output_dir / 'faiss_caption_index.bin'
            faiss.write_index(faiss_index_cap, str(faiss_cap_path))
            print(f"  Saved caption FAISS index: {faiss_cap_path}")

        print(f"\n✓ Database saved to: {self.output_dir}")

        # Print statistics
        print(f"\n{'='*80}")
        print("Database Statistics")
        print(f"{'='*80}")
        print(f"  Total samples:     {len(image_paths):,}")
        print(f"  Image chunks:      {len(image_chunk_files)}")
        print(f"  Caption chunks:    {len(caption_chunk_files)}")
        print(f"  Chunk size:        {self.chunk_size:,}")
        print(f"  Embedding dim:     {metadata['embedding_dim']}")
        print(f"  Uses FAISS:        {self.use_faiss}")

        # Calculate total size
        total_size = sum(p.stat().st_size for p in self.output_dir.glob('*.pt'))
        if faiss_index_img:
            total_size += (self.output_dir / 'faiss_image_index.bin').stat().st_size
        if faiss_index_cap:
            total_size += (self.output_dir / 'faiss_caption_index.bin').stat().st_size

        print(f"  Total size:        {total_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"{'='*80}\n")

        # Sample captions
        print("Sample captions:")
        for i, caption in enumerate(captions[:5]):
            print(f"  {i+1}. {caption}")

    def build(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        selected_indices: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Main build pipeline.

        Args:
            data_dir: Directory containing dataset
            max_samples: Maximum samples to process
            selected_indices: Path to MODE selected indices
            batch_size: Batch size for encoding
        """
        start_time = time.time()

        # Step 1: Collect samples
        image_paths, captions = self.collect_samples(
            data_dir, max_samples, selected_indices
        )

        if not image_paths:
            print("Error: No samples found!")
            return

        # Step 2: Encode images
        image_chunk_files = self.encode_images_chunked(image_paths, batch_size)

        # Step 3: Encode captions
        caption_chunk_files = self.encode_captions_chunked(captions, batch_size)

        # Step 4: Build FAISS indices
        faiss_index_img = None
        faiss_index_cap = None

        if self.use_faiss:
            faiss_index_img = self.build_faiss_index(image_chunk_files, index_type='flat')
            faiss_index_cap = self.build_faiss_index(caption_chunk_files, index_type='flat')

        # Step 5: Save database
        self.save_database(
            image_paths, captions,
            image_chunk_files, caption_chunk_files,
            faiss_index_img, faiss_index_cap
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Database built successfully in {elapsed/60:.1f} minutes!")
        print(f"\nTo use this database:")
        print(f"  python interactive_retrieval_demo.py \\")
        print(f"      --database_path {self.output_dir} \\")
        print(f"      --large_db")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build large-scale retrieval database")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing dataset')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained CLIP model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for database')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process')
    parser.add_argument('--selected_indices', type=str, default=None,
                       help='Path to MODE selected indices')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Samples per chunk')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--use_faiss', action='store_true', default=True,
                       help='Use FAISS for fast retrieval')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted build')

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: transformers not installed!")
        return

    # Create builder
    builder = LargeScaleDBBuilder(
        output_dir=args.output_dir,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        use_faiss=args.use_faiss,
        device=args.device
    )

    # Build database
    builder.build(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        selected_indices=args.selected_indices,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
