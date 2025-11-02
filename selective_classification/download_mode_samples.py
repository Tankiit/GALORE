#!/usr/bin/env python3
"""
Download MODE-Selected Samples from Webdataset to Local Directory

This script downloads the images selected by MODE from the streaming webdataset
and saves them locally for fast retrieval demo.

Usage:
    python download_mode_samples.py \
        --selected_indices ./datacomp_mode_cache/selected_indices.pt \
        --output_dir ./datacomp_data \
        --num_samples 30000 \
        --batch_size 128

Output structure:
    datacomp_data/
    ├── images/
    │   ├── 0000000.jpg
    │   ├── 0000001.jpg
    │   └── ...
    ├── captions.json       # {index: caption}
    ├── metadata.json       # Dataset info
    └── image_paths.txt     # List of all image paths
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Optional
import hashlib

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    sys.exit(1)

try:
    import webdataset as wds
except ImportError:
    print("Warning: 'webdataset' not found. Will try datasets library only.")
    wds = None


class MODESampleDownloader:
    """Download MODE-selected samples from webdataset to local directory."""

    def __init__(
        self,
        selected_indices_path: str,
        output_dir: str,
        num_samples: int = 30000,
        cache_dir: str = "./datacomp_cache",
        max_dataset_samples: int = 100000
    ):
        self.selected_indices_path = Path(selected_indices_path)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.cache_dir = cache_dir
        self.max_dataset_samples = max_dataset_samples

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load selected indices
        print(f"Loading selected indices from {self.selected_indices_path}...")
        self.selected_indices = self._load_indices()
        print(f"Loaded {len(self.selected_indices)} selected indices")

        # Limit to num_samples
        if len(self.selected_indices) > num_samples:
            print(f"Limiting to first {num_samples} samples")
            self.selected_indices = self.selected_indices[:num_samples]

        # Convert to set for fast lookup
        self.selected_set = set(self.selected_indices.tolist())

    def _load_indices(self) -> torch.Tensor:
        """Load selected indices from file."""
        if not self.selected_indices_path.exists():
            raise FileNotFoundError(f"Selected indices not found: {self.selected_indices_path}")

        indices = torch.load(self.selected_indices_path, map_location='cpu')

        # Handle different formats
        if isinstance(indices, dict):
            if 'selected_indices' in indices:
                indices = indices['selected_indices']
            elif 'indices' in indices:
                indices = indices['indices']

        if isinstance(indices, list):
            indices = torch.tensor(indices)

        return indices

    def download_samples(self, batch_size: int = 128):
        """Download selected samples from webdataset."""
        print("\n" + "="*80)
        print("DOWNLOADING MODE-SELECTED SAMPLES")
        print("="*80)
        print(f"Selected indices: {len(self.selected_indices)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Cache directory: {self.cache_dir}")
        print("="*80 + "\n")

        # Try to load dataset
        try:
            print("Loading DataComp dataset from HuggingFace...")
            dataset = load_dataset(
                "mlfoundations/datacomp_small",
                split="train",
                streaming=True,
                cache_dir=self.cache_dir
            )
            print("Dataset loaded successfully!")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative method...")
            try:
                dataset = load_dataset(
                    "webdataset",
                    data_dir=self.cache_dir,
                    split="train",
                    streaming=True
                )
                print("Dataset loaded via webdataset!")
            except Exception as e2:
                print(f"Error: {e2}")
                print("\nCannot load dataset. Please check:")
                print("1. Internet connection")
                print("2. HuggingFace datasets library is installed")
                print("3. Cache directory exists and is accessible")
                sys.exit(1)

        # Download samples
        downloaded = 0
        captions_dict = {}
        image_paths = []

        print(f"\nDownloading {len(self.selected_indices)} selected samples...")
        print("This may take 1-3 hours depending on network speed.\n")

        # Create progress bar
        pbar = tqdm(total=len(self.selected_indices), desc="Downloading")

        # Iterate through dataset
        for idx, sample in enumerate(dataset):
            # Stop if we've exceeded max dataset samples
            if idx >= self.max_dataset_samples:
                break

            # Check if this sample is selected
            if idx not in self.selected_set:
                continue

            try:
                # Extract image and caption
                if isinstance(sample, dict):
                    if 'jpg' in sample:
                        image = sample['jpg']
                    elif 'png' in sample:
                        image = sample['png']
                    elif 'image' in sample:
                        image = sample['image']
                    else:
                        print(f"Warning: No image key in sample {idx}")
                        continue

                    caption = sample.get('txt', sample.get('caption', f"Sample {idx}"))
                else:
                    print(f"Warning: Unexpected sample format at index {idx}")
                    continue

                # Convert to PIL Image if needed
                if not isinstance(image, Image.Image):
                    if isinstance(image, bytes):
                        from io import BytesIO
                        image = Image.open(BytesIO(image))
                    else:
                        print(f"Warning: Cannot convert image at index {idx}")
                        continue

                # Save image
                image_filename = f"{idx:07d}.jpg"
                image_path = self.images_dir / image_filename

                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image.save(image_path, 'JPEG', quality=95)

                # Store caption and path
                captions_dict[idx] = caption
                image_paths.append(str(image_path))

                downloaded += 1
                pbar.update(1)

                # Stop if we've downloaded all selected samples
                if downloaded >= len(self.selected_indices):
                    break

            except Exception as e:
                print(f"\nWarning: Error processing sample {idx}: {e}")
                continue

        pbar.close()

        print(f"\n✓ Successfully downloaded {downloaded} samples!")

        # Save metadata
        print("\nSaving metadata...")
        self._save_metadata(captions_dict, image_paths, downloaded)

        return downloaded

    def _save_metadata(self, captions_dict: Dict, image_paths: List[str], num_downloaded: int):
        """Save metadata files."""
        # Save captions
        captions_file = self.output_dir / "captions.json"
        with open(captions_file, 'w') as f:
            json.dump(captions_dict, f, indent=2)
        print(f"✓ Saved captions: {captions_file}")

        # Save image paths
        paths_file = self.output_dir / "image_paths.txt"
        with open(paths_file, 'w') as f:
            f.write('\n'.join(image_paths))
        print(f"✓ Saved image paths: {paths_file}")

        # Save metadata
        metadata = {
            "num_samples": num_downloaded,
            "selected_indices_file": str(self.selected_indices_path),
            "source": "datacomp_small",
            "format": "MODE-selected subset",
            "image_format": "JPEG",
            "image_directory": str(self.images_dir),
            "captions_file": str(captions_file),
            "paths_file": str(paths_file)
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")

        # Print summary
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"Downloaded: {num_downloaded} samples")
        print(f"Location: {self.output_dir}")
        print(f"Images: {self.images_dir}")
        print(f"Disk usage: ~{num_downloaded * 50 / 1024:.1f} MB")
        print("="*80 + "\n")

        print("Next steps:")
        print("1. Build retrieval database:")
        print(f"   ./build_demo_pipeline.sh --data_dir {self.output_dir}")
        print("\n2. Or use the images directly in your own experiments")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download MODE-selected samples from webdataset to local directory"
    )
    parser.add_argument(
        "--selected_indices",
        type=str,
        required=True,
        help="Path to selected indices .pt file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datacomp_data",
        help="Output directory for downloaded samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30000,
        help="Number of samples to download (default: 30000)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./datacomp_cache",
        help="Cache directory for webdataset"
    )
    parser.add_argument(
        "--max_dataset_samples",
        type=int,
        default=100000,
        help="Max samples to iterate through in dataset (default: 100000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for downloading (not used currently)"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = MODESampleDownloader(
        selected_indices_path=args.selected_indices,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        max_dataset_samples=args.max_dataset_samples
    )

    # Download samples
    try:
        num_downloaded = downloader.download_samples(batch_size=args.batch_size)

        if num_downloaded == 0:
            print("\nError: No samples were downloaded!")
            print("Please check:")
            print("1. Selected indices file is valid")
            print("2. Dataset is accessible")
            print("3. Internet connection is stable")
            sys.exit(1)

        print("\n✓ SUCCESS! Ready to build demo databases.")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Partial download saved. Run again to continue.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
