#!/usr/bin/env python3
"""
Initialize and save all WikiText-103 dataset variations
"""

import os
import sys
import logging
from tqdm import tqdm

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_datasets import (
    create_wikitext_datasets,
    save_dataset_metadata,
    WikiText103Dataset,
    stratify_by_quality,
    stratify_by_length,
    stratify_by_topic,
    stratify_by_complexity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_all_datasets(max_samples: int = None, data_dir: str = "/Users/mukher74/research/data"):
    """
    Initialize and save all WikiText-103 dataset variations
    """
    logger.info("=" * 60)
    logger.info("Initializing WikiText-103 Dataset Variations")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Create all dataset variations
        logger.info("\nCreating dataset variations...")
        datasets = create_wikitext_datasets(max_samples=max_samples, cache=True)
        
        # Display statistics
        logger.info("\n" + "=" * 60)
        logger.info("Dataset Statistics")
        logger.info("=" * 60)
        
        for name, dataset in datasets.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Total samples: {len(dataset)}")
            
            if hasattr(dataset, 'get_stratum_stats'):
                stats = dataset.get_stratum_stats()
                logger.info(f"  Stratification type: {stats['type']}")
                logger.info(f"  Number of strata: {stats['num_strata']}")
                logger.info(f"  Balanced: {stats['balanced']}")
                logger.info(f"  Strata distribution:")
                for stratum, size in stats['strata_sizes'].items():
                    logger.info(f"    {stratum}: {size} samples")
        
        # Save metadata
        logger.info("\n" + "=" * 60)
        logger.info("Saving Metadata")
        logger.info("=" * 60)
        save_dataset_metadata(datasets, data_dir)
        
        # Create a summary file
        summary_file = os.path.join(data_dir, "wikitext103_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("WikiText-103 Dataset Variations Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Available Datasets:\n")
            f.write("-" * 40 + "\n")
            for name, dataset in datasets.items():
                f.write(f"{name}: {len(dataset)} samples\n")
            
            f.write("\n\nUsage Examples:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Load full dataset:\n")
            f.write("   from llm_datasets import create_wikitext_datasets\n")
            f.write("   datasets = create_wikitext_datasets()\n")
            f.write("   full_dataset = datasets['wikitext103_full']\n\n")
            
            f.write("2. Load quality-stratified dataset:\n")
            f.write("   quality_dataset = datasets['wikitext103_quality_stratified']\n\n")
            
            f.write("3. Load length-stratified dataset:\n")
            f.write("   length_dataset = datasets['wikitext103_length_stratified']\n\n")
            
            f.write("4. Load topic-stratified dataset:\n")
            f.write("   topic_dataset = datasets['wikitext103_domain_stratified']\n\n")
            
            f.write("5. Use with LLM experiments:\n")
            f.write("   python llm_experiments.py --dataset wikitext103_quality_stratified\n\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Test loading cached datasets
        logger.info("\n" + "=" * 60)
        logger.info("Testing Cached Dataset Loading")
        logger.info("=" * 60)
        
        # Try loading from cache
        cached_datasets = create_wikitext_datasets(max_samples=max_samples, cache=True)
        logger.info(f"✓ Successfully loaded {len(cached_datasets)} datasets from cache")
        
        logger.info("\n" + "=" * 60)
        logger.info("Dataset Initialization Complete!")
        logger.info("=" * 60)
        logger.info(f"All datasets saved to: {data_dir}")
        logger.info("\nYou can now use these datasets in your experiments:")
        logger.info("  python llm_experiments.py --dataset wikitext103_quality_stratified")
        logger.info("  python llm_experiments.py --dataset wikitext103_length_stratified")
        logger.info("  python llm_experiments.py --dataset wikitext103_domain_stratified")
        
        return datasets
        
    except Exception as e:
        logger.error(f"Error initializing datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize WikiText-103 datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to use (None for all)")
    parser.add_argument("--data_dir", type=str, 
                       default="/Users/mukher74/research/data",
                       help="Directory to save datasets")
    parser.add_argument("--small", action="store_true",
                       help="Create small test datasets (1000 samples)")
    
    args = parser.parse_args()
    
    # Override max_samples if --small is specified
    if args.small:
        args.max_samples = 1000
        logger.info("Creating small test datasets with 1000 samples")
    
    # Initialize datasets
    datasets = initialize_all_datasets(
        max_samples=args.max_samples,
        data_dir=args.data_dir
    )
    
    if datasets:
        print(f"\n✓ Successfully initialized {len(datasets)} dataset variations")
        print(f"✓ All datasets saved to: {args.data_dir}")
    else:
        print("\n✗ Failed to initialize datasets")
        sys.exit(1)