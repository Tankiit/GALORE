#!/usr/bin/env python3
"""
Test script for TensorBoard Log Analyzer
"""

from tensorboard_log_analyzer import TensorBoardLogExtractor
import pandas as pd

def test_tensorboard_extractor():
    """Test the TensorBoard log extractor with existing experiment data"""
    print("Testing TensorBoard Log Extractor...")

    # Initialize extractor with experiments directory
    extractor = TensorBoardLogExtractor("./experiments")

    # Discover experiments
    print("1. Discovering experiments...")
    experiments = extractor.discover_experiments()
    print(f"   Found {len(experiments)} experiments:")
    for name, info in experiments.items():
        print(f"   - {name}")
        print(f"     Config: {info['config']}")

    if not experiments:
        print("   No experiments found! Please check that ./experiments directory exists with TensorBoard logs.")
        return

    # Extract scalar data
    print("\n2. Extracting scalar data...")
    summary_data = extractor.extract_all_scalars()

    # Print available metrics for first experiment
    if summary_data:
        first_exp = next(iter(summary_data.keys()))
        print(f"\n3. Available metrics for '{first_exp}':")
        for metric, data in summary_data[first_exp].items():
            if not metric.startswith('_'):
                if isinstance(data, dict) and 'values' in data:
                    print(f"   - {metric}: {len(data['values'])} values")

    # Create summary DataFrame
    print("\n4. Creating summary DataFrame...")
    summary_df = extractor.create_summary_dataframe()

    if not summary_df.empty:
        print("\nSummary DataFrame:")
        print(summary_df.to_string(index=False))

        print(f"\nStatistics:")
        print(f"- Total experiments: {len(summary_df)}")
        print(f"- Average best validation accuracy: {summary_df['best_val_accuracy'].mean():.4f}")
        print(f"- Best performance: {summary_df['best_val_accuracy'].max():.4f}")
        print(f"- Worst performance: {summary_df['best_val_accuracy'].min():.4f}")
    else:
        print("No data extracted - summary DataFrame is empty")

    # Test export functionality
    print("\n5. Testing export functionality...")
    try:
        extractor.export_to_csv("./analysis")
        print("   CSV export successful!")

        extractor.generate_report("./analysis")
        print("   HTML report generation successful!")
    except Exception as e:
        print(f"   Export failed: {e}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_tensorboard_extractor()