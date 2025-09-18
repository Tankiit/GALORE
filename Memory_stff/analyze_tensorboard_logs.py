#!/usr/bin/env python3
"""
Command-line TensorBoard log analyzer for MASCS experiments
Usage: python analyze_tensorboard_logs.py [--log_dir ./experiments] [--output_dir ./analysis]
"""

import argparse
from tensorboard_log_analyzer import TensorBoardLogExtractor

def main():
    parser = argparse.ArgumentParser(description="Analyze TensorBoard logs from MASCS experiments")
    parser.add_argument("--log_dir", default="./experiments", help="Directory containing experiment logs")
    parser.add_argument("--output_dir", default="./analysis", help="Output directory for analysis results")
    parser.add_argument("--export_csv", action="store_true", help="Export data to CSV files")
    parser.add_argument("--generate_report", action="store_true", help="Generate HTML report")

    args = parser.parse_args()

    print(f"Analyzing TensorBoard logs from: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)

    # Initialize extractor
    extractor = TensorBoardLogExtractor(args.log_dir)

    # Discover experiments
    experiments = extractor.discover_experiments()
    if not experiments:
        print("No experiments found!")
        return

    # Extract data
    print("Extracting scalar data...")
    extractor.extract_all_scalars()

    # Create summary
    summary_df = extractor.create_summary_dataframe()

    if not summary_df.empty:
        print("\nExperiment Summary:")
        print(summary_df.to_string(index=False))

        print(f"\nKey Statistics:")
        print(f"- Total experiments: {len(summary_df)}")
        print(f"- Average best validation accuracy: {summary_df['best_val_accuracy'].mean():.4f}")
        print(f"- Best performance: {summary_df['best_val_accuracy'].max():.4f}")
        print(f"- Worst performance: {summary_df['best_val_accuracy'].min():.4f}")

        # Budget analysis
        if 'budget' in summary_df.columns:
            budget_analysis = summary_df.groupby('budget')['best_val_accuracy'].agg(['mean', 'std', 'count'])
            print(f"\nBudget Analysis:")
            print(budget_analysis)
    else:
        print("No data could be extracted from the experiments.")
        return

    # Export if requested
    if args.export_csv:
        print(f"\nExporting CSV files to {args.output_dir}...")
        extractor.export_to_csv(args.output_dir)

    if args.generate_report:
        print(f"\nGenerating HTML report in {args.output_dir}...")
        extractor.generate_report(args.output_dir)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()