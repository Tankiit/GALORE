#!/usr/bin/env python3
"""
Quick script to view TensorBoard statistics from the command line.
Useful for checking results without launching the full TensorBoard UI.

Usage:
    python view_tensorboard_stats.py
    python view_tensorboard_stats.py --logdir ./runs/training
"""

import argparse
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def format_number(value):
    """Format numbers nicely"""
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        elif abs(value) < 100:
            return f"{value:.2f}"
        else:
            return f"{value:,.0f}"
    return str(value)


def view_tensorboard_logs(logdir: str, show_all_steps: bool = False):
    """
    View TensorBoard logs from command line

    Args:
        logdir: Path to TensorBoard log directory
        show_all_steps: If True, show all steps instead of just latest
    """
    logdir = Path(logdir)

    if not logdir.exists():
        print(f"Error: Directory '{logdir}' does not exist")
        return

    # Find all subdirectories with event files
    log_dirs = []

    # Check current directory
    if any(f.name.startswith('events.out.tfevents') for f in logdir.iterdir()):
        log_dirs.append(('root', logdir))

    # Check subdirectories
    for subdir in logdir.iterdir():
        if subdir.is_dir():
            if any(f.name.startswith('events.out.tfevents') for f in subdir.iterdir()):
                log_dirs.append((subdir.name, subdir))

    if not log_dirs:
        print(f"No TensorBoard event files found in '{logdir}'")
        print("\nNote: Make sure you've run training with enable_tensorboard=True")
        return
    print_header("TensorBoard Statistics Viewer")
    print(f"Log directory: {logdir.absolute()}")
    print(f"Found {len(log_dirs)} log directory(ies)")

    # Process each log directory
    for name, dir_path in sorted(log_dirs):
        print_header(f"{name.upper()}")

        try:
            # Load event file
            event_acc = EventAccumulator(str(dir_path))
            event_acc.Reload()

            # Get all tags
            tags = event_acc.Tags()

            # Process scalars
            if tags.get('scalars'):
                print("\nScalar Metrics:")
                print("-" * 80)

                # Group metrics by category
                categories = {}
                for tag in tags['scalars']:
                    if '/' in tag:
                        category, metric = tag.split('/', 1)
                    else:
                        category = 'other'
                        metric = tag

                    if category not in categories:
                        categories[category] = []
                    categories[category].append(tag)

                # Print by category
                for category in sorted(categories.keys()):
                    print(f"\n  [{category.upper()}]")

                    for tag in sorted(categories[category]):
                        data = event_acc.Scalars(tag)

                        if not data:
                            continue

                        # Get metric name
                        metric_name = tag.split('/', 1)[1] if '/' in tag else tag

                        if show_all_steps:
                            print(f"\n    {metric_name}:")
                            for i, event in enumerate(data):
                                if i < 5 or i >= len(data) - 2:  # Show first 5 and last 2
                                    print(f"      Step {event.step:6d}: {format_number(event.value)}")
                                elif i == 5:
                                    print(f"      ... ({len(data) - 7} more steps)")
                        else:
                            # Show summary statistics
                            values = [e.value for e in data]
                            latest = data[-1]

                            print(f"    {metric_name:30s} ", end="")
                            print(f"Latest: {format_number(latest.value):>12s}  ", end="")
                            print(f"(step {latest.step})  ", end="")
                            print(f"[{len(data)} points]")

            # Process histograms
            if tags.get('histograms'):
                print(f"\n\nHistograms: {len(tags['histograms'])} available")
                for tag in sorted(tags['histograms'])[:5]:  # Show first 5
                    print(f"    - {tag}")
                if len(tags['histograms']) > 5:
                    print(f"    ... and {len(tags['histograms']) - 5} more")

            # Process images
            if tags.get('images'):
                print(f"\n\nImages: {len(tags['images'])} available")

        except Exception as e:
            print(f"\nError processing {name}: {e}")

    print("\n" + "=" * 80)
    print("\nTo view in browser:")
    print(f"   tensorboard --logdir={logdir} --port=6006")
    print("   Then open: http://localhost:6006")
    print("\nTo see all steps:")
    print(f"   python {__file__} --logdir {logdir} --all-steps")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='View TensorBoard statistics from command line',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--logdir',
        default='./runs',
        help='Path to TensorBoard log directory (default: ./runs)'
    )
    parser.add_argument(
        '--all-steps',
        action='store_true',
        help='Show all logged steps instead of just latest'
    )

    args = parser.parse_args()

    view_tensorboard_logs(args.logdir, show_all_steps=args.all_steps)


if __name__ == '__main__':
    main()
