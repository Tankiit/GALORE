#!/usr/bin/env python3
"""
Quick Analysis Script - Analyze existing logs from your current runs
"""

import os
import glob

def find_log_directories(base_path="."):
    """Find all TensorBoard log directories in the current setup"""
    potential_paths = [
        "./logs",
        "./logs/*",
        "../logs", 
        "./tensorboard_logs",
        "./tb_logs"
    ]
    
    found_logs = []
    
    for path_pattern in potential_paths:
        expanded_paths = glob.glob(path_pattern)
        for path in expanded_paths:
            if os.path.isdir(path):
                # Check if it contains event files or subdirectories with event files
                for root, dirs, files in os.walk(path):
                    if any(f.startswith('events.out.tfevents') for f in files):
                        found_logs.append(root)
                        break
                    # Also check subdirectories
                    for d in dirs:
                        subdir_path = os.path.join(root, d)
                        subdir_files = os.listdir(subdir_path) if os.path.isdir(subdir_path) else []
                        if any(f.startswith('events.out.tfevents') for f in subdir_files):
                            found_logs.append(subdir_path)
    
    # Remove duplicates and sort
    found_logs = sorted(list(set(found_logs)))
    return found_logs

def main():
    print("🔍 MASCS Quick Log Analysis")
    print("=" * 40)
    
    # Find log directories
    log_dirs = find_log_directories()
    
    if not log_dirs:
        print("❌ No TensorBoard log directories found in common locations.")
        print("💡 Make sure you have run some MASCS experiments first!")
        print("\nCommon log locations checked:")
        print("  - ./logs")
        print("  - ../logs") 
        print("  - ./tensorboard_logs")
        print("  - ./tb_logs")
        return
    
    print(f"✅ Found {len(log_dirs)} TensorBoard log directories:")
    for i, log_dir in enumerate(log_dirs, 1):
        print(f"  {i:2d}. {log_dir}")
    
    # Determine base directory
    if log_dirs:
        # Find common parent directory
        base_dir = os.path.dirname(log_dirs[0])
        if all(d.startswith(base_dir) for d in log_dirs):
            print(f"\n📁 Detected base log directory: {base_dir}")
        else:
            base_dir = "."
            print(f"\n📁 Using current directory as base: {base_dir}")
        
        print("\n🚀 Ready to analyze! Run one of these commands:")
        print("\n1️⃣  Analyze all logs with batch analyzer:")
        print(f"   python batch_analyze_logs.py --log_dir {base_dir}")
        
        print("\n2️⃣  Analyze specific pattern (e.g., only CIFAR10):")
        print(f"   python batch_analyze_logs.py --log_dir {base_dir} --pattern CIFAR10")
        
        print("\n3️⃣  Analyze with custom output directory:")
        print(f"   python batch_analyze_logs.py --log_dir {base_dir} --output_dir ./my_analysis")
        
        print("\n4️⃣  Quick individual experiment analysis:")
        print(f"   python mascs_mdp.py --analyze_logs {base_dir}")
        
        # Check for intermediate tracking files
        tracking_dirs = glob.glob("./mascs_intermediate*") + glob.glob("./tracking*")
        if tracking_dirs:
            print(f"\n📊 Found intermediate tracking directories:")
            for track_dir in tracking_dirs:
                print(f"   - {track_dir}")
            print(f"\n5️⃣  Include intermediate tracking analysis:")
            print(f"   python batch_analyze_logs.py --log_dir {base_dir} --intermediate_dir {tracking_dirs[0]}")
    
    print(f"\n💡 Tip: You can also run the analysis while experiments are still running!")
    print(f"   This is useful to monitor progress and compare current results.")

if __name__ == "__main__":
    main()