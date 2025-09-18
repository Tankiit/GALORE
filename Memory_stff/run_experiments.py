#!/usr/bin/env python3
"""
Experiment Runner for MASCS with CIFAR10/100 and multiple coreset percentages
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def check_experiment_completed(experiment_name, log_dir):
    """Check if experiment is already completed"""
    results_file = f"results_{experiment_name}.json"
    intermediate_dir = f"intermediate_{experiment_name}"

    # Check if results file exists and has final results
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                if 'final_accuracy' in data:
                    return True
        except:
            pass

    # Check if intermediate files suggest completion
    if os.path.exists(intermediate_dir):
        checkpoint_file = os.path.join(intermediate_dir, 'experiment_checkpoint.json')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    if data.get('status') == 'completed':
                        return True
            except:
                pass

    return False

def save_experiment_checkpoint(experiment_name, log_dir, status, epoch=0, total_epochs=30):
    """Save experiment checkpoint for resume capability"""
    intermediate_dir = f"intermediate_{experiment_name}"
    os.makedirs(intermediate_dir, exist_ok=True)

    checkpoint = {
        'experiment_name': experiment_name,
        'status': status,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_file = os.path.join(intermediate_dir, 'experiment_checkpoint.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def run_experiment(dataset, architecture, data_percentage, budget, epochs=30, force_restart=False, device='cuda'):
    """Run a single experiment configuration"""
    
    # Calculate budget based on dataset size
    # CIFAR10: 50,000 training samples
    # CIFAR100: 50,000 training samples
    base_budget = 50000  # Approximate size of CIFAR training sets
    actual_budget = int(base_budget * (budget / 100))
    
    experiment_name = f"{dataset}_{architecture}_{data_percentage}%"
    log_dir = "./experiment_logs"
    data_dir = "./data"

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Check if experiment is already completed
    if not force_restart and check_experiment_completed(experiment_name, log_dir):
        print(f"⏭️  Skipping {experiment_name} - already completed")
        return True
    
    cmd = [
        "python", "mascs_mdp.py"
        "--datasets", dataset,
        "--architectures", architecture,
        "--data_percentages", str(data_percentage),
        "--budget", str(actual_budget),
        "--epochs", str(epochs),
        "--batch_size", "64",
        "--lr", "1e-3",
        "--data_dir", data_dir,
        "--log_dir", log_dir,
        "--device", device,
        "--save_results", f"results_{experiment_name}.json",
        "--save_intermediate", f"intermediate_{experiment_name}",
        "--save_frequency", "10",
        "--optimize_weights",
        "--gp_calls", "10",
        "--weight_optimization_epoch", "10"
    ]
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Dataset: {dataset} | Architecture: {architecture} | Data: {data_percentage}% | Budget: {actual_budget}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    # Save start checkpoint
    save_experiment_checkpoint(experiment_name, log_dir, 'running', 0, epochs)

    # Run the experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

        if result.returncode == 0:
            print(f"✅ Experiment {experiment_name} completed successfully!")
            save_experiment_checkpoint(experiment_name, log_dir, 'completed', epochs, epochs)
            print(f"STDOUT (last 300 chars): {result.stdout[-300:] if len(result.stdout) > 300 else result.stdout}")
            return True
        else:
            print(f"❌ Experiment {experiment_name} failed!")
            save_experiment_checkpoint(experiment_name, log_dir, 'failed', 0, epochs)
            print(f"STDERR (last 300 chars): {result.stderr[-300:] if len(result.stderr) > 300 else result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_name} timed out!")
        save_experiment_checkpoint(experiment_name, log_dir, 'timeout', 0, epochs)
        return False
    except Exception as e:
        print(f"💥 Experiment {experiment_name} crashed: {e}")
        save_experiment_checkpoint(experiment_name, log_dir, 'error', 0, epochs)
        return False

def run_experiment_wrapper(args):
    """Wrapper for parallel execution"""
    dataset, architecture, data_percentage, budget, epochs, force_restart, device = args
    return run_experiment(dataset, architecture, data_percentage, budget, epochs, force_restart, device)

def create_shared_dataset_cache():
    """Pre-load datasets to avoid repeated downloads/processing"""
    print("🔄 Pre-loading datasets for caching...")

    # This will ensure datasets are downloaded and cached
    cache_cmd = [
        "python", "-c",
        "import torchvision; import torchvision.transforms as T; "
        "T_norm = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]); "
        "torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=T_norm); "
        "torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=T_norm); "
        "print('✅ Dataset cache ready')"
    ]

    try:
        result = subprocess.run(cache_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Dataset caching completed")
        else:
            print(f"⚠️  Dataset caching had issues: {result.stderr[:200]}")
    except Exception as e:
        print(f"⚠️  Dataset caching failed: {e}")

def main():
    """Run all experiments"""
    
    print("🚀 Starting MASCS Experiments (Optimized)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add command line arguments for optimization
    import argparse
    parser = argparse.ArgumentParser(description='Run optimized MASCS experiments')
    parser.add_argument('--parallel', type=int, default=min(2, cpu_count()//2),
                       help='Number of parallel experiments (default: 2)')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart all experiments (ignore completed ones)')
    parser.add_argument('--skip_cache', action='store_true',
                       help='Skip dataset pre-caching')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (cuda, cpu, mps)')

    args = parser.parse_args() if len(sys.argv) > 1 else argparse.Namespace(parallel=2, force_restart=False, skip_cache=False, device='cuda')

    print(f"⚡ Parallel workers: {args.parallel}")
    print(f"🔄 Force restart: {args.force_restart}")
    print(f"🖥️  Device: {args.device}")

    # Pre-cache datasets if not skipping
    if not args.skip_cache:
        create_shared_dataset_cache()
    
    # Experiment configurations
    datasets = ["CIFAR10", "CIFAR100"]
    architectures = ["resnet18"]  # Start with one architecture for testing
    data_percentages = [10, 20, 30, 50, 70, 100]

    # Create experiment parameter list
    experiment_params = []
    for dataset in datasets:
        for architecture in architectures:
            for data_percentage in data_percentages:
                experiment_params.append((
                    dataset, architecture, data_percentage, data_percentage, 30, args.force_restart, args.device
                ))
    
    total_experiments = len(datasets) * len(architectures) * len(data_percentages)
    completed_experiments = 0
    successful_experiments = 0
    
    print(f"Total experiments planned: {total_experiments}")
    
    # Run experiments in parallel or sequential based on args.parallel
    if args.parallel > 1:
        print(f"🔄 Running experiments with {args.parallel} parallel workers...")

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all experiments
            future_to_params = {executor.submit(run_experiment_wrapper, params): params
                               for params in experiment_params}

            try:
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    dataset, architecture, data_percentage, budget, epochs, force_restart, device = params
                    experiment_name = f"{dataset}_{architecture}_{data_percentage}%"

                    try:
                        success = future.result()
                        completed_experiments += 1
                        if success:
                            successful_experiments += 1

                        print(f"\n📊 Progress: {completed_experiments}/{total_experiments} completed "
                              f"({successful_experiments} successful)")
                        print(f"   Latest: {experiment_name} - {'✅ Success' if success else '❌ Failed'}")

                    except Exception as exc:
                        print(f"💥 {experiment_name} generated exception: {exc}")
                        completed_experiments += 1

            except KeyboardInterrupt:
                print("\n🛑 Experiments interrupted by user")
                print(f"Final results: {successful_experiments}/{completed_experiments} successful")
                return

    else:
        print("🔄 Running experiments sequentially...")

        for params in experiment_params:
            dataset, architecture, data_percentage, budget, epochs, force_restart, device = params
            experiment_name = f"{dataset}_{architecture}_{data_percentage}%"

            try:
                success = run_experiment_wrapper(params)
                completed_experiments += 1
                if success:
                    successful_experiments += 1

                print(f"\n📊 Progress: {completed_experiments}/{total_experiments} completed "
                      f"({successful_experiments} successful)")

                # Small delay between sequential experiments
                if not success:  # Only delay on failures to avoid resource conflicts
                    time.sleep(2)

            except KeyboardInterrupt:
                print("\n🛑 Experiments interrupted by user")
                print(f"Final results: {successful_experiments}/{completed_experiments} successful")
                return
            except Exception as e:
                print(f"💥 Unexpected error with {experiment_name}: {e}")
                completed_experiments += 1
    
    print(f"\n🎉 All experiments completed!")
    print(f"✅ Successful: {successful_experiments}/{completed_experiments}")
    print(f"⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show summary of experiment statuses
    if completed_experiments > 0:
        success_rate = (successful_experiments / completed_experiments) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")

        # Show failed experiments for debugging
        failed_count = completed_experiments - successful_experiments
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} experiments failed - check individual logs for details")

if __name__ == "__main__":
    main()