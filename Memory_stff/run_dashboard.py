#!/usr/bin/env python3
"""
Launch the TensorBoard Log Analysis Dashboard
Usage: python run_dashboard.py [--port 8501]
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard Log Analysis Dashboard")
    parser.add_argument("--port", default="8501", help="Port to run the Streamlit app on")

    args = parser.parse_args()

    # Get the path to the tensorboard_log_analyzer.py file
    script_path = Path(__file__).parent / "tensorboard_log_analyzer.py"

    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        sys.exit(1)

    print(f"Starting TensorBoard Log Analysis Dashboard on port {args.port}")
    print(f"Access the dashboard at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(script_path),
            "--server.port", args.port
        ])
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()