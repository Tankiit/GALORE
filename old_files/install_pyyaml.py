import subprocess
import sys

# Try to install PyYAML using different methods
methods = [
    ["pip", "install", "--user", "--no-cache-dir", "pyyaml==6.0.3"],
    ["python", "-m", "pip", "install", "--user", "--no-cache-dir", "pyyaml==6.0.3"],
]

for method in methods:
    print(f"Trying: {' '.join(method)}")
    try:
        result = subprocess.run(method, capture_output=True, text=True)
        if result.returncode == 0:
            print("Success!")
            print(result.stdout)
            sys.exit(0)
        else:
            print("Failed!")
            print(result.stderr)
    except Exception as e:
        print(f"Exception: {e}")
    print("-" * 50)

print("All methods failed. You may need to fix permissions or use a different approach.")
