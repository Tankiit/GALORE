import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic features
np.random.seed(42)
n_points = 500
features = np.random.randn(n_points, 2)
# Assign clusters (simulating classes)
labels = np.random.choice(3, n_points)

# Simulate static coreset (random subset)
budget = mo.ui.slider(10, 200, value=50, label="Coreset Size")

@mo.cell
def plot_static_coreset():
    subset_idx = np.random.choice(range(n_points), size=budget.value, replace=False)
    coreset = features[subset_idx]
    plt.figure(figsize=(5,5))
    plt.scatter(features[:,0], features[:,1], c=labels, alpha=0.2, label="Full Dataset")
    plt.scatter(coreset[:,0], coreset[:,1], edgecolors="black", c="none", s=70, label="Coreset")
    plt.title(f"Static Coreset Selection (|C|={budget.value})")
    plt.legend()
    mo.ui.display(plt.gcf())