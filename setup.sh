#!/usr/bin/env bash
set -e

echo "=== Enhanced Theme Clustering Setup ==="

# Create venv
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing required packages..."
pip install \
  pandas numpy matplotlib seaborn \
  scikit-learn hdbscan umap-learn \
  torch transformers sentence-transformers

echo "Setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Run with: python enhanced_theme_clustering.py --csv your_dataset.csv"
