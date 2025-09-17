This file gives instructions to AI coding agents (e.g. Codex, GPT-5-Codex, etc.) so that generated code and suggestions align with this project's structure, conventions, and requirements.

---

## Project Overview

- Python-based pipeline that:  
  1. loads and cleans article data from CSV,  
  2. generates sentence embeddings,  
  3. reduces dimensionality (UMAP or PCA),  
  4. clusters with HDBSCAN,  
  5. extracts keywords,  
  6. generates narrative summaries via map-reduce using T5 (fallback DistilBART),  
  7. visualizes cluster layouts;  
  8. exports results (CSV, PNG).  
- Designed for reproducibility: fixed random seeds, fallbacks for missing libraries.  

---

## Dev / Environment Setup

- Use Python ≥ 3.8.  
- Use virtual environment (`venv`) or similar, then install dependencies via `pip`.  
- Key libraries: `sentence-transformers`, `transformers`, `torch`, `hdbscan`, `umap-learn`, `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.  
- If GPU is available, summarization should run on GPU; otherwise CPU fallback works.  

---

## Build / Run / Test Commands

- To install dependencies: `pip install -r requirements.txt` (or via the setup script).  
- To run the full analysis:  
  ```bash
  python enhanced_theme_clustering.py --csv path/to/data.csv --min-cluster-size 5
