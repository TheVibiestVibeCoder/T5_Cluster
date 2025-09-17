# Enhanced Theme Clustering with AI Narratives

A production-oriented Python tool that clusters news or article corpora into coherent themes, extracts representative keywords, and generates human-readable narrative summaries. It uses Sentence-BERT embeddings, dimensionality reduction, HDBSCAN clustering, TF-IDF keywording, and a map-reduce summarization pipeline with T5 (fallback to DistilBART). The script also exports CSVs and PNG visualizations for downstream analysis.

---

Execute command beispiel: 

python cluster_narrative_engine.py --csv "FILE_PATH" --min-cluster-size XYZ --topic "XYZ" --skip_summaries

## Key features

- **Robust semantic embeddings**  
  Uses `sentence-transformers/all-MiniLM-L6-v2` with **normalized embeddings** for stable cosine behavior.

- **Stabilized clustering**  
  Optional **UMAP** or **PCA** to 50 dims, then **HDBSCAN** on reduced space for non-parametric discovery of themes. Noise is supported.

- **Better text summarization**  
  T5-base summarization with task prefix `summarize:` and **map-reduce** chunking for long text. Falls back to DistilBART if T5 is not available.

- **Representative keyword extraction**  
  Optimized TF-IDF with enhanced stop word set for news/business. Adds proper-noun harvesting from headlines for named entities.

- **Concise theme naming**  
  Extracts short cluster names from AI summaries by selecting key entities or meaningful terms.

- **Coverage analytics**  
  Per-cluster context on coverage intensity, author diversity, article depth, and time span.

- **Determinism and resilience**  
  Fixed random seeds, clear error handling, sane fallbacks. GPU used if available.

- **Ready exports and visuals**  
  CSVs for articles, summaries, and narratives, plus two PNG plots: an overview scatter and a detailed view with titles for small clusters.

---

## Input expectations

The script expects a CSV with at least these columns:

- `Date`  
  Parsable to datetime.  
- `Headline`  
  Short title per article.  
- `Article`  
  The full text or body. May be empty.  
- `Journalists`  
  Either a Python-list-like string such as `["Alice", "Bob"]` or a single name.

During loading it creates:
- `Full_Text` as `Headline + " " + Article`
- `Text_Length` and `Word_Count`
- `Journalists_List` as a Python list via `ast.literal_eval` when possible

Very short items (`Text_Length <= 50`) are dropped by default.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip

pip install \
  pandas numpy matplotlib seaborn \
  scikit-learn hdbscan umap-learn \
  torch transformers sentence-transformers
````

GPU is optional. If CUDA is available, the summarizer runs on GPU.

---

## Quick start

```bash
python enhanced_theme_clustering.py \
  --csv path/to/your_dataset.csv \
  --sample-size 500 \
  --min-cluster-size 5 \
  --topic Russia election
```

Arguments:

* `--csv` Path to the input CSV. Required.
* `--sample-size` Optional cap on analyzed rows for faster runs.
* `--min-cluster-size` HDBSCAN minimum cluster size. Default 5.
* `--topic` One or more keywords. Applies a case-insensitive OR filter to `Full_Text`.

---

## What the pipeline does

1. **Load and clean**

   * Reads CSV, parses `Date`
   * Normalizes `Journalists` into `Journalists_List`
   * Builds `Full_Text`, drops too short items, optional keyword filter

2. **Embed**

   * Encodes `Full_Text` with Sentence-BERT
   * **normalize\_embeddings=True** for better cosine geometry

3. **Reduce**

   * UMAP to 50 dims if available, else PCA to 50

4. **Cluster**

   * HDBSCAN on reduced vectors
   * Parameters: `metric='euclidean'`, `min_samples=5`, `cluster_selection_epsilon=0.1`, `cluster_selection_method='eom'`
   * Noise labeled as `-1`

5. **Keywords**

   * TF-IDF with extended stop words for news/business domains
   * N-grams 1 to 4, improved token pattern for hyphenated and slash terms
   * Augments with simple proper-noun extraction from headlines
   * Returns top ten de-duplicated terms per cluster

6. **Cluster summaries**

   * Sizes, date ranges, average word counts
   * Top headlines by length
   * Most frequent authors

7. **Narratives**

   * Prepares representative text using a **centroid-based selection** of the most similar articles
   * Map-reduce summarization with T5 (beam search, no repeat n-grams)
   * Adds **Coverage Analysis**: intensity, author diversity, article depth, significance
   * Derives a concise **cluster name** from the AI summary

8. **Visualize**

   * 2D scatter via UMAP or PCA
   * Annotates cluster centers with names
   * A second figure optionally annotates headlines for smaller clusters

9. **Export**

   * `*_articles.csv` with per-article clusters
   * `*_summaries.csv` with per-cluster stats
   * `*_narratives.csv` with names, AI summaries, analytical context
   * `cluster_visualization.png` and `detailed_cluster_view.png`

---

## Output files

* `enhanced_theme_clustering_articles.csv`
  Columns include original fields plus `Cluster`.

* `enhanced_theme_clustering_summaries.csv`
  Per cluster: `cluster_id`, `size`, `date_range`, `avg_word_count`, `keywords`, `top_authors`.

* `enhanced_theme_clustering_narratives.csv`
  Per cluster: `cluster_id`, `cluster_name`, `size`, `ai_summary`, `analytical_context`, `full_narrative`, `keywords`.

* `cluster_visualization.png`
  Theme overview scatter with labeled cluster centers.

* `detailed_cluster_view.png`
  2D scatter with per-article annotations for small clusters.

---

## Practical tips

* **Data quality**
  Better results if `Article` is non-empty and `Headline` is informative.
  Ensure `Date` is parseable and in a consistent timezone.

* **Performance**
  Use `--sample-size` during iteration.
  If UMAP is slow, rely on PCA.
  GPU accelerates summarization.

* **Clustering sensitivity**
  Tune `--min-cluster-size`. Larger values yield fewer, broader clusters. Smaller values split themes.

* **Keyword relevance**
  Adjust stop words in `extract_cluster_keywords` to your domain.
  The proper-noun regex is simple by design. For high accuracy, integrate spaCy NER.

* **Narratives**
  The map-reduce summarizer keeps outputs concise while retaining coverage of long contexts.
  If T5 loads slowly or VRAM is low, the code falls back to DistilBART.

---

## Troubleshooting

* **UMAP missing**
  The script logs a notice and falls back to PCA automatically.

* **T5 load error**
  The pipeline switches to DistilBART. Check installed `transformers`, `torch`, and GPU drivers.

* **Weird `Journalists` parsing**
  Provide a proper Python-list-like string or a single name. The loader uses `ast.literal_eval` when possible.

* **Empty or tiny clusters**
  Increase `--min-cluster-size`, or provide more data.
  Very short `Full_Text` entries are dropped.

* **Overcrowded labels**
  The detailed view only annotates headlines for clusters with at most 15 items.

---

## Limitations

* Proper-noun extraction uses a regex, which is fast but naive.
* Summaries depend on the quality of the underlying model and the representativeness of selected articles.
* HDBSCAN parameters can affect stability on very small corpora.
* The tool is English-centric for stop words. Extend for multilingual use.

---

## Extending the tool

* Swap embeddings to a multilingual model such as `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* Add spaCy NER for better entity-level keywords.
* Feed `Narrative Capture` pipelines or dashboards from the output CSVs.
* Add a per-cluster timeline and volume burst detection.

---

## Security and ethics

Use responsibly. If the corpus contains sensitive or personal data, ensure proper anonymization and compliance with applicable laws and policies. Summaries and clusters can surface unintended inferences. Review outputs before distribution.

---

## License

Add your preferred license here.

```

If you want this saved into an actual `README.md` file in the workspace, say so and tell me the desired filename or path.
```
