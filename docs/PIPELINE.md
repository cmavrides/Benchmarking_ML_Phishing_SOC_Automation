# Pipeline

The phishing detection benchmark follows a modular pipeline that can be orchestrated via CLI commands or executed end-to-end using `python -m phishing_benchmark.cli.run_all`.

## High-Level Flow

```
Download → Load → Clean HTML → Normalize Text → Label Map → Merge → Deduplicate → Stratified Split → Featurize → Train → Evaluate → Report
```

Each stage is implemented as a composable function with configuration options exposed in `config.py` and the Typer CLI modules.

## Stage Details

### 1. Download
- Command: `python -m phishing_benchmark.cli.download_data`
- Uses `datasets.load_dataset` to fetch artifacts from Hugging Face.
- Persists raw CSV/Parquet files in `data/raw/` for reproducibility.

### 2. Load & Map Schema
- `phishing_benchmark.data.loaders.load_zefang_liu`
- `phishing_benchmark.data.loaders.load_cyradar`
- Converts source-specific fields into the unified schema defined in [docs/DATASETS.md](DATASETS.md).

### 3. Clean HTML & Normalize Text
- `phishing_benchmark.data.cleaning.strip_html`
- `phishing_benchmark.data.cleaning.normalize_text`
- Removes `<script>`/`<style>` blocks, decodes HTML entities, standardizes whitespace, and optionally lowercases text.

### 4. Label Normalization
- Maps source labels to binary integers (`1` phishing, `0` legitimate).
- Additional hooks allow re-labeling or exclusion of ambiguous samples.

### 5. Merge & Deduplicate
- `phishing_benchmark.data.merge.merge_and_dedup`
- Concatenates datasets, hashes `subject + body_clean`, and removes duplicates while preferring email-like entries when conflicts occur.

### 6. Stratified Splitting
- `phishing_benchmark.data.splits.make_stratified_splits`
- Performs stratified train/validation/test splits on label while keeping dataset proportions balanced.
- Secondary stratification on `source_dataset` reduces domain leakage.

### 7. Feature Engineering
- `phishing_benchmark.features.vectorizers.get_tfidf`
- Supports TF-IDF, hashing, and tokenization pipelines for various models.
- Text normalization utilities handle placeholder substitution for URLs, numbers, and user references.

### 8. Model Training
- Classical ML (`phishing_benchmark.models.classical`)
- Deep sequence models (`phishing_benchmark.models.deep`)
- Transformers (`phishing_benchmark.models.transformers`)
- LLM inference wrappers (`phishing_benchmark.models.llm_inference`)
- Training commands log to MLflow and store artifacts under `models/` or the specified checkpoint directory.

### 9. Evaluation & Reporting
- `phishing_benchmark.eval.metrics.compute_binary_metrics`
- `phishing_benchmark.eval.reporting`
- Generates metrics JSON, confusion matrices, precision-recall curves, and leaderboard CSV.

## CLI Overview

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `python -m phishing_benchmark.cli.download_data` | Fetch raw datasets | `--zefang-liu`, `--cyradar`, `--sample`, `--force` |
| `python -m phishing_benchmark.cli.preprocess` | Clean, normalize, split | `--lower/--no-lower`, `--strip-html/--keep-html`, `--min-length`, `--max-length`, `--val-size`, `--test-size`, `--seed` |
| `python -m phishing_benchmark.cli.train` | Train selected model | `--model`, `--max-features`, `--ngram-range`, `--epochs`, `--batch-size`, `--lr`, `--checkpoint-dir` |
| `python -m phishing_benchmark.cli.evaluate` | Evaluate trained model | `--model`, `--checkpoint-dir`, `--split` |
| `python -m phishing_benchmark.cli.run_all` | Execute full pipeline | `--models`, `--seed`, `--strip-html`, etc. |

## Configuration

- Default settings are defined in `phishing_benchmark.config.DefaultConfig`.
- Configuration can be overridden via CLI flags or environment variables.
- MLflow tracking URI defaults to a local `mlruns/` directory; customize via `MLFLOW_TRACKING_URI`.

## Extending the Pipeline

1. Implement new data loaders in `src/phishing_benchmark/data/` and register them in the CLI.
2. Add additional feature extractors or models under `src/phishing_benchmark/features/` and `src/phishing_benchmark/models/`.
3. Update tests to cover new functionality and ensure CI compatibility.
4. Document new steps in this file and in the relevant notebooks.
