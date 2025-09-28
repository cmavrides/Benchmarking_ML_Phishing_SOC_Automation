# Phishing Detection Benchmark

Phishing emails and social engineering messages continue to evolve, making it difficult for defenders to maintain accurate detection systems. **phishing-detection-benchmark** consolidates leading open-source phishing corpora, provides a modular preprocessing pipeline, and benchmarks a spectrum of models ranging from classical machine learning to large language models (LLMs). The repository is designed for both research exploration and production-grade experimentation.

## Key Features

- **Dataset fusion** – combines the [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) and [huynq3Cyradar/Phishing_Detection_Dataset](https://huggingface.co/datasets/huynq3Cyradar/Phishing_Detection_Dataset) corpora into a unified schema.
- **Reproducible preprocessing** – configurable cleaning, normalization, deduplication, and stratified splitting pipelines with CLI tooling.
- **Model zoo** – supports TF-IDF + classical ML, deep sequence encoders, Transformers, and LLM zero/few-shot inference through Typer-based CLIs.
- **Experiment tracking** – MLflow integration for parameters, metrics, and artifacts.
- **Comprehensive documentation** – detailed dataset notes, pipeline explanations, and result reporting templates.

## Repository Layout

```
phishing-detection-benchmark/
├── data/                 # raw, interim, processed datasets (not versioned)
├── docs/                 # dataset, pipeline, and results documentation
├── notebooks/            # exploratory and benchmarking notebooks
├── src/phishing_benchmark
│   ├── data/             # loaders, cleaners, mergers, and splitters
│   ├── features/         # text normalization and vectorization utilities
│   ├── models/           # classical ML, deep learning, transformers, LLMs
│   ├── eval/             # metrics and reporting helpers
│   └── cli/              # Typer-powered command line interface
├── tests/                # pytest-based unit tests
├── scripts/              # (optional) additional utilities
└── .github/workflows     # CI pipeline
```

## Getting Started

### 1. Clone & Environment Setup

```bash
git clone <repo-url> phishing-detection-benchmark
cd phishing-detection-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

Alternatively, use `make setup` to bootstrap dependencies (virtual environment management remains manual).

### 2. Configure Environment Variables

Duplicate `.env.example` to `.env` and populate any required secrets (e.g., `OPENAI_API_KEY` for LLM inference). Hugging Face caches default to the local `.cache` directory.

### 3. Download Datasets

```bash
python -m phishing_benchmark.cli.download_data --zefang-liu --cyradar
```

This command retrieves the datasets from Hugging Face using the `datasets` library and stores the raw files in `data/raw/`. Use `--sample` to limit the number of examples for quick experiments and `--force` to overwrite existing artifacts.

### 4. Preprocess & Merge

```bash
python -m phishing_benchmark.cli.preprocess --strip-html --lower --seed 42
```

The preprocessing CLI loads each dataset, applies configurable cleaning (HTML stripping, whitespace normalization, length filtering), deduplicates records, and produces stratified train/validation/test splits. Outputs are written to `data/interim/` and `data/processed/` in parquet format. Consult [docs/DATASETS.md](docs/DATASETS.md) and [docs/PIPELINE.md](docs/PIPELINE.md) for schema and pipeline details.

### 5. Train Baselines

```bash
python -m phishing_benchmark.cli.train --model lr
python -m phishing_benchmark.cli.train --model svm
python -m phishing_benchmark.cli.train --model distilbert --epochs 2 --batch-size 16
```

Model artifacts, metrics, and configuration metadata are tracked with MLflow (default local `mlruns/` directory). Use the CLI flags to control feature extraction, network architecture, and optimization parameters.

### 6. Evaluate & Build Leaderboard

```bash
python -m phishing_benchmark.cli.evaluate --model distilbert
python -m phishing_benchmark.cli.run_all
```

Evaluation exports JSON metric reports, diagnostic plots, and updates a `reports/leaderboard.csv`. Summaries should be reflected in [docs/RESULTS.md](docs/RESULTS.md).

## Makefile Shortcuts

```
make setup        # install Python dependencies and optional tools
make download     # download datasets from Hugging Face
make preprocess   # run preprocessing pipeline
make train        # train baseline models
make eval         # evaluate models and refresh leaderboard
make all          # execute download->preprocess->train->eval
make format       # run code formatters
make lint         # execute linters and static type checks
make test         # run pytest suite
```

## Notebooks

- `00_data_exploration.ipynb` – dataset profiling (class balance, length distributions, HTML prevalence).
- `10_feature_baselines.ipynb` – TF-IDF feature exploration with classical models.
- `20_transformers_eval.ipynb` – transformer fine-tuning experiments, learning curves, and error analysis.

## Reproducing Results

1. Ensure datasets are downloaded and preprocessed.
2. Train selected models via CLI or notebooks.
3. Run `python -m phishing_benchmark.cli.evaluate --model <name>` to compute metrics.
4. Update `docs/RESULTS.md` with new metrics and observations.
5. Share MLflow runs or exported artifacts for peer review.

## Contributing

1. Fork the repository and create a feature branch.
2. Run `make format lint test` before submitting a pull request.
3. Document new experiments in `docs/RESULTS.md` and provide reproducibility notes.

## Citation

If you use this repository in academic work, please cite the Hugging Face datasets and reference this project:

```
@misc{phishing_detection_benchmark,
  title        = {phishing-detection-benchmark},
  author       = {Contributors},
  year         = {2024},
  url          = {https://github.com/<org>/phishing-detection-benchmark}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
