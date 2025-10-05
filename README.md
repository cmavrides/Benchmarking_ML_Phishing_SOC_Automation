# Phishing Detection Suite

The **Phishing Detection Suite** is a lightweight two-part project that benchmarks traditional ML, transformer, and LLM-based approaches for phishing detection and exposes the best-performing model via a simple SOC automation API and web UI.

## Project Overview

- **Task A – Benchmarking:**
  - Load phishing datasets from `data/zefang_liu.csv` and `data/cyradar.csv`.
  - Clean and normalize text, combine datasets, and create train/validation/test splits stored under `data/processed/`.
  - Train baseline ML models (Logistic Regression, Linear SVM, Random Forest, XGBoost) using TF-IDF features and evaluate them alongside a fine-tuned DistilBERT model.
  - Optionally record zero/few-shot results from an LLM when an `OPENAI_API_KEY` is available.
  - Persist models, tokenizers, and metrics to the `artifacts/` and `reports/` directories.

- **Task B – SOC Automation Prototype:**
  - Load the best-performing model from Task A.
  - Provide a FastAPI service that classifies submitted text/emails as phishing or legitimate and extracts basic indicators of compromise (IOCs).
  - Serve a minimal JavaScript UI for interactive testing of the API.

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── zefang_liu.csv           # Place raw datasets here (not tracked)
│   ├── cyradar.csv
│   └── processed/
├── artifacts/                   # Saved models and vectorizers
├── reports/
│   ├── figures/
│   └── results/
└── src/
    ├── common/
    ├── task_a_benchmark/
    └── task_b_soc/
```

> **Note:** The raw CSV datasets are not committed to version control. Download or copy them into the `data/` directory before running the pipelines.

## Getting Started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ensure the datasets `data/zefang_liu.csv` and `data/cyradar.csv` exist before executing the tasks below.

### Task A: Benchmark Models

```bash
python src/task_a_benchmark/run_all.py
```

This command performs the full workflow: loading and merging datasets, preprocessing text, training classical ML models, optionally fine-tuning DistilBERT, and evaluating all models. Outputs include:

- Processed splits under `data/processed/`.
- Serialized models/vectorizers under `artifacts/`.
- Metrics and plots under `reports/results/` and `reports/figures/`.

### Task B: Launch SOC Automation API

```bash
uvicorn src.task_b_soc.api:app --reload
```

Navigate to `http://127.0.0.1:8000/ui/` to open the minimal UI. Submit sample emails or text snippets to receive predictions and IOC extraction results.

### Optional: LLM Zero/Few-Shot Benchmark

If you have an OpenAI API key, export it before running the pipeline to collect zero/few-shot results:

```bash
export OPENAI_API_KEY=your_key_here
python src/task_a_benchmark/train_llm.py
```

The script writes predictions to `reports/results/llm_predictions.csv` and can be incorporated into further analysis.

## Key Modules

- `src/common/`: Reusable utilities for text cleaning, serialization, and metric calculation.
- `src/task_a_benchmark/`: Scripts for dataset loading, preprocessing, model training, and evaluation.
- `src/task_b_soc/`: SOC automation pipeline, enrichment helpers, API definition, and simple UI assets.

## License

This project is provided for educational and prototyping purposes. Review dataset licenses before distributing derived artifacts.
