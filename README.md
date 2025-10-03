# phishing-detection-suite

The **phishing-detection-suite** repository provides an end-to-end workflow for benchmarking machine learning and language models on phishing detection datasets (Task A) and deploying a lightweight SOC automation microservice (Task B) that serves the best model over an API and a tiny UI.

## Repository Layout

```
phishing-detection-suite/
├── configs/              # Configuration files for both tasks
├── data/                 # Drop the raw CSV datasets here (see below)
├── reports/              # Generated evaluation artefacts
├── src/                  # Python packages (common utils, task-specific code)
├── tests/                # Automated tests
├── notebooks/            # Optional exploratory notebooks
├── Dockerfile.task_a     # Container for Task A workflows
├── Dockerfile.task_b     # Container for Task B microservice
├── docker-compose.yml    # Optional convenience launcher
└── artifacts/            # Exported models (Task A → Task B)
```

Both tasks share the `src/common_utils` package for configuration, logging, cleaning, hashing, and metrics.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
pre-commit install
```

Copy the phishing datasets (`zefang_liu.csv` and `cyradar.csv`) into `data/raw/`.

### Configuration

Edit the files under `configs/` to adjust defaults:

- `global.yaml` – shared settings (paths, random seeds, MLflow).
- `task_a.yaml` – preprocessing, feature, and model defaults for benchmarking.
- `task_b.yaml` – service configuration including model artifact location.
- `.env.example` – example environment variables (API keys, ports).

Environment variables override config values when both are defined.

---

## Task A — Benchmarking Workflow

Task A trains and evaluates classical ML, transformer, and (optionally) LLM-based classifiers on the unified phishing dataset. The CLI is implemented with [Typer](https://typer.tiangolo.com/).

### Preprocess

```bash
python -m task_a_benchmark.cli preprocess --strip-html --lower --seed 42
```

The command:

1. Loads the raw CSVs from `data/raw/` (see schema mapping in `loaders.py`).
2. Cleans text (optional HTML stripping, normalisation).
3. Deduplicates using the SHA-1 of `subject + body_clean`.
4. Applies minimum/maximum length filters.
5. Creates stratified train/validation/test splits (default 80/10/10) that retain a balanced share of each source dataset.
6. Persists the unified parquet files to `data/processed/`.

### Train

```bash
python -m task_a_benchmark.cli train --model lr
```

Supported models:

- Classical: `lr`, `svm`, `nb`, `rf`, `xgb`
- Transformers: `distilbert`, `roberta`
- API LLMs: `llm_zero`, `llm_few` (require `OPENAI_API_KEY`)

The CLI builds features (TF-IDF / hashing) or fine-tunes transformers, tracks experiments with MLflow (local file backend), and exports trained artefacts into `artifacts/` (best model also copied to `artifacts/best_model/`).

### Evaluate

```bash
python -m task_a_benchmark.cli evaluate --model lr
```

Evaluation loads the saved model, scores the held-out test split, generates metrics (accuracy, precision/recall/f1, ROC-AUC, PR-AUC), and writes JSON + text reports plus PR/ROC/confusion matrix plots under `reports/`.

### Leaderboard

```bash
python -m task_a_benchmark.cli leaderboard
```

Collects all MLflow runs and materialises `reports/leaderboard.csv` sorted by binary F1 and PR-AUC. The CLI also prints a compact table.

### Notebooks

Two optional exploratory notebooks are provided as stubs:

- `notebooks/01_explore.ipynb`
- `notebooks/02_baselines.ipynb`

Run them with Jupyter after preprocessing to inspect distributions and baseline results.

---

## Task B — SOC Automation Microservice

Task B exposes the best Task A model through a FastAPI application with an accompanying static UI and CLI utilities.

### API Server

```bash
uvicorn task_b_soc.serving:app --reload --port 8000
```

Endpoints:

- `GET /healthz` – simple readiness check.
- `POST /classify` – classify text or HTML payloads.
- `POST /classify-eml` – upload `.eml` email messages (multipart form data).

Each classification response includes the predicted label, phishing probability, optional feature attributions (for classical models), and extracted indicators of compromise (URLs, domains, IPs, emails, hashes).

### Web UI

Navigate to `http://localhost:8000` after starting the API. The single-page interface lets analysts paste or upload content and view structured results. The UI uses fetch to call `/classify` and renders the JSON output.

### CLI Batch Scoring

```bash
python -m task_b_soc.cli score-file --csv path/to/messages.csv --text-col body --subject-col subject --html-col body_is_html --out scored.csv
```

The command loads the configured model, applies the same preprocessing pipeline, and appends `pred_label` and `pred_score` columns.

### Model Artifacts

Task A writes trained artefacts under `artifacts/`. Update `configs/task_b.yaml` to point the microservice at the selected model directory. Classical models expect `model.pkl`, `vectorizer.pkl`, and `preproc_config.json`.

---

## Makefile Targets

```bash
make preprocess     # Task A preprocessing pipeline
make train MODEL=lr # Train a chosen model
make eval MODEL=lr  # Evaluate on the test split
make leaderboard    # Aggregate MLflow runs
make api            # Launch Task B FastAPI server
make ui             # Serve the UI (FastAPI static)
make score CSV=...  # Batch score a CSV via Task B CLI
make lint           # Run ruff + black --check
make typecheck      # Run mypy
make test           # Run pytest
```

---

## Continuous Integration

GitHub Actions under `.github/workflows/ci.yaml` run linting, type checking, and tests on Python 3.10 and 3.11. Transformer-heavy tests are marked `slow` and excluded by default.

---

## Licensing

This project is released under the MIT License. See [LICENSE](LICENSE).

## Example API Request

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Invoice update","body":"<p>Click here to verify</p>","as_html":true}'
```

The response resembles:

```json
{
  "label": 1,
  "score": 0.94,
  "explanations": {
    "top_positive": [["click", 1.45], ["verify", 1.12]],
    "top_negative": [["newsletter", -0.73], ["unsubscribe", -0.51]]
  },
  "iocs": {
    "urls": ["http://example.com"],
    "domains": ["example.com"],
    "emails": [],
    "ips": [],
    "hashes": []
  }
}
```

---

## Troubleshooting

- Ensure the raw datasets exist under `data/raw/` before preprocessing.
- Transformer fine-tuning runs on CPU but will be slow; use classical models for quick iterations.
- Set `OPENAI_API_KEY` in a `.env` file (or environment) to enable LLM-based classifiers.
- When running inside Docker, mount the `data/` and `artifacts/` directories as volumes for persistence.

Enjoy building a robust phishing detection workflow!
