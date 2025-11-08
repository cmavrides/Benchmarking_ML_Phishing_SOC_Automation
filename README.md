# Phishing Detection Kaggle Notebooks

This repository contains Kaggle-ready notebooks for benchmarking phishing detection models and demonstrating a lightweight SOC automation workflow.

## Project Structure

- `TaskA_Benchmark.ipynb` – preprocesses two phishing datasets, trains multiple models (classical ML and transformers), evaluates them, and exports the best-performing artifact bundle to `/kaggle/working/artifacts/`.
- `TaskB_SOC_Automation.ipynb` – loads the saved artifact bundle and provides an interactive SOC-style triage experience with IOC extraction and explainability helpers.

## Using the Notebooks on Kaggle

1. Create a **private Kaggle Dataset** that contains both `zefang_liu.csv` and `cyradar.csv`.
2. Open a new Kaggle notebook, click **Add data → Your Datasets**, and attach your dataset.
3. In the notebooks, update the dataset slug in the configuration cell so that the CSVs are found at paths like `/kaggle/input/<dataset-slug>/zefang_liu.csv`.

If you enable internet access on Kaggle, you may set `HF_DOWNLOAD=True` in the configuration cell to fetch the datasets directly from Hugging Face as a fallback.

## Local Quickstart

When you run the local benchmarking pipeline, the raw datasets are fetched automatically from Hugging Face if they are not already present under `data/`:

```bash
python src/task_a_benchmark/run_all.py
```

To pre-download the CSVs (or refresh them) without training models, run the downloader module directly:

```bash
python -m src.common.dataset_downloader
```

Set the optional environment variable `CYRADAR_MAX_ROWS` to limit how many rows are retained from the large CyRadar export during development runs.

All processed data, models, and reports are written to `/kaggle/working/` so they persist across notebook cells and can be downloaded after execution.

### Minimal Run Guide

1. Launch **TaskA_Benchmark.ipynb** on Kaggle, ensure the dataset paths are correct, and run all cells. The notebook saves trained models and processed data to `/kaggle/working/`.
2. Launch **TaskB_SOC_Automation.ipynb** in the same Kaggle session (or attach the exported artifacts as a dataset) and run all cells to interact with the SOC automation demo.

