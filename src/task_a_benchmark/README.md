# Task A – Benchmarking

This package contains preprocessing, feature engineering, modelling, and evaluation workflows for phishing detection benchmarking.

## CLI Usage

All commands share a base configuration file (default `configs/task_a.yaml`). Override parameters via flags or environment variables.

### Preprocess

```bash
python -m task_a_benchmark.cli preprocess --config configs/task_a.yaml --strip-html --lower --seed 42
```

### Train

```bash
python -m task_a_benchmark.cli train --config configs/task_a.yaml --model lr
```

### Evaluate

```bash
python -m task_a_benchmark.cli evaluate --config configs/task_a.yaml --model lr
```

### Leaderboard

```bash
python -m task_a_benchmark.cli leaderboard --config configs/task_a.yaml
```

Consult the module docstrings for details on schema mapping, feature generation, and model definitions.
