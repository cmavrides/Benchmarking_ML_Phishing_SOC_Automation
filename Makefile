PYTHON ?= python
PACKAGE ?= phishing-detection-suite
MODEL ?= lr
CSV ?=

.PHONY: preprocess train eval leaderboard api ui score lint format typecheck test notebooks

preprocess:
$(PYTHON) -m task_a_benchmark.cli preprocess --config configs/task_a.yaml

train:
$(PYTHON) -m task_a_benchmark.cli train --config configs/task_a.yaml --model $(MODEL)

eval:
$(PYTHON) -m task_a_benchmark.cli evaluate --config configs/task_a.yaml --model $(MODEL)

leaderboard:
$(PYTHON) -m task_a_benchmark.cli leaderboard --config configs/task_a.yaml

api:
uvicorn task_b_soc.serving:app --host 0.0.0.0 --port 8000

ui: api

score:
$(PYTHON) -m task_b_soc.cli score-file --csv $(CSV)

lint:
ruff check src tests
black --check src tests

format:
black src tests

typecheck:
mypy src

test:
pytest

notebooks:
jupyter notebook notebooks/
