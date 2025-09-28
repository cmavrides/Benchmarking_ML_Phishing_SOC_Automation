.PHONY: help setup download preprocess train eval all format lint test

PROJECT=phishing_benchmark
PYTHON?=python
PIP?=pip

help:
@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## install dependencies for development
$(PIP) install -U pip
$(PIP) install -e .
$(PIP) install -r requirements.txt

_download_cmd = $(PYTHON) -m $(PROJECT).cli.download_data
_preprocess_cmd = $(PYTHON) -m $(PROJECT).cli.preprocess
_train_cmd = $(PYTHON) -m $(PROJECT).cli.train
_evaluate_cmd = $(PYTHON) -m $(PROJECT).cli.evaluate
_run_all_cmd = $(PYTHON) -m $(PROJECT).cli.run_all

download: ## download datasets from Hugging Face
$(_download_cmd)

preprocess: ## preprocess datasets and create splits
$(_preprocess_cmd)

train: ## train baseline models defined in configuration
$(_train_cmd)

eval: ## evaluate models and update leaderboard
$(_evaluate_cmd)

all: ## run the full pipeline (download -> preprocess -> train -> eval)
$(_run_all_cmd)

format: ## format code using black and ruff
$(PYTHON) -m black src tests
$(PYTHON) -m ruff check src tests --fix

lint: ## run static analysis (ruff, mypy)
$(PYTHON) -m ruff check src tests
$(PYTHON) -m mypy src

test: ## run pytest test suite
$(PYTHON) -m pytest
