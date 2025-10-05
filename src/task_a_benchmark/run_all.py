"""Run the full Task A benchmarking pipeline."""
from __future__ import annotations

import argparse

from src.task_a_benchmark import evaluate, preprocess, train_ml, train_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task A benchmarking pipeline")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip DistilBERT fine-tuning step")
    parser.add_argument("--skip-ml", action="store_true", help="Skip classical ML training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    print("=== Preprocessing datasets ===")
    preprocess.preprocess_and_split()

    if not args.skip_ml:
        print("=== Training ML models ===")
        train_ml.train_models()

    if not args.skip_transformer:
        print("=== Fine-tuning DistilBERT ===")
        train_transformer.train_transformer()

    if not args.skip_eval:
        print("=== Evaluating models ===")
        evaluate.evaluate_all()


if __name__ == "__main__":
    main()
