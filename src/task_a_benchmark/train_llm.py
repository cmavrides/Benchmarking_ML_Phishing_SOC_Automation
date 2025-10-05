"""Zero/few-shot phishing detection using an API-based LLM."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("reports/results/llm_predictions.csv")
DEFAULT_MODEL = "gpt-3.5-turbo"


PROMPT_TEMPLATE = (
    "You are a security analyst. Decide if the following message is phishing or legitimate. "
    "Respond with a single word: 'phishing' or 'legitimate'.\n\nMessage:\n{message}\n"
)


def _load_split(name: str = "test") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run preprocess.py first.")
    return pd.read_csv(path)


def _parse_response(text: str) -> int:
    text = text.strip().lower()
    if "phish" in text:
        return 1
    if "legit" in text or "not phishing" in text:
        return 0
    # default to legitimate when uncertain
    return 0


def _load_client():
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'openai' package to run LLM inference.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    openai.api_key = api_key
    return openai


def run_llm_inference(model: str = DEFAULT_MODEL) -> Dict[str, float]:
    try:
        openai = _load_client()
    except (ImportError, EnvironmentError) as err:
        print(f"Skipping LLM inference: {err}")
        return {}

    df = _load_split("test")
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM inference"):
        message = row.get("clean_text") or row.get("text")
        prompt = PROMPT_TEMPLATE.format(message=message)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Classify phishing emails."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=5,
            )
            content = response["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to classify sample: {exc}")
            content = "legitimate"

        label = _parse_response(content)
        predictions.append({
            "id": row.get("id"),
            "text": message,
            "predicted_label": label,
            "raw_response": content,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved LLM predictions to {OUTPUT_PATH}")
    return {"samples": float(len(predictions))}


if __name__ == "__main__":
    run_llm_inference()
