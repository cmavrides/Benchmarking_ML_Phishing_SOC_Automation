"""LLM-based classification helpers (zero/few-shot)."""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List

from common_utils.logging_conf import get_logger

LOGGER = get_logger(__name__)


class MissingAPIKeyError(RuntimeError):
    pass


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is required for LLM-based classification."
        )
    return api_key


def _build_prompt(subject: str, body: str, mode: str) -> str:
    system = "You are a security analyst classifying emails as phishing (1) or legitimate (0)."
    instructions = (
        "Return a JSON object with keys 'label' (0 or 1) and 'reason'. "
        "Base your decision on common phishing traits."
    )
    if mode == "few":
        examples = (
            "Examples:\n"
            "1. Subject: 'Urgent account verification' Body: 'Click this link to avoid closure' => label 1\n"
            "2. Subject: 'Team lunch plans' Body: 'Let's meet at noon' => label 0"
        )
    else:
        examples = ""
    return f"{system}\n{examples}\n{instructions}\nSubject: {subject}\nBody: {body}"


def classify_texts(texts: Iterable[Dict[str, str]], mode: str = "zero") -> List[int]:
    _require_api_key()
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for LLM modes") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]

    predictions: List[int] = []
    for item in texts:
        subject = item.get("subject", "")
        body = item.get("body", "")
        prompt = _build_prompt(subject, body, mode)
        response = client.responses.create(  # type: ignore[attr-defined]
            model="gpt-3.5-turbo-instruct" if mode == "zero" else "gpt-3.5-turbo",
            input=prompt,
            temperature=0,
        )
        content = response.output[0].content[0].text  # type: ignore[index]
        label = _parse_label(content)
        predictions.append(label)
        time.sleep(0.5)
    return predictions


def _parse_label(text: str) -> int:
    text = text.strip().lower()
    if "\"label\"" in text:
        import json

        try:
            data = json.loads(text)
            label = int(data.get("label", 0))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to parse JSON response: %s", exc)
            label = 0
    else:
        label = 1 if "phishing" in text or "1" in text[:10] else 0
    return 1 if label == 1 else 0
