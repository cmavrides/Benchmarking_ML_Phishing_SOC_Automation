"""LLM-assisted phishing detection via zero/few-shot prompting."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import openai
except ImportError:  # pragma: no cover
    openai = None

from ..utils import sha1_hash

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a cybersecurity analyst who classifies messages as phishing or legitimate."
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Subject: Account verification required\nBody: Please verify your account by clicking this link: http://malicious.example.com",
    },
    {"role": "assistant", "content": "phishing"},
    {
        "role": "user",
        "content": "Subject: Meeting Agenda\nBody: Attached is the agenda for tomorrow's meeting.",
    },
    {"role": "assistant", "content": "legitimate"},
]


@dataclass
class LLMConfig:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0


def _call_openai(messages: List[dict], config: LLMConfig) -> str:
    if openai is None:
        raise RuntimeError("openai package not available")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    for attempt in range(config.max_retries):
        try:
            completion = openai.ChatCompletion.create(
                model=config.model_name,
                temperature=config.temperature,
                messages=messages,
            )
            return completion.choices[0].message["content"].strip()
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("OpenAI request failed (%s/%s): %s", attempt + 1, config.max_retries, exc)
            time.sleep(config.retry_delay)
    raise RuntimeError("Failed to obtain LLM response after retries")


def classify_messages(messages: Iterable[str], caller: Callable[[List[dict], LLMConfig], str] | None = None, config: Optional[LLMConfig] = None) -> List[int]:
    """Classify messages using an LLM."""

    config = config or LLMConfig()
    caller = caller or _call_openai
    results: List[int] = []
    for message in messages:
        prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT_EXAMPLES + [
            {"role": "user", "content": message[:4000]}
        ]
        response = caller(prompt_messages, config)
        normalized = response.strip().lower()
        if "phishing" in normalized or normalized == "1":
            results.append(1)
        else:
            results.append(0)
    return results
