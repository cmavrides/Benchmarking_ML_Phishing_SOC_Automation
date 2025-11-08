"""IOC extraction utilities."""
from __future__ import annotations

import re
from typing import Dict, List

IOC_PATTERNS = {
    "urls": re.compile(r"https?://[\w\-./?=&%]+", re.IGNORECASE),
    "ips": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "domains": re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE),
    "emails": re.compile(r"[\w\.-]+@[\w\.-]+\.[a-z]{2,}", re.IGNORECASE),
}


def _unique(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_iocs(text: str) -> Dict[str, List[str]]:
    """Extract simple indicators of compromise from text."""

    iocs: Dict[str, List[str]] = {}
    for key, pattern in IOC_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            iocs[key] = _unique(matches)
    return iocs


def summarize_iocs(iocs: Dict[str, List[str]]) -> Dict[str, int]:
    """Provide lightweight counts for IOC categories."""

    summary = {key: len(values) for key, values in iocs.items()}
    summary["total"] = sum(summary.values())
    return summary


__all__ = ["extract_iocs", "summarize_iocs"]
