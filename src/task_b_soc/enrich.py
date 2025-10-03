"""IOC extraction utilities."""

from __future__ import annotations

import re
from typing import Dict, List

IOC_PATTERNS = {
    "urls": re.compile(r"https?://[\w./%-]+", re.IGNORECASE),
    "domains": re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE),
    "emails": re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE),
    "ips": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "hashes": re.compile(r"\b[a-f0-9]{32}\b|\b[a-f0-9]{40}\b|\b[a-f0-9]{64}\b", re.IGNORECASE),
}


def deobfuscate(text: str) -> str:
    return text.replace("[.]", ".").replace("(dot)", ".")


def extract_iocs(text: str, max_items: int = 20) -> Dict[str, List[str]]:
    cleaned = deobfuscate(text)
    results: Dict[str, List[str]] = {}
    for name, pattern in IOC_PATTERNS.items():
        matches = pattern.findall(cleaned)
        unique = []
        for match in matches:
            norm = match.strip().strip('"')
            if norm and norm not in unique:
                unique.append(norm)
            if len(unique) >= max_items:
                break
        results[name] = unique
    return results
