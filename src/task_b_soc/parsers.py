"""Parsers for incoming data formats."""

from __future__ import annotations

import email
from email import policy
from email.message import Message
from typing import Tuple


from common_utils.logging_conf import get_logger

LOGGER = get_logger(__name__)


def _extract_body(message: Message) -> Tuple[str, bool]:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return part.get_content(), False
            if content_type == "text/html":
                return part.get_content(), True
    payload = message.get_body(preferencelist=("plain", "html"))
    if payload:
        content = payload.get_content()
        return content, payload.get_content_type() == "text/html"
    return message.get_content(), False


def parse_eml(data: bytes) -> Tuple[str, str, bool]:
    message = email.message_from_bytes(data, policy=policy.default)
    subject = message.get("subject", "")
    body, is_html = _extract_body(message)
    return subject or "", body or "", is_html
