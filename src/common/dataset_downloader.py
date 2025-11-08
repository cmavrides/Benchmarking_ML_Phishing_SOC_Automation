"""Utilities for downloading the raw phishing datasets on demand."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm


DATA_DIR = Path("data")


@dataclass(frozen=True)
class DatasetSource:
    """Description of a remotely hosted dataset."""

    name: str
    filename: str
    url: str
    format: str = "csv"
    max_rows_env: Optional[str] = None

    def destination(self, data_dir: Path) -> Path:
        return data_dir / self.filename


DATA_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="zefang_liu",
        filename="zefang_liu.csv",
        url=(
            "https://huggingface.co/datasets/zefang-liu/phishing-email-"
            "dataset/resolve/main/Phishing_Email.csv?download=1"
        ),
    ),
    DatasetSource(
        name="cyradar",
        filename="cyradar.csv",
        url=(
            "https://huggingface.co/datasets/huynq3Cyradar/Phishing_"
            "Detection_Dataset/resolve/main/combined_reduced.csv?download=1"
        ),
        max_rows_env="CYRADAR_MAX_ROWS",
    ),
)


def _maybe_parse_positive_int(value: str | None) -> Optional[int]:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid integer value: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {parsed}")
    return parsed


def _write_binary_stream(response: requests.Response, dest: Path) -> None:
    total = int(response.headers.get("Content-Length") or 0)
    dest_tmp = dest.with_suffix(dest.suffix + ".tmp")
    with dest_tmp.open("wb") as file_obj:
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            leave=False,
        )
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file_obj.write(chunk)
            progress.update(len(chunk))
        progress.close()
    dest_tmp.replace(dest)


def _write_text_stream(
    response: requests.Response,
    dest: Path,
    max_rows: Optional[int] = None,
) -> None:
    dest_tmp = dest.with_suffix(dest.suffix + ".tmp")
    iterator = response.iter_lines(decode_unicode=True)
    with dest_tmp.open("w", encoding="utf-8", newline="") as file_obj:
        progress = tqdm(unit="rows", desc=f"{dest.name} rows", leave=False)
        header_written = False
        data_rows = 0
        for line in iterator:
            if line is None:
                continue
            file_obj.write(line)
            file_obj.write("\n")
            if not header_written:
                header_written = True
                continue
            data_rows += 1
            progress.update(1)
            if max_rows is not None and data_rows >= max_rows:
                break
        progress.close()
    dest_tmp.replace(dest)


def _convert_jsonl_to_csv(source_path: Path, dest_path: Path) -> None:
    import csv

    with source_path.open("r", encoding="utf-8") as jsonl_file, dest_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "source", "text", "label"])
        for line in jsonl_file:
            if not line.strip():
                continue
            record = json.loads(line)
            text = f"{record.get('subject', '')}\n{record.get('body', '')}".strip()
            label = 1 if str(record.get("label", "")).lower() == "phishing" else 0
            writer.writerow([record.get("id", ""), "cyradar", text, label])


def _download_source(source: DatasetSource, data_dir: Path) -> Path:
    destination = source.destination(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {source.name} dataset to {destination}")
    with requests.get(source.url, stream=True, timeout=120) as response:
        response.raise_for_status()
        if source.format == "csv":
            max_rows = None
            if source.max_rows_env:
                max_rows = _maybe_parse_positive_int(os.getenv(source.max_rows_env))
            if response.headers.get("Content-Type", "").startswith("text/"):
                _write_text_stream(response, destination, max_rows=max_rows)
            else:
                _write_binary_stream(response, destination)
        elif source.format == "jsonl":
            tmp_path = destination.with_suffix(destination.suffix + ".jsonl")
            _write_binary_stream(response, tmp_path)
            _convert_jsonl_to_csv(tmp_path, destination)
            tmp_path.unlink(missing_ok=True)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {source.format}")

    return destination


def ensure_raw_datasets(data_dir: Path = DATA_DIR) -> Iterable[Path]:
    """Download the required datasets if they are missing."""

    resolved_paths = []
    for source in DATA_SOURCES:
        destination = source.destination(data_dir)
        if destination.exists():
            resolved_paths.append(destination)
            continue
        resolved_paths.append(_download_source(source, data_dir))
    return resolved_paths


def main() -> None:  # pragma: no cover - convenience CLI
    ensure_raw_datasets(DATA_DIR)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()

