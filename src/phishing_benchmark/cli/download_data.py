"""Download phishing datasets from Hugging Face into the local data/raw directory."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from datasets import DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

app = typer.Typer(add_completion=False, help=__doc__)


def _ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_and_save(dataset_name: str, sample: Optional[int], force: bool) -> None:
    typer.echo(f"Downloading dataset '{dataset_name}'...")
    dataset: DatasetDict = load_dataset(dataset_name)
    dataset_dir = RAW_DATA_DIR / dataset_name.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in dataset.items():
        if sample is not None:
            sample_size = min(sample, len(split_ds))
            typer.echo(f"  Sampling first {sample_size} rows from split '{split_name}'.")
            split_ds = split_ds.select(range(sample_size))

        output_path = dataset_dir / f"{split_name}.parquet"
        if output_path.exists() and not force:
            typer.echo(f"  Skipping existing file: {output_path}")
            continue

        typer.echo(f"  Saving split '{split_name}' to {output_path}")
        split_df = split_ds.to_pandas()
        split_df.to_parquet(output_path, index=False)


@app.command()
def main(
    zefang_liu: bool = typer.Option(True, help="Download the zefang-liu/phishing-email-dataset."),
    cyradar: bool = typer.Option(True, help="Download the huynq3Cyradar/Phishing_Detection_Dataset."),
    sample: Optional[int] = typer.Option(
        None,
        min=1,
        help="Optionally limit each split to the first N records for quick experimentation.",
    ),
    force: bool = typer.Option(False, help="Overwrite existing files."),
) -> None:
    """Download selected phishing datasets from Hugging Face."""

    if not zefang_liu and not cyradar:
        raise typer.BadParameter("At least one dataset must be selected for download.")

    _ensure_directories()

    if zefang_liu:
        _download_and_save("zefang-liu/phishing-email-dataset", sample=sample, force=force)

    if cyradar:
        _download_and_save("huynq3Cyradar/Phishing_Detection_Dataset", sample=sample, force=force)

    typer.echo("Download complete.")


if __name__ == "__main__":
    app()
