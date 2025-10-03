"""CLI utilities for Task B."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .pipeline import get_pipeline

app = typer.Typer(help="SOC automation CLI")


@app.command()
def score_file(
    csv: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Path to CSV file."),
    text_col: str = typer.Option("body", help="Column containing message body."),
    subject_col: str = typer.Option("subject", help="Column containing subject."),
    html_col: Optional[str] = typer.Option(None, help="Column indicating HTML content (bool)."),
    out: Path = typer.Option(Path("scored.csv"), help="Output CSV path."),
    config: Path = typer.Option(Path("configs/task_b.yaml"), help="Config file path."),
) -> None:
    df = pd.read_csv(csv)
    pipeline = get_pipeline(str(config))
    scores = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        subject = row_dict.get(subject_col, "") if subject_col in df.columns else ""
        body = str(row_dict.get(text_col, ""))
        as_html = bool(row_dict.get(html_col)) if html_col and html_col in df.columns else False
        result = pipeline.score_text(subject, body, as_html)
        scores.append(result)
    df["pred_label"] = [item["label"] for item in scores]
    df["pred_score"] = [item["score"] for item in scores]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"Wrote scored file to {out}")


if __name__ == "__main__":
    app()
