"""FastAPI application exposing the classification pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common_utils.logging_conf import get_logger

from .parsers import parse_eml
from .pipeline import get_pipeline

LOGGER = get_logger(__name__)

app = FastAPI(title="Phishing Detection SOC Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ui_dir = Path(__file__).resolve().parent / "ui"
app.mount("/static", StaticFiles(directory=_ui_dir), name="static")


class ClassifyRequest(BaseModel):
    subject: str | None = ""
    body: str
    as_html: bool = False


@app.on_event("startup")
async def startup_event() -> None:
    config_path = os.getenv("TASK_B_CONFIG", "configs/task_b.yaml")
    try:
        get_pipeline(config_path)
    except FileNotFoundError as exc:
        LOGGER.error("Pipeline initialisation failed: %s", exc)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_ui_dir / "index.html")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/classify")
async def classify(req: ClassifyRequest) -> Dict[str, Any]:
    pipeline = get_pipeline(os.getenv("TASK_B_CONFIG", "configs/task_b.yaml"))
    result = pipeline.score_text(req.subject or "", req.body, req.as_html)
    return result


@app.post("/classify-eml")
async def classify_eml(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    subject, body, is_html = parse_eml(data)
    pipeline = get_pipeline(os.getenv("TASK_B_CONFIG", "configs/task_b.yaml"))
    result = pipeline.score_text(subject, body, is_html)
    return {
        "subject": subject,
        "body_length": len(body),
        "as_html": is_html,
        "result": result,
    }
