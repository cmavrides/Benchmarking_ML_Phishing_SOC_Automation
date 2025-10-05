"""FastAPI service exposing phishing classification."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.task_b_soc import enrich, pipeline

STATIC_DIR = Path(__file__).parent / "ui"


class ClassifyRequest(BaseModel):
    text: str
    is_html: bool = False


app = FastAPI(title="Phishing Detection SOC API", version="0.1.0")


@app.post("/classify")
def classify(request: ClassifyRequest) -> JSONResponse:
    if not request.text:
        raise HTTPException(status_code=400, detail="'text' field is required")

    result = pipeline.classify_text(request.text, is_html=request.is_html)
    iocs = enrich.extract_iocs(request.text)
    payload = {"label": result["label"], "score": result["score"], "iocs": iocs}
    return JSONResponse(payload)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "Phishing Detection Suite API"})


if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
