# Task B – SOC Automation Microservice

Implements a FastAPI-based microservice that loads the best phishing detection model, classifies incoming content, enriches results with lightweight IOC extraction, and exposes both API and CLI interfaces.

## Running the API

```bash
uvicorn task_b_soc.serving:app --reload --port 8000
```

Visit `http://localhost:8000` to use the bundled single-page UI. Use `curl` or any HTTP client to interact with the `/classify` and `/classify-eml` endpoints programmatically.

## CLI Batch Scoring

```bash
python -m task_b_soc.cli score-file --csv path/to/messages.csv --text-col body --subject-col subject --html-col body_is_html --out scored.csv
```

## Configuration

The microservice reads `configs/task_b.yaml` (merged with `configs/global.yaml`) and environment variables. Update the `model.artifact_dir` to point to the trained artefacts exported by Task A.
