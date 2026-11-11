"""
Backend API for our modular RAG pipeline (built with FastAPI).

The frontend (frontend/index.html) talks to these endpoints:
    GET  /api/health    -> is the server alive?
    POST /api/pipeline  -> upload PDFs + settings, build the pipeline   (batch 4)
    POST /api/chat      -> ask a question, get an answer + sources      (batch 5)

Run from the project root:
    python main.py
Then open:
    http://localhost:8000          -> the frontend
    http://localhost:8000/docs     -> FastAPI's automatic API docs (try endpoints in the browser)
"""

import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from utils.pipeline import PipelineConfig, RAGPipeline

FRONTEND_DIR = Path(__file__).parent / "frontend"
UPLOAD_DIR = Path(__file__).parent / "uploads"   # uploaded PDFs are saved here, one folder per pipeline

# All built pipelines live here while the server runs:  {"3f9a1c2e": RAGPipeline, ...}
# (it's just a dict in memory - restarting the server empties it)
pipelines: dict[str, RAGPipeline] = {}

# The application object - every endpoint gets attached to it
app = FastAPI(title="RAG Studio API")

# CORS = which websites may call this API from a browser.
# "*" (everyone) is fine for local development, e.g. when index.html is opened by double-clicking.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoint 1: health check - the simplest possible endpoint
# ---------------------------------------------------------------------------

@app.get("/api/health")          # GET request to /api/health runs this function
def health():
    print("[health] someone checked if the server is alive")
    return {"status": "ok"}      # FastAPI turns the dict into JSON automatically


# ---------------------------------------------------------------------------
# Endpoint 2: build a pipeline from uploaded PDFs + settings
#
# The frontend sends "form-data" (the format browsers use to upload files):
#   files  = one or more PDF files
#   config = the settings as a JSON string
# ---------------------------------------------------------------------------

@app.post("/api/pipeline")
def create_pipeline(
    files: list[UploadFile] = File(...),   # File(...) = required uploaded file(s)
    config: str = Form(...),               # Form(...) = required text field
):
    # 1. Check the input BEFORE doing any expensive work ---------------------
    try:
        pipeline_config = PipelineConfig.model_validate_json(config)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=f"Invalid config: {error}")

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

    # 2. One pipeline at a time: remove the previous one(s) -------------------
    for old_id in list(pipelines):
        delete_pipeline(old_id)

    # 3. Create the pipeline and save the PDFs into uploads/<pipeline id>/ ----
    pipeline = RAGPipeline(pipeline_config)
    folder = UPLOAD_DIR / pipeline.id
    folder.mkdir(parents=True, exist_ok=True)

    pdf_paths = []
    for file in files:
        path = folder / Path(file.filename).name     # .name drops any folder part of the filename
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)       # copy the uploaded bytes into the file on disk
        pdf_paths.append(path)
    print(f"[pipeline] saved {len(pdf_paths)} file(s) to {folder}")

    # 4. Build: chunk -> embed -> store ---------------------------------------
    try:
        num_chunks = pipeline.build(pdf_paths)
    except ValueError as error:                      # e.g. no text in the PDF, overlap >= chunk size
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error))

    pipelines[pipeline.id] = pipeline
    print(f"[pipeline] {pipeline.id} ready: {num_chunks} chunks from {pipeline.documents}")

    # 5. Tell the frontend which pipeline to use when chatting ----------------
    return {"pipeline_id": pipeline.id, "num_chunks": num_chunks, "documents": pipeline.documents}


def delete_pipeline(pipeline_id: str) -> None:
    """Forget a pipeline: drop its Chroma collection and its uploaded files."""
    pipeline = pipelines.pop(pipeline_id)
    pipeline.store.client.delete_collection(pipeline.store.collection.name)
    shutil.rmtree(UPLOAD_DIR / pipeline_id, ignore_errors=True)
    print(f"[pipeline] deleted old pipeline {pipeline_id}")


# ---------------------------------------------------------------------------
# Serve the frontend at http://localhost:8000
# Must be LAST: it catches every URL that no endpoint above has claimed.
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    # reload=True restarts the server automatically whenever you save a file
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
