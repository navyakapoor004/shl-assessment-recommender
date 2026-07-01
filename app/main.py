"""
SHL Assessment Recommender — FastAPI entrypoint.

Endpoints:
  POST /chat    -> ChatRequest in, ChatResponse out (see app/schemas.py)
  GET  /health  -> {"status": "ok"}

Run locally:
  export GROQ_API_KEY=sk-...
  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.orchestrator import handle_conversation
from app.services.retrieval import retrieval_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

REQUEST_TIMEOUT_SECONDS = 30

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments from a grounded catalog.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    logger.info("Loading catalog + building FAISS index...")
    retrieval_service.load()
    logger.info("Loaded %d catalog items.", len(retrieval_service.catalog))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.monotonic()
    try:
        response = handle_conversation(request.messages)
    except ValidationError as exc:
        logger.exception("Schema validation failed")
        raise HTTPException(status_code=502, detail=f"Invalid model output: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — top-level safety net for a public API
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal error handling chat request") from exc

    elapsed = time.monotonic() - start
    if elapsed > REQUEST_TIMEOUT_SECONDS:
        logger.warning("Request exceeded %ss budget: %.2fs", REQUEST_TIMEOUT_SECONDS, elapsed)
    else:
        logger.info("Handled /chat in %.2fs", elapsed)

    return response
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")