"""FastAPI application for the Travel Reimbursement Approval Agent."""

import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from config import settings
from models.claim import ClaimRequest
from models.decision import DecisionResponse

# Logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

from llm.loader import ModelLoader
from llm.inference import InferenceEngine
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSStore
from rag.retriever import PolicyRetriever
from agent.graph import ReimbursementAgent

logger = structlog.get_logger()

agent = None
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("startup_begin")

    loader = ModelLoader.get_instance(settings.model_name)
    loader.load()
    engine = InferenceEngine(loader)

    embedding_model = EmbeddingModel.get_instance(settings.embedding_model)
    store = FAISSStore(embedding_model)
    retriever = PolicyRetriever(settings.policy_path, store)
    retriever.initialize()

    agent = ReimbursementAgent(inference_engine=engine, retriever=retriever)

    logger.info("startup_complete")
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    description="Agentic AI workflow: LLM (Qwen2.5-1.5B) selects tools and generates reasoning. Tools (receipt_validator + expense_limit_checker) provide deterministic analysis. RAG (FAISS + all-MiniLM-L6-v2) retrieves policy context. Decision derived from tool outputs. Manual Review routes to human reviewer.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "healthy" if agent else "loading", "model": settings.model_name}


@app.post("/claims/evaluate", response_model=DecisionResponse)
async def evaluate_claim(claim: ClaimRequest):
    if agent is None:
        raise HTTPException(503, "Agent not ready")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, agent.process_claim, claim)


@app.get("/claims/samples")
async def list_samples():
    samples = []
    for p in sorted(Path("data/claims").glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        samples.append({"file": p.name, "claim_id": d["claim_id"], "employee": d["employee"]["name"], "total": d["total_amount"]})
    return samples


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port)
