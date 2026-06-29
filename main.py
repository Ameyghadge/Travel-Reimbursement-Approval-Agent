"""FastAPI application for the Travel Reimbursement Approval Agent."""

import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from config import settings
from models.claim import ClaimRequest
from models.decision import DecisionResponse
from llm.loader import ModelLoader
from llm.inference import InferenceEngine
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSStore
from rag.retriever import PolicyRetriever
from tools import (
    PolicyLookupTool,
    ReceiptValidationTool,
    ExpenseLimitChecker,
    DuplicateClaimChecker,
    ApprovalMatrixTool,
    OutputValidationTool,
)
from agent.graph import ReimbursementAgent

logger = structlog.get_logger()

# Global agent instance and thread pool
agent = None
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components at startup."""
    global agent
    logger.info("startup_begin")

    # 1. Load LLM
    loader = ModelLoader.get_instance(settings.model_name)
    loader.load()
    engine = InferenceEngine(loader)

    # 2. Initialize RAG pipeline
    embedding_model = EmbeddingModel.get_instance(settings.embedding_model)
    store = FAISSStore(embedding_model)
    retriever = PolicyRetriever(settings.policy_path, store)
    retriever.initialize()

    # 3. Initialize tools
    policy_lookup = PolicyLookupTool(retriever)
    receipt_validator = ReceiptValidationTool()
    limit_checker = ExpenseLimitChecker()
    duplicate_checker = DuplicateClaimChecker()
    approval_matrix = ApprovalMatrixTool()
    output_validator = OutputValidationTool()

    # 4. Build agent
    agent = ReimbursementAgent(
        inference_engine=engine,
        policy_lookup=policy_lookup,
        receipt_validator=receipt_validator,
        limit_checker=limit_checker,
        duplicate_checker=duplicate_checker,
        approval_matrix=approval_matrix,
        output_validator=output_validator,
    )

    logger.info("startup_complete")
    yield
    executor.shutdown(wait=False)
    logger.info("shutdown")


app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    description="AI-powered travel expense claim evaluation using Qwen2.5-3B-Instruct, LangGraph, and RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if agent else "initializing",
        "model": settings.model_name,
        "embedding_model": settings.embedding_model,
    }


@app.post("/claims/evaluate", response_model=DecisionResponse)
async def evaluate_claim(claim: ClaimRequest):
    """Evaluate a travel reimbursement claim (async — non-blocking)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, agent.process_claim, claim)
        return result
    except Exception as e:
        logger.error("claim_processing_error", error=str(e), claim_id=claim.claim_id)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/claims/sample/{claim_id}")
async def get_sample_claim(claim_id: str):
    """Return a sample claim for testing."""
    claims_dir = Path("data/claims")
    for path in claims_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if data.get("claim_id") == claim_id or claim_id in path.stem:
            return data
    raise HTTPException(status_code=404, detail=f"Sample claim '{claim_id}' not found")


@app.get("/claims/samples")
async def list_sample_claims():
    """List available sample claims."""
    claims_dir = Path("data/claims")
    samples = []
    for path in sorted(claims_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        samples.append({
            "file": path.name,
            "claim_id": data.get("claim_id"),
            "employee": data.get("employee", {}).get("name"),
            "destination": data.get("trip", {}).get("destination"),
            "total_amount": data.get("total_amount"),
        })
    return samples


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
