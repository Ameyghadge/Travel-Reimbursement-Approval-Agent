# Travel Reimbursement Approval Agent

An AI-powered agent that evaluates employee travel reimbursement claims against company policy using **Qwen2.5-3B-Instruct** (local LLM), **LangGraph** (workflow orchestration), **RAG** (policy retrieval via FAISS + BGE-small embeddings), and deterministic validation tools.

100% open-source. No proprietary API dependencies.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                          │
│   FastAPI (/claims/evaluate)  │  CLI  │  Streamlit UI   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LangGraph Workflow                          │
│  validate → retrieve_policy → tools → llm → validate   │
└──────┬───────────┬──────────────────┬───────────────────┘
       │           │                  │
┌──────▼───┐ ┌────▼────┐  ┌──────────▼──────────────────┐
│   RAG    │ │  Tools  │  │    Qwen2.5-3B-Instruct      │
│  FAISS   │ │ Receipt │  │    (Local HF Transformers)   │
│ BGE-small│ │ Limits  │  └─────────────────────────────┘
│          │ │ Dupes   │
│          │ │ Approval│
└──────────┘ └─────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- ~16GB RAM (CPU mode) or ~8GB VRAM (GPU mode)
- ~6GB disk for Qwen model download (first run)

### Installation

```bash
git clone <repo-url>
cd AI-Agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run the CLI (process all sample claims)

```bash
python cli.py
```

### Run the CLI (single claim)

```bash
python cli.py --claim data/claims/claim_001.json --verbose
```

### Run the API server

```bash
python main.py
# Or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Then test:
```bash
curl -X POST http://localhost:8000/claims/evaluate \
  -H "Content-Type: application/json" \
  -d @data/claims/claim_001.json
```

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Project Structure

```
├── main.py                 # FastAPI app
├── cli.py                  # CLI entry point
├── streamlit_app.py        # Demo UI
├── config.py               # Centralized settings
├── llm/
│   ├── loader.py           # Singleton model loading
│   └── inference.py        # generate(prompt) interface
├── rag/
│   ├── embeddings.py       # BGE-small wrapper
│   ├── vector_store.py     # FAISS index
│   └── retriever.py        # Policy chunking + retrieval
├── agent/
│   ├── graph.py            # LangGraph workflow
│   ├── state.py            # AgentState definition
│   └── prompts.py          # System + reasoning prompts
├── tools/
│   ├── policy_lookup.py    # RAG-based policy retrieval
│   ├── receipt_validator.py
│   ├── expense_limit_checker.py
│   ├── duplicate_checker.py
│   ├── approval_matrix.py
│   └── output_validator.py
├── models/
│   ├── claim.py            # ClaimRequest schema
│   └── decision.py         # DecisionResult schema
├── data/
│   ├── travel_policy.md    # Company policy (RAG source)
│   ├── limits.json         # Category expense limits
│   ├── approval_matrix.json
│   └── claims/             # 5 sample claims
└── sample_outputs/         # Pre-generated decisions
```

## How It Works

1. **Claim Intake** — JSON claim validated by Pydantic schema
2. **Policy Retrieval** — Top-3 relevant policy chunks retrieved via FAISS similarity search
3. **Tool Execution** — Receipt validation, expense limits, duplicate detection, approval matrix
4. **LLM Reasoning** — Qwen2.5-3B-Instruct combines policy context + tool outputs to reason
5. **Output Validation** — JSON output validated against DecisionResult schema (retry up to 3x)
6. **Decision** — Approve, Partially Approve, Reject, or Manual Review with full audit trail

## Tools

| Tool | Purpose |
|------|---------|
| Policy Lookup | Retrieves Top-3 relevant policy sections via RAG |
| Receipt Validator | Checks receipt presence/consistency per expense |
| Expense Limit Checker | Validates amounts against category limits |
| Duplicate Checker | Detects potential duplicate submissions |
| Approval Matrix | Determines required approval level |
| Output Validator | Validates LLM JSON against schema |

## Sample Claims

| File | Scenario | Expected Decision |
|------|----------|-------------------|
| claim_001.json | All compliant, reasonable amounts | Approve |
| claim_002.json | Hotel over per-diem, meal over limit | Partially Approve |
| claim_003.json | First class, missing receipts, spa, alcohol | Reject |
| claim_004.json | Emergency trip, missing receipts, no pre-approval | Manual Review |
| claim_005.json | International, business class (approved), all compliant | Approve |

## Configuration

All settings in `config.py` / `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_NAME | Qwen/Qwen2.5-3B-Instruct | HuggingFace model ID |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Embedding model |
| MAX_RETRIES | 3 | LLM output retry attempts |
| TOP_K_CHUNKS | 3 | Policy chunks retrieved |
| CHUNK_SIZE | 512 | Policy chunk size (chars) |
| TEMPERATURE | 0.1 | LLM generation temperature |

## Design Decisions & Trade-offs

- **Qwen2.5-3B-Instruct**: Lightweight enough for local execution, reliable at structured JSON output with low temperature
- **LangGraph over LangChain agents**: Explicit workflow control, easier debugging, deterministic tool ordering
- **FAISS IndexFlatL2**: Exact search is fine for <30 policy chunks, no approximate indexing overhead
- **Deterministic tools + LLM reasoning**: Tools provide ground truth; LLM synthesizes and explains
- **Retry with correction**: Small models sometimes produce malformed JSON; retrying with error context fixes most cases
- **Manual Review fallback**: Never force an incorrect decision; route uncertainty to humans

## Assumptions & Limitations

- Uses mock claim history (duplicate detection always reports "no history")
- Single-currency (USD) — no real-time FX conversion
- Receipts are metadata only (no OCR/image parsing)
- Policy corpus is small enough for full FAISS flat search
- No persistent storage of decisions between runs
- No authentication on API endpoints (prototype scope)

## What I'd Improve Next

1. **Persistent claim history** for real duplicate detection
2. **Receipt OCR** via vision model for actual document verification
3. **RAG evaluation** — measure retrieval quality with labeled queries
4. **Larger model option** for nuanced edge cases (7B+ when hardware allows)
5. **Approval workflow integration** — route Manual Review to actual approvers
6. **Evaluation harness** — automated scoring of decisions against labeled test set
