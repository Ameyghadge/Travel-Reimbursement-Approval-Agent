# Travel Reimbursement Approval Agent

A lightweight agentic AI prototype that evaluates travel expense claims against company policy.

**Stack:** Python + LangGraph + Qwen2.5-1.5B-Instruct (local) + RAG (FAISS + all-MiniLM-L6-v2) + 2 tools + Streamlit UI


## How It Works

```
Claim JSON
    │
    ▼
┌───────────────────────────────────────────┐
│       LangGraph Workflow (4 nodes)         │
│                                           │
│  [1] LLM selects tools (agentic)         │
│  [2] RAG retrieves policy (FAISS)        │
│  [3] Tools execute:                       │
│      • Receipt Validator                  │
│      • Expense Limit Checker              │
│  [4] LLM generates reasoning text        │
│                                           │
│  Decision = derived from tool outputs     │
│  (no LLM hallucination on decision type)  │
└───────────────────────────────────────────┘
    │
    ▼
Structured JSON Decision + Audit Trail
    │
    ▼ (if manual_review)
Human reviewer approves/rejects via Streamlit UI
```

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run Streamlit UI (recommended for demo)
streamlit run streamlit_app.py

# Or run API server
python main.py
```

First run downloads models (~3GB LLM + ~90MB embeddings). Subsequent starts: ~20s.

## Agentic AI Workflow

The LLM participates in **two** steps:
1. **Tool Planning** — LLM decides which tools to call given the claim
2. **Reasoning** — LLM generates explanation text from tool results

The **decision type** (approve/partially_approve/reject/manual_review) is derived deterministically from tool outputs to eliminate hallucination.

## Tools (2 meaningful tools)

| Tool | What it does |
|------|-------------|
| **Receipt Validator** | Checks receipt presence (required >$25) and amount match |
| **Expense Limit Checker** | Compares each expense against category limits from `data/limits.json` |

## Sample Claims & Expected Outputs

| Claim | Scenario | Decision | Approved |
|-------|----------|----------|----------|
| 001 | All compliant | ✅ approve | $932 |
| 002 | Hotel $289 > $200 limit (3 nights) | ⚠️ partially_approve | $1,550 |
| 003 | First class + spa + alcohol + missing receipts | ❌ reject | $100 |
| 004 | Missing receipt + receipt amount mismatch | 🔍 manual_review | $220 |
| 005 | International, all within intl limits | ✅ approve | $3,840 |

## Manual Review

When the agent returns "Manual Review", the Streamlit UI shows:
- Why it can't decide automatically
- What information is missing
- **✅ Manually Approve** and **❌ Manually Reject** buttons for human reviewer

## Project Structure

```
├── main.py              # FastAPI server (async)
├── streamlit_app.py     # Demo UI with manual review buttons
├── config.py            # All settings in one place
├── agent/
│   ├── graph.py         # LangGraph workflow (4 nodes)
│   ├── prompts.py       # LLM prompt template
│   └── state.py         # AgentState TypedDict
├── llm/
│   ├── loader.py        # Singleton model loading
│   └── inference.py     # generate(prompt) → str
├── rag/
│   ├── embeddings.py    # all-MiniLM-L6-v2 (singleton)
│   ├── vector_store.py  # FAISS IndexFlatL2
│   └── retriever.py     # Policy chunking + retrieval
├── tools/
│   ├── receipt_validator.py
│   └── expense_limit_checker.py
├── models/
│   ├── claim.py         # ClaimRequest schema
│   └── decision.py      # DecisionResult + AuditTrail
├── data/
│   ├── travel_policy.md # Policy (RAG source)
│   ├── limits.json      # Expense limits
│   └── claims/          # 5 sample claims
└── sample_outputs/      # Pre-generated examples
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_NAME | Qwen/Qwen2.5-1.5B-Instruct | Local LLM |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | RAG embeddings |
| MAX_NEW_TOKENS | 120 | LLM output limit |
| TEMPERATURE | 0.0 | Greedy decoding |
| TOP_K_CHUNKS | 2 | Policy chunks retrieved |

## Design Decisions

- **2 tools only** — focused on highest-value checks (receipt + limits)
- **LLM selects tools** — demonstrates agentic capability
- **Decision from tool outputs** — eliminates LLM hallucination on classification
- **LLM provides reasoning** — natural language explanation still AI-generated
- **Manual Review with buttons** — uncertain cases go to humans
- **RAG for policy** — only relevant sections sent to LLM
- **LangGraph** — clear, inspectable 4-node workflow
- **Lightweight** — practical prototype, not enterprise system

## Assumptions & Limitations

- Mock data only (no real employee data)
- Single currency (USD)
- Receipts are metadata (no OCR)
- No persistent storage between runs
- No authentication (prototype scope)
- ~15-20s per claim on CPU (model inference bound)
