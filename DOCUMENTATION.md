# Travel Reimbursement Approval Agent — Complete Documentation

## 1. Overview

This is a lightweight agentic AI prototype that evaluates employee travel reimbursement claims against company policy. It demonstrates:

- **GenAI reasoning**: LLM selects tools and generates explanations
- **Tool usage**: 2 deterministic tools (receipt validator + expense limit checker)
- **Context grounding**: RAG retrieves relevant policy sections via FAISS
- **Structured output**: Consistent JSON with decision, amounts, confidence, audit trail
- **Manual review handling**: Routes uncertain cases to human with approve/reject UI

**Technology**: Python, LangGraph, Qwen2.5-1.5B-Instruct (local), FAISS, all-MiniLM-L6-v2, FastAPI, Streamlit, Pydantic

**No proprietary APIs.** Runs entirely on a developer laptop.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                            │
│          FastAPI (POST /claims/evaluate)                      │
│          Streamlit UI (http://localhost:8501)                 │
└──────────────────────┬───────────────────────────────────────┘
                       │ ClaimRequest JSON
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               LangGraph StateGraph (4 nodes)                 │
│                                                              │
│  ┌────────────┐   ┌─────────────┐   ┌──────────┐   ┌─────┐│
│  │[1] Plan    │──▶│[2] Retrieve │──▶│[3] Run   │──▶│[4]  ││
│  │   Tools    │   │   Policy    │   │   Tools  │   │LLM  ││
│  │  (LLM)    │   │   (RAG)     │   │          │   │Reason││
│  └────────────┘   └─────────────┘   └──────────┘   └─────┘│
│                                                              │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   DecisionResponse JSON                       │
│  • decision: approve|partially_approve|reject|manual_review  │
│  • confidence_score: 0.0–1.0                                 │
│  • expense_decisions: per-item breakdown                     │
│  • overall_reasoning: LLM-generated explanation              │
│  • audit_trail: tools used, outputs, policy chunks, timing   │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼ (if manual_review)
┌──────────────────────────────────────────────────────────────┐
│          Streamlit UI: Manual Approve / Reject buttons        │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Processing Flow (Step by Step)

### Step 0: Claim Submission

User submits a JSON claim via FastAPI endpoint or Streamlit UI.
The claim is validated against `ClaimRequest` Pydantic schema.
Invalid claims are rejected with 422 and field-level errors.

### Step 1: LLM Tool Planning (Node: `plan_tools`)

**What happens:**
- The LLM receives the claim summary and list of available tools
- It decides which tools should be run for this particular claim
- This demonstrates the "agentic" capability — LLM drives tool selection

**Prompt sent to LLM:**
```
Claim: CLM-2025-001 | Alice Johnson | $932 | Categories: flight, hotel, meals, transport

Available tools: receipt_validator, expense_limit_checker
Which tools should I run? List them:
```

**LLM responds:** `"receipt_validator, expense_limit_checker"`

**Code parses response** and extracts tool names. If LLM doesn't respond clearly, both tools are run as default.

**File:** `agent/graph.py` → `_plan_tools()` method
**Max tokens:** 20 (only needs a short list)

### Step 2: Policy Retrieval (Node: `retrieve_policy`)

**What happens:**
- Builds a search query from claim categories + destination
- Encodes query with all-MiniLM-L6-v2 embedding model (384 dimensions)
- Searches FAISS IndexFlatL2 for Top-2 most similar policy chunks
- Returns chunk text + section name + relevance score

**Query example:** `"travel policy flight hotel meals transport Austin, TX"`

**Returns:** 2 chunks like:
- [Flight Policy]: "Economy class is the standard for all domestic flights..."
- [Hotel Policy]: "Domestic per-diem hotel rate is $200 per night..."

**File:** `rag/retriever.py` → `retrieve()` method
**Index:** Built at startup from `data/travel_policy.md` (12 chunks)

### Step 3: Tool Execution (Node: `run_tools`)

**What happens:** Both tools execute on the claim's expenses.

#### Tool 1: Receipt Validator (`tools/receipt_validator.py`)

For each expense item:
- If `receipt.has_receipt == False` AND `amount > $25`:
  - Amount > $75 → severity "error" (needs manager exception)
  - Amount $25–$75 → severity "warning" (declaration OK)
- If `receipt.receipt_amount_matches == False`:
  - Flag as "amount_mismatch" with severity "error"
- Otherwise → valid

**Output:** `ReceiptValidationResult`
```json
{"all_valid": false, "valid_count": 3, "missing_receipts": [...], "mismatched_amounts": [...], "total_issues": 2}
```

#### Tool 2: Expense Limit Checker (`tools/expense_limit_checker.py`)

For each expense item:
- Looks up the category limit from `data/limits.json`
- If `is_international == True`, uses international limits
- If `amount > limit` → violation with overage

**Limits (`data/limits.json`):**
```json
{
  "flight": {"domestic": 800, "international": 2500},
  "hotel": {"domestic_per_night": 200, "international_per_night": 350},
  "meals": {"per_day": 75, "single_meal_max": 50},
  "transport": {"per_day": 100},
  "conference": {"registration_max": 1500},
  "miscellaneous": {"per_trip": 150}
}
```

**Output:** `ExpenseLimitResult`
```json
{"all_compliant": false, "violations": [{"category": "hotel", "amount": 289, "limit": 200, "overage": 89}], "compliant_count": 5, "total_overage": 267.0}
```

### Step 4: LLM Reasoning (Node: `llm_decide`)

**What happens:**
- Builds a prompt with: policy context + claim details + tool summaries
- LLM generates a JSON response with decision type + reasoning text
- Code extracts the reasoning text from LLM output
- **Decision type is determined by tool outputs (not LLM)** — eliminates hallucination

**Prompt sent:**
```
You are a Travel Reimbursement Approval Agent.

POLICY:
Economy class is the standard for all domestic flights...
Domestic per-diem hotel rate is $200 per night...

CLAIM: CLM-2025-002 | Carlos Rivera | New York, NY | $1837
EXPENSES:
  flight $650 | Round-trip DEN to JFK | ✓
  hotel $289 | Night 1 - NYC hotel | ✓
  hotel $289 | Night 2 | ✓
  ...

TOOL RESULTS:
- Receipt Check: All receipts OK
- Limit Check: hotel $289>$200, hotel $289>$200, hotel $289>$200

DECIDE: approve / partially_approve / reject / manual_review
Return JSON: {"decision":"<type>","reasoning":"<1 sentence>"}
```

**LLM responds:** `{"decision":"partially_approve","reasoning":"Hotel exceeds domestic per-diem limit of $200/night."}`

**Decision logic (deterministic, in `_parse_response()`):**

| Condition | Decision |
|-----------|----------|
| Receipt amount mismatch found | manual_review |
| Missing critical receipt (>$75) + some approved items | manual_review |
| All items rejected | reject |
| 50%+ items rejected | reject |
| All items approved (zero violations) | approve |
| Some reduced or rejected + some approved | partially_approve |

**Non-reimbursable keyword detection:**
If expense description contains: `spa`, `alcohol`, `minibar`, `gym`, `movie`, `first class` → that item is auto-rejected.

### Step 5: Response Assembly

Combines:
- Decision type (from tool logic)
- Reasoning text (from LLM)
- Per-expense decisions (from tool outputs)
- Policy references (from RAG chunks)
- Audit trail (tools run, outputs, LLM response, timing)

Returns `DecisionResponse` JSON to the client.

---

## 4. Data Models

### Input: ClaimRequest

```python
class ClaimRequest:
    claim_id: str                  # "CLM-2025-001"
    employee: EmployeeInfo         # {employee_id, name, department, level}
    trip: TripDetails              # {destination, purpose, start_date, end_date, is_international}
    expenses: List[ExpenseItem]    # min 1 item
    total_amount: float            # > 0
    notes: str                     # free text
    submitted_at: str              # ISO datetime

class ExpenseItem:
    date: date                     # "2025-03-10"
    category: ExpenseCategory      # flight|hotel|meals|transport|conference|miscellaneous
    amount: float                  # > 0
    currency: str                  # "USD"
    vendor: str                    # "United Airlines"
    description: str               # "Round-trip SFO to AUS"
    receipt: ReceiptInfo           # {has_receipt, receipt_format, receipt_amount_matches}
```

### Output: DecisionResponse

```python
class DecisionResponse:
    decision: DecisionResult
    audit_trail: AuditTrail

class DecisionResult:
    claim_id: str
    decision: str                  # approve|partially_approve|reject|manual_review
    confidence_score: float        # 0.0–1.0
    overall_reasoning: str         # LLM-generated explanation
    expense_decisions: List[ExpenseDecision]  # per-item breakdown
    policy_references: List[PolicyReference]  # RAG chunks used
    total_approved_amount: float   # sum of approved amounts
    flags: List[str]               # warnings/notes
    requires_additional_info: List[str]  # what's missing (for manual_review)

class ExpenseDecision:
    category: str                  # "hotel"
    amount: float                  # claimed amount
    approved_amount: float         # approved (may be reduced or 0)
    status: str                    # "approved"|"reduced"|"rejected"
    reason: str                    # "Exceeds $200 limit" or "Within policy"

class AuditTrail:
    retrieved_chunks: List[str]    # policy text used
    tools_executed: List[str]      # ["receipt_validator", "expense_limit_checker"]
    tool_outputs: Dict             # structured results from each tool
    llm_raw_response: str          # unmodified LLM output
    retry_count: int               # 0 in normal flow
    processing_time_ms: float      # end-to-end time
```

---

## 5. RAG Pipeline

### Policy Document: `data/travel_policy.md`

Covers 10 sections:
- General Principles (pre-approval, 30-day submission, receipts >$25)
- Flight Policy ($800 domestic, $2500 international, economy only)
- Hotel Policy ($200/night domestic, $350/night international)
- Meals Policy ($75/day, no alcohol)
- Ground Transportation ($100/day)
- Conference Expenses ($1500 registration cap)
- Non-Reimbursable (spa, alcohol, minibar, gym, movies, first class)
- Miscellaneous ($150/trip)
- Approval Thresholds ($500→manager, $2000→director, $5000→VP)
- Documentation Requirements (receipts >$25, declarations $25–$75)

### Chunking

- Split by `##` markdown headers → preserves logical sections
- Further split chunks >512 chars with 50-char overlap
- Result: **12 chunks** stored in FAISS

### Embedding Model: all-MiniLM-L6-v2

- 22.7M parameters, 384 dimensions
- ~90MB download
- Loaded once as singleton, cached

### FAISS Index

- `IndexFlatL2` — exact L2 distance search
- Appropriate for 12 chunks (no approximation needed)
- Built once at startup, kept in memory

### Retrieval

- Query built from: expense categories + destination
- Returns Top-2 chunks with relevance scores
- Relevance = `1 / (1 + L2_distance)`

---

## 6. LLM Details

### Model: Qwen/Qwen2.5-1.5B-Instruct

| Property | Value |
|----------|-------|
| Parameters | 1.5 billion |
| Download size | ~3 GB |
| RAM (CPU, float32) | ~6 GB |
| Context window | 32K tokens |
| Inference | ~30-40 tokens/sec on CPU |

### How it's loaded (`llm/loader.py`)

- Singleton pattern — `ModelLoader.get_instance()` ensures one load
- Auto-detects CUDA → float16; CPU → float32
- Sets model to `.eval()` mode
- Loads tokenizer alongside model

### How it generates (`llm/inference.py`)

- Uses `tokenizer.apply_chat_template()` for Qwen format
- Greedy decoding (`temperature=0.0`, `do_sample=False`)
- `torch.no_grad()` during inference
- Input truncated to 2048 tokens
- Returns only generated tokens (strips input echo)

### LLM's two roles in the workflow

| Step | What LLM does | Max tokens | Purpose |
|------|--------------|------------|---------|
| [1/4] plan_tools | Lists which tools to run | 20 | Demonstrates agentic tool selection |
| [4/4] llm_decide | Generates reasoning JSON | 120 | Provides natural language explanation |

### What the LLM does NOT do

- Does NOT determine the decision type (tools do that)
- Does NOT produce per-expense breakdowns (deterministic logic does that)
- Does NOT validate its own output (code handles parsing)

---

## 7. Tools

### Tool 1: Receipt Validator

**File:** `tools/receipt_validator.py`
**Class:** `ReceiptValidationTool`

**Input:** List of `ExpenseItem` objects
**Output:** `ReceiptValidationResult`

**Logic per expense:**
```
if no receipt AND amount > $75:     → severity "error" (needs manager exception)
if no receipt AND $25 < amount ≤ $75: → severity "warning" (declaration OK)
if no receipt AND amount ≤ $25:     → valid (receipt not required)
if receipt AND amount_matches=False: → "amount_mismatch" (severity "error")
otherwise:                          → valid
```

**Threshold:** $25 (configurable via `RECEIPT_THRESHOLD`)

### Tool 2: Expense Limit Checker

**File:** `tools/expense_limit_checker.py`
**Class:** `ExpenseLimitChecker`

**Input:** List of `ExpenseItem` + `is_international` flag
**Output:** `ExpenseLimitResult`

**Logic per expense:**
```
limit = get_limit(category, is_international)
if amount > limit:  → violation with overage = amount - limit
else:               → compliant
```

**Limit lookup logic:**
- `flight`: domestic=$800, international=$2500
- `hotel`: domestic=$200/night, international=$350/night
- `meals`: $75/day
- `transport`: $100/day
- `conference`: $1500
- `miscellaneous`: $150/trip
- Unknown category: returns warning (not auto-reject)

---

## 8. Decision Classification Logic

The decision is derived **deterministically** from tool outputs in `_parse_response()`:

```python
# Per-item decisions built from:
# 1. Non-reimbursable keywords in description → rejected
# 2. Receipt amount mismatch → rejected
# 3. Receipt missing (>$25) → rejected
# 4. Over category limit → reduced to limit
# 5. Otherwise → approved

# Overall decision from item counts:
if has_receipt_mismatch:                    → "manual_review"
elif has_missing_critical AND some_approved: → "manual_review"
elif all_rejected:                          → "reject"
elif 50%+ rejected:                         → "reject"
elif all_approved:                          → "approve"
elif some_reduced_or_rejected:              → "partially_approve"
else:                                       → "approve"
```

### Non-Reimbursable Keywords

Checked against `expense.description.lower()`:
`spa`, `alcohol`, `minibar`, `mini-bar`, `gym`, `movie`, `first class`

If found → that expense is auto-rejected regardless of amount or receipt status.

---

## 9. Sample Claims & Expected Outputs

### Claim 001 — ✅ APPROVE ($932)
- All 6 expenses within limits
- All receipts present and matching
- No non-reimbursable keywords

### Claim 002 — ⚠️ PARTIALLY APPROVE ($1,550)
- Hotel $289/night exceeds $200 domestic limit (3 nights, reduced to $200 each)
- Meals $95 exceeds $75/day (reduced)
- Other expenses approved

### Claim 003 — ❌ REJECT ($100)
- "First class" in flight description → rejected
- "alcohol" in meals description → rejected
- "Spa" in miscellaneous description → rejected
- 3 hotel nights with missing receipts → rejected
- Only transport $150 partially survives (reduced to $100 limit)

### Claim 004 — 🔍 MANUAL REVIEW ($220)
- Flight $480 receipt missing (severity "error", >$75)
- Lunch $38 receipt amount mismatch (split bill)
- Dinner $42 receipt missing (severity "warning")
- Hotel $165 + Transport $55 approved (receipts OK, within limits)
- **UI shows Approve/Reject buttons for human reviewer**

### Claim 005 — ✅ APPROVE ($3,840)
- International trip → uses international limits ($2500 flight, $350/night hotel)
- All 11 expenses within international limits
- All receipts present and matching

---

## 10. Manual Review Feature

When the agent returns `decision: "manual_review"`, the Streamlit UI displays:

1. **Why:** The `overall_reasoning` explaining what's ambiguous
2. **What's missing:** The `requires_additional_info` list
3. **Per-expense breakdown:** Shows which items are OK and which need review
4. **Action buttons:**
   - ✅ **Manually Approve** — Human overrides to approve
   - ❌ **Manually Reject** — Human confirms rejection

**Triggers for manual_review:**
- Receipt amount mismatch (could be legitimate split bill, needs verification)
- Missing critical receipt (>$75) but some expenses are valid (not a full reject)

---

## 11. API Reference

### POST /claims/evaluate

**Input:** `ClaimRequest` JSON
**Output:** `DecisionResponse` JSON
**Errors:** 422 (validation), 503 (agent not ready)

```bash
curl -X POST http://localhost:8000/claims/evaluate \
  -H "Content-Type: application/json" \
  -d @data/claims/claim_001.json
```

### GET /health

```json
{"status": "healthy", "model": "Qwen/Qwen2.5-1.5B-Instruct"}
```

### GET /claims/samples

Lists all available sample claims with metadata.

---

## 12. Configuration

All in `config.py` / `.env`:

| Variable | Default | What it controls |
|----------|---------|-----------------|
| MODEL_NAME | Qwen/Qwen2.5-1.5B-Instruct | LLM for tool planning + reasoning |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | RAG embeddings |
| MAX_NEW_TOKENS | 120 | LLM output length limit |
| TEMPERATURE | 0.0 | Greedy decoding (deterministic) |
| MAX_RETRIES | 2 | Not currently used (no retry loop) |
| TOP_K_CHUNKS | 2 | Policy chunks retrieved per query |
| CHUNK_SIZE | 512 | Max chars per policy chunk |
| CHUNK_OVERLAP | 50 | Overlap between adjacent chunks |
| POLICY_PATH | data/travel_policy.md | Path to policy document |
| HOST | 0.0.0.0 | Server bind address |
| PORT | 8000 | Server port |

---

## 13. Project Structure

```
AI-Agent/
├── main.py                      # FastAPI server (async, ThreadPoolExecutor)
├── streamlit_app.py             # Demo UI with manual review buttons
├── config.py                    # Centralized settings (pydantic-settings + .env)
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── .gitignore
├── README.md                    # Quick-start guide
├── DOCUMENTATION.md             # This file
│
├── agent/                       # LangGraph workflow
│   ├── __init__.py              # Exports ReimbursementAgent
│   ├── graph.py                 # StateGraph with 4 nodes + decision logic
│   ├── prompts.py               # REASONING_PROMPT template
│   └── state.py                 # AgentState TypedDict definition
│
├── llm/                         # LLM abstraction
│   ├── __init__.py              # Exports ModelLoader, InferenceEngine
│   ├── loader.py                # Singleton model/tokenizer loading
│   └── inference.py             # generate(prompt) → str interface
│
├── rag/                         # RAG pipeline
│   ├── __init__.py              # Exports EmbeddingModel, FAISSStore, PolicyRetriever
│   ├── embeddings.py            # all-MiniLM-L6-v2 singleton wrapper
│   ├── vector_store.py          # FAISS IndexFlatL2 build + search
│   └── retriever.py             # Policy chunking + retrieval
│
├── tools/                       # Deterministic business tools
│   ├── __init__.py              # Exports ReceiptValidationTool, ExpenseLimitChecker
│   ├── receipt_validator.py     # Receipt presence + amount match checking
│   └── expense_limit_checker.py # Category limit enforcement
│
├── models/                      # Pydantic schemas
│   ├── __init__.py              # Exports all models
│   ├── claim.py                 # ClaimRequest, ExpenseItem, ReceiptInfo, etc.
│   └── decision.py              # DecisionResult, AuditTrail, DecisionResponse
│
├── data/                        # Static data
│   ├── travel_policy.md         # Company travel policy (RAG source)
│   ├── limits.json              # Expense category limits
│   └── claims/                  # 5 sample claims
│       ├── claim_001.json       # → approve
│       ├── claim_002.json       # → partially_approve
│       ├── claim_003.json       # → reject
│       ├── claim_004.json       # → manual_review
│       └── claim_005.json       # → approve (international)
│
├── sample_outputs/              # Pre-generated decision examples
│   ├── CLM-2025-001_decision.json
│   ├── CLM-2025-002_decision.json
│   ├── CLM-2025-003_decision.json
│   ├── CLM-2025-004_decision.json
│   └── CLM-2025-005_decision.json
│
└── tests/
    └── test_tools.py            # Unit tests for both tools (19 assertions)
```

---

## 14. Setup & Running

### Prerequisites
- Python 3.9+
- ~6 GB RAM (CPU mode with 1.5B model)
- ~3.5 GB disk (model + embeddings + deps)

### Install
```bash
git clone https://github.com/Ameyghadge/Travel-Reimbursement-Approval-Agent.git
cd Travel-Reimbursement-Approval-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run Streamlit (recommended for demo)
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Run FastAPI
```bash
python main.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### First run
Models download from HuggingFace automatically:
- Qwen2.5-1.5B-Instruct: ~3 GB
- all-MiniLM-L6-v2: ~90 MB
- Cached in `~/.cache/huggingface/` for future runs

---

## 15. Performance

| Metric | Value |
|--------|-------|
| Startup (models cached) | ~20 seconds |
| Claim evaluation | ~15-20 seconds (CPU) |
| Tool execution | <10ms |
| RAG retrieval | ~50ms |
| LLM inference (2 calls) | ~15 seconds total |
| Memory usage | ~6 GB |

### Why 15-20 seconds?
Two LLM calls on CPU (tool planning ~3s + reasoning ~12s). With GPU, this drops to ~3-4s total.

---

## 16. Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| 2 tools only | Assignment says "at least 2". Kept focused on highest value checks. |
| LLM selects tools | Satisfies "LLM decides when to use tools" requirement |
| Decision from tool outputs | Small LLMs hallucinate decision types. Tools are deterministic. |
| LLM generates reasoning | Natural language explanation is AI-generated (GenAI value) |
| RAG not full policy | Only Top-2 chunks → smaller prompt → faster inference |
| Greedy decoding | Deterministic, fastest possible inference |
| Manual review buttons | Satisfies "route uncertain cases to Manual Review" |
| LangGraph | Clear, inspectable workflow. Easy to explain in interview. |
| No retry loop | Decision is tool-derived, can't fail. Simpler flow. |
| Pydantic everywhere | Type safety at all boundaries, clear schemas |

---

## 17. Assumptions & Limitations

- **Mock data only** — no real employee or company data
- **Single currency** (USD) — no exchange rate handling
- **Receipts are metadata** — `has_receipt: true/false`, no actual OCR
- **No persistent storage** — each server restart is fresh
- **No authentication** — API is open (prototype scope)
- **Expense limits per-item** — not aggregated daily totals
- **No duplicate detection** — removed for simplicity (was overcomplicated)
- **~15-20s latency** — bound by CPU inference of 1.5B model

---

## 18. What I'd Improve Next

1. **GPU inference** — drops latency to 3-4 seconds
2. **Receipt OCR** — validate actual receipt images/PDFs
3. **Persistent history** — enable real duplicate detection
4. **Larger model** (Qwen 3B/7B) — better reasoning quality when hardware allows
5. **Evaluation harness** — automated scoring against labeled test set
6. **Multi-currency** — exchange rate conversion for international claims
7. **Approval workflow** — actually route manual_review to approvers via email/Slack
