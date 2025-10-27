## Technical Implementation

This backend implements a **Retrieval-Augmented Generation (RAG)** chatbot for Hong Kong elderly-care policy documents.
 It is built with **FastAPI**, **FAISS**, and **BM25**, and integrates a remote **LLM service (SmartCare API)** for natural-language responses.

### ① System Architecture

**Main workflow:**

1. **Document ingestion (`app/rag.py`)**
    Markdown or bilingual policy documents are parsed and chunked (sentence-aware for Chinese).
    Each chunk is embedded using a **SentenceTransformer model**, and stored in a FAISS vector index with BM25 tokens for hybrid retrieval.
2. **Hybrid retrieval (`hybrid_retrieve`)**
    Combines **semantic similarity (FAISS)** and **lexical relevance (BM25)** with adaptive weighting.
   - English queries emphasize FAISS (semantic).
   - Chinese or mixed queries emphasize BM25 (token overlap).
   - Page-level penalties (from feedback) are applied dynamically.
3. **Chat interaction (`/chat`)**
    User queries are rewritten into context-complete standalone questions via SmartCare LLM.
    The backend retrieves relevant document chunks, builds a structured prompt, and sends it to the LLM for final generation.
    Inline citations (e.g., `[Source 1]`) are attached to each answer.
4. **Feedback and adaptation**
    User feedback (thumb-up/down) updates per-page counters and penalties, influencing subsequent retrieval.
    Thresholds (e.g., `MIN_VEC_SIM`) are auto-tuned daily via `scripts/tune_thresholds.py`.

### ② Major Components

| Module                      | Key Responsibilities                                         |
| --------------------------- | ------------------------------------------------------------ |
| **`app/main.py`**           | Core API endpoints (`/chat`, `/feedback`, `/metrics`, etc.); request handling, clarifying questions, feedback logging |
| **`app/rag.py`**            | Document ingestion, FAISS index, BM25 retrieval, hybrid scoring, penalty integration |
| **`app/admin_docs.py`**     | Upload/delete/list documents; rebuild index on change        |
| **`app/ingest_manager.py`** | Background process control for re-indexing                   |
| **`app/llm_client.py`**     | SmartCare LLM interface (stream & non-stream modes); translation & query rewriting |
| **`app/security.py`**       | Bearer-token authentication and AES-GCM encryption for stored data |
| **`app/utils.py`**          | Markdown parsing and logical page inference                  |
| **`scripts/\*.py`**         | Utility scripts for indexing, penalty building, and analytics |

### ③ Key Implementation Details

- **RAG Fusion**: `score = α·vec_sim + β·bm25_score − penalty`
   where α/β vary with language (`α=0.6–0.9`), and `penalty` is derived from negative feedback.

- **Chunking**:

  - Sentence-aware segmentation for Chinese (`。！？；：`)
  - Overlapping window: 1500 chars + 200 overlap
  - Both English words and Chinese 2/3-grams retained (for bilingual text)

- **Clarification logic**:

  - Ambiguous or too-short queries trigger `_clarify_question_smart()`
  - Suggestions are generated based on entity extraction and retrieval context

- **Security**:

  - Optional AES-GCM encryption for FAISS and metadata files
  - Configurable Bearer authentication
  - CORS enabled for `localhost:5173`

- **Logging & Metrics**:

  - Every chat logged to `data/logs/chat_usage.jsonl`
  - `/metrics/usage` and `/feedback/metrics` expose daily and feedback stats for admin dashboards

- **Instant Feedback Loop**:
   Each feedback immediately updates:

  ```
  data/feedback/
    ├── feedback.jsonl       # raw records
    ├── penalty_counts.json  # per-page up/down
    ├── penalty.json         # derived penalties
  ```

### ④ Models Used

| Purpose                           | Model                                                        | Source                                         |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **Embeddings**                    | `sentence-transformers/all-MiniLM-L6-v2` *(or multilingual: `paraphrase-multilingual-MiniLM-L12-v2`)* | [SentenceTransformers](https://www.sbert.net/) |
| **LLM responses**                 | SmartCare LLM API (`https://smartlab.cse.ust.hk/smartcare/dev/llm_chat/`) | Provided by HKUST SmartCare platform           |
| **Query rewriting & translation** | Same SmartCare LLM endpoint (non-stream mode)                | —                                              |

### ⑤ Recent Improvements

#### Feedback Collection

When a user clicks a feedback button in the frontend, the app sends a structured JSON payload to the /feedback endpoint:

{
  "label": "up",
  "userQuery": "Tell me about allowance.",
  "answer": "...",
  "citations": [
    {"file": "Manual_2024.md", "page": 32},
    {"file": "Annexes.md", "page": 149}
  ]
}


The record is appended to data/feedback/feedback.jsonl for auditing and offline analysis.

#### Aggregation and Penalty Update

The backend maintains two incremental statistics:

penalty_counts.json → tracks up/down counts per page

{"Manual_2024.md::32": {"up":5, "down":1}}


penalty.json → derived penalty weights applied during retrieval
computed as

penalty = min(1.0, BASE + STEP * max(0, down - up))


(default BASE=0.10, STEP=0.05)

Each time feedback is received, these files are updated atomically.

#### Real-Time Effect in Retrieval

During hybrid retrieval (rag.hybrid_retrieve), every chunk’s final score is adjusted:

combo = α * vec_sim + β * bm25_score - penalty


Pages with more 👎 feedback get a higher penalty and drop in ranking.

Pages with 👍 feedback remain unpenalized and surface more often.

The penalty table is re-loaded lazily, so new feedback takes effect immediately — no service restart needed.

#### Global Performance Adaptation

The /feedback/metrics and /metrics/usage endpoints expose weekly statistics such as up-rate and no-hit ratio.
A maintenance script scripts/tune_thresholds.py adjusts retrieval thresholds automatically:

If up-rate < 0.7 → increase MIN_VEC_SIM / MIN_BM25_SCORE (stricter filtering)

If up-rate > 0.8 → relax thresholds (broader recall)

This closes the feedback → metric → threshold optimization loop.

### ⑥ Recent Improvements

- Fixed **English-query retrieval for bilingual documents** by switching BM25 tokenization from *mutually exclusive* (CJK-only) to *union-based* (retain both English and Chinese tokens).
   → Ensures that mixed English-Chinese content can be retrieved from either language query.
- Enhanced async file upload with hot-reload index rebuild.
- Added per-page penalty updates for real-time feedback learning.
