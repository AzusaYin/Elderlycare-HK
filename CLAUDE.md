# CLAUDE.md - AI Assistant Guide for ElderlyCare HK

## Project Overview

**ElderlyCare HK** is a bilingual (English/Traditional Chinese) RAG-powered chatbot for Hong Kong elderly care policy documents. It combines semantic search (FAISS), lexical search (BM25), and an LLM service to provide accurate, source-cited answers to policy queries.

### Tech Stack

**Backend (Python)**
- FastAPI for REST API
- FAISS (vector search) + BM25 (lexical search) for hybrid retrieval
- sentence-transformers for embeddings
- cryptography for data encryption
- httpx for async HTTP calls to SmartCare LLM API

**Frontend (React + TypeScript)**
- React 19 with TypeScript
- Vite for build tooling
- Tailwind CSS 4 for styling
- react-markdown for rendering responses
- LocalStorage for chat history management

**Infrastructure**
- Python 3.x with virtual environment
- Node.js/npm for frontend tooling
- HKUST SmartCare LLM API (external service)

---

## Repository Structure

```
Elderlycare-HK/
├── app/                          # FastAPI backend application
│   ├── main.py                   # Core API endpoints, request handling, feedback
│   ├── rag.py                    # RAG engine: indexing, FAISS, BM25, hybrid retrieval
│   ├── admin_docs.py             # Admin routes: upload/delete/list documents
│   ├── ingest_manager.py         # Background process control for re-indexing
│   ├── llm_client.py             # SmartCare LLM interface (stream/non-stream)
│   ├── security.py               # Bearer auth + AES-GCM encryption
│   ├── settings.py               # Pydantic settings (loads from .env)
│   ├── schemas.py                # Pydantic models for API requests/responses
│   └── utils.py                  # Markdown parsing, page inference
│
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── App.tsx               # Main app component (chat UI, threads, i18n)
│   │   ├── AdminDocs.tsx         # Admin panel for document management
│   │   ├── main.tsx              # React entry point
│   │   └── assets/               # Static assets (images, etc.)
│   ├── public/                   # Public static files
│   ├── package.json              # Frontend dependencies
│   ├── vite.config.ts            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── tsconfig.json             # TypeScript config
│
├── scripts/                      # Utility scripts
│   ├── ingest.py                 # Manual index building script
│   ├── tune_thresholds.py        # Auto-tune retrieval thresholds based on feedback
│   ├── build_penalty.py          # Build penalty map from feedback
│   ├── analyze_usage.py          # Analyze chat usage logs
│   └── summarize_usage.py        # Generate usage summary stats
│
├── data/                         # Data directory (gitignored except structure)
│   ├── docs/                     # Markdown policy documents (source)
│   ├── all_docs/                 # All available documents
│   ├── index/                    # FAISS index, metadata, BM25 tokens (encrypted)
│   ├── feedback/                 # User feedback logs, penalty counts
│   └── logs/                     # Chat usage logs (JSONL)
│
├── .env                          # Environment variables (SECRET - not in git)
├── .gitignore                    # Git ignore patterns
├── requirements.txt              # Python dependencies
├── package.json                  # Root package.json (Tailwind deps)
├── README.md                     # User-facing setup guide
├── Technical Implementation.md   # Detailed technical documentation
├── PROJECT_FRAMEWORK.md          # Auto-generated file tree
└── CLAUDE.md                     # This file (AI assistant guide)
```

---

## Key Architectural Concepts

### 1. Hybrid RAG Retrieval

The system uses **dual-mode retrieval** combining:
- **FAISS (vector similarity)**: Semantic matching via embeddings
- **BM25 (lexical matching)**: Token-based keyword matching

**Scoring formula**:
```python
combo = α * vec_sim + β * bm25_score - penalty
```

**Adaptive weighting**:
- English queries: higher α (0.7-0.9), emphasize semantic
- Chinese queries: higher β (0.5-0.7), emphasize lexical
- Detected automatically based on CJK character presence

**Penalties**: Per-page penalties derived from user feedback (👎 > 👍) reduce problematic page rankings.

### 2. Document Ingestion Flow

1. Markdown files placed in `data/docs/`
2. Run `python -m scripts.ingest` or use admin API `/docs/upload`
3. Documents are chunked (1500 chars, 200 overlap, sentence-aware for Chinese)
4. Each chunk embedded with sentence-transformers
5. FAISS index + BM25 tokens + metadata saved to `data/index/` (encrypted if enabled)
6. Hot-reload: In-memory index cleared, next request loads new index

### 3. Query Processing Pipeline

```
User query → Query rewriting (resolve pronouns) → Hybrid retrieval →
Threshold gating (min_vec_sim, min_bm25_score) → Prompt building →
SmartCare LLM → Response with inline citations → Frontend render
```

**Clarification logic**: If query is ambiguous/short/lacks context, backend suggests clarifications before retrieval.

### 4. Feedback Loop

- User clicks 👍/👎 in UI → POST `/feedback`
- Backend updates `data/feedback/penalty_counts.json` (per-page counters)
- Penalty computed: `penalty = min(1.0, BASE + STEP * max(0, down - up))`
- Penalty cached in `data/feedback/penalty.json`
- Next retrieval uses updated penalties (no restart needed)
- Daily: `scripts/tune_thresholds.py` adjusts global thresholds based on metrics

---

## Development Workflows

### Backend Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file (see Environment Variables section)

# 4. Prepare data directories
mkdir -p data/docs data/index data/feedback data/logs

# 5. Add documents to data/docs/
# (Copy .md files or use admin API)

# 6. Build index
python -m scripts.ingest

# 7. Run backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Frontend Setup

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Create .env file
# VITE_BACKEND_URL=http://localhost:8001
# VITE_API_TOKEN=<your_bearer_token>

# 3. Run dev server
npm run dev

# 4. Build for production
npm run build

# 5. Preview production build
npm run preview
```

### Running Both Simultaneously

**Terminal 1** (Backend):
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

**Terminal 2** (Frontend):
```bash
cd frontend
npm run dev
```

Frontend runs on `http://localhost:5173` by default.

---

## Environment Variables

### Backend (.env in root)

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Security
API_BEARER_TOKEN=<random_secure_token>
REQUIRE_AUTH=true
ENCRYPT_DATA=true
ENCRYPTION_KEY_B64=<base64_encoded_32_byte_key>
ALLOWED_ORIGINS=["http://localhost:5173"]

# RAG
CHUNK_SIZE=1800
CHUNK_OVERLAP=250
TOP_K=8
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # or "cuda" for GPU

# SmartCare LLM
SMARTCARE_BASE_URL=https://smartlab.cse.ust.hk/smartcare/dev/llm_chat/
TEMPERATURE=0.0
MAX_TOKENS=1024
ENABLE_BM25=true
ENABLE_QUERY_REWRITE=true

# Retrieval Thresholds (auto-tuned)
MIN_VEC_SIM=0.35
MIN_BM25_SCORE=3.5
MIN_SOURCES_REQUIRED=1
```

**To generate encryption key**:
```python
from cryptography.fernet import Fernet
import base64
key = Fernet.generate_key()
print(base64.b64encode(key).decode())
```

### Frontend (.env in frontend/)

```bash
VITE_BACKEND_URL=http://localhost:8001
VITE_API_TOKEN=<same_as_API_BEARER_TOKEN>
```

---

## Code Conventions and Patterns

### Backend (Python)

**File Organization**:
- `app/main.py`: Route handlers, middleware, application lifecycle
- `app/*.py`: Domain modules (single responsibility)
- `scripts/*.py`: Standalone utilities (runnable with `python -m scripts.<name>`)

**Key Patterns**:
1. **Pydantic everywhere**: Settings, request/response models
2. **Async by default**: All API routes and LLM calls use async/await
3. **Atomic file writes**: Use temp file + replace for JSON updates
4. **Lazy loading**: Embedder and Index cached in module globals, loaded on first request
5. **Encryption abstraction**: `encrypt_bytes()/decrypt_bytes()` check settings internally

**Error Handling**:
- Use FastAPI's `HTTPException` for API errors
- Catch broad exceptions in background tasks, log to status files
- Return safe defaults (empty lists, None) rather than crashing

**Naming**:
- Private functions: `_prefix` (e.g., `_safe_read_json`)
- Endpoints: lowercase with underscores (e.g., `/docs/upload`)
- Settings: `snake_case` matching env var names

### Frontend (TypeScript/React)

**Component Structure**:
- Single-file component (`App.tsx`) for main app
- Separate `AdminDocs.tsx` for admin panel
- Minimal abstraction (avoid premature component splitting)

**State Management**:
- `useState` for local UI state
- `localStorage` for chat threads (JSON serialized)
- No external state library (Redux, Zustand, etc.)

**Styling**:
- Tailwind CSS utility classes (v4)
- Fallback inline styles for non-Tailwind environments
- High-contrast mode toggle via className

**API Calls**:
- Direct `fetch()` calls (no axios/swr)
- Stream handling with `ReadableStream` for chat responses
- Bearer token from env var

**Naming**:
- Components: PascalCase (e.g., `AdminDocs`)
- Functions: camelCase (e.g., `handleSendMessage`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_BACKEND`)

---

## Testing Strategy

**Backend** (Currently minimal):
- Manual testing via `curl` commands (see README.md)
- Verify endpoints: `/healthz`, `/chat`, `/docs/list`, etc.
- Test streaming with `curl -N`

**Frontend**:
- Manual browser testing
- eslint for linting: `npm run lint`
- TypeScript type checking: `tsc -b`

**Integration Testing**:
1. Start backend with sample documents
2. Start frontend
3. Test chat flow: ask question → verify citations → give feedback
4. Test admin flow: upload doc → verify reindex → test retrieval

**Future Recommendations**:
- Add pytest for backend unit tests (RAG logic, chunking, scoring)
- Add Vitest/React Testing Library for frontend
- Add E2E tests with Playwright

---

## Common Tasks and How-To

### Add/Update Documents

**Via filesystem**:
```bash
# 1. Copy .md files to data/docs/
cp new_policy.md data/docs/

# 2. Rebuild index
python -m scripts.ingest
```

**Via admin API** (hot-reload):
```bash
curl -X POST http://localhost:8001/docs/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@new_policy.md"
```

### Delete Documents

```bash
curl -X DELETE http://localhost:8001/docs/filename.md \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### View Metrics

```bash
# Usage stats
curl http://localhost:8001/metrics/usage

# Feedback stats
curl http://localhost:8001/feedback/metrics
```

### Tune Retrieval Thresholds

```bash
# Run daily or after significant feedback accumulation
python -m scripts.tune_thresholds
```

This adjusts `MIN_VEC_SIM`, `MIN_BM25_SCORE` in settings based on up-rate and no-hit-rate.

### Inspect Encrypted Index

```bash
# Decrypt and preview
python - <<'PY'
from app.security import decrypt_bytes
p = "data/index/meta.json"
print(decrypt_bytes(open(p,"rb").read())[:200])
PY
```

### Debug Retrieval

Enable debug prints in `app/main.py`:
```python
DEBUG_CLARIFY = True  # Already present
```
Check console output for:
- Query rewriting results
- Retrieved chunk scores
- Threshold gating decisions

### Add New API Endpoint

1. Edit `app/main.py` or create new router in `app/`
2. Add Pydantic models to `app/schemas.py`
3. Add route handler with `@app.get()`/`@app.post()`
4. Use `Depends(require_bearer)` if auth required
5. Test with `curl`
6. Update frontend API calls if needed

### Modify Chunking Logic

Edit `app/rag.py`:
- `simple_char_chunk()`: Character-based chunking
- `smart_sent_chunk_cn()`: Sentence-aware Chinese chunking
- Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP` in settings

Then rebuild index: `python -m scripts.ingest`

### Switch Embedding Model

1. Update `EMBEDDING_MODEL_NAME` in `.env`
2. Rebuild index: `python -m scripts.ingest`
3. Restart backend (will load new model on first request)

Popular alternatives:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (better multilingual)
- `sentence-transformers/all-mpnet-base-v2` (better English)

### Enable GPU Acceleration

1. Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
2. Set `EMBEDDING_DEVICE=cuda` in `.env`
3. Optional: Use `faiss-gpu` instead of `faiss-cpu`

---

## Important Security Considerations

### Secrets Management

**NEVER commit to git**:
- `.env` files (both root and frontend)
- `data/index/**` (encrypted data, but keys are in .env)
- `data/feedback/**` (contains user data)

**Token generation**:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### Authentication

- All admin endpoints (`/docs/*`, `/feedback/metrics`) require `Bearer` token
- Token configured via `API_BEARER_TOKEN` env var
- Frontend sends token in `Authorization: Bearer <token>` header
- Disable auth: Set `REQUIRE_AUTH=false` (NOT recommended for production)

### CORS Configuration

- Default allows `http://localhost:5173` (frontend dev server)
- For production, update `ALLOWED_ORIGINS` in settings to match your domain
- Example: `["https://elderlycare.example.com"]`

### Data Encryption

- All index files encrypted with AES-GCM (Fernet)
- Key stored in `ENCRYPTION_KEY_B64` (must be base64-encoded 32-byte key)
- Disable encryption: Set `ENCRYPT_DATA=false` (NOT recommended)

---

## Debugging Tips

### Backend not starting

1. Check virtual environment is activated: `which python` should show `.venv/bin/python`
2. Check dependencies: `pip install -r requirements.txt`
3. Check port availability: `lsof -i :8001` (kill other processes if needed)
4. Check .env file exists and has valid values

### Frontend not connecting to backend

1. Check `VITE_BACKEND_URL` in `frontend/.env`
2. Check CORS settings in backend `.env` → `ALLOWED_ORIGINS`
3. Open browser DevTools → Network tab → Check request URLs and responses
4. Verify backend is running: `curl http://localhost:8001/healthz`

### No retrieval results

1. Check if index exists: `ls -la data/index/`
2. Rebuild index: `python -m scripts.ingest`
3. Lower thresholds temporarily in `.env`: `MIN_VEC_SIM=0.2`, `MIN_BM25_SCORE=1.0`
4. Check document content: `ls data/docs/` (should have .md files)

### Slow responses

1. Check embedding device: `EMBEDDING_DEVICE=cpu` is slow for large indexes
   - Use GPU if available: `EMBEDDING_DEVICE=cuda`
2. Reduce `TOP_K` in `.env` (fewer retrieved chunks)
3. Check SmartCare LLM API latency (external service)
4. Reduce `MAX_TOKENS` for faster LLM responses

### Encoding errors (Chinese text)

- Ensure all files are UTF-8: `file -i data/docs/*.md`
- Check environment locale: `locale` should show UTF-8
- Python should default to UTF-8 (configured in code with `encoding="utf-8"`)

---

## AI Assistant Guidelines

### When Reading Code

1. **Start with**:
   - `README.md` for user setup
   - `Technical Implementation.md` for architecture
   - This file (`CLAUDE.md`) for conventions
   - `app/settings.py` for configuration schema

2. **Critical files**:
   - `app/main.py`: All API routes and main logic
   - `app/rag.py`: RAG engine (most complex logic)
   - `frontend/src/App.tsx`: Frontend UI and state

3. **Trace requests**:
   - User input → `App.tsx:handleSendMessage`
   - → `POST /chat` → `app/main.py:chat_endpoint`
   - → `hybrid_retrieve()` in `app/rag.py`
   - → SmartCare LLM call in `app/llm_client.py`
   - → Response rendered in `App.tsx` with ReactMarkdown

### When Modifying Code

**DO**:
- Read existing code first (use Read tool, not assumptions)
- Follow existing patterns (async, Pydantic, atomic writes)
- Test changes with curl/browser before committing
- Update documentation if changing APIs or workflows
- Keep changes minimal and focused (avoid over-engineering)
- Use existing error handling patterns
- Preserve bilingual support (English + Traditional Chinese)

**DON'T**:
- Add features not requested
- Refactor unrelated code
- Add dependencies without discussing first
- Break backwards compatibility without migration plan
- Commit secrets or sensitive data
- Use sync code where async is expected
- Add type: ignore or similar suppressions without good reason

### When Adding Features

1. **Understand the request**: Ask clarifying questions if ambiguous
2. **Plan the change**:
   - Backend: Which module? New endpoint or modify existing?
   - Frontend: New component or extend existing?
   - Database: Need new fields in metadata?
3. **Check dependencies**: Will this break existing functionality?
4. **Implement incrementally**: Backend → Test → Frontend → Test
5. **Update docs**: Add to this file if it's a new pattern/workflow

### Common Pitfalls

- **Forgetting to rebuild index** after changing chunking/embedding logic
- **Not invalidating cache** (`_index = None`) after document changes
- **Using sync file I/O** in async route handlers (use `asyncio.to_thread()`)
- **Hardcoding values** instead of using settings
- **Not handling stream cleanup** (abort controller in frontend)
- **Mixing Chinese/English** in user-facing strings (use i18n pattern)

---

## Performance Optimization

### Backend

1. **Use GPU for embeddings**: `EMBEDDING_DEVICE=cuda` (10-50x faster)
2. **Reduce TOP_K**: Fewer chunks to process and send to LLM
3. **Cache embeddings**: For frequently asked queries (not implemented yet)
4. **Preload models**: On startup instead of first request
5. **Use faiss-gpu**: For very large indexes (>100k chunks)

### Frontend

1. **Lazy load admin panel**: Suspense boundary around AdminDocs
2. **Virtualize chat history**: For threads with 100+ messages
3. **Debounce typing indicators**: Avoid re-renders on every keystroke
4. **Optimize markdown rendering**: Use react-markdown's memo components

### Retrieval

1. **Tune thresholds**: Run `scripts/tune_thresholds.py` regularly
2. **Prune bad pages**: Review penalty.json, remove consistently poor sources
3. **Optimize chunk size**: Smaller chunks (1000) = more precise but more API calls
4. **Hybrid weight tuning**: Adjust α/β based on language distribution

---

## Deployment Checklist

### Backend

- [ ] Set `DEBUG=false` in production .env
- [ ] Use strong random `API_BEARER_TOKEN`
- [ ] Generate new `ENCRYPTION_KEY_B64`
- [ ] Update `ALLOWED_ORIGINS` to production domain
- [ ] Set `REQUIRE_AUTH=true`
- [ ] Use reverse proxy (nginx) with SSL/TLS
- [ ] Configure firewall (only expose 443/80, not 8001)
- [ ] Set up systemd service for auto-restart
- [ ] Configure log rotation for `data/logs/`
- [ ] Schedule `tune_thresholds.py` daily (cron)
- [ ] Backup `data/index/` and `data/feedback/` regularly
- [ ] Monitor disk usage (`data/logs/` can grow large)

### Frontend

- [ ] Build production bundle: `npm run build`
- [ ] Update `VITE_BACKEND_URL` to production API
- [ ] Serve from CDN or nginx (not `npm run dev`)
- [ ] Enable gzip/brotli compression
- [ ] Set up CSP headers
- [ ] Configure cache headers for assets
- [ ] Test on multiple browsers/devices
- [ ] Verify HTTPS and SSL certificate

---

## Git Workflow

### Branch Strategy

- `main`: Stable production branch
- `claude/claude-md-*`: AI assistant working branches (auto-generated)
- Feature branches: `feature/<name>` (if working on major features)

### Commit Conventions

**Good commits**:
- `fix: resolve BM25 scoring for bilingual documents`
- `feat: add document upload progress indicator`
- `refactor: extract query rewriting to separate function`
- `docs: update CLAUDE.md with deployment checklist`

**Bad commits**:
- `update` (too vague)
- `fix bug` (which bug?)
- `wip` (work in progress, should not be pushed)

### Before Committing

1. **Test locally**: Both backend and frontend
2. **Run linter**: `npm run lint` (frontend)
3. **Check types**: `tsc -b` (frontend)
4. **Review changes**: `git diff` (look for debug prints, secrets)
5. **Verify .env not staged**: `git status` should not show .env

### Pushing Changes

```bash
# Always push to claude/ branches with matching session ID
git push -u origin claude/claude-md-<session-id>
```

**Retry on network errors**: Up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

---

## Useful Commands Reference

### Backend

```bash
# Activate venv
source .venv/bin/activate

# Run server (dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Run server (production)
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4

# Rebuild index
python -m scripts.ingest

# Tune thresholds
python -m scripts.tune_thresholds

# Analyze usage
python -m scripts.analyze_usage
python -m scripts.summarize_usage

# Check index status
curl http://localhost:8001/docs/status -H "Authorization: Bearer $TOKEN"

# Health check
curl http://localhost:8001/healthz
```

### Frontend

```bash
# Install deps
npm install

# Dev server
npm run dev

# Build
npm run build

# Preview build
npm run preview

# Lint
npm run lint

# Type check
npx tsc -b
```

### Git

```bash
# Check status
git status

# Stage changes
git add <file>

# Commit
git commit -m "type: description"

# Push to claude branch
git push -u origin claude/claude-md-<session-id>

# View recent commits
git log --oneline -10

# View diff
git diff
git diff --staged
```

---

## Troubleshooting Matrix

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| 401 Unauthorized | Missing/wrong Bearer token | Check `API_BEARER_TOKEN` matches in backend .env and frontend `VITE_API_TOKEN` |
| CORS error in browser | Origin not allowed | Add frontend URL to `ALLOWED_ORIGINS` in backend .env |
| No search results | Empty index or high thresholds | Run `python -m scripts.ingest`, lower `MIN_VEC_SIM` |
| Slow embedding | Using CPU | Set `EMBEDDING_DEVICE=cuda` (if GPU available) |
| Encryption error | Wrong key or missing key | Regenerate `ENCRYPTION_KEY_B64`, rebuild index |
| Chinese text garbled | Encoding issue | Ensure files are UTF-8: `file -i data/docs/*.md` |
| Stream disconnects | Timeout too short | Increase httpx timeout in `llm_client.py` |
| Frontend build fails | Missing deps or type errors | Run `npm install`, check `npx tsc -b` |
| Backend won't start | Port in use | Kill process on 8001: `lsof -i :8001` then `kill -9 <PID>` |
| Feedback not working | Penalty file locked | Delete `data/feedback/*.json.part` temp files |

---

## Further Reading

- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **BM25 Overview**: https://en.wikipedia.org/wiki/Okapi_BM25
- **Sentence Transformers**: https://www.sbert.net/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React 19 Docs**: https://react.dev/
- **Vite Guide**: https://vite.dev/guide/
- **Pydantic Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

---

## Contact and Contribution

For questions, issues, or contributions:
1. Open an issue on GitHub (if repository is public)
2. Review `Technical Implementation.md` for architecture details
3. Follow the conventions in this guide
4. Test thoroughly before submitting changes

**Remember**: This is a production system serving elderly care policy information. Accuracy and reliability are critical. When in doubt, ask before changing core retrieval or LLM logic.

---

**Last Updated**: 2025-12-07
**Version**: 1.0
**Maintained by**: AI Assistants working on ElderlyCare HK
