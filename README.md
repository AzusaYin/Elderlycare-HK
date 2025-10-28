# ElderlyCare HK — Backend (FastAPI + RAG)

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/docs data/index
# Put your Markdown files under data/docs/
```

Create a `.env` (optional):
```
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# RAG
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
TOP_K=5
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# SmartCare
SMARTCARE_BASE_URL=https://smartlab.cse.ust.hk/smartcare/dev/llm_chat/
TEMPERATURE=0.0
MAX_TOKENS=1024
ENABLE_BM25=true
```

## 2) Build the index
```bash
python -m scripts.ingest
```
This creates FAISS index + metadata under `data/index/`.

## 3) Run the server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## 4) Call the API

> If the Server and the Client are not on the same machine, then it is required ```ip a | grep 'inet '``` in bash to get the IP address of server.

### Health
```bash
curl http://localhost:8001/healthz
```

### Chat (non-stream)
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"What about Eligibility for Operating Subvented Welfare?"}
    ],
    "stream": false
  }' -w '\n'
```

curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer K3SYw0UNdctORRP2gpSaPcRMLgrXgL_q7tS7MT5PirQ" \
  -d '{
    "messages": [
      {"role": "user", "content": "What about Eligibility for Operating Subvented Welfare?"}
    ],
    "stream": false
  }' -w '\n'

### Stream (raw lines)
```bash
curl -N -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"Eligibility for Old Age Living Allowance?"}
    ],
    "stream": true
  }' -w '\n'
```

hexdump -C -n 64 data/index/meta.json
hexdump -C -n 64 data/index/bm25.json
hexdump -C -n 32 data/index/faiss.index

python - <<'PY'
from app.security import decrypt_bytes
p = "data/index/meta.json"
print(decrypt_bytes(open(p,"rb").read())[:200])
PY

python - <<'PY'
from app.security import decrypt_bytes
p = "data/index/bm25.json"
print(decrypt_bytes(open(p,"rb").read())[:200])
PY

## Notes
- You should better run ```scripts/tune_thresholds.py``` once per day to improve the performance of chat robot.
- Citations list the file name and page if detectable from markdown. If your MD lacks page markers, the backend still cites the file.
- For better page/snippet support, extend `infer_page_map` to parse your MD structure (e.g., headings) and store character spans for snippets.
- To switch to GPU embeddings, set `EMBEDDING_DEVICE=cuda` and ensure CUDA is available.
- You can later add caching, rate limiting, and feedback logging.