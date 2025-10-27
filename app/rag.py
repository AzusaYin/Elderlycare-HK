import json
import threading
import faiss
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .security import encrypt_bytes, decrypt_bytes
from .utils import read_markdown_files, infer_page_map

DOCS_DIR = Path("data/docs")

@dataclass
class Chunk:
    text: str
    meta: Dict

class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))

class Index:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss = None
        self.meta: List[Dict] = []
        self.bm25 = None
        self.bm25_corpus_tokens: List[List[str]] = []

    def save(self):
        # faiss index
        if self.faiss is not None:
            raw_path = self.index_dir / "faiss.index"
            faiss.write_index(self.faiss, str(raw_path))
            # 用加密覆盖写回；encrypt_bytes 内部会根据 settings.encrypt_data 决定是否加密/报错
            data = raw_path.read_bytes()
            raw_path.write_bytes(encrypt_bytes(data))

        # meta.json
        meta_path = self.index_dir / "meta.json"
        meta_bytes = json.dumps(self.meta, ensure_ascii=False).encode("utf-8")
        meta_path.write_bytes(encrypt_bytes(meta_bytes))

        # bm25.json
        if self.bm25 is not None:
            bm25_path = self.index_dir / "bm25.json"
            bm = {"tokens": self.bm25_corpus_tokens}
            bm_bytes = json.dumps(bm, ensure_ascii=False).encode("utf-8")
            bm25_path.write_bytes(encrypt_bytes(bm_bytes))

    def load(self):
        # faiss
        faiss_path = self.index_dir / "faiss.index"
        meta_path = self.index_dir / "meta.json"
        if faiss_path.exists() and meta_path.exists():
            blob = decrypt_bytes(faiss_path.read_bytes())
            tmp = self.index_dir / ".faiss.tmp"
            tmp.write_bytes(blob)
            self.faiss = faiss.read_index(str(tmp))
            tmp.unlink(missing_ok=True)

            mb = decrypt_bytes(meta_path.read_bytes())
            self.meta = json.loads(mb.decode("utf-8"))

        # bm25
        bm25_path = self.index_dir / "bm25.json"
        if bm25_path.exists():
            bb = decrypt_bytes(bm25_path.read_bytes())
            data = json.loads(bb.decode("utf-8"))
            self.bm25_corpus_tokens = data.get("tokens", [])
            if self.bm25_corpus_tokens:
                self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

                
    def build(self, embeddings: np.ndarray, meta: List[Dict], bm25_tokens: Optional[List[List[str]]] = None):
        dim = embeddings.shape[1] if embeddings.size else 384
        index = faiss.IndexFlatIP(dim)
        if embeddings.size:
            index.add(embeddings.astype(np.float32))
        self.faiss = index
        self.meta = meta
        if bm25_tokens:
            self.bm25_corpus_tokens = bm25_tokens
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.faiss is None or self.faiss.ntotal == 0:
            return []
        D, I = self.faiss.search(query_emb.astype(np.float32), k)
        return list(zip(I[0].tolist(), D[0].tolist()))

# --- Chunking (safe & fast) ---
def simple_char_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    step = max(1, chunk_size - overlap)
    n = len(text)
    if n == 0:
        return []
    return [text[i:i+chunk_size] for i in range(0, n, step)]

# --- Sentence-aware chunking for CJK ---
_SENT_SPLIT = re.compile(r"[。！？；：]\s*")  # 粗粒度分句

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # 先按中文标点粗分句，再在每个句段内二次裁切
    parts = []
    segs = [s for s in _SENT_SPLIT.split(text) if s]
    for seg in segs:
        parts.extend(simple_char_chunk(seg, chunk_size, overlap))
    if not segs:  # 没分出来就退回原策略
        parts = simple_char_chunk(text, chunk_size, overlap)
    return parts

# --- Tokenization (CJK-friendly) ---

_CJK = re.compile(r"[\u4e00-\u9fff]")

def _to_halfwidth(s: str) -> str:
    # 全角转半角（常见中文数字/标点）
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)

def tokenize(text: str) -> list[str]:
    text = _to_halfwidth(text)
    tokens: list[str] = []

    # 1) 英文/数字词，始终保留
    tokens += re.findall(r"[A-Za-z0-9_]+", text.lower())

    # 2) CJK 2/3-gram，只对 CJK 段落追加
    if _CJK.search(text):
        # 只取 CJK 字符做 n-gram，避免把英文一起碾碎
        cjk_only = "".join(ch for ch in re.sub(r"\s+", "", text) if _CJK.match(ch))
        if cjk_only:
            if len(cjk_only) >= 2:
                tokens += [cjk_only[i:i+2] for i in range(len(cjk_only)-1)]
            if len(cjk_only) >= 3:
                tokens += [cjk_only[i:i+3] for i in range(len(cjk_only)-2)]
    return tokens


# --- Ingestion ---
def ingest_corpus(docs_dir: str, index_dir: str) -> Tuple[int, int]:
    docs = read_markdown_files(docs_dir)
    print(f"[ingest] loaded {len(docs)} doc(s)")
    embedder = Embedder(settings.embedding_model_name, settings.embedding_device)

    all_chunks: List[Chunk] = []
    for di, doc in enumerate(docs, 1):
        page_ranges = infer_page_map(doc["text"])  # best-effort
        print(f"[ingest] doc {di}/{len(docs)} -> {len(page_ranges)} page(s) (logical)")
        for pr in page_ranges:
            page_text = doc["text"][pr["start"]:pr["end"]]
            pieces = chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
            for i, piece in enumerate(pieces):
                meta = {"file": doc["path"], "page": pr["page"], "chunk_id": i, "text": piece}
                all_chunks.append(Chunk(text=piece, meta=meta))
    print(f"[ingest] total chunks: {len(all_chunks)}; start embedding...")

    if not all_chunks:
        index = Index(index_dir)
        index.build(np.zeros((0, 384), dtype=np.float32), [], None)
        index.save()
        return len(docs), 0

    texts = [c.text for c in all_chunks]

    # batch embedding
    BATCH = 512
    emb_list = []
    for i in range(0, len(texts), BATCH):
        emb_list.append(embedder.encode(texts[i:i+BATCH]))
    embeddings = np.vstack(emb_list).astype(np.float32)
    print(f"[ingest] embedding done, shape={embeddings.shape}")

    bm25_tokens = [tokenize(t) for t in texts] if settings.enable_bm25 else None

    meta = [c.meta for c in all_chunks]
    index = Index(index_dir)
    index.build(embeddings, meta, bm25_tokens)
    index.save()
    print("[ingest] index saved")

    return len(docs), len(all_chunks)

def exact_phrase_fallback(query: str, limit=3):
    q = (query or "").strip()
    if len(q) < 8:
        return []
    pat = re.escape(q)
    hits = []
    for p in _DOCS_DIR.glob("**/*.md"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if not m:
            continue
        s = max(0, m.start() - 300); e = min(len(txt), m.end() + 300)
        hits.append({
            "id": f"{p.name}::0",
            "file": p.name,
            "page": None,
            "score": 1.0,
            "text": txt[s:e]
        })
        if len(hits) >= limit:
            break
    return hits

# --- Retrieval ---
from .settings import settings

_PENALTY = None           # type: dict | None
_PENALTY_MTIME = 0.0
_PENALTY_LOCK = threading.Lock()

def _load_penalty() -> dict:
    """
    懒加载 + mtime 变更时重载：
    返回 { "file.md::12": 0.20, ... }，键为 文件名::页码；值为扣分(>=0)。
    """
    global _PENALTY, _PENALTY_MTIME
    p = Path("data/feedback/penalty.json")
    try:
        mtime = p.stat().st_mtime
    except FileNotFoundError:
        with _PENALTY_LOCK:
            _PENALTY = {}
            _PENALTY_MTIME = 0.0
        return {}

    with _PENALTY_LOCK:
        if _PENALTY is None or mtime != _PENALTY_MTIME:
            try:
                _PENALTY = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                _PENALTY = {}
            _PENALTY_MTIME = mtime
        return _PENALTY or {}

def hybrid_retrieve(query: str, index: Index, embedder: Embedder, k: int, *, soft: bool=False) -> List[Dict]:
    def _tokenize_q(q: str) -> list[str]:
        def _to_halfwidth(s: str) -> str:
            out = []
            for ch in s:
                code = ord(ch)
                if code == 0x3000: code = 0x20
                elif 0xFF01 <= code <= 0xFF5E: code -= 0xFEE0
                out.append(chr(code))
            return re.sub(r"\s+", " ", "".join(out)).strip()

        s = _to_halfwidth(q)
        toks: list[str] = []

        # 英文/数字词，始终加入
        toks += re.findall(r"[A-Za-z0-9_]+", s.lower())

        # 若含 CJK，再追加 CJK n-gram（只对 CJK 字符做）
        if _CJK.search(s):
            cjk_only = "".join(ch for ch in s.replace(" ", "") if _CJK.match(ch))
            if cjk_only:
                if len(cjk_only) >= 2:
                    toks += [cjk_only[i:i+2] for i in range(len(cjk_only)-1)]
                if len(cjk_only) >= 3:
                    toks += [cjk_only[i:i+3] for i in range(len(cjk_only)-2)]
        return toks

    is_cjk = bool(_CJK.search(query))
    q_emb = embedder.encode([query])

    # 向量检索
    vec_hits: List[Tuple[int, float]] = []
    if index.faiss is not None and index.faiss.ntotal > 0:
        D, I = index.faiss.search(q_emb.astype(np.float32), max(k, 50))
        vec_hits = [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]

    # BM25（中文分词改造）
    bm25_hits: List[Tuple[int, float]] = []
    if index.bm25 is not None and index.bm25_corpus_tokens:
        q_tokens = _tokenize_q(query)
        if q_tokens:
            scores = index.bm25.get_scores(q_tokens)
            top_ids = np.argsort(scores)[::-1][:max(k, 50)]
            bm25_hits = [(int(i), float(scores[i])) for i in top_ids]

    # 合并分数（权重随 CJK 调整）
    # 中文：BM25 更重要；英文：向量为主
    alpha_vec = 0.60 if is_cjk else 0.90
    alpha_bm25 = 0.40 if is_cjk else 0.10

    score_map: Dict[int, Dict[str, float]] = {}
    for idx_i, sim in vec_hits:
        m = score_map.setdefault(idx_i, {"vec": -1e9, "bm25": -1e9})
        m["vec"] = max(m["vec"], sim)
    for idx_i, s in bm25_hits:
        m = score_map.setdefault(idx_i, {"vec": -1e9, "bm25": -1e9})
        m["bm25"] = max(m["bm25"], s)

    # 自适应阈值
    vec_thr = settings.min_vec_sim * (0.7 if (soft or is_cjk) else 1.0)
    bm25_thr = settings.min_bm25_score * (0.6 if (soft or is_cjk) else 1.0)

    # 书名号短语（如《津貼及服務協議》）用于加权
    phrase_boost = 0.35 if is_cjk else 0.20
    m_phrase = re.search(r"《(.+?)》", query)
    phrase = m_phrase.group(1).strip() if m_phrase else None

    # 读取一次惩罚表
    _PENALTY = None
    def _load_penalty():
        nonlocal _PENALTY
        if _PENALTY is None:
            from pathlib import Path
            p = Path("data/feedback/penalty.json")
            _PENALTY = json.loads(p.read_text("utf-8")) if p.exists() else {}
        return _PENALTY

    # 阈值过滤 + 融合 + phrase 加权 + 惩罚
    passed: List[Tuple[int, float]] = []
    pen = _load_penalty()
    for idx_i, sig in score_map.items():
        vec_ok = (sig["vec"] >= vec_thr)
        bm_ok = (sig["bm25"] >= bm25_thr)
        if not (vec_ok or bm_ok):
            continue

        # 基础融合分（注意：FAISS D 已是余弦，BM25 是原始分）
        combo = alpha_vec * max(sig["vec"], 0.0) + alpha_bm25 * max(sig["bm25"], 0.0)

        # 书名号短语命中加权
        meta = index.meta[idx_i]
        meta_text = meta.get("text") or (index.texts[idx_i] if hasattr(index, "texts") and index.texts else "")
        if phrase and meta_text and phrase in meta_text:
            combo += phrase_boost

        # 应用惩罚（👎反馈）
        from pathlib import Path
        key = f"{Path(meta['file']).name}::{meta.get('page')}"
        penalty = float(pen.get(key, 0.0))  # 例如 0.15~0.30
        combo -= penalty

        passed.append((idx_i, combo))

    # 排序+截断
    passed.sort(key=lambda x: x[1], reverse=True)
    passed = passed[:k]

    # 出结果：补齐 text 字段，便于后续逻辑判断与渲染
    results: List[Dict] = []
    for idx_i, combo in passed:
        meta = index.meta[idx_i]
        text = meta.get("text") or (index.texts[idx_i] if hasattr(index, "texts") and index.texts else None)
        results.append({"text": text, "meta": meta, "idx": idx_i, "score": float(combo)})
    return results

# --- Prompt & Citations ---

from pathlib import Path
def build_prompt(messages: List[Dict], contexts: List[Dict]) -> List[Dict]:
    citation_blocks = []
    for i, c in enumerate(contexts, 1):
        file = Path(c["meta"]["file"]).name
        page = c["meta"].get("page")
        ref = f"[Source {i}] {file}" + (f", page {page}" if page is not None else "")
        raw = c["meta"].get("text") or ""
        snippet = raw[:1200]
        citation_blocks.append(f"{ref}\n\n{snippet}")

    K = len(contexts)
    system = {
        "role": "system",
        "content": (
            "You are ElderlyCare HK, a helpful assistant that answers strictly based on the provided Hong Kong Social Welfare Department documents.\n"
            f"- You are given exactly {K} sources. If you cite, you must use ONLY these tokens: "
            + ", ".join(f"[Source {i}]" for i in range(1, K+1)) + ".\n"
            "- Never invent new source indices. If information is not in the sources, clearly say it is not found.\n"
            "- Write plain text only (no JSON). Do NOT include a separate 'Sources' section.\n"
            "- If you cite inline, use the provided tokens verbatim (e.g., ... [Source 1]).\n"
            "- Treat each request as a fresh conversation and DO NOT use any memory beyond the messages provided in this request.\n"
            "- Resolve pronouns in the last user message using the chat history provided in this request.\n"
            "- If the user question is in Traditional Chinese, answer in Traditional Chinese.\n"
            "- If the question is in English, answer in English.\n"
        )
    }
    
    context_msg = {
        "role": "system",
        "content": "Relevant sources:\n\n" + "\n\n---\n\n".join(citation_blocks)
    }
    return [system, context_msg] + messages

def format_citations(contexts: List[Dict]) -> List[Dict]:
    from pathlib import Path
    seen = set()
    out = []
    for c in contexts:
        file = Path(c["meta"]["file"]).name
        page = c["meta"].get("page")
        key = (file, page)
        if key not in seen:
            seen.add(key)
            out.append({"file": file, "page": page, "snippet": None})
    return out
