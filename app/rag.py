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
from .utils import read_markdown_files, read_all_documents, infer_page_map

DOCS_DIR = Path("data/docs")

# PDF页码标记正则（用于从PDF提取的文本中识别页码）
_PDF_PAGE_MARKER = re.compile(r"---\s*Page\s*(\d+)\s*---")

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
        # 文件哈希记录（用于增量索引）
        self.file_hashes: Dict[str, str] = {}

    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

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

        # file_hashes.json（用于增量索引）
        hashes_path = self.index_dir / "file_hashes.json"
        hashes_bytes = json.dumps(self.file_hashes, ensure_ascii=False).encode("utf-8")
        hashes_path.write_bytes(encrypt_bytes(hashes_bytes))

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

        # file_hashes.json
        hashes_path = self.index_dir / "file_hashes.json"
        if hashes_path.exists():
            try:
                hb = decrypt_bytes(hashes_path.read_bytes())
                self.file_hashes = json.loads(hb.decode("utf-8"))
            except Exception:
                self.file_hashes = {}

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

    def add_vectors(self, embeddings: np.ndarray, meta: List[Dict], bm25_tokens: Optional[List[List[str]]] = None):
        """
        增量添加向量到现有索引。
        用于增量索引场景。
        """
        if self.faiss is None:
            # 如果索引为空，直接build
            self.build(embeddings, meta, bm25_tokens)
            return

        if embeddings.size == 0:
            return

        # 添加到FAISS索引
        self.faiss.add(embeddings.astype(np.float32))

        # 添加元数据
        self.meta.extend(meta)

        # 更新BM25（需要重建，因为BM25不支持增量）
        if bm25_tokens:
            self.bm25_corpus_tokens.extend(bm25_tokens)
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def remove_file(self, file_path: str):
        """
        从索引中移除指定文件的所有chunk。
        注意：FAISS IndexFlatIP不支持直接删除，需要重建。
        """
        if not self.meta:
            return

        # 找出要保留的索引
        keep_indices = []
        for i, m in enumerate(self.meta):
            if m.get("file") != file_path:
                keep_indices.append(i)

        if len(keep_indices) == len(self.meta):
            # 没有需要删除的
            return

        # 重建索引（只保留需要的部分）
        # 这是一个昂贵的操作，但FAISS IndexFlatIP不支持删除
        if keep_indices:
            # 获取需要保留的向量
            all_vectors = faiss.rev_swig_ptr(self.faiss.get_xb(), self.faiss.ntotal * self.faiss.d)
            all_vectors = all_vectors.reshape(self.faiss.ntotal, self.faiss.d)
            kept_vectors = all_vectors[keep_indices]

            # 重建FAISS索引
            dim = self.faiss.d
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(kept_vectors.astype(np.float32))
            self.faiss = new_index

            # 更新元数据
            self.meta = [self.meta[i] for i in keep_indices]

            # 更新BM25
            if self.bm25_corpus_tokens:
                self.bm25_corpus_tokens = [self.bm25_corpus_tokens[i] for i in keep_indices]
                self.bm25 = BM25Okapi(self.bm25_corpus_tokens) if self.bm25_corpus_tokens else None
        else:
            # 全部删除，清空索引
            dim = self.faiss.d if self.faiss else 384
            self.faiss = faiss.IndexFlatIP(dim)
            self.meta = []
            self.bm25_corpus_tokens = []
            self.bm25 = None

        # 更新文件哈希
        if file_path in self.file_hashes:
            del self.file_hashes[file_path]

    def get_indexed_files(self) -> set:
        """获取已索引的文件列表"""
        return set(m.get("file") for m in self.meta if m.get("file"))

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

# --- Enhanced semantic chunking ---
# 段落分隔符
_PARA_SPLIT = re.compile(r'\n\s*\n')
# 句子分隔符（中英文混合）
_SENTENCE_SPLIT = re.compile(r'(?<=[。！？.!?])\s+|(?<=\n)')
# 标题检测（Markdown风格）
_HEADING_RE = re.compile(r'^#{1,6}\s+.+$|^.+\n[=\-]{2,}$', re.MULTILINE)

def semantic_chunk(text: str, chunk_size: int, overlap: int, respect_paragraphs: bool = True) -> List[str]:
    """
    语义感知的文本分块：
    1. 首先按段落分割
    2. 如果段落太长，按句子分割
    3. 合并小段落直到达到chunk_size
    4. 保持语义完整性
    """
    if not text or not text.strip():
        return []

    chunks = []
    current_chunk = ""

    # 步骤1: 按段落分割
    if respect_paragraphs:
        paragraphs = _PARA_SPLIT.split(text)
    else:
        paragraphs = [text]

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果段落本身就很短，尝试合并
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            continue

        # 如果当前chunk已经有内容，先保存
        if current_chunk:
            chunks.append(current_chunk)
            # 保留overlap部分
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""

        # 如果段落太长，按句子分割
        if len(para) > chunk_size:
            sentences = _split_into_sentences(para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    if current_chunk:
                        current_chunk += " " + sent
                    else:
                        current_chunk = sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # 如果单个句子就超过chunk_size，强制分割
                    if len(sent) > chunk_size:
                        sub_chunks = simple_char_chunk(sent, chunk_size, overlap)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = sent
        else:
            current_chunk = para

    # 保存最后一个chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def _split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    # 中文句子结束标志
    cn_pattern = r'([。！？；])'
    # 英文句子结束标志
    en_pattern = r'([.!?])\s+'

    # 先按中文标点分割
    parts = re.split(cn_pattern, text)
    sentences = []
    i = 0
    while i < len(parts):
        sent = parts[i]
        # 如果下一个是标点，合并
        if i + 1 < len(parts) and re.match(cn_pattern, parts[i+1]):
            sent += parts[i+1]
            i += 2
        else:
            i += 1
        if sent.strip():
            sentences.append(sent.strip())

    # 再对每个部分按英文标点分割
    final_sentences = []
    for sent in sentences:
        if re.search(r'[.!?]\s+[A-Z]', sent):
            # 包含英文句子
            en_parts = re.split(en_pattern, sent)
            j = 0
            while j < len(en_parts):
                s = en_parts[j]
                if j + 1 < len(en_parts) and re.match(en_pattern, en_parts[j+1] + ' '):
                    s += en_parts[j+1]
                    j += 2
                else:
                    j += 1
                if s.strip():
                    final_sentences.append(s.strip())
        else:
            final_sentences.append(sent)

    return final_sentences

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    智能文本分块：
    - 优先使用语义分块（保持段落和句子完整性）
    - 对于纯中文文本，使用原有的CJK感知分块作为备选
    """
    # 检测是否主要是中文
    cjk_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / max(1, len(text))

    # 尝试语义分块
    chunks = semantic_chunk(text, chunk_size, overlap)

    # 如果语义分块效果不好（块数太少或块太大），回退到原有策略
    if not chunks:
        # 回退到原有的CJK感知分块
        parts = []
        segs = [s for s in _SENT_SPLIT.split(text) if s]
        for seg in segs:
            parts.extend(simple_char_chunk(seg, chunk_size, overlap))
        if not segs:
            parts = simple_char_chunk(text, chunk_size, overlap)
        return parts

    return chunks

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
def _get_page_ranges_for_doc(doc: Dict) -> List[Dict]:
    """
    根据文档格式获取页码范围。
    对于PDF文档，优先使用read_pdf_file返回的page_ranges。
    对于其他格式，使用infer_page_map从文本中推断。
    """
    # 如果文档已有page_ranges（来自PDF解析），直接使用
    if "page_ranges" in doc and doc["page_ranges"]:
        return doc["page_ranges"]

    # 否则从文本中推断页码
    return infer_page_map(doc["text"])

def _clean_text(text: str) -> str:
    """
    清洗文本：去除多余空白、特殊字符等
    """
    if not text:
        return ""

    # 去除多余空白行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 去除控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()

def ingest_corpus(docs_dir: str, index_dir: str) -> Tuple[int, int]:
    docs = read_all_documents(docs_dir)
    print(f"[ingest] loaded {len(docs)} doc(s)")
    embedder = Embedder(settings.embedding_model_name, settings.embedding_device)

    all_chunks: List[Chunk] = []
    for di, doc in enumerate(docs, 1):
        doc_format = doc.get("format", "unknown")
        page_ranges = _get_page_ranges_for_doc(doc)
        print(f"[ingest] doc {di}/{len(docs)} ({doc_format}) -> {len(page_ranges)} page(s)")

        for pr in page_ranges:
            page_text = doc["text"][pr["start"]:pr["end"]]
            # 清洗文本
            page_text = _clean_text(page_text)
            if not page_text.strip():
                continue

            pieces = chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
            for i, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                meta = {
                    "file": doc["path"],
                    "page": pr["page"],
                    "chunk_id": i,
                    "text": piece,
                    "format": doc_format
                }
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

    # 更新文件哈希记录
    for doc in docs:
        index.file_hashes[doc["path"]] = index._compute_file_hash(doc["path"])

    index.save()
    print("[ingest] index saved")

    return len(docs), len(all_chunks)

def ingest_file_incremental(file_path: str, index_dir: str) -> Tuple[int, bool]:
    """
    增量索引单个文件。
    只对新文件或已修改的文件进行索引，而不是重建整个索引。

    返回: (chunks_added, was_updated)
    - chunks_added: 添加的chunk数量
    - was_updated: 文件是否被更新（True）或是新文件（False）
    """
    from .utils import read_document

    # 加载现有索引
    index = Index(index_dir)
    index.load()

    embedder = Embedder(settings.embedding_model_name, settings.embedding_device)

    # 计算文件哈希
    file_path_str = str(file_path)
    new_hash = index._compute_file_hash(file_path_str)

    # 检查文件是否已经索引且未修改
    old_hash = index.file_hashes.get(file_path_str)
    if old_hash == new_hash and old_hash is not None:
        print(f"[ingest] File {file_path} unchanged, skipping")
        return 0, False

    # 如果文件已存在但被修改，先删除旧的
    was_updated = False
    if old_hash is not None:
        print(f"[ingest] File {file_path} modified, removing old chunks")
        index.remove_file(file_path_str)
        was_updated = True

    # 读取文档
    doc = read_document(Path(file_path))
    if not doc["text"]:
        print(f"[ingest] File {file_path} is empty or unreadable")
        return 0, was_updated

    # 处理文档
    doc_format = doc.get("format", "unknown")
    page_ranges = _get_page_ranges_for_doc(doc)
    print(f"[ingest] Processing {file_path} ({doc_format}) -> {len(page_ranges)} page(s)")

    all_chunks: List[Chunk] = []
    for pr in page_ranges:
        page_text = doc["text"][pr["start"]:pr["end"]]
        page_text = _clean_text(page_text)
        if not page_text.strip():
            continue

        pieces = chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            if not piece.strip():
                continue
            meta = {
                "file": doc["path"],
                "page": pr["page"],
                "chunk_id": i,
                "text": piece,
                "format": doc_format
            }
            all_chunks.append(Chunk(text=piece, meta=meta))

    if not all_chunks:
        print(f"[ingest] No chunks generated from {file_path}")
        # 更新文件哈希（即使没有内容）
        index.file_hashes[file_path_str] = new_hash
        index.save()
        return 0, was_updated

    # 生成嵌入
    texts = [c.text for c in all_chunks]
    BATCH = 512
    emb_list = []
    for i in range(0, len(texts), BATCH):
        emb_list.append(embedder.encode(texts[i:i+BATCH]))
    embeddings = np.vstack(emb_list).astype(np.float32)
    print(f"[ingest] Incremental embedding done, shape={embeddings.shape}")

    # BM25 tokens
    bm25_tokens = [tokenize(t) for t in texts] if settings.enable_bm25 else None

    # 添加到索引
    meta = [c.meta for c in all_chunks]
    index.add_vectors(embeddings, meta, bm25_tokens)

    # 更新文件哈希
    index.file_hashes[file_path_str] = new_hash

    index.save()
    print(f"[ingest] Incremental index updated: {len(all_chunks)} chunks added")

    return len(all_chunks), was_updated

def remove_file_from_index(file_path: str, index_dir: str) -> bool:
    """
    从索引中删除指定文件的所有chunk。

    返回: True 如果文件被删除，False 如果文件不在索引中
    """
    index = Index(index_dir)
    index.load()

    file_path_str = str(file_path)
    if file_path_str not in index.file_hashes:
        print(f"[ingest] File {file_path} not in index")
        return False

    print(f"[ingest] Removing {file_path} from index")
    index.remove_file(file_path_str)
    index.save()
    print(f"[ingest] File removed from index")

    return True

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
