
# ============================================================================
# File: app/admin_docs.py
# ============================================================================
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import shutil, json, time, os, tempfile
import asyncio
from typing import List, Dict, Any

from .settings import settings
from .rag import ingest_corpus
from .security import require_bearer
from .ingest_manager import start as ingest_start, cancel as ingest_cancel, status as ingest_status


router = APIRouter(prefix="/docs", tags=["docs"])

DOCS_DIR = Path(settings.docs_dir)
STATUS_PATH = Path("data/status.json")
TMP_DIR = Path("data/tmp_uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def _write_status(payload: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    # 原子写：先写临时文件，再 replace
    with tempfile.NamedTemporaryFile(dir=str(STATUS_PATH.parent), delete=False) as tf:
        tf.write(data)
        tmpname = tf.name
    os.replace(tmpname, STATUS_PATH)

def _reindex_job(note: str):
    _write_status({"status": "indexing", "note": note, "start_ts": int(time.time())})
    try:
        # ingest_corpus 可以是同步函数；若你实现的是 async，可在这里用 anyio.run 调用
        if asyncio.iscoroutinefunction(ingest_corpus):
            import anyio
            anyio.run(ingest_corpus)
        else:
            ingest_corpus()
        _write_status({"status": "ready", "note": note, "last_built": int(time.time())})
    except Exception as e:
        _write_status({"status": "error", "note": f"{note}: {e}", "ts": int(time.time())})

@router.get("/status", dependencies=[Depends(require_bearer)])
def get_status():
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "unknown"}
    return {"status": "ready"}

@router.get("/list", dependencies=[Depends(require_bearer)])
def list_docs():
    items = []
    for p in sorted(list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.markdown"))):
        items.append({
            "filename": p.name,
            "size": p.stat().st_size,
            "modified": int(p.stat().st_mtime),
        })
    return {"docs": items}

@router.post("/upload", dependencies=[Depends(require_bearer)])
async def upload_doc(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".md") or name.endswith(".markdown")):  # 如需同时支持 pdf，可加 or name.endswith(".pdf")
        raise HTTPException(400, "Only Markdown (.md/.markdown) is supported")
    
    tmp_path = TMP_DIR / (file.filename + ".part")
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    dest = DOCS_DIR / file.filename
    tmp_path.replace(dest)  # 原子移动

    # ===== 状态：indexing =====
    _write_status({"status": "indexing", "note": f"uploaded: {file.filename}", "start_ts": int(time.time())})
   
    # —— 关键：同步重建（阻塞直到完成）——
    # 若 ingest_corpus 是 CPU/IO 密集，同步调用会卡住 event loop；
    # 用 asyncio.to_thread 让其在线程池执行，但这里依然“等它完成再返回”，体验等价于阻塞式。
    await asyncio.to_thread(ingest_corpus, settings.docs_dir, settings.index_dir)

    # 关键：失效内存索引缓存
    from . import main as _main
    _main._index = None
    _main._embedder = None

    # ===== 状态：ready =====
    _write_status({"status": "ready", "note": f"uploaded: {file.filename}", "last_built": int(time.time())})
    return {"ok": True, "message": f"{file.filename} uploaded. Index rebuilt (hot-loaded)."}

@router.delete("/{filename}", dependencies=[Depends(require_bearer)])
async def delete_doc(filename: str):
    # 基础校验，阻止路径穿越
    if "/" in filename or "\\" in filename:
        raise HTTPException(400, "Bad filename")

    # 允许省略后缀、大小写不敏感匹配
    candidates = [
        DOCS_DIR / filename,
        DOCS_DIR / (filename if filename.lower().endswith(".md") else filename + ".md"),
        DOCS_DIR / (filename if filename.lower().endswith(".markdown") else filename + ".markdown"),
    ]

    # 如果都不存在，尝试在目录里做一次“大小写不敏感”/近似匹配
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        low = filename.lower()
        for p in DOCS_DIR.glob("*"):
            if p.name.lower() == low or p.name.lower() == (low + ".md") or p.name.lower() == (low + ".markdown"):
                path = p
                break

    # 能找到就删，找不到也继续重建（保证索引与磁盘一致）
    if path and path.exists():
        path.unlink()
    
    # —— 状态：indexing ——
    _write_status({"status": "indexing", "note": f"deleted: {filename}", "start_ts": int(time.time())})

    try:
        # —— 同步重建（热加载）——
        await asyncio.to_thread(ingest_corpus, settings.docs_dir, settings.index_dir)

        # 关键：失效内存索引缓存
        from . import main as _main
        _main._index = None
        _main._embedder = None

        # —— 状态：ready ——
        _write_status({"status": "ready", "note": f"deleted: {filename}", "last_built": int(time.time())})
        return {"ok": True, "message": f"{filename} deleted (if existed). Index rebuilt (hot-loaded)."}
    except Exception as e:
        # —— 状态：error（可在前端提示）——
        _write_status({"status": "error", "note": f"deleted: {filename}: {e}", "ts": int(time.time())})
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
        
@router.post("/cancel", dependencies=[Depends(require_bearer)])
def cancel_reindex():
    killed = ingest_cancel()
    return {"ok": True, "killed": killed}

# ============================================================================
# File: app/ingest_manager.py
# ============================================================================
from __future__ import annotations
from multiprocessing import Process
from pathlib import Path
import json, os, tempfile, time, traceback
from typing import Dict, Any, Optional
from .rag import ingest_corpus
from .settings import settings

STATUS_PATH = Path("data/status.json")
STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

_proc: Optional[Process] = None
_note: str = ""
_start_ts: Optional[int] = None

def _write_status(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=str(STATUS_PATH.parent), delete=False) as tf:
        tf.write(data)
        tmpname = tf.name
    os.replace(tmpname, STATUS_PATH)

def _target(note: str):
    # 子进程里执行真正的重建
    _write_status({"status":"indexing", "note":note, "start_ts":int(time.time())})
    try:
        ingest_corpus(settings.docs_dir, settings.index_dir)
        _write_status({"status":"ready", "note":note, "last_built":int(time.time())})
    except Exception:
        err = traceback.format_exc()
        _write_status({"status":"error", "note":f"{note}: {err}", "ts":int(time.time())})

def start(note: str) -> bool:
    global _proc, _note, _start_ts
    if _proc is not None and _proc.is_alive():
        return False  # 已有任务在跑，返回 False
    _note = note
    _start_ts = int(time.time())
    _proc = Process(target=_target, args=(note,), daemon=True)
    _proc.start()
    return True

def cancel() -> bool:
    global _proc
    if _proc is None:
        return False
    alive = _proc.is_alive()
    if alive:
        _proc.terminate()
        _proc.join(timeout=5)
        _write_status({"status":"canceled", "note":"canceled by user", "ts":int(time.time())})
    _proc = None
    return alive

def status() -> Dict[str, Any]:
    s: Dict[str, Any]
    if STATUS_PATH.exists():
        try:
            s = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            s = {"status":"unknown"}
    else:
        s = {"status":"ready"}
    if _proc is not None:
        s["running"] = _proc.is_alive()
        s["pid"] = _proc.pid
    return s

# ============================================================================
# File: app/llm_client.py
# ============================================================================
import httpx
from typing import List, AsyncGenerator
from .settings import settings

# stream & non-stream are split to avoid async generator return conflicts
async def smartcare_chat_stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    payload = {
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", settings.smartcare_base_url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line

async def smartcare_chat(messages: List[dict]) -> dict:
    payload = {
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(settings.smartcare_base_url, json=payload)
        r.raise_for_status()
        try:
            data = r.json()
        except Exception:
            text = r.text
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}
        if isinstance(data, dict) and "choices" not in data:
            content = data.get("content") or data.get("answer") or data.get("message") or data.get("data")
            if isinstance(content, str):
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        return data

async def smartcare_rewrite_query(messages: list[dict]) -> str:
    try:
        sys = {
            "role": "system",
            "content": (
                "You are a query rewriter. Rewrite ONLY the last user message into a standalone, context-complete question. "
                "Resolve pronouns (it/its/this/that/they/their; 它/其/這/該 等) to the most recent explicit entity mentioned "
                "in the conversation, especially policy/program names (e.g., Old Age Allowance, Old Age Living Allowance, LSG Subvention Manual).\n"
                "Keep the original language. Output the rewritten question text only.\n\n"
                "Examples:\n"
                "User history: What is Old Age Allowance?\n"
                "Last user: What are its eligibility requirements?\n"
                "Rewrite: What are the eligibility requirements for the Old Age Allowance?\n\n"
                "User history: 什麼是長者生活津貼？\n"
                "Last user: 申請要哪些文件？\n"
                "Rewrite: 長者生活津貼的申請需要哪些文件？"
            ),
        }
        payload_msgs = [sys] + messages
        data = await smartcare_chat(payload_msgs)
        text = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        return (text or "").strip() or messages[-1]["content"]
    except Exception:
        return messages[-1]["content"]

async def smartcare_translate_to_en(text: str) -> str:
    """
    将輸入翻譯為英文（保持原意、用於檢索），若失敗則回退原文。
    """
    try:
        sys = {
            "role": "system",
            "content": (
                "Translate the user's text into natural English suitable for information retrieval. "
                "Preserve the meaning. Output English only, no explanations."
            ),
        }
        data = await smartcare_chat([sys, {"role": "user", "content": text}])
        out = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return out.strip() or text
    except Exception:
        return text

# ============================================================================
# File: app/main.py
# ============================================================================
import uvicorn
import numpy as np
import re, json, time, statistics, collections
from pathlib import Path
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .schemas import ChatRequest, ChatAnswer, IngestResponse, FeedbackIn
from .rag import Index, Embedder, hybrid_retrieve, build_prompt, format_citations, ingest_corpus
from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_translate_to_en
from .security import require_bearer
from .admin_docs import router as admin_docs_router

app = FastAPI(title="ElderlyCare HK — Backend")
app.include_router(admin_docs_router)

_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_USAGE_LOG = _LOG_DIR / "chat_usage.jsonl"

def _append_jsonl_safe(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _now_ms() -> int:
    return int(time.time() * 1000)

_FB_DIR = Path("data/feedback")
_FB_DIR.mkdir(parents=True, exist_ok=True)
_FB_JSONL = _FB_DIR / "feedback.jsonl"         # 已存在：主反馈日志
_FB_METRICS = _FB_DIR / "metrics.json"         # 已存在：聚合指标
_FB_COUNTS = _FB_DIR / "penalty_counts.json"   # 新增：按页计数 { (file,page): {up,down} }
_FB_PENALTY = _FB_DIR / "penalty.json"         # 已存在于 rag；我们会即时生成/更新

def _safe_read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json_atomic(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _recompute_penalty_from_counts(counts: dict) -> dict:
    """
    把 { "file::page": {"up":x,"down":y}, ... } 转成 { "file::page": penalty_float }
    简单规则：penalty = min(1.0, BASE + STEP * max(0, down - up))
    你也可以换成更平滑的函数（例如 sigmoid）。
    """
    BASE, STEP = 0.10, 0.05   # 可按需要微调
    out = {}
    for k, v in counts.items():
        up = int(v.get("up", 0))
        down = int(v.get("down", 0))
        excess = max(0, down - up)
        pen = min(1.0, BASE + STEP * excess)
        if pen > 0:
            out[k] = round(pen, 4)
    return out

def _bump_counts_and_penalty(recs: list[dict]):
    """
    根据这次反馈里附带的 citations 批量更新 counts 与 penalty。
    recs: [{ "file": "...", "page": 12 }, ...]
    """
    counts = _safe_read_json(_FB_COUNTS, {})
    # 每条 citation 累计 up/down；由调用者决定当前反馈是 up 还是 down
    def inc(key: str, field: str):
        row = counts.get(key) or {}
        row[field] = int(row.get(field, 0)) + 1
        counts[key] = row

    # 上层会把 label 透传进来；这里不区分每条 citation 的 label
    for r in recs:
        file = r.get("file")
        page = r.get("page")
        if not file:
            continue
        key = f"{file}::{page}"
        # 占位，实际增量在调用处做（见 feedback_in 内）
        counts.setdefault(key, counts.get(key, {"up": 0, "down": 0}))

    return counts, inc

_CITE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]")
_PRONOUN_PAT = re.compile(r"\b(it|its|this|that|they|their)\b|[它其這該]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 基本漢字

# 简单别名映射（可继续补充）
ALIASES = {
    "OAA": "Old Age Allowance",
    "OALA": "Old Age Living Allowance",
    "LSG": "Lump Sum Grant",
    "LSGSS": "Lump Sum Grant Subvention System",
}

# 用“非字母数字”前后视图替代 \b，避免在中文里失效
# (?<![A-Za-z0-9]) …… (?![A-Za-z0-9])
_ENTITY_EXTRACT_RE = re.compile(
    r"(?<![A-Za-z0-9])("                                   # 英文正式名稱（首字母大寫的多詞短語 + 後綴）
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,7}\s"
    r"(?:Allowance|Scheme|Program|Programme|Manual|Handbook|Guide|Guidance\s+Notes|Notes|Policy|"
    r"Ordinance|Regulation|Circular|Arrangement|Grant|Service|Subvention|System|Framework|Code|"
    r"Plan|Charter|Protocol|Directive|Guideline)s?"         # 允許可選複數 s
    r")(?![A-Za-z0-9])"
    r"|(?:Old\s+Age\s+Allowance|Old\s+Age\s+Living\s+Allowance|Disability\s+Allowance|"
    r"Comprehensive\s+Social\s+Security\s+Assistance)"      # 常見英文全名
    r"|(?:OAA|OALA|CSSA|DA|LSG|LSGSS)"                      # 常見英文縮寫
    r"|(?:長者生活津貼|高齡津貼|老年津貼|傷殘津貼|"
    r"綜合社會保障援助|資助福利服務|統一撥款|資助手冊|撥款制度|"
    r"政策|計劃|津貼|手冊|指引|通告|規例|條例|方案|制度|安排)"  # 繁中後綴/同義詞擴充
    , re.I
)

# 這個比上面的稍窄，用於你的“實體提示/澄清”檢測（不需要過多干擾詞）
_ENTITY_PAT = re.compile(
    r"(?<![A-Za-z0-9])("
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,6}\s"
    r"(?:Allowance|Scheme|Program|Programme|Manual|Handbook|Policy|Ordinance|Regulation|"
    r"Guideline|Circular|Grant|Service|Subvention|System)s?"
    r")(?![A-Za-z0-9])"
    r"|(?:Old\s+Age\s+Allowance|Old\s+Age\s+Living\s+Allowance|Operating\s+Subvented\s+Welfare|"
    r"LSG\s+Subvention\s+Manual|Disability\s+Allowance|Comprehensive\s+Social\s+Security\s+Assistance)"
    r"|(?:OAA|OALA|CSSA|DA|LSG|LSGSS)"
    r"|(?:長者生活津貼|高齡津貼|老年津貼|傷殘津貼|資助福利服務|統一撥款|資助手冊|"
    r"撥款制度|計劃|津貼|手冊|指引|通告|規例|條例|制度)"
    , re.I
)

# --- Helpers for zh-Hant queries ---
def _merge_dedup_hits(h1: list[dict], h2: list[dict], k: int) -> list[dict]:
    """以 (file, page, chunk_id) 去重；分数取较大值；保留来源标记便于调试"""
    seen = {}
    for src, tag in ((h1, "zh"), (h2, "en")):
        for h in (src or []):
            key = (h.get("file"), h.get("page"), h.get("chunk_id"))
            score = float(h.get("score", 0))
            if key not in seen or score > seen[key]["score"]:
                seen[key] = {**h, "score": score, "src": tag}
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]

def _expand_aliases_zh(q: str) -> str:
    """把繁中的俗称/简称扩成 (A OR B OR 英文名)；你可把表慢慢补充起来"""
    table = {
        "生果金": ["高齡津貼", "Old Age Allowance", "OAA"],
        "綜援": ["綜合社會保障援助", "Comprehensive Social Security Assistance", "CSSA"],
    }
    # 先处理繁中文本
    for k, vs in table.items():
        if k in q:
            q = q.replace(k, f"({ ' OR '.join([k] + vs) })")
    # 再套用你现有的英文缩写表
    for short, full in ALIASES.items():
        if short in q:
            q = q.replace(short, f"({short} OR {full})")
    return q

def _norm_for_entity(s: str) -> str:
    # 全角->半角 + 去掉多餘空白
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000: code = 0x20
        elif 0xFF01 <= code <= 0xFF5E: code -= 0xFEE0
        out.append(chr(code))
    return re.sub(r"\s+", " ", "".join(out)).strip()

def _extract_focus_phrase(s: str) -> str | None:
    s = _norm_for_entity(s or "")
    # 书名号优先：如《津貼及服務協議》
    m = re.search(r"《(.+?)》", s)
    if m: return m.group(1).strip()
    # 退化：连续大写开头词 + 关键尾词（Accounts/Allowance/Manual/...）
    m = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,6}\s(?:Accounts?|Allowance|Manual|Programme|Scheme|Policy|System|Subvention))", s)
    return m.group(1).strip() if m else None

def _boost_by_phrase(contexts: list[dict], phrase: str | None, boost: float = 0.35) -> list[dict]:
    if not phrase: 
        return contexts
    for c in contexts:
        t = (c.get("text") 
             or (c.get("meta") or {}).get("text") 
             or "")
        if t and phrase in t:
            c["score"] = float(c.get("score", 0.0)) + boost
    return sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True)

def _extract_entities_from_text(text: str) -> list[str]:
    if not text: 
        return []
    return [m.group(0).strip() for m in _ENTITY_EXTRACT_RE.finditer(text)]

def _suggest_entities_for(query: str, idx: "Index", emb: "Embedder", top_k: int = 5) -> list[str]:
    """
    用當前查詢在索引裡做一次輕量檢索，從命中的 chunk 文本中抽取實體名，按頻次去重排序返回。
    """
    try:
        contexts = hybrid_retrieve(query, idx, emb, k=20, soft=True)
    except Exception:
        return []
    freq: dict[str, int] = {}
    for c in contexts:
        t = c["meta"].get("text") or ""
        for ent in _extract_entities_from_text(t):
            # 合併大小寫/空白差異
            key = re.sub(r"\s+", " ", ent).strip()
            freq[key] = freq.get(key, 0) + 1
    # 排序 + 去掉過於籠統的詞
    bad = {"Allowance","Scheme","Manual","Programme","Grant","Service","Subvention","System","Policy","計劃","津貼","手冊","政策"}
    items = [e for e,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True) if e not in bad]
    # 保留不同前綴的前 top_k 個
    out = []
    seen_lower = set()
    for e in items:
        low = e.lower()
        if low in seen_lower: 
            continue
        seen_lower.add(low)
        out.append(e)
        if len(out) >= top_k:
            break
    return out

def _looks_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))
# 轻量启发式参数
_MIN_TOKENS = 3
_GENERIC_Q_RE = re.compile(
    r"^(what|how|why|tell me|can you|could you|explain|give me|i want to know|說說|介紹|解釋|請講講|我想知道)\b",
    re.I,
)

def _looks_specific(q: str, contexts: list[dict]) -> bool:
    """
    返回 True 表示“这个查询已经足够具体”，不要再触发澄清。
    判据：
      - 含《》书名号（通常是确指标题）
      - 命中你定义的实体正则（政策/津贴名等）
      - 在 topN 候选文本里出现了原样短语（严格包含）
    """
    qn = _norm_for_entity(q)
    if "《" in qn and "》" in qn: return True
    if _ENTITY_PAT.search(qn):    return True
    inner = None
    m = re.search(r"《(.+?)》", qn)
    if m: inner = m.group(1).strip()
    phrase = (inner or qn).strip()
    if phrase:
        for c in contexts[:10]:
            t = (c.get("text") or (c.get("meta") or {}).get("text") or "")
            if t and phrase in t:
                return True
    return False

def _tokenize_simple(s: str) -> list[str]:
    return [t for t in re.findall(r"\w+|[\u4e00-\u9fff]", s or "") if t.strip()]

def _is_ambiguous_heuristic(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return True
    toks = _tokenize_simple(q)
    if len(toks) <= _MIN_TOKENS:
        return True
    uniq_ratio = len(set(toks)) / max(1, len(toks))
    if uniq_ratio < 0.5:
        return True
    has_named = bool(re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b", q))
    if _GENERIC_Q_RE.search(q) and not has_named:
        return True
    return False

# 语义模板（极小集合，捕捉“泛问/扩写/解释一下”这类）
_GENERIC_TEMPLATES = [
    "Tell me about it",
    "Tell me about this",
    "What is it",
    "Explain this",
    "Give me details",
    "I want to know more",
    "What about it",
    "請介紹一下",
    "這是什麼",
    "說說看",
]

_GENERIC_EMB: np.ndarray | None = None  # 懒加载缓存

def _ensure_generic_emb() -> np.ndarray:
    global _GENERIC_EMB
    if _GENERIC_EMB is None:
        emb = get_embedder()  # 复用现有的 SentenceTransformer（已做 normalize）
        _GENERIC_EMB = emb.encode(_GENERIC_TEMPLATES)
        if _GENERIC_EMB.ndim == 1:
            _GENERIC_EMB = _GENERIC_EMB.reshape(1, -1)
    return _GENERIC_EMB

def _is_ambiguous_semantic(q: str, thr: float = 0.78) -> bool:
    q = (q or "").strip()
    if not q:
        return True
    emb = get_embedder()
    qe = emb.encode([q])[0]  # 已归一化
    G = _ensure_generic_emb()  # (m, d)
    sims = (G @ qe)  # 余弦相似度
    return float(np.max(sims)) >= thr

def _should_clarify_smart(user_query: str) -> bool:
    # A：启发式先判
    if _is_ambiguous_heuristic(user_query):
        return True
    # B：与“泛问模板”相似则判模糊
    if _is_ambiguous_semantic(user_query):
        return True
    return False

def _clarify_question(user_query: str, lang: str | None) -> str:
    """
    当检测到用户问题模糊时，生成一条追问句，用于提示用户具体化问题。
    """
    if not user_query:
        user_query = "your question"

    # 中文界面
    if lang == "zh-Hant":
        return (
            f"你的問題（「{user_query}」）目前範圍過大。"
            "我在相關的官方文檔中找不到有關該主題的信息。"
        )

    # 英文界面
    return (
        f"Your question (“{user_query}”) is a bit broad. "
        "I couldn't find any information on that topic in documents related to elderly care."
    )

def _clarify_question_smart(user_query: str, lang: str | None, idx: "Index", emb: "Embedder") -> str:
    q = (user_query or "").strip()
    # 先做候選：比如 "allowance", "policy", "scheme", "津貼", "政策"
    keywords = ["allowance", "policy", "scheme", "manual", "programme", "津貼", "政策", "計劃", "手冊"]
    need_list = any(kw in q.lower() for kw in keywords) or _is_ambiguous_semantic(q)

    if need_list:
        cands = _suggest_entities_for(q, idx, emb, top_k=5)
        if cands:
            if lang == "zh-Hant":
                opts = "、".join(cands[:4])
                return f"你的問題較為籠統（「{q}」）。你是在問 {opts}，還是其他？"
            else:
                opts = ", ".join(cands[:4])
                return f'Your question (“{q}”) is a bit broad. Are you asking about {opts}, or something else?'

    # 候選空時，回退到原來的通用提示
    return _clarify_question(user_query, lang)

def _expand_aliases(text: str) -> str:
    out = text
    for k, v in ALIASES.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return out

def _guess_entity_from_history(msgs: list[dict]) -> str | None:
    """從對話歷史中猜測最近出現的明確政策/津貼名（支援繁中與全角）"""
    for m in reversed(msgs):
        txt = _expand_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        # 🔹在匹配前做正規化
        normed = _norm_for_entity(txt)
        hit = _ENTITY_PAT.search(normed)
        if hit:
            return hit.group(0)

    # 若仍未命中，退回首輪訊息再試
    for m in msgs:
        txt = _expand_aliases((m.get("content") or "").strip())
        normed = _norm_for_entity(txt)
        hit = _ENTITY_PAT.search(normed)
        if hit:
            return hit.group(0)
    return None

_FEEDBACK_DIR = Path("data/feedback")
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_PATH = _FEEDBACK_DIR / "feedback.jsonl"
_METRICS_PATH  = _FEEDBACK_DIR / "metrics.json"   # 用于极简在线指标

def _append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _bump_metrics(label: str):
    # 超迷你在线指标：累计 up/down 次数；可扩展为分桶(相似度区间)统计
    m = {"up": 0, "down": 0}
    if _METRICS_PATH.exists():
        try: m = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    m[label] = m.get(label, 0) + 1
    _METRICS_PATH.write_text(json.dumps(m), encoding="utf-8")

def _iter_feedback_jsonl():
    if not _FB_JSONL.exists():
        return
    for line in _FB_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

import datetime as _dt
def _iter_usage():
    if not _USAGE_LOG.exists():
        return
    for line in _USAGE_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip(): 
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

@app.get("/metrics/usage")
def metrics_usage(days: int = 7, _auth=Depends(require_bearer)):
    """
    基础使用统计：最近 N 天（默认 7）
    返回总请求数 / clarifications / 无命中 / 平均与P95时延 / 关键词Top-N（英文/数字简单切词）
    """
    now_ms = _now_ms()
    cutoff = now_ms - int(days) * 24 * 60 * 60 * 1000

    total = 0
    clarified = 0
    nohit = 0
    found = 0
    durs = []
    ans_lens = []
    word_counter = collections.Counter()
    per_day = collections.Counter()  # YYYY-MM-DD -> count

    for r in _iter_usage() or []:
        ts = r.get("ts")
        if isinstance(ts, (int, float)) and ts < cutoff:
            continue
        total += 1
        if r.get("clarified"): clarified += 1
        if not r.get("found"): nohit += 1
        else: found += 1
        dur = r.get("duration_ms")
        if isinstance(dur, (int, float)) and dur >= 0:
            durs.append(float(dur))
        al = r.get("answer_len")
        if isinstance(al, int):
            ans_lens.append(al)
        # 简单关键词（英文/数字），如需中英文混合可在此处替换为更好的分词逻辑
        q = r.get("user_query") or ""
        for w in re.findall(r"[A-Za-z0-9]{3,}", q):
            word_counter[w.lower()] += 1

        # day bucket
        if isinstance(ts, (int, float)):
            day = _dt.datetime.utcfromtimestamp(ts/1000.0).strftime("%Y-%m-%d")
            per_day[day] += 1

    avg_dur = statistics.mean(durs) if durs else None
    p95_dur = (statistics.quantiles(durs, n=20)[18] if len(durs) >= 20 else (max(durs) if durs else None))
    avg_ans = statistics.mean(ans_lens) if ans_lens else None

    top_keywords = [{"term": k, "count": v} for k, v in word_counter.most_common(20)]
    series = [{"day": d, "count": c} for d, c in sorted(per_day.items())]

    return {
        "window_days": int(days),
        "total": total,
        "found": found,
        "nohit": nohit,
        "clarified": clarified,
        "avg_duration_ms": round(avg_dur, 2) if avg_dur is not None else None,
        "p95_duration_ms": round(p95_dur, 2) if p95_dur is not None else None,
        "avg_answer_len": round(avg_ans, 2) if avg_ans is not None else None,
        "top_keywords": top_keywords,
        "daily_series": series,
        "updated": now_ms
    }

@app.get("/feedback/metrics")
def feedback_metrics(days: int = 7, _auth=Depends(require_bearer)):
    """
    返回最近 N 天的 up/down 统计与 up-rate。days 默认 7。
    数据源：data/feedback/feedback.jsonl
    """
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - int(days) * 24 * 60 * 60 * 1000

    up = down = total = 0
    for rec in _iter_feedback_jsonl() or []:
        ts = rec.get("ts")  # 我们写入的是毫秒时间戳
        if isinstance(ts, (int, float)):
            if ts < cutoff_ms:
                continue
        # 无时间戳的老数据：可以选择纳入或跳过；这里纳入
        if rec.get("label") == "up":
            up += 1
        elif rec.get("label") == "down":
            down += 1
        total += 1

    uprate = (up / total) if total > 0 else 0.0
    return {
        "days": int(days),
        "up": up,
        "down": down,
        "total": total,
        "uprate": round(uprate, 4),
        "updated": now_ms
    }

@app.post("/feedback")
async def feedback_in(body: FeedbackIn, _auth=Depends(require_bearer)):
    rec = body.model_dump()
    rec["ts"] = int(time.time() * 1000)

    # 1) 先写主日志 & 聚合计数（与你现有逻辑一致）
    try:
        _append_jsonl(_FEEDBACK_PATH, rec)
        _bump_metrics(body.label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write feedback failed: {e}")

    # 2) 即时把这条反馈“转译”为页级 counts & penalty，立刻生效
    try:
        # 载入现有 counts，并得到一个累加器 inc()
        counts, inc = _bump_counts_and_penalty(rec.get("citations") or [])
        # 对本条反馈对应的所有 citation 做 up/down 增量
        fld = "up" if body.label == "up" else "down"
        for c in rec.get("citations") or []:
            file = c.get("file")
            page = c.get("page")
            if not file:
                continue
            key = f"{file}::{page}"
            # 初始化如果不存在
            if key not in counts:
                counts[key] = {"up": 0, "down": 0}
            counts[key][fld] = int(counts[key].get(fld, 0)) + 1

        # 写回 counts
        _write_json_atomic(_FB_COUNTS, counts)

        # 计算/写回 penalty.json —— rag.hybrid_retrieve 会在下一次检索时读取
        penalty = _recompute_penalty_from_counts(counts)
        _write_json_atomic(_FB_PENALTY, penalty)
    except Exception as e:
        # 不中断主流程；只是记录这次“即时惩罚更新”失败
        print(f"[WARN] feedback penalty update failed: {e}")

    return {"ok": True}

@app.middleware("http")
async def add_pna_header(request: Request, call_next):
    # 让 Chrome 的 Private Network Access 预检通过
    response: Response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_index: Index | None = None
_embedder: Embedder | None = None

def _extract_used_indices(answer_text: str, max_k: int) -> List[int]:
    """从答案正文里提取实际被使用的 [Source n]，去重并按首次出现顺序返回。"""
    seen = set()
    order: List[int] = []
    for m in _CITE_TAG_RE.finditer(answer_text or ""):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= max_k and n not in seen:
            seen.add(n)
            order.append(n)
    return order

def _citations_by_usage(answer_text: str, contexts: List[Dict]) -> List[Dict]:
    """只返回正文里实际使用到的 [Source n] 所对应的 citations。"""
    from .rag import format_citations
    all_cites = format_citations(contexts)  # 顺序与 [Source 1..K] 对应
    used = _extract_used_indices(answer_text, len(all_cites))
    return [all_cites[i - 1] for i in used]  # i 从 1 开始

def _sanitize_inline_citations(text: str, max_k: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_k else ""
    return _CITE_TAG_RE.sub(repl, text)

def get_index() -> Index:
    global _index
    if _index is None:
        idx = Index(settings.index_dir)
        idx.load()
        if idx.faiss is None:
            raise HTTPException(status_code=503, detail="Index not built yet. Run /ingest or scripts/ingest.py")
        _index = idx
    return _index

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.embedding_model_name, settings.embedding_device)
    return _embedder

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(_auth=Depends(require_bearer)):
    docs_count, chunks_count = ingest_corpus(settings.docs_dir, settings.index_dir)
    global _index, _embedder
    _index = None
    _embedder = None
    return IngestResponse(documents_indexed=docs_count, chunks_indexed=chunks_count)

def _extract_answer_text(data: dict) -> str:
    # 新增：处理 "answer" 是嵌套字符串 JSON 的情况
    if "answer" in data and isinstance(data["answer"], str):
        try:
            parsed = json.loads(data["answer"].replace("'", '"'))  # 兼容单引号
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed["response"]
        except Exception:
            pass  # fallback below

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        c0 = data["choices"][0]
        if isinstance(c0, dict) and "text" in c0 and isinstance(c0["text"], str):
            return c0["text"]
    for key in ("content", "answer", "message", "data"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for kk in ("content", "text", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv
    return str(data)

def _normalize_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    # 1) 解析 {"response": "..."} 或 {'response': '...'}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("response"), str):
            text = obj["response"]
    except Exception:
        m = re.match(r"\s*\{[^}]*'response'\s*:\s*'(.*?)'\s*\}\s*$", text, flags=re.S)
        if m:
            text = m.group(1)

    # 2) 去掉文末内嵌的 Sources 段
    text = re.sub(r"\n+Sources\s*\n(?:\[[^\n]+\].*\n?)+\s*$", "", text, flags=re.I)

    # 3) 统一换行，并把字面量 \\n 转成真正换行
    text = text.replace("\r\n", "\n")
    if "\\n" in text:
        text = text.replace("\\n", "\n")

    return text.strip()

def _extract_stream_token_preserve(text_line: str) -> str | None:
    line = text_line
    if not line:
        return None

    # 处理 SSE 前缀
    s = line.strip()
    if s.lower() == "[done]":
        return None
    if s.startswith("data:"):
        s = s[5:].strip()

    # 尝试解析 JSON；如果不是 JSON，直接返回原行（不改动）
    try:
        obj = json.loads(s)
    except Exception:
        return line  # 非 JSON，原样返回（保留其中换行/空格）

    # 在常见路径中取字符串值（原样返回）
    def get_path(o, path):
        cur = o
        for p in path:
            if isinstance(p, int):
                if isinstance(cur, list) and len(cur) > p:
                    cur = cur[p]
                else:
                    return None
            else:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
        return cur if isinstance(cur, str) else None

    candidates = [
        ("choices", 0, "delta", "content"),
        ("choices", 0, "text"),
        ("response",),
        ("content",),
        ("answer",),
        ("message", "content"),
        ("data", "content"),
    ]
    for path in candidates:
        val = get_path(obj, path)
        if isinstance(val, str):
            return val  # 原样返回，不做任何替换或去除

    # 没命中就把整行 JSON 丢弃（避免再把 JSON 文本回传前端）
    return None

def _extract_stream_piece(line: str) -> str:
    """
    从 SmartCare 流式每行里提取纯文本。
    兼容以下情况：
      - {"response": " ... "}
      - {"text": "..."} / {"content": "..."} / {"data": "..."}
      - 纯文本（直接返回）
      - SSE 风格的 'data: {...}'（可选兼容）
    """
    if not line:
        return ""

    # 1) 去掉可能的 SSE 前缀
    if line.startswith("data:"):
        line = line[len("data:"):].strip()

    # 2) 优先 JSON 解析
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            for key in ("response", "text", "content", "data"):
                val = obj.get(key)
                if isinstance(val, str):
                    return val
        # 如果是数组或其他结构，这里不处理
    except Exception:
        pass

    # 3) 兼容非严格 JSON，简单用正则兜底（例如 {'response': '...'}）
    m = re.search(r"'response'\s*:\s*'(.*?)'", line)
    if m:
        return m.group(1)

    # 4) 如果 line 不是 JSON，就当作纯文本（少见但安全）
    #    但要排除纯粹的空白/心跳
    if line.strip():
        return line
    return ""

def _not_found_text(lang: str | None) -> str:
    if lang == "zh-Hant":
        return ("抱歉，我在目前納入的社會福利文件中沒有找到你這個問題的具體答案。"
                "你可以嘗試：\n"
                "• 換一種說法或補充更具體的名詞（例如津貼名稱、服務單位）\n"
                "• 指定文件或年份（例如 2024 年 LSG Subvention Manual）\n"
                "如果需要，我可以幫你重述查詢或列出相關章節供你查閱。")
    return ("Sorry, I couldn't find a specific answer to this question in the indexed documents.\n"
            "You can try:\n"
            "• Rephrasing or adding more specific terms (e.g., the exact allowance name/service unit)\n"
            "• Mentioning a document or year (e.g., 2024 LSG Subvention Manual)\n"
            "If you like, I can help refine your query or show nearby sections.")

from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_rewrite_query
@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest, _auth=Depends(require_bearer)):
    idx = get_index()
    emb = get_embedder()

    # 1) 取得 messages（前端现在会带最近若干轮）
    msgs = [m.dict() if hasattr(m, "dict") else m for m in req.messages]

    # 2) 改写为独立问题（有历史时更有效）
    #    如果 messages 很短（只有一条 user），就直接用它
    if len(msgs) >= 2:
        rewritten = await smartcare_rewrite_query(msgs)
        user_query = rewritten
    else:
        user_query = msgs[-1]["content"]

    # ★ 規則兜底：若仍含代詞，嘗試補上最近實體
    if _PRONOUN_PAT.search(user_query):
        ent = _guess_entity_from_history(msgs)
        if ent:
            # 不粗暴替换用户原句，只给检索信号加注释
            user_query = f"{user_query} (about {ent})"
    
    # 3) 用改写后的独立问题进行检索（繁中：原文 + 英译 双通道合并）
    is_followup_pronoun = bool(_PRONOUN_PAT.search(msgs[-1]["content"]))

    if req.language == "zh-Hant":
        # 3.1 别名/俗称扩展（只影响检索，不改动原 messages）
        query_zh = _expand_aliases_zh(user_query)

        # 3.2 同时用繁中与英译检索
        try:
            query_en = await smartcare_translate_to_en(query_zh)
        except Exception:
            query_en = None

        hits_zh = hybrid_retrieve(query_zh, idx, emb, settings.top_k, soft=is_followup_pronoun)
        hits_en = hybrid_retrieve(query_en, idx, emb, settings.top_k, soft=True) if query_en else []

        # 3.3 合并去重（保留较高分，截到 top_k）
        contexts = _merge_dedup_hits(hits_zh, hits_en, settings.top_k)

        # （可选）把“实际用于检索的 query”记下来便于日志排查
        user_query = query_zh
    else:
        contexts = hybrid_retrieve(user_query, idx, emb, settings.top_k, soft=is_followup_pronoun)

    # 从“当前问句”和“上一个问句/回答”中抓一个焦点短语
    focus = (_extract_focus_phrase(msgs[-1]["content"]) or
            (len(msgs) >= 2 and _extract_focus_phrase(msgs[-2]["content"])) or
            _guess_entity_from_history(msgs))

    # 按焦点短语重排（把包含该短语的分片往前推）
    contexts = _boost_by_phrase(contexts, focus, boost=0.35)

    # === 先判斷是否找得到來源 ===
    if len(contexts) < settings.min_sources_required:
        text = _not_found_text(req.language)
        if req.stream:
            async def event_stream():
                yield text
                yield "\nCITATIONS:[]\n"
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            return ChatAnswer(answer=text, citations=[])
    
    # === 再檢查是否過於籠統（但已有來源） ===
    if _should_clarify_smart(user_query) and not _looks_specific(user_query, contexts):
        text = _clarify_question_smart(user_query, req.language, idx, emb)
        if req.stream:
            async def event_stream():
                yield text
                yield "\nCITATIONS:[]\n"
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            return ChatAnswer(answer=text, citations=[])

    # 4) 正常拼 prompt（把原 messages 发给模型，这样它能“按上下文口吻”回答）
    prompt_msgs = build_prompt(msgs, contexts)
    if req.language == "zh-Hant":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": (
                "請使用繁體中文回答所有問題，"
                "語氣親切、句子簡短，避免使用艱深詞彙，"
                "讓長者能容易明白。"
            )
        })
    elif req.language == "en":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": (
                "Please answer in clear, short English sentences suitable for older adults. "
                "Avoid jargon and keep the response friendly."
            )
        })

    try:
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": int(time.time() * 1000),
            "lang": req.language,
            "user_query": user_query,
            "context_hits": len(contexts),
            "answer_len": len(answer_text),
            "citations": citations,
            "clarified": _should_clarify_smart(user_query),
        }
        with open("data/logs/chat_usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 不影响主流程

    if req.stream:
        async def event_stream():
            buffer = ""
            async for line in smartcare_chat_stream(prompt_msgs):
                piece = _extract_stream_piece(line)
                if piece:
                    buffer += piece
                    yield piece
            clean_final = _sanitize_inline_citations(buffer, len(contexts))
            cites = _citations_by_usage(clean_final, contexts)
            trailer = "CITATIONS:" + json.dumps(cites, ensure_ascii=False)
            yield "\n" + trailer + "\n"
        # === 发送完毕后记录 usage 日志 ===
        try:
            _append_jsonl_safe(_USAGE_LOG, {
                "ts": _now_ms(),
                "lang": req.language,
                "duration_ms": None,
                "user_query": user_query,
                "clarified": _should_clarify_smart(user_query) and not _looks_specific(user_query, contexts),
                "context_hits": len(contexts),
                "found": len(contexts) >= settings.min_sources_required,
                "answer_len": len(clean_final or ""),
                "citations_cnt": len(cites or []),
                "stream": True
            })
        except Exception:
            pass

        return StreamingResponse(event_stream(), media_type="text/plain")

    data = await smartcare_chat(prompt_msgs)
    answer_text = _normalize_answer_text(_extract_answer_text(data))
    answer_text = _sanitize_inline_citations(answer_text, len(contexts))
    citations = _citations_by_usage(answer_text, contexts)

    try:
        _append_jsonl_safe(_USAGE_LOG, {
            "ts": _now_ms(),
            "lang": req.language,
            "duration_ms": None,
            "user_query": user_query,
            "clarified": _should_clarify_smart(user_query) and not _looks_specific(user_query, contexts),
            "context_hits": len(contexts),
            "found": len(contexts) >= settings.min_sources_required,
            "answer_len": len(answer_text or ""),
            "citations_cnt": len(citations or []),
            "stream": False
        })
    except Exception:
        pass

    return ChatAnswer(answer=answer_text, citations=citations)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

# ============================================================================
# File: app/rag.py
# ============================================================================
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
    # 轻量标准化
    text = _to_halfwidth(text)
    if _CJK.search(text):
        # 2-gram + 3-gram，适配繁中/简中
        s = re.sub(r"\s+", "", text)
        toks2 = [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else ([s] if s else [])
        toks3 = [s[i:i+3] for i in range(len(s)-2)] if len(s) >= 3 else []
        return toks2 + toks3
    # 英文/数字：保留原逻辑但更稳健的正则
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

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
    import re
    _CJK = re.compile(r"[\u4e00-\u9fff]")

    def _tokenize_q(q: str) -> List[str]:
        # 轻量规范：全角->半角，压缩空白
        def _to_halfwidth(s: str) -> str:
            out = []
            for ch in s:
                code = ord(ch)
                if code == 0x3000:  # 全角空格
                    code = 0x20
                elif 0xFF01 <= code <= 0xFF5E:
                    code -= 0xFEE0
                out.append(chr(code))
            return re.sub(r"\s+", " ", "".join(out)).strip()

        s = _to_halfwidth(q)
        if _CJK.search(s):
            s = s.replace(" ", "")
            toks2 = [s[i:i+2] for i in range(len(s)-1)] if len(s) >= 2 else ([s] if s else [])
            toks3 = [s[i:i+3] for i in range(len(s)-2)] if len(s) >= 3 else []
            return toks2 + toks3
        # 英文/数字：单词正则更稳
        return re.findall(r"[A-Za-z0-9_]+", s.lower())

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

# ============================================================================
# File: app/schemas.py
# ============================================================================
from pydantic import BaseModel
from typing import List, Optional, Literal

class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    language: Optional[Literal["en", "zh-Hant"]] = None

class Citation(BaseModel):
    file: str
    page: Optional[int] = None
    snippet: Optional[str] = None

class ChatAnswer(BaseModel):
    answer: str
    citations: List[Citation]

class FeedbackIn(BaseModel):
    threadId: str
    messageId: str
    label: Literal["up", "down"]
    userQuery: str = ""
    answer: str = ""
    language: Optional[Literal["en", "zh-Hant"]] = None
    citations: List[Citation] = []
    # 可选：一些检索与响应元数据，便于分析/自动调参
    meta: Optional[dict] = None

# ============================================================================
# File: app/security.py
# ============================================================================
import base64, os
from fastapi import HTTPException, Request
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .settings import settings

# --------- 访问控制 ----------
def require_bearer(request: Request):
    if not settings.require_auth:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    expected = (settings.api_bearer_token or "").strip()
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

# --------- 加解密 ----------
def _get_aesgcm():
    key_b64 = (settings.encryption_key_b64 or "").strip()
    if not key_b64:
        return None
    key = base64.urlsafe_b64decode(key_b64)
    return AESGCM(key)
        
def encrypt_bytes(plain: bytes) -> bytes:
    from secrets import token_bytes
    aes = _get_aesgcm()
    if settings.encrypt_data and aes is None:
        raise RuntimeError("ENCRYPT_DATA=true but no ENCRYPTION_KEY_B64 provided")
    if aes is None:
        return plain
    nonce = token_bytes(12)
    return nonce + aes.encrypt(nonce, plain, b"")

def decrypt_bytes(blob: bytes) -> bytes:
    aes = _get_aesgcm()
    if settings.encrypt_data and aes is None:
        raise RuntimeError("ENCRYPT_DATA=true but no ENCRYPTION_KEY_B64 provided")
    if aes is None:
        return blob
    if len(blob) < 13:
        raise ValueError("ciphertext too short")
    nonce, ct = blob[:12], blob[12:]
    return aes.decrypt(nonce, ct, b"")

# ============================================================================
# File: app/settings.py
# ============================================================================
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Paths
    docs_dir: str = "data/docs"
    index_dir: str = "data/index"

    # RAG chunking
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 5

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # set to "cuda" if available

    # HKUST LLM API
    smartcare_base_url: str = "https://smartlab.cse.ust.hk/smartcare/dev/llm_chat/"
    temperature: float = 0.0
    max_tokens: int = 1024

    # Feature flags
    enable_bm25: bool = True
    enable_query_rewrite: bool = True

    # Retrieval gating
    min_vec_sim: float = 0.35    # 余弦相似度(Inner Product, 归一化后)
    min_bm25_score: float = 4   # BM25 最小分（0~几十，语料而定）
    min_sources_required: int = 1  # 需要至少 N 个命中的来源才认为“找到了”

    # Security
    api_bearer_token: Optional[str] = None
    encryption_key_b64: Optional[str] = None
    allowed_origins: list[str] = ["http://localhost:5173/"]  # 生产建议白名单
    require_auth: bool = True  # 是否强制 Bearer
    encrypt_data: bool = True  # 是否对数据落盘加密

    # Pydantic v2 的配置写法（忽略额外环境变量；读取 .env）
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# ============================================================================
# File: app/utils.py
# ============================================================================
import re
from pathlib import Path
from typing import List, Dict

# —— 新：更鲁棒的多样式页码匹配 —— 
# 说明：
#  - 每个样式只有一个捕获组是页码数字；下面的 infer_page_map 会找出命中的那个分组
_PAGE_PATTERNS = [
    r"(?:^|\n)\s*Page\s*(\d+)\s*(?:\n|$)",            # Page 12
    r"(?:^|\n)\s*p\.\s*(\d+)\s*(?:\n|$)",             # p. 12
    r"(?:^|\n)\s*頁\s*(\d+)\s*(?:\n|$)",              # 頁 12
    r"(?:^|\n)\s*第\s*(\d+)\s*頁\s*(?:\n|$)",          # 第 12 頁
    r"(?:^|\n)\s*Page\s*(\d+)\s*of\s*\d+\s*(?:\n|$)", # Page 12 of 200
    # —— HTML / PDF 转 Markdown 常见锚点 —— 
    r"<span[^>]*\bid=['\"]page-(\d+)(?:-[^'\"]+)?['\"][^>]*>",   # <span id="page-30-0">
    r"<a[^>]*\b(?:id|name)=['\"]page-(\d+)['\"][^>]*>",          # <a id="page-30"> / <a name="page-30">
    r"<div[^>]*\bclass=['\"][^'\"]*\bpage\b[^'\"]*['\"][^>]*\bdata-page=['\"](\d+)['\"][^>]*>",  # <div class="page" data-page="30">
]
_PAGE_RE = re.compile("|".join(f"(?:{p})" for p in _PAGE_PATTERNS), re.IGNORECASE)

def read_markdown_files(docs_dir: str) -> List[Dict]:
    paths = list(Path(docs_dir).glob("**/*.md")) + list(Path(docs_dir).glob("**/*.markdown"))
    docs = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": str(p), "text": text})
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    return docs

def infer_page_map(md_text: str) -> List[Dict]:
    matches = list(_PAGE_RE.finditer(md_text))
    if not matches:
        return [{"page": None, "start": 0, "end": len(md_text)}]

    def first_int_group(m: re.Match) -> int | None:
        # 由于我们是一个大 OR，groups() 里只有一个是数字，其它是 None
        for g in m.groups():
            if g and g.isdigit():
                return int(g)
        return None

    pages = []
    for i, m in enumerate(matches):
        page_no = first_int_group(m)
        if page_no is None:
            # 理论不会发生；防御性处理
            continue
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
        pages.append({"page": page_no, "start": start, "end": end})
    return pages

# ============================================================================
# File: scripts/analyze_usage.py
# ============================================================================
from pathlib import Path
import json, collections, re

LOG = Path("data/logs/chat_usage.jsonl")
if not LOG.exists():
    print("No log file found.")
    raise SystemExit

qs = []
clar = cnt_nohit = 0
for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    qs.append(r["user_query"])
    if r.get("context_hits", 0) == 0: cnt_nohit += 1
    if r.get("clarified"): clar += 1

# 统计最常见的问题关键词
words = re.findall(r"\b[A-Za-z]{3,}\b", " ".join(qs))
freq = collections.Counter(words).most_common(20)

print("Top 20 keywords:", freq)
print(f"Total Qs: {len(qs)} | Clarifications: {clar} | No hits: {cnt_nohit}")

# ============================================================================
# File: scripts/build_penalty.py
# ============================================================================
from pathlib import Path
import json, collections

# 让路径相对于项目根目录，而不是当前脚本文件夹
ROOT = Path(__file__).resolve().parent.parent
FB = ROOT / "data/feedback/feedback.jsonl"
OUT = ROOT / "data/feedback/penalty.json"

cnt = collections.Counter()
if not FB.exists():
    print(f"[WARN] Feedback file not found: {FB}")
    raise SystemExit(0)

for line in FB.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception as e:
        print(f"[WARN] bad line: {e}")
        continue
    if r.get("label") != "down":
        continue
    for c in r.get("citations", []):
        key = (c.get("file"), c.get("page"))
        cnt[key] += 1

# 阈值：>=3 次 down 记入惩罚
pen = {f"{k[0]}::{k[1]}": min(1.0, 0.15 + 0.05*v) for k,v in cnt.items() if v >= 3}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(pen, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] Penalized pages: {len(pen)} → {OUT}")

# ============================================================================
# File: scripts/ingest.py
# ============================================================================
from app.rag import ingest_corpus
from app.settings import settings

if __name__ == "__main__":
    docs, chunks = ingest_corpus(settings.docs_dir, settings.index_dir)
    print(f"Indexed documents: {docs}, chunks: {chunks}")

# ============================================================================
# File: scripts/summarize_usage.py
# ============================================================================
from pathlib import Path
import json, re, collections, statistics, datetime as dt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/logs/chat_usage.jsonl"
OUT = ROOT / "data/logs/summary.json"

if not LOG.exists():
    print("[WARN] usage log not found:", LOG)
    raise SystemExit(0)

total = clarified = nohit = 0
durs = []
ans_lens = []
kw = collections.Counter()
per_day = collections.Counter()

for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.strip(): 
        continue
    try:
        r = json.loads(line)
    except:
        continue
    total += 1
    if r.get("clarified"): clarified += 1
    if not r.get("found"): nohit += 1
    dur = r.get("duration_ms")
    if isinstance(dur, (int, float)) and dur >= 0: durs.append(float(dur))
    al = r.get("answer_len")
    if isinstance(al, int): ans_lens.append(al)
    q = r.get("user_query") or ""
    for w in re.findall(r"[A-Za-z0-9]{3,}", q):
        kw[w.lower()] += 1
    ts = r.get("ts")
    if isinstance(ts, (int, float)):
        day = dt.datetime.utcfromtimestamp(ts/1000.0).strftime("%Y-%m-%d")
        per_day[day] += 1

summary = {
    "total": total,
    "clarified": clarified,
    "nohit": nohit,
    "avg_duration_ms": (statistics.mean(durs) if durs else None),
    "p95_duration_ms": (statistics.quantiles(durs, n=20)[18] if len(durs) >= 20 else (max(durs) if durs else None)),
    "avg_answer_len": (statistics.mean(ans_lens) if ans_lens else None),
    "top_keywords": kw.most_common(50),
    "daily_series": sorted(per_day.items()),
    "generated_at": int(dt.datetime.utcnow().timestamp() * 1000)
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {OUT}")

# ============================================================================
# File: scripts/tune_thresholds.py
# ============================================================================
import json, statistics, os, re
from pathlib import Path

FB = Path("data/feedback/feedback.jsonl")
ENV = Path(".env")

def load_feedback():
    if not FB.exists(): return []
    for line in FB.read_text(encoding="utf-8").splitlines():
        try: yield json.loads(line)
        except: pass

def window(feeds, days=7):
    # 简化：不按时间窗口也可，先全量
    return list(feeds)

def compute_uprate(items):
    up = sum(1 for x in items if x.get("label")=="up")
    down = sum(1 for x in items if x.get("label")=="down")
    return up / max(1, up+down)

def rewrite_env(**kv):
    old = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    for k,v in kv.items():
        if f"{k}=" in old:
            old = re.sub(rf"^{k}=.*$", f"{k}={v}", old, flags=re.M)
        else:
            old += f"\n{k}={v}"
    ENV.write_text(old, encoding="utf-8")

if __name__ == "__main__":
    data = list(window(load_feedback(), days=7))
    rate = compute_uprate(data)
    # 读取当前阈值（也可用 settings）
    min_vec = float(os.getenv("MIN_VEC_SIM", "0.40"))
    min_bm25 = float(os.getenv("MIN_BM25_SCORE", "5"))
    # 简单策略
    if rate < 0.7:
        min_vec = min(min_vec + 0.02, 0.60)
        min_bm25 = min_bm25 + 0.5
    elif rate > 0.8:
        min_vec = max(min_vec - 0.02, 0.30)
        min_bm25 = max(min_bm25 - 0.5, 3.0)
    # 写回 .env（下次进程重启或热加载读取）
    rewrite_env(MIN_VEC_SIM=min_vec, MIN_BM25_SCORE=min_bm25)
    print({"up_rate": rate, "MIN_VEC_SIM": min_vec, "MIN_BM25_SCORE": min_bm25})

# ============================================================================
# File: requirements.txt
# ============================================================================
# FastAPI & server
fastapi==0.115.2
uvicorn[standard]==0.30.6
httpx==0.27.2
cryptography==43.0.3

# RAG
faiss-cpu==1.8.0.post1
rank-bm25==0.2.2
sentence-transformers==3.0.1
numpy==1.26.4
pydantic==2.9.2
pydantic-settings==2.6.1

# ============================================================================
# File: README.md
# ============================================================================
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

## Notes
- You should better run ```scripts/tune_thresholds.py``` once per day to improve the performance of chat robot.
- Citations list the file name and page if detectable from markdown. If your MD lacks page markers, the backend still cites the file.
- For better page/snippet support, extend `infer_page_map` to parse your MD structure (e.g., headings) and store character spans for snippets.
- To switch to GPU embeddings, set `EMBEDDING_DEVICE=cuda` and ensure CUDA is available.
- You can later add caching, rate limiting, and feedback logging.
