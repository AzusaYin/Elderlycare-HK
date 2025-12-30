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
    # 列出所有支持的文档格式：Markdown和PDF
    all_files = (
        list(DOCS_DIR.glob("*.md")) +
        list(DOCS_DIR.glob("*.markdown")) +
        list(DOCS_DIR.glob("*.pdf"))
    )
    for p in sorted(all_files):
        items.append({
            "filename": p.name,
            "size": p.stat().st_size,
            "modified": int(p.stat().st_mtime),
        })
    return {"docs": items}

@router.post("/upload", dependencies=[Depends(require_bearer)])
async def upload_doc(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".md") or name.endswith(".markdown") or name.endswith(".pdf")):
        raise HTTPException(400, "Only Markdown (.md/.markdown) and PDF (.pdf) are supported")
    
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
        DOCS_DIR / (filename if filename.lower().endswith(".pdf") else filename + ".pdf"),
    ]

    # 如果都不存在，尝试在目录里做一次"大小写不敏感"/近似匹配
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        low = filename.lower()
        for p in DOCS_DIR.glob("*"):
            if (p.name.lower() == low or
                p.name.lower() == (low + ".md") or
                p.name.lower() == (low + ".markdown") or
                p.name.lower() == (low + ".pdf")):
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