from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import shutil, json, time, os, tempfile
import asyncio
from typing import List, Dict, Any

from .settings import settings
from .rag import ingest_corpus, ingest_file_incremental, remove_file_from_index
from .security import require_bearer
from .ingest_manager import start as ingest_start, cancel as ingest_cancel, status as ingest_status
from .utils import SUPPORTED_EXTENSIONS


router = APIRouter(prefix="/docs", tags=["docs"])

DOCS_DIR = Path(settings.docs_dir)
STATUS_PATH = Path("data/status.json")
TMP_DIR = Path("data/tmp_uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件格式（用于上传验证）
ALLOWED_EXTENSIONS = {'.md', '.markdown', '.pdf', '.txt', '.docx'}

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
    # 收集所有支持格式的文件
    all_files = []
    for ext in ALLOWED_EXTENSIONS:
        all_files.extend(DOCS_DIR.glob(f"*{ext}"))

    for p in sorted(set(all_files)):
        items.append({
            "filename": p.name,
            "size": p.stat().st_size,
            "modified": int(p.stat().st_mtime),
            "format": p.suffix.lower().lstrip('.'),
        })
    return {"docs": items}

@router.get("/supported-formats")
def get_supported_formats():
    """返回支持的文件格式列表"""
    return {
        "formats": list(ALLOWED_EXTENSIONS),
        "descriptions": {
            ".md": "Markdown document",
            ".markdown": "Markdown document",
            ".pdf": "PDF document",
            ".txt": "Plain text file",
            ".docx": "Microsoft Word document"
        }
    }

@router.post("/upload", dependencies=[Depends(require_bearer)])
async def upload_doc(file: UploadFile = File(...), incremental: bool = None):
    """
    上传文档到知识库。

    参数:
    - file: 要上传的文件
    - incremental: 是否使用增量索引（可选，默认由settings控制）
    """
    name = (file.filename or "")
    suffix = Path(name).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Unsupported file format. Supported formats: {supported}")

    tmp_path = TMP_DIR / (file.filename + ".part")
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    dest = DOCS_DIR / file.filename
    tmp_path.replace(dest)  # 原子移动

    # 确定是否使用增量索引
    use_incremental = incremental if incremental is not None else settings.enable_incremental_index

    # ===== 状态：indexing =====
    mode_note = "incremental" if use_incremental else "full rebuild"
    _write_status({"status": "indexing", "note": f"uploaded: {file.filename} ({mode_note})", "start_ts": int(time.time())})

    if use_incremental:
        # 增量索引：只索引新上传的文件
        chunks_added, was_updated = await asyncio.to_thread(
            ingest_file_incremental, str(dest), settings.index_dir
        )
        action = "updated" if was_updated else "added"
        message = f"{file.filename} {action}. {chunks_added} chunks indexed (incremental)."
    else:
        # 全量重建：重新索引所有文件
        await asyncio.to_thread(ingest_corpus, settings.docs_dir, settings.index_dir)
        message = f"{file.filename} uploaded. Index rebuilt (full)."

    # 关键：失效内存索引缓存
    from . import main as _main
    _main._index = None
    _main._embedder = None

    # ===== 状态：ready =====
    _write_status({"status": "ready", "note": f"uploaded: {file.filename}", "last_built": int(time.time())})
    return {"ok": True, "message": message}

@router.delete("/{filename}", dependencies=[Depends(require_bearer)])
async def delete_doc(filename: str):
    # 基础校验，阻止路径穿越
    if "/" in filename or "\\" in filename:
        raise HTTPException(400, "Bad filename")

    # 构建所有可能的候选路径（支持多种格式）
    candidates = [DOCS_DIR / filename]
    base_name = Path(filename).stem
    for ext in ALLOWED_EXTENSIONS:
        candidates.append(DOCS_DIR / (base_name + ext))

    # 如果都不存在，尝试在目录里做一次"大小写不敏感"/近似匹配
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        low = filename.lower()
        for p in DOCS_DIR.glob("*"):
            p_lower = p.name.lower()
            if p_lower == low:
                path = p
                break
            # 检查是否匹配任何支持的扩展名
            for ext in ALLOWED_EXTENSIONS:
                if p_lower == (low + ext) or p_lower == (base_name.lower() + ext):
                    path = p
                    break
            if path:
                break

    # 记录文件路径（用于增量删除）
    file_path_to_remove = str(path) if path else None

    # 能找到就删，找不到也继续重建（保证索引与磁盘一致）
    if path and path.exists():
        path.unlink()

    # 确定是否使用增量索引
    use_incremental = settings.enable_incremental_index

    # —— 状态：indexing ——
    mode_note = "incremental" if use_incremental else "full rebuild"
    _write_status({"status": "indexing", "note": f"deleted: {filename} ({mode_note})", "start_ts": int(time.time())})

    try:
        if use_incremental and file_path_to_remove:
            # 增量删除：只从索引中删除该文件
            await asyncio.to_thread(remove_file_from_index, file_path_to_remove, settings.index_dir)
            message = f"{filename} deleted. Index updated (incremental)."
        else:
            # 全量重建：重新索引所有文件
            await asyncio.to_thread(ingest_corpus, settings.docs_dir, settings.index_dir)
            message = f"{filename} deleted (if existed). Index rebuilt (full)."

        # 关键：失效内存索引缓存
        from . import main as _main
        _main._index = None
        _main._embedder = None

        # —— 状态：ready ——
        _write_status({"status": "ready", "note": f"deleted: {filename}", "last_built": int(time.time())})
        return {"ok": True, "message": message}
    except Exception as e:
        # —— 状态：error（可在前端提示）——
        _write_status({"status": "error", "note": f"deleted: {filename}: {e}", "ts": int(time.time())})
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
        
@router.post("/cancel", dependencies=[Depends(require_bearer)])
def cancel_reindex():
    killed = ingest_cancel()
    return {"ok": True, "killed": killed}