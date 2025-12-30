import re
from pathlib import Path
from typing import List, Dict
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

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

def read_pdf_file(pdf_path: Path) -> str:
    """
    从PDF文件中提取文本，保留页码标记
    使用PyMuPDF (fitz)提取，每页前添加 "Page N" 标记
    """
    if not HAS_PDF:
        raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")

    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            # 在每页开头添加页码标记（与现有的页码推断模式兼容）
            pages_text.append(f"Page {page_num + 1}\n{text}")

        doc.close()
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF {pdf_path}: {e}")
        raise


def read_markdown_files(docs_dir: str) -> List[Dict]:
    """
    读取文档目录中的所有Markdown和PDF文件
    返回格式: [{"path": "...", "text": "..."}, ...]
    """
    # 收集Markdown文件
    md_paths = list(Path(docs_dir).glob("**/*.md")) + list(Path(docs_dir).glob("**/*.markdown"))
    # 收集PDF文件
    pdf_paths = list(Path(docs_dir).glob("**/*.pdf")) if HAS_PDF else []

    docs = []

    # 处理Markdown文件
    for p in md_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": str(p), "text": text})
        except Exception as e:
            print(f"[WARN] Failed to read Markdown {p}: {e}")

    # 处理PDF文件
    for p in pdf_paths:
        try:
            text = read_pdf_file(p)
            docs.append({"path": str(p), "text": text})
        except Exception as e:
            print(f"[WARN] Failed to read PDF {p}: {e}")

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