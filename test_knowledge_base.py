#!/usr/bin/env python3
"""
知识库系统测试工具
支持PDF和Markdown文档的上传、索引构建和检索测试

使用方法:
    python test_knowledge_base.py --help
    python test_knowledge_base.py upload sample.pdf
    python test_knowledge_base.py query "香港安老服务有哪些？"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# PDF处理
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("⚠️  警告: PyMuPDF未安装，无法处理PDF文件")
    print("   安装命令: pip install PyMuPDF")

# RAG依赖
try:
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    HAS_RAG = True
except ImportError as e:
    HAS_RAG = False
    print(f"⚠️  警告: RAG依赖未安装: {e}")
    print("   安装命令: pip install faiss-cpu rank-bm25 sentence-transformers numpy")


# ============ 配置 ============
TEST_DOCS_DIR = Path("test_data/docs")
TEST_INDEX_DIR = Path("test_data/index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 5


# ============ PDF处理 ============
def extract_text_from_pdf(pdf_path: Path) -> str:
    """从PDF文件中提取文本，保留页码标记"""
    if not HAS_PDF:
        raise ImportError("PyMuPDF未安装，无法处理PDF文件")

    print(f"📄 正在提取PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages_text.append(f"Page {page_num + 1}\n{text}")
        print(f"   ✓ 已提取第 {page_num + 1}/{len(doc)} 页")

    doc.close()
    full_text = "\n\n".join(pages_text)
    print(f"✅ PDF提取完成: 共 {len(doc)} 页, {len(full_text)} 字符\n")
    return full_text


# ============ 文档读取 ============
def read_documents(docs_dir: Path) -> List[Dict]:
    """读取目录中的所有Markdown和PDF文件"""
    if not docs_dir.exists():
        print(f"❌ 文档目录不存在: {docs_dir}")
        return []

    # 收集文件
    md_files = list(docs_dir.glob("**/*.md")) + list(docs_dir.glob("**/*.markdown"))
    pdf_files = list(docs_dir.glob("**/*.pdf")) if HAS_PDF else []

    docs = []

    # 处理Markdown
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": str(p), "filename": p.name, "text": text, "type": "markdown"})
            print(f"📝 读取Markdown: {p.name} ({len(text)} 字符)")
        except Exception as e:
            print(f"⚠️  读取失败 {p.name}: {e}")

    # 处理PDF
    for p in pdf_files:
        try:
            text = extract_text_from_pdf(p)
            docs.append({"path": str(p), "filename": p.name, "text": text, "type": "pdf"})
        except Exception as e:
            print(f"⚠️  读取失败 {p.name}: {e}")

    return docs


# ============ 文本分块 ============
def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """简单的字符级分块"""
    step = max(1, chunk_size - overlap)
    n = len(text)
    if n == 0:
        return []
    return [text[i:i+chunk_size] for i in range(0, n, step)]


_SENT_SPLIT = re.compile(r"[。！？；：]\s*")

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """中文友好的句子感知分块"""
    parts = []
    segs = [s for s in _SENT_SPLIT.split(text) if s]
    for seg in segs:
        parts.extend(simple_chunk(seg, chunk_size, overlap))
    if not segs:
        parts = simple_chunk(text, chunk_size, overlap)
    return parts


# ============ 页码推断 ============
_PAGE_PATTERNS = [
    r"(?:^|\n)\s*Page\s*(\d+)\s*(?:\n|$)",
    r"(?:^|\n)\s*p\.\s*(\d+)\s*(?:\n|$)",
    r"(?:^|\n)\s*頁\s*(\d+)\s*(?:\n|$)",
    r"(?:^|\n)\s*第\s*(\d+)\s*頁\s*(?:\n|$)",
]
_PAGE_RE = re.compile("|".join(f"(?:{p})" for p in _PAGE_PATTERNS), re.IGNORECASE)

def infer_pages(text: str) -> List[Dict]:
    """推断文本中的页码信息"""
    matches = list(_PAGE_RE.finditer(text))
    if not matches:
        return [{"page": None, "start": 0, "end": len(text)}]

    def first_int(m):
        for g in m.groups():
            if g and g.isdigit():
                return int(g)
        return None

    pages = []
    for i, m in enumerate(matches):
        page_no = first_int(m)
        if page_no is None:
            continue
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        pages.append({"page": page_no, "start": start, "end": end})
    return pages


# ============ 索引构建 ============
class KnowledgeBase:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if not HAS_RAG:
            raise ImportError("RAG依赖未安装")

        print(f"🤖 加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.bm25 = None
        self.bm25_tokens = []

    def build_index(self, docs: List[Dict]):
        """构建知识库索引"""
        print(f"\n📚 开始构建索引 (共 {len(docs)} 个文档)...")

        all_chunks = []
        for doc in docs:
            pages = infer_pages(doc["text"])
            print(f"   📖 {doc['filename']}: {len(pages)} 页")

            for page_info in pages:
                page_text = doc["text"][page_info["start"]:page_info["end"]]
                pieces = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

                for i, piece in enumerate(pieces):
                    all_chunks.append({
                        "text": piece,
                        "file": doc["filename"],
                        "page": page_info["page"],
                        "chunk_id": i
                    })

        print(f"   ✂️  分块完成: {len(all_chunks)} 个片段")

        if not all_chunks:
            print("❌ 没有可索引的内容")
            return

        self.chunks = all_chunks
        texts = [c["text"] for c in all_chunks]

        # 生成嵌入
        print("   🧮 生成嵌入向量...")
        embeddings = np.array(self.model.encode(texts, normalize_embeddings=True))

        # 构建FAISS索引
        print("   🔍 构建FAISS索引...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        # 构建BM25索引
        print("   📊 构建BM25索引...")
        self.bm25_tokens = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

        print(f"✅ 索引构建完成!\n")

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词（英文词 + 中文2-gram）"""
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())

        # 中文2-gram
        cjk = re.compile(r"[\u4e00-\u9fff]")
        if cjk.search(text):
            cjk_chars = "".join(ch for ch in text if cjk.match(ch))
            if len(cjk_chars) >= 2:
                tokens += [cjk_chars[i:i+2] for i in range(len(cjk_chars)-1)]

        return tokens

    def search(self, query: str, k: int = TOP_K) -> List[Dict]:
        """混合检索：向量检索 + BM25"""
        if self.index is None:
            print("❌ 索引未构建")
            return []

        print(f"\n🔎 搜索: {query}")

        # 向量检索
        q_emb = np.array(self.model.encode([query], normalize_embeddings=True))
        D, I = self.index.search(q_emb.astype(np.float32), min(k * 2, len(self.chunks)))

        vec_hits = {int(I[0][i]): float(D[0][i]) for i in range(len(I[0]))}

        # BM25检索
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens) if self.bm25 else []
        bm25_hits = {i: float(s) for i, s in enumerate(bm25_scores)} if bm25_scores.size > 0 else {}

        # 融合分数
        is_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
        alpha_vec = 0.60 if is_cjk else 0.90
        alpha_bm25 = 0.40 if is_cjk else 0.10

        combined = {}
        all_ids = set(vec_hits.keys()) | set(bm25_hits.keys())

        for idx in all_ids:
            vec_score = vec_hits.get(idx, 0.0)
            bm25_score = bm25_hits.get(idx, 0.0)
            combined[idx] = alpha_vec * vec_score + alpha_bm25 * bm25_score

        # 排序并返回
        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in sorted_ids:
            chunk = self.chunks[idx]
            results.append({
                "score": score,
                "text": chunk["text"],
                "file": chunk["file"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            })

        return results

    def save(self, index_dir: Path):
        """保存索引到磁盘"""
        index_dir.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(index_dir / "faiss.index"))

        # 保存元数据
        with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # 保存BM25
        with open(index_dir / "bm25.json", "w", encoding="utf-8") as f:
            json.dump({"tokens": self.bm25_tokens}, f, ensure_ascii=False)

        print(f"💾 索引已保存到: {index_dir}")

    def load(self, index_dir: Path):
        """从磁盘加载索引"""
        if not index_dir.exists():
            print(f"❌ 索引目录不存在: {index_dir}")
            return False

        # 加载FAISS索引
        self.index = faiss.read_index(str(index_dir / "faiss.index"))

        # 加载元数据
        with open(index_dir / "meta.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 加载BM25
        with open(index_dir / "bm25.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.bm25_tokens = data["tokens"]
            self.bm25 = BM25Okapi(self.bm25_tokens)

        print(f"📂 索引已加载: {len(self.chunks)} 个片段")
        return True


# ============ 命令行接口 ============
def cmd_upload(args):
    """上传文档到测试知识库"""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    # 创建测试目录
    TEST_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 复制文件
    import shutil
    dest = TEST_DOCS_DIR / file_path.name
    shutil.copy(file_path, dest)
    print(f"✅ 文件已上传: {dest}\n")

    # 重建索引
    if args.rebuild:
        cmd_build(args)


def cmd_build(args):
    """构建或重建索引"""
    docs = read_documents(TEST_DOCS_DIR)
    if not docs:
        print("❌ 没有找到任何文档")
        return

    kb = KnowledgeBase()
    kb.build_index(docs)
    kb.save(TEST_INDEX_DIR)


def cmd_query(args):
    """查询知识库"""
    kb = KnowledgeBase()

    # 尝试加载现有索引
    if not kb.load(TEST_INDEX_DIR):
        print("\n❌ 索引不存在，请先上传文档并构建索引")
        print("   命令: python test_knowledge_base.py upload <your_file.pdf>")
        return

    # 执行查询
    results = kb.search(args.query, k=args.top_k)

    # 显示结果
    print(f"\n📊 找到 {len(results)} 个相关片段:\n")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        print(f"\n【结果 {i}】 (得分: {r['score']:.4f})")
        print(f"文件: {r['file']}")
        if r['page']:
            print(f"页码: {r['page']}")
        print(f"\n内容预览:\n{r['text'][:300]}...")
        print("-" * 80)


def cmd_list(args):
    """列出所有文档"""
    docs = read_documents(TEST_DOCS_DIR)

    if not docs:
        print("📂 文档目录为空")
        return

    print(f"\n📚 共 {len(docs)} 个文档:\n")
    for doc in docs:
        size_kb = len(doc['text']) / 1024
        print(f"  • {doc['filename']} ({doc['type']}) - {size_kb:.1f} KB")


def cmd_demo(args):
    """运行完整演示"""
    print("=" * 80)
    print("🎯 知识库系统演示")
    print("=" * 80)

    # 检查示例文件
    if not TEST_DOCS_DIR.exists() or not list(TEST_DOCS_DIR.iterdir()):
        print("\n📝 创建示例文档...")
        TEST_DOCS_DIR.mkdir(parents=True, exist_ok=True)

        sample_text = """Page 1
香港安老服务概述

香港特别行政区为长者提供多元化的安老服务，包括：

1. 社区照顾服务
   - 长者中心
   - 日间护理中心
   - 家居照顾服务

2. 院舍照顾服务
   - 安老院
   - 护理安老院
   - 护养院

Page 2
申请资格

申请安老服务的基本资格：
- 年龄：60岁或以上
- 居住：香港永久性居民
- 需要：经社工评估确认需要服务

查询热线：2343 2255
网址：www.swd.gov.hk
"""

        (TEST_DOCS_DIR / "sample.md").write_text(sample_text, encoding="utf-8")
        print("   ✓ 已创建示例文档: sample.md")

    # 构建索引
    print("\n" + "=" * 80)
    class Args:
        rebuild = True
    cmd_build(Args())

    # 示例查询
    print("\n" + "=" * 80)
    print("🔍 示例查询")
    print("=" * 80)

    queries = [
        "香港安老服务有哪些？",
        "申请资格是什么？",
        "查询热线"
    ]

    for q in queries:
        class QueryArgs:
            query = q
            top_k = 2
        cmd_query(QueryArgs())
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="知识库系统测试工具 - 支持PDF和Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传PDF文件并自动构建索引
  python test_knowledge_base.py upload document.pdf

  # 上传Markdown文件
  python test_knowledge_base.py upload notes.md

  # 查询知识库
  python test_knowledge_base.py query "香港安老服务有哪些？"

  # 列出所有文档
  python test_knowledge_base.py list

  # 运行完整演示
  python test_knowledge_base.py demo
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # upload命令
    upload_parser = subparsers.add_parser("upload", help="上传文档")
    upload_parser.add_argument("file", help="文件路径 (.pdf 或 .md)")
    upload_parser.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                              help="不自动重建索引")

    # build命令
    build_parser = subparsers.add_parser("build", help="构建索引")

    # query命令
    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument("query", help="查询内容")
    query_parser.add_argument("-k", "--top-k", type=int, default=TOP_K,
                             help=f"返回结果数量 (默认: {TOP_K})")

    # list命令
    list_parser = subparsers.add_parser("list", help="列出所有文档")

    # demo命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 执行命令
    commands = {
        "upload": cmd_upload,
        "build": cmd_build,
        "query": cmd_query,
        "list": cmd_list,
        "demo": cmd_demo
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
