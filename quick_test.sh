#!/bin/bash
# 知识库快速测试脚本

set -e

echo "🚀 知识库系统快速测试"
echo "======================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo ""

# 检查并安装依赖
echo "📦 检查依赖..."
if ! python3 -c "import fitz" 2>/dev/null; then
    echo "   安装 PyMuPDF..."
    pip install PyMuPDF
fi

if ! python3 -c "import numpy, faiss, sentence_transformers" 2>/dev/null; then
    echo "   安装 RAG 依赖..."
    pip install numpy faiss-cpu rank-bm25 sentence-transformers
fi

echo "✅ 依赖检查完成"
echo ""

# 运行演示
echo "🎯 运行知识库演示..."
echo "======================="
python3 test_knowledge_base.py demo

echo ""
echo "✅ 测试完成！"
echo ""
echo "📝 后续步骤:"
echo "   1. 上传你的PDF: python test_knowledge_base.py upload your.pdf"
echo "   2. 查询知识库: python test_knowledge_base.py query '你的问题'"
echo "   3. 查看文档: python test_knowledge_base.py list"
echo ""
echo "📖 完整文档: TEST_KNOWLEDGE_BASE.md"
