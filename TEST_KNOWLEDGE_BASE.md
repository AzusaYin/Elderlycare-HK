# 知识库系统测试指南

这是一个独立的知识库测试工具，可以测试PDF和Markdown文档的处理、索引构建和检索功能。

## 📦 安装依赖

首先确保安装了所有必需的依赖：

```bash
pip install -r requirements.txt
```

或者单独安装：

```bash
pip install PyMuPDF faiss-cpu rank-bm25 sentence-transformers numpy
```

## 🚀 快速开始

### 1. 运行演示（推荐）

最简单的方式是运行内置演示：

```bash
python test_knowledge_base.py demo
```

这将：
- ✅ 创建示例文档
- ✅ 构建知识库索引
- ✅ 运行示例查询
- ✅ 展示完整工作流程

### 2. 上传你的PDF文件

```bash
# 上传PDF文件（自动构建索引）
python test_knowledge_base.py upload your_document.pdf

# 上传Markdown文件
python test_knowledge_base.py upload your_notes.md

# 上传但不自动构建索引
python test_knowledge_base.py upload document.pdf --no-rebuild
```

### 3. 查询知识库

```bash
# 基本查询
python test_knowledge_base.py query "香港安老服务有哪些？"

# 返回更多结果
python test_knowledge_base.py query "申请资格" -k 10

# 英文查询
python test_knowledge_base.py query "What are the elderly care services?"
```

### 4. 列出所有文档

```bash
python test_knowledge_base.py list
```

### 5. 重建索引

```bash
python test_knowledge_base.py build
```

## 📁 文件结构

测试工具会创建以下目录结构：

```
test_data/
├── docs/          # 存储上传的文档
│   ├── sample.md
│   ├── document1.pdf
│   └── document2.pdf
└── index/         # 存储索引文件
    ├── faiss.index    # FAISS向量索引
    ├── meta.json      # 文档元数据
    └── bm25.json      # BM25关键词索引
```

## 🔧 核心功能

### PDF处理
- ✅ 自动提取PDF文本
- ✅ 保留页码信息
- ✅ 支持中英文混合
- ✅ 进度显示

### 文本分块
- ✅ 智能句子感知分块
- ✅ 中文标点符号识别
- ✅ 可配置块大小和重叠

### 混合检索
- ✅ FAISS向量检索（语义相似度）
- ✅ BM25关键词检索
- ✅ 自适应融合（中英文不同权重）

### 索引管理
- ✅ 保存/加载索引
- ✅ 增量更新
- ✅ 元数据追踪

## 🎯 使用示例

### 示例1: 测试PDF处理

```bash
# 1. 准备一个PDF文件
wget https://example.com/sample.pdf

# 2. 上传并处理
python test_knowledge_base.py upload sample.pdf

# 3. 查询测试
python test_knowledge_base.py query "文档中的关键信息"
```

### 示例2: 测试Markdown处理

```bash
# 1. 创建测试文件
cat > test.md << 'EOF'
Page 1
# 测试文档

这是一个测试文档，包含中文和English混合内容。

## 第一部分
内容描述...

Page 2
## 第二部分
更多内容...
EOF

# 2. 上传处理
python test_knowledge_base.py upload test.md

# 3. 查询
python test_knowledge_base.py query "第一部分"
```

### 示例3: 批量测试

```bash
# 上传多个文档
python test_knowledge_base.py upload doc1.pdf --no-rebuild
python test_knowledge_base.py upload doc2.md --no-rebuild
python test_knowledge_base.py upload doc3.pdf --no-rebuild

# 统一构建索引
python test_knowledge_base.py build

# 查询
python test_knowledge_base.py query "查询内容" -k 5
```

## 🧪 测试场景

### 1. PDF文本提取测试
```bash
python test_knowledge_base.py upload sample.pdf
# 检查输出中的页码和字符数
```

### 2. 中文分词测试
```bash
python test_knowledge_base.py query "香港長者服務"
python test_knowledge_base.py query "申請資格要求"
```

### 3. 混合语言测试
```bash
python test_knowledge_base.py query "Hong Kong elderly services 申请"
```

### 4. 页码定位测试
```bash
python test_knowledge_base.py query "第一章" -k 3
# 检查结果中的页码信息
```

## 📊 性能参考

| 操作 | 典型耗时 |
|------|---------|
| PDF提取 (10页) | ~2秒 |
| 首次加载模型 | ~5秒 |
| 构建索引 (100个分块) | ~10秒 |
| 单次查询 | <1秒 |

## ⚙️ 配置参数

在 `test_knowledge_base.py` 中可以修改以下参数：

```python
# 文档处理
CHUNK_SIZE = 1500       # 分块大小（字符）
CHUNK_OVERLAP = 200     # 重叠大小（字符）

# 检索配置
TOP_K = 5               # 默认返回结果数

# 模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## 🐛 故障排除

### 问题1: PyMuPDF导入失败
```bash
# 解决方法
pip install --upgrade PyMuPDF
```

### 问题2: FAISS安装失败
```bash
# macOS/Linux
pip install faiss-cpu

# Windows
conda install -c pytorch faiss-cpu
```

### 问题3: 模型下载慢
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
python test_knowledge_base.py demo
```

### 问题4: 内存不足
```python
# 修改批处理大小
# 在代码中找到并修改：
BATCH = 256  # 默认512，可减小
```

## 📝 输出说明

### 上传输出示例
```
📄 正在提取PDF: sample.pdf
   ✓ 已提取第 1/5 页
   ✓ 已提取第 2/5 页
   ...
✅ PDF提取完成: 共 5 页, 12580 字符

📚 开始构建索引 (共 1 个文档)...
   📖 sample.pdf: 5 页
   ✂️  分块完成: 15 个片段
   🧮 生成嵌入向量...
   🔍 构建FAISS索引...
   📊 构建BM25索引...
✅ 索引构建完成!
```

### 查询输出示例
```
🔎 搜索: 香港安老服务有哪些？

📊 找到 5 个相关片段:

================================================================================

【结果 1】 (得分: 0.8756)
文件: sample.pdf
页码: 1

内容预览:
香港特别行政区为长者提供多元化的安老服务，包括：
1. 社区照顾服务
   - 长者中心
   - 日间护理中心
...
```

## 🔗 相关文档

- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
- [FAISS 文档](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [BM25 算法](https://en.wikipedia.org/wiki/Okapi_BM25)

## 💡 提示

1. **首次运行**会下载嵌入模型（~100MB），需要等待
2. **PDF质量**影响提取效果，扫描版PDF需要OCR
3. **分块大小**影响检索精度，可根据文档特点调整
4. **混合检索**对中文查询特别有效

## 🎉 完整测试流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行演示（验证环境）
python test_knowledge_base.py demo

# 3. 上传你的PDF
python test_knowledge_base.py upload your_document.pdf

# 4. 测试查询
python test_knowledge_base.py query "你的问题"

# 5. 查看所有文档
python test_knowledge_base.py list

# 6. 完成！
```

祝测试顺利！🚀
