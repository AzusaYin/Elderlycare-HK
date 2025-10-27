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
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
    min_bm25_score: float = 3.5   # BM25 最小分（0~几十，语料而定）
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