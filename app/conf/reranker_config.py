from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    base_url: str
    api_key: str
    model: str


reranker_config = RerankerConfig(
    base_url=os.getenv("RERANK_BASE_URL") or os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("RERANK_API_KEY") or os.getenv("EMBEDDING_API_KEY"),
    model=os.getenv("RERANK_MODEL", "Qwen3-Reranker-8B"),
)
