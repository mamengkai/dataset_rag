from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    dim: int


embedding_config = EmbeddingConfig(
    base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B"),
    dim=int(os.getenv("EMBEDDING_DIM", "4096")),
)
