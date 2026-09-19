from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from app.conf.embedding_config import embedding_config
from app.core.logger import logger

# 缓存键：模型名
_embedding_client_cache = {}


def get_embedding_client(model: Optional[str] = None) -> OpenAIEmbeddings:
    """
    仅稠密向量，维度见 embedding_config.dim（如 Qwen3-Embedding-8B = 4096）。
    """
    target_model = model or embedding_config.model
    if not target_model:
        raise ValueError("[Embedding客户端] 未配置模型：请在.env中设置 EMBEDDING_MODEL")
    if target_model in _embedding_client_cache:
        logger.debug(f"[Embedding客户端] 缓存命中：模型={target_model}")
        return _embedding_client_cache[target_model]

    if not embedding_config.api_key:
        raise ValueError("[Embedding客户端] 配置缺失：请在.env中配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY")
    if not embedding_config.base_url:
        raise ValueError("[Embedding客户端] 配置缺失：请在.env中配置 EMBEDDING_BASE_URL 或 OPENAI_BASE_URL")

    logger.info(
        f"[Embedding客户端] 初始化：模型={target_model}，"
        f"配置dim={embedding_config.dim}（仅作校验，不传给接口），"
        f"base_url={embedding_config.base_url}"
    )

    # 兼容非 OpenAI 网关：发原文而非 token id，强制 float 编码。
    # 不传 dimensions：网关会把该参数当成 Matryoshka 降维并 400 拒绝。
    client = OpenAIEmbeddings(
        model=target_model,
        api_key=embedding_config.api_key,
        base_url=embedding_config.base_url,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    _embedding_client_cache[target_model] = client
    return client


def embed_documents(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """批量文本向量化。"""
    if not texts:
        return []
    return get_embedding_client(model).embed_documents(texts)


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """单条查询向量化。"""
    return get_embedding_client(model).embed_query(text)


if __name__ == "__main__":
    logger.info("===== 开始 Embedding 客户端测试 =====")
    try:
        vec = embed_query("烫金机电源线安全注意事项")
        logger.info(f"向量维度={len(vec)}，期望={embedding_config.dim}")
        if len(vec) != embedding_config.dim:
            logger.warning(
                f"返回维度与 EMBEDDING_DIM 不一致：实际={len(vec)}，配置={embedding_config.dim}"
            )
    except Exception as e:
        logger.opt(exception=True).error("Embedding 客户端测试失败：{}", e)
    finally:
        logger.info("===== Embedding 客户端测试结束 =====")
