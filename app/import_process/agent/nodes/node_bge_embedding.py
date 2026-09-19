import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding) 将 Chunk 文本转为稠密向量。

    使用 OpenAI 兼容 Embedding（见 EMBEDDING_* / get_embedding_client），
    不再加载本地 BGE-M3；当前方案仅 dense，无 sparse。
    todo:
    1. 调用 embed_documents 对每个 Chunk 生成稠密向量（dim=EMBEDDING_DIM）。
    2. 组装写入 Milvus 的数据结构（仅 dense_vector）。
    3. 写入 state["embeddings_content"]。
    """
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    return state