import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import embed_documents
from app.utils.task_utils import add_done_task, add_running_task

BATCH_SIZE = 20


def _build_embed_text(chunk: dict) -> str:
    """核心词前置，便于检索时命中商品名。"""
    item_name = (chunk.get("item_name") or "").strip()
    item_content = (chunk.get("content") or "").strip()
    if item_name and item_content:
        return f"商品：{item_name}，内容介绍：{item_content}"
    if item_content:
        return item_content
    if item_name:
        return f"商品：{item_name}"
    return ""


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding) 将 Chunk 文本转为稠密向量。
    使用 OpenAI 兼容 Embedding（见 EMBEDDING_* / get_embedding_client）
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点：{function_name}")
    add_running_task(state.get("task_id", ""), function_name)

    try:
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValueError("chunks数据无效，检查数据格式")

        final_chunks = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch_items = chunks[start: start + BATCH_SIZE]
            current_texts = [_build_embed_text(item) for item in batch_items]

            empty_indexes = [idx for idx, text in enumerate(current_texts) if not text]
            if empty_indexes:
                raise ValueError(
                    f"批次起始={start} 存在空文本 chunk，索引={empty_indexes}，无法向量化"
                )

            vectors = embed_documents(current_texts)
            if len(vectors) != len(batch_items):
                raise RuntimeError(
                    f"Embedding 返回数量与批次不一致：期望={len(batch_items)}，实际={len(vectors)}"
                )

            for chunk, dense_vector in zip(batch_items, vectors):
                chunk_item = chunk.copy()
                chunk_item["dense_vector"] = dense_vector
                final_chunks.append(chunk_item)

        state["chunks"] = final_chunks
        state["embeddings_content"] = final_chunks
        logger.info(f">>> [{function_name}] 完成向量化，共 {len(final_chunks)} 条")
        return state
    except Exception as e:
        logger.opt(exception=True).error(">>> 执行节点报错: {}，异常信息如下: {}", function_name, e)
        raise
    finally:
        logger.info(f">>> 执行节点结束: {function_name}")
        add_done_task(state.get("task_id"), function_name)
