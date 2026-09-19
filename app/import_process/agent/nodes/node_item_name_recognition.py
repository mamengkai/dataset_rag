import os.path
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import embed_documents
from app.lm.llm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

DEFAULT_ITEM_NAME_CHUNK_K = 5
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_chunks(state):
    """
    获取chunks和file_title
    :param state:
    :return:
    """
    chunks = state.get('chunks')
    file_title = state.get('file_title')

    if not chunks:
        raise ValueError("chunks is None")
    if not file_title:
        # 从md_path中获取文件名
        file_title = os.path.basename(state.get('md_path'))
        logger.info(f"file_title缺失，获取md_path中文件名称{file_title}")
        state['file_title'] = file_title

    return chunks, file_title


def step_2_build_context(chunks):
    """
    根据 chunks 切片的 content 拼接上下文（最多取前 K 个）。
    单片截断到 SINGLE_CHUNK_CONTENT_MAX_LEN，总长不超过 CONTEXT_TOTAL_MAX_CHARS。
    """
    parts = []
    total_chars = 0

    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        title = chunk.get("title") or ""
        content = (chunk.get("content") or "")[:SINGLE_CHUNK_CONTENT_MAX_LEN]
        data = f"切片: {index}, 标题: {title}, 内容: {content}"
        if total_chars + len(data) > CONTEXT_TOTAL_MAX_CHARS and parts:
            break
        parts.append(data)
        total_chars += len(data)
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            break

    return "\n\n".join(parts)[:CONTEXT_TOTAL_MAX_CHARS]


def step_3_call_llm(context, file_title):
    """
    调用模型获取 item_name，失败或空结果时用 file_title 兜底。
    """
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")

    llm = get_llm_client(json_mode=False)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)
    item_name = (response.content or "").strip()
    if not item_name:
        return file_title
    return item_name


def step_4_update_chunks_and_state(state, item_name, chunks):
    state["item_name"] = item_name

    for chunk in chunks:
        chunk["item_name"] = item_name

    state["chunks"] = chunks
    logger.info("完成chunks和state[item_name]赋值和修改")


def step_5_generate_embeddings(item_name):
    """
    根据 item_name 生成稠密向量。
    """
    dense_vector = embed_documents([item_name])[0]
    logger.info(f"已根据item_name生成dense_vector，维度={len(dense_vector)}")
    return dense_vector


def step_6_save_to_vector_db(file_title, item_name, dense_vector):
    """
    将向量和对应字段写入 Milvus item_name 集合。
    """
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus 未连接，请检查 MILVUS_URL 与服务是否启动")

    collection = milvus_config.item_name_collection
    if not collection:
        raise RuntimeError("未配置 ITEM_NAME_COLLECTION")

    if not milvus_client.has_collection(collection_name=collection):
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=embedding_config.dim,
        )

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        milvus_client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
        )

    milvus_client.load_collection(collection_name=collection)
    safe_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=collection,
        filter=f'item_name == "{safe_name}"',
    )

    milvus_client.insert(
        collection_name=collection,
        data=[{
            "item_name": item_name,
            "file_title": file_title,
            "dense_vector": dense_vector,
        }],
    )
    logger.info(f"保存了item_name: {item_name}的数据到向量数据库中")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition) 识别文档核心描述的物品/商品名称 (Item Name)。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 验证取值
        chunks, file_title = step_1_get_chunks(state)

        # 构建上下文环境 chunks -> top 5 -> 拼接成context文本
        context = step_2_build_context(chunks)

        # 调用模型，拼接提示词，识别chunks对应item_name
        item_name = step_3_call_llm(context, file_title)

        # 修改state chunks -> item_name
        step_4_update_chunks_and_state(state, item_name, chunks)

        # item_name生成向量
        dense_vector = step_5_generate_embeddings(item_name)

        # 将向量存储到向量数据库中
        step_6_save_to_vector_db(file_title, item_name, dense_vector)

    except Exception as e:
        logger.opt(exception=True).error(">>> [{}] 执行异常，异常信息: {}", function_name, e)
        raise
    finally:
        logger.info(f">>> [{function_name}] 执行结束，当前状态为: {state}")
        add_done_task(state['task_id'], function_name)

    return state


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Embedding、Milvus 服务可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.opt(exception=True).error("商品名称识别节点本地测试失败，原因：{}", e)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()