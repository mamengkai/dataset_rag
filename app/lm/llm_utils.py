from typing import Optional

from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

from app.conf.lm_config import lm_config
from app.core.logger import logger

# 缓存键：(模型名, json_mode)
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """对话模型客户端，默认使用 lm_config.llm_model。"""
    target_model = model or lm_config.llm_model
    if not target_model:
        raise ValueError("[LLM客户端] 未配置对话模型：请在.env中设置 LLM_DEFAULT_MODEL")
    return _get_chat_client(target_model, json_mode)


def get_vl_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """多模态（视觉）模型客户端，默认使用 lm_config.vl_model。"""
    target_model = model or lm_config.vl_model
    if not target_model:
        raise ValueError("[VL客户端] 未配置视觉模型：请在.env中设置 VL_MODEL")
    return _get_chat_client(target_model, json_mode)


def _get_chat_client(target_model: str, json_mode: bool) -> ChatOpenAI:
    cache_key = (target_model, json_mode)
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置 OPENAI_API_KEY")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置 OPENAI_BASE_URL")

    logger.info(f"[LLM客户端] 初始化：模型={target_model}，JSON模式={json_mode}")

    extra_body = {}
    if "qwen" in target_model.lower():
        extra_body["enable_thinking"] = False

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body or None,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败：{str(e)}") from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 已缓存：模型={target_model}，JSON模式={json_mode}")
    return llm_client


if __name__ == "__main__":
    logger.info("===== 开始执行LLM客户端工具测试 =====")
    try:
        chat_client = get_llm_client()
        logger.info(f"对话模型：{chat_client.model_name}")

        vl_client = get_vl_client()
        logger.info(f"多模态模型：{vl_client.model_name}")

        chat_again = get_llm_client()
        logger.info(f"对话缓存命中：{chat_client is chat_again}")

        json_client = get_llm_client(json_mode=True)
        logger.info(f"JSON模式为独立实例：{json_client is not chat_client}")
    except Exception as e:
        logger.error(f"LLM客户端工具测试失败：{str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM客户端工具测试结束 =====")
