import base64
import mimetypes
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.llm_utils import get_vl_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task, add_running_task

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def is_supported_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _image_data_url(image_path: str, image_base64: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return f"data:{mime};base64,{image_base64}"


def step_1_get_content(state) -> Tuple[str, Path, Path]:
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md_path不能为空")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path}文件不存在")

    md_content = state.get("md_content") or ""
    if not md_content:
        with md_path_obj.open(mode="r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content

    images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_dir_obj


def find_image_in_md_content(
        md_content: str,
        image_file: str,
        context_length: int = 100,
) -> Optional[Tuple[str, str]]:
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
    match = pattern.search(md_content)
    if not match:
        return None

    start, end = match.span()
    pre_text = md_content[max(start - context_length, 0):start]
    post_text = md_content[end:min(end + context_length, len(md_content))]
    logger.info(f"当前图片：{image_file} 上下文：{(pre_text, post_text)}")
    return pre_text, post_text


def step_2_scan_images(md_content, images_dir_obj) -> List[Tuple[str, str, Tuple[str, str]]]:
    targets = []
    for image_file in os.listdir(images_dir_obj):
        if not is_supported_image(image_file):
            logger.info(f"当前文件: {image_file} 不是图片格式，无需处理")
            continue

        context_data = find_image_in_md_content(md_content, image_file)
        if not context_data:
            logger.info(f"图片: {image_file}没有在md内容中使用，上下文为空")
            continue
        targets.append((image_file, str(images_dir_obj / image_file), context_data))
    return targets


def step_3_generate_img_summaries(targets, stem) -> dict:
    summaries = {}
    if not targets:
        return summaries

    vl_model = get_vl_client()
    request_times = deque()
    for image_file, image_path, context in targets:
        apply_api_rate_limit(request_times, max_requests=100)
        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path, image_base64)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = vl_model.invoke(messages)
        summary = (response.content or "").strip().replace("\n", "")
        if not summary:
            logger.warning(f"图片 {image_file} 视觉模型未返回正文，跳过")
            continue
        summaries[image_file] = summary

    return summaries


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img) 处理 Markdown 中的图片资源 (Image)。
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id")
    logger.info(f">>> 执行节点: {function_name},当前状态: {state}")
    add_running_task(task_id, function_name)

    try:
        md_content, md_path_obj, images_dir_obj = step_1_get_content(state)

        if not images_dir_obj.exists():
            logger.info(f">>> [{function_name}]没有图片，直接返回state")
            state["image_summaries"] = {}
            return state

        targets = step_2_scan_images(md_content, images_dir_obj)
        summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)
        state["image_summaries"] = summaries
        # todo 上传图片至minio、更新md内容  数据最终处理和备份
        return state
    except Exception as e:
        logger.error(f">>> 执行节点错误: {function_name},异常信息: {e}")
        raise
    finally:
        add_done_task(task_id, function_name)


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hl3040网络说明书", "hl3040网络说明书.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "image_summaries": {},
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
