import sys
from pathlib import Path
from typing import Tuple, List

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


def step1_get_content(state) -> Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    md_file_path = state.get("md_path")
    if not md_file_path:
        raise ValueError("md_path不能为空")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path}文件不存在")

    md_content = state.get("md_content") or ""
    if not md_content:
        # 没有再读取；从 pdf 节点过来时 content 已经赋值
        with md_path_obj.open(mode="r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content

    images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_dir_obj


def step2_scan_images(md_content, images_dir_obj) -> List[Tuple[str, str, Tuple[str, str]]]:
    pass


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img) 处理 Markdown 中的图片资源 (Image)。
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id")
    logger.info(f">>> 执行节点: {function_name},当前状态: {state}")
    add_running_task(task_id, function_name)

    # 校验并获取本次操作的数据
    md_content, md_path_obj, images_dir_obj = step1_get_content(state)

    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}]没有图片，直接返回state")
        return state

    # 识别md中使用过的图片，进行图片总结
    targets = step2_scan_images(md_content, images_dir_obj)

    return state
