import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry) 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """

    # 进入节点日志输出，记录任务状态
    function_name = sys._getframe().f_code.co_name

    logger.info(f">>> 执行节点: {function_name},当前状态: {state}")
    task_id = state.get("task_id") or ""
    add_running_task(task_id, function_name)

    try:
        local_file_path = state.get("local_file_path")
        if not local_file_path:
            logger.error(f">>> 执行节点错误: {function_name}, 没有输入文件，无法继续解析!")
            return state

        suffix = Path(local_file_path).suffix.lower()
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = local_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = local_file_path
        else:
            logger.error(f">>> 执行节点错误: {function_name}, 文件格式错误，无法继续解析!")
            return state

        state["file_title"] = Path(local_file_path).stem
        logger.info(f">>> 执行节点结束: {function_name},当前状态: {state}")
        return state
    finally:
        add_done_task(task_id, function_name)
