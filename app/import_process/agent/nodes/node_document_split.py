import json
import os.path
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 单个Chunk最大字符长度，超过则二次切分
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500


def step_1_get_content(state):
    """
    读取要切片的内容
    :param state:
    :return:
    """
    md_content = state.get("md_content")
    if not md_content:
        logger.error(f"[step_1_get_content] 无有效md内容")
        raise Exception("检查输入文件路径是否正确")

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "default")
    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    """
    语义切割，根据标题进行切割
    :param md_content:
    :param file_title:
    :return:
    """
    title_pattern = r'^\s*#{1,6}\s+.+'
    lines = md_content.split("\n")

    current_title = ""
    current_lines = []
    title_count = 0
    is_code_block = False
    sections = []

    for line in lines:
        strip = line.strip()

        # 判断是否是代码块
        if strip.startswith('```') or strip.startswith('~~~'):
            is_code_block = not is_code_block
            current_lines.append(line)
            continue

        # 判断是否是标题
        is_title = (not is_code_block) and re.match(title_pattern, strip)
        if is_title:
            if current_title:
                # 已有上一节：先落盘再开新节
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            elif any(line.strip() for line in current_lines):
                # 第一个标题之前的正文：单独成「前言」节，避免丢失
                sections.append({
                    "title": "前言",
                    "content": "\n".join(current_lines).strip(),
                    "file_title": file_title
                })

            current_title = strip
            current_lines = [current_title]
            title_count += 1
        else:
            # 不是标题
            current_lines.append(line)

    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    elif any(line.strip() for line in current_lines):
        # 全文无标题但有正文：整篇作为前言（与 title_count==0 兜底互补）
        sections.append({
            "title": "前言",
            "content": "\n".join(current_lines).strip(),
            "file_title": file_title
        })

    return sections, title_count, len(lines)


def split_long_section(section, max_length):
    """
    将当前chunk内容超长的进行二次切割
    :param section:
    :param max_length:
    :return: 切割后的[{}, {}]
    """
    content = section.get("content")
    if len(content) <= max_length:
        logger.info(f"[split_long_section]: {content}当前chunk长度小于{max_length}, 不做切割")
        return [section]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=['\n\n', '\n', '。', '，', '!', '；', ' ']
    )

    sub_sections = []
    for index, chunk in enumerate(splitter.split_text(content), start=1):
        text = chunk.strip()
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part
        })

    return sub_sections


def merge_short_sections(final_sections, min_length):
    """
    如果上次切得太碎需要合并
    1. content长度小于min_length
    2. 同一个parent_title才可以合并
    :param final_sections:
    :param min_length:
    :return:
    """
    merged_sections = []  # 存储合并结果
    pre_section = None  # 当前处理的块

    for section in final_sections:
        if pre_section is None:
            pre_section = section
            continue

        is_current_short = len(pre_section.get("content")) < min_length
        is_same_parent_title = pre_section.get("parent_title") and (section.get("parent_title") == pre_section.get("parent_title"))

        if is_current_short and is_same_parent_title:
            current_content = section.get("content")
            pre_section['content'] += "\n\n" + current_content
            pre_section['part'] = section.get("part")
        else:
            merged_sections.append(pre_section)
            pre_section = section

    if pre_section is not None:
        merged_sections.append(pre_section)

    return merged_sections


def step_3_refine_chunks(sections, max_length, min_length):
    """
    内容精细切割
    1. 超过了MIN_CONTENT_LENGTH需要切割
    2. 小于了MIN_CONTENT_LENGTH需要合并
    :param min_length:
    :param max_length:
    :param sections:
    :return: sections
    """
    final_sections = []  # 存储处理后的块

    # 超过的先截断
    for section in sections:
        sub_section = split_long_section(section, max_length)
        final_sections.extend(sub_section)

    # 小的要合并
    final_sections = merge_short_sections(final_sections, min_length)

    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')

    return final_sections


def step_4_backup_chunks(state, sections):
    """
    将切割玩的碎片进行存储
    :param state:
    :param sections: 要存储的内容
    :return:
    """
    local_dir = state.get("local_dir")
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            sections,
            f,
            ensure_ascii=False,
            indent=4,
        )

    logger.info(f"内容已备份至: {backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split) 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id")
    logger.info(f">>> 执行节点: {function_name},当前状态: {state}")
    add_running_task(state.get("task_id"), function_name)

    try:
        # 参数校验
        md_content, file_title = step_1_get_content(state)

        # 粗粒度切割
        sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)

        # 特殊场景，如果一个文档没有标题
        if title_count == 0:
            sections = [{"title": "没有主题", "content": md_content, "file_title": file_title}]

        # 细粒度切割
        sections = step_3_refine_chunks(sections, DEFAULT_MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH)

        state['chunks'] = sections
        step_4_backup_chunks(state, sections)

    except Exception as e:
        logger.error(f">>> 执行节点报错: {function_name}, 异常信息如下: {e}")
        raise
    finally:
        logger.info(f">>> 执行节点结束: {function_name}, 当前状态: {state}")
        add_done_task(state.get("task_id"), function_name)

    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")