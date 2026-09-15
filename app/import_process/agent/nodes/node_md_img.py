import base64
import mimetypes
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.llm_utils import get_vl_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task, add_running_task

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def is_supported_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _guess_image_mime(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        return "image/jpeg"
    return mime


def _image_data_url(image_path: str, image_base64: str) -> str:
    return f"data:{_guess_image_mime(image_path)};base64,{image_base64}"


def _public_image_url(object_name: str) -> str:
    scheme = "https" if minio_config.minio_secure else "http"
    return f"{scheme}://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"


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


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
    图片上传到minio服务器，替换原md中图片和描述
    :param summaries: 图片名: 描述
    :param targets: (图片名, 原地址, (上文, 下文))
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md
    """
    minio_client = get_minio_client()
    if minio_client is None:
        raise RuntimeError("MinIO 客户端未初始化，请检查 MINIO_* 配置及服务是否可用")

    prefix = f"{minio_config.minio_img_dir}/{stem}"
    object_list = minio_client.list_objects(
        minio_config.bucket_name,
        prefix=prefix,
        recursive=True,
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_object_list)
    for error in errors:
        logger.error(
            f"清空 MinIO 对象失败: bucket={minio_config.bucket_name}, "
            f"object={error.name}, code={error.code}, message={error.message}"
        )
    logger.info(f"已经完成{stem}下对象清空，本次删除了：{len(delete_object_list)}个文件")

    images_url = {}
    for image_file, image_path, _ in targets:
        object_name = f"{minio_config.minio_img_dir}/{stem}/{image_file}"
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=image_path,
                content_type=_guess_image_mime(image_path),
            )
            images_url[image_file] = _public_image_url(object_name)
            logger.info(f"完成图片上传: {image_file}, 访问地址为: {images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片失败: {image_file}, 失败原因: {e}")
            raise RuntimeError(f"上传图片失败: {image_file}") from e

    image_infos = {}
    for image_file, summary in summaries.items():
        url = images_url.get(image_file)
        if not url:
            raise RuntimeError(f"图片 {image_file} 已有摘要但缺少上传地址，中止替换以免混入本地路径")
        image_infos[image_file] = (summary, url)
    logger.info(f"图片处理汇总结果: {image_infos}")

    if image_infos:
        for image_file, (summary, url) in image_infos.items():
            rep = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = rep.sub(f"![{summary}]({url})", md_content)
        logger.info(f"完成md内容替换，新的内容为: {md_content}")

    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    新md的磁盘备份
    :param new_md_content: 新内容
    :param md_path_obj: 旧地址
    :return: 新地址
    """
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"
    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"完成新内容写入，新的地址为: {new_md_path_str}")

    return new_md_path_str


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

        new_md_content = step_4_upload_images_and_replace_md(summaries, targets, md_content, md_path_obj.stem)

        new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)

        state["md_path"] = new_md_file_path
        state["md_content"] = new_md_content

        return state
    except Exception as e:
        logger.error(f">>> 执行节点错误: {function_name},异常信息: {e}")
        raise
    finally:
        add_done_task(task_id, function_name)


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
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
