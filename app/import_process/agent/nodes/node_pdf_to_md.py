import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config


def step_1_validate_paths(state: ImportGraphState):
    """
    路径校验
    pdf失效，抛异常
    local_dir缺失，给予默认值
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths] - 文件格式校验, 传入参数: {state}")

    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.error(f">>> [step_1_validate_paths] - 未检测到输入文件，无法继续解析！")
        raise ValueError("[step_1_validate_paths] - 未检测到输入文件，无法继续解析！")

    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        logger.info(f">>> [step_1_validate_paths] - 检测到local_dir未赋值，给予默认值: {local_dir}")

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f">>> [step_1_validate_paths] - 检测到pdf_path不存在，请检查输入文件路径是否正确！")
        raise FileNotFoundError(f"[step_1_validate_paths] - 检测到pdf_path不存在，请检查输入文件路径是否正确！")

    if not local_dir_obj.exists():
        logger.info(f">>> [step_1_validate_paths] - 检测到local_dir不存在，将主动创建对应文件夹！")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj) -> str:
    """
    将pdf文件使用minerU解析，并获取md对应的下载url地址
    :param pdf_path_obj: 上传解析pdf文件的path对象
    :return: url， minerU解析后md文件zip压缩包的下载地址
    """
    # 申请上传解析的地址
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200 or response.json()['code'] != 0:
        logger.error(f">>> [step_2_upload_and_poll] - 请求minerU接口获取上传地址失败")
        raise RuntimeError(f"[step_2_upload_and_poll] - 请求minerU接口获取上传地址失败，请重试！")
    upload_url = response.json()['data']['file_urls'][0]
    batch_id = response.json()['data']['batch_id']

    # 将文件上传到对应的解析地址
    http_session = requests.Session()
    http_session.trust_env = False  # 禁止走代理
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(upload_url, data=file_data)
        if upload_response.status_code != 200:
            logger.error(f">>> [step_2_upload_and_poll] - 请求minerU接口上传文件失败")
            raise RuntimeError(f"[step_2_upload_and_poll] - 请求minerU接口上传文件失败，请重试！")
    except Exception as e:
        logger.error(f">>> [step_2_upload_and_poll] - 请求minerU接口上传文件失败")
        raise RuntimeError(f"[step_2_upload_and_poll] - 请求minerU接口上传文件失败，请重试！")
    finally:
        http_session.close()

    # 轮询获取结果
    timeout_seconds = 600
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    poll_interval = 3  # 间隔时间
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            logger.error(f">>> [step_2_upload_and_poll] - 请求minerU接口获取解析后文件超时")
            raise TimeoutError("[step_2_upload_and_poll] - 请求minerU接口获取解析后文件超时，请重试！")

        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll] - minerU服务器异常，返回状态码{res.status_code}")

        json_data = res.json()
        if json_data['code'] != 0:
            raise RuntimeError(
                f"[step_2_upload_and_poll] - minerU服务器异常，返回错误{json_data['code']}信息{json_data['msg']}")

        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            # 解析完成，可以获取到结果
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"已经完成pdf解析，耗时： {time.time() - start_time}秒，解析结果：{full_zip_url}")
            return full_zip_url
        else:
            # 还没有解析完成
            time.sleep(poll_interval)


def step_3_download_and_extract(zip_url, local_dir_obj, stem) -> str:
    """
    下载并解压指定md.zip文件
    :param zip_url: 要下载的地址
    :param local_dir_obj: 存储的文件夹
    :param stem: pdf的文件名称
    :return: 解压后的md文件地址
    """
    # 下载zip文件
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_download_and_extract] - 下载文件失败，状态码{response.status_code}，文件下载地址{zip_url}")
        raise RuntimeError("[step_3_download_and_extract] - 下载文件失败，请重试!")

    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"[step_3_download_and_extract] - 下载文件成功，文件保存地址{zip_save_path}")

    # 清空旧目录
    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)

    # 创建新目录
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    # 对zip进行解压
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_object:
        zip_file_object.extractall(extract_target_dir)

    # 返回md文件的地址
    md_file_list = list(extract_target_dir.rglob('*.md'))
    target_md_file: Path = None  # 存储最终md文件

    if not md_file_list:
        logger.error(f"[step_3_download_and_extract] - 未找到md文件")
        raise RuntimeError("[step_3_download_and_extract] - 未找到md文件!")

    for md_file in md_file_list:
        if md_file.name == stem + ".md":
            target_md_file = md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    if not target_md_file:
        target_md_file = md_file_list[0]

    # 统一命名
    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step_3_download_and_extract] - 完成md文件解压，最终存储md文件路径：{final_md_str_path}")
    return final_md_str_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md) 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    function_name = sys._getframe().f_code.co_name
    task_id = state.get("task_id")
    logger.info(f">>> 执行节点: {function_name},当前状态: {state}")
    add_running_task(task_id, function_name)

    try:
        # 校验
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)

        # 调用minerU
        zip_url = step_2_upload_and_poll(pdf_path_obj)

        # 下载zip，解析，提取
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)

        # 更新数据，读取md文件内容到md_content
        state["md_path"] = md_path
        state['local_dir'] = str(local_dir_obj)
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()

    except Exception as e:
        logger.error(f">>> 执行节点错误: {function_name},异常信息: {e}")
        raise
    else:
        logger.info(f">>> 执行节点结束: {function_name},当前状态: {state}")
        return state
    finally:
        add_done_task(task_id, function_name)


if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
