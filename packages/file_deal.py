import base64
from datetime import datetime
import os
from pathlib import Path
import pickle as pk

import numpy as np
import pandas as pd


from packages.my_logger import get_logger

logger = get_logger()


def get_output_dir(input_file):
    """
    从输入文件路径中提取最后两级目录作为输出目录名
    """
    dir_path = os.path.dirname(input_file)
    parent_dir = os.path.basename(dir_path)
    filename = os.path.splitext(os.path.basename(input_file))[0]
    timestamp_short = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(
        "api_results", f"{timestamp_short}_parallel_results_{parent_dir}_{filename}"
    )


def get_current_script_dir():
    """
    获取脚本相关路径
    """
    script_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_abs_path)
    evaluation_dir = os.path.dirname(os.path.dirname(script_dir))
    api_test_script = os.path.join(
        evaluation_dir, "code/02_llm_ability_eval/get_foodname_parallel_by_api.py"
    )

    logger.info(f"脚本所在的绝对路径: {script_dir}")
    logger.info(f"evaluation目录: {evaluation_dir}")
    logger.info(f"API测试脚本路径: {api_test_script}")

    return script_dir, evaluation_dir, api_test_script


def get_latest_file(directory):
    """
    获取目录下最新的文件
    """
    files = list(Path(directory).glob("*"))
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    return str(max(files, key=os.path.getmtime))


def encode_image(image_path):
    """
    base 64 编码格式
    """
    with open(image_path, "rb") as image_file:
        content = image_file.read()
        return base64.b64encode(content).decode("utf-8")


def encode_image_with_resize(image_path, max_pixels=800 * 600):
    """
    base 64 编码格式，带缩放，默认480k像素
    """
    from PIL import Image
    from io import BytesIO

    # 打开图像并调整大小
    with Image.open(image_path) as img:
        # 获取原始尺寸
        width, height = img.size
        original_pixels = width * height
        if original_pixels > max_pixels:
            scale_factor = (max_pixels / original_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # 如果图片是RGBA模式，转换为RGB
        if img.mode == "RGBA":
            # 创建白色背景
            background = Image.new("RGB", img.size, (255, 255, 255))
            # 将原图复制到白色背景上
            background.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 将图像保存到内存中的字节流
        img_bytes = BytesIO()
        save_format = "JPEG"  # 统一使用JPEG格式，因为我们已经处理了透明通道
        img.save(img_bytes, format=save_format, quality=95)  # 使用较高的质量设置
        # 编码为 base64
        return save_format.lower(), base64.b64encode(img_bytes.getvalue()).decode(
            "utf-8"
        )


def pk_dump(data, index, dir):
    """
    存储单组数据结果
    """
    filename = os.path.join(dir, f"{index}.pk")
    with open(filename, "wb") as f:
        pk.dump(data, f, protocol=pk.HIGHEST_PROTOCOL)
    return


def pk_load(index, dir):
    """
    读取单组数据结果
    """
    filename = os.path.join(dir, f"{index}.pk")
    with open(filename, "rb") as f:
        data = pk.load(f)
    return data


def search_pk(dir):
    """
    搜索目标路径下的pk文件，返回搜索到的index。用于判断成功任务和失败任务index，以及用于断点续传
    """
    pk_indexs = [
        int(filename.strip(".pk"))
        for filename in os.listdir(dir)
        if filename.endswith(".pk")
    ]
    return pk_indexs

def read_dataset(file_path, required=None):
    if required is None:
        required = []
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".csv":
        return read_dataset_csv(file_path, required)
    elif suffix in [".xls", ".xlsx"]:
        return read_dataset_excel(file_path, required)
    else:
        logger.error(f"[ERR] Unsupported file format: {suffix}")
        exit(1)


def read_dataset_csv(csv_path, required=None):
    if required is None:
        required = []
    logger.info(f"Loading dataset from CSV: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
        miss = [c for c in required if c not in df.columns]
        if miss:
            logger.info(f"[ERR] Missing columns: {', '.join(miss)}")
            logger.error(f"[ERR] Missing columns: {', '.join(miss)}")

    except Exception as e:
        logger.info(f"[ERR] load csv: {e}")
        logger.error(f"[ERR] load csv: {e}")
        exit(1)

    return df

def read_dataset_excel(excel_path, required=None):
    if required is None:
        required = []
    logger.info(f"Loading dataset from Excel: {excel_path} ...")
    try:
        df = pd.read_excel(excel_path)
        miss = [c for c in required if c not in df.columns]
        if miss:
            logger.info(f"[ERR] Missing columns: {', '.join(miss)}")
            logger.error(f"[ERR] Missing columns: {', '.join(miss)}")

    except Exception as e:
        logger.info(f"[ERR] load excel: {e}")
        logger.error(f"[ERR] load excel: {e}")
        exit(1)

    return df

def save_data_result(result, output_path):
    """
    保存结果到文件
    """
    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, list):
        df = pd.DataFrame(result)
    else:
        logger.error(f"[ERR] Unsupported data type: {type(result)}")
        return
    suffix = os.path.splitext(output_path)[1].lower()
    if suffix == ".csv":
        save_data_result_to_csv(df, output_path, encoding='gbk')
    elif suffix in [".xls", ".xlsx"]:
        save_data_result_to_excel(df, output_path)
    else:
        logger.error(f"[ERR] Unsupported file format: {suffix}")
        exit(1)

def save_data_result_to_excel(df, output_path):
    """
    保存结果到Excel
    """
    try:
        df.to_excel(output_path, index=False)
        logger.info(f"结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果到Excel时出错: {e}")
        
def save_data_result_to_csv(df, output_path):
    """
    保存结果到CSV
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果到CSV时出错: {e}")

def shuffle_excel_rows(input_file, output_file=None, random_seed=None):
    """
    将Excel文件的数据行打乱顺序（保留表头不变）

    参数:
    input_file: 输入的Excel文件路径
    output_file: 输出的Excel文件路径（可选）
    random_seed: 随机种子（可选，用于结果可重现）
    """
    try:
        # 设置随机种子（如果提供）
        if random_seed is not None:
            np.random.seed(random_seed)

        # 读取Excel文件（第一行作为表头）
        df = pd.read_excel(input_file)

        # 检查数据是否足够
        if len(df) < 1:
            logger.warning("警告：文件没有数据行")
            return
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # 打乱所有数据行的顺序（不包括表头）
        shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # 生成输出文件名（如果未提供）
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            extension = os.path.splitext(input_file)[1]
            output_file = f"{base_name}_shuffled_{current_time}{extension}"

        # 保存到新文件（保留原表头）
        shuffled_df.to_excel(output_file, index=False)

        logger.info(f"成功处理文件！")
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"表头已保留: {list(df.columns)}")
        logger.info(f"数据行数: {len(df)}")
        logger.info(f"已打乱数据行顺序")

    except FileNotFoundError:
        logger.error(f"错误：找不到文件 '{input_file}'")
    except Exception as e:
        logger.error(f"处理文件时发生错误: {str(e)}")
    return output_file


if __name__ == "__main__":
    print(read_dataset(r'D:\crwu\zjts_foodname\data\evaluation\eval_res_of_llm\20250930122257_evaluation_results_qwen2.5-vl-72b-instruct_by_api.xlsx'))
    print(read_dataset_excel(r'D:\crwu\zjts_foodname\data\evaluation\eval_res_of_llm\20250930122257_evaluation_results_qwen2.5-vl-72b-instruct_by_api.xlsx'))
