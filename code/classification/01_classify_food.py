import csv
import json
import os
import pandas as pd
from openai import OpenAI
from pathlib2 import Path
import pickle as pk
import time
from multiprocessing import Pool
from tqdm import tqdm
from io import StringIO  # 添加 StringIO 导入
from wcwidth import wcwidth

from packages.constants import (
    PROJECT_ROOT,
    API_URL,
    API_KEY_DISCOUNT,
)
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)
from packages.text_match import _BOX_RE, exact_match, extract_final_cn
from packages.call_api import call_openai_with_timeout
from packages.file_deal import (
    pk_dump,
    search_pk,
    pk_load,
)

def chinese_ljust(text, width, fillchar=' '):
    """支持中文字符的左对齐"""
    current_width = sum(wcwidth(char) for char in text)
    padding = max(0, width - current_width)
    return text + fillchar * padding

def format_csv_to_aligned_columns(csv_file_path, output_txt_path=None):
    """
    将CSV文件格式化为对齐的列

    Args:
        csv_file_path: 输入的CSV文件路径
        output_txt_path: 输出的文本文件路径（可选）

    Returns:
        str: 格式化后的文本
    """
    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    # 计算每列的最大显示宽度（考虑中文字符）
    if not rows:
        return ""

    num_columns = len(rows[0])
    column_widths = [0] * num_columns

    for row in rows:
        for i, cell in enumerate(row):
            cell_width = sum(wcwidth(char) for char in str(cell))
            column_widths[i] = max(column_widths[i], cell_width)

    # 构建格式化字符串
    formatted_lines = []
    for row in rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            # 左对齐，宽度为该列最大宽度
            formatted_cell = chinese_ljust(str(cell), column_widths[i])
            formatted_cells.append(formatted_cell)
        formatted_lines.append(",".join(formatted_cells))  # 用逗号分隔

    formatted_text = "\n".join(formatted_lines)

    # 输出到文件（如果指定了输出路径）
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(formatted_text)

    return formatted_text

def get_unfinished_tasks(temp_result_dir, total_tasks):
    """
    通过比较临时文件夹中的结果文件和样本数据集，确定未完成的任务

    Args:
        samples_df: 样本数据集DataFrame
        temp_result_dir: 临时结果文件夹路径
        max_tasks: 最大任务数限制，如果指定则只检查前max_tasks个任务
    """
    # 扫描临时文件夹中的结果文件
    completed_tasks = set(search_pk(temp_result_dir))
    all_tasks = set(range(total_tasks))
    return list(all_tasks - completed_tasks)


def parse_csv_from_response(response_text, batch_size):
    """
    从OpenAI响应中解析CSV数据，处理多种可能的格式
    Args:
        response_text: 包含CSV数据的响应文本
    Returns:
        pandas DataFrame对象
    """
    if not isinstance(response_text, str):
        logger.error(f"输入类型错误：预期字符串，实际得到 {type(response_text)}")
        return None

    start_id = 0
    lines = [line.strip(" `") for line in response_text.splitlines()]
    for id, line in enumerate(lines):
        if line.startswith("菜品名,"):
            start_id = id + 1
            break

    if not start_id:
        # 未找到任何CSV数据
        logger.error("未找到CSV格式数据，原文内容:")
        logger.error("-" * 50)
        logger.error(
            response_text[:200] + "..." if len(response_text) > 200 else response_text
        )
        logger.error("-" * 50)
        return None
    if start_id + batch_size > len(response_text.splitlines()):
        logger.warning("警告: 预期的行数超过响应中的实际行数，可能数据不完整")
    #     return None
    for id, line in enumerate(lines[start_id : start_id + batch_size]):
        if len(line.split(",")) != 14:
            logger.warning("警告: 解析的CSV列数不匹配预期，可能数据格式有误")
            logger.warning(f"line id={start_id + id + 1}, line={line}")
            return None

    csv_content = "\n".join(lines[start_id : start_id + batch_size])
    if csv_content:
        try:
            # return pd.read_csv(StringIO(csv_content))
            return csv_content

        except pd.errors.EmptyDataError:
            logger.error("CSV数据为空")
            return None
        except Exception as e:
            logger.error(f"解析CSV数据时出错: {str(e)}")
            logger.error("问题数据内容:")
            logger.error("-" * 50)
            logger.error(
                response_text[:200] + "..."
                if len(response_text) > 200
                else response_text
            )
            logger.error("-" * 50)
            return None


def process_foodname(model_name, batch_num, foodname_list, message_list, fn):
    logger.info(f"Processing {batch_num}...")
    prompt = "\n".join(foodname_list)
    messages = message_list + [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 使用带重试和超时的API调用
        response = call_openai_with_timeout(
            client=client,
            model=model_name,
            messages=messages,
            timeout=60,  # 设置60秒超时
        )

        ans = [batch_num, response.choices[0].message.content]
        csv_content = parse_csv_from_response(ans[1], len(foodname_list))
        if csv_content is None:
            return ''
        pk_dump(ans, batch_num, fn)
        logger.info(f"Complete batch {batch_num}, saved to {fn}")
        return ans
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return ""

def create_context(model_name):
    logger.info(f"Creating context for {model_name}...")
    propmpt_json_path = r"/date0/crwu/zjts_foodname/data/classification/data/prompt.json"
    with open(propmpt_json_path, 'r') as f:
        message_list = json.load(f)
    return message_list

if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = API_URL
    os.environ["OPENAI_API_KEY"] = API_KEY_DISCOUNT

    version = 'V100R25C20'
    client = OpenAI()
    file_name = "food_name_test"
    FD = pd.read_csv(f"/date0/crwu/zjts_foodname/data/classification/{file_name}.csv")
    temp_pk_target = Path(
        f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/temp_pk"
    )
    os.makedirs(temp_pk_target, exist_ok=True)
    FD["outfn"] = FD["菜名"].apply(lambda s: temp_pk_target / f"{s}.pk")

    # prepare for mapping
    FD_ = FD[FD["outfn"].apply(lambda fn: not os.path.exists(fn))]
    # inputs = FD_[["菜名", "outfn"]].values  # convert to tuple for map function

    model_name = "gemini-2.5-pro"  # "gemini-2.5-flash-preview-04-17-nothinking" #"gemini-2.5-pro" #"qwen-vl-max-latest"

    t_start = time.time()


    food_name_list = FD_["菜名"].tolist()
    # random.shuffle(food_name_list)
    test_num = 30000
    food_name_list = food_name_list[:test_num]

    logger.info(f"{len(food_name_list)} food names remaining for processing...")


    batch_size = 5
    batch_list = [
        food_name_list[i : i + batch_size] for i in range(0, len(food_name_list), batch_size)
    ]
    unfinished_batch_num_list = get_unfinished_tasks(temp_pk_target, len(batch_list))
    unfinished_batch_list = [batch_list[i] for i in unfinished_batch_num_list]
    logger.info(f"{len(unfinished_batch_list)} batches remaining for processing...")
    # test_num = 10
    # unfinished_batch_list=unfinished_batch_list[:test_num]
    failed_batches = []
    while unfinished_batch_num_list:
        failed_batches = []
        message_list = create_context(model_name)

        create_context_time = time.time() - t_start
        logger.info(f"Context creation time: {create_context_time:.2f} seconds")

        with Pool(150) as pool:
            # 用于存储所有异步任务
            async_results = []

            for batch_num, foodname_list in tqdm(
                zip(unfinished_batch_num_list, unfinished_batch_list)
            ):
                logger.info(
                    f"Processing batch {batch_num} with {len(foodname_list)} items..."
                )
                # 存储异步任务结果
                result = pool.apply_async(
                    func=process_foodname,
                    args=(
                        model_name,
                        batch_num,
                        foodname_list,
                        message_list,
                        temp_pk_target,
                    ),
                )
                async_results.append(result)
                # break

            # 等待所有任务完成并获取结果
            logger.info("\n等待所有异步任务完成...")
            for i, async_result in enumerate(tqdm(async_results)):
                try:
                    result = async_result.get(timeout=300)  # 300秒超时
                    if not result:  # 如果返回空字符串，说明处理失败
                        failed_batches.append(unfinished_batch_num_list[i])
                except Exception as e:
                    logger.info(f"Batch {unfinished_batch_num_list[i]} failed: {str(e)}")
                    failed_batches.append(unfinished_batch_num_list[i])
            unfinished_batch_num_list = failed_batches

            # 等待所有进程完成
            pool.close()
            pool.join()

        logger.info("\n所有异步任务已完成...")

    # 从pk文件中读取并合并结果
    logger.info("\n开始合并结果...")
    all_results = []
    pk_files = list(temp_pk_target.glob("*.pk"))
    logger.info(f"\n开始处理 {len(pk_files)} 个pk文件...")

    fail_pk_files = []
    for pk_file in tqdm(pk_files):
        try:
            with open(pk_file, "rb") as f:
                data = pk.load(f)
                if isinstance(data, list) and len(data) == 2:
                    batch_num, response_text = data
                    # 解析CSV数据
                    # df = parse_csv_from_response(response_text, batch_size)
                    csv_text = parse_csv_from_response(response_text, batch_size)
                    # print(df)
                    if csv_text is not None:
                        all_results.append((batch_num, csv_text))
                    else:
                        logger.error(f"文件 {pk_file} 中未能解析出有效的DataFrame")
                        fail_pk_files.append(pk_file)
        except Exception as e:
            logger.error(f"处理文件 {pk_file} 时出错: {str(e)}")
            fail_pk_files.append(pk_file)
        # break

    # 合并所有DataFrame
    if all_results:
        print(f"\n成功解析 {len(all_results)} 个结果文件")
        # 按batch_num排序
        all_results.sort(key=lambda x: x[0])
        # 提取所有DataFrame并合并
        # all_df = pd.concat([df for _, df in all_results], ignore_index=True)

        # 保存合并后的结果        
        output_path = Path(
            f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/merged_results.csv"
        )
        output_path2 = Path(
            f"/date0/crwu/zjts_foodname/data/classification/outputs_by_type_{file_name}_{version}/merged_results_formatted.csv"
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            headers = '菜品名,第一级别,第二级别,第三级别,第四级别,主要食材种类,烹饪方式,菜系（地域）分类,消费场景,营养建议分类,记忆/推理菜,早中晚餐消费场景,食用频率,熟悉程度\n'
            f.writelines(headers)
        line_num = 0
        for _, csv_text in all_results:
            with open(output_path, 'a') as f:
                for line in csv_text.splitlines():
                    f.writelines(line+'\n')
                    line_num +=1
        format_csv_to_aligned_columns(output_path, output_path2)
        print(f"保存合并结果到: {output_path}")
        print(f"保存format结果到: {output_path2}")
        print(f"总计处理了 {line_num} 条数据")
    else:
        print("没有找到有效的数据可以合并")

    t_end = time.time()
    t = t_end - t_start

    logger.info("=== 执行统计 ===")
    logger.info(f"* 待处理批次数: {len(unfinished_batch_list)}")
    if len(failed_batches) > 0:
        logger.error(f"* 失败批次数: {len(failed_batches)}")
        logger.error(f"* 失败批次编号: {failed_batches}")

    logger.info(f"* 待解析文件数: {len(pk_files)}")
    logger.info(f"* 成功解析文件数: {len(all_results)}")
    logger.error(f"* 失败解析文件数: {len(fail_pk_files)}")
    logger.error(f"* 失败解析文件: {fail_pk_files}")
    logger.info(f"* 总执行时间: {t:.2f} 秒")
    for fail_file in fail_pk_files:
        # os.remove(fail_file)
        logger.info(f"删除失败文件: {fail_file}")