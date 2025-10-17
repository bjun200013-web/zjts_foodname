#!/usr/bin/env python3
from math import ceil
import os
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from packages.constants import (
    PROJECT_ROOT,
    EVAL_RES_OUTPUT_PATH,
    EVAL_DATA_IMAGE_ROOT,
    EVAL_DATA_EXCEL_PATH,
    LLM_SCORE_OUTPUT_PATH,
    API_URL,
    API_KEY_DISCOUNT,
    GPT_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
)
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)
from packages.text_match import exact_match
from packages.call_api import call_openai_with_timeout
from packages.file_deal import (
    pk_dump,
    read_dataset,
    search_pk,
    pk_load,
)

from packages.file_deal import get_output_dir, get_current_script_dir, get_latest_file


def run_model_evaluation(
    api_url: str,
    api_key: str,
    model_name: str,
    max_test_img_num: int,
    output_path: str,
    evaluation_code_dir: str,
    shuffle=False,
):
    """
    执行模型评估
    """
    if model_name.startswith('.') or model_name.startswith('/') or model_name.endswith('/'):
        api_script_path = os.path.join(
        evaluation_code_dir, "02_llm_ability_eval", "get_foodname_local_infer_with_torch.py"
    )
        cmd = [
            "python",
            api_script_path,
            "--model_path",
            model_name,
            "--max_test_img_num",
            str(max_test_img_num),
            "--output_path",
            output_path,
            "--input_excel_path",
            EVAL_DATA_EXCEL_PATH,
            "--input_img_path",
            EVAL_DATA_IMAGE_ROOT,
        ]
    else:
        api_script_path = os.path.join(
        evaluation_code_dir, "02_llm_ability_eval", "get_foodname_parallel_by_api.py"
    )
        cmd = [
            "python",
            api_script_path,
            "--api_url",
            api_url,
            "--api_key",
            api_key,
            "--model_name",
            model_name,
            "--max_test_img_num",
            str(max_test_img_num),
            "--output_path",
            output_path,
            "--shuffle",
            str(shuffle).lower(),
        ]
    subprocess.run(cmd, check=True)


def run_scoring_batch(
    start, num_samples, input_file, output_file, script_dir, log_file
):
    """
    运行单个评分批次
    """
    cmd = [
        "python",
        os.path.join(script_dir, "main.py"),
        "--start",
        str(start),
        "--num",
        str(num_samples),
        "--input",
        input_file,
        "--output",
        output_file,
    ]

    with open(log_file, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_test_img_num",
        type=int,
        default=0,
        help="最多评估的图片数量,0代表全部测试",
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen2.5-vl-72b-instruct", help="模型名称"
    )
    parser.add_argument(
        "--eval_only", type=bool, default=False, help="是否仅评估打分, 不进行图片推理"
    )
    parser.add_argument(
        "--eval_input", type=str, default="", help="仅评估打分场景下输入的excel路径"
    )
    parser.add_argument("--api_url", type=str, default=API_URL, help="API的URL")
    parser.add_argument(
        "--api_key",
        type=str,
        default=API_KEY_DISCOUNT,
        help="API的令牌",
    )
    parser.add_argument(
        "--score_times", type=int, default=1, help="对一次评估结果的评分次数"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="是否随机打乱数据集顺序",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default='',
        help="最终输出结果的前缀",
    )

    args = parser.parse_args()

    # 获取脚本相关路径
    script_abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_abs_path)
    evaluation_code_dir = os.path.dirname(script_dir)
    if not args.eval_only:
        # 执行模型评估
        run_model_evaluation(
            args.api_url,
            args.api_key,
            args.model_name,
            args.max_test_img_num,
            EVAL_RES_OUTPUT_PATH,
            evaluation_code_dir,
            args.shuffle,
        )
        eval_end_time = time.time()
        logger.info(f"模型评估完成, 耗时 {eval_end_time - start_time:.2f} 秒")

        # 获取最新的评估结果文件
        data_file = get_latest_file(EVAL_RES_OUTPUT_PATH)
        total = args.max_test_img_num  # 评估数据条数
    elif args.eval_input and os.path.exists(args.eval_input):
        data_file = args.eval_input        
        eval_end_time = time.time()
    else:
        logger.error(f"开启了eval_only且路径{args.eval_input}不合法")
    if args.max_test_img_num == 0:
        total = len(read_dataset(data_file))  # 评估数据条数
    else:
        total = args.max_test_img_num  # 评估数据条数
    logger.info(f"评估结果文件: {data_file}, 数据共{total}条")
    

    # 对评估结果进行多次评分
    for i in range(args.score_times):
        logger.info(f"第 {i+1} 次评分...")

        # 创建输出目录
        out_dir = os.path.join(LLM_SCORE_OUTPUT_PATH, get_output_dir(data_file))
        logger.info(f"输出目录: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        # 设置批处理参数

        num_samples = max(ceil(total / 10), 1)  # 每组样本数
        num_batches = (total + num_samples - 1) // num_samples

        # 启动所有批处理任务
        processes = []
        for batch in range(num_batches):
            start = batch * num_samples
            out_file = os.path.join(out_dir, f"eval_{start}.csv")
            log_file = os.path.join(out_dir, f"log_{batch}.txt")

            logger.info(f"启动第{batch}批：样本 {start} ~ {start+num_samples-1}, 日志: {log_file}")
            run_scoring_batch(
                start, num_samples, data_file, out_file, script_dir, log_file
            )

        # 等待所有进程完成
        expected_files = {
            os.path.join(out_dir, f"eval_{b*num_samples}.csv")
            for b in range(num_batches)
        }
        while True:
            # 检查是否所有预期的输出文件都已生成
            existing_files = set(f for f in expected_files if os.path.exists(f))
            if len(existing_files) == len(expected_files):
                break

            time.sleep(5)  # 每5秒检查一次

        logger.info(f"第 {i+1} 次评分所有批处理完成")
        scoring_end_time = time.time()
        logger.info(f"评分完成, 耗时 {scoring_end_time - eval_end_time:.2f} 秒")
        # 合并结果
        if args.output_prefix:
            output_prefix = args.output_prefix
        elif os.path.basename(data_file).startswith('results_'):
            output_prefix = os.path.basename(data_file)[8:]
        else:
            output_prefix = ''
        merge_cmd = [
            "python",
            os.path.join(script_dir, "merge.py"),
            "--input_dir",
            out_dir,
            '--output_prefix',
            output_prefix,
        ]
        subprocess.run(merge_cmd, check=True)

    total_end_time = time.time()
    logger.info(f"所有评分完成, 耗时 {total_end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
