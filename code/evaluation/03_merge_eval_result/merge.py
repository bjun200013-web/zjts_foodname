import glob
import pandas as pd
import re
import json
import argparse
import os

from packages.constants import (
    PROJECT_ROOT,
    GPT_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
)
from packages.file_deal import read_dataset, save_data_result
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)


def get_score_from_str(s: str):
    """
    获取字符串里第一个数字
    """
    if isinstance(s, (int, float)):
        return s
    elif isinstance(s,str):
        for letter in s:
            if letter.isdigit():
                return int(letter)
        else:
            logger.error(f"解析错误: {s}")
    return None


def extract_score(val):
    try:
        if pd.isna(val):
            return None

        text = str(val).strip()

        # 1. 如果是纯数字, 直接返回
        if text.isdigit():
            return int(text)

        # 2. 清除 markdown、<think>标签等包裹
        text = re.sub(r"```json|```|<think>.*?</think>", "", text, flags=re.DOTALL)

        # 3. 恢复引号（如 ""score"" → "score"）
        text = text.replace('""', '"').strip()

        # 4. 提取所有 JSON 对象, 选最后一个（更健壮）
        json_matches = list(re.finditer(r"\{[\s\S]*?\}", text))
        if not json_matches:
            return None

        json_str = json_matches[-1].group(0)

        # 5. 加载 JSON 并提取 score
        parsed = json.loads(json_str)

        return int(parsed["score"])
    except Exception as e:
        logger.error(f"解析错误: {val}\n错误信息: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="合并评估结果并提取分数")
    parser.add_argument("-i", "--input_dir", required=True, help="输入结果目录路径")
    parser.add_argument("--output_dir", help="输出目录路径（默认与输入目录相同）")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            GEMINI_MODEL_NAME,
            GPT_MODEL_NAME,
            DEEPSEEK_MODEL_NAME,
            CLAUDE_MODEL_NAME,
        ],
        help="要评估的模型列表",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default='',
        help="最终输出结果的前缀",
    )

    args = parser.parse_args()

    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else args.input_dir

    # 合并所有批次结果
    all_files = sorted(
        glob.glob(os.path.join(args.input_dir, "eval_*.csv")),
        key=lambda x: int(re.search(r"eval_(\d+)\.csv", x).group(1)),
    )
    if not all_files:
        logger.error(f"错误: 在目录 {args.input_dir} 中没有找到 eval_*.csv 文件")
        return

    logger.info(f"找到 {len(all_files)} 个文件需要合并")
    dfs = [read_dataset(f) for f in all_files]
    all_results = pd.concat(dfs, ignore_index=True)

    # 保存合并结果
    merged_output = os.path.join(output_dir, "final_merged_results.csv")
    save_data_result(all_results, merged_output)
    logger.info(f"合并完毕, 总条数: {len(all_results)}")
    logger.info(f"合并结果保存至: {merged_output}")

    # 提取分数
    df = all_results.copy()

    # 应用于每个评分列
    exist_model_score = []
    for model in args.models:
        score_column = f"{model}的评分"
        if score_column in df.columns:
            exist_model_score.append(model)
            df[model] = df[score_column].apply(get_score_from_str)
        else:
            print(f"警告: 未找到列 '{score_column}', 跳过模型 {model}")

    # 计算每个模型的平均分
    logger.info("\n=== 各模型平均分 ===")
    model_scores = []
    for model in args.models:
        if model in df.columns:
            avg_score = df[model].mean()
            model_scores.append(avg_score)
            logger.info(f"{model}的评分: {avg_score:.3f}")
    # model_scores = []
    # for model in args.models:
    #     if f'{model}的评分' in df.columns:
    #         avg_score = df[f'{model}的评分'].mean()
    #         model_scores.append(avg_score)
    #         logger.info(f"{model}的评分: {avg_score:.3f}")

    # 计算所有模型的平均分
    if model_scores:
        overall_avg = sum(model_scores) / len(model_scores)
        logger.info(f"\n=== 所有模型平均分 ===")
        logger.info(f"总体平均分: {overall_avg:.3f}")

    # 创建要添加的新行数据
    new_row = pd.DataFrame({column: [df[column].mean()] for column in exist_model_score})
    # 使用concat添加新行
    df = pd.concat([df, new_row])

    df["菜名打分平均值"] = df[exist_model_score].mean(axis=1)  # axis=1 表示按行计算
    df["菜名打分标准差"] = df[exist_model_score].std(axis=1)

    # 保存结果
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = 'final_merged_scores'
    scores_output = os.path.join(
        output_dir, f"{output_prefix}_{overall_avg:.3f}.csv"
    )
    save_data_result(df, scores_output)

    logger.info(f"✅ 提取完成, 结果已保存为: {scores_output}")


if __name__ == "__main__":
    main()
