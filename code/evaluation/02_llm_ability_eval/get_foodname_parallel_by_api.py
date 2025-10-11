from multiprocessing import Pool
import time
import os
from datetime import datetime
import argparse
import shutil  # 用于删除临时文件夹

from openai import OpenAI
from tqdm import tqdm
import pandas as pd

from packages.constants import (
    PROJECT_ROOT,
    EVAL_RES_OUTPUT_PATH,
    EVAL_DATA_IMAGE_ROOT,
    EVAL_DATA_EXCEL_PATH,
    API_URL,
    API_KEY_DISCOUNT,
)
from packages.my_logger import setup_logging

log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_res_of_llm")
logger = setup_logging(log_dir=log_dir)
from packages.text_match import _BOX_RE, exact_match, extract_final_cn
from packages.call_api import call_openai_with_timeout
from packages.file_deal import (
    encode_image_with_resize,
    pk_dump,
    search_pk,
    pk_load,
    read_dataset,
    shuffle_excel_rows,
)


def get_unfinished_tasks(samples_df, temp_result_dir, max_tasks=None):
    """
    通过比较临时文件夹中的结果文件和样本数据集, 确定未完成的任务

    Args:
        samples_df: 样本数据集DataFrame
        temp_result_dir: 临时结果文件夹路径
        max_tasks: 最大任务数限制, 如果指定则只检查前max_tasks个任务
    """
    # 扫描临时文件夹中的结果文件
    completed_tasks = set(search_pk(temp_result_dir))
    # 找出未完成的任务, 考虑最大任务数限制
    total_tasks = max_tasks if max_tasks is not None else len(samples_df)
    all_tasks = set(range(total_tasks))
    return list(all_tasks - completed_tasks)


def request_single_img(input):
    idx, row, img_root, model_name, temp_dir = input
    orig_path = row["image_path"]
    gt = str(row["food_name"])
    ttype = str(row["type"])
    first_level = str(row["first_level"])
    second_level = str(row["second_level"])
    main_ingredients = str(row["main_ingredients"])
    cuisine_region_classification = str(row["cuisine_region_classification"])
    cooking_method = str(row["cooking_method"])
    consumption_scene = str(row["consumption_scene"])
    nutritional_advice_categories = str(row["nutritional_advice_categories"])

    img_path = os.path.join(img_root, os.path.basename(orig_path))

    # 检查是否已经处理过此样本
    result_file = os.path.join(temp_dir, f"{idx}.pk")
    if os.path.exists(result_file):
        return None

    if not os.path.exists(img_path):
        logger.warning(
            f"[WARN] image missing: {img_path} (index={idx}, orig={orig_path})"
        )
        result = {
            "sample_index": idx,
            "image_path": img_path,
            "problem": prompt,
            "ground_truth_answer": gt,
            "type": ttype,
            "model_prediction": "",
            "model_thinking": "",
            "is_correct": False,
            "raw_answer": "",
            "first_level": first_level,
            "second_level": second_level,
            "main_ingredients": main_ingredients,
            "cuisine_region_classification": cuisine_region_classification,
            "cooking_method": cooking_method,
            "consumption_scene": consumption_scene,
            "nutritional_advice_categories": nutritional_advice_categories,
        }
        pk_dump(result, idx, temp_dir)
        return result
    logger.info(f"Processing {img_path}...")

    client = OpenAI()
    # prompt = (
    #     ""
    #     + f"你是⼀个从业多年的餐饮专家, 请仔细观察图像内容, 综合图像中的线索报出图中食物的最准确中文名。\n"
    #     + f"1.请分析图中食物是否分为多个菜品/食物品类, 如果有, 请分别报出名称\n"
    #     + f"2.请分析食物的形状、颜色、烹饪方式、包含的可能食材的种类, 结合各地域菜系的特色, 推理最可能的菜品名称\n"
    #     + f"3.如果无法准确推断食品名称, 减少定语, 描述最可能的食品品类, 如：不确定是否是“烤包子”或“煎包子”时, 可以输出“包子”\n"
    #     + f"4.输出结果时菜名请输出中文。请严格按照这个格式输出: Answer: $ \\boxed{{answer}}$\n"
    # )

    prompt = "你是一个专业的中文美食识别AI。请根据图像内容仔细分析主要食材、烹饪方法、颜色特征、形状特征和摆盘样式, 再综合所有线索推理出图中食物的最可能的中文菜名。输出分析过程, 在结尾按照如下格式输出答案: Answer: $ \\boxed{{answer}}$"

    try:
        format, base64_image = encode_image_with_resize(img_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意, 传入Base64, 图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {
                            "url": f"data:image/{format};base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 使用带重试和超时的API调用
        response = call_openai_with_timeout(
            client=client,
            model=model_name,
            messages=messages,
            timeout=30,  # 设置30秒超时
        )

        pred = response.choices[0].message.content
        logger.info(str(response))

        model_thinking = (
            response.choices[0].message.reasoning_content.strip()
            if hasattr(response.choices[0].message, "reasoning_content")
            else ""
        )
        model_answer = extract_final_cn(pred) if extract_final_cn(pred) else pred
        is_ok = exact_match(model_answer, gt)

        result = {
            "sample_index": idx,
            "image_path": img_path,
            "problem": prompt,
            "ground_truth_answer": gt,
            "type": ttype,
            "model_prediction": model_answer,
            "model_thinking": model_thinking,
            "is_correct": is_ok,
            "raw_answer": response,
            "first_level": first_level,
            "second_level": second_level,
            "main_ingredients": main_ingredients,
            "cuisine_region_classification": cuisine_region_classification,
            "cooking_method": cooking_method,
            "consumption_scene": consumption_scene,
            "nutritional_advice_categories": nutritional_advice_categories,
        }

        # 立即保存单个结果
        pk_dump(result, idx, temp_dir)
        return result

    except Exception as e:
        logger.error(f"处理样本 {idx} 时发生错误: {str(e)}")
        # 出现异常时返回None, 让任务重试
        return None


if __name__ == "__main__":

    ap = argparse.ArgumentParser("Evaluate model (Torch/HF) on Excel dataset.")
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="qwen2.5-vl-72b-instruct",  # "gemini-2.5-flash-preview-04-17-nothinking" #"gemini-2.5-pro" #"qwen-vl-max-latest"
        help="api调用的模型名称",
    )
    ap.add_argument(
        "-e",
        "--excel_path",
        type=str,
        default=EVAL_DATA_EXCEL_PATH,
        help="测试数据集的excel文档路径",
    )
    ap.add_argument(
        "--continue_path",
        type=str,
        default="",
        help=".pk文件的存放路径, 继续上次测试；为空表示不继续；有值表示根据上次的临时文件断点续传",
    )
    ap.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=EVAL_RES_OUTPUT_PATH,
        help="输出结果文件夹位置",
    )
    ap.add_argument(
        "-i",
        "--img_path",
        type=str,
        default=EVAL_DATA_IMAGE_ROOT,
        help="图片所在根路径",
    )
    ap.add_argument(
        "--max_test_img_num",
        type=int,
        default=0,
        help="最大要测试的图片数量, 0代表测试全部",
    )
    ap.add_argument(
        "--api_url",
        type=str,
        default=API_URL,
        help="调用的API的URL, 默认为青云的https://api.qingyuntop.top/v1",
    )
    ap.add_argument(
        "--api_key",
        type=str,
        default=API_KEY_DISCOUNT,
        help="API的令牌",
    )
    ap.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="是否随机打乱数据集顺序",
    )

    args = ap.parse_args()
    os.environ["OPENAI_BASE_URL"] = args.api_url
    os.environ["OPENAI_API_KEY"] = args.api_key

    model_name = args.model_name
    excel_path = args.excel_path
    # 随机打乱数据顺序, 方便调试
    if args.shuffle:
        excel_path = shuffle_excel_rows(excel_path)

    # 读取数据集
    samples_df = read_dataset(
        excel_path, required=["image_path", "food_name", "type"]
    )
    samples_df.dropna(subset=["food_name"], inplace=True)
    samples_df["food_name"] = samples_df["food_name"].astype(str)
    samples_df["type"] = samples_df["type"].astype(str).fillna("")

    # 创建临时结果文件夹
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.continue_path:
        temp_result_dir = args.continue_path
        logger.info(f"继续任务: {os.path.abspath(temp_result_dir)}")
        unfinished_tasks = get_unfinished_tasks(
            samples_df, temp_result_dir, args.max_test_img_num
        )
    else:
        temp_result_dir = os.path.join(
            EVAL_RES_OUTPUT_PATH, f"{current_time}_temp_results_api"
        )
        os.makedirs(temp_result_dir, exist_ok=True)
        logger.info(f"临时结果文件夹: {os.path.abspath(temp_result_dir)}")
        # 初始任务列表
        unfinished_tasks = list(range(len(samples_df)))
        if args.max_test_img_num > 0:
            unfinished_tasks = unfinished_tasks[: args.max_test_img_num]

    t_start = time.time()
    results = []

    with Pool(processes=190) as pool:
        while unfinished_tasks:
            # 准备当前批次的任务
            current_tasks = [
                (idx, samples_df.iloc[idx], args.img_path, model_name, temp_result_dir)
                for idx in unfinished_tasks
            ]

            # 使用imap_unordered处理任务, 每完成一个就能得到结果
            for result in tqdm(
                pool.imap_unordered(request_single_img, current_tasks),
                total=len(current_tasks),
                desc="Processing images",
            ):
                if result is not None:
                    results.append(result)

            # 更新未完成任务列表, 传入最大任务数限制
            max_tasks = args.max_test_img_num if args.max_test_img_num > 0 else None
            unfinished_tasks = get_unfinished_tasks(
                samples_df, temp_result_dir, max_tasks
            )
            if unfinished_tasks:
                logger.info(f"还有 {len(unfinished_tasks)} 个任务未完成, 继续处理...")

    results = []
    # 所有任务完成, 保存最终结果
    final_output_path = os.path.join(
        args.output_path, f"{current_time}_evaluation_results_{model_name}_by_api.xlsx"
    )
    pk_indexs = search_pk(temp_result_dir)
    for pk_index in pk_indexs:
        result = pk_load(pk_index, temp_result_dir)
        results.append(result)
    results.sort(key=lambda x: x["sample_index"])
    pd.DataFrame(results).to_excel(final_output_path, index=False)
    logger.info(f"已保存最终结果到: {final_output_path}")

    # 清理临时文件夹
    shutil.rmtree(temp_result_dir)
    logger.info(f"已清理临时文件夹: {temp_result_dir}")

    t_end = time.time()
    logger.info(f"总用时: {t_end - t_start:.2f}秒")
