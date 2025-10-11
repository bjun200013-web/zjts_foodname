from multiprocessing import Pool, Manager
import argparse
import os
from openai import OpenAI

import pandas as pd
from tqdm import tqdm

from packages.constants import (
    PROJECT_ROOT,
    API_URL,
    API_KEY_DEFAULT,
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
    save_data_result,
    search_pk,
    pk_load,
)
from packages.call_api import call_openai_with_timeout


def get_score_from_str(s:str):
    """
    获取字符串里第一个数字
    """
    for letter in s:
        if letter.isdigit():
            return int(letter)
    return None


def evaluate_wrapper(args):
    """
    包装函数, 用于进程池调用
    """
    index, local_food_id, model_name, real, predicted, output_path = args
    # 在每个子进程中重新创建客户端
    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY_DEFAULT
    )
    # todo 这里要看下是否要优化, 比如当predicted的范围包含real可以给4分, 反过来给1分；明确每个分数等级的规则
    messages = [
        {
            "role": "user",
            "content": f"""
            请判断以下两个菜名是否指代同一种食物或非常相似, 并按 0 到 5 分之间的整数进行打分：
            - 0 分：完全不同
            - 5 分：完全相同或几乎相同

            真实菜名：{real}
            预测菜名：{predicted}

            请严格按照以下 JSON 格式输出, **不要添加任何解释、换行、说明文字或额外字段**, 仅输出一行合法 JSON：
            ```json
            {{ "score": <0到5之间的整数> }}
            ```""",
        }
    ]
    try:
        response = call_openai_with_timeout(client, model_name, messages)
        res = response.choices[0].message.content
        score = get_score_from_str(res)
        if score is None:
            logger.error(f"获取分数失败, 详情: {response}")
            return (index, local_food_id, real, predicted, model_name, None, False)
        data = (local_food_id, real, predicted, model_name, res)
        logger.info(f'菜品: {real}, id: {local_food_id}, 模型预测: {predicted}, 模型{model_name}打分: {score}')
        pk_dump(data, index, output_path)
        return (index, local_food_id, real, predicted, model_name, score, True)
    except Exception as e:
        logger.error(f"{model_name} API调用异常: {e}")
        return (index, local_food_id, real, predicted, model_name, None, False)


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    start = args.start
    num = args.num
    file_path = args.input
    output_path = args.output
    pk_path = os.path.join(os.path.dirname(output_path), "temp_pk")
    os.makedirs(pk_path, exist_ok=True)

    # 读取当前批次数据
    df = read_dataset(file_path)
    batch_df = df.iloc[start : start + num].copy()

    real_dishes = batch_df["ground_truth_answer"]
    predicted_dishes = batch_df["model_prediction"]

    models = [
        # GEMINI_MODEL_NAME,
        GPT_MODEL_NAME,
        CLAUDE_MODEL_NAME,
    ]
    columns = (
        ['sample_index', 'image_path', "ground_truth_answer", "model_prediction"]
        + [model + "的评分" for model in models]
    )
    results_df = pd.DataFrame(columns=columns)
    for column in columns:
        if column in batch_df:
            results_df[column] = batch_df[column].values

    # 环境变量
    os.environ["OPENAI_BASE_URL"] = API_URL
    os.environ["OPENAI_API_KEY"] = API_KEY_DEFAULT

    # 准备所有任务
    # 一个一共要跑num个菜名, 每个菜名要跑len(models)个评分, 每个批次要跑num*len(models)个评分
    # 按model顺序评估, 那么0~num-1是models[0]的评分, 第x菜的第y个model评分就是start*len(models) + (y-1)*num + (x-1)
    tasks = []
    target_index_list = list(
        range(start * len(models), start * len(models) + len(real_dishes) * len(models))
    )


    # 初始化进度条
    pbar = tqdm(total=len(target_index_list), desc="API调用进度")

    for index in target_index_list:
        model = models[(index - start * len(models)) // len(real_dishes)]
        local_food_id = (index - start * len(models)) % len(real_dishes)
        real = real_dishes.values[local_food_id]
        pred = predicted_dishes.values[local_food_id]

        if pd.isna(real) or pd.isna(pred):
            # 直接处理空值情况
            pk_dump((local_food_id, real, pred, model, '0'), index, pk_path)
            logger.info(f'菜品: {real}, id: {local_food_id}, 模型预测: {pred}, 模型{model}打分: {0}')
            continue
        if exact_match(real, pred):
            # 直接处理完全匹配情况
            pk_dump((local_food_id, real, pred, model, '5'), index, pk_path)
            logger.info(f'菜品: {real}, id: {local_food_id}, 模型预测: {pred}, 模型{model}打分: {5}')
            continue

        # 需要API调用的任务
        tasks.append((index, local_food_id, model, real, pred, pk_path))

    # 处理已完成的任务
    remain_index_list = list(set(target_index_list) - set(search_pk(pk_path)))
    logger.info(f'已完成的菜品数量: {len(target_index_list) - len(remain_index_list)}')
    count = 0
    for _ in list(set(target_index_list) - set(remain_index_list)):
        count += 1
        pbar.update(1)
    logger.info(f'更新进度{count}')
    tasks = [task for task in tasks if task[0] in remain_index_list]
    logger.info(f'待完成的菜品: {remain_index_list}')
    # 分批处理任务, 避免一次性提交太多
    # todo 试下190有没有提升
    batch_size = 60
    while remain_index_list:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            with Pool(min(batch_size, len(batch_tasks))) as pool:
                # 使用 imap_unordered 获取完成顺序的结果
                for result in pool.imap_unordered(evaluate_wrapper, batch_tasks):
                    index, local_food_id, real, pred, model, score, success = result
                    if success and score is not None:
                        # 结果已经在 evaluate_wrapper 中保存到文件了
                        pbar.update(1)
                        pass
        # 检查已完成的pk并重试
        remain_index_list = list(set(target_index_list) - set(search_pk(pk_path)))
        tasks = [task for task in tasks if task[0] in remain_index_list]
        logger.info(f'剩余未完成的菜品: {remain_index_list}')

    pbar.close()

    logger.info(f"✅ 菜品{start}~{start+num}评估完成。")

    # 从临时文件加载所有结果
    pk_indexs = search_pk(pk_path)
    for pk_index in pk_indexs:
        if pk_index not in target_index_list:
            continue
        local_food_id, real, pred, model, res = pk_load(pk_index, pk_path)
        results_df.loc[local_food_id, model + "的评分"] = res

    save_data_result(results_df, output_path)
    logger.info(f"✅ 批次评估结果已保存至 {output_path}")